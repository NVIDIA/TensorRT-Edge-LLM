# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GR00T N1.7 action head for ONNX export: one AlternateVLDiT flow-matching step.
The graph emits the velocity; the runtime owns the Euler update and binds the timestep-only
constants (:mod:`.schedule`) per step. The embodiment's rows are export-time initializers.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import BLOCK_NORM_EPS, OUT_NORM_EPS, Geometry
from .modeling_gr00t_vl_backbone import Gr00tAttentionFeedForward

logger = logging.getLogger(__name__)

# Finite "blocked" logit, as the ONNX-less lowering uses; a fully masked row
# then averages instead of producing NaN.
_MASKED_LOGIT = -65504.0


def _affine_free_layer_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm with ``elementwise_affine=False``, reduction in fp32."""
    return F.layer_norm(x.float(), x.shape[-1:], eps=eps).to(x.dtype)


class CategoryLinear(nn.Module):
    """One embodiment's slice of ``CategorySpecificLinear`` ``W [categories, in, out]``."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class CategoryMLP(nn.Module):
    """``CategorySpecificMLP``: ``layer2(relu(layer1(x)))``."""

    def __init__(self, in_features: int, hidden: int,
                 out_features: int) -> None:
        super().__init__()
        self.layer1 = CategoryLinear(in_features, hidden)
        self.layer2 = CategoryLinear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    """``MultiEmbodimentActionEncoder`` with the timestep half of ``W2`` folded:
    ``W2 @ concat(a, tau) + b == W2_a @ a + (W2_tau @ tau + b)``, and the second term is
    schedule-only, so it arrives as ``action_time_bias``."""

    def __init__(self, action_dim: int, inner_dim: int) -> None:
        super().__init__()
        self.w1 = CategoryLinear(action_dim, inner_dim)
        self.w2_action = nn.Linear(inner_dim, inner_dim, bias=False)
        self.w3 = CategoryLinear(inner_dim, inner_dim)

    def forward(self, actions: torch.Tensor,
                time_bias: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.w2_action(self.w1(actions)) + time_bias)
        return self.w3(gated)


def _modulate(x: torch.Tensor, row: torch.Tensor, eps: float,
              shift_first: bool) -> torch.Tensor:
    """Split one hoisted modulation row and apply it to an affine-free norm."""
    width = x.shape[-1]
    first, second = row[..., :width], row[..., width:]
    shift, scale = (first, second) if shift_first else (second, first)
    return _affine_free_layer_norm(x, eps) * (scale + 1.0) + shift


class Gr00tCrossAttention(nn.Module):
    """diffusers ``Attention`` reading precomputed K/V; the mask is per key."""

    def __init__(self, inner_dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(inner_dim, heads * head_dim)
        self.to_out = nn.ModuleList([nn.Linear(heads * head_dim, inner_dim)])

    def forward(self, x: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        from ..ops import attention_onnx

        bsz, tokens, _ = x.shape
        query = self.to_q(x).view(bsz, tokens, self.heads,
                                  self.head_dim).transpose(1, 2)
        additive = torch.where(mask, 0.0, _MASKED_LOGIT).to(query.dtype)
        attended = attention_onnx(query,
                                  key.transpose(1, 2),
                                  value.transpose(1, 2),
                                  attn_mask=additive,
                                  is_causal=False,
                                  scale=self.head_dim**-0.5)
        return self.to_out[0](attended.transpose(1,
                                                 2).reshape(bsz, tokens, -1))


class Gr00tSelfAttentionBlockAttention(nn.Module):
    """diffusers ``Attention`` over the state+action tokens."""

    def __init__(self, inner_dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(inner_dim, heads * head_dim)
        self.to_k = nn.Linear(inner_dim, heads * head_dim)
        self.to_v = nn.Linear(inner_dim, heads * head_dim)
        self.to_out = nn.ModuleList([nn.Linear(heads * head_dim, inner_dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from ..ops import attention_onnx

        bsz, tokens, _ = x.shape
        shape = (bsz, tokens, self.heads, self.head_dim)
        query = self.to_q(x).view(shape).transpose(1, 2)
        key = self.to_k(x).view(shape).transpose(1, 2)
        value = self.to_v(x).view(shape).transpose(1, 2)
        attended = attention_onnx(query,
                                  key,
                                  value,
                                  attn_mask=None,
                                  is_causal=False,
                                  scale=self.head_dim**-0.5)
        return self.to_out[0](attended.transpose(1,
                                                 2).reshape(bsz, tokens, -1))


class DiTBlock(nn.Module):
    """One AlternateVLDiT block: ada-norm attention then a feed-forward."""

    def __init__(self, index: int, geometry: Geometry) -> None:
        super().__init__()
        self.is_cross = index % 2 == 0
        inner = geometry.inner_dim
        if self.is_cross:
            self.attn1 = Gr00tCrossAttention(inner, geometry.num_heads,
                                             geometry.head_dim)
        else:
            self.attn1 = Gr00tSelfAttentionBlockAttention(
                inner, geometry.num_heads, geometry.head_dim)
        self.ff = Gr00tAttentionFeedForward(inner, 4 * inner)

    def forward(self,
                x: torch.Tensor,
                row: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normed = _modulate(x, row, BLOCK_NORM_EPS, shift_first=False)
        if self.is_cross:
            x = x + self.attn1(normed, key, value, mask)
        else:
            x = x + self.attn1(normed)
        return x + self.ff(_affine_free_layer_norm(x, BLOCK_NORM_EPS))


class Gr00tActionModel(nn.Module):
    """One flow-matching denoise step of the GR00T N1.7 action head."""

    def __init__(self, geometry: Geometry) -> None:
        super().__init__()
        self.geometry = geometry
        inner, out = geometry.inner_dim, geometry.output_dim
        state_width = geometry.state_history_length * geometry.state_dim
        self.state_encoder = CategoryMLP(state_width, out, inner)
        self.action_encoder = ActionEncoder(geometry.action_dim, inner)
        self.action_decoder = CategoryMLP(out, out, geometry.action_dim)
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(index, geometry) for index in range(geometry.num_layers)
        ])
        self.proj_out_2 = nn.Linear(inner, out)
        self.position_embedding = (nn.Parameter(
            torch.zeros(geometry.action_horizon, inner), requires_grad=False)
                                   if geometry.add_pos_embed else None)

    def forward(self, noise_trajectory: torch.Tensor, state: torch.Tensor,
                adanorm_modulation: torch.Tensor,
                action_time_bias: torch.Tensor,
                text_attention_mask: torch.Tensor,
                image_attention_mask: torch.Tensor, *cross_kv:
                torch.Tensor) -> torch.Tensor:
        """``cross_kv`` is every cross block's K then V, in block order."""
        geometry = self.geometry
        io_type = self.proj_out_2.weight.dtype
        bsz = noise_trajectory.shape[0]

        state_features = self.state_encoder(
            state.to(io_type).reshape(bsz, 1, -1))
        action_features = self.action_encoder(noise_trajectory.to(io_type),
                                              action_time_bias.view(1, 1, -1))
        if self.position_embedding is not None:
            action_features = action_features + self.position_embedding.view(
                1, geometry.action_horizon, geometry.inner_dim)
        hidden = torch.cat([state_features, action_features], dim=1)

        text_blocks = set(geometry.text_blocks)
        cross_index = 0
        for index, block in enumerate(self.transformer_blocks):
            row = adanorm_modulation[index].view(1, 1, -1)
            if block.is_cross:
                key, value = cross_kv[cross_index], cross_kv[cross_index + 1]
                cross_index += 2
                mask = (text_attention_mask
                        if index in text_blocks else image_attention_mask)
                hidden = block(hidden, row, key, value, mask)
            else:
                hidden = block(hidden, row)

        final_row = adanorm_modulation[geometry.num_layers].view(1, 1, -1)
        hidden = _modulate(hidden, final_row, OUT_NORM_EPS, shift_first=True)
        hidden = self.proj_out_2(hidden)[:, 1:1 + geometry.action_horizon]
        return self.action_decoder(hidden).float()

    def cross_kv_names(self) -> List[str]:
        names = []
        for index in self.geometry.cross_blocks:
            names.append(f"cross_k_layer{index:02d}")
            names.append(f"cross_v_layer{index:02d}")
        return names

    def get_onnx_export_args(self,
                             device: str) -> Tuple[tuple, list, list, tuple]:
        geometry = self.geometry
        # Batch 2 keeps the batch axis symbolic (torch.export specializes size-1).
        b, s = 2, 16
        noise = torch.randn(b,
                            geometry.action_horizon,
                            geometry.action_dim,
                            device=device,
                            dtype=torch.float32)
        state = torch.zeros(b,
                            geometry.state_history_length,
                            geometry.state_dim,
                            device=device,
                            dtype=torch.float32)
        modulation = torch.zeros(geometry.num_modulation_sites,
                                 geometry.modulation_dim,
                                 device=device,
                                 dtype=torch.float16)
        time_bias = torch.zeros(geometry.inner_dim,
                                device=device,
                                dtype=torch.float16)
        masks = [
            torch.ones(b, 1, 1, s, device=device, dtype=torch.bool)
            for _ in range(2)
        ]
        cross = [
            torch.zeros(b,
                        s,
                        geometry.num_heads,
                        geometry.head_dim,
                        device=device,
                        dtype=torch.float16)
            for _ in range(2 * len(geometry.cross_blocks))
        ]
        args = (noise, state, modulation, time_bias, *masks, *cross)
        input_names = [
            "noise_trajectory", "state", "adanorm_modulation",
            "action_time_bias", "text_attention_mask", "image_attention_mask"
        ] + self.cross_kv_names()
        batch = torch.export.Dim("batch_size", min=1, max=256)
        vl_len = torch.export.Dim("vl_len", min=2, max=32768)
        # torch.export sees the varargs as one element, so their shapes nest.
        dynamic_shapes = ({
            0: batch
        }, {
            0: batch
        }, {}, {}, {
            0: batch,
            3: vl_len
        }, {
            0: batch,
            3: vl_len
        }, tuple({
            0: batch,
            1: vl_len
        } for _ in cross))
        return args, input_names, ["velocity"], dynamic_shapes


def _category_slice(head: Dict[str, torch.Tensor], prefix: str,
                    embodiment_id: int) -> Dict[str, torch.Tensor]:
    stacked = head[f"{prefix}.W"]
    bias = head[f"{prefix}.b"]
    if not 0 <= embodiment_id < stacked.shape[0]:
        raise ValueError(
            f"embodiment id {embodiment_id} is outside {prefix}.W with "
            f"{stacked.shape[0]} categories")
    # nn.Linear stores [out, in]; the checkpoint stacks [categories, in, out].
    return {
        "weight": stacked[embodiment_id].T.contiguous(),
        "bias": bias[embodiment_id].contiguous(),
    }


def load_weights(model: Gr00tActionModel, head: Dict[str, torch.Tensor],
                 dtype: torch.dtype) -> None:
    geometry = model.geometry
    embodiment = geometry.embodiment_id
    state: Dict[str, torch.Tensor] = {}

    def baked(target: str, source: str) -> None:
        for name, tensor in _category_slice(head, source, embodiment).items():
            state[f"{target}.linear.{name}"] = tensor

    for module in ("state_encoder", "action_decoder"):
        baked(f"{module}.layer1", f"{module}.layer1")
        baked(f"{module}.layer2", f"{module}.layer2")
    baked("action_encoder.w1", "action_encoder.W1")
    baked("action_encoder.w3", "action_encoder.W3")
    width = geometry.inner_dim
    stacked = head["action_encoder.W2.W"][embodiment]
    if tuple(stacked.shape) != (2 * width, width):
        raise ValueError(
            f"action_encoder.W2.W has shape {tuple(stacked.shape)}"
            f" per category, expected {(2 * width, width)}")
    # The timestep half of W2 is folded into action_time_bias by the schedule.
    state["action_encoder.w2_action.weight"] = stacked[:width].T.contiguous()

    for key, tensor in head.items():
        if key.startswith("model.transformer_blocks."):
            _, _, index, module, *rest = key.split(".")
            block = model.transformer_blocks[int(index)]
            # The cross blocks' to_k/to_v and every norm1.linear live elsewhere:
            # the K/V projections in the backbone graph, the modulation in the
            # schedule table.
            if module == "norm1" or (block.is_cross and module == "attn1"
                                     and rest[0] in ("to_k", "to_v")):
                continue
            state[key[len("model."):]] = tensor
        elif key == "model.proj_out_2.weight" or key == "model.proj_out_2.bias":
            state[key[len("model."):]] = tensor
        elif key == "position_embedding.weight" and model.position_embedding is not None:
            state["position_embedding"] = tensor[:geometry.action_horizon]

    state = {
        k: (v.to(dtype) if v.is_floating_point() else v)
        for k, v in state.items()
    }
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise KeyError("GR00T action parameters received no checkpoint "
                       "tensor: " + ", ".join(incompatible.missing_keys[:8]))
    if incompatible.unexpected_keys:
        logger.warning("Unexpected action checkpoint keys (first 5): %s",
                       incompatible.unexpected_keys[:5])


def build_gr00t_action(geometry: Geometry, head: Dict[str, torch.Tensor],
                       dtype: torch.dtype) -> Gr00tActionModel:
    model = Gr00tActionModel(geometry).to(dtype)
    load_weights(model, head, dtype)
    model.eval()
    return model
