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
"""GR00T N1.7 vision-language backbone for ONNX export: the truncated Cosmos-Reason2-2B
decoder, VL LayerNorm and VL self-attention refinement, then the DiT cross-attention K/V
projections. Engine I/O matches the ONNX-less ``Gr00tVLBackbone``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import BLOCK_NORM_EPS, Geometry

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """Qwen3 RMSNorm: reduction in fp32, weight applied after the cast back."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        normed = x32 / torch.sqrt(
            torch.mean(x32 * x32, dim=-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight


class Gr00tDecoderAttention(nn.Module):
    """Causal Qwen3-VL self-attention with per-head q/k RMSNorm."""

    def __init__(self, text: dict) -> None:
        super().__init__()
        hidden = int(text["hidden_size"])
        self.num_heads = int(text["num_attention_heads"])
        self.num_kv_heads = int(text["num_key_value_heads"])
        self.head_dim = int(text["head_dim"])
        eps = float(text.get("rms_norm_eps", 1e-6))
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim,
                                False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim,
                                False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, False)
        self.q_norm = RMSNorm(self.head_dim, eps)
        self.k_norm = RMSNorm(self.head_dim, eps)

    def forward(self, hidden_states: torch.Tensor, rope_cos: torch.Tensor,
                rope_sin: torch.Tensor,
                position_ids: torch.Tensor) -> torch.Tensor:
        from ..ops import attention_onnx, rope_onnx

        bsz, seq_len, _ = hidden_states.shape
        query = self.q_norm(
            self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads,
                                            self.head_dim))
        key = self.k_norm(
            self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads,
                                            self.head_dim))
        value = self.v_proj(hidden_states).view(bsz, seq_len,
                                                self.num_kv_heads,
                                                self.head_dim)
        repeats = self.num_heads // self.num_kv_heads
        key = key.repeat_interleave(repeats, dim=2)
        value = value.repeat_interleave(repeats, dim=2)

        query = rope_onnx(query.transpose(1, 2), rope_cos, rope_sin,
                          position_ids)
        key = rope_onnx(key.transpose(1, 2), rope_cos, rope_sin, position_ids)
        attended = attention_onnx(query,
                                  key,
                                  value.transpose(1, 2),
                                  attn_mask=None,
                                  is_causal=True,
                                  scale=self.head_dim**-0.5)
        attended = attended.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(attended)


class Gr00tGatedMLP(nn.Module):

    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, False)
        self.up_proj = nn.Linear(hidden, intermediate, False)
        self.down_proj = nn.Linear(intermediate, hidden, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Gr00tDecoderLayer(nn.Module):

    def __init__(self, text: dict) -> None:
        super().__init__()
        hidden = int(text["hidden_size"])
        eps = float(text.get("rms_norm_eps", 1e-6))
        self.input_layernorm = RMSNorm(hidden, eps)
        self.self_attn = Gr00tDecoderAttention(text)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.mlp = Gr00tGatedMLP(hidden, int(text["intermediate_size"]))

    def forward(self, hidden_states: torch.Tensor, rope_cos: torch.Tensor,
                rope_sin: torch.Tensor, position_ids: torch.Tensor,
                deepstack: Optional[torch.Tensor]) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), rope_cos, rope_sin,
            position_ids)
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states))
        if deepstack is not None:
            hidden_states = hidden_states + deepstack
        return hidden_states


class Gr00tAttentionFeedForward(nn.Module):
    """diffusers ``FeedForward`` with ``gelu-approximate``."""

    def __init__(self, dim: int, inner: int) -> None:
        super().__init__()
        self.net = nn.ModuleList([
            nn.Module(),
            nn.Identity(),
            nn.Linear(inner, dim),
        ])
        self.net[0].proj = nn.Linear(dim, inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net[2](F.gelu(self.net[0].proj(x), approximate="tanh"))


class Gr00tSelfAttention(nn.Module):
    """diffusers ``Attention`` over one sequence: q/k/v/out with biases."""

    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(dim, heads * head_dim)
        self.to_k = nn.Linear(dim, heads * head_dim)
        self.to_v = nn.Linear(dim, heads * head_dim)
        self.to_out = nn.ModuleList([nn.Linear(heads * head_dim, dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from ..ops import attention_onnx

        bsz, seq_len, _ = x.shape
        shape = (bsz, seq_len, self.heads, self.head_dim)
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
                                                 2).reshape(bsz, seq_len, -1))


class Gr00tVlSelfAttentionBlock(nn.Module):
    """Non-causal refinement block: diffusers ``BasicTransformerBlock`` with
    layer norms and ``gelu-approximate``."""

    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=BLOCK_NORM_EPS)
        self.attn1 = Gr00tSelfAttention(dim, heads, head_dim)
        self.norm3 = nn.LayerNorm(dim, eps=BLOCK_NORM_EPS)
        self.ff = Gr00tAttentionFeedForward(dim, 4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn1(self.norm1(x))
        return x + self.ff(self.norm3(x))


class _Gr00tLanguageModel(nn.Module):
    """Container matching the checkpoint's ``model.layers`` / ``model.norm`` nesting."""

    def __init__(self, text: dict) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            Gr00tDecoderLayer(text)
            for _ in range(int(text["num_hidden_layers"]))
        ])
        self.norm = RMSNorm(int(text["hidden_size"]),
                            float(text.get("rms_norm_eps", 1e-6)))


class _Gr00tVlSelfAttention(nn.Module):

    def __init__(self, geometry: Geometry) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([
            Gr00tVlSelfAttentionBlock(geometry.backbone_dim,
                                      geometry.vl_sa_heads,
                                      geometry.vl_sa_head_dim)
            for _ in range(geometry.vl_sa_layers)
        ])


class Gr00tVLBackbone(nn.Module):
    """Prefill VL tower emitting the DiT's step-invariant cross-attention K/V."""

    def __init__(self, text: dict, geometry: Geometry,
                 num_deepstack: int) -> None:
        super().__init__()
        self.text = dict(text)
        self.geometry = geometry
        self.num_deepstack = num_deepstack
        self.head_dim = int(text["head_dim"])
        self.model = _Gr00tLanguageModel(text)
        self.vlln = nn.LayerNorm(geometry.backbone_dim, eps=1e-5)
        self.vl_self_attention = _Gr00tVlSelfAttention(geometry)
        # Only the even (cross-attention) blocks' K/V projections live here.
        self.cross_k = nn.ModuleDict({
            str(index):
            nn.Linear(geometry.backbone_dim, geometry.inner_dim)
            for index in geometry.cross_blocks
        })
        self.cross_v = nn.ModuleDict({
            str(index):
            nn.Linear(geometry.backbone_dim, geometry.inner_dim)
            for index in geometry.cross_blocks
        })

    def forward(self, inputs_embeds: torch.Tensor,
                rope_rotary_cos_sin: torch.Tensor,
                attention_pos_id: torch.Tensor, *deepstack_embeds:
                torch.Tensor) -> Tuple[torch.Tensor, ...]:
        half = self.head_dim // 2
        rope_cos = rope_rotary_cos_sin[..., :half].reshape(-1, half).to(
            torch.float16)
        rope_sin = rope_rotary_cos_sin[...,
                                       half:].reshape(-1,
                                                      half).to(torch.float16)

        hidden = inputs_embeds
        for index, layer in enumerate(self.model.layers):
            injected = deepstack_embeds[index] if index < len(
                deepstack_embeds) else None
            hidden = layer(hidden, rope_cos, rope_sin, attention_pos_id,
                           injected)

        features = self.vlln(self.model.norm(hidden))
        for block in self.vl_self_attention.transformer_blocks:
            features = block(features)

        bsz, seq_len, _ = features.shape
        heads, head_dim = self.geometry.num_heads, self.geometry.head_dim
        outputs: List[torch.Tensor] = []
        for index in self.geometry.cross_blocks:
            key = str(index)
            outputs.append(self.cross_k[key](features).view(
                bsz, seq_len, heads, head_dim))
            outputs.append(self.cross_v[key](features).view(
                bsz, seq_len, heads, head_dim))
        return tuple(outputs)

    def output_names(self) -> List[str]:
        names = []
        for index in self.geometry.cross_blocks:
            names.append(f"cross_k_layer{index:02d}")
            names.append(f"cross_v_layer{index:02d}")
        return names

    def get_onnx_export_args(self,
                             device: str) -> Tuple[tuple, list, list, tuple]:
        hidden = int(self.text["hidden_size"])
        # Batch 2 keeps the batch axis symbolic (torch.export specializes size-1).
        b, s = 2, 16
        embeds = torch.zeros(b, s, hidden, device=device, dtype=torch.float16)
        rope = torch.zeros(b,
                           s,
                           self.head_dim,
                           device=device,
                           dtype=torch.float32)
        pos = torch.arange(s, device=device,
                           dtype=torch.int32).unsqueeze(0).expand(b, -1)
        deepstack = [
            torch.zeros(b, s, hidden, device=device, dtype=torch.float16)
            for _ in range(self.num_deepstack)
        ]
        args = (embeds, rope, pos, *deepstack)
        input_names = [
            "inputs_embeds", "rope_rotary_cos_sin", "attention_pos_id"
        ] + [f"deepstack_embeds_{i}" for i in range(self.num_deepstack)]
        batch = torch.export.Dim("batch_size", min=1, max=256)
        vl_len = torch.export.Dim("vl_len", min=2, max=32768)
        per_input = {0: batch, 1: vl_len}
        # torch.export sees the varargs as one element, so their shapes nest.
        dynamic_shapes = (dict(per_input), dict(per_input), dict(per_input),
                          tuple(dict(per_input) for _ in deepstack))
        return args, input_names, self.output_names(), dynamic_shapes


def load_weights(model: Gr00tVLBackbone, backbone: Dict[str, torch.Tensor],
                 head: Dict[str, torch.Tensor], dtype: torch.dtype) -> None:
    """Assign the backbone and the action head's VL-side tensors."""
    state: Dict[str, torch.Tensor] = {}
    for key, tensor in backbone.items():
        state[key] = tensor.to(dtype)
    for key, tensor in head.items():
        if key.startswith("vlln.") or key.startswith("vl_self_attention."):
            state[key] = tensor.to(dtype)
            continue
        if key.startswith("model.transformer_blocks."):
            _, _, index, module, *rest = key.split(".")
            if module == "attn1" and rest[0] in (
                    "to_k", "to_v") and str(index) in model.cross_k:
                target = "cross_k" if rest[0] == "to_k" else "cross_v"
                state[f"{target}.{index}.{rest[-1]}"] = tensor.to(dtype)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise KeyError("GR00T vl_backbone parameters received no checkpoint "
                       "tensor: " + ", ".join(incompatible.missing_keys[:8]))
    if incompatible.unexpected_keys:
        logger.warning("Unexpected vl_backbone checkpoint keys (first 5): %s",
                       incompatible.unexpected_keys[:5])


def build_gr00t_vl_backbone(text: dict, geometry: Geometry, num_deepstack: int,
                            backbone: Dict[str, torch.Tensor],
                            head: Dict[str, torch.Tensor],
                            dtype: torch.dtype) -> Gr00tVLBackbone:
    model = Gr00tVLBackbone(text, geometry, num_deepstack).to(dtype)
    load_weights(model, backbone, head, dtype)
    model.eval()
    return model
