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
"""LFM3-VL vision tower and multimodal projector.

Checkpoint weight key prefixes:
    ``model.vision_tower.*`` and ``model.multi_modal_projector.*``

Architecture:
    Linear patchifier -> RMSNorm -> layers -> projector (LayerNorm, 2x2 merge, MLP)

Each layer is either bidirectional GQA attention (QK RMSNorm, axial 2D RoPE,
attention within one frame via ``cu_seqlens``) or a gated depthwise 3x3
convolution over the frame's patch grid, followed by a SwiGLU feed-forward.

Patches arrive in the Qwen-VL 2x2 merge-block order shared with the runtime
preprocessing. The packed sequence holds frames of different shapes, so the
convolution reads each token's 3x3 neighbourhood through the runtime-built
``conv_neighbor_index`` ``[T, 9]`` table; out-of-frame taps index an appended
zero row, which reproduces the convolution's zero padding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..linear import make_linear
from ..ops import (is_trt_native_attention_enabled, trt_ragged_attention,
                   vit_attention_plugin)
from ..qwen3_vl.modeling_qwen3_vl_visual import (LayerNorm, RMSNorm,
                                                 apply_rotary_pos_emb_vision)

if TYPE_CHECKING:
    from ...config import ModelConfig

__all__ = [
    "Lfm3VlVisualModel",
    "build_lfm3_vl_visual",
    "conv_neighbor_index",
]

_CONV_TAPS = 9


def conv_neighbor_index(frame_shapes: Sequence[Tuple[int, int]],
                        merge_size: int) -> torch.Tensor:
    """Reference ``conv_neighbor_index`` for frames packed in merge-block order.

    Args:
        frame_shapes: ``(grid_h, grid_w)`` of each frame, in sequence order.
        merge_size: Spatial merge size of the merge-block token order.

    Returns:
        ``[T, 9]`` int32. Entry ``3 * (dy + 1) + (dx + 1)`` is the sequence
        index of the patch at ``(y + dy, x + dx)``, or ``T`` when that patch
        lies outside the frame.
    """
    total = sum(h * w for h, w in frame_shapes)
    out = torch.full((total, _CONV_TAPS), total, dtype=torch.int32)
    base = 0
    for grid_h, grid_w in frame_shapes:
        blocks_w = grid_w // merge_size
        y = torch.arange(grid_h)[:, None].expand(grid_h, grid_w)
        x = torch.arange(grid_w)[None, :].expand(grid_h, grid_w)

        def seq_index(yy: torch.Tensor, xx: torch.Tensor) -> torch.Tensor:
            block = (yy // merge_size) * blocks_w + xx // merge_size
            return base + (block * merge_size +
                           yy % merge_size) * merge_size + xx % merge_size

        own = seq_index(y, x).flatten()
        for tap in range(_CONV_TAPS):
            dy, dx = tap // 3 - 1, tap % 3 - 1
            ny, nx = y + dy, x + dx
            inside = (ny >= 0) & (ny < grid_h) & (nx >= 0) & (nx < grid_w)
            neighbor = seq_index(ny.clamp(0, grid_h - 1),
                                 nx.clamp(0, grid_w - 1))
            out[own, tap] = torch.where(inside, neighbor,
                                        torch.full_like(neighbor,
                                                        total)).flatten().to(
                                                            torch.int32)
        base += grid_h * grid_w
    return out


class Lfm3VlMLP(nn.Module):
    """SwiGLU feed-forward: ``w2(silu(w1(x)) * w3(x))``."""

    def __init__(self, hidden_size: int, intermediate_size: int,
                 model_config: "ModelConfig", prefix: str) -> None:
        super().__init__()
        self.w1 = make_linear(model_config,
                              hidden_size,
                              intermediate_size,
                              module_name=f"{prefix}.w1")
        self.w3 = make_linear(model_config,
                              hidden_size,
                              intermediate_size,
                              module_name=f"{prefix}.w3")
        self.w2 = make_linear(model_config,
                              intermediate_size,
                              hidden_size,
                              module_name=f"{prefix}.w2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Lfm3VlVisionAttention(nn.Module):
    """Bidirectional GQA over one frame, with RMSNorm on Q and K heads."""

    def __init__(self, config: dict, model_config: "ModelConfig",
                 prefix: str) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = hidden_size // self.num_heads
        self.attention_scale = self.head_dim**-0.5
        kv_size = self.num_kv_heads * self.head_dim
        self.q_proj = make_linear(model_config,
                                  hidden_size,
                                  hidden_size,
                                  module_name=f"{prefix}.q_proj")
        self.k_proj = make_linear(model_config,
                                  hidden_size,
                                  kv_size,
                                  module_name=f"{prefix}.k_proj")
        self.v_proj = make_linear(model_config,
                                  hidden_size,
                                  kv_size,
                                  module_name=f"{prefix}.v_proj")
        self.out_proj = make_linear(model_config,
                                    hidden_size,
                                    hidden_size,
                                    module_name=f"{prefix}.out_proj")
        self.q_layernorm = RMSNorm(self.head_dim, eps=config["norm_eps"])
        self.k_layernorm = RMSNorm(self.head_dim, eps=config["norm_eps"])
        self._use_trt_attn = is_trt_native_attention_enabled()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        kv_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q = self.q_layernorm(
            self.q_proj(hidden_states).reshape(seq_length, self.num_heads,
                                               self.head_dim))
        k = self.k_layernorm(
            self.k_proj(hidden_states).reshape(seq_length, self.num_kv_heads,
                                               self.head_dim))
        v = self.v_proj(hidden_states).reshape(seq_length, self.num_kv_heads,
                                               self.head_dim)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        # The ViT attention kernels take one head count for Q, K and V.
        groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        q, k, v = (t.to(torch.float16) for t in (q, k, v))
        if self._use_trt_attn:
            attn_output = trt_ragged_attention(
                q,
                k,
                v,
                cu_seqlens,
                kv_lengths,
                num_heads=self.num_heads,
                head_size=self.head_dim,
                attention_scale=self.attention_scale)
        else:
            attn_output = vit_attention_plugin(
                q,
                k,
                v,
                cu_seqlens,
                max_seqlen_carrier,
                num_heads=self.num_heads,
                head_size=self.head_dim,
                attention_scale=self.attention_scale,
                qkv_scales=[1.0, 1.0, 1.0],
            )
        return self.out_proj(attn_output.reshape(seq_length, -1))


class Lfm3VlSpatialConvWeight(nn.Module):
    """Depthwise 3x3 kernel ``[hidden, 1, 3, 3]`` (``spatial_conv.weight``)."""

    def __init__(self, hidden_size: int, kernel_size: int) -> None:
        super().__init__()
        self.register_buffer(
            "weight",
            torch.zeros(hidden_size,
                        1,
                        kernel_size,
                        kernel_size,
                        dtype=torch.float16))


class Lfm3VlVisionConv2d(nn.Module):
    """Gated depthwise 3x3 convolution over each frame's patch grid."""

    def __init__(self, config: dict, model_config: "ModelConfig",
                 prefix: str) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        if config.get("conv_kernel_size", 3) != 3:
            raise NotImplementedError(
                "LFM3-VL vision convolution supports conv_kernel_size=3 only.")
        self.hidden_size = hidden_size
        self.in_proj = make_linear(model_config,
                                   hidden_size,
                                   3 * hidden_size,
                                   module_name=f"{prefix}.in_proj")
        self.spatial_conv = Lfm3VlSpatialConvWeight(hidden_size, 3)
        self.out_proj = make_linear(model_config,
                                    hidden_size,
                                    hidden_size,
                                    module_name=f"{prefix}.out_proj")

    def forward(self, hidden_states: torch.Tensor,
                neighbor_index: torch.Tensor) -> torch.Tensor:
        b_gate, c_gate, x = self.in_proj(hidden_states).chunk(3, dim=-1)
        gated = b_gate * x
        padded = torch.cat([gated, gated.new_zeros(1, self.hidden_size)])
        taps = self.spatial_conv.weight.reshape(self.hidden_size,
                                                _CONV_TAPS).float()
        conv = torch.zeros_like(gated, dtype=torch.float32)
        for tap in range(_CONV_TAPS):
            neighbors = torch.index_select(padded, 0, neighbor_index[:, tap])
            conv = conv + neighbors.float() * taps[:, tap]
        return self.out_proj(c_gate * conv.to(gated.dtype))


class Lfm3VlVisionLayer(nn.Module):

    def __init__(self, config: dict, layer_idx: int,
                 model_config: "ModelConfig") -> None:
        super().__init__()
        prefix = f"vision_tower.layers.{layer_idx}"
        hidden_size = config["hidden_size"]
        self.is_attention = (
            config["layer_types"][layer_idx] == "full_attention")
        if self.is_attention:
            self.self_attn = Lfm3VlVisionAttention(config, model_config,
                                                   f"{prefix}.self_attn")
        else:
            self.conv = Lfm3VlVisionConv2d(config, model_config,
                                           f"{prefix}.conv")
        self.feed_forward = Lfm3VlMLP(hidden_size, config["intermediate_size"],
                                      model_config, f"{prefix}.feed_forward")
        self.operator_norm = RMSNorm(hidden_size, eps=config["norm_eps"])
        self.ffn_norm = RMSNorm(hidden_size, eps=config["norm_eps"])

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                max_seqlen_carrier: Optional[torch.Tensor],
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                neighbor_index: torch.Tensor,
                kv_lengths: Optional[torch.Tensor]) -> torch.Tensor:
        normed = self.operator_norm(hidden_states)
        if self.is_attention:
            operator_out = self.self_attn(normed,
                                          cu_seqlens,
                                          max_seqlen_carrier,
                                          position_embeddings,
                                          kv_lengths=kv_lengths)
        else:
            operator_out = self.conv(normed, neighbor_index)
        hidden_states = hidden_states + operator_out
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))


class Lfm3VlVisionTower(nn.Module):

    def __init__(self, config: dict, model_config: "ModelConfig") -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        patch_dim = (config.get("in_channels", 3) *
                     config.get("temporal_patch_size", 2) *
                     config["patch_size"]**2)
        self.patchifier = make_linear(model_config,
                                      patch_dim,
                                      hidden_size,
                                      module_name="vision_tower.patchifier")
        self.embedding_norm = RMSNorm(hidden_size, eps=config["norm_eps"])
        self.layers = nn.ModuleList([
            Lfm3VlVisionLayer(config, i, model_config)
            for i in range(config["num_hidden_layers"])
        ])


class Lfm3VlProjector(nn.Module):
    """LayerNorm, merge each 2x2 block into one token, then a GELU MLP."""

    def __init__(self, config: dict, model_config: "ModelConfig") -> None:
        super().__init__()
        vision_hidden = config["vision_config"]["hidden_size"]
        self.merged_size = vision_hidden * config["downsample_factor"]**2
        bias = config.get("projector_bias", True)
        if config.get("projector_hidden_act", "gelu") != "gelu":
            raise NotImplementedError(
                "LFM3-VL projector supports projector_hidden_act='gelu' only.")
        self.layer_norm = LayerNorm(vision_hidden,
                                    eps=config["projector_layer_norm_eps"])
        self.linear_1 = make_linear(
            model_config,
            self.merged_size,
            config["projector_hidden_size"],
            bias=bias,
            module_name="multi_modal_projector.linear_1")
        self.linear_2 = make_linear(
            model_config,
            config["projector_hidden_size"],
            config["text_config"]["hidden_size"],
            bias=bias,
            module_name="multi_modal_projector.linear_2")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        merged = self.layer_norm(hidden_states).reshape(-1, self.merged_size)
        return self.linear_2(F.gelu(self.linear_1(merged)))


class Lfm3VlVisualModel(nn.Module):
    """Vision tower plus projector, exported as one visual engine."""

    def __init__(self, config: dict, model_config: "ModelConfig") -> None:
        super().__init__()
        vision_config = config["vision_config"]
        if vision_config["rope_parameters"].get("rope_type",
                                                "axial") != "axial":
            raise NotImplementedError(
                "LFM3-VL vision RoPE supports rope_type='axial' only.")
        self.head_dim = (vision_config["hidden_size"] //
                         vision_config["num_attention_heads"])
        self.vision_tower = Lfm3VlVisionTower(vision_config, model_config)
        self.multi_modal_projector = Lfm3VlProjector(config, model_config)
        self._use_trt_attn = is_trt_native_attention_enabled()

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        cu_seqlens: torch.Tensor,
        conv_neighbor_index: torch.Tensor,
        max_seqlen_carrier: Optional[torch.Tensor] = None,
        kv_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tower = self.vision_tower
        hidden_states = tower.embedding_norm(tower.patchifier(hidden_states))
        # rotary_pos_emb is [T, head_dim / 2] = concat(freq_h, freq_w); the
        # rotation spans the whole head.
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        for layer in tower.layers:
            hidden_states = layer(hidden_states, cu_seqlens,
                                  max_seqlen_carrier, position_embeddings,
                                  conv_neighbor_index, kv_lengths)
        return self.multi_modal_projector(hidden_states)

    def get_onnx_export_args(self, config: dict, device: str):
        """Return (dynamo_inputs, onnx_input_names, output_names, dynamic_shapes)."""
        vision_config = config["vision_config"]
        patch_size = vision_config["patch_size"]
        in_dim = (vision_config.get("in_channels", 3) *
                  vision_config.get("temporal_patch_size", 2) * patch_size *
                  patch_size)
        frame_shapes = [(16, 16)]
        num_patches = 256
        dynamo_inputs = {
            "hidden_states":
            torch.zeros(num_patches,
                        in_dim,
                        dtype=torch.float16,
                        device=device),
            "rotary_pos_emb":
            torch.zeros(num_patches,
                        self.head_dim // 2,
                        dtype=torch.float32,
                        device=device),
            "cu_seqlens":
            torch.tensor([0, num_patches], dtype=torch.int32, device=device),
            "conv_neighbor_index":
            conv_neighbor_index(
                frame_shapes, vision_config["spatial_merge_size"]).to(device),
        }
        onnx_input_names: List[str] = [
            "input", "rotary_pos_emb", "cu_seqlens", "conv_neighbor_index"
        ]
        total_tokens = torch.export.Dim("total_tokens")
        dynamic_shapes = {
            "hidden_states": {
                0: total_tokens
            },
            "rotary_pos_emb": {
                0: total_tokens
            },
            "cu_seqlens": {
                0: torch.export.Dim("batch_p1")
            },
            "conv_neighbor_index": {
                0: total_tokens
            },
        }
        if self._use_trt_attn:
            onnx_input_names.append("kv_lengths")
            dynamo_inputs["kv_lengths"] = torch.tensor([0, num_patches],
                                                       dtype=torch.int32,
                                                       device=device)
            dynamic_shapes["kv_lengths"] = {0: torch.export.Dim("kv_batch_p1")}
        else:
            onnx_input_names.append("max_seqlen_carrier")
            dynamo_inputs["max_seqlen_carrier"] = torch.zeros(
                num_patches, dtype=torch.int32, device=device)
            # Profiled independently of total_tokens by the C++ builder.
            dynamic_shapes["max_seqlen_carrier"] = {
                0: torch.export.Dim("max_seqlen", min=1)
            }
        return dynamo_inputs, onnx_input_names, ["output"], dynamic_shapes


def build_lfm3_vl_visual(
        config: dict,
        weights: dict,
        model_config: "ModelConfig",
        dtype: torch.dtype = torch.float16) -> Lfm3VlVisualModel:
    """Instantiate and load an LFM3-VL vision tower and projector.

    Args:
        config:       Full ``config.json`` dict (vision, text and projector fields).
        weights:      Safetensors weight dict; ``model.vision_tower.*`` and
                      ``model.multi_modal_projector.*`` are consumed here.
        model_config: Top-level ``ModelConfig`` for ``make_linear`` dispatch.
        dtype:        Target dtype (default float16).
    """
    from ...checkpoint.loader import load_submodule_weights

    model = Lfm3VlVisualModel(config, model_config=model_config)
    model.to(dtype)

    def remap(key: str) -> Optional[str]:
        for component in ("vision_tower.", "multi_modal_projector."):
            if key.startswith("model." + component):
                return key[len("model."):]
        return None

    load_submodule_weights(model, weights, remap, label="lfm3_vl visual")
    model.eval()
    return model
