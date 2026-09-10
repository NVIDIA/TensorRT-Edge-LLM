# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
From-scratch Qwen3-VL and Qwen3.5-VL vision encoder implementations.

Checkpoint weight key prefixes:
    Qwen3-VL / Qwen3.5-VL:  ``model.visual.*``

Architecture:
    PatchEmbed → positional embeddings → VisionBlocks → PatchMerger

Qwen3-VL uses standard GELU FFN (linear_fc1/fc2), LayerNorm, full attention only,
learned positional embedding, deepstack side outputs, and norm/linear_fc1/linear_fc2
merger. Qwen3.5-VL is the same but without deepstack_visual_indexes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import config as config_module
from ..linear import make_linear
from ..ops import (init_fp8_mha, is_trt_native_attention_enabled,
                   quantize_qkv_for_fp8_mha, trt_ragged_attention,
                   vit_attention_plugin)

if TYPE_CHECKING:
    from ...config import ModelConfig

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K in vision attention.

    cos/sin are FP32 (rotary_pos_emb is a FP32 graph input); q/k are FP16.
    Cast q/k to FP32 for the computation and cast the output back to FP16
    so the vit_attention_plugin receives FP16 inputs.
    """
    orig_dtype = q.dtype
    q = q.float()
    k = k.float()
    cos_e = cos.unsqueeze(-2)
    sin_e = sin.unsqueeze(-2)
    q_embed = (q * cos_e) + (_rotate_half(q) * sin_e)
    k_embed = (k * cos_e) + (_rotate_half(k) * sin_e)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class RMSNorm(nn.Module):
    """RMSNorm for Qwen3-VL."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(input_dtype) * x.to(input_dtype)


class LayerNorm(nn.Module):
    """Manual LayerNorm that decomposes into primitive ONNX ops.

    ``nn.LayerNorm`` exports as ONNX ``LayerNormalization``, which TRT 10.x
    fuses into a ForeignNode together with surrounding ops and then fails to
    find a Myelin implementation.  This class generates the same computation
    using Cast / ReduceMean / Sub / Pow / Rsqrt / Mul / Add so that each op
    can be lowered individually by TRT's ONNX parser.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        mean = x.mean(-1, keepdim=True)
        x = x - mean
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight.float() * x + self.bias.float()).to(input_dtype)


# ---------------------------------------------------------------------------
# Qwen3-VL Visual Model
# ---------------------------------------------------------------------------


class Qwen3VLPatchEmbed(nn.Module):
    """Conv3d patch embedding (temporal_patch_size, patch_size, patch_size)."""

    def __init__(self,
                 in_channels: int,
                 hidden_size: int,
                 patch_size: int,
                 temporal_patch_size: int,
                 bias: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        kernel = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels,
                              hidden_size,
                              kernel_size=kernel,
                              stride=kernel,
                              bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(-1, self.in_channels,
                                           self.temporal_patch_size,
                                           self.patch_size, self.patch_size)
        return self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.hidden_size)


class Qwen3VLMLP(nn.Module):
    """Standard GELU FFN used in Qwen3-VL vision blocks.

    Checkpoint keys: ``visual.blocks.N.mlp.linear_fc1.*``, ``linear_fc2.*``
    """

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 model_config: "ModelConfig",
                 name_prefix: str = "") -> None:
        super().__init__()
        self.linear_fc1 = make_linear(
            model_config,
            hidden_size,
            intermediate_size,
            bias=True,
            module_name=f"{name_prefix}.linear_fc1" if name_prefix else "")
        self.linear_fc2 = make_linear(
            model_config,
            intermediate_size,
            hidden_size,
            bias=True,
            module_name=f"{name_prefix}.linear_fc2" if name_prefix else "")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


class Qwen3VLVisionAttention(nn.Module):
    """Qwen3-VL vision attention replaced with vit_attention_plugin."""

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 attention_scale: float,
                 model_config: "ModelConfig",
                 name_prefix: str = "") -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attention_scale = attention_scale
        self.qkv = make_linear(
            model_config,
            hidden_size,
            hidden_size * 3,
            bias=True,
            module_name=f"{name_prefix}.qkv" if name_prefix else "")
        self.proj = make_linear(
            model_config,
            hidden_size,
            hidden_size,
            bias=True,
            module_name=f"{name_prefix}.proj" if name_prefix else "")
        self._use_trt_attn = is_trt_native_attention_enabled()
        # Fused-qkv attention: init_fp8_mha creates synthetic k_proj/v_proj
        # placeholders for the modelopt scale paths (see its docstring).
        self.enable_fp8_mha = init_fp8_mha(self, model_config, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        kv_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(
            seq_length, 3 * self.num_heads,
            self.head_dim).split(self.num_heads, dim=1)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
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
            q, k, v, qkv_scales = quantize_qkv_for_fp8_mha(self, q, k, v)
            attn_output = vit_attention_plugin(
                q,
                k,
                v,
                cu_seqlens,
                max_seqlen_carrier,
                num_heads=self.num_heads,
                head_size=self.head_dim,
                attention_scale=self.attention_scale,
                qkv_scales=qkv_scales,
            )
        attn_output = attn_output.reshape(seq_length, -1)
        return self.proj(attn_output)


class Qwen3VLVisionBlock(nn.Module):
    """Single Qwen3-VL vision transformer block.

    Checkpoint keys: ``visual.blocks.N.norm1.*``, ``norm2.*``, ``attn.*``, ``mlp.*``
    """

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 num_heads: int,
                 attention_scale: float,
                 model_config: "ModelConfig",
                 name_prefix: str = "") -> None:
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = LayerNorm(hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(
            hidden_size,
            num_heads,
            attention_scale,
            model_config,
            name_prefix=f"{name_prefix}.attn" if name_prefix else "")
        self.mlp = Qwen3VLMLP(
            hidden_size,
            intermediate_size,
            model_config,
            name_prefix=f"{name_prefix}.mlp" if name_prefix else "")

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        kv_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens,
            max_seqlen_carrier,
            position_embeddings,
            kv_lengths=kv_lengths,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLPatchMerger(nn.Module):
    """Qwen3-VL patch merger: LayerNorm → spatial-merge reshape → fc1 → GELU → fc2.

    For the final merger (use_postshuffle_norm=False):
        norm.weight shape: [hidden_size]          (pre-merge)
        linear_fc1.weight shape: [merged, merged]
        linear_fc2.weight shape: [merged, out]

    For deepstack mergers (use_postshuffle_norm=True):
        norm.weight shape: [merged]               (post-merge)
        linear_fc1/fc2: same shapes
    """

    def __init__(self,
                 hidden_size: int,
                 out_hidden_size: int,
                 spatial_merge_size: int,
                 model_config: "ModelConfig",
                 use_postshuffle_norm: bool = False,
                 name_prefix: str = "") -> None:
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.merge_unit = spatial_merge_size * spatial_merge_size
        self.merged_size = hidden_size * self.merge_unit
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.merged_size if use_postshuffle_norm else hidden_size
        self.norm = LayerNorm(norm_dim, eps=1e-6)
        self.linear_fc1 = make_linear(
            model_config,
            self.merged_size,
            self.merged_size,
            bias=True,
            module_name=f"{name_prefix}.linear_fc1" if name_prefix else "")
        self.linear_fc2 = make_linear(
            model_config,
            self.merged_size,
            out_hidden_size,
            bias=True,
            module_name=f"{name_prefix}.linear_fc2" if name_prefix else "")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            # Reshape first, then norm
            x = self.norm(x.view(-1,
                                 self.merged_size)).view(-1, self.merged_size)
        else:
            # Norm first, then reshape
            x = self.norm(x).view(-1, self.merged_size)
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


class Qwen3VLVisualModel(nn.Module):
    """From-scratch Qwen3-VL vision encoder.

    Supports qwen3_vl, qwen3_omni, qwen3_5 model_type values.

    All parameters are loaded from the checkpoint under the ``visual.*`` prefix.
    """

    def __init__(self, config: dict, model_config: "ModelConfig") -> None:
        super().__init__()
        self.hidden_size: int = config["hidden_size"]
        # AWQ/quantized checkpoints may name this `num_attention_heads`.
        self.num_heads: int = config.get("num_heads",
                                         config.get("num_attention_heads"))
        self.head_dim: int = self.hidden_size // self.num_heads
        self.in_channels: int = config.get("in_channels", 3)
        self.patch_size: int = config["patch_size"]
        self.temporal_patch_size: int = config.get("temporal_patch_size", 2)
        self.spatial_merge_size: int = config.get("spatial_merge_size", 2)
        self.out_hidden_size: int = config["out_hidden_size"]
        num_position_embeddings: int = config["num_position_embeddings"]
        depth: int = config["depth"]
        intermediate_size: int = config["intermediate_size"]
        self.deepstack_visual_indexes: List[int] = list(
            config.get("deepstack_visual_indexes", []))
        self.num_grid_per_side: int = int(num_position_embeddings**0.5)
        self.rotary_pos_emb_dim = self.head_dim // 2

        self.patch_embed = Qwen3VLPatchEmbed(self.in_channels,
                                             self.hidden_size,
                                             self.patch_size,
                                             self.temporal_patch_size,
                                             bias=True)
        self.pos_embed = nn.Embedding(num_position_embeddings,
                                      self.hidden_size)
        attention_scale = config_module._get_attention_scaling(
            config, self.head_dim, 1.0 / (float(self.head_dim)**0.5))
        # Module-name prefixes match the keys ``layer_overrides`` carries for
        # MIXED_PRECISION checkpoints (full HF path with the leading ``model.``
        # stripped) so per-layer FP8 / NVFP4 / ... overrides resolve correctly.
        self.blocks = nn.ModuleList([
            Qwen3VLVisionBlock(self.hidden_size,
                               intermediate_size,
                               self.num_heads,
                               attention_scale,
                               model_config,
                               name_prefix=f"visual.blocks.{i}")
            for i in range(depth)
        ])
        self.merger = Qwen3VLPatchMerger(self.hidden_size,
                                         self.out_hidden_size,
                                         self.spatial_merge_size,
                                         use_postshuffle_norm=False,
                                         model_config=model_config,
                                         name_prefix="visual.merger")
        self.deepstack_merger_list = nn.ModuleList([
            Qwen3VLPatchMerger(self.hidden_size,
                               self.out_hidden_size,
                               self.spatial_merge_size,
                               use_postshuffle_norm=True,
                               model_config=model_config,
                               name_prefix=f"visual.deepstack_merger_list.{i}")
            for i in range(len(self.deepstack_visual_indexes))
        ])

        # Rotary embedding (not a stored parameter — computed at runtime)
        theta = float(config.get("rope_theta", 10000.0))
        inv_freq = 1.0 / (theta**(
            torch.arange(0, self.rotary_pos_emb_dim, 2, dtype=torch.float) /
            self.rotary_pos_emb_dim))
        self.register_buffer("_rotary_inv_freq", inv_freq, persistent=False)
        self._use_trt_attn = is_trt_native_attention_enabled()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _compute_rotary_pos_emb(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen,
                           device=self._rotary_inv_freq.device,
                           dtype=self._rotary_inv_freq.dtype)
        freqs = torch.outer(seq, self._rotary_inv_freq)
        return freqs  # [seqlen, rotary_pos_emb_dim // 2]

    def fast_pos_embed_interpolate(
            self, grid_thw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pre-compute positional embedding interpolation indices and weights.

        Matches the optimized implementation from ``Qwen3VLVisionModelPatch``
        in tensorrt_edgellm: permutes before embedding so no post-permute needed.
        """
        merge_size = self.spatial_merge_size
        idx_list: List[List[int]] = [[], [], [], []]
        weight_list: List[List[float]] = [[], [], [], []]

        for t, h, w in grid_thw.tolist():
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_floor + 1).clip(max=self.num_grid_per_side - 1)
            w_ceil = (w_floor + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            base_h = h_floor * self.num_grid_per_side
            base_h_c = h_ceil * self.num_grid_per_side
            merged_h, merged_w = h // merge_size, w // merge_size

            indices_all = [
                (base_h.reshape(merged_h, 1, merge_size, 1) +
                 w_floor.reshape(1, merged_w, 1, merge_size)).flatten(),
                (base_h.reshape(merged_h, 1, merge_size, 1) +
                 w_ceil.reshape(1, merged_w, 1, merge_size)).flatten(),
                (base_h_c.reshape(merged_h, 1, merge_size, 1) +
                 w_floor.reshape(1, merged_w, 1, merge_size)).flatten(),
                (base_h_c.reshape(merged_h, 1, merge_size, 1) +
                 w_ceil.reshape(1, merged_w, 1, merge_size)).flatten(),
            ]
            weights_all = [
                ((1 - dh).reshape(merged_h, 1, merge_size, 1) *
                 (1 - dw).reshape(1, merged_w, 1, merge_size)).flatten(),
                ((1 - dh).reshape(merged_h, 1, merge_size, 1) *
                 dw.reshape(1, merged_w, 1, merge_size)).flatten(),
                (dh.reshape(merged_h, 1, merge_size, 1) *
                 (1 - dw).reshape(1, merged_w, 1, merge_size)).flatten(),
                (dh.reshape(merged_h, 1, merge_size, 1) *
                 dw.reshape(1, merged_w, 1, merge_size)).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices_all[i].tolist())
                weight_list[i].extend(weights_all[i].tolist())

        idx_tensor = torch.tensor(idx_list,
                                  dtype=torch.long,
                                  device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(weight_list,
                                     dtype=self.pos_embed.weight.dtype,
                                     device=self.pos_embed.weight.device)
        return idx_tensor, weight_tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        cu_seqlens: torch.Tensor,
        fast_pos_embed_idx: torch.Tensor,  # [4, T] int64
        fast_pos_embed_weight: torch.Tensor,  # [4, T] float16
        max_seqlen_carrier: Optional[torch.Tensor] = None,
        kv_lengths: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        hidden_states = self.patch_embed(hidden_states)

        # Apply positional embedding via pre-computed bilinear indices/weights.
        # 2-D Gather on pos_embed.weight: [4, T] indices → [4, T, H] embeddings.
        # fast_pos_embed_weight is already FP16 (same dtype as hidden_states).
        pos_embeds = self.pos_embed(fast_pos_embed_idx) * \
            fast_pos_embed_weight[:, :, None]   # [4, T, H]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + \
            pos_embeds[2] + pos_embeds[3]       # [T, H]
        hidden_states = hidden_states + patch_pos_embeds

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        deepstack_features: List[torch.Tensor] = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states,
                                cu_seqlens,
                                max_seqlen_carrier,
                                position_embeddings,
                                kv_lengths=kv_lengths)
            if (self.deepstack_visual_indexes
                    and layer_num in self.deepstack_visual_indexes):
                ds_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_features.append(
                    self.deepstack_merger_list[ds_idx](hidden_states))

        hidden_states = self.merger(hidden_states)

        if not self.deepstack_visual_indexes:
            return hidden_states
        return hidden_states, deepstack_features

    def get_onnx_export_args(self, config: dict, device: str):
        """Return (dynamo_inputs, onnx_input_names, output_names, dynamic_shapes) for ONNX export."""
        patch_size = config["patch_size"]
        temporal = config.get("temporal_patch_size", 2)
        num_patches = 256

        in_channels = config.get("in_channels",
                                 3) * temporal * patch_size * patch_size
        pixel_values = torch.zeros(num_patches,
                                   in_channels,
                                   dtype=torch.float16,
                                   device=device)
        rotary_pos_emb = torch.zeros(num_patches,
                                     self.head_dim // 2,
                                     dtype=torch.float32,
                                     device=device)
        cu_seqlens = torch.tensor([0, num_patches],
                                  dtype=torch.int32,
                                  device=device)
        fast_idx = torch.zeros(4,
                               num_patches,
                               dtype=torch.int64,
                               device=device)
        fast_weight = torch.zeros(4,
                                  num_patches,
                                  dtype=torch.float16,
                                  device=device)

        onnx_input_names = [
            "input", "rotary_pos_emb", "cu_seqlens", "fast_pos_embed_idx",
            "fast_pos_embed_weight"
        ]
        dynamo_inputs = {
            "hidden_states": pixel_values,
            "rotary_pos_emb": rotary_pos_emb,
            "cu_seqlens": cu_seqlens,
            "fast_pos_embed_idx": fast_idx,
            "fast_pos_embed_weight": fast_weight,
        }

        output_names = (["output"] + [
            f"deepstack_features_{i}"
            for i in range(len(self.deepstack_visual_indexes))
        ])
        T = torch.export.Dim("total_tokens")
        dynamic_shapes = {
            "hidden_states": {
                0: T
            },
            "rotary_pos_emb": {
                0: T
            },
            "cu_seqlens": {
                0: torch.export.Dim("batch_p1")
            },
            "fast_pos_embed_idx": {
                1: T
            },
            "fast_pos_embed_weight": {
                1: T
            },
        }

        if self._use_trt_attn:
            onnx_input_names.extend(["kv_lengths"])
            kv_lengths = torch.tensor([0, num_patches],
                                      dtype=torch.int32,
                                      device=device)
            dynamo_inputs["kv_lengths"] = kv_lengths
            dynamic_shapes["kv_lengths"] = {0: torch.export.Dim("kv_batch_p1")}
        else:
            onnx_input_names.extend(["max_seqlen_carrier"])
            max_seqlen_carrier = torch.zeros(num_patches,
                                             dtype=torch.int32,
                                             device=device)
            dynamo_inputs["max_seqlen_carrier"] = max_seqlen_carrier
            # max_seqlen_carrier must use an INDEPENDENT dynamic dim.
            # The C++ builder profiles kMaxSeqLenCarrier independently from T
            # (min=1, opt=maxSeqLen, max=maxSeqLen).  Linking it to T causes a
            # shape-profile conflict ("expect min <= common <= max") when the
            # C++ opt value for max_seqlen_carrier differs from that for total_tokens.
            _max_seqlen = torch.export.Dim("max_seqlen", min=1)
            dynamic_shapes["max_seqlen_carrier"] = {0: _max_seqlen}
        return dynamo_inputs, onnx_input_names, output_names, dynamic_shapes


def build_qwen3_vl_visual(
        config: dict,
        weights: dict,
        model_config: "ModelConfig",
        dtype: torch.dtype = torch.float16) -> Qwen3VLVisualModel:
    """Instantiate and load weights for a Qwen3-VL visual encoder.

    Args:
        config:       Parsed ``vision_config`` dict from ``config.json``.
        weights:      Safetensors weight dict (all model weights; keys with
                      ``model.visual.*`` prefix are consumed here).
        model_config: Top-level ``ModelConfig``.  Visual layers dispatch
                      through ``make_linear`` so quantized checkpoints are
                      honoured; an FP16 checkpoint yields ``FP16Linear``.
        dtype:        Target dtype (default float16).
    """
    model = Qwen3VLVisualModel(config, model_config=model_config)
    model.to(dtype)
    _load_weights(model, weights, prefix="model.visual")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Weight loading helper
# ---------------------------------------------------------------------------


def _load_weights(model: nn.Module, weights: dict, prefix: str = "") -> None:
    """Load weights from a flat safetensors dict into *model*.

    Strips ``<prefix>.`` from each checkpoint key, then defers to
    :func:`tensorrt_edgellm.checkpoint.loader.load_submodule_weights` (which uses
    ``_set_tensor`` so quantized weight dtypes are preserved and
    ``apply_all_repacking`` runs at the end).
    """
    from ...checkpoint.loader import load_submodule_weights

    strip = prefix + "." if prefix else ""

    def _remap(k: str) -> "str | None":
        if strip and not k.startswith(strip):
            return None
        return k[len(strip):] if strip else k

    load_submodule_weights(model, weights, _remap, label=prefix or "model")


__all__ = [
    "Qwen3VLPatchEmbed",
    "Qwen3VLMLP",
    "Qwen3VLVisionAttention",
    "Qwen3VLVisionBlock",
    "Qwen3VLPatchMerger",
    "Qwen3VLVisualModel",
    "build_qwen3_vl_visual",
    "_load_weights",
    "apply_rotary_pos_emb_vision",
    "RMSNorm",
]
