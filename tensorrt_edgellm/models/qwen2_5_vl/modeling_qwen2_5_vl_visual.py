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
From-scratch Qwen2.5-VL vision encoder implementation.

Checkpoint weight key prefixes:
    Qwen2.5-VL / Qwen2-VL:  ``visual.*``

Architecture:
    PatchEmbed → rotary positional embeddings → VisionBlocks (with window attention)
    → PatchMerger

Differences from Qwen3-VL:
    - SwiGLU MLP (gate_proj/up_proj/down_proj), RMSNorm, window attention,
      ln_q/mlp merger (instead of Qwen3-VL's GELU FFN / LayerNorm / fc1-fc2 merger)

Shared utilities (_rotate_half, apply_rotary_pos_emb_vision, RMSNorm) are
imported from the ``qwen3_vl`` module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import config as config_module
from ..linear import make_linear
from ..ops import (init_fp8_mha, is_trt_native_attention_enabled,
                   quantize_qkv_for_fp8_mha, trt_ragged_attention,
                   vit_attention_plugin)
from ..qwen3_vl.modeling_qwen3_vl_visual import (RMSNorm, _load_weights,
                                                 apply_rotary_pos_emb_vision)

if TYPE_CHECKING:
    from ...config import ModelConfig

# ---------------------------------------------------------------------------
# Qwen2.5-VL Visual Model
# ---------------------------------------------------------------------------


class Qwen2_5VLPatchEmbed(nn.Module):
    """Conv3d patch embedding for Qwen2.5-VL (no bias)."""

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int,
                 temporal_patch_size: int) -> None:
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
                              bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(-1, self.in_channels,
                                           self.temporal_patch_size,
                                           self.patch_size, self.patch_size)
        return self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.hidden_size)


class Qwen2_5VLMLP(nn.Module):
    """SwiGLU MLP for Qwen2.5-VL vision blocks (with bias).

    Checkpoint keys: ``visual.blocks.N.mlp.gate_proj.*``,
                     ``up_proj.*``, ``down_proj.*``
    """

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 model_config: "ModelConfig",
                 name_prefix: str = "") -> None:
        super().__init__()
        self.gate_proj = make_linear(
            model_config,
            hidden_size,
            intermediate_size,
            bias=True,
            module_name=f"{name_prefix}.gate_proj" if name_prefix else "")
        self.up_proj = make_linear(
            model_config,
            hidden_size,
            intermediate_size,
            bias=True,
            module_name=f"{name_prefix}.up_proj" if name_prefix else "")
        self.down_proj = make_linear(
            model_config,
            intermediate_size,
            hidden_size,
            bias=True,
            module_name=f"{name_prefix}.down_proj" if name_prefix else "")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2_5VLVisionAttention(nn.Module):
    """Qwen2.5-VL vision attention with vit_attention_plugin.

    Window blocks and full-attention blocks share the same attention module;
    the distinction is in which cu_seqlens / max_seqlen_carrier is passed.
    """

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


class Qwen2_5VLVisionBlock(nn.Module):
    """Single Qwen2.5-VL vision block.

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
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Qwen2_5VLVisionAttention(
            hidden_size,
            num_heads,
            attention_scale,
            model_config,
            name_prefix=f"{name_prefix}.attn" if name_prefix else "")
        self.mlp = Qwen2_5VLMLP(
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
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states),
                                                  cu_seqlens,
                                                  max_seqlen_carrier,
                                                  position_embeddings,
                                                  kv_lengths=kv_lengths)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5VLPatchMerger(nn.Module):
    """Qwen2.5-VL patch merger: RMSNorm → spatial-merge → Sequential(Linear, GELU, Linear).

    Checkpoint keys: ``visual.merger.ln_q.*``,
                     ``merger.mlp.0.*``, ``merger.mlp.2.*``
    """

    def __init__(self,
                 hidden_size: int,
                 out_hidden_size: int,
                 spatial_merge_size: int,
                 model_config: "ModelConfig",
                 name_prefix: str = "") -> None:
        super().__init__()
        self.merge_unit = spatial_merge_size * spatial_merge_size
        self.merged_size = hidden_size * self.merge_unit
        self.ln_q = RMSNorm(hidden_size, eps=1e-6)
        # nn.Sequential indices 0 and 2 match checkpoint keys mlp.0.*, mlp.2.*
        self.mlp = nn.Sequential(
            make_linear(
                model_config,
                self.merged_size,
                self.merged_size,
                bias=True,
                module_name=f"{name_prefix}.mlp.0" if name_prefix else ""),
            nn.GELU(),
            make_linear(
                model_config,
                self.merged_size,
                out_hidden_size,
                bias=True,
                module_name=f"{name_prefix}.mlp.2" if name_prefix else ""),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln_q(x).view(-1, self.merged_size))


class Qwen2_5VLVisualModel(nn.Module):
    """From-scratch Qwen2.5-VL vision encoder.

    Supports qwen2_5_vl and qwen2_vl model_type values.

    Qwen2-VL is identical to Qwen2.5-VL except it has no window attention
    (all blocks use full attention). Window blocks are identified by
    ``fullatt_block_indexes`` in the config; absent → all blocks full attention.

    All parameters are loaded from the checkpoint under the ``visual.*`` prefix.
    """

    def __init__(self, config: dict, model_config: "ModelConfig") -> None:
        super().__init__()
        self.hidden_size: int = config["hidden_size"]
        # AWQ/quantized Qwen2.5-VL checkpoints re-serialize the vision config
        # via a newer transformers, which names this field `num_attention_heads`
        # while the stock release uses `num_heads`.
        self.num_heads: int = config.get("num_heads",
                                         config.get("num_attention_heads"))
        self.head_dim = self.hidden_size // self.num_heads
        self.in_channels: int = config.get("in_chans",
                                           config.get("in_channels", 3))
        self.patch_size: int = config["patch_size"]
        self.temporal_patch_size: int = config.get("temporal_patch_size", 2)
        self.spatial_merge_size: int = config.get("spatial_merge_size", 2)
        self.out_hidden_size: int = config["out_hidden_size"]
        depth: int = config["depth"]
        intermediate_size: int = config["intermediate_size"]
        window_size: int = config.get("window_size", 0)
        self.fullatt_block_indexes: List[int] = list(
            config.get("fullatt_block_indexes", list(range(depth))))
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        if window_size > 0:
            self.window_max_seqlen = (window_size // self.patch_size)**2
        else:
            self.window_max_seqlen = 0

        self.patch_embed = Qwen2_5VLPatchEmbed(self.in_channels,
                                               self.hidden_size,
                                               self.patch_size,
                                               self.temporal_patch_size)
        attention_scale = config_module._get_attention_scaling(
            config, self.head_dim, 1.0 / (float(self.head_dim)**0.5))
        self.blocks = nn.ModuleList([
            Qwen2_5VLVisionBlock(self.hidden_size,
                                 intermediate_size,
                                 self.num_heads,
                                 attention_scale,
                                 model_config,
                                 name_prefix=f"visual.blocks.{i}")
            for i in range(depth)
        ])
        self.merger = Qwen2_5VLPatchMerger(self.hidden_size,
                                           self.out_hidden_size,
                                           self.spatial_merge_size,
                                           model_config,
                                           name_prefix="visual.merger")

        # Rotary embedding
        theta = float(config.get("rope_theta", 10000.0))
        rotary_dim = self.head_dim // 2
        inv_freq = 1.0 / (theta**(
            torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        self.register_buffer("_rotary_inv_freq", inv_freq, persistent=False)
        self._use_trt_attn = is_trt_native_attention_enabled()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_window_seqlens: torch.Tensor,
        window_index: torch.Tensor,
        reverse_window_index: torch.Tensor,
        max_seqlen_carrier: Optional[torch.Tensor] = None,
        kv_lengths: Optional[torch.Tensor] = None,
        kv_lengths_window: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)

        seq_len = hidden_states.size(0)
        # Reorder tokens into window-first order.
        # window_index has num_groups = seq_len // merge_unit elements.
        # Use reshape(-1, merge_unit, H) to infer num_groups without //,
        # avoiding unprovable floor-division symbolic constraints when
        # T = merge_unit * _T_base and window_index has _T_base elements.
        hidden_states = hidden_states.reshape(
            -1, self.spatial_merge_unit,
            self.hidden_size)[window_index, :, :].reshape(
                seq_len, self.hidden_size)
        rotary_pos_emb_dim = rotary_pos_emb.shape[-1]
        rotary_pos_emb = rotary_pos_emb.reshape(
            -1, self.spatial_merge_unit,
            rotary_pos_emb_dim)[window_index, :, :].reshape(
                seq_len, rotary_pos_emb_dim)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_now = cu_seqlens
                max_now = max_seqlen_carrier
                kvl_now = kv_lengths
            else:
                cu_now = cu_window_seqlens
                max_now = torch.zeros(self.window_max_seqlen,
                                      dtype=torch.int32,
                                      device=hidden_states.device)
                kvl_now = kv_lengths_window
            hidden_states = blk(hidden_states,
                                cu_now,
                                max_now,
                                position_embeddings,
                                kv_lengths=kvl_now)

        hidden_states = self.merger(hidden_states)
        # Undo window ordering
        hidden_states = hidden_states[reverse_window_index, :]
        return hidden_states

    def get_onnx_export_args(self, config: dict, device: str):
        """Return (dynamo_inputs, onnx_input_names, output_names, dynamic_shapes) for ONNX export."""
        patch_size = config["patch_size"]
        temporal = config.get("temporal_patch_size", 2)
        # num_patches must be divisible by spatial_merge_unit (= merge_size^2).
        # Use 196 (= 14*14) which is divisible by merge_unit=4.
        num_patches = 196
        num_groups = num_patches // self.spatial_merge_unit  # 49

        in_channels = config.get("in_channels", config.get(
            "in_chans", 3)) * temporal * patch_size * patch_size
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

        window_size = config.get("window_size", 0)
        if window_size > 0:
            window_patch_side = window_size // patch_size
            window_patches = window_patch_side * window_patch_side
        else:
            window_patches = num_patches
        cu_window_seqlens = torch.tensor([0, window_patches],
                                         dtype=torch.int32,
                                         device=device)

        # window_index and reverse_window_index select groups (not tokens):
        # size = num_patches // spatial_merge_unit.
        # Must be INT64 for ONNX dynamo export (aten.index requires INT64 indices).
        window_index = torch.arange(num_groups,
                                    dtype=torch.int64,
                                    device=device)
        reverse_window_index = torch.arange(num_groups,
                                            dtype=torch.int64,
                                            device=device)

        onnx_input_names = [
            "input", "rotary_pos_emb", "cu_seqlens", "cu_window_seqlens",
            "window_index", "reverse_window_index"
        ]
        dynamo_inputs = {
            "hidden_states": pixel_values,
            "rotary_pos_emb": rotary_pos_emb,
            "cu_seqlens": cu_seqlens,
            "cu_window_seqlens": cu_window_seqlens,
            "window_index": window_index,
            "reverse_window_index": reverse_window_index,
        }

        output_names = ["output"]
        # total_tokens (T) must be divisible by spatial_merge_unit (= 4).
        # Express T = merge_unit * G where G = num_groups, to avoid unprovable
        # symbolic floor-division guards like Eq((A*G)//(B*G), A//B).
        # With T = 4*G, the view(-1, 5120) constraint becomes (5120*G)//5120 = G
        # which sympy CAN prove (division by a constant).
        _G = torch.export.Dim("num_groups", min=1)
        T = self.spatial_merge_unit * _G

        # Keys must match the forward() parameter names (not ONNX input_names).
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
            "cu_window_seqlens": {
                0: torch.export.Dim("num_windows_p1")
            },
            "window_index": {
                0: _G
            },
            "reverse_window_index": {
                0: _G
            },
        }

        if self._use_trt_attn:
            onnx_input_names.extend([
                "kv_lengths",
                "kv_lengths_window",
            ])
            kv_lengths = torch.tensor([0, num_patches],
                                      dtype=torch.int32,
                                      device=device)
            kv_lengths_window = torch.tensor([0, window_patches],
                                             dtype=torch.int32,
                                             device=device)
            dynamo_inputs["kv_lengths"] = kv_lengths
            dynamo_inputs["kv_lengths_window"] = kv_lengths_window
            dynamic_shapes["kv_lengths"] = {0: torch.export.Dim("kv_batch_p1")}
            dynamic_shapes["kv_lengths_window"] = {
                0: torch.export.Dim("kv_window_batch_p1")
            }
        else:
            onnx_input_names.extend(["max_seqlen_carrier"])
            max_seqlen_carrier = torch.zeros(num_patches,
                                             dtype=torch.int32,
                                             device=device)
            dynamo_inputs["max_seqlen_carrier"] = max_seqlen_carrier
            # max_seqlen_carrier must be INDEPENDENT from T.
            # The C++ builder profiles kMaxSeqLenCarrier as {min=1, opt=maxSeqLen/2, max=maxSeqLen}
            # independently from the total token count.  Linking it to T via {0: T} would cause
            # a shape profile inconsistency ("expect min <= common <= max on input 3").
            _max_seqlen = torch.export.Dim("max_seqlen", min=1)
            dynamic_shapes["max_seqlen_carrier"] = {0: _max_seqlen}

        return dynamo_inputs, onnx_input_names, output_names, dynamic_shapes


def build_qwen25_vl_visual(
        config: dict,
        weights: dict,
        model_config: "ModelConfig",
        dtype: torch.dtype = torch.float16) -> Qwen2_5VLVisualModel:
    """Instantiate and load weights for a Qwen2.5-VL (or Qwen2-VL) visual encoder.

    Args:
        config:       Parsed ``vision_config`` dict from ``config.json``.
        weights:      Safetensors weight dict (all model weights).
        model_config: Top-level ``ModelConfig`` for quantized Linear dispatch.
        dtype:        Target dtype (default float16).
    """
    model = Qwen2_5VLVisualModel(config, model_config=model_config)
    model.to(dtype)
    # Raw HF checkpoints store visual weights under ``visual.*``, while modelopt
    # quantization checkpoints nest them under ``model.visual.*`` (the whole
    # model is wrapped under ``model.`` on re-save). Pick the prefix that the
    # checkpoint actually uses so both layouts load correctly.
    prefix = "model.visual" if any(
        k.startswith("model.visual.") for k in weights) else "visual"
    _load_weights(model, weights, prefix=prefix)
    model.eval()
    return model


__all__ = [
    "Qwen2_5VLPatchEmbed",
    "Qwen2_5VLMLP",
    "Qwen2_5VLVisionAttention",
    "Qwen2_5VLVisionBlock",
    "Qwen2_5VLPatchMerger",
    "Qwen2_5VLVisualModel",
    "build_qwen25_vl_visual",
]
