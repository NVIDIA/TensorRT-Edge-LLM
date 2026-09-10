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
"""
From-scratch Cosmos3-Edge reasoner vision encoder implementation.

Architecture (SigLIP2 naflex packed-patch tower + Qwen3-VL-style PatchMerger):
    Cosmos3ReasonerVisionTransformer
        Cosmos3ReasonerVisionEmbeddings  (Linear patch embed 768→1152 +
                                          interpolated 16×16 position table)
        27 × Cosmos3ReasonerVisionEncoderLayer
            layer_norm1 / self_attn (q/k/v_proj, out_proj) / layer_norm2 /
            mlp (fc1, fc2, gelu_pytorch_tanh)
        post_layernorm
    Cosmos3ReasonerPatchMerger           (LayerNorm(1152) → 2×2 merge reshape
                                          → fc1 4608→11520 → GELU → fc2 →2048)

Checkpoint weight key prefixes:
    ``model.visual.*``    (embeddings, encoder layers, post_layernorm)
    ``model.projector.*`` (norm, linear_fc1, linear_fc2)

Forward I/O (ONNX) — mirrors the Qwen3-VL packed-patch contract (minus
``rotary_pos_emb``; SigLIP2 has no rotary embeddings) so the C++
qwenViTRunner family can drive the engine:
    Inputs:
        input                 [total_patches, 768]   float16
        cu_seqlens            [num_frames + 1]       int32
        fast_pos_embed_idx    [4, total_patches]     int64
        fast_pos_embed_weight [4, total_patches]     float16
        kv_lengths            [num_frames + 1]       int32  (USE_TRT_NATIVE_ATTN=1)
          — or —
        max_seqlen_carrier    [max_seqlen]           int32  (plugin path)
    Output:
        output                [total_patches / 4, 2048]  float16

``input`` carries the packed patches in 2×2 spatial-merge-block order per
frame (block-raster over the frame, ``h1``-major inside each block), with
each patch flattened channel-first by the shared Qwen runtime.  The Cosmos3
processor flattens each patch channel-last, so checkpoint loading permutes
the patch-embedding weight columns to preserve embedding parity.

``fast_pos_embed_idx`` / ``fast_pos_embed_weight`` are the host-precomputed
4-tap bilinear gather (indices into the flattened 16×16 position table and
blend weights) produced by
:meth:`Cosmos3ReasonerVisualModel.fast_pos_embed_interpolate`.  They match
the HF reference ``F.interpolate(..., mode="bilinear", align_corners=False)``
exactly for grids ≥ 16 patches per side (identity / upsampling, where
``antialias=True`` is a no-op); for downsampled grid axes (< 256 px) the
reference applies an antialiased kernel wider than 4 taps, which a 4-tap
gather approximates with plain bilinear weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import config as config_module
from ..linear import FP16Linear, make_linear
from ..ops import (is_trt_native_attention_enabled, trt_ragged_attention,
                   vit_attention_plugin)
from ..qwen3_vl.modeling_qwen3_vl_visual import LayerNorm

if TYPE_CHECKING:
    from ...config import ModelConfig

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _make_visual_linear(model_config: "ModelConfig | None", in_features: int,
                        out_features: int, module_name: str) -> nn.Module:
    """``make_linear`` when a ``ModelConfig`` is available, else plain FP16.

    The standalone :func:`build_cosmos3_reasoner_visual` factory loads the
    encoder straight from a checkpoint directory without a top-level
    ``ModelConfig``; quantized exports pass one through so ``layer_overrides``
    resolve per-module.
    """
    if model_config is None:
        return FP16Linear(in_features, out_features, bias=True)
    return make_linear(model_config,
                       in_features,
                       out_features,
                       bias=True,
                       module_name=module_name)


def _adapt_patch_embedding_weight_for_chw_input(
        weight: torch.Tensor, patch_size: int,
        num_channels: int) -> torch.Tensor:
    """Adapt Cosmos3 patch-embedding columns to the runtime patch layout."""
    expected_width = num_channels * patch_size**2
    if weight.ndim != 2 or weight.shape[1] != expected_width:
        raise ValueError(
            "Cosmos3 patch-embedding weight must have shape "
            f"[out_features, {expected_width}], got {tuple(weight.shape)}")
    return weight.reshape(weight.shape[0],
                          patch_size, patch_size, num_channels).permute(
                              0, 3, 1, 2).reshape_as(weight).contiguous()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class Cosmos3ReasonerVisionEmbeddings(nn.Module):
    """Linear patch embedding + learned position embedding table.

    Checkpoint keys (under ``model.visual.``):
        ``embeddings.patch_embedding.{weight,bias}``  [1152, 768] / [1152]
        ``embeddings.position_embedding.weight``      [256, 1152]

    The 16×16 position table is bilinearly interpolated to each frame's
    patch grid; the interpolation is pre-computed on the host as a 4-tap
    gather (``fast_pos_embed_idx`` / ``fast_pos_embed_weight`` graph inputs).
    """

    def __init__(self,
                 hidden_size: int,
                 num_patches: int,
                 num_channels: int,
                 patch_size: int,
                 model_config: "ModelConfig | None" = None) -> None:
        super().__init__()
        self.patch_embedding = _make_visual_linear(
            model_config, num_channels * patch_size * patch_size, hidden_size,
            "visual.embeddings.patch_embedding")
        self.position_embedding = nn.Embedding(num_patches, hidden_size)

    def forward(self, pixel_values: torch.Tensor,
                fast_pos_embed_idx: torch.Tensor,
                fast_pos_embed_weight: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embedding(pixel_values)
        # 2-D Gather on position_embedding.weight: [4, T] → [4, T, H].
        pos_embeds = self.position_embedding(fast_pos_embed_idx) * \
            fast_pos_embed_weight[:, :, None]   # [4, T, H]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + \
            pos_embeds[2] + pos_embeds[3]       # [T, H]
        return hidden_states + patch_pos_embeds


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class Cosmos3ReasonerVisionAttention(nn.Module):
    """Ragged multi-head self-attention (bidirectional per frame, no RoPE).

    Checkpoint keys (under ``model.visual.encoder.layers.N.self_attn``):
        q_proj.* / k_proj.* / v_proj.* / out_proj.*  (all with bias)
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 attention_scale: float,
                 model_config: "ModelConfig | None" = None,
                 name_prefix: str = "") -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attention_scale = attention_scale
        self.q_proj = _make_visual_linear(
            model_config, hidden_size, hidden_size,
            f"{name_prefix}.q_proj" if name_prefix else "")
        self.k_proj = _make_visual_linear(
            model_config, hidden_size, hidden_size,
            f"{name_prefix}.k_proj" if name_prefix else "")
        self.v_proj = _make_visual_linear(
            model_config, hidden_size, hidden_size,
            f"{name_prefix}.v_proj" if name_prefix else "")
        self.out_proj = _make_visual_linear(
            model_config, hidden_size, hidden_size,
            f"{name_prefix}.out_proj" if name_prefix else "")
        self._use_trt_attn = is_trt_native_attention_enabled()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: Optional[torch.Tensor],
        kv_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(seq_length, self.num_heads,
                                            self.head_dim)
        k = self.k_proj(hidden_states).view(seq_length, self.num_heads,
                                            self.head_dim)
        v = self.v_proj(hidden_states).view(seq_length, self.num_heads,
                                            self.head_dim)
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
                attention_scale=self.attention_scale)
        attn_output = attn_output.reshape(seq_length, -1)
        return self.out_proj(attn_output)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class Cosmos3ReasonerVisionMLP(nn.Module):
    """Two-layer FFN with ``gelu_pytorch_tanh`` activation.

    Checkpoint keys (under ``model.visual.encoder.layers.N.mlp``):
        fc1.* / fc2.*
    """

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 model_config: "ModelConfig | None" = None,
                 name_prefix: str = "") -> None:
        super().__init__()
        self.fc1 = _make_visual_linear(
            model_config, hidden_size, intermediate_size,
            f"{name_prefix}.fc1" if name_prefix else "")
        self.fc2 = _make_visual_linear(
            model_config, intermediate_size, hidden_size,
            f"{name_prefix}.fc2" if name_prefix else "")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


# ---------------------------------------------------------------------------
# Encoder layer
# ---------------------------------------------------------------------------


class Cosmos3ReasonerVisionEncoderLayer(nn.Module):
    """Single SigLIP2 encoder block (pre-norm).

    Checkpoint keys (under ``model.visual.encoder.layers.N``):
        layer_norm1.*, self_attn.*, layer_norm2.*, mlp.*
    """

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 num_heads: int,
                 layer_norm_eps: float,
                 attention_scale: float,
                 model_config: "ModelConfig | None" = None,
                 name_prefix: str = "") -> None:
        super().__init__()
        self.layer_norm1 = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.self_attn = Cosmos3ReasonerVisionAttention(
            hidden_size,
            num_heads,
            attention_scale,
            model_config,
            name_prefix=f"{name_prefix}.self_attn" if name_prefix else "")
        self.layer_norm2 = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = Cosmos3ReasonerVisionMLP(
            hidden_size,
            intermediate_size,
            model_config,
            name_prefix=f"{name_prefix}.mlp" if name_prefix else "")

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: Optional[torch.Tensor],
        kv_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states),
            cu_seqlens,
            max_seqlen_carrier,
            kv_lengths=kv_lengths,
        )
        hidden_states = hidden_states + self.mlp(
            self.layer_norm2(hidden_states))
        return hidden_states


class Cosmos3ReasonerVisionEncoder(nn.Module):
    """Stack of ``num_layers`` encoder layers.

    Checkpoint keys: ``model.visual.encoder.layers.N.*``
    """

    def __init__(self,
                 num_layers: int,
                 hidden_size: int,
                 intermediate_size: int,
                 num_heads: int,
                 layer_norm_eps: float,
                 attention_scale: float,
                 model_config: "ModelConfig | None" = None,
                 name_prefix: str = "") -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            Cosmos3ReasonerVisionEncoderLayer(
                hidden_size,
                intermediate_size,
                num_heads,
                layer_norm_eps,
                attention_scale,
                model_config,
                name_prefix=f"{name_prefix}.layers.{i}" if name_prefix else "")
            for i in range(num_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: Optional[torch.Tensor],
        kv_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states,
                                  cu_seqlens,
                                  max_seqlen_carrier,
                                  kv_lengths=kv_lengths)
        return hidden_states


# ---------------------------------------------------------------------------
# Vision transformer (SigLIP2 tower)
# ---------------------------------------------------------------------------


class Cosmos3ReasonerVisionTransformer(nn.Module):
    """SigLIP2 packed-patch vision transformer.

    Checkpoint keys: ``model.visual.*``
    """

    def __init__(self,
                 hidden_size: int,
                 num_layers: int,
                 num_heads: int,
                 intermediate_size: int,
                 num_patches: int,
                 num_channels: int,
                 patch_size: int,
                 layer_norm_eps: float,
                 attention_scale: float,
                 model_config: "ModelConfig | None" = None,
                 name_prefix: str = "") -> None:
        super().__init__()
        self.embeddings = Cosmos3ReasonerVisionEmbeddings(
            hidden_size, num_patches, num_channels, patch_size, model_config)
        self.encoder = Cosmos3ReasonerVisionEncoder(
            num_layers,
            hidden_size,
            intermediate_size,
            num_heads,
            layer_norm_eps,
            attention_scale,
            model_config,
            name_prefix=f"{name_prefix}.encoder" if name_prefix else "")
        self.post_layernorm = LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        pixel_values: torch.Tensor,
        cu_seqlens: torch.Tensor,
        fast_pos_embed_idx: torch.Tensor,
        fast_pos_embed_weight: torch.Tensor,
        max_seqlen_carrier: Optional[torch.Tensor],
        kv_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values, fast_pos_embed_idx,
                                        fast_pos_embed_weight)
        hidden_states = self.encoder(hidden_states,
                                     cu_seqlens,
                                     max_seqlen_carrier,
                                     kv_lengths=kv_lengths)
        return self.post_layernorm(hidden_states)


# ---------------------------------------------------------------------------
# Patch merger (projector)
# ---------------------------------------------------------------------------


class Cosmos3ReasonerPatchMerger(nn.Module):
    """Qwen3-VL-style patch merger: LayerNorm → 2×2 merge reshape → fc1 →
    GELU → fc2.

    Checkpoint keys (under ``model.projector.``):
        norm.{weight,bias}        [1152]
        linear_fc1.{weight,bias}  [11520, 4608] / [11520]
        linear_fc2.{weight,bias}  [2048, 11520] / [2048]

    Unlike Qwen3-VL's merger (tanh GELU), the HF reference uses ``nn.GELU()``
    with the exact (erf) formulation — keep it byte-faithful.
    """

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 out_hidden_size: int,
                 spatial_merge_size: int,
                 layer_norm_eps: float,
                 model_config: "ModelConfig | None" = None,
                 name_prefix: str = "") -> None:
        super().__init__()
        self.merged_size = hidden_size * spatial_merge_size**2
        self.norm = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.linear_fc1 = _make_visual_linear(
            model_config, self.merged_size, intermediate_size,
            f"{name_prefix}.linear_fc1" if name_prefix else "")
        self.linear_fc2 = _make_visual_linear(
            model_config, intermediate_size, out_hidden_size,
            f"{name_prefix}.linear_fc2" if name_prefix else "")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-shuffle norm (use_postshuffle_norm=False): norm over 1152,
        # then merge 4 consecutive patches (already in merge-block order).
        x = self.norm(x).view(-1, self.merged_size)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


# ---------------------------------------------------------------------------
# Top-level visual model
# ---------------------------------------------------------------------------


class Cosmos3ReasonerVisualModel(nn.Module):
    """Complete Cosmos3-Edge reasoner vision encoder (SigLIP2 tower + merger).

    ``config`` is the full ``config.json`` dict; the vision tower reads
    ``vision_config`` and the merger reads ``projector_config`` (with the
    top-level ``projector_hidden_size`` / text ``hidden_size`` as fallbacks).
    """

    def __init__(self,
                 config: dict,
                 model_config: "ModelConfig | None" = None) -> None:
        super().__init__()
        vision_config: dict = config.get("vision_config", config)
        projector_config: dict = config.get("projector_config", {})

        self.hidden_size: int = vision_config["hidden_size"]
        self.num_heads: int = vision_config["num_attention_heads"]
        self.head_dim: int = self.hidden_size // self.num_heads
        num_layers: int = vision_config["num_hidden_layers"]
        intermediate_size: int = vision_config["intermediate_size"]
        num_channels: int = vision_config.get("num_channels", 3)
        self.patch_size: int = vision_config["patch_size"]
        num_patches: int = vision_config.get("num_patches", 256)
        layer_norm_eps: float = vision_config.get("layer_norm_eps", 1e-6)
        self.spatial_merge_size: int = (
            projector_config.get("spatial_merge_size")
            or vision_config.get("spatial_merge_size", 2))
        merger_intermediate_size: int = (
            projector_config.get("merger_intermediate_size")
            or config["projector_hidden_size"])
        self.out_hidden_size: int = (projector_config.get("out_hidden_size")
                                     or config["text_config"]["hidden_size"])
        self.num_grid_per_side: int = int(num_patches**0.5)
        self.in_features: int = num_channels * self.patch_size**2
        attention_scale = config_module._get_attention_scaling(
            vision_config, self.head_dim, 1.0 / (float(self.head_dim)**0.5))

        # Module-name prefixes match the full HF checkpoint path with the
        # leading ``model.`` stripped, so per-layer quant ``layer_overrides``
        # resolve correctly for MIXED_PRECISION checkpoints.
        self.visual = Cosmos3ReasonerVisionTransformer(
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            num_heads=self.num_heads,
            intermediate_size=intermediate_size,
            num_patches=num_patches,
            num_channels=num_channels,
            patch_size=self.patch_size,
            layer_norm_eps=layer_norm_eps,
            attention_scale=attention_scale,
            model_config=model_config,
            name_prefix="visual")
        self.projector = Cosmos3ReasonerPatchMerger(
            hidden_size=self.hidden_size,
            intermediate_size=merger_intermediate_size,
            out_hidden_size=self.out_hidden_size,
            spatial_merge_size=self.spatial_merge_size,
            layer_norm_eps=layer_norm_eps,
            model_config=model_config,
            name_prefix="projector")
        self._use_trt_attn = is_trt_native_attention_enabled()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def fast_pos_embed_interpolate(
            self, grid_thw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pre-compute position-embedding interpolation indices and weights.

        For each frame the 16×16 position table is resized to the frame's
        ``(h, w)`` patch grid with bilinear interpolation
        (``align_corners=False``, matching the HF reference
        ``F.interpolate``), expressed as a 4-tap gather.  Indices/weights are
        emitted in 2×2 merge-block order — the same permutation the packed
        ``input`` patches use — and repeated ``t`` times per frame.

        Returns:
            ``(idx [4, T] int64, weight [4, T] pos-table dtype)``.
        """
        merge_size = self.spatial_merge_size
        side = self.num_grid_per_side
        idx_list: List[List[int]] = [[], [], [], []]
        weight_list: List[List[float]] = [[], [], [], []]

        for t, h, w in grid_thw.tolist():
            # align_corners=False source coordinates, clamped to the border.
            h_idxs = ((torch.arange(h, dtype=torch.float32) + 0.5) *
                      (side / h) - 0.5).clamp(0, side - 1)
            w_idxs = ((torch.arange(w, dtype=torch.float32) + 0.5) *
                      (side / w) - 0.5).clamp(0, side - 1)
            h_floor = h_idxs.floor()
            w_floor = w_idxs.floor()
            dh = h_idxs - h_floor
            dw = w_idxs - w_floor
            h_lo = h_floor.long()
            w_lo = w_floor.long()
            h_hi = (h_lo + 1).clamp(max=side - 1)
            w_hi = (w_lo + 1).clamp(max=side - 1)

            base_lo = h_lo * side
            base_hi = h_hi * side
            merged_h, merged_w = h // merge_size, w // merge_size

            def _blocks(vec_h: torch.Tensor,
                        vec_w: torch.Tensor) -> torch.Tensor:
                # [h] + [w] → merge-block order (mh, mw, h1, w1), flattened.
                return (vec_h.reshape(merged_h, 1, merge_size, 1) +
                        vec_w.reshape(1, merged_w, 1, merge_size)).flatten()

            indices_all = [
                _blocks(base_lo, w_lo),
                _blocks(base_lo, w_hi),
                _blocks(base_hi, w_lo),
                _blocks(base_hi, w_hi),
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
                frame_idx = indices_all[i].tolist()
                frame_weight = weights_all[i].tolist()
                for _ in range(t):
                    idx_list[i].extend(frame_idx)
                    weight_list[i].extend(frame_weight)

        pos_weight = self.visual.embeddings.position_embedding.weight
        idx_tensor = torch.tensor(idx_list,
                                  dtype=torch.long,
                                  device=pos_weight.device)
        weight_tensor = torch.tensor(weight_list,
                                     dtype=pos_weight.dtype,
                                     device=pos_weight.device)
        return idx_tensor, weight_tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        fast_pos_embed_idx: torch.Tensor,  # [4, T] int64
        fast_pos_embed_weight: torch.Tensor,  # [4, T] float16
        max_seqlen_carrier: Optional[torch.Tensor] = None,
        kv_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.visual(hidden_states,
                                    cu_seqlens,
                                    fast_pos_embed_idx,
                                    fast_pos_embed_weight,
                                    max_seqlen_carrier,
                                    kv_lengths=kv_lengths)
        return self.projector(hidden_states)

    def get_onnx_export_args(self, config: dict, device: str):
        """Return (dynamo_inputs, onnx_input_names, output_names, dynamic_shapes) for ONNX export."""
        num_patches = 256

        pixel_values = torch.zeros(num_patches,
                                   self.in_features,
                                   dtype=torch.float16,
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
            "input", "cu_seqlens", "fast_pos_embed_idx",
            "fast_pos_embed_weight"
        ]
        dynamo_inputs = {
            "hidden_states": pixel_values,
            "cu_seqlens": cu_seqlens,
            "fast_pos_embed_idx": fast_idx,
            "fast_pos_embed_weight": fast_weight,
        }

        output_names = ["output"]
        T = torch.export.Dim("total_tokens")
        dynamic_shapes = {
            "hidden_states": {
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
            # max_seqlen_carrier must use an INDEPENDENT dynamic dim: the C++
            # builder profiles kMaxSeqLenCarrier separately from total_tokens
            # (see modeling_qwen3_vl_visual.get_onnx_export_args).
            _max_seqlen = torch.export.Dim("max_seqlen", min=1)
            dynamic_shapes["max_seqlen_carrier"] = {0: _max_seqlen}
        return dynamo_inputs, onnx_input_names, output_names, dynamic_shapes


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def load_cosmos3_reasoner_visual_checkpoint(
        checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """Read only the vision-tower tensors from the checkpoint shards.

    The Cosmos3-Edge root ``model.safetensors.index.json`` maps the
    ``model.visual.*`` / ``model.projector.*`` tensors into per-component
    shard files (e.g. ``vision_encoder/model.safetensors``); this reads just
    those keys (lazy per-key ``safe_open`` reads) instead of the full
    multi-gigabyte checkpoint.
    """
    from safetensors import safe_open

    from ...checkpoint.loader import _build_shard_map

    prefixes = ("model.visual.", "model.projector.")
    shard_to_keys: Dict[str, List[str]] = {}
    for key, shard_path in _build_shard_map(checkpoint_dir).items():
        if key.startswith(prefixes):
            shard_to_keys.setdefault(shard_path, []).append(key)

    weights: Dict[str, torch.Tensor] = {}
    for shard_path, keys in shard_to_keys.items():
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in keys:
                weights[key] = f.get_tensor(key)
    return weights


def load_cosmos3_reasoner_visual_weights(
        model: Cosmos3ReasonerVisualModel,
        weights: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
    """Load ``model.visual.*`` / ``model.projector.*`` checkpoint weights.

    Selects only the vision-tower tensors from the flat checkpoint dict,
    strips the leading ``model.`` so keys land on this module tree, and
    assigns through the shared ``load_submodule_weights`` pipeline
    (``_set_tensor`` + repacking).

    Returns:
        ``(missing, unexpected)`` — model parameters that received no
        checkpoint tensor, and checkpoint keys that matched no parameter.
        Both are empty for an FP16 load of an intact checkpoint.
    """
    from ...checkpoint.loader import load_submodule_weights

    prefixes = ("model.visual.", "model.projector.")
    strip = "model."

    def _remap(k: str) -> "str | None":
        return k[len(strip):] if k.startswith(prefixes) else None

    def _transform(remapped_key: str, tensor: torch.Tensor) -> torch.Tensor:
        if remapped_key == "visual.embeddings.patch_embedding.weight":
            num_channels = model.in_features // model.patch_size**2
            return _adapt_patch_embedding_weight_for_chw_input(
                tensor, model.patch_size, num_channels)
        return tensor

    load_submodule_weights(model,
                           weights,
                           _remap,
                           transform=_transform,
                           label="Cosmos3ReasonerVisualModel")

    remapped = {k[len(strip):] for k in weights if k.startswith(prefixes)}
    model_keys = set(model.state_dict().keys())
    missing = sorted(model_keys - remapped)
    unexpected = sorted(remapped - model_keys)
    return missing, unexpected


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_cosmos3_reasoner_visual(
    config: dict,
    weights: Dict[str, torch.Tensor],
    model_config: "ModelConfig | None" = None,
    dtype: torch.dtype = torch.float16,
) -> Cosmos3ReasonerVisualModel:
    """Build and return a :class:`Cosmos3ReasonerVisualModel` with loaded weights.

    Matches the ``export_encoder`` visual-family ``build_fn`` contract.

    Args:
        config:       Full checkpoint ``config.json`` dict (``vision_config``
                      + ``projector_config`` + token IDs).
        weights:      Flat ``{key: tensor}`` checkpoint dict.
        model_config: Optional top-level ``ModelConfig`` for quantized
                      Linear dispatch; ``None`` builds plain FP16 linears.
        dtype:        Weight dtype (default ``float16``).
    """
    model = Cosmos3ReasonerVisualModel(config,
                                       model_config=model_config).to(dtype)
    load_cosmos3_reasoner_visual_weights(model, weights)
    model.eval()
    return model


__all__ = [
    "Cosmos3ReasonerVisionEmbeddings",
    "Cosmos3ReasonerVisionAttention",
    "Cosmos3ReasonerVisionMLP",
    "Cosmos3ReasonerVisionEncoderLayer",
    "Cosmos3ReasonerVisionEncoder",
    "Cosmos3ReasonerVisionTransformer",
    "Cosmos3ReasonerPatchMerger",
    "Cosmos3ReasonerVisualModel",
    "load_cosmos3_reasoner_visual_weights",
    "build_cosmos3_reasoner_visual",
]
