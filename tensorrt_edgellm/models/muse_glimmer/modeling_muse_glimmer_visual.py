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
"""Muse-Glimmer vision encoder for the ONNX frontend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import config as config_module
from ..linear import FP16Linear, make_linear
from ..ops import (is_trt_native_attention_enabled, trt_ragged_attention,
                   vit_attention_plugin)

if TYPE_CHECKING:
    from ...config import ModelConfig


def _make_visual_linear(model_config: "ModelConfig | None",
                        in_features: int,
                        out_features: int,
                        module_name: str,
                        bias: bool = True) -> nn.Module:
    """``make_linear`` when a ``ModelConfig`` is available, else plain FP16.

    The standalone :func:`build_muse_glimmer_visual` factory can load the
    encoder straight from a checkpoint directory without a top-level
    ``ModelConfig``; quantized exports pass one through so ``layer_overrides``
    resolve per-module.
    """
    if model_config is None:
        return FP16Linear(in_features, out_features, bias=bias)
    return make_linear(model_config,
                       in_features,
                       out_features,
                       bias=bias,
                       module_name=module_name)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_dtype = q.dtype
    k_dtype = k.dtype
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q = q.float()
    k = k.float()
    return ((q * cos + _rotate_half(q) * sin).to(q_dtype),
            (k * cos + _rotate_half(k) * sin).to(k_dtype))


class LayerNorm(nn.Module):
    """LayerNorm decomposed into TensorRT-supported primitives."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        x = x - x.mean(-1, keepdim=True)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * x + self.bias.float()).to(input_dtype)


class MuseGlimmerRMSNorm(nn.Module):
    """Scaleless RMSNorm (``with_scale=False``) computed in FP32.

    Used for the ``perception_emb_norm`` applied to the final projected
    vision embeddings.  Has no learnable parameters — matches the reference
    ``MuseGlimmerRMSNorm(with_scale=False)``.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return x.to(input_dtype)


# ---------------------------------------------------------------------------
# Patch embedder
# ---------------------------------------------------------------------------


class MuseGlimmerVisionPatchEmbed(nn.Module):
    """Linear patch embedding + learned, interpolated position table.

    Checkpoint keys (under ``vision_tower.patch_embedder.``):
        ``patch_embedding.weight``            [1536, 1176]  (bias=False)
        ``position_embedding_table.weight``   [1024, 1536]

    The 32x32 position table is bilinearly interpolated to each frame's patch
    grid (``align_corners=False``); the interpolation is pre-computed on the
    host as a 4-tap gather (``fast_pos_embed_idx`` / ``fast_pos_embed_weight``
    graph inputs).
    """

    def __init__(self,
                 in_features: int,
                 hidden_size: int,
                 num_position_embeddings: int,
                 model_config: "ModelConfig | None" = None) -> None:
        super().__init__()
        self.patch_embedding = _make_visual_linear(
            model_config,
            in_features,
            hidden_size,
            "vision_tower.patch_embedder.patch_embedding",
            bias=False)
        self.position_embedding_table = nn.Embedding(num_position_embeddings,
                                                     hidden_size)

    def forward(self, pixel_values: torch.Tensor,
                fast_pos_embed_idx: torch.Tensor,
                fast_pos_embed_weight: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embedding(pixel_values)
        # 2-D gather on the position table: [4, T] -> [4, T, H].
        pos_embeds = self.position_embedding_table(fast_pos_embed_idx) * \
            fast_pos_embed_weight[:, :, None]        # [4, T, H]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + \
            pos_embeds[2] + pos_embeds[3]            # [T, H]
        return hidden_states + patch_pos_embeds


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class MuseGlimmerVisionAttention(nn.Module):
    """Ragged multi-head self-attention with 2D interleaved RoPE.

    Checkpoint keys (under ``vision_tower.layers.N.attn.``):
        q_proj.* / k_proj.* / v_proj.* / proj.*  (all with bias)
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
        self.proj = _make_visual_linear(
            model_config, hidden_size, hidden_size,
            f"{name_prefix}.proj" if name_prefix else "")
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
        q = self.q_proj(hidden_states).view(seq_length, self.num_heads,
                                            self.head_dim)
        k = self.k_proj(hidden_states).view(seq_length, self.num_heads,
                                            self.head_dim)
        v = self.v_proj(hidden_states).view(seq_length, self.num_heads,
                                            self.head_dim)
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
        return self.proj(attn_output)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MuseGlimmerVisionMLP(nn.Module):
    """Two-layer FFN with (erf) GELU activation.

    Checkpoint keys (under ``vision_tower.layers.N.mlp.``):
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
        # hidden_act == "gelu" -> exact (erf) GELU, not the tanh approximation.
        return self.fc2(F.gelu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Encoder layer
# ---------------------------------------------------------------------------


class MuseGlimmerVisionEncoderLayer(nn.Module):
    """Single pre-norm vision transformer block.

    Checkpoint keys (under ``vision_tower.layers.N.``):
        norm1.*, attn.*, norm2.*, mlp.*
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
        self.norm1 = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = MuseGlimmerVisionAttention(
            hidden_size,
            num_heads,
            attention_scale,
            model_config,
            name_prefix=f"{name_prefix}.attn" if name_prefix else "")
        self.mlp = MuseGlimmerVisionMLP(
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


# ---------------------------------------------------------------------------
# Vision tower
# ---------------------------------------------------------------------------


class MuseGlimmerVisionModel(nn.Module):
    """Muse-Glimmer vision tower (patch embed -> encoder -> ln_post -> merge).

    Checkpoint keys: ``vision_tower.*``
    """

    def __init__(self,
                 hidden_size: int,
                 num_layers: int,
                 num_heads: int,
                 intermediate_size: int,
                 in_features: int,
                 num_position_embeddings: int,
                 merge_size: int,
                 layer_norm_eps: float,
                 layer_types: List[str],
                 window_max_seqlen: int,
                 attention_scale: float,
                 model_config: "ModelConfig | None" = None,
                 name_prefix: str = "vision_tower") -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.merge_size = merge_size
        self.merge_unit = merge_size * merge_size
        self.merged_size = hidden_size * self.merge_unit
        self.layer_types = list(layer_types)
        self.window_max_seqlen = window_max_seqlen

        self.patch_embedder = MuseGlimmerVisionPatchEmbed(
            in_features, hidden_size, num_position_embeddings, model_config)
        self.ln_pre = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layers = nn.ModuleList([
            MuseGlimmerVisionEncoderLayer(
                hidden_size,
                intermediate_size,
                num_heads,
                layer_norm_eps,
                attention_scale,
                model_config,
                name_prefix=f"{name_prefix}.layers.{i}")
            for i in range(num_layers)
        ])
        self.ln_post = LayerNorm(hidden_size, eps=layer_norm_eps)
        self._use_trt_attn = is_trt_native_attention_enabled()

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_window_seqlens: torch.Tensor,
        window_index: torch.Tensor,
        reverse_window_index: torch.Tensor,
        fast_pos_embed_idx: torch.Tensor,
        fast_pos_embed_weight: torch.Tensor,
        max_seqlen_carrier: Optional[torch.Tensor] = None,
        kv_lengths: Optional[torch.Tensor] = None,
        kv_lengths_window: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Patch embed + interpolated learned position embedding (raster order).
        hidden_states = self.patch_embedder(hidden_states, fast_pos_embed_idx,
                                            fast_pos_embed_weight)
        hidden_states = self.ln_pre(hidden_states)

        # Per-token (spatial_merge_size=1) window reorder.  Rotary is reordered
        # the same way (equivalent to reordering position_ids before the rope).
        hidden_states = hidden_states[window_index, :]
        rotary_pos_emb = rotary_pos_emb[window_index, :]

        # rotary_pos_emb per token = [freq_w, freq_h] (head_dim // 2).  Doubling
        # yields [freq_w, freq_h, freq_w, freq_h] (head_dim) — the reference
        # freq layout — before cos/sin and half-rotation.
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for i, block in enumerate(self.layers):
            if self.layer_types[i] == "full_attention":
                cu_now = cu_seqlens
                max_now = max_seqlen_carrier
                kvl_now = kv_lengths
            else:  # window_attention
                cu_now = cu_window_seqlens
                max_now = torch.zeros(self.window_max_seqlen,
                                      dtype=torch.int32,
                                      device=hidden_states.device)
                kvl_now = kv_lengths_window
            hidden_states = block(hidden_states,
                                  cu_now,
                                  max_now,
                                  position_embeddings,
                                  kv_lengths=kvl_now)

        # Undo the window ordering and simultaneously group tokens into 2x2
        # output blocks (reverse_window_index is host-precomputed to do both).
        hidden_states = hidden_states[reverse_window_index, :]
        hidden_states = self.ln_post(hidden_states)

        # pixel_shuffle (merge_size=2): concatenate each 2x2 block dim-major.
        # [T, H] -> [G, merge_unit, H] -> [G, H, merge_unit] -> [G, H*merge_unit]
        hidden_states = hidden_states.view(-1, self.merge_unit,
                                           self.hidden_size)
        hidden_states = hidden_states.permute(0, 2, 1).contiguous()
        hidden_states = hidden_states.reshape(-1, self.merged_size)
        return hidden_states


# ---------------------------------------------------------------------------
# Adapter / projection
# ---------------------------------------------------------------------------


class MuseGlimmerVisionAdapter(nn.Module):
    """Vision adapter: fc1 -> gelu -> fc2 -> gelu (both linears bias-free).

    Checkpoint keys (under ``vision_adapter.``):
        fc1.weight  [4096, 6144]
        fc2.weight  [4096, 4096]
    """

    def __init__(self,
                 out_hidden_size: int,
                 projector_hidden_size: int,
                 model_config: "ModelConfig | None" = None,
                 name_prefix: str = "vision_adapter") -> None:
        super().__init__()
        self.fc1 = _make_visual_linear(model_config,
                                       out_hidden_size,
                                       projector_hidden_size,
                                       f"{name_prefix}.fc1",
                                       bias=False)
        self.fc2 = _make_visual_linear(model_config,
                                       projector_hidden_size,
                                       projector_hidden_size,
                                       f"{name_prefix}.fc2",
                                       bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.fc2(F.gelu(self.fc1(x))))


# ---------------------------------------------------------------------------
# Top-level visual model
# ---------------------------------------------------------------------------


class MuseGlimmerVisualModel(nn.Module):
    """Complete Muse-Glimmer vision encoder (tower + adapter + projection).

    ``config`` is the full ``config.json`` dict; the vision tower reads
    ``vision_config`` and the projection reads the top-level
    ``out_hidden_size`` / ``projector_hidden_size`` / ``projector_hidden_act``
    plus the text hidden size.

    Reproduces ``MuseGlimmerModel.get_image_features`` (minus the per-image
    split): vision_tower -> vision_adapter -> vision_projection ->
    perception_emb_norm.
    """

    def __init__(self,
                 config: dict,
                 model_config: "ModelConfig | None" = None) -> None:
        super().__init__()
        vision_config: dict = config.get("vision_config", config)

        self.hidden_size: int = vision_config["hidden_size"]
        self.num_heads: int = vision_config["num_attention_heads"]
        self.head_dim: int = self.hidden_size // self.num_heads
        num_layers: int = vision_config["num_hidden_layers"]
        intermediate_size: int = vision_config["intermediate_size"]
        self.patch_size: int = vision_config.get("patch_size", 14)
        self.patch_temporal: int = vision_config.get("patch_temporal", 2)
        num_channels: int = vision_config.get("num_channels", 3)
        self.merge_size: int = vision_config.get("merge_size", 2)
        self.pos_emb_height: int = vision_config.get("pos_emb_height", 32)
        self.pos_emb_width: int = vision_config.get("pos_emb_width", 32)
        layer_norm_eps: float = vision_config.get("layer_norm_eps", 1e-5)

        self.num_grid_per_side: int = self.pos_emb_height
        num_position_embeddings: int = self.pos_emb_height * self.pos_emb_width
        self.in_features: int = (self.patch_temporal * num_channels *
                                 self.patch_size**2)
        self.rotary_pos_emb_dim: int = self.head_dim // 2

        # Window blocks reorder at spatial_merge_size=1 with a
        # window_size = pos_emb_height * patch_size (px) window; the maximum
        # token count per window is (window_size // patch_size)**2.
        self.window_size: int = self.pos_emb_height * self.patch_size
        self.window_max_seqlen: int = (self.window_size // self.patch_size)**2

        # Alternating window / full attention per layer_types.
        layer_types = vision_config.get("layer_types")
        if layer_types is None:
            layer_types = [
                "full_attention" if
                (i + 1) % 4 == 0 or i == num_layers - 1 else "window_attention"
                for i in range(num_layers)
            ]
        self.layer_types: List[str] = list(layer_types)

        attention_scale = config_module._get_attention_scaling(
            vision_config, self.head_dim, 1.0 / (float(self.head_dim)**0.5))

        # rope base for the optional host-side rotary helper.
        rope_params = vision_config.get("rope_parameters") or {}
        self.rope_theta: float = float(rope_params.get("rope_theta", 10000.0))

        self.vision_tower = MuseGlimmerVisionModel(
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            num_heads=self.num_heads,
            intermediate_size=intermediate_size,
            in_features=self.in_features,
            num_position_embeddings=num_position_embeddings,
            merge_size=self.merge_size,
            layer_norm_eps=layer_norm_eps,
            layer_types=self.layer_types,
            window_max_seqlen=self.window_max_seqlen,
            attention_scale=attention_scale,
            model_config=model_config,
            name_prefix="vision_tower")

        # Top-level (multimodal) projection config.
        self.out_hidden_size: int = config.get(
            "out_hidden_size", self.hidden_size * self.merge_size**2)
        projector_hidden_size: int = config.get("projector_hidden_size", 4096)
        text_config: dict = config.get("text_config", {})
        text_hidden_size: int = text_config.get("hidden_size", 6656)
        text_rms_eps: float = text_config.get("rms_norm_eps", 1e-5)

        self.vision_adapter = MuseGlimmerVisionAdapter(
            self.out_hidden_size,
            projector_hidden_size,
            model_config,
            name_prefix="vision_adapter")
        self.vision_projection = _make_visual_linear(model_config,
                                                     projector_hidden_size,
                                                     text_hidden_size,
                                                     "vision_projection",
                                                     bias=False)
        self.perception_emb_norm = MuseGlimmerRMSNorm(eps=text_rms_eps)
        self._use_trt_attn = is_trt_native_attention_enabled()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def fast_pos_embed_interpolate(
            self, grid_thw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pre-compute the 4-tap bilinear position-table gather (raster order).

        Reproduces the reference ``get_vision_bilinear_indices_and_weights``
        with ``spatial_merge_size=1`` (raster token order): equivalent to
        ``F.grid_sample(..., align_corners=False, padding="zeros")`` on the
        ``num_grid_per_side x num_grid_per_side`` position table.

        Returns:
            ``(idx [4, T] int64, weight [4, T] pos-table dtype)``.
        """
        side = self.num_grid_per_side
        idx_parts: List[List[torch.Tensor]] = [[], [], [], []]
        weight_parts: List[List[torch.Tensor]] = [[], [], [], []]

        for t, h, w in grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)
            h_grid = (torch.arange(h, dtype=torch.float32) + 0.5) * (side /
                                                                     h) - 0.5
            w_grid = (torch.arange(w, dtype=torch.float32) + 0.5) * (side /
                                                                     w) - 0.5

            h_floor = torch.floor(h_grid).long()
            w_floor = torch.floor(w_grid).long()
            h_ceil = h_floor + 1
            w_ceil = w_floor + 1
            h_frac = h_grid - h_floor.float()
            w_frac = w_grid - w_floor.float()

            h_floor_valid = (h_floor >= 0) & (h_floor <= side - 1)
            h_ceil_valid = (h_ceil >= 0) & (h_ceil <= side - 1)
            w_floor_valid = (w_floor >= 0) & (w_floor <= side - 1)
            w_ceil_valid = (w_ceil >= 0) & (w_ceil <= side - 1)
            h_floor = h_floor.clamp(0, side - 1)
            h_ceil = h_ceil.clamp(0, side - 1)
            w_floor = w_floor.clamp(0, side - 1)
            w_ceil = w_ceil.clamp(0, side - 1)

            h_floor_offset = h_floor * side
            h_ceil_offset = h_ceil * side

            corner_indices = [
                (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
                (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
                (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
                (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
            ]
            corner_weights = [
                ((1 - h_frac)[:, None] * (1 - w_frac)[None, :] *
                 (h_floor_valid[:, None] & w_floor_valid[None, :])).flatten(),
                ((1 - h_frac)[:, None] * w_frac[None, :] *
                 (h_floor_valid[:, None] & w_ceil_valid[None, :])).flatten(),
                (h_frac[:, None] * (1 - w_frac)[None, :] *
                 (h_ceil_valid[:, None] & w_floor_valid[None, :])).flatten(),
                (h_frac[:, None] * w_frac[None, :] *
                 (h_ceil_valid[:, None] & w_ceil_valid[None, :])).flatten(),
            ]
            # spatial_merge_size == 1 -> plain raster order, repeated t times.
            reorder = torch.arange(h * w).repeat(t)
            for i in range(4):
                idx_parts[i].append(corner_indices[i][reorder])
                weight_parts[i].append(corner_weights[i][reorder])

        pos_weight = self.vision_tower.patch_embedder.position_embedding_table.weight
        idx_tensor = torch.stack([torch.cat(p) for p in idx_parts
                                  ]).to(dtype=torch.long,
                                        device=pos_weight.device)
        weight_tensor = torch.stack([torch.cat(p) for p in weight_parts
                                     ]).to(dtype=pos_weight.dtype,
                                           device=pos_weight.device)
        return idx_tensor, weight_tensor

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Host-side 2D interleaved rotary positions in raster order.

        Returns ``[T, head_dim // 2]`` where each token's row is
        ``concat(freq_w, freq_h)`` and ``freq_{w,h} = pos * inv_freq``.  The
        reference offsets positions by ``+1`` (``position_ids.flip(-1) + 1``);
        that offset is folded into the ``pos`` values here.
        """
        spatial_dim = self.head_dim // 2
        inv_freq = 1.0 / (self.rope_theta**(torch.arange(
            0, spatial_dim, 2, dtype=torch.float32) / spatial_dim))
        rows: List[torch.Tensor] = []
        for t, h, w in grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)
            hpos = torch.arange(h, dtype=torch.float32) + 1
            wpos = torch.arange(w, dtype=torch.float32) + 1
            hh = hpos[:, None].expand(h, w).reshape(-1)
            ww = wpos[None, :].expand(h, w).reshape(-1)
            freq_w = torch.outer(ww, inv_freq)  # [h*w, spatial_dim//2]
            freq_h = torch.outer(hh, inv_freq)  # [h*w, spatial_dim//2]
            frame = torch.cat([freq_w, freq_h], dim=-1)  # [h*w, spatial_dim]
            rows.append(frame.repeat(t, 1))
        return torch.cat(rows, dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_window_seqlens: torch.Tensor,
        window_index: torch.Tensor,
        reverse_window_index: torch.Tensor,
        fast_pos_embed_idx: torch.Tensor,  # [4, T] int64
        fast_pos_embed_weight: torch.Tensor,  # [4, T] float16
        max_seqlen_carrier: Optional[torch.Tensor] = None,
        kv_lengths: Optional[torch.Tensor] = None,
        kv_lengths_window: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.vision_tower(
            hidden_states,
            rotary_pos_emb,
            cu_seqlens,
            cu_window_seqlens,
            window_index,
            reverse_window_index,
            fast_pos_embed_idx,
            fast_pos_embed_weight,
            max_seqlen_carrier=max_seqlen_carrier,
            kv_lengths=kv_lengths,
            kv_lengths_window=kv_lengths_window,
        )
        hidden_states = self.vision_adapter(hidden_states)
        hidden_states = self.vision_projection(hidden_states)
        hidden_states = self.perception_emb_norm(hidden_states)
        return hidden_states

    def get_onnx_export_args(self, config: dict, device: str):
        """Return (dynamo_inputs, onnx_input_names, output_names, dynamic_shapes)."""
        # total_tokens must be divisible by merge_unit (= merge_size**2) for the
        # final view(-1, merge_unit, H).  Use num_patches = merge_unit * G.
        num_groups = 64
        num_patches = self.vision_tower.merge_unit * num_groups  # 256

        pixel_values = torch.zeros(num_patches,
                                   self.in_features,
                                   dtype=torch.float16,
                                   device=device)
        rotary_pos_emb = torch.zeros(num_patches,
                                     self.rotary_pos_emb_dim,
                                     dtype=torch.float32,
                                     device=device)
        cu_seqlens = torch.tensor([0, num_patches],
                                  dtype=torch.int32,
                                  device=device)
        # Window covers all tokens when num_patches <= window_max_seqlen.
        window_patches = min(num_patches, self.window_max_seqlen)
        cu_window_seqlens = torch.tensor([0, window_patches],
                                         dtype=torch.int32,
                                         device=device)
        # Per-token (spatial_merge_size=1) reorder indices -> INT64 for
        # aten.index during dynamo export.
        window_index = torch.arange(num_patches,
                                    dtype=torch.int64,
                                    device=device)
        reverse_window_index = torch.arange(num_patches,
                                            dtype=torch.int64,
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
            "input", "rotary_pos_emb", "cu_seqlens", "cu_window_seqlens",
            "window_index", "reverse_window_index", "fast_pos_embed_idx",
            "fast_pos_embed_weight"
        ]
        dynamo_inputs = {
            "hidden_states": pixel_values,
            "rotary_pos_emb": rotary_pos_emb,
            "cu_seqlens": cu_seqlens,
            "cu_window_seqlens": cu_window_seqlens,
            "window_index": window_index,
            "reverse_window_index": reverse_window_index,
            "fast_pos_embed_idx": fast_idx,
            "fast_pos_embed_weight": fast_weight,
        }

        output_names = ["output"]
        # Express T = merge_unit * G so the view(-1, merge_unit, H) reshape's
        # floor-division guard is provable (division by a constant).
        _G = torch.export.Dim("num_groups", min=1)
        T = self.vision_tower.merge_unit * _G
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
                0: T
            },
            "reverse_window_index": {
                0: T
            },
            "fast_pos_embed_idx": {
                1: T
            },
            "fast_pos_embed_weight": {
                1: T
            },
        }

        if self._use_trt_attn:
            onnx_input_names.extend(["kv_lengths", "kv_lengths_window"])
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
            # max_seqlen_carrier uses an INDEPENDENT dynamic dim: the C++
            # builder profiles kMaxSeqLenCarrier separately from total_tokens.
            _max_seqlen = torch.export.Dim("max_seqlen", min=1)
            dynamic_shapes["max_seqlen_carrier"] = {0: _max_seqlen}
        return dynamo_inputs, onnx_input_names, output_names, dynamic_shapes


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

_MUSE_GLIMMER_VISUAL_PREFIXES = (
    "model.vision_tower.",
    "model.vision_adapter.",
    "model.vision_projection.",
    "model.perception_emb_norm.",
)


def load_muse_glimmer_visual_weights(
        model: MuseGlimmerVisualModel,
        weights: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
    """Load ``model.vision_tower.*`` / ``model.vision_adapter.*`` /
    ``model.vision_projection.*`` checkpoint weights.

    Strips the leading ``model.`` so keys land on this module tree and assigns
    through the shared ``load_submodule_weights`` pipeline.

    Returns ``(missing, unexpected)``.
    """
    from ...checkpoint.loader import load_submodule_weights

    strip = "model."

    def _remap(k: str) -> "str | None":
        return k[len(strip):] if k.startswith(
            _MUSE_GLIMMER_VISUAL_PREFIXES) else None

    load_submodule_weights(model,
                           weights,
                           _remap,
                           label="MuseGlimmerVisualModel")

    remapped = {
        k[len(strip):]
        for k in weights if k.startswith(_MUSE_GLIMMER_VISUAL_PREFIXES)
    }
    model_keys = set(model.state_dict().keys())
    missing = sorted(model_keys - remapped)
    unexpected = sorted(remapped - model_keys)
    return missing, unexpected


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_muse_glimmer_visual(
    config: dict,
    weights: Dict[str, torch.Tensor],
    model_config: "ModelConfig | None" = None,
    dtype: torch.dtype = torch.float16,
) -> MuseGlimmerVisualModel:
    """Build and return a :class:`MuseGlimmerVisualModel` with loaded weights.

    Matches the ``export_encoder`` visual-family ``build_fn`` contract.

    Args:
        config:       Full checkpoint ``config.json`` dict (``vision_config``
                      + top-level projector fields + token IDs).
        weights:      Flat ``{key: tensor}`` checkpoint dict.
        model_config: Optional top-level ``ModelConfig`` for quantized Linear
                      dispatch; ``None`` builds plain FP16 linears.
        dtype:        Weight dtype (default ``float16``).
    """
    model = MuseGlimmerVisualModel(config, model_config=model_config).to(dtype)
    load_muse_glimmer_visual_weights(model, weights)
    model.eval()
    return model


__all__ = [
    "MuseGlimmerRMSNorm",
    "MuseGlimmerVisionPatchEmbed",
    "MuseGlimmerVisionAttention",
    "MuseGlimmerVisionMLP",
    "MuseGlimmerVisionEncoderLayer",
    "MuseGlimmerVisionModel",
    "MuseGlimmerVisionAdapter",
    "MuseGlimmerVisualModel",
    "load_muse_glimmer_visual_weights",
    "build_muse_glimmer_visual",
]
