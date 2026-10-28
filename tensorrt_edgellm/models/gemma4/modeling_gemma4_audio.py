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
From-scratch Gemma4 audio encoder for E2B/E4B models.

Architecture:
    SubsampleConvProjection (4x temporal downsample):
        Conv2d(1->128, k=3, s=2) + LayerNorm + ReLU
        Conv2d(128->32, k=3, s=2) + LayerNorm + ReLU
        Linear(32 * feat/4 -> 1024)

    RelPositionalEncoding (sinusoidal countdown, context_size//2+1 positions)

    12x AudioLayer:
        FFN1: clamp -> RMSNorm -> ClippableLinear(1024->4096) -> SiLU
              -> ClippableLinear(4096->1024) -> clamp -> RMSNorm -> x*0.5 + residual
        Attn: clamp -> RMSNorm -> ChunkedLocalAttn(chunk=12, ctx=24, softcap=50)
              -> clamp -> RMSNorm + residual
        LightConv1d: RMSNorm -> Linear(1024->2048) -> GLU
                     -> CausalConv1d(k=5, depthwise) -> clamp -> RMSNorm -> SiLU
                     -> Linear(1024->1024) + residual
        FFN2: (same as FFN1)
        clamp -> RMSNorm(norm_out)

    output_proj: Linear(1024->1536, bias=True)

Checkpoint weight key prefix:
    ``model.audio_tower.*``

Module member names match HuggingFace checkpoint keys directly (no remapping needed).

ONNX Forward I/O (with embedder, default):
    Inputs:
        input_features   [1, seq_len, num_mel_bins]  float16
        valid            [1, seq_len//4]  bool  (True for real tokens after downsample)
    Output:
        last_hidden_state  [1, seq_len/4, text_hidden_size]  float16

    The multimodal embedder (RMSNorm no-scale -> Linear) projects from
    output_proj_dims (1536) to text_hidden_size (1536 for E2B, 2560 for E4B).
    HF weight prefix: model.embed_audio.*
"""

from __future__ import annotations

import logging
import math
from functools import cached_property
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Gemma4AudioConfig

from ...config import ModelConfig
from ..linear import make_linear

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture defaults sourced from HF Gemma4AudioConfig.
# intermediate_size and num_mel_bins are architecture-fixed (hidden_size * 4
# and feature_extractor feature_size respectively) and not in the config class.
# All other values are read from config.json at runtime; these serve only as
# fallback defaults for the sub-module __init__ signatures.
# ---------------------------------------------------------------------------
_ac = Gemma4AudioConfig()

_CONV_KERNEL_SIZE = _ac.conv_kernel_size
_ATTENTION_CHUNK_SIZE = _ac.attention_chunk_size
_ATTENTION_CONTEXT_LEFT = _ac.attention_context_left
_ATTENTION_CONTEXT_RIGHT = _ac.attention_context_right
_ATTENTION_LOGIT_CAP = _ac.attention_logit_cap
_RESIDUAL_WEIGHT = _ac.residual_weight
_GRADIENT_CLIPPING = _ac.gradient_clipping
_RMS_NORM_EPS = _ac.rms_norm_eps

# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class Gemma4RMSNorm(nn.Module):
    """RMSNorm with optional learned scale parameter.

    Args:
        hidden_size: Dimension to normalize.
        eps: Epsilon for numerical stability.
        with_scale: If True, includes a learnable weight parameter.
    """

    def __init__(self,
                 hidden_size: int,
                 eps: float = _RMS_NORM_EPS,
                 with_scale: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.with_scale:
            return (self.weight.to(input_dtype) * x.to(input_dtype))
        return x.to(input_dtype)


# ---------------------------------------------------------------------------
# ClippableLinear
# ---------------------------------------------------------------------------


class Gemma4ClippableLinear(nn.Module):
    """Linear layer with learned input/output clamping bounds.

    When ``use_clipped_linears=True``, each linear has 4 buffer values:
        input_min, input_max, output_min, output_max
    that clamp the input before and output after the linear operation.

    Checkpoint keys (under the parent's name):
        ``linear.weight``, ``linear.bias`` (optional)
        ``input_min``, ``input_max``, ``output_min``, ``output_max``
    """

    def __init__(self,
                 model_config: ModelConfig,
                 in_features: int,
                 out_features: int,
                 bias: bool = False,
                 use_clipping: bool = True,
                 module_name: str = "") -> None:
        super().__init__()
        self.use_clipping = use_clipping
        self.linear = make_linear(model_config,
                                  in_features,
                                  out_features,
                                  bias=bias,
                                  module_name=module_name)
        if use_clipping:
            self.register_buffer("input_min",
                                 torch.tensor(-_GRADIENT_CLIPPING))
            self.register_buffer("input_max", torch.tensor(_GRADIENT_CLIPPING))
            self.register_buffer("output_min",
                                 torch.tensor(-_GRADIENT_CLIPPING))
            self.register_buffer("output_max",
                                 torch.tensor(_GRADIENT_CLIPPING))
            # Plain floats for ONNX export (avoids tensor Clip inputs that
            # TRT's ONNX parser cannot resolve)
            self._input_min_val = -_GRADIENT_CLIPPING
            self._input_max_val = _GRADIENT_CLIPPING
            self._output_min_val = -_GRADIENT_CLIPPING
            self._output_max_val = _GRADIENT_CLIPPING

    def _sync_clipping_vals(self):
        """Sync plain-float clipping values from tensor buffers."""
        if self.use_clipping:
            self._input_min_val = self.input_min.item()
            self._input_max_val = self.input_max.item()
            self._output_min_val = self.output_min.item()
            self._output_max_val = self.output_max.item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_clipping:
            x = torch.nn.functional.hardtanh(x, self._input_min_val,
                                             self._input_max_val)
        x = self.linear(x)
        if self.use_clipping:
            x = torch.nn.functional.hardtanh(x, self._output_min_val,
                                             self._output_max_val)
        return x


# ---------------------------------------------------------------------------
# Subsample convolution projection (4x temporal downsample)
# ---------------------------------------------------------------------------


class _ConvNormBlock(nn.Module):
    """Conv2d + LayerNorm block for SubSampleConvProjection."""

    def __init__(self, in_channels: int, out_channels: int,
                 norm_eps: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias=False)
        self.norm = nn.LayerNorm(out_channels, eps=norm_eps, bias=False)


class Gemma4AudioSubSampleConvProjection(nn.Module):
    """2x Conv2d stride-2 with LayerNorm + ReLU, then Linear projection.

    Input:  [B, T, num_mel_bins]
    Output: [B, T//4, hidden_size]

    Checkpoint keys (under ``subsample_conv_projection.``):
        layer0.conv.weight
        layer0.norm.weight
        layer1.conv.weight
        layer1.norm.weight
        input_proj_linear.weight
    """

    def __init__(self,
                 num_mel_bins: int,
                 hidden_size: int,
                 conv_channels: list = None,
                 norm_eps: float = _RMS_NORM_EPS) -> None:
        super().__init__()
        if conv_channels is None:
            conv_channels = list(_ac.subsampling_conv_channels)

        c1, c2 = conv_channels
        self.layer0 = _ConvNormBlock(1, c1, norm_eps)
        self.layer1 = _ConvNormBlock(c1, c2, norm_eps)
        # After 2x stride-2 on freq axis: freq_out = num_mel_bins // 4
        freq_out = num_mel_bins // 4
        self.input_proj_linear = nn.Linear(c2 * freq_out,
                                           hidden_size,
                                           bias=False)

    def forward(
        self, x: torch.Tensor, input_features_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, T, mel_bins] -> [B, T//4, hidden_size]"""
        x = x.unsqueeze(1)  # [B, 1, T, mel]
        for block in (self.layer0, self.layer1):
            x = x * input_features_mask[:, None, :, None]
            x = block.conv(x)  # [B, C, T', F']
            # LayerNorm over channel dim: permute to [B, T', F', C]
            x = x.permute(0, 2, 3, 1)
            x = block.norm(x)
            x = x.permute(0, 3, 1, 2)  # back to [B, C, T', F']
            x = F.relu(x)
            input_features_mask = input_features_mask[:, ::2]
        # x: [B, c2, T//4, mel//4]
        B, C, T, F_ = x.shape
        # HF uses permute(0, 2, 3, 1) → [B, T, F, C] then reshape
        x = x.permute(0, 2, 3, 1).contiguous().reshape(B, T, -1)
        x = self.input_proj_linear(x)
        return x, input_features_mask


# ---------------------------------------------------------------------------
# Relative positional encoding (sinusoidal countdown)
# ---------------------------------------------------------------------------


class Gemma4AudioRelPositionalEncoding(nn.Module):
    """Sinusoidal countdown positional encoding for chunked attention.

    Generates position embeddings for context_size//2+1 positions (countdown
    from context_size//2 to 0). Uses concatenated [sin, cos] layout matching
    HF Gemma4AudioRelPositionalEncoding.

    Output shape: [context_size//2+1, hidden_size]
    """

    inv_timescales: torch.Tensor

    def __init__(self, hidden_size: int, context_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_positions = context_size // 2 + 1  # 13 for context_size=24

        # Match HF: min_timescale=1.0, max_timescale=10000.0
        min_timescale = 1.0
        max_timescale = 10000.0
        num_timescales = hidden_size // 2
        log_timescale_increment = (math.log(max_timescale / min_timescale) /
                                   max(num_timescales - 1, 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales) * -log_timescale_increment)
        # Store as [1, 1, num_timescales] for broadcasting
        self.register_buffer("inv_timescales",
                             inv_timescales.unsqueeze(0).unsqueeze(0),
                             persistent=False)

    @torch.no_grad()
    def forward(self) -> torch.Tensor:
        """Return [context_size//2+1, hidden_size] position embeddings."""
        # Countdown: [context_size//2, ..., 1, 0]
        position_ids = torch.arange(self.context_size // 2,
                                    -1,
                                    -1,
                                    device=self.inv_timescales.device)
        position_ids = position_ids[..., None]  # [num_pos, 1]
        scaled_time = position_ids * self.inv_timescales.squeeze(0)
        # Concatenate [sin, cos] (not interleaved)
        pos_embed = torch.cat([torch.sin(scaled_time),
                               torch.cos(scaled_time)],
                              dim=-1)
        return pos_embed


# ---------------------------------------------------------------------------
# Feed-forward module
# ---------------------------------------------------------------------------


class Gemma4AudioFeedForward(nn.Module):
    """Audio FFN: clamp -> RMSNorm -> Linear -> SiLU -> Linear -> clamp ->
    RMSNorm -> x*residual_weight + residual.

    Checkpoint keys (under ``feed_forward1.`` or ``feed_forward2.``):
        pre_layer_norm.weight
        ffw_layer_1.{linear.weight, input_min, input_max, output_min, output_max}
        ffw_layer_2.{linear.weight, input_min, input_max, output_min, output_max}
        post_layer_norm.weight
    """

    def __init__(self,
                 model_config: ModelConfig,
                 hidden_size: int,
                 intermediate_size: int,
                 residual_weight: float = _RESIDUAL_WEIGHT,
                 gradient_clipping: float = _GRADIENT_CLIPPING,
                 use_clipped_linears: bool = True,
                 name_prefix: str = "") -> None:
        super().__init__()
        self.residual_weight = residual_weight
        self.gradient_clipping = gradient_clipping
        self.pre_layer_norm = Gemma4RMSNorm(hidden_size)
        self.ffw_layer_1 = Gemma4ClippableLinear(
            model_config,
            hidden_size,
            intermediate_size,
            bias=False,
            use_clipping=use_clipped_linears,
            module_name=f"{name_prefix}.ffw_layer_1.linear"
            if name_prefix else "")
        self.ffw_layer_2 = Gemma4ClippableLinear(
            model_config,
            intermediate_size,
            hidden_size,
            bias=False,
            use_clipping=use_clipped_linears,
            module_name=f"{name_prefix}.ffw_layer_2.linear"
            if name_prefix else "")
        self.post_layer_norm = Gemma4RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        gc = min(self.gradient_clipping, torch.finfo(x.dtype).max)
        x = F.hardtanh(x, -gc, gc)
        x = self.pre_layer_norm(x)
        x = self.ffw_layer_1(x)
        x = F.silu(x)
        x = self.ffw_layer_2(x)
        x = F.hardtanh(x, -gc, gc)
        x = self.post_layer_norm(x)
        x = x * self.residual_weight + residual
        return x


# ---------------------------------------------------------------------------
# Causal Conv1d (left-padded depthwise)
# ---------------------------------------------------------------------------


class Gemma4AudioCausalConv1d(nn.Conv1d):
    """Causal depthwise Conv1d with left-padding.

    Inherits from nn.Conv1d so that ``self.weight`` is stored directly
    (matching HF checkpoint key ``depthwise_conv1d.weight``).
    """

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__(channels,
                         channels,
                         kernel_size=kernel_size,
                         groups=channels,
                         bias=False)

    @cached_property
    def left_pad(self):
        return (self.kernel_size[0] -
                1) * self.dilation[0] + 1 - self.stride[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] -> [B, C, T] (causal: left-padded)"""
        x = F.pad(x, (self.left_pad, 0))
        return super().forward(x)


# ---------------------------------------------------------------------------
# LightConv1d module
# ---------------------------------------------------------------------------


class Gemma4AudioLightConv1d(nn.Module):
    """LightConv: RMSNorm -> Linear -> GLU -> CausalConv1d -> clamp ->
    RMSNorm -> SiLU -> Linear + residual.

    Checkpoint keys (under ``lconv1d.``):
        pre_layer_norm.weight
        linear_start.{linear.weight, input_min, ...}
        depthwise_conv1d.weight
        conv_norm.weight
        linear_end.{linear.weight, input_min, ...}
    """

    def __init__(self,
                 model_config: ModelConfig,
                 hidden_size: int,
                 conv_kernel_size: int = _CONV_KERNEL_SIZE,
                 gradient_clipping: float = _GRADIENT_CLIPPING,
                 use_clipped_linears: bool = True,
                 name_prefix: str = "") -> None:
        super().__init__()
        self.gradient_clipping = gradient_clipping
        self.pre_layer_norm = Gemma4RMSNorm(hidden_size)
        # Linear -> GLU splits output in half: 2*hidden -> hidden
        self.linear_start = Gemma4ClippableLinear(
            model_config,
            hidden_size,
            hidden_size * 2,
            bias=False,
            use_clipping=use_clipped_linears,
            module_name=f"{name_prefix}.linear_start.linear"
            if name_prefix else "")
        self.depthwise_conv1d = Gemma4AudioCausalConv1d(
            hidden_size, conv_kernel_size)
        self.conv_norm = Gemma4RMSNorm(hidden_size)
        self.linear_end = Gemma4ClippableLinear(
            model_config,
            hidden_size,
            hidden_size,
            bias=False,
            use_clipping=use_clipped_linears,
            module_name=f"{name_prefix}.linear_end.linear"
            if name_prefix else "")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        gc = min(self.gradient_clipping, torch.finfo(x.dtype).max)
        x = self.pre_layer_norm(x)
        x = self.linear_start(x)
        x = F.glu(x, dim=-1)  # [B, T, 2*H] -> [B, T, H]
        x = self.depthwise_conv1d(x.transpose(1, 2)).transpose(1, 2)
        x = F.hardtanh(x, -gc, gc)
        x = self.conv_norm(x)
        x = F.silu(x)
        x = self.linear_end(x)
        x = x + residual
        return x


# ---------------------------------------------------------------------------
# Chunked local attention with relative position bias
# ---------------------------------------------------------------------------


class Gemma4AudioAttention(nn.Module):
    """Chunked local attention with relative position bias and softcap.

    Splits sequence into non-overlapping chunks of size ``chunk_size``, then
    each chunk attends to a context window of ``context_size`` (left context +
    chunk itself). Uses relative positional bias via a learned projection of
    sinusoidal countdown embeddings.

    Checkpoint keys (under ``self_attn.``):
        q_proj.{linear.weight, input_min, ...}
        k_proj.{linear.weight, input_min, ...}
        v_proj.{linear.weight, input_min, ...}
        post.{linear.weight, input_min, ...}
        relative_k_proj.weight
        per_dim_scale   — [head_dim]
    """

    def __init__(self,
                 model_config: ModelConfig,
                 hidden_size: int,
                 num_heads: int,
                 chunk_size: int = _ATTENTION_CHUNK_SIZE,
                 context_left: int = _ATTENTION_CONTEXT_LEFT,
                 context_right: int = _ATTENTION_CONTEXT_RIGHT,
                 logit_cap: float = _ATTENTION_LOGIT_CAP,
                 gradient_clipping: float = _GRADIENT_CLIPPING,
                 use_clipped_linears: bool = True,
                 name_prefix: str = "") -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.chunk_size = chunk_size
        self.context_left = context_left
        self.context_right = context_right
        # context_size = chunk_size + (context_left - 1) + context_right = 24
        self.context_size = chunk_size + (context_left - 1) + context_right
        self.logit_cap = logit_cap
        self.gradient_clipping = gradient_clipping

        self.q_proj = Gemma4ClippableLinear(
            model_config,
            hidden_size,
            hidden_size,
            bias=False,
            use_clipping=use_clipped_linears,
            module_name=f"{name_prefix}.q_proj.linear" if name_prefix else "")
        self.k_proj = Gemma4ClippableLinear(
            model_config,
            hidden_size,
            hidden_size,
            bias=False,
            use_clipping=use_clipped_linears,
            module_name=f"{name_prefix}.k_proj.linear" if name_prefix else "")
        self.v_proj = Gemma4ClippableLinear(
            model_config,
            hidden_size,
            hidden_size,
            bias=False,
            use_clipping=use_clipped_linears,
            module_name=f"{name_prefix}.v_proj.linear" if name_prefix else "")
        self.post = Gemma4ClippableLinear(
            model_config,
            hidden_size,
            hidden_size,
            bias=False,
            use_clipping=use_clipped_linears,
            module_name=f"{name_prefix}.post.linear" if name_prefix else "")
        # Relative position key projection (no clipping)
        self.relative_k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Per-dim scale: learnable parameter [head_dim] (broadcast over heads)
        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim))

    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """Forward using the fused Gemma4AudioAttentionPlugin.

        The plugin fuses Q/K scaling, chunked context gather, content +
        relative-position scores, soft-cap, masking, softmax, and value mix.

        Args:
            x: [B, T, hidden_size] — already normed by caller
            pos_embed: [P, hidden_size] from RelPositionalEncoding
            valid: [B, T] bool — True for real tokens, False for padding

        Returns:
            [B, T, hidden_size]
        """
        from ..ops import gemma4_audio_attention_plugin

        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        # Project Q, K, V (raw — kernel handles scaling)
        q_raw = self.q_proj(x).view(B, T, H, D)
        k_raw = self.k_proj(x).view(B, T, H, D)
        v = self.v_proj(x).view(B, T, H, D)

        # Relative position keys: [P, H, D]
        rel_key = self.relative_k_proj(pos_embed).view(-1, H, D)

        # gamma = per_dim_scale (fp32)
        gamma = self.per_dim_scale.float()

        # seq_len carrier (shape hint for TRT)
        seq_len_carrier = torch.tensor([T], dtype=torch.int32, device=x.device)

        # Call the plugin custom op
        out = gemma4_audio_attention_plugin(
            q_raw,
            k_raw,
            v,
            gamma,
            rel_key,
            valid,
            seq_len_carrier,
            chunk_size=self.chunk_size,
            left_horizon=self.context_left - 1,
            context_size=self.context_size,
            logit_cap=self.logit_cap,
        )

        # Reshape [B, T, H, D] -> [B, T, H*D] and apply output projection
        out = out.reshape(B, T, -1)
        out = self.post(out.to(dtype=self.post.linear.weight.dtype))
        return out


# ---------------------------------------------------------------------------
# Audio layer
# ---------------------------------------------------------------------------


class Gemma4AudioLayer(nn.Module):
    """Single Gemma4 audio encoder layer: FFN1 -> Attn -> LightConv -> FFN2 -> norm.

    Structure matches HF Gemma4AudioLayer:
        feed_forward1 → norm_pre_attn → self_attn → norm_post_attn → lconv1d
        → feed_forward2 → norm_out

    Checkpoint keys (under ``layers.N.``):
        feed_forward1.*
        self_attn.*
        norm_pre_attn.weight
        norm_post_attn.weight
        lconv1d.*
        feed_forward2.*
        norm_out.weight
    """

    def __init__(self,
                 model_config: ModelConfig,
                 hidden_size: int,
                 num_heads: int,
                 intermediate_size: int,
                 conv_kernel_size: int,
                 chunk_size: int,
                 context_left: int,
                 context_right: int,
                 logit_cap: float,
                 residual_weight: float,
                 gradient_clipping: float,
                 use_clipped_linears: bool,
                 name_prefix: str = "") -> None:
        super().__init__()
        self.gradient_clipping = gradient_clipping

        self.feed_forward1 = Gemma4AudioFeedForward(
            model_config,
            hidden_size,
            intermediate_size,
            residual_weight=residual_weight,
            gradient_clipping=gradient_clipping,
            use_clipped_linears=use_clipped_linears,
            name_prefix=f"{name_prefix}.feed_forward1" if name_prefix else "")
        self.norm_pre_attn = Gemma4RMSNorm(hidden_size)
        self.self_attn = Gemma4AudioAttention(
            model_config,
            hidden_size,
            num_heads,
            chunk_size=chunk_size,
            context_left=context_left,
            context_right=context_right,
            logit_cap=logit_cap,
            gradient_clipping=gradient_clipping,
            use_clipped_linears=use_clipped_linears,
            name_prefix=f"{name_prefix}.self_attn" if name_prefix else "")
        self.norm_post_attn = Gemma4RMSNorm(hidden_size)
        self.lconv1d = Gemma4AudioLightConv1d(
            model_config,
            hidden_size,
            conv_kernel_size=conv_kernel_size,
            gradient_clipping=gradient_clipping,
            use_clipped_linears=use_clipped_linears,
            name_prefix=f"{name_prefix}.lconv1d" if name_prefix else "")
        self.feed_forward2 = Gemma4AudioFeedForward(
            model_config,
            hidden_size,
            intermediate_size,
            residual_weight=residual_weight,
            gradient_clipping=gradient_clipping,
            use_clipped_linears=use_clipped_linears,
            name_prefix=f"{name_prefix}.feed_forward2" if name_prefix else "")
        self.norm_out = Gemma4RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        gradient_clipping = min(
            self.gradient_clipping,
            torch.finfo(self.norm_pre_attn.weight.dtype).max)

        x = self.feed_forward1(x)

        # Attention with layer-level norms and residual
        residual = x
        x = F.hardtanh(x, -gradient_clipping, gradient_clipping)
        x = self.norm_pre_attn(x)
        x = self.self_attn(x, pos_embed, valid)
        x = F.hardtanh(x, -gradient_clipping, gradient_clipping)
        x = self.norm_post_attn(x)
        x = x + residual

        x = self.lconv1d(x)
        x = self.feed_forward2(x)

        x = F.hardtanh(x, -gradient_clipping, gradient_clipping)
        x = self.norm_out(x)
        return x


# ---------------------------------------------------------------------------
# Full audio model
# ---------------------------------------------------------------------------


class Gemma4AudioModel(nn.Module):
    """From-scratch Gemma4 audio encoder (E2B/E4B).

    Output: [batch, encoded_seq_len, output_proj_dims]

    This is the raw encoder only. For the full pipeline including the
    multimodal embedder projection to text_hidden_size, use
    :class:`Gemma4AudioWithEmbedder` (or ``build_gemma4_audio()`` which
    uses it by default).
    """

    def __init__(self,
                 config: Dict[str, Any],
                 *,
                 model_config: ModelConfig,
                 name_prefix: str = "audio_tower") -> None:
        super().__init__()
        ac = config.get("audio_config", config)

        # All values read from config.json's audio_config section.
        # intermediate_size and num_mel_bins are architecture-fixed (not in HF config).
        hidden_size = ac["hidden_size"]
        num_heads = ac["num_attention_heads"]
        num_layers = ac["num_hidden_layers"]
        intermediate_size = ac.get("intermediate_size", hidden_size * 4)
        conv_kernel_size = ac["conv_kernel_size"]
        chunk_size = ac["attention_chunk_size"]
        context_left = ac["attention_context_left"]
        context_right = ac["attention_context_right"]
        logit_cap = ac["attention_logit_cap"]
        output_proj_dims = ac["output_proj_dims"]
        residual_weight = ac["residual_weight"]
        gradient_clipping = ac["gradient_clipping"]
        rms_norm_eps = ac["rms_norm_eps"]
        num_mel_bins = ac.get("num_mel_bins", 128)
        conv_channels = ac["subsampling_conv_channels"]
        use_clipped_linears = ac.get("use_clipped_linears", True)

        # context_size = chunk + (context_left - 1) + context_right = 24
        context_size = chunk_size + (context_left - 1) + context_right

        self.config: Dict[str, Any] = {
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "num_hidden_layers": num_layers,
            "intermediate_size": intermediate_size,
            "conv_kernel_size": conv_kernel_size,
            "attention_chunk_size": chunk_size,
            "attention_context_left": context_left,
            "attention_context_right": context_right,
            "attention_logit_cap": logit_cap,
            "output_proj_dims": output_proj_dims,
            "residual_weight": residual_weight,
            "gradient_clipping": gradient_clipping,
            "rms_norm_eps": rms_norm_eps,
            "num_mel_bins": num_mel_bins,
            "context_size": context_size,
            "use_clipped_linears": use_clipped_linears,
        }

        self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(
            num_mel_bins=num_mel_bins,
            hidden_size=hidden_size,
            conv_channels=list(conv_channels)
            if not isinstance(conv_channels, list) else conv_channels,
            norm_eps=rms_norm_eps,
        )
        self.pos_enc = Gemma4AudioRelPositionalEncoding(
            hidden_size, context_size)
        self.layers = nn.ModuleList([
            Gemma4AudioLayer(
                model_config,
                hidden_size,
                num_heads,
                intermediate_size,
                conv_kernel_size,
                chunk_size=chunk_size,
                context_left=context_left,
                context_right=context_right,
                logit_cap=logit_cap,
                residual_weight=residual_weight,
                gradient_clipping=gradient_clipping,
                use_clipped_linears=use_clipped_linears,
                name_prefix=f"{name_prefix}.layers.{i}" if name_prefix else "")
            for i in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_size, output_proj_dims, bias=True)

    def forward(self, input_features: torch.Tensor,
                input_features_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: [B, seq_len, num_mel_bins]
            input_features_mask: [B, seq_len] bool — True for real mel frames

        Returns:
            [B, seq_len//4, output_proj_dims]
        """
        x, valid = self.subsample_conv_projection(input_features,
                                                  input_features_mask)
        pos_embed = self.pos_enc()

        for layer in self.layers:
            x = layer(x, pos_embed, valid)

        x = self.output_proj(x)
        return x

    def get_onnx_export_args(self, config: Dict[str, Any], device: str):
        """Return (args, input_names, output_names, dynamic_shapes)."""
        num_mel_bins = self.config["num_mel_bins"]
        seq_len = 200  # example length
        dtype = next(self.parameters()).dtype

        input_features = torch.zeros(1,
                                     seq_len,
                                     num_mel_bins,
                                     dtype=dtype,
                                     device=device)
        input_features_mask = torch.ones(1,
                                         seq_len,
                                         dtype=torch.bool,
                                         device=device)

        args = (input_features, input_features_mask)
        input_names = ["input_features", "input_features_mask"]
        output_names = ["last_hidden_state"]

        S = torch.export.Dim("seq_len", min=8, max=16384)
        dynamic_shapes = {
            "input_features": {
                1: S
            },
            "input_features_mask": {
                1: S
            },
        }
        return args, input_names, output_names, dynamic_shapes


# ---------------------------------------------------------------------------
# Multimodal embedder (RMSNorm no-scale -> Linear projection to text dim)
# ---------------------------------------------------------------------------


class Gemma4AudioMultimodalEmbedder(nn.Module):
    """Project Gemma4 audio encoder output to text hidden size.

    Architecture: RMSNorm(no_scale) -> Linear(output_proj_dims -> text_hidden_size)
    HF weight prefix: ``model.embed_audio.*``
    """

    def __init__(self, audio_config: Dict[str, Any], text_config: Dict[str,
                                                                       Any],
                 model_config: ModelConfig) -> None:
        super().__init__()
        multimodal_hidden_size = int(
            audio_config.get("output_proj_dims", _ac.output_proj_dims))
        text_hidden_size = int(text_config["hidden_size"])
        eps = float(audio_config.get("rms_norm_eps", _RMS_NORM_EPS))
        self.embedding_pre_projection_norm = Gemma4RMSNorm(
            multimodal_hidden_size, eps=eps, with_scale=False)
        self.embedding_projection = make_linear(
            model_config,
            multimodal_hidden_size,
            text_hidden_size,
            bias=False,
            module_name="embed_audio.embedding_projection")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embedding_pre_projection_norm(hidden_states)
        return self.embedding_projection(hidden_states.to(torch.float16))


class Gemma4AudioWithEmbedder(nn.Module):
    """Gemma4 audio encoder plus multimodal projection to text hidden size.

    Output: [batch, encoded_seq_len, text_hidden_size]
    """

    def __init__(self,
                 config: Dict[str, Any],
                 *,
                 model_config: ModelConfig,
                 name_prefix: str = "audio_tower") -> None:
        super().__init__()
        audio_config = config.get("audio_config", config)
        text_config = config.get("text_config") or {}
        if "hidden_size" not in text_config:
            raise ValueError(
                "Gemma4 audio export with embedder requires text_config "
                "with 'hidden_size'")
        self.audio_tower = Gemma4AudioModel(config,
                                            model_config=model_config,
                                            name_prefix=name_prefix)
        self.embed_audio = Gemma4AudioMultimodalEmbedder(
            audio_config, text_config, model_config)

    def forward(self, input_features: torch.Tensor,
                input_features_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.audio_tower(input_features, input_features_mask)
        return self.embed_audio(hidden_states)

    def get_onnx_export_args(self, config: Dict[str, Any], device: str):
        """Return (args, input_names, output_names, dynamic_shapes)."""
        return self.audio_tower.get_onnx_export_args(config, device)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

_CANDIDATE_PREFIXES = (
    "model.audio_tower.",
    "audio_tower.",
    "",
)


def _load_audio_weights(model: nn.Module,
                        weights: Dict[str, torch.Tensor],
                        prefix: Optional[str] = None) -> None:
    """Load checkpoint weights into model.

    Handles both Gemma4AudioModel (audio_tower only) and
    Gemma4AudioWithEmbedder (audio_tower + embed_audio).
    """
    from ...checkpoint.loader import load_submodule_weights

    if isinstance(model, Gemma4AudioWithEmbedder):
        # Combined model: remap HF keys to submodule paths
        # model.audio_tower.* -> audio_tower.*
        # model.embed_audio.* -> embed_audio.*
        def _remap(key: str) -> "str | None":
            for pfx in ("model.", ""):
                candidate = key[len(pfx
                                    ):] if pfx and key.startswith(pfx) else key
                if (candidate.startswith("audio_tower.")
                        or candidate.startswith("embed_audio.")):
                    return candidate
            return None

        load_submodule_weights(model,
                               weights,
                               _remap,
                               label="Gemma4AudioWithEmbedder",
                               log=logger)
    else:
        if prefix is None:
            for cand in _CANDIDATE_PREFIXES:
                if cand == "" or any(
                        k.startswith(cand) for k in weights.keys()):
                    prefix = cand
                    break
            else:
                prefix = ""

        stripped: Dict[str, torch.Tensor] = {}
        for k, v in weights.items():
            if not k.startswith(prefix):
                continue
            stripped[k[len(prefix):]] = v

        load_submodule_weights(
            model,
            stripped,
            key_remap=lambda k: k,
            label="Gemma4AudioModel",
            log=logger,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_gemma4_audio(
    config: Dict[str, Any],
    weights: Dict[str, torch.Tensor],
    dtype: torch.dtype = torch.float16,
    prefix: Optional[str] = None,
    *,
    model_config: ModelConfig,
    name_prefix: str = "audio_tower",
    include_embedder: bool = True,
) -> nn.Module:
    """Build and return a Gemma4 audio model with loaded weights.

    Args:
        config:  Full parsed ``config.json`` dict (must contain ``audio_config``).
        weights: Flat ``{key: tensor}`` dict from safetensors.
        dtype:   Target dtype (default ``float16``).
        prefix:  Checkpoint key prefix to strip. ``None`` = auto-detect.
        model_config: Top-level :class:`ModelConfig` carrying the
                 :class:`QuantConfig` for ``make_linear`` dispatch.
        name_prefix: Module-name prefix for ``make_linear`` module paths.
        include_embedder: If True (default), includes the multimodal embedder
                 (RMSNorm + Linear) that projects to text_hidden_size.
                 Requires ``text_config`` in config. If False, returns raw
                 encoder output at ``output_proj_dims``.

    Returns:
        :class:`Gemma4AudioWithEmbedder` if include_embedder is True,
        otherwise :class:`Gemma4AudioModel`.
    """
    if include_embedder:
        model: nn.Module = Gemma4AudioWithEmbedder(config,
                                                   model_config=model_config,
                                                   name_prefix=name_prefix)
    else:
        model = Gemma4AudioModel(config,
                                 model_config=model_config,
                                 name_prefix=name_prefix)
    model = model.to(dtype=dtype)
    _load_audio_weights(model, weights, prefix)
    # Sync plain-float clipping values from loaded buffers for ONNX export
    for m in model.modules():
        if isinstance(m, Gemma4ClippableLinear):
            m._sync_clipping_vals()
    model.eval()
    return model


__all__ = [
    "Gemma4AudioModel",
    "Gemma4AudioMultimodalEmbedder",
    "Gemma4AudioWithEmbedder",
    "build_gemma4_audio",
]
