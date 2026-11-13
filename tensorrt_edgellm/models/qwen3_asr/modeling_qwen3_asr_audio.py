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
From-scratch Qwen3-ASR / Qwen3-Omni audio encoder implementation.

Architecture (Whisper-style encoder):
    CNN stack:  conv2d1 → conv2d2 → conv2d3  (depthwise downsampling)
    conv_out:   Linear(downsample_hidden_size * freq_bins, d_model)
    positional_embedding:  SinusoidsPositionEmbedding (fixed, non-learned)
    layers:     32 × QwenAudioEncoderLayer
        self_attn_layer_norm, self_attn (q/k/v_proj, out_proj), final_layer_norm, fc1, fc2
    ln_post:    LayerNorm
    proj1, act, proj2:  output projection

Checkpoint weight key prefix:
    Qwen3-ASR standalone:  ``audio_tower.*``   (or no prefix if a pure audio checkpoint)
    Qwen3-Omni / Qwen3-TTS embedded:  ``thinker.audio_tower.*`` or ``audio_tower.*``

ONNX Forward I/O:
    Inputs:
        padded_feature                 [num_chunks, num_mel_bins, n_window*2]  float16
        padded_mask_after_cnn_indices  [num_attention_elems, 2]               int64
        cu_seqlens                     [num_windows+1]                         int32
        max_seqlen_carrier             [max_window_len]            int32, shape only
    Output:
        last_hidden_state              [num_attention_elems, output_dim]       float16

Attention runs on ViTAttentionPlugin over the chunk windows ``cu_seqlens``
describes. ``USE_AUDIO_ATTN_PLUGIN=0`` exports the dense graph instead, which
takes ``attention_mask`` [num_attention_elems, num_attention_elems] float16 in
place of the two ragged inputs. That mask is block diagonal over the same
windows, so both graphs compute the same attention.

The ``padded_mask_after_cnn_indices`` approach avoids non-zero ONNX nodes that are
TensorRT-unfriendly. The C++ runtime computes this tensor during pre-processing.

All Linear layers route through ``make_linear`` so the same modeling tree
handles FP16, FP8 (and future NVFP4) checkpoints: ``make_linear`` reads
the supplied :class:`ModelConfig`'s :class:`QuantConfig` and returns
``FP16Linear`` / ``FP8Linear`` / ... so the Linear emits the right Q/DQ
ops in its forward. ``module_name`` is the strip-``model.`` canonical
path (e.g. ``audio_tower.layers.0.self_attn.q_proj``) so it matches the
keys ``hf_quant_config.json``'s ``exclude_modules`` / ``layer_overrides``
use after :func:`tensorrt_edgellm.config._normalize_module_name`.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import config as config_module
from .. import ops
from ..linear import make_linear

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default architecture constants (Qwen3-ASR / Qwen3-Omni)
# ---------------------------------------------------------------------------

_D_MODEL = 1280
_NUM_LAYERS = 32
_NUM_HEADS = 20
_FFN_DIM = 5120
_NUM_MEL_BINS = 128
_MAX_SOURCE_POSITIONS = 1500
_OUTPUT_DIM = 3584  # LLM hidden size (Qwen3-7.5B)
_DOWNSAMPLE_HIDDEN = 480
_N_WINDOW = 50  # Both 0.6B and 1.7B use 50; runtime overrides from HF config.

# ---------------------------------------------------------------------------
# Sinusoidal positional embedding (fixed, not learned)
# ---------------------------------------------------------------------------


class SinusoidsPositionEmbedding(nn.Module):
    """Fixed sinusoidal position encoding as used in Whisper / Qwen3-ASR.

    Checkpoint buffer: ``positional_embedding``  [max_source_positions, d_model]
    """

    def __init__(self,
                 length: int = _MAX_SOURCE_POSITIONS,
                 channels: int = _D_MODEL,
                 max_timescale: int = 10000) -> None:
        super().__init__()
        assert channels % 2 == 0
        log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment *
            torch.arange(channels // 2, dtype=torch.float32))
        scaled_time = torch.arange(
            length,
            dtype=torch.float32).unsqueeze(1) * inv_timescales.unsqueeze(0)
        pos_emb = torch.cat([torch.sin(scaled_time),
                             torch.cos(scaled_time)],
                            dim=1)
        self.register_buffer("positional_embedding", pos_emb, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        return self.positional_embedding[:seqlen, :]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class QwenAudioAttention(nn.Module):
    """Multi-head self-attention for the audio encoder.

    Checkpoint keys (under encoder prefix + ``layers.N.self_attn``):
        q_proj.{weight,bias}, k_proj.{weight,bias},
        v_proj.{weight,bias}, out_proj.{weight,bias}

    Runtime module layout:
        qkv.{weight,bias}, out_proj.{weight,bias}
    """

    def __init__(self,
                 model_config: config_module.ModelConfig,
                 attention_scale: float,
                 d_model: int = _D_MODEL,
                 num_heads: int = _NUM_HEADS,
                 name_prefix: str = "") -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attention_scale = attention_scale
        self.q_proj = make_linear(
            model_config,
            d_model,
            d_model,
            bias=True,
            module_name=f"{name_prefix}.q_proj" if name_prefix else "")
        self.k_proj = make_linear(
            model_config,
            d_model,
            d_model,
            bias=True,
            module_name=f"{name_prefix}.k_proj" if name_prefix else "")
        self.v_proj = make_linear(
            model_config,
            d_model,
            d_model,
            bias=True,
            module_name=f"{name_prefix}.v_proj" if name_prefix else "")
        self.out_proj = make_linear(
            model_config,
            d_model,
            d_model,
            bias=True,
            module_name=f"{name_prefix}.out_proj" if name_prefix else "")
        self._use_attn_plugin = ops.is_audio_attention_plugin_enabled()
        # Both paths claim cu_seqlens, so they cannot share one exported graph.
        self._use_trt_attn = (ops.is_trt_native_attention_enabled()
                              and not self._use_attn_plugin)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            cu_seqlens: Optional[torch.Tensor] = None,
            kv_lengths: Optional[torch.Tensor] = None,
            max_seqlen_carrier: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [T, d_model] — ragged sequence (all chunks concatenated)
            attention_mask: [T, T] additive mask (0 = attend, -inf = ignore)
            cu_seqlens: [num_windows+1] int32 window boundaries (plugin) or [batch+1] (TRT)
            kv_lengths: [batch+1] int32 — TRT path only
            max_seqlen_carrier: [max_seqlen] int32, shape-only — plugin path only
        """
        T = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(T, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(T, self.num_heads, self.head_dim)

        if self._use_attn_plugin:
            out = ops.vit_attention_plugin(
                q.to(torch.float16),
                k.to(torch.float16),
                v.to(torch.float16),
                cu_seqlens,
                max_seqlen_carrier,
                num_heads=self.num_heads,
                head_size=self.head_dim,
                attention_scale=self.attention_scale,
            ).reshape(T, -1)
        elif self._use_trt_attn:
            # TODO: Enable these paths when supported
            raise RuntimeError(
                "Qwen3-ASR TRT-native attention is currently not supported.")
            """
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

            out = ops.trt_ragged_attention(
                q,
                k,
                v,
                cu_seqlens,
                kv_lengths,
                num_heads=self.num_heads,
                head_size=self.head_dim,
                attention_scale=self.attention_scale,
                mask=attention_mask,
            )
            # attn_output: [T, num_heads, head_dim] → [T, num_heads * head_dim]
            out = out.reshape(T, -1)
            """
        else:
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            # q/k/v: [num_heads, T, head_dim]
            # Explicit softmax attention (avoids SDPA op which TRT ONNX parser rejects).
            # scores: [num_heads, T, T]
            scores = torch.matmul(q, k.transpose(-2, -1))
            if self.attention_scale != 1.0:
                scores = scores * self.attention_scale
            # attention_mask: [T, T] → [1, T, T] (broadcast over heads)
            scores = scores + attention_mask.unsqueeze(0)
            attn_weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            out = torch.matmul(attn_weights, v)
            # out: [num_heads, T, head_dim] → [T, num_heads * head_dim]
            out = out.transpose(0, 1).reshape(T, -1)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Encoder Layer
# ---------------------------------------------------------------------------


class QwenAudioEncoderLayer(nn.Module):
    """Single Qwen3-ASR encoder block (pre-norm).

    Checkpoint keys (under encoder prefix + ``layers.N``):
        self_attn_layer_norm.{weight,bias}
        self_attn.*
        final_layer_norm.{weight,bias}
        fc1.{weight,bias}
        fc2.{weight,bias}
    """

    def __init__(self,
                 model_config: config_module.ModelConfig,
                 attention_scale: float,
                 d_model: int = _D_MODEL,
                 num_heads: int = _NUM_HEADS,
                 ffn_dim: int = _FFN_DIM,
                 name_prefix: str = "") -> None:
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = QwenAudioAttention(
            model_config,
            attention_scale,
            d_model,
            num_heads,
            name_prefix=f"{name_prefix}.self_attn" if name_prefix else "")
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = make_linear(
            model_config,
            d_model,
            ffn_dim,
            bias=True,
            module_name=f"{name_prefix}.fc1" if name_prefix else "")
        self.fc2 = make_linear(
            model_config,
            ffn_dim,
            d_model,
            bias=True,
            module_name=f"{name_prefix}.fc2" if name_prefix else "")

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            cu_seqlens: Optional[torch.Tensor] = None,
            kv_lengths: Optional[torch.Tensor] = None,
            max_seqlen_carrier: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask,
                                       cu_seqlens, kv_lengths,
                                       max_seqlen_carrier)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = F.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_val = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states,
                                        min=-clamp_val,
                                        max=clamp_val)
        return hidden_states


# ---------------------------------------------------------------------------
# Full audio encoder
# ---------------------------------------------------------------------------


class QwenAudioEncoder(nn.Module):
    """Complete Qwen3-ASR / Qwen3-Omni audio encoder.

    ONNX forward:
        padded_feature                 [C, num_mel_bins, n_window*2]  float16
        padded_mask_after_cnn_indices  [T, 2]                        int64
        cu_seqlens                     [num_windows+1]               int32
        max_seqlen_carrier             [max_window_len]              int32
        → last_hidden_state            [T, output_dim]               float16

    With ``USE_AUDIO_ATTN_PLUGIN=0`` the two ragged inputs are replaced by
    ``attention_mask`` [T, T] float16.

    Checkpoint keys are directly under the constructor prefix stripped by
    :func:`build_qwen_audio`. The instance attribute ``config`` exposes the
    chunking knobs (``n_window`` / ``num_mel_bins`` / etc.) that host-side
    preprocessing needs to mirror the C++ runtime.
    """

    def __init__(self,
                 num_mel_bins: int = _NUM_MEL_BINS,
                 d_model: int = _D_MODEL,
                 num_layers: int = _NUM_LAYERS,
                 num_heads: int = _NUM_HEADS,
                 ffn_dim: int = _FFN_DIM,
                 max_source_positions: int = _MAX_SOURCE_POSITIONS,
                 output_dim: int = _OUTPUT_DIM,
                 downsample_hidden: int = _DOWNSAMPLE_HIDDEN,
                 *,
                 attention_scale: float,
                 model_config: config_module.ModelConfig,
                 name_prefix: str = "audio_tower") -> None:
        super().__init__()
        self.config: Dict[str, Any] = {
            "num_mel_bins": num_mel_bins,
            "d_model": d_model,
            "encoder_layers": num_layers,
            "encoder_attention_heads": num_heads,
            "encoder_ffn_dim": ffn_dim,
            "max_source_positions": max_source_positions,
            "output_dim": output_dim,
            "downsample_hidden_size": downsample_hidden,
        }

        # CNN downsampling stack
        self.conv2d1 = nn.Conv2d(1,
                                 downsample_hidden,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1)
        self.conv2d2 = nn.Conv2d(downsample_hidden,
                                 downsample_hidden,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1)
        self.conv2d3 = nn.Conv2d(downsample_hidden,
                                 downsample_hidden,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1)
        # Compute frequency bins after 3 stride-2 convolutions
        freq_bins = ((((num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2)
        self.conv_out = make_linear(
            model_config,
            downsample_hidden * freq_bins,
            d_model,
            bias=False,
            module_name=f"{name_prefix}.conv_out" if name_prefix else "")
        self.positional_embedding = SinusoidsPositionEmbedding(
            max_source_positions, d_model)
        self.layers = nn.ModuleList([
            QwenAudioEncoderLayer(
                model_config,
                attention_scale,
                d_model,
                num_heads,
                ffn_dim,
                name_prefix=f"{name_prefix}.layers.{i}" if name_prefix else "")
            for i in range(num_layers)
        ])
        self._use_attn_plugin = ops.is_audio_attention_plugin_enabled()
        self._use_trt_attn = (ops.is_trt_native_attention_enabled()
                              and not self._use_attn_plugin)
        self.ln_post = nn.LayerNorm(d_model)
        self.proj1 = make_linear(
            model_config,
            d_model,
            d_model,
            bias=True,
            module_name=f"{name_prefix}.proj1" if name_prefix else "")
        self.proj2 = make_linear(
            model_config,
            d_model,
            output_dim,
            bias=True,
            module_name=f"{name_prefix}.proj2" if name_prefix else "")

    def forward(
            self,
            padded_feature: torch.Tensor,
            padded_mask_after_cnn_indices: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            cu_seqlens: Optional[torch.Tensor] = None,
            kv_lengths: Optional[torch.Tensor] = None,
            max_seqlen_carrier: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            padded_feature: [num_chunks, num_mel_bins, n_window*2]
            padded_mask_after_cnn_indices: [num_attention_elems, 2]  int64
            attention_mask: [num_attention_elems, num_attention_elems]  float16
            cu_seqlens: [num_windows+1] int32 window boundaries (plugin) or [batch+1] (TRT)
            kv_lengths: [batch+1] int32 — TRT path only
            max_seqlen_carrier: [max_seqlen] int32, shape-only — plugin path only
        Returns:
            [num_attention_elems, output_dim]
        """
        x = padded_feature.unsqueeze(1)  # [C, 1, mel, t]
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))

        # x: [C, D, freq, T_out]
        b, c, f, t = x.shape
        x = self.conv_out(x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
        # x: [C, T_out, d_model]

        # Add positional encoding
        pos = self.positional_embedding.positional_embedding[:x.shape[
            1], :].unsqueeze(0).to(x.dtype)  # type: ignore[attr-defined]
        x = x + pos

        # Gather valid tokens using pre-computed nonzero indices
        hidden_states = x[padded_mask_after_cnn_indices[:, 0],
                          padded_mask_after_cnn_indices[:, 1]]  # [T, d_model]

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, cu_seqlens,
                                  kv_lengths, max_seqlen_carrier)

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states

    def get_onnx_export_args(self, config: dict, device: str):
        """Return (dynamo_inputs, onnx_input_names, output_names, dynamic_shapes) for ONNX export."""
        num_mel_bins = config.get("num_mel_bins", 128)
        n_window = config.get("n_window", 100)
        num_chunks = 3
        t_out = n_window * 2 // 8  # after 3× stride-2 CNN layers
        num_attention_elems = num_chunks * t_out - 1

        padded_feature = torch.zeros(num_chunks,
                                     num_mel_bins,
                                     n_window * 2,
                                     dtype=torch.float16,
                                     device=device)
        padded_mask_after_cnn_indices = torch.zeros(num_attention_elems,
                                                    2,
                                                    dtype=torch.int64,
                                                    device=device)
        onnx_input_names = [
            "padded_feature",
            "padded_mask_after_cnn_indices",
        ]
        dynamo_inputs = {
            "padded_feature": padded_feature,
            "padded_mask_after_cnn_indices": padded_mask_after_cnn_indices,
        }

        output_names = ["last_hidden_state"]

        T = torch.export.Dim("num_attention_elems")
        dynamic_shapes = {
            "padded_feature": {
                0: torch.export.Dim("num_chunks")
            },
            "padded_mask_after_cnn_indices": {
                0: T
            },
        }

        if not self._use_attn_plugin:
            onnx_input_names.append("attention_mask")
            dynamo_inputs["attention_mask"] = torch.zeros(num_attention_elems,
                                                          num_attention_elems,
                                                          dtype=torch.float16,
                                                          device=device)
            dynamic_shapes["attention_mask"] = {0: T, 1: T}
        else:
            # Block-diagonal windows expressed as segment boundaries. The carrier
            # is shape-only: the plugin reads its length as the max segment length.
            onnx_input_names.extend(["cu_seqlens", "max_seqlen_carrier"])
            dynamo_inputs["cu_seqlens"] = torch.tensor(
                [0, t_out, 2 * t_out, num_attention_elems],
                dtype=torch.int32,
                device=device)
            dynamo_inputs["max_seqlen_carrier"] = torch.zeros(
                t_out, dtype=torch.int32, device=device)
            dynamic_shapes["cu_seqlens"] = {
                0: torch.export.Dim("num_windows_p1")
            }
            dynamic_shapes["max_seqlen_carrier"] = {
                0: torch.export.Dim("max_window_len")
            }

        if self._use_trt_attn:
            onnx_input_names.extend(["cu_seqlens", "kv_lengths"])
            cu_seqlens = torch.tensor([0, num_attention_elems],
                                      dtype=torch.int32,
                                      device=device)
            kv_lengths = torch.tensor([0, num_attention_elems],
                                      dtype=torch.int32,
                                      device=device)

            dynamo_inputs["cu_seqlens"] = cu_seqlens
            dynamo_inputs["kv_lengths"] = kv_lengths
            dynamic_shapes["cu_seqlens"] = {0: torch.export.Dim("batch_p1")}
            dynamic_shapes["kv_lengths"] = {0: torch.export.Dim("kv_batch_p1")}
        return dynamo_inputs, onnx_input_names, output_names, dynamic_shapes


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

_CANDIDATE_PREFIXES = (
    "audio_tower.",
    "thinker.audio_tower.",
    "model.audio_tower.",
    "",  # direct keys (pure audio checkpoint)
)


def _load_audio_weights(model: QwenAudioEncoder,
                        weights: Dict[str, torch.Tensor],
                        prefix: Optional[str] = None) -> None:
    """Load safetensors weights into *model*, stripping *prefix*.

    Uses :func:`load_submodule_weights` so quantized buffers (FP8 / packed
    int8 / ...) keep their original dtype — ``nn.Module.load_state_dict``'s
    ``Tensor.copy_()`` silently casts FP8 → FP16, defeating the point of
    FP8 buffers on ``FP8Linear`` modules. The per-module dispatch
    (``FP16Linear`` vs ``FP8Linear``) is decided by ``make_linear`` reading
    the same ``hf_quant_config.json`` that produced the safetensors, so
    incoming tensor dtypes always match their target buffers' dtypes -- we
    just hand the dict straight to the loader.
    """
    from ...checkpoint.loader import load_submodule_weights

    if prefix is None:
        for cand in _CANDIDATE_PREFIXES:
            if cand == "" or any(k.startswith(cand) for k in weights.keys()):
                prefix = cand
                break
        else:
            prefix = ""

    stripped: Dict[str, torch.Tensor] = {}
    for k, v in weights.items():
        if not k.startswith(prefix):
            continue
        stripped[k[len(prefix):]] = v

    def _identity_remap(key: str) -> Optional[str]:
        return key

    load_submodule_weights(
        model,
        stripped,
        key_remap=_identity_remap,
        label="QwenAudioEncoder",
        log=logger,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_qwen_audio(
    config: Dict[str, Any],
    weights: Dict[str, torch.Tensor],
    dtype: torch.dtype = torch.float16,
    prefix: Optional[str] = None,
    *,
    model_config: config_module.ModelConfig,
    name_prefix: str = "audio_tower",
) -> QwenAudioEncoder:
    """Build and return a :class:`QwenAudioEncoder` with loaded weights.

    Args:
        config:  Model config dict; recognized keys mirror
                 ``Qwen3ASRAudioEncoderConfig`` field names. May be a
                 top-level qwen3_asr ``config.json`` (``audio_config``
                 looked up under ``thinker_config`` or directly), or an
                 audio-only sub-config.
        weights: Flat ``{key: tensor}`` dict from safetensors.
        dtype:   Target dtype for FP16 parameters (default ``float16``).
                 Quantized buffers (FP8 / NVFP4) are loaded as-is.
        prefix:  Checkpoint key prefix to strip. ``None`` = auto-detect.
        model_config: Top-level :class:`ModelConfig` carrying the
                 :class:`QuantConfig` for ``make_linear`` dispatch.
        name_prefix: Module-name prefix passed to ``make_linear`` so
                 generated ``module_name`` strings match the canonical
                 strip-``model.`` paths used in
                 ``hf_quant_config.json::exclude_modules`` /
                 ``layer_overrides``.
    """
    thinker_cfg = config.get("thinker_config", {})
    audio_cfg = (thinker_cfg.get("audio_config") or config.get("audio_config")
                 or config)

    def _get(key: str, default: Any) -> Any:
        return audio_cfg.get(key, config.get(key, default))

    d_model = _get("d_model", _D_MODEL)
    num_heads = _get("encoder_attention_heads", _NUM_HEADS)
    head_dim = d_model // num_heads
    attention_scale = config_module._get_attention_scaling(
        audio_cfg, head_dim, 1.0 / (float(head_dim)**0.5))

    model = QwenAudioEncoder(
        num_mel_bins=_get("num_mel_bins", _NUM_MEL_BINS),
        d_model=d_model,
        num_layers=_get("encoder_layers", _NUM_LAYERS),
        num_heads=num_heads,
        ffn_dim=_get("encoder_ffn_dim", _FFN_DIM),
        max_source_positions=_get("max_source_positions",
                                  _MAX_SOURCE_POSITIONS),
        output_dim=_get("output_dim", _OUTPUT_DIM),
        downsample_hidden=_get("downsample_hidden_size", _DOWNSAMPLE_HIDDEN),
        attention_scale=attention_scale,
        model_config=model_config,
        name_prefix=name_prefix,
    )
    # Stash chunking knobs on the model so host-side helpers can read them
    # without separately threading the config through.
    model.config["n_window"] = _get("n_window", _N_WINDOW)
    # 0.6B and 1.7B both ship n_window_infer = 800 = 16 * 50; runtime
    # always overrides this from HF config, the default is just for
    # tests that build the encoder from scratch.
    model.config["n_window_infer"] = _get("n_window_infer", _N_WINDOW * 16)

    # Cast FP16 components (LayerNorm / Conv2d / FP16Linear) to ``dtype``
    # *before* loading weights. load_submodule_weights' _set_tensor
    # overwrites buffers in-place (preserving FP8 dtype on FP8Linear); if
    # we cast after load, .to(fp16) would silently downgrade FP8 buffers
    # that have just been assigned.
    model = model.to(dtype=dtype)
    _load_audio_weights(model, weights, prefix)
    model.eval()
    return model


__all__ = [
    "SinusoidsPositionEmbedding",
    "QwenAudioAttention",
    "QwenAudioEncoderLayer",
    "QwenAudioEncoder",
    "build_qwen_audio",
]
