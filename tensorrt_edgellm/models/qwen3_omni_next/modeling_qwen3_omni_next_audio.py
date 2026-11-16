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
"""Qwen3-Next Omni audio encoder — adds one extra Conv2d downsample stage."""
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..qwen3_asr import modeling_qwen3_asr_audio as qwen3_asr_audio
from ..qwen3_asr.modeling_qwen3_asr_audio import QwenAudioEncoder

if TYPE_CHECKING:
    from ...config import ModelConfig

__all__ = [
    "Qwen3OmniNextAudioEncoder",
    "build_qwen3_omni_next_audio",
]


class Qwen3OmniNextAudioEncoder(QwenAudioEncoder):
    """:class:`QwenAudioEncoder` with a 4th stride-2 Conv2d (8x downsample)."""

    def __init__(
        self,
        num_mel_bins: int = qwen3_asr_audio._NUM_MEL_BINS,
        d_model: int = qwen3_asr_audio._D_MODEL,
        num_layers: int = qwen3_asr_audio._NUM_LAYERS,
        num_heads: int = qwen3_asr_audio._NUM_HEADS,
        ffn_dim: int = qwen3_asr_audio._FFN_DIM,
        max_source_positions: int = qwen3_asr_audio._MAX_SOURCE_POSITIONS,
        output_dim: int = qwen3_asr_audio._OUTPUT_DIM,
        downsample_hidden: int = qwen3_asr_audio._DOWNSAMPLE_HIDDEN,
        *,
        attention_scale: float,
        model_config: "ModelConfig",
        name_prefix: str = "audio_tower",
    ) -> None:
        super().__init__(
            num_mel_bins=num_mel_bins,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            max_source_positions=max_source_positions,
            output_dim=output_dim,
            downsample_hidden=downsample_hidden,
            attention_scale=attention_scale,
            model_config=model_config,
            name_prefix=name_prefix,
        )
        self.conv2d4 = nn.Conv2d(downsample_hidden,
                                 downsample_hidden,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1)
        # Recompute freq_bins with one extra halving and resize conv_out.
        freq_bins_4 = (((((num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2 + 1) //
                       2)
        self.conv_out = qwen3_asr_audio.make_linear(
            model_config,
            downsample_hidden * freq_bins_4,
            d_model,
            bias=False,
            module_name=f"{name_prefix}.conv_out" if name_prefix else "")

    def get_onnx_export_args(self, config: dict, device: str):
        """Return (dynamo_inputs, onnx_input_names, output_names, dynamic_shapes) for ONNX export.

        Overrides the qwen3-asr / Qwen3-Omni 3-stage variant to account for
        the extra stride-2 Conv2d (16x downsample instead of 8x).
        """
        # Reuse the parent implementation and rescale ``t_out`` -- the parent
        # bakes ``t_out = n_window * 2 // 8`` (3-stage stack); the 4-stage
        # stack here needs ``// 16``. Everything else (mel bins, num_chunks,
        # dynamic shapes) is identical.
        args = super().get_onnx_export_args(config, device)
        dynamo_inputs, onnx_input_names, output_names, dynamic_shapes = (
            args if len(args) == 4 else args[:4])
        num_mel_bins = config.get("num_mel_bins", 128)
        n_window = config.get("n_window", 100)
        num_chunks = 3
        t_out = max(n_window * 2 // 16, 2)
        num_attention_elems = num_chunks * t_out - 1
        dynamo_inputs = dict(dynamo_inputs)
        dynamo_inputs["padded_feature"] = torch.zeros(num_chunks,
                                                      num_mel_bins,
                                                      n_window * 2,
                                                      dtype=torch.float16,
                                                      device=device)
        dynamo_inputs["padded_mask_after_cnn_indices"] = torch.zeros(
            num_attention_elems, 2, dtype=torch.int64, device=device)
        if "attention_mask" in dynamo_inputs:
            dynamo_inputs["attention_mask"] = torch.zeros(num_attention_elems,
                                                          num_attention_elems,
                                                          dtype=torch.float16,
                                                          device=device)
        if "cu_seqlens" in dynamo_inputs:
            dynamo_inputs["cu_seqlens"] = torch.tensor(
                [0, t_out, 2 * t_out, num_attention_elems],
                dtype=torch.int32,
                device=device)
            dynamo_inputs["max_seqlen_carrier"] = torch.zeros(
                t_out, dtype=torch.int32, device=device)
        return dynamo_inputs, onnx_input_names, output_names, dynamic_shapes

    def forward(
        self,
        padded_feature: torch.Tensor,
        padded_mask_after_cnn_indices: torch.Tensor,
        attention_mask: "torch.Tensor | None" = None,
        cu_seqlens: "torch.Tensor | None" = None,
        max_seqlen_carrier: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        x = padded_feature.unsqueeze(1)  # [C, 1, mel, t]
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        x = F.gelu(self.conv2d4(x))

        b, c, f, t = x.shape
        x = self.conv_out(x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        pos = self.positional_embedding.positional_embedding[:x.shape[
            1], :].unsqueeze(0).to(x.dtype)  # type: ignore[attr-defined]
        x = x + pos

        hidden_states = x[padded_mask_after_cnn_indices[:, 0],
                          padded_mask_after_cnn_indices[:, 1]]

        for layer in self.layers:
            hidden_states = layer(hidden_states,
                                  attention_mask,
                                  cu_seqlens,
                                  max_seqlen_carrier=max_seqlen_carrier)

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states


def build_qwen3_omni_next_audio(
    config: dict,
    weights: dict,
    dtype: torch.dtype = torch.float16,
    prefix: "str | None" = None,
    *,
    model_config: "ModelConfig",
    name_prefix: str = "audio_tower",
) -> Qwen3OmniNextAudioEncoder:
    audio_cfg = config.get("audio_config", config)

    def _get(key: str, default):
        return audio_cfg.get(key, config.get(key, default))

    d_model = _get("d_model", qwen3_asr_audio._D_MODEL)
    num_heads = _get("encoder_attention_heads", qwen3_asr_audio._NUM_HEADS)
    head_dim = d_model // num_heads
    attention_scale = qwen3_asr_audio.config_module._get_attention_scaling(
        audio_cfg, head_dim, 1.0 / (float(head_dim)**0.5))

    model = Qwen3OmniNextAudioEncoder(
        num_mel_bins=_get("num_mel_bins", qwen3_asr_audio._NUM_MEL_BINS),
        d_model=d_model,
        num_layers=_get("encoder_layers", qwen3_asr_audio._NUM_LAYERS),
        num_heads=num_heads,
        ffn_dim=_get("encoder_ffn_dim", qwen3_asr_audio._FFN_DIM),
        max_source_positions=_get("max_source_positions",
                                  qwen3_asr_audio._MAX_SOURCE_POSITIONS),
        output_dim=_get("output_dim", qwen3_asr_audio._OUTPUT_DIM),
        downsample_hidden=_get("downsample_hidden_size",
                               qwen3_asr_audio._DOWNSAMPLE_HIDDEN),
        attention_scale=attention_scale,
        model_config=model_config,
        name_prefix=name_prefix,
    )
    # Cast before loading — _set_tensor overwrites buffers in place and
    # preserves FP8 dtypes; casting after load silently downgrades them.
    model = model.to(dtype=dtype)
    qwen3_asr_audio._load_audio_weights(model, weights, prefix)
    model.eval()
    return model
