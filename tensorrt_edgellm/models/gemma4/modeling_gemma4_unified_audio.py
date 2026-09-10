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
"""Encoder-free Gemma 4 Unified audio embedding export model.

Each input token is a frame of 640 raw 16-kHz PCM samples.  Unlike the
smaller Gemma 4 checkpoints there is no mel frontend, convolutional
subsampling, attention stack, or audio tower.  The complete graph is a
weightless RMSNorm followed by the ``model.embed_audio`` projection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .modeling_gemma4_unified_visual import Gemma4UnifiedMultimodalEmbedder

if TYPE_CHECKING:
    from ...config import ModelConfig

logger = logging.getLogger(__name__)


class Gemma4UnifiedAudioModel(nn.Module):
    """Map framed PCM samples directly into the language-model hidden space."""

    def __init__(self, config: dict, model_config: "ModelConfig") -> None:
        super().__init__()
        audio_config = config.get("audio_config", config)
        text_config = config.get("text_config") or {}
        if "hidden_size" not in text_config:
            raise ValueError(
                "Gemma4 Unified audio export requires text_config.hidden_size")
        audio_embed_dim_value = audio_config.get(
            "audio_embed_dim", audio_config.get("hidden_size"))
        if audio_embed_dim_value is None:
            raise ValueError(
                "Gemma4 Unified audio export requires audio_config."
                "audio_embed_dim")
        audio_embed_dim = int(audio_embed_dim_value)
        # ``Gemma4UnifiedAudioConfig`` does not declare ``output_proj_dims``,
        # so re-serializing the typed config tree drops it from every quantized
        # checkpoint. The fallback is exact, not a guess: the check below
        # rejects any checkpoint where the two differ.
        output_proj_dims = int(
            audio_config.get("output_proj_dims", audio_embed_dim))
        if audio_embed_dim != output_proj_dims:
            raise ValueError(
                "Gemma4 Unified audio requires audio_embed_dim to equal "
                f"output_proj_dims, got {audio_embed_dim} and "
                f"{output_proj_dims}.")
        self.embed_audio = Gemma4UnifiedMultimodalEmbedder(
            audio_config,
            text_config,
            model_config,
            module_name="embed_audio.embedding_projection",
            multimodal_hidden_size=output_proj_dims,
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.embed_audio(input_features)

    def get_onnx_export_args(self, config: dict, device: str):
        """Return framed-PCM I/O and dynamic-shape metadata.

        ONNX bindings:
          - ``input_features``: ``[1, num_frames, audio_embed_dim]`` FP16
          - ``last_hidden_state``: ``[1, num_frames, text_hidden_size]`` FP16
        """
        audio_config = config.get("audio_config", config)
        audio_embed_dim_value = audio_config.get(
            "audio_embed_dim", audio_config.get("hidden_size"))
        if audio_embed_dim_value is None:
            raise ValueError(
                "Gemma4 Unified audio export requires audio_config."
                "audio_embed_dim")
        audio_embed_dim = int(audio_embed_dim_value)
        input_features = torch.zeros(1,
                                     4,
                                     audio_embed_dim,
                                     dtype=torch.float16,
                                     device=device)
        num_frames = torch.export.Dim("num_frames", min=1)
        args = (input_features, )
        input_names = ["input_features"]
        output_names = ["last_hidden_state"]
        dynamic_shapes = {"input_features": {1: num_frames}}
        return args, input_names, output_names, dynamic_shapes


def _load_weights(model: nn.Module, weights: dict) -> None:
    """Strictly load the single Unified audio projection tensor."""
    from ...checkpoint.loader import load_submodule_weights

    def _remap(key: str) -> "str | None":
        candidate = key[len("model."):] if key.startswith("model.") else key
        if candidate.startswith("embed_audio."):
            return candidate
        return None

    mapped_keys = {
        mapped
        for key in weights if (mapped := _remap(key)) is not None
    }
    expected_keys = set(model.state_dict())
    missing = sorted(expected_keys - mapped_keys)
    unexpected = sorted(mapped_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "Gemma4 Unified audio checkpoint schema mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}")

    load_submodule_weights(model,
                           weights,
                           _remap,
                           label="gemma4 unified audio",
                           log=logger)


def build_gemma4_unified_audio(
        config: dict,
        weights: dict,
        model_config: "ModelConfig",
        dtype: torch.dtype = torch.float16) -> Gemma4UnifiedAudioModel:
    """Instantiate the encoder-free audio graph and load checkpoint weights."""
    model = Gemma4UnifiedAudioModel(config, model_config=model_config)
    model.to(dtype)
    _load_weights(model, weights)
    model.eval()
    return model


__all__ = [
    "Gemma4UnifiedAudioModel",
    "build_gemma4_unified_audio",
]
