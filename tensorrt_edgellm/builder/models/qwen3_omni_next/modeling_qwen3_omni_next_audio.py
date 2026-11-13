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
"""Checkpoint-direct Qwen3-Omni-Next audio encoder."""

import tensorrt as trt

from ...core import contracts
from ...ops import (AudioPositionEmbedding, AudioTransformer, LayerNorm,
                    Linear, Module, NetworkModule)
from ...ops import functional as F
from ...ops.audio import AudioConv2d

__all__ = [
    "Qwen3OmniNextAudioConvSubsampler",
    "Qwen3OmniNextAudioTransformer",
    "Qwen3OmniNextAudioEncoder",
]


class Qwen3OmniNextAudioConvSubsampler(Module):
    """Four-stage stride-2 frontend unique to the Next audio tower."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx, "audio")
        self.convolutions = [
            AudioConv2d(
                ctx,
                self.weights.find_suffix(f"conv2d{index}.weight",
                                         "audio")[:-len(".weight")])
            for index in (1, 2, 3, 4)
        ]
        output = self.weights.find_suffix("conv_out.weight",
                                          "audio")[:-len(".weight")]
        self.conv_out = Linear(ctx, output, rank=3, tensor_parallel=False)

    def forward(self, features):
        hidden_states = features.unsqueeze(1, 3)
        for convolution in self.convolutions:
            hidden_states = convolution(hidden_states)
        hidden_states = hidden_states.transpose((0, 3, 1, 2)).reshape(
            (0, 0, -1))
        return self.conv_out(hidden_states)


class Qwen3OmniNextAudioTransformer(AudioTransformer):
    """Next audio stack with provider activation and FP16 clamping."""

    def __init__(self, ctx, audio_config: dict, activation: str) -> None:
        super().__init__(ctx,
                         audio_config,
                         activation=activation,
                         clamp_fp16=True)


class Qwen3OmniNextAudioEncoder(NetworkModule):
    """End-to-end audio component with a 16x convolutional time reduction."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle)

    def __init__(self, ctx, bundle) -> None:
        super().__init__(ctx, "audio")
        self.audio_config = bundle.component_dict(contracts.Component.AUDIO)
        self.activation = str(
            self.audio_config.get("activation_function", "gelu"))
        self.conv_subsampler = Qwen3OmniNextAudioConvSubsampler(ctx)
        self.positions = AudioPositionEmbedding(ctx, self.audio_config)
        self.transformer = Qwen3OmniNextAudioTransformer(
            ctx, self.audio_config, self.activation)
        ln_post = self.weights.find_suffix("ln_post.weight",
                                           "audio")[:-len(".weight")]
        proj1 = self.weights.find_suffix("proj1.weight",
                                         "audio")[:-len(".weight")]
        proj2 = self.weights.find_suffix("proj2.weight",
                                         "audio")[:-len(".weight")]
        self.ln_post = LayerNorm(ctx, ln_post, 1e-5, 2)
        self.proj1 = Linear(ctx, proj1, rank=2, tensor_parallel=False)
        self.proj2 = Linear(ctx, proj2, rank=2, tensor_parallel=False)

    def input_tensors(self):
        mel_bins = int(self.audio_config.get("num_mel_bins", 128))
        n_window = int(self.audio_config.get("n_window", 100))
        return {
            "features":
            self.add_input("padded_feature", trt.float16,
                           (-1, mel_bins, n_window * 2)),
            "indices":
            self.add_input("padded_mask_after_cnn_indices", trt.int64,
                           (-1, 2)),
            "attention_mask":
            self.add_input("attention_mask", trt.float16, (-1, -1)),
        }

    def forward(self, features, indices, attention_mask):
        hidden_states = self.conv_subsampler(features)
        hidden_states = self.positions(hidden_states)
        hidden_states = F.gather_nd(hidden_states, indices)
        hidden_states = self.transformer(hidden_states, attention_mask)
        hidden_states = self.proj1(self.ln_post(hidden_states)).activation(
            self.activation)
        return {"last_hidden_state": self.proj2(hidden_states)}
