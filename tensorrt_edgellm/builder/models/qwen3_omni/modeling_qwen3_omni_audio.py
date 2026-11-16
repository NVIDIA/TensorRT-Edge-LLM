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
"""Qwen3-Omni audio component and engine I/O contract."""

import tensorrt as trt

from ...core import contracts
from ...ops import (AudioConvSubsampler, AudioPositionEmbedding,
                    AudioTransformer, LayerNorm, Linear, NetworkModule)
from ...ops import functional as F


class Qwen3OmniAudioTransformer(AudioTransformer):
    """Qwen3-Omni audio stack with model-specific activation and clamping."""

    def __init__(self, ctx, config, activation: str) -> None:
        super().__init__(ctx, config, activation=activation, clamp_fp16=True)


class Qwen3OmniAudioEncoder(NetworkModule):
    """End-to-end Qwen3-Omni audio encoder graph."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle)

    def __init__(self, ctx, bundle) -> None:
        super().__init__(ctx, "audio")
        self.config = bundle.component_dict(contracts.Component.AUDIO)
        self.activation = str(self.config.get("activation_function", "gelu"))
        self.conv_subsampler = AudioConvSubsampler(ctx)
        self.positions = AudioPositionEmbedding(ctx, self.config)
        self.transformer = Qwen3OmniAudioTransformer(ctx, self.config,
                                                     self.activation)
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
        mel_bins = int(self.config.get("num_mel_bins", 128))
        n_window = int(self.config.get("n_window", 100))
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
