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
"""Gemma4 Unified encoder-free framed-PCM audio graph."""

import numpy as np
import tensorrt as trt

from ...ops import Linear, Module, NetworkModule
from ...ops import functional as F


class Gemma4UnifiedAudioEmbedder(Module):
    """Weightless RMSNorm and projection into the text hidden space."""

    def __init__(self, ctx, config: dict) -> None:
        super().__init__(ctx, "embed_audio")
        hidden_size = int(
            config.get("audio_embed_dim", config.get("hidden_size", 0)))
        output_size = int(config.get("output_proj_dims", hidden_size))
        if hidden_size <= 0 or hidden_size != output_size:
            raise ValueError(
                "Gemma4 Unified audio_embed_dim must equal output_proj_dims")
        self.hidden_size = hidden_size
        self.eps = float(config.get("rms_norm_eps", 1e-6))
        self.projection = Linear(ctx,
                                 self.key("embedding_projection"),
                                 tensor_parallel=False)

    def forward(self, hidden):
        weight = np.ones(self.hidden_size, dtype=np.float16)
        hidden = F.rms_norm(hidden, weight, self.eps, hidden.rank)
        return self.projection(hidden)


class Gemma4UnifiedAudioModel(NetworkModule):
    """Framed raw PCM samples to language-model embeddings."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle.root)

    def __init__(self, ctx, root: dict) -> None:
        super().__init__(ctx)
        self.embed_audio = Gemma4UnifiedAudioEmbedder(
            ctx,
            root.get("audio_config") or root)

    def input_tensors(self):
        return {
            "features":
            self.add_input("input_features", trt.float16,
                           (1, -1, self.embed_audio.hidden_size))
        }

    def forward(self, features):
        return {"last_hidden_state": self.embed_audio(features)}
