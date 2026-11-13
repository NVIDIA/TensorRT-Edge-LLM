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
"""Qwen code2wav checkpoint-direct graph."""

import numpy as np
import tensorrt as trt

from ...core import contracts
from ...ops import Module, NetworkModule
from ...ops import functional as F
from ...ops.code2wav import QwenCode2WavPreTransformer, QwenCode2WavVocoder


class Qwen3OmniCodeEmbedding(Module):
    """Qwen3-Omni codebook frontend before the shared Code2Wav decoder."""

    def __init__(self, ctx, config: dict) -> None:
        super().__init__(ctx, "code2wav")
        self.config = config
        self.num_quantizers = int(
            config.get("num_quantizers", config.get("n_q", 16)))
        self.embedding_keys = [
            key for key in self.weights.keys()
            if key.endswith(("code_embedding.weight", "embedding_sum")) and (
                "code2wav" in key or "decoder" in key)
        ]
        if not self.embedding_keys:
            raise KeyError("Code2Wav checkpoint has no codebook embedding")

    def _table(self, key: str):
        table = self.weights.store.get_f16(key)
        if key.endswith("embedding_sum"):
            usage_key = key[:-len("embedding_sum")] + "cluster_usage"
            if self.weights.store.has(usage_key):
                usage = np.maximum(self.weights.store.get_f32(usage_key), 1e-5)
                table = table.astype(np.float32) / usage[:, None]
                table = table.astype(np.float16)
        return table

    def _single_embedding(self, codes):
        table = self.weights.store.get_f16(self.embedding_keys[0])
        offsets = np.arange(self.num_quantizers, dtype=np.int64) * int(
            self.config.get("codebook_size",
                            table.shape[0] // self.num_quantizers))
        shifted = codes + F.constant(
            offsets.reshape(1, self.num_quantizers, 1), "offset")
        embedded = F.embedding_lookup(table, shifted)
        return embedded.reduce(trt.ReduceOperation.AVG, 1 << 1, False)

    def _multi_embedding(self, codes):
        quantized = []
        for index, key in enumerate(
                sorted(self.embedding_keys)[:self.num_quantizers]):
            table = self._table(key)
            code = codes.slice_axis(1, index, 1, 3)
            embedded = F.embedding_lookup(table, code).reshape(
                (0, 0, int(table.shape[-1])))
            projection = key.rsplit("._codebook.", 1)[0] + ".project_out"
            if self.weights.store.has(projection + ".weight"):
                embedded = F.linear(embedded, projection, 3)
            quantized.append(embedded)
        hidden = quantized[0]
        for tensor in quantized[1:]:
            hidden = hidden + tensor
        return hidden.reshape((0, 0, int(hidden.shape[-1])))

    def forward(self, codes):
        if len(self.embedding_keys) == 1:
            return self._single_embedding(codes)
        return self._multi_embedding(codes)


class Qwen3OmniCode2WavModel(NetworkModule):
    """End-to-end Qwen3-Omni Code2Wav graph."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle, ctx.args.max_code_len)

    def __init__(self, ctx, bundle, max_code_len: int) -> None:
        super().__init__(ctx, "code2wav")
        self.config = bundle.component_dict(contracts.Component.CODE2WAV)
        self.embedding = Qwen3OmniCodeEmbedding(ctx, self.config)
        self.pre_transformer = QwenCode2WavPreTransformer(
            ctx, self.config, max_code_len, self.key("pre_transformer"))
        self.vocoder = QwenCode2WavVocoder(ctx,
                                           self.config,
                                           self.prefix,
                                           symmetric_padding=True)

    def input_tensors(self):
        return {
            "codes":
            self.add_input("codes", trt.int64,
                           (-1, self.embedding.num_quantizers, -1))
        }

    def forward(self, codes):
        hidden = self.embedding(codes)
        hidden = self.pre_transformer(hidden)
        return {"waveform": self.vocoder(hidden).cast(trt.float16)}
