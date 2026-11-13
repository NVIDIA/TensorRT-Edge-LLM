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
"""Qwen3-TTS Code2Wav component and engine I/O contract."""

import tensorrt as trt

from ...core import contracts
from ...ops import NetworkModule
from ...ops.code2wav import (CausalConv1d, QwenCode2WavPreTransformer,
                             QwenCode2WavVocoder, QwenQuantizerGroup)


class Qwen3TTSCode2WavModel(NetworkModule):
    """End-to-end Qwen3-TTS Code2Wav graph."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle, ctx.args.max_code_len)

    def __init__(self, ctx, bundle, max_code_len: int) -> None:
        super().__init__(ctx, "decoder")
        self.config = bundle.component_dict(contracts.Component.CODE2WAV)
        self.num_quantizers = int(self.config.get("num_quantizers", 16))
        semantic = int(self.config.get("num_semantic_quantizers", 1))
        self.first = QwenQuantizerGroup(ctx, self.key("quantizer.rvq_first"),
                                        tuple(range(semantic)))
        self.rest = QwenQuantizerGroup(
            ctx, self.key("quantizer.rvq_rest"),
            tuple(range(semantic, self.num_quantizers)))
        self.pre_conv = CausalConv1d(ctx, self.key("pre_conv.conv"))
        self.pre_transformer = QwenCode2WavPreTransformer(
            ctx, self.config, max_code_len, self.key("pre_transformer"))
        self.vocoder = QwenCode2WavVocoder(ctx, self.config, self.prefix)

    def input_tensors(self):
        return {
            "codes":
            self.add_input("codes", trt.int64, (-1, self.num_quantizers, -1))
        }

    def forward(self, codes):
        hidden_states = self.first(codes) + self.rest(codes)
        hidden_states = self.pre_conv(hidden_states.transpose((0, 2, 1)))
        hidden_states = self.pre_transformer(hidden_states.transpose(
            (0, 2, 1)))
        return {"waveform": self.vocoder(hidden_states).cast(trt.float32)}
