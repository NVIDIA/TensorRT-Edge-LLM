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
"""Qwen3-Omni talker model."""

from typing import Dict

import tensorrt as trt

from ...ops import (DecoderLayer, DecoderModel, Linear, NetworkModule,
                    QKNormDecoderAttention)
from ...ops import functional as F
from ...ops.ragged import RaggedDecoderInputs, add_ragged_decoder_inputs


class Qwen3OmniTalkerAttention(QKNormDecoderAttention):
    """Qwen3-Omni talker attention extension point."""


class Qwen3OmniTalkerDecoderLayer(DecoderLayer):
    """Qwen3-Omni talker layer composed from shared primitive modules."""

    attention_class = Qwen3OmniTalkerAttention


class Qwen3OmniTalkerModel(DecoderModel):
    """Qwen3-Omni talker stack and model-family extension point."""

    layer_class = Qwen3OmniTalkerDecoderLayer


class Qwen3OmniTalker(NetworkModule):
    """Qwen3-Omni autoregressive talker and its engine I/O contract."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.model = Qwen3OmniTalkerModel(ctx)
        self.codec_head = Linear(ctx, "codec_head")

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        io = {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, cfg.hidden_size)),
            "past_key_values": [
                self.add_input(f"past_key_values_{index}", kv_dtype,
                               (2, -1, F.KV_PAGE_SIZE, cfg.num_key_value_heads,
                                cfg.head_dim))
                for index in range(cfg.num_hidden_layers)
            ],
            "rope":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, cfg.rotary_dim)),
        }
        io.update(add_ragged_decoder_inputs(self.add_input).as_dict())
        return io

    def forward(self, **io):
        ragged = RaggedDecoderInputs.from_dict(io)
        hidden, present, _ = self.model(io["inputs_embeds"],
                                        io["past_key_values"], io["rope"],
                                        ragged)
        selected = F.gather_token_rows(hidden, ragged.logits_indices)
        outputs = {
            "logits": self.codec_head(selected).cast(trt.float32),
            "hidden_states": hidden,
        }
        for index, tensor in enumerate(present):
            outputs[f"present_key_values_{index}"] = tensor
        return outputs
