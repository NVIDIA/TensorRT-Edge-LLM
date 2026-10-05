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
"""Qwen3-Omni code-predictor model."""

from typing import Dict

import tensorrt as trt

from ...ops import (DecoderLayer, DecoderModel, DynamicLinear, FP32GatedMLP,
                    NetworkModule, QKNormDecoderAttention)
from ...ops import functional as F
from ...ops.ragged import RaggedDecoderInputs, add_ragged_decoder_inputs


class Qwen3OmniCodePredictorMLP(FP32GatedMLP):
    """Qwen3-Omni CodePredictor precision-preserving MLP."""


class Qwen3OmniCodePredictorAttention(QKNormDecoderAttention):
    """Qwen3-Omni code-predictor attention."""


class Qwen3OmniCodePredictorDecoderLayer(DecoderLayer):
    """Qwen3-Omni code-predictor decoder layer."""

    attention_class = Qwen3OmniCodePredictorAttention
    mlp_class = Qwen3OmniCodePredictorMLP


class Qwen3OmniCodePredictorModel(DecoderModel):
    """Qwen3-Omni code-predictor decoder stack."""

    layer_class = Qwen3OmniCodePredictorDecoderLayer


class Qwen3OmniCodePredictor(NetworkModule):
    """Qwen3-Omni auxiliary predictor and its distinct I/O contract."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.model = Qwen3OmniCodePredictorModel(ctx)
        self.lm_head = DynamicLinear(ctx, ctx.cfg.hidden_size)

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        num_heads = int(
            cfg.raw_component.get("num_code_groups",
                                  (cfg.raw_root.get("talker_config")
                                   or {}).get("num_code_groups", 16))) - 1
        if num_heads < 1:
            raise ValueError(
                "Qwen3-Omni CodePredictor requires num_code_groups > 1")
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
            "lm_heads":
            self.add_input("lm_heads", trt.float16,
                           (num_heads, cfg.vocab_size, cfg.hidden_size)),
            "lm_head_idx":
            self.add_input("lm_head_idx", trt.int32, (1, )),
        }
        io.update(add_ragged_decoder_inputs(self.add_input).as_dict())
        return io

    def forward(self, **io):
        ragged = RaggedDecoderInputs.from_dict(io)
        hidden, present, _ = self.model(io["inputs_embeds"],
                                        io["past_key_values"], io["rope"],
                                        ragged)
        selected = F.gather_token_rows(hidden, ragged.logits_indices)
        head = io["lm_heads"].gather(io["lm_head_idx"].cast(trt.int64),
                                     0).reshape((self.cfg.vocab_size,
                                                 self.cfg.hidden_size))
        outputs = {
            "logits": self.lm_head(selected, head).cast(trt.float32),
            "hidden_states": hidden,
        }
        for index, tensor in enumerate(present):
            outputs[f"present_key_values_{index}"] = tensor
        return outputs
