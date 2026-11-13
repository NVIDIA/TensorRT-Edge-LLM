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
"""InternVL3.5 dense text model family."""

from typing import Dict

import tensorrt as trt

from ...ops import (BuildContext, DecoderAttention, DecoderLayer, DecoderModel,
                    Linear, Module, NetworkModule)
from ...ops import functional as F
from ...ops.ragged import RaggedDecoderInputs, add_ragged_decoder_inputs

__all__ = [
    "BuildContext",
    "Linear",
    "InternVL35TextAttention",
    "InternVL35TextDecoderLayer",
    "InternVL35TextModel",
    "InternVL35CausalLM",
    "Module",
    "NetworkModule",
]


class InternVL35TextAttention(DecoderAttention):
    """InternVL3.5 nested Qwen2 attention extension point."""


class InternVL35TextDecoderLayer(DecoderLayer):
    """InternVL3.5 text layer composed from shared primitive modules."""

    attention_class = InternVL35TextAttention


class InternVL35TextModel(DecoderModel):
    """InternVL3.5's nested Qwen2 decoder stack."""

    layer_class = InternVL35TextDecoderLayer


class InternVL35CausalLM(NetworkModule):
    """Dense checkpoint-backed causal language-model graph."""

    def __init__(self, ctx: BuildContext) -> None:
        super().__init__(ctx)
        self.model = InternVL35TextModel(ctx)
        lm_head = ("lm_head" if ctx.weights.has("lm_head.weight")
                   or ctx.weights.has("lm_head.qweight") else
                   "model.embed_tokens")
        self.lm_head = Linear(ctx, lm_head)

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        result: Dict[str, object] = {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, cfg.hidden_size)),
            "past_key_values": [
                self.add_input(f"past_key_values_{index}", kv_dtype,
                               (2, -1, F.KV_PAGE_SIZE, cfg.num_key_value_heads,
                                cfg.head_dim))
                for index in range(cfg.num_hidden_layers)
            ],
            "rope_rotary_cos_sin":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, cfg.rotary_dim)),
        }
        result.update(add_ragged_decoder_inputs(self.add_input).as_dict())
        if cfg.engine_role == "base":
            result["attention_pos_id"] = self.add_input(
                "attention_position_ids", trt.int32, (-1, ))
            result["attention_mask"] = self.add_input("packed_attention_mask",
                                                      trt.int32, (-1, -1))
        else:
            result["attention_pos_id"] = None
            result["attention_mask"] = None
        return result

    def forward(self, **io):
        ragged = RaggedDecoderInputs.from_dict(io)
        outputs = {}
        hidden_states, present_key_values, all_hidden_states = self.model(
            io["inputs_embeds"],
            io["past_key_values"],
            io["rope_rotary_cos_sin"],
            ragged,
            attention_mask=io["attention_mask"],
            attention_pos_id=io["attention_pos_id"])
        selected = F.gather_token_rows(hidden_states, ragged.logits_indices)
        logits = F.cast(self.lm_head(selected), trt.float32)
        outputs["logits"] = logits
        if self.cfg.engine_role == "base":
            outputs["hidden_states"] = F.hidden_state_feedback(
                hidden_states, all_hidden_states, self.cfg)
        for layer_index, present in enumerate(present_key_values):
            outputs[f"present_key_values_{layer_index}"] = present
        return outputs
