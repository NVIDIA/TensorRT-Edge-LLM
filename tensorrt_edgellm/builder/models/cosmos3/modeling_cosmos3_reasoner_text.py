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
"""Checkpoint-direct Cosmos3 reasoner language model."""

from __future__ import annotations

from typing import Dict

import tensorrt as trt

from ...ops import (BuildContext, DecoderAttention, DecoderLayer, DecoderModel,
                    Linear, Module, NetworkModule)
from ...ops import functional as F
from ...ops.ragged import RaggedDecoderInputs, add_ragged_decoder_inputs


class Cosmos3ReasonerMLP(Module):
    """Nemotron-style non-gated squared-ReLU feed-forward block."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.up_proj = Linear(ctx, self.key("up_proj"))
        self.down_proj = Linear(ctx, self.key("down_proj"))

    def forward(self, hidden_states):
        activated = self.up_proj(hidden_states).relu()
        return self.down_proj(activated * activated)


class Cosmos3ReasonerAttention(DecoderAttention):
    """Reasoner grouped-query attention without Q/K normalization."""


class Cosmos3ReasonerDecoderLayer(DecoderLayer):
    """One Cosmos3 reasoner attention and relu2 feed-forward layer."""

    attention_class = Cosmos3ReasonerAttention
    mlp_class = Cosmos3ReasonerMLP


class Cosmos3ReasonerModel(DecoderModel):
    """Model-owned Cosmos3 reasoner decoder stack."""

    layer_class = Cosmos3ReasonerDecoderLayer


class Cosmos3ReasonerForCausalLM(NetworkModule):
    """Autoregressive reasoner engine and its explicit I/O contract."""

    def __init__(self, ctx: BuildContext) -> None:
        super().__init__(ctx)
        self.model = Cosmos3ReasonerModel(ctx)
        self.lm_head = Linear(ctx, ctx.weights.causal_lm_head_prefix())

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
            "rope_rotary_cos_sin":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, cfg.rotary_dim)),
        }
        io.update(add_ragged_decoder_inputs(self.add_input).as_dict())
        if cfg.engine_role == "base":
            io["attention_mask"] = self.add_input("packed_attention_mask",
                                                  trt.int32, (-1, -1))
            io["attention_pos_id"] = self.add_input("attention_position_ids",
                                                    trt.int32, (-1, ))
        else:
            io["attention_mask"] = None
            io["attention_pos_id"] = None
        return io

    def forward(self, **io):
        ragged = RaggedDecoderInputs.from_dict(io)
        hidden_states, present_key_values, all_hidden_states = self.model(
            io["inputs_embeds"],
            io["past_key_values"],
            io["rope_rotary_cos_sin"],
            ragged,
            attention_mask=io["attention_mask"],
            attention_pos_id=io["attention_pos_id"])
        selected = F.gather_token_rows(hidden_states, ragged.logits_indices)
        outputs = {"logits": self.lm_head(selected).cast(trt.float32)}
        if self.cfg.engine_role == "base":
            outputs["hidden_states"] = F.hidden_state_feedback(
                hidden_states, all_hidden_states, self.cfg)
        outputs.update({
            f"present_key_values_{index}": present
            for index, present in enumerate(present_key_values)
        })
        return outputs
