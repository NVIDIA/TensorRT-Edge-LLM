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
"""Alpamayo dense text model family."""

from typing import Dict

import tensorrt as trt

from ...ops import (BuildContext, DecoderLayer, DecoderModel, Linear,
                    NetworkModule, QKNormDecoderAttention)
from ...ops import functional as F
from ...ops.ragged import RaggedDecoderInputs, add_ragged_decoder_inputs


class AlpamayoAttention(QKNormDecoderAttention):
    """Alpamayo text attention extension point."""

    def __init__(self,
                 ctx,
                 prefix,
                 layer_index=None,
                 sliding_window_size=None):
        super().__init__(ctx, prefix, layer_index, sliding_window_size=-1)


class AlpamayoDecoderLayer(DecoderLayer):
    """Alpamayo decoder layer composed from shared primitive modules."""

    attention_class = AlpamayoAttention


class AlpamayoModel(DecoderModel):
    """Alpamayo text stack and model-family extension point."""

    layer_class = AlpamayoDecoderLayer


class AlpamayoForCausalLM(NetworkModule):
    """Alpamayo language model and its explicit engine I/O contract."""

    def __init__(self, ctx: BuildContext) -> None:
        super().__init__(ctx)
        self.model = AlpamayoModel(ctx)
        lm_head = ("lm_head" if ctx.weights.has("lm_head.weight")
                   or ctx.weights.has("lm_head.qweight") else
                   "model.embed_tokens")
        self.lm_head = Linear(ctx, lm_head)

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
            "deepstack_embeds": [
                self.add_input(f"deepstack_embeds_{index}", trt.float16,
                               (-1, cfg.hidden_size))
                for index in range(cfg.num_deepstack_features)
            ],
        }
        io.update(add_ragged_decoder_inputs(self.add_input).as_dict())
        return io

    def forward(self, **io):
        ragged = RaggedDecoderInputs.from_dict(io)
        hidden_states, present_key_values, _ = self.model(
            io["inputs_embeds"], io["past_key_values"],
            io["rope_rotary_cos_sin"], ragged, io["deepstack_embeds"])
        selected = F.gather_token_rows(hidden_states, ragged.logits_indices)
        outputs = {"logits": F.cast(self.lm_head(selected), trt.float32)}
        outputs.update({
            f"present_key_values_{index}": present
            for index, present in enumerate(present_key_values)
        })
        return outputs
