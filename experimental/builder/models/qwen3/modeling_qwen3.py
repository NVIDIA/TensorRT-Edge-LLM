# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Qwen3 checkpoint-direct graph aligned with Transformers."""

from typing import Dict

import tensorrt as trt

from ...core import quantization
from ...ops import (BuildContext, DecoderLayer, DecoderModel, Linear,
                    NetworkModule, QKNormDecoderAttention)
from ...ops import functional as F
from . import weights as weight_conversion


class Qwen3Attention(QKNormDecoderAttention):
    """Qwen3 attention extension point with Q/K normalization."""

    def packed_qkv(self, hidden_states):
        """Use the provider's fused NVFP4 QKV projection when compatible."""
        projections = (self.q_proj, self.k_proj, self.v_proj)
        supported_execution = ((self.cfg.tp_size == 1
                                and self.cfg.kv_cache_quant != "fp8")
                               or self.cfg.tp_size == 2)
        can_fuse = (self.ctx.backend == "edgellm" and supported_execution
                    and all(projection.quant_type() == quantization.QUANT_NVFP4
                            for projection in projections)
                    and not any(projection.has_adapter()
                                for projection in projections))
        if not can_fuse:
            return super().packed_qkv(hidden_states)

        descriptor = weight_conversion.fuse_nvfp4_qkv(
            tuple(projection.weight_descriptor()
                  for projection in projections))
        if descriptor is None:
            return super().packed_qkv(hidden_states)
        return F.linear_from_weights(hidden_states,
                                     descriptor,
                                     name=self.key("qkv_proj_fused"))


class Qwen3DecoderLayer(DecoderLayer):
    """Qwen3 decoder layer composed from shared primitive modules."""

    attention_class = Qwen3Attention


class Qwen3Model(DecoderModel):
    """Qwen3 decoder stack and family extension point."""

    layer_class = Qwen3DecoderLayer


class Qwen3ForCausalLM(NetworkModule):
    """Qwen3 engine component and its explicit runtime I/O contract."""

    def __init__(self, ctx: BuildContext) -> None:
        super().__init__(ctx)
        self.model = Qwen3Model(ctx)
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
                           (-1, -1, cfg.hidden_size)),
            "past_key_values": [
                self.add_input(f"past_key_values_{index}", kv_dtype,
                               (2, -1, F.KV_PAGE_SIZE, cfg.num_key_value_heads,
                                cfg.head_dim))
                for index in range(cfg.num_hidden_layers)
            ],
            "rope_rotary_cos_sin":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, -1, cfg.rotary_dim)),
            "context_lengths":
            self.add_input("context_lengths", trt.int32, (-1, )),
            "kvcache_start_index":
            self.add_input("kvcache_start_index", trt.int32, (-1, )),
            "kv_page_table":
            self.add_input("kv_page_table", trt.int32, (-1, 2, -1)),
            "last_token_ids":
            self.add_input("last_token_ids", trt.int64,
                           (-1, -1) if cfg.engine_role == "base" else (-1, 1)),
        }
        if cfg.engine_role == "base":
            io["attention_pos_id"] = self.add_input("attention_pos_id",
                                                    trt.int32, (-1, -1))
            io["attention_mask"] = self.add_input("attention_mask", trt.int32,
                                                  (-1, -1, -1))
        else:
            io["attention_pos_id"] = None
            io["attention_mask"] = None
        return io

    def forward(self, **io):
        outputs = {}
        hidden_states, present_key_values, all_hidden_states = self.model(
            io["inputs_embeds"],
            io["past_key_values"],
            io["rope_rotary_cos_sin"],
            io["context_lengths"],
            io["kvcache_start_index"],
            io["kv_page_table"],
            attention_mask=io["attention_mask"],
            attention_pos_id=io["attention_pos_id"])
        selected = F.gather_last_tokens(hidden_states, io["last_token_ids"])
        outputs["logits"] = F.cast(self.lm_head(selected), trt.float32)
        if self.cfg.engine_role == "base":
            outputs["hidden_states"] = F.hidden_state_feedback(
                hidden_states, all_hidden_states, self.cfg)
        for index, present in enumerate(present_key_values):
            outputs[f"present_key_values_{index}"] = present
        return outputs
