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
"""Qwen3.5 MTP checkpoint-direct draft graph."""

from typing import Dict

import tensorrt as trt

from ...ops import GatedMLP, Linear, Module, NetworkModule, RMSNorm
from ...ops import functional as F
from .modeling_qwen3_5_text import Qwen3_5Attention


class Qwen35MtpDecoderLayer(Module):
    """One Qwen3.5 MTP tree-attention decoder layer."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        eps = ctx.cfg.rms_norm_eps
        self.input_norm = RMSNorm(ctx,
                                  self.key("input_layernorm"),
                                  eps,
                                  unit_offset=True)
        self.post_norm = RMSNorm(ctx,
                                 self.key("post_attention_layernorm"),
                                 eps,
                                 unit_offset=True)
        self.attention = Qwen3_5Attention(ctx, self.key("self_attn"))
        self.mlp = GatedMLP(ctx, self.key("mlp"))

    def forward(self, hidden, past, rope, context_lengths, cache_start,
                kv_page_table, attention_mask, attention_pos_id):
        attention, present = self.attention(self.input_norm(hidden), past,
                                            rope, context_lengths, cache_start,
                                            kv_page_table, attention_mask,
                                            attention_pos_id)
        hidden = hidden + attention
        feed_forward = self.mlp(self.post_norm(hidden))
        hidden = hidden + feed_forward
        return hidden, present


class Qwen35MtpDraftModel(NetworkModule):
    """Qwen3.5 MTP draft model."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        eps = ctx.cfg.rms_norm_eps
        self.pre_embed_norm = RMSNorm(ctx,
                                      "pre_fc_norm_embedding",
                                      eps,
                                      unit_offset=True)
        self.pre_hidden_norm = RMSNorm(ctx,
                                       "pre_fc_norm_hidden",
                                       eps,
                                       unit_offset=True)
        self.fc = Linear(ctx, "fc")
        self.layers = [
            Qwen35MtpDecoderLayer(ctx, f"layers.{index}")
            for index in range(ctx.cfg.num_hidden_layers)
        ]
        self.norm = RMSNorm(ctx, "norm", eps, unit_offset=True)
        self.lm_head = Linear(ctx,
                              ctx.weights.causal_lm_head_prefix("mtp.lm_head"))

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        return {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, -1, cfg.hidden_size)),
            "past_key_values": [
                self.add_input(f"past_key_values_{index}", kv_dtype,
                               (2, -1, F.KV_PAGE_SIZE, cfg.num_key_value_heads,
                                cfg.head_dim))
                for index in range(cfg.num_hidden_layers)
            ],
            "rope":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, -1, cfg.rotary_dim)),
            "context_lengths":
            self.add_input("context_lengths", trt.int32, (-1, )),
            "cache_start":
            self.add_input("kvcache_start_index", trt.int32, (-1, )),
            "kv_page_table":
            self.add_input("kv_page_table", trt.int32, (-1, 2, -1)),
            "base_hidden":
            self.add_input("hidden_states_input", trt.float16,
                           (-1, -1, cfg.hidden_size)),
            "draft_hidden":
            self.add_input("hidden_states_from_draft", trt.float16,
                           (-1, -1, cfg.hidden_size)),
            "attention_pos_id":
            self.add_input("attention_pos_id", trt.int32, (-1, -1)),
            "attention_mask":
            self.add_input("attention_mask", trt.int32, (-1, -1, -1)),
            "last_token_ids":
            self.add_input("last_token_ids", trt.int64, (-1, 1)),
        }

    def forward(self, **io):
        outputs = {}
        source = io["base_hidden"] + io["draft_hidden"]
        merged = F.concatenate((self.pre_embed_norm(
            io["inputs_embeds"]), self.pre_hidden_norm(source)), 2)
        hidden = self.fc(merged)
        present = []
        for index, layer in enumerate(self.layers):
            hidden, cache = layer(hidden, io["past_key_values"][index],
                                  io["rope"], io["context_lengths"],
                                  io["cache_start"], io["kv_page_table"],
                                  io["attention_mask"], io["attention_pos_id"])
            present.append(cache)
        selected = F.gather_last_tokens(hidden, io["last_token_ids"])
        logits = self.lm_head(self.norm(selected)).cast(trt.float32)
        outputs["logits"] = logits.log_softmax(2)
        outputs["hidden_states"] = selected
        for index, tensor in enumerate(present):
            outputs[f"present_key_values_{index}"] = tensor
        return outputs
