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
"""Qwen3-Omni Next checkpoint-direct MTP draft graph."""

from typing import Dict

import tensorrt as trt

from ...ops import (GatedDecoderAttention, GatedExperts, GatedMLP, Linear,
                    Module, NetworkModule, RMSNorm, TopKRouter)
from ...ops import functional as F
from . import weights as weight_conversion


class Qwen3OmniMtpAttention(GatedDecoderAttention):
    """The draft's gated full-attention block."""


class Qwen3OmniMtpSparseMoeBlock(Module):
    """Unquantized routed experts plus the gated shared expert."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.gate = TopKRouter(ctx, self.key("gate"))
        self.experts = GatedExperts(ctx, self.key("experts"))
        self.shared_expert = GatedMLP(ctx, self.key("shared_expert"))
        self.shared_expert_gate = Linear(ctx, self.key("shared_expert_gate"))

    def forward(self, hidden_states):
        cfg = self.cfg
        expert_weights = self.weights.parameter_value(
            "fp16",
            self.experts.prefix,
            lambda: weight_conversion.fp16_expert_specs(
                self.weights, self.experts.prefix, cfg.num_experts),
            lambda: weight_conversion.prepare_fp16_experts(
                self.weights,
                self.experts.prefix,
                cfg.num_experts,
                cfg.hidden_size,
                cfg.moe_intermediate_size,
            ),
        )
        bindings = weight_conversion.fp16_expert_bindings(
            self.weights, self.experts.prefix, cfg.num_experts)
        routed = F.fp16_moe(
            self.gate(hidden_states),
            hidden_states,
            expert_weights,
            cfg.num_experts,
            cfg.num_experts_per_tok,
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            weight_prefix=self.experts.prefix,
            weight_bindings=bindings,
            norm_topk_prob=int(cfg.norm_topk_prob),
        )
        shared = self.shared_expert(hidden_states)
        return routed + shared * self.shared_expert_gate(
            hidden_states).sigmoid()


class Qwen3OmniMtpDecoderLayer(Module):
    """One full-attention MTP layer with an FP16 sparse MoE."""

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
        self.attention = Qwen3OmniMtpAttention(ctx, self.key("self_attn"))
        self.mlp = Qwen3OmniMtpSparseMoeBlock(ctx, self.key("mlp"))

    def forward(self, hidden, past, rope, context_lengths, cache_start,
                kv_page_table, attention_mask, attention_pos_id):
        attention, present = self.attention(self.input_norm(hidden), past,
                                            rope, context_lengths, cache_start,
                                            kv_page_table, attention_mask,
                                            attention_pos_id)
        hidden = hidden + attention
        hidden = hidden + self.mlp(self.post_norm(hidden))
        return hidden, present


class Qwen3OmniMtpDraftModel(NetworkModule):
    """Qwen3-Omni Next native MTP draft and runtime I/O contract."""

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
            Qwen3OmniMtpDecoderLayer(ctx, f"layers.{index}")
            for index in range(ctx.cfg.num_hidden_layers)
        ]
        self.norm = RMSNorm(ctx, "norm", eps, unit_offset=True)
        self.lm_head = Linear(
            ctx,
            ctx.weights.causal_lm_head_prefix("thinker.mtp.lm_head",
                                              "mtp.lm_head"))

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
        merged = F.concatenate(
            (self.pre_embed_norm(io["inputs_embeds"]),
             self.pre_hidden_norm(io["base_hidden"] + io["draft_hidden"])), 2)
        hidden = self.fc(merged)
        present = []
        for index, layer in enumerate(self.layers):
            hidden, cache = layer(hidden, io["past_key_values"][index],
                                  io["rope"], io["context_lengths"],
                                  io["cache_start"], io["kv_page_table"],
                                  io["attention_mask"], io["attention_pos_id"])
            present.append(cache)
        selected = F.gather_last_tokens(hidden, io["last_token_ids"])
        outputs["logits"] = self.lm_head(self.norm(selected)).cast(
            trt.float32).log_softmax(2)
        outputs["hidden_states"] = selected
        for index, tensor in enumerate(present):
            outputs[f"present_key_values_{index}"] = tensor
        return outputs
