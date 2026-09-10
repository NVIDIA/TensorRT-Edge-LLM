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
"""DSpark checkpoint-direct draft backbone."""

from typing import Dict

import tensorrt as trt

from ...core import quantization
from ...ops import (GatedMLP, Linear, Module, NetworkModule, RMSNorm,
                    TreeAttention)
from ...ops import functional as F
from ...ops import pack_qkv


class DSparkProposalAttention(TreeAttention):
    """Proposal attention over persistent target-derived and proposal K/V."""

    def forward(self, hidden, hidden_delta, past, rope, context_lengths,
                cache_start, kv_page_table, delta_lengths, attention_mask,
                attention_pos_id):
        cfg = self.cfg
        key_delta = self.k_proj(hidden_delta).reshape(
            (0, 0, cfg.num_key_value_heads, cfg.head_dim))
        key_delta = self.k_norm(key_delta, 4)
        value_delta = self.v_proj(hidden_delta).reshape(
            (0, 0, cfg.num_key_value_heads, cfg.head_dim))
        pages_per_slot = (self.ctx.args.max_kv_cache_capacity +
                          F.KV_PAGE_SIZE - 1) // F.KV_PAGE_SIZE
        updated = F.update_dflash_target_cache(key_delta,
                                               value_delta,
                                               past,
                                               rope,
                                               cache_start,
                                               delta_lengths,
                                               kv_page_table,
                                               pages_per_slot=pages_per_slot)

        query = self.q_proj(hidden).reshape(
            (0, 0, cfg.num_attention_heads, cfg.head_dim))
        query = self.q_norm(query, 4).reshape(
            (0, 0, cfg.num_attention_heads * cfg.head_dim))
        key = self.k_proj(hidden).reshape(
            (0, 0, cfg.num_key_value_heads, cfg.head_dim))
        key = self.k_norm(key, 4).reshape(
            (0, 0, cfg.num_key_value_heads * cfg.head_dim))
        value = self.v_proj(hidden)
        qkv = pack_qkv(query, key, value, self.v_proj)
        attention, present = F.attention(
            qkv,
            updated,
            context_lengths,
            rope,
            cache_start,
            kv_page_table,
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_size=cfg.head_dim,
            enable_fp8_kv_cache=False,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
        )
        return self.o_proj(attention), present


class DSparkDecoderLayer(Module):
    """One DSpark cached proposal decoder layer."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.input_norm = RMSNorm(ctx, self.key("input_layernorm"),
                                  ctx.cfg.rms_norm_eps)
        self.attention = DSparkProposalAttention(ctx, self.key("self_attn"))
        self.post_norm = RMSNorm(ctx, self.key("post_attention_layernorm"),
                                 ctx.cfg.rms_norm_eps)
        self.mlp = GatedMLP(ctx, self.key("mlp"))

    def forward(self, hidden, hidden_delta, past, rope, context_lengths,
                cache_start, kv_page_table, delta_lengths, attention_mask,
                attention_pos_id):
        attention, present = self.attention(self.input_norm(hidden),
                                            hidden_delta, past, rope,
                                            context_lengths, cache_start,
                                            kv_page_table, delta_lengths,
                                            attention_mask, attention_pos_id)
        hidden = hidden + attention
        feed_forward = self.mlp(self.post_norm(hidden))
        return hidden + feed_forward, present


class DSparkTargetProjection(Module):
    """Project concatenated target states in FP32 before normalization."""

    def forward(self, hidden):
        descriptor = self.weights.linear_descriptor(self.prefix,
                                                    quantization.QUANT_FP16)
        return F.linear_f32_from_weights(hidden, descriptor, self.prefix)


class DSparkDraftModel(NetworkModule):
    """Parallel proposal backbone with hidden output for sequential heads."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        if not (ctx.weights.has("lm_head.weight")
                or ctx.weights.has("lm_head.qweight")):
            raise ValueError("DSpark draft checkpoint must provide lm_head")
        self.fc = DSparkTargetProjection(ctx, "fc")
        self.hidden_norm = RMSNorm(ctx, "hidden_norm", ctx.cfg.rms_norm_eps)
        self.layers = [
            DSparkDecoderLayer(ctx, f"layers.{index}")
            for index in range(ctx.cfg.num_hidden_layers)
        ]
        self.norm = RMSNorm(ctx, "norm", ctx.cfg.rms_norm_eps)
        self.lm_head = Linear(ctx, "lm_head")

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        target_layers = cfg.dspark_target_layer_ids
        return {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, -1, cfg.hidden_size)),
            "past_key_values": [
                self.add_input(f"past_key_values_{index}", trt.float16,
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
            self.add_input("dflash_target_hidden_concat", trt.float16,
                           (-1, -1, len(target_layers) * cfg.hidden_size)),
            "attention_pos_id":
            self.add_input("attention_pos_id", trt.int32, (-1, -1)),
            "attention_mask":
            self.add_input("attention_mask", trt.int32, (-1, -1, -1)),
            "delta_lengths":
            self.add_input("dflash_delta_lengths", trt.int32, (-1, )),
        }

    def forward(self, inputs_embeds, past_key_values, rope, context_lengths,
                cache_start, kv_page_table, base_hidden, attention_pos_id,
                attention_mask, delta_lengths):
        hidden = inputs_embeds
        delta = self.hidden_norm(self.fc(base_hidden))
        present = []
        for index, layer in enumerate(self.layers):
            hidden, cache = layer(hidden, delta, past_key_values[index], rope,
                                  context_lengths, cache_start, kv_page_table,
                                  delta_lengths, attention_mask,
                                  attention_pos_id)
            present.append(cache)
        hidden = self.norm(hidden)
        outputs = {
            "logits": F.cast(self.lm_head(hidden), trt.float32),
            "dspark_hidden_states": F.cast(hidden, trt.float16),
        }
        for index, tensor in enumerate(present):
            outputs[f"present_key_values_{index}"] = tensor
        return outputs
