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
"""Checkpoint-direct Qwen3-Omni-Next native MTP draft."""

from typing import Dict

import tensorrt as trt

from ...ops import GatedMLP, Linear, Module, NetworkModule, RMSNorm
from ...ops import functional as F
from ...ops.ragged import RaggedDecoderInputs, add_ragged_decoder_inputs
from .modeling_qwen3_omni_next_moe_text import Qwen3OmniNextSparseMoeBlock
from .modeling_qwen3_omni_next_text import Qwen3OmniNextAttention

__all__ = [
    "Qwen3OmniNextMtpDecoderLayer",
    "Qwen3OmniNextMtpDraftModel",
]


class Qwen3OmniNextMtpDecoderLayer(Module):
    """One full-attention MTP block with dense or sparse FFN."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        eps = ctx.cfg.rms_norm_eps
        self.input_layernorm = RMSNorm(ctx,
                                       self.key("input_layernorm"),
                                       eps,
                                       unit_offset=True)
        self.post_attention_layernorm = RMSNorm(
            ctx, self.key("post_attention_layernorm"), eps, unit_offset=True)
        self.self_attn = Qwen3OmniNextAttention(ctx, self.key("self_attn"))
        self.mlp = (Qwen3OmniNextSparseMoeBlock(ctx, self.key("mlp"))
                    if ctx.cfg.num_experts > 0 else GatedMLP(
                        ctx, self.key("mlp")))

    def forward(self, hidden_states, past_key_value, rope, ragged,
                attention_mask, attention_pos_id):
        attention, present = self.self_attn(
            self.input_layernorm(hidden_states), past_key_value, rope, ragged,
            attention_mask, attention_pos_id)
        hidden_states = hidden_states + attention
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states))
        return hidden_states, present


class Qwen3OmniNextMtpDraftModel(NetworkModule):
    """Native MTP draft over the Thinker's checkpoint-owned MTP layers."""

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
            Qwen3OmniNextMtpDecoderLayer(ctx, f"layers.{index}")
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
            "base_hidden":
            self.add_input("hidden_states_input", trt.float16,
                           (-1, cfg.hidden_size)),
            "draft_hidden":
            self.add_input("hidden_states_from_draft", trt.float16,
                           (-1, cfg.hidden_size)),
            "attention_pos_id":
            self.add_input("attention_position_ids", trt.int32, (-1, )),
            "attention_mask":
            self.add_input("packed_attention_mask", trt.int32, (-1, -1)),
        }
        io.update(add_ragged_decoder_inputs(self.add_input).as_dict())
        return io

    def forward(self, **io):
        ragged = RaggedDecoderInputs.from_dict(io)
        merged = F.concatenate(
            (self.pre_embed_norm(io["inputs_embeds"]),
             self.pre_hidden_norm(io["base_hidden"] + io["draft_hidden"])), 1)
        hidden_states = self.fc(merged)
        present = []
        for index, layer in enumerate(self.layers):
            hidden_states, cache = layer(hidden_states,
                                         io["past_key_values"][index],
                                         io["rope"], ragged,
                                         io["attention_mask"],
                                         io["attention_pos_id"])
            present.append(cache)
        selected = F.gather_token_rows(hidden_states, ragged.logits_indices)
        logits = self.lm_head(self.norm(selected)).cast(
            trt.float32).log_softmax(1)
        outputs = {
            "logits": logits,
            "hidden_states": selected,
        }
        for index, tensor in enumerate(present):
            outputs[f"present_key_values_{index}"] = tensor
        return outputs
