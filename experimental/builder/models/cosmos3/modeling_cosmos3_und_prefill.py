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
"""Checkpoint-direct Cosmos3 understanding-prefill tower."""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from ...ops import Linear, Module, NetworkModule, RMSNorm
from ...ops import functional as F


class Cosmos3UndMLP(Module):
    """Checkpoint-selected gated or non-gated understanding MLP."""

    def __init__(self, ctx, prefix: str, hidden_act: str) -> None:
        super().__init__(ctx, prefix)
        self.hidden_act = hidden_act
        self.up_proj = Linear(ctx, self.key("up_proj"), tensor_parallel=False)
        self.gate_proj = (
            Linear(ctx, self.key("gate_proj"), tensor_parallel=False)
            if self.weights.has(self.key("gate_proj.weight")) else None)
        self.down_proj = Linear(ctx,
                                self.key("down_proj"),
                                tensor_parallel=False)

    def forward(self, hidden_states):
        up = self.up_proj(hidden_states)
        if self.gate_proj is None:
            intermediate = up.activation(self.hidden_act)
        else:
            intermediate = self.gate_proj(hidden_states).activation(
                self.hidden_act) * up
        return self.down_proj(intermediate)


class Cosmos3UndAttention(Module):
    """Causal UND self-attention that also exposes per-layer GEN K/V."""

    def __init__(self, ctx, prefix: str, config: dict) -> None:
        super().__init__(ctx, prefix)
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        eps = float(config.get("rms_norm_eps", 1e-6))
        self.to_q = Linear(ctx, self.key("to_q"), tensor_parallel=False)
        self.to_k = Linear(ctx, self.key("to_k"), tensor_parallel=False)
        self.to_v = Linear(ctx, self.key("to_v"), tensor_parallel=False)
        self.to_out = Linear(ctx, self.key("to_out"), tensor_parallel=False)
        self.norm_q = (RMSNorm(ctx, self.key("norm_q"), eps, rank=4) if
                       self.weights.has(self.key("norm_q.weight")) else None)
        self.norm_k = (RMSNorm(ctx, self.key("norm_k"), eps, rank=4) if
                       self.weights.has(self.key("norm_k.weight")) else None)
        self.gen_k_norm = (
            RMSNorm(ctx, self.key("k_norm_und_for_gen"), eps, rank=4) if
            self.weights.has(self.key("k_norm_und_for_gen.weight")) else None)

    def forward(self, hidden_states, rope_cos, rope_sin, position_ids):
        query = self.to_q(hidden_states).reshape(
            (0, 0, self.num_heads, self.head_dim))
        key = self.to_k(hidden_states).reshape(
            (0, 0, self.num_kv_heads, self.head_dim))
        value = self.to_v(hidden_states).reshape(
            (0, 0, self.num_kv_heads, self.head_dim))
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = query.transpose((0, 2, 1, 3))
        key_heads = key.transpose((0, 2, 1, 3))
        value_heads = value.transpose((0, 2, 1, 3))
        query = F.rotary_embedding(query.cast(trt.float16), rope_cos, rope_sin,
                                   position_ids, self.head_dim)
        self_key = F.rotary_embedding(key_heads.cast(trt.float16), rope_cos,
                                      rope_sin, position_ids, self.head_dim)
        query = query * np.float16(self.head_dim**-0.5)
        attended = F.scaled_dot_product_attention(query,
                                                  self_key,
                                                  value_heads,
                                                  scale=1.0,
                                                  is_causal=True)
        attended = attended.transpose((0, 2, 1, 3)).reshape(
            (0, 0, self.num_heads * self.head_dim))

        gen_key = key
        if self.gen_k_norm is not None:
            gen_key = self.gen_k_norm(gen_key)
        gen_key = F.rotary_embedding(
            gen_key.transpose((0, 2, 1, 3)).cast(trt.float16), rope_cos,
            rope_sin, position_ids, self.head_dim)
        return (self.to_out(attended), gen_key.transpose((0, 2, 1, 3)), value)


class Cosmos3UndDecoderLayer(Module):
    """One model-owned Cosmos3 understanding decoder layer."""

    def __init__(self, ctx, index: int, config: dict) -> None:
        prefix = f"layers.{index}"
        super().__init__(ctx, prefix)
        eps = float(config.get("rms_norm_eps", 1e-6))
        self.input_layernorm = RMSNorm(ctx, self.key("input_layernorm"), eps)
        self.self_attn = Cosmos3UndAttention(ctx, self.key("self_attn"),
                                             config)
        self.post_attention_layernorm = RMSNorm(
            ctx, self.key("post_attention_layernorm"), eps)
        self.mlp = Cosmos3UndMLP(ctx, self.key("mlp"),
                                 str(config["hidden_act"]))

    def forward(self, hidden_states, rope_cos, rope_sin, position_ids):
        attention, gen_key, gen_value = self.self_attn(
            self.input_layernorm(hidden_states), rope_cos, rope_sin,
            position_ids)
        hidden_states = hidden_states + attention
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states))
        return hidden_states, gen_key, gen_value


class Cosmos3UndPrefillModel(NetworkModule):
    """Prefill-only understanding tower with explicit per-layer K/V outputs."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle)

    def __init__(self, ctx, bundle) -> None:
        super().__init__(ctx)
        self.config = dict(bundle.root["_direct_transformer_config"])
        self.hidden_size = int(self.config["hidden_size"])
        self.num_layers = int(self.config["num_hidden_layers"])
        self.num_kv_heads = int(self.config["num_key_value_heads"])
        self.head_dim = int(self.config["head_dim"])
        self.layers = [
            Cosmos3UndDecoderLayer(ctx, index, self.config)
            for index in range(self.num_layers)
        ]
        self.norm = RMSNorm(ctx, "norm",
                            float(self.config.get("rms_norm_eps", 1e-6)))

    def input_tensors(self):
        return {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, -1, self.hidden_size)),
            "rope_rotary_cos_sin":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, -1, self.head_dim)),
            "attention_pos_id":
            self.add_input("attention_position_ids", trt.int32, (-1, -1)),
        }

    def forward(self, inputs_embeds, rope_rotary_cos_sin, attention_pos_id):
        half = self.head_dim // 2
        rope_cos = rope_rotary_cos_sin.slice_last_dim(0, half, 3).reshape(
            (-1, half)).cast(trt.float16)
        rope_sin = rope_rotary_cos_sin.slice_last_dim(half, half, 3).reshape(
            (-1, half)).cast(trt.float16)
        hidden_states = inputs_embeds
        keys = []
        values = []
        for layer in self.layers:
            hidden_states, key, value = layer(hidden_states, rope_cos,
                                              rope_sin, attention_pos_id)
            keys.append(key)
            values.append(value)
        outputs = {
            f"und_k_layer{index:02d}": key
            for index, key in enumerate(keys)
        }
        outputs.update({
            f"und_v_layer{index:02d}": value
            for index, value in enumerate(values)
        })
        outputs["hidden_states"] = self.norm(hidden_states)
        return outputs
