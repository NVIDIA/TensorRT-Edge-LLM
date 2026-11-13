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
"""Checkpoint-direct pi0.5 bidirectional prefix tower."""

from __future__ import annotations

import tensorrt as trt

from ...ops import Linear, Module, NetworkModule, RMSNorm
from ...ops import functional as F
from .configuration import Pi05Geometry


class Pi05MLP(Module):
    """Gemma GeGLU feed-forward block."""

    def __init__(self, ctx, prefix: str, rank: int = 3) -> None:
        super().__init__(ctx, prefix)
        self.gate_proj = Linear(ctx,
                                self.key("gate_proj"),
                                rank=rank,
                                tensor_parallel=False)
        self.up_proj = Linear(ctx,
                              self.key("up_proj"),
                              rank=rank,
                              tensor_parallel=False)
        self.down_proj = Linear(ctx,
                                self.key("down_proj"),
                                rank=rank,
                                tensor_parallel=False)

    def forward(self, hidden_states):
        return self.down_proj(
            self.gate_proj(hidden_states).gelu_tanh() *
            self.up_proj(hidden_states))


class Pi05PrefixAttention(Module):
    """Bidirectional GQA that exposes post-RoPE K and plain V."""

    def __init__(self, ctx, prefix: str, config: dict) -> None:
        super().__init__(ctx, prefix)
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.hidden_size = int(config["hidden_size"])
        self.q_proj = Linear(ctx,
                             self.key("q_proj"),
                             rank=3,
                             tensor_parallel=False)
        self.k_proj = Linear(ctx,
                             self.key("k_proj"),
                             rank=3,
                             tensor_parallel=False)
        self.v_proj = Linear(ctx,
                             self.key("v_proj"),
                             rank=3,
                             tensor_parallel=False)
        self.o_proj = Linear(ctx,
                             self.key("o_proj"),
                             rank=3,
                             tensor_parallel=False)

    def forward(self, hidden_states, rope_cos, rope_sin, position_ids):
        query = self.q_proj(hidden_states).reshape(
            (0, 0, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        key = self.k_proj(hidden_states).reshape(
            (0, 0, self.num_kv_heads, self.head_dim)).transpose((0, 2, 1, 3))
        value = self.v_proj(hidden_states).reshape(
            (0, 0, self.num_kv_heads, self.head_dim)).transpose((0, 2, 1, 3))
        query = F.rotary_embedding(query, rope_cos, rope_sin, position_ids,
                                   self.head_dim)
        key = F.rotary_embedding(key, rope_cos, rope_sin, position_ids,
                                 self.head_dim)
        attended = F.scaled_dot_product_attention(query,
                                                  key,
                                                  value,
                                                  scale=self.head_dim**-0.5)
        attended = attended.transpose((0, 2, 1, 3)).reshape(
            (0, 0, self.num_heads * self.head_dim))
        return (self.o_proj(attended), key.transpose(
            (0, 2, 1, 3)), value.transpose((0, 2, 1, 3)))


class Pi05PrefixLayer(Module):
    """One unit-offset Gemma prefix layer."""

    def __init__(self, ctx, index: int, config: dict) -> None:
        prefix = f"layers.{index}"
        super().__init__(ctx, prefix)
        eps = float(config.get("rms_norm_eps", 1e-6))
        self.input_layernorm = RMSNorm(ctx,
                                       self.key("input_layernorm"),
                                       eps,
                                       rank=3,
                                       unit_offset=True)
        self.self_attn = Pi05PrefixAttention(ctx, self.key("self_attn"),
                                             config)
        self.post_attention_layernorm = RMSNorm(
            ctx,
            self.key("post_attention_layernorm"),
            eps,
            rank=3,
            unit_offset=True)
        self.mlp = Pi05MLP(ctx, self.key("mlp"))

    def forward(self, hidden_states, rope_cos, rope_sin, position_ids):
        attention, key, value = self.self_attn(
            self.input_layernorm(hidden_states), rope_cos, rope_sin,
            position_ids)
        hidden_states = hidden_states + attention
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states))
        return hidden_states, key, value


class Pi05PrefixModel(NetworkModule):
    """Run prefix prefill once and return per-layer K/V tensors."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, Pi05Geometry.from_bundle(ctx.bundle, ctx.args))

    def __init__(self, ctx, geometry: Pi05Geometry) -> None:
        super().__init__(ctx)
        self.config = geometry.prefix
        self.hidden_size = int(self.config["hidden_size"])
        self.head_dim = int(self.config["head_dim"])
        self.num_layers = int(self.config["num_hidden_layers"])
        self.layers = [
            Pi05PrefixLayer(ctx, index, self.config)
            for index in range(self.num_layers)
        ]

    def input_tensors(self):
        return {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, -1, self.hidden_size)),
            "rope_rotary_cos_sin":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, -1, self.head_dim)),
            "attention_pos_id":
            self.add_input("attention_pos_id", trt.int32, (-1, -1)),
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
            f"k_layer{index:02d}": key
            for index, key in enumerate(keys)
        }
        outputs.update({
            f"v_layer{index:02d}": value
            for index, value in enumerate(values)
        })
        return outputs
