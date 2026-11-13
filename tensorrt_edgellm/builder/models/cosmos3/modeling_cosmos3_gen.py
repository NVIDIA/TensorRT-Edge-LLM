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
"""Checkpoint-direct Cosmos3 generation expert."""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from ...ops import Linear, Module, NetworkModule, RMSNorm
from ...ops import functional as F
from .configuration import Cosmos3PolicyGeometry
from .weights import selected_domain_linear


class Cosmos3TimestepEmbedding(Module):
    """Sinusoidal timestep embedding followed by the checkpoint MLP."""

    def __init__(self,
                 ctx,
                 hidden_size: int,
                 frequency_size: int = 256,
                 max_period: int = 10000) -> None:
        super().__init__(ctx, "time_embedder")
        self.linear_1 = Linear(ctx,
                               self.key("linear_1"),
                               rank=2,
                               tensor_parallel=False)
        self.linear_2 = Linear(ctx,
                               self.key("linear_2"),
                               rank=2,
                               tensor_parallel=False)
        half = frequency_size // 2
        frequencies = np.exp(-math.log(max_period) *
                             np.arange(half, dtype=np.float32) / half).reshape(
                                 1, half)
        self.frequencies = frequencies.astype(np.float32)

    def forward(self, timestep):
        phase = timestep.reshape(
            (-1, 1)) * F.constant(self.frequencies, "timestep_frequencies")
        embedding = F.concatenate((phase.cos(), phase.sin()), 1)
        return self.linear_2(self.linear_1(embedding.cast(trt.float16)).silu())


class Cosmos3GenMLP(Module):
    """Checkpoint-selected gated or non-gated generation MLP."""

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


class Cosmos3GenAttention(Module):
    """GEN queries attending to the concatenated UND and GEN context."""

    def __init__(self, ctx, prefix: str, config: dict) -> None:
        super().__init__(ctx, prefix)
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.eps = float(config.get("rms_norm_eps", 1e-6))
        self.to_q = Linear(ctx, self.key("add_q_proj"), tensor_parallel=False)
        self.to_k = Linear(ctx, self.key("add_k_proj"), tensor_parallel=False)
        self.to_v = Linear(ctx, self.key("add_v_proj"), tensor_parallel=False)
        self.to_out = Linear(ctx,
                             self.key("to_add_out"),
                             tensor_parallel=False)
        self.norm_q = RMSNorm(ctx, self.key("norm_added_q"), self.eps, rank=4)
        self.norm_k = RMSNorm(ctx, self.key("norm_added_k"), self.eps, rank=4)

    def forward(self, hidden_states, und_key, und_value, rope_cos, rope_sin,
                position_ids):
        query = self.to_q(hidden_states).reshape(
            (0, 0, self.num_heads, self.head_dim))
        key = self.to_k(hidden_states).reshape(
            (0, 0, self.num_kv_heads, self.head_dim))
        value = self.to_v(hidden_states).reshape(
            (0, 0, self.num_kv_heads, self.head_dim))
        query = self.norm_q(query).transpose((0, 2, 1, 3))
        key = self.norm_k(key).transpose((0, 2, 1, 3))
        value = value.transpose((0, 2, 1, 3))
        query = F.rotary_embedding(query.cast(trt.float16), rope_cos, rope_sin,
                                   position_ids, self.head_dim)
        key = F.rotary_embedding(key.cast(trt.float16), rope_cos, rope_sin,
                                 position_ids, self.head_dim)
        query = query * np.float16(self.head_dim**-0.5)
        key = F.concatenate((und_key.transpose((0, 2, 1, 3)), key), 2)
        value = F.concatenate((und_value.transpose((0, 2, 1, 3)), value), 2)
        attended = F.scaled_dot_product_attention(query, key, value, scale=1.0)
        attended = attended.transpose((0, 2, 1, 3)).reshape(
            (0, 0, self.num_heads * self.head_dim))
        return self.to_out(attended)


class Cosmos3GenDecoderLayer(Module):
    """Pre-normalized generation cross-attention and feed-forward block."""

    def __init__(self, ctx, index: int, config: dict) -> None:
        prefix = f"layers.{index}"
        super().__init__(ctx, prefix)
        eps = float(config.get("rms_norm_eps", 1e-6))
        self.cross_attention = Cosmos3GenAttention(ctx, self.key("self_attn"),
                                                   config)
        self.input_layernorm = RMSNorm(ctx,
                                       self.key("input_layernorm_moe_gen"),
                                       eps)
        self.post_attention_layernorm = RMSNorm(
            ctx, self.key("post_attention_layernorm_moe_gen"), eps)
        self.mlp = Cosmos3GenMLP(ctx, self.key("mlp_moe_gen"),
                                 str(config["hidden_act"]))

    def forward(self, hidden_states, und_key, und_value, rope_cos, rope_sin,
                position_ids):
        hidden_states = hidden_states + self.cross_attention(
            self.input_layernorm(hidden_states), und_key, und_value, rope_cos,
            rope_sin, position_ids)
        return hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states))


class Cosmos3GenModel(NetworkModule):
    """One complete Cosmos3 video/action flow-matching prediction step."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle)

    def __init__(self, ctx, bundle) -> None:
        super().__init__(ctx)
        self.config = dict(bundle.root["_direct_transformer_config"])
        self.geometry = Cosmos3PolicyGeometry.from_bundle(bundle, ctx.args)
        self.hidden_size = int(self.config["hidden_size"])
        self.num_layers = int(self.config["num_hidden_layers"])
        self.num_kv_heads = int(self.config["num_key_value_heads"])
        self.head_dim = int(self.config["head_dim"])
        self.latent_channel = int(self.config.get("latent_channel", 48))
        self.patch_size = int(self.config.get("latent_patch_size", 2))
        self.max_action_dim = int(self.config.get("max_action_dim", 64))
        self.patch_height = ((self.geometry.latent_h + self.patch_size - 1) //
                             self.patch_size)
        self.patch_width = ((self.geometry.latent_w + self.patch_size - 1) //
                            self.patch_size)
        self.video_tokens = (self.geometry.latent_t * self.patch_height *
                             self.patch_width)
        self.action_tokens = self.geometry.action_token_count
        self.gen_tokens = self.video_tokens + self.action_tokens
        self.proj_in = Linear(ctx, "proj_in", rank=3, tensor_parallel=False)
        self.proj_out = Linear(ctx, "proj_out", rank=3, tensor_parallel=False)
        self.time_embedder = Cosmos3TimestepEmbedding(ctx, self.hidden_size)
        self.layers = [
            Cosmos3GenDecoderLayer(ctx, index, self.config)
            for index in range(self.num_layers)
        ]
        self.norm_moe_gen = RMSNorm(
            ctx, "norm_moe_gen", float(self.config.get("rms_norm_eps", 1e-6)))
        self.action_in_weight, self.action_in_bias = selected_domain_linear(
            self.weights, "action_proj_in", self.geometry.domain_id,
            self.max_action_dim, self.hidden_size)
        self.action_out_weight, self.action_out_bias = selected_domain_linear(
            self.weights, "action_proj_out", self.geometry.domain_id,
            self.hidden_size, self.max_action_dim)
        self.action_modality_embed = self.weights.f16(
            "action_modality_embed").reshape(1, 1, self.hidden_size)

    def input_tensors(self):
        batch = -1
        return {
            "video_latent":
            self.add_input("video_latent", trt.float32,
                           (batch, self.latent_channel, self.geometry.latent_t,
                            self.geometry.latent_h, self.geometry.latent_w)),
            "action_latent":
            self.add_input("action_latent", trt.float32,
                           (batch, self.action_tokens, self.max_action_dim)),
            "timestep":
            self.add_input("timestep", trt.float32, (batch, )),
            "token_noisy_mask":
            self.add_input("token_noisy_mask", trt.float32,
                           (batch, self.video_tokens, 1)),
            "action_noisy_mask":
            self.add_input("action_noisy_mask", trt.float32,
                           (batch, self.action_tokens, 1)),
            "rope_rotary_cos_sin":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (batch, self.gen_tokens, self.head_dim)),
            "attention_pos_id":
            self.add_input("attention_position_ids", trt.int32,
                           (batch, self.gen_tokens)),
            "und_keys": [
                self.add_input(f"und_k_layer{index:02d}", trt.float16,
                               (batch, -1, self.num_kv_heads, self.head_dim))
                for index in range(self.num_layers)
            ],
            "und_values": [
                self.add_input(f"und_v_layer{index:02d}", trt.float16,
                               (batch, -1, self.num_kv_heads, self.head_dim))
                for index in range(self.num_layers)
            ],
        }

    def _patchify(self, latent):
        batch, channels = 0, self.latent_channel
        time = self.geometry.latent_t
        patch = self.patch_size
        if self.patch_height * patch != self.geometry.latent_h:
            zero_row = latent.slice_axis(3, 0, 1, 5) * np.float16(0.0)
            latent = F.concatenate((latent, zero_row), 3)
        if self.patch_width * patch != self.geometry.latent_w:
            zero_column = latent.slice_axis(4, 0, 1, 5) * np.float16(0.0)
            latent = F.concatenate((latent, zero_column), 4)
        height = self.patch_height
        width = self.patch_width
        latent = latent.reshape(
            (batch, channels, time, height, patch, width, patch))
        latent = latent.transpose((0, 2, 3, 5, 4, 6, 1))
        return latent.reshape(
            (batch, self.video_tokens, patch * patch * channels))

    def _unpatchify(self, tokens):
        patch = self.patch_size
        height = self.patch_height
        width = self.patch_width
        tokens = tokens.reshape((0, self.geometry.latent_t, height, width,
                                 patch, patch, self.latent_channel))
        tokens = tokens.transpose((0, 6, 1, 2, 4, 3, 5))
        tokens = tokens.reshape(
            (0, self.latent_channel, self.geometry.latent_t, height * patch,
             width * patch))
        if height * patch != self.geometry.latent_h:
            tokens = tokens.slice_axis(3, 0, self.geometry.latent_h, 5)
        if width * patch != self.geometry.latent_w:
            tokens = tokens.slice_axis(4, 0, self.geometry.latent_w, 5)
        return tokens

    def forward(self, video_latent, action_latent, timestep, token_noisy_mask,
                action_noisy_mask, rope_rotary_cos_sin, attention_pos_id,
                und_keys, und_values):
        video_tokens = self.proj_in(
            self._patchify(video_latent.cast(trt.float16)))
        action_tokens = F.linear_with_weights(action_latent.cast(trt.float16),
                                              self.action_in_weight,
                                              self.action_in_bias,
                                              rank=3)
        action_tokens = action_tokens + F.constant(self.action_modality_embed,
                                                   "action_modality_embed")
        timestep = timestep * np.float32(
            self.config.get("timestep_scale", 0.001))
        time_embedding = self.time_embedder(timestep).unsqueeze(1, 2)
        video_tokens = video_tokens + time_embedding * token_noisy_mask.cast(
            trt.float16)
        action_tokens = action_tokens + time_embedding * action_noisy_mask.cast(
            trt.float16)
        hidden_states = F.concatenate((video_tokens, action_tokens), 1)

        half = self.head_dim // 2
        rope_cos = rope_rotary_cos_sin.slice_last_dim(0, half, 3).reshape(
            (-1, half)).cast(trt.float16)
        rope_sin = rope_rotary_cos_sin.slice_last_dim(half, half, 3).reshape(
            (-1, half)).cast(trt.float16)
        for index, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, und_keys[index],
                                  und_values[index], rope_cos, rope_sin,
                                  attention_pos_id)

        hidden_states = self.norm_moe_gen(hidden_states)
        video_hidden = hidden_states[:, :self.video_tokens, :]
        action_hidden = hidden_states[:, self.video_tokens:, :]
        video_prediction = self._unpatchify(self.proj_out(video_hidden)).cast(
            trt.float32)
        action_prediction = F.linear_with_weights(action_hidden,
                                                  self.action_out_weight,
                                                  self.action_out_bias,
                                                  rank=3).cast(trt.float32)
        return {
            "video_pred": video_prediction,
            "action_pred": action_prediction,
        }
