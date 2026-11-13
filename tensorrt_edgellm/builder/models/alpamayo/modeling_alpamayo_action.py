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
"""Alpamayo action checkpoint-direct graph."""

import re

import numpy as np
import tensorrt as trt

from ...ops import Linear, Module, NetworkModule
from ...ops import functional as F


class AlpamayoActionNorm(Module):
    """Checkpoint-backed action RMSNorm or LayerNorm."""

    def __init__(self, ctx, prefix: str, eps: float) -> None:
        super().__init__(ctx, prefix)
        self.eps = eps

    def forward(self, hidden, rank: int = 3):
        return F.normalization(hidden, self.prefix, self.eps, rank)


class AlpamayoActionInputProjection(Module):
    """Fourier action/noise embedding plus action_in_proj MLP."""

    def __init__(self, ctx, config: dict, hidden_size: int) -> None:
        super().__init__(ctx, "action_in_proj")
        self.hidden_size = hidden_size
        self.feature_count = int(config.get("num_fourier_feats", 20))
        self.max_frequency = float(config.get("max_freq", 100.0))
        trunk_prefixes = sorted({
            int(match.group(2)): match.group(1)
            for key in self.weights.keys() if (match := re.search(
                r"(action_in_proj\.encoder\.trunk\.(\d+))\.weight$", key))
        }.items())
        self.trunk = [(AlpamayoActionNorm(ctx, prefix, 1e-5)
                       if self.weights.f16(prefix + ".weight").ndim == 1 else
                       Linear(ctx, prefix, rank=3, tensor_parallel=False))
                      for _, prefix in trunk_prefixes]
        norm_prefix = self.key("norm")
        self.norm = (AlpamayoActionNorm(ctx, norm_prefix, 1e-5)
                     if self.weights.has(norm_prefix + ".weight") else None)

    def forward(self, noise, time_steps):
        noise = noise
        features = [
            F.fourier_features(noise.slice_last_dim(index, 1, 3),
                               self.feature_count, self.max_frequency)
            for index in range(2)
        ]
        time = time_steps.reshape((-1, 1, 1))
        time_features = F.fourier_features(time, self.feature_count,
                                           self.max_frequency)
        carrier = noise.slice_last_dim(0, 1, 3) * F.constant(
            np.zeros((1, 1, 1), dtype=np.float16), "zero")
        features.append(time_features + carrier)
        hidden = F.concatenate(features, 2)
        for position, module in enumerate(self.trunk):
            hidden = module(hidden)
            if (isinstance(module, Linear) and position + 1 < len(self.trunk)):
                hidden = hidden.silu()
        if self.norm is not None:
            hidden = self.norm(hidden)
        if int(hidden.shape[-1]) not in (-1, self.hidden_size):
            raise ValueError("action input projection has unexpected width")
        return hidden


class AlpamayoActionDecoderLayer(Module):
    """Expert decoder layer used by Alpamayo action generation."""

    def __init__(self, ctx, prefix: str, num_heads: int, num_kv_heads: int,
                 head_dim: int, eps: float) -> None:
        super().__init__(ctx, prefix)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.eps = eps
        self.input_layernorm = AlpamayoActionNorm(ctx,
                                                  self.key("input_layernorm"),
                                                  eps)
        attention_prefix = self.key("self_attn")
        self.q_proj = Linear(ctx,
                             attention_prefix + ".q_proj",
                             rank=3,
                             tensor_parallel=False)
        self.k_proj = Linear(ctx,
                             attention_prefix + ".k_proj",
                             rank=3,
                             tensor_parallel=False)
        self.v_proj = Linear(ctx,
                             attention_prefix + ".v_proj",
                             rank=3,
                             tensor_parallel=False)
        self.q_norm = AlpamayoActionNorm(ctx, attention_prefix + ".q_norm",
                                         eps)
        self.k_norm = AlpamayoActionNorm(ctx, attention_prefix + ".k_norm",
                                         eps)
        self.o_proj = Linear(ctx,
                             attention_prefix + ".o_proj",
                             rank=3,
                             tensor_parallel=False)
        self.post_attention_layernorm = AlpamayoActionNorm(
            ctx, self.key("post_attention_layernorm"), eps)
        self.gate_proj = Linear(ctx,
                                self.key("mlp.gate_proj"),
                                rank=3,
                                tensor_parallel=False)
        self.up_proj = Linear(ctx,
                              self.key("mlp.up_proj"),
                              rank=3,
                              tensor_parallel=False)
        self.down_proj = Linear(ctx,
                                self.key("mlp.down_proj"),
                                rank=3,
                                tensor_parallel=False)

    def forward(self, hidden, *attention_inputs):
        normalized = self.input_layernorm(hidden)
        query = self.q_proj(normalized).reshape(
            (0, 0, self.num_heads, self.head_dim))
        key = self.k_proj(normalized).reshape(
            (0, 0, self.num_kv_heads, self.head_dim))
        query = self.q_norm(query, 4)
        key = self.k_norm(key, 4)
        value = self.v_proj(normalized).reshape(
            (0, 0, self.num_kv_heads, self.head_dim))
        attention, *present = self.attention(query, key, value,
                                             *attention_inputs)
        hidden = hidden + self.o_proj(attention)
        normalized = self.post_attention_layernorm(hidden)
        feed_forward = self.down_proj(
            self.gate_proj(normalized).silu() * self.up_proj(normalized))
        return (hidden + feed_forward, *present)

    def attention(self, query, key, value, past_key, past_value, rope,
                  context_lengths, cache_start, attention_pos_id):
        query = query.transpose((0, 2, 1, 3))
        key = key.transpose((0, 2, 1, 3))
        value = value.transpose((0, 2, 1, 3))
        half_dim = self.head_dim // 2
        rope_cos = rope.slice_last_dim(0, half_dim, 3).reshape(
            (-1, half_dim)).cast(trt.float16)
        rope_sin = rope.slice_last_dim(half_dim, half_dim, 3).reshape(
            (-1, half_dim)).cast(trt.float16)
        query = F.rotary_embedding(query, rope_cos, rope_sin, attention_pos_id,
                                   self.head_dim)
        key = F.rotary_embedding(key, rope_cos, rope_sin, attention_pos_id,
                                 self.head_dim)
        present_key = F.kv_cache_update(past_key, key, cache_start)
        present_value = F.kv_cache_update(past_value, value, cache_start)
        attention = F.scaled_dot_product_attention(
            query,
            present_key,
            present_value,
            key_value_lengths=context_lengths,
        )
        attention = attention.transpose((0, 2, 1, 3)).reshape(
            (0, 0, self.num_heads * self.head_dim))
        return attention, present_key, present_value


class AlpamayoActionModel(NetworkModule):
    """One Alpamayo flow-matching action denoising step."""

    layer_class = AlpamayoActionDecoderLayer

    @classmethod
    def from_config(cls, ctx):
        if ctx.args.reuse_llm_kv_cache:
            from .modeling_alpamayo_action_paged import \
                PagedAlpamayoActionModel
            return PagedAlpamayoActionModel(ctx, ctx.bundle)
        return cls(ctx, ctx.bundle)

    def __init__(self, ctx, bundle) -> None:
        super().__init__(ctx, "expert")
        self.root = bundle.root
        self.action = (self.root.get("expert_cfg")
                       or self.root.get("action_config") or {})
        self.layer_prefixes = sorted(
            {
                match.group(1)
                for key in self.weights.keys() if (match := re.search(
                    r"(expert\.layers\.\d+)\.input_layernorm\.weight$", key))
            },
            key=lambda prefix: int(prefix.rsplit(".", 1)[-1]))
        self.num_layers = len(self.layer_prefixes) or int(
            self.action.get("num_hidden_layers", 0))
        if self.num_layers <= 0:
            raise ValueError("action checkpoint has no expert decoder layers")
        self.num_heads = int(self.action.get("num_attention_heads", 16))
        self.head_dim = int(self.action.get("head_dim", 128))
        k_weight = self.weights.find_suffix(
            "expert.layers.0.self_attn.k_proj.weight")
        self.num_kv_heads = self.weights.store.shape(
            k_weight)[0] // self.head_dim
        self.hidden_size = int(
            self.action.get("hidden_size", self.num_heads * self.head_dim))
        self.diffusion_tokens = int((self.root.get("action_space_cfg")
                                     or {}).get("n_waypoints", 64))
        self.eps = float(self.action.get("rms_norm_eps", 1e-6))
        self.input_projection = AlpamayoActionInputProjection(
            ctx,
            self.root.get("action_in_proj_cfg") or {}, self.hidden_size)
        self.layers = [
            self.layer_class(ctx, prefix, self.num_heads, self.num_kv_heads,
                             self.head_dim, self.eps)
            for prefix in self.layer_prefixes
        ]
        self.norm = AlpamayoActionNorm(ctx, self.prefix + ".norm", self.eps)
        self.action_out_proj = Linear(ctx,
                                      "action_out_proj",
                                      rank=3,
                                      tensor_parallel=False)

    def input_tensors(self):
        cache_capacity = self.ctx.args.max_kv_cache_capacity
        return {
            "noise":
            self.add_input("noise_trajectory", trt.float32,
                           (-1, self.diffusion_tokens, 2)),
            "time0":
            self.add_input("time_steps_t0", trt.float32, (1, )),
            "time1":
            self.add_input("time_steps_t1", trt.float32, (1, )),
            "cache_start":
            self.add_input("kvcache_start_index", trt.int32, (-1, )),
            "rope":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, self.diffusion_tokens, self.head_dim)),
            "attention_pos_id":
            self.add_input("attention_pos_id", trt.int32,
                           (-1, self.diffusion_tokens)),
            "key_caches": [
                self.add_input(
                    f"k_cache_{index}", trt.float16,
                    (-1, self.num_kv_heads, cache_capacity, self.head_dim))
                for index in range(self.num_layers)
            ],
            "value_caches": [
                self.add_input(
                    f"v_cache_{index}", trt.float16,
                    (-1, self.num_kv_heads, cache_capacity, self.head_dim))
                for index in range(self.num_layers)
            ],
        }

    def forward(self, noise, time0, time1, cache_start, rope, attention_pos_id,
                key_caches, value_caches):
        hidden = self.input_projection(
            noise.cast(trt.float16),
            time0.reshape((-1, 1)).cast(trt.float32))
        diffusion = F.constant(np.array(self.diffusion_tokens, dtype=np.int32),
                               "diffusion_tokens")
        context_lengths = cache_start + diffusion
        # RotaryEmbedding indexes a shared 2D table; select each request's rows.
        rope_row_offsets = F.constant(
            (np.arange(self.ctx.args.max_batch_size, dtype=np.int32) *
             self.diffusion_tokens).reshape((-1, 1)), "rope_row_offsets")
        rope_row_offsets = F.dynamic_slice(rope_row_offsets, (0, 0),
                                           (F.shape_of(rope)[:1], 1))
        attention_pos_id = attention_pos_id + rope_row_offsets
        present_keys = []
        present_values = []
        for index, layer in enumerate(self.layers):
            hidden, present_key, present_value = layer(
                hidden, key_caches[index], value_caches[index], rope,
                context_lengths, cache_start, attention_pos_id)
            present_keys.append(present_key)
            present_values.append(present_value)
        prediction = self.action_out_proj(self.norm(hidden)).cast(trt.float32)
        delta = (time1 - time0).reshape((0, 1, 1)).cast(trt.float32)
        outputs = {
            "denoised_trajectory": noise + delta * prediction,
        }
        for index, tensor in enumerate(present_keys):
            outputs[f"present_k_cache_{index}"] = tensor
        for index, tensor in enumerate(present_values):
            outputs[f"present_v_cache_{index}"] = tensor
        return outputs
