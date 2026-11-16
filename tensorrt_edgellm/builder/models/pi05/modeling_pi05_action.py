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
"""Checkpoint-direct pi0.5 action and hoisted-condition graphs."""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from tensorrt_edgellm.models.pi05.component_contracts import (
    DIFFUSION_DENOISE_PHASE, KV_TOKENS_PER_PAGE, packed_mask_words)

from ...ops import Linear, Module, NetworkModule
from ...ops import functional as F
from ...ops.ragged import RaggedDecoderInputs
from .configuration import Pi05Geometry
from .modeling_pi05_prefix import Pi05MLP

_TIME_MIN_PERIOD = 4e-3
_TIME_MAX_PERIOD = 4.0


class Pi05TimeEmbedder(Module):
    """Sinusoidal timestep embedding followed by two SiLU linears."""

    def __init__(self, ctx, hidden_size: int) -> None:
        super().__init__(ctx)
        if hidden_size % 2:
            raise ValueError("pi0.5 action hidden size must be even")
        self.hidden_size = hidden_size
        fraction = np.linspace(0.0, 1.0, hidden_size // 2, dtype=np.float32)
        period = _TIME_MIN_PERIOD * (_TIME_MAX_PERIOD /
                                     _TIME_MIN_PERIOD)**fraction
        self.frequency = (2.0 * math.pi / period).reshape(1, -1)
        self.time_mlp_in = Linear(ctx,
                                  "time_mlp_in",
                                  rank=2,
                                  tensor_parallel=False)
        self.time_mlp_out = Linear(ctx,
                                   "time_mlp_out",
                                   rank=2,
                                   tensor_parallel=False)

    def forward(self, timestep):
        phase = timestep.reshape(
            (-1, 1)) * F.constant(self.frequency, "pi05_timestep_frequency")
        embedding = F.concatenate((phase.sin(), phase.cos()),
                                  1).cast(trt.float16)
        return self.time_mlp_out(self.time_mlp_in(embedding).silu()).silu()


class Pi05AdaRMSNorm(Module):
    """RMSNorm modulated by one packed scale/shift/gate row."""

    def __init__(self, ctx, prefix: str, hidden_size: int, eps: float,
                 site_index: int) -> None:
        super().__init__(ctx, prefix)
        self.hidden_size = hidden_size
        self.eps = eps
        self.site_index = site_index

    def forward(self, hidden_states, modulation):
        normalized = F.rms_norm(hidden_states,
                                np.ones(self.hidden_size, dtype=np.float16),
                                self.eps,
                                rank=3).cast(trt.float32)
        packed = modulation.slice_axis(1, self.site_index, 1, 3)
        scale = packed.slice_last_dim(0, self.hidden_size, 3).cast(trt.float32)
        shift = packed.slice_last_dim(self.hidden_size, self.hidden_size,
                                      3).cast(trt.float32)
        gate = packed.slice_last_dim(2 * self.hidden_size, self.hidden_size, 3)
        modulated = normalized * (scale + np.float32(1.0)) + shift
        return modulated.cast(trt.float16), gate


class Pi05ActionAttention(Module):
    """Paged tree attention over prefix and action tokens."""

    def __init__(self, ctx, prefix: str, config: dict) -> None:
        super().__init__(ctx, prefix)
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.q_proj = Linear(ctx,
                             self.key("q_proj"),
                             rank=2,
                             tensor_parallel=False)
        self.k_proj = Linear(ctx,
                             self.key("k_proj"),
                             rank=2,
                             tensor_parallel=False)
        self.v_proj = Linear(ctx,
                             self.key("v_proj"),
                             rank=2,
                             tensor_parallel=False)
        self.o_proj = Linear(ctx,
                             self.key("o_proj"),
                             rank=2,
                             tensor_parallel=False)

    def forward(self, hidden_states, kv_cache, rope_rotary_cos_sin, ragged,
                attention_mask, attention_pos_id):
        tokens = hidden_states.reshape((-1, int(hidden_states.shape[-1])))
        qkv = F.concatenate(
            (self.q_proj(tokens), self.k_proj(tokens), self.v_proj(tokens)), 1)
        attended, present = F.attention(
            qkv,
            kv_cache,
            rope_rotary_cos_sin,
            ragged,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            sliding_window_size=0,
            attention_scale=self.head_dim**-0.5,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
        )
        attended = attended.reshape((-1, self.num_heads * self.head_dim))
        return self.o_proj(attended), present


class Pi05ActionLayer(Module):
    """One action expert block with gated residuals."""

    def __init__(self, ctx, index: int, config: dict, first_site: int) -> None:
        prefix = f"layers.{index}"
        super().__init__(ctx, prefix)
        hidden_size = int(config["hidden_size"])
        eps = float(config.get("rms_norm_eps", 1e-6))
        self.input_layernorm = Pi05AdaRMSNorm(ctx, self.key("input_layernorm"),
                                              hidden_size, eps, first_site)
        self.self_attn = Pi05ActionAttention(ctx, self.key("self_attn"),
                                             config)
        self.post_attention_layernorm = Pi05AdaRMSNorm(
            ctx, self.key("post_attention_layernorm"), hidden_size, eps,
            first_site + 1)
        self.mlp = Pi05MLP(ctx, self.key("mlp"))

    def forward(self, hidden_states, modulation, kv_cache, rope_rotary_cos_sin,
                ragged, attention_mask, attention_pos_id, horizon: int):
        normalized, gate = self.input_layernorm(hidden_states, modulation)
        attention, present = self.self_attn(normalized, kv_cache,
                                            rope_rotary_cos_sin, ragged,
                                            attention_mask, attention_pos_id)
        attention = attention.reshape(
            (-1, horizon, int(hidden_states.shape[-1])))
        hidden_states = hidden_states + attention * gate
        normalized, gate = self.post_attention_layernorm(
            hidden_states, modulation)
        hidden_states = hidden_states + self.mlp(normalized) * gate
        return hidden_states, present


class Pi05ActionModel(NetworkModule):
    """One velocity-prediction step of the pi0.5 denoising loop."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, Pi05Geometry.from_bundle(ctx.bundle, ctx.args))

    def __init__(self, ctx, geometry: Pi05Geometry) -> None:
        super().__init__(ctx)
        self.geometry = geometry
        self.config = geometry.action
        self.hidden_size = int(self.config["hidden_size"])
        self.num_layers = int(self.config["num_hidden_layers"])
        self.num_kv_heads = int(self.config["num_key_value_heads"])
        self.head_dim = int(self.config["head_dim"])
        self.action_in_proj = Linear(ctx,
                                     "action_in_proj",
                                     rank=3,
                                     tensor_parallel=False)
        self.layers = [
            Pi05ActionLayer(ctx, index, self.config, 2 * index)
            for index in range(self.num_layers)
        ]
        self.norm = Pi05AdaRMSNorm(
            ctx, "norm", self.hidden_size,
            float(self.config.get("rms_norm_eps", 1e-6)), 2 * self.num_layers)
        self.action_out_proj = Linear(ctx,
                                      "action_out_proj",
                                      rank=3,
                                      tensor_parallel=False)

    def input_tensors(self):
        horizon = self.geometry.action_horizon
        pages = self.geometry.pages_per_seq
        return {
            "noise_trajectory":
            self.add_input("noise_trajectory", trt.float32,
                           (-1, horizon, self.geometry.action_dim)),
            "adarms_modulation":
            self.add_input("adarms_modulation", trt.float16,
                           (-1, self.geometry.num_adarms_sites,
                            self.geometry.modulation_dim)),
            "rope_rotary_cos_sin":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, self.head_dim)),
            "attention_pos_id":
            self.add_input("attention_pos_id", trt.int32, (-1, )),
            "query_lengths":
            self.add_input("query_lengths", trt.int32, (-1, )),
            "kvcache_start_index":
            self.add_input("kvcache_start_index", trt.int32, (-1, )),
            "kv_page_table":
            self.add_input("kv_page_table", trt.int32, (-1, 2, pages)),
            "attention_mask":
            self.add_input("attention_mask", trt.int32,
                           (-1, packed_mask_words(horizon))),
            "query_start_offsets":
            self.add_input("query_start_offsets", trt.int32, (-1, )),
            "attention_sequence_lengths":
            self.add_input("attention_sequence_lengths", trt.int32, (-1, )),
            "execution_phase_marker":
            self.add_input("execution_phase_marker", trt.int32,
                           (DIFFUSION_DENOISE_PHASE, )),
            "context_sequence_count_carrier":
            self.add_input("context_sequence_count_carrier", trt.int32, (0, )),
            "kv_caches": [
                self.add_input(f"kv_cache_layer{index:02d}", trt.float16,
                               (2, -1, KV_TOKENS_PER_PAGE, self.num_kv_heads,
                                self.head_dim))
                for index in range(self.num_layers)
            ],
        }

    def forward(self, noise_trajectory, adarms_modulation, rope_rotary_cos_sin,
                attention_pos_id, query_lengths, kvcache_start_index,
                kv_page_table, attention_mask, query_start_offsets,
                attention_sequence_lengths, execution_phase_marker,
                context_sequence_count_carrier, kv_caches):
        ragged = RaggedDecoderInputs(
            positions=None,
            query_start_offsets=query_start_offsets,
            query_lengths=query_lengths,
            past_lengths=kvcache_start_index,
            attention_sequence_lengths=attention_sequence_lengths,
            state_indices=None,
            logits_indices=None,
            execution_phase_marker=execution_phase_marker,
            context_sequence_count_carrier=context_sequence_count_carrier,
            kv_page_table=kv_page_table,
        )
        hidden_states = self.action_in_proj(noise_trajectory.cast(trt.float16))
        presents = []
        for index, layer in enumerate(self.layers):
            hidden_states, present = layer(hidden_states, adarms_modulation,
                                           kv_caches[index],
                                           rope_rotary_cos_sin, ragged,
                                           attention_mask, attention_pos_id,
                                           self.geometry.action_horizon)
            presents.append(present)
        hidden_states, _ = self.norm(hidden_states, adarms_modulation)
        outputs = {
            "action_pred":
            self.action_out_proj(hidden_states).cast(trt.float32)
        }
        outputs.update({
            f"present_kv_cache_layer{index:02d}": present
            for index, present in enumerate(presents)
        })
        return outputs


class Pi05CondModel(NetworkModule):
    """Precompute every AdaRMS modulation row for a denoise schedule."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, Pi05Geometry.from_bundle(ctx.bundle, ctx.args))

    def __init__(self, ctx, geometry: Pi05Geometry) -> None:
        super().__init__(ctx)
        self.geometry = geometry
        hidden_size = int(geometry.action["hidden_size"])
        self.time_embedder = Pi05TimeEmbedder(ctx, hidden_size)
        prefixes = []
        for index in range(int(geometry.action["num_hidden_layers"])):
            prefixes.extend((f"layers.{index}.input_layernorm.dense",
                             f"layers.{index}.post_attention_layernorm.dense"))
        prefixes.append("norm.dense")
        self.modulation_weight = np.concatenate(
            [self.weights.f16(prefix + ".weight") for prefix in prefixes],
            axis=0)
        self.modulation_bias = np.concatenate(
            [self.weights.f16(prefix + ".bias") for prefix in prefixes],
            axis=0)

    def input_tensors(self):
        return {"timestep": self.add_input("timestep", trt.float32, (-1, ))}

    def forward(self, timestep):
        condition = self.time_embedder(timestep)
        modulation = F.linear_with_weights(condition,
                                           self.modulation_weight,
                                           self.modulation_bias,
                                           rank=2)
        return {
            "adarms_modulation":
            modulation.reshape((-1, self.geometry.num_adarms_sites,
                                self.geometry.modulation_dim))
        }
