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
"""GR00T N1.7 vision-language backbone: the truncated Cosmos-Reason2-2B decoder,
its VL LayerNorm and VL self-attention refinement, then the DiT cross-attention
K/V projections, hoisted here because they read only the step-invariant VL context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import tensorrt as trt

from tensorrt_edgellm.models.gr00t_n1d7.geometry import BLOCK_NORM_EPS

from ...ops import (BuildContext, LayerNorm, Linear, Module, NetworkModule,
                    RMSNorm, Tensor)
from ...ops import functional as F
from .runtime_config import geometry_from_bundle

if TYPE_CHECKING:
    from ...core.builder import BuildArgs
    from ...core.bundle import BundleConfig


def repeat_kv(heads: Tensor, num_kv_heads: int, repeats: int,
              head_dim: int) -> Tensor:
    """Expand ``[B, S, H_kv, D]`` to ``[B, S, H_kv * repeats, D]``."""
    if repeats == 1:
        return heads
    expanded = F.concatenate([heads.unsqueeze(3, 4)] * repeats, dim=3)
    return expanded.reshape((0, 0, num_kv_heads * repeats, head_dim))


class Gr00tGatedMLP(Module):
    """Qwen3 SwiGLU feed-forward."""

    def __init__(self, ctx: BuildContext, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.gate_proj = Linear(ctx,
                                self.key("gate_proj"),
                                tensor_parallel=False)
        self.up_proj = Linear(ctx, self.key("up_proj"), tensor_parallel=False)
        self.down_proj = Linear(ctx,
                                self.key("down_proj"),
                                tensor_parallel=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate = self.gate_proj(hidden_states).silu()
        return self.down_proj(gate * self.up_proj(hidden_states))


class Gr00tDecoderAttention(Module):
    """Causal Qwen3-VL self-attention with per-head q/k RMSNorm."""

    def __init__(self, ctx: BuildContext, prefix: str, config: dict) -> None:
        super().__init__(ctx, prefix)
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.repeats = self.num_heads // self.num_kv_heads
        eps = float(config.get("rms_norm_eps", 1e-6))
        self.q_proj = Linear(ctx, self.key("q_proj"), tensor_parallel=False)
        self.k_proj = Linear(ctx, self.key("k_proj"), tensor_parallel=False)
        self.v_proj = Linear(ctx, self.key("v_proj"), tensor_parallel=False)
        self.o_proj = Linear(ctx, self.key("o_proj"), tensor_parallel=False)
        self.q_norm = RMSNorm(ctx, self.key("q_norm"), eps, rank=4)
        self.k_norm = RMSNorm(ctx, self.key("k_norm"), eps, rank=4)

    def forward(self, hidden_states: Tensor, rope_cos: Tensor,
                rope_sin: Tensor, position_ids: Tensor) -> Tensor:
        query = self.q_proj(hidden_states).reshape(
            (0, 0, self.num_heads, self.head_dim))
        key = self.k_proj(hidden_states).reshape(
            (0, 0, self.num_kv_heads, self.head_dim))
        query = self.q_norm(query)
        key = self.k_norm(key)
        value = self.v_proj(hidden_states).reshape(
            (0, 0, self.num_kv_heads, self.head_dim))

        key = repeat_kv(key, self.num_kv_heads, self.repeats, self.head_dim)
        value = repeat_kv(value, self.num_kv_heads, self.repeats,
                          self.head_dim)

        query = F.rotary_embedding(
            query.transpose((0, 2, 1, 3)).cast(trt.float16), rope_cos,
            rope_sin, position_ids, self.head_dim)
        key = F.rotary_embedding(
            key.transpose((0, 2, 1, 3)).cast(trt.float16), rope_cos, rope_sin,
            position_ids, self.head_dim)
        value = value.transpose((0, 2, 1, 3))
        attended = F.scaled_dot_product_attention(query,
                                                  key,
                                                  value,
                                                  scale=self.head_dim**-0.5,
                                                  is_causal=True)
        attended = attended.transpose((0, 2, 1, 3)).reshape(
            (0, 0, self.num_heads * self.head_dim))
        return self.o_proj(attended)


class Gr00tDecoderLayer(Module):
    """One truncated Cosmos-Reason2-2B decoder layer."""

    def __init__(self, ctx: BuildContext, index: int, config: dict) -> None:
        super().__init__(ctx, f"model.layers.{index}")
        eps = float(config.get("rms_norm_eps", 1e-6))
        self.input_layernorm = RMSNorm(ctx, self.key("input_layernorm"), eps)
        self.self_attn = Gr00tDecoderAttention(ctx, self.key("self_attn"),
                                               config)
        self.post_attention_layernorm = RMSNorm(
            ctx, self.key("post_attention_layernorm"), eps)
        self.mlp = Gr00tGatedMLP(ctx, self.key("mlp"))

    def forward(self, hidden_states: Tensor, rope_cos: Tensor,
                rope_sin: Tensor, position_ids: Tensor,
                deepstack: Optional[Tensor]) -> Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), rope_cos, rope_sin,
            position_ids)
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states))
        if deepstack is not None:
            hidden_states = hidden_states + deepstack
        return hidden_states


class Gr00tVlSelfAttention(Module):
    """Non-causal refinement block over the VL context: diffusers
    ``BasicTransformerBlock`` with layer norms and ``gelu-approximate``."""

    def __init__(self, ctx: BuildContext, prefix: str, num_heads: int,
                 head_dim: int) -> None:
        super().__init__(ctx, prefix)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.norm1 = LayerNorm(ctx, self.key("norm1"), BLOCK_NORM_EPS, rank=3)
        self.norm3 = LayerNorm(ctx, self.key("norm3"), BLOCK_NORM_EPS, rank=3)
        self.to_q = Linear(ctx, self.key("attn1.to_q"), tensor_parallel=False)
        self.to_k = Linear(ctx, self.key("attn1.to_k"), tensor_parallel=False)
        self.to_v = Linear(ctx, self.key("attn1.to_v"), tensor_parallel=False)
        self.to_out = Linear(ctx,
                             self.key("attn1.to_out.0"),
                             tensor_parallel=False)
        self.ff_in = Linear(ctx,
                            self.key("ff.net.0.proj"),
                            tensor_parallel=False)
        self.ff_out = Linear(ctx, self.key("ff.net.2"), tensor_parallel=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        normed = self.norm1(hidden_states)
        shape = (0, 0, self.num_heads, self.head_dim)
        query = self.to_q(normed).reshape(shape).transpose((0, 2, 1, 3))
        key = self.to_k(normed).reshape(shape).transpose((0, 2, 1, 3))
        value = self.to_v(normed).reshape(shape).transpose((0, 2, 1, 3))
        attended = F.scaled_dot_product_attention(query,
                                                  key,
                                                  value,
                                                  scale=self.head_dim**-0.5)
        attended = attended.transpose((0, 2, 1, 3)).reshape(
            (0, 0, self.num_heads * self.head_dim))
        hidden_states = hidden_states + self.to_out(attended)
        return hidden_states + self.ff_out(
            self.ff_in(self.norm3(hidden_states)).gelu_tanh())


class Gr00tVLBackbone(NetworkModule):
    """Prefill VL tower emitting the DiT's step-invariant cross-attention K/V."""

    @classmethod
    def from_config(cls, ctx: BuildContext) -> Gr00tVLBackbone:
        return cls(ctx, ctx.bundle, ctx.args)

    def __init__(self, ctx: BuildContext, bundle: BundleConfig,
                 args: BuildArgs) -> None:
        super().__init__(ctx)
        from ...core import contracts
        config = bundle.component_dict(contracts.Component.VL_BACKBONE)
        self.geometry = geometry_from_bundle(bundle, args)
        self.hidden_size = int(config["hidden_size"])
        self.head_dim = int(config["head_dim"])
        self.num_layers = int(config["num_hidden_layers"])
        self.num_deepstack = len(
            bundle.component_dict(
                contracts.Component.VISUAL).get("deepstack_visual_indexes")
            or [])
        self.layers = [
            Gr00tDecoderLayer(ctx, index, config)
            for index in range(self.num_layers)
        ]
        self.norm = RMSNorm(ctx, "model.norm",
                            float(config.get("rms_norm_eps", 1e-6)))
        self.vlln = LayerNorm(ctx, "vlln", 1e-5, rank=3)
        self.vl_self_attention = [
            Gr00tVlSelfAttention(
                ctx, f"vl_self_attention.transformer_blocks.{index}",
                self.geometry.vl_sa_heads, self.geometry.vl_sa_head_dim)
            for index in range(self.geometry.vl_sa_layers)
        ]
        self.cross_k = {
            index:
            Linear(ctx,
                   f"model.transformer_blocks.{index}.attn1.to_k",
                   tensor_parallel=False)
            for index in self.geometry.cross_blocks
        }
        self.cross_v = {
            index:
            Linear(ctx,
                   f"model.transformer_blocks.{index}.attn1.to_v",
                   tensor_parallel=False)
            for index in self.geometry.cross_blocks
        }

    def input_tensors(self) -> Dict[str, Tensor]:
        tensors = {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, -1, self.hidden_size)),
            "rope_rotary_cos_sin":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, -1, self.head_dim)),
            "attention_pos_id":
            self.add_input("attention_pos_id", trt.int32, (-1, -1)),
        }
        tensors["deepstack_embeds"] = [
            self.add_input(f"deepstack_embeds_{index}", trt.float16,
                           (-1, -1, self.hidden_size))
            for index in range(self.num_deepstack)
        ]
        return tensors

    def forward(self, inputs_embeds: Tensor, rope_rotary_cos_sin: Tensor,
                attention_pos_id: Tensor,
                deepstack_embeds: list) -> Dict[str, Tensor]:
        half = self.head_dim // 2
        rope_cos = rope_rotary_cos_sin.slice_last_dim(0, half, 3).reshape(
            (-1, half)).cast(trt.float16)
        rope_sin = rope_rotary_cos_sin.slice_last_dim(half, half, 3).reshape(
            (-1, half)).cast(trt.float16)

        hidden_states = inputs_embeds
        for index, layer in enumerate(self.layers):
            injected = (deepstack_embeds[index]
                        if index < len(deepstack_embeds) else None)
            hidden_states = layer(hidden_states, rope_cos, rope_sin,
                                  attention_pos_id, injected)

        features = self.vlln(self.norm(hidden_states))
        for block in self.vl_self_attention:
            features = block(features)

        heads = self.geometry.num_heads
        head_dim = self.geometry.head_dim
        outputs = {}
        for index in self.geometry.cross_blocks:
            outputs[f"cross_k_layer{index:02d}"] = self.cross_k[index](
                features).reshape((0, 0, heads, head_dim))
            outputs[f"cross_v_layer{index:02d}"] = self.cross_v[index](
                features).reshape((0, 0, heads, head_dim))
        return outputs
