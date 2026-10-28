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
"""Gemma4 visual checkpoint-direct graph."""

import numpy as np
import tensorrt as trt

from ...core import contracts
from ...ops import Linear, Module, NetworkModule
from ...ops import functional as F
from . import modeling_gemma4_layers


class Gemma4VisionNorm(Module):
    """Checkpoint-backed Gemma4 vision normalization."""

    def __init__(self, ctx, prefix: str, eps: float, rank: int = 2) -> None:
        super().__init__(ctx, prefix)
        self.eps = eps
        self.rank = rank

    def forward(self, hidden):
        return F.normalization(hidden, self.prefix, self.eps, self.rank)


class Gemma4VisionUnitRMSNorm(Module):
    """Gemma4 provider RMSNorm with an implicit unit scale."""

    def __init__(self, ctx, width: int, eps: float, rank: int) -> None:
        super().__init__(ctx, "vision_tower.unit_norm")
        self.weight = np.ones(width, dtype=np.float16)
        self.eps = eps
        self.rank = rank

    def forward(self, hidden):
        return F.rms_norm(hidden, self.weight, self.eps, self.rank)


class Gemma4VisionPatchEmbedder(Module):
    """Gemma4 flattened patch projection plus 2-D position embedding."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx, "vision_tower.patch_embedder")
        self.input_projection = modeling_gemma4_layers.Gemma4ClippableLinear(
            ctx, self.key("input_proj"), rank=2)

    def forward(self, pixels, positions):
        pixels = (pixels - np.float16(0.5)) * np.float16(2.0)
        hidden = self.input_projection(pixels)
        table = self.weights.f16(self.key("position_embedding_table"))
        positions = positions
        x_index = positions.slice_axis(1, 0, 1, 2).reshape((-1, ))
        y_index = positions.slice_axis(1, 1, 1, 2).reshape((-1, ))
        x_embedding = F.constant(table[0], "position_x").gather(x_index, 0)
        y_embedding = F.constant(table[1], "position_y").gather(y_index, 0)
        return hidden + x_embedding + y_embedding


class Gemma4VisionAttention(Module):
    """Gemma4 ViT attention block with q/k RMSNorm and rotary embedding."""

    def __init__(self, ctx, prefix: str, num_heads: int, num_kv_heads: int,
                 head_dim: int, eps: float) -> None:
        super().__init__(ctx, prefix)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = modeling_gemma4_layers.Gemma4ClippableLinear(
            ctx, self.key("q_proj"), rank=2)
        self.k_proj = modeling_gemma4_layers.Gemma4ClippableLinear(
            ctx, self.key("k_proj"), rank=2)
        self.v_proj = modeling_gemma4_layers.Gemma4ClippableLinear(
            ctx, self.key("v_proj"), rank=2)
        self.o_proj = modeling_gemma4_layers.Gemma4ClippableLinear(
            ctx, self.key("o_proj"), rank=2)
        self.q_norm = Gemma4VisionNorm(ctx, self.key("q_norm"), eps, 3)
        self.k_norm = Gemma4VisionNorm(ctx, self.key("k_norm"), eps, 3)
        self.v_norm = Gemma4VisionUnitRMSNorm(ctx, head_dim, eps, 3)

    def _repeat_kv(self, hidden):
        if self.num_heads == self.num_kv_heads:
            return hidden
        if self.num_heads % self.num_kv_heads:
            raise ValueError("vision KV heads must divide query heads")
        groups = self.num_heads // self.num_kv_heads
        heads = []
        for index in range(self.num_kv_heads):
            head = hidden.slice_axis(1, index, 1, 3)
            heads.extend([head] * groups)
        return F.concatenate(heads, 1)

    def forward(self, hidden, rotary, cu_seqlens, max_seqlen):
        query = self.q_proj(hidden)
        key = self.k_proj(hidden)
        value = self.v_proj(hidden)
        query = query.reshape((0, self.num_heads, self.head_dim))
        key = key.reshape((0, self.num_kv_heads, self.head_dim))
        value = value.reshape((0, self.num_kv_heads, self.head_dim))
        query = self.q_norm(query)
        key = self.k_norm(key)
        value = self.v_norm(value)
        query, key = F.apply_multidimensional_rope(query, key, rotary,
                                                   self.head_dim, 2)
        key = self._repeat_kv(key)
        value = self._repeat_kv(value)
        query = query * np.float16(np.sqrt(self.head_dim))
        attention = F.vit_attention(query, key, value, cu_seqlens, max_seqlen,
                                    self.num_heads, self.head_dim)
        attention = attention.reshape((0, self.num_heads * self.head_dim))
        return self.o_proj(attention)


class Gemma4VisionMLP(Module):
    """Gemma4 visual gated MLP."""

    def __init__(self, ctx, prefix: str, hidden_activation: str) -> None:
        super().__init__(ctx, prefix)
        self.hidden_activation = hidden_activation
        self.compute_fp32 = bool(
            ctx.bundle.root.get("vision_config", {}).get("standardize", False))
        self.gate_proj = modeling_gemma4_layers.Gemma4ClippableLinear(
            ctx, self.key("gate_proj"), rank=2)
        self.up_proj = modeling_gemma4_layers.Gemma4ClippableLinear(
            ctx, self.key("up_proj"), rank=2)
        self.down_proj = modeling_gemma4_layers.Gemma4ClippableLinear(
            ctx, self.key("down_proj"), rank=2)

    def forward(self, hidden):
        gate = self.gate_proj(hidden,
                              compute_fp32=self.compute_fp32).activation(
                                  self.hidden_activation)
        up = self.up_proj(hidden, compute_fp32=self.compute_fp32)
        return self.down_proj(gate * up, compute_fp32=self.compute_fp32)


class Gemma4VisionEncoderLayer(Module):
    """One Gemma4 vision encoder layer."""

    def __init__(self, ctx, prefix: str, num_heads: int, num_kv_heads: int,
                 head_dim: int, eps: float, hidden_activation: str) -> None:
        super().__init__(ctx, prefix)
        self.self_attn = Gemma4VisionAttention(ctx, self.key("self_attn"),
                                               num_heads, num_kv_heads,
                                               head_dim, eps)
        self.mlp = Gemma4VisionMLP(ctx, self.key("mlp"), hidden_activation)
        self.input_layernorm = Gemma4VisionNorm(ctx,
                                                self.key("input_layernorm"),
                                                eps)
        self.post_attention_layernorm = Gemma4VisionNorm(
            ctx, self.key("post_attention_layernorm"), eps)
        self.pre_feedforward_layernorm = Gemma4VisionNorm(
            ctx, self.key("pre_feedforward_layernorm"), eps)
        self.post_feedforward_layernorm = Gemma4VisionNorm(
            ctx, self.key("post_feedforward_layernorm"), eps)

    def forward(self, hidden, rotary, cu_seqlens, max_seqlen):
        residual = hidden
        normalized = self.input_layernorm(hidden)
        attention = self.self_attn(normalized, rotary, cu_seqlens, max_seqlen)
        attention = self.post_attention_layernorm(attention)
        hidden = residual + attention
        residual = hidden
        normalized = self.pre_feedforward_layernorm(hidden)
        feed_forward = self.mlp(normalized)
        feed_forward = self.post_feedforward_layernorm(feed_forward)
        return residual + feed_forward


class Gemma4VisionPooler(Module):
    """Pool Gemma4 visual tokens and project to the text embedding width."""

    def __init__(self, ctx, hidden_size: int, eps: float) -> None:
        super().__init__(ctx, "vision_tower")
        self.hidden_size = hidden_size
        self.norm = Gemma4VisionUnitRMSNorm(ctx, hidden_size, eps, 2)
        self.projection = Linear(ctx,
                                 "embed_vision.embedding_projection",
                                 rank=2,
                                 tensor_parallel=False)

    def forward(self, hidden, pooling):
        pooled = pooling.cast(trt.float32).matmul(hidden.cast(trt.float32))
        pooled = pooled * np.float32(np.sqrt(self.hidden_size))
        if self.weights.has(self.key("std_bias")):
            pooled = ((pooled - F.constant(
                self.weights.f32(self.key("std_bias")).reshape(1, -1),
                "std_bias")) * F.constant(
                    self.weights.f32(self.key("std_scale")).reshape(1, -1),
                    "std_scale"))
        pooled = pooled.cast(trt.float16)
        return self.projection(self.norm(pooled))


class Gemma4VisionModel(NetworkModule):
    """HF-style Gemma4 vision tower graph."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle)

    def __init__(self, ctx, bundle) -> None:
        super().__init__(ctx, "vision_tower")
        self.visual = bundle.component_dict(contracts.Component.VISUAL)
        self.hidden_size = int(self.visual["hidden_size"])
        self.num_heads = int(self.visual["num_attention_heads"])
        self.num_kv_heads = int(
            self.visual.get("num_key_value_heads", self.num_heads))
        self.head_dim = int(
            self.visual.get("head_dim", self.hidden_size // self.num_heads))
        self.patch_size = int(self.visual["patch_size"])
        self.eps = float(self.visual.get("rms_norm_eps", 1e-6))
        hidden_activation = str(
            self.visual.get("hidden_activation", "gelu_pytorch_tanh"))
        self.patch_embedder = Gemma4VisionPatchEmbedder(ctx)
        self.layers = [
            Gemma4VisionEncoderLayer(ctx, prefix, self.num_heads,
                                     self.num_kv_heads, self.head_dim,
                                     self.eps, hidden_activation)
            for prefix in self.weights.layer_prefixes((
                r"(.+vision_tower\.encoder\.layers\.\d+)\.input_layernorm\.weight$",
            ))
        ]
        self.pooler = Gemma4VisionPooler(ctx, self.hidden_size, self.eps)

    def input_tensors(self):
        return {
            "pixels":
            self.add_input("input", trt.float16,
                           (-1, 3 * self.patch_size * self.patch_size)),
            "positions":
            self.add_input("pixel_position_ids", trt.int64, (-1, 2)),
            "rotary":
            self.add_input("rotary_pos_emb", trt.float32, (-1, self.head_dim)),
            "cu_seqlens":
            self.add_input("cu_seqlens", trt.int32, (-1, )),
            "max_seqlen":
            self.add_input("max_seqlen_carrier", trt.int32, (-1, )),
            "pooling":
            self.add_input("pooling_weights", trt.float16, (-1, -1)),
        }

    def forward(self, pixels, positions, rotary, cu_seqlens, max_seqlen,
                pooling):
        hidden = self.patch_embedder(pixels, positions)
        for layer in self.layers:
            hidden = layer(hidden, rotary, cu_seqlens, max_seqlen)
        return {"output": self.pooler(hidden, pooling)}
