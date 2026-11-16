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
"""Checkpoint-direct Qwen3-VL text tower that conditions MiniMax-H3.

A text-only prompt gives every multimodal RoPE axis the same position, so the
interleaved three-axis RoPE reduces to ordinary one-dimensional RoPE.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from ...ops import Embedding, Linear, Module, NetworkModule, RMSNorm
from ...ops import functional as F
from . import configuration
from .layers import apply_rotary, scaled_projection

_PREFIX = "model.language_model"


class Qwen3VLTextAttention(Module):
    """Causal grouped-query attention with per-head Q/K RMSNorm."""

    def __init__(self, ctx, prefix: str, config: dict) -> None:
        super().__init__(ctx, prefix)
        eps = float(config["rms_norm_eps"])
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.q_proj = Linear(ctx, self.key("q_proj"), tensor_parallel=False)
        self.k_proj = Linear(ctx, self.key("k_proj"), tensor_parallel=False)
        self.v_proj = Linear(ctx, self.key("v_proj"), tensor_parallel=False)
        self.o_proj = Linear(ctx, self.key("o_proj"), tensor_parallel=False)
        self.q_norm = RMSNorm(ctx, self.key("q_norm"), eps, rank=4)
        self.k_norm = RMSNorm(ctx, self.key("k_norm"), eps, rank=4)

    def forward(self, hidden, rope):
        query = self.q_norm(
            self.q_proj(hidden).reshape((0, 0, self.num_heads, self.head_dim)))
        key = self.k_norm(
            self.k_proj(hidden).reshape(
                (0, 0, self.num_kv_heads, self.head_dim)))
        value = self.v_proj(hidden).reshape(
            (0, 0, self.num_kv_heads, self.head_dim))
        query_rope, key_rope = rope
        attended = F.scaled_dot_product_attention(
            apply_rotary(query, *query_rope).transpose((0, 2, 1, 3)),
            apply_rotary(key, *key_rope).transpose((0, 2, 1, 3)),
            value.transpose((0, 2, 1, 3)),
            scale=1.0,
            is_causal=True)
        attended = attended.transpose((0, 2, 1, 3)).reshape(
            (0, 0, self.num_heads * self.head_dim))
        return scaled_projection(self.o_proj, attended)


class Qwen3VLTextDecoderLayer(Module):

    def __init__(self, ctx, index: int, config: dict) -> None:
        super().__init__(ctx, f"{_PREFIX}.layers.{index}")
        eps = float(config["rms_norm_eps"])
        self.input_layernorm = RMSNorm(ctx, self.key("input_layernorm"), eps)
        self.self_attn = Qwen3VLTextAttention(ctx, self.key("self_attn"),
                                              config)
        self.post_attention_layernorm = RMSNorm(
            ctx, self.key("post_attention_layernorm"), eps)
        self.gate_proj = Linear(ctx,
                                self.key("mlp.gate_proj"),
                                tensor_parallel=False)
        self.up_proj = Linear(ctx,
                              self.key("mlp.up_proj"),
                              tensor_parallel=False)
        self.down_proj = Linear(ctx,
                                self.key("mlp.down_proj"),
                                tensor_parallel=False)

    def forward(self, hidden, rope):
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), rope)
        normed = self.post_attention_layernorm(hidden)
        activated = (self.gate_proj(normed).cast(trt.float32).silu() *
                     self.up_proj(normed).cast(trt.float32))
        return hidden + scaled_projection(self.down_proj, activated)


class MiniMaxH3TextEncoder(NetworkModule):
    """Return the unnormalized hidden state after the conditioning layer."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle)

    def __init__(self, ctx, bundle) -> None:
        super().__init__(ctx)
        config = configuration.component_config(
            bundle.root, configuration.Component.TEXT_ENCODER)["text_config"]
        num_layers = int(config["num_hidden_layers"])
        if num_layers < configuration.TEXT_ENCODER_LAYERS:
            raise ValueError("MiniMax-H3 text encoder has too few layers")
        self.head_dim = int(config["head_dim"])
        self.rope_theta = float(config["rope_theta"])
        self.embed_tokens = Embedding(ctx, f"{_PREFIX}.embed_tokens")
        self.layers = [
            Qwen3VLTextDecoderLayer(ctx, index, config)
            for index in range(configuration.TEXT_ENCODER_LAYERS)
        ]

    def input_tensors(self):
        return {"input_ids": self.add_input("input_ids", trt.int32, (1, -1))}

    def _rope(self, input_ids):
        """Return the query and key RoPE ``(cos, sin, rotary_dim)`` tables.

        The query tables carry the softmax scale: TensorRT 10.16 drops the
        causal mask of dynamic-shape attention whose FP16 query or key is
        scaled just before it.
        """
        inverse = 1.0 / self.rope_theta**(
            np.arange(0, self.head_dim, 2, dtype=np.float32) / self.head_dim)
        angles = (np.arange(configuration.MAX_TEXT_TOKENS,
                            dtype=np.float32)[:, None] *
                  inverse.astype(np.float32)[None])
        angles = np.concatenate((angles, angles), axis=-1)
        length = F.shape_of(input_ids)[1:2]

        def tables(prefix, scale):
            result = []
            for name, function in (("cos", np.cos), ("sin", np.sin)):
                constant = F.constant(
                    (function(angles) * scale).astype(np.float32),
                    f"{prefix}_rope_{name}")
                result.append(
                    F.dynamic_slice(constant, (0, 0),
                                    (length, self.head_dim)).reshape(
                                        (1, -1, 1, self.head_dim)))
            return (*result, self.head_dim)

        return tables("query", self.head_dim**-0.5), tables("key", 1.0)

    def forward(self, input_ids):
        rope = self._rope(input_ids)
        hidden = self.embed_tokens(input_ids).cast(trt.float32)
        for layer in self.layers:
            hidden = layer(hidden, rope)
        return {"text_hidden": hidden.cast(trt.float16)}
