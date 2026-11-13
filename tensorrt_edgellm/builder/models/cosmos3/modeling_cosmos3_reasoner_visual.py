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
"""Checkpoint-direct SigLIP2 tower for the Cosmos3 reasoner."""

from __future__ import annotations

import math

import tensorrt as trt

from ...ops import LayerNorm, Linear, Module, NetworkModule
from ...ops import functional as F


class Cosmos3ReasonerVisionEmbeddings(Module):
    """Flattened patch projection and host-indexed position interpolation."""

    def __init__(self, ctx, hidden_size: int) -> None:
        super().__init__(ctx, "model.visual.embeddings")
        self.patch_embedding = Linear(ctx,
                                      self.key("patch_embedding"),
                                      rank=2,
                                      tensor_parallel=False)
        self.position_key = self.key("position_embedding.weight")
        position_shape = self.weights.store.shape(
            self.weights.checkpoint_key(self.position_key))
        if tuple(position_shape[1:]) != (hidden_size, ):
            raise ValueError(
                "Cosmos3 reasoner position embedding width is inconsistent")

    def forward(self, pixel_values, position_indices, position_weights):
        hidden_states = self.patch_embedding(pixel_values)
        position_table = F.constant(self.weights.f16(self.position_key),
                                    "cosmos_position_embedding")
        position = position_table.gather(position_indices, 0)
        position = position * position_weights.reshape((4, -1, 1))
        return hidden_states + position.sum(dim=0)


class Cosmos3ReasonerVisionAttention(Module):
    """Packed bidirectional visual self-attention."""

    def __init__(self, ctx, prefix: str, hidden_size: int,
                 num_heads: int) -> None:
        super().__init__(ctx, prefix)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
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
        self.out_proj = Linear(ctx,
                               self.key("out_proj"),
                               rank=2,
                               tensor_parallel=False)

    def forward(self, hidden_states, cu_seqlens, max_seqlen):
        query = self.q_proj(hidden_states).reshape(
            (-1, self.num_heads, self.head_dim))
        key = self.k_proj(hidden_states).reshape(
            (-1, self.num_heads, self.head_dim))
        value = self.v_proj(hidden_states).reshape(
            (-1, self.num_heads, self.head_dim))
        attended = F.vit_attention(query, key, value, cu_seqlens, max_seqlen,
                                   self.num_heads, self.head_dim)
        return self.out_proj(attended.reshape((-1, self.hidden_size)))


class Cosmos3ReasonerVisionMLP(Module):
    """SigLIP2 tanh-GELU feed-forward block."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.fc1 = Linear(ctx, self.key("fc1"), rank=2, tensor_parallel=False)
        self.fc2 = Linear(ctx, self.key("fc2"), rank=2, tensor_parallel=False)

    def forward(self, hidden_states):
        return self.fc2(self.fc1(hidden_states).gelu_tanh())


class Cosmos3ReasonerVisionLayer(Module):
    """One pre-normalized SigLIP2 encoder layer."""

    def __init__(self, ctx, index: int, hidden_size: int, num_heads: int,
                 eps: float) -> None:
        prefix = f"model.visual.encoder.layers.{index}"
        super().__init__(ctx, prefix)
        self.layer_norm1 = LayerNorm(ctx, self.key("layer_norm1"), eps, rank=2)
        self.self_attn = Cosmos3ReasonerVisionAttention(
            ctx, self.key("self_attn"), hidden_size, num_heads)
        self.layer_norm2 = LayerNorm(ctx, self.key("layer_norm2"), eps, rank=2)
        self.mlp = Cosmos3ReasonerVisionMLP(ctx, self.key("mlp"))

    def forward(self, hidden_states, cu_seqlens, max_seqlen):
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states), cu_seqlens, max_seqlen)
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class Cosmos3ReasonerVisionTransformer(Module):
    """Complete SigLIP2 patch encoder."""

    def __init__(self, ctx, config: dict) -> None:
        super().__init__(ctx, "model.visual")
        self.hidden_size = int(config["hidden_size"])
        self.num_heads = int(config["num_attention_heads"])
        self.embeddings = Cosmos3ReasonerVisionEmbeddings(
            ctx, self.hidden_size)
        eps = float(config.get("layer_norm_eps", 1e-6))
        self.layers = [
            Cosmos3ReasonerVisionLayer(ctx, index, self.hidden_size,
                                       self.num_heads, eps)
            for index in range(int(config["num_hidden_layers"]))
        ]
        self.post_layernorm = LayerNorm(ctx,
                                        self.key("post_layernorm"),
                                        eps,
                                        rank=2)

    def forward(self, pixel_values, cu_seqlens, position_indices,
                position_weights, max_seqlen):
        hidden_states = self.embeddings(pixel_values, position_indices,
                                        position_weights)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens, max_seqlen)
        return self.post_layernorm(hidden_states)


class Cosmos3ReasonerPatchMerger(Module):
    """Normalize, merge 2D patch blocks, and project to decoder width."""

    def __init__(self, ctx, hidden_size: int, merge_size: int,
                 eps: float) -> None:
        super().__init__(ctx, "model.projector")
        self.merged_size = hidden_size * merge_size**2
        self.norm = LayerNorm(ctx, self.key("norm"), eps, rank=2)
        self.linear_fc1 = Linear(ctx,
                                 self.key("linear_fc1"),
                                 rank=2,
                                 tensor_parallel=False)
        self.linear_fc2 = Linear(ctx,
                                 self.key("linear_fc2"),
                                 rank=2,
                                 tensor_parallel=False)

    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states).reshape(
            (-1, self.merged_size))
        return self.linear_fc2(self.linear_fc1(hidden_states).gelu())


class Cosmos3ReasonerVisualModel(NetworkModule):
    """Cosmos3 reasoner visual engine with packed-patch I/O."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle)

    def __init__(self, ctx, bundle) -> None:
        super().__init__(ctx)
        root = bundle.root
        self.config = dict(root.get("vision_config") or {})
        projector = dict(root.get("projector_config") or {})
        self.hidden_size = int(self.config["hidden_size"])
        self.num_heads = int(self.config["num_attention_heads"])
        self.head_dim = self.hidden_size // self.num_heads
        default_scale = self.head_dim**-0.5
        configured_scale = float(
            self.config.get("attention_scale", default_scale))
        if not math.isclose(
                configured_scale, default_scale, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "Cosmos3 visual attention scaling is not supported by "
                "VitAttentionPlugin")
        channels = int(self.config.get("num_channels", 3))
        patch_size = int(self.config["patch_size"])
        self.input_size = channels * patch_size**2
        self.merge_size = int(
            projector.get("spatial_merge_size",
                          self.config.get("spatial_merge_size", 2)))
        eps = float(self.config.get("layer_norm_eps", 1e-6))
        self.visual = Cosmos3ReasonerVisionTransformer(ctx, self.config)
        self.projector = Cosmos3ReasonerPatchMerger(ctx, self.hidden_size,
                                                    self.merge_size, eps)

    def input_tensors(self):
        return {
            "pixel_values":
            self.add_input("input", trt.float16, (-1, self.input_size)),
            "cu_seqlens":
            self.add_input("cu_seqlens", trt.int32, (-1, )),
            "position_indices":
            self.add_input("fast_pos_embed_idx", trt.int64, (4, -1)),
            "position_weights":
            self.add_input("fast_pos_embed_weight", trt.float16, (4, -1)),
            "max_seqlen":
            self.add_input("max_seqlen_carrier", trt.int32, (-1, )),
        }

    def forward(self, pixel_values, cu_seqlens, position_indices,
                position_weights, max_seqlen):
        hidden_states = self.visual(pixel_values, cu_seqlens, position_indices,
                                    position_weights, max_seqlen)
        return {"output": self.projector(hidden_states)}
