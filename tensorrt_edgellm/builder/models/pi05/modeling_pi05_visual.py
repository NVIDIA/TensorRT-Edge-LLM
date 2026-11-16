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
"""Checkpoint-direct pi0.5 SigLIP vision tower."""

from __future__ import annotations

import tensorrt as trt

from ...ops import LayerNorm, Linear, Module, NetworkModule
from ...ops import functional as F


class Pi05VisionEmbeddings(Module):
    """Patch convolution plus the learned 16x16 position table."""

    def __init__(self, ctx, hidden_size: int, patch_size: int) -> None:
        super().__init__(ctx, "vision_tower.vision_model.embeddings")
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.patch_weight = self.weights.fp16_parameter(
            self.key("patch_embedding.weight"))
        self.patch_bias = self.weights.fp16_parameter(
            self.key("patch_embedding.bias"))
        self.position_key = self.key("position_embedding.weight")

    def forward(self, pixel_values):
        patches = F.convolution(pixel_values,
                                self.patch_weight,
                                self.patch_bias,
                                stride=(self.patch_size, self.patch_size))
        patches = patches.transpose((0, 2, 3, 1)).reshape(
            (0, -1, self.hidden_size))
        position = F.constant(
            self.weights.f16(self.position_key).reshape(
                (1, -1, self.hidden_size)), "pi05_visual_position_embedding")
        return patches + position


class Pi05VisionAttention(Module):
    """Bidirectional per-view self-attention without RoPE."""

    def __init__(self, ctx, prefix: str, hidden_size: int,
                 num_heads: int) -> None:
        super().__init__(ctx, prefix)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
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
        self.out_proj = Linear(ctx,
                               self.key("out_proj"),
                               rank=3,
                               tensor_parallel=False)

    def forward(self, hidden_states):
        query = self.q_proj(hidden_states).reshape(
            (0, 0, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        key = self.k_proj(hidden_states).reshape(
            (0, 0, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        value = self.v_proj(hidden_states).reshape(
            (0, 0, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        attended = F.scaled_dot_product_attention(query,
                                                  key,
                                                  value,
                                                  scale=self.head_dim**-0.5)
        attended = attended.transpose((0, 2, 1, 3)).reshape(
            (0, 0, self.hidden_size))
        return self.out_proj(attended)


class Pi05VisionMLP(Module):
    """SigLIP tanh-GELU feed-forward block."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.fc1 = Linear(ctx, self.key("fc1"), rank=3, tensor_parallel=False)
        self.fc2 = Linear(ctx, self.key("fc2"), rank=3, tensor_parallel=False)

    def forward(self, hidden_states):
        return self.fc2(self.fc1(hidden_states).gelu_tanh())


class Pi05VisionLayer(Module):
    """One pre-normalized SigLIP encoder layer."""

    def __init__(self, ctx, index: int, hidden_size: int, num_heads: int,
                 eps: float) -> None:
        prefix = f"vision_tower.vision_model.encoder.layers.{index}"
        super().__init__(ctx, prefix)
        self.layer_norm1 = LayerNorm(ctx, self.key("layer_norm1"), eps, rank=3)
        self.self_attn = Pi05VisionAttention(ctx, self.key("self_attn"),
                                             hidden_size, num_heads)
        self.layer_norm2 = LayerNorm(ctx, self.key("layer_norm2"), eps, rank=3)
        self.mlp = Pi05VisionMLP(ctx, self.key("mlp"))

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states))
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class Pi05VisualModel(NetworkModule):
    """Map each 224x224 view to 256 prefix embeddings."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx)

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.hidden_size = 1152
        self.num_heads = 16
        self.num_layers = 27
        self.patch_size = 14
        self.image_size = 224
        self.projection_dim = 2048
        self.embeddings = Pi05VisionEmbeddings(ctx, self.hidden_size,
                                               self.patch_size)
        self.layers = [
            Pi05VisionLayer(ctx, index, self.hidden_size, self.num_heads, 1e-6)
            for index in range(self.num_layers)
        ]
        self.post_layernorm = LayerNorm(
            ctx, "vision_tower.vision_model.post_layernorm", 1e-6, rank=3)
        self.projector = Linear(ctx,
                                "multi_modal_projector.linear",
                                rank=3,
                                tensor_parallel=False)

    def input_tensors(self):
        return {
            "pixel_values":
            self.add_input("pixel_values", trt.float16,
                           (-1, 3, self.image_size, self.image_size))
        }

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        return {"image_features": self.projector(hidden_states)}
