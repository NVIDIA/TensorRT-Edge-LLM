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
"""Shared modules for equivalent packed vision-transformer components."""

from . import functional as F
from .linear import Linear
from .module import Module
from .normalization import LayerNorm


class PatchEmbedding(Module):
    """Project flattened temporal image patches into the visual width."""

    def __init__(self, ctx, hidden_size: int, *, bias: bool = True) -> None:
        super().__init__(ctx, "visual.patch_embed")
        self.hidden_size = hidden_size
        self.use_bias = bias
        self.patch_key = next(
            key for key in self.weights.keys()
            if "visual" in key and key.endswith("patch_embed.proj.weight"))
        self.patch_prefix = self.patch_key[:-len(".weight")]
        self.visual_prefix = self.patch_prefix.rsplit(".patch_embed", 1)[0]

    def forward(self, pixels):
        weight = self.weights.f16(self.patch_key).reshape(self.hidden_size, -1)
        bias = (self.weights.opt_f16(self.patch_prefix +
                                     ".bias") if self.use_bias else None)
        return F.linear_with_weights(pixels, weight, bias, rank=2)


class FastPositionEmbedding(Module):
    """Apply the bilinearly blended Qwen fast position embedding."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx, "visual")
        self.position_key = self.weights.find_suffix(".pos_embed.weight",
                                                     "visual")

    def forward(self, hidden_states, indices, blend_weights):
        table = F.constant(self.weights.f16(self.position_key),
                           "visual_pos_embed")
        position = table.gather(indices, 0)
        position = position * blend_weights.reshape((4, -1, 1))
        return hidden_states + position.sum(dim=0)


class VisionMLP(Module):
    """Two-layer visual feed-forward block with checkpoint activation."""

    def __init__(self, ctx, prefix: str, hidden_act: str) -> None:
        super().__init__(ctx, prefix)
        self.hidden_act = hidden_act
        self.linear_fc1 = Linear(ctx,
                                 self.key("linear_fc1"),
                                 rank=2,
                                 tensor_parallel=False)
        self.linear_fc2 = Linear(ctx,
                                 self.key("linear_fc2"),
                                 rank=2,
                                 tensor_parallel=False)

    def forward(self, hidden_states):
        return self.linear_fc2(
            self.linear_fc1(hidden_states).activation(self.hidden_act))


class PackedVisionAttention(Module):
    """QKV-packed variable-length visual self-attention."""

    def __init__(self, ctx, prefix: str, hidden_size: int,
                 num_heads: int) -> None:
        super().__init__(ctx, prefix)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.qkv = Linear(ctx, self.key("qkv"), rank=2, tensor_parallel=False)
        self.proj = Linear(ctx,
                           self.key("proj"),
                           rank=2,
                           tensor_parallel=False)

    def forward(self, hidden_states, rotary, cu_seqlens, max_seqlen):
        qkv = self.qkv(hidden_states)
        query = qkv[..., :self.hidden_size].reshape(
            (0, self.num_heads, self.head_size))
        key = qkv[..., self.hidden_size:self.hidden_size * 2].reshape(
            (0, self.num_heads, self.head_size))
        value = qkv[..., self.hidden_size * 2:].reshape(
            (0, self.num_heads, self.head_size))
        query, key = F.apply_rope(query, key, rotary, self.num_heads,
                                  self.head_size)
        output = F.vit_attention(query, key, value, cu_seqlens, max_seqlen,
                                 self.num_heads, self.head_size)
        return self.proj(output.reshape((0, self.hidden_size)))


class VisionTransformerBlock(Module):
    """Pre-normalized packed visual self-attention and feed-forward block."""

    def __init__(self, ctx, prefix: str, hidden_size: int, num_heads: int,
                 hidden_act: str) -> None:
        super().__init__(ctx, prefix)
        self.norm1 = LayerNorm(ctx, self.key("norm1"), 1e-6, 2)
        self.norm2 = LayerNorm(ctx, self.key("norm2"), 1e-6, 2)
        self.attn = PackedVisionAttention(ctx, self.key("attn"), hidden_size,
                                          num_heads)
        self.mlp = VisionMLP(ctx, self.key("mlp"), hidden_act)

    def forward(self, hidden_states, rotary, cu_seqlens, max_seqlen):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), rotary, cu_seqlens, max_seqlen)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class VisionPatchMerger(Module):
    """Normalize, spatially merge, and project packed visual features."""

    def __init__(self,
                 ctx,
                 prefix: str,
                 hidden_size: int,
                 merge_unit: int,
                 postshuffle_norm: bool,
                 *,
                 norm_name: str = "norm",
                 fc1_name: str = "linear_fc1",
                 fc2_name: str = "linear_fc2") -> None:
        super().__init__(ctx, prefix)
        self.hidden_size = hidden_size
        self.merge_unit = merge_unit
        self.postshuffle_norm = postshuffle_norm
        self.norm = LayerNorm(ctx, self.key(norm_name), 1e-6, 2)
        self.fc1 = Linear(ctx,
                          self.key(fc1_name),
                          rank=2,
                          tensor_parallel=False)
        self.fc2 = Linear(ctx,
                          self.key(fc2_name),
                          rank=2,
                          tensor_parallel=False)

    def forward(self, hidden_states):
        if self.postshuffle_norm:
            hidden_states = hidden_states.reshape(
                (-1, self.hidden_size * self.merge_unit))
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states = self.norm(hidden_states)
            hidden_states = hidden_states.reshape(
                (-1, self.hidden_size * self.merge_unit))
        return self.fc2(self.fc1(hidden_states).gelu())
