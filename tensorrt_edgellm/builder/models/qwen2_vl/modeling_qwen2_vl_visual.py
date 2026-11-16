# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Qwen2-VL vision model aligned with Transformers' provider implementation."""

import tensorrt as trt

from ...core import contracts
from ...ops import Linear, Module, NetworkModule
from ...ops import functional as F


class Qwen2VLPatchEmbed(Module):
    """Provider Conv3d patch projection, flattened to packed image tokens."""

    def __init__(self, ctx, embed_dim: int) -> None:
        super().__init__(ctx, "visual.patch_embed")
        self.embed_dim = embed_dim
        self.weight_key = self.weights.find_suffix(
            "visual.patch_embed.proj.weight")
        self.prefix = self.weight_key[:-len(".weight")]
        self.visual_prefix = self.prefix.rsplit(".patch_embed", 1)[0]

    def forward(self, hidden_states):
        weight = self.weights.f16(self.weight_key).reshape(self.embed_dim, -1)
        return F.linear_with_weights(hidden_states, weight, None, rank=2)


class Qwen2VLLayerNorm(Module):
    """Biased LayerNorm used by Qwen2-VL's vision transformer."""

    def forward(self, hidden_states):
        return F.normalization(hidden_states, self.prefix, 1e-6, 2)


class Qwen2VLVisionMlp(Module):
    """Provider ``fc1 -> quick_gelu -> fc2`` vision MLP."""

    def __init__(self, ctx, prefix: str, hidden_act: str) -> None:
        super().__init__(ctx, prefix)
        self.fc1 = Linear(ctx, self.key("fc1"), rank=2, tensor_parallel=False)
        self.fc2 = Linear(ctx, self.key("fc2"), rank=2, tensor_parallel=False)
        self.hidden_act = hidden_act

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states).activation(self.hidden_act)
        return self.fc2(hidden_states)


class Qwen2VLVisionAttention(Module):
    """Packed Qwen2-VL visual self-attention."""

    def __init__(self, ctx, prefix: str, embed_dim: int,
                 num_heads: int) -> None:
        super().__init__(ctx, prefix)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = Linear(ctx, self.key("qkv"), rank=2, tensor_parallel=False)
        self.proj = Linear(ctx,
                           self.key("proj"),
                           rank=2,
                           tensor_parallel=False)

    def forward(self, hidden_states, rotary_pos_emb, cu_seqlens,
                max_seqlen_carrier):
        qkv = self.qkv(hidden_states)
        query = qkv.slice_last_dim(0, self.embed_dim, 2).reshape(
            (0, self.num_heads, self.head_dim))
        key = qkv.slice_last_dim(self.embed_dim, self.embed_dim, 2).reshape(
            (0, self.num_heads, self.head_dim))
        value = qkv.slice_last_dim(self.embed_dim * 2, self.embed_dim,
                                   2).reshape(
                                       (0, self.num_heads, self.head_dim))
        query, key = F.apply_rope(query, key, rotary_pos_emb, self.num_heads,
                                  self.head_dim)
        attention = F.vit_attention(query, key, value, cu_seqlens,
                                    max_seqlen_carrier, self.num_heads,
                                    self.head_dim)
        attention = attention.reshape((0, self.embed_dim))
        return self.proj(attention)


class Qwen2VLVisionBlock(Module):
    """One provider Qwen2-VL pre-normalized vision block."""

    def __init__(self, ctx, prefix: str, embed_dim: int, num_heads: int,
                 hidden_act: str) -> None:
        super().__init__(ctx, prefix)
        self.norm1 = Qwen2VLLayerNorm(ctx, self.key("norm1"))
        self.norm2 = Qwen2VLLayerNorm(ctx, self.key("norm2"))
        self.attn = Qwen2VLVisionAttention(ctx, self.key("attn"), embed_dim,
                                           num_heads)
        self.mlp = Qwen2VLVisionMlp(ctx, self.key("mlp"), hidden_act)

    def forward(self, hidden_states, rotary_pos_emb, cu_seqlens,
                max_seqlen_carrier):
        hidden_states = hidden_states
        attention = self.attn(self.norm1(hidden_states), rotary_pos_emb,
                              cu_seqlens, max_seqlen_carrier)
        hidden_states = hidden_states + attention
        mlp = self.mlp(self.norm2(hidden_states))
        return hidden_states + mlp


class Qwen2VLPatchMerger(Module):
    """Provider LayerNorm, spatial merge, and exact-GELU projection."""

    def __init__(self, ctx, prefix: str, embed_dim: int,
                 merge_unit: int) -> None:
        super().__init__(ctx, prefix)
        self.embed_dim = embed_dim
        self.merge_unit = merge_unit
        self.ln_q = Qwen2VLLayerNorm(ctx, self.key("ln_q"))
        self.fc1 = Linear(ctx,
                          self.key("mlp.0"),
                          rank=2,
                          tensor_parallel=False)
        self.fc2 = Linear(ctx,
                          self.key("mlp.2"),
                          rank=2,
                          tensor_parallel=False)

    def forward(self, hidden_states):
        hidden_states = self.ln_q(hidden_states).reshape(
            (-1, self.embed_dim * self.merge_unit))
        hidden_states = self.fc1(hidden_states).gelu()
        return self.fc2(hidden_states)


class Qwen2VLVisualEncoder(NetworkModule):
    """One TensorRT network implementing the complete Qwen2-VL ViT."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle)

    def __init__(self, ctx, bundle) -> None:
        super().__init__(ctx, "visual")
        config = bundle.component_dict(contracts.Component.VISUAL)
        self.embed_dim = int(config["embed_dim"])
        self.num_heads = int(config["num_heads"])
        self.patch_size = int(config.get("patch_size", 14))
        self.temporal_patch_size = int(config.get("temporal_patch_size", 2))
        self.in_channels = int(config.get("in_channels", 3))
        self.merge_unit = int(config.get("spatial_merge_size", 2))**2
        hidden_act = str(config.get("hidden_act", "quick_gelu"))
        self.patch_embed = Qwen2VLPatchEmbed(ctx, self.embed_dim)
        block_prefixes = self.weights.layer_prefixes(
            (r"(.+\.blocks\.\d+)\.norm1\.weight$", ))
        self.blocks = [
            Qwen2VLVisionBlock(ctx, prefix, self.embed_dim, self.num_heads,
                               hidden_act) for prefix in block_prefixes
        ]
        self.merger = Qwen2VLPatchMerger(
            ctx, self.patch_embed.visual_prefix + ".merger", self.embed_dim,
            self.merge_unit)

    def input_tensors(self):
        input_width = (self.in_channels * self.temporal_patch_size *
                       self.patch_size * self.patch_size)
        return {
            "hidden_states":
            self.add_input("input", trt.float16, (-1, input_width)),
            "rotary_pos_emb":
            self.add_input("rotary_pos_emb", trt.float32,
                           (-1, self.embed_dim // self.num_heads // 2)),
            "cu_seqlens":
            self.add_input("cu_seqlens", trt.int32, (-1, )),
            "max_seqlen_carrier":
            self.add_input("max_seqlen_carrier", trt.int32, (-1, )),
        }

    def forward(self, **io):
        outputs = {}
        hidden_states = self.patch_embed(io["hidden_states"])
        for block in self.blocks:
            hidden_states = block(hidden_states, io["rotary_pos_emb"],
                                  io["cu_seqlens"], io["max_seqlen_carrier"])
        outputs["output"] = self.merger(hidden_states)
        return outputs
