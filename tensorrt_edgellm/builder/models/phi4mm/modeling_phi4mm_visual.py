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
"""Phi-4 Multimodal visual checkpoint-direct graph."""

import tensorrt as trt

from ...core import contracts
from ...ops import Linear, Module, NetworkModule
from ...ops import functional as F


class Phi4MultimodalVisionEmbeddings(Module):
    """Patch and position embeddings for the Phi-4 image tower."""

    def __init__(self, ctx, image_size: int, patch_size: int,
                 hidden_size: int) -> None:
        super().__init__(ctx, "img_processor.embeddings")
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.side = image_size // patch_size
        self.num_patches = self.side * self.side
        self.patch_key = self.weights.find_suffix(
            "img_processor.embeddings.patch_embedding.weight")
        self.patch_prefix = self.patch_key[:-len(".weight")]
        self.position_key = self.weights.find_suffix(
            "img_processor.embeddings.position_embedding.weight")

    def forward(self, pixels):
        hidden = F.convolution(pixels,
                               self.weights.f16(self.patch_key),
                               self.weights.opt_f16(self.patch_prefix +
                                                    ".bias"),
                               stride=(self.patch_size, self.patch_size))
        hidden = hidden.reshape(
            (0, self.hidden_size, self.num_patches)).transpose((0, 2, 1))
        position = F.constant(
            self.weights.f16(self.position_key).reshape(
                1, self.num_patches, -1), "position_embedding")
        return hidden + position


class Phi4VisualLayerNorm(Module):
    """Phi-4 visual LayerNorm."""

    def __init__(self, ctx, prefix: str, eps: float) -> None:
        super().__init__(ctx, prefix)
        self.eps = eps

    def forward(self, hidden):
        return F.normalization(hidden, self.prefix, self.eps, 3)


class Phi4MultimodalVisionAttention(Module):
    """Phi-4 visual multi-head self-attention."""

    def __init__(self, ctx, prefix: str, hidden_size: int,
                 num_heads: int) -> None:
        super().__init__(ctx, prefix)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        qkv = self.key("qkv")
        if self.weights.has(qkv + ".weight"):
            self.qkv = Linear(ctx, qkv, rank=3, tensor_parallel=False)
            self.separate_qkv = False
        else:
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
            self.separate_qkv = True
        output = next(candidate
                      for candidate in (self.key("out_proj"),
                                        self.key("o_proj"), self.key("proj"),
                                        self.key("projection_layer"))
                      if self.weights.has(candidate + ".weight"))
        self.out_proj = Linear(ctx, output, rank=3, tensor_parallel=False)

    def forward(self, hidden):
        if self.separate_qkv:
            query = self.q_proj(hidden)
            key = self.k_proj(hidden)
            value = self.v_proj(hidden)
        else:
            qkv = self.qkv(hidden)
            query = qkv.slice_last_dim(0, self.hidden_size, 3)
            key = qkv.slice_last_dim(self.hidden_size, self.hidden_size, 3)
            value = qkv.slice_last_dim(self.hidden_size * 2, self.hidden_size,
                                       3)
        query = query.reshape((0, 0, self.num_heads, self.head_dim)).transpose(
            (0, 2, 1, 3))
        key = key.reshape((0, 0, self.num_heads, self.head_dim)).transpose(
            (0, 2, 1, 3))
        value = value.reshape((0, 0, self.num_heads, self.head_dim)).transpose(
            (0, 2, 1, 3))
        attention = F.scaled_dot_product_attention(query,
                                                   key,
                                                   value,
                                                   scale=self.head_dim**-0.5)
        attention = attention.transpose((0, 2, 1, 3)).reshape(
            (0, 0, self.hidden_size))
        return self.out_proj(attention)


class Phi4MultimodalVisionMLP(Module):
    """Phi-4 visual feed-forward module."""

    def __init__(self, ctx, prefix: str, hidden_act: str) -> None:
        super().__init__(ctx, prefix)
        self.hidden_act = hidden_act
        self.fc1 = Linear(ctx, self.key("fc1"), rank=3, tensor_parallel=False)
        self.fc2 = Linear(ctx, self.key("fc2"), rank=3, tensor_parallel=False)

    def forward(self, hidden):
        hidden = self.fc1(hidden).activation(self.hidden_act)
        return self.fc2(hidden)


class Phi4MultimodalVisionEncoderLayer(Module):
    """One pre-norm Phi-4 vision transformer layer."""

    def __init__(self, ctx, prefix: str, hidden_size: int, num_heads: int,
                 eps: float, hidden_act: str) -> None:
        super().__init__(ctx, prefix)
        self.num_heads = num_heads
        self.eps = eps
        self.layer_norm1 = Phi4VisualLayerNorm(ctx, self.key("layer_norm1"),
                                               eps)
        self.layer_norm2 = Phi4VisualLayerNorm(ctx, self.key("layer_norm2"),
                                               eps)
        self.self_attn = Phi4MultimodalVisionAttention(ctx,
                                                       self.key("self_attn"),
                                                       hidden_size, num_heads)
        self.mlp = Phi4MultimodalVisionMLP(ctx, self.key("mlp"), hidden_act)

    def forward(self, hidden):
        hidden = hidden
        normalized = self.layer_norm1(hidden)
        attention = self.self_attn(normalized)
        hidden = hidden + attention
        normalized = self.layer_norm2(hidden)
        feed_forward = self.mlp(normalized)
        return hidden + feed_forward


class Phi4MultimodalImageProjection(Module):
    """Average-pool Phi-4 patch pairs and project to language hidden size."""

    def __init__(self, ctx, side: int, hidden_size: int) -> None:
        super().__init__(ctx, "img_projection")
        self.side = side
        self.hidden_size = hidden_size
        projection_key = self.weights.find_suffix("img_projection.0.weight")
        projection_root = projection_key[:-len("0.weight")]
        self.linear1 = Linear(ctx,
                              projection_root + "0",
                              rank=3,
                              tensor_parallel=False)
        self.linear2 = Linear(ctx,
                              projection_root + "2",
                              rank=3,
                              tensor_parallel=False)

    def forward(self, feature):
        feature = feature.reshape(
            (0, self.side // 2, 2, self.side // 2, 2, self.hidden_size))
        feature = feature.reduce(trt.ReduceOperation.AVG, (1 << 2) | (1 << 4),
                                 False)
        feature = feature.reshape((0, (self.side // 2)**2, self.hidden_size))
        return self.linear2(self.linear1(feature).gelu())


class Phi4MultimodalVisionModel(NetworkModule):
    """End-to-end Phi-4 visual model graph."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle)

    def __init__(self, ctx, bundle) -> None:
        super().__init__(ctx, "img_processor")
        visual = bundle.component_dict(contracts.Component.VISUAL)
        self.image_size = int(visual.get("image_size", 448))
        self.patch_size = int(visual.get("patch_size", 14))
        self.hidden_size = int(visual.get("hidden_size", 1152))
        self.num_heads = int(visual.get("num_attention_heads", 16))
        feature_layer = int(
            visual.get("feature_layer", visual.get("embd_layer", -2)))
        self.channels = int(visual.get("num_channels", 3))
        self.embeddings = Phi4MultimodalVisionEmbeddings(
            ctx, self.image_size, self.patch_size, self.hidden_size)
        layers = self.weights.layer_prefixes(
            (r"(.+img_processor\.encoder\.layers\.\d+)\.layer_norm1\.weight$",
             ))
        self.feature_index = (len(layers) + 1 + feature_layer
                              if feature_layer < 0 else feature_layer)
        if not 0 <= self.feature_index <= len(layers):
            raise ValueError(f"Phi-4 feature_layer {feature_layer} is "
                             "outside the vision hidden-state range")
        eps = float(visual.get("layer_norm_eps", 1e-6))
        hidden_act = str(visual.get("hidden_act", "gelu_pytorch_tanh"))
        self.layers = [
            Phi4MultimodalVisionEncoderLayer(ctx, prefix, self.hidden_size,
                                             self.num_heads, eps, hidden_act)
            for prefix in layers[:self.feature_index]
        ]
        self.projector = Phi4MultimodalImageProjection(ctx,
                                                       self.embeddings.side,
                                                       self.hidden_size)

    def input_tensors(self):
        return {
            "pixels":
            self.add_input(
                "input", trt.float16,
                (-1, self.channels, self.image_size, self.image_size))
        }

    def forward(self, pixels):
        hidden = self.embeddings(pixels)
        for layer in self.layers:
            hidden = layer(hidden)
        output = self.projector(hidden)
        return {"output": output.reshape((-1, int(output.shape[-1])))}
