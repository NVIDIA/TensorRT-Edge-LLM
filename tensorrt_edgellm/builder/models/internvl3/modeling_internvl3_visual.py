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
"""InternVL visual checkpoint-direct graph."""

import tensorrt as trt

from ...core import contracts
from ...ops import Linear, Module, NetworkModule
from ...ops import functional as F


class InternVLEmbeddings(Module):
    """Patch, class-token, and position embeddings for InternVL ViTs."""

    def __init__(self, ctx, image_size: int, patch_size: int, hidden_size: int,
                 num_patches: int) -> None:
        super().__init__(ctx, "vision")
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_patches = num_patches
        self.patch_key = self.weights.find_suffix(
            "vision_model.embeddings.patch_embedding.weight")
        embedding_prefix = self.patch_key[:-len("patch_embedding.weight")]
        self.class_key = embedding_prefix + "class_embedding"
        self.position_key = embedding_prefix + "position_embedding"
        self.patch_prefix = self.patch_key[:-len(".weight")]
        self.image_size = image_size

    def forward(self, pixels):
        hidden = F.convolution(pixels,
                               self.weights.f16(self.patch_key),
                               self.weights.opt_f16(self.patch_prefix +
                                                    ".bias"),
                               stride=(self.patch_size, self.patch_size))
        hidden = hidden.reshape(
            (0, self.hidden_size, self.num_patches)).transpose((0, 2, 1))
        class_token = F.batch_token(hidden, self.weights.f16(self.class_key))
        hidden = F.concatenate((class_token, hidden), 1)
        if self.weights.has(self.position_key):
            hidden = hidden + F.constant(self.weights.f16(self.position_key),
                                         "position_embedding")
        return hidden


class InternVLVisualLayerNorm(Module):
    """InternVL visual LayerNorm."""

    def __init__(self, ctx, prefix: str, eps: float) -> None:
        super().__init__(ctx, prefix)
        self.eps = eps

    def forward(self, hidden):
        return F.normalization(hidden, self.prefix, self.eps, 3)


class InternVLVisualAttention(Module):
    """InternVL visual multi-head self-attention."""

    def __init__(self, ctx, prefix: str, hidden_size: int, num_heads: int,
                 eps: float, qk_normalization: bool) -> None:
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
        output = self.key("proj")
        if not self.weights.has(output + ".weight"):
            output = next(candidate
                          for candidate in (self.key("projection_layer"),
                                            self.key("o_proj"),
                                            self.key("out_proj"))
                          if self.weights.has(candidate + ".weight"))
        self.proj = Linear(ctx, output, rank=3, tensor_parallel=False)
        self.q_norm = (InternVLVisualLayerNorm(ctx, self.key("q_norm"), eps)
                       if qk_normalization else None)
        self.k_norm = (InternVLVisualLayerNorm(ctx, self.key("k_norm"), eps)
                       if qk_normalization else None)

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
        if self.q_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
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
        return self.proj(attention)


class InternVLVisualMLP(Module):
    """InternVL visual feed-forward module."""

    def __init__(self, ctx, prefix: str, hidden_act: str) -> None:
        super().__init__(ctx, prefix)
        self.hidden_act = hidden_act
        self.fc1 = Linear(ctx, self.key("fc1"), rank=3, tensor_parallel=False)
        self.fc2 = Linear(ctx, self.key("fc2"), rank=3, tensor_parallel=False)

    def forward(self, hidden):
        hidden = self.fc1(hidden).activation(self.hidden_act)
        return self.fc2(hidden)


class InternVLEncoderLayer(Module):
    """One InternVL visual transformer layer."""

    def __init__(self, ctx, prefix: str, hidden_size: int, num_heads: int,
                 eps: float, hidden_act: str, qk_normalization: bool) -> None:
        super().__init__(ctx, prefix)
        self.num_heads = num_heads
        self.eps = eps
        self.norm1 = self.key("norm1")
        self.norm2 = self.key("norm2")
        self.attention_prefix = self.key("attn")
        self.scale1 = self.key("ls1")
        self.scale2 = self.key("ls2")
        self.layer_norm1 = InternVLVisualLayerNorm(ctx, self.norm1, eps)
        self.layer_norm2 = InternVLVisualLayerNorm(ctx, self.norm2, eps)
        self.attention = InternVLVisualAttention(ctx, self.attention_prefix,
                                                 hidden_size, num_heads, eps,
                                                 qk_normalization)
        self.mlp = InternVLVisualMLP(ctx, self.key("mlp"), hidden_act)

    def _apply_scale(self, hidden, scale_key: str):
        hidden = hidden
        if self.weights.has(scale_key):
            hidden = hidden * F.constant(
                self.weights.f16(scale_key).reshape(1, 1, -1), "layer_scale")
        return hidden

    def forward(self, hidden):
        normalized = self.layer_norm1(hidden)
        attention = self.attention(normalized)
        attention = self._apply_scale(attention, self.scale1)
        hidden = hidden + attention
        normalized = self.layer_norm2(hidden)
        feed_forward = self.mlp(normalized)
        feed_forward = self._apply_scale(feed_forward, self.scale2)
        return hidden + feed_forward


class InternVLProjector(Module):
    """InternVL multimodal projector after pixel unshuffle."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx, "projector")
        norm_prefix = next(
            key.rsplit(".", 1)[0] for key in self.weights.keys()
            if key.endswith("mlp1.0.weight"))
        root = norm_prefix.rsplit(".", 1)[0]
        self.norm = InternVLVisualLayerNorm(ctx, norm_prefix, 1e-5)
        self.fc1 = Linear(ctx, root + ".1", rank=3, tensor_parallel=False)
        self.fc2 = Linear(ctx, root + ".3", rank=3, tensor_parallel=False)

    def forward(self, hidden):
        return self.fc2(self.fc1(self.norm(hidden)).gelu())


class InternVL3VisualEncoder(NetworkModule):
    """HF-style InternVL3 vision tower plus projector."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle)

    def __init__(self, ctx, bundle) -> None:
        super().__init__(ctx, "vision")
        self.visual = bundle.component_dict(contracts.Component.VISUAL)
        image_size = self.visual["image_size"]
        if isinstance(image_size, list):
            image_size = image_size[0]
        patch_size = self.visual["patch_size"]
        if isinstance(patch_size, list):
            patch_size = patch_size[0]
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.hidden_size = int(self.visual["hidden_size"])
        self.num_heads = int(self.visual["num_attention_heads"])
        self.channels = int(self.visual.get("num_channels", 3))
        self.side = self.image_size // self.patch_size
        self.num_patches = self.side * self.side
        self.eps = float(self.visual.get("layer_norm_eps", 1e-6))
        hidden_act = str(self.visual.get("hidden_act", "gelu"))
        qk_normalization = bool(
            self.visual.get("qk_normalization",
                            self.visual.get("use_qk_norm", False)))
        layer_pattern = r"(.*vision_model\.encoder\.layers\.\d+)\.norm1\.weight$"
        self.embeddings = InternVLEmbeddings(ctx, self.image_size,
                                             self.patch_size, self.hidden_size,
                                             self.num_patches)
        self.layers = [
            InternVLEncoderLayer(ctx, prefix, self.hidden_size, self.num_heads,
                                 self.eps, hidden_act, qk_normalization)
            for prefix in self.weights.layer_prefixes((layer_pattern, ))
        ]
        select_layer = int(bundle.root.get("select_layer", -1))
        self.feature_index = (len(self.layers) + 1 + select_layer
                              if select_layer < 0 else select_layer)
        if not 0 <= self.feature_index <= len(self.layers):
            raise ValueError(f"InternVL3 select_layer {select_layer} is "
                             "outside the vision hidden-state range")
        self.downsample_ratio = float(bundle.root.get("downsample_ratio", 0.5))
        self.projector = InternVLProjector(ctx)

    def input_tensors(self):
        return {
            "pixels":
            self.add_input(
                "input", trt.float16,
                (-1, self.channels, self.image_size, self.image_size))
        }

    def forward(self, pixels):
        hidden = self.embeddings(pixels)
        feature = hidden if self.feature_index == 0 else None
        for index, layer in enumerate(self.layers, 1):
            hidden = layer(hidden)
            if index == self.feature_index:
                feature = hidden
        hidden = feature.slice_axis(1, 1, self.num_patches, 3)
        scale = int(round(1.0 / self.downsample_ratio))
        hidden = F.pixel_unshuffle(hidden, self.side, scale, self.hidden_size)
        hidden = self.projector(hidden)
        return {"output": hidden.reshape((-1, int(hidden.shape[-1])))}
