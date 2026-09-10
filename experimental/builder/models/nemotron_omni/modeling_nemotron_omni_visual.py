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
"""Nemotron-Omni visual checkpoint-direct graph."""

import tensorrt as trt

from ...ops import Linear, Module, NetworkModule
from ...ops import functional as F


class NemotronVisionPatchEmbeddings(Module):
    """Prepend RADIO register tokens to runtime-computed patch embeddings."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx, "vision_model")
        self.register_key = self.weights.find_suffix(
            "patch_generator.cls_token.token", "vision")
        self.num_registers = int(self.weights.f16(self.register_key).shape[0])

    def forward(self, patch_embeddings):
        registers = F.batch_token(patch_embeddings,
                                  self.weights.f16(self.register_key))
        return F.concatenate((registers, patch_embeddings), 1)


class NemotronRadioLayerNorm(Module):
    """RADIO visual LayerNorm."""

    def forward(self, hidden):
        return F.normalization(hidden, self.prefix, 1e-6, 3)


class NemotronProjectorRMSNorm(Module):
    """RMSNorm used by the RADIO-to-LLM projector."""

    def forward(self, hidden):
        return F.rms_norm(hidden, self.weights.f16(self.key("weight")), 1e-5,
                          3)


class NemotronRadioAttention(Module):
    """RADIO multi-head self-attention."""

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
        output = self.key("proj")
        if not self.weights.has(output + ".weight"):
            output = next(candidate
                          for candidate in (self.key("projection_layer"),
                                            self.key("o_proj"),
                                            self.key("out_proj"))
                          if self.weights.has(candidate + ".weight"))
        self.proj = Linear(ctx, output, rank=3, tensor_parallel=False)

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
        return self.proj(attention)


class NemotronRadioMLP(Module):
    """RADIO visual feed-forward module."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        first = self.key("fc1")
        second = self.key("fc2")
        if not self.weights.has(first + ".weight"):
            first, second = self.key("linear_fc1"), self.key("linear_fc2")
        self.fc1 = Linear(ctx, first, rank=3, tensor_parallel=False)
        self.fc2 = Linear(ctx, second, rank=3, tensor_parallel=False)

    def forward(self, hidden):
        return self.fc2(self.fc1(hidden).gelu())


class NemotronRadioBlock(Module):
    """One pre-normalized RADIO visual transformer block."""

    def __init__(self, ctx, prefix: str, hidden_size: int,
                 num_heads: int) -> None:
        super().__init__(ctx, prefix)
        self.norm1 = NemotronRadioLayerNorm(ctx, self.key("norm1"))
        self.norm2 = NemotronRadioLayerNorm(ctx, self.key("norm2"))
        self.attn = NemotronRadioAttention(ctx, self.key("attn"), hidden_size,
                                           num_heads)
        self.mlp = NemotronRadioMLP(ctx, self.key("mlp"))
        self.scale1 = self.key("ls1")
        self.scale2 = self.key("ls2")

    def _scale(self, hidden, key: str):
        hidden = hidden
        if self.weights.has(key):
            hidden = hidden * F.constant(
                self.weights.f16(key).reshape(1, 1, -1), "layer_scale")
        return hidden

    def forward(self, hidden):
        hidden = hidden
        attention = self._scale(self.attn(self.norm1(hidden)), self.scale1)
        hidden = hidden + attention
        feed_forward = self._scale(self.mlp(self.norm2(hidden)), self.scale2)
        return hidden + feed_forward


class NemotronVisionProjector(Module):
    """Dynamic pixel-unshuffle gather plus RADIO-to-LLM projector."""

    def __init__(self, ctx, root: dict, hidden_size: int) -> None:
        super().__init__(ctx, "mlp1")
        self.hidden_size = hidden_size
        self.scale = int(round(1.0 / float(root["downsample_ratio"])))
        norm_key = self.weights.find_suffix("mlp1.0.weight")
        root_prefix = norm_key[:-len("0.weight")]
        self.norm = NemotronProjectorRMSNorm(ctx, norm_key[:-len(".weight")])
        self.linear1 = Linear(ctx,
                              root_prefix + "1",
                              rank=3,
                              tensor_parallel=False)
        self.linear2 = Linear(ctx,
                              root_prefix + "3",
                              rank=3,
                              tensor_parallel=False)

    def forward(self, hidden, shuffle_indices):
        hidden = hidden.gather(shuffle_indices.reshape((-1, )), axis=1)
        hidden_shape = F.shape_of(hidden)
        shuffle_shape = F.shape_of(shuffle_indices)
        hidden = F.dynamic_reshape(
            hidden, (hidden_shape[0:1], shuffle_shape[0:1],
                     self.hidden_size * self.scale * self.scale))
        hidden = self.linear1(self.norm(hidden)).relu()
        hidden = hidden * hidden
        return self.linear2(hidden)


class NemotronVisualEncoder(NetworkModule):
    """HF-style Nemotron-Omni RADIO vision encoder."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle)

    def __init__(self, ctx, bundle) -> None:
        super().__init__(ctx, "vision_model")
        self.root = bundle.root
        self.visual = self.root.get("vision_config") or {}
        self.image_size = int(self.root["force_image_size"])
        self.patch_size = int(self.root["patch_size"])
        self.hidden_size = int(self.root["vit_hidden_size"])
        self.scale = int(round(1.0 / float(self.root["downsample_ratio"])))
        if self.scale <= 0:
            raise ValueError("downsample_ratio must define a positive scale")
        self.num_heads = int(
            self.visual.get("num_attention_heads", self.hidden_size // 80))
        self.embeddings = NemotronVisionPatchEmbeddings(ctx)
        self.blocks = [
            NemotronRadioBlock(ctx, prefix, self.hidden_size, self.num_heads)
            for prefix in self.weights.layer_prefixes((
                r"(vision_model\.radio_model\.model\.blocks\.\d+)\.norm1\.weight$",
            ))
        ]
        expected_layers = int(self.visual.get("num_hidden_layers", 32))
        if len(self.blocks) != expected_layers:
            raise ValueError(
                f"expected {expected_layers} RADIO blocks, found {len(self.blocks)}"
            )
        self.projector = NemotronVisionProjector(ctx, self.root,
                                                 self.hidden_size)

    def input_tensors(self):
        return {
            "patch_embeddings":
            self.add_input("input", trt.float16, (-1, -1, self.hidden_size)),
            "shuffle_indices":
            self.add_input("shuffle_indices", trt.int64,
                           (-1, self.scale * self.scale)),
        }

    def forward(self, patch_embeddings, shuffle_indices):
        hidden = self.embeddings(patch_embeddings)
        for block in self.blocks:
            hidden = block(hidden)
        shape = F.shape_of(hidden)
        hidden = F.dynamic_slice(
            hidden, (0, self.embeddings.num_registers, 0),
            (shape[0:1], shape[1:2] - self.embeddings.num_registers,
             self.hidden_size))
        hidden = self.projector(hidden, shuffle_indices)
        return {"output": hidden}
