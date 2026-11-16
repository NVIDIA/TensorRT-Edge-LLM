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
"""Shared Qwen audio-encoder modules."""

import numpy as np

from . import functional as F
from .linear import Linear
from .module import Module
from .normalization import LayerNorm


class AudioConv2d(Module):
    """One convolution in the Qwen audio subsampling frontend."""

    def forward(self, hidden_states):
        return F.convolution(hidden_states,
                             self.weights.fp16_parameter(self.key("weight")),
                             self.weights.opt_fp16_parameter(self.key("bias")),
                             stride=(2, 2),
                             padding=(1, 1)).gelu()


class AudioConvSubsampler(Module):
    """Convolutional audio frontend followed by its output projection."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx, "audio")
        self.convolutions = [
            AudioConv2d(
                ctx,
                self.weights.find_suffix(f"conv2d{index}.weight",
                                         "audio")[:-len(".weight")])
            for index in (1, 2, 3)
        ]
        output = self.weights.find_suffix("conv_out.weight",
                                          "audio")[:-len(".weight")]
        self.conv_out = Linear(ctx, output, rank=3, tensor_parallel=False)

    def forward(self, features):
        hidden_states = features.unsqueeze(1, 3)
        for convolution in self.convolutions:
            hidden_states = convolution(hidden_states)
        hidden_states = hidden_states.transpose((0, 3, 1, 2)).reshape(
            (0, 0, -1))
        return self.conv_out(hidden_states)


class AudioPositionEmbedding(Module):
    """Dynamic sinusoidal position embedding used by Qwen audio encoders."""

    def __init__(self, ctx, config: dict) -> None:
        super().__init__(ctx, "audio.position")
        self.max_positions = int(config.get("max_source_positions", 1500))

    def forward(self, hidden_states):
        width = int(hidden_states.shape[-1])
        if width % 2:
            raise ValueError("Qwen audio position width must be even")
        half = width // 2
        increment = np.log(10000.0) / (half - 1)
        frequencies = np.exp(-increment * np.arange(half, dtype=np.float32))
        scaled_time = (
            np.arange(self.max_positions, dtype=np.float32)[:, None] *
            frequencies[None, :])
        table = np.concatenate((np.sin(scaled_time), np.cos(scaled_time)),
                               axis=1)
        positions = F.constant(table.astype(np.float16), "audio_position")
        time_extent = F.shape_of(hidden_states)[1:2]
        size = F.concatenate(
            (time_extent,
             F.constant(np.asarray([width], dtype=np.int32), "hidden_size")),
            0)
        positions = F.dynamic_slice(positions, (0, 0), size)
        positions = positions.unsqueeze(0, 2)
        return hidden_states + positions


class AudioAttention(Module):
    """Multi-head self-attention used by one Qwen audio layer."""

    def __init__(self, ctx, prefix: str, hidden_size: int,
                 num_heads: int) -> None:
        super().__init__(ctx, prefix)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        qkv = self.key("qkv")
        if self.weights.has(qkv + ".weight"):
            self.qkv = Linear(ctx, qkv, rank=2, tensor_parallel=False)
            self.separate_qkv = False
        else:
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
            self.separate_qkv = True
        output = self.key("proj")
        if not self.weights.has(output + ".weight"):
            output = next(candidate
                          for candidate in (self.key("out_proj"),
                                            self.key("o_proj"),
                                            self.key("projection_layer"))
                          if self.weights.has(candidate + ".weight"))
        self.out_proj = Linear(ctx, output, rank=2, tensor_parallel=False)

    def forward(self, hidden_states, attention_mask):
        if self.separate_qkv:
            query = self.q_proj(hidden_states)
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)
        else:
            qkv = self.qkv(hidden_states)
            query = qkv[..., :self.hidden_size]
            key = qkv[..., self.hidden_size:self.hidden_size * 2]
            value = qkv[..., self.hidden_size * 2:]
        query = query.reshape((0, self.num_heads, self.head_dim)).transpose(
            (1, 0, 2))
        key = key.reshape((0, self.num_heads, self.head_dim)).transpose(
            (1, 0, 2))
        value = value.reshape((0, self.num_heads, self.head_dim)).transpose(
            (1, 0, 2))
        query = query.unsqueeze(0, 3)
        key = key.unsqueeze(0, 3)
        value = value.unsqueeze(0, 3)
        mask = (None if attention_mask is None else attention_mask.unsqueeze(
            0, 2).unsqueeze(0, 3))
        output = F.scaled_dot_product_attention(query,
                                                key,
                                                value,
                                                mask=mask,
                                                scale=self.head_dim**-0.5)[0]
        output = output.transpose((1, 0, 2)).reshape((0, self.hidden_size))
        return self.out_proj(output)


class AudioMLP(Module):
    """Two-layer audio feed-forward block."""

    def __init__(self, ctx, prefix: str, activation: str) -> None:
        super().__init__(ctx, prefix)
        self.activation = activation
        fc1 = self.key("fc1")
        fc2 = self.key("fc2")
        if not self.weights.has(fc1 + ".weight"):
            fc1, fc2 = self.key("mlp.fc1"), self.key("mlp.fc2")
        self.fc1 = Linear(ctx, fc1, rank=2, tensor_parallel=False)
        self.fc2 = Linear(ctx, fc2, rank=2, tensor_parallel=False)

    def forward(self, hidden_states):
        return self.fc2(self.fc1(hidden_states).activation(self.activation))


class AudioEncoderLayer(Module):
    """One pre-normalized Qwen audio transformer layer."""

    def __init__(self, ctx, prefix: str, hidden_size: int, num_heads: int,
                 activation: str, clamp_fp16: bool) -> None:
        super().__init__(ctx, prefix)
        norm1 = (self.key("self_attn_layer_norm") if self.weights.has(
            self.key("self_attn_layer_norm.weight")) else self.key("norm1"))
        norm2 = (self.key("final_layer_norm") if self.weights.has(
            self.key("final_layer_norm.weight")) else self.key("norm2"))
        self.norm1 = LayerNorm(ctx, norm1, 1e-5, 2)
        self.norm2 = LayerNorm(ctx, norm2, 1e-5, 2)
        self.self_attn = AudioAttention(ctx, self.key("self_attn"),
                                        hidden_size, num_heads)
        self.mlp = AudioMLP(ctx, self.prefix, activation)
        self.clamp_fp16 = clamp_fp16

    def forward(self, hidden_states, attention_mask):
        hidden_states = hidden_states + self.self_attn(
            self.norm1(hidden_states), attention_mask)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        if self.clamp_fp16:
            limit = np.float16(np.finfo(np.float16).max - 1000)
            hidden_states = hidden_states.maximum(-limit).minimum(limit)
        return hidden_states


class AudioTransformer(Module):
    """Configuration-driven Qwen audio transformer stack."""

    def __init__(self,
                 ctx,
                 config: dict,
                 *,
                 activation: str,
                 clamp_fp16: bool = False) -> None:
        super().__init__(ctx, "audio")
        hidden_size = int(config.get("d_model", config.get("hidden_size",
                                                           512)))
        num_heads = int(
            config.get("encoder_attention_heads",
                       config.get("num_attention_heads", 8)))
        prefixes = self.weights.layer_prefixes((
            r"(.+audio.+\.layers\.\d+)\.self_attn_layer_norm\.weight$",
            r"(.+audio.+\.layers\.\d+)\.norm1\.weight$",
        ))
        self.layers = [
            AudioEncoderLayer(ctx, prefix, hidden_size, num_heads, activation,
                              clamp_fp16) for prefix in prefixes
        ]

    def forward(self, hidden_states, attention_mask):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states
