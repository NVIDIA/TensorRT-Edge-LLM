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
"""Nemotron-Omni audio checkpoint-direct graph."""

import math
import re

import numpy as np
import tensorrt as trt

from ...core import contracts
from ...ops import Linear, Module, NetworkModule
from ...ops import functional as F


class NemotronAudioNorm(Module):
    """Checkpoint-backed Nemotron-Omni audio normalization."""

    def __init__(self, ctx, prefix: str, eps: float = 1e-5) -> None:
        super().__init__(ctx, prefix)
        self.eps = eps

    def forward(self, hidden):
        return F.normalization(hidden, self.prefix, self.eps, 3)


class NemotronAudioConv1d(Module):
    """Checkpoint-backed convolution in a Nemotron audio block."""

    def __init__(self, ctx, prefix: str, depthwise: bool = False) -> None:
        super().__init__(ctx, prefix)
        self.depthwise = depthwise

    def forward(self, hidden):
        kernel = self.weights.fp16_parameter(self.key("weight"))
        groups = int(kernel.shape[0]) if self.depthwise else 1
        padding = (int(kernel.shape[-1]) // 2, ) if self.depthwise else (0, )
        return F.convolution(hidden,
                             kernel,
                             self.weights.opt_fp16_parameter(self.key("bias")),
                             padding=padding,
                             groups=groups)


class NemotronAudioSubsamplingLayer(Module):
    """One model-owned convolutional acoustic subsampling stage."""

    def __init__(self, ctx, prefix: str, index: int) -> None:
        super().__init__(ctx, prefix)
        kernel = self.weights.parameter_spec(self.key("weight"))
        self.groups = int(
            kernel.shape[0]) if kernel.shape[1] == 1 and index else 1
        self.stride = (2, 2) if kernel.shape[-1] == 3 else (1, 1)
        self.padding = (1, 1) if kernel.shape[-1] == 3 else (0, 0)
        self.activation = index in (0, 3, 6)

    def forward(self, hidden):
        hidden = F.convolution(hidden,
                               self.weights.fp16_parameter(self.key("weight")),
                               self.weights.opt_fp16_parameter(
                                   self.key("bias")),
                               stride=self.stride,
                               padding=self.padding,
                               groups=self.groups)
        return hidden.relu() if self.activation else hidden


class NemotronAudioSubsampling(Module):
    """Convolutional acoustic subsampling frontend."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx, "sound_encoder.subsampling")
        layer_prefixes = {}
        for key in self.weights.keys():
            match = re.search(r"(.+subsampling\.layers\.(\d+))\.weight$", key)
            if match:
                layer_prefixes[int(match.group(2))] = match.group(1)
        self.layers = [
            NemotronAudioSubsamplingLayer(ctx, layer_prefixes[index], index)
            for index in sorted(layer_prefixes)
        ]
        linear_key = self.weights.find_suffix("subsampling.linear.weight",
                                              "sound")
        self.linear = Linear(ctx,
                             linear_key[:-len(".weight")],
                             rank=3,
                             tensor_parallel=False)

    def forward(self, features):
        hidden = features.unsqueeze(1, 3)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = hidden.transpose((0, 2, 1, 3)).reshape((0, 0, -1))
        return self.linear(hidden)


class NemotronRelativePositionEmbedding(Module):
    """Runtime-length Conformer relative position embedding."""

    def __init__(self, ctx, max_time_steps: int) -> None:
        super().__init__(ctx, "sound_encoder.relative_position")
        self.max_time_steps = max_time_steps

    def forward(self, hidden):
        hidden_size = int(hidden.shape[-1])
        max_encoded_steps = (self.max_time_steps + 7) // 8
        positions = np.arange(max_encoded_steps - 1,
                              -max_encoded_steps,
                              -1,
                              dtype=np.float32)[:, None]
        frequencies = np.exp(
            np.arange(0, hidden_size, 2, dtype=np.float32) *
            -(math.log(10000.0) / hidden_size))
        table = np.empty((1, positions.shape[0], hidden_size),
                         dtype=np.float16)
        table[0, :, 0::2] = np.sin(positions * frequencies).astype(np.float16)
        table[0, :, 1::2] = np.cos(positions * frequencies).astype(np.float16)

        sequence = F.shape_of(hidden)[1:2]
        one = F.constant(np.array([1], dtype=np.int32), "one")
        maximum = F.constant(np.array([max_encoded_steps], dtype=np.int32),
                             "max_encoded_steps")
        start = maximum - sequence
        length = sequence * np.int32(2) - one
        position = F.constant(table, "relative_position")
        return F.dynamic_slice(position, (0, start, 0),
                               (one, length, hidden_size))


def _relative_shift(scores):
    score_shape = F.shape_of(scores)
    batch = score_shape[0:1]
    heads = score_shape[1:2]
    sequence = score_shape[2:3]
    relative = score_shape[3:4]
    relative_plus_one = relative + np.int32(1)

    zero = scores.slice_last_dim(0, 1, 4) * np.float16(0.0)
    padded = F.concatenate((zero, scores), 3)
    shifted = F.dynamic_reshape(padded,
                                (batch, heads, relative_plus_one, sequence))
    shifted = F.dynamic_slice(shifted, (0, 0, 1, 0),
                              (batch, heads, relative, sequence))
    shifted = F.dynamic_reshape(shifted, (batch, heads, sequence, relative))
    return F.dynamic_slice(shifted, (0, 0, 0, 0),
                           (batch, heads, sequence, sequence))


class NemotronRelativeAttention(Module):
    """Conformer relative self-attention."""

    def __init__(self, ctx, prefix: str, num_heads: int) -> None:
        super().__init__(ctx, prefix)
        self.num_heads = num_heads
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
        self.relative_k_proj = Linear(ctx,
                                      self.key("relative_k_proj"),
                                      rank=3,
                                      tensor_parallel=False)
        self.o_proj = Linear(ctx,
                             self.key("o_proj"),
                             rank=3,
                             tensor_parallel=False)

    def forward(self, hidden, position):
        hidden_size = int(hidden.shape[-1])
        head_size = hidden_size // self.num_heads
        query = self.q_proj(hidden).reshape(
            (0, 0, self.num_heads, head_size)).transpose((0, 2, 1, 3))
        key = self.k_proj(hidden).reshape(
            (0, 0, self.num_heads, head_size)).transpose((0, 2, 1, 3))
        value = self.v_proj(hidden).reshape(
            (0, 0, self.num_heads, head_size)).transpose((0, 2, 1, 3))

        bias_shape = (1, self.num_heads, 1, head_size)
        bias_u = F.constant(
            self.weights.f16(self.key("bias_u")).reshape(bias_shape),
            "relative_bias_u")
        bias_v = F.constant(
            self.weights.f16(self.key("bias_v")).reshape(bias_shape),
            "relative_bias_v")
        relative_key = self.relative_k_proj(position)
        relative_key = relative_key.reshape(
            (1, -1, self.num_heads, head_size)).transpose((0, 2, 1, 3))
        position_score = (query + bias_v).matmul(
            relative_key, rhs_op=trt.MatrixOperation.TRANSPOSE)
        position_score = _relative_shift(position_score)
        scale = head_size**-0.5
        output = F.scaled_dot_product_attention(query + bias_u,
                                                key,
                                                value,
                                                mask=position_score *
                                                np.float16(scale),
                                                scale=scale)
        output = output.transpose((0, 2, 1, 3)).reshape((0, 0, hidden_size))
        return self.o_proj(output)


class NemotronConformerConv(Module):
    """Conformer convolution module with GLU and folded batch norm."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.pointwise_conv1 = NemotronAudioConv1d(ctx,
                                                   self.key("pointwise_conv1"))
        self.depthwise_conv = NemotronAudioConv1d(ctx,
                                                  self.key("depthwise_conv"),
                                                  depthwise=True)
        self.pointwise_conv2 = NemotronAudioConv1d(ctx,
                                                   self.key("pointwise_conv2"))

    def forward(self, hidden):
        width = int(hidden.shape[-1])
        hidden = hidden.transpose((0, 2, 1))
        hidden = self.pointwise_conv1(hidden)
        first = hidden.slice_axis(1, 0, width, 3)
        second = hidden.slice_axis(1, width, width, 3).sigmoid()
        hidden = first * second
        hidden = self.depthwise_conv(hidden)
        norm = self.key("norm")
        running_mean = self.weights.f32(norm + ".running_mean")
        running_var = self.weights.f32(norm + ".running_var")
        gamma = self.weights.f32(norm + ".weight")
        beta = self.weights.f32(norm + ".bias")
        scale = gamma / np.sqrt(running_var + 1e-5)
        bias = beta - running_mean * scale
        hidden = hidden * F.constant(
            scale.astype(np.float16).reshape(1, -1, 1), "batch_scale")
        hidden = hidden + F.constant(
            bias.astype(np.float16).reshape(1, -1, 1), "batch_bias")
        hidden = self.pointwise_conv2(hidden.silu())
        return hidden.transpose((0, 2, 1))


class NemotronFeedForward(Module):
    """Conformer feed-forward branch."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.linear1 = Linear(ctx,
                              self.key("linear1"),
                              rank=3,
                              tensor_parallel=False)
        self.linear2 = Linear(ctx,
                              self.key("linear2"),
                              rank=3,
                              tensor_parallel=False)

    def forward(self, hidden):
        return self.linear2(self.linear1(hidden).silu())


class NemotronConformerLayer(Module):
    """One Nemotron-Omni sound encoder Conformer layer."""

    def __init__(self, ctx, prefix: str, num_heads: int) -> None:
        super().__init__(ctx, prefix)
        self.feed_forward1 = NemotronFeedForward(ctx,
                                                 self.key("feed_forward1"))
        self.self_attn = NemotronRelativeAttention(ctx, self.key("self_attn"),
                                                   num_heads)
        self.conv = NemotronConformerConv(ctx, self.key("conv"))
        self.feed_forward2 = NemotronFeedForward(ctx,
                                                 self.key("feed_forward2"))
        self.norm_feed_forward1 = NemotronAudioNorm(
            ctx, self.key("norm_feed_forward1"))
        self.norm_self_att = NemotronAudioNorm(ctx, self.key("norm_self_att"))
        self.norm_conv = NemotronAudioNorm(ctx, self.key("norm_conv"))
        self.norm_feed_forward2 = NemotronAudioNorm(
            ctx, self.key("norm_feed_forward2"))
        self.norm_out = NemotronAudioNorm(ctx, self.key("norm_out"))

    def _half_residual(self, hidden, branch):
        return hidden + branch * np.float16(0.5)

    def forward(self, hidden, position):
        normalized = self.norm_feed_forward1(hidden)
        hidden = self._half_residual(hidden, self.feed_forward1(normalized))
        normalized = self.norm_self_att(hidden)
        hidden = hidden + self.self_attn(normalized, position)
        normalized = self.norm_conv(hidden)
        hidden = hidden + self.conv(normalized)
        normalized = self.norm_feed_forward2(hidden)
        hidden = self._half_residual(hidden, self.feed_forward2(normalized))
        return self.norm_out(hidden)


class NemotronSoundProjection(Module):
    """Model-owned sound projector after the Conformer stack."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.norm = NemotronAudioNorm(ctx, self.key("norm"))
        self.linear1 = Linear(ctx,
                              self.key("linear1"),
                              rank=3,
                              tensor_parallel=False)
        self.linear2 = Linear(ctx,
                              self.key("linear2"),
                              rank=3,
                              tensor_parallel=False)

    def forward(self, hidden):
        hidden = self.linear1(self.norm(hidden)).relu()
        return self.linear2(hidden * hidden)


class NemotronOmniAudioEncoder(NetworkModule):
    """End-to-end Nemotron-Omni sound encoder graph."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle, ctx.args.max_time_steps)

    def __init__(self, ctx, bundle, max_time_steps: int) -> None:
        super().__init__(ctx, "sound_encoder")
        self.config = bundle.component_dict(contracts.Component.AUDIO)
        self.root = bundle.root
        self.mel_bins = int(self.config.get("num_mel_bins", 128))
        sound = self.root.get("sound_config", self.root)
        num_heads = int(sound.get("num_attention_heads", 8))
        self.subsampling = NemotronAudioSubsampling(ctx)
        self.position = NemotronRelativePositionEmbedding(ctx, max_time_steps)
        layer_prefixes = self.weights.layer_prefixes((
            r"(sound_encoder\.encoder\.layers\.\d+)\.norm_feed_forward1\.weight$",
        ))
        expected_layers = int(sound.get("num_hidden_layers", 24))
        if len(layer_prefixes) != expected_layers:
            raise ValueError(f"expected {expected_layers} conformer layers, "
                             f"found {len(layer_prefixes)}")
        self.layers = [
            NemotronConformerLayer(ctx, prefix, num_heads)
            for prefix in layer_prefixes
        ]
        norm_key = self.weights.find_suffix("sound_projection.norm.weight")
        self.projection = NemotronSoundProjection(
            ctx, norm_key[:-len(".norm.weight")])

    def input_tensors(self):
        return {
            "features":
            self.add_input("input_features", trt.float16,
                           (1, -1, self.mel_bins))
        }

    def forward(self, features):
        hidden = self.subsampling(features)
        position = self.position(hidden)
        for layer in self.layers:
            hidden = layer(hidden, position)
        return {"last_hidden_state": self.projection(hidden)}
