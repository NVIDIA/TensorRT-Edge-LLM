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
"""Nemotron-3.5-ASR checkpoint-direct FastConformer encoder."""

import math

import numpy as np
import tensorrt as trt

from ...ops import Linear, Module, NetworkModule
from ...ops import functional as F


def _subsampling_layer_count(config: dict) -> int:
    factor = int(config.get("subsampling_factor", 8))
    stride = int(config.get("subsampling_conv_stride", 2))
    layer_count = int(math.log2(factor)) if factor > 0 else 0
    if stride != 2 or layer_count < 1 or 2**layer_count != factor:
        raise ValueError(
            "Nemotron-3.5-ASR requires power-of-two stride-two subsampling")
    return layer_count


class NemotronLayerNorm(Module):
    """Checkpoint-backed LayerNorm used throughout the FastConformer."""

    def __init__(self, ctx, prefix: str, eps: float = 1e-5) -> None:
        super().__init__(ctx, prefix)
        self.eps = eps

    def forward(self, hidden):
        return F.layer_norm(hidden, self.weights.f16(self.key("weight")),
                            self.weights.f16(self.key("bias")), self.eps,
                            hidden.rank)


class CausalSubsamplingStage(Module):
    """Provider depthwise-separable causal stride-two convolution."""

    def __init__(self, ctx, prefix: str, channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__(ctx, prefix)
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, hidden):
        depthwise = F.convolution(
            hidden,
            self.weights.fp16_parameter(self.key("depthwise_conv.weight")),
            self.weights.opt_fp16_parameter(self.key("depthwise_conv.bias")),
            stride=(self.stride, self.stride),
            groups=self.channels,
            pre_padding=(self.kernel_size - 1, self.kernel_size - 1),
            post_padding=(self.stride - 1, self.stride - 1),
        )
        return F.convolution(
            depthwise,
            self.weights.fp16_parameter(self.key("pointwise_conv.weight")),
            self.weights.opt_fp16_parameter(self.key("pointwise_conv.bias")),
        )


class CausalSubsampling(Module):
    """Causal factor-eight convolutional subsampling frontend."""

    def __init__(self, ctx, config: dict) -> None:
        super().__init__(ctx, "encoder.subsampling")
        self.channels = int(config.get("subsampling_conv_channels", 256))
        self.kernel_size = int(config.get("subsampling_conv_kernel_size", 3))
        self.stride = int(config.get("subsampling_conv_stride", 2))
        layer_count = _subsampling_layer_count(config)
        self.layers = [
            CausalSubsamplingStage(ctx, self.key(f"layers.{index}"),
                                   self.channels, self.kernel_size,
                                   self.stride)
            for index in range(layer_count - 1)
        ]
        self.projection = Linear(ctx,
                                 self.key("linear"),
                                 rank=3,
                                 tensor_parallel=False)

    def forward(self, features):
        hidden = features.unsqueeze(1, features.rank)
        hidden = F.convolution(
            hidden,
            self.weights.fp16_parameter(self.key("conv_in.weight")),
            self.weights.opt_fp16_parameter(self.key("conv_in.bias")),
            stride=(self.stride, self.stride),
            pre_padding=(self.kernel_size - 1, self.kernel_size - 1),
            post_padding=(self.stride - 1, self.stride - 1),
        ).relu()
        for layer in self.layers:
            hidden = layer(hidden).relu()
        hidden = hidden.transpose((0, 2, 1, 3)).reshape((0, 0, -1))
        return self.projection(hidden)


class RelativePositionEmbedding(Module):
    """Dynamic slice of the provider's descending sinusoidal table."""

    def __init__(self, ctx, hidden_size: int, max_length: int) -> None:
        super().__init__(ctx, "encoder.encode_positions")
        self.hidden_size = hidden_size
        self.max_length = max_length
        position = np.arange(max_length - 1, -max_length, -1,
                             dtype=np.float32)[:, None]
        inverse = 1.0 / (10000.0**(
            np.arange(0, hidden_size, 2, dtype=np.float32) / hidden_size))
        frequency = position * inverse[None, :]
        table = np.zeros((1, 2 * max_length - 1, hidden_size),
                         dtype=np.float32)
        table[:, :, 0::2] = np.sin(frequency)
        table[:, :, 1::2] = np.cos(frequency)
        self.table = table.astype(np.float16)

    def forward(self, hidden):
        length = F.shape_of(hidden)[1:2]
        start = (0, self.max_length - length, 0)
        size = (1, length * 2 - 1, self.hidden_size)
        return F.dynamic_slice(F.constant(self.table, "relative_position"),
                               start, size)


class RelativePositionAttention(Module):
    """Transformer-XL relative attention with chunk-limited masking."""

    def __init__(self, ctx, prefix: str, hidden_size: int,
                 num_heads: int) -> None:
        super().__init__(ctx, prefix)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = np.float16(self.head_dim**-0.5)
        self.q_proj = Linear(ctx, self.key("q_proj"), tensor_parallel=False)
        self.k_proj = Linear(ctx, self.key("k_proj"), tensor_parallel=False)
        self.v_proj = Linear(ctx, self.key("v_proj"), tensor_parallel=False)
        self.o_proj = Linear(ctx, self.key("o_proj"), tensor_parallel=False)
        self.relative_k_proj = Linear(ctx,
                                      self.key("relative_k_proj"),
                                      tensor_parallel=False)

    @staticmethod
    def _relative_shift(scores):
        shape = F.shape_of(scores)
        batch = shape[0:1]
        heads = shape[1:2]
        length = shape[2:3]
        relative = shape[3:4]
        zero = scores.slice_axis(3, 0, 1, 4) * np.float16(0.0)
        padded = F.concatenate((zero, scores), 3)
        shifted = F.dynamic_reshape(padded,
                                    (batch, heads, relative + 1, length))
        shifted = F.dynamic_slice(shifted, (0, 0, 1, 0),
                                  (batch, heads, relative, length))
        shifted = F.dynamic_reshape(shifted, (batch, heads, length, relative))
        return F.dynamic_slice(shifted, (0, 0, 0, 0),
                               (batch, heads, length, length))

    def forward(self, hidden, position, mask):
        query = self.q_proj(hidden).reshape(
            (0, 0, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        key = self.k_proj(hidden).reshape(
            (0, 0, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        value = self.v_proj(hidden).reshape(
            (0, 0, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        relative_key = self.relative_k_proj(position).reshape(
            (1, -1, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        bias_u = F.constant(
            self.weights.f16(self.key("bias_u")).reshape(
                1, self.num_heads, 1, self.head_dim), "bias_u")
        bias_v = F.constant(
            self.weights.f16(self.key("bias_v")).reshape(
                1, self.num_heads, 1, self.head_dim), "bias_v")
        content = F.matmul(query + bias_u, key, transpose_rhs=True)
        positional = self._relative_shift(
            F.matmul(query + bias_v, relative_key, transpose_rhs=True))
        probabilities = ((content + positional) * self.scale + mask).softmax(3)
        output = F.matmul(probabilities, value).transpose((0, 2, 1, 3))
        return self.o_proj(output.reshape((0, 0, self.hidden_size)))


class ConformerConvolution(Module):
    """Pointwise, GLU, causal depthwise, norm, and pointwise branch."""

    def __init__(self, ctx, prefix: str, hidden_size: int,
                 kernel_size: int) -> None:
        super().__init__(ctx, prefix)
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.norm = NemotronLayerNorm(ctx, self.key("norm"))

    def forward(self, hidden):
        hidden = hidden.transpose((0, 2, 1))
        hidden = F.convolution(
            hidden,
            self.weights.fp16_parameter(self.key("pointwise_conv1.weight")),
            self.weights.opt_fp16_parameter(self.key("pointwise_conv1.bias")),
        )
        first = hidden.slice_axis(1, 0, self.hidden_size, 3)
        second = hidden.slice_axis(1, self.hidden_size, self.hidden_size, 3)
        hidden = first * second.sigmoid()
        hidden = F.convolution(
            hidden,
            self.weights.fp16_parameter(self.key("depthwise_conv.weight")),
            self.weights.opt_fp16_parameter(self.key("depthwise_conv.bias")),
            groups=self.hidden_size,
            pre_padding=(self.kernel_size - 1, ),
            post_padding=(0, ),
        )
        hidden = self.norm(hidden.transpose((0, 2, 1))).transpose((0, 2, 1))
        hidden = F.convolution(
            hidden.silu(),
            self.weights.fp16_parameter(self.key("pointwise_conv2.weight")),
            self.weights.opt_fp16_parameter(self.key("pointwise_conv2.bias")),
        )
        return hidden.transpose((0, 2, 1))


class ConformerFeedForward(Module):
    """Provider Linear-SiLU-Linear feed-forward branch."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.linear1 = Linear(ctx, self.key("linear1"), tensor_parallel=False)
        self.linear2 = Linear(ctx, self.key("linear2"), tensor_parallel=False)

    def forward(self, hidden):
        return self.linear2(self.linear1(hidden).silu())


class ConformerBlock(Module):
    """One macaron FastConformer block."""

    def __init__(self, ctx, prefix: str, config: dict) -> None:
        super().__init__(ctx, prefix)
        hidden_size = int(config["hidden_size"])
        self.norm_ff1 = NemotronLayerNorm(ctx, self.key("norm_feed_forward1"))
        self.ff1 = ConformerFeedForward(ctx, self.key("feed_forward1"))
        self.norm_attention = NemotronLayerNorm(ctx, self.key("norm_self_att"))
        self.attention = RelativePositionAttention(
            ctx, self.key("self_attn"), hidden_size,
            int(config["num_attention_heads"]))
        self.norm_conv = NemotronLayerNorm(ctx, self.key("norm_conv"))
        self.convolution = ConformerConvolution(
            ctx, self.key("conv"), hidden_size,
            int(config.get("conv_kernel_size", 9)))
        self.norm_ff2 = NemotronLayerNorm(ctx, self.key("norm_feed_forward2"))
        self.ff2 = ConformerFeedForward(ctx, self.key("feed_forward2"))
        self.norm_out = NemotronLayerNorm(ctx, self.key("norm_out"))

    def forward(self, hidden, position, mask):
        hidden = hidden + self.ff1(self.norm_ff1(hidden)) * np.float16(0.5)
        hidden = hidden + self.attention(self.norm_attention(hidden), position,
                                         mask)
        hidden = hidden + self.convolution(self.norm_conv(hidden))
        hidden = hidden + self.ff2(self.norm_ff2(hidden)) * np.float16(0.5)
        return self.norm_out(hidden)


def _chunked_mask(max_length: int, sliding_window: int,
                  lookahead: int) -> np.ndarray:
    chunk_size = lookahead + 1
    left_chunks = (sliding_window - 1) // chunk_size
    index = np.arange(max_length)
    chunk = index // chunk_size
    difference = chunk[:, None] - chunk[None, :]
    allowed = (difference >= 0) & (difference <= left_chunks)
    return np.where(allowed, 0.0,
                    np.finfo(np.float16).min).astype(np.float16).reshape(
                        1, 1, max_length, max_length)


class NemotronFastConformer(Module):
    """Causal subsampling and provider FastConformer stack."""

    def __init__(self, ctx, config: dict, max_length: int) -> None:
        super().__init__(ctx, "encoder")
        hidden_size = int(config["hidden_size"])
        max_positions = int(config.get("max_position_embeddings", 5000))
        if max_length > max_positions:
            raise ValueError(
                f"audio profile produces {max_length} frames, exceeding "
                f"max_position_embeddings={max_positions}")
        self.subsampling = CausalSubsampling(ctx, config)
        self.position = RelativePositionEmbedding(ctx, hidden_size,
                                                  max_positions)
        self.mask = _chunked_mask(
            max_length, int(config.get("sliding_window", 57)),
            int(config.get("default_num_lookahead_tokens", 3)))
        self.layers = [
            ConformerBlock(ctx, self.key(f"layers.{index}"), config)
            for index in range(int(config["num_hidden_layers"]))
        ]

    def forward(self, features):
        hidden = self.subsampling(features)
        position = self.position(hidden)
        length = F.shape_of(hidden)[1:2]
        mask = F.dynamic_slice(F.constant(self.mask, "attention_mask"),
                               (0, 0, 0, 0), (1, 1, length, length))
        for layer in self.layers:
            hidden = layer(hidden, position, mask)
        return hidden


class PromptProjector(Module):
    """Fuse the provider's one-hot language prompt into every frame."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx, "prompt_projector")
        self.linear1 = Linear(ctx, self.key("linear_1"), tensor_parallel=False)
        self.linear2 = Linear(ctx, self.key("linear_2"), tensor_parallel=False)

    def forward(self, hidden):
        return self.linear2(self.linear1(hidden).relu())


class Nemotron3_5AsrAudioEncoder(NetworkModule):
    """FastConformer, language prompt, and RNN-T joint-space projection."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        root = ctx.bundle.root
        self.config = root["encoder_config"]
        self.mel_bins = int(self.config.get("num_mel_bins", 128))
        self.num_prompts = int(root.get("num_prompts", 128))
        max_length = int(ctx.args.max_time_steps)
        for _ in range(_subsampling_layer_count(self.config)):
            max_length = max_length // 2 + 1
        self.encoder = NemotronFastConformer(ctx, self.config, max_length)
        self.prompt_projector = PromptProjector(ctx)
        self.encoder_projector = Linear(ctx,
                                        "encoder_projector",
                                        tensor_parallel=False)
        self.prompt_eye = np.eye(self.num_prompts, dtype=np.float16)

    def input_tensors(self):
        return {
            "input_features":
            self.add_input("input_features", trt.float16,
                           (1, -1, self.mel_bins)),
            "prompt_ids":
            self.add_input("prompt_ids", trt.int64, (1, )),
        }

    def forward(self, input_features, prompt_ids):
        hidden = self.encoder(input_features)
        prompt = F.constant(self.prompt_eye,
                            "prompt_eye").gather(prompt_ids, 0)
        prompt = prompt.unsqueeze(1, prompt.rank)
        frame_zeros = hidden.slice_last_dim(0, 1, 3) * np.float16(0.0)
        prompt = frame_zeros + prompt
        fused = self.prompt_projector(F.concatenate((hidden, prompt), 2))
        return {"encoder_frames": self.encoder_projector(fused)}
