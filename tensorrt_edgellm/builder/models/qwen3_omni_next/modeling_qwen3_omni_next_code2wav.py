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
"""Checkpoint-direct Qwen3-Omni-Next online codec decoder."""

from typing import Dict

import numpy as np
import tensorrt as trt

from ...core import contracts
from ...ops import Linear, Module, NetworkModule
from ...ops import functional as F
from ...ops.code2wav import (CausalConv1d, QwenCode2WavVocoder,
                             QwenQuantizerGroup)

__all__ = [
    "Qwen3OmniNextCodecAttention",
    "Qwen3OmniNextCodecTransformerLayer",
    "Qwen3OmniNextCodecTransformer",
    "Qwen3OmniNextCode2WavModel",
]


def _attention_constants(hidden_states, config: Dict, max_code_len: int,
                         window_size: int):
    head_dim = int(config["head_dim"])
    rope_theta = float(config.get("rope_theta", 10000.0))
    positions = np.arange(max_code_len, dtype=np.float32)[:, None]
    frequencies = 1.0 / (rope_theta**(
        np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angles = positions * frequencies[None, :]
    rotary = np.concatenate((angles, angles), axis=-1)
    cosine = F.constant(
        np.cos(rotary).astype(np.float16).reshape(1, 1, max_code_len,
                                                  head_dim), "rope_cosine")
    sine = F.constant(
        np.sin(rotary).astype(np.float16).reshape(1, 1, max_code_len,
                                                  head_dim), "rope_sine")

    query_position = np.arange(max_code_len)[:, None]
    key_position = np.arange(max_code_len)[None, :]
    valid = ((key_position > query_position - window_size)
             & (key_position <= query_position))
    mask = np.where(valid, 0.0, np.finfo(np.float16).min).astype(np.float16)
    mask = F.constant(mask.reshape(1, 1, max_code_len, max_code_len),
                      "attention_mask")

    length = F.shape_of(hidden_states)[1:2]
    position_size = (1, 1, length, head_dim)
    mask_size = (1, 1, length, length)
    return (
        F.dynamic_slice(cosine, (0, 0, 0, 0), position_size),
        F.dynamic_slice(sine, (0, 0, 0, 0), position_size),
        F.dynamic_slice(mask, (0, 0, 0, 0), mask_size),
    )


class Qwen3OmniNextCodecRMSNorm(Module):
    """Standard, non-unit-offset RMSNorm used by the online codec."""

    def __init__(self, ctx, prefix: str, eps: float) -> None:
        super().__init__(ctx, prefix)
        self.eps = eps

    def forward(self, hidden_states):
        return F.rms_norm(hidden_states, self.weights.f16(self.key("weight")),
                          self.eps, 3)


class Qwen3OmniNextCodecAttention(Module):
    """Provider fused-QKV, grouped-query sliding-window attention."""

    def __init__(self, ctx, prefix: str, config: Dict) -> None:
        super().__init__(ctx, prefix)
        self.num_heads = int(config["num_attention_heads"])
        self.num_key_value_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        if self.num_heads % self.num_key_value_heads:
            raise ValueError("codec key/value heads must divide query heads")
        self.query_size = self.num_heads * self.head_dim
        self.key_value_size = self.num_key_value_heads * self.head_dim
        self.wqkv = Linear(ctx,
                           self.key("wqkv"),
                           rank=3,
                           tensor_parallel=False)
        self.wo = Linear(ctx, self.key("wo"), rank=3, tensor_parallel=False)

    def _apply_rope(self, tensor, cosine, sine):
        first = tensor.slice_last_dim(0, self.head_dim // 2, 4)
        second = tensor.slice_last_dim(self.head_dim // 2, self.head_dim // 2,
                                       4)
        rotated = F.concatenate((-second, first), 3)
        return tensor * cosine + rotated * sine

    def _repeat_key_value(self, tensor):
        if self.num_heads == self.num_key_value_heads:
            return tensor
        repeats = self.num_heads // self.num_key_value_heads
        indices = np.arange(self.num_heads, dtype=np.int64) // repeats
        return tensor.gather(F.constant(indices, "gqa_head_indices"), 1)

    def forward(self, hidden_states, cosine, sine, mask):
        projected = self.wqkv(hidden_states)
        query = projected.slice_last_dim(0, self.query_size, 3)
        key = projected.slice_last_dim(self.query_size, self.key_value_size, 3)
        value = projected.slice_last_dim(self.query_size + self.key_value_size,
                                         self.key_value_size, 3)
        query = query.reshape((0, 0, self.num_heads, self.head_dim)).transpose(
            (0, 2, 1, 3))
        key = key.reshape(
            (0, 0, self.num_key_value_heads, self.head_dim)).transpose(
                (0, 2, 1, 3))
        value = value.reshape(
            (0, 0, self.num_key_value_heads, self.head_dim)).transpose(
                (0, 2, 1, 3))
        query = self._apply_rope(query, cosine, sine)
        key = self._repeat_key_value(self._apply_rope(key, cosine, sine))
        value = self._repeat_key_value(value)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            mask=mask,
            scale=self.head_dim**-0.5,
        )
        output = output.transpose((0, 2, 1, 3)).reshape(
            (0, 0, self.query_size))
        return self.wo(output)


class Qwen3OmniNextCodecFeedForward(Module):
    """Provider-named SwiGLU feed-forward block."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.w1 = Linear(ctx, self.key("w1"), rank=3, tensor_parallel=False)
        self.w3 = Linear(ctx, self.key("w3"), rank=3, tensor_parallel=False)
        self.w2 = Linear(ctx, self.key("w2"), rank=3, tensor_parallel=False)

    def forward(self, hidden_states):
        return self.w2(self.w1(hidden_states).silu() * self.w3(hidden_states))


class Qwen3OmniNextCodecTransformerLayer(Module):
    """One fused-attention codec transformer block with LayerScale."""

    def __init__(self, ctx, prefix: str, config: Dict) -> None:
        super().__init__(ctx, prefix)
        eps = float(config.get("rms_norm_eps", 1e-5))
        self.attention_norm = Qwen3OmniNextCodecRMSNorm(
            ctx, self.key("attention_norm"), eps)
        self.attention = Qwen3OmniNextCodecAttention(ctx,
                                                     self.key("attention"),
                                                     config)
        self.ffn_norm = Qwen3OmniNextCodecRMSNorm(ctx, self.key("ffn_norm"),
                                                  eps)
        self.feed_forward = Qwen3OmniNextCodecFeedForward(
            ctx, self.key("feed_forward"))

    def _layer_scale(self, suffix: str):
        return F.constant(
            self.weights.f16(self.key(suffix)).reshape(1, 1, -1),
            suffix.rsplit(".", 1)[0])

    def forward(self, hidden_states, cosine, sine, mask):
        attention = self.attention(self.attention_norm(hidden_states), cosine,
                                   sine, mask)
        hidden_states = hidden_states + attention * self._layer_scale(
            "attention_layer_scale.gamma")
        feed_forward = self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states + feed_forward * self._layer_scale(
            "ffn_layer_scale.gamma")


class Qwen3OmniNextCodecTransformer(Module):
    """Channels-first online-codec transformer with a bounded causal window."""

    def __init__(self, ctx, config: Dict, max_code_len: int,
                 input_dimension: int, window_size: int) -> None:
        super().__init__(ctx, "pre_transformer")
        max_positions = int(config["max_position_embeddings"])
        if max_code_len <= 0 or max_code_len > max_positions:
            raise ValueError("Qwen3-Omni-Next max_code_len must be in "
                             f"[1, {max_positions}], got {max_code_len}")
        if window_size <= 0:
            raise ValueError(
                "Qwen3-Omni-Next codec window size must be positive")
        self.config = config
        self.max_code_len = max_code_len
        self.window_size = window_size
        hidden_size = int(config["hidden_size"])
        self.input_proj = (
            Linear(ctx, self.key("input_proj"), rank=3, tensor_parallel=False)
            if input_dimension != hidden_size
            or self.weights.has(self.key("input_proj.weight")) else None)
        self.output_proj = (
            Linear(ctx, self.key("output_proj"), rank=3, tensor_parallel=False)
            if input_dimension != hidden_size
            or self.weights.has(self.key("output_proj.weight")) else None)
        self.layers = [
            Qwen3OmniNextCodecTransformerLayer(ctx,
                                               self.key(f"layers.{index}"),
                                               config)
            for index in range(int(config["num_hidden_layers"]))
        ]
        self.norm = Qwen3OmniNextCodecRMSNorm(
            ctx, self.key("norm"), float(config.get("rms_norm_eps", 1e-5)))

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose((0, 2, 1))
        if self.input_proj is not None:
            hidden_states = self.input_proj(hidden_states)
        cosine, sine, mask = _attention_constants(hidden_states, self.config,
                                                  self.max_code_len,
                                                  self.window_size)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cosine, sine, mask)
        hidden_states = self.norm(hidden_states)
        if self.output_proj is not None:
            hidden_states = self.output_proj(hidden_states)
        return hidden_states


class Qwen3OmniNextCode2WavModel(NetworkModule):
    """Split-RVQ, transformer, ConvNeXt, and DAC waveform decoder."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle, ctx.args.max_code_len)

    def __init__(self, ctx, bundle, max_code_len: int) -> None:
        super().__init__(ctx)
        self.codec_config = bundle.component_dict(contracts.Component.CODE2WAV)
        num_quantizers = int(self.codec_config["num_quantizers"])
        semantic_count = int(self.codec_config["num_semantic_quantizers"])
        if not 0 < semantic_count < num_quantizers:
            raise ValueError(
                "Qwen3-Omni-Next codec requires semantic and acoustic RVQ")
        self.num_quantizers = num_quantizers
        self.semantic_quantizer = QwenQuantizerGroup(
            ctx, "quantizer.rvq_first", tuple(range(semantic_count)))
        self.acoustic_quantizer = QwenQuantizerGroup(
            ctx, "quantizer.rvq_rest",
            tuple(range(semantic_count, num_quantizers)))
        self.pre_conv = CausalConv1d(ctx, "pre_conv.conv")
        transformer = dict(self.codec_config["transformer"])
        self.pre_transformer = Qwen3OmniNextCodecTransformer(
            ctx,
            transformer,
            max_code_len,
            int(self.codec_config["pre_transformer_input_dimension"]),
            int(self.codec_config["pre_transformer_window_size"]),
        )
        vocoder_config = {
            "upsampling_ratios": list(self.codec_config["pre_upsample_rates"]),
            "upsample_rates": list(self.codec_config["decoder_rates"]),
        }
        self.vocoder = QwenCode2WavVocoder(ctx,
                                           vocoder_config,
                                           root="",
                                           symmetric_padding=False)

    def input_tensors(self):
        return {
            "codes":
            self.add_input("codes", trt.int64, (-1, self.num_quantizers, -1))
        }

    def forward(self, codes):
        hidden_states = (self.semantic_quantizer(codes) +
                         self.acoustic_quantizer(codes))
        hidden_states = self.pre_conv(hidden_states.transpose((0, 2, 1)))
        hidden_states = self.pre_transformer(hidden_states)
        return {"waveform": self.vocoder(hidden_states).cast(trt.float16)}
