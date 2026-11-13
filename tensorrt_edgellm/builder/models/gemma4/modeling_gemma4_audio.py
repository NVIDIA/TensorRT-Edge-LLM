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
"""Gemma4 audio checkpoint-direct graph."""

import numpy as np
import tensorrt as trt

from ...core import contracts
from ...ops import Module, NetworkModule
from ...ops import functional as F
from . import modeling_gemma4_layers


def _clamp(hidden_states, limit: np.float16):
    return hidden_states.maximum(-limit).minimum(limit)


def _fp16_clamp_limit(config: dict) -> np.float16:
    limit = min(float(config.get("gradient_clipping", 1e10)),
                float(np.finfo(np.float16).max))
    return np.float16(limit)


class Gemma4AudioRMSNorm(Module):
    """Checkpoint-backed Gemma4 audio RMSNorm."""

    def __init__(self, ctx, prefix: str, eps: float) -> None:
        super().__init__(ctx, prefix)
        self.eps = eps

    def forward(self, hidden):
        return F.rms_norm(hidden, self.weights.f16(self.key("weight")),
                          self.eps, 3)


class Gemma4AudioLayerNorm(Module):
    """Checkpoint-backed Gemma4 audio LayerNorm."""

    def __init__(self, ctx, prefix: str, eps: float) -> None:
        super().__init__(ctx, prefix)
        self.eps = eps

    def forward(self, hidden):
        weight = self.weights.f16(self.key("weight"))
        bias = self.weights.opt_f16(self.key("bias"))
        if bias is None:
            bias = np.zeros_like(weight)
        return F.layer_norm(hidden, weight, bias, self.eps, hidden.rank)


class Gemma4AudioUnitRMSNorm(Module):
    """RMSNorm with the provider's implicit unit scale."""

    def __init__(self, ctx, eps: float) -> None:
        super().__init__(ctx, "audio_tower.output_norm")
        self.eps = eps

    def forward(self, hidden):
        weight = np.ones(int(hidden.shape[-1]), dtype=np.float16)
        return F.rms_norm(hidden, weight, self.eps, 3)


class Gemma4AudioDepthwiseConv1d(Module):
    """Depthwise convolution used by the Gemma4 local-convolution block."""

    def forward(self, hidden):
        kernel = self.weights.fp16_parameter(self.key("weight"))
        return F.convolution(hidden,
                             kernel,
                             self.weights.opt_fp16_parameter(self.key("bias")),
                             groups=int(kernel.shape[0]))


class Gemma4AudioConv2d(Module):
    """Stride-two convolution used by Gemma4 audio subsampling."""

    def forward(self, hidden):
        return F.convolution(hidden,
                             self.weights.fp16_parameter(self.key("weight")),
                             self.weights.opt_fp16_parameter(self.key("bias")),
                             stride=(2, 2),
                             padding=(1, 1))


class Gemma4AudioFeedForward(Module):
    """Feed-forward branch with Gemma audio residual scaling."""

    def __init__(self, ctx, prefix: str, config: dict) -> None:
        super().__init__(ctx, prefix)
        self.residual_weight = np.float16(config.get("residual_weight", 0.5))
        self.eps = float(config.get("rms_norm_eps", 1e-6))
        self.activation = str(config.get("hidden_act", "silu"))
        self.clamp_limit = _fp16_clamp_limit(config)
        self.ffw_layer_1 = modeling_gemma4_layers.Gemma4ClippableLinear(
            ctx, self.key("ffw_layer_1"))
        self.ffw_layer_2 = modeling_gemma4_layers.Gemma4ClippableLinear(
            ctx, self.key("ffw_layer_2"))
        self.pre_layer_norm = Gemma4AudioRMSNorm(ctx,
                                                 self.key("pre_layer_norm"),
                                                 self.eps)
        self.post_layer_norm = Gemma4AudioRMSNorm(ctx,
                                                  self.key("post_layer_norm"),
                                                  self.eps)

    def forward(self, hidden):
        residual = hidden
        hidden = _clamp(residual, self.clamp_limit)
        hidden = self.pre_layer_norm(hidden)
        hidden = self.ffw_layer_1(hidden).activation(self.activation)
        hidden = self.ffw_layer_2(hidden)
        hidden = _clamp(hidden, self.clamp_limit)
        hidden = self.post_layer_norm(hidden)
        return residual + hidden * self.residual_weight


class Gemma4AudioLightConv1d(Module):
    """Depthwise local convolution branch used inside Gemma audio layers."""

    def __init__(self, ctx, prefix: str, config: dict) -> None:
        super().__init__(ctx, prefix)
        self.eps = float(config.get("rms_norm_eps", 1e-6))
        self.activation = str(config.get("hidden_act", "silu"))
        self.clamp_limit = _fp16_clamp_limit(config)
        self.linear_start = modeling_gemma4_layers.Gemma4ClippableLinear(
            ctx, self.key("linear_start"))
        self.linear_end = modeling_gemma4_layers.Gemma4ClippableLinear(
            ctx, self.key("linear_end"))
        self.pre_layer_norm = Gemma4AudioRMSNorm(ctx,
                                                 self.key("pre_layer_norm"),
                                                 self.eps)
        self.depthwise_conv1d = Gemma4AudioDepthwiseConv1d(
            ctx, self.key("depthwise_conv1d"))
        self.conv_norm = Gemma4AudioRMSNorm(ctx, self.key("conv_norm"),
                                            self.eps)

    def forward(self, hidden):
        residual = hidden
        hidden = self.pre_layer_norm(residual)
        hidden = self.linear_start(hidden)
        width = int(hidden.shape[-1]) // 2
        first = hidden.slice_last_dim(0, width, 3)
        second = hidden.slice_last_dim(width, width, 3)
        hidden = (first * second.sigmoid()).transpose((0, 2, 1))
        kernel = self.weights.parameter_spec(
            self.key("depthwise_conv1d.weight"))
        zero = hidden.slice_axis(2, 0, 1, 3) * np.float16(0.0)
        hidden = F.concatenate(
            tuple([zero] * (kernel.shape[-1] - 1)) + (hidden, ), 2)
        hidden = self.depthwise_conv1d(hidden)
        hidden = hidden.transpose((0, 2, 1))
        hidden = _clamp(hidden, self.clamp_limit)
        hidden = self.conv_norm(hidden)
        hidden = hidden.activation(self.activation)
        hidden = self.linear_end(hidden)
        return residual + hidden


class Gemma4AudioAttention(Module):
    """Chunked relative attention for the Gemma4 audio encoder."""

    def __init__(self, ctx, prefix: str, config: dict) -> None:
        super().__init__(ctx, prefix)
        self.config = config
        self.num_heads = int(config["num_attention_heads"])
        self.chunk_size = int(config.get("attention_chunk_size", 12))
        self.context_left = int(config.get("attention_context_left", 13))
        self.context_right = int(config.get("attention_context_right", 0))
        self.context_size = (self.chunk_size + self.context_left - 1 +
                             self.context_right)
        self.logit_cap = float(config.get("attention_logit_cap", 50.0))
        self.q_proj = modeling_gemma4_layers.Gemma4ClippableLinear(
            ctx, self.key("q_proj"))
        self.k_proj = modeling_gemma4_layers.Gemma4ClippableLinear(
            ctx, self.key("k_proj"))
        self.v_proj = modeling_gemma4_layers.Gemma4ClippableLinear(
            ctx, self.key("v_proj"))
        self.post = modeling_gemma4_layers.Gemma4ClippableLinear(
            ctx, self.key("post"))

    def _relative_position(self, hidden_size: int, head_dim: int):
        positions = np.arange(self.context_size // 2, -1, -1,
                              dtype=np.float32)[:, None]
        count = hidden_size // 2
        increment = np.log(10000.0) / max(count - 1, 1)
        timescales = np.exp(-np.arange(count, dtype=np.float32) * increment)
        embedding = np.concatenate(
            (np.sin(positions * timescales), np.cos(positions * timescales)),
            axis=1).astype(np.float16)
        relative = F.linear(F.constant(embedding, "relative_position"),
                            self.key("relative_k_proj"), 2)
        return relative.reshape((-1, self.num_heads, head_dim))

    def forward(self, hidden, valid):
        hidden_size = int(hidden.shape[-1])
        head_dim = hidden_size // self.num_heads
        query = self.q_proj(hidden).reshape((0, 0, self.num_heads, head_dim))
        key = self.k_proj(hidden).reshape((0, 0, self.num_heads, head_dim))
        value = self.v_proj(hidden).reshape((0, 0, self.num_heads, head_dim))
        gamma = F.constant(self.weights.f32(self.key("per_dim_scale")),
                           "per_dim_scale")
        relative = self._relative_position(hidden_size, head_dim)
        sequence_length = F.shape_of(hidden)[1:2]
        hidden = F.gemma4_attention(query, key, value, gamma, relative, valid,
                                    sequence_length, self.chunk_size,
                                    self.context_left - 1, self.context_size,
                                    self.logit_cap)
        hidden = hidden.reshape((0, 0, hidden_size))
        return self.post(hidden)


class Gemma4AudioSubSampleLayer(Module):
    """One convolution, normalization, and activation subsampling stage."""

    def __init__(self, ctx, prefix: str, eps: float) -> None:
        super().__init__(ctx, prefix)
        self.convolution = Gemma4AudioConv2d(ctx, self.key("conv"))
        self.norm = Gemma4AudioLayerNorm(ctx, self.key("norm"), eps)

    def forward(self, hidden):
        hidden = self.convolution(hidden)
        hidden = hidden.transpose((0, 2, 3, 1))
        hidden = self.norm(hidden)
        return hidden.transpose((0, 3, 1, 2)).relu()


class Gemma4AudioSubSampleConvProjection(Module):
    """Model-owned convolutional subsampling frontend."""

    def __init__(self, ctx, config: dict) -> None:
        super().__init__(ctx, "audio_tower.subsample_conv_projection")
        self.eps = float(config.get("rms_norm_eps", 1e-6))
        self.root = self.weights.find_suffix(
            "subsample_conv_projection.layer0.conv.weight"
        )[:-len("layer0.conv.weight")]
        self.layers = [
            Gemma4AudioSubSampleLayer(ctx, self.root + f"layer{index}",
                                      self.eps) for index in range(2)
        ]
        self.input_projection = modeling_gemma4_layers.Gemma4ClippableLinear(
            ctx, self.root + "input_proj_linear")

    @staticmethod
    def _downsample_mask(mask):
        shape = F.shape_of(mask)
        output_length = (shape[1:2] + np.int32(1)) / np.int32(2)
        output_shape = F.concatenate((shape[0:1], output_length), 0)
        return F.dynamic_slice(mask, (0, 0), output_shape, stride=(1, 2))

    def forward(self, features, input_features_mask):
        hidden = features.unsqueeze(1, 3)
        for layer in self.layers:
            mask = input_features_mask.cast(trt.float16)
            mask = mask.unsqueeze(1, 2).unsqueeze(3, 3)
            hidden = hidden * mask
            hidden = layer(hidden)
            input_features_mask = self._downsample_mask(input_features_mask)
        hidden = hidden.transpose((0, 2, 3, 1)).reshape((0, 0, -1))
        return self.input_projection(hidden), input_features_mask


class Gemma4AudioLayer(Module):
    """One Gemma4 audio conformer-style layer."""

    def __init__(self, ctx, prefix: str, config: dict) -> None:
        super().__init__(ctx, prefix)
        self.eps = float(config.get("rms_norm_eps", 1e-6))
        self.clamp_limit = _fp16_clamp_limit(config)
        self.feed_forward1 = Gemma4AudioFeedForward(ctx,
                                                    self.key("feed_forward1"),
                                                    config)
        self.self_attn = Gemma4AudioAttention(ctx, self.key("self_attn"),
                                              config)
        self.lconv1d = Gemma4AudioLightConv1d(ctx, self.key("lconv1d"), config)
        self.feed_forward2 = Gemma4AudioFeedForward(ctx,
                                                    self.key("feed_forward2"),
                                                    config)
        self.norm_pre_attn = Gemma4AudioRMSNorm(ctx, self.key("norm_pre_attn"),
                                                self.eps)
        self.norm_post_attn = Gemma4AudioRMSNorm(ctx,
                                                 self.key("norm_post_attn"),
                                                 self.eps)
        self.norm_out = Gemma4AudioRMSNorm(ctx, self.key("norm_out"), self.eps)

    def forward(self, hidden, valid):
        hidden = self.feed_forward1(hidden)
        residual = hidden
        hidden = _clamp(residual, self.clamp_limit)
        normalized = self.norm_pre_attn(hidden)
        attention = self.self_attn(normalized, valid)
        attention = _clamp(attention, self.clamp_limit)
        attention = self.norm_post_attn(attention)
        hidden = residual + attention
        hidden = self.lconv1d(hidden)
        hidden = self.feed_forward2(hidden)
        hidden = _clamp(hidden, self.clamp_limit)
        return self.norm_out(hidden)


class Gemma4AudioOutputProjection(Module):
    """Final Gemma4 audio projection and unit-weight RMS normalization."""

    def __init__(self, ctx, output_prefix: str, embedding_prefix: str,
                 eps: float) -> None:
        super().__init__(ctx, "audio_tower.output")
        self.output_projection = modeling_gemma4_layers.Gemma4ClippableLinear(
            ctx, output_prefix)
        self.embedding_projection = modeling_gemma4_layers.Gemma4ClippableLinear(
            ctx, embedding_prefix)
        self.norm = Gemma4AudioUnitRMSNorm(ctx, eps)

    def forward(self, hidden):
        hidden = self.output_projection(hidden)
        hidden = self.norm(hidden)
        return self.embedding_projection(hidden)


class Gemma4AudioModel(NetworkModule):
    """End-to-end Gemma4 audio encoder graph."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle)

    def __init__(self, ctx, bundle) -> None:
        super().__init__(ctx, "audio_tower")
        self.config = bundle.component_dict(contracts.Component.AUDIO)
        self.mel_bins = int(self.config.get("num_mel_bins", 128))
        self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(
            ctx, self.config)
        layer_prefixes = self.weights.layer_prefixes((
            r"(.+audio_tower\.layers\.\d+)\.feed_forward1\.pre_layer_norm\.weight$",
        ))
        self.layers = [
            Gemma4AudioLayer(ctx, prefix, self.config)
            for prefix in layer_prefixes
        ]
        output_key = self.weights.find_suffix("audio_tower.output_proj.weight")
        output_proj = output_key[:-len(".weight")]
        embed_key = self.weights.find_suffix(
            "embed_audio.embedding_projection.weight")
        embedding_projection = embed_key[:-len(".weight")]
        self.output_projection = Gemma4AudioOutputProjection(
            ctx, output_proj, embedding_projection,
            float(self.config.get("rms_norm_eps", 1e-6)))

    def input_tensors(self):
        return {
            "features":
            self.add_input("input_features", trt.float16,
                           (1, -1, self.mel_bins)),
            "input_features_mask":
            self.add_input("input_features_mask", trt.bool, (1, -1)),
        }

    def forward(self, features, input_features_mask):
        hidden, valid = self.subsample_conv_projection(features,
                                                       input_features_mask)
        for layer in self.layers:
            hidden = layer(hidden, valid)
        return {"last_hidden_state": self.output_projection(hidden)}
