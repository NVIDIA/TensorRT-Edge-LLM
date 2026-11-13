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
"""Shared Qwen Code2Wav decoder modules."""

from typing import Dict, Tuple

import numpy as np

from . import functional as F
from .linear import Linear
from .module import Module


def _codebook(weights, prefix: str) -> np.ndarray:
    embedding = weights.f32(prefix + ".embedding_sum")
    usage = np.maximum(weights.f32(prefix + ".cluster_usage"), 1e-5)
    return np.ascontiguousarray(embedding / usage[:, None], dtype=np.float16)


class QwenCodebookEmbedding(Module):
    """One residual-vector-quantizer codebook lookup."""

    def __init__(self, ctx, prefix: str, code_index: int) -> None:
        super().__init__(ctx, prefix)
        self.code_index = code_index

    def forward(self, codes):
        code = codes.slice_axis(1, self.code_index, 1, 3).reshape((0, -1))
        return F.embedding_lookup(_codebook(self.weights, self.prefix), code)


class QwenQuantizerProjection(Module):
    """Project a summed RVQ group into the decoder feature space."""

    def forward(self, hidden):
        projection = self.weights.f16(self.key("weight"))
        if projection.shape[-1] != 1:
            raise ValueError(
                f"unsupported quantizer projection {projection.shape}")
        return F.linear_with_weights(hidden, projection[:, :, 0], rank=3)


class QwenQuantizerGroup(Module):
    """RVQ codebook group plus output projection."""

    def __init__(self, ctx, prefix: str, code_indices: Tuple[int,
                                                             ...]) -> None:
        super().__init__(ctx, prefix)
        self.codebooks = [
            QwenCodebookEmbedding(
                ctx, self.key(f"vq.layers.{layer_index}._codebook"),
                code_index)
            for layer_index, code_index in enumerate(code_indices)
        ]
        self.output_proj = QwenQuantizerProjection(ctx,
                                                   self.key("output_proj"))

    def forward(self, codes):
        quantized = None
        for codebook in self.codebooks:
            embedded = codebook(codes)
            quantized = embedded if quantized is None else quantized + embedded
        return self.output_proj(quantized)


class CausalConv1d(Module):
    """Causal 1D convolution wrapper used by Code2Wav blocks."""

    def __init__(self,
                 ctx,
                 prefix: str,
                 dilation: int = 1,
                 groups: int = 1) -> None:
        super().__init__(ctx, prefix)
        self.dilation = dilation
        self.groups = groups

    def forward(self, hidden):
        kernel = self.weights.f16(self.key("weight"))
        effective_kernel = (int(kernel.shape[-1]) - 1) * self.dilation + 1
        return F.convolution(hidden,
                             kernel,
                             self.weights.opt_f16(self.key("bias")),
                             dilation=(self.dilation, ),
                             groups=self.groups,
                             pre_padding=(effective_kernel - 1, ),
                             post_padding=(0, ))


class CausalDeconv1d(Module):
    """Causal transposed convolution wrapper used by upsampling blocks."""

    def __init__(self,
                 ctx,
                 prefix: str,
                 stride: int,
                 symmetric_padding: bool = False) -> None:
        super().__init__(ctx, prefix)
        self.stride = stride
        self.symmetric_padding = symmetric_padding

    def forward(self, hidden):
        kernel = self.weights.f16(self.key("weight"))
        crop = int(kernel.shape[-1]) - self.stride
        return F.deconvolution(
            hidden,
            kernel,
            self.weights.opt_f16(self.key("bias")),
            stride=(self.stride, ),
            pre_padding=(crop if self.symmetric_padding else 0, ),
            post_padding=(crop, ))


class QwenCode2WavNorm(Module):
    """Checkpoint-backed Qwen Code2Wav normalization."""

    def __init__(self, ctx, prefix: str, eps: float) -> None:
        super().__init__(ctx, prefix)
        self.eps = eps

    def forward(self, hidden):
        return F.normalization(hidden, self.prefix, self.eps, 3)


class QwenCode2WavMLP(Module):
    """Gated feed-forward block used by the Code2Wav transformer."""

    def __init__(self, ctx, prefix: str, hidden_act: str) -> None:
        super().__init__(ctx, prefix)
        self.hidden_act = hidden_act
        self.gate_proj = Linear(ctx,
                                self.key("gate_proj"),
                                rank=3,
                                tensor_parallel=False)
        self.up_proj = Linear(ctx,
                              self.key("up_proj"),
                              rank=3,
                              tensor_parallel=False)
        self.down_proj = Linear(ctx,
                                self.key("down_proj"),
                                rank=3,
                                tensor_parallel=False)

    def forward(self, hidden):
        gate = self.gate_proj(hidden).activation(self.hidden_act)
        return self.down_proj(gate * self.up_proj(hidden))


def _attention_constants(hidden, config: Dict, max_code_len: int):
    head_dim = int(
        config.get("head_dim")
        or int(config["hidden_size"]) / int(config["num_attention_heads"]))
    rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
    rope_theta = float(
        rope.get("rope_theta", config.get("rope_theta", 10000.0)))
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

    position = np.arange(max_code_len)
    query_position = position[:, None]
    key_position = position[None, :]
    window = int(config.get("sliding_window", max_code_len))
    valid = ((key_position > query_position - window)
             & (key_position <= query_position))
    mask = np.where(valid, 0.0, np.finfo(np.float16).min).astype(np.float16)
    mask = F.constant(mask.reshape(1, 1, max_code_len, max_code_len),
                      "attention_mask")

    length = F.shape_of(hidden)[1:2]
    position_size = (1, 1, length, head_dim)
    mask_size = (1, 1, length, length)
    return (F.dynamic_slice(cosine, (0, 0, 0, 0), position_size),
            F.dynamic_slice(sine, (0, 0, 0, 0), position_size),
            F.dynamic_slice(mask, (0, 0, 0, 0), mask_size))


class QwenCode2WavAttention(Module):
    """Decoder self-attention with static RoPE tables and causal mask slices."""

    def __init__(self, ctx, prefix: str, config: Dict) -> None:
        super().__init__(ctx, prefix)
        self.config = config
        self.num_heads = int(config["num_attention_heads"])
        num_kv_heads = int(config.get("num_key_value_heads", self.num_heads))
        if num_kv_heads != self.num_heads:
            raise ValueError("Qwen Code2Wav requires equal Q and KV heads")
        self.head_dim = int(
            config.get("head_dim")
            or int(config["hidden_size"]) / self.num_heads)
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
        self.o_proj = Linear(ctx,
                             self.key("o_proj"),
                             rank=3,
                             tensor_parallel=False)

    def _apply_rope(self, tensor, cosine, sine):
        tensor = tensor
        first = tensor.slice_last_dim(0, self.head_dim // 2, 4)
        second = tensor.slice_last_dim(self.head_dim // 2, self.head_dim // 2,
                                       4)
        rotated = F.concatenate((-second, first), 3)
        return tensor * cosine + rotated * sine

    def forward(self, hidden, cosine, sine, mask):
        query = self.q_proj(hidden)
        key = self.k_proj(hidden)
        value = self.v_proj(hidden)
        query = query.reshape((0, 0, self.num_heads, self.head_dim)).transpose(
            (0, 2, 1, 3))
        key = key.reshape((0, 0, self.num_heads, self.head_dim)).transpose(
            (0, 2, 1, 3))
        value = value.reshape((0, 0, self.num_heads, self.head_dim)).transpose(
            (0, 2, 1, 3))
        query = self._apply_rope(query, cosine, sine)
        key = self._apply_rope(key, cosine, sine)
        output = F.scaled_dot_product_attention(query,
                                                key,
                                                value,
                                                mask=mask,
                                                scale=self.head_dim**-0.5)
        output = output.transpose((0, 2, 1, 3)).reshape(
            (0, 0, self.num_heads * self.head_dim))
        return self.o_proj(output)


class QwenCode2WavTransformerLayer(Module):
    """One Code2Wav pre-transformer decoder layer."""

    def __init__(self, ctx, prefix: str, config: Dict) -> None:
        super().__init__(ctx, prefix)
        self.config = config
        self.eps = float(config.get("rms_norm_eps", 1e-5))
        self.hidden_act = str(config.get("hidden_act", "silu"))
        self.self_attn = QwenCode2WavAttention(ctx, self.key("self_attn"),
                                               config)
        self.input_layernorm = QwenCode2WavNorm(ctx,
                                                self.key("input_layernorm"),
                                                self.eps)
        self.post_attention_layernorm = QwenCode2WavNorm(
            ctx, self.key("post_attention_layernorm"), self.eps)
        self.mlp = QwenCode2WavMLP(ctx, self.key("mlp"), self.hidden_act)

    def forward(self, hidden, cosine, sine, mask):
        residual = hidden
        normalized = self.input_layernorm(residual)
        attention = self.self_attn(normalized, cosine, sine, mask)
        attention = attention * F.constant(
            self.weights.f16(self.key("self_attn_layer_scale.scale")).reshape(
                1, 1, -1), "attention_layer_scale")
        hidden = residual + attention

        residual = hidden
        normalized = self.post_attention_layernorm(hidden)
        feed_forward = self.mlp(normalized)
        feed_forward = feed_forward * F.constant(
            self.weights.f16(self.key("mlp_layer_scale.scale")).reshape(
                1, 1, -1), "mlp_layer_scale")
        return residual + feed_forward


class QwenCode2WavPreTransformer(Module):
    """Transformer that refines code embeddings before the waveform decoder."""

    def __init__(self,
                 ctx,
                 config: Dict,
                 max_code_len: int,
                 root: str = "decoder.pre_transformer") -> None:
        super().__init__(ctx, root)
        self.config = config
        self.max_code_len = max_code_len
        self.eps = float(config.get("rms_norm_eps", 1e-5))
        self.input_proj = (
            Linear(ctx, self.key("input_proj"), rank=3, tensor_parallel=False)
            if self.weights.has(self.key("input_proj.weight")) else None)
        self.output_proj = (
            Linear(ctx, self.key("output_proj"), rank=3, tensor_parallel=False)
            if self.weights.has(self.key("output_proj.weight")) else None)
        self.norm = QwenCode2WavNorm(ctx, self.key("norm"), self.eps)
        self.layers = [
            QwenCode2WavTransformerLayer(ctx, self.key(f"layers.{index}"),
                                         config)
            for index in range(int(config["num_hidden_layers"]))
        ]

    def forward(self, hidden):
        if self.input_proj is not None:
            hidden = self.input_proj(hidden)
        cosine, sine, mask = _attention_constants(hidden, self.config,
                                                  self.max_code_len)
        for layer in self.layers:
            hidden = layer(hidden, cosine, sine, mask)
        hidden = self.norm(hidden)
        if self.output_proj is not None:
            hidden = self.output_proj(hidden)
        return hidden


class SnakeBeta(Module):
    """SnakeBeta activation used by the audio decoder."""

    def forward(self, hidden):
        alpha = np.exp(self.weights.f32(self.key("alpha"))).astype(np.float16)
        beta = np.exp(self.weights.f32(self.key("beta"))).astype(np.float16)
        hidden = hidden
        phase = hidden * F.constant(alpha.reshape(1, -1, 1), "snake_alpha")
        sine = phase.sin()
        update = sine * sine * F.constant((1.0 / (beta + 1e-9)).astype(
            np.float16).reshape(1, -1, 1), "snake_inverse_beta")
        return hidden + update


class ConvNextUpsample(Module):
    """One ConvNeXt-style upsampling block."""

    def __init__(self,
                 ctx,
                 stage: int,
                 root: str = "decoder.upsample",
                 symmetric_padding: bool = False) -> None:
        super().__init__(ctx, f"{root}.{stage}")
        self.deconv = CausalDeconv1d(ctx, self.key("0.conv"), 2,
                                     symmetric_padding)
        dwconv = self.key("1.dwconv.conv")
        groups = int(self.weights.parameter_spec(dwconv + ".weight").shape[0])
        self.dwconv = CausalConv1d(ctx, dwconv, groups=groups)
        self.norm = QwenCode2WavNorm(ctx, self.key("1.norm"), 1e-6)
        self.pwconv1 = Linear(ctx,
                              self.key("1.pwconv1"),
                              rank=3,
                              tensor_parallel=False)
        self.pwconv2 = Linear(ctx,
                              self.key("1.pwconv2"),
                              rank=3,
                              tensor_parallel=False)

    def forward(self, hidden):
        hidden = self.deconv(hidden)
        residual = hidden
        hidden = self.dwconv(hidden)
        hidden = hidden.transpose((0, 2, 1))
        hidden = self.pwconv2(self.pwconv1(self.norm(hidden)).gelu())
        hidden = hidden * F.constant(
            self.weights.f16(self.key("1.gamma")).reshape(1, 1, -1),
            "convnext_gamma")
        hidden = hidden.transpose((0, 2, 1))
        return residual + hidden


class DecoderResidualUnit(Module):
    """HiFi-GAN residual unit used inside each decoder upsampling stage."""

    def __init__(self, ctx, prefix: str, dilation: int) -> None:
        super().__init__(ctx, prefix)
        self.act1 = SnakeBeta(ctx, self.key("act1"))
        self.conv1 = CausalConv1d(ctx, self.key("conv1.conv"), dilation)
        self.act2 = SnakeBeta(ctx, self.key("act2"))
        self.conv2 = CausalConv1d(ctx, self.key("conv2.conv"))

    def forward(self, hidden):
        residual = hidden
        hidden = self.conv1(self.act1(hidden))
        hidden = self.conv2(self.act2(hidden))
        return residual + hidden


class QwenCode2WavVocoder(Module):
    """Waveform decoder after the Code2Wav pre-transformer."""

    def __init__(self,
                 ctx,
                 config: Dict,
                 root: str = "decoder",
                 symmetric_padding: bool = False) -> None:
        super().__init__(ctx, root)
        self.upsamples = [
            ConvNextUpsample(ctx, stage, self.key("upsample"),
                             symmetric_padding)
            for stage in range(len(config.get("upsampling_ratios", (2, 2))))
        ]
        self.initial_conv = CausalConv1d(ctx, self.key("decoder.0.conv"))
        self.stage_rates = list(config.get("upsample_rates", (8, 5, 4, 3)))
        self.stage_acts = [
            SnakeBeta(ctx, self.key(f"decoder.{stage}.block.0"))
            for stage in range(1,
                               len(self.stage_rates) + 1)
        ]
        self.stage_deconvs = [
            CausalDeconv1d(ctx, self.key(f"decoder.{stage}.block.1.conv"),
                           int(rate), symmetric_padding)
            for stage, rate in enumerate(self.stage_rates, start=1)
        ]
        self.stage_units = [[
            DecoderResidualUnit(ctx, self.key(f"decoder.{stage}.block.{unit}"),
                                dilation)
            for unit, dilation in zip((2, 3, 4), (1, 3, 9))
        ] for stage in range(1,
                             len(self.stage_rates) + 1)]
        self.final_act = SnakeBeta(ctx, self.key("decoder.5"))
        self.final_conv = CausalConv1d(ctx, self.key("decoder.6.conv"))

    def forward(self, hidden):
        hidden = hidden.transpose((0, 2, 1))
        for upsample in self.upsamples:
            hidden = upsample(hidden)
        hidden = self.initial_conv(hidden)
        for act, deconv, units in zip(self.stage_acts, self.stage_deconvs,
                                      self.stage_units):
            hidden = deconv(act(hidden))
            for unit in units:
                hidden = unit(hidden)
        hidden = self.final_conv(self.final_act(hidden))
        return hidden.maximum(np.float16(-1.0)).minimum(np.float16(1.0))
