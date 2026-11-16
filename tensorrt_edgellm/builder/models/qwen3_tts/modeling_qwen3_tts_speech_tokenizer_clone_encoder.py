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
"""Qwen3-TTS Base speech-tokenizer clone encoder.

This is the provider Mimi encoder, causal transformer, 2x downsampler, and
split residual-vector quantizer. The static 40-second contract matches
``CloneEncoderRunner``:

``wav`` FP32 ``[1, 960000]`` -> ``codes`` INT64 ``[500, 16]``.
"""

import json
import os
from typing import Dict, Tuple

import numpy as np
import tensorrt as trt

from ...ops import LayerNorm, Linear, Module, NetworkModule
from ...ops import functional as F

_CLONE_BUCKET_SECONDS = 40


def _tokenizer_config(ctx) -> Dict:
    config = ctx.bundle.root.get("_speech_tokenizer_config")
    if config is not None:
        return config
    path = os.path.join(ctx.bundle.model_dir, "speech_tokenizer",
                        "config.json")
    with open(path) as config_file:
        return json.load(config_file)


def _replicate_pad(hidden_states, left: int, right: int, length: int):
    pieces = []
    if left:
        first = hidden_states.slice_axis(2, 0, 1, 3)
        pieces.extend([first] * left)
    pieces.append(hidden_states)
    if right:
        last = hidden_states.slice_axis(2, length - 1, 1, 3)
        pieces.extend([last] * right)
    return F.concatenate(pieces, 2)


class Qwen3TTSMimiCausalConv1d(Module):
    """One static-shape provider Mimi causal convolution."""

    def __init__(self,
                 ctx,
                 prefix: str,
                 input_length: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 pad_mode: str = "constant") -> None:
        super().__init__(ctx, prefix)
        self.input_length = input_length
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.pad_mode = pad_mode
        effective_kernel = (kernel_size - 1) * dilation + 1
        self.left_padding = effective_kernel - stride
        self.right_padding = (-input_length) % stride
        self.output_length = (input_length + self.left_padding +
                              self.right_padding -
                              effective_kernel) // stride + 1

    def forward(self, hidden_states):
        pre_padding = self.left_padding
        post_padding = self.right_padding
        if self.pad_mode == "replicate":
            hidden_states = _replicate_pad(hidden_states, pre_padding,
                                           post_padding, self.input_length)
            pre_padding = 0
            post_padding = 0
        elif self.pad_mode != "constant":
            raise ValueError(
                f"unsupported Mimi encoder padding mode {self.pad_mode!r}")
        return F.convolution(hidden_states,
                             self.weights.fp16_parameter(self.key("weight")),
                             self.weights.opt_fp16_parameter(self.key("bias")),
                             stride=(self.stride, ),
                             dilation=(self.dilation, ),
                             groups=self.groups,
                             pre_padding=(pre_padding, ),
                             post_padding=(post_padding, ))


class Qwen3TTSMimiResnetBlock(Module):
    """Provider SEANet residual block."""

    def __init__(self, ctx, prefix: str, config: Dict, channels: int,
                 length: int, dilation: int) -> None:
        super().__init__(ctx, prefix)
        hidden_channels = channels // int(config["compress"])
        self.conv1 = Qwen3TTSMimiCausalConv1d(
            ctx,
            self.key("block.1.conv"),
            length,
            int(config["residual_kernel_size"]),
            dilation=dilation,
            pad_mode=str(config["pad_mode"]))
        self.conv2 = Qwen3TTSMimiCausalConv1d(ctx,
                                              self.key("block.3.conv"),
                                              length,
                                              1,
                                              pad_mode=str(config["pad_mode"]))
        first_shape = self.conv1.weights.parameter_spec(
            self.conv1.key("weight")).shape
        if first_shape[:2] != (hidden_channels, channels):
            raise ValueError("Mimi residual checkpoint channels do not match "
                             "the encoder configuration")

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.conv1(hidden_states.elu())
        hidden_states = self.conv2(hidden_states.elu())
        return residual + hidden_states


class Qwen3TTSMimiEncoder(Module):
    """Provider Mimi SEANet encoder."""

    def __init__(self, ctx, config: Dict, input_length: int) -> None:
        super().__init__(ctx, "encoder.encoder")
        self.config = config
        self.layers = []
        length = input_length
        channels = int(config["num_filters"])
        self.layers.append(
            Qwen3TTSMimiCausalConv1d(ctx,
                                     self.key("layers.0.conv"),
                                     length,
                                     int(config["kernel_size"]),
                                     pad_mode=str(config["pad_mode"])))

        layer_index = 1
        scaling = 1
        for ratio in reversed(
                tuple(int(value) for value in config["upsampling_ratios"])):
            current_channels = scaling * channels
            for residual_index in range(int(config["num_residual_layers"])):
                dilation = int(config["dilation_growth_rate"])**residual_index
                self.layers.append(
                    Qwen3TTSMimiResnetBlock(ctx,
                                            self.key(f"layers.{layer_index}"),
                                            config, current_channels, length,
                                            dilation))
                layer_index += 1
            layer_index += 1  # ELU has no checkpoint state.
            downsample = Qwen3TTSMimiCausalConv1d(
                ctx,
                self.key(f"layers.{layer_index}.conv"),
                length,
                ratio * 2,
                stride=ratio,
                pad_mode=str(config["pad_mode"]))
            self.layers.append(downsample)
            length = downsample.output_length
            scaling *= 2
            layer_index += 1

        layer_index += 1  # Final ELU has no checkpoint state.
        final = Qwen3TTSMimiCausalConv1d(
            ctx,
            self.key(f"layers.{layer_index}.conv"),
            length,
            int(config["last_kernel_size"]),
            pad_mode=str(config["pad_mode"]))
        self.layers.append(final)
        self.output_length = final.output_length

    def forward(self, hidden_states):
        for index, layer in enumerate(self.layers):
            if index and isinstance(layer, Qwen3TTSMimiCausalConv1d):
                hidden_states = hidden_states.elu()
            hidden_states = layer(hidden_states)
        return hidden_states


def _attention_constants(config: Dict,
                         sequence_length: int) -> Tuple[np.ndarray, ...]:
    head_dim = int(config["head_dim"])
    positions = np.arange(sequence_length, dtype=np.float32)[:, None]
    frequencies = 1.0 / (float(config["rope_theta"])**(
        np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angles = positions * frequencies[None, :]
    rotary = np.concatenate((angles, angles), axis=-1)
    cosine = np.cos(rotary).astype(np.float16).reshape(1, 1, sequence_length,
                                                       head_dim)
    sine = np.sin(rotary).astype(np.float16).reshape(1, 1, sequence_length,
                                                     head_dim)

    positions_i64 = np.arange(sequence_length)
    query = positions_i64[:, None]
    key = positions_i64[None, :]
    window = int(config["sliding_window"])
    valid = (key <= query) & (key > query - window)
    mask = np.where(valid, 0.0, np.finfo(np.float16).min).astype(np.float16)
    return cosine, sine, mask.reshape(1, 1, sequence_length, sequence_length)


class Qwen3TTSMimiAttention(Module):
    """Provider Mimi sliding-window causal self-attention."""

    def __init__(self, ctx, prefix: str, config: Dict) -> None:
        super().__init__(ctx, prefix)
        self.num_heads = int(config["num_attention_heads"])
        self.num_key_value_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.hidden_size = int(config["hidden_size"])
        if self.num_heads != self.num_key_value_heads:
            raise ValueError(
                "Qwen3-TTS Mimi clone encoder requires equal Q and KV heads")
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

    def _apply_rope(self, hidden_states, cosine, sine):
        first = hidden_states.slice_last_dim(0, self.head_dim // 2, 4)
        second = hidden_states.slice_last_dim(self.head_dim // 2,
                                              self.head_dim // 2, 4)
        rotated = F.concatenate((-second, first), 3)
        return hidden_states * cosine + rotated * sine

    def forward(self, hidden_states, cosine, sine, mask):
        query = self.q_proj(hidden_states).reshape(
            (0, 0, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        key = self.k_proj(hidden_states).reshape(
            (0, 0, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        value = self.v_proj(hidden_states).reshape(
            (0, 0, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        query = self._apply_rope(query, cosine, sine)
        key = self._apply_rope(key, cosine, sine)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, mask=mask, scale=self.head_dim**-0.5)
        hidden_states = hidden_states.transpose((0, 2, 1, 3)).reshape(
            (0, 0, self.hidden_size))
        return self.o_proj(hidden_states)


class Qwen3TTSMimiMLP(Module):
    """Provider Mimi GELU feed-forward network."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.fc1 = Linear(ctx, self.key("fc1"), rank=3, tensor_parallel=False)
        self.fc2 = Linear(ctx, self.key("fc2"), rank=3, tensor_parallel=False)

    def forward(self, hidden_states):
        return self.fc2(self.fc1(hidden_states).gelu())


class Qwen3TTSMimiTransformerLayer(Module):
    """One provider Mimi pre-norm transformer layer."""

    def __init__(self, ctx, prefix: str, config: Dict) -> None:
        super().__init__(ctx, prefix)
        epsilon = float(config["norm_eps"])
        self.input_layernorm = LayerNorm(ctx,
                                         self.key("input_layernorm"),
                                         epsilon,
                                         rank=3)
        self.self_attn = Qwen3TTSMimiAttention(ctx, self.key("self_attn"),
                                               config)
        self.post_attention_layernorm = LayerNorm(
            ctx, self.key("post_attention_layernorm"), epsilon, rank=3)
        self.mlp = Qwen3TTSMimiMLP(ctx, self.key("mlp"))
        self.attention_scale = self.weights.f16(
            self.key("self_attn_layer_scale.scale")).reshape(1, 1, -1)
        self.mlp_scale = self.weights.f16(
            self.key("mlp_layer_scale.scale")).reshape(1, 1, -1)

    def forward(self, hidden_states, cosine, sine, mask):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states),
                                       cosine, sine, mask)
        hidden_states = residual + hidden_states * F.constant(
            self.attention_scale, "attention_layer_scale")
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states * F.constant(self.mlp_scale,
                                                     "mlp_layer_scale")


class Qwen3TTSMimiTransformer(Module):
    """Provider Mimi encoder transformer."""

    def __init__(self, ctx, config: Dict, sequence_length: int) -> None:
        super().__init__(ctx, "encoder.encoder_transformer")
        self.layers = [
            Qwen3TTSMimiTransformerLayer(ctx, self.key(f"layers.{index}"),
                                         config)
            for index in range(int(config["num_hidden_layers"]))
        ]
        self.cosine, self.sine, self.mask = _attention_constants(
            config, sequence_length)

    def forward(self, hidden_states):
        cosine = F.constant(self.cosine, "rotary_cosine")
        sine = F.constant(self.sine, "rotary_sine")
        mask = F.constant(self.mask, "causal_attention_mask")
        for layer in self.layers:
            hidden_states = layer(hidden_states, cosine, sine, mask)
        return hidden_states


class Qwen3TTSMimiCodebook(Module):
    """One provider Euclidean residual-vector-quantizer codebook."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        embedding_sum = self.weights.f32(self.key("embed_sum"))
        usage = np.maximum(self.weights.f32(self.key("cluster_usage")), 1e-5)
        self.embedding = np.ascontiguousarray(embedding_sum / usage[:, None],
                                              dtype=np.float32)
        self.embedding_norm = np.ascontiguousarray(np.sum(
            self.embedding * self.embedding, axis=1)[None, None, :],
                                                   dtype=np.float32)

    def forward(self, hidden_states):
        embedding = F.constant(self.embedding, "rvq_embedding")
        embedding_matrix = F.constant(self.embedding[None, :, :],
                                      "rvq_embedding_matrix")
        hidden_norm = (hidden_states * hidden_states).sum(2, keepdim=True)
        distances = (
            hidden_norm -
            2.0 * F.matmul(hidden_states, embedding_matrix, transpose_rhs=True)
            + F.constant(self.embedding_norm, "rvq_embedding_norm"))
        _, indices = F.topk(-distances, 1, 2)
        quantized = F.embedding_lookup(embedding, indices).reshape(
            (0, 0, self.embedding.shape[1]))
        codes = indices.reshape((0, 0)).transpose((1, 0))
        return codes.cast(trt.int64), quantized


class Qwen3TTSMimiResidualVectorQuantizer(Module):
    """One semantic or acoustic provider RVQ group."""

    def __init__(self, ctx, prefix: str, num_quantizers: int) -> None:
        super().__init__(ctx, prefix)
        self.input_projection = Qwen3TTSMimiCausalConv1d(
            ctx, self.key("input_proj"), 1, 1, pad_mode="constant")
        self.codebooks = [
            Qwen3TTSMimiCodebook(ctx, self.key(f"layers.{index}.codebook"))
            for index in range(num_quantizers)
        ]

    def forward(self, hidden_states):
        hidden_states = self.input_projection(hidden_states)
        residual = hidden_states.transpose((0, 2, 1)).cast(trt.float32)
        codes = []
        for codebook in self.codebooks:
            indices, quantized = codebook(residual)
            residual = residual - quantized
            codes.append(indices)
        return codes


class Qwen3TTSMimiSplitResidualVectorQuantizer(Module):
    """Provider semantic/acoustic split RVQ."""

    def __init__(self, ctx, config: Dict, num_quantizers: int) -> None:
        super().__init__(ctx, "encoder.quantizer")
        semantic_count = int(config["num_semantic_quantizers"])
        if num_quantizers < semantic_count:
            raise ValueError("valid quantizers cannot exclude semantic RVQ")
        if num_quantizers > int(config["num_quantizers"]):
            raise ValueError("valid quantizers exceed Mimi codebooks")
        self.semantic = Qwen3TTSMimiResidualVectorQuantizer(
            ctx, self.key("semantic_residual_vector_quantizer"),
            semantic_count)
        self.acoustic = Qwen3TTSMimiResidualVectorQuantizer(
            ctx, self.key("acoustic_residual_vector_quantizer"),
            num_quantizers - semantic_count)

    def forward(self, hidden_states):
        codes = self.semantic(hidden_states)
        codes.extend(self.acoustic(hidden_states))
        return F.concatenate(codes, 1)


class Qwen3TTSSpeechTokenizerCloneEncoder(NetworkModule):
    """Raw-waveform Qwen3-TTS Mimi encoder and RVQ engine."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        root_config = _tokenizer_config(ctx)
        self.config = dict(root_config["encoder_config"])
        self.sample_rate = int(root_config["input_sample_rate"])
        self.bucket_samples = self.sample_rate * _CLONE_BUCKET_SECONDS
        self.num_quantizers = int(root_config["encoder_valid_num_quantizers"])

        self.encoder = Qwen3TTSMimiEncoder(ctx, self.config,
                                           self.bucket_samples)
        self.transformer = Qwen3TTSMimiTransformer(ctx, self.config,
                                                   self.encoder.output_length)

        encodec_frame_rate = self.sample_rate
        for ratio in self.config["upsampling_ratios"]:
            encodec_frame_rate /= int(ratio)
        frame_rate = float(self.config["_frame_rate"])
        downsample_ratio = int(encodec_frame_rate / frame_rate)
        self.downsample = Qwen3TTSMimiCausalConv1d(ctx,
                                                   "encoder.downsample.conv",
                                                   self.encoder.output_length,
                                                   2 * downsample_ratio,
                                                   stride=2,
                                                   pad_mode="replicate")
        self.quantizer = Qwen3TTSMimiSplitResidualVectorQuantizer(
            ctx, self.config, self.num_quantizers)
        self.output_frames = self.downsample.output_length

        expected_frames = self.bucket_samples // int(
            root_config["encode_downsample_rate"])
        if self.output_frames != expected_frames:
            raise ValueError(
                f"Mimi bucket produces {self.output_frames} frames, expected "
                f"{expected_frames}")

    def input_tensors(self):
        return {
            "wav": self.add_input("wav", trt.float32, (1, self.bucket_samples))
        }

    def forward(self, wav):
        hidden_states = self.encoder(
            wav.cast(trt.float16).reshape((1, 1, self.bucket_samples)))
        hidden_states = self.transformer(hidden_states.transpose(
            (0, 2, 1))).transpose((0, 2, 1))
        hidden_states = self.downsample(hidden_states)
        return {"codes": self.quantizer(hidden_states)}
