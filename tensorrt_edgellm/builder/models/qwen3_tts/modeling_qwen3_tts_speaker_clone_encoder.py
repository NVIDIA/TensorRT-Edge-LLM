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
"""Qwen3-TTS Base speaker-clone encoder.

The network follows the provider's raw-waveform preprocessing and ECAPA-TDNN
speaker encoder. Its engine contract is the one consumed by
``CloneEncoderRunner``:

``wav`` FP32 ``[1, T]`` -> ``speaker_embedding`` FP32 ``[1, enc_dim]``.
"""

import math
from typing import Dict, Sequence

import numpy as np
import tensorrt as trt

from ...ops import Module, NetworkModule
from ...ops import functional as F

_SAMPLE_RATE = 24000
_N_FFT = 1024
_HOP_LENGTH = 256
_STFT_PAD = (_N_FFT - _HOP_LENGTH) // 2

_SPEAKER_DEFAULTS = {
    "mel_dim": 128,
    "enc_dim": 1024,
    "enc_channels": (512, 512, 512, 512, 1536),
    "enc_kernel_sizes": (5, 3, 3, 3, 1),
    "enc_dilations": (1, 2, 3, 4, 1),
    "enc_attention_channels": 128,
    "enc_res2net_scale": 8,
    "enc_se_channels": 128,
    "sample_rate": _SAMPLE_RATE,
}


def _speaker_config(root: Dict) -> Dict:
    config = dict(_SPEAKER_DEFAULTS)
    config.update(root.get("speaker_encoder_config") or {})
    for name in ("enc_channels", "enc_kernel_sizes", "enc_dilations"):
        config[name] = tuple(int(value) for value in config[name])
    if not (len(config["enc_channels"]) == len(config["enc_kernel_sizes"]) ==
            len(config["enc_dilations"])):
        raise ValueError("Qwen3-TTS speaker encoder channel, kernel, and "
                         "dilation lists must have the same length")
    if int(config["sample_rate"]) != _SAMPLE_RATE:
        raise ValueError("CloneEncoderRunner requires 24 kHz reference audio")
    return config


def _reflect_pad_last_dim(hidden_states, padding: int):
    """Compose PyTorch reflect padding from dynamic slices."""
    if padding == 0:
        return hidden_states
    shape = F.shape_of(hidden_states)
    size = (shape[0:1], shape[1:2], padding)
    left = F.dynamic_slice(hidden_states, (0, 0, padding),
                           size,
                           stride=(1, 1, -1))
    right = F.dynamic_slice(hidden_states, (0, 0, shape[2:3] - 2),
                            size,
                            stride=(1, 1, -1))
    return F.concatenate((left, hidden_states, right), 2)


def _slaney_mel_filterbank(num_mels: int) -> np.ndarray:
    """Return the librosa-compatible Slaney-normalized mel filterbank."""

    def hz_to_mel(frequency):
        frequency = np.asarray(frequency, dtype=np.float64)
        linear = 3.0 * frequency / 200.0
        logarithmic = 15.0 + np.log(
            np.maximum(frequency, 1e-10) / 1000.0) / (math.log(6.4) / 27.0)
        return np.where(frequency >= 1000.0, logarithmic, linear)

    def mel_to_hz(mel):
        mel = np.asarray(mel, dtype=np.float64)
        linear = 200.0 * mel / 3.0
        logarithmic = 1000.0 * np.exp((math.log(6.4) / 27.0) * (mel - 15.0))
        return np.where(mel >= 15.0, logarithmic, linear)

    fft_frequencies = (np.arange(_N_FFT // 2 + 1, dtype=np.float64) *
                       _SAMPLE_RATE / _N_FFT)
    mel_points = mel_to_hz(
        np.linspace(hz_to_mel(0.0), hz_to_mel(_SAMPLE_RATE / 2.0),
                    num_mels + 2))
    frequency_delta = mel_points[1:] - mel_points[:-1]
    ramps = mel_points[:, None] - fft_frequencies[None, :]
    lower = -ramps[:-2] / frequency_delta[:-1, None]
    upper = ramps[2:] / frequency_delta[1:, None]
    weights = np.maximum(np.minimum(lower, upper), 0.0)
    normalization = 2.0 / (mel_points[2:num_mels + 2] - mel_points[:num_mels])
    return np.ascontiguousarray(weights * normalization[:, None],
                                dtype=np.float32)


def _dft_kernels() -> np.ndarray:
    samples = np.arange(_N_FFT, dtype=np.float64)
    bins = np.arange(_N_FFT // 2 + 1, dtype=np.float64)
    angles = 2.0 * math.pi * bins[:, None] * samples[None, :] / _N_FFT
    window = 0.5 - 0.5 * np.cos(2.0 * math.pi * samples / _N_FFT)
    basis = np.concatenate((np.cos(angles), -np.sin(angles)), axis=0)
    return np.ascontiguousarray((basis * window[None, :])[:, None, :],
                                dtype=np.float32)


class Qwen3TTSMelFrontend(Module):
    """Provider-equivalent 24 kHz log-mel preprocessing."""

    def __init__(self, ctx, num_mels: int) -> None:
        super().__init__(ctx)
        self.num_mels = num_mels
        self.dft = _dft_kernels()
        self.mel_filterbank = _slaney_mel_filterbank(num_mels)

    def forward(self, wav):
        wav_shape = F.shape_of(wav)
        hidden_states = F.dynamic_reshape(wav.cast(trt.float16),
                                          (wav_shape[0:1], 1, wav_shape[1:2]))
        hidden_states = _reflect_pad_last_dim(hidden_states, _STFT_PAD)
        spectrum = F.convolution(hidden_states,
                                 self.dft,
                                 stride=(_HOP_LENGTH, ))
        num_bins = _N_FFT // 2 + 1
        real = spectrum.slice_axis(1, 0, num_bins, 3).cast(trt.float32)
        imaginary = spectrum.slice_axis(1, num_bins, num_bins,
                                        3).cast(trt.float32)
        magnitude = (real * real + imaginary * imaginary + 1e-9).sqrt()
        mel = F.matmul(
            F.constant(self.mel_filterbank[None, :, :], "mel_filterbank"),
            magnitude)
        mel = mel.maximum(1e-5).log()
        return mel.transpose((0, 2, 1)).cast(trt.float16)


class Qwen3TTSTimeDelayNetBlock(Module):
    """One provider ECAPA time-delay convolution and ReLU."""

    def __init__(self, ctx, prefix: str, kernel_size: int,
                 dilation: int) -> None:
        super().__init__(ctx, prefix)
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, hidden_states):
        effective_kernel = (self.kernel_size - 1) * self.dilation + 1
        if effective_kernel % 2 != 1:
            raise ValueError("ECAPA same-padding convolution must be odd")
        hidden_states = _reflect_pad_last_dim(hidden_states,
                                              effective_kernel // 2)
        hidden_states = F.convolution(
            hidden_states,
            self.weights.fp16_parameter(self.key("conv.weight")),
            self.weights.opt_fp16_parameter(self.key("conv.bias")),
            dilation=(self.dilation, ))
        return hidden_states.relu()


class Qwen3TTSRes2NetBlock(Module):
    """Channel-split residual hierarchy used inside ECAPA."""

    def __init__(self, ctx, prefix: str, channels: int, scale: int,
                 kernel_size: int, dilation: int) -> None:
        super().__init__(ctx, prefix)
        if channels % scale:
            raise ValueError("ECAPA Res2Net channels must divide its scale")
        self.width = channels // scale
        self.scale = scale
        self.blocks = [
            Qwen3TTSTimeDelayNetBlock(ctx, self.key(f"blocks.{index}"),
                                      kernel_size, dilation)
            for index in range(scale - 1)
        ]

    def forward(self, hidden_states):
        outputs = []
        previous = None
        for index in range(self.scale):
            part = hidden_states.slice_axis(1, index * self.width, self.width,
                                            3)
            if index == 0:
                output = part
            elif index == 1:
                output = self.blocks[index - 1](part)
            else:
                output = self.blocks[index - 1](part + previous)
            outputs.append(output)
            previous = output
        return F.concatenate(outputs, 1)


class Qwen3TTSSqueezeExcitation(Module):
    """ECAPA squeeze-excitation gate."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)

    def _conv(self, hidden_states, name: str):
        return F.convolution(
            hidden_states,
            self.weights.fp16_parameter(self.key(f"{name}.weight")),
            self.weights.opt_fp16_parameter(self.key(f"{name}.bias")))

    def forward(self, hidden_states):
        pooled = hidden_states.mean(2, keepdim=True)
        gate = self._conv(pooled, "conv1").relu()
        gate = self._conv(gate, "conv2").sigmoid()
        return hidden_states * gate


class Qwen3TTSSqueezeExcitationRes2NetBlock(Module):
    """TDNN-Res2Net-TDNN-SE residual block."""

    def __init__(self, ctx, prefix: str, channels: int, scale: int,
                 kernel_size: int, dilation: int) -> None:
        super().__init__(ctx, prefix)
        self.tdnn1 = Qwen3TTSTimeDelayNetBlock(ctx, self.key("tdnn1"), 1, 1)
        self.res2net = Qwen3TTSRes2NetBlock(ctx, self.key("res2net_block"),
                                            channels, scale, kernel_size,
                                            dilation)
        self.tdnn2 = Qwen3TTSTimeDelayNetBlock(ctx, self.key("tdnn2"), 1, 1)
        self.se = Qwen3TTSSqueezeExcitation(ctx, self.key("se_block"))

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.tdnn1(hidden_states)
        hidden_states = self.res2net(hidden_states)
        hidden_states = self.tdnn2(hidden_states)
        return self.se(hidden_states) + residual


class Qwen3TTSAttentiveStatisticsPooling(Module):
    """Provider ECAPA attentive mean and standard-deviation pooling."""

    def __init__(self, ctx, prefix: str, channels: int,
                 attention_channels: int) -> None:
        super().__init__(ctx, prefix)
        del attention_channels
        self.channels = channels
        self.eps = 1e-12
        self.tdnn = Qwen3TTSTimeDelayNetBlock(ctx, self.key("tdnn"), 1, 1)

    def _attention_conv(self, hidden_states):
        return F.convolution(
            hidden_states,
            self.weights.fp16_parameter(self.key("conv.weight")),
            self.weights.opt_fp16_parameter(self.key("conv.bias")))

    def _statistics(self, hidden_states, weights):
        mean = (weights * hidden_states).sum(2, keepdim=True)
        centered = hidden_states - mean
        variance = (weights * centered * centered).sum(2, keepdim=True)
        return mean, variance.maximum(self.eps).sqrt()

    def forward(self, hidden_states):
        uniform_mean = hidden_states.mean(2, keepdim=True)
        centered = hidden_states - uniform_mean
        uniform_variance = (centered * centered).mean(2, keepdim=True)
        uniform_std = uniform_variance.maximum(self.eps).sqrt()
        expanded_mean = hidden_states * 0.0 + uniform_mean
        expanded_std = hidden_states * 0.0 + uniform_std
        attention = F.concatenate((hidden_states, expanded_mean, expanded_std),
                                  1)
        attention = self._attention_conv(self.tdnn(attention).tanh())
        attention = attention.softmax(2)
        mean, std = self._statistics(hidden_states, attention)
        return F.concatenate((mean, std), 1)


class Qwen3TTSECAPASpeakerEncoder(Module):
    """Qwen3-TTS provider ECAPA-TDNN hierarchy."""

    def __init__(self, ctx, config: Dict) -> None:
        super().__init__(ctx, "speaker_encoder")
        channels: Sequence[int] = config["enc_channels"]
        kernels: Sequence[int] = config["enc_kernel_sizes"]
        dilations: Sequence[int] = config["enc_dilations"]
        self.blocks = [
            Qwen3TTSTimeDelayNetBlock(ctx, self.key("blocks.0"), kernels[0],
                                      dilations[0])
        ]
        for index in range(1, len(channels) - 1):
            self.blocks.append(
                Qwen3TTSSqueezeExcitationRes2NetBlock(
                    ctx, self.key(f"blocks.{index}"), channels[index],
                    int(config["enc_res2net_scale"]), kernels[index],
                    dilations[index]))
        self.mfa = Qwen3TTSTimeDelayNetBlock(ctx, self.key("mfa"), kernels[-1],
                                             dilations[-1])
        self.asp = Qwen3TTSAttentiveStatisticsPooling(
            ctx, self.key("asp"), channels[-1],
            int(config["enc_attention_channels"]))

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose((0, 2, 1))
        intermediate = []
        for block in self.blocks:
            hidden_states = block(hidden_states)
            intermediate.append(hidden_states)
        hidden_states = self.mfa(F.concatenate(intermediate[1:], 1))
        hidden_states = self.asp(hidden_states)
        hidden_states = F.convolution(
            hidden_states, self.weights.fp16_parameter(self.key("fc.weight")),
            self.weights.opt_fp16_parameter(self.key("fc.bias")))
        return hidden_states.reshape((0, -1))


class Qwen3TTSSpeakerCloneEncoder(NetworkModule):
    """Raw-waveform Qwen3-TTS speaker-clone engine."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.config = _speaker_config(ctx.bundle.root)
        self.mel_frontend = Qwen3TTSMelFrontend(ctx,
                                                int(self.config["mel_dim"]))
        self.speaker_encoder = Qwen3TTSECAPASpeakerEncoder(ctx, self.config)

    def input_tensors(self):
        return {"wav": self.add_input("wav", trt.float32, (1, -1))}

    def forward(self, wav):
        embedding = self.speaker_encoder(self.mel_frontend(wav))
        return {"speaker_embedding": embedding.cast(trt.float32)}
