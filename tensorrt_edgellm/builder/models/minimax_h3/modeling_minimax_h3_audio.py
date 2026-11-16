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
"""Checkpoint-direct MiniMax-H3 audio VAE decoder.

BigVGAN decoder with alias-free SnakeBeta activations, evaluated in FP32 as in
Diffusers. The two stereo channels are the batch. Latent normalization is
folded in: the input is the GEN audio latent and the output is the clamped
waveform.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from ...ops import Module, NetworkModule
from ...ops import functional as F
from . import configuration

_RESAMPLE_RATIO = 2
_RESAMPLE_KERNEL = 12


def _weight_norm(weights, prefix: str) -> np.ndarray:
    """Fold ``weight_g * weight_v / ||weight_v||`` over all but dim 0."""
    direction = weights.f32(prefix + ".weight_v").astype(np.float64)
    norm = np.sqrt((direction**2).sum(axis=tuple(range(1, direction.ndim)),
                                      keepdims=True))
    return (weights.f32(prefix + ".weight_g") * direction / norm).astype(
        np.float32)


def _conv(hidden,
          weight,
          bias=None,
          padding: int = 0,
          dilation: int = 1,
          stride: int = 1,
          groups: int = 1):
    return F.convolution(hidden,
                         weight,
                         bias,
                         stride=(stride, ),
                         padding=(padding, ),
                         dilation=(dilation, ),
                         groups=groups,
                         dtype=np.float32)


def _length(hidden):
    return F.shape_of(hidden)[2:3]


def _replicate_pad(hidden, left: int, right: int):
    channels = int(hidden.shape[1])
    first = hidden.slice_axis(2, 0, 1, 3)
    last = F.dynamic_slice(hidden, (0, 0, _length(hidden) - 1),
                           (2, channels, 1))
    return F.concatenate([first] * left + [hidden] + [last] * right, 2)


class AliasFreeSnakeBeta(Module):
    """Upsample by two, apply SnakeBeta, and downsample by two."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.alpha = np.exp(self.weights.f32(self.key("act.alpha")))
        self.inverse_beta = 1.0 / (
            np.exp(self.weights.f32(self.key("act.beta"))) + 1e-9)
        self.channels = self.alpha.size
        self.up_filter = self.weights.f32(self.key("upsample.filter"))
        self.down_filter = self.weights.f32(
            self.key("downsample.lowpass.filter"))

    def forward(self, hidden):
        depthwise = (self.channels, 1, 1)
        pad = _RESAMPLE_KERNEL // _RESAMPLE_RATIO - 1
        crop = pad * _RESAMPLE_RATIO + (_RESAMPLE_KERNEL -
                                        _RESAMPLE_RATIO) // 2
        upsampled = F.deconvolution(_replicate_pad(hidden, pad, pad),
                                    np.tile(self.up_filter, depthwise),
                                    stride=(_RESAMPLE_RATIO, ),
                                    groups=self.channels,
                                    dtype=np.float32) * float(_RESAMPLE_RATIO)
        upsampled = F.dynamic_slice(
            upsampled, (0, 0, crop),
            (2, self.channels, _length(upsampled) - 2 * crop))
        periodic = (
            upsampled *
            F.constant(self.alpha.reshape(1, -1, 1), self.key("alpha"))).sin()
        activated = upsampled + F.constant(self.inverse_beta.reshape(
            1, -1, 1), self.key("inverse_beta")) * periodic * periodic
        return _conv(_replicate_pad(activated, pad, pad + 1),
                     np.tile(self.down_filter, depthwise),
                     stride=_RESAMPLE_RATIO,
                     groups=self.channels)


class AMPBlock(Module):
    """BigVGAN residual block of dilated and undilated convolution pairs."""

    def __init__(self, ctx, prefix: str, kernel: int, dilations) -> None:
        super().__init__(ctx, prefix)
        self.kernel = kernel
        self.dilations = tuple(int(dilation) for dilation in dilations)
        self.activations = [
            AliasFreeSnakeBeta(ctx, self.key(f"activations.{index}"))
            for index in range(2 * len(self.dilations))
        ]

    def _conv(self, name: str, index: int, hidden, dilation: int):
        prefix = self.key(f"{name}.{index}")
        return _conv(hidden,
                     _weight_norm(self.weights, prefix),
                     self.weights.f32(prefix + ".bias"),
                     padding=(self.kernel - 1) * dilation // 2,
                     dilation=dilation)

    def forward(self, hidden):
        for index, dilation in enumerate(self.dilations):
            residual = self._conv("convs1", index,
                                  self.activations[2 * index](hidden),
                                  dilation)
            residual = self._conv("convs2", index,
                                  self.activations[2 * index + 1](residual), 1)
            hidden = hidden + residual
        return hidden


class MiniMaxH3AudioDecoder(NetworkModule):
    """Decode ``[2, C, L]`` stereo latents to a ``[2, L * hop]`` waveform."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle)

    def __init__(self, ctx, bundle) -> None:
        super().__init__(ctx)
        config = configuration.component_config(
            bundle.root, configuration.Component.AUDIO_DECODER)
        self.config = config
        self.rates = tuple(int(rate) for rate in config["decoder_rates"])
        self.kernels = tuple(
            int(kernel) for kernel in config["decoder_kernel_sizes"])
        resblock_kernels = config["resblock_kernel_sizes"]
        self.resblocks = [[
            AMPBlock(
                ctx,
                f"decoder.resblocks.{stage * len(resblock_kernels) + index}",
                int(kernel), dilations)
            for index, (kernel, dilations) in enumerate(
                zip(resblock_kernels, config["resblock_dilation_sizes"]))
        ] for stage in range(len(self.rates))]
        self.activation_post = AliasFreeSnakeBeta(ctx,
                                                  "decoder.activation_post")

    def input_tensors(self):
        return {
            "audio_latents":
            self.add_input("audio_latents", trt.float32,
                           (2, int(self.config["latent_channels"]), -1))
        }

    def forward(self, audio_latents):
        weights = self.weights
        mean, std = (np.asarray(self.config[name],
                                np.float32).reshape(1, -1, 1)
                     for name in ("latents_mean", "latents_std"))
        hidden = (audio_latents * F.constant(std, "latents_std") +
                  F.constant(mean, "latents_mean"))
        hidden = _conv(hidden, weights.f32("dec_in_proj.weight"),
                       weights.f32("dec_in_proj.bias"))
        hidden = _conv(hidden,
                       _weight_norm(weights, "decoder.conv_pre"),
                       weights.f32("decoder.conv_pre.bias"),
                       padding=3)
        for stage, (rate, kernel) in enumerate(zip(self.rates, self.kernels)):
            prefix = f"decoder.ups.{stage}.0"
            hidden = F.deconvolution(hidden,
                                     _weight_norm(weights, prefix),
                                     weights.f32(prefix + ".bias"),
                                     stride=(rate, ),
                                     padding=((kernel - rate) // 2, ),
                                     dtype=np.float32)
            blocks = self.resblocks[stage]
            total = blocks[0](hidden)
            for block in blocks[1:]:
                total = total + block(hidden)
            hidden = total / float(len(blocks))
        hidden = _conv(self.activation_post(hidden),
                       _weight_norm(weights, "decoder.conv_post"),
                       padding=3)
        waveform = hidden.maximum(-1.0).minimum(1.0)
        return {"waveform": waveform.reshape((2, -1))}
