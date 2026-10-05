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
"""Checkpoint-direct Wan VAE encoder used by Cosmos3 policy inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import tensorrt as trt

from ...ops import Module, NetworkModule
from ...ops import functional as F
from .configuration import Cosmos3PolicyGeometry

_CACHE_FRAMES = 2


@dataclass
class _FeatureCache:
    """Graph-construction cursor for the reference causal-conv cache slots."""

    slots: list = field(default_factory=list)
    index: int = 0

    def reset(self) -> None:
        self.index = 0

    def exchange(self, value):
        index = self.index
        self.index += 1
        if index == len(self.slots):
            self.slots.append(value)
            return None
        previous = self.slots[index]
        self.slots[index] = value
        return previous


class Cosmos3CausalConv3d(Module):
    """Checkpoint-backed 3D convolution with causal temporal padding."""

    def __init__(self,
                 ctx,
                 prefix: str,
                 *,
                 stride=(1, 1, 1),
                 padding=(0, 0, 0)) -> None:
        super().__init__(ctx, prefix)
        self.stride = tuple(int(value) for value in stride)
        self.padding = tuple(int(value) for value in padding)
        self.weight = self.weights.fp16_parameter(self.key("weight"))
        self.bias = self.weights.opt_fp16_parameter(self.key("bias"))

    def forward(self, hidden_states, cached=None):
        temporal_padding = 2 * self.padding[0]
        if cached is not None:
            hidden_states = F.concatenate((cached, hidden_states), 2)
            temporal_padding -= int(cached.shape[2])
        if temporal_padding < 0:
            raise ValueError("Cosmos3 causal-conv cache exceeds its padding")
        return F.convolution(
            hidden_states,
            self.weight,
            self.bias,
            stride=self.stride,
            pre_padding=(temporal_padding, self.padding[1], self.padding[2]),
            post_padding=(0, self.padding[1], self.padding[2]),
        )


class Cosmos3Conv2d(Module):
    """Checkpoint-backed 2D convolution."""

    def __init__(
        self,
        ctx,
        prefix: str,
        *,
        stride=(1, 1),
        pre_padding=(0, 0),
        post_padding=(0, 0)) -> None:
        super().__init__(ctx, prefix)
        self.stride = stride
        self.pre_padding = pre_padding
        self.post_padding = post_padding
        self.weight = self.weights.fp16_parameter(self.key("weight"))
        self.bias = self.weights.opt_fp16_parameter(self.key("bias"))

    def forward(self, hidden_states):
        return F.convolution(hidden_states,
                             self.weight,
                             self.bias,
                             stride=self.stride,
                             pre_padding=self.pre_padding,
                             post_padding=self.post_padding)


class Cosmos3ChannelRMSNorm(Module):
    """RMS normalization over channels, matching the Wan reference layout."""

    def __init__(self, ctx, prefix: str, video: bool) -> None:
        super().__init__(ctx, prefix)
        self.video = video
        self.gamma = self.weights.f16(self.key("gamma")).reshape(-1)
        self.eps = 1e-24 / self.gamma.size

    def forward(self, hidden_states):
        if self.video:
            hidden_states = hidden_states.transpose((0, 2, 3, 4, 1))
            normalized = F.rms_norm(hidden_states,
                                    self.gamma,
                                    self.eps,
                                    rank=5)
            return normalized.transpose((0, 4, 1, 2, 3))
        hidden_states = hidden_states.transpose((0, 2, 3, 1))
        normalized = F.rms_norm(hidden_states, self.gamma, self.eps, rank=4)
        return normalized.transpose((0, 3, 1, 2))


def _tail(hidden_states, frames: int):
    size = min(frames, int(hidden_states.shape[2]))
    start = int(hidden_states.shape[2]) - size
    return hidden_states.slice_axis(2, start, size, 5)


def _cached_conv(module: Cosmos3CausalConv3d, hidden_states,
                 cache: _FeatureCache):
    cached_tail = _tail(hidden_states, _CACHE_FRAMES)
    previous = cache.exchange(cached_tail)
    if int(cached_tail.shape[2]) < _CACHE_FRAMES and previous is not None:
        cached_tail = F.concatenate((_tail(previous, 1), cached_tail), 2)
        cache.slots[cache.index - 1] = cached_tail
    return module(hidden_states, previous)


class Cosmos3VaeResidualBlock(Module):
    """Channel norm, SiLU, and causal convolution residual block."""

    def __init__(self, ctx, prefix: str, in_channels: int,
                 out_channels: int) -> None:
        super().__init__(ctx, prefix)
        self.norm1 = Cosmos3ChannelRMSNorm(ctx, self.key("norm1"), video=True)
        self.conv1 = Cosmos3CausalConv3d(ctx,
                                         self.key("conv1"),
                                         padding=(1, 1, 1))
        self.norm2 = Cosmos3ChannelRMSNorm(ctx, self.key("norm2"), video=True)
        self.conv2 = Cosmos3CausalConv3d(ctx,
                                         self.key("conv2"),
                                         padding=(1, 1, 1))
        self.shortcut = (Cosmos3CausalConv3d(ctx, self.key("conv_shortcut"))
                         if in_channels != out_channels else None)

    def forward(self, hidden_states, cache):
        residual = (self.shortcut(hidden_states)
                    if self.shortcut is not None else hidden_states)
        hidden_states = _cached_conv(self.conv1,
                                     self.norm1(hidden_states).silu(), cache)
        hidden_states = _cached_conv(self.conv2,
                                     self.norm2(hidden_states).silu(), cache)
        return hidden_states + residual


class Cosmos3AverageDownsample(Module):
    """Parameter-free temporal/spatial average residual projection."""

    def __init__(self, ctx, in_channels: int, out_channels: int,
                 temporal_factor: int, spatial_factor: int) -> None:
        super().__init__(ctx)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temporal_factor = temporal_factor
        self.spatial_factor = spatial_factor
        total = in_channels * temporal_factor * spatial_factor**2
        if total % out_channels:
            raise ValueError("Cosmos3 average shortcut channels do not divide")
        self.group_size = total // out_channels

    def forward(self, hidden_states):
        time = int(hidden_states.shape[2])
        height = int(hidden_states.shape[3])
        width = int(hidden_states.shape[4])
        spatial = self.spatial_factor
        if height % spatial or width % spatial:
            raise ValueError("Cosmos3 VAE downsample input is not aligned")
        temporal_padding = (-time) % self.temporal_factor
        if temporal_padding:
            zero = hidden_states.slice_axis(2, 0, 1, 5) * np.float16(0.0)
            hidden_states = F.concatenate((zero, hidden_states), 2)
            time += temporal_padding
        hidden_states = hidden_states.reshape(
            (0, self.in_channels, time // self.temporal_factor,
             self.temporal_factor, height // spatial, spatial,
             width // spatial, spatial))
        hidden_states = hidden_states.transpose((0, 1, 3, 5, 7, 2, 4, 6))
        hidden_states = hidden_states.reshape(
            (0, self.out_channels, self.group_size,
             time // self.temporal_factor, height // spatial,
             width // spatial))
        return hidden_states.mean(dim=2)


class Cosmos3VaeDownsample(Module):
    """Spatial convolution and optional causal temporal downsampling."""

    def __init__(self, ctx, prefix: str, channels: int,
                 temporal: bool) -> None:
        super().__init__(ctx, prefix)
        self.channels = channels
        self.spatial = Cosmos3Conv2d(
            ctx,
            self.key("resample.1"),
            stride=(2, 2),
            post_padding=(1, 1),
        )
        self.temporal = (Cosmos3CausalConv3d(
            ctx, self.key("time_conv"), stride=(2, 1,
                                                1)) if temporal else None)

    def forward(self, hidden_states, cache):
        time = int(hidden_states.shape[2])
        height = int(hidden_states.shape[3])
        width = int(hidden_states.shape[4])
        flattened = hidden_states.transpose((0, 2, 1, 3, 4)).reshape(
            (-1, self.channels, height, width))
        flattened = self.spatial(flattened)
        down_height = int(flattened.shape[2])
        down_width = int(flattened.shape[3])
        hidden_states = flattened.reshape(
            (-1, time, self.channels, down_height, down_width)).transpose(
                (0, 2, 1, 3, 4))
        if self.temporal is None:
            return hidden_states
        previous = cache.exchange(_tail(hidden_states, 1))
        if previous is None:
            return hidden_states
        temporal_input = F.concatenate((_tail(previous, 1), hidden_states), 2)
        return self.temporal(temporal_input)


class Cosmos3VaeResidualDownBlock(Module):
    """Wan residual down block and average-pooled shortcut."""

    def __init__(self, ctx, prefix: str, in_channels: int, out_channels: int,
                 num_res_blocks: int, temporal: bool,
                 downsample: bool) -> None:
        super().__init__(ctx, prefix)
        self.shortcut = Cosmos3AverageDownsample(ctx, in_channels,
                                                 out_channels,
                                                 2 if temporal else 1,
                                                 2 if downsample else 1)
        channels = in_channels
        self.resnets = []
        for index in range(num_res_blocks):
            self.resnets.append(
                Cosmos3VaeResidualBlock(ctx, self.key(f"resnets.{index}"),
                                        channels, out_channels))
            channels = out_channels
        self.downsampler = (Cosmos3VaeDownsample(
            ctx, self.key("downsampler"), out_channels, temporal=temporal)
                            if downsample else None)

    def forward(self, hidden_states, cache):
        residual = self.shortcut(hidden_states)
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, cache)
        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, cache)
        return hidden_states + residual


class Cosmos3VaeAttention(Module):
    """Single-head spatial attention in the Wan encoder mid block."""

    def __init__(self, ctx, prefix: str, channels: int) -> None:
        super().__init__(ctx, prefix)
        self.channels = channels
        self.norm = Cosmos3ChannelRMSNorm(ctx, self.key("norm"), video=False)
        self.to_qkv = Cosmos3Conv2d(ctx, self.key("to_qkv"))
        self.proj = Cosmos3Conv2d(ctx, self.key("proj"))

    def forward(self, hidden_states):
        identity = hidden_states
        time = int(hidden_states.shape[2])
        height = int(hidden_states.shape[3])
        width = int(hidden_states.shape[4])
        spatial_tokens = height * width
        frames = hidden_states.transpose((0, 2, 1, 3, 4)).reshape(
            (-1, self.channels, height, width))
        qkv = self.to_qkv(self.norm(frames)).reshape(
            (-1, self.channels * 3, spatial_tokens)).transpose((0, 2, 1))
        query = qkv[..., :self.channels].reshape(
            (-1, 1, spatial_tokens, self.channels))
        key = qkv[..., self.channels:self.channels * 2].reshape(
            (-1, 1, spatial_tokens, self.channels))
        value = qkv[..., self.channels * 2:].reshape(
            (-1, 1, spatial_tokens, self.channels))
        attended = F.scaled_dot_product_attention(query,
                                                  key,
                                                  value,
                                                  scale=self.channels**-0.5)
        attended = attended.reshape(
            (-1, spatial_tokens, self.channels)).transpose((0, 2, 1)).reshape(
                (-1, self.channels, height, width))
        attended = self.proj(attended).reshape(
            (-1, time, self.channels, height, width)).transpose(
                (0, 2, 1, 3, 4))
        return attended + identity


class Cosmos3VaeMidBlock(Module):
    """Residual, spatial-attention, residual bottleneck."""

    def __init__(self, ctx, prefix: str, channels: int) -> None:
        super().__init__(ctx, prefix)
        self.resnet_0 = Cosmos3VaeResidualBlock(ctx, self.key("resnets.0"),
                                                channels, channels)
        self.attention = Cosmos3VaeAttention(ctx, self.key("attentions.0"),
                                             channels)
        self.resnet_1 = Cosmos3VaeResidualBlock(ctx, self.key("resnets.1"),
                                                channels, channels)

    def forward(self, hidden_states, cache):
        hidden_states = self.resnet_0(hidden_states, cache)
        hidden_states = self.attention(hidden_states)
        return self.resnet_1(hidden_states, cache)


class Cosmos3WanEncoder(Module):
    """Wan 2.2 causal encoder tower."""

    def __init__(self, ctx, config: dict) -> None:
        super().__init__(ctx, "encoder")
        base = int(config["base_dim"])
        dimensions = [base] + [
            base * int(multiplier) for multiplier in config["dim_mult"]
        ]
        self.conv_in = Cosmos3CausalConv3d(ctx,
                                           self.key("conv_in"),
                                           padding=(1, 1, 1))
        self.down_blocks = []
        temporal = list(config["temperal_downsample"])
        last = len(config["dim_mult"]) - 1
        for index, (in_channels, out_channels) in enumerate(
                zip(dimensions[:-1], dimensions[1:])):
            self.down_blocks.append(
                Cosmos3VaeResidualDownBlock(
                    ctx,
                    self.key(f"down_blocks.{index}"),
                    in_channels,
                    out_channels,
                    int(config["num_res_blocks"]),
                    temporal=bool(temporal[index]) if index != last else False,
                    downsample=index != last,
                ))
        channels = dimensions[-1]
        self.mid_block = Cosmos3VaeMidBlock(ctx, self.key("mid_block"),
                                            channels)
        self.norm_out = Cosmos3ChannelRMSNorm(ctx,
                                              self.key("norm_out"),
                                              video=True)
        self.conv_out = Cosmos3CausalConv3d(ctx,
                                            self.key("conv_out"),
                                            padding=(1, 1, 1))

    def forward(self, hidden_states, cache):
        hidden_states = _cached_conv(self.conv_in, hidden_states, cache)
        for block in self.down_blocks:
            hidden_states = block(hidden_states, cache)
        hidden_states = self.mid_block(hidden_states, cache)
        hidden_states = self.norm_out(hidden_states).silu()
        return _cached_conv(self.conv_out, hidden_states, cache)


class Cosmos3VaeEncoder(NetworkModule):
    """Full conditioning-frame VAE encoder with a fixed policy geometry."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle)

    def __init__(self, ctx, bundle) -> None:
        super().__init__(ctx)
        self.config = dict(bundle.root["_direct_vae_config"])
        self.geometry = Cosmos3PolicyGeometry.from_bundle(bundle, ctx.args)
        if not self.config.get("is_residual", False):
            raise ValueError("Cosmos3 requires the residual Wan VAE")
        if self.config.get("attn_scales"):
            raise ValueError(
                "Cosmos3 VAE per-scale attention is not supported")
        self.patch_size = int(self.config["patch_size"])
        self.z_dim = int(self.config["z_dim"])
        expected_channels = 3 * self.patch_size**2
        if int(self.config["in_channels"]) != expected_channels:
            raise ValueError(
                "Cosmos3 VAE input channels do not match pixel patchification")
        if self.geometry.height % self.patch_size:
            raise ValueError("Cosmos3 VAE height is not patch-aligned")
        if self.geometry.width % self.patch_size:
            raise ValueError("Cosmos3 VAE width is not patch-aligned")
        self.encoder = Cosmos3WanEncoder(ctx, self.config)
        self.quant_conv = Cosmos3CausalConv3d(ctx, "quant_conv")
        self.latents_mean = np.asarray(self.config["latents_mean"],
                                       dtype=np.float16).reshape(
                                           1, self.z_dim, 1, 1, 1)
        self.latents_std = np.asarray(self.config["latents_std"],
                                      dtype=np.float16).reshape(
                                          1, self.z_dim, 1, 1, 1)

    def input_tensors(self):
        return {
            "pixel_values":
            self.add_input("pixel_values", trt.float32,
                           (-1, 3, self.geometry.conditioning_frames,
                            self.geometry.height, self.geometry.width))
        }

    def _patchify(self, pixel_values):
        if self.patch_size == 1:
            return pixel_values
        patch = self.patch_size
        height = self.geometry.height // patch
        width = self.geometry.width // patch
        pixel_values = pixel_values.reshape(
            (0, 3, self.geometry.conditioning_frames, height, patch, width,
             patch))
        pixel_values = pixel_values.transpose((0, 1, 6, 4, 2, 3, 5))
        return pixel_values.reshape(
            (0, 3 * patch * patch, self.geometry.conditioning_frames, height,
             width))

    def forward(self, pixel_values):
        hidden_states = self._patchify(pixel_values.cast(trt.float16))
        cache = _FeatureCache()
        chunks = []
        chunk_count = 1 + (self.geometry.conditioning_frames - 1) // 4
        for index in range(chunk_count):
            start = 0 if index == 0 else 1 + 4 * (index - 1)
            end = 1 if index == 0 else 1 + 4 * index
            chunk = hidden_states.slice_axis(2, start, end - start, 5)
            cache.reset()
            chunks.append(self.encoder(chunk, cache))
        encoded = (chunks[0] if len(chunks) == 1 else F.concatenate(chunks, 2))
        posterior = self.quant_conv(encoded).slice_axis(1, 0, self.z_dim, 5)
        mean = F.constant(self.latents_mean, "vae_latents_mean")
        std = F.constant(self.latents_std, "vae_latents_std")
        latent = ((posterior - mean) / std).cast(trt.float32)
        # The reference policy encodes the complete reflection-padded canvas,
        # then removes the padding in latent space using the original image size.
        if self.geometry.content_height != self.geometry.height:
            latent = latent.slice_axis(3, 0, self.geometry.latent_h, 5)
        if self.geometry.content_width != self.geometry.width:
            latent = latent.slice_axis(4, 0, self.geometry.latent_w, 5)
        return {"cond_latent": latent}
