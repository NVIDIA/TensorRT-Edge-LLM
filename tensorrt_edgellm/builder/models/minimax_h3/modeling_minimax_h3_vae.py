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
"""Checkpoint-direct MiniMax-H3 video VAE decoder for one latent tile.

The engine decodes the fixed tile of Diffusers' tiled decode; the runtime
blends tiles and clips. Latent and ImageNet pixel normalization are folded in:
the input is a GEN latent and the output is RGB in ``[0, 1]`` before clamping.
The ViT decoder runs under FP16 autocast in Diffusers: projections are FP16
and normalization, LayerScale, and the residual stream are FP32.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from ...ops import Linear, Module, NetworkModule
from ...ops import functional as F
from . import configuration
from .layers import apply_rotary

_PIXEL_MEAN = (0.485, 0.456, 0.406)
_PIXEL_STD = (0.229, 0.224, 0.225)
#: Latent tile of Diffusers' 256-pixel tiled decode of a 7-latent clip.
TILE_LATENT_FRAMES = 7
TILE_LATENT_SIZE = 16


def _rgb_constant(values, name: str):
    """Per-channel constant broadcast over ``[1, 3, T, H, W]`` pixels."""
    return F.constant(
        np.asarray(values, np.float32).reshape(1, 3, 1, 1, 1), name)


def _rms_norm(hidden, eps: float, weight=None):
    """FP32 RMSNorm over the last axis with an optional FP32 affine."""
    rank = hidden.ndim
    hidden = hidden.cast(trt.float32)
    mean_square = F.reduce(hidden * hidden,
                           trt.ReduceOperation.AVG,
                           1 << (rank - 1),
                           keep_dims=True)
    hidden = hidden / (
        mean_square + F.constant(np.full(
            (1, ) * rank, eps, dtype=np.float32), "rms_eps")).sqrt()
    if weight is None:
        return hidden
    return hidden * F.constant(
        weight.reshape((1, ) * (rank - 1) +
                       (-1, )).astype(np.float32), "rms_weight")


class MiniMaxH3VaeBlock(Module):
    """Pre-norm ViT block with LayerScale residual updates."""

    def __init__(self, ctx, prefix: str, config: dict) -> None:
        super().__init__(ctx, prefix)
        self.eps = float(config["decoder_norm_eps"])
        self.num_heads = int(config["decoder_num_attention_heads"])
        self.head_dim = int(config["decoder_attention_head_dim"])
        self.inner_dim = (int(config["decoder_ffn_mult"]) * self.num_heads *
                          self.head_dim)
        self.to_q = Linear(ctx, self.key("attn.to_q"), tensor_parallel=False)
        self.to_k = Linear(ctx, self.key("attn.to_k"), tensor_parallel=False)
        self.to_v = Linear(ctx, self.key("attn.to_v"), tensor_parallel=False)
        self.to_out = Linear(ctx,
                             self.key("attn.to_out.0"),
                             tensor_parallel=False)
        self.ff_proj = Linear(ctx,
                              self.key("ff.net.0.proj"),
                              tensor_parallel=False)
        self.ff_down = Linear(ctx, self.key("ff.net.2"), tensor_parallel=False)

    def _norm(self, name: str, hidden):
        return _rms_norm(hidden, self.eps,
                         self.weights.f32(self.key(name + ".weight")))

    def _layer_scale(self, name: str):
        return F.constant(
            self.weights.f32(self.key(name)).reshape(1, 1, -1), self.key(name))

    def _attention(self, hidden, rope):
        heads = (0, 0, self.num_heads, self.head_dim)
        query, key = (_rms_norm(projection(hidden).reshape(heads),
                                self.eps).cast(trt.float16)
                      for projection in (self.to_q, self.to_k))
        attended = F.scaled_dot_product_attention(
            apply_rotary(query, *rope).transpose((0, 2, 1, 3)),
            apply_rotary(key, *rope).transpose((0, 2, 1, 3)),
            self.to_v(hidden).reshape(heads).transpose((0, 2, 1, 3)),
            scale=self.head_dim**-0.5).transpose((0, 2, 1, 3))
        return self.to_out(
            attended.reshape((0, 0, self.num_heads * self.head_dim)))

    def forward(self, hidden, rope):
        normed = self._norm("norm1", hidden).cast(trt.float16)
        hidden = hidden + self._attention(normed, rope).cast(
            trt.float32) * self._layer_scale("scale1")
        fused = self.ff_proj(self._norm("norm2", hidden).cast(trt.float16))
        fed = fused.slice_last_dim(
            0, self.inner_dim, 3) * fused.slice_last_dim(
                self.inner_dim, self.inner_dim, 3).silu()
        return hidden + self.ff_down(fed).cast(
            trt.float32) * self._layer_scale("scale2")


class MiniMaxH3VaeDecoder(NetworkModule):
    """Decode ``[1, C, 7, 16, 16]`` latents to ``[1, 3, 28, 256, 256]`` RGB."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle)

    def __init__(self, ctx, bundle) -> None:
        super().__init__(ctx)
        config = configuration.component_config(
            bundle.root, configuration.Component.VAE_DECODER)
        self.config = config
        self.latent_channels = int(config["latent_channels"])
        self.patch_size = math.prod(config["spatial_downsample_factors"])
        self.patch_size_t = math.prod(config["temporal_downsample_factors"])
        self.num_registers = int(config["decoder_num_register_tokens"])
        self.dim = (int(config["decoder_num_attention_heads"]) *
                    int(config["decoder_attention_head_dim"]))
        self.proj_in = Linear(ctx, "decoder.proj_in", tensor_parallel=False)
        self.blocks = [
            MiniMaxH3VaeBlock(ctx, f"decoder.transformer_blocks.{index}",
                              config)
            for index in range(int(config["decoder_num_layers"]))
        ]
        self.proj_out = Linear(ctx, "decoder.proj_out", tensor_parallel=False)

    def input_tensors(self):
        return {
            "latents":
            self.add_input("latents", trt.float32,
                           (1, self.latent_channels, TILE_LATENT_FRAMES,
                            TILE_LATENT_SIZE, TILE_LATENT_SIZE))
        }

    def _rope(self):
        """Decoder RoPE of the tile's normalized ``(t, h, w)`` grid."""
        config = self.config
        rotary_dim = int(config["decoder_attention_head_dim"] *
                         config["decoder_rope_dim_ratio"])
        inverse = 1.0 / float(config["decoder_rope_theta"])**np.arange(
            0, 1, 6 / rotary_dim, dtype=np.float32)
        grids = [
            2.0 * (np.arange(0.5, size, dtype=np.float32) / size) - 1.0
            for size in (TILE_LATENT_FRAMES, TILE_LATENT_SIZE,
                         TILE_LATENT_SIZE)
        ]
        positions = np.stack(np.meshgrid(*grids, indexing="ij"),
                             axis=-1).reshape(-1, 3)
        positions = np.concatenate(
            (positions, np.zeros((self.num_registers + 1, 3), np.float32)))
        angles = F.constant(positions.reshape(
            -1, 3, 1), "rope_positions") * F.constant(
                (2.0 * np.pi * inverse).reshape(1, 1, -1), "rope_inv_freq")
        angles = angles.reshape((-1, rotary_dim // 2))
        angles = F.concatenate((angles, angles), 1).reshape(
            (1, -1, 1, rotary_dim))
        return angles.cos(), angles.sin(), rotary_dim

    def forward(self, latents):
        config = self.config
        channels = self.latent_channels
        num_patches = TILE_LATENT_FRAMES * TILE_LATENT_SIZE**2
        mean, std = (np.asarray(config[name],
                                np.float32).reshape(1, channels, 1, 1, 1)
                     for name in ("latents_mean", "latents_std"))
        latents = (latents * F.constant(std, "latents_std") +
                   F.constant(mean, "latents_mean"))
        tokens = latents.transpose((0, 2, 3, 4, 1)).reshape(
            (1, num_patches, channels))
        # post_quant_conv is a 1x1x1 Conv3d, i.e. a per-voxel linear.
        weight = self.weights.f32("post_quant_conv.weight").reshape(
            channels, channels)
        bias = self.weights.f32("post_quant_conv.bias").reshape(1, 1, -1)
        weight = F.constant(
            np.ascontiguousarray(weight.T)[None], "post_quant_conv")
        tokens = (F.matmul(tokens, weight) +
                  F.constant(bias, "post_quant_conv_bias"))

        hidden = self.proj_in(tokens.cast(trt.float16)).cast(trt.float32)
        hidden = F.concatenate(
            (hidden,
             F.constant(self.weights.f32("decoder.register_tokens"),
                        "register_tokens"),
             F.constant(np.zeros(
                 (1, 1, self.dim), np.float32), "cls_token")), 1)
        rope = self._rope()
        for block in self.blocks:
            hidden = block(hidden, rope)
        hidden = F.layer_norm(hidden,
                              self.weights.f32("decoder.norm_out.weight"),
                              self.weights.f32("decoder.norm_out.bias"),
                              float(config["decoder_norm_eps"]), 3)
        hidden = self.proj_out(hidden.cast(trt.float16)).cast(trt.float32)

        patch, patch_t = self.patch_size, self.patch_size_t
        pixels = hidden.slice_axis(1, 0, num_patches, 3).reshape(
            (1, TILE_LATENT_FRAMES, TILE_LATENT_SIZE, TILE_LATENT_SIZE, 3,
             patch_t, patch, patch)).transpose((0, 4, 1, 5, 2, 6, 3, 7))
        pixels = pixels.reshape(
            (1, 3, TILE_LATENT_FRAMES * patch_t, TILE_LATENT_SIZE * patch,
             TILE_LATENT_SIZE * patch))
        pixels = (pixels * _rgb_constant(_PIXEL_STD, "pixel_std") +
                  _rgb_constant(_PIXEL_MEAN, "pixel_mean"))
        return {"pixels": pixels.cast(trt.float16)}
