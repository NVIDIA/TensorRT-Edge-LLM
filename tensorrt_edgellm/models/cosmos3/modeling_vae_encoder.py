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
"""From-scratch Wan2.2 VAE *encoder* for the Cosmos3 policy conditioning frame.

Policy mode conditions the GEN diffusion on the VAE latent of the conditioning
image (frame-0).  Only the encoder is needed (video pixel decode is out of
scope).  Every module below is implemented from scratch in this file — pixel
patchify, causal 3D conv tower with residual down blocks (Wan 2.2 layout),
single mid attention, ``quant_conv``, posterior mean, and the per-channel
latent normalization — with submodule attribute paths matching the ``vae/``
checkpoint keys (``encoder.*`` / ``quant_conv.*``) so the safetensors weights
load directly.

Numerics follow the reference exactly:

* Causal 3D convs left-pad time by ``2 * pad_t`` and consume a 2-frame feature
  cache so a clip can be encoded in causal chunks of 4 frames (the reference
  ``1 + (F - 1) // 4`` chunking, reproduced in :meth:`Cosmos3VaeEncoder.forward`).
* The single-head mid attention is written as explicit
  ``softmax(q kᵀ / sqrt(c)) v`` — TensorRT has no fused-attention kernel for
  this shape/dtype, so the decomposed form is required for the engine build.
* The forward returns the posterior **mode** (mean channels of the
  ``2 * z_dim`` encoder output) normalized by ``latents_mean`` /
  ``latents_std`` from the VAE config, exactly as the reference policy
  pipeline consumes it.  Pixel preprocessing (``uint8 -> /127.5 - 1``) is
  applied by the caller.
"""

from __future__ import annotations

import json
import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy_geometry import (DEFAULT_HEIGHT, DEFAULT_NUM_FRAMES,
                              DEFAULT_WIDTH, VAE_SPATIAL_DOWNSAMPLE_FACTOR)

logger = logging.getLogger(__name__)

# Frames of features cached per causal conv between temporal chunks.
_CACHE_T = 2


class _CausalConv3d(nn.Conv3d):
    """3D convolution, causal in time: left-pads time by ``2 * pad_t`` and can
    consume a cached tail of the previous chunk instead of zero padding."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0) -> None:
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride=stride,
                         padding=padding)
        self._causal_padding = (self.padding[2], self.padding[2],
                                self.padding[1], self.padding[1],
                                2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self,
                x: torch.Tensor,
                cache_x: "torch.Tensor | None" = None) -> torch.Tensor:
        padding = list(self._causal_padding)
        if cache_x is not None and self._causal_padding[4] > 0:
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        return super().forward(F.pad(x, padding))


class _ChannelRMSNorm(nn.Module):
    """Channel-wise RMS norm (``F.normalize`` over dim 1, computed in fp32)
    scaled by ``sqrt(dim) * gamma``; ``gamma`` broadcasts over (T,)H,W."""

    def __init__(self, dim: int, video: bool = False) -> None:
        super().__init__()
        shape = (dim, 1, 1, 1) if video else (dim, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.scale = dim**0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(x.float(), dim=1).to(x.dtype)
        return normalized * self.scale * self.gamma


class _ResidualBlock(nn.Module):
    """norm -> silu -> causal conv, twice, with a 1x1 causal-conv shortcut."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm1 = _ChannelRMSNorm(in_dim, video=True)
        self.conv1 = _CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = _ChannelRMSNorm(out_dim, video=True)
        self.conv2 = _CausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = _CausalConv3d(
            in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache, feat_idx):
        h = self.conv_shortcut(x)
        x = F.silu(self.norm1(x))
        x = _cached_conv(self.conv1, x, feat_cache, feat_idx)
        x = F.silu(self.norm2(x))
        x = _cached_conv(self.conv2, x, feat_cache, feat_idx)
        return x + h


class _AttentionBlock(nn.Module):
    """Single-head per-frame spatial self-attention, decomposed explicitly."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = _ChannelRMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.norm(x)
        qkv = self.to_qkv(x).reshape(b * t, c * 3, h * w).permute(0, 2, 1)
        q, k, v = qkv.chunk(3, dim=-1)  # [b*t, h*w, c] each
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(c)
        x = torch.matmul(torch.softmax(attn, dim=-1), v)
        x = x.permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return x + identity


class _MidBlock(nn.Module):
    """resnet -> attention -> resnet."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attentions = nn.ModuleList([_AttentionBlock(dim)])
        self.resnets = nn.ModuleList(
            [_ResidualBlock(dim, dim),
             _ResidualBlock(dim, dim)])

    def forward(self, x, feat_cache, feat_idx):
        x = self.resnets[0](x, feat_cache, feat_idx)
        x = self.attentions[0](x)
        x = self.resnets[1](x, feat_cache, feat_idx)
        return x


class _AvgDown3d(nn.Module):
    """Parameter-free space-to-channel average pooling (residual shortcut)."""

    def __init__(self, in_channels: int, out_channels: int, factor_t: int,
                 factor_s: int) -> None:
        super().__init__()
        self.factor_t = factor_t
        self.factor_s = factor_s
        factor = factor_t * factor_s * factor_s
        assert in_channels * factor % out_channels == 0
        self.out_channels = out_channels
        self.group_size = in_channels * factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        b, c, t, h, w = x.shape
        x = x.view(b, c, t // self.factor_t, self.factor_t, h // self.factor_s,
                   self.factor_s, w // self.factor_s, self.factor_s)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(b, self.out_channels, self.group_size, t // self.factor_t,
                   h // self.factor_s, w // self.factor_s)
        return x.mean(dim=2)


class _Downsample(nn.Module):
    """Strided spatial conv; the 3D variant adds a strided causal time conv."""

    def __init__(self, dim: int, temporal: bool) -> None:
        super().__init__()
        self.temporal = temporal
        self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)),
                                      nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        if temporal:
            self.time_conv = _CausalConv3d(dim,
                                           dim, (3, 1, 1),
                                           stride=(2, 1, 1))

    def forward(self, x, feat_cache, feat_idx):
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2),
                   x.size(3)).permute(0, 2, 1, 3, 4)
        if self.temporal:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                # First chunk: the strided time conv is skipped; its cache seeds
                # from this chunk's output (reference WanResample downsample3d).
                feat_cache[idx] = x.clone()
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :].clone()
                x = self.time_conv(
                    torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
        return x


class _ResidualDownBlock(nn.Module):
    """Wan 2.2 residual down block: resnets (+ optional downsample) plus an
    average-pooled space-to-channel shortcut."""

    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int,
                 temporal_downsample: bool, down_flag: bool) -> None:
        super().__init__()
        self.avg_shortcut = _AvgDown3d(in_dim, out_dim,
                                       2 if temporal_downsample else 1,
                                       2 if down_flag else 1)
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(_ResidualBlock(in_dim, out_dim))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.downsampler = _Downsample(
            out_dim, temporal=temporal_downsample) if down_flag else None

    def forward(self, x, feat_cache, feat_idx):
        shortcut = self.avg_shortcut(x)
        for resnet in self.resnets:
            x = resnet(x, feat_cache, feat_idx)
        if self.downsampler is not None:
            x = self.downsampler(x, feat_cache, feat_idx)
        return x + shortcut


def _cached_conv(conv: _CausalConv3d, x: torch.Tensor, feat_cache: list,
                 feat_idx: list) -> torch.Tensor:
    """Run a causal conv with the reference 2-frame feature cache protocol."""
    idx = feat_idx[0]
    cache_x = x[:, :, -_CACHE_T:, :, :].clone()
    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
        # Chunk shorter than the cache: keep the last frame of the previous cache.
        cache_x = torch.cat(
            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
    x = conv(x, feat_cache[idx])
    feat_cache[idx] = cache_x
    feat_idx[0] += 1
    return x


class _Encoder3d(nn.Module):
    """Wan 2.2 3D encoder: conv_in -> residual down blocks -> mid -> conv_out."""

    def __init__(self, in_channels: int, dim: int, z_dim: int, dim_mult: list,
                 num_res_blocks: int, temporal_downsample: list) -> None:
        super().__init__()
        dims = [dim * u for u in [1] + list(dim_mult)]
        self.conv_in = _CausalConv3d(in_channels, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([
            _ResidualDownBlock(
                in_dim,
                out_dim,
                num_res_blocks,
                temporal_downsample=temporal_downsample[i]
                if i != len(dim_mult) - 1 else False,
                down_flag=i != len(dim_mult) - 1,
            ) for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]))
        ])
        self.mid_block = _MidBlock(dims[-1])
        self.norm_out = _ChannelRMSNorm(dims[-1], video=True)
        self.conv_out = _CausalConv3d(dims[-1], z_dim, 3, padding=1)

    def forward(self, x, feat_cache, feat_idx):
        x = _cached_conv(self.conv_in, x, feat_cache, feat_idx)
        for block in self.down_blocks:
            x = block(x, feat_cache, feat_idx)
        x = self.mid_block(x, feat_cache, feat_idx)
        x = F.silu(self.norm_out(x))
        x = _cached_conv(self.conv_out, x, feat_cache, feat_idx)
        return x


class Cosmos3VaeEncoder(nn.Module):
    """Wan 2.2 VAE encoder head for ONNX export.

    ``forward(pixel_values[B, 3, T, H, W]) -> cond_latent[B, z_dim, t, h, w]``:
    pixel patchify, causal chunked encode, ``quant_conv``, posterior mean, and
    per-channel latent normalization — the exact latent the reference policy
    pipeline conditions on.
    """

    def __init__(self,
                 vae_config: dict,
                 content_height: "int | None" = None,
                 content_width: "int | None" = None) -> None:
        super().__init__()
        self.patch_size = int(vae_config["patch_size"])
        self.z_dim = int(vae_config["z_dim"])
        self.content_height = content_height
        self.content_width = content_width
        assert vae_config.get("is_residual", False), \
            "Cosmos3 uses the Wan 2.2 residual-down-block VAE"
        assert not vae_config.get("attn_scales"), \
            "Wan 2.2 encoder has no per-scale attention (mid attention only)"
        self.encoder = _Encoder3d(
            in_channels=int(vae_config["in_channels"]),
            dim=int(vae_config["base_dim"]),
            z_dim=self.z_dim * 2,
            dim_mult=vae_config["dim_mult"],
            num_res_blocks=int(vae_config["num_res_blocks"]),
            temporal_downsample=vae_config["temperal_downsample"],
        )
        self.quant_conv = _CausalConv3d(self.z_dim * 2, self.z_dim * 2, 1)
        # One cache slot per causal conv in the encoder (a temporal
        # downsampler's slot belongs to its strided time_conv).
        self._num_cached_convs = sum(
            isinstance(m, _CausalConv3d) for m in self.encoder.modules())
        mean = torch.tensor(vae_config["latents_mean"]).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae_config["latents_std"]).view(1, -1, 1, 1, 1)
        self.register_buffer("latents_mean", mean)
        self.register_buffer("latents_std", std)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_size == 1:
            return x
        p = self.patch_size
        b, c, f, h, w = x.shape
        x = x.view(b, c, f, h // p, p, w // p, p)
        x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
        return x.view(b, c * p * p, f, h // p, w // p)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        dtype = self.quant_conv.weight.dtype
        x = self._patchify(pixel_values.to(dtype))
        num_frames = x.shape[2]

        # Causal chunked encode (frame 0, then chunks of 4), reproducing the
        # reference streaming protocol so the latent matches it exactly.
        feat_cache: list = [None] * self._num_cached_convs
        chunks = []
        for i in range(1 + (num_frames - 1) // 4):
            start, end = (0, 1) if i == 0 else (1 + 4 * (i - 1), 1 + 4 * i)
            chunks.append(
                self.encoder(x[:, :, start:end], feat_cache, feat_idx=[0]))
        out = torch.cat(chunks, dim=2) if len(chunks) > 1 else chunks[0]

        latent = self.quant_conv(out)[:, :self.z_dim]  # posterior mean
        latent = (latent - self.latents_mean.to(latent.dtype)) / \
            self.latents_std.to(latent.dtype)
        if self.content_height is not None:
            latent = latent[:, :, :, :self.content_height //
                            VAE_SPATIAL_DOWNSAMPLE_FACTOR, :]
        if self.content_width is not None:
            latent = latent[:, :, :, :, :self.content_width //
                            VAE_SPATIAL_DOWNSAMPLE_FACTOR]
        return latent.float()


def build_cosmos3_vae_encoder(
        vae_dir: str,
        dtype: torch.dtype,
        content_height: "int | None" = None,
        content_width: "int | None" = None) -> Cosmos3VaeEncoder:
    """Build the from-scratch encoder and load the ``vae/`` checkpoint weights."""
    from safetensors import safe_open

    with open(os.path.join(vae_dir, "config.json")) as f:
        vae_config = json.load(f)
    model = Cosmos3VaeEncoder(vae_config, content_height,
                              content_width).to(dtype)

    weights = {}
    for name in sorted(os.listdir(vae_dir)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(vae_dir, name),
                       framework="pt",
                       device="cpu") as f:
            for key in f.keys():
                if key.startswith(("encoder.", "quant_conv.")):
                    t = f.get_tensor(key)
                    weights[key] = t.to(dtype) if t.is_floating_point() else t

    missing, unexpected = model.load_state_dict(weights, strict=False)
    # Buffers (latents_mean/std) come from the config, not the checkpoint.
    missing = [m for m in missing if not m.startswith("latents_")]
    if missing or unexpected:
        raise KeyError(
            f"Cosmos3 VAE encoder load mismatch: missing={missing[:5]} "
            f"unexpected={unexpected[:5]}")
    logger.info("Cosmos3 VAE encoder load: assigned=%d missing=0 unexpected=0",
                len(weights))
    model.eval()
    return model


def get_vae_onnx_export_args(height: int = DEFAULT_HEIGHT,
                             width: int = DEFAULT_WIDTH,
                             device: str = "cpu",
                             dtype: torch.dtype = torch.float16,
                             num_frames: int = DEFAULT_NUM_FRAMES):
    """Full-clip inputs + names + dynamic shapes for VAE encoder export.

    The policy pipeline VAE-encodes the *full* conditioning clip
    ``[B, 3, num_frames, H, W]`` (the conditioning image broadcast to
    ``num_frames``) and keeps frame 0 via the vision condition mask; this
    replicates that, so the algorithm — and the measured VAE workload — is
    identical to the reference.
    """
    # Runtime preprocessing produces float32 pixels in [-1, 1]. Keep the ONNX
    # boundary fp32 and cast internally to the VAE dtype. Sample batch of 2
    # keeps the batch axis symbolic (torch.export specializes size-1 dims).
    pixel_values = torch.zeros(2,
                               3,
                               num_frames,
                               height,
                               width,
                               device=device,
                               dtype=torch.float32)
    input_names = ["pixel_values"]
    output_names = ["cond_latent"]
    batch = torch.export.Dim("batch_size", min=1, max=256)
    dynamic_shapes = ({0: batch}, )
    return (pixel_values, ), input_names, output_names, dynamic_shapes


# ---------------------------------------------------------------------------
# Component contract (consumed by the experimental Cosmos3 builder/runtime)
# ---------------------------------------------------------------------------


def make_vae_encoder_config(height: int = DEFAULT_HEIGHT,
                            width: int = DEFAULT_WIDTH,
                            num_frames: int = DEFAULT_NUM_FRAMES,
                            content_height: "int | None" = None,
                            content_width: "int | None" = None) -> dict:
    """Return the VAE encoder component ``config.json`` payload."""
    content_height = height if content_height is None else content_height
    content_width = width if content_width is None else content_width
    return {
        "component": "vae_encoder",
        "onnx_filename": "model.onnx",
        "engine_filename": "vae_encoder.engine",
        "optimization_profile": {
            "pixel_values": {
                "min": [1, 3, num_frames, height, width],
                "opt": [1, 3, num_frames, height, width],
                "max": [1, 3, num_frames, height, width],
            },
        },
        "tensor_contract": {
            "inputs": {
                "pixel_values": ["batch", 3, num_frames, height, width],
            },
            "outputs": {
                "cond_latent": ["batch", "latent_channel", "t", "h", "w"],
            },
        },
        "builder_config": {
            "max_batch_size": 1,
            "height": height,
            "width": width,
            "content_height": content_height,
            "content_width": content_width,
            "num_frames": num_frames,
        },
    }
