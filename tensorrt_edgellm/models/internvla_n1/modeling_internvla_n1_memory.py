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
"""
From-scratch InternVLA-N1 System-1 memory block for ONNX export.

Turns a short window of navigation frames into the 32 memory tokens the
trajectory expert cross-attends against:

    DINOv2 ViT -> temporal encoder -> concat -> QFormer resampler

Only the DINOv2 tower needs real reimplementation. The encoder and the resampler
are stock ``nn.TransformerEncoder`` / ``nn.TransformerDecoder`` in the reference,
so they are reproduced directly.

Three things about the reference are easy to get wrong and are pinned here:

* The tower is the **DINOv2 backbone of DepthAnythingV2, not the depth head**.
  The checkpoint carries no DPT tensors at all, and the depth head is never run.
* Input is **pre-normalized**. ``generate_traj`` divides by the ResNet statistics
  before this block; the block itself does not normalize, so feeding raw pixels
  produces plausible-looking but wrong tokens.
* ``QFormer.visual_proj`` exists in the checkpoint and is **never called** by the
  reference forward, exactly like ``patch_embedder`` in the trajectory expert.
  It is named and skipped rather than dropped silently.

The positional embedding is stored for a 37x37 grid (518 px) and is bicubically
resampled to the 16x16 grid of a 224 px frame. That resample is reproduced
faithfully rather than approximated -- ``interpolate_offset`` in particular is
0.1, not 0, and a wrong offset shifts every patch slightly with no other symptom.
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

#: DINOv2-small as used by DepthAnythingV2 ``vits``.
DEFAULT_IMAGE_SIZE = 224
DEFAULT_PATCH_SIZE = 14
DEFAULT_EMBED_DIM = 384
DEFAULT_DEPTH = 12
DEFAULT_NUM_HEADS = 6
DEFAULT_MLP_RATIO = 4.0
#: DINOv2 (without registers) works around a float rounding issue in the
#: positional-embedding resample with a 0.1 offset. Do not "clean" this to 0.
INTERPOLATE_OFFSET = 0.1


@dataclass
class MemoryConfig:
    image_size: int = DEFAULT_IMAGE_SIZE
    patch_size: int = DEFAULT_PATCH_SIZE
    embed_dim: int = DEFAULT_EMBED_DIM
    depth: int = DEFAULT_DEPTH
    num_heads: int = DEFAULT_NUM_HEADS
    mlp_ratio: float = DEFAULT_MLP_RATIO
    num_pos_tokens: int = 1370          # 37 * 37 patches + cls
    memory_hidden: int = 384
    memory_heads: int = 6
    memory_layers: int = 3
    memory_max_len: int = 512
    num_query: int = 32
    qformer_hidden: int = 768
    qformer_heads: int = 12
    qformer_layers: int = 3


class PatchEmbed(nn.Module):
    def __init__(self, cfg: MemoryConfig) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, cfg.embed_dim, kernel_size=cfg.patch_size,
                              stride=cfg.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """ViT self-attention with a fused QKV projection."""

    def __init__(self, cfg: MemoryConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.embed_dim // cfg.num_heads
        self.qkv = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3, bias=True)
        self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(b, n, c))


class LayerScale(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, cfg: MemoryConfig) -> None:
        super().__init__()
        hidden = int(cfg.embed_dim * cfg.mlp_ratio)
        self.fc1 = nn.Linear(cfg.embed_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, cfg.embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: MemoryConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim, eps=1e-6)
        self.attn = Attention(cfg)
        self.ls1 = LayerScale(cfg.embed_dim)
        self.norm2 = nn.LayerNorm(cfg.embed_dim, eps=1e-6)
        self.mlp = Mlp(cfg)
        self.ls2 = LayerScale(cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class Dinov2VisionTower(nn.Module):
    """DINOv2-small, exposing the last block's patch tokens after the final norm.

    Equivalent to ``get_intermediate_layers(x)`` with the reference defaults
    (``n=1, norm=True, reshape=False, return_class_token=False``): last block,
    normed, class token dropped.
    """

    def __init__(self, cfg: Optional[MemoryConfig] = None) -> None:
        super().__init__()
        cfg = cfg or MemoryConfig()
        self.config = cfg
        self.patch_size = cfg.patch_size
        self.patch_embed = PatchEmbed(cfg)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, cfg.num_pos_tokens, cfg.embed_dim))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int,
                                 h: int) -> torch.Tensor:
        npatch = x.shape[1] - 1
        n = self.pos_embed.shape[1] - 1
        if npatch == n and w == h:
            return self.pos_embed
        dtype = x.dtype
        pos_embed = self.pos_embed.float()
        class_pos = pos_embed[:, 0]
        patch_pos = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size + INTERPOLATE_OFFSET
        h0 = h // self.patch_size + INTERPOLATE_OFFSET
        sqrt_n = math.sqrt(n)
        patch_pos = F.interpolate(
            patch_pos.reshape(1, int(sqrt_n), int(sqrt_n), dim).permute(0, 3, 1, 2),
            scale_factor=(float(w0) / sqrt_n, float(h0) / sqrt_n),
            mode="bicubic",
            antialias=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat([class_pos.unsqueeze(0), patch_pos], dim=1).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 1:]          # drop the class token


class MemoryEncoder(nn.Module):
    """Temporal encoder over the flattened patch sequence."""

    def __init__(self, cfg: MemoryConfig) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=cfg.memory_hidden,
                                           nhead=cfg.memory_heads,
                                           batch_first=True,
                                           dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.memory_layers)
        self.memory_pos = nn.Parameter(
            torch.zeros(cfg.memory_max_len, cfg.memory_hidden))

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        n = memory.shape[1]
        return self.encoder(memory + self.memory_pos[:n].unsqueeze(0))


class QFormer(nn.Module):
    """Resamples a variable-length visual sequence to a fixed set of queries."""

    def __init__(self, cfg: MemoryConfig) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(
            torch.zeros(cfg.num_query, cfg.qformer_hidden))
        self.query_pos = nn.Parameter(
            torch.zeros(cfg.num_query, cfg.qformer_hidden))
        layer = nn.TransformerDecoderLayer(d_model=cfg.qformer_hidden,
                                           nhead=cfg.qformer_heads,
                                           batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.qformer_layers)
        # Present in the checkpoint, never called by the reference forward.
        self.visual_proj = nn.Linear(cfg.qformer_hidden, cfg.qformer_hidden)

    def forward(self, visual_feats: torch.Tensor) -> torch.Tensor:
        batch = visual_feats.shape[0]
        queries = self.query_tokens.unsqueeze(0).expand(batch, -1, -1)
        queries = queries + self.query_pos.unsqueeze(0)
        return self.decoder(queries, visual_feats)


class InternVLAN1MemoryBlock(nn.Module):
    """Frames -> memory tokens, as one exportable module.

    Input is ``[T, 3, 224, 224]``, already ResNet-normalized; output is
    ``[1, num_query, 768]``.
    """

    def __init__(self, cfg: Optional[MemoryConfig] = None) -> None:
        super().__init__()
        cfg = cfg or MemoryConfig()
        self.config = cfg
        self.rgb_model = Dinov2VisionTower(cfg)
        self.memory_encoder = MemoryEncoder(cfg)
        self.rgb_resampler = QFormer(cfg)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.rgb_model(images).unflatten(0, (1, -1))   # [1, T, Np, C]
        flat = feat.flatten(1, 2)                             # [1, T*Np, C]
        encoded = self.memory_encoder(flat)
        return self.rgb_resampler(torch.cat([flat, encoded], dim=-1))


#: Checkpoint prefixes for the three sub-modules.
MEMORY_PREFIXES = {
    "rgb_model": "model.rgb_model.",
    "memory_encoder": "model.memory_encoder.",
    "rgb_resampler": "model.rgb_resampler.",
}

#: Present in the checkpoint, unreachable in ``forward`` -- see the docstring.
DEAD_TENSORS = ("rgb_resampler.visual_proj.weight",
                "rgb_resampler.visual_proj.bias")


def load_memory_weights(model: InternVLAN1MemoryBlock, weights: dict) -> dict:
    """Load the memory block from checkpoint tensors; return a load report."""
    own = dict(model.named_parameters())
    loaded, dead, unexpected = [], [], []
    for key, tensor in weights.items():
        name = None
        for attr, prefix in MEMORY_PREFIXES.items():
            if key.startswith(prefix):
                name = f"{attr}.{key[len(prefix):]}"
                break
        if name is None:
            continue
        if name in DEAD_TENSORS:
            dead.append(name)
            continue
        param = own.get(name)
        if param is None:
            unexpected.append(name)
            continue
        if tuple(param.shape) != tuple(tensor.shape):
            raise ValueError(f"shape mismatch for {name}: model "
                             f"{tuple(param.shape)} vs checkpoint "
                             f"{tuple(tensor.shape)}")
        param.data.copy_(tensor.to(param.dtype))
        loaded.append(name)

    covered = set(loaded) | set(DEAD_TENSORS)
    return {
        "loaded": len(loaded),
        "dead": sorted(dead),
        "unexpected": sorted(unexpected),
        "missing": sorted(set(own) - covered),
    }


def build_internvla_n1_memory(weights: dict,
                              cfg: Optional[MemoryConfig] = None,
                              dtype: torch.dtype = torch.bfloat16
                              ) -> InternVLAN1MemoryBlock:
    """Build the memory block and load it, refusing a partial load."""
    model = InternVLAN1MemoryBlock(cfg).to(dtype).eval()
    report = load_memory_weights(model, weights)
    if report["missing"] or report["unexpected"]:
        raise ValueError(
            "InternVLA-N1 memory block did not load cleanly: "
            f"{len(report['missing'])} parameter(s) never assigned, "
            f"{len(report['unexpected'])} checkpoint key(s) unmatched. "
            f"missing={report['missing'][:5]} "
            f"unexpected={report['unexpected'][:5]}")
    return model


__all__ = [
    "InternVLAN1MemoryBlock",
    "MemoryConfig",
    "build_internvla_n1_memory",
    "load_memory_weights",
    "MEMORY_PREFIXES",
]
