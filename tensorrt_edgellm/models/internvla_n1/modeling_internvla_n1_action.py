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
From-scratch InternVLA-N1 System-1 trajectory expert for ONNX export.

Wraps one flow-matching denoising step: the noisy trajectory latents are
cross-attended against ``z_latents`` from System 2 and the block stack predicts
the flow. The Euler step and the sampler loop stay outside the graph.

The reference implementation (``NextDiTCrossAttn`` over a modified
``LuminaNextDiT2DModel``) is built on ``diffusers`` primitives, which this repo
does not depend on, so the pieces are reimplemented here in the same style as
``models/alpamayo``.

Reimplementing rather than importing also removes a whole class of specialization
this particular model invites. The reference is a general 2-D Lumina DiT, but
InternVLA-N1 drives it in a much narrower regime:

* ``image_rotary_emb`` is ``None`` on every call, so there is **no RoPE** here.
* ``patch_embedder`` is constructed but never called in ``forward``; its weights
  exist in the checkpoint and are dead. They are deliberately not loaded.
* Both attention masks are all-ones, so attention is plain SDPA with no mask.
* ``num_kv_heads == num_attention_heads``, so the GQA repeat is the identity.

Two details are load-bearing and must not be "simplified":

* ``attn1`` has **no** output projection -- the reference sets it to Identity and
  reuses ``attn2.to_out[0]`` for the summed self+cross result. There is exactly
  one projection per block, not two.
* The feed-forward SiLU runs in fp32 in the reference (``FP32SiLU``). Keeping
  that is what makes bf16 export match.
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Reference config (``NextDiTCrossAttnConfig`` with ``LatentEmbSize = 768``).
#: These are constants of the released checkpoint, not tunables.
DEFAULT_DIM = 384
DEFAULT_N_LAYERS = 12
DEFAULT_N_HEADS = 6
DEFAULT_FFN_MULTIPLE_OF = 256
DEFAULT_FFN_DIM_MULTIPLIER = 0.6667
DEFAULT_NORM_EPS = 1e-5
DEFAULT_FREQ_EMBED_SIZE = 256


@dataclass
class TrajDitConfig:
    """Shape parameters for the trajectory expert."""

    dim: int = DEFAULT_DIM
    num_layers: int = DEFAULT_N_LAYERS
    num_attention_heads: int = DEFAULT_N_HEADS
    num_kv_heads: int = DEFAULT_N_HEADS
    in_channels: int = DEFAULT_DIM
    latent_dim: int = 768
    multiple_of: int = DEFAULT_FFN_MULTIPLE_OF
    ffn_dim_multiplier: float = DEFAULT_FFN_DIM_MULTIPLIER
    norm_eps: float = DEFAULT_NORM_EPS
    frequency_embedding_size: int = DEFAULT_FREQ_EMBED_SIZE

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_attention_heads

    @property
    def cond_dim(self) -> int:
        """Width of the AdaLN conditioning vector (``min(dim, 1024)``)."""
        return min(self.dim, 1024)


class RMSNorm(nn.Module):
    """RMS normalization with a learned scale, computed in fp32."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep features, cosine first.

    Mirrors ``Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0.0)``: the
    cosine half precedes the sine half, and the frequency exponent is not
    shifted. Both choices are visible in the output ordering, so a mismatch here
    is silent -- the model still runs, it just conditions on the wrong phase.
    """
    half = dim // 2
    exponent = -math.log(10000.0) * torch.arange(
        half, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / half
    emb = timesteps.float()[:, None] * torch.exp(exponent)[None, :]
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(sample)))


class CombinedTimestepCaptionEmbedding(nn.Module):
    """AdaLN conditioning: timestep features plus mask-pooled z_latents."""

    def __init__(self, cfg: TrajDitConfig) -> None:
        super().__init__()
        self.frequency_embedding_size = cfg.frequency_embedding_size
        self.timestep_embedder = TimestepEmbedder(cfg.frequency_embedding_size,
                                                  cfg.cond_dim)
        self.caption_embedder = nn.Sequential(
            nn.LayerNorm(cfg.dim),
            nn.Linear(cfg.dim, cfg.cond_dim, bias=True),
        )

    def forward(self, timestep: torch.Tensor, caption_feat: torch.Tensor,
                caption_mask: torch.Tensor) -> torch.Tensor:
        freq = timestep_embedding(timestep, self.frequency_embedding_size)
        time_embed = self.timestep_embedder(freq.to(caption_feat.dtype))
        mask = caption_mask.to(caption_feat.dtype).unsqueeze(-1)
        pooled = (caption_feat * mask).sum(dim=1) / mask.sum(dim=1)
        return time_embed + self.caption_embedder(pooled)


class CaptionProjection(nn.Module):
    """``z_latents`` (latent_dim) -> model width, with a tanh-GELU in between."""

    def __init__(self, in_features: int, hidden_size: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size, bias=True)
        self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act_1(self.linear_1(caption)))


class FeedForward(nn.Module):
    """SwiGLU. The gate activation is evaluated in fp32, as in the reference."""

    def __init__(self, cfg: TrajDitConfig) -> None:
        super().__init__()
        inner = int(cfg.ffn_dim_multiplier * 4 * cfg.dim)
        inner = cfg.multiple_of * (
            (inner + cfg.multiple_of - 1) // cfg.multiple_of)
        self.inner_dim = inner
        self.linear_1 = nn.Linear(cfg.dim, inner, bias=False)
        self.linear_2 = nn.Linear(inner, cfg.dim, bias=False)
        self.linear_3 = nn.Linear(cfg.dim, inner, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.linear_1(x).float()).to(x.dtype)
        return self.linear_2(gate * self.linear_3(x))


class RMSNormZero(nn.Module):
    """Adaptive RMS norm producing the block's three gates."""

    def __init__(self, cfg: TrajDitConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(cfg.cond_dim, 4 * cfg.dim, bias=True)
        self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        emb = self.linear(F.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
        return self.norm(x) * (1 + scale_msa[:, None]), gate_msa, scale_mlp, gate_mlp


class LayerNormContinuous(nn.Module):
    """AdaLN output head: scale by a projection of the conditioning, then project."""

    def __init__(self, cfg: TrajDitConfig, out_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(cfg.cond_dim, cfg.dim, bias=True)
        self.norm = nn.LayerNorm(cfg.dim, eps=1e-6, elementwise_affine=False)
        self.linear_2 = nn.Linear(cfg.dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale = self.linear_1(F.silu(conditioning).to(x.dtype))
        return self.linear_2(self.norm(x) * (1 + scale)[:, None, :])


class Attention(nn.Module):
    """Self- or cross-attention with LayerNorm applied across all heads.

    Returns the per-head result **unflattened**; the block sums self and cross
    before the single shared output projection, so projecting here would be
    wrong.
    """

    def __init__(self, cfg: TrajDitConfig, kv_dim: int,
                 with_out_proj: bool) -> None:
        super().__init__()
        self.heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        inner = cfg.num_attention_heads * cfg.head_dim
        kv_inner = cfg.num_kv_heads * cfg.head_dim
        self.to_q = nn.Linear(cfg.dim, inner, bias=False)
        self.to_k = nn.Linear(kv_dim, kv_inner, bias=False)
        self.to_v = nn.Linear(kv_dim, kv_inner, bias=False)
        # "layer_norm_across_heads": one LayerNorm over the whole projection,
        # applied before the head split.
        self.norm_q = nn.LayerNorm(inner, eps=1e-5)
        self.norm_k = nn.LayerNorm(kv_inner, eps=1e-5)
        self.to_out = nn.ModuleList(
            [nn.Linear(inner, cfg.dim, bias=False)]) if with_out_proj else None

    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        batch = hidden_states.shape[0]
        query = self.norm_q(self.to_q(hidden_states))
        key = self.norm_k(self.to_k(encoder_hidden_states))
        value = self.to_v(encoder_hidden_states)

        query = query.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)

        # Masks are all-ones in this model, so SDPA runs unmasked.
        out = F.scaled_dot_product_attention(query, key, value)
        return out.transpose(1, 2)          # [B, S, H, D], not flattened


class TrajDitBlock(nn.Module):
    def __init__(self, cfg: TrajDitConfig) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(cfg.num_attention_heads))
        self.attn1 = Attention(cfg, kv_dim=cfg.dim, with_out_proj=False)
        self.attn2 = Attention(cfg, kv_dim=cfg.dim, with_out_proj=True)
        self.feed_forward = FeedForward(cfg)
        self.norm1 = RMSNormZero(cfg)
        self.ffn_norm1 = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.norm2 = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.ffn_norm2 = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.norm1_context = RMSNorm(cfg.dim, eps=cfg.norm_eps)

    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                temb: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        normed, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)

        self_out = self.attn1(normed, normed)
        cross_out = self.attn2(normed, self.norm1_context(encoder_hidden_states))
        cross_out = cross_out * self.gate.tanh().view(1, 1, -1, 1)

        mixed = (self_out + cross_out).flatten(-2)
        hidden_states = self.attn2.to_out[0](mixed)
        hidden_states = residual + gate_msa.unsqueeze(1).tanh() * self.norm2(
            hidden_states)

        mlp_out = self.feed_forward(
            self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1)))
        return hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(
            mlp_out)


class InternVLAN1TrajDit(nn.Module):
    """One denoising step of the InternVLA-N1 trajectory expert."""

    def __init__(self, cfg: Optional[TrajDitConfig] = None) -> None:
        super().__init__()
        cfg = cfg or TrajDitConfig()
        self.config = cfg
        self.caption_projection = CaptionProjection(cfg.latent_dim, cfg.dim)
        self.time_caption_embed = CombinedTimestepCaptionEmbedding(cfg)
        self.layers = nn.ModuleList(
            [TrajDitBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm_out = LayerNormContinuous(cfg, out_dim=cfg.in_channels)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor,
                z_latents: torch.Tensor) -> torch.Tensor:
        encoder_hidden_states = self.caption_projection(z_latents)
        # The reference passes an all-ones encoder mask, so pooling is a plain
        # mean over the conditioning sequence.
        mask = torch.ones(encoder_hidden_states.shape[:2],
                          dtype=encoder_hidden_states.dtype,
                          device=encoder_hidden_states.device)
        temb = self.time_caption_embed(timestep, encoder_hidden_states, mask)
        hidden_states = x
        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states, temb)
        return self.norm_out(hidden_states, temb)


class SinusoidalPositionalEncoding(nn.Module):
    """Waypoint-index encoding, sine half first.

    Parameter-free. The ordering is sin-then-cos, the opposite of the timestep
    embedding above -- both orderings appear in this model and swapping either
    is silent.
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        half = self.embedding_dim // 2
        exponent = -torch.arange(half, dtype=torch.float32,
                                 device=positions.device) * (
            math.log(10000.0) / half)
        freqs = positions.float().unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class InternVLAN1TrajDitStep(nn.Module):
    """One denoising step end to end, in trajectory space.

    Wraps the expert with the projections that surround it, so the engine takes
    and returns waypoints ``[batch, waypoints, 3]`` instead of 384-wide features.
    Everything folded in here is a fixed linear map or a parameter-free encoding;
    leaving them outside would force the runtime to carry two GEMMs and reproduce
    the positional encoding, for no benefit.

    The sampler loop, the classifier-free-guidance blend and the Euler update
    stay outside -- they are control flow, not compute.
    """

    def __init__(self, cfg: Optional[TrajDitConfig] = None,
                 action_dim: int = 3) -> None:
        super().__init__()
        cfg = cfg or TrajDitConfig()
        self.config = cfg
        self.action_encoder = nn.Linear(action_dim, cfg.dim, bias=True)
        self.pos_encoding = SinusoidalPositionalEncoding(cfg.dim)
        self.traj_dit = InternVLAN1TrajDit(cfg)
        self.action_decoder = nn.Linear(cfg.dim, action_dim, bias=True)

    def forward(self, latents: torch.Tensor, timestep: torch.Tensor,
                z_latents: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(latents.shape[1], device=latents.device)
        positions = positions.reshape(1, -1).expand(latents.shape[0], -1)
        feats = self.action_encoder(latents) + self.pos_encoding(positions).to(
            latents.dtype)
        return self.action_decoder(self.traj_dit(feats, timestep, z_latents))


#: Checkpoint prefixes for the projections folded into :class:`InternVLAN1TrajDitStep`.
STEP_PREFIXES = {
    "action_encoder": "model.action_encoder.",
    "action_decoder": "model.action_decoder.",
}


def build_internvla_n1_traj_dit_step(weights: dict,
                                     cfg: Optional[TrajDitConfig] = None,
                                     dtype: torch.dtype = torch.bfloat16
                                     ) -> InternVLAN1TrajDitStep:
    """Build the full denoising step and load it, refusing a partial load."""
    model = InternVLAN1TrajDitStep(cfg).to(dtype).eval()
    report = load_traj_dit_weights(model.traj_dit, weights)
    if report["missing"] or report["unexpected"]:
        raise ValueError(
            "InternVLA-N1 traj_dit did not load cleanly: "
            f"missing={report['missing'][:5]} "
            f"unexpected={report['unexpected'][:5]}")
    for attr, prefix in STEP_PREFIXES.items():
        module = getattr(model, attr)
        for suffix in ("weight", "bias"):
            key = prefix + suffix
            if key not in weights:
                raise ValueError(f"InternVLA-N1: checkpoint is missing {key}")
            param = getattr(module, suffix)
            param.data.copy_(weights[key].to(param.dtype))
    return model


#: Checkpoint prefix for the trajectory expert. The doubled ``model`` is the
#: reference wrapper (``NextDiTCrossAttn.model``), not a typo.
TRAJ_DIT_PREFIX = "model.traj_dit.model."

#: Present in the checkpoint, unreachable in ``forward`` -- see the module
#: docstring. Listed so the loader can account for every tensor it skips rather
#: than dropping unknown keys silently.
DEAD_TENSORS = ("patch_embedder.proj.weight", "patch_embedder.proj.bias")


def load_traj_dit_weights(model: InternVLAN1TrajDit, weights: dict) -> dict:
    """Load ``model.traj_dit.*`` into *model*; return a load report.

    Every checkpoint tensor is accounted for as loaded, dead, or unexpected, and
    every model parameter must be covered. A quiet partial load leaves the expert
    partly random and still exports and runs, which is the failure this guards.
    """
    own = dict(model.named_parameters())
    loaded, dead, unexpected = [], [], []
    for key, tensor in weights.items():
        if not key.startswith(TRAJ_DIT_PREFIX):
            continue
        name = key[len(TRAJ_DIT_PREFIX):]
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

    missing = sorted(set(own) - set(loaded))
    return {
        "loaded": len(loaded),
        "dead": sorted(dead),
        "unexpected": sorted(unexpected),
        "missing": missing,
    }


def build_internvla_n1_traj_dit(weights: dict,
                                cfg: Optional[TrajDitConfig] = None,
                                dtype: torch.dtype = torch.bfloat16
                                ) -> InternVLAN1TrajDit:
    """Build the trajectory expert and load it, refusing a partial load."""
    model = InternVLAN1TrajDit(cfg).to(dtype).eval()
    report = load_traj_dit_weights(model, weights)
    if report["missing"] or report["unexpected"]:
        raise ValueError(
            "InternVLA-N1 traj_dit did not load cleanly: "
            f"{len(report['missing'])} parameter(s) never assigned, "
            f"{len(report['unexpected'])} checkpoint key(s) unmatched. "
            f"missing={report['missing'][:5]} "
            f"unexpected={report['unexpected'][:5]}")
    return model


__all__ = [
    "InternVLAN1TrajDit",
    "InternVLAN1TrajDitStep",
    "build_internvla_n1_traj_dit_step",
    "TrajDitConfig",
    "build_internvla_n1_traj_dit",
    "load_traj_dit_weights",
    "TRAJ_DIT_PREFIX",
]
