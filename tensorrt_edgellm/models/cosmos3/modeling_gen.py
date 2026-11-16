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
"""Cosmos3 GEN (diffusion) expert for ONNX export.

Wraps ONE flow-matching denoising step of the Cosmos3-Omni generation pathway:

    patchify(video) + proj_in           ->  video tokens
    action_proj_in(action) + modality   ->  action tokens
    + time_embed (added directly, NO adaLN)
    -> N GEN cross-attention layers (each attends to [k_und ; k_gen])
    -> norm_moe_gen
    -> proj_out + unpatchify             ->  video_pred (velocity)
    -> action_proj_out                   ->  action_pred (velocity)

The host runtime owns the UniPC flow-matching scheduler step; this graph only
emits the model prediction.  The per-layer UND key/value tensors are *inputs*
(step-invariant: the runtime computes them once with the understanding tower and
binds them once for the whole denoising loop).

Attention uses the TRT-native ``trt::rope_onnx`` / ``trt::attention_onnx`` ops
(default-domain RotaryEmbedding / Attention nodes), non-causal. Numerics follow
the reference ``transformer_cosmos3.py`` implementation
(Cosmos3CrossAttention / Cosmos3GenDecoderLayer).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy_geometry import (DEFAULT_ACTION_CHUNK_SIZE, DEFAULT_DOMAIN,
                              DEFAULT_DOMAIN_ID, DEFAULT_FPS,
                              DEFAULT_RAW_ACTION_DIM,
                              TEMPORAL_COMPRESSION_FACTOR)

logger = logging.getLogger(__name__)


@dataclass
class Cosmos3GenConfig:
    """Hyperparameters for the Cosmos3 GEN expert (one denoising step)."""

    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9216
    rms_norm_eps: float = 1e-6
    hidden_act: str = "relu2"

    # Diffusion / latent geometry.
    latent_channel: int = 48
    latent_patch_size: int = 2
    max_action_dim: int = 64
    action_chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE
    raw_action_dim: int = DEFAULT_RAW_ACTION_DIM
    state_rows: int = 0
    num_embodiment_domains: int = 32

    # Timestep embedder.
    frequency_embedding_size: int = 256
    timestep_max_period: int = 10000
    timestep_scale: float = 0.001

    # Video latent grid [t, h, w]; the graph is dynamic in t only, so h and w
    # are baked into the exported model.
    latent_t: int = 5
    latent_h: int = 34
    latent_w: int = 46

    # Embodiment domain baked into action_proj_in / action_proj_out at export.
    domain_id: int = DEFAULT_DOMAIN_ID

    @property
    def patch_latent_dim(self) -> int:
        return self.latent_patch_size * self.latent_patch_size * self.latent_channel

    @property
    def hp(self) -> int:
        p = self.latent_patch_size
        return ((self.latent_h + p - 1) // p)

    @property
    def wp(self) -> int:
        p = self.latent_patch_size
        return ((self.latent_w + p - 1) // p)

    @property
    def num_video_tokens(self) -> int:
        return self.latent_t * self.hp * self.wp

    @property
    def num_gen_tokens(self) -> int:
        return self.num_video_tokens + self.action_token_count

    @property
    def action_token_count(self) -> int:
        return self.action_chunk_size + self.state_rows


class RMSNorm(nn.Module):
    """T5-style RMSNorm computed in fp32 (matches the reference ``F.rms_norm``)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.type(dtype)) * self.weight


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding -> Linear -> SiLU -> Linear (no adaLN)."""

    def __init__(self, cfg: Cosmos3GenConfig) -> None:
        super().__init__()
        self.frequency_embedding_size = cfg.frequency_embedding_size
        self.linear_1 = nn.Linear(cfg.frequency_embedding_size,
                                  cfg.hidden_size,
                                  bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        half = cfg.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(cfg.timestep_max_period) *
                          torch.arange(0, half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t[:, None].float() * self.freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.linear_2(
            self.act(self.linear_1(t_freq.type_as(self.linear_1.weight))))


class Cosmos3GenAttention(nn.Module):
    """GEN cross-attention: GEN-token Q attends to ``[k_und ; k_gen]`` (non-causal)."""

    def __init__(self, cfg: Cosmos3GenConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.to_q = nn.Linear(cfg.hidden_size,
                              self.num_heads * self.head_dim,
                              bias=False)
        self.to_k = nn.Linear(cfg.hidden_size,
                              self.num_kv_heads * self.head_dim,
                              bias=False)
        self.to_v = nn.Linear(cfg.hidden_size,
                              self.num_kv_heads * self.head_dim,
                              bias=False)
        self.to_out = nn.Linear(self.num_heads * self.head_dim,
                                cfg.hidden_size,
                                bias=False)
        self.norm_q = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.norm_k = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.qk_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        from tensorrt_edgellm.models.ops import attention_onnx, rope_onnx

        bsz, s_gen, _ = hidden_states.shape
        io_type = hidden_states.dtype
        compute_type = torch.float16

        q = self.to_q(hidden_states).view(bsz, s_gen, self.num_heads,
                                          self.head_dim).transpose(1, 2)
        k = self.to_k(hidden_states).view(bsz, s_gen, self.num_kv_heads,
                                          self.head_dim).transpose(1, 2)
        v = self.to_v(hidden_states).view(bsz, s_gen, self.num_kv_heads,
                                          self.head_dim).transpose(1, 2)

        # Per-head QK-norm (GEN always has qk-norm: qk_norm_for_diffusion=True).
        q = self.norm_q(q)
        k = self.norm_k(k)

        # Qwen3-style RoPE on the GEN-token positions (unified_3d_mrope cos/sin).
        q = rope_onnx(q.to(compute_type), rope_cos, rope_sin,
                      position_ids).to(io_type)
        k = rope_onnx(k.to(compute_type), rope_cos, rope_sin,
                      position_ids).to(io_type)

        q = q * self.qk_scale

        # UND context K/V arrive seq-major [B, S_und, H_kv, D] (already post-norm/
        # RoPE from the understanding tower); transpose to [B, H_kv, S_und, D] and
        # concatenate in front of the freshly computed GEN K/V along the seq axis
        # (reference Cosmos3CrossAttention._forward_local: cat([k_und, k_gen])).
        k_und = k_und.transpose(1, 2).to(k.dtype)
        v_und = v_und.transpose(1, 2).to(v.dtype)
        k_all = torch.cat([k_und, k], dim=2)
        v_all = torch.cat([v_und, v], dim=2)

        attn_output = attention_onnx(q,
                                     k_all,
                                     v_all,
                                     attn_mask=None,
                                     is_causal=False,
                                     scale=1.0)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, s_gen, -1)
        return self.to_out(attn_output)


class Cosmos3GenMLP(nn.Module):
    """Non-gated MLP ``down(act(up(x)))`` — the GEN expert is the same
    Nemotron-H block as the text tower (``hidden_act='relu2'`` = squared
    ReLU); the checkpoint schema has exactly ``up_proj`` / ``down_proj``."""

    def __init__(self, cfg: Cosmos3GenConfig) -> None:
        super().__init__()
        if cfg.hidden_act == "relu2":
            self.act_fn = lambda x: F.relu(x).square()
        elif cfg.hidden_act == "silu":
            self.act_fn = F.silu
        else:
            raise ValueError(
                f"Unsupported Cosmos3 GEN hidden_act: {cfg.hidden_act!r}")
        self.up_proj = nn.Linear(cfg.hidden_size,
                                 cfg.intermediate_size,
                                 bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size,
                                   cfg.hidden_size,
                                   bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(x)))


class Cosmos3GenDecoderLayer(nn.Module):
    """Pre-norm GEN block: cross-attention (to UND K/V) + MLP."""

    def __init__(self, cfg: Cosmos3GenConfig) -> None:
        super().__init__()
        self.cross_attention = Cosmos3GenAttention(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size,
                                                eps=cfg.rms_norm_eps)
        self.mlp = Cosmos3GenMLP(cfg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.cross_attention(hidden_states, k_und, v_und,
                                             rope_cos, rope_sin, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class Cosmos3Gen(nn.Module):
    """One Cosmos3 GEN denoising step for ONNX export.

    The action embodiment domain is baked at construction (``cfg.domain_id``):
    ``action_proj_in`` / ``action_proj_out`` become plain matmul + bias for that
    domain (DomainAwareLinear row gather folded into constants).
    """

    def __init__(self, cfg: Cosmos3GenConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj_in = nn.Linear(cfg.patch_latent_dim,
                                 cfg.hidden_size,
                                 bias=True)
        self.proj_out = nn.Linear(cfg.hidden_size,
                                  cfg.patch_latent_dim,
                                  bias=True)
        self.time_embedder = TimestepEmbedder(cfg)
        self.action_modality_embed = nn.Parameter(torch.zeros(cfg.hidden_size))
        # Baked DomainAwareLinear (domain row folded in _load_gen_weights):
        #   action_in:  x[.,max_action_dim] @ W_in[max_action_dim, hidden] + b_in[hidden]
        #   action_out: x[.,hidden]         @ W_out[hidden, max_action_dim] + b_out[max_action_dim]
        self.action_in_weight = nn.Parameter(
            torch.zeros(cfg.max_action_dim, cfg.hidden_size))
        self.action_in_bias = nn.Parameter(torch.zeros(cfg.hidden_size))
        self.action_out_weight = nn.Parameter(
            torch.zeros(cfg.hidden_size, cfg.max_action_dim))
        self.action_out_bias = nn.Parameter(torch.zeros(cfg.max_action_dim))
        self.layers = nn.ModuleList([
            Cosmos3GenDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)
        ])
        self.norm_moe_gen = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    # -- patchify / unpatchify (reference transformer_cosmos3.patchify) ------
    def _patchify(self, latents: torch.Tensor) -> torch.Tensor:
        b, c, t, _, _ = latents.shape
        p = self.cfg.latent_patch_size
        hp, wp = self.cfg.hp, self.cfg.wp
        pad_h = hp * p - self.cfg.latent_h
        pad_w = wp * p - self.cfg.latent_w
        if pad_h or pad_w:
            latents = F.pad(latents, (0, pad_w, 0, pad_h))
        x = latents.reshape(b, c, t, hp, p, wp, p)
        x = x.permute(0, 2, 3, 5, 4, 6, 1)  # [B, t, hp, wp, p, p, C]
        return x.reshape(b, t * hp * wp, p * p * c)

    def _unpatchify(self, tokens: torch.Tensor, t: int) -> torch.Tensor:
        b = tokens.shape[0]
        p, c = self.cfg.latent_patch_size, self.cfg.latent_channel
        hp, wp = self.cfg.hp, self.cfg.wp
        x = tokens.reshape(b, t, hp, wp, p, p, c)
        x = x.permute(0, 6, 1, 2, 4, 3, 5)  # [B, C, t, hp, p, wp, p]
        x = x.reshape(b, c, t, hp * p, wp * p)
        return x[:, :, :, :self.cfg.latent_h, :self.cfg.latent_w]

    def forward(
        self,
        video_latent: torch.Tensor,
        action_latent: torch.Tensor,
        timestep: torch.Tensor,
        token_noisy_mask: torch.Tensor,
        action_noisy_mask: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        attention_pos_id: torch.Tensor,
        *und_kv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        n = cfg.num_hidden_layers
        k_und = und_kv[:n]
        v_und = und_kv[n:]
        io_type = self.proj_in.weight.dtype

        video_latent = video_latent.to(io_type)
        action_latent = action_latent.to(io_type)
        t_lat = video_latent.shape[2]

        # Video tokens.
        video_tokens = self.proj_in(self._patchify(video_latent))
        # Action tokens (baked domain matmul) + modality embedding.
        action_tokens = torch.matmul(
            action_latent, self.action_in_weight) + self.action_in_bias
        action_tokens = action_tokens + self.action_modality_embed

        # Timestep added directly to noisy tokens (clean tokens masked to zero).
        t_embed = self.time_embedder(timestep.float() *
                                     cfg.timestep_scale).to(io_type)
        video_tokens = video_tokens + t_embed[:,
                                              None, :] * token_noisy_mask.to(
                                                  io_type)
        action_tokens = action_tokens + t_embed[:,
                                                None, :] * action_noisy_mask.to(
                                                    io_type)

        hidden = torch.cat([video_tokens, action_tokens], dim=1)

        half = cfg.head_dim // 2
        rope_cos = rope_rotary_cos_sin[..., :half].reshape(-1, half).to(
            torch.float16)
        rope_sin = rope_rotary_cos_sin[...,
                                       half:].reshape(-1,
                                                      half).to(torch.float16)

        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, k_und[i], v_und[i], rope_cos, rope_sin,
                           attention_pos_id)

        hidden = self.norm_moe_gen(hidden)
        s_video = video_tokens.shape[1]
        video_part = hidden[:, :s_video, :]
        action_part = hidden[:, s_video:, :]

        video_pred = self._unpatchify(self.proj_out(video_part),
                                      t_lat).to(torch.float32)
        action_pred = (torch.matmul(action_part, self.action_out_weight) +
                       self.action_out_bias).to(torch.float32)
        return video_pred, action_pred

    def get_onnx_export_args(self, max_und_len: int,
                             device: str) -> Tuple[tuple, list, list, tuple]:
        cfg = self.cfg
        n, hkv, d = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
        # Sample batch of 2 keeps the batch axis symbolic (torch.export
        # specializes size-1 dims), so engines can be built for any max batch.
        b = 2
        s_video = cfg.num_video_tokens
        s_gen = cfg.num_gen_tokens

        video_latent = torch.randn(b,
                                   cfg.latent_channel,
                                   cfg.latent_t,
                                   cfg.latent_h,
                                   cfg.latent_w,
                                   device=device,
                                   dtype=torch.float32)
        action_latent = torch.randn(b,
                                    cfg.action_token_count,
                                    cfg.max_action_dim,
                                    device=device,
                                    dtype=torch.float32)
        timestep = torch.tensor([500.0], device=device,
                                dtype=torch.float32).repeat(b)
        token_noisy_mask = torch.ones(b,
                                      s_video,
                                      1,
                                      device=device,
                                      dtype=torch.float32)
        action_noisy_mask = torch.ones(b,
                                       cfg.action_token_count,
                                       1,
                                       device=device,
                                       dtype=torch.float32)
        rope_cos_sin = torch.randn(b,
                                   s_gen,
                                   d,
                                   device=device,
                                   dtype=torch.float32)
        pos_ids = torch.arange(s_gen, device=device,
                               dtype=torch.int32).unsqueeze(0).expand(b, -1)

        # UND K/V arrive seq-major: [B, S_und, H_kv, D] (matches the reference
        # dump und_k_layerNN / und_v_layerNN and the runtime KV repack).
        und_shape = (b, max_und_len, hkv, d)
        k_und = [
            torch.randn(und_shape, device=device, dtype=torch.float16)
            for _ in range(n)
        ]
        v_und = [
            torch.randn(und_shape, device=device, dtype=torch.float16)
            for _ in range(n)
        ]

        args = tuple([
            video_latent, action_latent, timestep, token_noisy_mask,
            action_noisy_mask, rope_cos_sin, pos_ids
        ] + k_und + v_und)
        input_names = ([
            "video_latent",
            "action_latent",
            "timestep",
            "token_noisy_mask",
            "action_noisy_mask",
            "rope_rotary_cos_sin",
            "attention_position_ids",
        ] + [f"und_k_layer{i:02d}"
             for i in range(n)] + [f"und_v_layer{i:02d}" for i in range(n)])
        output_names = ["video_pred", "action_pred"]

        batch = torch.export.Dim("batch_size", min=1, max=256)
        und = torch.export.Dim("und_len", min=1, max=131072)
        action = torch.export.Dim("action_len", min=0, max=4096)
        gen = torch.export.Dim("gen_len", min=1, max=262144)
        vtok = torch.export.Dim("video_tokens", min=1, max=262144)
        # Temporal extent is dynamic; the engine profile owns a fixed spatial grid.
        lt = torch.export.Dim("latent_t", min=1, max=256)
        dynamic_shapes = (
            {
                0: batch,
                2: lt
            },  # video_latent [B,C,t,h,w]
            {
                0: batch,
                1: action
            },  # action_latent
            {
                0: batch
            },  # timestep
            {
                0: batch,
                1: vtok
            },  # token_noisy_mask [B, S_video, 1]
            {
                0: batch,
                1: action
            },  # action_noisy_mask
            {
                0: batch,
                1: gen
            },  # rope_rotary_cos_sin  (S_gen = S_video + S_action)
            {
                0: batch,
                1: gen
            },  # attention_pos_id
            tuple({
                0: batch,
                1: und
            } for _ in range(n * 2)),  # *und_kv
        )
        return args, input_names, output_names, dynamic_shapes


def _load_gen_weights(model: Cosmos3Gen, weights: dict,
                      dtype: torch.dtype) -> int:
    """Assign GEN weights from the split ``{key: tensor}`` dict (see weights_cosmos3).

    Expected keys (already remapped by the splitter to GEN naming):
      proj_in.{weight,bias}, proj_out.{weight,bias}
      time_embedder.linear_1.{weight,bias}, time_embedder.linear_2.{weight,bias}
      action_modality_embed
      action_proj_in.fc / action_proj_in.bias       (DomainAwareLinear tables)
      action_proj_out.fc / action_proj_out.bias
      norm_moe_gen.weight
      layers.{i}.cross_attention.{to_q,to_k,to_v,to_out,norm_q,norm_k}.weight
      layers.{i}.input_layernorm.weight, layers.{i}.post_attention_layernorm.weight
      layers.{i}.mlp.{up_proj,down_proj[,gate_proj]}.weight
    """
    cfg = model.cfg
    d = cfg.domain_id
    loaded = 0
    assigned: set = set()

    def assign(path: str, tensor: torch.Tensor) -> None:
        nonlocal loaded
        obj = model
        parts = path.split(".")
        for p in parts[:-1]:
            obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
        param = getattr(obj, parts[-1])
        t = tensor.to(dtype) if tensor.is_floating_point() else tensor
        param.data.copy_(t)
        assigned.add(path)
        loaded += 1

    for key, tensor in weights.items():
        if key == "action_modality_embed":
            assign("action_modality_embed", tensor.reshape(-1))
        elif key in ("action_proj_in.fc", "action_proj_in.fc.weight"):
            w = tensor.reshape(cfg.num_embodiment_domains, cfg.max_action_dim,
                               cfg.hidden_size)[d]
            assign("action_in_weight", w)
        elif key in ("action_proj_in.bias", "action_proj_in.bias.weight"):
            assign(
                "action_in_bias",
                tensor.reshape(cfg.num_embodiment_domains, cfg.hidden_size)[d])
        elif key in ("action_proj_out.fc", "action_proj_out.fc.weight"):
            w = tensor.reshape(cfg.num_embodiment_domains, cfg.hidden_size,
                               cfg.max_action_dim)[d]
            assign("action_out_weight", w)
        elif key in ("action_proj_out.bias", "action_proj_out.bias.weight"):
            assign(
                "action_out_bias",
                tensor.reshape(cfg.num_embodiment_domains,
                               cfg.max_action_dim)[d])
        else:
            try:
                assign(key, tensor)
            except (AttributeError, IndexError):
                raise KeyError(
                    f"Cosmos3 GEN checkpoint key {key!r} matches no model "
                    "parameter") from None

    # Every model parameter must have received a checkpoint tensor: a silently
    # random-initialized module (e.g. a structural mismatch with the
    # checkpoint schema) corrupts the denoising output.
    param_names = {n
                   for n, _ in model.named_parameters()} | {
                       "action_in_weight", "action_in_bias",
                       "action_out_weight", "action_out_bias"
                   }
    missing = sorted(param_names - assigned)
    if missing:
        raise KeyError(
            "Cosmos3 GEN parameters received no checkpoint tensor: " +
            ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))
    logger.info("Loaded %d Cosmos3 GEN tensors (missing=0, unexpected=0)",
                loaded)
    return loaded


def build_cosmos3_gen(cfg: Cosmos3GenConfig, weights: dict,
                      dtype: torch.dtype) -> Cosmos3Gen:
    model = Cosmos3Gen(cfg).to(dtype)
    _load_gen_weights(model, weights, dtype)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Component contract (consumed by the experimental Cosmos3 builder/runtime)
# ---------------------------------------------------------------------------

# Largest video-subsample factor the GEN engine's dynamic profile must admit. The
# regular request (vsf=1) is the profile opt/max; the vsf-reduced request is the
# profile min. See _latent_t_for_subsample.
DEFAULT_MAX_VIDEO_SUBSAMPLE_FACTOR = 4
DEFAULT_NUM_INFERENCE_STEPS = 4
DEFAULT_FLOW_SHIFT = 5.0


def gen_config_from_transformer(
        tcfg: dict,
        action_chunk_size: "int | None" = None,
        num_frames: "int | None" = None,
        raw_action_dim: int = DEFAULT_RAW_ACTION_DIM,
        state_rows: int = 0,
        latent_height: "int | None" = None,
        latent_width: "int | None" = None,
        domain_id: "int | None" = None) -> Cosmos3GenConfig:
    """Build the GEN config object from a Cosmos3 transformer config.

    ``action_chunk_size`` / ``num_frames`` are request-time parameters (the
    canonical policy request uses 16 / 17); ``num_frames`` sets the temporal
    latent extent ``latent_t = (num_frames - 1) // 4 + 1``.
    """
    if num_frames is not None and (num_frames -
                                   1) % TEMPORAL_COMPRESSION_FACTOR != 0:
        raise ValueError(
            f"num_frames must be 4k+1 (got {num_frames}); the VAE compresses "
            "time 4x with a single leading conditioning frame.")
    return Cosmos3GenConfig(
        hidden_size=tcfg["hidden_size"],
        num_hidden_layers=tcfg["num_hidden_layers"],
        num_attention_heads=tcfg["num_attention_heads"],
        num_key_value_heads=tcfg["num_key_value_heads"],
        head_dim=tcfg["head_dim"],
        intermediate_size=tcfg["intermediate_size"],
        rms_norm_eps=tcfg.get("rms_norm_eps", 1e-6),
        # Required: the activation defines the MLP graph (Cosmos3-Edge uses
        # the Nemotron-H squared-ReLU 'relu2'); no silent default.
        hidden_act=str(tcfg["hidden_act"]),
        latent_channel=tcfg.get("latent_channel", 48),
        latent_patch_size=tcfg.get("latent_patch_size", 2),
        max_action_dim=tcfg.get("max_action_dim", 64),
        num_embodiment_domains=tcfg.get("num_embodiment_domains", 32),
        timestep_scale=tcfg.get("timestep_scale", 0.001),
        action_chunk_size=(action_chunk_size if action_chunk_size is not None
                           else tcfg.get("action_chunk_size",
                                         DEFAULT_ACTION_CHUNK_SIZE)),
        raw_action_dim=raw_action_dim,
        state_rows=state_rows,
        latent_t=((num_frames - 1) // TEMPORAL_COMPRESSION_FACTOR +
                  1 if num_frames is not None else Cosmos3GenConfig.latent_t),
        latent_h=(latent_height
                  if latent_height is not None else Cosmos3GenConfig.latent_h),
        latent_w=(latent_width
                  if latent_width is not None else Cosmos3GenConfig.latent_w),
        domain_id=(domain_id if domain_id is not None else tcfg.get(
            "domain_id", DEFAULT_DOMAIN_ID)),
    )


def _latent_t_for_subsample(
        action_chunk_size: int,
        video_subsample_factor: int,
        temporal_compression: int = TEMPORAL_COMPRESSION_FACTOR) -> int:
    """Latent temporal planes for a video-subsample factor.

    ``t_frames = action_chunk_size // vsf + 1`` generated frames, compressed 4x
    by the Wan VAE with a single leading conditioning frame:
    ``latent_t = (t_frames - 1) // 4 + 1`` (regular vsf=1, optimized vsf=4).
    """
    t_frames = action_chunk_size // video_subsample_factor + 1
    return (t_frames - 1) // temporal_compression + 1


def make_gen_config(
        cfg: Cosmos3GenConfig,
        tcfg: dict,
        max_und_len: int,
        fps: float = DEFAULT_FPS,
        domain: "str | None" = DEFAULT_DOMAIN,
        max_video_subsample_factor: int = DEFAULT_MAX_VIDEO_SUBSAMPLE_FACTOR,
        min_action_chunk: "int | None" = None,
        max_action_chunk: "int | None" = None) -> dict:
    """Return the GEN component ``config.json`` payload."""
    rope_scaling = tcfg.get("rope_scaling", {}) or {}
    v_tok = cfg.num_video_tokens
    g_tok = cfg.num_gen_tokens
    n_kv = cfg.num_key_value_heads
    hd = cfg.head_dim
    mad = cfg.max_action_dim
    # Action-token axis is DYNAMIC over [min_action .. max_action]; the canonical
    # chunk (cfg.action_chunk_size) is the opt point, so the regular request stays
    # the optimization target. Defaults keep min == opt == max = the canonical
    # chunk (action axis fixed, behavior identical) unless the caller widens it.
    action_chunk = cfg.action_chunk_size
    max_action_chunk = max(
        action_chunk,
        max_action_chunk if max_action_chunk is not None else action_chunk)
    min_action_chunk = min(
        action_chunk,
        min_action_chunk if min_action_chunk is not None else action_chunk)
    action_len = cfg.action_token_count
    min_action = min_action_chunk + cfg.state_rows
    max_action = max_action_chunk + cfg.state_rows
    opt_und = min(32, max_und_len)

    # The video-token sequence axis is DYNAMIC: the regular request (vsf=1,
    # cfg.latent_t) is the profile opt/max, and the largest supported subsample
    # (vsf=max_video_subsample_factor, fewer frames) is the profile min. One
    # engine then serves any request whose video-token count lies in [min, max];
    # the runtime binds the per-request temporal extent (Cosmos3PolicyRunner).
    min_latent_t = min(
        _latent_t_for_subsample(action_chunk, max_video_subsample_factor),
        cfg.latent_t)
    min_v_tok = min_latent_t * cfg.hp * cfg.wp
    # GEN sequence = video tokens + action tokens; both axes vary, so the gen-len
    # profile spans (min video + min action) .. (max video + max action) with the
    # regular request (v_tok + canonical chunk) as opt.
    min_g_tok = min_v_tok + min_action
    opt_g_tok = v_tok + action_len
    max_g_tok = v_tok + max_action

    def _fix(shape: list) -> dict:
        return {"min": shape, "opt": shape, "max": shape}

    def _dyn(min_shape: list, opt_shape: list) -> dict:
        return {"min": min_shape, "opt": opt_shape, "max": opt_shape}

    def _range(min_shape: list, opt_shape: list, max_shape: list) -> dict:
        return {"min": min_shape, "opt": opt_shape, "max": max_shape}

    profile = {
        "video_latent":
        _dyn(
            [1, cfg.latent_channel, min_latent_t, cfg.latent_h, cfg.latent_w],
            [1, cfg.latent_channel, cfg.latent_t, cfg.latent_h, cfg.latent_w]),
        "action_latent":
        _range([1, min_action, mad], [1, action_len, mad],
               [1, max_action, mad]),
        "timestep":
        _fix([1]),
        "token_noisy_mask":
        _dyn([1, min_v_tok, 1], [1, v_tok, 1]),
        "action_noisy_mask":
        _range([1, min_action, 1], [1, action_len, 1], [1, max_action, 1]),
        "rope_rotary_cos_sin":
        _range([1, min_g_tok, hd], [1, opt_g_tok, hd], [1, max_g_tok, hd]),
        "attention_position_ids":
        _range([1, min_g_tok], [1, opt_g_tok], [1, max_g_tok]),
    }
    for i in range(cfg.num_hidden_layers):
        und = {
            "min": [1, 1, n_kv, hd],
            "opt": [1, opt_und, n_kv, hd],
            "max": [1, max_und_len, n_kv, hd],
        }
        profile[f"und_k_layer{i:02d}"] = und
        profile[f"und_v_layer{i:02d}"] = und

    return {
        "component": "gen",
        "onnx_filename": "model.onnx",
        "engine_filename": "gen.engine",
        "optimization_profile": profile,
        "tensor_contract": {
            "inputs": {
                "video_latent": ["batch", "latent_channel", "t", "h", "w"],
                "action_latent":
                ["batch", "action_chunk_size", "max_action_dim"],
                "und_k_layerNN":
                ["batch", "und_len", "num_kv_heads", "head_dim"],
                "und_v_layerNN":
                ["batch", "und_len", "num_kv_heads", "head_dim"],
            },
            "outputs": {
                "video_pred": ["batch", "latent_channel", "t", "h", "w"],
                "action_pred":
                ["batch", "action_chunk_size", "max_action_dim"],
            },
        },
        "builder_config": {
            "max_batch_size": 1,
            "max_und_len": max_und_len,
            "num_und_kv_inputs": cfg.num_hidden_layers * 2,
        },
        # Required: rope geometry is architecture-defining; no silent defaults.
        "rope_theta": float(tcfg["rope_theta"]),
        "rope_scaling": {
            "mrope_section": rope_scaling["mrope_section"],
            "mrope_interleaved": rope_scaling.get("mrope_interleaved", True),
            "rope_type": "mrope",
        },
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "rms_norm_eps": cfg.rms_norm_eps,
        "hidden_act": cfg.hidden_act,
        "latent_channel": cfg.latent_channel,
        "latent_patch_size": cfg.latent_patch_size,
        "num_video_tokens": cfg.num_video_tokens,
        "action_chunk_size": cfg.action_chunk_size,
        "action_token_count": cfg.action_token_count,
        "state_rows": cfg.state_rows,
        "history_length": cfg.state_rows,
        "use_state": cfg.state_rows > 0,
        "raw_action_dim": cfg.raw_action_dim,
        "max_action_dim": cfg.max_action_dim,
        "num_embodiment_domains": cfg.num_embodiment_domains,
        "domain": domain,
        "domain_id": cfg.domain_id,
        "timestep_scale": cfg.timestep_scale,
        "num_inference_steps": DEFAULT_NUM_INFERENCE_STEPS,
        "flow_shift": DEFAULT_FLOW_SHIFT,
        "video_latent_frames": cfg.latent_t,
        "min_video_latent_frames": min_latent_t,
        "max_video_subsample_factor": max_video_subsample_factor,
        "min_action_chunk_size": min_action_chunk,
        "max_action_chunk_size": max_action_chunk,
        "fps": float(fps),
        "base_fps": 24.0,
        "temporal_compression_factor": TEMPORAL_COMPRESSION_FACTOR,
        "temporal_modality_margin": 15000,
        "action_start_frame_offset": 0 if cfg.state_rows else 1,
    }
