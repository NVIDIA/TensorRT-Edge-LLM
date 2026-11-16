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
From-scratch Alpamayo action expert for dynamo ONNX export.

Wraps one flow-matching denoising step:
    action_in_proj -> expert transformer -> action_out_proj -> Euler step.

The expert uses TRT native attention ops (RotaryEmbedding, TensorScatter,
Attention in ONNX default domain) with non-causal masking.
"""

from __future__ import annotations

import logging
import math
from typing import List, Tuple

import torch
import torch.nn as nn

from ...config import ActionConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action input projection
# ---------------------------------------------------------------------------


class FourierEncoder(nn.Module):
    """Fourier feature encoder with logarithmically-spaced frequencies."""

    def __init__(self, dim: int, max_freq: float = 100.0) -> None:
        super().__init__()
        half = dim // 2
        freqs = torch.logspace(0, math.log10(max_freq), steps=half)
        self.out_dim = dim
        self.register_buffer("freqs", freqs[None, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arg = x[..., None] * self.freqs * 2 * torch.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


class MLPEncoder(nn.Module):
    """Basic MLP encoder."""

    def __init__(self, num_input_feats: int, num_enc_layers: int,
                 hidden_size: int, outdim: int) -> None:
        super().__init__()
        assert 1 <= num_enc_layers, f"{num_enc_layers=} must be >= 1"
        layers: list[nn.Module] = [
            nn.Linear(num_input_feats, hidden_size),
            nn.SiLU(),
        ]
        for i in range(num_enc_layers):
            if i < num_enc_layers - 1:
                layers.extend([
                    RMSNorm(hidden_size, eps=1e-5),
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                ])
            else:
                layers.extend([
                    RMSNorm(hidden_size, eps=1e-5),
                    nn.Linear(hidden_size, outdim),
                ])
        self.trunk = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class ActionInProjection(nn.Module):
    """Per-waypoint action input projection module.

    It uses FourierEncoder with logarithmically-spaced frequencies and includes layer normalization. Projects
    action sequences with timestep information into a higher-dimensional representation.
    """

    def __init__(self, cfg: ActionConfig) -> None:
        super().__init__()
        n_action_dims = 2  # (accel, curvature)
        nf = cfg.in_proj_num_fourier_feats
        self.sinus = nn.ModuleList([
            FourierEncoder(dim=nf, max_freq=cfg.in_proj_max_freq)
            for _ in range(n_action_dims)
        ])
        self.timestep_fourier_encoder = FourierEncoder(
            dim=nf, max_freq=cfg.in_proj_max_freq)
        num_input_feats = sum(
            s.out_dim
            for s in self.sinus) + self.timestep_fourier_encoder.out_dim
        self.encoder = MLPEncoder(
            num_input_feats=num_input_feats,
            num_enc_layers=cfg.in_proj_num_enc_layers,
            hidden_size=cfg.in_proj_hidden_size,
            outdim=cfg.hidden_size,
        )
        self.norm = nn.LayerNorm(cfg.hidden_size)

    def forward(self, x: torch.Tensor,
                timesteps: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        action_feats = torch.cat(
            [s(x[:, :, i]) for i, s in enumerate(self.sinus)], dim=-1)
        timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1])
        timestep_feats = timestep_feats.repeat(1, T, 1)
        x = torch.cat((action_feats, timestep_feats), dim=-1)
        return self.norm(self.encoder(x.flatten(0, 1)).reshape(B, T, -1))


# ---------------------------------------------------------------------------
# Expert transformer (TRT native attention path)
# ---------------------------------------------------------------------------


class ActionAttention(nn.Module):
    """TRT native attention for the action expert (non-causal).

    Uses trt::rope_onnx, trt::kv_cache_update_onnx, trt::attention_onnx
    custom ops that emit RotaryEmbedding, TensorScatter, Attention ONNX
    nodes in the default domain.
    """

    def __init__(self, cfg: ActionConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.hidden_size = cfg.hidden_size
        self.attention_scale = float(cfg.attention_scaling)

        self.q_proj = nn.Linear(cfg.hidden_size,
                                self.num_heads * self.head_dim,
                                bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size,
                                self.num_kv_heads * self.head_dim,
                                bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size,
                                self.num_kv_heads * self.head_dim,
                                bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                cfg.hidden_size,
                                bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        position_ids: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from ..ops import attention_onnx, kv_cache_update_onnx, rope_onnx

        bsz, q_len, _ = hidden_states.shape
        io_type = hidden_states.dtype
        compute_type = torch.float16

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads,
                   self.head_dim).transpose(1, 2)

        # QK-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        q = rope_onnx(q.to(compute_type), rope_cos, rope_sin, position_ids)
        k = rope_onnx(k.to(compute_type), rope_cos, rope_sin, position_ids)

        q = q.to(io_type)
        k = k.to(io_type)
        v = v.to(io_type)

        # KV cache update
        present_k = kv_cache_update_onnx(k_cache, k, kvcache_start_index)
        present_v = kv_cache_update_onnx(v_cache, v, kvcache_start_index)

        # Pass full cache to attention; the mask handles invalid positions.
        attn_output = attention_onnx(
            q,
            present_k,
            present_v,
            attn_mask=attention_mask,
            is_causal=False,
            scale=self.attention_scale,
        )

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, present_k, present_v


class ActionDecoderLayer(nn.Module):
    """Pre-norm transformer block for the action expert."""

    def __init__(self,
                 cfg: ActionConfig,
                 attention: nn.Module | None = None) -> None:
        super().__init__()
        self.self_attn = attention if attention is not None else ActionAttention(
            cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size,
                                                eps=cfg.rms_norm_eps)
        # SwiGLU MLP
        self.mlp_gate_proj = nn.Linear(cfg.hidden_size,
                                       cfg.intermediate_size,
                                       bias=False)
        self.mlp_up_proj = nn.Linear(cfg.hidden_size,
                                     cfg.intermediate_size,
                                     bias=False)
        self.mlp_down_proj = nn.Linear(cfg.intermediate_size,
                                       cfg.hidden_size,
                                       bias=False)

    def forward(self, hidden_states: torch.Tensor, *attention_inputs:
                torch.Tensor) -> tuple:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, *present_kv = self.self_attn(hidden_states,
                                                    *attention_inputs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        gate = self.mlp_gate_proj(hidden_states)
        up = self.mlp_up_proj(hidden_states)
        hidden_states = nn.functional.silu(gate) * up
        hidden_states = self.mlp_down_proj(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, *present_kv)


class ActionExpert(nn.Module):
    """Stack of ActionDecoderLayers + final RMSNorm."""

    def __init__(self, cfg: ActionConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [ActionDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        k_caches: Tuple[torch.Tensor, ...],
        v_caches: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        hidden_states = inputs_embeds
        present_ks: List[torch.Tensor] = []
        present_vs: List[torch.Tensor] = []
        for layer, k_cache, v_cache in zip(self.layers, k_caches, v_caches):
            hidden_states, pk, pv = layer(
                hidden_states,
                k_cache,
                v_cache,
                rope_cos,
                rope_sin,
                position_ids,
                kvcache_start_index,
                attention_mask,
            )
            present_ks.append(pk)
            present_vs.append(pv)
        hidden_states = self.norm(hidden_states)
        return hidden_states, present_ks, present_vs


# ---------------------------------------------------------------------------
# Top-level model (one flow-matching step)
# ---------------------------------------------------------------------------


class AlpamayoAction(nn.Module):
    """Wraps one flow-matching denoising step for ONNX export."""

    def __init__(self,
                 cfg: ActionConfig,
                 expert: nn.Module | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_layers = cfg.num_hidden_layers
        self.n_diffusion_tokens = cfg.n_diffusion_tokens
        self.action_in_proj = ActionInProjection(cfg)
        self.expert = expert if expert is not None else ActionExpert(cfg)
        self.action_out_proj = nn.Linear(cfg.hidden_size, 2, bias=True)

    def forward(
        self,
        noise_trajectory: torch.Tensor,
        time_steps_t0: torch.Tensor,
        time_steps_t1: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        attention_pos_id: torch.Tensor,
        *cache_tensors: torch.Tensor,
    ) -> tuple:
        n_layers = self.n_layers
        k_caches = tuple(cache_tensors[:n_layers])
        v_caches = tuple(cache_tensors[n_layers:])
        bsz = noise_trajectory.shape[0]

        noise_trajectory = noise_trajectory.to(torch.float16)
        time_steps_t0 = time_steps_t0.to(torch.float16)
        time_steps_t1 = time_steps_t1.to(torch.float16)

        # Expand dt and t_start to (B, 1, 1) matching action_space_dims
        dt = (time_steps_t1 - time_steps_t0).view(-1, 1, 1).expand(bsz, 1, 1)
        t_start = time_steps_t0.view(-1, 1, 1).expand(bsz, 1, 1)

        embeds = self.action_in_proj(noise_trajectory, t_start)

        # Split rope_rotary_cos_sin into cos and sin
        half_dim = self.cfg.head_dim // 2
        rope_cos = rope_rotary_cos_sin[:, :, :half_dim]
        rope_sin = rope_rotary_cos_sin[:, :, half_dim:]
        rope_cos = rope_cos.reshape(-1, half_dim).to(torch.float16)
        rope_sin = rope_sin.reshape(-1, half_dim).to(torch.float16)
        # RotaryEmbedding indexes a shared 2D table; select each request's rows.
        rope_row_offsets = torch.arange(
            bsz, device=attention_pos_id.device,
            dtype=attention_pos_id.dtype).view(-1, 1) * self.n_diffusion_tokens
        attention_pos_id = attention_pos_id + rope_row_offsets

        # Non-causal attention mask: 0.0 for valid, -inf beyond valid length
        max_kv_capacity = k_caches[0].shape[2]
        q_len = self.n_diffusion_tokens
        valid_lengths = (kvcache_start_index + q_len).view(-1, 1, 1, 1)
        position_indices = torch.arange(
            max_kv_capacity,
            device=noise_trajectory.device,
            dtype=torch.int32,
        ).view(1, 1, 1, -1)
        mask = torch.full(
            (bsz, 1, q_len, max_kv_capacity),
            0.0,
            device=noise_trajectory.device,
            dtype=torch.float16,
        )
        mask = mask.masked_fill(position_indices >= valid_lengths,
                                float("-inf"))

        hidden, present_ks, present_vs = self.expert(
            embeds,
            kvcache_start_index,
            rope_cos,
            rope_sin,
            attention_pos_id,
            mask,
            k_caches,
            v_caches,
        )

        pred = self.action_out_proj(hidden[:, -self.n_diffusion_tokens:]).view(
            bsz, self.n_diffusion_tokens, 2)
        denoised = (noise_trajectory + dt * pred).to(torch.float32)

        return (denoised, *present_ks, *present_vs)

    def get_onnx_export_args(
        self,
        max_kv_cache_capacity: int,
        device: str,
    ) -> Tuple[tuple, list, list, dict]:
        """Build dummy inputs, names, and dynamic shapes for ONNX export."""
        cfg = self.cfg
        n_layers = cfg.num_hidden_layers
        n_kv_heads = cfg.num_key_value_heads
        head_dim = cfg.head_dim
        n_diff = cfg.n_diffusion_tokens
        B = 1

        noise_trajectory = torch.randn(B,
                                       n_diff,
                                       2,
                                       device=device,
                                       dtype=torch.float32)
        t0 = torch.tensor([0.0], device=device, dtype=torch.float32)
        t1 = torch.tensor([0.1], device=device, dtype=torch.float32)
        kvcache_start_index = torch.full((B, ),
                                         3027,
                                         device=device,
                                         dtype=torch.int32)
        rope_cos_sin = torch.randn(B,
                                   n_diff,
                                   head_dim,
                                   device=device,
                                   dtype=torch.float32)
        pos_ids = torch.arange(n_diff, device=device,
                               dtype=torch.int32).unsqueeze(0).expand(B, -1)

        cache_shape = (B, n_kv_heads, max_kv_cache_capacity, head_dim)
        k_caches = [
            torch.randn(cache_shape, device=device, dtype=torch.float16)
            for _ in range(n_layers)
        ]
        v_caches = [
            torch.randn(cache_shape, device=device, dtype=torch.float16)
            for _ in range(n_layers)
        ]

        args = tuple([
            noise_trajectory, t0, t1, kvcache_start_index, rope_cos_sin,
            pos_ids
        ] + k_caches + v_caches)

        input_names = ([
            "noise_trajectory",
            "time_steps_t0",
            "time_steps_t1",
            "kvcache_start_index",
            "rope_rotary_cos_sin",
            "attention_pos_id",
        ] + [f"k_cache_{i}" for i in range(n_layers)] +
                       [f"v_cache_{i}" for i in range(n_layers)])

        output_names = (["denoised_trajectory"] +
                        [f"present_k_cache_{i}" for i in range(n_layers)] +
                        [f"present_v_cache_{i}" for i in range(n_layers)])

        batch = torch.export.Dim("batch_size")
        # 7 entries matching forward() signature: 6 named args + *cache_tensors.
        dynamic_shapes: tuple = (
            {
                0: batch
            },  # noise_trajectory
            None,  # time_steps_t0
            None,  # time_steps_t1
            {
                0: batch
            },  # kvcache_start_index
            {
                0: batch
            },  # rope_rotary_cos_sin
            {
                0: batch
            },  # attention_pos_id
            tuple({0: batch} for _ in range(n_layers * 2)),  # *cache_tensors
        )

        return args, input_names, output_names, dynamic_shapes


# ---------------------------------------------------------------------------
# Weight loading + factory
# ---------------------------------------------------------------------------


def _load_action_weights(model: AlpamayoAction, weights: dict,
                         dtype: torch.dtype) -> None:
    """Assign weights from the flat checkpoint dict to the model."""
    loaded = 0
    _ACTION_PREFIXES = ("expert.", "action_in_proj.", "action_out_proj.")
    for ckpt_key, tensor in weights.items():
        if not ckpt_key.startswith(_ACTION_PREFIXES):
            continue

        # Checkpoint uses expert.layers.N.mlp.{gate,up,down}_proj but
        # our ActionDecoderLayer flattens mlp into mlp_{gate,up,down}_proj.
        model_key = ckpt_key
        model_key = model_key.replace(".mlp.gate_proj", ".mlp_gate_proj")
        model_key = model_key.replace(".mlp.up_proj", ".mlp_up_proj")
        model_key = model_key.replace(".mlp.down_proj", ".mlp_down_proj")

        # Navigate to the parameter
        parts = model_key.split(".")
        obj = model
        try:
            for part in parts[:-1]:
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            param_name = parts[-1]
            param = getattr(obj, param_name, None)
            if param is None:
                logger.debug("Key not found in model: %s", model_key)
                continue
            if isinstance(param, nn.Parameter):
                t = tensor.to(dtype) if tensor.is_floating_point() else tensor
                param.data.copy_(t)
                loaded += 1
            elif isinstance(param, torch.Tensor):
                # buffer (e.g. freqs)
                param.copy_(tensor)
                loaded += 1
        except (AttributeError, IndexError):
            logger.debug("Failed to set: %s", model_key)
            continue

    logger.info("Loaded %d action expert tensors", loaded)


def build_alpamayo_1_action(cfg: ActionConfig,
                            weights: dict,
                            dtype: torch.dtype,
                            tp_size: int = 1) -> AlpamayoAction:
    """Build paged action for TP=1, or dense action for a TP-sharded LLM."""
    if tp_size < 1:
        raise ValueError("tp_size must be positive")
    if tp_size == 1:
        from .modeling_alpamayo_action_paged import PagedAlpamayoAction
        model = PagedAlpamayoAction(cfg)
    else:
        model = AlpamayoAction(cfg)
    model = model.to(dtype)
    _load_action_weights(model, weights, dtype)
    model.eval()
    return model
