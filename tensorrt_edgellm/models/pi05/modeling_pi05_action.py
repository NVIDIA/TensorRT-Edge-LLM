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
"""pi0.5 action expert (gemma_300m) for ONNX export.

Wraps ONE flow-matching denoising step:

    action_in_proj(x_t) + adaRMS(time)  ->  suffix tokens
    -> N expert layers, each attending to [prefix_kv ; suffix_kv]
    -> adaRMS final norm -> action_out_proj  ->  velocity v_t

The host runtime owns the Euler step; this graph only emits the prediction.
The per-layer prefix K/V are *inputs* (step-invariant: the runtime computes
them once with the prefix tower and binds them for the whole denoising loop).

Attention runs over a K/V cache the prefix tower fills in place, non-causal and
unmasked -- the runtime compacts the prefix so every key is valid. It is the
shared ``trt_edgellm::AttentionPlugin`` over a paged pool, with RoPE, the Q/K/V
split and the cache append fused into it.
Numerics follow openpi ``pi0_pytorch.PI0Pytorch`` plus the patched
``transformers_replace`` ``GemmaRMSNorm`` / ``_gated_residual``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modeling_pi05_prefix import Pi05MLP

logger = logging.getLogger(__name__)

# openpi ``create_sinusoidal_pos_embedding`` sensitivity range for the
# flow-matching timestep (pi0_pytorch.embed_suffix).
_TIME_MIN_PERIOD = 4e-3
_TIME_MAX_PERIOD = 4.0


@dataclass
class Pi05ActionConfig:
    """Hyperparameters for the pi0.5 action expert (one denoising step)."""

    hidden_size: int = 1024
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    intermediate_size: int = 4096
    rms_norm_eps: float = 1e-6

    action_dim: int = 32
    action_horizon: int = 10

    @property
    def cond_dim(self) -> int:
        return self.hidden_size


def num_adarms_sites(cfg: "Pi05ActionConfig") -> int:
    """Two AdaRMS norms per layer plus the final norm."""
    return 2 * cfg.num_hidden_layers + 1


# XQA walks the cache in CTA tiles this wide.
_XQA_CACHE_TILE = 256

#: ``rt::ExecutionPhase::kDiffusionDenoise`` (cpp/common/executionPhase.h). The
#: AttentionPlugin reads this as the LENGTH of ``execution_phase_marker``, and with
#: tree attention on it selects the tree kernel, whose shape a denoise step has.
DIFFUSION_DENOISE_PHASE = 6


def xqa_cache_capacity(max_prefix_len: int, action_horizon: int) -> int:
    """Slots per sequence in the XQA KV cache: the prefix plus the action tokens.

    The prefix tower writes slots ``[0, prefix_len)`` of each half directly and
    the plugin appends the action tokens at ``[prefix_len, prefix_len + H)``, so
    both towers must agree on this number.
    """
    slots = max_prefix_len + action_horizon
    return -(-slots // _XQA_CACHE_TILE) * _XQA_CACHE_TILE  # whole CTA tiles


# Page size of the shared paged-KV pool; must equal ``rt::kTOKENS_PER_PAGE``.
KV_TOKENS_PER_PAGE = 128


def kv_pages_per_seq(max_prefix_len: int, action_horizon: int) -> int:
    """Pages one sequence's cache spans. ``_XQA_CACHE_TILE`` is a multiple of the
    page size, so a sequence occupies whole pages and an identity page table
    reproduces the contiguous layout byte for byte."""
    capacity = xqa_cache_capacity(max_prefix_len, action_horizon)
    return capacity // KV_TOKENS_PER_PAGE


def packed_mask_words(action_horizon: int) -> int:
    """INT32 words per query token in the packed spec-decode mask."""
    return -(-action_horizon // 32)


class AdaRMSNorm(nn.Module):
    """Gemma RMSNorm with a Dense-produced (scale, shift, gate) modulation.

    pi0.5 expert norms have no ``weight``: the modulation Dense replaces it.
    The reduction runs in fp32 -- keeping it in fp16 loses the small-variance
    range the gate amplifies.

    With ``mod_index`` set, ``dense`` leaves the graph and ``cond`` is the
    packed ``[B, sites, 3*dim]`` modulation this norm slices its row out of.
    """

    def __init__(self,
                 dim: int,
                 cond_dim: int,
                 eps: float = 1e-6,
                 mod_index: "int | None" = None) -> None:
        super().__init__()
        self.eps = eps
        self.mod_index = mod_index
        self.dense = nn.Linear(cond_dim, dim * 3, bias=True)

    def forward(self, x: torch.Tensor,
                cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        normed = x.float() * torch.rsqrt(var + self.eps)

        if self.mod_index is None:
            modulation = self.dense(cond).unsqueeze(1)
        else:
            modulation = cond[:, self.mod_index:self.mod_index + 1, :]
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        normed = normed * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)


def _gated_residual(x: torch.Tensor, y: torch.Tensor,
                    gate: torch.Tensor) -> torch.Tensor:
    return x + y * gate


class Pi05TimeEmbedder(nn.Module):
    """Sinusoidal timestep embedding -> Linear -> SiLU -> Linear -> SiLU."""

    def __init__(self, cfg: Pi05ActionConfig) -> None:
        super().__init__()
        dim = cfg.hidden_size
        if dim % 2 != 0:
            raise ValueError(f"hidden_size ({dim}) must be divisible by 2")
        self.half_dim = dim // 2
        self.time_mlp_in = nn.Linear(dim, dim, bias=True)
        self.time_mlp_out = nn.Linear(dim, dim, bias=True)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        # Built here, not held as a buffer: nn.Module.to(fp16) converts float buffers, and
        # rounding the top frequency moves the embedding by up to 0.45 at t=1. Export folds
        # this to an fp32 constant.
        fraction = torch.linspace(0.0,
                                  1.0,
                                  self.half_dim,
                                  dtype=torch.float32,
                                  device=timestep.device)
        period = _TIME_MIN_PERIOD * (_TIME_MAX_PERIOD /
                                     _TIME_MIN_PERIOD)**fraction
        scaling_factor = 1.0 / period * 2 * math.pi
        sin_input = scaling_factor[None, :] * timestep[:, None].float()
        time_emb = torch.cat(
            [torch.sin(sin_input), torch.cos(sin_input)], dim=1)
        time_emb = time_emb.type_as(self.time_mlp_in.weight)
        return F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_emb))))


class Pi05ActionAttention(nn.Module):
    """Expert attention over ``[prefix_kv ; suffix_kv]`` (non-causal).

    The shared ``AttentionPlugin`` over a paged pool, running its XQA spec-decode
    kernel under an all-ones mask.
    """

    def __init__(self, cfg: Pi05ActionConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.qk_scale = self.head_dim**-0.5
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_io: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from ..ops import attention_plugin

        (rope_cos_sin, attention_pos_id, query_lengths, kvcache_start_index,
         kv_page_table, attention_mask, query_start_offsets,
         attention_sequence_lengths, execution_phase_marker,
         context_sequence_count_carrier) = attn_io
        bsz, q_len, _ = hidden_states.shape
        # The plugin's ABI is token-major: one row per query token, requests concatenated,
        # so T_exec is exactly B * H at a uniform horizon. The mask and position ids are
        # exported at that rank already and need no reshape node.
        tokens = hidden_states.reshape(bsz * q_len, -1)

        qkv = torch.cat([
            self.q_proj(tokens),
            self.k_proj(tokens),
            self.v_proj(tokens),
        ],
                        dim=-1)

        # Tree decoding with an all-ones mask and relative position ids over a
        # cos/sin table the runtime already sliced to the action span: the query
        # tokens see the whole prefix and each other, which is what pi0.5 needs.
        attn_output, present_kv = attention_plugin(
            qkv,
            kv_cache,
            query_lengths,
            rope_cos_sin,
            kvcache_start_index,
            kv_page_table,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            sliding_window_size=0,
            enable_tree_attention=True,
            enable_fp8_kv_cache=False,
            # The plugin applies the scale itself, so q is not pre-scaled here.
            attention_scale=self.qk_scale,
            enable_context_mask_selector=False,
            enable_vision_block_attention=False,
            skip_softmax_scale_factor=0.0,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
            qkv_scales=[1.0, 1.0, 1.0],
            query_start_offsets=query_start_offsets,
            attention_sequence_lengths=attention_sequence_lengths,
            execution_phase_marker=execution_phase_marker,
            context_sequence_count_carrier=context_sequence_count_carrier)
        # The width is spelled out rather than inferred: with a token-major plugin
        # output, a -1 here leaves TensorRT two unknowns in one Reshape.
        attn_output = attn_output.reshape(bsz, q_len,
                                          self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present_kv


class Pi05ActionDecoderLayer(nn.Module):
    """Pre-norm expert block with adaRMS modulation and gated residuals."""

    def __init__(self, cfg: Pi05ActionConfig) -> None:
        super().__init__()
        self.input_layernorm = AdaRMSNorm(cfg.hidden_size, cfg.cond_dim,
                                          cfg.rms_norm_eps)
        self.post_attention_layernorm = AdaRMSNorm(cfg.hidden_size,
                                                   cfg.cond_dim,
                                                   cfg.rms_norm_eps)
        self.self_attn = Pi05ActionAttention(cfg)
        self.mlp = Pi05MLP(cfg.hidden_size, cfg.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        adarms_cond: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_io: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, adarms_cond)
        hidden_states, present_kv = self.self_attn(hidden_states, kv_cache,
                                                   attn_io)
        hidden_states = _gated_residual(residual, hidden_states, gate)

        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(
            hidden_states, adarms_cond)
        hidden_states = self.mlp(hidden_states)
        return _gated_residual(residual, hidden_states, gate), present_kv


class _Pi05ActionModel(nn.Module):
    """Container matching the checkpoint's ``layers`` / ``norm`` nesting."""

    def __init__(self, cfg: Pi05ActionConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            Pi05ActionDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)
        ])
        self.norm = AdaRMSNorm(cfg.hidden_size, cfg.cond_dim, cfg.rms_norm_eps)


def adarms_norms(model: "_Pi05ActionModel") -> "list[AdaRMSNorm]":
    """Modulation site order, shared by the action graph's slice indices and the
    cond graph's output rows -- the two must not drift apart."""
    norms: "list[AdaRMSNorm]" = []
    for layer in model.layers:
        norms += [layer.input_layernorm, layer.post_attention_layernorm]
    norms.append(model.norm)
    return norms


class Pi05Action(nn.Module):
    """One pi0.5 denoising step; emits the velocity, not the Euler update."""

    def __init__(
        self,
        cfg: Pi05ActionConfig,
        hoist_cond: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.hoist_cond = hoist_cond
        self.action_in_proj = nn.Linear(cfg.action_dim,
                                        cfg.hidden_size,
                                        bias=True)
        self.action_out_proj = nn.Linear(cfg.hidden_size,
                                         cfg.action_dim,
                                         bias=True)
        self.time_embedder = Pi05TimeEmbedder(cfg)
        self.model = _Pi05ActionModel(cfg)
        if hoist_cond:
            for index, norm in enumerate(adarms_norms(self.model)):
                norm.mod_index = index

    def forward(
        self,
        noise_trajectory: torch.Tensor,
        cond_input: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        attention_pos_id: torch.Tensor,
        *prefix_kv: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """``cond_input`` is the timestep ``[B]``, or, when the modulation is
        hoisted, the precomputed ``[B, sites, 3*hidden_size]`` triples.

        Returns the velocity followed by one appended cache per layer; the
        caches are the plugin's in-place update, exposed so TensorRT sees the
        write."""
        io_type = self.action_in_proj.weight.dtype

        adarms_cond = (cond_input if self.hoist_cond else
                       self.time_embedder(cond_input)).to(io_type)
        hidden = self.action_in_proj(noise_trajectory.to(io_type))

        # The eight AttentionPlugin metadata inputs lead the per-layer paged pools.
        kv_cache = prefix_kv[8:]
        attn_io = (rope_rotary_cos_sin, attention_pos_id) + prefix_kv[:8]

        present_kv: List[torch.Tensor] = []
        for i, layer in enumerate(self.model.layers):
            hidden, present = layer(hidden, adarms_cond, kv_cache[i], attn_io)
            present_kv.append(present)

        hidden, _ = self.model.norm(hidden, adarms_cond)
        return (self.action_out_proj(hidden).to(torch.float32), *present_kv)

    def get_onnx_export_args(self, max_prefix_len: int,
                             device: str) -> Tuple[tuple, list, list, tuple]:
        cfg = self.cfg
        n, hkv, d = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
        # Batch 2 keeps the batch axis symbolic (torch.export specializes size-1).
        batch_size, horizon = 2, cfg.action_horizon

        noise = torch.randn(batch_size,
                            horizon,
                            cfg.action_dim,
                            device=device,
                            dtype=torch.float32)
        if self.hoist_cond:
            cond_name = "adarms_modulation"
            cond = torch.randn(batch_size,
                               num_adarms_sites(cfg),
                               cfg.hidden_size * 3,
                               device=device,
                               dtype=torch.float16)
        else:
            cond_name = "timestep"
            cond = torch.full((batch_size, ),
                              1.0,
                              device=device,
                              dtype=torch.float32)
        batch_dim = torch.export.Dim("batch_size", min=1, max=256)
        # Token-major ABI: every per-token input is one row per query token,
        # requests concatenated, so T_exec = B * H.
        exec_tokens = batch_dim * horizon
        rope = torch.randn(batch_size * horizon,
                           d,
                           device=device,
                           dtype=torch.float32)
        pos = torch.arange(horizon, device=device,
                           dtype=torch.int32).repeat(batch_size)

        pages = kv_pages_per_seq(max_prefix_len, horizon)
        num_pages = torch.export.Dim("num_pages", min=pages, max=256 * pages)
        kv_args = [
            torch.full(
                (batch_size, ), horizon, device=device, dtype=torch.int32),
            # Values unused: tree decoding takes the cache append offset from
            # attention_sequence_lengths.
            torch.zeros(batch_size, device=device, dtype=torch.int32),
            torch.zeros(
                (batch_size, 2, pages), device=device, dtype=torch.int32),
            torch.zeros((batch_size * horizon, packed_mask_words(horizon)),
                        device=device,
                        dtype=torch.int32),
            torch.arange(0,
                         batch_size * horizon + 1,
                         horizon,
                         device=device,
                         dtype=torch.int32),
            torch.full((batch_size, ),
                       max_prefix_len + horizon,
                       device=device,
                       dtype=torch.int32),
            # Shape-only carriers: the plugin reads their LENGTH, never their
            # payload, and both lengths are fixed for this model -- the denoise
            # phase, and no context sequences -- so they are exported static.
            torch.zeros(
                DIFFUSION_DENOISE_PHASE, device=device, dtype=torch.int32),
            torch.zeros(0, device=device, dtype=torch.int32),
        ] + [
            torch.randn((2, batch_size * pages, KV_TOKENS_PER_PAGE, hkv, d),
                        device=device,
                        dtype=torch.float16) for _ in range(n)
        ]
        kv_names = ([
            "query_lengths",
            "kvcache_start_index",
            "kv_page_table",
            "attention_mask",
            "query_start_offsets",
            "attention_sequence_lengths",
            "execution_phase_marker",
            "context_sequence_count_carrier",
        ] + [f"kv_cache_layer{i:02d}" for i in range(n)])
        kv_dyn = ([{
            0: batch_dim
        }, {
            0: batch_dim
        }, {
            0: batch_dim
        }, {
            0: exec_tokens
        }, {
            0: batch_dim + 1
        }, {
            0: batch_dim
        }, {}, {}] + [{
            1: num_pages
        } for _ in range(n)])
        args = tuple([noise, cond, rope, pos] + kv_args)
        input_names = ([
            "noise_trajectory",
            cond_name,
            "rope_rotary_cos_sin",
            "attention_pos_id",
        ] + kv_names)
        output_names = (["action_pred"] +
                        [f"present_kv_cache_layer{i:02d}" for i in range(n)])

        dynamic_shapes = (
            {
                0: batch_dim
            },  # noise_trajectory
            {
                0: batch_dim
            },  # timestep / adarms_modulation
            {
                0: exec_tokens
            },  # rope_rotary_cos_sin
            {
                0: exec_tokens
            },  # attention_pos_id
            tuple(kv_dyn),  # *prefix_kv / cache tail
        )
        return args, input_names, output_names, dynamic_shapes


def _load_action_weights(model: Pi05Action, weights: dict,
                         dtype: torch.dtype) -> None:
    """Assign expert weights from the split dict (see ``weights.py``).

    Every parameter must receive a checkpoint tensor: a silently
    random-initialized module corrupts the denoising output.
    """
    state = {}
    for key, tensor in weights.items():
        if key.startswith(("action_in_proj.", "action_out_proj.")):
            state[key] = tensor
        elif key.startswith(("time_mlp_in.", "time_mlp_out.")):
            state["time_embedder." + key] = tensor
        else:
            state["model." + key] = tensor

    incompatible = model.load_state_dict(
        {
            k: (v.to(dtype) if v.is_floating_point() else v)
            for k, v in state.items()
        },
        strict=False)
    if incompatible.missing_keys:
        raise KeyError("pi0.5 action parameters received no checkpoint "
                       "tensor: " + ", ".join(incompatible.missing_keys[:8]))
    if incompatible.unexpected_keys:
        logger.warning("Unexpected action checkpoint keys (first 5): %s",
                       incompatible.unexpected_keys[:5])
    logger.info("Loaded %d pi0.5 action tensors", len(state))


class Pi05CondModulation(nn.Module):
    """Timestep -> packed AdaRMS ``(scale, shift, gate)`` for every norm site.

    The schedule is a function of the step count and the batch alone, not of the
    observation, so the runtime evaluates all N steps once and binds one row per
    step until either changes. The 37
    per-site Denses are fused into one Gemm; they are mathematically
    independent rows of the same ``[sites * 3 * hidden, cond_dim]`` matmul.
    """

    def __init__(self, action: Pi05Action) -> None:
        super().__init__()
        norms = adarms_norms(action.model)
        weight = torch.cat([norm.dense.weight for norm in norms], dim=0)
        bias = torch.cat([norm.dense.bias for norm in norms], dim=0)
        self.num_sites = len(norms)
        self.mod_dim = norms[0].dense.out_features
        self.time_embedder = action.time_embedder
        self.dense = nn.Linear(action.cfg.cond_dim,
                               weight.shape[0],
                               bias=True,
                               dtype=weight.dtype)
        with torch.no_grad():
            self.dense.weight.copy_(weight)
            self.dense.bias.copy_(bias)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        cond = self.time_embedder(timestep)
        return self.dense(cond).unflatten(-1, (self.num_sites, self.mod_dim))

    def get_onnx_export_args(self, opt_steps: int,
                             device: str) -> Tuple[tuple, list, list, tuple]:
        # Batch 2 keeps the step axis symbolic (torch.export specializes size-1).
        timestep = torch.linspace(1.0,
                                  0.0,
                                  max(2, opt_steps),
                                  device=device,
                                  dtype=torch.float32)
        steps = torch.export.Dim("num_steps", min=1, max=256)
        return ((timestep, ), ["timestep"], ["adarms_modulation"], ({
            0: steps
        }, ))


def build_pi05_action(
    cfg: Pi05ActionConfig,
    weights: dict,
    dtype: torch.dtype,
    hoist_cond: bool = False,
) -> Pi05Action:
    model = Pi05Action(cfg, hoist_cond=hoist_cond).to(dtype)
    _load_action_weights(model, weights, dtype)
    model.eval()
    return model


def build_pi05_cond(action: Pi05Action) -> Pi05CondModulation:
    model = Pi05CondModulation(action)
    model.eval()
    return model
