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
"""LiLiCorr candidate-lattice head: transition scores between per-slot top-K candidates.

Features, norms, attention softmax, vector normalization and scores run in
FP32. Linear layers keep their checkpoint format: NVFP4 linears run as packed
W4A16 GEMMs, BF16 linears as FP16. Submodule and parameter names mirror the
checkpoint tensors under ``lilicorr.``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import ModelConfig
from ..default.modeling_default import RMSNorm
from ..linear import FP16Linear, make_linear

_NUM_CANDIDATE_FEATURES = 5
_PREFIX = "lilicorr"
_FEATURE_NORM_EPS = 1e-5


def build_lattice_attention_bias(relative_slot_bias: torch.Tensor,
                                 same_slot_bias: torch.Tensor, num_slots: int,
                                 topk: int, block_size: int) -> torch.Tensor:
    """Return the FP32 additive lattice attention bias ``[heads, S*K, S*K]``.

    ``bias[h, q, k] = relative_slot_bias[h, slot(q) - slot(k) + block_size - 1]
    + same_slot_bias[h] * [slot(q) == slot(k)]`` with ``slot(i) = i // topk``.
    """
    slots = torch.arange(
        num_slots, device=relative_slot_bias.device).repeat_interleave(topk)
    distance = slots[:, None] - slots[None, :] + (block_size - 1)
    same_slot = (slots[:, None] == slots[None, :]).to(torch.float32)
    return (relative_slot_bias.to(torch.float32)[:, distance] +
            same_slot_bias.to(torch.float32)[:, None, None] * same_slot)


def _linear(module: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    if isinstance(module, FP16Linear):
        weight = module.weight
        return F.linear(hidden.to(weight.dtype), weight, module.bias)
    # Packed GEMM plugins take token-major [M, K] activations.
    out = module(hidden.reshape(-1, hidden.shape[-1]).to(torch.float16))
    return out.reshape(*hidden.shape[:-1], out.shape[-1])


def _normalize(vectors: torch.Tensor, eps: float) -> torch.Tensor:
    norm = vectors.pow(2).sum(dim=-1, keepdim=True).sqrt()
    return vectors / norm.clamp_min(eps)


class _FeatureLayerNorm(nn.Module):

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features,
                                               dtype=torch.float16),
                                   requires_grad=False)
        self.bias = nn.Parameter(torch.empty(num_features,
                                             dtype=torch.float16),
                                 requires_grad=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(features.to(torch.float32), (features.shape[-1], ),
                            self.weight.to(torch.float32),
                            self.bias.to(torch.float32), _FEATURE_NORM_EPS)


class _LiLiCorrMLP(nn.Module):
    """``down_proj(silu(up_proj(x)))`` with biases; FP32 output."""

    def __init__(self, config: ModelConfig, prefix: str, in_features: int,
                 intermediate: int, out_features: int) -> None:
        super().__init__()
        self.up_proj = make_linear(config,
                                   in_features,
                                   intermediate,
                                   bias=True,
                                   module_name=f"{prefix}.up_proj")
        self.down_proj = make_linear(config,
                                     intermediate,
                                     out_features,
                                     bias=True,
                                     module_name=f"{prefix}.down_proj")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(_linear(self.up_proj, hidden).to(torch.float32))
        return _linear(self.down_proj, hidden).to(torch.float32)


class _LiLiCorrLatticeAttention(nn.Module):
    """Bidirectional multi-head attention over the lattice with an additive bias."""

    def __init__(self, config: ModelConfig, prefix: str, hidden_size: int,
                 num_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * hidden_size,
                                                       hidden_size,
                                                       dtype=torch.float16),
                                           requires_grad=False)
        self.in_proj_bias = nn.Parameter(torch.empty(3 * hidden_size,
                                                     dtype=torch.float16),
                                         requires_grad=False)
        self.out_proj = make_linear(config,
                                    hidden_size,
                                    hidden_size,
                                    bias=True,
                                    module_name=f"{prefix}.out_proj")

    def forward(self, hidden: torch.Tensor,
                attention_bias: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden.shape
        qkv = F.linear(hidden.to(self.in_proj_weight.dtype),
                       self.in_proj_weight, self.in_proj_bias)
        q, k, v = (
            t.reshape(batch, length, self.num_heads,
                      self.head_dim).transpose(1, 2)
            for t in qkv.to(torch.float32).split(self.hidden_size, dim=-1))
        scores = torch.matmul(q, k.transpose(
            -1, -2)) * (1.0 / math.sqrt(self.head_dim)) + attention_bias
        context = torch.matmul(torch.softmax(scores, dim=-1), v)
        context = context.transpose(1, 2).reshape(batch, length,
                                                  self.hidden_size)
        return _linear(self.out_proj, context).to(torch.float32)


class _LiLiCorrLayer(nn.Module):
    """Pre-RMSNorm lattice transformer layer on an FP32 residual stream."""

    def __init__(self, config: ModelConfig, prefix: str, hidden_size: int,
                 num_heads: int, mlp_size: int, rms_norm_eps: float) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size, rms_norm_eps)
        self.attn = _LiLiCorrLatticeAttention(config, f"{prefix}.attn",
                                              hidden_size, num_heads)
        self.mlp_norm = RMSNorm(hidden_size, rms_norm_eps)
        self.mlp = _LiLiCorrMLP(config, f"{prefix}.mlp", hidden_size, mlp_size,
                                hidden_size)

    def forward(self, hidden: torch.Tensor,
                attention_bias: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attn(self.attn_norm(hidden), attention_bias)
        return hidden + self.mlp(self.mlp_norm(hidden))


class LiLiCorrHead(nn.Module):
    """Scores transitions between the top-K candidates of consecutive draft slots.

    Call :meth:`materialize` after the weights are loaded (and after any
    device or dtype move) to build the FP32 lattice attention bias.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.lilicorr_hidden_size
        num_heads = config.lilicorr_num_heads
        if num_heads <= 0 or hidden % num_heads:
            raise ValueError(
                f"LiLiCorr hidden size {hidden} must be divisible "
                f"by num_heads {num_heads}")
        self.block_size = config.dflash2_block_size
        self.num_slots = self.block_size - 1
        self.topk = config.lilicorr_candidate_topk
        self.hidden_size = hidden
        self.vector_eps = config.lilicorr_vector_eps
        self.logit_scale = config.lilicorr_logit_scale
        model_hidden = config.hidden_size
        factor_dim = config.lilicorr_factor_dim

        def linear(name: str, in_features: int, out_features: int):
            return make_linear(config,
                               in_features,
                               out_features,
                               bias=True,
                               module_name=f"{_PREFIX}.{name}")

        self.token_proj = linear("token_proj", model_hidden, hidden)
        self.pass_hidden_proj = linear("pass_hidden_proj", model_hidden,
                                       hidden)
        self.feature_norm = _FeatureLayerNorm(_NUM_CANDIDATE_FEATURES)
        self.feature_mlp = _LiLiCorrMLP(config, f"{_PREFIX}.feature_mlp",
                                        _NUM_CANDIDATE_FEATURES, hidden,
                                        hidden)
        self.slot_embedding = nn.Parameter(torch.empty(1,
                                                       1,
                                                       self.num_slots,
                                                       1,
                                                       hidden,
                                                       dtype=torch.float16),
                                           requires_grad=False)
        self.rank_embedding = nn.Parameter(torch.empty(1,
                                                       1,
                                                       1,
                                                       self.topk,
                                                       hidden,
                                                       dtype=torch.float16),
                                           requires_grad=False)
        self.relative_slot_bias = nn.Parameter(torch.empty(
            num_heads, 2 * self.block_size - 1, dtype=torch.float16),
                                               requires_grad=False)
        self.same_slot_bias = nn.Parameter(torch.empty(num_heads,
                                                       dtype=torch.float16),
                                           requires_grad=False)
        self.context_proj = linear("context_proj", model_hidden, hidden)
        mlp_size = int(hidden * config.lilicorr_mlp_ratio)
        self.layers = nn.ModuleList([
            _LiLiCorrLayer(config, f"{_PREFIX}.layers.{index}", hidden,
                           num_heads, mlp_size, config.rms_norm_eps)
            for index in range(config.lilicorr_num_layers)
        ])
        self.output_norm = RMSNorm(hidden, config.rms_norm_eps)
        self.anchor_norm = RMSNorm(hidden, config.rms_norm_eps)
        self.factor_input_proj = linear("factor_input_proj", 3 * hidden,
                                        hidden)
        if not isinstance(self.factor_input_proj, FP16Linear):
            raise ValueError(
                "LiLiCorr factor_input_proj must be dense: its weight is "
                "split by input column block")
        self.out_head = linear("out_head", hidden, factor_dim)
        self.in_head = linear("in_head", hidden, factor_dim)
        self.anchor_out_head = linear("anchor_out_head", hidden, factor_dim)

        lattice = self.num_slots * self.topk
        self.register_buffer("lattice_attention_bias",
                             torch.full((num_heads, lattice, lattice),
                                        float("nan"),
                                        dtype=torch.float32),
                             persistent=False)
        self._lattice_bias_materialized = False

    @torch.no_grad()
    def materialize(self) -> None:
        """Rebuild the lattice attention bias from the loaded slot biases."""
        self.lattice_attention_bias = build_lattice_attention_bias(
            self.relative_slot_bias, self.same_slot_bias, self.num_slots,
            self.topk, self.block_size)
        self._lattice_bias_materialized = True

    def forward(self, token_embeddings: torch.Tensor,
                candidate_log_probs: torch.Tensor, pass_hidden: torch.Tensor,
                anchor_hidden: torch.Tensor,
                anchor_valid: torch.Tensor) -> torch.Tensor:
        """Return FP32 transition scores ``[B, S, K(prev), K(next)]``.

        Args:
            token_embeddings: ``[B, S, K, H]`` target input embeddings of the candidates.
            candidate_log_probs: ``[B, S, K]`` full-vocabulary log-probabilities.
            pass_hidden: ``[B, S, H]`` drafter hidden of the predicted slots.
            anchor_hidden: ``[B, H]`` ``hidden_norm(fc(...))`` of the last valid delta row.
            anchor_valid: ``[B]`` bool; False zeroes the projected anchor.

        Slot 0 holds the anchor start scores broadcast over ``prev``; slot
        ``s > 0`` scores candidate ``prev`` of slot ``s - 1`` against
        candidate ``next`` of slot ``s``. Scores are already scaled by
        ``logit_scale``.
        """
        attention_bias = self.lattice_attention_bias
        if (not self._lattice_bias_materialized
                or attention_bias.dtype != torch.float32):
            raise RuntimeError(
                "LiLiCorrHead.materialize() must run after loading weights "
                "and after any dtype change")
        batch = token_embeddings.shape[0]
        topk = self.topk

        log_probs = candidate_log_probs.to(torch.float32)
        rank = torch.arange(topk, dtype=torch.float32, device=log_probs.device)
        rank_fraction = rank / float(max(topk - 1, 1))
        is_top1 = (rank == 0).to(torch.float32)
        features = torch.stack([
            log_probs,
            log_probs.exp(),
            log_probs - log_probs.amax(dim=-1, keepdim=True),
            rank_fraction.expand_as(log_probs),
            is_top1.expand_as(log_probs),
        ],
                               dim=-1)

        nodes = _linear(self.token_proj, token_embeddings).to(torch.float32)
        nodes = nodes + _linear(self.pass_hidden_proj, pass_hidden).to(
            torch.float32)[:, :, None, :]
        nodes = nodes + self.feature_mlp(self.feature_norm(features))
        nodes = nodes + self.slot_embedding[0, 0].to(torch.float32)
        nodes = nodes + self.rank_embedding[0, 0].to(torch.float32)

        lattice = nodes.reshape(batch, self.num_slots * topk, self.hidden_size)
        for layer in self.layers:
            lattice = layer(lattice, attention_bias)
        states = self.output_norm(lattice).reshape(batch, self.num_slots, topk,
                                                   self.hidden_size)

        anchor = _linear(self.context_proj, anchor_hidden).to(torch.float32)
        anchor = self.anchor_norm(anchor *
                                  anchor_valid.to(torch.float32)[:, None])

        # factor_input_proj consumes concat(X, a, X*a); split its columns so
        # W_a @ a is computed once per sequence instead of per candidate.
        weight = self.factor_input_proj.weight
        hidden = self.hidden_size
        anchor_row = anchor[:, None, None, :]
        pre = F.linear(states.to(weight.dtype), weight[:, :hidden],
                       self.factor_input_proj.bias).to(torch.float32)
        pre = pre + F.linear(anchor_row.to(weight.dtype),
                             weight[:, hidden:2 * hidden]).to(torch.float32)
        pre = pre + F.linear((states * anchor_row).to(weight.dtype),
                             weight[:, 2 * hidden:]).to(torch.float32)
        factor = F.silu(pre)

        out_vec = _normalize(
            _linear(self.out_head, factor).to(torch.float32), self.vector_eps)
        in_vec = _normalize(
            _linear(self.in_head, factor).to(torch.float32), self.vector_eps)
        anchor_out = _normalize(
            _linear(self.anchor_out_head, anchor).to(torch.float32),
            self.vector_eps)

        start = torch.matmul(in_vec[:, 0], anchor_out[:, :, None]).squeeze(-1)
        pair = torch.matmul(out_vec[:, :-1], in_vec[:, 1:].transpose(-1, -2))
        first = start[:, None, None, :].expand(batch, 1, topk, topk)
        return self.logit_scale * torch.cat([first, pair], dim=1)
