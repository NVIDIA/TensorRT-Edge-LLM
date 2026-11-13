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

Follows ``tensorrt_edgellm.models.lilicorr.modeling_lilicorr_head`` op for op.
Features, norms, attention softmax, vector normalization and scores run in
FP32. Linears keep their checkpoint format: NVFP4 linears run as W4A16 GEMMs
on token-major rows, BF16 linears as FP16.
"""

import math

import numpy as np
import tensorrt as trt

from ...ops import Linear, Module
from ...ops import functional as F

_FEATURE_NORM_EPS = 1e-5


def lattice_attention_bias(relative_slot_bias: np.ndarray,
                           same_slot_bias: np.ndarray, num_slots: int,
                           topk: int, block_size: int) -> np.ndarray:
    """Return the FP32 additive lattice attention bias ``[heads, S*K, S*K]``.

    ``bias[h, q, k] = relative_slot_bias[h, slot(q) - slot(k) + block_size - 1]
    + same_slot_bias[h] * [slot(q) == slot(k)]`` with ``slot(i) = i // topk``.
    """
    slots = np.repeat(np.arange(num_slots), topk)
    distance = slots[:, None] - slots[None, :] + (block_size - 1)
    same_slot = (slots[:, None] == slots[None, :]).astype(np.float32)
    return (relative_slot_bias.astype(np.float32)[:, distance] +
            same_slot_bias.astype(np.float32)[:, None, None] * same_slot)


def _last_axis_constant(values: np.ndarray, rank: int, name: str):
    return F.constant(
        np.asarray(values,
                   dtype=np.float32).reshape((1, ) * (rank - 1) + (-1, )),
        name)


def _rms_norm_f32(hidden, weight: np.ndarray, eps: float):
    variance = (hidden * hidden).mean(-1, keepdim=True)
    normalized = hidden * (1.0 / (variance + eps).sqrt())
    return normalized * _last_axis_constant(weight, hidden.ndim,
                                            "lilicorr_rms_weight")


def _layer_norm_f32(hidden, weight: np.ndarray, bias: np.ndarray, eps: float):
    centered = hidden - hidden.mean(-1, keepdim=True)
    variance = (centered * centered).mean(-1, keepdim=True)
    normalized = centered / (variance + eps).sqrt()
    return (normalized *
            _last_axis_constant(weight, hidden.ndim, "lilicorr_ln_weight") +
            _last_axis_constant(bias, hidden.ndim, "lilicorr_ln_bias"))


def _normalize(vectors, eps: float):
    norm = (vectors * vectors).sum(-1, keepdim=True).sqrt()
    return vectors / norm.maximum(eps)


def _project(linear: Linear, hidden, out_features: int):
    """Apply ``linear`` to FP16 rows of ``hidden`` and restore its leading dims."""
    rank = hidden.ndim
    leading = F.shape_of(hidden)[0:rank - 1]
    if hidden.dtype != trt.float16:
        hidden = hidden.cast(trt.float16)
    rows = hidden.reshape((-1, int(hidden.shape[-1])))
    return F.dynamic_reshape(linear(rows), (leading, out_features))


class _LatticeLayer(Module):
    """Pre-RMSNorm lattice transformer layer on an FP32 residual stream."""

    def __init__(self, ctx, prefix: str, hidden_size: int, num_heads: int,
                 mlp_size: int) -> None:
        super().__init__(ctx, prefix)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_size = mlp_size
        self.attn_norm = self.weights.f16(self.key("attn_norm.weight"))
        self.mlp_norm = self.weights.f16(self.key("mlp_norm.weight"))
        self.in_proj_weight = self.weights.f16(self.key("attn.in_proj_weight"))
        self.in_proj_bias = self.weights.f16(self.key("attn.in_proj_bias"))
        self.out_proj = _head_linear(ctx, self.key("attn.out_proj"))
        self.up_proj = _head_linear(ctx, self.key("mlp.up_proj"))
        self.down_proj = _head_linear(ctx, self.key("mlp.down_proj"))

    def _attention(self, hidden, attention_bias):
        hidden_size, num_heads = self.hidden_size, self.num_heads
        head_dim = hidden_size // num_heads
        length = int(hidden.shape[1])
        qkv = F.linear_with_weights(hidden.cast(trt.float16),
                                    self.in_proj_weight, self.in_proj_bias,
                                    3).cast(trt.float32)

        def heads(offset: int):
            return qkv.slice_last_dim(offset, hidden_size, 3).reshape(
                (0, length, num_heads, head_dim)).transpose((0, 2, 1, 3))

        query, key, value = (heads(index * hidden_size) for index in range(3))
        scores = F.matmul(query, key, transpose_rhs=True) * (
            1.0 / math.sqrt(head_dim)) + attention_bias
        context = F.matmul(scores.softmax(3), value).transpose(
            (0, 2, 1, 3)).reshape((0, length, hidden_size))
        return _project(self.out_proj, context, hidden_size).cast(trt.float32)

    def _mlp(self, hidden):
        hidden = _project(self.up_proj, hidden,
                          self.mlp_size).cast(trt.float32).silu()
        return _project(self.down_proj, hidden,
                        self.hidden_size).cast(trt.float32)

    def forward(self, hidden, attention_bias):
        eps = self.cfg.rms_norm_eps
        hidden = hidden + self._attention(
            _rms_norm_f32(hidden, self.attn_norm, eps), attention_bias)
        return hidden + self._mlp(_rms_norm_f32(hidden, self.mlp_norm, eps))


def _head_linear(ctx, prefix: str) -> Linear:
    # Every head projection has a bias; the checkpoint loader treats a missing
    # one as absent rather than as an error.
    if not ctx.weights.has(prefix + ".bias"):
        raise KeyError(f"LiLiCorr checkpoint is missing {prefix}.bias")
    return Linear(ctx, prefix, tensor_parallel=False)


class LiLiCorrHead(Module):
    """Scores transitions between the top-K candidates of consecutive draft slots."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        cfg = ctx.cfg
        hidden = cfg.lilicorr_hidden_size
        num_heads = cfg.lilicorr_num_heads
        if num_heads <= 0 or hidden % num_heads:
            raise ValueError(
                f"LiLiCorr hidden size {hidden} must be divisible "
                f"by num_heads {num_heads}")
        self.hidden_size = hidden
        self.num_slots = cfg.dflash2_block_size - 1
        self.topk = cfg.lilicorr_candidate_topk
        self.factor_dim = cfg.lilicorr_factor_dim
        self.mlp_size = int(hidden * cfg.lilicorr_mlp_ratio)

        def linear(name: str) -> Linear:
            return _head_linear(ctx, self.key(name))

        self.token_proj = linear("token_proj")
        self.pass_hidden_proj = linear("pass_hidden_proj")
        self.context_proj = linear("context_proj")
        self.feature_up_proj = linear("feature_mlp.up_proj")
        self.feature_down_proj = linear("feature_mlp.down_proj")
        self.out_head = linear("out_head")
        self.in_head = linear("in_head")
        self.anchor_out_head = linear("anchor_out_head")
        self.layers = [
            _LatticeLayer(ctx, self.key(f"layers.{index}"), hidden, num_heads,
                          self.mlp_size)
            for index in range(cfg.lilicorr_num_layers)
        ]
        self.feature_norm_weight = self.weights.f16(
            self.key("feature_norm.weight"))
        self.feature_norm_bias = self.weights.f16(
            self.key("feature_norm.bias"))
        self.output_norm = self.weights.f16(self.key("output_norm.weight"))
        self.anchor_norm = self.weights.f16(self.key("anchor_norm.weight"))
        self.slot_embedding = self._embedding("slot_embedding",
                                              (1, self.num_slots, 1, hidden))
        self.rank_embedding = self._embedding("rank_embedding",
                                              (1, 1, self.topk, hidden))
        self.attention_bias = lattice_attention_bias(
            self.weights.f16(self.key("relative_slot_bias")),
            self.weights.f16(self.key("same_slot_bias")), self.num_slots,
            self.topk, cfg.dflash2_block_size)[None]
        # factor_input_proj consumes concat(X, a, X*a); its weight is split by
        # input column block so W_a @ a is computed once per sequence.
        factor = self.key("factor_input_proj")
        if self.weights.is_nvfp4(factor):
            raise ValueError(
                "LiLiCorr factor_input_proj must be dense: its weight is "
                "split by input column block")
        weight = self.weights.f16(factor + ".weight")
        self.factor_weights = tuple(
            np.ascontiguousarray(weight[:,
                                        index * hidden:(index + 1) * hidden])
            for index in range(3))
        self.factor_bias = self.weights.f16(factor + ".bias")

    def _embedding(self, name: str, shape) -> np.ndarray:
        value = self.weights.f16(self.key(name))
        if value.size != int(np.prod(shape)):
            raise ValueError(
                f"{self.key(name)} has shape {value.shape}, expected "
                f"{shape} elements")
        return value.astype(np.float32).reshape(shape)

    def _candidate_features(self, log_probs):
        """Return FP32 ``[B, S, K, 5]``: log p, p, log p - max log p, rank / (K - 1), [rank == 0]."""
        topk = self.topk
        log_probs = log_probs.unsqueeze(3, 3)
        rank = np.arange(topk, dtype=np.float32)
        shape = F.shape_of(log_probs)

        def broadcast(values: np.ndarray, name: str):
            constant = F.constant(values.reshape(1, 1, topk, 1), name)
            return F.dynamic_slice(constant, (0, 0, 0, 0),
                                   shape,
                                   stride=(0, 0, 1, 1))

        return F.concatenate([
            log_probs,
            log_probs.exp(),
            log_probs -
            log_probs.reduce(trt.ReduceOperation.MAX, 1 << 2, True),
            broadcast(rank / float(max(topk - 1, 1)),
                      "lilicorr_rank_fraction"),
            broadcast((rank == 0).astype(np.float32), "lilicorr_is_top1"),
        ], 3)

    def _feature_mlp(self, features):
        hidden = _project(self.feature_up_proj, features,
                          self.hidden_size).cast(trt.float32).silu()
        return _project(self.feature_down_proj, hidden,
                        self.hidden_size).cast(trt.float32)

    def forward(self, token_embeddings, log_probs, pass_hidden, anchor_hidden,
                anchor_valid):
        """Return FP32 transition scores ``[B, S, K(prev), K(next)]``.

        Args:
            token_embeddings: ``[B, S, K, H]`` target input embeddings of the candidates.
            log_probs: ``[B, S, K]`` FP32 full-vocabulary log-probabilities.
            pass_hidden: ``[B, S, H]`` drafter hidden of the predicted slots.
            anchor_hidden: ``[B, H]`` ``hidden_norm(fc(...))`` of the last valid delta row.
            anchor_valid: ``[B, 1]`` FP32; 0 zeroes the projected anchor.

        Slot 0 holds the anchor start scores broadcast over ``prev``; slot
        ``s > 0`` scores candidate ``prev`` of slot ``s - 1`` against
        candidate ``next`` of slot ``s``. Scores are scaled by ``logit_scale``.
        """
        cfg = self.cfg
        hidden, slots, topk = self.hidden_size, self.num_slots, self.topk
        eps = cfg.rms_norm_eps
        batch = F.shape_of(log_probs)[0:1]

        nodes = _project(self.token_proj, token_embeddings,
                         hidden).cast(trt.float32)
        nodes = nodes + _project(self.pass_hidden_proj, pass_hidden,
                                 hidden).cast(trt.float32).unsqueeze(2, 3)
        nodes = nodes + self._feature_mlp(
            _layer_norm_f32(self._candidate_features(log_probs),
                            self.feature_norm_weight, self.feature_norm_bias,
                            _FEATURE_NORM_EPS))
        nodes = nodes + F.constant(self.slot_embedding,
                                   "lilicorr_slot_embedding")
        nodes = nodes + F.constant(self.rank_embedding,
                                   "lilicorr_rank_embedding")

        lattice = nodes.reshape((0, slots * topk, hidden))
        attention_bias = F.constant(self.attention_bias,
                                    "lilicorr_lattice_attention_bias")
        for layer in self.layers:
            lattice = layer(lattice, attention_bias)
        states = _rms_norm_f32(lattice, self.output_norm, eps).reshape(
            (0, slots, topk, hidden))

        anchor = _project(self.context_proj, anchor_hidden,
                          hidden).cast(trt.float32)
        anchor = _rms_norm_f32(anchor * anchor_valid, self.anchor_norm, eps)
        anchor_row = anchor.reshape((0, 1, 1, hidden))
        state_weight, anchor_weight, product_weight = self.factor_weights
        pre = F.linear_with_weights(states.cast(trt.float16), state_weight,
                                    self.factor_bias, 4).cast(trt.float32)
        pre = pre + F.linear_with_weights(anchor_row.cast(
            trt.float16), anchor_weight, None, 4).cast(trt.float32)
        pre = pre + F.linear_with_weights((states * anchor_row).cast(
            trt.float16), product_weight, None, 4).cast(trt.float32)
        factor = pre.silu()

        factor_dim, vector_eps = self.factor_dim, cfg.lilicorr_vector_eps
        out_vec = _normalize(
            _project(self.out_head, factor, factor_dim).cast(trt.float32),
            vector_eps)
        in_vec = _normalize(
            _project(self.in_head, factor, factor_dim).cast(trt.float32),
            vector_eps)
        anchor_out = _normalize(
            _project(self.anchor_out_head, anchor,
                     factor_dim).cast(trt.float32), vector_eps)

        start = F.matmul(in_vec[:, 0:1],
                         anchor_out.reshape((0, 1, factor_dim, 1))).reshape(
                             (0, 1, 1, topk))
        first = F.dynamic_slice(start, (0, 0, 0, 0), (batch, 1, topk, topk),
                                stride=(1, 1, 0, 1))
        pair = F.matmul(out_vec[:, 0:slots - 1],
                        in_vec[:, 1:slots],
                        transpose_rhs=True)
        return F.concatenate([first, pair], 1) * cfg.lilicorr_logit_scale
