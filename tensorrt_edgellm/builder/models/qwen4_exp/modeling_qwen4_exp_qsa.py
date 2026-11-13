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
"""Qwen4-Exp sparse attention: the projections around QsaAttentionPlugin.

The implementation owns qk-norm, partial RoPE, the paged-KV write, the block
indexer and the sparse GQA itself. This module owns the projections, the packed
QKV layout and the output gate. Like Qwen3.5 gated attention, ``q_proj`` emits
query and gate together: per head the first ``head_dim`` values are the query
and the second ``head_dim`` are the gate.

The sparse-attention layer uses the token-major ragged ``QsaAttentionPlugin``
interface (issue #1027); an older plugin is rejected while the network is built.
"""

import numpy as np

from ...ops import BuildContext, Linear, Module, RaggedDecoderInputs, Tensor
from ...ops import functional as F

__all__ = ["Qwen4ExpQsaAttention"]


class Qwen4ExpQsaAttention(Module):
    """Projections, output gate and indexer around one sparse-attention op."""

    def __init__(self, ctx: BuildContext, prefix: str) -> None:
        super().__init__(ctx, prefix)
        cfg = ctx.cfg
        for name in ("indexer_n_heads", "indexer_head_dim", "indexer_budget",
                     "indexer_compress_ratio"):
            if getattr(cfg, name) <= 0:
                raise ValueError(
                    f"Qwen4-Exp sparse attention requires {name} in config.json"
                )
        # index_qk packs the indexer's query heads plus a single KV head, so a
        # wider one would silently mis-size the projection.
        if cfg.indexer_kv_heads != 1:
            raise ValueError("QsaAttentionPlugin packs one indexer KV head; "
                             f"config.json says {cfg.indexer_kv_heads}")
        self.q_proj = Linear(ctx, self.key("q_proj"))
        self.k_proj = Linear(ctx, self.key("k_proj"))
        self.v_proj = Linear(ctx, self.key("v_proj"))
        self.o_proj = Linear(ctx, self.key("o_proj"))
        self.index_qk_proj = Linear(ctx, self.key("indexer.index_qk_proj"))

    def _gamma(self, prefix: str, width: int, unit_offset: bool) -> Tensor:
        """Materialize one norm gain in the convention ``F.qsa_attention`` documents."""
        weight = self.weights.f16(self.key(prefix))
        if weight.shape != (width, ):
            raise ValueError(f"{self.key(prefix)} must have shape {(width,)}, "
                             f"got {weight.shape}")
        if unit_offset:
            weight = weight + np.float16(1.0)
        return F.constant(weight, prefix.replace(".", "_"))

    def forward(self,
                hyper_states: Tensor,
                past_key_value: Tensor,
                rope_rotary_cos_sin: Tensor,
                ragged: RaggedDecoderInputs,
                tree_parent_ids: Tensor = None,
                tree_depths: Tensor = None):
        cfg = self.cfg
        heads, head_dim = cfg.num_attention_heads, cfg.head_dim
        projected = self.q_proj(hyper_states).reshape((0, heads, head_dim * 2))
        query = projected.slice_last_dim(0, head_dim, 3)
        gate = projected.slice_last_dim(head_dim, head_dim, 3)
        qkv = F.concatenate((query.reshape(
            (0, heads * head_dim)), self.k_proj(hyper_states),
                             self.v_proj(hyper_states)), 1)

        output, present = F.qsa_attention(
            qkv,
            self.index_qk_proj(hyper_states),
            past_key_value,
            rope_rotary_cos_sin,
            ragged,
            self._gamma("q_norm.weight", head_dim, unit_offset=True),
            self._gamma("k_norm.weight", head_dim, unit_offset=True),
            self._gamma("indexer.q_layernorm.weight",
                        cfg.indexer_head_dim,
                        unit_offset=False),
            self._gamma("indexer.k_layernorm.weight",
                        cfg.indexer_head_dim,
                        unit_offset=False),
            num_q_heads=heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_size=head_dim,
            indexer_n_heads=cfg.indexer_n_heads,
            indexer_head_dim=cfg.indexer_head_dim,
            indexer_budget=cfg.indexer_budget,
            indexer_compress_ratio=cfg.indexer_compress_ratio,
            attention_scale=cfg.attention_scaling,
            rms_norm_eps=cfg.rms_norm_eps,
            tree_parent_ids=tree_parent_ids,
            tree_depths=tree_depths)
        output = (output * gate.sigmoid()).reshape((0, heads * head_dim))
        return self.o_proj(output), present
