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
"""Qwen3-MoE projection, normalization, attention, and Qwen3MoeMLP modules."""

from typing import Tuple

from ...core import quantization
from ...ops import BuildContext, Linear, Module, RMSNorm, Tensor
from ...ops import functional as F
from ...ops import pack_qkv
from ...ops.ragged import RaggedDecoderInputs
from . import weights as weight_conversion

__all__ = [
    "BuildContext",
    "Module",
    "Qwen3MoeAttention",
    "Linear",
]


class Qwen3MoeAttention(Module):
    """Qwen3MoeAttention block with q/k/v/o projections."""

    def __init__(self, ctx: BuildContext, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.q_proj = Linear(ctx, self.key("q_proj"))
        self.k_proj = Linear(ctx, self.key("k_proj"))
        self.v_proj = Linear(ctx, self.key("v_proj"))
        self.o_proj = Linear(ctx, self.key("o_proj"))
        self.q_norm = RMSNorm(ctx, self.key("q_norm"), ctx.cfg.rms_norm_eps)
        self.k_norm = RMSNorm(ctx, self.key("k_norm"), ctx.cfg.rms_norm_eps)

    def project_qkv(self,
                    hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply the three Transformers projections or their GPTQ fusion."""
        projections = (self.q_proj, self.k_proj, self.v_proj)
        plugin_version = self.ctx.options.int4_gemm_plugin_version
        cfg = self.cfg
        query_size = cfg.num_attention_heads * cfg.head_dim
        key_value_size = cfg.num_key_value_heads * cfg.head_dim
        can_attempt_fusion = (
            self.ctx.backend == "edgellm"
            and all(projection.quant_type() == quantization.QUANT_INT4_GPTQ
                    for projection in projections)
            and not any(projection.has_adapter() for projection in projections)
            and (plugin_version == 1 or
                 (query_size % 128 == 0 and key_value_size % 128 == 0)))
        if not can_attempt_fusion:
            return tuple(
                projection(hidden_states) for projection in projections)

        descriptors = tuple(projection.weight_descriptor()
                            for projection in projections)
        descriptor = weight_conversion.fuse_gptq_qkv(descriptors,
                                                     plugin_version)
        packed = F.linear_from_weights(hidden_states,
                                       descriptor,
                                       name=self.key("qkv_proj"))
        return (packed[..., :query_size],
                packed[..., query_size:query_size + key_value_size],
                packed[..., query_size + key_value_size:query_size +
                       2 * key_value_size])

    def forward(
        self,
        hidden_states: Tensor,
        past_key_value: Tensor,
        rope_rotary_cos_sin: Tensor,
        ragged: RaggedDecoderInputs,
        attention_mask: Tensor = None,
        attention_pos_id: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        cfg = self.cfg
        q, k, v = self.project_qkv(hidden_states)

        q3 = q.reshape((0, cfg.num_attention_heads, cfg.head_dim))
        q = self.q_norm(q3, rank=3).reshape(
            (0, cfg.num_attention_heads * cfg.head_dim))
        k3 = k.reshape((0, cfg.num_key_value_heads, cfg.head_dim))
        k = self.k_norm(k3, rank=3).reshape(
            (0, cfg.num_key_value_heads * cfg.head_dim))

        qkv = pack_qkv(q, k, v, self.v_proj)
        attn, present_key_value = F.attention(
            qkv,
            past_key_value,
            rope_rotary_cos_sin,
            ragged,
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_size=cfg.head_dim,
            sliding_window_size=cfg.sliding_window_size,
            enable_fp8_kv_cache=cfg.kv_cache_quant == "fp8",
            qkv_scales=self.weights.qkv_scales(self.prefix),
            skip_softmax_scale_factor=cfg.skip_softmax_scale_factor,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
        )
        attn = attn.reshape((0, cfg.num_attention_heads * cfg.head_dim))
        return self.o_proj(attn), present_key_value
