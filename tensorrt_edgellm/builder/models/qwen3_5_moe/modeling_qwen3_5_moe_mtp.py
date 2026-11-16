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
"""Qwen3.5 MoE native-MTP checkpoint-direct draft graph."""

from ...ops import Linear, NetworkModule, RMSNorm
from ...ops import functional as F
from ..qwen3_5.modeling_qwen3_5_mtp import (Qwen35MtpDecoderLayer,
                                            Qwen35MtpDraftModel)
from . import weights as weight_conversion
from .modeling_qwen3_5_moe_sparse_moe import Qwen3_5MoeSparseMoeBlock

__all__ = ["Qwen3_5MoeMtpDecoderLayer", "Qwen3_5MoeMtpDraftModel"]


class Qwen3_5MoeMtpSparseMoeBlock(Qwen3_5MoeSparseMoeBlock):
    """The MTP experts are stacked BF16 tensors, not base-layer NVFP4 experts."""

    def forward(self, hidden_states):
        cfg = self.cfg
        experts_prefix = self.experts.prefix
        expert_weights = self.weights.parameter_value(
            "fp16",
            experts_prefix,
            lambda: weight_conversion.mtp_fp16_expert_specs(
                self.weights, experts_prefix, cfg.num_experts, cfg.hidden_size,
                cfg.moe_intermediate_size),
            lambda: weight_conversion.prepare_mtp_fp16_experts(
                self.weights, experts_prefix, cfg.num_experts, cfg.hidden_size,
                cfg.moe_intermediate_size),
        )
        routed = F.fp16_moe(
            self.gate(hidden_states),
            hidden_states,
            expert_weights,
            cfg.num_experts,
            cfg.num_experts_per_tok,
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            weight_prefix=experts_prefix,
            weight_bindings=weight_conversion.mtp_fp16_expert_bindings(
                self.weights, experts_prefix),
            norm_topk_prob=self.norm_topk_prob,
        )
        shared = self.shared_expert(hidden_states)
        gate = self.shared_expert_gate(hidden_states).sigmoid()
        return routed + shared * gate


class Qwen3_5MoeMtpDecoderLayer(Qwen35MtpDecoderLayer):
    """Qwen3.5 native-MTP layer with the model family's sparse MoE FFN."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.mlp = Qwen3_5MoeMtpSparseMoeBlock(ctx, self.key("mlp"))


class Qwen3_5MoeMtpDraftModel(Qwen35MtpDraftModel):
    """Qwen3.5 MoE native-MTP draft model."""

    def __init__(self, ctx) -> None:
        NetworkModule.__init__(self, ctx)
        eps = ctx.cfg.rms_norm_eps
        self.pre_embed_norm = RMSNorm(ctx,
                                      "pre_fc_norm_embedding",
                                      eps,
                                      unit_offset=True)
        self.pre_hidden_norm = RMSNorm(ctx,
                                       "pre_fc_norm_hidden",
                                       eps,
                                       unit_offset=True)
        self.fc = Linear(ctx, "fc")
        self.layers = [
            Qwen3_5MoeMtpDecoderLayer(ctx, f"layers.{index}")
            for index in range(ctx.cfg.num_hidden_layers)
        ]
        self.norm = RMSNorm(ctx, "norm", eps, unit_offset=True)
        self.lm_head = Linear(ctx,
                              ctx.weights.causal_lm_head_prefix("mtp.lm_head"))
