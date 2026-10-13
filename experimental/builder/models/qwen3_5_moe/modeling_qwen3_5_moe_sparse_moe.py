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
"""Qwen3.5-MoE router and expert modules.

The graph reads q/k/v/o projections, router weights, expert gate/up/down
weights, final normalization, and language-model head tensors from the
checkpoint. Expert tensors are packed into the inputs required by the
architecture-selected NVFP4 MoE implementation.
"""

from functools import partial

from ...core import quantization
from ...ops import (BuildContext, GatedExperts, GatedMLP, Linear, Module,
                    Tensor, TopKRouter)
from ...ops import functional as F
from ...ops import prepare_gated_int4_weights, prepare_gated_nvfp4_weights
from . import weights as weight_conversion

__all__ = [
    "Qwen3_5MoeSparseMoeBlock",
]


class Qwen3_5MoeSparseMoeBlock(Module):
    """Sparse MoE block containing the router and experts.

    ``prefix`` is ``model.layers.{i}.mlp``.
    """

    def __init__(self, ctx: BuildContext, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.gate = TopKRouter(ctx, self.key("gate"))
        self.experts = GatedExperts(ctx, self.key("experts"))
        # Operation attributes.
        self.activation_type = F.MoeActivation.SWIGLU
        self.routing_mode = F.MoeRouting.SOFTMAX_TOPK
        self.n_group = ctx.cfg.n_group
        self.topk_group = ctx.cfg.topk_group
        self.norm_topk_prob = 1
        self.routed_scaling_factor = ctx.cfg.routed_scaling_factor
        self.shared_expert = GatedMLP(ctx, self.key("shared_expert"))
        self.shared_expert_gate = Linear(ctx, self.key("shared_expert_gate"))

    def forward(self, hidden_states: Tensor) -> Tensor:
        cfg = self.cfg
        router_logits = self.gate(hidden_states)
        expert_quant_type = self.weights.module_quant_type(
            self.experts.key("0.up_proj"))
        if expert_quant_type == quantization.QUANT_FP16:
            moe_weights = self.weights.parameter_value(
                "fp16",
                self.experts.prefix,
                lambda: weight_conversion.fp16_expert_specs(
                    self.weights, self.experts.prefix, cfg.num_experts),
                lambda: weight_conversion.prepare_fp16_experts(
                    self.weights,
                    self.experts.prefix,
                    cfg.num_experts,
                    cfg.hidden_size,
                    cfg.moe_intermediate_size,
                ),
            )
            bindings = weight_conversion.fp16_expert_bindings(
                self.weights, self.experts.prefix, cfg.num_experts)
            routed = F.fp16_moe(
                router_logits,
                hidden_states,
                moe_weights,
                cfg.num_experts,
                cfg.num_experts_per_tok,
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                weight_prefix=self.experts.prefix,
                weight_bindings=bindings,
                norm_topk_prob=self.norm_topk_prob,
            )
        elif expert_quant_type == quantization.QUANT_NVFP4:
            moe_weights = self.weights.parameter_value(
                "nvfp4_moe",
                self.experts.prefix,
                lambda: weight_conversion.nvfp4_expert_specs(
                    self.weights, self.experts.prefix, cfg.num_experts),
                lambda: prepare_gated_nvfp4_weights(
                    self.ctx, self.experts, weight_conversion.
                    repack_nvfp4_experts),
            )
            bindings = weight_conversion.nvfp4_expert_bindings(
                self.weights, self.experts.prefix, cfg.num_experts,
                self.ctx.options.sm12x)
            routed = F.nvfp4_moe(router_logits,
                                 hidden_states,
                                 moe_weights,
                                 cfg.num_experts,
                                 cfg.num_experts_per_tok,
                                 cfg.hidden_size,
                                 cfg.moe_intermediate_size,
                                 self.activation_type,
                                 self.n_group,
                                 self.topk_group,
                                 self.norm_topk_prob,
                                 self.routed_scaling_factor,
                                 self.routing_mode,
                                 self.ctx.options.sm12x,
                                 weight_prefix=self.experts.prefix,
                                 weight_bindings=bindings)
        elif expert_quant_type == quantization.QUANT_INT4_GPTQ:

            def materialize_int4():
                load_projection = partial(self.weights.gptq_expert_projection,
                                          self.experts.prefix)
                return prepare_gated_int4_weights(self.ctx, load_projection)

            moe_weights = self.weights.parameter_value(
                "int4_moe",
                self.experts.prefix,
                lambda: weight_conversion.int4_expert_specs(
                    self.weights, self.experts.prefix, cfg.num_experts),
                materialize_int4,
            )
            bindings = weight_conversion.int4_expert_bindings(
                self.weights, self.experts.prefix, cfg.num_experts,
                cfg.group_size, cfg.quant.gptq_zero_point_offset)
            routed = F.int4_moe(
                router_logits,
                hidden_states,
                moe_weights,
                cfg.num_experts,
                cfg.num_experts_per_tok,
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                cfg.group_size,
                weight_prefix=self.experts.prefix,
                weight_bindings=bindings,
                zero_point_offset=cfg.quant.gptq_zero_point_offset)
        else:
            raise ValueError(
                "Qwen3.5 MoE experts require FP16, NVFP4, or INT4 GPTQ; "
                f"got {expert_quant_type!r}")
        shared = self.shared_expert(hidden_states)
        gate = self.shared_expert_gate(hidden_states).sigmoid()
        return routed + shared * gate
