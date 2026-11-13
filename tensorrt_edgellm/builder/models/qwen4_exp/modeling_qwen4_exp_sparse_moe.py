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
"""Qwen4-Exp router, routed experts, and shared expert.

The official checkpoint quantizes the routed experts and nothing else in this
block: the router, the shared expert and its gate stay in checkpoint
precision, so only the expert bank reaches a quantized MoE operation. The
decoder stack's banks are NVFP4; the MTP draft's bank can be FP8 or BF16.
"""

import logging

from ...core import quantization
from ...ops import (BuildContext, GatedExperts, GatedMLP, Linear, Module,
                    Tensor, TopKRouter)
from ...ops import functional as F
from ...ops import (gated_w4a16_group_size, prepare_gated_nvfp4_weights,
                    prepare_gated_w4a16_weights)
from ...weight_packing import nvfp4 as nvfp4_pack
from . import weights as weight_conversion

LOGGER = logging.getLogger("builder.qwen4_exp")

__all__ = ["Qwen4ExpSparseMoeBlock"]


class Qwen4ExpSparseMoeBlock(Module):
    """Sparse MoE block containing the router, experts and shared expert.

    ``prefix`` is ``model.layers.{i}.mlp``.
    """

    def __init__(self, ctx: BuildContext, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.gate = TopKRouter(ctx, self.key("gate"))
        self.experts = GatedExperts(ctx, self.key("experts"))
        self.activation_type = F.MoeActivation.SWIGLU
        self.routing_mode = F.MoeRouting.SOFTMAX_TOPK
        self.n_group = ctx.cfg.n_group
        self.topk_group = ctx.cfg.topk_group
        self.norm_topk_prob = int(ctx.cfg.norm_topk_prob)
        self.routed_scaling_factor = ctx.cfg.routed_scaling_factor
        self.shared_expert = GatedMLP(ctx, self.key("shared_expert"))
        self.shared_expert_gate = Linear(ctx,
                                         self.key("shared_expert_gate"),
                                         tensor_parallel=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        cfg = self.cfg
        expert_quant = weight_conversion.expert_quant_type(
            self.weights, self.experts.prefix)
        router_logits = self.gate(hidden_states)
        if cfg.engine_role == "draft" and expert_quant in (
                quantization.QUANT_FP16, quantization.QUANT_FP8,
                quantization.QUANT_FP8_BLOCK):
            return self._floating_routed(
                router_logits, hidden_states) + self._shared(hidden_states)
        if expert_quant != quantization.QUANT_NVFP4:
            raise ValueError(
                "Qwen4-Exp routed experts require NVFP4 (floating experts "
                f"only for the MTP draft); {self.experts.prefix} is "
                f"{expert_quant!r}")
        moe_weights = self.weights.parameter_value(
            "nvfp4_moe",
            self.experts.prefix,
            lambda: weight_conversion.nvfp4_expert_specs(
                self.weights, self.experts.prefix, cfg.num_experts),
            lambda: prepare_gated_nvfp4_weights(
                self.ctx, self.experts, nvfp4_pack.pack_gated_nvfp4_experts),
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
        return routed + self._shared(hidden_states)

    def _floating_routed(self, router_logits: Tensor,
                         hidden_states: Tensor) -> Tensor:
        """Use FP16 when supported, otherwise requantize the draft to W4A16."""
        cfg = self.cfg
        if weight_conversion.fp16_moe_supports(cfg.num_experts,
                                               cfg.num_experts_per_tok):
            LOGGER.info("%s: floating experts converted to FP16", self.prefix)
            return F.fp16_moe(router_logits,
                              hidden_states,
                              weight_conversion.dequantized_fp16_experts(
                                  self.experts, cfg.hidden_size,
                                  cfg.moe_intermediate_size),
                              cfg.num_experts,
                              cfg.num_experts_per_tok,
                              cfg.hidden_size,
                              cfg.moe_intermediate_size,
                              weight_prefix=self.experts.prefix,
                              weight_bindings={},
                              norm_topk_prob=self.norm_topk_prob,
                              activation_type=self.activation_type,
                              routing_mode=self.routing_mode,
                              n_group=self.n_group,
                              topk_group=self.topk_group,
                              routed_scaling_factor=self.routed_scaling_factor)
        LOGGER.info(
            "%s: floating experts requantized to W4A16 (%d experts, "
            "top-%d is outside the FP16 MoE operation)", self.prefix,
            cfg.num_experts, cfg.num_experts_per_tok)
        # The W4A16 operation takes neither knob: it always renormalizes the
        # top-k weights and never scales them. Both match this checkpoint, and
        # a checkpoint that asked for anything else would be routed wrongly
        # without any sign of it.
        if not self.norm_topk_prob or self.routed_scaling_factor != 1.0:
            raise ValueError(
                f"{self.experts.prefix}: the W4A16 MoE operation renormalizes "
                "top-k and ignores routed_scaling_factor, but this checkpoint "
                f"asks for norm_topk_prob={bool(self.norm_topk_prob)}, "
                f"routed_scaling_factor={self.routed_scaling_factor}")
        group_size = gated_w4a16_group_size(cfg.hidden_size,
                                            cfg.moe_intermediate_size)
        return F.int4_moe(router_logits,
                          hidden_states,
                          prepare_gated_w4a16_weights(self.ctx,
                                                      self.experts,
                                                      group_size=group_size),
                          cfg.num_experts,
                          cfg.num_experts_per_tok,
                          cfg.hidden_size,
                          cfg.moe_intermediate_size,
                          group_size,
                          weight_prefix=self.experts.prefix,
                          weight_bindings={},
                          zero_point_offset=0)

    def _shared(self, hidden_states: Tensor) -> Tensor:
        shared = self.shared_expert(hidden_states)
        gate = self.shared_expert_gate(hidden_states).sigmoid()
        return shared * gate
