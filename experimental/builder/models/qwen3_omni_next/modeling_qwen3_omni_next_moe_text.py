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
"""Checkpoint-direct Qwen3-Omni-Next sparse-MoE Thinker."""

from functools import partial

from ...core import quantization
from ...ops import GatedExperts, GatedMLP, Linear, Module, Tensor, TopKRouter
from ...ops import functional as F
from ...ops import prepare_gated_int4_weights, prepare_gated_nvfp4_weights
from . import weights as weight_conversion
from .modeling_qwen3_omni_next_text import (Qwen3OmniNextDecoderLayer,
                                            Qwen3OmniNextThinker,
                                            Qwen3OmniNextThinkerModel)

__all__ = [
    "Qwen3OmniNextSparseMoeBlock",
    "Qwen3OmniNextMoeDecoderLayer",
    "Qwen3OmniNextMoeThinkerModel",
    "Qwen3OmniNextMoeThinker",
]


class Qwen3OmniNextSparseMoeBlock(Module):
    """Routed experts plus the checkpoint's sigmoid-gated shared expert."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.gate = TopKRouter(ctx, self.key("gate"))
        self.experts = GatedExperts(ctx, self.key("experts"))
        self.shared_expert = GatedMLP(ctx, self.key("shared_expert"))
        self.shared_expert_gate = Linear(ctx, self.key("shared_expert_gate"))

    def _fp16(self, hidden_states: Tensor, router_logits: Tensor) -> Tensor:
        cfg = self.cfg
        expert_weights = self.weights.parameter_value(
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
        return F.fp16_moe(
            router_logits,
            hidden_states,
            expert_weights,
            cfg.num_experts,
            cfg.num_experts_per_tok,
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            weight_prefix=self.experts.prefix,
            weight_bindings=bindings,
            norm_topk_prob=int(cfg.norm_topk_prob),
        )

    def _nvfp4(self, hidden_states: Tensor, router_logits: Tensor) -> Tensor:
        cfg = self.cfg
        expert_weights = self.weights.parameter_value(
            "nvfp4_moe",
            self.experts.prefix,
            lambda: weight_conversion.nvfp4_expert_specs(
                self.weights, self.experts.prefix, cfg.num_experts),
            lambda: prepare_gated_nvfp4_weights(
                self.ctx, self.experts, weight_conversion.repack_nvfp4_experts
            ),
        )
        bindings = weight_conversion.nvfp4_expert_bindings(
            self.weights, self.experts.prefix, cfg.num_experts,
            self.ctx.options.sm12x)
        return F.nvfp4_moe(
            router_logits,
            hidden_states,
            expert_weights,
            cfg.num_experts,
            cfg.num_experts_per_tok,
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            F.MoeActivation.SWIGLU,
            cfg.n_group,
            cfg.topk_group,
            int(cfg.norm_topk_prob),
            cfg.routed_scaling_factor,
            F.MoeRouting.SOFTMAX_TOPK,
            self.ctx.options.sm12x,
            weight_prefix=self.experts.prefix,
            weight_bindings=bindings,
        )

    def _int4(self, hidden_states: Tensor, router_logits: Tensor) -> Tensor:
        cfg = self.cfg

        def materialize():
            load_projection = partial(self.weights.gptq_expert_projection,
                                      self.experts.prefix)
            return prepare_gated_int4_weights(self.ctx, load_projection)

        expert_weights = self.weights.parameter_value(
            "int4_moe",
            self.experts.prefix,
            lambda: weight_conversion.int4_expert_specs(
                self.weights, self.experts.prefix, cfg.num_experts),
            materialize,
        )
        bindings = weight_conversion.int4_expert_bindings(
            self.weights, self.experts.prefix, cfg.num_experts, cfg.group_size,
            cfg.quant.gptq_zero_point_offset)
        return F.int4_moe(
            router_logits,
            hidden_states,
            expert_weights,
            cfg.num_experts,
            cfg.num_experts_per_tok,
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            cfg.group_size,
            weight_prefix=self.experts.prefix,
            weight_bindings=bindings,
            zero_point_offset=cfg.quant.gptq_zero_point_offset,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        router_logits = self.gate(hidden_states)
        if self.cfg.quant_type == quantization.QUANT_FP16:
            routed = self._fp16(hidden_states, router_logits)
        elif self.cfg.quant_type == quantization.QUANT_NVFP4:
            routed = self._nvfp4(hidden_states, router_logits)
        elif self.cfg.quant_type == quantization.QUANT_INT4_GPTQ:
            routed = self._int4(hidden_states, router_logits)
        else:
            raise ValueError(
                "Qwen3-Omni-Next experts require FP16, NVFP4, or INT4 GPTQ; "
                f"got {self.cfg.quant_type!r}")
        shared = self.shared_expert(hidden_states)
        shared_gate = self.shared_expert_gate(hidden_states).sigmoid()
        return routed + shared * shared_gate


class Qwen3OmniNextMoeDecoderLayer(Qwen3OmniNextDecoderLayer):
    """One hybrid Next layer with a routed and shared-expert FFN."""

    mlp_class = Qwen3OmniNextSparseMoeBlock


class Qwen3OmniNextMoeThinkerModel(Qwen3OmniNextThinkerModel):
    """Sparse-MoE variant of the model-owned Next Thinker stack."""

    layer_class = Qwen3OmniNextMoeDecoderLayer


class Qwen3OmniNextMoeThinker(Qwen3OmniNextThinker):
    """Sparse-MoE Next Thinker with the same explicit runtime contract."""

    model_class = Qwen3OmniNextMoeThinkerModel
