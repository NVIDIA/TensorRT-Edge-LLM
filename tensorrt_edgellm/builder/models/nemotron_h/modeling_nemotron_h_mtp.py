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
"""Nemotron-H checkpoint-direct MTP draft graph."""

from typing import Dict

import numpy as np
import tensorrt as trt

from ...ops import (GroupedSigmoidRouter, Linear, Module, NetworkModule,
                    RaggedDecoderInputs, RMSNorm, add_ragged_decoder_inputs)
from ...ops import functional as F
from . import weights as weight_conversion
from .modeling_nemotron_h import NemotronHAttention


class _BiasFreeLayerNorm(Module):
    """Checkpoint-backed LayerNorm with no additive bias."""

    def __init__(self, ctx, prefix: str, eps: float) -> None:
        super().__init__(ctx, prefix)
        self.eps = eps

    def forward(self, hidden_states):
        weight = self.weights.fp16_parameter(self.key("weight"))
        bias = np.zeros(self.cfg.hidden_size, dtype=np.float16)
        return F.layer_norm(hidden_states, weight, bias, self.eps, rank=2)


class _Relu2MLP(Module):
    """Dense non-gated ReLU-squared expert."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.up_proj = Linear(ctx, self.key("up_proj"))
        self.down_proj = Linear(ctx, self.key("down_proj"))

    def forward(self, hidden_states):
        activated = self.up_proj(hidden_states).relu()
        return self.down_proj(activated * activated)


class _MtpFp16MoE(Module):
    """Sigmoid-grouped routed FP16 experts plus a shared expert."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.gate = GroupedSigmoidRouter(ctx, self.key("gate"))
        self.experts_prefix = self.key("experts")
        self.shared_experts = _Relu2MLP(ctx, self.key("shared_experts"))

    def forward(self, hidden_states):
        cfg = self.cfg
        expert_weights = self.weights.parameter_value(
            "fp16",
            self.experts_prefix,
            lambda: weight_conversion.mtp_fp16_expert_specs(
                cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size),
            lambda: weight_conversion.prepare_mtp_fp16_experts(
                self.weights, self.experts_prefix, cfg.num_experts, cfg.
                hidden_size, cfg.moe_intermediate_size),
        )
        bindings = weight_conversion.mtp_fp16_expert_bindings(
            self.weights, self.experts_prefix, cfg.num_experts)
        routed = F.fp16_moe(
            self.gate(hidden_states),
            hidden_states,
            expert_weights,
            cfg.num_experts,
            cfg.num_experts_per_tok,
            cfg.hidden_size,
            expert_weights["padded_intermediate"],
            weight_prefix=self.experts_prefix,
            weight_bindings=bindings,
            norm_topk_prob=int(cfg.norm_topk_prob),
            activation_type=F.MoeActivation.RELU2,
            routing_mode=F.MoeRouting.SIGMOID_GROUP_TOPK,
            n_group=cfg.n_group,
            topk_group=cfg.topk_group,
            routed_scaling_factor=cfg.routed_scaling_factor,
            e_score_correction_bias=F.constant(self.gate.correction,
                                               "e_score_correction_bias"),
        )
        return routed + self.shared_experts(hidden_states)


class _MtpAttentionLayer(Module):
    """Fusion and attention block at the start of the predictor."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        eps = ctx.cfg.rms_norm_eps
        self.enorm = RMSNorm(ctx, self.key("enorm"), eps)
        self.hnorm = RMSNorm(ctx, self.key("hnorm"), eps)
        self.eh_proj = Linear(ctx, self.key("eh_proj"))
        self.norm = RMSNorm(ctx, self.key("norm"), eps)
        self.mixer = NemotronHAttention(ctx, self.key("mixer"))

    def forward(self, inputs_embeds, hidden_states, past, rope, ragged,
                attention_mask, attention_pos_id):
        merged = F.concatenate(
            (self.enorm(inputs_embeds), self.hnorm(hidden_states)), 1)
        hidden_states = self.eh_proj(merged)
        attention, present = self.mixer(self.norm(hidden_states), past, rope,
                                        ragged, attention_mask,
                                        attention_pos_id)
        return hidden_states + attention, present


class _MtpMoeLayer(Module):
    """Residual MoE block at the end of the predictor."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.norm = RMSNorm(ctx, self.key("norm"), ctx.cfg.rms_norm_eps)
        self.mixer = _MtpFp16MoE(ctx, self.key("mixer"))
        self.final_norm = _BiasFreeLayerNorm(ctx, self.key("final_layernorm"),
                                             ctx.cfg.rms_norm_eps)

    def forward(self, hidden_states):
        return hidden_states + self.mixer(self.norm(hidden_states))


class NemotronHMtpDraftModel(NetworkModule):
    """Nemotron-H embedded MTP predictor and runtime I/O contract."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        if ctx.cfg.layer_types != ["attention", "moe"]:
            raise ValueError(
                "Nemotron-H direct MTP currently requires attention/MoE blocks"
            )
        self.attention = _MtpAttentionLayer(ctx, "layers.0")
        self.moe = _MtpMoeLayer(ctx, "layers.1")
        self.lm_head = Linear(ctx,
                              ctx.weights.causal_lm_head_prefix("mtp.lm_head"))

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        io = {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, cfg.hidden_size)),
            "past":
            self.add_input("past_key_values_0", kv_dtype,
                           (2, -1, F.KV_PAGE_SIZE, cfg.num_key_value_heads,
                            cfg.head_dim)),
            "rope":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, cfg.rotary_dim)),
            "base_hidden":
            self.add_input("hidden_states_input", trt.float16,
                           (-1, cfg.hidden_size)),
            "draft_hidden":
            self.add_input("hidden_states_from_draft", trt.float16,
                           (-1, cfg.hidden_size)),
            "attention_pos_id":
            self.add_input("attention_position_ids", trt.int32, (-1, )),
            "attention_mask":
            self.add_input("packed_attention_mask", trt.int32, (-1, -1)),
        }
        io.update(add_ragged_decoder_inputs(self.add_input).as_dict())
        return io

    def forward(self, **io):
        ragged = RaggedDecoderInputs.from_dict(io)
        source = io["base_hidden"] + io["draft_hidden"]
        hidden, present = self.attention(io["inputs_embeds"], source,
                                         io["past"], io["rope"], ragged,
                                         io["attention_mask"],
                                         io["attention_pos_id"])
        hidden = self.moe(hidden)
        selected = F.gather_token_rows(hidden, ragged.logits_indices)
        selected = self.moe.final_norm(selected)
        logits = self.lm_head(selected).cast(trt.float32).log_softmax(1)
        return {
            "logits": logits,
            "hidden_states": selected,
            "present_key_values_0": present,
        }
