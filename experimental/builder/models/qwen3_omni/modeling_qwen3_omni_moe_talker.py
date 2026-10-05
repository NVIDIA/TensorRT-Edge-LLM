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
"""Qwen3-Omni-MoE talker network.

Unlike the thinker, every Transformers talker decoder layer uses routed
experts plus a gated shared expert. Attention and normalization are the only
submodules intentionally reused from the thinker, matching the provider code.
"""

import logging
from functools import partial
from typing import Dict, List, Tuple

import tensorrt as trt

from ...core import quantization
from ...ops import (BuildContext, GatedExperts, GatedMLP, Linear, Module,
                    NetworkModule, RMSNorm, Tensor, TopKRouter)
from ...ops import functional as F
from ...ops import prepare_gated_int4_weights, prepare_gated_nvfp4_weights
from ...ops.ragged import RaggedDecoderInputs, add_ragged_decoder_inputs
from . import weights as weight_conversion
from .modeling_qwen3_omni_moe_text import Qwen3OmniMoeThinkerTextAttention

LOGGER = logging.getLogger("builder.qwen3_omni_moe.talker")

__all__ = [
    "Qwen3OmniMoeTalkerTextSparseMoeBlock",
    "Qwen3OmniMoeTalkerDecoderLayer",
    "Qwen3OmniMoeTalkerModel",
    "Qwen3OmniMoeTalker",
]


class Qwen3OmniMoeTalkerTextSparseMoeBlock(Module):
    """Talker routed experts combined with a sigmoid-gated shared expert."""

    def __init__(self, ctx: BuildContext, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.gate = TopKRouter(ctx, self.key("gate"))
        self.experts = GatedExperts(ctx, self.key("experts"))
        self.shared_expert = GatedMLP(ctx, self.key("shared_expert"))
        self.shared_expert_gate = Linear(ctx, self.key("shared_expert_gate"))

    def forward(self, hidden_states: Tensor) -> Tensor:
        cfg = self.cfg
        router_logits = self.gate(hidden_states)
        if cfg.quant_type == quantization.QUANT_FP16:
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
                norm_topk_prob=int(cfg.norm_topk_prob),
            )
        elif cfg.quant_type == quantization.QUANT_NVFP4:
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
                                 F.MoeActivation.SWIGLU,
                                 cfg.n_group,
                                 cfg.topk_group,
                                 int(cfg.norm_topk_prob),
                                 cfg.routed_scaling_factor,
                                 F.MoeRouting.SOFTMAX_TOPK,
                                 self.ctx.options.sm12x,
                                 weight_prefix=self.experts.prefix,
                                 weight_bindings=bindings)
        elif cfg.quant_type == quantization.QUANT_INT4_GPTQ:

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
                "Qwen3-Omni-MoE talker experts require FP16, NVFP4, or INT4 "
                f"GPTQ; got {cfg.quant_type!r}")

        shared = self.shared_expert(hidden_states)
        shared_gate = self.shared_expert_gate(hidden_states).sigmoid()
        return routed + shared * shared_gate


class Qwen3OmniMoeTalkerDecoderLayer(Module):
    """One pre-normalized talker layer with an always-sparse MLP block."""

    def __init__(self, ctx: BuildContext, prefix: str,
                 layer_index: int) -> None:
        del layer_index
        super().__init__(ctx, prefix)
        self.self_attn = Qwen3OmniMoeThinkerTextAttention(
            ctx, self.key("self_attn"))
        self.input_layernorm = RMSNorm(ctx, self.key("input_layernorm"),
                                       ctx.cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            ctx, self.key("post_attention_layernorm"), ctx.cfg.rms_norm_eps)
        self.mlp = Qwen3OmniMoeTalkerTextSparseMoeBlock(ctx, self.key("mlp"))

    def forward(
        self,
        hidden_states: Tensor,
        past_key_value: Tensor,
        rope_rotary_cos_sin: Tensor,
        ragged: RaggedDecoderInputs,
    ) -> Tuple[Tensor, Tensor]:
        attention, present = self.self_attn(
            self.input_layernorm(hidden_states), past_key_value,
            rope_rotary_cos_sin, ragged)
        hidden_states = hidden_states + attention
        feed_forward = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = hidden_states + feed_forward
        return hidden_states, present


class Qwen3OmniMoeTalkerModel(Module):
    """Autoregressive codec-token decoder used by the talker runtime."""

    def __init__(self, ctx: BuildContext, prefix: str = "model") -> None:
        super().__init__(ctx, prefix)
        self.layers = [
            Qwen3OmniMoeTalkerDecoderLayer(ctx, self.key(f"layers.{index}"),
                                           index)
            for index in range(ctx.cfg.num_hidden_layers)
        ]
        self.norm = RMSNorm(ctx, self.key("norm"), ctx.cfg.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: Tensor,
        past_key_values: List[Tensor],
        rope_rotary_cos_sin: Tensor,
        ragged: RaggedDecoderInputs,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        hidden_states = inputs_embeds
        present_key_values = []
        all_hidden_states = []
        for layer_index, layer in enumerate(self.layers):
            LOGGER.info("building talker layer %d/%d", layer_index + 1,
                        len(self.layers))
            hidden_states, present = layer(hidden_states,
                                           past_key_values[layer_index],
                                           rope_rotary_cos_sin, ragged)
            present_key_values.append(present)
            all_hidden_states.append(hidden_states)
        return self.norm(hidden_states), present_key_values, all_hidden_states


class Qwen3OmniMoeTalker(NetworkModule):
    """Qwen3-Omni-MoE talker network and codec-token I/O contract."""

    def __init__(self, ctx: BuildContext) -> None:
        super().__init__(ctx, "")
        self.model = Qwen3OmniMoeTalkerModel(ctx, "model")
        self.codec_head = Linear(ctx, "codec_head")

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        io = {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, cfg.hidden_size)),
            "past_key_values": [
                self.add_input(f"past_key_values_{index}", kv_dtype,
                               (2, -1, F.KV_PAGE_SIZE, cfg.num_key_value_heads,
                                cfg.head_dim))
                for index in range(cfg.num_hidden_layers)
            ],
            "rope_rotary_cos_sin":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, cfg.rotary_dim)),
        }
        io.update(add_ragged_decoder_inputs(self.add_input).as_dict())
        return io

    def forward(self, **io):
        ragged = RaggedDecoderInputs.from_dict(io)
        outputs = {}
        hidden_states, present_key_values, _ = self.model(
            io["inputs_embeds"], io["past_key_values"],
            io["rope_rotary_cos_sin"], ragged)
        selected = F.gather_token_rows(hidden_states, ragged.logits_indices)
        outputs["logits"] = self.codec_head(selected).cast(trt.float32)
        outputs["hidden_states"] = hidden_states
        for layer_index, present in enumerate(present_key_values):
            outputs[f"present_key_values_{layer_index}"] = present
        return outputs
