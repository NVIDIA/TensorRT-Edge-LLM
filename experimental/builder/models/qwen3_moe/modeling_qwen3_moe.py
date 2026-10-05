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
"""Qwen3-MoE checkpoint-direct model.

The graph reads q/k/v/o projections, router weights, expert gate/up/down
weights, final normalization, and language-model head tensors from the
checkpoint. Expert tensors are packed into the inputs required by the
architecture-selected NVFP4 MoE implementation.
"""

import logging
from functools import partial
from typing import Dict, List, Tuple

import tensorrt as trt

from ...core import quantization
from ...core.config import DeviceConfig
from ...ops import (BuildContext, GatedExperts, GatedMLP, Linear, Module,
                    NetworkModule, RMSNorm, Tensor, TopKRouter)
from ...ops import functional as F
from ...ops import prepare_gated_int4_weights, prepare_gated_nvfp4_weights
from ...ops.ragged import RaggedDecoderInputs, add_ragged_decoder_inputs
from . import weights as weight_conversion
from .modeling_qwen3_moe_layers import Qwen3MoeAttention

LOGGER = logging.getLogger("builder.qwen3_moe")

__all__ = [
    "Qwen3MoeSparseMoeBlock",
    "Qwen3MoeDecoderLayer",
    "Qwen3MoeModel",
    "Qwen3MoeForCausalLM",
]


def _is_moe_layer(config: DeviceConfig, layer_idx: int) -> bool:
    if layer_idx in config.mlp_only_layers:
        return False
    return (config.num_experts > 0
            and (layer_idx + 1) % config.decoder_sparse_step == 0)


class Qwen3MoeSparseMoeBlock(Module):
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
        self.norm_topk_prob = int(ctx.cfg.norm_topk_prob)
        self.routed_scaling_factor = ctx.cfg.routed_scaling_factor

    def forward(self, hidden_states: Tensor) -> Tensor:
        cfg = self.cfg
        router_logits = self.gate(hidden_states)
        if cfg.quant_type == quantization.QUANT_NVFP4:
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
            raise ValueError("Qwen MoE experts require NVFP4 or INT4 GPTQ; "
                             f"got {cfg.quant_type!r}")
        return routed


class Qwen3MoeDecoderLayer(Module):
    """Decoder layer: attention and feed-forward with residuals."""

    def __init__(self, ctx: BuildContext, prefix: str, layer_idx: int) -> None:
        super().__init__(ctx, prefix)
        eps = ctx.cfg.rms_norm_eps
        self.self_attn = Qwen3MoeAttention(ctx, self.key("self_attn"))
        self.input_layernorm = RMSNorm(ctx, self.key("input_layernorm"), eps)
        self.post_attention_layernorm = RMSNorm(
            ctx, self.key("post_attention_layernorm"), eps)
        if _is_moe_layer(ctx.cfg, layer_idx):
            self.mlp = Qwen3MoeSparseMoeBlock(ctx, self.key("mlp"))
        else:
            self.mlp = GatedMLP(ctx, self.key("mlp"))

    def forward(
        self,
        hidden_states: Tensor,
        past_key_value: Tensor,
        rope_rotary_cos_sin: Tensor,
        ragged: RaggedDecoderInputs,
        attention_mask: Tensor = None,
        attention_pos_id: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        normed = self.input_layernorm(hidden_states)
        attn_output, present_key_value = self.self_attn(
            normed, past_key_value, rope_rotary_cos_sin, ragged,
            attention_mask, attention_pos_id)
        hidden_states = hidden_states + attn_output

        normed = self.post_attention_layernorm(hidden_states)
        feed_forward = self.mlp(normed)
        hidden_states = hidden_states + feed_forward
        return hidden_states, present_key_value


class Qwen3MoeModel(Module):
    """Decoder stack and final norm.

    ``inputs_embeds`` is a graph input, so the embedding table is not emitted.
    """

    def __init__(self, ctx: BuildContext, prefix: str = "model") -> None:
        super().__init__(ctx, prefix)
        self.layers = [
            Qwen3MoeDecoderLayer(ctx, self.key(f"layers.{i}"), i)
            for i in range(ctx.cfg.num_hidden_layers)
        ]
        self.norm = RMSNorm(ctx, self.key("norm"), ctx.cfg.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: Tensor,
        past_key_values: List[Tensor],
        rope_rotary_cos_sin: Tensor,
        ragged: RaggedDecoderInputs,
        attention_mask: Tensor = None,
        attention_pos_id: Tensor = None,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        hidden_states = inputs_embeds
        present_key_values: List[Tensor] = []
        all_hidden_states: List[Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            LOGGER.info("building layer %d/%d", layer_idx + 1,
                        len(self.layers))
            hidden_states, present = layer(hidden_states,
                                           past_key_values[layer_idx],
                                           rope_rotary_cos_sin, ragged,
                                           attention_mask, attention_pos_id)
            present_key_values.append(present)
            all_hidden_states.append(hidden_states)
        return self.norm(hidden_states), present_key_values, all_hidden_states


class Qwen3MoeForCausalLM(NetworkModule):
    """Top-level transformer, language-model head, and engine I/O declaration."""

    def __init__(self, ctx: BuildContext) -> None:
        super().__init__(ctx, "")
        self.model = Qwen3MoeModel(ctx, "model")
        lm_key = ("lm_head" if ctx.weights.has("lm_head.weight") or
                  ctx.weights.has("lm_head.qweight") else "model.embed_tokens")
        self.lm_head = Linear(ctx, lm_key)

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        hidden = cfg.hidden_size
        result = {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16, (-1, hidden)),
            "past_key_values": [
                self.add_input(f"past_key_values_{i}", kv_dtype,
                               (2, -1, F.KV_PAGE_SIZE, cfg.num_key_value_heads,
                                cfg.head_dim))
                for i in range(cfg.num_hidden_layers)
            ],
            "rope_rotary_cos_sin":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, cfg.rotary_dim)),
        }
        result.update(add_ragged_decoder_inputs(self.add_input).as_dict())
        if cfg.engine_role == "base":
            result["attention_pos_id"] = self.add_input(
                "attention_position_ids", trt.int32, (-1, ))
            result["attention_mask"] = self.add_input("packed_attention_mask",
                                                      trt.int32, (-1, -1))
        else:
            result["attention_pos_id"] = None
            result["attention_mask"] = None
        return result

    def forward(self, **io):
        ragged = RaggedDecoderInputs.from_dict(io)
        outputs = {}
        hidden_states, present_key_values, all_hidden_states = self.model(
            io["inputs_embeds"], io["past_key_values"],
            io["rope_rotary_cos_sin"], ragged, io["attention_mask"],
            io["attention_pos_id"])
        selected = F.gather_token_rows(hidden_states, ragged.logits_indices)
        logits16 = self.lm_head(selected)
        logits = F.cast(logits16, trt.float32)

        outputs["logits"] = logits
        if self.cfg.engine_role == "base":
            outputs["hidden_states"] = F.hidden_state_feedback(
                hidden_states, all_hidden_states, self.cfg)
        for layer_idx, present in enumerate(present_key_values):
            outputs[f"present_key_values_{layer_idx}"] = present
        return outputs
