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
"""Qwen3-Omni-MoE thinker text network.

The module boundaries follow ``transformers.models.qwen3_omni_moe``. The
thinker has optional dense decoder layers, routed experts without a shared
expert, per-head Q/K normalization, and DeepStack visual inputs.
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
from ...ops import (pack_qkv, prepare_gated_int4_weights,
                    prepare_gated_nvfp4_weights)
from ...ops.ragged import RaggedDecoderInputs, add_ragged_decoder_inputs
from . import weights as weight_conversion

LOGGER = logging.getLogger("builder.qwen3_omni_moe.thinker")

__all__ = [
    "Linear",
    "Qwen3OmniMoeThinkerTextAttention",
    "Qwen3OmniMoeThinkerTextSparseMoeBlock",
    "Qwen3OmniMoeThinkerTextDecoderLayer",
    "Qwen3OmniMoeThinkerTextModel",
    "Qwen3OmniMoeThinker",
]


def _is_moe_layer(config: DeviceConfig, layer_index: int) -> bool:
    return (layer_index not in config.mlp_only_layers
            and config.num_experts > 0
            and (layer_index + 1) % config.decoder_sparse_step == 0)


class Qwen3OmniMoeThinkerTextAttention(Module):
    """Qwen3-Omni-MoE attention with mandatory per-head Q/K RMSNorm."""

    def __init__(self, ctx: BuildContext, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.q_proj = Linear(ctx, self.key("q_proj"))
        self.k_proj = Linear(ctx, self.key("k_proj"))
        self.v_proj = Linear(ctx, self.key("v_proj"))
        self.o_proj = Linear(ctx, self.key("o_proj"))
        self.q_norm = RMSNorm(ctx, self.key("q_norm"), ctx.cfg.rms_norm_eps)
        self.k_norm = RMSNorm(ctx, self.key("k_norm"), ctx.cfg.rms_norm_eps)

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
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self.q_norm(query.reshape(
            (0, cfg.num_attention_heads, cfg.head_dim)),
                            rank=3).reshape(
                                (0, cfg.num_attention_heads * cfg.head_dim))
        key = self.k_norm(key.reshape(
            (0, cfg.num_key_value_heads, cfg.head_dim)),
                          rank=3).reshape(
                              (0, cfg.num_key_value_heads * cfg.head_dim))

        qkv = pack_qkv(query, key, value, self.v_proj)
        attention, present_key_value = F.attention(
            qkv,
            past_key_value,
            rope_rotary_cos_sin,
            ragged,
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_size=cfg.head_dim,
            sliding_window_size=-1,
            enable_fp8_kv_cache=cfg.kv_cache_quant == "fp8",
            qkv_scales=self.weights.qkv_scales(self.prefix),
            skip_softmax_scale_factor=cfg.skip_softmax_scale_factor,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
        )
        attention = attention.reshape(
            (0, cfg.num_attention_heads * cfg.head_dim))
        return self.o_proj(attention), present_key_value


class Qwen3OmniMoeThinkerTextSparseMoeBlock(Module):
    """Thinker router and routed experts; no shared expert is present."""

    def __init__(self, ctx: BuildContext, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.gate = TopKRouter(ctx, self.key("gate"))
        self.experts = GatedExperts(ctx, self.key("experts"))

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
            return F.fp16_moe(
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
            return F.nvfp4_moe(router_logits,
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
        if cfg.quant_type == quantization.QUANT_INT4_GPTQ:

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
            return F.int4_moe(
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
        raise ValueError(
            "Qwen3-Omni-MoE experts require FP16, NVFP4, or INT4 GPTQ; "
            f"got {cfg.quant_type!r}")


class Qwen3OmniMoeThinkerTextDecoderLayer(Module):
    """One pre-normalized thinker decoder layer."""

    def __init__(self, ctx: BuildContext, prefix: str,
                 layer_index: int) -> None:
        super().__init__(ctx, prefix)
        self.self_attn = Qwen3OmniMoeThinkerTextAttention(
            ctx, self.key("self_attn"))
        self.input_layernorm = RMSNorm(ctx, self.key("input_layernorm"),
                                       ctx.cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            ctx, self.key("post_attention_layernorm"), ctx.cfg.rms_norm_eps)
        self.mlp = (Qwen3OmniMoeThinkerTextSparseMoeBlock(
            ctx, self.key("mlp")) if _is_moe_layer(ctx.cfg, layer_index) else
                    GatedMLP(ctx, self.key("mlp")))

    def forward(
        self,
        hidden_states: Tensor,
        past_key_value: Tensor,
        rope_rotary_cos_sin: Tensor,
        ragged: RaggedDecoderInputs,
        attention_mask: Tensor = None,
        attention_pos_id: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        attention, present = self.self_attn(
            self.input_layernorm(hidden_states), past_key_value,
            rope_rotary_cos_sin, ragged, attention_mask, attention_pos_id)
        hidden_states = hidden_states + attention
        feed_forward = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = hidden_states + feed_forward
        return hidden_states, present


class Qwen3OmniMoeThinkerTextModel(Module):
    """Thinker decoder stack with DeepStack visual feature injection."""

    def __init__(self, ctx: BuildContext, prefix: str = "model") -> None:
        super().__init__(ctx, prefix)
        self.layers = [
            Qwen3OmniMoeThinkerTextDecoderLayer(ctx,
                                                self.key(f"layers.{index}"),
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
        deepstack_embeds: List[Tensor],
        attention_mask: Tensor = None,
        attention_pos_id: Tensor = None,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        hidden_states = inputs_embeds
        present_key_values = []
        all_hidden_states = []
        for layer_index, layer in enumerate(self.layers):
            LOGGER.info("building thinker layer %d/%d", layer_index + 1,
                        len(self.layers))
            hidden_states, present = layer(hidden_states,
                                           past_key_values[layer_index],
                                           rope_rotary_cos_sin, ragged,
                                           attention_mask, attention_pos_id)
            if layer_index < len(deepstack_embeds):
                hidden_states = (hidden_states + deepstack_embeds[layer_index])
            present_key_values.append(present)
            all_hidden_states.append(hidden_states)
        return self.norm(hidden_states), present_key_values, all_hidden_states


class Qwen3OmniMoeThinker(NetworkModule):
    """Qwen3-Omni-MoE thinker network and engine I/O contract."""

    def __init__(self, ctx: BuildContext) -> None:
        super().__init__(ctx, "")
        self.model = Qwen3OmniMoeThinkerTextModel(ctx, "model")
        lm_key = ("lm_head" if ctx.weights.has("lm_head.weight") or
                  ctx.weights.has("lm_head.qweight") else "model.embed_tokens")
        self.lm_head = Linear(ctx, lm_key)

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
            "deepstack_embeds": [
                self.add_input(f"deepstack_embeds_{index}", trt.float16,
                               (-1, cfg.hidden_size))
                for index in range(cfg.num_deepstack_features)
            ],
        }
        io.update(add_ragged_decoder_inputs(self.add_input).as_dict())
        if cfg.engine_role == "base":
            io["attention_pos_id"] = self.add_input("attention_position_ids",
                                                    trt.int32, (-1, ))
            io["attention_mask"] = self.add_input("packed_attention_mask",
                                                  trt.int32, (-1, -1))
        else:
            io["attention_pos_id"] = None
            io["attention_mask"] = None
        return io

    def forward(self, **io):
        ragged = RaggedDecoderInputs.from_dict(io)
        outputs = {}
        hidden_states, present_key_values, all_hidden_states = self.model(
            io["inputs_embeds"], io["past_key_values"],
            io["rope_rotary_cos_sin"], ragged, io["deepstack_embeds"],
            io["attention_mask"], io["attention_pos_id"])
        selected = F.gather_token_rows(hidden_states, ragged.logits_indices)
        outputs["logits"] = self.lm_head(selected).cast(trt.float32)

        if self.cfg.engine_role == "base":
            feedback = hidden_states
        else:
            accepted = self.cfg.accept_hidden_layer
            feedback = (all_hidden_states[accepted - 1]
                        if 1 <= accepted <= len(all_hidden_states) else
                        hidden_states)
        outputs["hidden_states"] = feedback
        for layer_index, present in enumerate(present_key_values):
            outputs[f"present_key_values_{layer_index}"] = present
        return outputs
