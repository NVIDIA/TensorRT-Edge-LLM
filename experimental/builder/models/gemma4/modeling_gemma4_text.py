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
"""Gemma4 text checkpoint-direct graph."""

import logging
from typing import Dict, List

import numpy as np
import tensorrt as trt

from ...ops import Module, NetworkModule
from ...ops import functional as F
from ...ops import pack_qkv
from ...ops.ragged import RaggedDecoderInputs, add_ragged_decoder_inputs
from . import modeling_gemma4_layers as layers
from . import weights as weight_conversion

LOGGER = logging.getLogger("builder.gemma4")


class Gemma4RMSNorm(layers.Gemma4RMSNormBase):
    """Gemma4 learned RMS normalization."""


class Gemma4TextMLP(Module):
    """Gated Gemma feed-forward block."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.gate = layers.Linear(ctx, self.key("gate_proj"))
        self.up = layers.Linear(ctx, self.key("up_proj"))
        self.down = layers.Linear(ctx, self.key("down_proj"))

    def forward(self, hidden_states):
        gate = self.gate(hidden_states).cast(trt.float32).activation(
            self.cfg.hidden_act)
        up = self.up(hidden_states).cast(trt.float32)
        product = (gate * up).maximum(np.float32(-2048.0)).minimum(
            np.float32(2048.0)).cast(trt.float16)
        return self.down(product)


class Gemma4TextAttention(Module):
    """Gemma attention with per-layer geometry and optional tree metadata."""

    def __init__(self, ctx, prefix: str, layer_index: int) -> None:
        super().__init__(ctx, prefix)
        self.layer_index = layer_index
        self.head_dim = ctx.cfg.layer_head_dim(layer_index)
        self.num_kv_heads = ctx.cfg.layer_num_kv_heads(layer_index)
        self.attention_type = ctx.cfg.attention_type(layer_index)
        self.use_alternative_attention = (ctx.cfg.attention_k_eq_v
                                          and self.attention_type
                                          == "full_attention")
        first_shared = ctx.cfg.num_hidden_layers - ctx.cfg.num_kv_shared_layers
        self.is_kv_shared = (first_shared > 0 and layer_index >= first_shared)
        self.q_proj = layers.Linear(ctx, self.key("q_proj"))
        self.o_proj = layers.Linear(ctx, self.key("o_proj"))
        self.q_norm = Gemma4RMSNorm(ctx, self.key("q_norm"),
                                    ctx.cfg.rms_norm_eps)
        if self.is_kv_shared:
            self.k_proj = None
            self.v_proj = None
            self.k_norm = None
        else:
            self.k_proj = layers.Linear(ctx, self.key("k_proj"))
            self.v_proj = (None if self.use_alternative_attention else
                           layers.Linear(ctx, self.key("v_proj")))
            self.k_norm = Gemma4RMSNorm(ctx, self.key("k_norm"),
                                        ctx.cfg.rms_norm_eps)

    def forward(self,
                hidden_states,
                past_key_value,
                rope,
                ragged,
                attention_mask=None,
                attention_pos_id=None):
        cfg = self.cfg
        query = self.q_proj(hidden_states)
        if not self.is_kv_shared:
            key = self.k_proj(hidden_states)
            value = (key if self.use_alternative_attention else
                     self.v_proj(hidden_states))
        query = query.reshape((0, cfg.num_attention_heads, self.head_dim))
        query = self.q_norm(query, 3)
        query = query.reshape((0, cfg.num_attention_heads * self.head_dim))
        if not self.is_kv_shared:
            key = key.reshape((0, self.num_kv_heads, self.head_dim))
            key = self.k_norm(key, 3)
            key = key.reshape((0, self.num_kv_heads * self.head_dim))
            if cfg.has_value_norm:
                value = value.reshape((0, self.num_kv_heads, self.head_dim))
                value = F.rms_norm(value,
                                   np.ones(self.head_dim, dtype=np.float16),
                                   cfg.rms_norm_eps, 3)
                value = value.reshape((0, self.num_kv_heads * self.head_dim))
        sliding_window = (cfg.sliding_window_size if self.attention_type
                          == "sliding_attention" else -1)
        qkv = (query if self.is_kv_shared else pack_qkv(
            query, key, value, self.v_proj or self.k_proj))
        attention, present = F.attention(
            qkv,
            past_key_value,
            rope,
            ragged,
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            sliding_window_size=sliding_window,
            enable_fp8_kv_cache=cfg.kv_cache_quant == "fp8",
            attention_scale=cfg.attention_scaling,
            # Sliding layers have no skippable long-range tiles.
            skip_softmax_scale_factor=(cfg.skip_softmax_scale_factor
                                       if self.attention_type
                                       == "full_attention" else 0.0),
            enable_kv_shared=self.is_kv_shared,
            qkv_scales=self.weights.qkv_scales(self.prefix),
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
        )
        attention = attention.reshape(
            (0, cfg.num_attention_heads * self.head_dim))
        return self.o_proj(attention), present


class Gemma4TextExperts(Module):
    """Gemma expert weights in HF-stacked or quantizer-split layout."""

    def _has_packed_nvfp4(self) -> bool:
        return self.weights.is_nvfp4(self.key("0.up_proj"))

    def plugin_intermediate_size(self) -> int:
        alignment = 128 if self.ctx.options.sm12x else 64
        size = self.cfg.moe_intermediate_size
        return ((size + alignment - 1) // alignment) * alignment

    def load_expert_dense(self, expert_index: int) -> dict:
        gate_up_key = self.key("gate_up_proj")
        down_key = self.key("down_proj")
        if self.weights.has(gate_up_key) and self.weights.has(down_key):
            gate_up = self.weights.f32(gate_up_key)[expert_index]
            gate, up = np.split(gate_up, 2, axis=0)
            down = self.weights.f32(down_key)[expert_index]
            return {
                "gate": np.ascontiguousarray(gate),
                "up": np.ascontiguousarray(up),
                "down": np.ascontiguousarray(down),
            }

        expert_prefix = self.key(str(expert_index))
        return {
            "gate":
            self.weights.expert_dense_f32(f"{expert_prefix}.gate_proj"),
            "up": self.weights.expert_dense_f32(f"{expert_prefix}.up_proj"),
            "down":
            self.weights.expert_dense_f32(f"{expert_prefix}.down_proj"),
        }

    def load_expert_raw_nvfp4(self, expert_index: int) -> dict:
        """Load one provider-packed Gemma expert without decoding it."""
        expert_prefix = self.key(str(expert_index))
        return {
            projection:
            self.weights.expert_raw_nvfp4(f"{expert_prefix}.{projection}_proj")
            for projection in ("gate", "up", "down")
        }

    def parameters(self, correction_key: str):
        cfg = self.cfg
        layout = "concat" if self.ctx.options.sm12x else "interleave"
        plugin_intermediate_size = self.plugin_intermediate_size()

        def materialize():
            pack = (weight_conversion.repack_nvfp4_experts
                    if self._has_packed_nvfp4() else
                    weight_conversion.pack_dense_nvfp4_experts)
            load = (self.load_expert_raw_nvfp4
                    if self._has_packed_nvfp4() else self.load_expert_dense)
            (fc1, fc1_scale, fc1_alpha, fc2, fc2_scale, fc2_alpha) = pack(
                load,
                cfg.num_experts,
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                cfg.group_size,
                fc1_layout=layout,
                plugin_intermediate_size=plugin_intermediate_size)
            correction = (self.weights.f32(correction_key)
                          if self.weights.has(correction_key) else np.ones(
                              cfg.num_experts, dtype=np.float32))
            return {
                "fc1_qweights":
                fc1,
                "fc1_blocks_scale":
                fc1_scale,
                "fc1_alpha":
                fc1_alpha,
                "fc2_qweights":
                fc2,
                "fc2_blocks_scale":
                fc2_scale,
                "fc2_alpha":
                fc2_alpha,
                "input_global_scale":
                np.ones(cfg.num_experts, np.float32),
                "down_input_scale":
                np.ones(cfg.num_experts, np.float32),
                "e_score_correction_bias":
                np.ascontiguousarray(correction, np.float32),
            }

        if not self._has_packed_nvfp4():
            return materialize(), None
        parameters = self.weights.parameter_value(
            "nvfp4_moe",
            self.prefix,
            lambda: weight_conversion.nvfp4_expert_specs(
                self.weights, self.prefix, cfg.num_experts,
                plugin_intermediate_size),
            materialize,
        )
        bindings = weight_conversion.nvfp4_expert_bindings(
            self.weights, self.prefix, correction_key, cfg.num_experts,
            self.ctx.options.sm12x)
        return parameters, bindings


class Gemma4TextRouter(Module):
    """Provider weightless RMSNorm and expert-score projection."""

    def forward(self, hidden_states):
        cfg = self.cfg
        flattened = hidden_states.reshape((-1, cfg.hidden_size))
        normalized = F.rms_norm(flattened,
                                np.ones(cfg.hidden_size, dtype=np.float16),
                                cfg.rms_norm_eps, 2)
        if self.weights.has(self.key("scale")):
            scale = self.weights.f16(self.key("scale"))
            scale = scale * np.float16(cfg.hidden_size**-0.5)
            normalized = normalized * F.constant(scale.reshape(1, -1),
                                                 "router_scale")
        router_weight = self.weights.f16(self.key("proj.weight"))
        router_logits = F.matmul(normalized,
                                 F.constant(router_weight, "router_weight"),
                                 transpose_rhs=True)
        return F.cast(router_logits, trt.float32)


class Gemma4TextMoE(Module):
    """Gemma4 router and expert collection."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.router = Gemma4TextRouter(ctx, self.key("router"))
        self.experts = Gemma4TextExperts(ctx, self.key("experts"))

    def forward(self, expert_input, router_input):
        cfg = self.cfg
        router_logits = self.router(router_input)
        packed_weights, bindings = self.experts.parameters(
            self.key("router.per_expert_scale"))
        return F.nvfp4_moe(router_logits,
                           expert_input,
                           packed_weights,
                           cfg.num_experts,
                           cfg.num_experts_per_tok,
                           cfg.hidden_size,
                           self.experts.plugin_intermediate_size(),
                           F.MoeActivation.GEGLU,
                           1,
                           1,
                           1,
                           1.0,
                           F.MoeRouting.SOFTMAX_TOPK_POST_SCALE,
                           self.ctx.options.sm12x,
                           weight_prefix=self.experts.prefix,
                           weight_bindings=bindings)


class Gemma4TextDecoderLayer(Module):
    """Four-normalization Gemma decoder layer."""

    def __init__(self, ctx, prefix: str, layer_index: int) -> None:
        super().__init__(ctx, prefix)
        eps = ctx.cfg.rms_norm_eps
        self.self_attn = Gemma4TextAttention(ctx, self.key("self_attn"),
                                             layer_index)
        self.input_layernorm = Gemma4RMSNorm(ctx, self.key("input_layernorm"),
                                             eps)
        self.post_attention_layernorm = Gemma4RMSNorm(
            ctx, self.key("post_attention_layernorm"), eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            ctx, self.key("pre_feedforward_layernorm"), eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            ctx, self.key("post_feedforward_layernorm"), eps)
        self.mlp = Gemma4TextMLP(ctx, self.key("mlp"))
        self.moe = (Gemma4TextMoE(ctx, prefix)
                    if ctx.cfg.enable_moe_block else None)
        if self.moe is not None:
            self.pre_moe_norm = Gemma4RMSNorm(
                ctx, self.key("pre_feedforward_layernorm_2"), eps)
            self.post_dense_norm = Gemma4RMSNorm(
                ctx, self.key("post_feedforward_layernorm_1"), eps)
            self.post_moe_norm = Gemma4RMSNorm(
                ctx, self.key("post_feedforward_layernorm_2"), eps)
        self.layer_scalar = (ctx.weights.f16(
            self.key("layer_scalar")) if ctx.weights.has(
                self.key("layer_scalar")) else np.ones(1, dtype=np.float16))
        ple_size = ctx.cfg.hidden_size_per_layer_input
        self.ple_gate = (layers.Linear(ctx, self.key("per_layer_input_gate"))
                         if ple_size > 0 else None)
        self.ple_projection = (layers.Linear(
            ctx, self.key("per_layer_projection")) if ple_size > 0 else None)
        self.ple_norm = (Gemma4RMSNorm(ctx,
                                       self.key("post_per_layer_input_norm"),
                                       eps) if ple_size > 0 else None)

    def forward(self,
                hidden_states,
                past,
                rope,
                ragged,
                attention_mask,
                attention_pos_id,
                ple_input=None):
        residual = hidden_states
        attention, present = self.self_attn(
            self.input_layernorm(hidden_states), past, rope, ragged,
            attention_mask, attention_pos_id)
        attention = self.post_attention_layernorm(attention)
        hidden_states = residual + attention
        residual = hidden_states
        dense = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        if self.moe is not None:
            dense = self.post_dense_norm(dense)
            expert_input = self.pre_moe_norm(residual)
            routed = self.moe(expert_input, residual)
            routed = self.post_moe_norm(routed)
            dense = dense + routed
        dense = self.post_feedforward_layernorm(dense)
        hidden_states = residual + dense
        if ple_input is not None:
            gated = self.ple_gate(hidden_states).activation(
                self.cfg.hidden_act)
            gated = self.ple_norm(self.ple_projection(gated * ple_input))
            hidden_states = hidden_states + gated
        hidden_states = hidden_states * F.constant(
            self.layer_scalar.reshape(1, 1), "layer_scalar")
        return hidden_states, present


class Gemma4ForCausalLM(NetworkModule):
    """Gemma4 base LLM with dual RoPE, PLE, and speculative verification."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.layers = [
            Gemma4TextDecoderLayer(ctx, f"model.layers.{index}", index)
            for index in range(ctx.cfg.num_hidden_layers)
        ]
        self.norm = Gemma4RMSNorm(ctx, "model.norm", ctx.cfg.rms_norm_eps)
        lm_head = ("lm_head" if ctx.weights.has("lm_head.weight") else
                   "model.embed_tokens")
        self.lm_head = layers.Linear(ctx, lm_head)
        if ctx.cfg.hidden_size_per_layer_input > 0:
            self.ple_projection = layers.Linear(
                ctx, "model.per_layer_model_projection")
            self.ple_projection_norm = Gemma4RMSNorm(
                ctx, "model.per_layer_projection_norm", ctx.cfg.rms_norm_eps)
        else:
            self.ple_projection = None

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        io: Dict[str, object] = {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, cfg.hidden_size)),
            "past": [
                self.add_input(
                    f"past_key_values_{index}", kv_dtype,
                    (2, -1, F.KV_PAGE_SIZE, cfg.layer_num_kv_heads(index),
                     cfg.layer_head_dim(index)))
                for index in range(cfg.num_hidden_layers)
            ],
        }
        io.update(add_ragged_decoder_inputs(self.add_input).as_dict())
        if cfg.uses_dual_rope:
            sliding_dim = cfg.rope_rotary_dim(cfg.sliding_rope_config,
                                              cfg.head_dim)
            full_dim = cfg.rope_rotary_dim(cfg.full_rope_config,
                                           cfg.global_head_dim or cfg.head_dim)
            io["rope_sliding"] = self.add_input("rope_rotary_cos_sin_sliding",
                                                trt.float32, (-1, sliding_dim))
            io["rope_full"] = self.add_input("rope_rotary_cos_sin_full",
                                             trt.float32, (-1, full_dim))
        else:
            io["rope"] = self.add_input("rope_rotary_cos_sin", trt.float32,
                                        (-1, cfg.rotary_dim))
        if cfg.engine_role == "base":
            io["attention_pos_id"] = self.add_input("attention_position_ids",
                                                    trt.int32, (-1, ))
            io["attention_mask"] = self.add_input("packed_attention_mask",
                                                  trt.int32, (-1, -1))
        else:
            io["attention_pos_id"] = None
            io["attention_mask"] = None
        io["ple"] = [
            self.add_input(f"ple_token_embeds_{index}", trt.float16,
                           (-1, cfg.hidden_size_per_layer_input))
            for index in range(cfg.num_hidden_layers)
        ] if cfg.hidden_size_per_layer_input > 0 else []
        return io

    def _layer_ple(self, hidden_states, token_ple: List[object], index: int):
        if self.ple_projection is None:
            return None
        cfg = self.cfg
        projected = self.ple_projection(hidden_states)
        projected = projected * np.float16(cfg.hidden_size**-0.5)
        projected = projected.reshape(
            (0, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input))
        projected = projected.transpose((0, 2, 1)).slice_last_dim(index, 1, 3)
        projected = projected.reshape((0, cfg.hidden_size_per_layer_input))
        projected = self.ple_projection_norm(projected)
        combined = projected + token_ple[index]
        return combined * np.float16(2**-0.5)

    def forward(self, **io):
        ragged = RaggedDecoderInputs.from_dict(io)
        outputs = {}
        hidden_states = io["inputs_embeds"]
        present = []
        pre_layer_hidden = []
        post_layer_hidden = []
        for index, layer in enumerate(self.layers):
            pre_layer_hidden.append(hidden_states)
            if self.cfg.uses_dual_rope:
                rope = (io["rope_full"] if self.cfg.attention_type(index)
                        == "full_attention" else io["rope_sliding"])
            else:
                rope = io["rope"]
            ple = self._layer_ple(io["inputs_embeds"], io["ple"], index)
            hidden_states, layer_present = layer(hidden_states,
                                                 io["past"][index], rope,
                                                 ragged, io["attention_mask"],
                                                 io["attention_pos_id"], ple)
            present.append(layer_present)
            post_layer_hidden.append(hidden_states)
        pre_norm_hidden = hidden_states
        hidden_states = self.norm(hidden_states)
        selected = F.gather_token_rows(hidden_states, ragged.logits_indices)
        logits = F.cast(self.lm_head(selected), trt.float32)
        if self.cfg.final_logit_softcapping is not None:
            cap_value = float(self.cfg.final_logit_softcapping)
            logits = ((logits / np.float32(cap_value)).tanh() *
                      np.float32(cap_value))
        outputs["logits"] = logits
        if self.cfg.engine_role == "base":
            if (self.cfg.spec_decode_type == "eagle3"
                    and self.cfg.eagle3_target_layer_ids):
                indices = self.cfg.eagle3_target_layer_ids
                feedback = F.concatenate(
                    tuple(post_layer_hidden[index] for index in indices), 1)
            elif (self.cfg.spec_decode_type == "eagle3"
                  and len(pre_layer_hidden) >= 4):
                indices = (2, len(pre_layer_hidden) // 2,
                           len(pre_layer_hidden) - 4)
                feedback = F.concatenate(
                    tuple(pre_layer_hidden[index] for index in indices), 1)
            elif self.cfg.spec_decode_type in ("dflash", "dspark"):
                feedback = F.hidden_state_feedback(
                    hidden_states,
                    post_layer_hidden,
                    self.cfg,
                    allow_eagle3=False,
                )
            elif self.cfg.spec_decode_type == "gemma4_mtp":
                feedback = pre_norm_hidden
            else:
                feedback = hidden_states
            outputs["hidden_states"] = feedback
        elif self.cfg.root_model_type == "gemma4":
            _ = pre_norm_hidden
        for index, tensor in enumerate(present):
            outputs[f"present_key_values_{index}"] = tensor
        return outputs
