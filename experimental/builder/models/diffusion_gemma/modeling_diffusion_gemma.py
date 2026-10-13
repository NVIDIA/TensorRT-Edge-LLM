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
"""DiffusionGemma phase-aware checkpoint-direct text backbone."""

from typing import Dict

import numpy as np
import tensorrt as trt

from ...ops import Embedding, GatedExperts, Linear, Module, NetworkModule
from ...ops import functional as F
from ...ops import pack_qkv
from ...ops.ragged import RaggedDecoderInputs, add_ragged_decoder_inputs
from . import weights as weight_conversion


class DiffusionGemmaRMSNorm(Module):
    """Provider RMSNorm with learned or implicit unit scale."""

    def __init__(self,
                 ctx,
                 prefix: str,
                 eps: float,
                 *,
                 with_scale: bool = True) -> None:
        super().__init__(ctx, prefix)
        self.eps = eps
        self.with_scale = with_scale

    def forward(self, hidden_states, rank: int = 2):
        weight = (self.weights.f32(self.key("weight")) if self.with_scale else
                  np.ones(self.cfg.hidden_size, dtype=np.float32))
        return F.rms_norm(hidden_states,
                          weight,
                          self.eps,
                          rank=rank,
                          weight_before_cast=True)


class DiffusionGemmaMLP(Module):
    """Provider GEGLU dense feed-forward block."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.gate_proj = Linear(ctx, self.key("gate_proj"))
        self.up_proj = Linear(ctx, self.key("up_proj"))
        self.down_proj = Linear(ctx, self.key("down_proj"))

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states).activation(self.cfg.hidden_act)
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class DiffusionGemmaAttention(Module):
    """Phase-selected causal/non-causal attention over a shared KV pool."""

    def __init__(self, ctx, prefix: str, layer_index: int) -> None:
        super().__init__(ctx, prefix)
        self.layer_index = layer_index
        self.head_dim = ctx.cfg.layer_head_dim(layer_index)
        self.num_kv_heads = ctx.cfg.layer_num_kv_heads(layer_index)
        self.attention_type = ctx.cfg.attention_type(layer_index)
        self.k_eq_v = self.attention_type == "full_attention"
        self.q_proj = Linear(ctx, self.key("q_proj"))
        self.k_proj = Linear(ctx, self.key("k_proj"))
        self.v_proj = (None
                       if self.k_eq_v else Linear(ctx, self.key("v_proj")))
        self.o_proj = Linear(ctx, self.key("o_proj"))
        self.q_norm = DiffusionGemmaRMSNorm(ctx, self.key("q_norm"),
                                            ctx.cfg.rms_norm_eps)
        self.k_norm = DiffusionGemmaRMSNorm(ctx, self.key("k_norm"),
                                            ctx.cfg.rms_norm_eps)

    def forward(self, hidden_states, past_key_value, rope, ragged,
                context_mask_selector):
        cfg = self.cfg
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = key if self.k_eq_v else self.v_proj(hidden_states)

        query = query.reshape((0, cfg.num_attention_heads, self.head_dim))
        query = self.q_norm(query, 3).reshape(
            (0, cfg.num_attention_heads * self.head_dim))
        key = key.reshape((0, self.num_kv_heads, self.head_dim))
        key = self.k_norm(key, 3).reshape(
            (0, self.num_kv_heads * self.head_dim))
        value = value.reshape((0, self.num_kv_heads, self.head_dim))
        value = F.rms_norm(value,
                           np.ones(self.head_dim, dtype=np.float32),
                           cfg.rms_norm_eps,
                           rank=3,
                           weight_before_cast=True)
        value = value.reshape((0, self.num_kv_heads * self.head_dim))

        qkv = pack_qkv(query, key, value, self.v_proj or self.k_proj)
        qkv_scales = list(self.weights.qkv_scales(self.prefix))
        if self.k_eq_v:
            qkv_scales[2] = qkv_scales[1]
        attention, present = F.attention(
            qkv,
            past_key_value,
            rope,
            ragged,
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            sliding_window_size=(cfg.sliding_window_size if self.attention_type
                                 == "sliding_attention" else -1),
            enable_fp8_kv_cache=cfg.kv_cache_quant == "fp8",
            qkv_scales=qkv_scales,
            attention_scale=cfg.attention_scaling,
            context_mask_selector=context_mask_selector,
        )
        attention = attention.reshape(
            (0, cfg.num_attention_heads * self.head_dim))
        return self.o_proj(attention), present


class DiffusionGemmaExperts(GatedExperts):
    """DiffusionGemma stacked or provider-packed GEGLU expert bank."""

    def _has_packed_nvfp4(self) -> bool:
        return self.weights.is_nvfp4(self.key("0.up_proj"))

    def plugin_intermediate_size(self) -> int:
        alignment = 128 if self.ctx.options.sm12x else 64
        size = self.cfg.moe_intermediate_size
        return ((size + alignment - 1) // alignment) * alignment

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
                plugin_intermediate_size=plugin_intermediate_size,
            )
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
                np.ones(cfg.num_experts, dtype=np.float32),
                "down_input_scale":
                np.ones(cfg.num_experts, dtype=np.float32),
                "e_score_correction_bias":
                np.ascontiguousarray(correction, dtype=np.float32),
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


class DiffusionGemmaRouter(Module):
    """Weightless RMSNorm followed by the provider router projection."""

    def forward(self, hidden_states):
        cfg = self.cfg
        flattened = hidden_states.reshape((-1, cfg.hidden_size))
        normalized = F.rms_norm(flattened,
                                np.ones(cfg.hidden_size, dtype=np.float32),
                                cfg.rms_norm_eps,
                                rank=2,
                                weight_before_cast=True)
        scale = self.weights.f16(self.key("scale"))
        scale = np.ascontiguousarray(scale * np.float16(cfg.hidden_size**-0.5))
        normalized = normalized * F.constant(scale.reshape(1, -1),
                                             "router_scale")
        router_weight = self.weights.f16(self.key("proj.weight"))
        logits = F.matmul(normalized,
                          F.constant(router_weight, "router_weight"),
                          transpose_rhs=True)
        return logits.cast(trt.float32)


class DiffusionGemmaMoE(Module):
    """Provider top-k router and GEGLU expert operation."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.router = DiffusionGemmaRouter(ctx, self.key("router"))
        self.experts = DiffusionGemmaExperts(ctx, self.key("experts"))

    def forward(self, expert_input, router_input):
        cfg = self.cfg
        router_logits = self.router(router_input)
        parameters, bindings = self.experts.parameters(
            self.key("router.per_expert_scale"))
        return F.nvfp4_moe(
            router_logits,
            expert_input,
            parameters,
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
            weight_bindings=bindings,
        )


class DiffusionGemmaDecoderLayer(Module):
    """Shared encoder/decoder layer with phase-specific residual scaling."""

    def __init__(self, ctx, prefix: str, layer_index: int) -> None:
        super().__init__(ctx, prefix)
        eps = ctx.cfg.rms_norm_eps
        self.self_attn = DiffusionGemmaAttention(ctx, self.key("self_attn"),
                                                 layer_index)
        self.mlp = DiffusionGemmaMLP(ctx, self.key("mlp"))
        self.moe = DiffusionGemmaMoE(ctx, prefix)
        self.input_layernorm = DiffusionGemmaRMSNorm(
            ctx, self.key("input_layernorm"), eps)
        self.post_attention_layernorm = DiffusionGemmaRMSNorm(
            ctx, self.key("post_attention_layernorm"), eps)
        self.pre_feedforward_layernorm = DiffusionGemmaRMSNorm(
            ctx, self.key("pre_feedforward_layernorm"), eps)
        self.post_feedforward_layernorm = DiffusionGemmaRMSNorm(
            ctx, self.key("post_feedforward_layernorm"), eps)
        self.post_dense_norm = DiffusionGemmaRMSNorm(
            ctx, self.key("post_feedforward_layernorm_1"), eps)
        self.pre_moe_norm = DiffusionGemmaRMSNorm(
            ctx, self.key("pre_feedforward_layernorm_2"), eps)
        self.post_moe_norm = DiffusionGemmaRMSNorm(
            ctx, self.key("post_feedforward_layernorm_2"), eps)
        self.encoder_scalar = ctx.weights.f16(self.key("encoder_layer_scalar"))
        self.decoder_scalar = ctx.weights.f16(self.key("decoder_layer_scalar"))

    def forward(self, hidden_states, ragged, past, rope, context_mask_selector,
                token_phase):
        residual = hidden_states
        attention, present = self.self_attn(
            self.input_layernorm(hidden_states), past, rope, ragged,
            context_mask_selector)
        hidden_states = residual + self.post_attention_layernorm(attention)

        residual = hidden_states
        dense = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        dense = self.post_dense_norm(dense)
        expert_input = self.pre_moe_norm(residual)
        routed = self.moe(expert_input, residual)
        routed = self.post_moe_norm(routed)
        hidden_states = residual + self.post_feedforward_layernorm(dense +
                                                                   routed)

        encoder = hidden_states * F.constant(self.encoder_scalar.reshape(1, 1),
                                             "encoder_layer_scalar")
        decoder = hidden_states * F.constant(self.decoder_scalar.reshape(1, 1),
                                             "decoder_layer_scalar")
        phase = token_phase.reshape((-1, 1)).equal(np.int32(1))
        return F.select(phase, encoder, decoder), present


class DiffusionGemmaSelfConditioning(Module):
    """Provider self-conditioning MLP over previous soft embeddings."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx, "self_conditioning")
        eps = ctx.cfg.rms_norm_eps
        self.pre_norm = DiffusionGemmaRMSNorm(ctx, self.key("pre_norm"), eps)
        self.gate_proj = Linear(ctx, self.key("gate_proj"))
        self.up_proj = Linear(ctx, self.key("up_proj"))
        self.down_proj = Linear(ctx, self.key("down_proj"))
        self.post_norm = DiffusionGemmaRMSNorm(ctx,
                                               self.key("post_norm"),
                                               eps,
                                               with_scale=False)

    def forward(self, inputs_embeds, self_conditioning_signal):
        normalized = self.pre_norm(self_conditioning_signal)
        gate = self.gate_proj(normalized).activation(self.cfg.hidden_act)
        up = self.up_proj(normalized)
        signal = self.down_proj(gate * up)
        return self.post_norm(inputs_embeds + signal)


class DiffusionGemmaForBlockDiffusion(NetworkModule):
    """Unified encoder, denoise, and commit backbone used by the C++ runtime."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        cfg = ctx.cfg
        self.embed_tokens = Embedding(ctx,
                                      "model.embed_tokens",
                                      scale=cfg.embedding_scale,
                                      runtime_embedding=True)
        self.self_conditioning = DiffusionGemmaSelfConditioning(ctx)
        self.layers = [
            DiffusionGemmaDecoderLayer(ctx, f"model.layers.{index}", index)
            for index in range(cfg.num_hidden_layers)
        ]
        self.norm = DiffusionGemmaRMSNorm(ctx, "model.norm", cfg.rms_norm_eps)

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        io: Dict[str, object] = {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, cfg.hidden_size)),
            "canvas_ids":
            self.add_input("canvas_ids", trt.int32, (-1, )),
            "prev_self_conditioning_embeds":
            self.add_input("prev_self_conditioning_embeds", trt.float16,
                           (-1, cfg.hidden_size)),
            "self_conditioning_temperature":
            self.add_input("self_conditioning_temperature", trt.float32,
                           (1, )),
            "past": [
                self.add_input(
                    f"past_key_values_{index}", kv_dtype,
                    (2, -1, F.KV_PAGE_SIZE, cfg.layer_num_kv_heads(index),
                     cfg.layer_head_dim(index)))
                for index in range(cfg.num_hidden_layers)
            ],
            "context_mask_selector":
            self.add_input("context_mask_selector", trt.int32, (-1, )),
            "phase_is_encoder":
            self.add_input("phase_is_encoder", trt.int32, (1, )),
            "select_token_indices":
            self.add_input("select_token_indices", trt.int64, (-1, )),
        }
        io.update(
            add_ragged_decoder_inputs(self.add_input,
                                      include_logits_indices=False).as_dict())
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
        return io

    def forward(self, **io):
        token_embeds = self.embed_tokens(io["canvas_ids"])
        conditioned = self.self_conditioning(
            token_embeds, io["prev_self_conditioning_embeds"])
        ragged = RaggedDecoderInputs.from_dict(io)
        phase = io["phase_is_encoder"].reshape((1, 1)).equal(np.int32(1))
        hidden_states = F.select(phase, io["inputs_embeds"], conditioned)

        present = []
        for index, layer in enumerate(self.layers):
            if self.cfg.uses_dual_rope:
                rope = (io["rope_full"] if self.cfg.attention_type(index)
                        == "full_attention" else io["rope_sliding"])
            else:
                rope = io["rope"]
            hidden_states, layer_present = layer(
                hidden_states,
                ragged,
                io["past"][index],
                rope,
                io["context_mask_selector"],
                token_phase,
            )
            present.append(layer_present)

        hidden_states = self.norm(hidden_states)
        selected = F.gather_token_rows(hidden_states,
                                       io["select_token_indices"])
        embedding_weight = self.embed_tokens.weight.reshape(
            (self.cfg.vocab_size, self.cfg.hidden_size))
        logits = selected.matmul(embedding_weight,
                                 rhs_op=trt.MatrixOperation.TRANSPOSE).cast(
                                     trt.float32)
        logits = logits / np.float32(self.cfg.embedding_scale)
        if self.cfg.final_logit_softcapping is not None:
            cap = np.float32(self.cfg.final_logit_softcapping)
            logits = (logits / cap).tanh() * cap

        temperature = io["self_conditioning_temperature"].reshape(
            (1, 1)).maximum(np.float32(1.0e-6))
        probabilities = (logits / temperature).softmax(1).cast(trt.float16)
        next_conditioning = probabilities.matmul(embedding_weight)

        outputs = {
            "logits": logits,
            "next_self_conditioning_embeds": next_conditioning,
        }
        for index, tensor in enumerate(present):
            outputs[f"present_key_values_{index}"] = tensor
        return outputs
