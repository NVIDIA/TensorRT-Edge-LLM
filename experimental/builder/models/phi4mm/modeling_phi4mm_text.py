# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Phi-4 Multimodal text model aligned with Transformers."""

import logging
from typing import Dict, List, Tuple

import tensorrt as trt

from ...ops import BuildContext, Linear, Module, NetworkModule
from ...ops import functional as F
from ...ops.ragged import RaggedDecoderInputs, add_ragged_decoder_inputs

LOGGER = logging.getLogger("builder.phi4_multimodal.text")


class Phi4MultimodalRMSNorm(Module):
    """Provider RMSNorm."""

    def __init__(self, ctx: BuildContext, prefix: str, eps: float) -> None:
        super().__init__(ctx, prefix)
        self.eps = eps

    def forward(self, hidden_states, rank: int = 2):
        return F.rms_norm(hidden_states, self.weights.f16(self.key("weight")),
                          self.eps, rank)


class Phi4MultimodalMLP(Module):
    """Provider fused gate/up MLP."""

    def __init__(self, ctx: BuildContext, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.gate_up_proj = Linear(ctx, self.key("gate_up_proj"))
        self.down_proj = Linear(ctx, self.key("down_proj"))

    def forward(self, hidden_states):
        gate_up = self.gate_up_proj(hidden_states)
        width = self.cfg.intermediate_size
        gate = gate_up.slice_last_dim(0, width, gate_up.ndim)
        up = gate_up.slice_last_dim(width, width, gate_up.ndim)
        gate = gate.activation(self.cfg.hidden_act)
        return self.down_proj(gate * up)


class Phi4MultimodalAttention(Module):
    """Provider fused-QKV attention with partial RoPE."""

    def __init__(self, ctx: BuildContext, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.qkv_proj = Linear(ctx, self.key("qkv_proj"))
        self.o_proj = Linear(ctx, self.key("o_proj"))

    def forward(self,
                hidden_states,
                past_key_value,
                rope_rotary_cos_sin,
                ragged,
                attention_mask=None,
                attention_pos_id=None) -> Tuple[object, object]:
        cfg = self.cfg
        qkv = self.qkv_proj(hidden_states)
        attention, present = F.attention(
            qkv,
            past_key_value,
            rope_rotary_cos_sin,
            ragged,
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_size=cfg.head_dim,
            sliding_window_size=cfg.sliding_window_size,
            enable_fp8_kv_cache=cfg.kv_cache_quant == "fp8",
            qkv_scales=self.weights.qkv_scales(self.prefix),
            skip_softmax_scale_factor=cfg.skip_softmax_scale_factor,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
        )
        attention = attention.reshape(
            (0, cfg.num_attention_heads * cfg.head_dim))
        return self.o_proj(attention), present


class Phi4MultimodalDecoderLayer(Module):
    """One provider Phi-4 decoder layer."""

    def __init__(self, ctx: BuildContext, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.self_attn = Phi4MultimodalAttention(ctx, self.key("self_attn"))
        self.mlp = Phi4MultimodalMLP(ctx, self.key("mlp"))
        self.input_layernorm = Phi4MultimodalRMSNorm(
            ctx, self.key("input_layernorm"), ctx.cfg.rms_norm_eps)
        self.post_attention_layernorm = Phi4MultimodalRMSNorm(
            ctx, self.key("post_attention_layernorm"), ctx.cfg.rms_norm_eps)

    def forward(self,
                hidden_states,
                past_key_value,
                rope_rotary_cos_sin,
                ragged,
                attention_mask=None,
                attention_pos_id=None) -> Tuple[object, object]:
        attention, present = self.self_attn(
            self.input_layernorm(hidden_states), past_key_value,
            rope_rotary_cos_sin, ragged, attention_mask, attention_pos_id)
        hidden_states = hidden_states + attention
        feed_forward = self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states + feed_forward, present


class Phi4MultimodalModel(Module):
    """Provider Phi-4 decoder stack."""

    def __init__(self, ctx: BuildContext, prefix: str = "model") -> None:
        super().__init__(ctx, prefix)
        self.layers = [
            Phi4MultimodalDecoderLayer(ctx, self.key(f"layers.{index}"))
            for index in range(ctx.cfg.num_hidden_layers)
        ]
        self.norm = Phi4MultimodalRMSNorm(ctx, self.key("norm"),
                                          ctx.cfg.rms_norm_eps)

    def forward(
            self,
            inputs_embeds,
            past_key_values,
            rope_rotary_cos_sin,
            ragged,
            attention_mask=None,
            attention_pos_id=None
    ) -> Tuple[object, List[object], List[object]]:
        hidden_states = inputs_embeds
        present_key_values = []
        all_hidden_states = []
        for index, layer in enumerate(self.layers):
            LOGGER.info("building layer %d/%d", index + 1, len(self.layers))
            hidden_states, present = layer(hidden_states,
                                           past_key_values[index],
                                           rope_rotary_cos_sin, ragged,
                                           attention_mask, attention_pos_id)
            present_key_values.append(present)
            all_hidden_states.append(hidden_states)
        return self.norm(hidden_states), present_key_values, all_hidden_states


class Phi4MultimodalForCausalLM(NetworkModule):
    """TensorRT network for Phi-4 Multimodal text generation."""

    def __init__(self, ctx: BuildContext) -> None:
        super().__init__(ctx)
        self.model = Phi4MultimodalModel(ctx)
        lm_head = ("lm_head" if ctx.weights.has("lm_head.weight")
                   or ctx.weights.has("lm_head.qweight") else
                   "model.embed_tokens")
        self.lm_head = Linear(ctx, lm_head)

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
            io["rope_rotary_cos_sin"], ragged, io["attention_mask"],
            io["attention_pos_id"])
        selected = F.gather_token_rows(hidden_states, ragged.logits_indices)
        outputs["logits"] = F.cast(self.lm_head(selected), trt.float32)
        if self.cfg.engine_role == "base":
            outputs["hidden_states"] = F.hidden_state_feedback(
                hidden_states, all_hidden_states, self.cfg)
        for index, present in enumerate(present_key_values):
            outputs[f"present_key_values_{index}"] = present
        return outputs
