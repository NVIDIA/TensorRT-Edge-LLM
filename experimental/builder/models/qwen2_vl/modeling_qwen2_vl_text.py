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
"""Qwen2-VL text model aligned with Transformers' provider hierarchy."""

import logging
from typing import Dict, List, Tuple

import tensorrt as trt

from ...ops import BuildContext, Linear, Module, NetworkModule
from ...ops import functional as F
from ...ops import pack_qkv
from ...ops.ragged import RaggedDecoderInputs, add_ragged_decoder_inputs

LOGGER = logging.getLogger("builder.qwen2_vl.text")


class Qwen2VLRMSNorm(Module):
    """Qwen2-VL RMSNorm."""

    def __init__(self, ctx: BuildContext, prefix: str, eps: float) -> None:
        super().__init__(ctx, prefix)
        self.eps = eps

    def forward(self, hidden_states, rank: int = 2):
        weight = self.weights.f16(self.key("weight"))
        return F.rms_norm(hidden_states, weight, self.eps, rank)


class Qwen2VLMLP(Module):
    """Provider gated MLP: ``down(act(gate) * up)``."""

    def __init__(self, ctx: BuildContext, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.gate_proj = Linear(ctx, self.key("gate_proj"))
        self.up_proj = Linear(ctx, self.key("up_proj"))
        self.down_proj = Linear(ctx, self.key("down_proj"))

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states).activation(self.cfg.hidden_act)
        return self.down_proj(gate * self.up_proj(hidden_states))


class Qwen2VLAttention(Module):
    """Qwen2-VL multimodal RoPE attention."""

    def __init__(self, ctx: BuildContext, prefix: str,
                 layer_index: int) -> None:
        super().__init__(ctx, prefix)
        self.layer_index = layer_index
        self.q_proj = Linear(ctx, self.key("q_proj"))
        self.k_proj = Linear(ctx, self.key("k_proj"))
        self.v_proj = Linear(ctx, self.key("v_proj"))
        self.o_proj = Linear(ctx, self.key("o_proj"))

    def forward(self,
                hidden_states,
                past_key_value,
                rope_rotary_cos_sin,
                ragged,
                attention_mask=None,
                attention_pos_id=None) -> Tuple[object, object]:
        sliding_window = (-1 if self.cfg.attention_type(
            self.layer_index) != "sliding_attention" else
                          self.cfg.sliding_window_size)
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        qkv = pack_qkv(query, key, value, self.v_proj)
        attention, present = F.attention(
            qkv,
            past_key_value,
            rope_rotary_cos_sin,
            ragged,
            num_q_heads=self.cfg.num_attention_heads,
            num_kv_heads=self.cfg.num_key_value_heads,
            head_size=self.cfg.head_dim,
            sliding_window_size=sliding_window,
            enable_fp8_kv_cache=self.cfg.kv_cache_quant == "fp8",
            qkv_scales=self.weights.qkv_scales(self.prefix),
            skip_softmax_scale_factor=self.cfg.skip_softmax_scale_factor,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
        )
        attention = attention.reshape(
            (0, self.cfg.num_attention_heads * self.cfg.head_dim))
        return self.o_proj(attention), present


class Qwen2VLDecoderLayer(Module):
    """One provider Qwen2-VL decoder layer."""

    def __init__(self, ctx: BuildContext, prefix: str,
                 layer_index: int) -> None:
        super().__init__(ctx, prefix)
        self.self_attn = Qwen2VLAttention(ctx, self.key("self_attn"),
                                          layer_index)
        self.mlp = Qwen2VLMLP(ctx, self.key("mlp"))
        self.input_layernorm = Qwen2VLRMSNorm(ctx, self.key("input_layernorm"),
                                              ctx.cfg.rms_norm_eps)
        self.post_attention_layernorm = Qwen2VLRMSNorm(
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
        mlp = self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states + mlp, present


class Qwen2VLTextModel(Module):
    """Provider Qwen2-VL decoder stack."""

    def __init__(self, ctx: BuildContext, prefix: str = "model") -> None:
        super().__init__(ctx, prefix)
        self.layers = [
            Qwen2VLDecoderLayer(ctx, self.key(f"layers.{index}"), index)
            for index in range(ctx.cfg.num_hidden_layers)
        ]
        self.norm = Qwen2VLRMSNorm(ctx, self.key("norm"), ctx.cfg.rms_norm_eps)

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


class Qwen2VLForConditionalGeneration(NetworkModule):
    """TensorRT network for Qwen2-VL text generation."""

    def __init__(self, ctx: BuildContext) -> None:
        super().__init__(ctx)
        self.model = Qwen2VLTextModel(ctx)
        lm_head = ("lm_head" if ctx.weights.has("lm_head.weight")
                   or ctx.weights.has("lm_head.qweight") else
                   "model.embed_tokens")
        self.lm_head = Linear(ctx, lm_head)

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        result = {
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
        outputs["logits"] = F.cast(self.lm_head(selected), trt.float32)
        if self.cfg.engine_role == "base":
            outputs["hidden_states"] = F.hidden_state_feedback(
                hidden_states, all_hidden_states, self.cfg)
        for index, present in enumerate(present_key_values):
            outputs[f"present_key_values_{index}"] = present
        return outputs
