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
"""Muse-Glimmer dense text decoder (checkpoint-direct graph).

Weightless QK-norm, a sigmoid ``gate_proj`` gate on the attention output,
sandwich normalization with a separate ``post_norm_eps`` for the two post
norms, a scaleless normed embedding, and ``output_multiplier`` + logit softcap.
The softmax scale and the NoPE dual-RoPE selection (full-attention layers use an
identity RoPE table) are resolved in ``configuration.py``.
"""

from typing import Dict

import numpy as np
import tensorrt as trt

from ...ops import (Linear, Module, NetworkModule, RaggedDecoderInputs,
                    RMSNorm, add_ragged_decoder_inputs)
from ...ops import functional as F
from ...ops import pack_qkv


def _scaleless_rms_norm(hidden_states, width: int, eps: float, rank: int):
    """Weightless RMSNorm over the last axis (QK-norm and embedding norm)."""
    return F.rms_norm(hidden_states, np.ones(width, dtype=np.float16), eps,
                      rank)


def _text_field(cfg, key, default):
    """Read a text-tower scalar not surfaced on ``DeviceConfig``.

    ``raw_component`` is the root config for the full VLM (fields nested under
    ``text_config``) and the text config itself for a text-only checkpoint.
    """
    raw = cfg.raw_component
    if key in raw:
        return raw[key]
    text_config = raw.get("text_config")
    if isinstance(text_config, dict) and key in text_config:
        return text_config[key]
    return default


class MuseGlimmerMLP(Module):
    """SwiGLU feed-forward block."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.gate = Linear(ctx, self.key("gate_proj"))
        self.up = Linear(ctx, self.key("up_proj"))
        self.down = Linear(ctx, self.key("down_proj"))

    def forward(self, hidden_states):
        gate = self.gate(hidden_states).activation(self.cfg.hidden_act)
        return self.down(gate * self.up(hidden_states))


class MuseGlimmerAttention(Module):
    """GQA attention with weightless QK-norm and a sigmoid output gate."""

    def __init__(self, ctx, prefix: str, layer_index: int) -> None:
        super().__init__(ctx, prefix)
        self.head_dim = ctx.cfg.head_dim
        self.num_kv_heads = ctx.cfg.num_key_value_heads
        self.attention_type = ctx.cfg.attention_type(layer_index)
        self.sliding_window = (ctx.cfg.sliding_window_size
                               if self.attention_type == "sliding_attention"
                               else -1)
        self.q_proj = Linear(ctx, self.key("q_proj"))
        self.k_proj = Linear(ctx, self.key("k_proj"))
        self.v_proj = Linear(ctx, self.key("v_proj"))
        self.o_proj = Linear(ctx, self.key("o_proj"))
        self.gate_proj = Linear(ctx, self.key("gate_proj"))

    def forward(self,
                hidden_states,
                past_key_value,
                rope,
                ragged,
                attention_mask=None,
                attention_pos_id=None):
        cfg = self.cfg
        eps = cfg.rms_norm_eps
        num_heads = cfg.num_attention_heads
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query = _scaleless_rms_norm(
            query.reshape((0, num_heads, self.head_dim)), self.head_dim, eps,
            3).reshape((0, num_heads * self.head_dim))
        key = _scaleless_rms_norm(
            key.reshape((0, self.num_kv_heads, self.head_dim)), self.head_dim,
            eps, 3).reshape((0, self.num_kv_heads * self.head_dim))
        qkv = pack_qkv(query, key, value, self.v_proj)
        attention, present = F.attention(
            qkv,
            past_key_value,
            rope,
            ragged,
            num_q_heads=num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            sliding_window_size=self.sliding_window,
            enable_fp8_kv_cache=cfg.kv_cache_quant == "fp8",
            attention_scale=cfg.attention_scaling,
            qkv_scales=self.weights.qkv_scales(self.prefix),
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
        )
        attention = attention.reshape((0, num_heads * self.head_dim))
        attention = attention * self.gate_proj(hidden_states).sigmoid()
        return self.o_proj(attention), present


class MuseGlimmerDecoderLayer(Module):
    """Sandwich-normalized dense decoder layer."""

    def __init__(self, ctx, prefix: str, layer_index: int) -> None:
        super().__init__(ctx, prefix)
        eps = ctx.cfg.rms_norm_eps
        post_eps = float(_text_field(ctx.cfg, "post_norm_eps", eps))
        self.self_attn = MuseGlimmerAttention(ctx, self.key("self_attn"),
                                              layer_index)
        self.mlp = MuseGlimmerMLP(ctx, self.key("mlp"))
        self.input_layernorm = RMSNorm(ctx,
                                       self.key("input_layernorm"),
                                       eps,
                                       unit_offset=True)
        self.post_attention_layernorm = RMSNorm(
            ctx,
            self.key("post_attention_layernorm"),
            post_eps,
            unit_offset=True)
        self.pre_feedforward_layernorm = RMSNorm(
            ctx, self.key("pre_feedforward_layernorm"), eps, unit_offset=True)
        self.post_feedforward_layernorm = RMSNorm(
            ctx,
            self.key("post_feedforward_layernorm"),
            post_eps,
            unit_offset=True)

    def forward(self, hidden_states, past, rope, ragged, attention_mask,
                attention_pos_id):
        residual = hidden_states
        attention, present = self.self_attn(
            self.input_layernorm(hidden_states), past, rope, ragged,
            attention_mask, attention_pos_id)
        hidden_states = residual + self.post_attention_layernorm(attention)
        residual = hidden_states
        dense = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        hidden_states = residual + self.post_feedforward_layernorm(dense)
        return hidden_states, present


class MuseGlimmerForCausalLM(NetworkModule):
    """Muse-Glimmer base LLM with dual RoPE and logit softcap."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.layers = [
            MuseGlimmerDecoderLayer(ctx, f"model.layers.{index}", index)
            for index in range(ctx.cfg.num_hidden_layers)
        ]
        self.norm = RMSNorm(ctx,
                            "model.norm",
                            ctx.cfg.rms_norm_eps,
                            unit_offset=True)
        self.lm_head = Linear(ctx, "lm_head")
        self.output_multiplier = float(
            _text_field(ctx.cfg, "output_multiplier", 1.0))

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
            if cfg.dflash_tree_base:
                io["tree_parent_ids"] = self.add_input("tree_parent_ids",
                                                       trt.int32, (-1, ))
                io["tree_depths"] = self.add_input("tree_depths", trt.int32,
                                                   (-1, ))
                io["valid_tree_counts"] = self.add_input(
                    "valid_tree_counts", trt.int32, (-1, ))
            else:
                io["tree_parent_ids"] = None
                io["tree_depths"] = None
                io["valid_tree_counts"] = None
        else:
            io["attention_pos_id"] = None
            io["attention_mask"] = None
            io["tree_parent_ids"] = None
            io["tree_depths"] = None
            io["valid_tree_counts"] = None
        return io

    def forward(self, **io):
        cfg = self.cfg
        outputs = {}
        ragged = RaggedDecoderInputs.from_dict(io)
        hidden_states = _scaleless_rms_norm(io["inputs_embeds"],
                                            cfg.hidden_size, cfg.rms_norm_eps,
                                            2)
        present = []
        post_layer_hidden = []
        for index, layer in enumerate(self.layers):
            if cfg.uses_dual_rope:
                rope = (io["rope_full"] if cfg.attention_type(index)
                        == "full_attention" else io["rope_sliding"])
            else:
                rope = io["rope"]
            hidden_states, layer_present = layer(hidden_states,
                                                 io["past"][index], rope,
                                                 ragged, io["attention_mask"],
                                                 io["attention_pos_id"])
            present.append(layer_present)
            post_layer_hidden.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        selected = F.gather_token_rows(hidden_states, ragged.logits_indices)
        logits = F.cast(self.lm_head(selected), trt.float32)
        if cfg.final_logit_softcapping is not None:
            # Fold output_multiplier into the softcap pre-tanh scale to emit a
            # single pre-tanh multiply, matching Gemma3/Gemma4.
            cap = float(cfg.final_logit_softcapping)
            pre_scale = np.float32(self.output_multiplier / cap)
            logits = (logits * pre_scale).tanh() * np.float32(cap)
        elif self.output_multiplier != 1.0:
            logits = logits * np.float32(self.output_multiplier)
        outputs["logits"] = logits
        if cfg.engine_role == "base" and cfg.spec_decode_type == "dflash":
            outputs["hidden_states"] = F.hidden_state_feedback(
                hidden_states, post_layer_hidden, cfg, allow_eagle3=False)
        for index, tensor in enumerate(present):
            outputs[f"present_key_values_{index}"] = tensor
        return outputs
