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
"""DFlash checkpoint-direct draft graph."""

from typing import Dict

import numpy as np
import tensorrt as trt

from ...core import config as core_config
from ...core import quantization
from ...ops import (GatedMLP, Linear, Module, NetworkModule, RMSNorm,
                    TreeAttention)
from ...ops import functional as F
from ...ops import pack_qkv
from ...ops.ragged import RaggedDecoderInputs, add_ragged_decoder_inputs
from .. import registry as model_registry
from ..gemma4.modeling_gemma4_text import Gemma4RMSNorm, Gemma4TextMLP


def _draft_rotary_dim(cfg) -> int:
    rope_config = {
        "rope_scaling": cfg.rope_scaling,
        "partial_rotary_factor": cfg.partial_rotary_factor,
    }
    return cfg.rope_rotary_dim(rope_config, cfg.head_dim)


class DFlashProposalAttention(TreeAttention):
    """Proposal attention after inserting target-hidden K/V deltas."""

    def __init__(self, ctx, prefix: str, layer_index: int) -> None:
        super().__init__(ctx, prefix)
        cfg = ctx.cfg
        self.attention_type = cfg.attention_type(layer_index)
        self.head_dim = cfg.layer_head_dim(layer_index)
        self.num_kv_heads = cfg.layer_num_kv_heads(layer_index)
        self.k_eq_v = (cfg.attention_k_eq_v
                       and self.attention_type == "full_attention")
        self.sliding_window_size = (cfg.sliding_window_size
                                    if self.attention_type
                                    == "sliding_attention" else -1)
        if self.k_eq_v:
            self.v_proj = None
        if str(cfg.model_type).startswith("gemma4"):
            self.q_norm = Gemma4RMSNorm(ctx, self.key("q_norm"),
                                        cfg.rms_norm_eps)
            self.k_norm = Gemma4RMSNorm(ctx, self.key("k_norm"),
                                        cfg.rms_norm_eps)

    def _normalize_value(self, value):
        value = value.reshape((0, self.num_kv_heads, self.head_dim))
        if self.cfg.has_value_norm:
            value = F.rms_norm(value, np.ones(self.head_dim, dtype=np.float16),
                               self.cfg.rms_norm_eps, 3)
        return value

    def project_delta_kv(self, hidden_delta):
        key_raw = self.k_proj(hidden_delta)
        value_raw = key_raw if self.k_eq_v else self.v_proj(hidden_delta)
        key = self.k_norm(
            key_raw.reshape((0, self.num_kv_heads, self.head_dim)), 3)
        return key, self._normalize_value(value_raw)

    def project_qkv(self, hidden):
        cfg = self.cfg
        query = self.q_norm(
            self.q_proj(hidden).reshape(
                (0, cfg.num_attention_heads, self.head_dim)), 3).reshape(
                    (0, cfg.num_attention_heads * self.head_dim))
        key_raw = self.k_proj(hidden)
        value_raw = key_raw if self.k_eq_v else self.v_proj(hidden)
        key = self.k_norm(
            key_raw.reshape((0, self.num_kv_heads, self.head_dim)), 3).reshape(
                (0, self.num_kv_heads * self.head_dim))
        value = self._normalize_value(value_raw).reshape(
            (0, self.num_kv_heads * self.head_dim))
        scales = list(self.weights.qkv_scales(self.prefix))
        if self.k_eq_v:
            scales[2] = scales[1]
        return pack_qkv(query, key, value, self.v_proj or self.k_proj), scales

    def forward(self, hidden, hidden_delta, past, rope, ragged, delta_rope,
                delta_positions, delta_token_to_sequence, attention_mask,
                attention_pos_id):
        cfg = self.cfg
        key_delta, value_delta = self.project_delta_kv(hidden_delta)
        updated = F.update_dflash_target_cache(key_delta, value_delta, past,
                                               delta_rope, delta_positions,
                                               delta_token_to_sequence,
                                               ragged.kv_page_table)

        qkv, qkv_scales = self.project_qkv(hidden)
        attention, present = F.attention(
            qkv,
            updated,
            rope,
            ragged,
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            sliding_window_size=self.sliding_window_size,
            enable_fp8_kv_cache=False,
            attention_scale=cfg.attention_scaling,
            qkv_scales=qkv_scales,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
        )
        attention = attention.reshape(
            (0, cfg.num_attention_heads * self.head_dim))
        return self.o_proj(attention), present


class DFlashDecoderLayer(Module):
    """DFlash decoder layer with target-cache update and proposal attention."""

    attention_class = DFlashProposalAttention

    def __init__(self, ctx, prefix: str, layer_index: int) -> None:
        super().__init__(ctx, prefix)
        cfg = ctx.cfg
        self.is_gemma4 = str(cfg.model_type).startswith("gemma4")
        norm_class = Gemma4RMSNorm if self.is_gemma4 else RMSNorm
        self.input_norm = norm_class(ctx, self.key("input_layernorm"),
                                     cfg.rms_norm_eps)
        self.attention = self.attention_class(ctx, self.key("self_attn"),
                                              layer_index)
        self.post_norm = norm_class(ctx, self.key("post_attention_layernorm"),
                                    cfg.rms_norm_eps)
        self.mlp = (Gemma4TextMLP(ctx, self.key("mlp"))
                    if self.is_gemma4 else GatedMLP(ctx, self.key("mlp")))
        if self.is_gemma4:
            self.pre_feedforward_norm = Gemma4RMSNorm(
                ctx, self.key("pre_feedforward_layernorm"), cfg.rms_norm_eps)
            self.post_feedforward_norm = Gemma4RMSNorm(
                ctx, self.key("post_feedforward_layernorm"), cfg.rms_norm_eps)
            self.layer_scalar = (ctx.weights.f16(self.key("layer_scalar"))
                                 if ctx.weights.has(self.key("layer_scalar"))
                                 else np.ones(1, dtype=np.float16))

    def forward(self, hidden, hidden_delta, past, rope, ragged, delta_rope,
                delta_positions, delta_token_to_sequence, attention_mask,
                attention_pos_id):
        attention, present = self.attention(self.input_norm(hidden),
                                            hidden_delta, past, rope, ragged,
                                            delta_rope, delta_positions,
                                            delta_token_to_sequence,
                                            attention_mask, attention_pos_id)
        if not self.is_gemma4:
            hidden = hidden + attention
            feed_forward = self.mlp(self.post_norm(hidden))
            return hidden + feed_forward, present

        hidden = hidden + self.post_norm(attention)
        residual = hidden
        feed_forward = self.mlp(self.pre_feedforward_norm(hidden))
        hidden = residual + self.post_feedforward_norm(feed_forward)
        hidden = hidden * F.constant(self.layer_scalar.reshape(1, 1),
                                     "layer_scalar")
        return hidden, present


class DFlashTargetProjection(Module):
    """Project concatenated target states before proposal decoding."""

    def forward(self, hidden):
        descriptor = self.weights.linear_descriptor(self.prefix,
                                                    quantization.QUANT_FP16)
        return F.linear_f32_from_weights(hidden, descriptor, self.prefix,
                                         hidden.ndim)


class DFlashDraftModel(NetworkModule):
    """DFlash draft model with target-cache update and proposal attention."""

    @classmethod
    def from_config(cls, ctx):
        args = ctx.args
        if (ctx.weights.has("lm_head.weight")
                or ctx.weights.has("lm_head.qweight")):
            return cls(ctx)

        base_cfg = core_config.DeviceConfig.from_pretrained(
            args.target_model_dir, tp_size=args.tp_size, tp_rank=args.tp_rank)
        target_bundle = core_config.BundleConfig.from_pretrained(
            args.target_model_dir)
        conversion = model_registry.weight_conversion_for(
            target_bundle.root_model_type)

        base_weights = ctx.open_weights(
            args.target_model_dir,
            group_size=base_cfg.group_size,
            quant=base_cfg.quant,
            component="llm",
            vocab_map=ctx.weights.vocab_map,
            conversion=conversion,
            int4_gemm_plugin_version=(args.int4_gemm_plugin_version),
            checkpoint_source="target",
            tie_word_embeddings=base_cfg.tie_word_embeddings)
        try:
            base_context = ctx.with_checkpoint(base_cfg, base_weights)
            model = cls(ctx,
                        lm_head=Linear(base_context,
                                       base_weights.causal_lm_head_prefix()))
        except Exception:
            base_weights.close()
            raise
        model._base_weights = base_weights
        return model

    def __init__(self, ctx, lm_head=None) -> None:
        super().__init__(ctx)
        self._base_weights = None
        norm_class = (Gemma4RMSNorm if str(
            ctx.cfg.model_type).startswith("gemma4") else RMSNorm)
        self.fc = DFlashTargetProjection(ctx, "fc")
        self.hidden_norm = norm_class(ctx, "hidden_norm", ctx.cfg.rms_norm_eps)
        self.layers = [
            DFlashDecoderLayer(ctx, f"layers.{index}", index)
            for index in range(ctx.cfg.num_hidden_layers)
        ]
        self.norm = norm_class(ctx, "norm", ctx.cfg.rms_norm_eps)
        self.lm_head = lm_head or Linear(ctx, "lm_head")

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        target_layers = cfg.dflash_target_layer_ids or [1, 8, 15, 22, 29]
        io = {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, cfg.hidden_size)),
            "past_key_values": [
                self.add_input(
                    f"past_key_values_{index}", kv_dtype,
                    (2, -1, F.KV_PAGE_SIZE, cfg.layer_num_kv_heads(index),
                     cfg.layer_head_dim(index)))
                for index in range(cfg.num_hidden_layers)
            ],
            "rope":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, _draft_rotary_dim(cfg))),
            "base_hidden":
            self.add_input("dflash_target_hidden_concat", trt.float16,
                           (-1, len(target_layers) * cfg.hidden_size)),
            "attention_pos_id":
            self.add_input("attention_position_ids", trt.int32, (-1, )),
            "attention_mask":
            self.add_input("packed_attention_mask", trt.int32, (-1, -1)),
            "delta_rope":
            self.add_input("dflash_delta_rope_cos_sin", trt.float32,
                           (-1, _draft_rotary_dim(cfg))),
            "delta_positions":
            self.add_input("dflash_delta_positions", trt.int32, (-1, )),
            "delta_token_to_sequence":
            self.add_input("dflash_delta_token_to_sequence", trt.int32,
                           (-1, )),
        }
        io.update(
            add_ragged_decoder_inputs(self.add_input,
                                      include_logits_indices=False).as_dict())
        return io

    def forward(self, **io):
        hidden = io["inputs_embeds"]
        delta = self.hidden_norm(self.fc(io["base_hidden"]))
        present = []
        for index, layer in enumerate(self.layers):
            hidden, cache = layer(hidden, delta,
                                  io["past_key_values"][index], io["rope"],
                                  RaggedDecoderInputs.from_dict(io),
                                  io["delta_rope"], io["delta_positions"],
                                  io["delta_token_to_sequence"],
                                  io["attention_mask"], io["attention_pos_id"])
            present.append(cache)
        logits = self.lm_head(self.norm(hidden)).cast(trt.float32)
        if self.cfg.final_logit_softcapping is not None:
            cap = np.float32(self.cfg.final_logit_softcapping)
            logits = (logits / cap).tanh() * cap
        outputs = {"logits": logits}
        for index, tensor in enumerate(present):
            outputs[f"present_key_values_{index}"] = tensor
        return outputs

    def close(self) -> None:
        if self._base_weights is not None:
            self._base_weights.close()
            self._base_weights = None
