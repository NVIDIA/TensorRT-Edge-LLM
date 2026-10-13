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
"""DSpark checkpoint-direct draft backbone."""

from typing import Dict

import numpy as np
import tensorrt as trt

from ...core import config as core_config
from ...core import quantization
from ...ops import Linear, Module, NetworkModule, RMSNorm
from ...ops import functional as F
from ...ops.ragged import RaggedDecoderInputs, add_ragged_decoder_inputs
from .. import registry as model_registry
from ..dflash.modeling_dflash_draft import (DFlashDecoderLayer,
                                            DFlashProposalAttention,
                                            _draft_rotary_dim)
from ..gemma4.modeling_gemma4_text import Gemma4RMSNorm


class DSparkProposalAttention(DFlashProposalAttention):
    """Proposal attention over persistent target-derived and proposal K/V."""

    def __init__(self, ctx, prefix: str, layer_index: int) -> None:
        super().__init__(ctx, prefix, layer_index)
        if not str(ctx.cfg.model_type).startswith("gemma4"):
            self.sliding_window_size = ctx.cfg.sliding_window_size
        self.attention_sinks = None
        if ctx.cfg.attention_sink_bias:
            name = self.key("attention_sink_bias")
            if not self.weights.has(name):
                raise ValueError(
                    f"DSpark attention sink is enabled but {name!r} is missing"
                )
            sinks = self.weights.f32(name).reshape(-1)
            if sinks.shape != (ctx.cfg.num_attention_heads, ):
                raise ValueError(f"{name} has shape {sinks.shape}, expected "
                                 f"{(ctx.cfg.num_attention_heads,)}")
            self.attention_sinks = np.ascontiguousarray(sinks, np.float32)

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
            attention_scale=cfg.attention_scaling,
            enable_fp8_kv_cache=False,
            qkv_scales=qkv_scales,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
            attention_sinks=(F.constant(self.attention_sinks,
                                        "attention_sinks")
                             if self.attention_sinks is not None else None),
            enable_contiguous_query_swa=(cfg.dspark_contiguous_query_swa
                                         and self.sliding_window_size > 0),
        )
        attention = attention.reshape(
            (0, cfg.num_attention_heads * self.head_dim))
        return self.o_proj(attention), present


class DSparkDecoderLayer(DFlashDecoderLayer):
    """One DSpark cached proposal decoder layer."""

    attention_class = DSparkProposalAttention


class DSparkTargetProjection(Module):
    """Project concatenated target states in FP32 before normalization."""

    def forward(self, hidden):
        descriptor = self.weights.linear_descriptor(self.prefix,
                                                    quantization.QUANT_FP16)
        return F.linear_f32_from_weights(hidden, descriptor, self.prefix,
                                         hidden.ndim)


class DSparkDraftModel(NetworkModule):
    """Parallel proposal backbone with hidden output for sequential heads."""

    @classmethod
    def from_config(cls, ctx):
        if (ctx.weights.has("lm_head.weight")
                or ctx.weights.has("lm_head.qweight")):
            return cls(ctx)

        args = ctx.args
        target_cfg = core_config.DeviceConfig.from_pretrained(
            args.target_model_dir, tp_size=args.tp_size, tp_rank=args.tp_rank)
        target_bundle = core_config.BundleConfig.from_pretrained(
            args.target_model_dir)
        conversion = model_registry.weight_conversion_for(
            target_bundle.root_model_type)
        target_weights = ctx.open_weights(
            args.target_model_dir,
            group_size=target_cfg.group_size,
            quant=target_cfg.quant,
            component="llm",
            vocab_map=ctx.weights.vocab_map,
            conversion=conversion,
            int4_gemm_plugin_version=args.int4_gemm_plugin_version,
            checkpoint_source="target",
            tie_word_embeddings=target_cfg.tie_word_embeddings)
        try:
            target_context = ctx.with_checkpoint(target_cfg, target_weights)
            model = cls(ctx,
                        lm_head=Linear(target_context,
                                       target_weights.causal_lm_head_prefix()))
        except Exception:
            target_weights.close()
            raise
        model._target_weights = target_weights
        return model

    def __init__(self, ctx, lm_head=None) -> None:
        super().__init__(ctx)
        self._target_weights = None
        norm_class = (Gemma4RMSNorm if str(
            ctx.cfg.model_type).startswith("gemma4") else RMSNorm)
        self.fc = DSparkTargetProjection(ctx, "fc")
        self.hidden_norm = norm_class(ctx, "hidden_norm", ctx.cfg.rms_norm_eps)
        self.layers = [
            DSparkDecoderLayer(ctx, f"layers.{index}", index)
            for index in range(ctx.cfg.num_hidden_layers)
        ]
        self.norm = norm_class(ctx, "norm", ctx.cfg.rms_norm_eps)
        self.lm_head = lm_head or Linear(ctx, "lm_head")

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        target_layers = cfg.dspark_target_layer_ids
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
        hidden = self.norm(hidden)
        logits = F.cast(self.lm_head(hidden), trt.float32)
        if self.cfg.final_logit_softcapping is not None:
            cap = np.float32(self.cfg.final_logit_softcapping)
            logits = (logits / cap).tanh() * cap
        outputs = {
            "logits": logits,
            "dspark_hidden_states": F.cast(hidden, trt.float16),
        }
        for index, tensor in enumerate(present):
            outputs[f"present_key_values_{index}"] = tensor
        return outputs

    def close(self) -> None:
        if self._target_weights is not None:
            self._target_weights.close()
            self._target_weights = None
