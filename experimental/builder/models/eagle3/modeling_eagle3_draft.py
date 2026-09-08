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
"""EAGLE3 checkpoint-direct draft graph."""

from typing import Dict

import tensorrt as trt

from ...core import config as core_config
from ...ops import (GatedMLP, Linear, Module, NetworkModule, RMSNorm,
                    TreeAttention)
from ...ops import functional as F
from .. import registry as model_registry


class Eagle3DecoderLayer(Module):
    """One EAGLE3 tree-attention decoder layer."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        eps = ctx.cfg.rms_norm_eps
        self.hidden_norm = RMSNorm(ctx, self.key("hidden_norm"), eps)
        self.input_norm = RMSNorm(ctx, self.key("input_layernorm"), eps)
        self.post_norm = RMSNorm(ctx, self.key("post_attention_layernorm"),
                                 eps)
        self.attention = TreeAttention(ctx, self.key("self_attn"))
        self.mlp = GatedMLP(ctx, self.key("mlp"))

    def forward(self, hidden, embeds, past, rope, context_lengths, cache_start,
                kv_page_table, attention_mask, attention_pos_id):
        attention_input = F.concatenate(
            (self.input_norm(embeds), self.hidden_norm(hidden)), 2)
        attention, present = self.attention(attention_input, past, rope,
                                            context_lengths, cache_start,
                                            kv_page_table, attention_mask,
                                            attention_pos_id)
        hidden = hidden + attention
        feed_forward = self.mlp(self.post_norm(hidden))
        hidden = hidden + feed_forward
        return hidden, present


class Eagle3DraftModel(NetworkModule):
    """EAGLE3 draft model with runtime tree-attention metadata."""

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
        self.fc = Linear(ctx, "fc")
        self.layers = [
            Eagle3DecoderLayer(ctx, f"layers.{index}")
            for index in range(ctx.cfg.num_hidden_layers)
        ]
        self.norm = RMSNorm(ctx, "norm", ctx.cfg.rms_norm_eps)
        self.lm_head = lm_head or Linear(ctx, "lm_head")

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        target_hidden = int(cfg.target_hidden_size or cfg.raw_component.get(
            "eagle3_target_hidden_size", cfg.hidden_size))
        target_layers = len(cfg.eagle3_target_layer_ids)
        if target_layers <= 0:
            raise ValueError("EAGLE3 draft requires target-layer metadata")
        return {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, -1, cfg.hidden_size)),
            "past_key_values": [
                self.add_input(f"past_key_values_{index}", kv_dtype,
                               (2, -1, F.KV_PAGE_SIZE, cfg.num_key_value_heads,
                                cfg.head_dim))
                for index in range(cfg.num_hidden_layers)
            ],
            "rope":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, -1, cfg.rotary_dim)),
            "context_lengths":
            self.add_input("context_lengths", trt.int32, (-1, )),
            "cache_start":
            self.add_input("kvcache_start_index", trt.int32, (-1, )),
            "kv_page_table":
            self.add_input("kv_page_table", trt.int32, (-1, 2, -1)),
            "base_hidden":
            self.add_input("hidden_states_input", trt.float16,
                           (-1, -1, target_hidden * target_layers)),
            "draft_hidden":
            self.add_input("hidden_states_from_draft", trt.float16,
                           (-1, -1, cfg.hidden_size)),
            "attention_pos_id":
            self.add_input("attention_pos_id", trt.int32, (-1, -1)),
            "attention_mask":
            self.add_input("attention_mask", trt.int32, (-1, -1, -1)),
            "last_token_ids":
            self.add_input("last_token_ids", trt.int64, (-1, -1)),
        }

    def forward(self, **io):
        outputs = {}
        hidden = self.fc(io["base_hidden"])
        hidden = hidden + io["draft_hidden"]
        present = []
        for index, layer in enumerate(self.layers):
            hidden, cache = layer(hidden, io["inputs_embeds"],
                                  io["past_key_values"][index], io["rope"],
                                  io["context_lengths"], io["cache_start"],
                                  io["kv_page_table"], io["attention_mask"],
                                  io["attention_pos_id"])
            present.append(cache)
        selected = F.gather_last_tokens(hidden, io["last_token_ids"])
        logits = self.lm_head(self.norm(selected)).cast(trt.float32)
        outputs["logits"] = logits.log_softmax(2)
        outputs["hidden_states"] = selected
        for index, tensor in enumerate(present):
            outputs[f"present_key_values_{index}"] = tensor
        return outputs

    def close(self) -> None:
        if self._target_weights is not None:
            self._target_weights.close()
            self._target_weights = None
