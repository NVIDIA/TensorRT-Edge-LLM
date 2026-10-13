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
"""Gemma4 assistant checkpoint-direct graph."""

from typing import Dict

import numpy as np
import tensorrt as trt

from ...core import config as core_config
from ...ops import Module, NetworkModule
from ...ops import functional as F
from ...ops.ragged import RaggedDecoderInputs, add_ragged_decoder_inputs
from .. import registry as model_registry
from . import modeling_gemma4_layers as layers
from . import modeling_gemma4_text


class Gemma4AssistantMaskedEmbedder(Module):
    """Ordered-vocabulary projection used by the provider assistant."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        cfg = ctx.cfg
        if cfg.num_centroids <= 0:
            raise ValueError("ordered embeddings require num_centroids > 0")
        if cfg.centroid_intermediate_top_k <= 0:
            raise ValueError(
                "ordered embeddings require centroid_intermediate_top_k > 0")
        if cfg.vocab_size % cfg.num_centroids:
            raise ValueError(
                "ordered embeddings require vocab_size divisible by num_centroids"
            )
        self.centroids = layers.Linear(ctx, self.key("centroids"))

    def forward(self, hidden_states, full_logits):
        cfg = self.cfg
        ordering = self.weights.array(self.key("token_ordering"))
        cluster_ids = np.arange(cfg.num_centroids, dtype=np.int32).repeat(
            cfg.vocab_size // cfg.num_centroids)
        token_to_centroid = np.empty(cfg.vocab_size, dtype=np.int32)
        token_to_centroid[ordering.astype(np.int64)] = cluster_ids

        centroid_logits = self.centroids(hidden_states)
        _, selected = F.topk(centroid_logits, cfg.centroid_intermediate_top_k,
                             1)
        selected = selected.reshape((0, 1, cfg.centroid_intermediate_top_k))
        mapping = F.constant(token_to_centroid.reshape(1, -1, 1),
                             "token_to_centroid")
        selected_mask = mapping.equal(selected)
        selected_mask = F.cast(selected_mask, trt.int32)
        selected_mask = F.reduce(selected_mask, trt.ReduceOperation.MAX,
                                 1 << 2, False)
        selected_mask = F.cast(selected_mask, trt.bool)

        upper = F.constant(np.full((1, 1), 65504.0, dtype=np.float16),
                           "unselected_logit_ceiling")
        selected_logits = F.select(selected_mask, full_logits, upper)
        mask_value = selected_logits.reduce(trt.ReduceOperation.MIN, (1 << 0) |
                                            (1 << 1), True) - np.float16(1.0)
        return F.select(selected_mask, full_logits, mask_value)


class Gemma4AssistantLMHead(Module):
    """Provider assistant output head with optional ordered-vocabulary mask."""

    def __init__(self, ctx, projection=None) -> None:
        super().__init__(ctx, "model.embed_tokens")
        self.projection = projection or layers.Linear(
            ctx, ctx.weights.causal_lm_head_prefix())
        self.masked_embedding = (Gemma4AssistantMaskedEmbedder(
            ctx, "masked_embedding")
                                 if self.cfg.use_ordered_embeddings else None)

    def forward(self, hidden_states):
        full_logits = self.projection(hidden_states)
        if self.masked_embedding is None:
            return full_logits
        return self.masked_embedding(hidden_states, full_logits)


class Gemma4AssistantForCausalLM(NetworkModule):
    """Paired assistant that attends to the target model's KV cache."""

    @classmethod
    def from_config(cls, ctx):
        try:
            ctx.weights.causal_lm_head_prefix()
        except KeyError:
            pass
        else:
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
            projection = layers.Linear(target_context,
                                       target_weights.causal_lm_head_prefix())
            model = cls(ctx, lm_head=Gemma4AssistantLMHead(ctx, projection))
        except Exception:
            target_weights.close()
            raise
        model._target_weights = target_weights
        return model

    def __init__(self, ctx, lm_head=None) -> None:
        super().__init__(ctx)
        self._target_weights = None
        cfg = ctx.cfg
        if cfg.backbone_hidden_size <= 0:
            raise ValueError("Gemma4 assistant requires backbone_hidden_size")
        self.pre_projection = layers.Linear(ctx, "pre_projection")
        self.layers = [
            Gemma4AssistantDecoderLayer(ctx, f"model.layers.{index}", index)
            for index in range(cfg.num_hidden_layers)
        ]
        self.norm = modeling_gemma4_text.Gemma4RMSNorm(ctx, "model.norm",
                                                       cfg.rms_norm_eps)
        self.post_projection = layers.Linear(ctx, "post_projection")
        self.lm_head = lm_head or Gemma4AssistantLMHead(ctx)

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        sliding_dim = cfg.rope_partial_rotary_dim(cfg.sliding_rope_config,
                                                  cfg.head_dim)
        full_dim = cfg.rope_partial_rotary_dim(
            cfg.full_rope_config, cfg.global_head_dim or cfg.head_dim)
        io = {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, cfg.backbone_hidden_size)),
            "hidden_states_input":
            self.add_input("hidden_states_input", trt.float16,
                           (-1, cfg.backbone_hidden_size)),
            "rope_sliding":
            self.add_input("rope_rotary_cos_sin_sliding", trt.float32,
                           (-1, sliding_dim)),
            "rope_full":
            self.add_input("rope_rotary_cos_sin_full", trt.float32,
                           (-1, full_dim)),
            "past": [
                self.add_input(
                    f"past_key_values_{index}", kv_dtype,
                    (2, -1, F.KV_PAGE_SIZE, cfg.layer_num_kv_heads(index),
                     cfg.layer_head_dim(index)))
                for index in range(cfg.num_hidden_layers)
            ],
        }
        io.update(
            add_ragged_decoder_inputs(self.add_input,
                                      include_logits_indices=False).as_dict())
        return io

    def forward(self, **io):
        ragged = RaggedDecoderInputs.from_dict(io)
        merged = F.concatenate(
            (io["inputs_embeds"], io["hidden_states_input"]), 1)
        hidden_states = self.pre_projection(merged)
        for index, layer in enumerate(self.layers):
            rope = (io["rope_full"] if self.cfg.attention_type(index)
                    == "full_attention" else io["rope_sliding"])
            hidden_states = layer(hidden_states, io["past"][index], ragged,
                                  rope)
        hidden_states = self.norm(hidden_states)
        logits = F.cast(self.lm_head(hidden_states), trt.float32)
        feedback = self.post_projection(hidden_states)
        return {"logits": logits, "hidden_states": feedback}

    def close(self) -> None:
        if self._target_weights is not None:
            self._target_weights.close()
            self._target_weights = None


class Gemma4AssistantSharedKVAttention(modeling_gemma4_text.Gemma4TextAttention
                                       ):
    """Q-only attention over the target cache."""

    def forward(self, hidden_states, target_cache, ragged, rope):
        cfg = self.cfg
        query = self.q_proj(hidden_states)
        query = query.reshape((0, cfg.num_attention_heads, self.head_dim))
        query = self.q_norm(query, 3)
        query = query.reshape((0, cfg.num_attention_heads * self.head_dim))
        attention, _ = F.attention(
            query,
            target_cache,
            rope,
            ragged,
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            sliding_window_size=(cfg.sliding_window_size if self.attention_type
                                 == "sliding_attention" else -1),
            enable_fp8_kv_cache=cfg.kv_cache_quant == "fp8",
            attention_scale=cfg.attention_scaling,
            enable_kv_shared=True,
        )
        attention = attention.reshape(
            (0, cfg.num_attention_heads * self.head_dim))
        return self.o_proj(attention)


class Gemma4AssistantDecoderLayer(Module):
    """Assistant decoder layer with shared target KV."""

    def __init__(self, ctx, prefix: str, layer_index: int) -> None:
        super().__init__(ctx, prefix)
        eps = ctx.cfg.rms_norm_eps
        self.input_norm = modeling_gemma4_text.Gemma4RMSNorm(
            ctx, self.key("input_layernorm"), eps)
        self.attention = Gemma4AssistantSharedKVAttention(
            ctx, self.key("self_attn"), layer_index)
        self.post_attention_norm = modeling_gemma4_text.Gemma4RMSNorm(
            ctx, self.key("post_attention_layernorm"), eps)
        self.pre_ffn_norm = modeling_gemma4_text.Gemma4RMSNorm(
            ctx, self.key("pre_feedforward_layernorm"), eps)
        self.mlp = modeling_gemma4_text.Gemma4TextMLP(ctx, self.key("mlp"))
        self.post_ffn_norm = modeling_gemma4_text.Gemma4RMSNorm(
            ctx, self.key("post_feedforward_layernorm"), eps)
        self.layer_scalar = (ctx.weights.f16(
            self.key("layer_scalar")) if ctx.weights.has(
                self.key("layer_scalar")) else np.ones(1, dtype=np.float16))

    def forward(self, hidden_states, cache, ragged, rope):
        attention = self.attention(self.input_norm(hidden_states), cache,
                                   ragged, rope)
        attention = self.post_attention_norm(attention)
        hidden_states = hidden_states + attention
        feed_forward = self.mlp(self.pre_ffn_norm(hidden_states))
        feed_forward = self.post_ffn_norm(feed_forward)
        hidden_states = hidden_states + feed_forward
        return hidden_states * F.constant(self.layer_scalar.reshape(1, 1),
                                          "layer_scalar")
