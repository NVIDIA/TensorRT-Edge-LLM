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
"""Reusable dense decoder modules with Transformers-style composition."""

import logging
from typing import List, Optional, Sequence, Tuple, Type

from ..core import quantization
from . import functional as F
from .linear import Linear
from .mlp import GatedMLP
from .module import BuildContext, Module
from .normalization import RMSNorm
from .ragged import RaggedDecoderInputs
from .tensor import Tensor

LOGGER = logging.getLogger("builder.ops.transformer")

_INT4_PLUGIN_TYPES = frozenset((
    quantization.QUANT_INT4_AWQ,
    quantization.QUANT_INT4_AWQ_MODELOPT,
    quantization.QUANT_INT4_GPTQ,
))


def pack_qkv(query: Tensor, key: Tensor, value: Tensor,
             value_projection: Linear) -> Tensor:
    """Pack attention projections while preserving V2 INT4 value storage."""
    needs_kv_boundary = (
        value_projection.ctx.backend == "edgellm"
        and value_projection.ctx.options.int4_gemm_plugin_version == 2
        and value_projection.quant_type() in _INT4_PLUGIN_TYPES)
    if needs_kv_boundary:
        # A direct three-way concat can let the optimizer reuse the value
        # plugin's transient storage before AttentionPlugin consumes every
        # prefill row. Materializing K/V first preserves the dependency without
        # changing the engine I/O contract.
        return F.concatenate((query, F.concatenate((key, value), 1)), 1)
    return F.concatenate((query, key, value), 1)


class DecoderAttention(Module):
    """Grouped-query decoder attention."""

    def __init__(self,
                 ctx: BuildContext,
                 prefix: str,
                 layer_index: int = None,
                 sliding_window_size: int = None) -> None:
        super().__init__(ctx, prefix)
        if sliding_window_size is not None:
            self.sliding_window_size = sliding_window_size
        elif (layer_index is not None
              and ctx.cfg.attention_type(layer_index) != "sliding_attention"):
            self.sliding_window_size = -1
        else:
            self.sliding_window_size = ctx.cfg.sliding_window_size
        self.q_proj = Linear(ctx, self.key("q_proj"))
        self.k_proj = Linear(ctx, self.key("k_proj"))
        self.v_proj = Linear(ctx, self.key("v_proj"))
        self.o_proj = Linear(ctx, self.key("o_proj"))

    def project_qkv(self,
                    hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Project Q/K/V; model-family subclasses may refine this step."""
        return (self.q_proj(hidden_states), self.k_proj(hidden_states),
                self.v_proj(hidden_states))

    def packed_qkv(self, hidden_states: Tensor) -> Tensor:
        """Return the packed projection consumed by decoder attention."""
        query, key, value = self.project_qkv(hidden_states)
        return pack_qkv(query, key, value, self.v_proj)

    def attention_kwargs(self) -> dict:
        """Return optional parameters owned by this attention family."""
        return {}

    def forward(
        self,
        hidden_states: Tensor,
        past_key_value: Tensor,
        rope_rotary_cos_sin: Tensor,
        ragged: RaggedDecoderInputs,
        attention_mask: Tensor = None,
        attention_pos_id: Tensor = None,
        tree_parent_ids: Tensor = None,
        tree_depths: Tensor = None,
        valid_tree_counts: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        cfg = self.cfg
        qkv = self.packed_qkv(hidden_states)
        output, present_key_value = F.attention(
            qkv,
            past_key_value,
            rope_rotary_cos_sin,
            ragged,
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_size=cfg.head_dim,
            sliding_window_size=self.sliding_window_size,
            enable_fp8_kv_cache=cfg.kv_cache_quant == "fp8",
            qkv_scales=self.weights.qkv_scales(self.prefix),
            skip_softmax_scale_factor=cfg.skip_softmax_scale_factor,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
            **self.attention_kwargs(),
        )
        output = output.reshape((0, cfg.num_attention_heads * cfg.head_dim))
        return self.o_proj(output), present_key_value


class QKNormDecoderAttention(DecoderAttention):
    """Grouped-query decoder attention with per-head Q/K normalization."""

    def __init__(self,
                 ctx: BuildContext,
                 prefix: str,
                 layer_index: int = None,
                 sliding_window_size: int = None) -> None:
        super().__init__(ctx, prefix, layer_index, sliding_window_size)
        self.q_norm = RMSNorm(ctx, self.key("q_norm"), ctx.cfg.rms_norm_eps)
        self.k_norm = RMSNorm(ctx, self.key("k_norm"), ctx.cfg.rms_norm_eps)

    def attention_kwargs(self) -> dict:
        """Fuse the provider's per-head Q/K normalization into attention."""
        q_norm = self.weights.fp16_parameter(self.q_norm.key("weight"))
        k_norm = self.weights.fp16_parameter(self.k_norm.key("weight"))
        return {
            "q_norm_gamma":
            F.parameter(q_norm.name,
                        q_norm.value,
                        "fp16",
                        recipe=q_norm.recipe),
            "k_norm_gamma":
            F.parameter(k_norm.name,
                        k_norm.value,
                        "fp16",
                        recipe=k_norm.recipe),
            "rms_norm_eps":
            self.cfg.rms_norm_eps,
        }


class GatedDecoderAttention(Module):
    """Qwen3.5-style decoder attention with a per-head output gate."""

    def __init__(self, ctx: BuildContext, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.q_proj = Linear(ctx, self.key("q_proj"))
        self.k_proj = Linear(ctx, self.key("k_proj"))
        self.v_proj = Linear(ctx, self.key("v_proj"))
        self.o_proj = Linear(ctx, self.key("o_proj"))
        self.q_norm = RMSNorm(ctx,
                              self.key("q_norm"),
                              ctx.cfg.rms_norm_eps,
                              unit_offset=True)
        self.k_norm = RMSNorm(ctx,
                              self.key("k_norm"),
                              ctx.cfg.rms_norm_eps,
                              unit_offset=True)

    def forward(self,
                hidden_states,
                past_key_value,
                rope_rotary_cos_sin,
                ragged,
                attention_mask=None,
                attention_pos_id=None,
                tree_parent_ids=None,
                tree_depths=None,
                valid_tree_counts=None):
        cfg = self.cfg
        projected = self.q_proj(hidden_states).reshape(
            (0, cfg.num_attention_heads, cfg.head_dim * 2))
        query = projected.slice_last_dim(0, cfg.head_dim, 3)
        gate = projected.slice_last_dim(cfg.head_dim, cfg.head_dim, 3)
        key = self.k_proj(hidden_states).reshape(
            (0, cfg.num_key_value_heads, cfg.head_dim))
        value = self.v_proj(hidden_states)
        query = self.q_norm(query, 3).reshape(
            (0, cfg.num_attention_heads * cfg.head_dim))
        key = self.k_norm(key, 3).reshape(
            (0, cfg.num_key_value_heads * cfg.head_dim))
        qkv = pack_qkv(query, key, value, self.v_proj)
        output, present = F.attention(
            qkv,
            past_key_value,
            rope_rotary_cos_sin,
            ragged,
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_size=cfg.head_dim,
            enable_fp8_kv_cache=cfg.kv_cache_quant == "fp8",
            qkv_scales=self.weights.qkv_scales(self.prefix),
            skip_softmax_scale_factor=cfg.skip_softmax_scale_factor,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
        )
        output = (output * gate.sigmoid()).reshape(
            (0, cfg.num_attention_heads * cfg.head_dim))
        return self.o_proj(output), present


class TreeAttention(Module):
    """Decoder attention accepting tree-verification metadata."""

    def __init__(self, ctx: BuildContext, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.q_proj = Linear(ctx, self.key("q_proj"))
        self.k_proj = Linear(ctx, self.key("k_proj"))
        self.v_proj = Linear(ctx, self.key("v_proj"))
        self.o_proj = Linear(ctx, self.key("o_proj"))
        self.q_norm = RMSNorm(ctx, self.key("q_norm"), ctx.cfg.rms_norm_eps)
        self.k_norm = RMSNorm(ctx, self.key("k_norm"), ctx.cfg.rms_norm_eps)

    def forward(self,
                hidden,
                past,
                rope,
                ragged,
                attention_mask,
                attention_pos_id,
                tree_parent_ids=None,
                tree_depths=None,
                valid_tree_counts=None):
        cfg = self.cfg
        query = self.q_proj(hidden)
        gate = None
        query_size = cfg.num_attention_heads * cfg.head_dim
        if cfg.attn_output_gate:
            gate = query[..., query_size:query_size * 2]
            query = query[..., :query_size]
        key = self.k_proj(hidden)
        value = self.v_proj(hidden)
        if self.weights.has(self.key("q_norm.weight")):
            query = self.q_norm(
                query.reshape((0, cfg.num_attention_heads, cfg.head_dim)),
                3).reshape((0, cfg.num_attention_heads * cfg.head_dim))
        if self.weights.has(self.key("k_norm.weight")):
            key = self.k_norm(
                key.reshape((0, cfg.num_key_value_heads, cfg.head_dim)),
                3).reshape((0, cfg.num_key_value_heads * cfg.head_dim))
        qkv = pack_qkv(query, key, value, self.v_proj)
        output, present = F.attention(
            qkv,
            past,
            rope,
            ragged,
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_size=cfg.head_dim,
            enable_fp8_kv_cache=cfg.kv_cache_quant == "fp8",
            qkv_scales=self.weights.qkv_scales(self.prefix),
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
        )
        output = output.reshape((0, cfg.num_attention_heads * cfg.head_dim))
        if gate is not None:
            output = output * gate.sigmoid()
        return self.o_proj(output), present


class DecoderLayer(Module):
    """Pre-normalized attention and gated feed-forward decoder layer."""

    attention_class = DecoderAttention
    mlp_class = GatedMLP
    norm_class = RMSNorm

    def __init__(self,
                 ctx: BuildContext,
                 prefix: str,
                 layer_index: int = None,
                 sliding_window_size: int = None) -> None:
        super().__init__(ctx, prefix)
        self.self_attn = self.attention_class(ctx, self.key("self_attn"),
                                              layer_index, sliding_window_size)
        self.input_layernorm = self.norm_class(ctx,
                                               self.key("input_layernorm"),
                                               ctx.cfg.rms_norm_eps)
        self.post_attention_layernorm = self.norm_class(
            ctx, self.key("post_attention_layernorm"), ctx.cfg.rms_norm_eps)
        self.mlp = self.mlp_class(ctx, self.key("mlp"))

    def forward(
        self,
        hidden_states: Tensor,
        past_key_value: Tensor,
        rope_rotary_cos_sin: Tensor,
        ragged: RaggedDecoderInputs,
        attention_mask: Tensor = None,
        attention_pos_id: Tensor = None,
        tree_parent_ids: Tensor = None,
        tree_depths: Tensor = None,
        valid_tree_counts: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        attention_output, present = self.self_attn(
            self.input_layernorm(hidden_states), past_key_value,
            rope_rotary_cos_sin, ragged, attention_mask, attention_pos_id,
            tree_parent_ids, tree_depths, valid_tree_counts)
        hidden_states = hidden_states + attention_output
        return (hidden_states +
                self.mlp(self.post_attention_layernorm(hidden_states)),
                present)


class DecoderModel(Module):
    """Shared execution of a model-specific dense decoder stack.

    Concrete model families provide ``layer_class``. This keeps the repeated
    stack traversal in one place without replacing the family-specific
    attention, layer, model, or engine-contract classes.
    """

    layer_class: Optional[Type[DecoderLayer]] = None
    norm_class = RMSNorm

    def __init__(self, ctx: BuildContext, prefix: str = "model") -> None:
        super().__init__(ctx, prefix)
        if self.layer_class is None:
            raise TypeError(
                "DecoderModel must be subclassed with a model-specific "
                "layer_class")
        self.layers = [
            self.layer_class(ctx, self.key(f"layers.{index}"), index)
            for index in range(ctx.cfg.num_hidden_layers)
        ]
        self.norm = self.norm_class(ctx, self.key("norm"),
                                    ctx.cfg.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: Tensor,
        past_key_values: List[Tensor],
        rope_rotary_cos_sin: Tensor,
        ragged: RaggedDecoderInputs,
        deepstack_embeds: Sequence[Tensor] = (),
        attention_mask: Tensor = None,
        attention_pos_id: Tensor = None,
        tree_parent_ids: Tensor = None,
        tree_depths: Tensor = None,
        valid_tree_counts: Tensor = None,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        hidden_states = inputs_embeds
        present_key_values = []
        all_hidden_states = []
        for layer_index, layer in enumerate(self.layers):
            LOGGER.debug("building layer %d/%d", layer_index + 1,
                         len(self.layers))
            hidden_states, present = layer(hidden_states,
                                           past_key_values[layer_index],
                                           rope_rotary_cos_sin, ragged,
                                           attention_mask, attention_pos_id,
                                           tree_parent_ids, tree_depths,
                                           valid_tree_counts)
            if layer_index < len(deepstack_embeds):
                hidden_states = hidden_states + deepstack_embeds[layer_index]
            all_hidden_states.append(hidden_states)
            present_key_values.append(present)
        return self.norm(hidden_states), present_key_values, all_hidden_states
