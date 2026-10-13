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
"""Muse-Glimmer token-major text decoder for checkpoint export."""

from __future__ import annotations

import itertools
import re
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import (QUANT_NVFP4, ModelConfig, module_quant_group_size,
                       module_quant_type)
from ..default.modeling_default import OnnxSpec
from ..linear import TPMode, make_linear
from ..ops import KV_PAGE_SIZE, attention_plugin

__all__ = [
    "MuseGlimmerRMSNorm",
    "MuseGlimmerScalelessRMSNorm",
    "MuseGlimmerAttention",
    "MuseGlimmerMLP",
    "MuseGlimmerDecoderLayer",
    "MuseGlimmerTransformer",
    "MuseGlimmerForCausalLM",
    "MUSE_GLIMMER_KEY_REMAP",
]


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------
class MuseGlimmerRMSNorm(nn.Module):
    """Centered RMSNorm ``(1 + w) * rmsnorm(x)`` with an f32 weight.

    Weight is stored/applied in f32 so large norm gains do not overflow f16 and
    both operands of the elementwise multiply share a type (TRT --stronglyTyped).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(
            torch.zeros(hidden_size, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        hidden_states = hidden_states * (1.0 + self.weight)
        return hidden_states.to(input_dtype)


class MuseGlimmerScalelessRMSNorm(nn.Module):
    """Weightless per-vector RMSNorm (used for QK-norm and the embedding norm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        return hidden_states.to(input_dtype)


class MuseGlimmerRuntimeEmbedding(nn.Module):
    """Packed provider embedding materialized as FP16 for the C++ runtime."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        group_size = module_quant_group_size("embed_tokens", config)
        self.group_size = group_size
        self.register_buffer(
            "weight",
            torch.empty(config.vocab_size,
                        config.hidden_size // 2,
                        dtype=torch.int8))
        self.register_buffer(
            "weight_scale",
            torch.empty(config.vocab_size,
                        config.hidden_size // group_size,
                        dtype=torch.float8_e4m3fn))
        self.register_buffer("weight_scale_2",
                             torch.empty(1, dtype=torch.float32))

    def runtime_weight(self) -> torch.Tensor:
        from ...checkpoint.repacking import decode_modelopt_nvfp4
        dense = decode_modelopt_nvfp4(self.weight, self.weight_scale,
                                      self.weight_scale_2, self.group_size)
        return torch.from_numpy(dense).to(torch.float16)


# ---------------------------------------------------------------------------
# Attention (gated, weightless QK-norm)
# ---------------------------------------------------------------------------
def _attention_type_for_layer(config: ModelConfig, layer_idx: int) -> str:
    return config.attention_layer_types[layer_idx]


class MuseGlimmerAttention(nn.Module):
    """GQA attention with weightless QK-norm and a sigmoid output gate."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        hidden_size = int(config.hidden_size)
        self.attention_type = _attention_type_for_layer(config, layer_idx)
        self.attention_scale = config.attention_scaling
        self.enable_fp8_kv_cache = config.quant.kv_cache_quant == "fp8"
        self.sliding_window_size = (config.sliding_window_size
                                    if self.attention_type
                                    == "sliding_attention" else -1)
        prefix = f"layers.{layer_idx}.self_attn"

        self.q_proj = make_linear(config,
                                  hidden_size,
                                  self.num_heads * self.head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{prefix}.q_proj",
                                  tp_mode=TPMode.COL)
        self.k_proj = make_linear(config,
                                  hidden_size,
                                  self.num_kv_heads * self.head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{prefix}.k_proj",
                                  tp_mode=TPMode.COL)
        self.v_proj = make_linear(config,
                                  hidden_size,
                                  self.num_kv_heads * self.head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{prefix}.v_proj",
                                  tp_mode=TPMode.COL)
        self.o_proj = make_linear(config,
                                  self.num_heads * self.head_dim,
                                  hidden_size,
                                  bias=config.attention_bias,
                                  module_name=f"{prefix}.o_proj",
                                  tp_mode=TPMode.ROW)
        self.gate_proj = make_linear(config,
                                     hidden_size,
                                     self.num_heads * self.head_dim,
                                     bias=False,
                                     module_name=f"{prefix}.gate_proj",
                                     tp_mode=TPMode.COL)
        # Weightless QK-norm (over head_dim), applied before RoPE.
        self.qk_norm = MuseGlimmerScalelessRMSNorm(self.head_dim,
                                                   config.rms_norm_eps)
        if self.enable_fp8_kv_cache:
            self.q_proj.register_buffer("q_scale", torch.ones(1))
            self.k_proj.register_buffer("k_scale", torch.ones(1))
            self.v_proj.register_buffer("v_scale", torch.ones(1))

    def forward_ragged(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        positions: torch.Tensor,
        query_start_offsets: torch.Tensor,
        query_lengths: torch.Tensor,
        past_lengths: torch.Tensor,
        attention_sequence_lengths: torch.Tensor,
        state_indices: torch.Tensor,
        execution_phase_marker: torch.Tensor,
        context_sequence_count_carrier: torch.Tensor,
        kv_page_table: torch.Tensor,
        attention_position_ids: "torch.Tensor | None" = None,
        packed_attention_mask: "torch.Tensor | None" = None,
        tree_parent_ids: "torch.Tensor | None" = None,
        tree_depths: "torch.Tensor | None" = None,
        valid_tree_counts: "torch.Tensor | None" = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del positions, state_indices, tree_parent_ids, tree_depths
        del valid_tree_counts
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = self.qk_norm(q.unflatten(
            -1, (self.num_heads, self.head_dim))).flatten(-2)
        k = self.qk_norm(k.unflatten(
            -1, (self.num_kv_heads, self.head_dim))).flatten(-2)

        enable_tree = packed_attention_mask is not None
        kwargs: dict = {
            "num_q_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_dim,
            "sliding_window_size": self.sliding_window_size,
            "enable_tree_attention": enable_tree,
            "enable_fp8_kv_cache": self.enable_fp8_kv_cache,
            "attention_scale": self.attention_scale,
            "enable_context_mask_selector": False,
            "enable_vision_block_attention": False,
            "skip_softmax_scale_factor": 0.0,
            "qkv_scales": getattr(self, "_qkv_scales_float", [1.0, 1.0, 1.0]),
            "query_start_offsets": query_start_offsets,
            "attention_sequence_lengths": attention_sequence_lengths,
            "execution_phase_marker": execution_phase_marker,
            "context_sequence_count_carrier": context_sequence_count_carrier,
        }
        if enable_tree:
            kwargs["attention_mask"] = packed_attention_mask
            kwargs["attention_pos_id"] = attention_position_ids

        qkv = torch.cat([q, k, v], dim=-1)
        attn_output, present_key_value = attention_plugin(
            qkv, past_key_value, query_lengths, rope_rotary_cos_sin,
            past_lengths, kv_page_table, **kwargs)
        attn_output = attn_output.flatten(-2)
        attn_output = attn_output * torch.sigmoid(
            self.gate_proj(hidden_states))
        return self.o_proj(attn_output), present_key_value


# ---------------------------------------------------------------------------
# MLP / decoder layer
# ---------------------------------------------------------------------------
class MuseGlimmerMLP(nn.Module):
    """SwiGLU MLP (SiLU gate)."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        prefix = f"layers.{layer_idx}.mlp"
        hidden_size = int(config.hidden_size)
        inter = int(config.intermediate_size)
        self.gate_proj = make_linear(config,
                                     hidden_size,
                                     inter,
                                     bias=False,
                                     module_name=f"{prefix}.gate_proj",
                                     tp_mode=TPMode.COL)
        self.up_proj = make_linear(config,
                                   hidden_size,
                                   inter,
                                   bias=False,
                                   module_name=f"{prefix}.up_proj",
                                   tp_mode=TPMode.COL)
        self.down_proj = make_linear(config,
                                     inter,
                                     hidden_size,
                                     bias=False,
                                     module_name=f"{prefix}.down_proj",
                                     tp_mode=TPMode.ROW)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MuseGlimmerDecoderLayer(nn.Module):
    """Muse-Glimmer sandwich-norm decoder layer with gated attention."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = MuseGlimmerAttention(config, layer_idx=layer_idx)
        self.mlp = MuseGlimmerMLP(config, layer_idx=layer_idx)
        rms_eps = float(config.rms_norm_eps)
        post_eps = float(getattr(config, "post_norm_eps", config.rms_norm_eps))
        self.input_layernorm = MuseGlimmerRMSNorm(config.hidden_size, rms_eps)
        self.post_attention_layernorm = MuseGlimmerRMSNorm(
            config.hidden_size, post_eps)
        self.pre_feedforward_layernorm = MuseGlimmerRMSNorm(
            config.hidden_size, rms_eps)
        self.post_feedforward_layernorm = MuseGlimmerRMSNorm(
            config.hidden_size, post_eps)

    def forward_ragged(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        **metadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn.forward_ragged(
            hidden_states, past_key_value, rope_rotary_cos_sin, **metadata)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_key_value


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------
def _select_rope_for_layer(layer, rope_sliding, rope_full):
    """Sliding layers use the real RoPE table; full (NoPE) layers the identity."""
    if layer.self_attn.attention_type == "sliding_attention":
        return rope_sliding
    return rope_full


class MuseGlimmerTransformer(nn.Module):
    """Muse-Glimmer decoder stack with a scaleless embedding norm + dual RoPE."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if module_quant_type("embed_tokens", config) == QUANT_NVFP4:
            self.embed_tokens = MuseGlimmerRuntimeEmbedding(config)
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size,
                                             config.hidden_size)
        self.embed_norm = MuseGlimmerScalelessRMSNorm(config.hidden_size,
                                                      config.rms_norm_eps)
        self.layers = nn.ModuleList([
            MuseGlimmerDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = MuseGlimmerRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.target_hidden_concat: "torch.Tensor | None" = None

    def forward_ragged(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin_sliding: torch.Tensor,
        rope_rotary_cos_sin_full: torch.Tensor,
        positions: torch.Tensor,
        query_start_offsets: torch.Tensor,
        query_lengths: torch.Tensor,
        past_lengths: torch.Tensor,
        attention_sequence_lengths: torch.Tensor,
        state_indices: torch.Tensor,
        execution_phase_marker: torch.Tensor,
        context_sequence_count_carrier: torch.Tensor,
        kv_page_table: torch.Tensor,
        attention_position_ids: "torch.Tensor | None" = None,
        packed_attention_mask: "torch.Tensor | None" = None,
        tree_parent_ids: "torch.Tensor | None" = None,
        tree_depths: "torch.Tensor | None" = None,
        valid_tree_counts: "torch.Tensor | None" = None,
        dflash_target_layer_ids: "List[int] | None" = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        hidden_states = self.embed_norm(inputs_embeds)
        present: List[torch.Tensor] = []
        target_hidden: dict[int, torch.Tensor] = {}
        target_set = set(dflash_target_layer_ids or [])
        for i, layer in enumerate(self.layers):
            rope = _select_rope_for_layer(layer, rope_rotary_cos_sin_sliding,
                                          rope_rotary_cos_sin_full)
            hidden_states, kv = layer.forward_ragged(
                hidden_states,
                past_key_values[i],
                rope,
                positions=positions,
                query_start_offsets=query_start_offsets,
                query_lengths=query_lengths,
                past_lengths=past_lengths,
                attention_sequence_lengths=attention_sequence_lengths,
                state_indices=state_indices,
                execution_phase_marker=execution_phase_marker,
                context_sequence_count_carrier=context_sequence_count_carrier,
                kv_page_table=kv_page_table,
                attention_position_ids=attention_position_ids,
                packed_attention_mask=packed_attention_mask,
                tree_parent_ids=tree_parent_ids,
                tree_depths=tree_depths,
                valid_tree_counts=valid_tree_counts)
            present.append(kv)
            if i in target_set:
                target_hidden[i] = hidden_states
        self.target_hidden_concat = (torch.cat(
            [target_hidden[i] for i in dflash_target_layer_ids], dim=-1)
                                     if dflash_target_layer_ids else None)
        return self.norm(hidden_states), tuple(present)


# ---------------------------------------------------------------------------
# CausalLM
# ---------------------------------------------------------------------------
class MuseGlimmerForCausalLM(nn.Module):
    """Muse-Glimmer causal LM: transformer + lm_head + logit softcap."""

    match_fp32_elementwise_initializers = True
    emit_hidden_states = False

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = MuseGlimmerTransformer(config)
        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")
        self.output_multiplier = float(
            getattr(config, "output_multiplier", 1.0))
        self.final_logit_softcapping = getattr(config,
                                               "final_logit_softcapping", None)

    def tie_weights(self) -> None:
        return

    def forward_ragged(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin_sliding: torch.Tensor,
        rope_rotary_cos_sin_full: torch.Tensor,
        positions: torch.Tensor,
        query_start_offsets: torch.Tensor,
        query_lengths: torch.Tensor,
        past_lengths: torch.Tensor,
        attention_sequence_lengths: torch.Tensor,
        state_indices: torch.Tensor,
        execution_phase_marker: torch.Tensor,
        context_sequence_count_carrier: torch.Tensor,
        kv_page_table: torch.Tensor,
        logits_indices: torch.Tensor,
        attention_position_ids: "torch.Tensor | None" = None,
        packed_attention_mask: "torch.Tensor | None" = None,
        tree_parent_ids: "torch.Tensor | None" = None,
        tree_depths: "torch.Tensor | None" = None,
        valid_tree_counts: "torch.Tensor | None" = None,
    ) -> Tuple:
        dflash_base = getattr(self.config, "dflash_base", False)
        target_layer_ids = (getattr(self.config, "dflash_target_layer_ids",
                                    None) if dflash_base else None)

        hidden_states, present_key_values = self.model.forward_ragged(
            inputs_embeds,
            past_key_values,
            rope_rotary_cos_sin_sliding,
            rope_rotary_cos_sin_full,
            positions,
            query_start_offsets,
            query_lengths,
            past_lengths,
            attention_sequence_lengths,
            state_indices,
            execution_phase_marker,
            context_sequence_count_carrier,
            kv_page_table,
            attention_position_ids=attention_position_ids,
            packed_attention_mask=packed_attention_mask,
            tree_parent_ids=tree_parent_ids,
            tree_depths=tree_depths,
            valid_tree_counts=valid_tree_counts,
            dflash_target_layer_ids=target_layer_ids)

        selected = torch.index_select(hidden_states, 0, logits_indices)
        logits = self.lm_head(selected).to(torch.float32)
        if self.final_logit_softcapping is not None:
            # Fold output_multiplier into the softcap pre-tanh scale to emit a
            # single pre-tanh multiply, matching Gemma3/Gemma4.
            cap = float(self.final_logit_softcapping)
            logits = torch.tanh(logits * (self.output_multiplier / cap)) * cap
        elif self.output_multiplier != 1.0:
            logits = logits * self.output_multiplier

        emitted_hidden = (self.model.target_hidden_concat
                          if dflash_base else None)
        return logits, emitted_hidden, present_key_values

    def onnx_export_spec(self) -> OnnxSpec:
        config = self.config
        num_layers = config.num_hidden_layers
        dflash_base = getattr(config, "dflash_base", False)
        device = next(itertools.chain(self.parameters(),
                                      self.buffers())).device
        num_sequences = 2
        query_length = 2
        physical_tokens = num_sequences * query_length

        inputs_embeds = torch.zeros(physical_tokens,
                                    config.hidden_size,
                                    dtype=torch.float16,
                                    device=device)
        kv_dtype = (torch.float8_e4m3fn
                    if config.quant.kv_cache_quant == "fp8" else torch.float16)
        past_key_values = [
            torch.zeros(2,
                        2,
                        KV_PAGE_SIZE,
                        config.num_key_value_heads,
                        config.head_dim,
                        dtype=kv_dtype,
                        device=device) for _ in range(num_layers)
        ]
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        rope_sliding = torch.zeros(physical_tokens,
                                   rotary_dim,
                                   dtype=torch.float32,
                                   device=device)
        rope_full = torch.zeros(physical_tokens,
                                rotary_dim,
                                dtype=torch.float32,
                                device=device)
        positions = torch.arange(query_length,
                                 dtype=torch.int32,
                                 device=device).repeat(num_sequences)
        query_start_offsets = torch.arange(0,
                                           physical_tokens + 1,
                                           query_length,
                                           dtype=torch.int32,
                                           device=device)
        query_lengths = torch.full((num_sequences, ),
                                   query_length,
                                   dtype=torch.int32,
                                   device=device)
        past_lengths = torch.zeros(num_sequences,
                                   dtype=torch.int32,
                                   device=device)
        attention_sequence_lengths = query_lengths.clone()
        state_indices = torch.arange(num_sequences,
                                     dtype=torch.int32,
                                     device=device)
        execution_phase_marker = torch.zeros(2,
                                             dtype=torch.int32,
                                             device=device)
        context_sequence_count_carrier = torch.zeros(num_sequences,
                                                     dtype=torch.int32,
                                                     device=device)
        kv_page_table = torch.zeros(num_sequences,
                                    2,
                                    2,
                                    dtype=torch.int32,
                                    device=device)
        logits_indices = (torch.tensor(
            [0, 2, 3], dtype=torch.int64, device=device) if dflash_base else
                          query_start_offsets[1:].to(torch.int64) - 1)

        args = (inputs_embeds, *past_key_values, rope_sliding, rope_full,
                positions, query_start_offsets, query_lengths, past_lengths,
                attention_sequence_lengths, state_indices,
                execution_phase_marker, context_sequence_count_carrier,
                kv_page_table, logits_indices)
        input_names = (
            ["inputs_embeds"] +
            [f"past_key_values_{i}" for i in range(num_layers)] + [
                "rope_rotary_cos_sin_sliding", "rope_rotary_cos_sin_full",
                "positions", "query_start_offsets", "query_lengths",
                "past_lengths", "attention_sequence_lengths", "state_indices",
                "execution_phase_marker", "context_sequence_count_carrier",
                "kv_page_table", "logits_indices"
            ])
        output_names = ["logits"] + [
            f"present_key_values_{i}" for i in range(num_layers)
        ]
        if dflash_base:
            output_names = (
                ["logits", "hidden_states"] +
                [f"present_key_values_{i}" for i in range(num_layers)])

        tokens = torch.export.Dim("physical_tokens", min=1, max=8_388_608)
        logits_rows = torch.export.Dim("logits_rows", min=1, max=8_388_608)
        sequences = torch.export.Dim("num_sequences", min=1, max=256)
        context_sequences = torch.export.Dim("num_context_sequences",
                                             min=0,
                                             max=256)
        max_pages = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
        num_pages = torch.export.Dim("num_pages", min=1, max=1048576)
        phase_extent = torch.export.Dim("execution_phase_extent", min=1, max=8)
        packed_mask_width = torch.export.Dim("packed_mask_width",
                                             min=1,
                                             max=64)

        shapes: list = [{0: tokens}]
        shapes.extend({1: num_pages} for _ in range(num_layers))
        shapes.extend([{
            0: tokens
        }, {
            0: tokens
        }, {
            0: tokens
        }, {
            0: sequences + 1
        }, {
            0: sequences
        }, {
            0: sequences
        }, {
            0: sequences
        }, {
            0: sequences
        }, {
            0: phase_extent
        }, {
            0: context_sequences
        }, {
            0: sequences,
            2: max_pages
        }, {
            0: logits_rows if dflash_base else sequences
        }])

        if dflash_base:
            attention_position_ids = positions.clone()
            packed_attention_mask = torch.zeros(physical_tokens,
                                                (query_length + 31) // 32,
                                                dtype=torch.int32,
                                                device=device)
            tree_parent_ids = torch.full((physical_tokens, ),
                                         -1,
                                         dtype=torch.int32,
                                         device=device)
            tree_depths = torch.zeros(physical_tokens,
                                      dtype=torch.int32,
                                      device=device)
            valid_tree_counts = query_lengths.clone()
            args += (attention_position_ids, packed_attention_mask,
                     tree_parent_ids, tree_depths, valid_tree_counts)
            input_names += [
                "attention_position_ids", "packed_attention_mask",
                "tree_parent_ids", "tree_depths", "valid_tree_counts"
            ]
            shapes.extend([{
                0: tokens
            }, {
                0: tokens,
                1: packed_mask_width
            }, {
                0: tokens
            }, {
                0: tokens
            }, {
                0: sequences
            }])

        wrapped = _make_muse_flat_wrapper(self,
                                          num_layers,
                                          dflash_base=dflash_base)
        wrapped.eval()
        return OnnxSpec(wrapped=wrapped,
                        args=args,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=shapes)


def _make_muse_flat_wrapper(model: nn.Module,
                            num_layers: int,
                            dflash_base: bool = False) -> nn.Module:
    """Build a flat wrapper for the token-major Muse-Glimmer ABI."""
    names: List[str] = (
        ["inputs_embeds"] +
        [f"past_key_values_{i}" for i in range(num_layers)] + [
            "rope_rotary_cos_sin_sliding", "rope_rotary_cos_sin_full",
            "positions", "query_start_offsets", "query_lengths",
            "past_lengths", "attention_sequence_lengths", "state_indices",
            "execution_phase_marker", "context_sequence_count_carrier",
            "kv_page_table", "logits_indices"
        ])
    if dflash_base:
        names += [
            "attention_position_ids", "packed_attention_mask",
            "tree_parent_ids", "tree_depths", "valid_tree_counts"
        ]
    kv_tuple = "({},)".format(", ".join(f"past_key_values_{i}"
                                        for i in range(num_layers)))
    tree_kwargs = (", attention_position_ids=attention_position_ids"
                   ", packed_attention_mask=packed_attention_mask"
                   ", tree_parent_ids=tree_parent_ids"
                   ", tree_depths=tree_depths"
                   ", valid_tree_counts=valid_tree_counts"
                   if dflash_base else "")
    call = (
        f"    logits, hidden_states, present = self._model.forward_ragged(\n"
        f"        inputs_embeds, {kv_tuple}, rope_rotary_cos_sin_sliding, "
        f"rope_rotary_cos_sin_full, positions, query_start_offsets, "
        f"query_lengths, past_lengths, attention_sequence_lengths, "
        f"state_indices, execution_phase_marker, "
        f"context_sequence_count_carrier, kv_page_table, logits_indices"
        f"{tree_kwargs})\n")
    if dflash_base:
        body = call + "    return (logits, hidden_states) + tuple(present)\n"
    else:
        body = call + "    return (logits,) + tuple(present)\n"
    src = "def _forward(self, {}):\n{}".format(", ".join(names), body)
    globs: dict = {}
    exec(src, globs)  # noqa: S102

    class _Wrapper(nn.Module):

        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self._model = m

    _Wrapper.forward = globs["_forward"]
    return _Wrapper(model)


# ---------------------------------------------------------------------------
# Checkpoint key remap
# ---------------------------------------------------------------------------
_VISION_PREFIX_RE = re.compile(
    r"^model\.(vision_tower|vision_adapter|vision_projection|"
    r"perception_emb_norm)\.")


def MUSE_GLIMMER_KEY_REMAP(key: str) -> "str | None":
    """Map HF Muse-Glimmer text keys onto the Edge-LLM module tree.

    HF checkpoint layout (VLM):
      ``model.language_model.layers.*`` / ``model.language_model.embed_tokens.*``
      / ``model.language_model.norm.*`` / ``lm_head.*`` and the vision tower under
      ``model.vision_tower.* / vision_adapter.* / vision_projection.*``.

    Edge-LLM text tree: ``model.layers.*`` / ``model.embed_tokens.*`` /
    ``model.norm.*`` / ``lm_head.*``.  So drop the vision tower and rewrite the
    ``language_model`` wrapper to ``model.``.  The weightless ``qk_norm`` /
    ``embed_norm`` carry no checkpoint tensors, so nothing to drop for them.
    """
    if _VISION_PREFIX_RE.match(key):
        return None
    key = re.sub(r"^model\.language_model\.", "model.", key)
    key = re.sub(r"^language_model\.", "model.", key)
    if key.endswith(".weight_scale_inv"):
        key = key[:-len("_inv")]
    return key
