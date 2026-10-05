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
"""
EAGLE3 Draft Model for speculative decoding.

The draft model predicts multiple tokens ahead to accelerate generation.
It fuses hidden states from three selected base-model layers via an ``fc``
projection, adds the previous draft hidden states, and runs a single
decoder layer to predict next-token logits.

Forward I/O matches the C++ ``EagleDraftEngineRunner`` binding names.

Checkpoint layout (HuggingFace EAGLE3 draft repos)
---------------------------------------------------
    d2t                              [draft_vocab_size]  int64 (draft-to-target map)
    fc.weight                        [hidden, target_hidden * 3]
    lm_head.weight                   [draft_vocab_size, hidden]
    norm.weight                      [hidden]
    midlayer.hidden_norm.weight      [hidden]
    midlayer.input_layernorm.weight  [hidden]
    midlayer.self_attn.{q,k,v,o}_proj.weight
    midlayer.mlp.{gate,up,down}_proj.weight
    midlayer.post_attention_layernorm.weight

``midlayer`` is remapped to ``layers.0`` on load; ``t2d`` keys are skipped.
"""

import itertools
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import ModelConfig, _is_gemma4_model_type
from ..default.modeling_default import OnnxSpec, RMSNorm
# yapf: disable
from ..gemma4.modeling_gemma4_text import (Gemma4MLP, Gemma4RMSNorm,
                                           Gemma4ValueRMSNorm,
                                           _attention_type_for_layer,
                                           _head_dim_for_attention_type,
                                           _num_kv_heads_for_attention_type,
                                           _rotary_dim_from_rope_config,
                                           _uses_attention_k_eq_v)
# yapf: enable
from ..linear import is_int4_linear, make_linear
from ..ops import KV_PAGE_SIZE, attention_plugin, qkv_concat

__all__ = ["Eagle3DraftModel"]

# ---------------------------------------------------------------------------
# Dummy-shape constants for ONNX export
# ---------------------------------------------------------------------------

_BATCH_SIZE = 1
_SEQ_LEN = 1
_PAST_LEN = 1
_MAX_POS = 4096

# ---------------------------------------------------------------------------
# Eagle3 Attention (with tree attention mask)
# ---------------------------------------------------------------------------


class Eagle3Attention(nn.Module):
    """GQA attention for EAGLE3 draft with explicit tree attention mask.

    Q/K/V projections accept ``in_features = 2 * hidden_size`` because the
    decoder layer concatenates normalised ``inputs_embeds`` and ``hidden_states``
    before the attention block.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        qkv_in_features = hidden_size * 2

        self.layer_idx = layer_idx
        self.is_gemma4 = _is_gemma4_model_type(config.model_type)
        if self.is_gemma4:
            self.attention_type = _attention_type_for_layer(config, layer_idx)
            self.attention_k_eq_v = _uses_attention_k_eq_v(
                config, self.attention_type)
            self.num_heads = int(config.num_attention_heads)
            self.num_kv_heads = _num_kv_heads_for_attention_type(
                config, self.attention_type)
            self.head_dim = _head_dim_for_attention_type(
                config, self.attention_type)
            self.sliding_window_size = (config.sliding_window_size
                                        if self.attention_type
                                        == "sliding_attention" else -1)
        else:
            self.attention_k_eq_v = False
            self.num_heads = config.num_attention_heads
            self.num_kv_heads = config.num_key_value_heads
            self.head_dim = config.head_dim
            self.sliding_window_size = -1
        self.attention_scale = config.attention_scaling

        self.q_proj = make_linear(config,
                                  qkv_in_features,
                                  self.num_heads * self.head_dim,
                                  bias=config.attention_bias)
        self.k_proj = make_linear(config,
                                  qkv_in_features,
                                  self.num_kv_heads * self.head_dim,
                                  bias=config.attention_bias)
        if not self.attention_k_eq_v:
            self.v_proj = make_linear(config,
                                      qkv_in_features,
                                      self.num_kv_heads * self.head_dim,
                                      bias=config.attention_bias)
        qkv_projections = [self.q_proj, self.k_proj]
        if not self.attention_k_eq_v:
            qkv_projections.append(self.v_proj)
        self._uses_int4_qkv = any(
            is_int4_linear(proj) for proj in qkv_projections)
        self.o_proj = make_linear(config, self.num_heads * self.head_dim,
                                  hidden_size)

        if config.has_qk_norm:
            norm_cls = Gemma4RMSNorm if self.is_gemma4 else RMSNorm
            self.q_norm = norm_cls(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = norm_cls(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.v_norm = (Gemma4ValueRMSNorm(self.head_dim, config.rms_norm_eps)
                       if self.is_gemma4 and config.has_value_norm else None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        attention_mask: torch.Tensor,
        attention_pos_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states_raw = self.k_proj(hidden_states)
        value_states = (key_states_raw if self.attention_k_eq_v else
                        self.v_proj(hidden_states))
        key_states = key_states_raw

        if self.q_norm is not None:
            query_states = self.q_norm(
                query_states.reshape(batch_size, seq_len, self.num_heads,
                                     self.head_dim)).reshape(
                                         batch_size, seq_len,
                                         self.num_heads * self.head_dim)
        if self.k_norm is not None:
            key_states = self.k_norm(
                key_states.reshape(batch_size, seq_len, self.num_kv_heads,
                                   self.head_dim)).reshape(
                                       batch_size, seq_len,
                                       self.num_kv_heads * self.head_dim)
        if self.v_norm is not None:
            value_states = self.v_norm(
                value_states.reshape(batch_size, seq_len, self.num_kv_heads,
                                     self.head_dim)).reshape(
                                         batch_size, seq_len,
                                         self.num_kv_heads * self.head_dim)

        qkv = (qkv_concat(query_states, key_states, value_states)
               if self._uses_int4_qkv else torch.cat(
                   [query_states, key_states, value_states], dim=-1))
        attn_output, present_key_value = attention_plugin(
            qkv,
            past_key_value,
            context_lengths,
            rope_rotary_cos_sin,
            kvcache_start_index,
            kv_page_table,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            sliding_window_size=self.sliding_window_size,
            enable_tree_attention=True,
            enable_fp8_kv_cache=False,
            attention_scale=self.attention_scale,
            enable_context_mask_selector=False,
            enable_vision_block_attention=False,
            skip_softmax_scale_factor=0.0,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
            qkv_scales=[1.0, 1.0, 1.0],
        )
        attn_output = attn_output.reshape(batch_size, seq_len,
                                          self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present_key_value

    def forward_ragged(self, hidden_states: torch.Tensor,
                       past_key_value: torch.Tensor, **m):
        tokens = hidden_states.shape[0]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = (key_states if self.attention_k_eq_v else
                        self.v_proj(hidden_states))
        if self.q_norm is not None:
            query_states = self.q_norm(
                query_states.reshape(tokens, self.num_heads,
                                     self.head_dim)).reshape(tokens, -1)
        if self.k_norm is not None:
            key_states = self.k_norm(
                key_states.reshape(tokens, self.num_kv_heads,
                                   self.head_dim)).reshape(tokens, -1)
        if self.v_norm is not None:
            value_states = self.v_norm(
                value_states.reshape(tokens, self.num_kv_heads,
                                     self.head_dim)).reshape(tokens, -1)
        qkv = (qkv_concat(query_states, key_states, value_states)
               if self._uses_int4_qkv else torch.cat(
                   (query_states, key_states, value_states), dim=-1))
        output, present = attention_plugin(
            qkv,
            past_key_value,
            m["query_lengths"],
            m["rope_rotary_cos_sin"],
            m["past_lengths"],
            m["kv_page_table"],
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            sliding_window_size=self.sliding_window_size,
            enable_tree_attention=True,
            enable_fp8_kv_cache=False,
            attention_scale=self.attention_scale,
            enable_context_mask_selector=False,
            enable_vision_block_attention=False,
            skip_softmax_scale_factor=0.0,
            attention_mask=m["packed_attention_mask"],
            attention_pos_id=m["attention_position_ids"],
            qkv_scales=[1.0, 1.0, 1.0],
            query_start_offsets=m["query_start_offsets"],
            attention_sequence_lengths=m["attention_sequence_lengths"],
            execution_phase_marker=m["execution_phase_marker"],
            context_sequence_count_carrier=m["context_sequence_count_carrier"])
        return self.o_proj(output.reshape(tokens, -1)), present


# ---------------------------------------------------------------------------
# MLP (same as default)
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """SwiGLU MLP: gate_proj, up_proj, down_proj."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = make_linear(config, config.hidden_size,
                                     config.intermediate_size)
        self.up_proj = make_linear(config, config.hidden_size,
                                   config.intermediate_size)
        self.down_proj = make_linear(config, config.intermediate_size,
                                     config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) *
            self.up_proj(hidden_states))


# ---------------------------------------------------------------------------
# Eagle3 Decoder Layer
# ---------------------------------------------------------------------------


class Eagle3DecoderLayer(nn.Module):
    """Decoder layer for EAGLE3 draft model.

    Before attention, both ``hidden_states`` and ``inputs_embeds`` are
    independently normalised and concatenated along the feature dimension,
    producing ``[batch, seq, 2 * hidden_size]`` as input to Q/K/V.

    Submodule names match checkpoint keys (after ``midlayer`` -> ``layers.0``):
        self_attn, mlp, hidden_norm, input_layernorm, post_attention_layernorm
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_gemma4 = _is_gemma4_model_type(config.model_type)
        self.self_attn = Eagle3Attention(config, layer_idx=layer_idx)
        self.mlp = (Gemma4MLP(config, layer_idx=layer_idx)
                    if self.is_gemma4 else MLP(config))
        norm_cls = Gemma4RMSNorm if self.is_gemma4 else RMSNorm
        self.hidden_norm = norm_cls(config.hidden_size, config.rms_norm_eps)
        self.input_layernorm = norm_cls(config.hidden_size,
                                        config.rms_norm_eps)
        self.post_attention_layernorm = norm_cls(config.hidden_size,
                                                 config.rms_norm_eps)
        if self.is_gemma4:
            self.pre_feedforward_layernorm = Gemma4RMSNorm(
                config.hidden_size, config.rms_norm_eps)
            self.post_feedforward_layernorm = Gemma4RMSNorm(
                config.hidden_size, config.rms_norm_eps)
            self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        attention_mask: torch.Tensor,
        attention_pos_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states

        # Norm + concat: [batch, seq, 2*hidden]
        normed_hidden = self.hidden_norm(hidden_states)
        normed_embeds = self.input_layernorm(inputs_embeds)
        concat_hidden = torch.cat((normed_embeds, normed_hidden), dim=-1)

        attn_output, present_key_value = self.self_attn(
            concat_hidden,
            past_key_value,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            kv_page_table,
            attention_mask,
            attention_pos_id,
        )
        if self.is_gemma4:
            hidden_states = self.post_attention_layernorm(attn_output)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.post_feedforward_layernorm(hidden_states)
            hidden_states = residual + hidden_states
            hidden_states = hidden_states * self.layer_scalar.to(
                dtype=hidden_states.dtype)
        else:
            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = residual + self.mlp(
                self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value

    def forward_ragged(self, hidden_states: torch.Tensor,
                       inputs_embeds: torch.Tensor,
                       past_key_value: torch.Tensor, **metadata):
        residual = hidden_states
        concat_hidden = torch.cat((self.input_layernorm(inputs_embeds),
                                   self.hidden_norm(hidden_states)),
                                  dim=-1)
        attention, present = self.self_attn.forward_ragged(
            concat_hidden, past_key_value, **metadata)
        if self.is_gemma4:
            hidden_states = residual + self.post_attention_layernorm(attention)
            residual = hidden_states
            hidden_states = residual + self.post_feedforward_layernorm(
                self.mlp(self.pre_feedforward_layernorm(hidden_states)))
            hidden_states = hidden_states * self.layer_scalar.to(
                dtype=hidden_states.dtype)
        else:
            hidden_states = residual + attention
            hidden_states = hidden_states + self.mlp(
                self.post_attention_layernorm(hidden_states))
        return hidden_states, present


# ---------------------------------------------------------------------------
# Eagle3 Draft Model
# ---------------------------------------------------------------------------


def _make_flat_wrapper_eagle3_ragged(model: nn.Module, num_layers: int):
    names = (["inputs_embeds"] +
             [f"past_key_values_{i}" for i in range(num_layers)] + [
                 "rope_rotary_cos_sin", "positions", "query_start_offsets",
                 "query_lengths", "past_lengths", "attention_sequence_lengths",
                 "state_indices", "execution_phase_marker",
                 "context_sequence_count_carrier", "kv_page_table",
                 "logits_indices", "hidden_states_input",
                 "hidden_states_from_draft", "attention_position_ids",
                 "packed_attention_mask", "tree_parent_ids", "tree_depths",
                 "valid_tree_counts"
             ])
    kv = "({},)".format(", ".join(f"past_key_values_{i}"
                                  for i in range(num_layers)))
    body = (
        f"    logits, hidden_states, present = self._model.forward_ragged(\n"
        f"        inputs_embeds, {kv}, logits_indices, hidden_states_input, "
        f"hidden_states_from_draft, rope_rotary_cos_sin=rope_rotary_cos_sin, "
        f"positions=positions, query_start_offsets=query_start_offsets, "
        f"query_lengths=query_lengths, "
        f"past_lengths=past_lengths, "
        f"attention_sequence_lengths=attention_sequence_lengths, "
        f"state_indices=state_indices, execution_phase_marker=execution_phase_marker, "
        f"context_sequence_count_carrier=context_sequence_count_carrier, "
        f"kv_page_table=kv_page_table, attention_position_ids=attention_position_ids, "
        f"packed_attention_mask=packed_attention_mask, tree_parent_ids=tree_parent_ids, "
        f"tree_depths=tree_depths, valid_tree_counts=valid_tree_counts)\n"
        f"    return (logits, hidden_states) + tuple(present)\n")
    globs: dict = {}
    exec("def _forward(self, {}):\n{}".format(", ".join(names), body),
         globs)  # noqa: S102

    class _Wrapper(nn.Module):

        def __init__(self, m):
            super().__init__()
            self._model = m

    _Wrapper.forward = globs["_forward"]
    return _Wrapper(model)


def _draft_ragged_export_spec(model: nn.Module, num_layers: int,
                              target_hidden_size: int, wrapper_factory):
    config = model.config
    device = next(itertools.chain(model.parameters(), model.buffers())).device
    n, q, tokens = 2, 64, 128
    inputs = torch.zeros(tokens,
                         config.hidden_size,
                         dtype=torch.float16,
                         device=device)
    kv = tuple(
        torch.zeros(2,
                    2,
                    KV_PAGE_SIZE,
                    config.num_key_value_heads,
                    config.head_dim,
                    dtype=torch.float16,
                    device=device) for _ in range(num_layers))
    rotary_dim = int(config.head_dim * config.partial_rotary_factor)
    if _is_gemma4_model_type(config.model_type):
        rotary_dim = _rotary_dim_from_rope_config(config, None,
                                                  config.head_dim)
    rope = torch.zeros(tokens, rotary_dim, dtype=torch.float32, device=device)
    positions = torch.arange(q, dtype=torch.int32, device=device).repeat(n)
    offsets = torch.arange(0, tokens + 1, q, dtype=torch.int32, device=device)
    lengths = torch.full((n, ), q, dtype=torch.int32, device=device)
    past = torch.zeros(n, dtype=torch.int32, device=device)
    state = torch.arange(n, dtype=torch.int32, device=device)
    phase = torch.zeros(2, dtype=torch.int32, device=device)
    context_count = torch.empty(n, dtype=torch.int32, device=device)
    page_table = torch.zeros(n, 2, 2, dtype=torch.int32, device=device)
    selected = offsets[1:].to(torch.int64) - 1
    target = torch.zeros(tokens,
                         target_hidden_size,
                         dtype=torch.float16,
                         device=device)
    draft = torch.zeros(tokens,
                        config.hidden_size,
                        dtype=torch.float16,
                        device=device)
    mask = torch.zeros(tokens, (q + 31) // 32,
                       dtype=torch.int32,
                       device=device)
    parents = torch.full((tokens, ), -1, dtype=torch.int32, device=device)
    depths = torch.zeros(tokens, dtype=torch.int32, device=device)
    args = (inputs, *kv, rope, positions, offsets, lengths, past,
            lengths.clone(), state, phase, context_count, page_table, selected,
            target, draft, positions.clone(), mask, parents, depths,
            lengths.clone())
    names = (["inputs_embeds"] +
             [f"past_key_values_{i}" for i in range(num_layers)] + [
                 "rope_rotary_cos_sin", "positions", "query_start_offsets",
                 "query_lengths", "past_lengths", "attention_sequence_lengths",
                 "state_indices", "execution_phase_marker",
                 "context_sequence_count_carrier", "kv_page_table",
                 "logits_indices", "hidden_states_input",
                 "hidden_states_from_draft", "attention_position_ids",
                 "packed_attention_mask", "tree_parent_ids", "tree_depths",
                 "valid_tree_counts"
             ])
    outputs = ["logits", "hidden_states"
               ] + [f"present_key_values_{i}" for i in range(num_layers)]
    token_dim = torch.export.Dim("physical_tokens", min=1, max=8_388_608)
    seq_dim = torch.export.Dim("num_sequences", min=1, max=256)
    context_seq_dim = torch.export.Dim("num_context_sequences", min=0, max=256)
    pages = torch.export.Dim("num_pages", min=1, max=1048576)
    max_pages = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
    phase_dim = torch.export.Dim("execution_phase_extent", min=1, max=8)
    selected_dim = torch.export.Dim("selected_rows", min=1, max=8_388_608)
    packed_mask_width = torch.export.Dim("packed_mask_width", min=1, max=64)
    shapes = [{0: token_dim}] + [{1: pages} for _ in range(num_layers)]
    shapes += [{
        0: token_dim
    }, {
        0: token_dim
    }, {
        0: seq_dim + 1
    }, {
        0: seq_dim
    }, {
        0: seq_dim
    }, {
        0: seq_dim
    }, {
        0: seq_dim
    }, {
        0: phase_dim
    }, {
        0: context_seq_dim
    }, {
        0: seq_dim,
        2: max_pages
    }, {
        0: selected_dim
    }, {
        0: token_dim
    }, {
        0: token_dim
    }, {
        0: token_dim
    }, {
        0: token_dim,
        1: packed_mask_width
    }, {
        0: token_dim
    }, {
        0: token_dim
    }, {
        0: seq_dim
    }]
    wrapped = wrapper_factory(model, num_layers)
    wrapped.eval()
    return OnnxSpec(wrapped, args, names, outputs, shapes)


class Eagle3DraftModel(nn.Module):
    """EAGLE3 draft model for speculative decoding.

    Module tree (matches checkpoint keys after remapping):
        fc               Linear(target_hidden * 3, hidden)
        layers.0         Eagle3DecoderLayer  (checkpoint: ``midlayer``)
        norm             RMSNorm
        lm_head          Linear(hidden, draft_vocab_size)
        d2t              buffer [draft_vocab_size] int32

    Note: ``embed_tokens`` is not used — the draft model receives
    ``inputs_embeds`` from the C++ runtime (via the base model's embedding).
    The C++ builder already skips ``embedding.safetensors`` for draft models.
    """

    match_fp32_elementwise_initializers = True

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        target_hidden = config.eagle3_target_hidden_size
        draft_vocab_size = config.draft_vocab_size or config.vocab_size

        self.fc = make_linear(config,
                              target_hidden * config.eagle3_num_target_layers,
                              hidden_size,
                              module_name="fc")

        self.layers = nn.ModuleList([
            Eagle3DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        norm_cls = Gemma4RMSNorm if _is_gemma4_model_type(
            config.model_type) else RMSNorm
        self.norm = norm_cls(hidden_size, config.rms_norm_eps)
        # Always pass module_name="lm_head" so that the excluded list and
        # tie_word_embeddings overrides work correctly (both force FP16).
        self.lm_head = make_linear(config,
                                   hidden_size,
                                   draft_vocab_size,
                                   bias=False,
                                   module_name="lm_head")

        d2t = (torch.arange(draft_vocab_size, dtype=torch.int32)
               if draft_vocab_size == config.vocab_size else torch.zeros(
                   draft_vocab_size, dtype=torch.int32))
        self.register_buffer("d2t", d2t)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        last_token_ids: torch.Tensor,
        hidden_states_from_base: torch.Tensor,
        hidden_states_from_draft: torch.Tensor,
        attention_pos_id: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass.

        Returns:
            ``(logits, hidden_states, present_key_values)``
        """
        # Fusion: project base hidden states and add draft hidden states
        hidden_states = self.fc(hidden_states_from_base)
        hidden_states = hidden_states.to(torch.float16)
        hidden_states_from_draft = hidden_states_from_draft.to(torch.float16)
        hidden_states = hidden_states_from_draft + hidden_states

        present_key_values: List[torch.Tensor] = []

        for idx, layer in enumerate(self.layers):
            hidden_states, present_kv = layer(
                hidden_states,
                inputs_embeds,
                past_key_values[idx],
                rope_rotary_cos_sin,
                context_lengths,
                kvcache_start_index,
                kv_page_table,
                attention_mask,
                attention_pos_id,
            )
            present_key_values.append(present_kv)

        # Select hidden states for specified token positions
        hidden_states = torch.ops.trt.gather_nd(hidden_states, last_token_ids)
        hidden_states_normed = self.norm(hidden_states)
        logits = self.lm_head(hidden_states_normed).to(torch.float32)
        final_logit_softcapping = getattr(self.config,
                                          "final_logit_softcapping", None)
        if final_logit_softcapping is not None:
            logits = torch.tanh(
                logits / final_logit_softcapping) * final_logit_softcapping
        logits = F.log_softmax(logits, dim=-1)

        return logits, hidden_states, tuple(present_key_values)

    def forward_ragged(self, inputs_embeds: torch.Tensor,
                       past_key_values: Tuple[torch.Tensor, ...],
                       logits_indices: torch.Tensor,
                       hidden_states_from_base: torch.Tensor,
                       hidden_states_from_draft: torch.Tensor, **metadata):
        hidden_states = (hidden_states_from_draft.to(torch.float16) +
                         self.fc(hidden_states_from_base).to(torch.float16))
        present = []
        for idx, layer in enumerate(self.layers):
            hidden_states, present_kv = layer.forward_ragged(
                hidden_states, inputs_embeds, past_key_values[idx], **metadata)
            present.append(present_kv)
        selected = torch.index_select(hidden_states, 0, logits_indices)
        logits = self.lm_head(self.norm(selected)).to(torch.float32)
        cap = getattr(self.config, "final_logit_softcapping", None)
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return F.log_softmax(logits, dim=-1), selected, tuple(present)

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def onnx_export_spec(self) -> OnnxSpec:
        """Return all model-specific parameters needed for ONNX export."""
        return self._ragged_onnx_export_spec()

    def _ragged_onnx_export_spec(self) -> OnnxSpec:
        return _draft_ragged_export_spec(
            self, self.config.num_hidden_layers,
            self.config.eagle3_target_hidden_size *
            self.config.eagle3_num_target_layers,
            _make_flat_wrapper_eagle3_ragged)
