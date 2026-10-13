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
"""Qwen3.5 dense MTP draft model."""

import itertools
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import ModelConfig
from ..default.modeling_default import OnnxSpec
from ..linear import is_int4_linear, make_linear
from ..ops import KV_PAGE_SIZE, attention_plugin, qkv_concat
from .modeling_qwen3_5_text import MLP, GatedAttention, Qwen3_5RMSNorm

__all__ = ["Qwen3_5MtpDraftModel", "Qwen3_5MtpDecoderLayer"]

_BATCH_SIZE = 2
_SEQ_LEN = 2
_PAST_LEN = 1
_MAX_POS = 4096


class Qwen3_5MtpDecoderLayer(nn.Module):
    """Single Qwen3.5 full-attention-only decoder block for MTP draft."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size,
                                              config.rms_norm_eps)
        self.self_attn = GatedAttention(config, layer_idx=layer_idx)
        self._uses_int4_qkv = any(
            is_int4_linear(proj)
            for proj in (self.self_attn.q_proj, self.self_attn.k_proj,
                         self.self_attn.v_proj))
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size,
                                                       config.rms_norm_eps)
        self.mlp = MLP(config, layer_idx=layer_idx)

    def _forward_attention(
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
        attn = self.self_attn

        q_output = attn.q_proj(hidden_states)
        q_output = q_output.view(batch_size, seq_len, attn.num_heads,
                                 attn.head_dim * 2)
        query_states, gate_states = q_output.chunk(2, dim=-1)

        key_states = attn.k_proj(hidden_states)
        value_states = attn.v_proj(hidden_states)

        query_states = attn.q_norm(query_states).reshape(
            batch_size, seq_len, attn.num_heads * attn.head_dim)
        key_states = attn.k_norm(
            key_states.reshape(batch_size, seq_len, attn.num_kv_heads,
                               attn.head_dim)).reshape(
                                   batch_size, seq_len,
                                   attn.num_kv_heads * attn.head_dim)

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
            num_q_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            head_size=attn.head_dim,
            sliding_window_size=attn.sliding_window_size,
            enable_tree_attention=True,
            enable_fp8_kv_cache=attn.enable_fp8_kv_cache,
            attention_scale=attn.attention_scale,
            enable_context_mask_selector=False,
            enable_vision_block_attention=False,
            skip_softmax_scale_factor=0.0,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
            qkv_scales=getattr(attn, "_qkv_scales_float", [1.0, 1.0, 1.0]),
        )
        attn_output = attn_output * torch.sigmoid(gate_states)
        attn_output = attn_output.reshape(batch_size, seq_len,
                                          attn.num_heads * attn.head_dim)
        return attn.o_proj(attn_output), present_key_value

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
        residual = hidden_states
        normed_hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_key_value = self._forward_attention(
            normed_hidden_states,
            past_key_value,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            kv_page_table,
            attention_mask,
            attention_pos_id,
        )
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

    def forward_ragged(self, hidden_states: torch.Tensor,
                       past_key_value: torch.Tensor, **metadata):
        residual = hidden_states
        attn_output, present_key_value = self.self_attn.forward_ragged(
            self.input_layernorm(hidden_states),
            past_key_value,
            metadata["rope_rotary_cos_sin"],
            metadata["positions"],
            metadata["query_start_offsets"],
            metadata["query_lengths"],
            metadata["past_lengths"],
            metadata["attention_sequence_lengths"],
            metadata["state_indices"],
            metadata["execution_phase_marker"],
            metadata["context_sequence_count_carrier"],
            metadata["kv_page_table"],
            attention_position_ids=metadata["attention_position_ids"],
            packed_attention_mask=metadata["packed_attention_mask"],
            tree_parent_ids=metadata["tree_parent_ids"],
            tree_depths=metadata["tree_depths"],
            valid_tree_counts=metadata["valid_tree_counts"])
        hidden_states = residual + attn_output
        return (hidden_states +
                self.mlp(self.post_attention_layernorm(hidden_states)),
                present_key_value)


def _make_flat_wrapper_qwen3_5_mtp_ragged(model: nn.Module,
                                          num_layers: int) -> nn.Module:
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
        f"        inputs_embeds, {kv}, rope_rotary_cos_sin, positions, "
        f"query_start_offsets, query_lengths, past_lengths, "
        f"attention_sequence_lengths, state_indices, "
        f"execution_phase_marker, context_sequence_count_carrier, "
        f"kv_page_table, logits_indices, "
        f"hidden_states_input, hidden_states_from_draft, "
        f"attention_position_ids, packed_attention_mask, tree_parent_ids, "
        f"tree_depths, valid_tree_counts)\n"
        f"    return (logits, hidden_states) + tuple(present)\n")
    globs: dict = {}
    exec("def _forward(self, {}):\n{}".format(", ".join(names), body),
         globs)  # noqa: S102

    class _Wrapper(nn.Module):

        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self._model = m

    _Wrapper.forward = globs["_forward"]
    return _Wrapper(model)


class Qwen3_5MtpDraftModel(nn.Module):
    """Qwen3.5 dense MTP draft model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.num_hidden_layers != 1:
            raise ValueError("Qwen3.5 dense MTP draft currently requires "
                             "num_hidden_layers == 1")

        self.config = config
        hidden_size = config.hidden_size

        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(hidden_size,
                                                    config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(hidden_size,
                                                 config.rms_norm_eps)
        self.fc = make_linear(config,
                              hidden_size * 2,
                              hidden_size,
                              bias=False,
                              module_name="fc")
        self.layers = nn.ModuleList([self._make_decoder_layer(config)])
        self.norm = Qwen3_5RMSNorm(hidden_size, config.rms_norm_eps)
        self.lm_head = make_linear(config,
                                   hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")

    def _make_decoder_layer(self, config: ModelConfig) -> nn.Module:
        """Decoder-layer factory; subclasses override to swap the layer type."""
        return Qwen3_5MtpDecoderLayer(config, layer_idx=0)

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
        # Merge base and draft hidden states via Add.  The runtime always
        # zeroes exactly one of the two inputs so Add acts as a multiplexer:
        #   - prefill / accept-token: base is real, draft is zero
        #   - draft proposal steps:   draft is real, base is zero
        hidden_states_input = hidden_states_from_base + hidden_states_from_draft
        normed_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        normed_hidden_states = self.pre_fc_norm_hidden(hidden_states_input)
        fused_hidden_states = self.fc(
            torch.cat((normed_embeds, normed_hidden_states), dim=-1))

        present_key_values: List[torch.Tensor] = []
        hidden_states = fused_hidden_states
        for idx, layer in enumerate(self.layers):
            hidden_states, present_key_value = layer(
                hidden_states,
                past_key_values[idx],
                rope_rotary_cos_sin,
                context_lengths,
                kvcache_start_index,
                kv_page_table,
                attention_mask,
                attention_pos_id,
            )
            present_key_values.append(present_key_value)

        # Select hidden states for specified token positions
        hidden_states = torch.ops.trt.gather_nd(hidden_states, last_token_ids)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states).to(torch.float32)
        logits = F.log_softmax(logits, dim=-1)

        return logits, hidden_states, tuple(present_key_values)

    def forward_ragged(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
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
        logits_indices: torch.Tensor,
        hidden_states_input: torch.Tensor,
        hidden_states_from_draft: torch.Tensor,
        attention_position_ids: torch.Tensor,
        packed_attention_mask: torch.Tensor,
        tree_parent_ids: torch.Tensor,
        tree_depths: torch.Tensor,
        valid_tree_counts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        hidden_states = self.fc(
            torch.cat((self.pre_fc_norm_embedding(inputs_embeds),
                       self.pre_fc_norm_hidden(hidden_states_input +
                                               hidden_states_from_draft)),
                      dim=-1))
        metadata = {
            "rope_rotary_cos_sin": rope_rotary_cos_sin,
            "positions": positions,
            "query_start_offsets": query_start_offsets,
            "query_lengths": query_lengths,
            "past_lengths": past_lengths,
            "attention_sequence_lengths": attention_sequence_lengths,
            "state_indices": state_indices,
            "execution_phase_marker": execution_phase_marker,
            "context_sequence_count_carrier": context_sequence_count_carrier,
            "kv_page_table": kv_page_table,
            "attention_position_ids": attention_position_ids,
            "packed_attention_mask": packed_attention_mask,
            "tree_parent_ids": tree_parent_ids,
            "tree_depths": tree_depths,
            "valid_tree_counts": valid_tree_counts,
        }
        present_key_values = []
        for idx, layer in enumerate(self.layers):
            hidden_states, present = layer.forward_ragged(
                hidden_states, past_key_values[idx], **metadata)
            present_key_values.append(present)
        selected_hidden = self.norm(
            torch.index_select(hidden_states, 0, logits_indices))
        logits = F.log_softmax(self.lm_head(selected_hidden).to(torch.float32),
                               dim=-1)
        return logits, selected_hidden, tuple(present_key_values)

    def onnx_export_spec(self) -> OnnxSpec:
        """Return all model-specific parameters needed for ONNX export."""
        return self._ragged_onnx_export_spec()

    def _ragged_onnx_export_spec(self) -> OnnxSpec:
        config = self.config
        num_layers = len(self.layers)
        device = next(itertools.chain(self.parameters(),
                                      self.buffers())).device
        n, q = 2, 64
        tokens = n * q
        dtype = torch.float16
        kv_dtype = (torch.float8_e4m3fn
                    if config.quant.kv_cache_quant == "fp8" else dtype)
        inputs_embeds = torch.zeros(tokens,
                                    config.hidden_size,
                                    dtype=dtype,
                                    device=device)
        past_key_values = tuple(
            torch.zeros(2,
                        2,
                        KV_PAGE_SIZE,
                        config.num_key_value_heads,
                        config.head_dim,
                        dtype=kv_dtype,
                        device=device) for _ in range(num_layers))
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        rope = torch.zeros(tokens,
                           rotary_dim,
                           dtype=torch.float32,
                           device=device)
        positions = torch.arange(q, dtype=torch.int32, device=device).repeat(n)
        offsets = torch.arange(0,
                               tokens + 1,
                               q,
                               dtype=torch.int32,
                               device=device)
        lengths = torch.full((n, ), q, dtype=torch.int32, device=device)
        past = torch.zeros(n, dtype=torch.int32, device=device)
        state_indices = torch.arange(n, dtype=torch.int32, device=device)
        phase = torch.zeros(2, dtype=torch.int32, device=device)
        context_count = torch.empty(n, dtype=torch.int32, device=device)
        page_table = torch.zeros(n, 2, 2, dtype=torch.int32, device=device)
        logits_indices = offsets[1:].to(torch.int64) - 1
        base_hidden = torch.zeros(tokens,
                                  config.hidden_size,
                                  dtype=dtype,
                                  device=device)
        draft_hidden = torch.zeros_like(base_hidden)
        attention_positions = positions.clone()
        attention_mask = torch.zeros(tokens, (q + 31) // 32,
                                     dtype=torch.int32,
                                     device=device)
        parents = torch.full((tokens, ), -1, dtype=torch.int32, device=device)
        depths = torch.zeros(tokens, dtype=torch.int32, device=device)
        counts = lengths.clone()
        args = (inputs_embeds,
                *past_key_values, rope, positions, offsets, lengths, past,
                lengths.clone(), state_indices, phase, context_count,
                page_table, logits_indices, base_hidden, draft_hidden,
                attention_positions, attention_mask, parents, depths, counts)
        input_names = (
            ["inputs_embeds"] +
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
        output_names = (["logits", "hidden_states"] +
                        [f"present_key_values_{i}" for i in range(num_layers)])
        token_dim = torch.export.Dim("physical_tokens", min=1, max=8_388_608)
        seq_dim = torch.export.Dim("num_sequences", min=1, max=256)
        context_seq_dim = torch.export.Dim("num_context_sequences",
                                           min=0,
                                           max=256)
        pages = torch.export.Dim("num_pages", min=1, max=1048576)
        max_pages = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
        phase_dim = torch.export.Dim("execution_phase_extent", min=1, max=8)
        selected_dim = torch.export.Dim("selected_rows", min=1, max=8_388_608)
        packed_mask_width = torch.export.Dim("packed_mask_width",
                                             min=1,
                                             max=64)
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
        wrapped = _make_flat_wrapper_qwen3_5_mtp_ragged(self, num_layers)
        wrapped.eval()
        return OnnxSpec(wrapped, args, input_names, output_names, shapes)
