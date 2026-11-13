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
"""Qwen4-Exp MTP draft model.

The draft carries its own copy of the multi-stream residual: the base hands it
the bus from before the final mixer, ``hc_count * hidden_size`` wide, and the
draft returns the same thing for its next step. The token embedding enters that
bus written to every stream with unit weight, which is why the two ``pre_fc``
norms are not symmetric: ``pre_fc_norm_embedding`` normalizes one stream and
``pre_fc_norm_hidden`` normalizes the whole bus at once.
"""

import itertools
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import ModelConfig
from ..default.modeling_default import OnnxSpec
from ..linear import make_linear
from ..ops import KV_PAGE_SIZE
from .modeling_qwen4_exp_text import (Qwen4ExpDecoderLayer,
                                      Qwen4ExpGatedResidual, Qwen4ExpRMSNorm)

__all__ = ["Qwen4ExpMtpDraftModel"]

_RAGGED_INPUTS = ("rope_rotary_cos_sin", "positions", "query_start_offsets",
                  "query_lengths", "past_lengths",
                  "attention_sequence_lengths", "state_indices",
                  "execution_phase_marker", "context_sequence_count_carrier",
                  "kv_page_table", "logits_indices", "hidden_states_input",
                  "hidden_states_from_draft", "attention_position_ids",
                  "packed_attention_mask", "tree_parent_ids", "tree_depths",
                  "valid_tree_counts", "spec_verify_layout")


def _make_flat_wrapper_ragged(model: nn.Module, num_layers: int) -> nn.Module:
    """Positional-argument wrapper ``torch.onnx.export`` traces."""
    names = (["inputs_embeds"] +
             [f"past_key_values_{i}"
              for i in range(num_layers)] + list(_RAGGED_INPUTS))
    kv = "({},)".format(", ".join(f"past_key_values_{i}"
                                  for i in range(num_layers)))
    body = (
        f"    logits, hidden_states, present = self._model.forward_ragged(\n"
        f"        inputs_embeds, {kv}, {', '.join(_RAGGED_INPUTS)})\n"
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


class Qwen4ExpMtpDraftModel(nn.Module):
    """Qwen4-Exp MTP draft: one decoder block over the multi-stream residual."""

    # The zero-centered norms multiply in float32; a folded ``1 + weight``
    # constant must not be downgraded to FP16 under that multiply.
    match_fp32_elementwise_initializers = True

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.num_hidden_layers != 1:
            raise ValueError("Qwen4-Exp MTP draft requires one predictor "
                             f"layer, got {config.num_hidden_layers}")
        self.config = config
        hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        self.hc_hidden_size = config.hc_count * hidden_size
        eps = config.rms_norm_eps
        self.pre_fc_norm_embedding = Qwen4ExpRMSNorm(hidden_size, eps=eps)
        self.pre_fc_norm_hidden = Qwen4ExpRMSNorm(self.hc_hidden_size, eps=eps)
        self.fc_embedding = make_linear(config,
                                        hidden_size,
                                        hidden_size,
                                        bias=False,
                                        module_name="fc_embedding")
        self.fc_hidden = make_linear(config,
                                     hidden_size,
                                     hidden_size,
                                     bias=False,
                                     module_name="fc_hidden")
        self.layers = nn.ModuleList([
            Qwen4ExpDecoderLayer(config,
                                 layer_idx=0,
                                 layer_type=config.layer_types[0])
        ])
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(
            config, "hyper_connection_mixer", use_combine=False)
        self.lm_head = make_linear(config,
                                   hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")

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
        spec_verify_layout: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        embeds = self.fc_embedding(self.pre_fc_norm_embedding(inputs_embeds))
        # The runtime zeroes whichever feedback is not in play this step.
        hyper_states = self.pre_fc_norm_hidden(hidden_states_input +
                                               hidden_states_from_draft)
        # One fc_hidden shared by every stream, applied stream by stream.
        hyper_states = self.fc_hidden(
            hyper_states.reshape(-1, self.config.hidden_size)).reshape(
                -1, self.hc_hidden_size)
        hyper_states = hyper_states + embeds.repeat(1, self.hc_count)
        ragged = {
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
            "spec_verify_layout": spec_verify_layout,
        }
        hyper_states, present_kv = self.layers[0].forward_attention(
            hyper_states, past_key_values[0], ragged)
        selected = torch.index_select(hyper_states, 0, logits_indices)
        logits = F.log_softmax(self.lm_head(
            self.hyper_connection_mixer(selected)).to(torch.float32),
                               dim=-1)
        return logits, selected, (present_kv, )

    def onnx_export_spec(self) -> OnnxSpec:
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
                        config.kv_pool_head_dim,
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
        # Both feedback bindings carry the multi-stream bus.
        base_hidden = torch.zeros(tokens,
                                  self.hc_hidden_size,
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
        verify_layout = torch.empty(2, dtype=torch.int32, device=device)
        args = (inputs_embeds, *past_key_values, rope, positions, offsets,
                lengths, past, lengths.clone(), state_indices, phase,
                context_count, page_table, logits_indices, base_hidden,
                draft_hidden, attention_positions, attention_mask, parents,
                depths, counts, verify_layout)
        input_names = (["inputs_embeds"] +
                       [f"past_key_values_{i}"
                        for i in range(num_layers)] + list(_RAGGED_INPUTS))
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
        }, {
            0: torch.export.Dim("spec_verify_layout_len", min=0, max=2)
        }]
        wrapped = _make_flat_wrapper_ragged(self, num_layers)
        wrapped.eval()
        return OnnxSpec(wrapped, args, input_names, output_names, shapes)
