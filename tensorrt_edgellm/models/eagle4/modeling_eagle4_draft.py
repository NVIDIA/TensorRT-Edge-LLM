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
"""
EAGLE-4 draft model for speculative decoding.

An EAGLE-4 drafter is an EAGLE-3 style single-layer draft with ``fc_norm``,
``norm_output`` or a low-rank candidate head. It runs in the EAGLE3 runtime
mode, so the graph keeps the EAGLE3 draft I/O except for the head outputs: a
candidate head emits ``spec_proposal_support_ids`` and
``spec_proposal_support_log_probs`` in place of ``logits``.

Checkpoint layout
-----------------
    fc.weight                        [hidden, target_hidden * 3]
    fc_norm.{0,1,2}.weight           [target_hidden]  (``fc_norm``)
    layers.0.hidden_norm.weight      [hidden]
    layers.0.input_layernorm.weight  [hidden]
    layers.0.self_attn.{q,k,v,o}_proj.weight
    layers.0.mlp.{gate,up,down}_proj.weight
    layers.0.post_attention_layernorm.weight
    norm.weight                      [hidden]
    lm_head.weight                   [draft_vocab_size, hidden]
    d2t                              [draft_vocab_size]  (dense head, optional)
    candidate head factors           [rank, hidden] and [vocab_size, rank]

``_eagle4_key_remap`` in ``model.py`` maps ``midlayer`` to ``layers.0``
and loads the candidate head factors into ``candidate_head.down_proj`` and
``candidate_head.up_proj``; ``embed_tokens`` is skipped because the draft reads
the base embedding.
"""

import itertools
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import QUANT_FP16, ModelConfig, module_quant_type
from ..candidate_head import LowRankCandidateHead, fp16_head_weight
from ..default.modeling_default import OnnxSpec, RMSNorm
from ..linear import make_linear, requires_qkv_concat
from ..ops import KV_PAGE_SIZE, attention_plugin, qkv_concat

__all__ = ["Eagle4DraftModel"]


def _layer_uses_sliding_attention(config: ModelConfig, layer_idx: int) -> bool:
    """Whether *layer_idx* runs sliding-window attention.

    Falls back to the config-level window when the checkpoint ships no
    per-layer labels, so drafts that only set ``sliding_window`` still apply it.
    """
    if config.sliding_window_size <= 0:
        return False
    labels = config.raw_layer_types or config.attention_layer_types
    if not labels:
        return True
    if layer_idx >= len(labels):
        return False
    return labels[layer_idx] == "sliding_attention"


class Eagle4Attention(nn.Module):
    """GQA tree attention over the concatenated ``[embeds, hidden]`` input (``2 * hidden_size``)."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        qkv_in_features = hidden_size * 2
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.sliding_window_size = (config.sliding_window_size
                                    if _layer_uses_sliding_attention(
                                        config, layer_idx) else -1)
        self.attention_scale = config.attention_scaling

        self.q_proj = make_linear(config,
                                  qkv_in_features,
                                  self.num_heads * self.head_dim,
                                  bias=config.attention_bias)
        self.k_proj = make_linear(config,
                                  qkv_in_features,
                                  self.num_kv_heads * self.head_dim,
                                  bias=config.attention_bias)
        self.v_proj = make_linear(config,
                                  qkv_in_features,
                                  self.num_kv_heads * self.head_dim,
                                  bias=config.attention_bias)
        self._uses_qkv_concat = requires_qkv_concat(self.q_proj, self.k_proj,
                                                    self.v_proj)
        self.o_proj = make_linear(config, self.num_heads * self.head_dim,
                                  hidden_size)
        if config.has_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward_ragged(self, hidden_states: torch.Tensor,
                       past_key_value: torch.Tensor, **m):
        tokens = hidden_states.shape[0]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        if self.q_norm is not None:
            query_states = self.q_norm(
                query_states.reshape(tokens, self.num_heads,
                                     self.head_dim)).reshape(tokens, -1)
        if self.k_norm is not None:
            key_states = self.k_norm(
                key_states.reshape(tokens, self.num_kv_heads,
                                   self.head_dim)).reshape(tokens, -1)
        qkv = (qkv_concat(query_states, key_states, value_states)
               if self._uses_qkv_concat else torch.cat(
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
            skip_softmax_prefill_input=0,
            attention_mask=m["packed_attention_mask"],
            attention_pos_id=m["attention_position_ids"],
            qkv_scales=[1.0, 1.0, 1.0],
            query_start_offsets=m["query_start_offsets"],
            attention_sequence_lengths=m["attention_sequence_lengths"],
            execution_phase_marker=m["execution_phase_marker"],
            context_sequence_count_carrier=m["context_sequence_count_carrier"])
        return self.o_proj(output.reshape(tokens, -1)), present


class Eagle4MLP(nn.Module):
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


class Eagle4DecoderLayer(nn.Module):
    """Normalises ``hidden_states`` and ``inputs_embeds`` separately and attends over their concatenation."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Eagle4Attention(config, layer_idx=layer_idx)
        self.mlp = Eagle4MLP(config)
        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps)

    def forward_ragged(self, hidden_states: torch.Tensor,
                       inputs_embeds: torch.Tensor,
                       past_key_value: torch.Tensor, **metadata):
        residual = hidden_states
        concat_hidden = torch.cat((self.input_layernorm(inputs_embeds),
                                   self.hidden_norm(hidden_states)),
                                  dim=-1)
        attention, present = self.self_attn.forward_ragged(
            concat_hidden, past_key_value, **metadata)
        hidden_states = residual + attention
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states))
        return hidden_states, present


_INPUT_NAMES_AFTER_KV = [
    "rope_rotary_cos_sin", "positions", "query_start_offsets", "query_lengths",
    "past_lengths", "attention_sequence_lengths", "state_indices",
    "execution_phase_marker", "context_sequence_count_carrier",
    "kv_page_table", "logits_indices", "hidden_states_input",
    "hidden_states_from_draft", "attention_position_ids",
    "packed_attention_mask", "tree_parent_ids", "tree_depths",
    "valid_tree_counts"
]
# Inputs that forward_ragged takes positionally; the rest are attention metadata.
_POSITIONAL_INPUTS = ("logits_indices", "hidden_states_input",
                      "hidden_states_from_draft")


def _input_names(num_layers: int):
    return (["inputs_embeds"] +
            [f"past_key_values_{i}"
             for i in range(num_layers)] + _INPUT_NAMES_AFTER_KV)


def _make_flat_wrapper(model: nn.Module, num_layers: int) -> nn.Module:
    """Wrap ``forward_ragged`` in a module whose forward takes every graph input by name."""
    names = _input_names(num_layers)
    kv = "({},)".format(", ".join(f"past_key_values_{i}"
                                  for i in range(num_layers)))
    metadata = ", ".join(f"{name}={name}" for name in _INPUT_NAMES_AFTER_KV
                         if name not in _POSITIONAL_INPUTS)
    body = (
        f"    head_outputs, hidden_states, present = self._model.forward_ragged(\n"
        f"        inputs_embeds, {kv}, logits_indices, hidden_states_input, "
        f"hidden_states_from_draft, {metadata})\n"
        f"    return tuple(head_outputs) + (hidden_states,) + tuple(present)\n"
    )
    globs: dict = {}
    exec("def _forward(self, {}):\n{}".format(", ".join(names), body),
         globs)  # noqa: S102

    class _Wrapper(nn.Module):

        def __init__(self, m):
            super().__init__()
            self._model = m

    _Wrapper.forward = globs["_forward"]
    return _Wrapper(model)


def _check_draft_precision(config: ModelConfig) -> None:
    """Reject checkpoint precisions that the candidate-head draft graph cannot represent."""
    if config.draft_candidate_count <= 0:
        return
    if config.quant.kv_cache_quant == "fp8":
        raise ValueError(
            "The candidate-head draft graph writes an FP16 KV cache, but this "
            "draft checkpoint quantizes its KV cache to FP8 "
            "(kv_cache_quant_algo FP8). Quantize the draft again without "
            "--kv-cache-quantization.")
    lm_head_quant = module_quant_type("lm_head", config)
    if lm_head_quant != QUANT_FP16:
        raise ValueError(
            "The low-rank candidate head gathers fp16 lm_head rows, but this "
            f"draft checkpoint stores lm_head as {lm_head_quant}. Quantize "
            "the draft with lm_head unquantized, or export it with "
            "--draft-dense-head.")


class Eagle4DraftModel(nn.Module):
    """EAGLE-4 draft model for speculative decoding.

    Module tree (matches checkpoint keys after remapping):
        fc               Linear(target_hidden * 3, hidden)
        fc_norm.{0,1,2}  RMSNorm(target_hidden)  (``eagle_fc_norm``)
        layers.0         Eagle4DecoderLayer
        norm             RMSNorm
        lm_head          Linear(hidden, draft_vocab_size)
        candidate_head   LowRankCandidateHead  (``draft_candidate_count > 0``)
        d2t              buffer [draft_vocab_size] int32  (dense head only)

    The draft receives ``inputs_embeds`` from the runtime (the base model's
    embedding), so it has no ``embed_tokens``.
    """

    match_fp32_elementwise_initializers = True

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        _check_draft_precision(config)
        self.config = config
        hidden_size = config.hidden_size
        target_hidden = config.eagle3_target_hidden_size
        draft_vocab_size = config.draft_vocab_size or config.vocab_size

        self.fc = make_linear(config,
                              target_hidden * config.eagle3_num_target_layers,
                              hidden_size,
                              module_name="fc")
        self.fc_norm = (nn.ModuleList([
            RMSNorm(target_hidden, config.rms_norm_eps)
            for _ in range(config.eagle3_num_target_layers)
        ]) if config.eagle_fc_norm else None)
        self.layers = nn.ModuleList([
            Eagle4DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size, config.rms_norm_eps)
        # module_name="lm_head" applies the excluded list and the
        # tie_word_embeddings override, which both force FP16.
        self.lm_head = make_linear(config,
                                   hidden_size,
                                   draft_vocab_size,
                                   bias=False,
                                   module_name="lm_head")

        if config.draft_candidate_count > 0:
            self.candidate_head = LowRankCandidateHead(config)
        else:
            self.candidate_head = None
            d2t = (torch.arange(draft_vocab_size, dtype=torch.int32)
                   if draft_vocab_size == config.vocab_size else torch.zeros(
                       draft_vocab_size, dtype=torch.int32))
            self.register_buffer("d2t", d2t)

    def _fuse_target_hidden(
            self, hidden_states_from_base: torch.Tensor) -> torch.Tensor:
        if self.fc_norm is not None:
            chunks = hidden_states_from_base.split(
                self.config.eagle3_target_hidden_size, dim=-1)
            hidden_states_from_base = torch.cat(
                [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks)],
                dim=-1)
        return self.fc(hidden_states_from_base).to(torch.float16)

    def _draft_head(
        self, selected: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """Return ``(head_outputs, fed-back hidden)`` for the selected rows.

        ``head_outputs`` is ``(log-probs [R, V],)`` for the dense head and
        ``(support_ids, support_log_probs)`` for the candidate head.
        """
        nrm = self.norm(selected)
        feedback = nrm if self.config.eagle_norm_output else selected
        if self.candidate_head is not None:
            return self.candidate_head(nrm, self.lm_head.weight), feedback
        logits = self.lm_head(nrm).to(torch.float32)
        cap = self.config.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return (F.log_softmax(logits, dim=-1), ), feedback

    def forward_ragged(self, inputs_embeds: torch.Tensor,
                       past_key_values: Tuple[torch.Tensor, ...],
                       logits_indices: torch.Tensor,
                       hidden_states_from_base: torch.Tensor,
                       hidden_states_from_draft: torch.Tensor, **metadata):
        """Return ``(head_outputs, fed-back hidden, present_key_values)``."""
        hidden_states = (hidden_states_from_draft.to(torch.float16) +
                         self._fuse_target_hidden(hidden_states_from_base))
        present = []
        for idx, layer in enumerate(self.layers):
            hidden_states, present_kv = layer.forward_ragged(
                hidden_states, inputs_embeds, past_key_values[idx], **metadata)
            present.append(present_kv)
        selected = torch.index_select(hidden_states, 0, logits_indices)
        head_outputs, feedback = self._draft_head(selected)
        return head_outputs, feedback, tuple(present)

    def onnx_export_spec(self) -> OnnxSpec:
        """Return all model-specific parameters needed for ONNX export."""
        if self.candidate_head is not None:
            self._cast_candidate_head_weights()
        return self._ragged_export_spec()

    def _cast_candidate_head_weights(self) -> None:
        head_modules = {
            "candidate_head.down_proj.weight": self.candidate_head.down_proj,
            "candidate_head.up_proj.weight": self.candidate_head.up_proj,
            "lm_head.weight": self.lm_head,
        }
        for name, module in head_modules.items():
            module.weight.data = fp16_head_weight(name, module.weight.data)

    def _ragged_export_spec(self) -> OnnxSpec:
        config = self.config
        num_layers = config.num_hidden_layers
        device = next(itertools.chain(self.parameters(),
                                      self.buffers())).device
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
        state = torch.arange(n, dtype=torch.int32, device=device)
        phase = torch.zeros(2, dtype=torch.int32, device=device)
        context_count = torch.empty(n, dtype=torch.int32, device=device)
        page_table = torch.zeros(n, 2, 2, dtype=torch.int32, device=device)
        selected = offsets[1:].to(torch.int64) - 1
        target = torch.zeros(tokens,
                             config.eagle3_target_hidden_size *
                             config.eagle3_num_target_layers,
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
                lengths.clone(), state, phase,
                context_count, page_table, selected, target, draft,
                positions.clone(), mask, parents, depths, lengths.clone())
        head_outputs = (["logits"] if config.draft_candidate_count <= 0 else [
            "spec_proposal_support_ids", "spec_proposal_support_log_probs"
        ])
        outputs = head_outputs + ["hidden_states"] + [
            f"present_key_values_{i}" for i in range(num_layers)
        ]
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
        wrapped = _make_flat_wrapper(self, num_layers)
        wrapped.eval()
        return OnnxSpec(wrapped, args, _input_names(num_layers), outputs,
                        shapes)
