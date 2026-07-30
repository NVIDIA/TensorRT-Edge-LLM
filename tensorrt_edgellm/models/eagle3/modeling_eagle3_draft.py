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

from ...config import ModelConfig
from ..default.modeling_default import OnnxSpec, RMSNorm
from ..linear import make_linear
from ..ops import KV_PAGE_SIZE, attention_plugin

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
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = config.head_dim
        hidden_size = config.hidden_size
        qkv_in_features = hidden_size * 2

        self.layer_idx = layer_idx
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.attention_scale = config.attention_scaling
        self.sliding_window_size = -1

        self.q_proj = make_linear(config,
                                  qkv_in_features,
                                  num_attention_heads * head_dim,
                                  bias=config.attention_bias)
        self.k_proj = make_linear(config,
                                  qkv_in_features,
                                  num_key_value_heads * head_dim,
                                  bias=config.attention_bias)
        self.v_proj = make_linear(config,
                                  qkv_in_features,
                                  num_key_value_heads * head_dim,
                                  bias=config.attention_bias)
        self.o_proj = make_linear(config, num_attention_heads * head_dim,
                                  hidden_size)

        if config.has_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

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
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

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

        attn_output, present_key_value = attention_plugin(
            query_states,
            key_states,
            value_states,
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
            enable_vision_block_attention=False,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
            qkv_scales=[1.0, 1.0, 1.0],
        )
        attn_output = attn_output.reshape(batch_size, seq_len,
                                          self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present_key_value


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
        self.self_attn = Eagle3Attention(config, layer_idx=layer_idx)
        self.mlp = MLP(config)
        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps)

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
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value


# ---------------------------------------------------------------------------
# Eagle3 Draft Model
# ---------------------------------------------------------------------------


def _make_flat_wrapper_eagle3(model: nn.Module, Na: int) -> nn.Module:
    """Build a flat-signature wrapper for EAGLE3 draft ONNX export.

    Named parameters avoid the PyTorch 2.10 variadic-tuple export bug.
    """
    param_names: List[str] = (
        ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
            "rope_rotary_cos_sin", "context_lengths", "kvcache_start_index",
            "kv_page_table", "last_token_ids", "hidden_states_input",
            "hidden_states_from_draft", "attention_pos_id", "attention_mask"
        ])

    past_kv_tuple = "({},)".format(", ".join(
        f"past_key_values_{i}" for i in range(Na))) if Na else "()"

    body = (
        f"    logits, hidden_states, present_key_values = self._model(\n"
        f"        inputs_embeds, {past_kv_tuple}, rope_rotary_cos_sin, "
        f"context_lengths, kvcache_start_index, kv_page_table, "
        f"last_token_ids, "
        f"hidden_states_input, hidden_states_from_draft, "
        f"attention_pos_id, attention_mask)\n"
        f"    return (logits, hidden_states) + tuple(present_key_values)\n")

    src = "def _forward(self, {}):\n{}".format(", ".join(param_names), body)
    globs: dict = {}
    exec(src, globs)  # noqa: S102

    class _Wrapper(nn.Module):

        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self._model = m

    _Wrapper.forward = globs["_forward"]
    return _Wrapper(model)


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

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        target_hidden = config.eagle3_target_hidden_size
        draft_vocab_size = config.draft_vocab_size or config.vocab_size

        self.fc = make_linear(config,
                              target_hidden * 3,
                              hidden_size,
                              module_name="fc")

        self.layers = nn.ModuleList([
            Eagle3DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size, config.rms_norm_eps)
        # Always pass module_name="lm_head" so that the excluded list and
        # tie_word_embeddings overrides work correctly (both force FP16).
        self.lm_head = make_linear(config,
                                   hidden_size,
                                   draft_vocab_size,
                                   bias=False,
                                   module_name="lm_head")

        self.register_buffer("d2t",
                             torch.zeros(draft_vocab_size, dtype=torch.int32))

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
        logits = F.log_softmax(logits, dim=-1)

        return logits, hidden_states, tuple(present_key_values)

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def onnx_export_spec(self) -> OnnxSpec:
        """Return all model-specific parameters needed for ONNX export."""
        config = self.config
        Na = config.num_hidden_layers
        target_hidden = config.eagle3_target_hidden_size
        device = next(itertools.chain(self.parameters(),
                                      self.buffers())).device
        dtype16 = torch.float16
        batch_size, seq_len, past_len, max_pos = (_BATCH_SIZE, _SEQ_LEN,
                                                  _PAST_LEN, _MAX_POS)

        inputs_embeds = torch.zeros(batch_size,
                                    seq_len,
                                    config.hidden_size,
                                    dtype=dtype16,
                                    device=device)
        # Paged KV pool binding: [2, num_pages, KV_PAGE_SIZE, num_kv_heads, head_dim].
        past_key_values_list: List[torch.Tensor] = [
            torch.zeros(2,
                        1,
                        KV_PAGE_SIZE,
                        config.num_key_value_heads,
                        config.head_dim,
                        dtype=dtype16,
                        device=device) for _ in range(Na)
        ]
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        rope_rotary_cos_sin = torch.zeros(batch_size,
                                          max_pos,
                                          rotary_dim,
                                          dtype=torch.float32,
                                          device=device)
        context_lengths = torch.zeros(batch_size,
                                      dtype=torch.int32,
                                      device=device)
        kvcache_start_index = torch.zeros(batch_size,
                                          dtype=torch.int32,
                                          device=device)
        kv_page_table = torch.zeros(batch_size,
                                    2,
                                    1,
                                    dtype=torch.int32,
                                    device=device)
        last_token_ids = torch.zeros(batch_size,
                                     1,
                                     dtype=torch.int64,
                                     device=device)
        hidden_states_input = torch.zeros(batch_size,
                                          seq_len,
                                          target_hidden * 3,
                                          dtype=dtype16,
                                          device=device)
        hidden_states_from_draft = torch.zeros(batch_size,
                                               seq_len,
                                               config.hidden_size,
                                               dtype=dtype16,
                                               device=device)
        attention_pos_id = torch.zeros(batch_size,
                                       seq_len,
                                       dtype=torch.int32,
                                       device=device)
        attention_mask = torch.zeros(batch_size,
                                     seq_len,
                                     seq_len + past_len,
                                     dtype=torch.int32,
                                     device=device)

        args = (inputs_embeds, *past_key_values_list, rope_rotary_cos_sin,
                context_lengths, kvcache_start_index, kv_page_table,
                last_token_ids, hidden_states_input, hidden_states_from_draft,
                attention_pos_id, attention_mask)

        input_names = (
            ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
                "rope_rotary_cos_sin", "context_lengths",
                "kvcache_start_index", "kv_page_table", "last_token_ids",
                "hidden_states_input", "hidden_states_from_draft",
                "attention_pos_id", "attention_mask"
            ])
        output_names = (["logits", "hidden_states"] +
                        [f"present_key_values_{i}" for i in range(Na)])

        batch = torch.export.Dim("batch", min=1, max=256)
        seq = torch.export.Dim("seq_len", min=1, max=32768)
        pos = torch.export.Dim("max_pos", min=1, max=32768)
        rope_batch = torch.export.Dim("rope_batch", min=1, max=256)
        kv_batch = torch.export.Dim("kv_batch", min=1, max=256)
        page_batch = torch.export.Dim("page_batch", min=1, max=256)
        max_pages = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
        num_pages = torch.export.Dim("num_pages", min=1, max=1048576)
        eagle_seq = torch.export.Dim("eagle_seq_len", min=1, max=32768)
        mask_kv_len = torch.export.Dim("mask_kv_len", min=1, max=65536)
        num_selected = torch.export.Dim("num_selected", min=1, max=256)

        all_shapes: list = [{0: batch, 1: seq}]  # inputs_embeds
        for _ in range(Na):
            all_shapes.append({1:
                               num_pages})  # past_key_values_i (pool-shaped)
        all_shapes.append({0: rope_batch, 1: pos})  # rope_rotary_cos_sin
        all_shapes.append({0: batch})  # context_lengths
        all_shapes.append({0: kv_batch})  # kvcache_start_index
        all_shapes.append({0: page_batch, 2: max_pages})  # kv_page_table
        all_shapes.append({0: batch, 1: num_selected})  # last_token_ids
        all_shapes.append({0: batch, 1: seq})  # hidden_states_input
        all_shapes.append({0: batch, 1: seq})  # hidden_states_from_draft
        all_shapes.append({0: batch, 1: eagle_seq})  # attention_pos_id
        all_shapes.append({
            0: batch,
            1: eagle_seq,
            2: mask_kv_len
        })  # attention_mask

        wrapped = _make_flat_wrapper_eagle3(self, Na)
        wrapped.eval()

        return OnnxSpec(wrapped=wrapped,
                        args=args,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=all_shapes)
