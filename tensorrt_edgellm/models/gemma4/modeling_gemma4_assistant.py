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
"""Gemma4 paired MTP assistant decoder model."""

from __future__ import annotations

import itertools
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import ModelConfig
from ..default.modeling_default import OnnxSpec, RMSNorm
from ..linear import TPMode, make_linear
from ..ops import KV_PAGE_SIZE, attention_plugin
from .modeling_gemma4_text import Gemma4MLP, _rotary_dim_from_rope_config

__all__ = ["Gemma4AssistantForCausalLM", "Gemma4AssistantDecoderLayer"]

_DUMMY_BATCH_SIZE = 1
_DUMMY_SEQ_LEN = 1
_DUMMY_ROPE_CACHE_LEN = 4096


def _make_gemma4_assistant_flat_wrapper(model: nn.Module,
                                        num_layers: int) -> nn.Module:
    """Build the token-major Gemma4 assistant ONNX wrapper."""
    param_names: List[str] = (
        ["inputs_embeds", "hidden_states_input"] +
        [f"past_key_values_{i}" for i in range(num_layers)] + [
            "rope_rotary_cos_sin_sliding", "rope_rotary_cos_sin_full",
            "positions", "query_start_offsets", "query_lengths",
            "past_lengths", "attention_sequence_lengths", "state_indices",
            "execution_phase_marker", "context_sequence_count_carrier",
            "kv_page_table"
        ])

    past_kv_tuple = "({},)".format(", ".join(
        f"past_key_values_{i}"
        for i in range(num_layers))) if num_layers else "()"
    body = (
        f"    logits, hidden_states = self._model.forward_ragged(\n"
        f"        inputs_embeds, hidden_states_input, {past_kv_tuple},\n"
        f"        rope_rotary_cos_sin_sliding=rope_rotary_cos_sin_sliding,\n"
        f"        rope_rotary_cos_sin_full=rope_rotary_cos_sin_full,\n"
        f"        positions=positions, query_start_offsets=query_start_offsets,\n"
        f"        query_lengths=query_lengths,\n"
        f"        past_lengths=past_lengths,\n"
        f"        attention_sequence_lengths=attention_sequence_lengths,\n"
        f"        state_indices=state_indices, execution_phase_marker=execution_phase_marker,\n"
        f"        context_sequence_count_carrier=context_sequence_count_carrier,\n"
        f"        kv_page_table=kv_page_table)\n"
        f"    return logits, hidden_states\n")

    src = "def _forward(self, {}):\n{}".format(", ".join(param_names), body)
    globs: dict = {}
    exec(src, globs)  # noqa: S102

    class _Wrapper(nn.Module):

        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self._model = m

    _Wrapper.forward = globs["_forward"]
    return _Wrapper(model)


class Gemma4SharedKVAttention(nn.Module):
    """Q-only attention over target/base KV cache."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = self._head_dim_for_layer(config, layer_idx)
        self.sliding_window_size = config.sliding_window_size
        module_prefix = f"layers.{layer_idx}.self_attn"

        self.q_proj = make_linear(config,
                                  config.hidden_size,
                                  self.num_heads * self.head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{module_prefix}.q_proj",
                                  tp_mode=TPMode.COL)
        self.o_proj = make_linear(config,
                                  self.num_heads * self.head_dim,
                                  config.hidden_size,
                                  module_name=f"{module_prefix}.o_proj",
                                  tp_mode=TPMode.ROW)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.enable_fp8_kv_cache = config.quant.kv_cache_quant == "fp8"
        layer_type = self._layer_type_for_layer(config, layer_idx)
        self.attention_type = layer_type
        if layer_type == "full_attention":
            self.sliding_window_size = -1

        self.num_kv_heads = self._num_kv_heads_for_layer(config, layer_idx)
        self.attention_scale = config.attention_scaling

    @staticmethod
    def _layer_type_for_layer(config: ModelConfig, layer_idx: int) -> str:
        raw_layer_types = config.raw_layer_types or config.layer_types
        if layer_idx >= len(raw_layer_types):
            raise ValueError(
                "Gemma4 assistant layer_types must have one entry per layer; "
                f"missing layer {layer_idx}.")
        return raw_layer_types[layer_idx]

    @staticmethod
    def _head_dim_for_layer(config: ModelConfig, layer_idx: int) -> int:
        layer_type = Gemma4SharedKVAttention._layer_type_for_layer(
            config, layer_idx)
        if layer_type == "full_attention" and config.global_head_dim:
            return config.global_head_dim
        return config.head_dim

    @staticmethod
    def _num_kv_heads_for_layer(config: ModelConfig, layer_idx: int) -> int:
        layer_type = Gemma4SharedKVAttention._layer_type_for_layer(
            config, layer_idx)
        if (layer_type == "full_attention" and config.attention_k_eq_v
                and config.num_global_key_value_heads):
            return config.num_global_key_value_heads
        return config.num_key_value_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_past_key_value: torch.Tensor,
        kv_page_table: torch.Tensor,
        context_lengths: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        query_states = self.q_norm(
            query_states.reshape(batch_size, seq_len, self.num_heads,
                                 self.head_dim)).reshape(
                                     batch_size, seq_len,
                                     self.num_heads * self.head_dim)

        attention_mask = torch.ones(batch_size,
                                    seq_len,
                                    seq_len,
                                    dtype=torch.int32,
                                    device=query_states.device)
        # Match HF Gemma4 SinglePosition MTP: all assistant draft steps use
        # the last seen token position while reading the target shared KV states.
        frontier_pos_id = torch.clamp(context_lengths - 1, min=0)
        attention_pos_id = frontier_pos_id.reshape(batch_size, 1).expand(
            batch_size, seq_len)
        # Shared-KV layer: qkv carries Q only; K/V are read from the donated
        # target cache without being written.
        attn_output, _ = attention_plugin(
            query_states,
            target_past_key_value,
            context_lengths,
            rope_rotary_cos_sin,
            context_lengths,
            kv_page_table,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            sliding_window_size=self.sliding_window_size,
            enable_tree_attention=True,
            enable_fp8_kv_cache=self.enable_fp8_kv_cache,
            attention_scale=self.attention_scale,
            enable_context_mask_selector=False,
            enable_vision_block_attention=False,
            skip_softmax_scale_factor=0.0,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
            qkv_scales=[1.0, 1.0, 1.0],
            enable_kv_shared=1,
        )
        return self.o_proj(
            attn_output.reshape(batch_size, seq_len,
                                self.num_heads * self.head_dim))

    def forward_ragged(self, hidden_states: torch.Tensor,
                       target_past_key_value: torch.Tensor,
                       rope_rotary_cos_sin: torch.Tensor,
                       **metadata) -> torch.Tensor:
        physical_tokens = hidden_states.shape[0]
        query_states = self.q_norm(
            self.q_proj(hidden_states).reshape(
                physical_tokens, self.num_heads,
                self.head_dim)).reshape(physical_tokens,
                                        self.num_heads * self.head_dim)
        attn_output, _ = attention_plugin(
            query_states,
            target_past_key_value,
            metadata["query_lengths"],
            rope_rotary_cos_sin,
            metadata["past_lengths"],
            metadata["kv_page_table"],
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            sliding_window_size=self.sliding_window_size,
            enable_tree_attention=False,
            enable_fp8_kv_cache=self.enable_fp8_kv_cache,
            attention_scale=self.attention_scale,
            enable_context_mask_selector=False,
            enable_vision_block_attention=False,
            skip_softmax_scale_factor=0.0,
            qkv_scales=[1.0, 1.0, 1.0],
            enable_kv_shared=1,
            query_start_offsets=metadata["query_start_offsets"],
            attention_sequence_lengths=metadata["attention_sequence_lengths"],
            execution_phase_marker=metadata["execution_phase_marker"],
            context_sequence_count_carrier=metadata[
                "context_sequence_count_carrier"])
        return self.o_proj(
            attn_output.reshape(physical_tokens,
                                self.num_heads * self.head_dim))


class Gemma4AssistantDecoderLayer(nn.Module):
    """Gemma4 assistant decoder layer with shared target-KV attention."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Gemma4SharedKVAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size,
                                                 config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size,
                                                  config.rms_norm_eps)
        self.mlp = Gemma4MLP(config, layer_idx=layer_idx)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_past_key_value: torch.Tensor,
        kv_page_table: torch.Tensor,
        context_lengths: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        attn_output = self.self_attn(self.input_layernorm(hidden_states),
                                     target_past_key_value, kv_page_table,
                                     context_lengths, rope_rotary_cos_sin)
        hidden_states = residual + self.post_attention_layernorm(attn_output)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return ((residual + hidden_states) * self.layer_scalar).to(
            hidden_states.dtype)

    def forward_ragged(self, hidden_states: torch.Tensor,
                       target_past_key_value: torch.Tensor,
                       rope_rotary_cos_sin: torch.Tensor,
                       **metadata) -> torch.Tensor:
        residual = hidden_states
        attn_output = self.self_attn.forward_ragged(
            self.input_layernorm(hidden_states), target_past_key_value,
            rope_rotary_cos_sin, **metadata)
        hidden_states = residual + self.post_attention_layernorm(attn_output)
        residual = hidden_states
        hidden_states = self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(hidden_states)))
        return ((residual + hidden_states) * self.layer_scalar).to(
            torch.float16)


class Gemma4AssistantMaskedEmbedder(nn.Module):
    """HF-compatible ordered embedding logits for Gemma4 assistant."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.num_centroids <= 0:
            raise ValueError(
                "Gemma4 ordered embeddings require num_centroids > 0.")
        if config.centroid_intermediate_top_k <= 0:
            raise ValueError(
                "Gemma4 ordered embeddings require centroid_intermediate_top_k > 0."
            )
        if config.vocab_size % config.num_centroids != 0:
            raise ValueError(
                "Gemma4 ordered embeddings require vocab_size divisible by num_centroids."
            )
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_centroids = config.num_centroids
        self.centroid_intermediate_top_k = config.centroid_intermediate_top_k
        self.vocab_size_per_centroid = config.vocab_size // config.num_centroids
        self.centroids = make_linear(config,
                                     self.hidden_size,
                                     self.num_centroids,
                                     bias=False,
                                     module_name="masked_embedding.centroids")
        self.register_buffer("token_ordering",
                             torch.empty(self.vocab_size, dtype=torch.long))
        self.register_buffer("token_to_centroid",
                             torch.empty(self.vocab_size, dtype=torch.long))

    def prepare_for_export(self) -> None:
        token_ordering = self.token_ordering.detach().to(torch.long).cpu()
        cluster_ids = torch.arange(self.num_centroids,
                                   dtype=torch.long).repeat_interleave(
                                       self.vocab_size_per_centroid)
        token_to_centroid = torch.empty(self.vocab_size, dtype=torch.long)
        token_to_centroid[token_ordering] = cluster_ids
        self.token_to_centroid.copy_(
            token_to_centroid.to(self.token_to_centroid.device))

    def forward(self, hidden_states: torch.Tensor,
                lm_head_weight: torch.Tensor) -> torch.Tensor:
        full_logits = F.linear(hidden_states,
                               lm_head_weight.to(hidden_states.dtype))
        centroid_logits = self.centroids(hidden_states)
        _, top_k_indices = torch.topk(centroid_logits,
                                      k=self.centroid_intermediate_top_k,
                                      dim=-1)

        token_to_centroid = self.token_to_centroid.view(
            *([1] * (full_logits.ndim - 1)), self.vocab_size, 1)
        selected_centroids = top_k_indices.unsqueeze(-2)
        selected_mask = (token_to_centroid == selected_centroids).any(dim=-1)

        mask_value = torch.full_like(full_logits, -65504.0)
        return torch.where(selected_mask, full_logits, mask_value)


class Gemma4AssistantForCausalLM(nn.Module):
    """Gemma4 paired MTP assistant with feedback hidden output."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if not config.backbone_hidden_size:
            raise ValueError(
                "Gemma4 assistant requires backbone_hidden_size in config.json."
            )
        self.config = config
        self.pre_projection = make_linear(config,
                                          config.backbone_hidden_size * 2,
                                          config.hidden_size,
                                          bias=False,
                                          module_name="pre_projection")
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size,
                                               config.hidden_size)
        self.model.layers = nn.ModuleList([
            Gemma4AssistantDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.model.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_projection = make_linear(config,
                                           config.hidden_size,
                                           config.backbone_hidden_size,
                                           bias=False,
                                           module_name="post_projection")
        self.masked_embedding = (Gemma4AssistantMaskedEmbedder(config)
                                 if config.use_ordered_embeddings else None)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        hidden_states_input: torch.Tensor,
        context_lengths: torch.Tensor,
        kv_page_table: torch.Tensor,
        rope_rotary_cos_sin_sliding: torch.Tensor,
        rope_rotary_cos_sin_full: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.pre_projection(
            torch.cat((inputs_embeds, hidden_states_input),
                      dim=-1)).to(torch.float16)
        for layer_idx, layer in enumerate(self.model.layers):
            rope_rotary_cos_sin = (
                rope_rotary_cos_sin_full if layer.self_attn.attention_type
                == "full_attention" else rope_rotary_cos_sin_sliding)
            hidden_states = layer(hidden_states, past_key_values[layer_idx],
                                  kv_page_table, context_lengths,
                                  rope_rotary_cos_sin)
        hidden_states = self.model.norm(hidden_states).to(torch.float16)
        if self.masked_embedding is not None:
            logits = self.masked_embedding(hidden_states,
                                           self.model.embed_tokens.weight)
        else:
            logits = F.linear(
                hidden_states,
                self.model.embed_tokens.weight.to(hidden_states.dtype))
        logits = logits.to(torch.float32)
        feedback_hidden = self.post_projection(hidden_states)
        return logits[:, -1, :], feedback_hidden

    def forward_ragged(self, inputs_embeds: torch.Tensor,
                       hidden_states_input: torch.Tensor,
                       past_key_values: Tuple[torch.Tensor, ...],
                       rope_rotary_cos_sin_sliding: torch.Tensor,
                       rope_rotary_cos_sin_full: torch.Tensor, **metadata):
        hidden_states = self.pre_projection(
            torch.cat((inputs_embeds, hidden_states_input), dim=-1))
        for layer_idx, layer in enumerate(self.model.layers):
            rope = (rope_rotary_cos_sin_full if layer.self_attn.attention_type
                    == "full_attention" else rope_rotary_cos_sin_sliding)
            hidden_states = layer.forward_ragged(hidden_states,
                                                 past_key_values[layer_idx],
                                                 rope, **metadata)
        hidden_states = self.model.norm(hidden_states)
        if self.masked_embedding is not None:
            logits = self.masked_embedding(hidden_states,
                                           self.model.embed_tokens.weight)
        else:
            logits = F.linear(
                hidden_states,
                self.model.embed_tokens.weight.to(hidden_states.dtype))
        return logits.to(torch.float32), self.post_projection(hidden_states)

    def onnx_export_spec(self) -> OnnxSpec:
        """Return the Gemma4 assistant ONNX I/O contract."""
        config = self.config
        num_layers = len(self.model.layers)
        if self.masked_embedding is not None:
            self.masked_embedding.prepare_for_export()
        device = next(itertools.chain(self.parameters(),
                                      self.buffers())).device
        dtype16 = torch.float16
        batch_size, seq_len = _DUMMY_BATCH_SIZE, _DUMMY_SEQ_LEN
        physical_tokens = batch_size * seq_len

        inputs_embeds = torch.zeros(physical_tokens,
                                    config.backbone_hidden_size,
                                    dtype=dtype16,
                                    device=device)
        hidden_states_input = torch.zeros(physical_tokens,
                                          config.backbone_hidden_size,
                                          dtype=dtype16,
                                          device=device)
        sliding_rotary_dim = _rotary_dim_from_rope_config(
            config, config.sliding_rope_config, config.head_dim)
        full_rotary_dim = _rotary_dim_from_rope_config(
            config, config.full_rope_config, config.global_head_dim
            or config.head_dim)
        rope_rotary_cos_sin_sliding = torch.zeros(physical_tokens,
                                                  sliding_rotary_dim,
                                                  dtype=torch.float32,
                                                  device=device)
        rope_rotary_cos_sin_full = torch.zeros(physical_tokens,
                                               full_rotary_dim,
                                               dtype=torch.float32,
                                               device=device)
        kv_dtype = (torch.float8_e4m3fn
                    if config.quant.kv_cache_quant == "fp8" else dtype16)
        # Target paged KV pool binding: [2, num_pages, KV_PAGE_SIZE, num_kv_heads,
        # head_dim] — the assistant reads the TARGET model's pool, so the dummy
        # mirrors the target's per-layer pool shape.
        past_key_values_list: List[torch.Tensor] = [
            torch.zeros(2,
                        1,
                        KV_PAGE_SIZE,
                        Gemma4SharedKVAttention._num_kv_heads_for_layer(
                            config, layer_idx),
                        Gemma4SharedKVAttention._head_dim_for_layer(
                            config, layer_idx),
                        dtype=kv_dtype,
                        device=device) for layer_idx in range(num_layers)
        ]
        kv_page_table = torch.zeros(batch_size,
                                    2,
                                    1,
                                    dtype=torch.int32,
                                    device=device)

        positions = torch.zeros(physical_tokens,
                                dtype=torch.int32,
                                device=device)
        query_start_offsets = torch.arange(batch_size + 1,
                                           dtype=torch.int32,
                                           device=device)
        query_lengths = torch.ones(batch_size,
                                   dtype=torch.int32,
                                   device=device)
        past_lengths = torch.zeros(batch_size,
                                   dtype=torch.int32,
                                   device=device)
        attention_sequence_lengths = torch.ones(batch_size,
                                                dtype=torch.int32,
                                                device=device)
        state_indices = torch.arange(batch_size,
                                     dtype=torch.int32,
                                     device=device)
        execution_phase_marker = torch.zeros(4,
                                             dtype=torch.int32,
                                             device=device)
        context_sequence_count_carrier = torch.empty(0,
                                                     dtype=torch.int32,
                                                     device=device)

        args = (inputs_embeds, hidden_states_input, *past_key_values_list,
                rope_rotary_cos_sin_sliding, rope_rotary_cos_sin_full,
                positions, query_start_offsets, query_lengths, past_lengths,
                attention_sequence_lengths, state_indices,
                execution_phase_marker, context_sequence_count_carrier,
                kv_page_table)
        input_names = (
            ["inputs_embeds", "hidden_states_input"] +
            [f"past_key_values_{i}" for i in range(num_layers)] + [
                "rope_rotary_cos_sin_sliding", "rope_rotary_cos_sin_full",
                "positions", "query_start_offsets", "query_lengths",
                "past_lengths", "attention_sequence_lengths", "state_indices",
                "execution_phase_marker", "context_sequence_count_carrier",
                "kv_page_table"
            ])
        output_names = ["logits", "hidden_states"]

        tokens = torch.export.Dim("physical_tokens", min=1, max=8_388_608)
        batch = torch.export.Dim("num_sequences", min=1, max=256)
        page_batch = torch.export.Dim("page_batch", min=1, max=256)
        max_pages = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
        num_pages = torch.export.Dim("num_pages", min=1, max=1048576)

        phase_extent = torch.export.Dim("execution_phase_extent", min=1, max=8)
        dynamic_shapes: list = [{0: tokens}, {0: tokens}]
        dynamic_shapes += [{1: num_pages} for _ in range(num_layers)]
        dynamic_shapes += [{
            0: tokens
        }, {
            0: tokens
        }, {
            0: tokens
        }, {
            0: batch + 1
        }, {
            0: batch
        }, {
            0: batch
        }, {
            0: batch
        }, {
            0: batch
        }, {
            0: phase_extent
        }, {}, {
            0: page_batch,
            2: max_pages
        }]

        wrapped = _make_gemma4_assistant_flat_wrapper(self, num_layers)
        wrapped.eval()
        return OnnxSpec(wrapped=wrapped,
                        args=args,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=dynamic_shapes)
