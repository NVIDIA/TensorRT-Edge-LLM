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
"""Qwen4-Exp (Qwen3.8-Flash-Next) text backbone components.

Qwen4-Exp replaces the single residual bus of a standard transformer with
``hc_count`` parallel residual streams ("Gated Residual", also published as
HyperConnections).  The stack entry tiles ``inputs_embeds`` into
``hc_count * hidden_size``; every block is wrapped by a
:class:`Qwen4ExpGatedResidual` that reads a ``hidden_size``-wide mixture out of
the streams, and writes the block output back with a per-stream weight; the
stack exit folds the streams back to ``hidden_size``.

These streams carry the only normalization on the residual path -- the
architecture has no per-layer ``input_layernorm`` / ``post_attention_layernorm``
and no final ``model.norm``; each ``hc_norm`` here takes their place.
"""

import itertools
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import LAYER_GDN, ModelConfig
from ..default.modeling_default import MLP, OnnxSpec
from ..linear import (FP16Linear, NVFP4A16Linear, TPMode, is_int4_linear,
                      make_linear)
from ..ops import (KV_PAGE_SIZE, causal_conv1d,
                   causal_conv1d_with_intermediate, qkv_concat)
from ..qwen3_5.modeling_qwen3_5_text import Conv1dBuffers, GdnMixer
from ..qwen3_moe.modeling_qwen3_moe import Qwen3SparseMoeBlock

# Dummy extents for the export trace. Every one of them is a dynamic axis, so
# the values only have to be large enough to keep the trace well formed; values at
# a Dim's min boundary risk being specialized to constants, hence 2 rather than 1.
_NUM_SEQUENCES = 2
_QUERY_LENGTH = 2
_STATE_POOL_ROWS = 4
_PLE_POOL_ROWS = 4

__all__ = [
    "Qwen4ExpRMSNorm",
    "Qwen4ExpPle",
    "Qwen4ExpQsaIndexer",
    "Qwen4ExpQsaAttention",
    "Qwen4ExpGatedResidual",
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpBackbone",
    "Qwen4ExpCausalLM",
]


class Qwen4ExpRMSNorm(nn.Module):
    """Zero-centered RMSNorm, optionally normalizing groups independently.

    With ``group_size`` set, the last dimension is split into
    ``dim // group_size`` groups that each get their own RMS. The Gated
    Residual needs this so that one residual stream's magnitude cannot rescale
    another's; PLE (n-gram embedding) reuses it for the same reason.

    Normalization and the ``(1 + weight)`` scale are both evaluated in float32
    and rounded once on the way out, matching the reference implementation.
    """

    def __init__(self,
                 dim: int,
                 group_size: "int | None" = None,
                 eps: float = 1e-6) -> None:
        super().__init__()
        if group_size is not None and dim % group_size != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by group_size ({group_size})")
        self.variance_epsilon = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.float16))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        if self.group_size is not None:
            x = x.unflatten(-1, (-1, self.group_size))
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        if self.group_size is not None:
            x = x.flatten(-2)
        return (x * (1.0 + self.weight.float())).to(input_dtype)


class Qwen4ExpGatedResidual(nn.Module):
    """Read/write gate pair wrapping one block of the multi-stream residual.

    :meth:`forward` is the read half ("Mix"): it normalizes the streams, builds
    an element-wise gate through a ``hc_lowrank`` bottleneck and averages the
    gated streams into one ``hidden_size``-wide block input.  It also emits the
    per-stream injection weights that :meth:`combine`, the write half, applies
    once the block has produced its output.

    ``use_combine=False`` builds the stack-exit mixer, which only folds the
    streams down and has no ``block_inject_weight``.
    """

    def __init__(self,
                 config: ModelConfig,
                 module_name: str,
                 use_combine: bool = True) -> None:
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        hc_hidden_size = self.hc_count * self.hidden_size
        self.hc_norm = Qwen4ExpRMSNorm(hc_hidden_size,
                                       group_size=self.hidden_size,
                                       eps=config.rms_norm_eps)
        self.input_mix_weight_down = make_linear(
            config,
            hc_hidden_size,
            config.hc_lowrank,
            module_name=f"{module_name}.input_mix_weight_down")
        self.input_mix_weight_up = make_linear(
            config,
            config.hc_lowrank,
            hc_hidden_size,
            module_name=f"{module_name}.input_mix_weight_up")
        if use_combine:
            self.block_inject_weight = make_linear(
                config,
                hc_hidden_size,
                self.hc_count,
                module_name=f"{module_name}.block_inject_weight")
        else:
            self.block_inject_weight = None

    def forward(self, hyper_input: torch.Tensor):
        """Mix the streams into one block input.

        Returns ``mixed_input`` alone for the exit mixer, else
        ``(mixed_input, injection_weights)``.
        """
        normed = self.hc_norm(hyper_input)
        # The /hc_count temperature keeps the gates off saturation at init; it
        # is a constant at inference and could be folded into the weights.
        mix = F.silu(self.input_mix_weight_down(normed) / self.hc_count)
        mix = torch.sigmoid(self.input_mix_weight_up(mix))
        streams = (self.hc_count, self.hidden_size)
        mixed_input = (mix.unflatten(-1, streams) *
                       normed.unflatten(-1, streams)).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed_input
        injection_weights = 2 * torch.sigmoid(
            self.block_inject_weight(normed) / self.hc_count)
        return mixed_input, injection_weights

    @staticmethod
    def combine(hyper_input: torch.Tensor, block_output: torch.Tensor,
                injection_weights: torch.Tensor) -> torch.Tensor:
        """Write ``block_output`` back into every stream, per-stream weighted.

        The un-normalized ``hyper_input`` is what carries forward: the
        normalized view exists only to decide the gates, and the residual
        trunk has to stay an identity path.
        """
        injection = block_output.unsqueeze(-2) * injection_weights.unsqueeze(
            -1)
        return hyper_input + injection.flatten(-2)


class NgramTableScale(nn.Module):
    """Dequantization scale for the n-gram table.

    Nested so the buffer lands on the checkpoint key
    ``layers.N.ple.ple_embedding.ngram_embedding.weight_scale``. Only this scalar reaches the
    graph; the table's rows stay in the checkpoint and are gathered from the runtime's row pool.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ngram_embedding = nn.Module()
        self.ngram_embedding.register_buffer(
            "weight_scale", torch.zeros(1, dtype=torch.float16))


class Qwen4ExpPle(nn.Module):
    """Hashed n-gram features written into the residual streams (PLE).

    Row IDs depend on ``input_ids`` alone -- no hidden state reaches the hash -- so the runtime
    computes them before the engine runs and hands in a row map into its embedding pool. The
    lookup itself is in the graph: ``ple_pool_gather`` reads the FP8 rows, dequantizes them and
    concatenates the heads. What remains is the decision of how much of each token's n-gram vector
    to write into each stream, and a dilated depthwise convolution that widens that decision over a
    few tokens of lexical context.

    The gate is asymmetric: one key per stream ("how much"), one value shared by all of them
    ("what").  :meth:`forward` returns the injection; the caller adds it to the streams.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.conv_dim = self.hc_count * self.hidden_size
        self.conv_kernel = config.ple_conv_kernel_size
        # Adjacent n-grams overlap, so dilating by the n-gram order samples non-overlapping windows.
        self.conv_dilation = config.ngram_size
        self.conv_state_len = (self.conv_kernel - 1) * self.conv_dilation
        prefix = f"layers.{layer_idx}.ple"
        self.key_proj = make_linear(config,
                                    config.ple_embed_dim,
                                    self.conv_dim,
                                    module_name=f"{prefix}.key_proj")
        self.value_proj = make_linear(config,
                                      config.ple_embed_dim,
                                      self.hidden_size,
                                      module_name=f"{prefix}.value_proj")
        self.norm_key = Qwen4ExpRMSNorm(self.conv_dim,
                                        group_size=self.hidden_size,
                                        eps=config.rms_norm_eps)
        self.norm_query = Qwen4ExpRMSNorm(self.conv_dim,
                                          group_size=self.hidden_size,
                                          eps=config.rms_norm_eps)
        self.norm_conv = Qwen4ExpRMSNorm(self.conv_dim,
                                         group_size=self.hidden_size,
                                         eps=config.rms_norm_eps)
        # Weight and bias as plain buffers, like the GDN mixer's: the convolution itself is the
        # plugin's, and upstream carries no bias, so a zero one keeps the binding uniform.
        self.conv1d = Conv1dBuffers(self.conv_dim, self.conv_kernel, ())
        self.ple_embedding = NgramTableScale()

    def forward(
        self,
        hidden_states: torch.Tensor,
        ple_cache_pool: torch.Tensor,
        ple_row_map: torch.Tensor,
        conv_state: torch.Tensor,
        query_start_offsets: torch.Tensor,
        query_lengths: torch.Tensor,
        state_indices: torch.Tensor,
        execution_phase_marker: torch.Tensor,
        context_sequence_count_carrier: torch.Tensor,
        tree_parent_ids: "torch.Tensor | None" = None,
        tree_depths: "torch.Tensor | None" = None,
        use_intermediate_state: bool = False,
        spec_verify_layout: "torch.Tensor | None" = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, "torch.Tensor | None"]:
        """Returns what to add to the residual streams, the next convolution state and, on a
        speculative base, the per-token states."""
        # The gather returns the scale dtype; the checkpoint may store it as F32.
        ngram_embeds = torch.ops.trt.ple_pool_gather(
            ple_cache_pool, ple_row_map,
            self.ple_embedding.ngram_embedding.weight_scale.to(
                hidden_states.dtype))
        key_normed = self.norm_key(self.key_proj(ngram_embeds)).unflatten(
            -1, (self.hc_count, self.hidden_size))
        value = self.value_proj(ngram_embeds)
        query_normed = self.norm_query(hidden_states).unflatten(
            -1, (self.hc_count, self.hidden_size))

        gate = (key_normed * query_normed).sum(
            dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        # Signed square root: magnitude compression, so the sigmoid below is not saturated.
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_normed = self.norm_conv(gated_value.flatten(-2))

        # No mask, unlike the reference: the plugin reads nothing past a sequence's
        # ``query_lengths`` and ends its state window at the last real token. That argument is
        # specific to the entry-padded layout; kNativeCompactRagged would need it redone.
        if use_intermediate_state:
            convolved, next_conv_state, intermediate_conv_state = causal_conv1d_with_intermediate(
                gated_value_normed,
                self.conv1d.weight,
                self.conv1d.bias,
                conv_state,
                query_lengths,
                query_start_offsets,
                state_indices,
                stride=1,
                padding=(self.conv_kernel - 1) * self.conv_dilation,
                dilation=self.conv_dilation,
                groups=self.conv_dim,
                execution_phase_marker=execution_phase_marker,
                context_sequence_count_carrier=context_sequence_count_carrier,
                tree_parent_ids=tree_parent_ids,
                tree_depths=tree_depths,
                spec_verify_layout=spec_verify_layout)
        else:
            convolved, next_conv_state, _ = causal_conv1d(
                gated_value_normed,
                self.conv1d.weight,
                self.conv1d.bias,
                conv_state,
                query_lengths,
                stride=1,
                padding=(self.conv_kernel - 1) * self.conv_dilation,
                dilation=self.conv_dilation,
                groups=self.conv_dim,
                query_start_offsets=query_start_offsets,
                state_indices=state_indices,
                execution_phase_marker=execution_phase_marker,
                context_sequence_count_carrier=context_sequence_count_carrier)
            intermediate_conv_state = None
        return (gated_value.flatten(-2) + F.silu(convolved), next_conv_state,
                intermediate_conv_state)


# ---------------------------------------------------------------------------
# Decoder stack
# ---------------------------------------------------------------------------


class Qwen4ExpQsaIndexer(nn.Module):
    """Projections for the block-compressed indexer that drives QSA.

    Only the packed index Q/K projection and its two norms live here; the
    scoring, block top-k and index expansion are inside ``QsaAttentionPlugin``.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        head_dim = config.indexer_head_dim
        packed = (config.indexer_n_heads + config.indexer_kv_heads) * head_dim
        self.index_qk_proj = make_linear(
            config,
            config.hidden_size,
            packed,
            module_name=f"layers.{layer_idx}.self_attn.indexer.index_qk_proj",
            tp_mode=TPMode.COL)
        self.q_layernorm = Qwen4ExpRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = Qwen4ExpRMSNorm(head_dim, eps=config.rms_norm_eps)


class Qwen4ExpQsaAttention(nn.Module):
    """Qwen Sparse Attention: the projections around ``QsaAttentionPlugin``.

    The plugin owns qk-norm, partial RoPE, the paged-KV write, the indexer and
    the sparse GQA itself; this module owns the projections, the packed QKV
    layout and the output gate.  Like Qwen3.5 gated attention, ``q_proj`` emits
    query and gate together: per head the first ``head_dim`` values are the
    query and the second ``head_dim`` are the gate.

    With the tree inputs the plugin also serves the speculative phases: each
    node writes its indexer tails into its own pool slot and the runtime
    commits the accepted rows.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.indexer_n_heads = config.indexer_n_heads
        self.indexer_head_dim = config.indexer_head_dim
        self.indexer_budget = config.indexer_budget
        self.indexer_compress_ratio = config.indexer_compress_ratio
        self.rms_norm_eps = config.rms_norm_eps
        # 0.0 tells the plugin to use its default 1/sqrt(head_size).
        self.attention_scale = float(config.attention_scaling or 0.0)
        prefix = f"layers.{layer_idx}.self_attn"

        self.q_proj = make_linear(config,
                                  config.hidden_size,
                                  self.num_heads * self.head_dim * 2,
                                  bias=config.attention_bias,
                                  module_name=f"{prefix}.q_proj",
                                  tp_mode=TPMode.COL)
        self.k_proj = make_linear(config,
                                  config.hidden_size,
                                  self.num_kv_heads * self.head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{prefix}.k_proj",
                                  tp_mode=TPMode.COL)
        self.v_proj = make_linear(config,
                                  config.hidden_size,
                                  self.num_kv_heads * self.head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{prefix}.v_proj",
                                  tp_mode=TPMode.COL)
        self.o_proj = make_linear(config,
                                  self.num_heads * self.head_dim,
                                  config.hidden_size,
                                  module_name=f"{prefix}.o_proj",
                                  tp_mode=TPMode.ROW)
        # Quantized projections must be packed by the QKV concat plugin, as in the default model.
        self._requires_qkv_concat = any(
            is_int4_linear(proj) or isinstance(proj, NVFP4A16Linear)
            for proj in (self.q_proj, self.k_proj, self.v_proj))
        self.q_norm = Qwen4ExpRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen4ExpRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.indexer = Qwen4ExpQsaIndexer(config, layer_idx)

        # Filled by ``_capture_qk_norm_gammas_for_export`` once weights are
        # loaded. Left empty here so that its guard can tell a capture that
        # never happened from a legitimately all-zero gamma.
        self._q_norm_gamma_list: list = []
        self._k_norm_gamma_list: list = []
        self._indexer_q_norm_gamma_list: list = []
        self._indexer_k_norm_gamma_list: list = []

    def _capture_qk_norm_gamma_lists(self) -> None:
        """Materialize the four norm gains as ``list[float]`` plugin attributes.

        A gamma list always holds the effective multiplier the fused kernel
        applies as-is.  For a llama-convention norm that is the raw weight; for
        the zero-centered norms here it is ``1 + w``, so the fold happens on
        this side.

        The indexer pair is the exception and stays raw: its kernel adds the 1
        itself in float32.  Both conventions are declared per input in
        ``cpp/plugins/qsaAttentionPlugin/qsaAttentionPlugin.cpp``.
        """

        def gain(module: nn.Module) -> torch.Tensor:
            return module.weight.detach().to(torch.float32).cpu().flatten()

        self._q_norm_gamma_list = (1.0 + gain(self.q_norm)).tolist()
        self._k_norm_gamma_list = (1.0 + gain(self.k_norm)).tolist()
        self._indexer_q_norm_gamma_list = gain(
            self.indexer.q_layernorm).tolist()
        self._indexer_k_norm_gamma_list = gain(
            self.indexer.k_layernorm).tolist()

    def forward_ragged(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        query_start_offsets: torch.Tensor,
        query_lengths: torch.Tensor,
        past_lengths: torch.Tensor,
        attention_sequence_lengths: torch.Tensor,
        execution_phase_marker: torch.Tensor,
        context_sequence_count_carrier: torch.Tensor,
        kv_page_table: torch.Tensor,
        tree_parent_ids: "torch.Tensor | None" = None,
        tree_depths: "torch.Tensor | None" = None,
        spec_verify_layout: "torch.Tensor | None" = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (tree_parent_ids is None) != (tree_depths is None):
            raise ValueError(
                "tree parent IDs and depths must be supplied together")
        if tree_parent_ids is not None and spec_verify_layout is None:
            raise ValueError("Verify state requires spec_verify_layout")
        num_tokens = hidden_states.shape[0]
        q_output = self.q_proj(hidden_states).reshape(num_tokens,
                                                      self.num_heads,
                                                      self.head_dim * 2)
        query_states, gate_states = q_output.chunk(2, dim=-1)
        pack = qkv_concat if self._requires_qkv_concat else (
            lambda q, k, v: torch.cat([q, k, v], dim=-1))
        qkv = pack(
            query_states.reshape(num_tokens, self.num_heads * self.head_dim),
            self.k_proj(hidden_states), self.v_proj(hidden_states))

        attn_output, present_key_value = torch.ops.trt.qsa_attention_plugin(
            qkv,
            self.indexer.index_qk_proj(hidden_states),
            past_key_value,
            query_lengths,
            rope_rotary_cos_sin,
            past_lengths,
            kv_page_table,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            indexer_n_heads=self.indexer_n_heads,
            indexer_head_dim=self.indexer_head_dim,
            indexer_budget=self.indexer_budget,
            indexer_compress_ratio=self.indexer_compress_ratio,
            attention_scale=self.attention_scale,
            rms_norm_eps=self.rms_norm_eps,
            q_norm_gamma=self._q_norm_gamma_list,
            k_norm_gamma=self._k_norm_gamma_list,
            indexer_q_norm_gamma=self._indexer_q_norm_gamma_list,
            indexer_k_norm_gamma=self._indexer_k_norm_gamma_list,
            query_start_offsets=query_start_offsets,
            attention_sequence_lengths=attention_sequence_lengths,
            execution_phase_marker=execution_phase_marker,
            context_sequence_count_carrier=context_sequence_count_carrier,
            use_spec_verify_state=int(tree_parent_ids is not None),
            tree_parent_ids=tree_parent_ids,
            tree_depths=tree_depths,
            spec_verify_layout=spec_verify_layout)

        attn_output = attn_output * torch.sigmoid(gate_states)
        return self.o_proj(
            attn_output.reshape(num_tokens, self.num_heads *
                                self.head_dim)), present_key_value


# Verification metadata the runtime binds on a speculative base.
_SPECULATIVE_INPUTS = ("attention_position_ids", "packed_attention_mask",
                       "tree_parent_ids", "tree_depths", "valid_tree_counts",
                       "spec_verify_layout")


class Qwen4ExpDecoderLayer(nn.Module):
    """One decoder layer over the ``hc_count``-stream residual.

    Both the mixer and the MLP are wrapped by their own read/write gate pair,
    so the blocks themselves never see more than ``hidden_size``.  The mixer's
    own state (GDN conv/recurrent, or the attention KV page) is passed straight
    back to the caller, which owns the per-layer indexing.
    """

    def __init__(self, config: ModelConfig, layer_idx: int,
                 layer_type: str) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.layer_idx = layer_idx
        prefix = f"layers.{layer_idx}"
        if layer_type == LAYER_GDN:
            assert config.gdn_cfg is not None, "linear_attention needs gdn_cfg"
            self.linear_attn = GdnMixer(config, config.gdn_cfg,
                                        f"{prefix}.linear_attn")
        else:
            self.self_attn = Qwen4ExpQsaAttention(config, layer_idx)
        self.mlp = (Qwen3SparseMoeBlock(config, module_prefix=f"{prefix}.mlp")
                    if config.num_experts > 0 else MLP(config,
                                                       layer_idx=layer_idx))
        # An empty ``ple_layer_ids`` must cost no module, no input, no state
        # and no binding anywhere in the stack.
        self.ple: "Qwen4ExpPle | None" = (Qwen4ExpPle(
            config, layer_idx) if layer_idx in config.ngram_layers else None)
        self.attn_hyper_connection = Qwen4ExpGatedResidual(
            config, f"{prefix}.attn_hyper_connection")
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(
            config, f"{prefix}.mlp_hyper_connection")

    def _apply_mlp(self, hyper_states: torch.Tensor) -> torch.Tensor:
        mixed, injection_weights = self.mlp_hyper_connection(hyper_states)
        return Qwen4ExpGatedResidual.combine(hyper_states, self.mlp(mixed),
                                             injection_weights)

    def forward_gdn(
        self,
        hyper_states: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        ragged: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        mixed, injection_weights = self.attn_hyper_connection(hyper_states)
        (mixer_out, conv_out, rec_out, intermediate_conv,
         intermediate_rec) = self.linear_attn.forward_ragged(
             mixed,
             conv_state,
             recurrent_state,
             ragged["query_start_offsets"],
             ragged["query_lengths"],
             ragged["state_indices"],
             ragged["execution_phase_marker"],
             ragged["context_sequence_count_carrier"],
             tree_parent_ids=ragged.get("tree_parent_ids"),
             tree_depths=ragged.get("tree_depths"),
             use_intermediate_state=ragged.get("use_intermediate_state",
                                               False),
             spec_verify_layout=ragged.get("spec_verify_layout"))
        hyper_states = Qwen4ExpGatedResidual.combine(hyper_states, mixer_out,
                                                     injection_weights)
        return (self._apply_mlp(hyper_states), conv_out, rec_out,
                intermediate_conv, intermediate_rec)

    def forward_attention(
        self,
        hyper_states: torch.Tensor,
        past_key_value: torch.Tensor,
        ragged: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mixed, injection_weights = self.attn_hyper_connection(hyper_states)
        attn_out, present_kv = self.self_attn.forward_ragged(
            mixed,
            past_key_value,
            ragged["rope_rotary_cos_sin"],
            ragged["query_start_offsets"],
            ragged["query_lengths"],
            ragged["past_lengths"],
            ragged["attention_sequence_lengths"],
            ragged["execution_phase_marker"],
            ragged["context_sequence_count_carrier"],
            ragged["kv_page_table"],
            tree_parent_ids=ragged.get("tree_parent_ids"),
            tree_depths=ragged.get("tree_depths"),
            spec_verify_layout=ragged.get("spec_verify_layout"))
        hyper_states = Qwen4ExpGatedResidual.combine(hyper_states, attn_out,
                                                     injection_weights)
        return self._apply_mlp(hyper_states), present_kv


class Qwen4ExpBackbone(nn.Module):
    """Decoder backbone: tile into the streams, run the layers, fold back.

    There is no ``norm`` member: ``hyper_connection_mixer`` is both the final
    normalization and the stream reduction, and the checkpoint carries no
    separate final-norm weight.  Anything appended after it would be a second
    normalization the reference does not have.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hc_count = config.hc_count
        self.ple_layers: List[int] = config.ngram_layers
        for layer_idx in self.ple_layers:
            if not 0 <= layer_idx < config.num_hidden_layers:
                raise ValueError(
                    f"ple_layer_ids={config.ple_layer_ids} is 1-indexed and names decoder layer "
                    f"{layer_idx}, which is outside a {config.num_hidden_layers}-layer stack."
                )
        self.layer_types: List[str] = config.layer_types
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen4ExpDecoderLayer(config, layer_idx=i, layer_type=lt)
            for i, lt in enumerate(config.layer_types)
        ])
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(
            config, "hyper_connection_mixer", use_combine=False)

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
        conv_states: Tuple[torch.Tensor, ...] = (),
        recurrent_states: Tuple[torch.Tensor, ...] = (),
        ple_cache_pool: "torch.Tensor | None" = None,
        ple_row_map: "torch.Tensor | None" = None,
        ngram_conv_states: Tuple[torch.Tensor, ...] = (),
        attention_position_ids: "torch.Tensor | None" = None,
        packed_attention_mask: "torch.Tensor | None" = None,
        tree_parent_ids: "torch.Tensor | None" = None,
        tree_depths: "torch.Tensor | None" = None,
        valid_tree_counts: "torch.Tensor | None" = None,
        spec_verify_layout: "torch.Tensor | None" = None,
        collect_intermediate_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple, Tuple, Tuple, Tuple, Tuple,
               Tuple, Tuple]:
        """Returns the folded stream, the bus before the fold, and the per-layer states."""
        speculative = {
            "attention_position_ids": attention_position_ids,
            "packed_attention_mask": packed_attention_mask,
            "tree_parent_ids": tree_parent_ids,
            "tree_depths": tree_depths,
            "valid_tree_counts": valid_tree_counts,
            "spec_verify_layout": spec_verify_layout,
        }
        hyper_states = inputs_embeds.repeat(1, self.hc_count)
        present_key_values_list: List[torch.Tensor] = []
        present_conv_states_list: List[torch.Tensor] = []
        present_recurrent_states_list: List[torch.Tensor] = []
        present_ngram_conv_states_list: List[torch.Tensor] = []
        intermediate_conv_states_list: List[torch.Tensor] = []
        intermediate_recurrent_states_list: List[torch.Tensor] = []
        intermediate_ngram_conv_states_list: List[torch.Tensor] = []
        attn_idx = 0
        gdn_idx = 0
        ple_idx = 0
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
            "use_intermediate_state": collect_intermediate_states,
            **speculative,
        }

        for layer, layer_type in zip(self.layers, self.layer_types):
            if layer.ple is not None:
                injection, ple_conv_out, ple_intermediate = layer.ple(
                    hyper_states,
                    ple_cache_pool,
                    ple_row_map,
                    ngram_conv_states[ple_idx],
                    query_start_offsets,
                    query_lengths,
                    state_indices,
                    execution_phase_marker,
                    context_sequence_count_carrier,
                    tree_parent_ids=tree_parent_ids,
                    tree_depths=tree_depths,
                    use_intermediate_state=collect_intermediate_states,
                    spec_verify_layout=spec_verify_layout)
                # A bare addition: unlike every other write to the streams, the per-stream gate
                # is already inside the injection.
                hyper_states = hyper_states + injection
                present_ngram_conv_states_list.append(ple_conv_out)
                if collect_intermediate_states:
                    intermediate_ngram_conv_states_list.append(
                        ple_intermediate)
                ple_idx += 1
            if layer_type == LAYER_GDN:
                (hyper_states, conv_out, rec_out, intermediate_conv,
                 intermediate_rec) = layer.forward_gdn(
                     hyper_states, conv_states[gdn_idx],
                     recurrent_states[gdn_idx], ragged)
                present_conv_states_list.append(conv_out)
                present_recurrent_states_list.append(rec_out)
                if collect_intermediate_states:
                    intermediate_conv_states_list.append(intermediate_conv)
                    intermediate_recurrent_states_list.append(intermediate_rec)
                gdn_idx += 1
            else:
                hyper_states, present_kv = layer.forward_attention(
                    hyper_states, past_key_values[attn_idx], ragged)
                present_key_values_list.append(present_kv)
                attn_idx += 1

        hidden_states = self.hyper_connection_mixer(hyper_states)
        return (hidden_states, hyper_states, tuple(present_key_values_list),
                tuple(present_conv_states_list),
                tuple(present_recurrent_states_list),
                tuple(present_ngram_conv_states_list),
                tuple(intermediate_conv_states_list),
                tuple(intermediate_recurrent_states_list),
                tuple(intermediate_ngram_conv_states_list))


def _make_flat_wrapper_ragged(model: nn.Module, num_attn: int, num_gdn: int,
                              num_ple: int, spec_base: bool) -> nn.Module:
    """Build the positional-argument wrapper ``torch.onnx.export`` traces.

    The exporter names graph inputs after the traced signature, so the
    per-layer state tensors have to appear as individually named parameters
    rather than tuples; the signature is generated because their count is a
    property of the config.
    """
    param_names: List[str] = (
        ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(num_attn)] +
        [
            "rope_rotary_cos_sin", "positions", "query_start_offsets",
            "query_lengths", "past_lengths", "attention_sequence_lengths",
            "state_indices", "execution_phase_marker",
            "context_sequence_count_carrier", "kv_page_table", "logits_indices"
        ] + [f"conv_state_{i}" for i in range(num_gdn)] +
        [f"recurrent_state_{i}" for i in range(num_gdn)] +
        (["ple_cache_pool", "ple_row_map"] if num_ple else []) +
        [f"ngram_conv_state_{i}" for i in range(num_ple)])
    if spec_base:
        param_names += list(_SPECULATIVE_INPUTS)

    def group(prefix: str, count: int) -> str:
        return "({},)".format(", ".join(
            f"{prefix}{i}" for i in range(count))) if count else "()"

    ple_args = (
        f", ple_cache_pool, ple_row_map, {group('ngram_conv_state_', num_ple)}"
        if num_ple else ", None, None, ()")
    spec_kwargs = ("".join(
        f", {name}={name}"
        for name in _SPECULATIVE_INPUTS) if spec_base else "")
    body = (
        f"    (logits, hidden_states, present_key_values, present_conv_states, "
        f"present_recurrent_states, present_ngram_conv_states, "
        f"intermediate_conv_states, intermediate_recurrent_states, "
        f"intermediate_ngram_conv_states"
        f") = self._model.forward_ragged(\n"
        f"        inputs_embeds, {group('past_key_values_', num_attn)}, "
        f"rope_rotary_cos_sin, positions, query_start_offsets, query_lengths, "
        f"past_lengths, attention_sequence_lengths, state_indices, "
        f"execution_phase_marker, context_sequence_count_carrier, "
        f"kv_page_table, logits_indices,\n"
        f"        {group('conv_state_', num_gdn)}, "
        f"{group('recurrent_state_', num_gdn)}{ple_args}{spec_kwargs})\n"
        f"    result = ((logits,)"
        f" + ((hidden_states,) if hidden_states is not None else ())\n"
        f"              + tuple(present_key_values)"
        f" + tuple(present_conv_states)"
        f" + tuple(present_recurrent_states)"
        f" + tuple(present_ngram_conv_states))\n" +
        ("    result += tuple(intermediate_conv_states)"
         " + tuple(intermediate_recurrent_states)"
         " + tuple(intermediate_ngram_conv_states)\n" if spec_base else "") +
        "    return result\n")

    src = "def _forward(self, {}):\n{}".format(", ".join(param_names), body)
    globs: dict = {}
    exec(src, globs)  # noqa: S102

    class _Wrapper(nn.Module):

        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self._model = m

    _Wrapper.forward = globs["_forward"]
    return _Wrapper(model)


def _is_mtp_base_export(config: ModelConfig) -> bool:
    return bool(getattr(config, "mtp_base", False))


class Qwen4ExpCausalLM(nn.Module):
    """Qwen4-Exp causal LM: multi-stream backbone + lm_head."""

    # Dictated by the ``trt_edgellm::gated_delta_net`` plugin schema, same as
    # Qwen3.5: ``h0_source`` is fp32, ``conv_state`` follows the fp16 tensor
    # type. Also written into the runtime config.json, so this is the single
    # source of truth for both.
    RECURRENT_STATE_DTYPE = torch.float32
    CONV_STATE_DTYPE = torch.float16

    # The n-gram injection's dilated conv keeps its own state, on layers that need not be
    # recurrent at all. Declared separately from ``CONV_STATE_DTYPE`` so a model can bind this
    # state without binding a mixer one.
    NGRAM_CONV_STATE_DTYPE = torch.float16

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen4ExpBackbone(config)
        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")

    def tie_weights(self) -> None:
        if not self.config.tie_word_embeddings:
            return
        if not isinstance(self.lm_head, FP16Linear):
            return
        self.lm_head.weight = nn.Parameter(
            self.model.embed_tokens.weight.detach().clone(),
            requires_grad=False)

    def materialize_checkpoint_defaults(self, device: str) -> None:
        """Fill in tensors the checkpoint does not carry.

        Neither the GDN conv1d nor the injection's conv1d has a bias upstream; both modules
        keep one so the plugin binding is uniform, and zero is the value that makes it a no-op.
        """
        for layer in self.model.layers:
            mixers = [getattr(layer, "linear_attn", None), layer.ple]
            for mixer in mixers:
                if mixer is None or mixer.conv1d.bias.device.type != "meta":
                    continue
                bias = mixer.conv1d.bias
                mixer.conv1d.bias = torch.zeros(bias.shape,
                                                dtype=bias.dtype,
                                                device=device)

    def onnx_export_spec(self) -> OnnxSpec:
        config = self.config
        gc = config.gdn_cfg
        num_attn = config.num_attn_layers
        num_gdn = config.num_gdn_layers
        num_ple = len(config.ngram_layers)
        mtp_base = _is_mtp_base_export(config)
        assert gc is not None or num_gdn == 0, (
            "Qwen4-Exp requires gdn_cfg when any layer is a GDN layer")
        device = next(itertools.chain(self.parameters(),
                                      self.buffers())).device
        dtype16 = torch.float16
        num_sequences = _NUM_SEQUENCES
        query_length = _QUERY_LENGTH
        physical_tokens = num_sequences * query_length
        pool_rows = _STATE_POOL_ROWS

        inputs_embeds = torch.zeros(physical_tokens,
                                    config.hidden_size,
                                    dtype=dtype16,
                                    device=device)
        kv_dtype = (torch.float8_e4m3fn
                    if config.quant.kv_cache_quant == "fp8" else dtype16)
        # A QSA layer's pool row carries an indexer tail, so it is wider than head_dim; the
        # builder checks each past_key_values_i profile against the matching kv_layer_configs entry.
        attn_layer_ids = [
            layer_idx
            for layer_idx, layer_type in enumerate(config.layer_types)
            if layer_type != LAYER_GDN
        ]
        past_key_values_list: List[torch.Tensor] = [
            torch.zeros(2,
                        2,
                        KV_PAGE_SIZE,
                        config.num_key_value_heads,
                        config.kv_pool_head_dim,
                        dtype=kv_dtype,
                        device=device) for layer_idx in attn_layer_ids
        ]
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        rope_rotary_cos_sin = torch.zeros(physical_tokens,
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
        # Deliberately not the identity: a traced identity mapping would let a state gather fold
        # away, and the runtime hands in resident slots that are not batch positions.
        state_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)
        execution_phase_marker = torch.zeros(2,
                                             dtype=torch.int32,
                                             device=device)
        context_sequence_count_carrier = torch.zeros(num_sequences,
                                                     dtype=torch.int32,
                                                     device=device)
        # [sequences, 2, maxPagesPerSeq]: the K and V page tables are separate rows,
        # which is what the builder's optimization profile expects.
        kv_page_table = torch.zeros(num_sequences,
                                    2,
                                    2,
                                    dtype=torch.int32,
                                    device=device)
        # A verify step selects several rows per sequence.
        logits_indices = (torch.tensor(
            [0, 2, 3], dtype=torch.int64, device=device) if mtp_base else
                          query_start_offsets[1:].to(torch.int64) - 1)
        conv_states: List[torch.Tensor] = [
            torch.zeros(pool_rows,
                        gc.conv_dim,
                        gc.conv_kernel,
                        dtype=self.CONV_STATE_DTYPE,
                        device=device) for _ in range(num_gdn)
        ]
        recurrent_states: List[torch.Tensor] = [
            torch.zeros(pool_rows,
                        gc.num_value_heads,
                        gc.key_head_dim,
                        gc.value_head_dim,
                        dtype=self.RECURRENT_STATE_DTYPE,
                        device=device) for _ in range(num_gdn)
        ]
        # The table is the runtime's row pool and the row IDs are hashed outside the engine, so
        # both reach the graph as inputs; the gather itself is in the graph.
        ple_args: List[torch.Tensor] = []
        if num_ple:
            ple_conv_dim = config.hc_count * config.hidden_size
            ple_state_len = ((config.ple_conv_kernel_size - 1) *
                             config.ngram_size)
            ple_args.append(
                torch.zeros(_PLE_POOL_ROWS,
                            config.ple_embed_dim // config.ngram_heads,
                            dtype=torch.float8_e4m3fn,
                            device=device))
            ple_args.append(
                torch.zeros(physical_tokens,
                            config.ngram_heads,
                            dtype=torch.int32,
                            device=device))
            ple_args += [
                torch.zeros(pool_rows,
                            ple_conv_dim,
                            ple_state_len,
                            dtype=self.NGRAM_CONV_STATE_DTYPE,
                            device=device) for _ in range(num_ple)
            ]

        args = (inputs_embeds, *past_key_values_list, rope_rotary_cos_sin,
                positions, query_start_offsets, query_lengths, past_lengths,
                attention_sequence_lengths, state_indices,
                execution_phase_marker, context_sequence_count_carrier,
                kv_page_table, logits_indices, *conv_states, *recurrent_states,
                *ple_args)
        input_names = (["inputs_embeds"] + [
            f"past_key_values_{i}" for i in range(num_attn)
        ] + [
            "rope_rotary_cos_sin", "positions", "query_start_offsets",
            "query_lengths", "past_lengths", "attention_sequence_lengths",
            "state_indices", "execution_phase_marker",
            "context_sequence_count_carrier", "kv_page_table", "logits_indices"
        ] + [f"conv_state_{i}" for i in range(num_gdn)] +
                       [f"recurrent_state_{i}" for i in range(num_gdn)] +
                       (["ple_cache_pool", "ple_row_map"] if num_ple else []) +
                       [f"ngram_conv_state_{i}" for i in range(num_ple)])
        output_names = (
            ["logits"] + [f"present_key_values_{i}" for i in range(num_attn)] +
            [f"present_conv_state_{i}" for i in range(num_gdn)] +
            [f"present_recurrent_state_{i}" for i in range(num_gdn)] +
            [f"present_ngram_conv_state_{i}" for i in range(num_ple)])
        if mtp_base:
            output_names.insert(1, "hidden_states")
            output_names += [
                f"intermediate_conv_state_{i}" for i in range(num_gdn)
            ]
            output_names += [
                f"intermediate_recurrent_state_{i}" for i in range(num_gdn)
            ]
            output_names += [
                f"intermediate_ngram_conv_state_{i}" for i in range(num_ple)
            ]

        tokens = torch.export.Dim("physical_tokens", min=1, max=8_388_608)
        logits_rows = torch.export.Dim("logits_rows", min=1, max=8_388_608)
        sequences = torch.export.Dim("num_sequences", min=1, max=256)
        context_sequences = torch.export.Dim("num_context_sequences",
                                             min=0,
                                             max=256)
        state_pool = torch.export.Dim("state_pool_rows", min=1, max=256)
        max_pages = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
        num_pages = torch.export.Dim("num_pages", min=1, max=1048576)
        phase_extent = torch.export.Dim("execution_phase_extent", min=1, max=8)
        pool_capacity = torch.export.Dim("ple_pool_capacity",
                                         min=2,
                                         max=2**31 - 1)

        all_shapes: list = [{0: tokens}]
        all_shapes += [{1: num_pages} for _ in range(num_attn)]
        all_shapes += [
            {
                0: tokens
            },
            {
                0: tokens
            },
            {
                0: sequences + 1
            },
            {
                0: sequences
            },
            {
                0: sequences
            },
            {
                0: sequences
            },
            {
                0: sequences
            },
            {
                0: phase_extent
            },
            {
                0: context_sequences
            },
            {
                0: sequences,
                2: max_pages
            },
            {
                0: logits_rows if mtp_base else sequences
            },
        ]
        all_shapes += [{0: state_pool} for _ in range(num_gdn)]
        all_shapes += [{0: state_pool} for _ in range(num_gdn)]
        if num_ple:
            all_shapes.append({0: pool_capacity})
            all_shapes.append({0: tokens})
            all_shapes += [{0: state_pool} for _ in range(num_ple)]
        if mtp_base:
            packed_mask_width = torch.export.Dim("packed_mask_width",
                                                 min=1,
                                                 max=64)
            args += (positions.clone(),
                     torch.zeros(physical_tokens, (query_length + 31) // 32,
                                 dtype=torch.int32,
                                 device=device),
                     torch.full((physical_tokens, ),
                                -1,
                                dtype=torch.int32,
                                device=device),
                     torch.zeros(physical_tokens,
                                 dtype=torch.int32,
                                 device=device), query_lengths.clone(),
                     torch.empty(2, dtype=torch.int32, device=device))
            input_names += list(_SPECULATIVE_INPUTS)
            all_shapes += [{
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
            }, {
                0:
                torch.export.Dim("spec_verify_layout_len", min=0, max=2)
            }]

        wrapped = _make_flat_wrapper_ragged(self, num_attn, num_gdn, num_ple,
                                            mtp_base)
        wrapped.eval()
        return OnnxSpec(wrapped=wrapped,
                        args=args,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=all_shapes)

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
        conv_states: Tuple[torch.Tensor, ...] = (),
        recurrent_states: Tuple[torch.Tensor, ...] = (),
        ple_cache_pool: "torch.Tensor | None" = None,
        ple_row_map: "torch.Tensor | None" = None,
        ngram_conv_states: Tuple[torch.Tensor, ...] = (),
        attention_position_ids: "torch.Tensor | None" = None,
        packed_attention_mask: "torch.Tensor | None" = None,
        tree_parent_ids: "torch.Tensor | None" = None,
        tree_depths: "torch.Tensor | None" = None,
        valid_tree_counts: "torch.Tensor | None" = None,
        spec_verify_layout: "torch.Tensor | None" = None,
    ) -> Tuple:
        mtp_base = _is_mtp_base_export(self.config)
        (hidden_states, pre_mixer_states, present_key_values,
         present_conv_states, present_recurrent_states,
         present_ngram_conv_states, intermediate_conv_states,
         intermediate_recurrent_states,
         intermediate_ngram_conv_states) = self.model.forward_ragged(
             inputs_embeds,
             past_key_values,
             rope_rotary_cos_sin,
             positions,
             query_start_offsets,
             query_lengths,
             past_lengths,
             attention_sequence_lengths,
             state_indices,
             execution_phase_marker,
             context_sequence_count_carrier,
             kv_page_table,
             conv_states,
             recurrent_states,
             ple_cache_pool,
             ple_row_map,
             ngram_conv_states,
             attention_position_ids=attention_position_ids,
             packed_attention_mask=packed_attention_mask,
             tree_parent_ids=tree_parent_ids,
             tree_depths=tree_depths,
             valid_tree_counts=valid_tree_counts,
             spec_verify_layout=spec_verify_layout,
             collect_intermediate_states=mtp_base)
        selected_hidden_states = torch.index_select(hidden_states, 0,
                                                    logits_indices)
        logits = self.lm_head(selected_hidden_states).to(torch.float32)
        # The MTP draft resumes the multi-stream bus, so it is fed the state
        # the mixer consumed, not the single stream the mixer produced.
        emitted_hidden = pre_mixer_states if mtp_base else None
        return (logits, emitted_hidden, present_key_values,
                present_conv_states, present_recurrent_states,
                present_ngram_conv_states, intermediate_conv_states,
                intermediate_recurrent_states, intermediate_ngram_conv_states)
