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
Gemma 4 (``gemma4_text``) causal LM modeling for ONNX export.

This is the second phase of Gemma 4 support (issue
https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/72): the Python modeling and
``onnx_export_spec`` for the ``gemma4_text`` backbone.  Phase 1 added checkpoint
recognition and config parsing; the C++/runtime support for the new ONNX I/O
this module introduces (a second RoPE table and the Per-Layer-Embedding input)
lands in a follow-up phase.

Why a dedicated modeling file instead of the dense
:class:`~tensorrt_edgellm.models.default.modeling_default.CausalLM`?  Gemma 4
(a Gemma-3n-class architecture, without that family's AltUp/Laurel blocks)
departs from the Llama-style dense template in ways that are *silently wrong* if
forced through the default path:

* **Per-layer-type attention** — local *sliding* layers use ``head_dim`` (256);
  global *full* layers use ``global_head_dim`` (512).  ``num_kv_heads``,
  ``sliding_window`` and the RoPE table all vary with the layer type.
* **Dual RoPE** — local layers use a plain RoPE (``theta=1e4`` over the full
  256-d head); global layers use a *proportional* RoPE (``theta=1e6`` over only
  ``partial_rotary_factor=0.25`` of the 512-d head, i.e. 128 dims).  The two
  precomputed cos/sin tables differ in width, so the model takes *two* RoPE
  inputs and feeds each attention layer the one for its type.
* **QK(V) norm** — query and key are RMSNorm'd over the head before RoPE; value
  is RMSNorm'd (scale-free) as well.
* **Unit attention scaling** — Gemma 4 uses ``scaling = 1.0`` rather than the
  usual ``1/sqrt(head_dim)``.  The attention plugin always applies its own
  ``1/sqrt(head_size)``, so the query is pre-scaled by ``sqrt(head_dim)`` to
  cancel it (RoPE is a rotation and commutes with this scalar).
* **Per-Layer Embeddings (PLE)** — a second embedding pathway feeds a residual
  signal into every decoder layer.  The token-identity component is a lookup the
  runtime supplies as the ``per_layer_inputs`` model input; the context-aware
  component is projected from ``inputs_embeds`` inside the graph.
* **KV sharing** — the trailing ``num_kv_shared_layers`` layers carry no
  k/v/k_norm/v_norm weights and reuse the keys/values of the last non-shared
  layer of the same type.
* **Sandwich norms, GeGLU (double-wide on KV-shared layers) MLP, scaled input
  embeddings, and final-logit soft-capping.**
"""

import itertools
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import Gemma4Config, ModelConfig
from ..default.modeling_default import OnnxSpec
from ..linear import FP16Linear, make_linear
from ..ops import attention_plugin

__all__ = [
    "Gemma4RMSNorm",
    "Gemma4Attention",
    "Gemma4MLP",
    "Gemma4DecoderLayer",
    "Gemma4Backbone",
    "Gemma4CausalLM",
]

# ONNX export dummy-shape constants (match modeling_default).
_BATCH_SIZE = 1
_SEQ_LEN = 1
_PAST_LEN = 1
_MAX_POS = 4096

# ---------------------------------------------------------------------------
# Flat export wrapper
# ---------------------------------------------------------------------------


def _make_flat_wrapper(model: nn.Module, na: int) -> nn.Module:
    """Build a wrapper with an explicit flat ``forward`` signature (no ``*args``).

    Mirrors ``modeling_default._make_flat_wrapper`` (an explicit parameter list
    avoids a PyTorch 2.10 non-strict-exporter bug with ``*args``), but for the
    Gemma 4 input contract: two RoPE tables (local + global) and a
    ``per_layer_inputs`` tensor, and no deepstack inputs.

    ``na`` is the number of attention (== decoder) layers.
    """
    param_names: List[str] = (
        ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(na)] + [
            "rope_rotary_cos_sin_local", "rope_rotary_cos_sin_global",
            "context_lengths", "kvcache_start_index", "last_token_ids",
            "per_layer_inputs"
        ])
    past_kv_tuple = "({},)".format(", ".join(
        f"past_key_values_{i}" for i in range(na))) if na else "()"
    body = (
        f"    logits, present_key_values = self._model(\n"
        f"        inputs_embeds, {past_kv_tuple}, rope_rotary_cos_sin_local,\n"
        f"        rope_rotary_cos_sin_global, context_lengths,\n"
        f"        kvcache_start_index, last_token_ids, per_layer_inputs)\n"
        f"    return (logits,) + tuple(present_key_values)\n")
    src = "def _forward(self, {}):\n{}".format(", ".join(param_names), body)
    globs: dict = {}
    exec(src, globs)  # noqa: S102

    class _Wrapper(nn.Module):

        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self._model = m

    _Wrapper.forward = globs["_forward"]
    return _Wrapper(model)


# ---------------------------------------------------------------------------
# RMSNorm  (Gemma convention: fp32 throughout, eps inside the rsqrt, optional
# learnable scale)
# ---------------------------------------------------------------------------


class Gemma4RMSNorm(nn.Module):
    """Gemma 4 RMS normalization.

    Differs from :class:`~tensorrt_edgellm.models.default.modeling_default.RMSNorm`
    in three ways that matter for parity: the whole computation (including the
    weight multiply) runs in float32, the epsilon is added *inside* the inverse
    square root, and the learnable scale is optional (``with_scale=False`` for
    the value norm).

    Buffer: ``weight`` [dim] (present only when ``with_scale``).
    """

    def __init__(self,
                 dim: int,
                 eps: float = 1e-6,
                 with_scale: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float16))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        normed = hidden_states * torch.rsqrt(variance + self.eps)
        if self.with_scale:
            normed = normed * self.weight.to(torch.float32)
        return normed.to(input_dtype)


# ---------------------------------------------------------------------------
# Attention  (per-layer-type head dim / RoPE / sliding window, QKV norm, unit
# scaling, KV sharing)
# ---------------------------------------------------------------------------


class Gemma4Attention(nn.Module):
    """Gemma 4 GQA attention for one decoder layer.

    The layer type (``sliding`` local vs ``full`` global) fixes the head
    dimension, KV-head count, sliding-window size and which RoPE table is used.
    Trailing KV-shared layers carry only ``q_proj``/``q_norm``/``o_proj`` and
    reuse the keys/values of the last non-shared layer of the same type, passed
    in via ``shared_key``/``shared_value``.

    Submodule names match checkpoint keys: ``q_proj``, ``k_proj``, ``v_proj``,
    ``o_proj``, ``q_norm``, ``k_norm``, ``v_norm``.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        gemma4: Gemma4Config = config.gemma4_cfg
        self.layer_idx = layer_idx
        self.is_global = gemma4.is_global_layer[layer_idx]
        self.num_heads = config.num_attention_heads
        self.head_dim = (gemma4.global_head_dim
                         if self.is_global else gemma4.local_head_dim)
        # ``attention_k_eq_v`` (global layers reuse k as v) only applies to
        # global layers; E2B leaves it False.
        self.k_eq_v = gemma4.attention_k_eq_v and self.is_global
        self.num_kv_heads = (gemma4.num_global_key_value_heads
                             if self.k_eq_v else config.num_key_value_heads)
        # Local (sliding) layers carry a finite window; global layers do not.
        self.sliding_window_size = (gemma4.sliding_window
                                    if not self.is_global else -1)
        # Gemma 4 attention scaling is 1.0; the plugin applies 1/sqrt(head_size)
        # internally, so pre-scale the query by sqrt(head_dim) to cancel it.
        self.q_prescale = float(math.sqrt(self.head_dim))

        first_shared = gemma4.first_kv_shared_layer_idx(
            config.num_hidden_layers)
        self.is_kv_shared_layer = (gemma4.num_kv_shared_layers > 0
                                   and layer_idx >= first_shared)
        # The store-full-length-kv layer for a type is the last non-shared layer
        # of that type; KV-shared layers of the same type reuse its k/v.
        self.store_full_length_kv = False
        if not self.is_kv_shared_layer:
            same_type_before = [
                j for j in range(first_shared)
                if gemma4.is_global_layer[j] == self.is_global
            ]
            self.store_full_length_kv = bool(
                same_type_before) and layer_idx == same_type_before[-1]

        prefix = f"layers.{layer_idx}.self_attn"
        self.q_proj = make_linear(config,
                                  config.hidden_size,
                                  self.num_heads * self.head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{prefix}.q_proj")
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if not self.is_kv_shared_layer:
            self.k_proj = make_linear(config,
                                      config.hidden_size,
                                      self.num_kv_heads * self.head_dim,
                                      bias=config.attention_bias,
                                      module_name=f"{prefix}.k_proj")
            self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.v_norm = Gemma4RMSNorm(self.head_dim,
                                        eps=config.rms_norm_eps,
                                        with_scale=False)
            self.v_proj = (None if self.k_eq_v else make_linear(
                config,
                config.hidden_size,
                self.num_kv_heads * self.head_dim,
                bias=config.attention_bias,
                module_name=f"{prefix}.v_proj"))
        else:
            self.k_proj = None
            self.v_proj = None
            self.k_norm = None
            self.v_norm = None
        self.o_proj = make_linear(config,
                                  self.num_heads * self.head_dim,
                                  config.hidden_size,
                                  bias=config.attention_bias,
                                  module_name=f"{prefix}.o_proj")

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        shared_key: Optional[torch.Tensor] = None,
        shared_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        q_flat = self.num_heads * self.head_dim
        kv_flat = self.num_kv_heads * self.head_dim

        query_states = self.q_proj(hidden_states)
        query_states = self.q_norm(
            query_states.reshape(batch_size, seq_len, self.num_heads,
                                 self.head_dim)).reshape(
                                     batch_size, seq_len, q_flat)
        # Unit attention scaling (see __init__): cancel the plugin's
        # 1/sqrt(head_size) by pre-scaling the (post-norm) query.
        query_states = query_states * self.q_prescale

        if self.is_kv_shared_layer:
            key_states = shared_key
            value_states = shared_value
        else:
            key_states = self.k_proj(hidden_states)
            key_states = self.k_norm(
                key_states.reshape(batch_size, seq_len, self.num_kv_heads,
                                   self.head_dim)).reshape(
                                       batch_size, seq_len, kv_flat)
            if self.v_proj is not None:
                value_states = self.v_proj(hidden_states)
            else:
                # attention_k_eq_v: reuse the (pre-norm) key projection as value.
                value_states = self.k_proj(hidden_states)
            value_states = self.v_norm(
                value_states.reshape(batch_size, seq_len, self.num_kv_heads,
                                     self.head_dim)).reshape(
                                         batch_size, seq_len, kv_flat)

        attn_output, present_key_value = attention_plugin(
            query_states,
            key_states,
            value_states,
            past_key_value,
            context_lengths,
            rope_rotary_cos_sin,
            kvcache_start_index,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            sliding_window_size=self.sliding_window_size,
            enable_tree_attention=False,
            enable_fp8_kv_cache=False,
            qkv_scales=[1.0, 1.0, 1.0],
        )
        attn_output = attn_output.reshape(batch_size, seq_len, q_flat)
        # Return the pre-RoPE key/value so the backbone can stash them for the
        # KV-shared layers of this type (the plugin applies RoPE internally, so
        # reusing the pre-RoPE projections reproduces the shared cache exactly).
        return self.o_proj(attn_output), present_key_value, \
            key_states, value_states


# ---------------------------------------------------------------------------
# MLP  (GeGLU; double-wide intermediate on KV-shared layers)
# ---------------------------------------------------------------------------


class Gemma4MLP(nn.Module):
    """Gemma 4 GeGLU MLP: ``down(gelu(gate(x)) * up(x))``.

    The activation is ``gelu_pytorch_tanh``.  On KV-shared layers (when
    ``use_double_wide_mlp`` is set) the intermediate size is doubled.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        gemma4: Gemma4Config = config.gemma4_cfg
        first_shared = gemma4.first_kv_shared_layer_idx(
            config.num_hidden_layers)
        is_kv_shared = (gemma4.num_kv_shared_layers > 0
                        and layer_idx >= first_shared)
        double_wide = gemma4.use_double_wide_mlp and is_kv_shared
        intermediate = config.intermediate_size * (2 if double_wide else 1)

        prefix = f"layers.{layer_idx}.mlp"
        self.gate_proj = make_linear(config,
                                     config.hidden_size,
                                     intermediate,
                                     bias=False,
                                     module_name=f"{prefix}.gate_proj")
        self.up_proj = make_linear(config,
                                   config.hidden_size,
                                   intermediate,
                                   bias=False,
                                   module_name=f"{prefix}.up_proj")
        self.down_proj = make_linear(config,
                                     intermediate,
                                     config.hidden_size,
                                     bias=False,
                                     module_name=f"{prefix}.down_proj")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(hidden_states), approximate="tanh")
        return self.down_proj(gate * self.up_proj(hidden_states))


# ---------------------------------------------------------------------------
# DecoderLayer  (sandwich norms + PLE block)
# ---------------------------------------------------------------------------


class Gemma4DecoderLayer(nn.Module):
    """Single Gemma 4 decoder layer.

    Sandwich-normed attention and MLP blocks (pre- and post-norm around each
    residual), followed by the Per-Layer-Embedding block when the model carries
    PLE.  Submodule names match checkpoint keys: ``self_attn``, ``mlp``,
    ``input_layernorm``, ``post_attention_layernorm``,
    ``pre_feedforward_layernorm``, ``post_feedforward_layernorm`` and, for PLE,
    ``per_layer_input_gate``, ``per_layer_projection``,
    ``post_per_layer_input_norm``.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        gemma4: Gemma4Config = config.gemma4_cfg
        eps = config.rms_norm_eps
        hidden = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma4Attention(config, layer_idx)
        self.mlp = Gemma4MLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = Gemma4RMSNorm(hidden, eps=eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(hidden, eps=eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(hidden, eps=eps)
        # Gemma 4 multiplies the layer output by a (loadable) scalar buffer.
        self.register_buffer("layer_scalar", torch.ones(1))

        self.ple_dim = gemma4.hidden_size_per_layer_input
        if self.ple_dim:
            prefix = f"layers.{layer_idx}"
            self.per_layer_input_gate = make_linear(
                config,
                hidden,
                self.ple_dim,
                bias=False,
                module_name=f"{prefix}.per_layer_input_gate")
            self.per_layer_projection = make_linear(
                config,
                self.ple_dim,
                hidden,
                bias=False,
                module_name=f"{prefix}.per_layer_projection")
            self.post_per_layer_input_norm = Gemma4RMSNorm(hidden, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        per_layer_input: Optional[torch.Tensor] = None,
        shared_key: Optional[torch.Tensor] = None,
        shared_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out, present_key_value, key_states, value_states = self.self_attn(
            normed,
            past_key_value,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            shared_key=shared_key,
            shared_value=shared_value,
        )
        attn_out = self.post_attention_layernorm(attn_out)
        hidden_states = residual + attn_out

        residual = hidden_states
        mlp_out = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        mlp_out = self.post_feedforward_layernorm(mlp_out)
        hidden_states = residual + mlp_out

        if self.ple_dim:
            residual = hidden_states
            gated = self.per_layer_input_gate(hidden_states)
            gated = F.gelu(gated, approximate="tanh")
            gated = gated * per_layer_input
            gated = self.per_layer_projection(gated)
            gated = self.post_per_layer_input_norm(gated)
            hidden_states = residual + gated

        hidden_states = hidden_states * self.layer_scalar.to(
            hidden_states.dtype)
        return hidden_states, present_key_value, key_states, value_states


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class Gemma4Backbone(nn.Module):
    """Gemma 4 decoder stack with scaled embeddings and the PLE projection.

    Stored as ``model`` inside :class:`Gemma4CausalLM` so parameter keys carry
    the ``model.`` prefix matching the checkpoint.  Submodules: ``embed_tokens``,
    ``embed_tokens_per_layer``, ``per_layer_model_projection``,
    ``per_layer_projection_norm``, ``layers``, ``norm``.

    ``forward`` receives ``inputs_embeds`` (the runtime's *unscaled* main
    embedding lookup) and ``per_layer_inputs`` (the runtime's *unscaled*
    per-layer embedding lookup, shape ``[batch, seq, num_layers, ple_dim]``).
    The two Gemma embedding scales (``sqrt(hidden)`` and ``sqrt(ple_dim)``) and
    the PLE context projection are applied here so the runtime only does plain
    gathers.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        gemma4: Gemma4Config = config.gemma4_cfg
        self.config = config
        hidden = config.hidden_size
        num_layers = config.num_hidden_layers
        self.embed_scale = float(math.sqrt(hidden))
        self.layer_types_global = list(gemma4.is_global_layer)

        self.embed_tokens = nn.Embedding(config.vocab_size, hidden)

        self.ple_dim = gemma4.hidden_size_per_layer_input
        if self.ple_dim:
            self.ple_embed_scale = float(math.sqrt(self.ple_dim))
            self.per_layer_input_scale = float(2.0**-0.5)
            self.per_layer_projection_scale = float(hidden**-0.5)
            self.embed_tokens_per_layer = nn.Embedding(
                gemma4.vocab_size_per_layer_input, num_layers * self.ple_dim)
            self.per_layer_model_projection = make_linear(
                config,
                hidden,
                num_layers * self.ple_dim,
                bias=False,
                module_name="per_layer_model_projection")
            self.per_layer_projection_norm = Gemma4RMSNorm(
                self.ple_dim, eps=config.rms_norm_eps)

        self.layers = nn.ModuleList(
            [Gemma4DecoderLayer(config, i) for i in range(num_layers)])
        self.norm = Gemma4RMSNorm(hidden, eps=config.rms_norm_eps)

    def _project_per_layer_inputs(
            self, inputs_embeds: torch.Tensor,
            per_layer_inputs: torch.Tensor) -> torch.Tensor:
        """Combine the context-aware and token-identity PLE components.

        ``inputs_embeds`` is the *scaled* main embedding; ``per_layer_inputs`` is
        the *unscaled* per-layer lookup ``[batch, seq, num_layers, ple_dim]``.
        """
        batch_size, seq_len, _ = inputs_embeds.shape
        num_layers = len(self.layers)
        projection = self.per_layer_model_projection(inputs_embeds)
        projection = projection * self.per_layer_projection_scale
        projection = projection.reshape(batch_size, seq_len, num_layers,
                                        self.ple_dim)
        projection = self.per_layer_projection_norm(projection)
        token_identity = per_layer_inputs * self.ple_embed_scale
        return (projection + token_identity) * self.per_layer_input_scale

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin_local: torch.Tensor,
        rope_rotary_cos_sin_global: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        per_layer_inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple]:
        hidden_states = inputs_embeds * self.embed_scale
        per_layer = None
        if self.ple_dim:
            per_layer = self._project_per_layer_inputs(hidden_states,
                                                       per_layer_inputs)

        present_key_values: List[torch.Tensor] = []
        # Pre-RoPE key/value of the last non-shared layer of each type, reused by
        # the trailing KV-shared layers (keyed by is_global).
        shared_key: Dict[bool, torch.Tensor] = {}
        shared_value: Dict[bool, torch.Tensor] = {}

        for layer_index, layer in enumerate(self.layers):
            is_global = self.layer_types_global[layer_index]
            rope = (rope_rotary_cos_sin_global
                    if is_global else rope_rotary_cos_sin_local)
            attn = layer.self_attn
            sk = shared_key.get(is_global) if attn.is_kv_shared_layer else None
            sv = shared_value.get(
                is_global) if attn.is_kv_shared_layer else None
            ple_in = (per_layer[:, :, layer_index, :]
                      if per_layer is not None else None)

            hidden_states, present, key_states, value_states = layer(
                hidden_states,
                past_key_values[layer_index],
                rope,
                context_lengths,
                kvcache_start_index,
                per_layer_input=ple_in,
                shared_key=sk,
                shared_value=sv,
            )
            present_key_values.append(present)
            if attn.store_full_length_kv:
                shared_key[is_global] = key_states
                shared_value[is_global] = value_states

        return self.norm(hidden_states), tuple(present_key_values)


# ---------------------------------------------------------------------------
# CausalLM
# ---------------------------------------------------------------------------


class Gemma4CausalLM(nn.Module):
    """Gemma 4 causal LM: backbone + (soft-capped) lm_head.

    The inner :class:`Gemma4Backbone` is stored as ``model`` so its parameters
    carry the ``model.`` prefix matching checkpoint keys.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.gemma4_cfg is None:
            raise ValueError(
                "Gemma4CausalLM requires a parsed gemma4_cfg on the config; "
                "got None (is this a gemma4_text checkpoint?).")
        self.config = config
        self.gemma4_cfg: Gemma4Config = config.gemma4_cfg
        self.final_logit_softcapping = self.gemma4_cfg.final_logit_softcapping
        self.model = Gemma4Backbone(config)
        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")

    def tie_weights(self) -> None:
        """Clone ``embed_tokens.weight`` into ``lm_head.weight`` when tied.

        Gemma 4 ties the output projection to the *unscaled* input embedding
        (the ``sqrt(hidden)`` scale is applied to the embedding pathway only, in
        :meth:`Gemma4Backbone.forward`).  Mirrors
        :meth:`~tensorrt_edgellm.models.default.modeling_default.CausalLM.tie_weights`.
        """
        if not self.config.tie_word_embeddings:
            return
        if not isinstance(self.lm_head, FP16Linear):
            return
        embed_weight = self.model.embed_tokens.weight
        self.lm_head.weight = nn.Parameter(embed_weight.detach().clone(),
                                           requires_grad=False)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin_local: torch.Tensor,
        rope_rotary_cos_sin_global: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        last_token_ids: torch.Tensor,
        per_layer_inputs: torch.Tensor,
    ) -> Tuple:
        hidden_states, present_key_values = self.model(
            inputs_embeds,
            past_key_values,
            rope_rotary_cos_sin_local,
            rope_rotary_cos_sin_global,
            context_lengths,
            kvcache_start_index,
            per_layer_inputs,
        )
        # Select hidden states for the requested token positions (GatherND with
        # batch_dims=1), matching modeling_default.
        selected_hidden_states = torch.ops.trt.gather_nd(
            hidden_states, last_token_ids)
        logits = self.lm_head(selected_hidden_states).to(torch.float32)
        if self.final_logit_softcapping:
            cap = float(self.final_logit_softcapping)
            logits = torch.tanh(logits / cap) * cap
        return logits, present_key_values

    def onnx_export_spec(self) -> OnnxSpec:
        """Return the dummy inputs / names / dynamic shapes for ONNX export.

        The Gemma 4 input contract extends the dense one with a second RoPE
        table (local + global, of different rotary widths) and a
        ``per_layer_inputs`` tensor (the runtime's per-layer embedding lookup).
        Per-layer caches differ in head dimension between local (256) and global
        (512) layers.
        """
        config = self.config
        gemma4 = self.gemma4_cfg
        na = config.num_hidden_layers
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
        past_key_values_list: List[torch.Tensor] = []
        for i in range(na):
            head_dim = (gemma4.global_head_dim if gemma4.is_global_layer[i]
                        else gemma4.local_head_dim)
            num_kv_heads = (gemma4.num_global_key_value_heads if
                            (gemma4.attention_k_eq_v
                             and gemma4.is_global_layer[i]) else
                            config.num_key_value_heads)
            past_key_values_list.append(
                torch.zeros(batch_size,
                            2,
                            num_kv_heads,
                            past_len,
                            head_dim,
                            dtype=dtype16,
                            device=device))

        local_rotary = int(gemma4.local_head_dim)
        global_rotary = int(gemma4.global_head_dim *
                            gemma4.global_partial_rotary_factor)
        rope_local = torch.zeros(batch_size,
                                 max_pos,
                                 local_rotary,
                                 dtype=torch.float32,
                                 device=device)
        rope_global = torch.zeros(batch_size,
                                  max_pos,
                                  global_rotary,
                                  dtype=torch.float32,
                                  device=device)
        context_lengths = torch.zeros(batch_size,
                                      dtype=torch.int32,
                                      device=device)
        kvcache_start_index = torch.zeros(batch_size,
                                          dtype=torch.int32,
                                          device=device)
        last_token_ids = torch.zeros(batch_size,
                                     1,
                                     dtype=torch.int64,
                                     device=device)
        per_layer_inputs = torch.zeros(batch_size,
                                       seq_len,
                                       na,
                                       gemma4.hidden_size_per_layer_input,
                                       dtype=dtype16,
                                       device=device)

        args = (inputs_embeds, *past_key_values_list, rope_local, rope_global,
                context_lengths, kvcache_start_index, last_token_ids,
                per_layer_inputs)

        input_names = (
            ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(na)] + [
                "rope_rotary_cos_sin_local", "rope_rotary_cos_sin_global",
                "context_lengths", "kvcache_start_index", "last_token_ids",
                "per_layer_inputs"
            ])
        output_names = (["logits"] +
                        [f"present_key_values_{i}" for i in range(na)])

        batch = torch.export.Dim("batch", min=1, max=256)
        seq = torch.export.Dim("seq_len", min=1, max=32768)
        pos = torch.export.Dim("max_pos", min=1, max=32768)
        past = torch.export.Dim("past_len", min=1, max=32768)
        rope_batch = torch.export.Dim("rope_batch", min=1, max=256)
        kv_batch = torch.export.Dim("kv_batch", min=1, max=256)

        all_shapes: list = [{0: batch, 1: seq}]  # inputs_embeds
        for _ in range(na):
            all_shapes.append({0: batch, 3: past})  # past_key_values_i
        all_shapes.append({0: rope_batch, 1: pos})  # rope_local
        all_shapes.append({0: rope_batch, 1: pos})  # rope_global
        all_shapes.append({0: batch})  # context_lengths
        all_shapes.append({0: kv_batch})  # kvcache_start_index
        all_shapes.append({0: batch})  # last_token_ids
        all_shapes.append({0: batch, 1: seq})  # per_layer_inputs

        wrapped = _make_flat_wrapper(self, na)
        wrapped.eval()

        return OnnxSpec(wrapped=wrapped,
                        args=args,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=all_shapes)
