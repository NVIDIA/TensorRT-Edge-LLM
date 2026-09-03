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
Qwen3.5 hybrid causal LM (GatedDeltaNet + gated full attention).

Qwen3.5 alternates between GDN (linear_attention) layers and gated full
attention layers.  Each layer has input_layernorm, mixer, post_attention_
layernorm, and a SwiGLU MLP.

Checkpoint key structure
------------------------
model.embed_tokens.weight                                  - token embedding
model.layers.{i}.input_layernorm.weight                    - pre-mixer RMSNorm
model.layers.{i}.post_attention_layernorm.weight           - pre-MLP RMSNorm
model.layers.{i}.mlp.{gate,up,down}_proj.weight            - SwiGLU MLP

GDN (linear_attention) layers:
model.layers.{i}.linear_attn.in_proj_qkv.weight            - fused QKV [conv_dim, hidden]
model.layers.{i}.linear_attn.in_proj_z.weight              - gate [value_dim, hidden]
model.layers.{i}.linear_attn.in_proj_b.weight              - beta [num_v_heads, hidden]
model.layers.{i}.linear_attn.in_proj_a.weight              - alpha [num_v_heads, hidden]
model.layers.{i}.linear_attn.conv1d.weight                 - [conv_dim, 1, kernel]
model.layers.{i}.linear_attn.A_log                         - [num_v_heads] float32
model.layers.{i}.linear_attn.dt_bias                       - [num_v_heads] float16
model.layers.{i}.linear_attn.norm.weight                   - [value_head_dim]
model.layers.{i}.linear_attn.out_proj.weight               - [hidden, value_dim]

Full attention (gated) layers:
model.layers.{i}.self_attn.q_proj.weight                   - [num_heads*head_dim*2, hidden]
model.layers.{i}.self_attn.k_proj.weight                   - [num_kv_heads*head_dim, hidden]
model.layers.{i}.self_attn.v_proj.weight                   - [num_kv_heads*head_dim, hidden]
model.layers.{i}.self_attn.o_proj.weight                   - [hidden, num_heads*head_dim]
model.layers.{i}.self_attn.q_norm.weight                   - [head_dim]
model.layers.{i}.self_attn.k_norm.weight                   - [head_dim]

model.norm.weight                                          - final RMSNorm
lm_head.weight                                             - output projection
"""

import itertools
import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import LAYER_GDN, GdnConfig, ModelConfig
from ..default.modeling_default import MLP, OnnxSpec, RMSNorm
from ..linear import (AWQLinear, FP16Linear, GPTQLinear,
                      ModelOptAWQPrepackedLinear, NVFP4LinearMethod,
                      ReplicatedLinear, TPMode, is_nvfp4_linear, make_linear)
from ..ops import (KV_PAGE_SIZE, attention_plugin, causal_conv1d,
                   causal_conv1d_with_intermediate, gated_delta_net,
                   gated_delta_net_with_intermediate, int4_gemm_plugin_version)

__all__ = ["Qwen3_5CausalLM"]

logger = logging.getLogger(__name__)

# Projection names in canonical concatenation order.
_GDN_PROJ_NAMES = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")

# NVFP4 scalar scale suffixes that must be identical for fusion.
_NVFP4_SCALAR_SCALE_SUFFIXES = ("input_scale", "weight_scale_2")

# ---------------------------------------------------------------------------
# Qwen3.5 RMSNorm  (residual-weight convention: effective = 1 + weight)
# ---------------------------------------------------------------------------


class Qwen3_5RMSNorm(nn.Module):
    """RMSNorm with Qwen3.5 residual-weight convention.

    HuggingFace ``Qwen3_5RMSNorm`` stores weights initialised to **zero**
    and computes ``(1 + weight) * RMSNorm(x)``.  This differs from the
    standard Llama-style RMSNorm (``weight * RMSNorm(x)`` with weight
    initialised to one).

    Used for ``input_layernorm``, ``post_attention_layernorm``, and the
    final ``model.norm`` — all non-gated norms in Qwen3.5.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(
            torch.zeros(hidden_size, dtype=torch.float16))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        hidden_states = hidden_states.to(input_dtype)
        return (1.0 + self.weight.to(input_dtype)) * hidden_states


# ---------------------------------------------------------------------------
# Conv1d buffer holder (shared with NemotronH pattern)
# ---------------------------------------------------------------------------


class Conv1dBuffers(nn.Module):
    """Holds conv1d weight and bias as plain buffers (not quantized).

    Named ``conv1d`` inside :class:`GdnMixer` so checkpoint keys like
    ``model.layers.N.linear_attn.conv1d.weight`` resolve correctly.
    """

    def __init__(self, conv_dim: int, conv_kernel: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.zeros(conv_dim, 1, conv_kernel))
        self.register_buffer("bias", torch.zeros(conv_dim))


# ---------------------------------------------------------------------------
# GdnMixer  (GatedDeltaNet linear attention)
# ---------------------------------------------------------------------------


class GdnMixer(nn.Module):
    """GatedDeltaNet linear attention computation module.

    Named ``linear_attn`` inside :class:`Qwen3_5DecoderLayer` to match
    checkpoint key prefix ``model.layers.N.linear_attn.*``.
    """

    def __init__(self, config: ModelConfig, gc: GdnConfig,
                 module_prefix: str) -> None:
        super().__init__()
        hidden_size = config.hidden_size

        # Always create 4 separate input projections matching checkpoint keys.
        # A post-load optimization pass (fuse_gdn_input_projections) may
        # replace them with a single in_proj_fused when conditions are met.
        self.in_proj_qkv = make_linear(
            config,
            hidden_size,
            gc.conv_dim,
            bias=False,
            module_name=f"{module_prefix}.in_proj_qkv")
        self.in_proj_z = make_linear(config,
                                     hidden_size,
                                     gc.value_dim,
                                     bias=False,
                                     module_name=f"{module_prefix}.in_proj_z")
        self.in_proj_b = make_linear(config,
                                     hidden_size,
                                     gc.num_value_heads,
                                     bias=False,
                                     module_name=f"{module_prefix}.in_proj_b")
        self.in_proj_a = make_linear(config,
                                     hidden_size,
                                     gc.num_value_heads,
                                     bias=False,
                                     module_name=f"{module_prefix}.in_proj_a")
        self._fused_splits: List[int] = [
            gc.conv_dim, gc.value_dim, gc.num_value_heads, gc.num_value_heads
        ]

        self.conv1d = Conv1dBuffers(gc.conv_dim, gc.conv_kernel)

        # GDN decay and bias: store as FP16 so Cast nodes appear in ONNX.
        self.register_buffer(
            "A_log", torch.zeros(gc.num_value_heads, dtype=torch.float16))
        self.register_buffer(
            "dt_bias", torch.zeros(gc.num_value_heads, dtype=torch.float16))

        # Per-head group norm on output
        self.norm = RMSNorm(gc.value_head_dim, eps=config.rms_norm_eps)

        # Output projection
        self.out_proj = make_linear(config,
                                    gc.value_dim,
                                    hidden_size,
                                    bias=False,
                                    module_name=f"{module_prefix}.out_proj")

        self.num_k_heads = gc.num_key_heads
        self.num_v_heads = gc.num_value_heads
        self.k_dim = gc.key_head_dim
        self.v_dim = gc.value_head_dim
        self.key_dim = gc.key_dim
        self.value_dim = gc.value_dim
        self.conv_dim = gc.conv_dim
        self.conv_kernel = gc.conv_kernel

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        context_lengths: torch.Tensor,
        spec_verify_phase_marker: "torch.Tensor | None" = None,
        tree_parent_ids: "torch.Tensor | None" = None,
        tree_depths: "torch.Tensor | None" = None,
        collect_intermediate_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Input projection(s) -> QKV, gate_z, beta, alpha
        if hasattr(self, "in_proj_fused"):
            fused_out = self.in_proj_fused(hidden_states)
            mixed_qkv, z, b, a = fused_out.split(self._fused_splits, dim=-1)
        elif hasattr(self, "in_proj_qkvz"):
            qkvz_out = self.in_proj_qkvz(hidden_states)
            mixed_qkv, z = qkvz_out.split(self._fused_splits[:2], dim=-1)
            ba_out = self.in_proj_ba(hidden_states)
            b, a = ba_out.split(self._fused_splits[2:], dim=-1)
        else:
            mixed_qkv = self.in_proj_qkv(hidden_states)
            z = self.in_proj_z(hidden_states)
            b = self.in_proj_b(hidden_states)
            a = self.in_proj_a(hidden_states)

        # 2. Causal conv1d (no activation baked in)
        if collect_intermediate_states:
            conv_outputs = causal_conv1d_with_intermediate(
                mixed_qkv,
                self.conv1d.weight,
                self.conv1d.bias,
                conv_state,
                context_lengths,
                stride=1,
                padding=self.conv_kernel - 1,
                dilation=1,
                groups=self.conv_dim,
                spec_verify_phase_marker=spec_verify_phase_marker,
                tree_parent_ids=tree_parent_ids,
                tree_depths=tree_depths,
                use_ddtree_state=tree_parent_ids is not None,
            )
            (mixed_qkv, conv_state_out,
             intermediate_conv_state_out) = conv_outputs
        else:
            mixed_qkv, conv_state_out, _ = causal_conv1d(
                mixed_qkv,
                self.conv1d.weight,
                self.conv1d.bias,
                conv_state,
                context_lengths,
                stride=1,
                padding=self.conv_kernel - 1,
                dilation=1,
                groups=self.conv_dim,
            )
            intermediate_conv_state_out = None
        mixed_qkv = F.silu(mixed_qkv)

        # 3. Split into Q, K, V and reshape to head dims
        query, key, value = mixed_qkv.split(
            [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        # Unflatten only the head dim: naming batch/seq_len here makes
        # torch.export add a ``seq_len != 1`` guard and reject decode shapes.
        query = query.unflatten(-1, (self.num_k_heads, self.k_dim))
        key = key.unflatten(-1, (self.num_k_heads, self.k_dim))
        value = value.unflatten(-1, (self.num_v_heads, self.v_dim))

        # 4. GDN plugin (handles g/beta, QK L2 norm, H/HV head mapping)
        A_log_f32 = self.A_log.to(torch.float32)
        if collect_intermediate_states:
            gdn_outputs = gated_delta_net_with_intermediate(
                query,
                key,
                value,
                a,
                b,
                A_log_f32,
                self.dt_bias,
                recurrent_state,
                context_lengths,
                self.k_dim,
                self.v_dim,
                spec_verify_phase_marker=spec_verify_phase_marker,
                tree_parent_ids=tree_parent_ids,
                tree_depths=tree_depths,
                use_ddtree_state=tree_parent_ids is not None,
            )
            (core_attn_out, recurrent_state_out,
             intermediate_recurrent_state_out) = gdn_outputs
        else:
            core_attn_out, recurrent_state_out, _ = gated_delta_net(
                query, key, value, a, b, A_log_f32, self.dt_bias,
                recurrent_state, context_lengths, self.k_dim, self.v_dim)
            intermediate_recurrent_state_out = None

        # 5. Gated norm: norm FIRST, then gate
        # HF Qwen3_5RMSNormGated: weight * RMSNorm(x) * silu(gate)
        core_attn_out = core_attn_out.reshape(-1, self.v_dim)
        z = z.reshape(-1, self.v_dim)
        core_attn_out = self.norm(core_attn_out)
        core_attn_out = core_attn_out * F.silu(z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        # 6. Output projection
        output = self.out_proj(core_attn_out)

        return (output, conv_state_out, recurrent_state_out,
                intermediate_conv_state_out, intermediate_recurrent_state_out)


# ---------------------------------------------------------------------------
# GatedAttention  (full attention with output gating)
# ---------------------------------------------------------------------------


class GatedAttention(nn.Module):
    """GQA attention with gated output (Qwen3.5 full_attention layers).

    ``q_proj`` packs both query and gate: output is
    ``[batch, seq, num_heads * head_dim * 2]``.  Per head, the first
    ``head_dim`` values are the query, the second ``head_dim`` are the gate.
    After attention: ``output = o_proj(attn_out * sigmoid(gate))``.

    Named ``self_attn`` inside :class:`Qwen3_5DecoderLayer` to match
    checkpoint key prefix ``model.layers.N.self_attn.*``.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        hidden_size = config.hidden_size

        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.attention_scale = config.attention_scaling
        self.enable_fp8_kv_cache = config.quant.kv_cache_quant == "fp8"
        self.sliding_window_size = -1
        module_prefix = f"layers.{layer_idx}.self_attn"

        # q_proj output is doubled: query + gate
        self.q_proj = make_linear(config,
                                  hidden_size,
                                  num_heads * head_dim * 2,
                                  bias=config.attention_bias,
                                  module_name=f"{module_prefix}.q_proj",
                                  tp_mode=TPMode.COL)
        self.k_proj = make_linear(config,
                                  hidden_size,
                                  num_kv_heads * head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{module_prefix}.k_proj",
                                  tp_mode=TPMode.COL)
        self.v_proj = make_linear(config,
                                  hidden_size,
                                  num_kv_heads * head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{module_prefix}.v_proj",
                                  tp_mode=TPMode.COL)
        self._materialize_int4_v = (isinstance(
            self.v_proj, (AWQLinear, GPTQLinear, ModelOptAWQPrepackedLinear))
                                    and int4_gemm_plugin_version() == 2)

        if self.enable_fp8_kv_cache:
            self.q_proj.register_buffer("q_scale", torch.ones(1))
            self.k_proj.register_buffer("k_scale", torch.ones(1))
            self.v_proj.register_buffer("v_scale", torch.ones(1))

        self.o_proj = make_linear(config,
                                  num_heads * head_dim,
                                  hidden_size,
                                  module_name=f"{module_prefix}.o_proj",
                                  tp_mode=TPMode.ROW)

        # Qwen3.5 full attention always has QK norm (residual-weight convention)
        self.q_norm = Qwen3_5RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        attention_mask: "torch.Tensor | None" = None,
        attention_pos_id: "torch.Tensor | None" = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        # Q projection with query/gate split
        q_output = self.q_proj(hidden_states)
        q_output = q_output.view(batch_size, seq_len, self.num_heads,
                                 self.head_dim * 2)
        query_states, gate_states = q_output.chunk(2, dim=-1)
        # query_states, gate_states: [batch, seq, num_heads, head_dim]

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self._materialize_int4_v:
            # A direct V2 plugin -> Concat edge lets TRT virtualize the plugin
            # output as a strided slice, although the plugin's LINEAR output is
            # contiguous.  This live-row mask is exactly one for valid requests
            # and gives TRT a native producer that can honor the Concat stride.
            active_rows = (context_lengths > 0).to(value_states.dtype)
            value_states = value_states * active_rows.reshape(batch_size, 1, 1)

        # QK norm (on reshaped per-head tensors)
        query_states = self.q_norm(query_states)
        query_states = query_states.reshape(batch_size, seq_len,
                                            self.num_heads * self.head_dim)

        key_states = self.k_norm(
            key_states.reshape(batch_size, seq_len, self.num_kv_heads,
                               self.head_dim)).reshape(
                                   batch_size, seq_len,
                                   self.num_kv_heads * self.head_dim)

        enable_tree = attention_mask is not None and attention_pos_id is not None
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
        }
        if enable_tree:
            kwargs["attention_mask"] = attention_mask
            kwargs["attention_pos_id"] = attention_pos_id
        # Always pass qkv_scales so torch.export includes a valid FLOATS
        # value in the FX graph for the unified ONNX translation.
        kwargs["qkv_scales"] = getattr(self, "_qkv_scales_float",
                                       [1.0, 1.0, 1.0])

        attn_output, present_key_value = attention_plugin(
            torch.cat([query_states, key_states, value_states],
                      dim=-1), past_key_value, context_lengths,
            rope_rotary_cos_sin, kvcache_start_index, kv_page_table, **kwargs)

        # attn_output: [batch, seq, num_heads, head_dim]
        # Apply gating: sigmoid(gate) * attn_output
        attn_output = attn_output * torch.sigmoid(gate_states)

        attn_output = attn_output.reshape(batch_size, seq_len,
                                          self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present_key_value


# ---------------------------------------------------------------------------
# Qwen3_5DecoderLayer
# ---------------------------------------------------------------------------


class Qwen3_5DecoderLayer(nn.Module):
    """Single Qwen3.5 decoder layer: pre-norm + mixer + post-norm + MLP.

    For GDN layers the mixer is ``linear_attn`` (GdnMixer).
    For full attention layers the mixer is ``self_attn`` (GatedAttention).
    Both share input_layernorm, post_attention_layernorm, and MLP.
    """

    def __init__(self, config: ModelConfig, gc: GdnConfig, layer_idx: int,
                 layer_type: str) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size,
                                              config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size,
                                                       config.rms_norm_eps)
        self.mlp = MLP(config, layer_idx=layer_idx)

        if layer_type == LAYER_GDN:
            module_prefix = f"layers.{layer_idx}.linear_attn"
            self.linear_attn = GdnMixer(config, gc, module_prefix)
        else:
            self.self_attn = GatedAttention(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        # Attention-specific (ignored by GDN layers)
        past_key_value: "torch.Tensor | None" = None,
        rope_rotary_cos_sin: "torch.Tensor | None" = None,
        context_lengths: "torch.Tensor | None" = None,
        kvcache_start_index: "torch.Tensor | None" = None,
        kv_page_table: "torch.Tensor | None" = None,
        attention_mask: "torch.Tensor | None" = None,
        attention_pos_id: "torch.Tensor | None" = None,
        # GDN-specific (ignored by attention layers)
        conv_state: "torch.Tensor | None" = None,
        recurrent_state: "torch.Tensor | None" = None,
        spec_verify_phase_marker: "torch.Tensor | None" = None,
        tree_parent_ids: "torch.Tensor | None" = None,
        tree_depths: "torch.Tensor | None" = None,
        collect_intermediate_states: bool = False,
    ):
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)

        if self.layer_type == LAYER_GDN:
            (mixer_out, conv_state_out, rec_state_out, intermediate_conv_out,
             intermediate_rec_out) = self.linear_attn(
                 normed,
                 conv_state,
                 recurrent_state,
                 context_lengths,
                 spec_verify_phase_marker=spec_verify_phase_marker,
                 tree_parent_ids=tree_parent_ids,
                 tree_depths=tree_depths,
                 collect_intermediate_states=collect_intermediate_states)
            hidden_states = residual + mixer_out
            residual = hidden_states
            hidden_states = residual + self.mlp(
                self.post_attention_layernorm(hidden_states))
            return (hidden_states, conv_state_out, rec_state_out,
                    intermediate_conv_out, intermediate_rec_out)
        else:
            attn_out, present_kv = self.self_attn(
                normed, past_key_value, rope_rotary_cos_sin, context_lengths,
                kvcache_start_index, kv_page_table, attention_mask,
                attention_pos_id)
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = residual + self.mlp(
                self.post_attention_layernorm(hidden_states))
            return hidden_states, present_kv


# ---------------------------------------------------------------------------
# Qwen3_5Backbone
# ---------------------------------------------------------------------------


class Qwen3_5Backbone(nn.Module):
    """Qwen3.5 hybrid decoder backbone.

    Stored as ``model`` inside :class:`Qwen3_5CausalLM` so parameter keys
    carry the ``model.`` prefix matching checkpoint key prefixes.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        gc = config.gdn_cfg
        assert gc is not None or LAYER_GDN not in config.layer_types, (
            "Qwen3.5 requires gdn_cfg when any layer is a GDN "
            "(linear_attention) layer")
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3_5DecoderLayer(config, gc, layer_idx=i, layer_type=lt)
            for i, lt in enumerate(config.layer_types)
        ])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layer_types: List[str] = config.layer_types

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        conv_states: Tuple[torch.Tensor, ...] = (),
        recurrent_states: Tuple[torch.Tensor, ...] = (),
        attention_mask: "torch.Tensor | None" = None,
        attention_pos_id: "torch.Tensor | None" = None,
        spec_verify_phase_marker: "torch.Tensor | None" = None,
        tree_parent_ids: "torch.Tensor | None" = None,
        tree_depths: "torch.Tensor | None" = None,
        collect_intermediate_states: bool = False,
        dflash_target_layer_ids: "List[int] | None" = None,
    ) -> Tuple[torch.Tensor, Tuple, Tuple, Tuple, Tuple, Tuple, object]:
        hidden_states = inputs_embeds
        present_key_values_list: List[torch.Tensor] = []
        present_conv_states_list: List[torch.Tensor] = []
        present_recurrent_states_list: List[torch.Tensor] = []
        intermediate_conv_states_list: List[torch.Tensor] = []
        intermediate_recurrent_states_list: List[torch.Tensor] = []
        dflash_hidden_list: List[torch.Tensor] = []
        dflash_target_set = set(dflash_target_layer_ids or [])
        attn_idx = 0
        gdn_idx = 0

        for layer_idx, (layer,
                        lt) in enumerate(zip(self.layers, self.layer_types)):
            if lt == LAYER_GDN:
                (hidden_states, conv_out, rec_out, intermediate_conv_out,
                 intermediate_rec_out) = layer(
                     hidden_states,
                     context_lengths=context_lengths,
                     conv_state=conv_states[gdn_idx],
                     recurrent_state=recurrent_states[gdn_idx],
                     spec_verify_phase_marker=spec_verify_phase_marker,
                     tree_parent_ids=tree_parent_ids,
                     tree_depths=tree_depths,
                     collect_intermediate_states=collect_intermediate_states,
                 )
                present_conv_states_list.append(conv_out)
                present_recurrent_states_list.append(rec_out)
                if collect_intermediate_states:
                    intermediate_conv_states_list.append(intermediate_conv_out)
                    intermediate_recurrent_states_list.append(
                        intermediate_rec_out)
                gdn_idx += 1
            else:
                hidden_states, present_kv = layer(
                    hidden_states,
                    past_key_value=past_key_values[attn_idx],
                    rope_rotary_cos_sin=rope_rotary_cos_sin,
                    context_lengths=context_lengths,
                    kvcache_start_index=kvcache_start_index,
                    kv_page_table=kv_page_table,
                    attention_mask=attention_mask,
                    attention_pos_id=attention_pos_id,
                )
                present_key_values_list.append(present_kv)
                attn_idx += 1

            if layer_idx in dflash_target_set:
                dflash_hidden_list.append(hidden_states)

        normed_hidden = self.norm(hidden_states)
        dflash_hidden_concat = (torch.cat(dflash_hidden_list, dim=-1)
                                if dflash_hidden_list else None)

        return (normed_hidden, tuple(present_key_values_list),
                tuple(present_conv_states_list),
                tuple(present_recurrent_states_list),
                tuple(intermediate_conv_states_list),
                tuple(intermediate_recurrent_states_list),
                dflash_hidden_concat)


# ---------------------------------------------------------------------------
# Flat ONNX wrapper
# ---------------------------------------------------------------------------

# Use dummy values > 1 for batch/seq so torch.export marks them as truly
# dynamic (values at the Dim min boundary may be specialized to constants).
_BATCH_SIZE = 2
_SEQ_LEN = 2
_PAST_LEN = 2
_MAX_POS = 4096


def _is_mtp_base_export(config: ModelConfig) -> bool:
    """Return True when exporting the Qwen3.5 hybrid base for MTP verify."""
    return bool(
        getattr(config, "mtp_base", False)
        or getattr(config, "export_component", "") == "mtp_base")


def _is_dflash_base_export(config: ModelConfig) -> bool:
    """Return True when exporting the Qwen3.5 hybrid base for DFlash verify."""
    return bool(getattr(config, "dflash_base", False))


def _is_dspark_base_export(config: ModelConfig) -> bool:
    """Return True when exporting the Qwen3.5 hybrid base for DSpark verify."""
    return bool(getattr(config, "dspark_base", False))


def _is_spec_tree_base_export(config: ModelConfig) -> bool:
    """Return True when exporting DDTree metadata for Qwen3.5 hybrid state.

    DFlash, JetSpec, and MTP tree bases consume the same
    ``tree_parent_ids`` / ``tree_depths`` verify inputs.
    """
    return (bool(getattr(config, "dflash_tree_base", False))
            or bool(getattr(config, "jetspec_tree_base", False))
            or bool(getattr(config, "mtp_tree_base", False)))


def _make_flat_wrapper_hybrid(model: nn.Module,
                              Na: int,
                              Ng: int,
                              mtp_base: bool = False,
                              dflash_base: bool = False,
                              dflash_tree_base: bool = False) -> nn.Module:
    """Build flat forward wrapper for Qwen3.5 hybrid (GDN + attention).

    Extends the transformer wrapper with ``conv_state_i`` and
    ``recurrent_state_i`` inputs and ``present_conv_state_i`` /
    ``present_recurrent_state_i`` outputs.

    ``Na`` = number of attention layers, ``Ng`` = number of GDN layers.
    """
    param_names: List[str] = (
        ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
            "rope_rotary_cos_sin", "context_lengths", "kvcache_start_index",
            "kv_page_table", "last_token_ids"
        ] + [f"conv_state_{i}"
             for i in range(Ng)] + [f"recurrent_state_{i}" for i in range(Ng)])
    spec_base = mtp_base or dflash_base
    if spec_base:
        param_names += [
            "attention_pos_id", "attention_mask", "spec_verify_phase_marker"
        ]
    if dflash_tree_base:
        param_names += ["tree_parent_ids", "tree_depths"]

    past_kv_tuple = "({},)".format(", ".join(
        f"past_key_values_{i}" for i in range(Na))) if Na else "()"
    conv_tuple = "({},)".format(", ".join(f"conv_state_{i}"
                                          for i in range(Ng))) if Ng else "()"
    rec_tuple = "({},)".format(", ".join(f"recurrent_state_{i}"
                                         for i in range(Ng))) if Ng else "()"

    mtp_kwargs = (", attention_pos_id=attention_pos_id"
                  ", attention_mask=attention_mask"
                  ", spec_verify_phase_marker=spec_verify_phase_marker"
                  if spec_base else "")
    if dflash_tree_base:
        mtp_kwargs += (", tree_parent_ids=tree_parent_ids"
                       ", tree_depths=tree_depths")

    if spec_base:
        body = (
            f"    logits, hidden_states, present_key_values, "
            f"present_conv_states, present_recurrent_states, "
            f"intermediate_conv_states, intermediate_recurrent_states = self._model(\n"
            f"        inputs_embeds, {past_kv_tuple}, rope_rotary_cos_sin, "
            f"context_lengths, kvcache_start_index, kv_page_table, "
            f"last_token_ids,\n"
            f"        {conv_tuple}, {rec_tuple}{mtp_kwargs})\n"
            f"    return ((logits, hidden_states) + tuple(present_key_values)\n"
            f"            + tuple(present_conv_states)"
            f" + tuple(present_recurrent_states)"
            f" + tuple(intermediate_conv_states)"
            f" + tuple(intermediate_recurrent_states))\n")
    else:
        body = (
            f"    logits, present_key_values, present_conv_states, "
            f"present_recurrent_states = self._model(\n"
            f"        inputs_embeds, {past_kv_tuple}, rope_rotary_cos_sin, "
            f"context_lengths, kvcache_start_index, kv_page_table, "
            f"last_token_ids,\n"
            f"        {conv_tuple}, {rec_tuple})\n"
            f"    return ((logits,) + tuple(present_key_values)\n"
            f"            + tuple(present_conv_states)"
            f" + tuple(present_recurrent_states))\n")

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
# Post-load optimisation: fuse GDN input projections
# ---------------------------------------------------------------------------


def _can_fuse_nvfp4_scales(mixer: "GdnMixer") -> bool:
    """Return True if all 4 NVFP4 GDN projections have identical scalar scales."""
    for suffix in _NVFP4_SCALAR_SCALE_SUFFIXES:
        tensors = []
        for name in _GDN_PROJ_NAMES:
            proj = getattr(mixer, name, None)
            if proj is None:
                return False
            t = getattr(proj, suffix, None)
            if t is None:
                return False
            tensors.append(t)
        if not all(torch.equal(tensors[0], t) for t in tensors[1:]):
            return False
    return True


def fuse_gdn_input_projections(model: nn.Module) -> int:
    """Post-load optimisation: fuse 4 GDN input projections into one GEMM.

    Iterates over all :class:`GdnMixer` modules.  For each mixer:
    - **FP16**: always fuse (concatenate weights along output dim).
    - **NVFP4**: fuse only if per-tensor scalar scales (``input_scale``,
      ``weight_scale_2``) are identical across all 4 projections.  When
      scales differ, a warning is logged and the layer stays unfused.
    - **Other quant types** (INT4, FP8, …): skip (weight layouts
      incompatible with simple concatenation).

    After fusion the 4 original sub-modules are deleted and replaced by
    a single ``in_proj_fused``.  The forward path auto-detects the fused
    module via ``hasattr(self, "in_proj_fused")``.

    Returns the number of layers fused.
    """

    fused_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, GdnMixer):
            continue
        mixer: GdnMixer = module

        # Check quant type of the first projection.
        first_proj = mixer.in_proj_qkv
        if isinstance(first_proj, FP16Linear):
            pass  # always fusible
        elif is_nvfp4_linear(first_proj):
            if not _can_fuse_nvfp4_scales(mixer):
                # Mixed layout (NVFP4 qkv/z + unquantized FP16 b/a): fuse
                # same-dtype pairs only — qkv+z into one NVFP4 GEMM, b+a
                # into one FP16 GEMM.  Pure concatenation, no re-quantization.
                qkv, zp = mixer.in_proj_qkv, mixer.in_proj_z
                bp, ap = mixer.in_proj_b, mixer.in_proj_a
                pairable = (is_nvfp4_linear(zp) and isinstance(bp, FP16Linear)
                            and isinstance(ap, FP16Linear) and all(
                                torch.equal(getattr(qkv, s), getattr(zp, s))
                                for s in _NVFP4_SCALAR_SCALE_SUFFIXES))
                if pairable:
                    splits = mixer._fused_splits
                    method = NVFP4LinearMethod(
                        group_size=qkv.quant_method.group_size)
                    qkvz = ReplicatedLinear(qkv.in_features,
                                            splits[0] + splits[1],
                                            bias=False,
                                            dtype=torch.float16,
                                            mapping=qkv.mapping,
                                            quant_method=method)
                    qkvz._buffers["weight"] = torch.cat(
                        [qkv.weight, zp.weight], dim=0)
                    qkvz._buffers["weight_scale"] = torch.cat(
                        [qkv.weight_scale, zp.weight_scale], dim=0)
                    qkvz._buffers["weight_scale_2"] = \
                        qkv.weight_scale_2.clone()
                    qkvz._buffers["input_scale"] = qkv.input_scale.clone()
                    ba = FP16Linear(qkv.in_features, splits[2] + splits[3])
                    ba.weight = nn.Parameter(torch.cat(
                        [bp.weight.data, ap.weight.data], dim=0),
                                             requires_grad=False)
                    mixer.in_proj_qkvz = qkvz
                    mixer.in_proj_ba = ba
                    for proj_name in _GDN_PROJ_NAMES:
                        delattr(mixer, proj_name)
                    fused_count += 1
                    logger.debug(
                        "Pair-fused GDN projections (NVFP4 qkvz + FP16 ba) "
                        "for %s", name)
                    continue
                logger.warning(
                    "GDN fusion skipped for %s: NVFP4 scalar scales "
                    "differ across projections. Re-quantize with "
                    "resmoothing enabled to equalise scales.", name)
                continue
        else:
            # INT4, FP8, MXFP8, etc. — not fusible.
            continue

        # --- Fuse: concatenate weights along output dim (dim 0) ----------
        fused_buffers: dict = {}
        proj_modules = [getattr(mixer, n) for n in _GDN_PROJ_NAMES]
        for attr in list(proj_modules[0]._buffers) + list(
                proj_modules[0]._parameters):
            parts = [getattr(p, attr) for p in proj_modules]
            if parts[0] is None:
                continue
            if parts[0].dim() >= 1:
                # Per-output-channel: concat along dim 0.
                fused_buffers[attr] = torch.cat(parts, dim=0)
            else:
                # Scalar / per-tensor: take first (already verified equal
                # for NVFP4; identical for FP16 which has no scales).
                fused_buffers[attr] = parts[0]

        # Build a fused linear with correct type.
        fused_out_dim = sum(mixer._fused_splits)
        in_features = first_proj.in_features
        if is_nvfp4_linear(first_proj):
            method = NVFP4LinearMethod(
                group_size=first_proj.quant_method.group_size)
            fused_linear = ReplicatedLinear(in_features,
                                            fused_out_dim,
                                            bias=False,
                                            dtype=torch.float16,
                                            mapping=first_proj.mapping,
                                            quant_method=method)
        else:
            fused_linear = FP16Linear(in_features, fused_out_dim)

        # Assign fused buffers/params.
        for attr, tensor in fused_buffers.items():
            if attr in fused_linear._buffers:
                fused_linear._buffers[attr] = tensor
            elif attr in fused_linear._parameters:
                fused_linear._parameters[attr] = nn.Parameter(
                    tensor, requires_grad=False)
            else:
                setattr(fused_linear, attr, tensor)

        # Replace: add fused, delete originals.
        mixer.in_proj_fused = fused_linear
        for proj_name in _GDN_PROJ_NAMES:
            delattr(mixer, proj_name)

        fused_count += 1
        logger.debug("Fused GDN input projections for %s", name)

    if fused_count:
        logger.info("Fused GDN input projections in %d layer(s)", fused_count)
    return fused_count


# ---------------------------------------------------------------------------
# Qwen3_5CausalLM
# ---------------------------------------------------------------------------


class Qwen3_5CausalLM(nn.Module):
    """Qwen3.5 hybrid causal LM: backbone + lm_head.

    The inner backbone is stored as attribute ``model`` so parameter keys
    carry the ``model.`` prefix matching checkpoint key prefixes.
    ``lm_head`` maps directly to ``lm_head.weight``.
    """

    # Dtypes of the GDN state tensors this model feeds the ONNX graph.
    # These drive (a) the dummy tensor dtypes in ``export_onnx`` and
    # (b) the ``recurrent_state_dtype`` / ``conv_state_dtype`` strings written
    # into ``config.json`` (see checkpoint_utils). They must stay in sync, so
    # the single source of truth is this class attribute, not a separate table.
    # Dtypes are dictated by the ``trt_edgellm::gated_delta_net`` plugin schema:
    # ``h0_source`` is typed ``T_A = tensor(float)`` (fp32), ``conv_state``
    # follows ``T = tensor(float16)``.
    RECURRENT_STATE_DTYPE = torch.float32
    CONV_STATE_DTYPE = torch.float16

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3_5Backbone(config)
        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")

    def tie_weights(self) -> None:
        """Clone embed_tokens.weight into lm_head when tie_word_embeddings=True."""
        if not self.config.tie_word_embeddings:
            return
        if not isinstance(self.lm_head, FP16Linear):
            return
        embed_weight = self.model.embed_tokens.weight
        self.lm_head.weight = nn.Parameter(embed_weight.detach().clone(),
                                           requires_grad=False)

    def onnx_export_spec(self) -> OnnxSpec:
        """Return all model-specific parameters needed for ONNX export."""
        config = self.config
        gc = config.gdn_cfg
        Na = config.num_attn_layers
        Ng = config.num_gdn_layers
        assert gc is not None or Ng == 0, (
            "Qwen3.5 requires gdn_cfg when any layer is GDN")
        mtp_base = _is_mtp_base_export(config)
        dflash_base = _is_dflash_base_export(config)
        dspark_base = _is_dspark_base_export(config)
        target_hidden_base = dflash_base or dspark_base
        spec_tree_base = _is_spec_tree_base_export(config)
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
        kv_dtype = (torch.float8_e4m3fn
                    if config.quant.kv_cache_quant == "fp8" else dtype16)
        # Paged KV pool binding: [2, num_pages, KV_PAGE_SIZE, num_kv_heads, head_dim].
        past_key_values_list: List[torch.Tensor] = [
            torch.zeros(2,
                        1,
                        KV_PAGE_SIZE,
                        config.num_key_value_heads,
                        config.head_dim,
                        dtype=kv_dtype,
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
        spec_base = mtp_base or target_hidden_base
        select_len = 2 if spec_base else 1
        last_token_ids = torch.zeros(batch_size,
                                     select_len,
                                     dtype=torch.int64,
                                     device=device)

        conv_states: List[torch.Tensor] = [
            torch.zeros(batch_size,
                        gc.conv_dim,
                        gc.conv_kernel,
                        dtype=self.CONV_STATE_DTYPE,
                        device=device) for _ in range(Ng)
        ]
        recurrent_states: List[torch.Tensor] = [
            torch.zeros(batch_size,
                        gc.num_value_heads,
                        gc.key_head_dim,
                        gc.value_head_dim,
                        dtype=self.RECURRENT_STATE_DTYPE,
                        device=device) for _ in range(Ng)
        ]

        args = (inputs_embeds, *past_key_values_list, rope_rotary_cos_sin,
                context_lengths, kvcache_start_index, kv_page_table,
                last_token_ids, *conv_states, *recurrent_states)

        input_names = (
            ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
                "rope_rotary_cos_sin", "context_lengths",
                "kvcache_start_index", "kv_page_table", "last_token_ids"
            ] + [f"conv_state_{i}" for i in range(Ng)] +
            [f"recurrent_state_{i}" for i in range(Ng)])
        output_names = (["logits"] +
                        [f"present_key_values_{i}" for i in range(Na)] +
                        [f"present_conv_state_{i}" for i in range(Ng)] +
                        [f"present_recurrent_state_{i}" for i in range(Ng)])

        batch = torch.export.Dim("batch", min=1, max=256)
        seq = torch.export.Dim("seq_len", min=1, max=32768)
        pos = torch.export.Dim("max_pos", min=1, max=32768)
        rope_batch = torch.export.Dim("rope_batch", min=1, max=256)
        kv_batch = torch.export.Dim("kv_batch", min=1, max=256)
        page_batch = torch.export.Dim("page_batch", min=1, max=256)
        max_pages = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
        num_pages = torch.export.Dim("num_pages", min=1, max=1048576)
        num_selected = torch.export.Dim("num_selected", min=1,
                                        max=256) if spec_base else None

        all_shapes: list = [{0: batch, 1: seq}]  # inputs_embeds
        for _ in range(Na):
            all_shapes.append({1:
                               num_pages})  # past_key_values_i (pool-shaped)
        all_shapes.append({0: rope_batch, 1: pos})  # rope_rotary_cos_sin
        all_shapes.append({0: batch})  # context_lengths
        all_shapes.append({0: kv_batch})  # kvcache_start_index
        all_shapes.append({0: page_batch, 2: max_pages})  # kv_page_table
        if spec_base:
            all_shapes.append({0: batch, 1: num_selected})  # last_token_ids
        else:
            all_shapes.append({0: batch})  # last_token_ids
        for _ in range(Ng):
            all_shapes.append({0: batch})  # conv_state_i
        for _ in range(Ng):
            all_shapes.append({0: batch})  # recurrent_state_i

        if spec_base:
            attention_pos_id = torch.zeros(batch_size,
                                           seq_len,
                                           dtype=torch.int32,
                                           device=device)
            attention_mask = torch.zeros(batch_size,
                                         seq_len,
                                         seq_len + past_len,
                                         dtype=torch.int32,
                                         device=device)
            spec_verify_phase_marker = torch.zeros(1,
                                                   dtype=torch.int32,
                                                   device=device)
            args = args + (attention_pos_id, attention_mask,
                           spec_verify_phase_marker)
            input_names = input_names + [
                "attention_pos_id", "attention_mask",
                "spec_verify_phase_marker"
            ]
            output_names = (
                ["logits", "hidden_states"] +
                [f"present_key_values_{i}" for i in range(Na)] +
                [f"present_conv_state_{i}" for i in range(Ng)] +
                [f"present_recurrent_state_{i}" for i in range(Ng)] +
                [f"intermediate_conv_state_{i}" for i in range(Ng)] +
                [f"intermediate_recurrent_state_{i}" for i in range(Ng)])

            verify_seq = torch.export.Dim("verify_seq_len", min=1, max=32768)
            mask_kv_len = torch.export.Dim("mask_kv_len", min=1, max=65536)
            all_shapes.append({0: batch, 1: verify_seq})  # attention_pos_id
            all_shapes.append({
                0: batch,
                1: verify_seq,
                2: mask_kv_len
            })  # attention_mask
            all_shapes.append({0: torch.export.Dim.AUTO
                               })  # spec_verify_phase_marker
            if spec_tree_base:
                tree_parent_ids = torch.zeros(batch_size,
                                              seq_len,
                                              dtype=torch.int32,
                                              device=device)
                tree_depths = torch.zeros(batch_size,
                                          seq_len,
                                          dtype=torch.int32,
                                          device=device)
                args = args + (tree_parent_ids, tree_depths)
                input_names = input_names + ["tree_parent_ids", "tree_depths"]
                all_shapes.append({0: batch, 1: verify_seq})  # tree_parent_ids
                all_shapes.append({0: batch, 1: verify_seq})  # tree_depths

        wrapped = _make_flat_wrapper_hybrid(self,
                                            Na,
                                            Ng,
                                            mtp_base=mtp_base,
                                            dflash_base=target_hidden_base,
                                            dflash_tree_base=spec_tree_base)
        wrapped.eval()

        return OnnxSpec(wrapped=wrapped,
                        args=args,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=all_shapes)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        last_token_ids: torch.Tensor,
        conv_states: Tuple[torch.Tensor, ...] = (),
        recurrent_states: Tuple[torch.Tensor, ...] = (),
        attention_pos_id: "torch.Tensor | None" = None,
        attention_mask: "torch.Tensor | None" = None,
        spec_verify_phase_marker: "torch.Tensor | None" = None,
        tree_parent_ids: "torch.Tensor | None" = None,
        tree_depths: "torch.Tensor | None" = None,
    ) -> Tuple:
        mtp_base = _is_mtp_base_export(self.config)
        dflash_base = _is_dflash_base_export(self.config)
        dspark_base = _is_dspark_base_export(self.config)
        target_hidden_base = dflash_base or dspark_base
        target_layer_ids = (
            self.config.dspark_target_layer_ids if dspark_base else
            self.config.dflash_target_layer_ids if dflash_base else None)
        (hidden_states, present_key_values, present_conv_states,
         present_recurrent_states, intermediate_conv_states,
         intermediate_recurrent_states, dflash_hidden_concat) = self.model(
             inputs_embeds,
             past_key_values,
             rope_rotary_cos_sin,
             context_lengths,
             kvcache_start_index,
             kv_page_table,
             conv_states,
             recurrent_states,
             attention_mask=attention_mask,
             attention_pos_id=attention_pos_id,
             spec_verify_phase_marker=spec_verify_phase_marker,
             tree_parent_ids=tree_parent_ids,
             tree_depths=tree_depths,
             collect_intermediate_states=(mtp_base or target_hidden_base),
             dflash_target_layer_ids=target_layer_ids,
         )
        # Select hidden states for specified token positions before lm_head.
        selected_hidden_states = torch.ops.trt.gather_nd(
            hidden_states, last_token_ids)

        logits = self.lm_head(selected_hidden_states).to(torch.float32)
        if target_hidden_base:
            return (logits, dflash_hidden_concat, present_key_values,
                    present_conv_states, present_recurrent_states,
                    intermediate_conv_states, intermediate_recurrent_states)
        if mtp_base:
            return (logits, hidden_states, present_key_values,
                    present_conv_states, present_recurrent_states,
                    intermediate_conv_states, intermediate_recurrent_states)
        return (logits, present_key_values, present_conv_states,
                present_recurrent_states)
