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
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import LAYER_GDN, GdnConfig, ModelConfig
from ...dflash import DFlashVersion
from ..default.modeling_default import (MLP, OnnxSpec, RMSNorm,
                                        _concat_hidden_in_provider_order)
from ..linear import (ColumnParallelLinear, FP16Linear, NVFP4LinearMethod,
                      ReplicatedLinear, TPMode, is_int4_linear,
                      is_nvfp4_a16_linear, is_nvfp4_linear, make_linear)
from ..ops import (KV_PAGE_SIZE, attention_plugin, causal_conv1d,
                   causal_conv1d_with_intermediate, gated_delta_net,
                   gated_delta_net_with_intermediate, qkv_concat)

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
        weight = self.weight.to(input_dtype)
        return (torch.ones_like(weight) + weight) * hidden_states


# ---------------------------------------------------------------------------
# Conv1d buffer holder (shared with NemotronH pattern)
# ---------------------------------------------------------------------------


class Conv1dBuffers(nn.Module):
    """Holds conv1d weight and bias as plain buffers (not quantized).

    Named ``conv1d`` inside :class:`GdnMixer` so checkpoint keys like
    ``model.layers.N.linear_attn.conv1d.weight`` resolve correctly.
    """

    def __init__(self, conv_dim: int, conv_kernel: int,
                 tp_split_sizes: Tuple[int, ...]) -> None:
        super().__init__()
        self.register_buffer("weight", torch.zeros(conv_dim, 1, conv_kernel))
        self.register_buffer("bias", torch.zeros(conv_dim))
        self.tp_split_sizes = tp_split_sizes or None

    def tp_split_dim(self, attr: str) -> Optional[int]:
        """Shard Q, K, and V convolution channels independently under TP."""
        if self.tp_split_sizes is not None and attr in ("weight", "bias"):
            return 0
        return None


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
            module_name=f"{module_prefix}.in_proj_qkv",
            tp_mode=TPMode.COL)
        self.in_proj_z = make_linear(config,
                                     hidden_size,
                                     gc.value_dim,
                                     bias=False,
                                     module_name=f"{module_prefix}.in_proj_z",
                                     tp_mode=TPMode.COL)
        self.in_proj_b = make_linear(config,
                                     hidden_size,
                                     gc.num_value_heads,
                                     bias=False,
                                     module_name=f"{module_prefix}.in_proj_b",
                                     tp_mode=TPMode.COL)
        self.in_proj_a = make_linear(config,
                                     hidden_size,
                                     gc.num_value_heads,
                                     bias=False,
                                     module_name=f"{module_prefix}.in_proj_a",
                                     tp_mode=TPMode.COL)
        qkv_split_sizes = (gc.key_dim, gc.key_dim, gc.value_dim)
        if config.tp_size > 1:
            self.in_proj_qkv.tp_split_sizes = qkv_split_sizes
        self._fused_splits: List[int] = [
            gc.conv_dim, gc.value_dim, gc.num_value_heads, gc.num_value_heads
        ]

        self.conv1d = Conv1dBuffers(
            gc.conv_dim, gc.conv_kernel,
            qkv_split_sizes if config.tp_size > 1 else ())

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
                                    module_name=f"{module_prefix}.out_proj",
                                    tp_mode=TPMode.ROW)

        self.tp_mode = (TPMode.COL
                        if config.tp_size > 1 else TPMode.REPLICATED)

        self.num_k_heads = gc.num_key_heads
        self.num_v_heads = gc.num_value_heads
        self.k_dim = gc.key_head_dim
        self.v_dim = gc.value_head_dim
        self.key_dim = gc.key_dim
        self.value_dim = gc.value_dim
        self.conv_dim = gc.conv_dim
        self.conv_kernel = gc.conv_kernel

    def tp_split_dim(self, attr: str) -> Optional[int]:
        """Shard per-value-head state parameters under TP."""
        if self.tp_mode == TPMode.COL and attr in ("A_log", "dt_bias"):
            return 0
        return None

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        context_lengths: torch.Tensor,
        execution_phase_marker: "torch.Tensor | None" = None,
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
                execution_phase_marker=execution_phase_marker,
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
                execution_phase_marker=execution_phase_marker,
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

    def forward_ragged(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        query_start_offsets: torch.Tensor,
        query_lengths: torch.Tensor,
        state_indices: torch.Tensor,
        execution_phase_marker: torch.Tensor,
        context_sequence_count_carrier: torch.Tensor,
        tree_parent_ids: "torch.Tensor | None" = None,
        tree_depths: "torch.Tensor | None" = None,
        use_intermediate_state: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        if hasattr(self, "in_proj_fused"):
            fused_out = self.in_proj_fused(hidden_states)
            mixed_qkv, z, b, a = fused_out.split(self._fused_splits, dim=-1)
        elif hasattr(self, "in_proj_qkvz"):
            qkvz_out = self.in_proj_qkvz(hidden_states)
            mixed_qkv, z = qkvz_out.split(self._fused_splits[:2], dim=-1)
            b, a = self.in_proj_ba(hidden_states).split(self._fused_splits[2:],
                                                        dim=-1)
        else:
            mixed_qkv = self.in_proj_qkv(hidden_states)
            z = self.in_proj_z(hidden_states)
            b = self.in_proj_b(hidden_states)
            a = self.in_proj_a(hidden_states)

        if not use_intermediate_state:
            mixed_qkv, conv_state_out, intermediate_conv_state = causal_conv1d(
                mixed_qkv,
                self.conv1d.weight,
                self.conv1d.bias,
                conv_state,
                query_lengths,
                stride=1,
                padding=self.conv_kernel - 1,
                dilation=1,
                groups=self.conv_dim,
                query_start_offsets=query_start_offsets,
                state_indices=state_indices,
                execution_phase_marker=execution_phase_marker,
                context_sequence_count_carrier=context_sequence_count_carrier)
        else:
            mixed_qkv, conv_state_out, intermediate_conv_state = causal_conv1d_with_intermediate(
                mixed_qkv,
                self.conv1d.weight,
                self.conv1d.bias,
                conv_state,
                query_lengths,
                query_start_offsets,
                state_indices,
                stride=1,
                padding=self.conv_kernel - 1,
                dilation=1,
                groups=self.conv_dim,
                execution_phase_marker=execution_phase_marker,
                context_sequence_count_carrier=context_sequence_count_carrier,
                tree_parent_ids=tree_parent_ids,
                tree_depths=tree_depths,
                use_ddtree_state=tree_parent_ids is not None)
        mixed_qkv = F.silu(mixed_qkv)
        query, key, value = mixed_qkv.split(
            [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.unflatten(-1, (self.num_k_heads, self.k_dim))
        key = key.unflatten(-1, (self.num_k_heads, self.k_dim))
        value = value.unflatten(-1, (self.num_v_heads, self.v_dim))
        if not use_intermediate_state:
            core_attn_out, recurrent_state_out, intermediate_recurrent_state = gated_delta_net(
                query,
                key,
                value,
                a,
                b,
                self.A_log.to(torch.float32),
                self.dt_bias,
                recurrent_state,
                query_lengths,
                self.k_dim,
                self.v_dim,
                query_start_offsets=query_start_offsets,
                state_indices=state_indices,
                execution_phase_marker=execution_phase_marker,
                context_sequence_count_carrier=context_sequence_count_carrier)
        else:
            core_attn_out, recurrent_state_out, intermediate_recurrent_state = gated_delta_net_with_intermediate(
                query,
                key,
                value,
                a,
                b,
                self.A_log.to(torch.float32),
                self.dt_bias,
                recurrent_state,
                query_lengths,
                query_start_offsets,
                state_indices,
                self.k_dim,
                self.v_dim,
                execution_phase_marker=execution_phase_marker,
                context_sequence_count_carrier=context_sequence_count_carrier,
                tree_parent_ids=tree_parent_ids,
                tree_depths=tree_depths,
                use_ddtree_state=tree_parent_ids is not None)
        core_attn_out = self.norm(core_attn_out.reshape(-1, self.v_dim))
        core_attn_out = core_attn_out * F.silu(z.reshape(-1, self.v_dim))
        core_attn_out = core_attn_out.reshape(hidden_states.shape[0],
                                              self.value_dim)
        return (self.out_proj(core_attn_out), conv_state_out,
                recurrent_state_out, intermediate_conv_state,
                intermediate_recurrent_state)


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
        self.skip_softmax_scale_factor = config.skip_softmax_scale_factor
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
        # INT4 / NVFP4-A16 QKV packs through qkv_concat to keep the Concat from
        # being elided on SM>=100 (which corrupts the strided V write).
        self._uses_qkv_concat = any(
            is_int4_linear(proj) or is_nvfp4_a16_linear(proj)
            for proj in (self.q_proj, self.k_proj, self.v_proj))
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
        skip_softmax_scale: "torch.Tensor | None" = None,
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
            "skip_softmax_scale_factor": self.skip_softmax_scale_factor,
        }
        if skip_softmax_scale is not None and self.skip_softmax_scale_factor > 0.0:
            kwargs["skip_softmax_scale"] = skip_softmax_scale
        if enable_tree:
            kwargs["attention_mask"] = attention_mask
            kwargs["attention_pos_id"] = attention_pos_id
        # Always pass qkv_scales so torch.export includes a valid FLOATS
        # value in the FX graph for the unified ONNX translation.
        kwargs["qkv_scales"] = getattr(self, "_qkv_scales_float",
                                       [1.0, 1.0, 1.0])

        qkv = (qkv_concat(query_states, key_states, value_states)
               if self._uses_qkv_concat else torch.cat(
                   [query_states, key_states, value_states], dim=-1))
        attn_output, present_key_value = attention_plugin(
            qkv, past_key_value, context_lengths, rope_rotary_cos_sin,
            kvcache_start_index, kv_page_table, **kwargs)

        # attn_output: [batch, seq, num_heads, head_dim]
        # Apply gating: sigmoid(gate) * attn_output
        attn_output = attn_output * torch.sigmoid(gate_states)

        attn_output = attn_output.reshape(batch_size, seq_len,
                                          self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present_key_value

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
        skip_softmax_scale: "torch.Tensor | None" = None,
        attention_position_ids: "torch.Tensor | None" = None,
        packed_attention_mask: "torch.Tensor | None" = None,
        tree_parent_ids: "torch.Tensor | None" = None,
        tree_depths: "torch.Tensor | None" = None,
        valid_tree_counts: "torch.Tensor | None" = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_tokens = hidden_states.shape[0]
        q_output = self.q_proj(hidden_states).reshape(num_tokens,
                                                      self.num_heads,
                                                      self.head_dim * 2)
        query_states, gate_states = q_output.chunk(2, dim=-1)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = self.q_norm(query_states).reshape(
            num_tokens, self.num_heads * self.head_dim)
        key_states = self.k_norm(
            key_states.reshape(num_tokens, self.num_kv_heads,
                               self.head_dim)).reshape(
                                   num_tokens,
                                   self.num_kv_heads * self.head_dim)
        qkv = (qkv_concat(query_states, key_states, value_states)
               if self._uses_qkv_concat else torch.cat(
                   [query_states, key_states, value_states], dim=-1))
        attn_output, present_key_value = attention_plugin(
            qkv,
            past_key_value,
            query_lengths,
            rope_rotary_cos_sin,
            past_lengths,
            kv_page_table,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            sliding_window_size=self.sliding_window_size,
            enable_tree_attention=packed_attention_mask is not None,
            enable_fp8_kv_cache=self.enable_fp8_kv_cache,
            attention_scale=self.attention_scale,
            enable_context_mask_selector=False,
            enable_vision_block_attention=False,
            skip_softmax_scale_factor=self.skip_softmax_scale_factor,
            qkv_scales=getattr(self, "_qkv_scales_float", [1.0, 1.0, 1.0]),
            query_start_offsets=query_start_offsets,
            attention_sequence_lengths=attention_sequence_lengths,
            execution_phase_marker=execution_phase_marker,
            context_sequence_count_carrier=context_sequence_count_carrier,
            attention_pos_id=attention_position_ids,
            attention_mask=packed_attention_mask,
            skip_softmax_scale=(skip_softmax_scale
                                if self.skip_softmax_scale_factor > 0.0 else
                                None))
        attn_output = attn_output * torch.sigmoid(gate_states)
        return self.o_proj(
            attn_output.reshape(num_tokens, self.num_heads *
                                self.head_dim)), present_key_value


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
        skip_softmax_scale: "torch.Tensor | None" = None,
        # GDN-specific (ignored by attention layers)
        conv_state: "torch.Tensor | None" = None,
        recurrent_state: "torch.Tensor | None" = None,
        execution_phase_marker: "torch.Tensor | None" = None,
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
                 execution_phase_marker=execution_phase_marker,
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
                normed,
                past_key_value,
                rope_rotary_cos_sin,
                context_lengths,
                kvcache_start_index,
                kv_page_table,
                attention_mask,
                attention_pos_id,
                skip_softmax_scale=skip_softmax_scale)
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = residual + self.mlp(
                self.post_attention_layernorm(hidden_states))
            return hidden_states, present_kv

    def forward_ragged(self, hidden_states: torch.Tensor, **kwargs):
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        if self.layer_type == LAYER_GDN:
            (mixer_out, conv_out, recurrent_out, intermediate_conv_out,
             intermediate_recurrent_out) = self.linear_attn.forward_ragged(
                 normed, kwargs["conv_state"], kwargs["recurrent_state"],
                 kwargs["query_start_offsets"], kwargs["query_lengths"],
                 kwargs["state_indices"], kwargs["execution_phase_marker"],
                 kwargs["context_sequence_count_carrier"],
                 kwargs.get("tree_parent_ids"), kwargs.get("tree_depths"),
                 kwargs.get("use_intermediate_state", False))
            mixer_out = mixer_out.reshape(-1, residual.shape[-1])
            hidden_states = residual + mixer_out
            hidden_states = hidden_states + self.mlp(
                self.post_attention_layernorm(hidden_states))
            return (hidden_states, conv_out, recurrent_out,
                    intermediate_conv_out, intermediate_recurrent_out)
        attn_out, present_kv = self.self_attn.forward_ragged(
            normed, kwargs["past_key_value"], kwargs["rope_rotary_cos_sin"],
            kwargs["positions"], kwargs["query_start_offsets"],
            kwargs["query_lengths"], kwargs["past_lengths"],
            kwargs["attention_sequence_lengths"], kwargs["state_indices"],
            kwargs["execution_phase_marker"],
            kwargs["context_sequence_count_carrier"], kwargs["kv_page_table"],
            kwargs.get("skip_softmax_scale"),
            kwargs.get("attention_position_ids"),
            kwargs.get("packed_attention_mask"), kwargs.get("tree_parent_ids"),
            kwargs.get("tree_depths"), kwargs.get("valid_tree_counts"))
        attn_out = attn_out.reshape(-1, residual.shape[-1])
        hidden_states = residual + attn_out
        hidden_states = hidden_states + self.mlp(
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
        skip_softmax_scale: "torch.Tensor | None" = None,
        execution_phase_marker: "torch.Tensor | None" = None,
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
        dflash_hidden_by_layer: dict[int, torch.Tensor] = {}
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
                     execution_phase_marker=execution_phase_marker,
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
                    skip_softmax_scale=skip_softmax_scale,
                )
                present_key_values_list.append(present_kv)
                attn_idx += 1

            if layer_idx in dflash_target_set:
                dflash_hidden_by_layer[layer_idx] = hidden_states

        normed_hidden = self.norm(hidden_states)
        dflash_hidden_concat = _concat_hidden_in_provider_order(
            dflash_hidden_by_layer, dflash_target_layer_ids)

        return (normed_hidden, tuple(present_key_values_list),
                tuple(present_conv_states_list),
                tuple(present_recurrent_states_list),
                tuple(intermediate_conv_states_list),
                tuple(intermediate_recurrent_states_list),
                dflash_hidden_concat)

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
        conv_states: Tuple[torch.Tensor, ...],
        recurrent_states: Tuple[torch.Tensor, ...],
        skip_softmax_scale: "torch.Tensor | None" = None,
        attention_position_ids: "torch.Tensor | None" = None,
        packed_attention_mask: "torch.Tensor | None" = None,
        tree_parent_ids: "torch.Tensor | None" = None,
        tree_depths: "torch.Tensor | None" = None,
        valid_tree_counts: "torch.Tensor | None" = None,
        collect_intermediate_states: bool = False,
        target_layer_ids: "List[int] | None" = None,
    ) -> Tuple[torch.Tensor, Tuple, Tuple, Tuple, Tuple, Tuple,
               "torch.Tensor | None"]:
        hidden_states = inputs_embeds
        present_key_values = []
        present_conv_states = []
        present_recurrent_states = []
        intermediate_conv_states = []
        intermediate_recurrent_states = []
        target_hidden_by_layer: dict[int, torch.Tensor] = {}
        target_layer_set = set(target_layer_ids or [])
        attn_idx = 0
        gdn_idx = 0
        common = {
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
            "skip_softmax_scale": skip_softmax_scale,
            "attention_position_ids": attention_position_ids,
            "packed_attention_mask": packed_attention_mask,
            "tree_parent_ids": tree_parent_ids,
            "tree_depths": tree_depths,
            "valid_tree_counts": valid_tree_counts,
            "use_intermediate_state": collect_intermediate_states,
        }
        for layer_idx, (layer, layer_type) in enumerate(
                zip(self.layers, self.layer_types)):
            if layer_type == LAYER_GDN:
                (hidden_states, conv_out, recurrent_out, intermediate_conv_out,
                 intermediate_recurrent_out) = layer.forward_ragged(
                     hidden_states,
                     conv_state=conv_states[gdn_idx],
                     recurrent_state=recurrent_states[gdn_idx],
                     **common)
                present_conv_states.append(conv_out)
                present_recurrent_states.append(recurrent_out)
                if collect_intermediate_states:
                    intermediate_conv_states.append(intermediate_conv_out)
                    intermediate_recurrent_states.append(
                        intermediate_recurrent_out)
                gdn_idx += 1
            else:
                hidden_states, present_kv = layer.forward_ragged(
                    hidden_states,
                    past_key_value=past_key_values[attn_idx],
                    **common)
                present_key_values.append(present_kv)
                attn_idx += 1
            self._capture_ragged_layer_output(layer_idx, hidden_states)
            if layer_idx in target_layer_set:
                target_hidden_by_layer[layer_idx] = hidden_states
        target_hidden_concat = _concat_hidden_in_provider_order(
            target_hidden_by_layer, target_layer_ids)
        return (self.norm(hidden_states), tuple(present_key_values),
                tuple(present_conv_states), tuple(present_recurrent_states),
                tuple(intermediate_conv_states),
                tuple(intermediate_recurrent_states), target_hidden_concat)

    def _capture_ragged_layer_output(self, layer_idx: int,
                                     hidden_states: torch.Tensor) -> None:
        del layer_idx, hidden_states


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


def _is_jetspec_base_export(config: ModelConfig) -> bool:
    """Return True when exporting the Qwen3.5 hybrid base for JetSpec verify."""
    return bool(getattr(config, "jetspec_base", False))


def _is_spec_tree_base_export(config: ModelConfig) -> bool:
    """Return True when exporting DDTree metadata for Qwen3.5 hybrid state.

    DFlash, JetSpec, MTP, and DSpark tree bases consume the same
    ``tree_parent_ids`` / ``tree_depths`` verify inputs.
    """
    dflash2_base = (bool(getattr(config, "dflash_base", False)) and getattr(
        config, "dflash_version", DFlashVersion.V1) == DFlashVersion.V2)
    return (dflash2_base or bool(getattr(config, "dflash_tree_base", False))
            or bool(getattr(config, "jetspec_tree_base", False))
            or bool(getattr(config, "mtp_tree_base", False))
            or bool(getattr(config, "dspark_tree_base", False)))


def _make_flat_wrapper_hybrid_ragged(model: nn.Module, Na: int, Ng: int,
                                     spec_base: bool, tree_attention: bool,
                                     tree_metadata: bool,
                                     emit_accept_hidden: bool) -> nn.Module:
    """Build the flat wrapper for the unified hybrid token-major ABI."""
    param_names: List[str] = (
        ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
            "rope_rotary_cos_sin", "positions", "query_start_offsets",
            "query_lengths", "past_lengths", "attention_sequence_lengths",
            "state_indices", "execution_phase_marker",
            "context_sequence_count_carrier", "kv_page_table", "logits_indices"
        ] + [f"conv_state_{i}" for i in range(Ng)] +
        [f"recurrent_state_{i}" for i in range(Ng)] + ["skip_softmax_scale"])
    if tree_attention:
        param_names += ["attention_position_ids", "packed_attention_mask"]
    if tree_metadata:
        param_names += ["tree_parent_ids", "tree_depths", "valid_tree_counts"]
    past_kv_tuple = "({},)".format(", ".join(
        f"past_key_values_{i}" for i in range(Na))) if Na else "()"
    conv_tuple = "({},)".format(", ".join(f"conv_state_{i}"
                                          for i in range(Ng))) if Ng else "()"
    rec_tuple = "({},)".format(", ".join(f"recurrent_state_{i}"
                                         for i in range(Ng))) if Ng else "()"
    tree_kwargs = (", attention_position_ids=attention_position_ids"
                   ", packed_attention_mask=packed_attention_mask"
                   if tree_attention else "")
    tree_kwargs += (", tree_parent_ids=tree_parent_ids"
                    ", tree_depths=tree_depths"
                    ", valid_tree_counts=valid_tree_counts"
                    if tree_metadata else "")
    unpack = (
        "    logits, hidden_states, accept_hidden_states, present_key_values, "
        "present_conv_states, present_recurrent_states, "
        "intermediate_conv_states, intermediate_recurrent_states = outputs\n"
        if emit_accept_hidden else
        "    logits, hidden_states, present_key_values, present_conv_states, "
        "present_recurrent_states, intermediate_conv_states, "
        "intermediate_recurrent_states = outputs\n")
    accept_result = (" + (accept_hidden_states,)"
                     if emit_accept_hidden else "")
    body = (
        f"    outputs = self._model.forward_ragged(\n"
        f"        inputs_embeds, {past_kv_tuple}, rope_rotary_cos_sin, "
        f"positions, query_start_offsets, query_lengths, "
        f"past_lengths, attention_sequence_lengths, "
        f"state_indices, execution_phase_marker, context_sequence_count_carrier, "
        f"kv_page_table, logits_indices, "
        f"{conv_tuple}, {rec_tuple}, skip_softmax_scale{tree_kwargs})\n" +
        unpack +
        f"    result = ((logits,) + ((hidden_states,) if hidden_states is not None else ()) "
        f"{accept_result} "
        f"+ tuple(present_key_values)\n"
        f"            + tuple(present_conv_states)"
        f" + tuple(present_recurrent_states))\n" +
        ("    result += (tuple(intermediate_conv_states)"
         " + tuple(intermediate_recurrent_states))\n" if spec_base else "") +
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


def _make_fused_nvfp4_linear(source: nn.Module,
                             out_features: int) -> nn.Module:
    """Create a fused NVFP4 projection with the source TP ownership."""
    method = NVFP4LinearMethod(group_size=source.quant_method.group_size)
    if source.tp_mode == TPMode.COL:
        return ColumnParallelLinear(source.in_features,
                                    out_features,
                                    bias=False,
                                    dtype=torch.float16,
                                    mapping=source.mapping,
                                    quant_method=method,
                                    tp_mode=TPMode.COL)
    return ReplicatedLinear(source.in_features,
                            out_features,
                            bias=False,
                            dtype=torch.float16,
                            mapping=source.mapping,
                            quant_method=method)


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
                    qkvz = _make_fused_nvfp4_linear(qkv, splits[0] + splits[1])
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
            fused_linear = _make_fused_nvfp4_linear(first_proj, fused_out_dim)
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

    def materialize_checkpoint_defaults(self, device: str) -> None:
        for layer in self.model.layers:
            mixer = getattr(layer, "linear_attn", None)
            if mixer is None or mixer.conv1d.bias.device.type != "meta":
                continue
            bias = mixer.conv1d.bias
            mixer.conv1d.bias = torch.zeros(bias.shape,
                                            dtype=bias.dtype,
                                            device=device)

    def onnx_export_spec(self) -> OnnxSpec:
        """Return all model-specific parameters needed for ONNX export."""
        config = self.config
        gc = config.gdn_cfg
        Ng = config.num_gdn_layers
        assert gc is not None or Ng == 0, (
            "Qwen3.5 requires gdn_cfg when any layer is GDN")
        return self._token_major_onnx_export_spec()

    def _token_major_onnx_export_spec(self) -> OnnxSpec:
        """Return the unified Qwen3.5 token-major export contract."""
        config = self.config
        gc = config.gdn_cfg
        Na = config.num_attn_layers
        Ng = config.num_gdn_layers
        assert gc is not None or Ng == 0
        mtp_base = _is_mtp_base_export(config)
        dflash_base = _is_dflash_base_export(config)
        jetspec_base = _is_jetspec_base_export(config)
        dspark_base = _is_dspark_base_export(config)
        target_hidden_base = dflash_base or jetspec_base or dspark_base
        spec_base = mtp_base or target_hidden_base
        tree_attention = spec_base
        tree_metadata = _is_spec_tree_base_export(config)
        device = next(itertools.chain(self.parameters(),
                                      self.buffers())).device
        num_sequences = _BATCH_SIZE
        query_length = _SEQ_LEN
        physical_tokens = num_sequences * query_length
        pool_rows = 4
        dtype16 = torch.float16
        kv_dtype = (torch.float8_e4m3fn
                    if config.quant.kv_cache_quant == "fp8" else dtype16)

        inputs_embeds = torch.zeros(physical_tokens,
                                    config.hidden_size,
                                    dtype=dtype16,
                                    device=device)
        past_key_values = [
            torch.zeros(2,
                        2,
                        KV_PAGE_SIZE,
                        config.num_key_value_heads,
                        config.head_dim,
                        dtype=kv_dtype,
                        device=device) for _ in range(Na)
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
        state_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)
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
            [0, 2, 3], dtype=torch.int64, device=device) if spec_base else
                          query_start_offsets[1:].to(torch.int64) - 1)
        conv_states = [
            torch.zeros(pool_rows,
                        gc.conv_dim,
                        gc.conv_kernel,
                        dtype=self.CONV_STATE_DTYPE,
                        device=device) for _ in range(Ng)
        ]
        recurrent_states = [
            torch.zeros(pool_rows,
                        gc.num_value_heads,
                        gc.key_head_dim,
                        gc.value_head_dim,
                        dtype=self.RECURRENT_STATE_DTYPE,
                        device=device) for _ in range(Ng)
        ]
        args = (inputs_embeds, *past_key_values, rope_rotary_cos_sin,
                positions, query_start_offsets, query_lengths, past_lengths,
                attention_sequence_lengths, state_indices,
                execution_phase_marker, context_sequence_count_carrier,
                kv_page_table, logits_indices, *conv_states, *recurrent_states)
        input_names = (
            ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
                "rope_rotary_cos_sin", "positions", "query_start_offsets",
                "query_lengths", "past_lengths", "attention_sequence_lengths",
                "state_indices", "execution_phase_marker",
                "context_sequence_count_carrier", "kv_page_table",
                "logits_indices"
            ] + [f"conv_state_{i}" for i in range(Ng)] +
            [f"recurrent_state_{i}" for i in range(Ng)])
        output_names = (["logits"] +
                        [f"present_key_values_{i}" for i in range(Na)] +
                        [f"present_conv_state_{i}" for i in range(Ng)] +
                        [f"present_recurrent_state_{i}" for i in range(Ng)])
        if spec_base or getattr(self, "emit_hidden_states", False):
            output_names.insert(1, "hidden_states")
        emit_accept_hidden = bool(
            spec_base and getattr(self, "emit_accept_hidden_states", False))
        if emit_accept_hidden:
            output_names.insert(2, "accept_hidden_states")
        if spec_base:
            output_names += [f"intermediate_conv_state_{i}" for i in range(Ng)]
            output_names += [
                f"intermediate_recurrent_state_{i}" for i in range(Ng)
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
        packed_mask_width = torch.export.Dim("packed_mask_width",
                                             min=1,
                                             max=64)
        all_shapes: list = [{0: tokens}]
        all_shapes.extend({1: num_pages} for _ in range(Na))
        all_shapes.extend([
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
                0: logits_rows if spec_base else sequences
            },
        ])
        all_shapes.extend({0: state_pool} for _ in range(Ng))
        all_shapes.extend({0: state_pool} for _ in range(Ng))

        skip_softmax_scale = torch.zeros(2, dtype=torch.int8, device=device)
        skip_dim = torch.export.Dim("skip_softmax_scale_len",
                                    min=0,
                                    max=1048576)
        args += (skip_softmax_scale, )
        input_names += ["skip_softmax_scale"]
        all_shapes.append({0: skip_dim})

        if tree_attention:
            attention_position_ids = positions.clone()
            packed_attention_mask = torch.zeros(physical_tokens,
                                                (query_length + 31) // 32,
                                                dtype=torch.int32,
                                                device=device)
            args += (attention_position_ids, packed_attention_mask)
            input_names += ["attention_position_ids", "packed_attention_mask"]
            all_shapes.extend([{0: tokens}, {0: tokens, 1: packed_mask_width}])

        if tree_metadata:
            tree_parent_ids = torch.full((physical_tokens, ),
                                         -1,
                                         dtype=torch.int32,
                                         device=device)
            tree_depths = torch.zeros(physical_tokens,
                                      dtype=torch.int32,
                                      device=device)
            valid_tree_counts = query_lengths.clone()
            args += (tree_parent_ids, tree_depths, valid_tree_counts)
            input_names += [
                "tree_parent_ids", "tree_depths", "valid_tree_counts"
            ]
            all_shapes.extend([{0: tokens}, {0: tokens}, {0: sequences}])

        wrapped = _make_flat_wrapper_hybrid_ragged(self, Na, Ng, spec_base,
                                                   tree_attention,
                                                   tree_metadata,
                                                   emit_accept_hidden)

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
        skip_softmax_scale: "torch.Tensor | None" = None,
        attention_position_ids: "torch.Tensor | None" = None,
        packed_attention_mask: "torch.Tensor | None" = None,
        tree_parent_ids: "torch.Tensor | None" = None,
        tree_depths: "torch.Tensor | None" = None,
        valid_tree_counts: "torch.Tensor | None" = None,
    ) -> Tuple:
        mtp_base = _is_mtp_base_export(self.config)
        dflash_base = _is_dflash_base_export(self.config)
        jetspec_base = _is_jetspec_base_export(self.config)
        dspark_base = _is_dspark_base_export(self.config)
        target_hidden_base = dflash_base or jetspec_base or dspark_base
        target_layer_ids = (
            self.config.dspark_target_layer_ids if dspark_base else
            self.config.jetspec_target_layer_ids if jetspec_base else
            self.config.dflash_target_layer_ids if dflash_base else None)
        (hidden_states, present_kv, present_conv, present_recurrent,
         intermediate_conv, intermediate_recurrent,
         target_hidden) = self.model.forward_ragged(
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
             skip_softmax_scale,
             attention_position_ids=attention_position_ids,
             packed_attention_mask=packed_attention_mask,
             tree_parent_ids=tree_parent_ids,
             tree_depths=tree_depths,
             valid_tree_counts=valid_tree_counts,
             collect_intermediate_states=(mtp_base or target_hidden_base),
             target_layer_ids=target_layer_ids)
        selected_hidden_states = torch.index_select(hidden_states, 0,
                                                    logits_indices)
        logits = self.lm_head(selected_hidden_states).to(torch.float32)
        emitted_hidden = target_hidden if target_hidden_base else (
            hidden_states if
            (mtp_base or getattr(self, "emit_hidden_states", False)) else None)
        return (logits, emitted_hidden, present_kv, present_conv,
                present_recurrent, intermediate_conv, intermediate_recurrent)

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
        skip_softmax_scale: "torch.Tensor | None" = None,
        execution_phase_marker: "torch.Tensor | None" = None,
        tree_parent_ids: "torch.Tensor | None" = None,
        tree_depths: "torch.Tensor | None" = None,
    ) -> Tuple:
        mtp_base = _is_mtp_base_export(self.config)
        dflash_base = _is_dflash_base_export(self.config)
        jetspec_base = _is_jetspec_base_export(self.config)
        dspark_base = _is_dspark_base_export(self.config)
        target_hidden_base = dflash_base or jetspec_base or dspark_base
        target_layer_ids = (
            self.config.dspark_target_layer_ids if dspark_base else
            self.config.jetspec_target_layer_ids if jetspec_base else
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
             skip_softmax_scale=skip_softmax_scale,
             execution_phase_marker=execution_phase_marker,
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
