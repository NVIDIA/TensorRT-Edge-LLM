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
DFlash Draft Model for speculative decoding — cached KV path.

The DFlash draft model generates an entire block of draft tokens in a SINGLE
forward pass.  Target-hidden-derived K/V is updated into a persistent
draft KV cache via the DFlashTargetKVCacheUpdate plugin; proposal self K/V
is written and attention is performed by the standard AttentionPlugin with
tree attention enabled.

Engine bindings (cached path):
    inputs_embeds        [B, BS, H]     Embedding of [y0, mask, mask, ..., mask]
    target_hidden_concat [B, L, Nl*H]   Target hidden DELTA from base
    past_key_values_i    [2, num_pages, KV_PAGE_SIZE, Hkv, D]  Draft combined KV cache —
                         the paged pool (same contract as the AttentionPlugin kv_cache
                         binding). maxBatch/capPadded are recovered at enqueue time from
                         numPages and the builder-configured pages_per_slot attribute
                         (see DFlashTargetKVCacheUpdatePlugin::setPagesPerSlot); this cache
                         has no page table of its own (identity-contiguous, reuse opt-out).
    rope_rotary_cos_sin  [1, capacity, rotaryDim]   Shared RoPE cache
    context_lengths      [B]            Total context length (target + proposal)
    kvcache_start_index  [B]            Draft cache start index (for delta write)
    kv_page_table        [B, 2, max_pages_per_seq]  int32 page table (proposal
                         self-attention only; the draft KV cache above has no
                         page table of its own)
    attention_mask       [B, BS, divUp(BS,32)]  Packed proposal mask
    attention_pos_id     [B, BS]        Position IDs for proposal tokens
    logits               [B, BS, V]     Full vocab logits
    present_key_values_i [2, num_pages, KV_PAGE_SIZE, Hkv, D]  Updated draft KV cache
"""

import itertools
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import ModelConfig
from ..default.modeling_default import OnnxSpec, RMSNorm
from ..linear import FP16Linear, make_linear
from ..ops import KV_PAGE_SIZE, attention_plugin, dflash_target_kv_cache_update

__all__ = ["DFlashDraftModel"]

# ---------------------------------------------------------------------------
# Dummy-shape constants for ONNX export
# ---------------------------------------------------------------------------

_BATCH_SIZE = 2
_BLOCK_SIZE = 16  # DFlash block size
_CTX_LEN = 2  # Delta length for dummy shapes
_KV_CAPACITY = 64  # Dummy KV cache capacity

# ---------------------------------------------------------------------------
# MLP (SwiGLU, same as Qwen3)
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """SwiGLU MLP: gate_proj, up_proj, down_proj."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        prefix = f"layers.{layer_idx}.mlp"
        self.gate_proj = make_linear(config,
                                     config.hidden_size,
                                     config.intermediate_size,
                                     module_name=f"{prefix}.gate_proj")
        self.up_proj = make_linear(config,
                                   config.hidden_size,
                                   config.intermediate_size,
                                   module_name=f"{prefix}.up_proj")
        self.down_proj = make_linear(config,
                                     config.intermediate_size,
                                     config.hidden_size,
                                     module_name=f"{prefix}.down_proj")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) *
            self.up_proj(hidden_states))


# ---------------------------------------------------------------------------
# DFlash Cached Attention Layer
# ---------------------------------------------------------------------------


class DFlashCachedAttention(nn.Module):
    """Cached attention for DFlash draft model.

    Per-layer: updates the draft KV cache with target delta K/V,
    then runs AttentionPlugin for proposal self-attention over the
    full context (persistent target K/V + temporary proposal K/V).
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.attention_scale = config.attention_scaling
        self.hidden_size = config.hidden_size

        prefix = f"layers.{layer_idx}.self_attn"
        self.q_proj = make_linear(config,
                                  config.hidden_size,
                                  self.num_heads * self.head_dim,
                                  bias=False,
                                  module_name=f"{prefix}.q_proj")
        self.k_proj = make_linear(config,
                                  config.hidden_size,
                                  self.num_kv_heads * self.head_dim,
                                  bias=False,
                                  module_name=f"{prefix}.k_proj")
        self.v_proj = make_linear(config,
                                  config.hidden_size,
                                  self.num_kv_heads * self.head_dim,
                                  bias=False,
                                  module_name=f"{prefix}.v_proj")
        self.o_proj = make_linear(config,
                                  self.num_heads * self.head_dim,
                                  config.hidden_size,
                                  module_name=f"{prefix}.o_proj")

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,  # [B, BS, H] proposal hidden
            h_delta: torch.
        Tensor,  # [B, L, H] target hidden delta (after fc+norm)
            past_key_value: torch.
        Tensor,  # paged pool [2, num_pages, KV_PAGE_SIZE, Hkv, D]
            rope_cos_sin: torch.Tensor,  # [ropeBatch, capacity, rotaryDim]
            kvcache_start_index: torch.Tensor,  # [B] INT32
            kv_page_table: torch.Tensor,  # [B, 2, maxPagesPerSeq] INT32
            delta_lengths: torch.Tensor,  # [B] INT32, per-batch delta lengths
            context_lengths: torch.Tensor,  # [B] INT32
            attention_mask: torch.Tensor,  # [B, BS, packedMaskLen] INT32
            attention_pos_id: torch.Tensor,  # [B, BS] INT32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with cached target K/V + proposal self-attention.

        Returns:
            (attn_output [B, BS, H], present_key_value [B, 2, Hkv, capacity, D])
        """
        B, BS, _ = hidden_states.shape
        L = h_delta.shape[1]

        # --- Target delta K/V: project and update cache ---
        k_delta = self.k_proj(h_delta)  # [B, L, Hkv*D]
        v_delta = self.v_proj(h_delta)  # [B, L, Hkv*D]
        k_delta = k_delta.reshape(B, L, self.num_kv_heads, self.head_dim)
        v_delta = v_delta.reshape(B, L, self.num_kv_heads, self.head_dim)
        k_delta = self.k_norm(k_delta)  # [B, L, Hkv, D]

        updated_kv = dflash_target_kv_cache_update(k_delta, v_delta,
                                                   past_key_value,
                                                   rope_cos_sin,
                                                   kvcache_start_index,
                                                   delta_lengths)

        # --- Proposal self Q/K/V ---
        q = self.q_proj(hidden_states)  # [B, BS, Hq*D]
        q = q.reshape(B, BS, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        q = q.reshape(B, BS, self.num_heads * self.head_dim)

        k_self = self.k_proj(hidden_states)  # [B, BS, Hkv*D]
        k_self = k_self.reshape(B, BS, self.num_kv_heads, self.head_dim)
        k_self = self.k_norm(k_self)
        k_self = k_self.reshape(B, BS, self.num_kv_heads * self.head_dim)

        v_self = self.v_proj(hidden_states)  # [B, BS, Hkv*D]

        # --- AttentionPlugin: proposal attention over full context ---
        attn_4d, present_kv = attention_plugin(
            q,
            k_self,
            v_self,
            updated_kv,
            context_lengths,
            rope_cos_sin,
            kvcache_start_index,
            kv_page_table,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            sliding_window_size=-1,
            enable_tree_attention=True,
            enable_fp8_kv_cache=False,
            attention_scale=self.attention_scale,
            enable_vision_block_attention=False,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
            qkv_scales=[1.0, 1.0, 1.0])

        # attn_4d: [B, BS, Hq, D] -> [B, BS, Hq*D]
        attn_output = attn_4d.reshape(B, BS, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_kv


# ---------------------------------------------------------------------------
# DFlash Cached Decoder Layer
# ---------------------------------------------------------------------------


class DFlashCachedDecoderLayer(nn.Module):
    """Decoder layer for cached DFlash draft model."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DFlashCachedAttention(config, layer_idx=layer_idx)
        self.mlp = MLP(config, layer_idx=layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        h_delta: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_cos_sin: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        delta_lengths: torch.Tensor,
        context_lengths: torch.Tensor,
        attention_mask: torch.Tensor,
        attention_pos_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)

        attn_output, present_kv = self.self_attn(
            normed, h_delta, past_key_value, rope_cos_sin, kvcache_start_index,
            kv_page_table, delta_lengths, context_lengths, attention_mask,
            attention_pos_id)

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states))

        return hidden_states, present_kv


# ---------------------------------------------------------------------------
# Flat ONNX wrapper
# ---------------------------------------------------------------------------


def _make_flat_wrapper_dflash(model: nn.Module, num_layers: int) -> nn.Module:
    """Build a flat-signature wrapper for cached DFlash draft ONNX export.

    Uses exec() to generate a forward() with explicit named parameters for
    each past_key_values_i, matching the pattern used by the default LLM
    wrapper. This is required because torch.export treats *args as a single
    tuple, causing dynamic_shapes mismatches.
    """
    # Build explicit parameter list
    param_names = ([
        "inputs_embeds", "dflash_target_hidden_concat", "rope_rotary_cos_sin",
        "context_lengths", "kvcache_start_index", "kv_page_table",
        "dflash_delta_lengths", "attention_mask", "attention_pos_id"
    ] + [f"past_key_values_{i}" for i in range(num_layers)])

    past_kv_tuple = "({},)".format(", ".join(f"past_key_values_{i}"
                                             for i in range(num_layers)))

    body = (
        f"    logits, present_kv_list = self._model(\n"
        f"        inputs_embeds, dflash_target_hidden_concat,\n"
        f"        rope_rotary_cos_sin, context_lengths,\n"
        f"        kvcache_start_index, kv_page_table, dflash_delta_lengths,\n"
        f"        attention_mask, attention_pos_id,\n"
        f"        list({past_kv_tuple}))\n"
        f"    return (logits,) + tuple(present_kv_list)\n")

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
# DFlash Draft Model (Cached)
# ---------------------------------------------------------------------------


class DFlashDraftModel(nn.Module):
    """DFlash draft model for speculative decoding — cached KV path.

    Module tree (matches checkpoint keys after remapping):
        fc              Linear(Nl * H, H, bias=False)   - feature fusion
        hidden_norm     RMSNorm(H)                       - normalize fused features
        layers.0..4     DFlashCachedDecoderLayer          - 5 decoder layers
        norm            RMSNorm(H)                        - final norm
        lm_head         Linear(H, V)                      - shared with base
    """

    match_fp32_elementwise_initializers = True

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        num_target_layers = len(config.dflash_target_layer_ids)

        self.fc = make_linear(config,
                              num_target_layers * hidden_size,
                              hidden_size,
                              bias=False,
                              module_name="fc")
        if not isinstance(self.fc, FP16Linear):
            raise ValueError(
                "DFlash draft fc projector must remain dense FP16 for the "
                "full-FP32 target-hidden projection. Exclude module 'fc' "
                "from draft quantization.")
        self.hidden_norm = RMSNorm(hidden_size, config.rms_norm_eps)

        self.layers = nn.ModuleList([
            DFlashCachedDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size, config.rms_norm_eps)
        self.lm_head = make_linear(config,
                                   hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")

    def forward(
        self,
        inputs_embeds: torch.Tensor,  # [B, BS, H]
        target_hidden_concat: torch.Tensor,  # [B, L, Nl*H] target hidden delta
        rope_rotary_cos_sin: torch.Tensor,  # [ropeBatch, capacity, rotaryDim]
        context_lengths: torch.Tensor,  # [B] INT32
        kvcache_start_index: torch.Tensor,  # [B] INT32
        kv_page_table: torch.Tensor,  # [B, 2, maxPagesPerSeq] INT32
        delta_lengths: torch.Tensor,  # [B] INT32, per-batch delta lengths
        attention_mask: torch.Tensor,  # [B, BS, packedMaskLen] INT32
        attention_pos_id: torch.Tensor,  # [B, BS] INT32
        past_key_values: List[
            torch.Tensor],  # list of [B, 2, Hkv, capacity, D]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.

        Returns:
            (logits [B, BS, V], present_key_values list)
        """
        B, BS, _ = inputs_embeds.shape

        # Project multi-layer hidden states: [B, L, Nl*H] -> [B, L, H]
        # Qwen3-8B target_hidden can spike above abs=2e4 for some first-token
        # channels. The visible pre-RMSNorm FC result must remain FP32; casting
        # an already-overflowed FP16 FC output back to FP32 is too late.
        bias = (self.fc.bias.to(torch.float32)
                if self.fc.bias is not None else None)
        h_delta_acc = F.linear(target_hidden_concat.to(torch.float32),
                               self.fc.weight.to(torch.float32), bias)
        h_delta = self.hidden_norm(h_delta_acc).to(inputs_embeds.dtype)

        # Run through decoder layers
        hidden_states = inputs_embeds.to(h_delta.dtype)
        present_key_values = []

        for i, layer in enumerate(self.layers):
            hidden_states, present_kv = layer(
                hidden_states, h_delta, past_key_values[i],
                rope_rotary_cos_sin, kvcache_start_index, kv_page_table,
                delta_lengths, context_lengths, attention_mask,
                attention_pos_id)
            present_key_values.append(present_kv)

        # Final norm + lm_head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states).to(torch.float32)

        return logits, present_key_values

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def onnx_export_spec(self) -> OnnxSpec:
        """Return all model-specific parameters needed for ONNX export."""
        config = self.config
        device = next(itertools.chain(self.parameters(),
                                      self.buffers())).device
        dtype16 = torch.float16
        num_target_layers = len(config.dflash_target_layer_ids)
        batch_size = _BATCH_SIZE
        block_size = config.dflash_block_size
        delta_len = _CTX_LEN
        kv_capacity = _KV_CAPACITY
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        num_layers = config.num_hidden_layers
        packed_mask_len = (block_size + 31) // 32

        inputs_embeds = torch.zeros(batch_size,
                                    block_size,
                                    config.hidden_size,
                                    dtype=dtype16,
                                    device=device)
        target_hidden_concat = torch.zeros(batch_size,
                                           delta_len,
                                           num_target_layers *
                                           config.hidden_size,
                                           dtype=dtype16,
                                           device=device)
        rope_rotary_cos_sin = torch.zeros(1,
                                          kv_capacity,
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
        delta_lengths = torch.zeros(batch_size,
                                    dtype=torch.int32,
                                    device=device)
        attention_mask = torch.zeros(batch_size,
                                     block_size,
                                     packed_mask_len,
                                     dtype=torch.int32,
                                     device=device)
        attention_pos_id = torch.zeros(batch_size,
                                       block_size,
                                       dtype=torch.int32,
                                       device=device)

        # Paged KV pool binding — same contract as the AttentionPlugin's kv_cache input:
        # [2, num_pages, KV_PAGE_SIZE, numKVHeads, headDim]. See
        # DFlashTargetKVCacheUpdatePlugin's input 2 doc; maxBatch/cap are recovered at
        # enqueue time from the builder-configured pages_per_slot attribute.
        past_key_values = [
            torch.zeros(2,
                        1,
                        KV_PAGE_SIZE,
                        num_kv_heads,
                        head_dim,
                        dtype=dtype16,
                        device=device) for _ in range(num_layers)
        ]

        args = (inputs_embeds, target_hidden_concat, rope_rotary_cos_sin,
                context_lengths, kvcache_start_index, kv_page_table,
                delta_lengths, attention_mask, attention_pos_id,
                *past_key_values)

        input_names = [
            "inputs_embeds",
            "dflash_target_hidden_concat",
            "rope_rotary_cos_sin",
            "context_lengths",
            "kvcache_start_index",
            "kv_page_table",
            "dflash_delta_lengths",
            "attention_mask",
            "attention_pos_id",
        ]
        for i in range(num_layers):
            input_names.append(f"past_key_values_{i}")

        output_names = ["logits"]
        for i in range(num_layers):
            output_names.append(f"present_key_values_{i}")

        batch = torch.export.Dim("batch", min=1, max=256)
        block_seq = torch.export.Dim("block_seq", min=1, max=64)
        delta_seq = torch.export.Dim("delta_seq", min=1, max=32768)
        kv_len = torch.export.Dim("kv_len", min=1, max=32768)
        packed_mask = torch.export.Dim("packed_mask", min=1, max=64)
        page_batch = torch.export.Dim("page_batch", min=1, max=256)
        max_pages = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
        num_pages = torch.export.Dim("num_pages", min=1, max=1048576)

        dynamic_shapes = [
            {
                0: batch,
                1: block_seq
            },  # inputs_embeds
            {
                0: batch,
                1: delta_seq
            },  # dflash_target_hidden_concat
            {
                1: kv_len
            },  # rope_rotary_cos_sin (ropeBatch=1 fixed)
            {
                0: batch
            },  # context_lengths
            {
                0: batch
            },  # kvcache_start_index
            {
                0: page_batch,
                2: max_pages
            },  # kv_page_table
            {
                0: batch
            },  # dflash_delta_lengths
            {
                0: batch,
                1: block_seq,
                2: packed_mask
            },  # attention_mask
            {
                0: batch,
                1: block_seq
            },  # attention_pos_id
        ]
        for _ in range(num_layers):
            dynamic_shapes.append({1: num_pages
                                   })  # past_key_values_i (pool-shaped)

        wrapped = _make_flat_wrapper_dflash(self, num_layers)
        wrapped.eval()

        return OnnxSpec(wrapped=wrapped,
                        args=args,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=dynamic_shapes)
