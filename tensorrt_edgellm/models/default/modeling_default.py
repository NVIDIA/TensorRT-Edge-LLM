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
Module definitions for the decoder stack (attention-only).

Forward-pass conventions
------------------------
``CausalLM.forward_ragged``:

    inputs_embeds        [physical_tokens, hidden_size]           float16/bfloat16
    past_key_values      tuple of [2, num_pages, KV_PAGE_SIZE, num_kv_heads, head_dim] per
                         attn-layer — the paged KV pool (in-place aliased; num_pages is a
                         fixed value per engine build, see llmBuilder.cpp setupKVCacheProfiles)
    rope_rotary_cos_sin  [physical_tokens, rotary_dim]             float32
    query metadata       token rows [physical_tokens], sequences [batch]
    kv_page_table        [batch, 2, max_pages_per_seq]  int32
    ──────────────────────────────────────────────────────────────────────
    -> logits             [selected_tokens, vocab_size]             float32
    -> present_key_values tuple of the same pool tensors as past_key_values (aliased, no growth)

Checkpoint key correspondence
------------------------------
Attention layer  -> model.layers.N.self_attn.*
MLP              -> model.layers.N.mlp.*
LayerNorm        -> model.layers.N.input_layernorm.*, post_attention_layernorm.*
"""

import itertools
import logging
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import ModelConfig
from ..linear import (FP16Linear, NVFP4LinearMethod, ReplicatedLinear, TPMode,
                      is_int4_linear, is_nvfp4_linear, make_linear)
from ..ops import KV_PAGE_SIZE, attention_plugin, qkv_concat

logger = logging.getLogger(__name__)


def _concat_hidden_in_provider_order(
    hidden_by_layer: dict[int, torch.Tensor],
    layer_ids: "List[int] | None",
) -> "torch.Tensor | None":
    if not layer_ids:
        return None
    missing = [
        layer_id for layer_id in layer_ids if layer_id not in hidden_by_layer
    ]
    if missing:
        raise ValueError(
            f"Target hidden layer IDs were not produced: {missing}")
    return torch.cat([hidden_by_layer[layer_id] for layer_id in layer_ids],
                     dim=-1)


__all__ = [
    "OnnxSpec",
    "RMSNorm",
    "Attention",
    "MLP",
    "DecoderLayer",
    "Transformer",
    "CausalLM",
]

# ---------------------------------------------------------------------------
# ONNX export spec
# ---------------------------------------------------------------------------


@dataclass
class OnnxSpec:
    """All model-specific parameters needed to call ``torch.onnx.export``.

    Produced by :meth:`CausalLM.onnx_export_spec`; consumed by
    :func:`~tensorrt_edgellm.onnx.export._export_model`.
    """
    wrapped: nn.Module
    args: tuple
    input_names: List[str]
    output_names: List[str]
    dynamic_shapes: list


def _make_flat_wrapper_ragged(model: nn.Module,
                              Na: int,
                              Nd: int,
                              tree_attention: bool = False) -> nn.Module:
    """Build the flat vanilla wrapper for the common token-major ABI."""
    param_names: List[str] = (
        ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
            "rope_rotary_cos_sin", "positions", "query_start_offsets",
            "query_lengths", "past_lengths", "attention_sequence_lengths",
            "state_indices", "execution_phase_marker",
            "context_sequence_count_carrier", "kv_page_table", "logits_indices"
        ] + [f"deepstack_embeds_{i}"
             for i in range(Nd)] + ["skip_softmax_scale"])
    if tree_attention:
        param_names += [
            "attention_position_ids", "packed_attention_mask",
            "tree_parent_ids", "tree_depths", "valid_tree_counts"
        ]
    past_kv_tuple = "({},)".format(", ".join(
        f"past_key_values_{i}" for i in range(Na))) if Na else "()"
    deepstack_tuple = "({},)".format(", ".join(
        f"deepstack_embeds_{i}" for i in range(Nd))) if Nd else "()"
    tree_kwargs = (", attention_position_ids=attention_position_ids"
                   ", packed_attention_mask=packed_attention_mask"
                   ", tree_parent_ids=tree_parent_ids"
                   ", tree_depths=tree_depths"
                   ", valid_tree_counts=valid_tree_counts"
                   if tree_attention else "")
    body = (
        f"    outputs = self._model.forward_ragged(\n"
        f"        inputs_embeds, {past_kv_tuple}, rope_rotary_cos_sin, "
        f"positions, query_start_offsets, query_lengths, "
        f"past_lengths, attention_sequence_lengths, "
        f"state_indices, execution_phase_marker, context_sequence_count_carrier, "
        f"kv_page_table, logits_indices, "
        f"{deepstack_tuple}, skip_softmax_scale{tree_kwargs})\n"
        f"    logits, hidden_states, present_key_values = outputs\n"
        f"    if hidden_states is None:\n"
        f"        return (logits,) + tuple(present_key_values)\n"
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


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation.

    Buffer: ``weight`` [hidden_size].
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size,
                                              dtype=torch.float16))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        hidden_states = hidden_states.to(input_dtype)
        return self.weight.to(input_dtype) * hidden_states


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head GQA attention (custom attention ONNX op).

    Submodule names match checkpoint keys:
        q_proj, k_proj, v_proj, o_proj
        q_norm, k_norm  (present only when config.has_qk_norm)
    """

    def __init__(self,
                 config: ModelConfig,
                 layer_idx: int,
                 in_features: int = 0) -> None:
        super().__init__()
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = config.head_dim
        hidden_size = config.hidden_size
        # in_features overrides the QKV projection input dimension.
        # When 0 (default), QKV projections use hidden_size as input.
        qkv_in_features = in_features or hidden_size

        self.layer_idx = layer_idx
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.attention_scale = config.attention_scaling
        self.enable_fp8_kv_cache = config.quant.kv_cache_quant == "fp8"
        self.sliding_window_size = config.sliding_window_size  # -1 means no sliding window
        # Skip-softmax (BLASST) calibrated scale factor S (0.0 = disabled).
        self.skip_softmax_scale_factor = config.skip_softmax_scale_factor
        module_prefix = f"layers.{layer_idx}.self_attn"

        self.q_proj = make_linear(config,
                                  qkv_in_features,
                                  num_attention_heads * head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{module_prefix}.q_proj",
                                  tp_mode=TPMode.COL)
        self.k_proj = make_linear(config,
                                  qkv_in_features,
                                  num_key_value_heads * head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{module_prefix}.k_proj",
                                  tp_mode=TPMode.COL)
        self.v_proj = make_linear(config,
                                  qkv_in_features,
                                  num_key_value_heads * head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{module_prefix}.v_proj",
                                  tp_mode=TPMode.COL)
        self._uses_int4_qkv = any(
            is_int4_linear(proj)
            for proj in (self.q_proj, self.k_proj, self.v_proj))

        # FP8 attention scales live on the projection modules (checkpoint keys
        # ``...{q,k,v}_proj.{q,k,v}_scale``); they are not part of FP8Linear's
        # per-tensor weight/input scales.
        if self.enable_fp8_kv_cache:
            self.q_proj.register_buffer("q_scale", torch.ones(1))
            self.k_proj.register_buffer("k_scale", torch.ones(1))
            self.v_proj.register_buffer("v_scale", torch.ones(1))

        self.o_proj = make_linear(config,
                                  num_attention_heads * head_dim,
                                  hidden_size,
                                  module_name=f"{module_prefix}.o_proj",
                                  tp_mode=TPMode.ROW)

        if config.has_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        # Cache RMSNorm eps as a plain float so the attention_plugin custom-op gets a stable
        # default-free kwarg in the FX graph (torch.export strips default-matching kwargs).
        self._rms_norm_eps = float(
            config.rms_norm_eps) if config.has_qk_norm else 1e-6
        # HunYuan V1 applies the per-head QK RMSNorm AFTER RoPE (gamma placement
        # differs from the Qwen3 norm-then-rotate convention).
        self._qk_norm_post_rope = int(
            bool(getattr(config, "qk_norm_post_rope", False)))

        # Per-head q/k_norm gamma weights as plain list[float] (NOT tensors): the
        # attention_plugin custom-op needs literal List[float] kwargs at trace time.
        self._q_norm_gamma_list: list = []
        self._k_norm_gamma_list: list = []
        if config.has_qk_norm:
            # Capture again whenever weights change (e.g. after `load_state_dict`).
            self.register_load_state_dict_post_hook(
                lambda *_args, **_kwargs: self._capture_qk_norm_gamma_lists())

    def _capture_qk_norm_gamma_lists(self) -> None:
        """Extract gamma weights from `self.q_norm` / `self.k_norm` into plain Python lists.

        Called from the post-state-dict-load hook so the lists reflect real checkpoint values
        (not the random `__init__` values). Idempotent — safe to call multiple times.

        Uses ``getattr`` defaults: subclasses may delete the norm submodules
        (e.g. Gemma4Attention removes ``k_norm`` on KV-shared layers) while
        still inheriting this method.
        """
        q_norm = getattr(self, "q_norm", None)
        k_norm = getattr(self, "k_norm", None)
        if q_norm is not None:
            self._q_norm_gamma_list = q_norm.weight.detach().to(
                torch.float32).cpu().flatten().tolist()
        if k_norm is not None:
            self._k_norm_gamma_list = k_norm.weight.detach().to(
                torch.float32).cpu().flatten().tolist()

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

        # Packed QKV: prefer the single fused GEMM installed by
        # `fuse_qkv_projections`; fall back to three projections + concat.
        if hasattr(self, "qkv_proj_fused"):
            qkv = self.qkv_proj_fused(hidden_states)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            if self._uses_int4_qkv:
                qkv = qkv_concat(q, k, v)
            else:
                qkv = torch.cat([q, k, v], dim=-1)

        # qk_norm is fused inside the AttentionPlugin — do NOT apply q_norm / k_norm here.
        # The modules stay registered only so checkpoint loading finds their weights.

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
        # Wire the runtime override carrier iff skip-softmax is enabled (scale
        # factor > 0).
        if skip_softmax_scale is not None and self.skip_softmax_scale_factor > 0.0:
            kwargs["skip_softmax_scale"] = skip_softmax_scale
        if enable_tree:
            kwargs["attention_mask"] = attention_mask
            kwargs["attention_pos_id"] = attention_pos_id
        # Always pass qkv_scales so torch.export includes a valid FLOATS
        # value in the FX graph for the unified ONNX translation.
        kwargs["qkv_scales"] = getattr(self, "_qkv_scales_float",
                                       [1.0, 1.0, 1.0])
        # Gamma kwargs are passed only when the model uses qk_norm; otherwise the ONNX
        # node carries no enable_qk_norm attribute.
        if self._q_norm_gamma_list or self._k_norm_gamma_list:
            kwargs["q_norm_gamma"] = self._q_norm_gamma_list
            kwargs["k_norm_gamma"] = self._k_norm_gamma_list
            kwargs["rms_norm_eps"] = float(self._rms_norm_eps)
            kwargs["enable_qk_norm"] = 1
            if self._qk_norm_post_rope:
                kwargs["qk_norm_post_rope"] = 1

        attn_output, present_key_value = attention_plugin(
            qkv,
            past_key_value,
            context_lengths,
            rope_rotary_cos_sin,
            kvcache_start_index,
            kv_page_table,
            **kwargs,
        )
        # AttentionPlugin returns [batch, seq_len, num_heads, head_dim]; reshape for o_proj.
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
        if hasattr(self, "qkv_proj_fused"):
            qkv = self.qkv_proj_fused(hidden_states)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            qkv = (qkv_concat(q, k, v)
                   if self._uses_int4_qkv else torch.cat([q, k, v], dim=-1))
        qkv = qkv.reshape(-1, qkv.shape[-1])
        kwargs: dict = {
            "num_q_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_dim,
            "sliding_window_size": self.sliding_window_size,
            "enable_tree_attention": packed_attention_mask is not None,
            "enable_fp8_kv_cache": self.enable_fp8_kv_cache,
            "attention_scale": self.attention_scale,
            "enable_context_mask_selector": False,
            "enable_vision_block_attention": False,
            "skip_softmax_scale_factor": self.skip_softmax_scale_factor,
            "qkv_scales": getattr(self, "_qkv_scales_float", [1.0, 1.0, 1.0]),
            "query_start_offsets": query_start_offsets,
            "attention_sequence_lengths": attention_sequence_lengths,
            "execution_phase_marker": execution_phase_marker,
            "context_sequence_count_carrier": context_sequence_count_carrier,
        }
        if packed_attention_mask is not None:
            kwargs["attention_mask"] = packed_attention_mask
            kwargs["attention_pos_id"] = attention_position_ids
        if skip_softmax_scale is not None and self.skip_softmax_scale_factor > 0.0:
            kwargs["skip_softmax_scale"] = skip_softmax_scale
        if self._q_norm_gamma_list or self._k_norm_gamma_list:
            kwargs["q_norm_gamma"] = self._q_norm_gamma_list
            kwargs["k_norm_gamma"] = self._k_norm_gamma_list
            kwargs["rms_norm_eps"] = float(self._rms_norm_eps)
            kwargs["enable_qk_norm"] = 1
            if self._qk_norm_post_rope:
                kwargs["qk_norm_post_rope"] = 1
        attn_output, present_key_value = attention_plugin(
            qkv, past_key_value, query_lengths, rope_rotary_cos_sin,
            past_lengths, kv_page_table, **kwargs)
        attn_output = attn_output.reshape(*hidden_states.shape[:-1],
                                          self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present_key_value


# ---------------------------------------------------------------------------
# Post-load optimisation: fuse attention Q/K/V projections into one GEMM
# (mirrors qwen3_5.fuse_gdn_input_projections)
# ---------------------------------------------------------------------------

_QKV_PROJ_NAMES = ("q_proj", "k_proj", "v_proj")
_FP8_KV_SCALE_NAMES = ("q_scale", "k_scale", "v_scale")
_NVFP4_SCALAR_SCALE_SUFFIXES = ("input_scale", "weight_scale_2")


def _fp8_kv_scales_as_floats(attn: nn.Module) -> List[float]:
    """Read Q/K/V attention scales before projection modules are fused."""
    q_buf = getattr(getattr(attn, "q_proj", None), "q_scale", None)
    k_buf = getattr(getattr(attn, "k_proj", None), "k_scale", None)
    v_buf = getattr(getattr(attn, "v_proj", None), "v_scale", None)
    if v_buf is None and getattr(attn, "attention_k_eq_v", False):
        v_buf = k_buf
    return [
        float(q_buf.item()) if q_buf is not None else 1.0,
        float(k_buf.item()) if k_buf is not None else 1.0,
        float(v_buf.item()) if v_buf is not None else 1.0,
    ]


def _can_fuse_nvfp4_scales(attn: "Attention") -> bool:
    """True if all 3 NVFP4 Q/K/V projections have identical scalar scales."""
    for suffix in _NVFP4_SCALAR_SCALE_SUFFIXES:
        tensors = []
        for name in _QKV_PROJ_NAMES:
            proj = getattr(attn, name, None)
            if proj is None:
                return False
            t = getattr(proj, suffix, None)
            if t is None:
                return False
            tensors.append(t)
        if not all(torch.equal(tensors[0], t) for t in tensors[1:]):
            return False
    return True


def fuse_qkv_projections(model: nn.Module) -> int:
    """Post-load optimisation: fuse attention Q/K/V projections into one GEMM.

    A single fused GEMM feeds the packed-QKV plugin input directly (no Concat,
    no reliance on the compiler backend's horizontal GEMM fusion).

    Per quant type: FP16 always fuses; NVFP4 fuses only when the per-tensor
    scales (``input_scale``, ``weight_scale_2``) match across Q/K/V (mismatch
    => warn and fall back to 3 GEMMs + concat); other quant types skip.
    For FP8 KV cache, the Q/K/V attention scales are preserved as Python
    floats before the original projection modules are removed.

    Fused layers replace the three sub-modules with ``qkv_proj_fused``
    (auto-detected in ``Attention.forward``). Returns the number fused.
    """

    fused_count = 0
    for name, module in model.named_modules():
        # Exact type match: subclasses (e.g. Gemma4Attention) have their own
        # forward() and may not own k_proj / v_proj.
        if type(module) is not Attention:
            continue
        attn: Attention = module
        if hasattr(attn, "qkv_proj_fused"):
            continue  # idempotent
        # Defensive: skip any layer that doesn't own all three projections.
        if any(getattr(attn, n, None) is None for n in _QKV_PROJ_NAMES):
            continue

        proj_modules = [getattr(attn, n) for n in _QKV_PROJ_NAMES]
        first_proj = attn.q_proj
        # Mixed quantization across Q/K/V cannot be fused into one GEMM.
        if any(type(p) is not type(first_proj) for p in proj_modules):
            logger.warning(
                "QKV fusion skipped for %s: mixed projection types (%s).",
                name, [type(p).__name__ for p in proj_modules])
            continue
        if isinstance(first_proj, FP16Linear):
            pass  # always fusible
        elif is_nvfp4_linear(first_proj):
            if not _can_fuse_nvfp4_scales(attn):
                logger.warning(
                    "QKV fusion skipped for %s: NVFP4 scalar scales differ "
                    "across projections. Re-quantize with resmoothing "
                    "enabled to equalise scales.", name)
                continue
        else:
            # INT4, FP8, MXFP8, etc. — not fusible.
            continue

        fp8_kv_scales = (_fp8_kv_scales_as_floats(attn)
                         if attn.enable_fp8_kv_cache else None)

        # --- Fuse: concatenate weights along output dim (dim 0) ----------
        fused_buffers: dict = {}
        # Union across all three projections so an attribute missing on any
        # side is caught (fail-loud below) instead of silently dropped.
        attr_names = list(
            dict.fromkeys(
                itertools.chain.from_iterable(
                    itertools.chain(p._buffers, p._parameters)
                    for p in proj_modules)))
        # These are AttentionPlugin attributes, not GEMM quantization state.
        # They live on separate Q/K/V projections only to match checkpoint
        # keys and were captured above before those modules are removed.
        attr_names = [
            attr for attr in attr_names if attr not in _FP8_KV_SCALE_NAMES
        ]
        # The checkpoint loader may rebind `bias` as a plain attribute
        # (outside _buffers/_parameters); include it explicitly.
        if "bias" not in attr_names and any(
                getattr(p, "bias", None) is not None for p in proj_modules):
            attr_names.append("bias")
        for attr in attr_names:
            parts = [getattr(p, attr, None) for p in proj_modules]
            if all(p is None for p in parts):
                continue
            # An attribute present on only a subset of Q/K/V would either
            # crash torch.cat or be silently dropped — fail loudly instead.
            if any(p is None for p in parts):
                raise RuntimeError(
                    f"QKV fusion: attribute '{attr}' present on only a "
                    "subset of q/k/v projections; cannot fuse.")
            if parts[0].numel() == 1:
                # Per-tensor scalar (0-d or (1,)-shaped): take first —
                # equality across Q/K/V is a fusion precondition.
                fused_buffers[attr] = parts[0]
            else:
                # Per-output-channel: concat along dim 0.
                fused_buffers[attr] = torch.cat(parts, dim=0)

        # Build a fused linear with correct type.
        fused_out_dim = sum(p.out_features for p in proj_modules)
        in_features = first_proj.in_features
        has_bias = getattr(first_proj, "bias", None) is not None
        if is_nvfp4_linear(first_proj):
            method = NVFP4LinearMethod(
                group_size=first_proj.quant_method.group_size)
            fused_linear = ReplicatedLinear(in_features,
                                            fused_out_dim,
                                            bias=has_bias,
                                            dtype=torch.float16,
                                            mapping=first_proj.mapping,
                                            quant_method=method)
        else:
            fused_linear = FP16Linear(in_features,
                                      fused_out_dim,
                                      bias=has_bias)

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
        attn.qkv_proj_fused = fused_linear
        if fp8_kv_scales is not None:
            attn._qkv_scales_float = fp8_kv_scales
        for proj_name in _QKV_PROJ_NAMES:
            delattr(attn, proj_name)

        fused_count += 1
        logger.debug("Fused QKV projections for %s", name)

    if fused_count:
        logger.info("Fused attention QKV projections in %d layer(s)",
                    fused_count)
    return fused_count


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """SwiGLU MLP: gate_proj, up_proj, down_proj."""

    def __init__(self, config: ModelConfig, layer_idx: int = -1) -> None:
        super().__init__()
        module_prefix = f"layers.{layer_idx}.mlp" if layer_idx >= 0 else ""
        self.gate_proj = make_linear(
            config,
            config.hidden_size,
            config.intermediate_size,
            module_name=f"{module_prefix}.gate_proj" if module_prefix else "",
            tp_mode=TPMode.COL)
        self.up_proj = make_linear(
            config,
            config.hidden_size,
            config.intermediate_size,
            module_name=f"{module_prefix}.up_proj" if module_prefix else "",
            tp_mode=TPMode.COL)
        self.down_proj = make_linear(
            config,
            config.intermediate_size,
            config.hidden_size,
            module_name=f"{module_prefix}.down_proj" if module_prefix else "",
            tp_mode=TPMode.ROW)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        use_token_matrix = (hidden_states.ndim > 2
                            and is_nvfp4_linear(self.gate_proj)
                            and is_nvfp4_linear(self.up_proj)
                            and is_nvfp4_linear(self.down_proj)
                            and self.down_proj.tp_size > 1)
        if use_token_matrix:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        output = self.down_proj(
            F.silu(self.gate_proj(hidden_states)) *
            self.up_proj(hidden_states))
        if use_token_matrix:
            output = output.reshape(*input_shape[:-1], output.shape[-1])
        return output


# ---------------------------------------------------------------------------
# DecoderLayer  (attention + MLP)
# ---------------------------------------------------------------------------


class DecoderLayer(nn.Module):
    """Single transformer decoder layer.

    Submodule names match checkpoint keys:
        self_attn, mlp, input_layernorm, post_attention_layernorm
    """

    #: Subclasses override to swap the per-layer feed-forward implementation
    #: (e.g. Cosmos3-Edge's non-gated squared-ReLU MLP) without rebuilding or
    #: patching modules after construction.
    mlp_cls = MLP

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Attention(config, layer_idx=layer_idx)
        self.mlp = type(self).mlp_cls(config, layer_idx=layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps)

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
        residual = hidden_states
        attn_output, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            past_key_value,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            kv_page_table,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
            skip_softmax_scale=skip_softmax_scale,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value

    def forward_ragged(self, hidden_states: torch.Tensor, **kwargs):
        residual = hidden_states
        attn_output, present_key_value = self.self_attn.forward_ragged(
            self.input_layernorm(hidden_states), **kwargs)
        hidden_states = residual + attn_output
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


class Transformer(nn.Module):
    """Full attention-only decoder stack.

    Stored as ``model`` inside :class:`CausalLM` so parameter keys
    carry the ``model.`` prefix matching safetensors checkpoint keys.

    Submodules: ``embed_tokens``, ``layers``, ``norm``.

    After ``forward``, the last-layer pre-norm hidden states are also exposed
    on ``self.last_pre_norm_hidden_states`` for subclasses that need to emit
    them as an extra ONNX output (see :class:`CausalLM.emit_hidden_states`).
    """

    #: Subclasses override to swap the decoder-layer implementation.
    decoder_layer_cls = DecoderLayer

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            type(self).decoder_layer_cls(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # Populated at the end of each forward; see module docstring.
        self.last_pre_norm_hidden_states: "torch.Tensor | None" = None

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        deepstack_embeds: Tuple[torch.Tensor, ...] = (),
        attention_mask: "torch.Tensor | None" = None,
        attention_pos_id: "torch.Tensor | None" = None,
        skip_softmax_scale: "torch.Tensor | None" = None,
        output_hidden_states: bool = False,
        dflash_target_layer_ids: "List[int] | None" = None,
    ) -> Tuple[torch.Tensor, Tuple, "Tuple | None", "torch.Tensor | None"]:
        hidden_states = inputs_embeds
        present_key_values_list: List[torch.Tensor] = []
        all_hidden_states: list = []
        dflash_hidden_by_layer: dict[int, torch.Tensor] = {}
        dflash_target_set = set(dflash_target_layer_ids or [])

        for layer_index, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states, next_key_value = layer(
                hidden_states,
                past_key_values[layer_index],
                rope_rotary_cos_sin,
                context_lengths,
                kvcache_start_index,
                kv_page_table,
                attention_mask=attention_mask,
                attention_pos_id=attention_pos_id,
                skip_softmax_scale=skip_softmax_scale,
            )
            present_key_values_list.append(next_key_value)

            # DFlash target hidden collection: after each target layer, before norm
            if layer_index in dflash_target_set:
                dflash_hidden_by_layer[layer_index] = hidden_states

            # Multimodal deepstack visual embedding (post-layer, first N layers).
            if layer_index < len(deepstack_embeds):
                hidden_states = hidden_states + deepstack_embeds[layer_index]

        # Expose the last-layer pre-norm residual stream (= final DecoderLayer
        # output) for subclasses that need it as an extra ONNX output.  This
        # matches the HuggingFace ``_can_record_outputs["hidden_states"] =
        # DecoderLayer`` hook point used by Qwen3-Omni Thinker, whose Talker
        # consumes exactly this tensor.
        self.last_pre_norm_hidden_states = hidden_states

        # Target-hidden concat for DFlash/DSpark-style draft feedback.
        # Stored as an attribute (like last_pre_norm_hidden_states) so that
        # Transformer.forward() keeps its 3-value return signature and
        # downstream callers (TTS talker, Qwen3.5 text, etc.) are unaffected.
        self.target_hidden_concat = _concat_hidden_in_provider_order(
            dflash_hidden_by_layer, dflash_target_layer_ids)
        # Backward-compatible alias used by existing DFlash callers.
        self.dflash_hidden_concat = self.target_hidden_concat

        normed = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(normed)

        return (normed, tuple(present_key_values_list),
                tuple(all_hidden_states) if output_hidden_states else None)

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
        deepstack_embeds: Tuple[torch.Tensor, ...] = (),
        skip_softmax_scale: "torch.Tensor | None" = None,
        output_hidden_states: bool = False,
        target_layer_ids: "List[int] | None" = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Tuple]:
        hidden_states = inputs_embeds.unsqueeze(1)
        present_key_values = []
        all_hidden_states = []
        target_hidden_by_layer: dict[int, torch.Tensor] = {}
        target_layer_set = set(target_layer_ids or [])
        for layer_index, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states, present_kv = layer.forward_ragged(
                hidden_states,
                past_key_value=past_key_values[layer_index],
                rope_rotary_cos_sin=rope_rotary_cos_sin,
                positions=positions,
                query_start_offsets=query_start_offsets,
                query_lengths=query_lengths,
                past_lengths=past_lengths,
                attention_sequence_lengths=attention_sequence_lengths,
                state_indices=state_indices,
                execution_phase_marker=execution_phase_marker,
                context_sequence_count_carrier=context_sequence_count_carrier,
                kv_page_table=kv_page_table,
                skip_softmax_scale=skip_softmax_scale,
                **kwargs)
            present_key_values.append(present_kv)
            if layer_index in target_layer_set:
                target_hidden_by_layer[layer_index] = hidden_states
            if layer_index < len(deepstack_embeds):
                hidden_states = hidden_states + deepstack_embeds[
                    layer_index].unsqueeze(1)
            self._after_ragged_layer(hidden_states, layer_index)
        self.last_pre_norm_hidden_states = hidden_states
        self.target_hidden_concat = _concat_hidden_in_provider_order(
            target_hidden_by_layer, target_layer_ids)
        self.dflash_hidden_concat = self.target_hidden_concat
        normed = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(normed)
        self.all_hidden_states = (tuple(all_hidden_states)
                                  if output_hidden_states else None)
        return normed, tuple(present_key_values)

    def _after_ragged_layer(self, hidden_states: torch.Tensor,
                            layer_index: int) -> None:
        """Allow subclasses to observe the post-deepstack residual."""


# ---------------------------------------------------------------------------
# CausalLM
# ---------------------------------------------------------------------------


class CausalLM(nn.Module):
    """Causal LM wrapper: Transformer + lm_head.

    The inner ``Transformer`` is stored as attribute ``model`` so all its
    parameters carry the ``model.`` prefix matching checkpoint key prefixes.

    Subclasses can set the class attribute ``emit_hidden_states = True`` to
    expose the full-sequence last-layer normed ``hidden_states`` as an extra
    ONNX output (needed by Qwen3-Omni for the thinker → talker handoff).
    This is independent of ``config.eagle_base``.
    """

    #: Subclasses override to True when the model must emit ``hidden_states``
    #: as an ONNX output in addition to ``logits``.
    emit_hidden_states: bool = False

    #: Subclasses override to swap the transformer stack implementation.
    transformer_cls = Transformer

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.model = type(self).transformer_cls(config)

        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")

    def tie_weights(self) -> None:
        """Clone embed_tokens.weight into lm_head.weight when tie_word_embeddings=True.

        HuggingFace checkpoints with ``tie_word_embeddings=True`` omit
        ``lm_head.weight`` from the saved file.  We clone (not share) the
        embedding weight so that lm_head and embed_tokens remain independent
        tensors, which is required for correct ONNX export.

        Called by :func:`~loader.load_weights` after checkpoint loading.

        Skipped when lm_head is a quantized module (e.g. FP8Linear in
        MIXED_PRECISION checkpoints) that stores its own weight.
        """
        if not self.config.tie_word_embeddings:
            return
        if not isinstance(self.lm_head, FP16Linear):
            return
        embed_weight = self.model.embed_tokens.weight
        self.lm_head.weight = nn.Parameter(embed_weight.detach().clone(),
                                           requires_grad=False)

    def onnx_export_spec(self) -> OnnxSpec:
        """Return all model-specific parameters needed for ONNX export.

        Builds dummy inputs, I/O name lists, and dynamic shape descriptors
        matching the token-major flat wrapper signature.

        When ``config.eagle_base``, ``config.dflash_base``,
        ``config.jetspec_base``, or ``config.dspark_base`` is True, extra inputs (``attention_pos_id``,
        ``attention_mask``) and an extra output (``hidden_states``) are added.
        For DFlash/JetSpec/DSpark base, hidden_states is the concatenated target-layer
        hidden (shape: [B, S, len(target_layer_ids)*H]).

        When ``config.eagle_base`` is True, extra inputs (``attention_pos_id``,
        ``attention_mask``) and an extra output (``hidden_states``) are added
        for EAGLE3 base-model tree-attention verification.
        """
        config = self.config
        dflash_base = getattr(config, 'dflash_base', False)
        jetspec_base = getattr(config, 'jetspec_base', False)
        dspark_base = getattr(config, 'dspark_base', False)
        target_hidden_base = dflash_base or jetspec_base or dspark_base
        # DFlash/JetSpec/DSpark base uses the same export structure as Eagle base
        # (tree-attention inputs + hidden_states output), so we treat it as
        # eagle_base for the wrapper.
        eagle_base = config.eagle_base or target_hidden_base
        return self._token_major_onnx_export_spec(tree_attention=eagle_base)

    def _token_major_onnx_export_spec(self,
                                      tree_attention: bool = False
                                      ) -> OnnxSpec:
        config = self.config
        Na = config.num_hidden_layers
        Nd = config.num_deepstack_features
        device = next(itertools.chain(self.parameters(),
                                      self.buffers())).device
        num_sequences = 2
        query_length = 2
        physical_tokens = num_sequences * query_length
        kv_dtype = (torch.float8_e4m3fn
                    if config.quant.kv_cache_quant == "fp8" else torch.float16)
        inputs_embeds = torch.zeros(physical_tokens,
                                    config.hidden_size,
                                    dtype=torch.float16,
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
        state_indices = torch.arange(num_sequences,
                                     dtype=torch.int32,
                                     device=device)
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
            [0, 2, 3], dtype=torch.int64, device=device) if tree_attention else
                          query_start_offsets[1:].to(torch.int64) - 1)
        deepstack_embeds = [
            torch.zeros(physical_tokens,
                        config.hidden_size,
                        dtype=torch.float16,
                        device=device) for _ in range(Nd)
        ]
        skip_softmax_scale = torch.zeros(2, dtype=torch.int8, device=device)
        args = (inputs_embeds, *past_key_values, rope_rotary_cos_sin,
                positions, query_start_offsets, query_lengths, past_lengths,
                attention_sequence_lengths, state_indices,
                execution_phase_marker, context_sequence_count_carrier,
                kv_page_table, logits_indices, *deepstack_embeds,
                skip_softmax_scale)
        input_names = (
            ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
                "rope_rotary_cos_sin", "positions", "query_start_offsets",
                "query_lengths", "past_lengths", "attention_sequence_lengths",
                "state_indices", "execution_phase_marker",
                "context_sequence_count_carrier", "kv_page_table",
                "logits_indices"
            ] + [f"deepstack_embeds_{i}"
                 for i in range(Nd)] + ["skip_softmax_scale"])
        dflash_base = getattr(config, "dflash_base", False)
        jetspec_base = getattr(config, "jetspec_base", False)
        dspark_base = getattr(config, "dspark_base", False)
        has_hidden_output = (self.emit_hidden_states or config.eagle_base
                             or dflash_base or jetspec_base or dspark_base)
        output_names = ["logits"
                        ] + [f"present_key_values_{i}" for i in range(Na)]
        if has_hidden_output:
            output_names.insert(1, "hidden_states")

        tokens = torch.export.Dim("physical_tokens", min=1, max=8_388_608)
        logits_rows = torch.export.Dim("logits_rows", min=1, max=8_388_608)
        sequences = torch.export.Dim("num_sequences", min=1, max=256)
        context_sequences = torch.export.Dim("num_context_sequences",
                                             min=0,
                                             max=256)
        max_pages = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
        num_pages = torch.export.Dim("num_pages", min=1, max=1048576)
        phase_extent = torch.export.Dim("execution_phase_extent", min=1, max=8)
        packed_mask_width = torch.export.Dim("packed_mask_width",
                                             min=1,
                                             max=64)
        skip_dim = torch.export.Dim("skip_softmax_scale_len",
                                    min=0,
                                    max=1048576)
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
                0: logits_rows if tree_attention else sequences
            },
        ])
        all_shapes.extend({0: tokens} for _ in range(Nd))
        all_shapes.append({0: skip_dim})
        if tree_attention:
            attention_position_ids = positions.clone()
            packed_attention_mask = torch.zeros(physical_tokens,
                                                (query_length + 31) // 32,
                                                dtype=torch.int32,
                                                device=device)
            tree_parent_ids = torch.full((physical_tokens, ),
                                         -1,
                                         dtype=torch.int32,
                                         device=device)
            tree_depths = torch.zeros(physical_tokens,
                                      dtype=torch.int32,
                                      device=device)
            valid_tree_counts = query_lengths.clone()
            args += (attention_position_ids, packed_attention_mask,
                     tree_parent_ids, tree_depths, valid_tree_counts)
            input_names += [
                "attention_position_ids", "packed_attention_mask",
                "tree_parent_ids", "tree_depths", "valid_tree_counts"
            ]
            all_shapes.extend([{
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
            }])
        wrapped = _make_flat_wrapper_ragged(self, Na, Nd, tree_attention)
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
        deepstack_embeds: Tuple[torch.Tensor, ...] = (),
        skip_softmax_scale: "torch.Tensor | None" = None,
        attention_position_ids: "torch.Tensor | None" = None,
        packed_attention_mask: "torch.Tensor | None" = None,
        tree_parent_ids: "torch.Tensor | None" = None,
        tree_depths: "torch.Tensor | None" = None,
        valid_tree_counts: "torch.Tensor | None" = None,
    ) -> Tuple:
        hidden_states, present_key_values = self.model.forward_ragged(
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
            deepstack_embeds,
            skip_softmax_scale,
            attention_position_ids=attention_position_ids,
            packed_attention_mask=packed_attention_mask,
            tree_parent_ids=tree_parent_ids,
            tree_depths=tree_depths,
            valid_tree_counts=valid_tree_counts,
            output_hidden_states=self.config.eagle_base,
            target_layer_ids=self._target_hidden_layer_ids())
        token_hidden_states = hidden_states.reshape(-1,
                                                    hidden_states.shape[-1])
        selected_hidden_states = torch.index_select(token_hidden_states, 0,
                                                    logits_indices)
        logits = self.lm_head(selected_hidden_states).to(torch.float32)
        logits = logits.reshape(-1, logits.shape[-1])
        emitted_hidden = self._ragged_emitted_hidden()
        return logits, emitted_hidden, present_key_values

    def _target_hidden_layer_ids(self) -> "List[int] | None":
        for enabled_name, layer_ids_name in (("dflash_base",
                                              "dflash_target_layer_ids"),
                                             ("jetspec_base",
                                              "jetspec_target_layer_ids"),
                                             ("dspark_base",
                                              "dspark_target_layer_ids")):
            if getattr(self.config, enabled_name, False):
                return getattr(self.config, layer_ids_name, None)
        return None

    def _ragged_emitted_hidden(self) -> "torch.Tensor | None":
        target_hidden = getattr(self.model, "target_hidden_concat", None)
        if target_hidden is not None:
            return target_hidden.reshape(-1, target_hidden.shape[-1])
        if self.config.eagle_base:
            all_hidden_states = self.model.all_hidden_states
            n_layers = len(all_hidden_states) - 1
            # The high feature is HF hidden_states[-4], including the final norm.
            layer_indices = [2, n_layers // 2, n_layers - 3]
            hidden_states = torch.cat(
                [all_hidden_states[i] for i in layer_indices],
                dim=-1).to(torch.float16)
            return hidden_states.reshape(-1, hidden_states.shape[-1])
        if self.emit_hidden_states:
            hidden_states = self.model.last_pre_norm_hidden_states
            return hidden_states.reshape(-1, hidden_states.shape[-1])
        return None

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        last_token_ids: torch.Tensor,
        deepstack_embeds: Tuple[torch.Tensor, ...] = (),
        attention_mask: "torch.Tensor | None" = None,
        attention_pos_id: "torch.Tensor | None" = None,
        skip_softmax_scale: "torch.Tensor | None" = None,
    ) -> Tuple:
        eagle_base = self.config.eagle_base
        dflash_base = getattr(self.config, 'dflash_base', False)
        jetspec_base = getattr(self.config, 'jetspec_base', False)
        dspark_base = getattr(self.config, 'dspark_base', False)
        target_hidden_base = dflash_base or jetspec_base or dspark_base
        dflash_target_layer_ids = getattr(self.config,
                                          'dflash_target_layer_ids', None)
        jetspec_target_layer_ids = getattr(self.config,
                                           'jetspec_target_layer_ids', None)
        dspark_target_layer_ids = getattr(self.config,
                                          'dspark_target_layer_ids', None)
        if jetspec_base:
            target_layer_ids = jetspec_target_layer_ids
        elif dspark_base:
            target_layer_ids = dspark_target_layer_ids
        else:
            target_layer_ids = dflash_target_layer_ids

        hidden_states, present_key_values, all_hidden_states = self.model(
            inputs_embeds,
            past_key_values,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            kv_page_table,
            deepstack_embeds,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
            skip_softmax_scale=skip_softmax_scale,
            output_hidden_states=eagle_base and not target_hidden_base,
            dflash_target_layer_ids=target_layer_ids
            if target_hidden_base else None,
        )
        target_hidden_concat = getattr(self.model, 'target_hidden_concat',
                                       None)

        # Select hidden states for specified token positions before lm_head.
        # last_token_ids: [batch, num_tokens] int64 -- indices into the seq dim.
        # Use trt::gather_nd so the ONNX export emits GatherND(batch_dims=1)
        # instead of GatherElements; TRT handles GatherND natively.
        selected_hidden_states = torch.ops.trt.gather_nd(
            hidden_states, last_token_ids)

        logits = self.lm_head(selected_hidden_states).to(torch.float32)

        if target_hidden_base and target_hidden_concat is not None:
            # DFlash/JetSpec/DSpark base: concatenate hidden states from target layers.
            # Output the full-sequence hidden states (NOT gathered) — the C++
            # runtime passes these to the draft engine per round.
            return logits, target_hidden_concat, present_key_values

        if eagle_base and all_hidden_states is not None:
            # EAGLE3 base: concatenate hidden states from 3 selected layers.
            # Layer indices: layer 2, middle layer, near-final layer.
            # Output the full-sequence hidden states (NOT gathered) — the C++
            # runtime selects accepted tokens from these after verification.
            n_layers = len(
                all_hidden_states) - 1  # last entry is normed output
            idx = [2, n_layers // 2, n_layers - 3]
            eagle_hidden = torch.cat([
                all_hidden_states[idx[0]],
                all_hidden_states[idx[1]],
                all_hidden_states[idx[2]],
            ],
                                     dim=-1).to(torch.float16)
            return logits, eagle_hidden, present_key_values

        if self.emit_hidden_states:
            # Full-sequence last-layer pre-norm residual, populated by
            # :meth:`Transformer.forward` (see its docstring for the HF
            # hidden_states hook-point alignment).
            return logits, self.model.last_pre_norm_hidden_states, \
                present_key_values

        return logits, present_key_values
