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
``torch.library.custom_op`` stubs for inference export.

Each op is a trace-time dummy (returns zero tensors of the correct shape/dtype)
paired with a ``register_fake`` for shape propagation in the dynamo exporter.
Domains ``trt::`` / ``trt_edgellm::`` map to ONNX nodes consumed by the
TensorRT plugin runtime.
"""

import logging
import os
from typing import List, Optional, Tuple

import torch

from .sm12x_nvfp4 import supports_sm12x_nvfp4_gemm_allreduce

logger = logging.getLogger(__name__)

#: Tokens per paged-KV page. Must match ``rt::kTOKENS_PER_PAGE``
#: (cpp/runtime/state/pagedKvTypes.h) — the AttentionPlugin's KV cache
#: binding is the pool ``[2, num_pages, KV_PAGE_SIZE, num_kv_heads, head_size]``.
KV_PAGE_SIZE = 128

from ..skip_softmax import SKIP_SOFTMAX_SCALE_TABLE_MAX_LEN  # noqa: F401

# ---------------------------------------------------------------------------
# NVFP4 target architecture selectors
# ---------------------------------------------------------------------------
#
# ``Nvfp4MoePlugin`` (SM100/101/103/110, split FC1/FC2) and ``NvFP4MoEPluginGeforce``
# (SM12x/Blackwell consumer, fused) share the same 11-input ONNX surface and
# attribute set, but consume **different FC1 weight layouts** for SwiGLU MoE:
#
#   * SM100/101/103/110 expect FC1 packed as the 64-row up/gate interleave that
#     ``repack_nvfp4_gated_moe_experts(..., fc1_layout="interleave")`` produces.
#   * SM12x expects FC1 packed as the plain ``[up_all, gate_all]`` concat
#     that ``repack_nvfp4_gated_moe_experts(..., fc1_layout="concat")`` produces.
#
# Repacking and modeling code call :func:`use_geforce_nvfp4_moe` to pick the
# matching plugin op and FC1 layout at export time. Override via env var:
#
#   EDGELLM_NVFP4_MOE_TARGET=sm100  -> Nvfp4MoePlugin
#   EDGELLM_NVFP4_MOE_TARGET=sm103  -> Nvfp4MoePlugin
#   EDGELLM_NVFP4_MOE_TARGET=sm110  -> Nvfp4MoePlugin            (default)
#   EDGELLM_NVFP4_MOE_TARGET=sm12x  -> NvFP4MoEPluginGeforce
#
# Accepted aliases for SM12x: ``sm120``, ``sm121``, ``geforce``.

_NVFP4_MOE_SM110_TARGET_ALIASES = frozenset(
    ("sm100", "sm101", "sm103", "sm110", "blackwell_dc", "thor", ""))
_NVFP4_GEMM_ALLREDUCE_FUSED_TARGET_ALIASES = frozenset(
    ("sm100", "sm101", "sm103", "sm110", "blackwell_dc", "thor", ""))
_NVFP4_SM12X_TARGET_ALIASES = frozenset(("sm12x", "sm120", "sm121", "geforce"))

_NVFP4_MOE_TARGET_ENV = "EDGELLM_NVFP4_MOE_TARGET"
_NVFP4_GEMM_ALLREDUCE_TARGET_ENV = \
    "EDGELLM_NVFP4_GEMM_ALLREDUCE_TARGET"


def targets_sm12x_nvfp4() -> bool:
    """Return whether host-side NVFP4 export targets SM120 or SM121."""
    target = os.environ.get(_NVFP4_GEMM_ALLREDUCE_TARGET_ENV,
                            "sm110").strip().lower()
    if target in _NVFP4_SM12X_TARGET_ALIASES:
        return True
    if target in _NVFP4_GEMM_ALLREDUCE_FUSED_TARGET_ALIASES:
        return False
    raise ValueError(
        f"{_NVFP4_GEMM_ALLREDUCE_TARGET_ENV}={target!r} is not recognized. "
        "Use 'sm100'/'sm110' or 'sm12x'. Aliases: "
        "sm101/sm103/blackwell_dc/thor, sm120/sm121/geforce.")


def use_sm12x_nvfp4_gemm_allreduce(config) -> bool:
    """Select the qualified SM12x CuTeDSL GEMM + NCCL all-reduce path.

    ONNX export is host-independent, so the target must remain explicit. Other
    model sizes and families keep the portable TensorRT GEMM plus AllReduce
    lowering.
    """
    return (targets_sm12x_nvfp4()
            and supports_sm12x_nvfp4_gemm_allreduce(config))


def use_generic_nvfp4_gemm_allreduce(config) -> bool:
    """Return whether an SM12x export must use portable GEMM + AllReduce."""
    return (targets_sm12x_nvfp4()
            and not supports_sm12x_nvfp4_gemm_allreduce(config))


def use_geforce_nvfp4_moe() -> bool:
    """Return True iff exporting for ``NvFP4MoEPluginGeforce`` (SM12x).

    The default is the SM100/101/103/110 path so existing export pipelines keep producing
    the 64-row up/gate interleave layout consumed by ``Nvfp4MoePlugin``.
    """
    val = os.environ.get(_NVFP4_MOE_TARGET_ENV, "sm110").strip().lower()
    if val in _NVFP4_SM12X_TARGET_ALIASES:
        return True
    if val in _NVFP4_MOE_SM110_TARGET_ALIASES:
        return False
    raise ValueError(
        f"{_NVFP4_MOE_TARGET_ENV}={val!r} is not recognized. Use 'sm100'/'sm103'/'sm110' "
        "(Nvfp4MoePlugin) or 'sm12x' (Blackwell consumer/NvFP4MoEPluginGeforce). "
        "Aliases: sm101/blackwell_dc/thor, sm120/sm121/geforce.")


# ---------------------------------------------------------------------------
# Custom op: trt::attention_plugin  (unified: vanilla / FP8-KV / EAGLE tree)
# ---------------------------------------------------------------------------


def _validate_attention_plugin_inputs(
        qkv: torch.Tensor, enable_tree_attention: bool,
        attention_mask: Optional[torch.Tensor],
        attention_pos_id: Optional[torch.Tensor]) -> None:
    if qkv.ndim != 2:
        raise ValueError("AttentionPlugin qkv must use token-major rank-2 ABI")
    if enable_tree_attention:
        if attention_mask is None or attention_mask.ndim != 2:
            raise ValueError(
                "AttentionPlugin tree mask must use rank-2 [T_exec, words] ABI"
            )
        if attention_pos_id is None or attention_pos_id.ndim != 1:
            raise ValueError(
                "AttentionPlugin tree position IDs must use rank-1 [T_exec] ABI"
            )


@torch.library.custom_op("trt::attention_plugin", mutates_args=())
def attention_plugin(
    qkv: torch.Tensor,
    past_key_value: torch.Tensor,
    query_lengths: torch.Tensor,
    rope_rotary_cos_sin: torch.Tensor,
    past_lengths: torch.Tensor,
    kv_page_table: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    sliding_window_size: int,
    enable_tree_attention: bool,
    enable_fp8_kv_cache: bool,
    attention_scale: float,
    enable_context_mask_selector: bool,
    enable_vision_block_attention: bool,
    skip_softmax_prefill_input: int,
    context_mask_selector: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    attention_pos_id: Optional[torch.Tensor] = None,
    qkv_scales: Optional[List[float]] = None,
    q_norm_gamma: Optional[List[float]] = None,
    k_norm_gamma: Optional[List[float]] = None,
    # Callers pass the real config value explicitly; the default matches the C++ side
    # (attentionPlugin.h: mRmsNormEps{1e-6f}) so a stripped attr still behaves correctly.
    rms_norm_eps: float = 1e-6,
    # Default 0 so torch.export strips the kwarg for non-qk_norm models.
    enable_qk_norm: int = 0,
    # QK-norm order: 0 = norm then RoPE (Qwen3), 1 = RoPE then norm (HunYuan V1).
    # Only meaningful when enable_qk_norm=1. Default 0 so torch.export strips it.
    qk_norm_post_rope: int = 0,
    # Whether this layer reads K/V from a donated (shared) cache. Ordinary full-cache
    # and spec layers carry Q only. Runtime-selectable SWA consumers also carry the
    # current donor K/V for bounded prefill. Default 0 for normal layers.
    enable_kv_shared: int = 0,
    # Tensor-parallel world size of the exporting rank. TP>1 decodes on XQA on every SM
    # (the CTM kernels serve TP=1 only). Default 1 so torch.export strips it.
    tp_size: int = 1,
    # Skip-softmax scale table: 1-D INT32 of flattened (seqlen_bin, S) pairs the
    # runtime uploads from the engine config.json (length 0 = dense cubin). Default
    # None so torch.export strips it for models without skip-softmax.
    skip_softmax_prefill_scale: Optional[torch.Tensor] = None,
    # Decode skip-softmax scale table, same format; wired iff
    # skip_softmax_decode_input == 1 (a decode table is deployed). Defaults
    # None / 0 so torch.export strips them otherwise.
    skip_softmax_decode_scale: Optional[torch.Tensor] = None,
    skip_softmax_decode_input: int = 0,
    # Shape-only runtime policy selector for bounded-capable SWA layers. A
    # length of 1 selects bounded O(W) storage; length 0 selects full storage.
    swa_kv_cache_mode: Optional[torch.Tensor] = None,
    attention_sinks: Optional[List[float]] = None,
    enable_attention_sink: int = 0,
    enable_contiguous_query_swa: int = 0,
    query_start_offsets: Optional[torch.Tensor] = None,
    attention_sequence_lengths: Optional[torch.Tensor] = None,
    execution_phase_marker: Optional[torch.Tensor] = None,
    context_sequence_count_carrier: Optional[torch.Tensor] = None,
    prefix_key_value: Optional[torch.Tensor] = None,
    prefix_page_table: Optional[torch.Tensor] = None,
    external_kv_prefix: int = 0,
    fp8_kv_prefix: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified stub for AttentionPlugin covering all feature combinations.

    Feature matrix (all map to the same TRT ``AttentionPlugin``):

    +-----------------------+--------------------+----------------------------+
    | Mode                  | enable_tree_attn   | enable_fp8_kv_cache        |
    +=======================+====================+============================+
    | Vanilla               | False              | False                      |
    +-----------------------+--------------------+----------------------------+
    | FP8 KV cache          | False              | True  (qkv_scales set)     |
    +-----------------------+--------------------+----------------------------+
    | EAGLE tree attention  | True               | False                      |
    +-----------------------+--------------------+----------------------------+
    | EAGLE + FP8 KV        | True               | True  (qkv_scales set)     |
    +-----------------------+--------------------+----------------------------+

    ``kv_page_table`` is a required ``[batch, 2, max_pages_per_seq]`` int32
    per-request page table (K page ids then derived V page ids).

    ``past_key_value`` / ``present_key_value`` are the same paged-pool
    allocation, in-place aliased by the TRT plugin (no growth per call —
    the pool is a fixed-size allocation and writes land at page-table
    positions). ``present_key_value`` therefore always has the exact same
    shape as ``past_key_value``. All models declare the pool shape
    ``[2, num_pages, kTOKENS_PER_PAGE, num_kv_heads, head_size]``.

    ``enable_tree_attention``, ``enable_fp8_kv_cache``,
    ``enable_context_mask_selector``, ``enable_vision_block_attention``,
    ``attention_scale``, and ``skip_softmax_prefill_input`` are required
    (no default) so that ``torch.export`` always includes them in the FX
    graph — default-matching kwargs get stripped, breaking ONNX translation.

    ``skip_softmax_prefill_input`` / ``skip_softmax_decode_input`` say which
    skip-softmax (BLASST) table inputs the layer carries; S itself comes from
    the ``skip_softmax_prefill_scale`` / ``skip_softmax_decode_scale`` tables
    the runtime uploads from the engine config.

    Callers must always pass ``qkv_scales=[1.0, 1.0, 1.0]`` explicitly so
    the FX graph contains a valid FLOATS value for the ONNX translation.

    When ``enable_tree_attention=True``, ``attention_mask`` and
    ``attention_pos_id`` must be provided (non-None).

    When ``enable_vision_block_attention=True``, the ``attention_mask`` input
    carries a ``[T_exec]`` INT32 vision-block-ID tensor instead of a
    tree mask.  ``-1`` means causal text/audio; equal non-negative IDs identify
    one contiguous image run whose tokens may attend bidirectionally to
    each other.  This mode is mutually exclusive with tree attention.

    ``qkv`` is the token-major PACKED projection output
    ``[T_exec, (Hq + 2*Hkv) * D]``
    (Q/K/V concatenated on the last dim — either a single fused QKV GEMM
    output or ``torch.cat`` of the three separate projections), or
    ``[T_exec, Hq * D]`` (Q only) for ordinary shared-KV layers, or
    ``[T_exec, (Hq + 2*Hkv) * D]`` when a bounded SWA consumer also carries the
    donor's current K/V transiently. Shared layers never write the donor cache.

    The presence of the shape-only ``swa_kv_cache_mode`` final optional input
    advertises bounded-storage capability. Its length selects bounded O(W) or
    full storage at runtime.

    The TRT AttentionPlugin kernel returns a 3-D tensor
    ``[T_exec, num_q_heads, head_size]``.
    The caller (``Attention.forward``) is responsible for reshaping to
    ``[T_exec, num_q_heads * head_size]``.
    """
    _validate_attention_plugin_inputs(qkv, enable_tree_attention,
                                      attention_mask, attention_pos_id)
    attn_output = torch.zeros(qkv.shape[0],
                              num_q_heads,
                              head_size,
                              dtype=qkv.dtype,
                              device=qkv.device)
    present_key_value = torch.zeros_like(past_key_value)
    return attn_output, present_key_value


@attention_plugin.register_fake
def _(
    qkv,
    past_key_value,
    query_lengths,
    rope_rotary_cos_sin,
    past_lengths,
    kv_page_table,
    num_q_heads,
    num_kv_heads,
    head_size,
    sliding_window_size,
    enable_tree_attention,
    enable_fp8_kv_cache,
    attention_scale,
    enable_context_mask_selector,
    enable_vision_block_attention,
    skip_softmax_prefill_input,
    context_mask_selector=None,
    attention_mask=None,
    attention_pos_id=None,
    qkv_scales=None,
    q_norm_gamma=None,
    k_norm_gamma=None,
    rms_norm_eps=1e-6,
    enable_qk_norm=0,
    qk_norm_post_rope=0,
    enable_kv_shared=0,
    tp_size=1,
    skip_softmax_prefill_scale=None,
    skip_softmax_decode_scale=None,
    skip_softmax_decode_input=0,
    swa_kv_cache_mode=None,
    attention_sinks=None,
    enable_attention_sink=0,
    enable_contiguous_query_swa=0,
    query_start_offsets=None,
    attention_sequence_lengths=None,
    execution_phase_marker=None,
    context_sequence_count_carrier=None,
    prefix_key_value=None,
    prefix_page_table=None,
    external_kv_prefix=0,
    fp8_kv_prefix=0,
):
    _validate_attention_plugin_inputs(qkv, enable_tree_attention,
                                      attention_mask, attention_pos_id)
    output_shape = (qkv.shape[0], num_q_heads, head_size)
    return (torch.empty(*output_shape, dtype=qkv.dtype,
                        device=qkv.device), torch.empty_like(past_key_value))


# ---------------------------------------------------------------------------
# Custom op: trt::vit_attention_plugin  (ViT ragged self-attention)
# ---------------------------------------------------------------------------

# Head dims with an FP8 ViT FMHA kernel (build_cutedsl.py
# ``vit_fmha_d{64,80,96,128}_fp8``; mirrored by ViTAttentionPlugin's runtime
# check). d80 pads the MMA tiler K to 96 inside the kernel. d72 is blocked
# by TMA 16-byte stride alignment (72 FP8 bytes) and has no direct kernel;
# d72 models (Phi-4 SigLIP) widen their q/k/v projections to d=80 with zero
# weight rows at load time instead (see Phi4MMVisionAttention) and gate on
# the padded dim. Visual models gate ``enable_fp8_mha`` on this.
FP8_VIT_HEAD_DIMS = (64, 80, 96, 128)


def init_fp8_mha(attn: torch.nn.Module,
                 model_config,
                 head_dim: int,
                 *,
                 k_proj: Optional[torch.nn.Module] = None,
                 v_proj: Optional[torch.nn.Module] = None) -> bool:
    """Gate the FP8 ViT MHA path and register its dequant-scale buffers.

    Returns True when the checkpoint requests FP8 visual MHA
    (``quant.visual_mha_quant == "fp8"``) and *head_dim* has an FP8 kernel
    (``FP8_VIT_HEAD_DIMS``). On True, registers per-tensor scale buffers at
    the exact paths modelopt's calibration frontend writes:

    - ``<attn>.q_scale`` — module-level (modelopt has no q_bmm rename; see
      ``_surface_visual_q_scales`` in ``tensorrt_edgellm/quantization``),
    - ``<attn>.k_proj.k_scale`` / ``<attn>.v_proj.v_scale`` — on the real
      *k_proj* / *v_proj* linears when passed (separate-projection
      attentions), else on synthetic ``nn.Module()`` placeholders created
      here (fused-qkv attentions) so ``load_state_dict`` still picks the
      buffers up.

    ``setup_fp8_qkv_scales_for_export`` reads these at export time and
    stores ``_qkv_scales_float`` on *attn*, which
    :func:`quantize_qkv_for_fp8_mha` threads to the ViT attention plugin.
    """
    enabled = (getattr(model_config.quant, "visual_mha_quant", None) == "fp8"
               and head_dim in FP8_VIT_HEAD_DIMS)
    if not enabled:
        return False
    attn.register_buffer("q_scale", torch.ones((), dtype=torch.float16))
    if k_proj is None:
        k_proj = torch.nn.Module()
        attn.k_proj = k_proj
    if v_proj is None:
        v_proj = torch.nn.Module()
        attn.v_proj = v_proj
    k_proj.register_buffer("k_scale", torch.ones((), dtype=torch.float16))
    v_proj.register_buffer("v_scale", torch.ones((), dtype=torch.float16))
    return True


def quantize_qkv_for_fp8_mha(
    attn: torch.nn.Module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[float]]:
    """FP8-quantize FP16 q/k/v with the calibrated per-tensor scales when
    *attn*'s FP8 MHA gate is on; pass-through otherwise. Returns
    ``(q, k, v, qkv_scales)`` ready for the vit_attention_plugin call."""
    qkv_scales = getattr(attn, "_qkv_scales_float", [1.0, 1.0, 1.0])
    if attn.enable_fp8_mha:
        q = fp8_quantize(q, attn.q_scale)
        k = fp8_quantize(k, attn.k_proj.k_scale)
        v = fp8_quantize(v, attn.v_proj.v_scale)
    return q, k, v, qkv_scales


@torch.library.custom_op("trt::vit_attention_plugin", mutates_args=())
def vit_attention_plugin(
    query_states: torch.Tensor,  # [T, num_heads, head_size]
    key_states: torch.Tensor,  # [T, num_heads, head_size]
    value_states: torch.Tensor,  # [T, num_heads, head_size]
    cu_seqlens: torch.Tensor,  # [batch+1] int32
    max_seqlen_carrier: torch.Tensor,  # [max_seqlen] int32, length only
    num_heads: int,
    head_size: int,
    attention_scale: float,
    qkv_scales: "list[float] | None" = None,
) -> torch.Tensor:
    """ViT ragged self-attention.

    In eager mode, implements varlen SDPA using cu_seqlens to process each
    sequence segment independently.  During dynamo/ONNX tracing the
    register_fake shape propagation is used and this body is not executed.

    Unlike AttentionPlugin, ViT attention has no KV cache and takes ragged
    input with cu_seqlens instead of context_lengths.  RoPE is applied before
    this call.

    FP8 mode is carried by the tensors themselves: when Q/K/V arrive as
    float8_e4m3fn (quantized by preceding QuantizeLinear nodes in the graph),
    the plugin runs the FP8 FMHA kernel and ``qkv_scales`` carries the
    per-tensor dequant scales it folds into the softmax/output scaling.
    Output stays FP16.
    """
    import torch.nn.functional as F
    if query_states.dtype == torch.float8_e4m3fn:
        # Eager reference: dequantize back to FP16 with the per-tensor scales,
        # mirroring the plugin's internal fold.
        scales = qkv_scales or [1.0, 1.0, 1.0]
        query_states = query_states.to(torch.float16) * scales[0]
        key_states = key_states.to(torch.float16) * scales[1]
        value_states = value_states.to(torch.float16) * scales[2]
    out = torch.empty_like(query_states, dtype=torch.float16)
    seqlens = cu_seqlens.tolist()
    for i in range(len(seqlens) - 1):
        start, end = int(seqlens[i]), int(seqlens[i + 1])
        if start >= end:
            continue
        # q/k/v: [S, H, D] -> [1, H, S, D] for SDPA
        q = query_states[start:end].permute(1, 0, 2).unsqueeze(0)
        k = key_states[start:end].permute(1, 0, 2).unsqueeze(0)
        v = value_states[start:end].permute(1, 0, 2).unsqueeze(0)
        attn = F.scaled_dot_product_attention(
            q, k, v, scale=attention_scale)  # [1, H, S, D]
        out[start:end] = attn.squeeze(0).permute(1, 0, 2)
    return out


@vit_attention_plugin.register_fake
def _(query_states,
      key_states,
      value_states,
      cu_seqlens,
      max_seqlen_carrier,
      num_heads,
      head_size,
      attention_scale,
      qkv_scales=None):
    # The plugin always emits FP16 (FP8 inputs are dequantized internally),
    # so the fake must not inherit an FP8 input dtype.
    return torch.empty_like(query_states, dtype=torch.float16)


# ---------------------------------------------------------------------------
# Custom op: trt::trt_ragged_attention  (TRT-native ViT ragged self-attention)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::trt_ragged_attention", mutates_args=())
def trt_ragged_attention(
        query_states: torch.Tensor,  # [T, num_heads, head_size]
        key_states: torch.Tensor,  # [T, num_heads, head_size]
        value_states: torch.Tensor,  # [T, num_heads, head_size]
        query_lengths: torch.Tensor,  # [batch+1] int32
        kv_lengths: torch.Tensor,  # [batch+1] int32
        num_heads: int,
        head_size: int,
        attention_scale: float,
        mask: Optional[torch.Tensor] = None,  # optional attention mask
) -> torch.Tensor:
    """TRT-native ragged self-attention proxy op (TRT >= 11).

    Emits trt::TRT_Attention ONNX node instead of the edgellm plugin.
    ``attention_scale`` is the absolute multiplier applied to QK^T. The ONNX
    translation folds non-identity values into Q before emitting TRT attention.
    query_lengths and kv_lengths must be separate tensors (not the same
    object) — TRT requires distinct inputs for these positions.
    ``mask`` is optional; pass ``None`` for unmasked attention.
    """
    return torch.empty_like(query_states)


@trt_ragged_attention.register_fake
def _(query_states,
      key_states,
      value_states,
      query_lengths,
      kv_lengths,
      num_heads,
      head_size,
      attention_scale,
      mask=None):
    return torch.empty_like(query_states)


def is_trt_native_attention_enabled():
    """Check if TRT-native attention is enabled."""
    is_enabled = os.environ.get("USE_TRT_NATIVE_ATTN") == "1"
    if is_enabled:
        msg = "Using TRT-native attention (TRT_Attention)"
    else:
        msg = "Using attention plugin or non-fused attention"
    logger.info(msg)
    return is_enabled


def is_audio_attention_plugin_enabled():
    """Check if the audio encoder routes attention through ViTAttentionPlugin.

    On by default; ``USE_AUDIO_ATTN_PLUGIN=0`` exports the dense-mask graph.
    """
    return os.environ.get("USE_AUDIO_ATTN_PLUGIN", "1") != "0"


# ---------------------------------------------------------------------------
# Custom op: trt::fp8_quantize
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::dflash_target_kv_cache_update
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::dflash_target_kv_cache_update",
                         mutates_args=())
def dflash_target_kv_cache_update(
    k_delta: torch.Tensor,
    v_delta: torch.Tensor,
    past_key_value: torch.Tensor,
    token_aligned_rope_cos_sin: torch.Tensor,
    delta_positions: torch.Tensor,
    delta_token_to_sequence: torch.Tensor,
    kv_page_table: torch.Tensor,
) -> torch.Tensor:
    """Update the draft combined KV cache with target-hidden-derived K/V delta.

    k_delta: [T_delta, numKVHeads, headDim] FP16, k_normed, not RoPE-applied.
    v_delta: [T_delta, numKVHeads, headDim] FP16.
    past_key_value: [2, num_pages, KV_PAGE_SIZE, numKVHeads, headDim] FP16 —
        the paged KV pool (same contract as the AttentionPlugin kv_cache
        binding).
    token_aligned_rope_cos_sin: [T_delta, rotaryDim] FP32.
    delta_positions: [T_delta] INT32; padding rows use -1.
    delta_token_to_sequence: [T_delta] INT32; padding rows use -1.
    kv_page_table: [B, 2, maxPagesPerSeq] INT32 with canonical K page ids in
        [0, num_pages) and V page ids in [num_pages, 2 * num_pages).

    Applies RoPE to k_delta and writes k_rope + v_delta into the KV cache at
    positions [delta_start, delta_start + t) for each batch element, where
    t < delta_lengths[b]. Positions beyond delta_lengths[b] are skipped.
    Returns present_key_value (same shape as past_key_value — aliased in TRT).
    """
    return past_key_value.clone()


@dflash_target_kv_cache_update.register_fake
def _(k_delta, v_delta, past_key_value, token_aligned_rope_cos_sin,
      delta_positions, delta_token_to_sequence, kv_page_table):
    return torch.empty_like(past_key_value)


# ---------------------------------------------------------------------------
# DFlash2 fixed linear-path proposal plugins
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::dflash2_grouped_dynamic_conv",
                         mutates_args=())
def dflash2_grouped_dynamic_conv(hidden_states: torch.Tensor,
                                 delta: torch.Tensor,
                                 base_kernel: torch.Tensor,
                                 residual: Optional[torch.Tensor],
                                 block_size: int, kernel_size: int,
                                 group_size: int,
                                 fuse_residual: int) -> torch.Tensor:
    """Trace-time proxy for DFlash2GroupedDynamicConvPlugin."""
    del delta, base_kernel, block_size, kernel_size, group_size, fuse_residual
    if residual is not None:
        return torch.empty_like(residual)
    return torch.empty_like(hidden_states)


@dflash2_grouped_dynamic_conv.register_fake
def _(hidden_states, delta, base_kernel, residual, block_size, kernel_size,
      group_size, fuse_residual):
    del delta, base_kernel, block_size, kernel_size, group_size, fuse_residual
    if residual is not None:
        return torch.empty_like(residual)
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Low-rank candidate head (LowRankCandidateHeadPlugin)
# ---------------------------------------------------------------------------


def _candidate_order_keys(values: torch.Tensor,
                          token_ids: torch.Tensor) -> torch.Tensor:
    """Pack fp32 ``values`` and token IDs into int64 keys ordered as the plugin's.

    The plugin key is ``(orderable(value) << 32) | (0xFFFFFFFF - id)`` as u64
    (larger value first, then smaller ID). NaN maps to orderable 0, below every
    real value, and -0 maps to +0. Subtracting 2**31 from the orderable word
    keeps the same order in signed int64.
    """
    values = torch.where(values == 0, torch.zeros_like(values), values)
    bits = values.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    orderable = torch.where(bits >= 0x80000000, bits ^ 0xFFFFFFFF,
                            bits | 0x80000000)
    orderable = torch.where(torch.isnan(values), torch.zeros_like(orderable),
                            orderable)
    return (orderable - 0x80000000) * 0x100000000 + (0xFFFFFFFF - token_ids)


@torch.library.custom_op("trt_edgellm::low_rank_candidate_head",
                         mutates_args=())
def low_rank_candidate_head(
        hidden: torch.Tensor, down_proj: torch.Tensor, up_proj: torch.Tensor,
        lm_head: torch.Tensor,
        num_candidates: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select and score ``num_candidates`` full-vocabulary tokens per row.

    Args:
        hidden: ``[R, H]`` fp16 final-norm hidden states.
        down_proj: ``[r, H]`` fp16 low-rank factor B.
        up_proj: ``[V, r]`` fp16 low-rank factor A.
        lm_head: ``[V, H]`` fp16 output projection.
        num_candidates: Kc.

    Returns:
        ``support_ids [R, Kc]`` int32 sorted by (exact logit desc, id asc) and
        ``support_log_probs [R, Kc]`` fp32, the log_softmax over the Kc exact
        logits. Candidates are the top Kc of ``A (B h)``: ``B h`` accumulates
        in fp32 and is stored in fp16, the scores ``A s`` are fp32. Exact
        logits ``lm_head[id] . h`` are fp32 dot products; a non-finite one
        sorts as -inf and gets log-prob -inf.
    """
    vocab = up_proj.shape[0]
    hidden32 = hidden.float()
    projected = (hidden32 @ down_proj.float().t()).to(torch.float16)
    scores = projected.float() @ up_proj.float().t()
    vocab_ids = torch.arange(vocab, dtype=torch.int64, device=hidden.device)
    candidate_ids = torch.topk(_candidate_order_keys(scores, vocab_ids),
                               num_candidates,
                               dim=-1).indices
    logits = torch.einsum("rkh,rh->rk", lm_head[candidate_ids].float(),
                          hidden32)
    finite = torch.isfinite(logits)
    ranked = torch.where(finite, logits,
                         torch.full_like(logits, float("-inf")))
    order = torch.argsort(_candidate_order_keys(ranked, candidate_ids),
                          dim=-1,
                          descending=True)
    candidate_ids = candidate_ids.gather(-1, order)
    ranked = ranked.gather(-1, order)
    finite = finite.gather(-1, order)
    log_norm = torch.logsumexp(ranked, dim=-1, keepdim=True)
    log_probs = torch.where(finite, ranked - log_norm,
                            torch.full_like(ranked, float("-inf")))
    return candidate_ids.to(torch.int32), log_probs


@low_rank_candidate_head.register_fake
def _(hidden, down_proj, up_proj, lm_head, num_candidates):
    del down_proj, up_proj, lm_head
    rows = hidden.shape[0]
    return (hidden.new_empty((rows, num_candidates), dtype=torch.int32),
            hidden.new_empty((rows, num_candidates), dtype=torch.float32))


# ---------------------------------------------------------------------------
# FP8 fake-quant eager helpers (numeric-validation golden)
#
# As with the NVFP4 helpers above, the eager bodies of the FP8 ops were zero
# stubs (export-only). The golden runs the model eagerly, so they must compute
# the real per-tensor FP8 (E4M3) quantize / dequantize. ``fp8_quantize`` keeps
# the fp16 output dtype of register_fake (it holds the fp8-grid value of x/scale
# in fp16), so ``fp8_dequantize`` -- which multiplies by the scale -- works
# uniformly whether fed the quantized activation or the raw fp8 weight buffer.
# ---------------------------------------------------------------------------

# E4M3 finite max (float8_e4m3fn has no inf; overflow casts to NaN, so saturate first).
_FP8_E4M3_MAX = 448.0


@torch.library.custom_op("trt::ple_pool_lookup", mutates_args=())
def ple_pool_lookup(pool: torch.Tensor, row_map: torch.Tensor) -> torch.Tensor:
    """Select raw rows with shape [*row_map.shape, row_width]."""
    if pool.dtype == torch.float8_e4m3fn:
        return pool.view(torch.uint8)[row_map.long()].view(pool.dtype)
    return pool[row_map.long()]


@ple_pool_lookup.register_fake
def _(pool, row_map):
    return torch.empty((*row_map.shape, pool.shape[1]),
                       dtype=pool.dtype,
                       device=pool.device)


@torch.library.custom_op("trt::ple_pool_gather", mutates_args=())
def ple_pool_gather(pool: torch.Tensor, row_map: torch.Tensor,
                    scale: torch.Tensor) -> torch.Tensor:
    """Gather FP8 PLE rows before dequantizing with a scalar output-dtype scale.

    Args:
        pool: FP8 E4M3 row pool with shape [capacity, row_width].
        row_map: INT32 indices [tokens, heads] or [batch, sequence, heads].
        scale: Scalar BF16 or FP16 global dequantization scale.

    Returns:
        Embeddings [*row_map.shape[:-1], heads * row_width] in the scale dtype.
    """
    rows = pool.view(torch.uint8)[row_map.long()].view(torch.float8_e4m3fn)
    return (rows.float() * scale.float()).to(scale.dtype).flatten(-2)


@ple_pool_gather.register_fake
def _(pool, row_map, scale):
    return torch.empty(
        (*row_map.shape[:-1], row_map.shape[-1] * pool.shape[1]),
        dtype=scale.dtype,
        device=pool.device)


def _fp8_quantize_eager(hidden_states: torch.Tensor,
                        scale: torch.Tensor) -> torch.Tensor:
    """Per-tensor FP8 E4M3 quantize: round (x / scale) onto the FP8 grid.

    Returns fp16 holding the fp8-grid value (still divided by ``scale``); the
    paired ``fp8_dequantize`` multiplies the scale back. Saturates to the E4M3
    max so an out-of-range value clamps (like the kernel) instead of becoming NaN.
    """
    s = scale.to(torch.float32)
    q = (hidden_states.to(torch.float32) / s).clamp(-_FP8_E4M3_MAX,
                                                    _FP8_E4M3_MAX)
    return q.to(torch.float8_e4m3fn).to(torch.float16)


def _fp8_dequantize_eager(weight: torch.Tensor,
                          weight_scale: torch.Tensor) -> torch.Tensor:
    """Per-tensor FP8 dequantize: value * scale -> fp16.

    ``weight`` is either the fp8 weight buffer or the fp16 output of
    ``_fp8_quantize_eager`` (already on the fp8 grid); ``.to(float16)`` is the
    real fp8 value in both cases.
    """
    return (weight.to(torch.float32) * weight_scale.to(torch.float32)).to(
        torch.float16)


@torch.library.custom_op("trt::fp8_quantize", mutates_args=())
def fp8_quantize(
        hidden_states: torch.Tensor,  # float16 input
        scale: torch.Tensor,  # float16 per-tensor scale (scalar)
) -> torch.Tensor:
    """Quantize float16 -> FP8 (per-tensor E4M3); ONNX export -> QuantizeLinear.

    Eager body computes the real fake-quant for the numeric-validation golden; export uses
    register_fake and emits the ONNX node, so the real body does not affect it.
    """
    return _fp8_quantize_eager(hidden_states, scale)


@fp8_quantize.register_fake
def _(hidden_states, scale):
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Custom op: trt::fp8_dequantize
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::fp8_dequantize", mutates_args=())
def fp8_dequantize(
        weight: torch.Tensor,  # fp8_e4m3fn [out, in]
        weight_scale: torch.Tensor,  # float16 per-tensor scale (scalar)
) -> torch.Tensor:
    """Dequantize FP8 -> float16 (value * scale); ONNX export -> DequantizeLinear.

    Eager body computes the real dequant for the numeric-validation golden; export uses
    register_fake and emits the ONNX node, so the real body does not affect it.
    """
    return _fp8_dequantize_eager(weight, weight_scale)


@fp8_dequantize.register_fake
def _(weight, weight_scale):
    return torch.empty_like(weight, dtype=torch.float16)


# ---------------------------------------------------------------------------
# NVFP4 fake-quant eager helpers (numeric-validation golden)
#
# The custom ops below normally only emit ONNX nodes; their eager bodies were
# zero stubs. For the PyTorch golden we run the model eagerly, so the bodies
# must compute the real NVFP4 fake-quant. These helpers mirror EdgeLLM's own
# offline decoder ``checkpoint/repacking.decode_modelopt_nvfp4`` exactly (same
# E2M1 levels, same nibble order: low nibble = even index, same
# value*block_scale*scale_2 formula) so the golden matches the engine's NVFP4
# definition rather than a third-party convention.
# ---------------------------------------------------------------------------

# E2M1 positive levels (index 0..7) and the midpoints used to round to them.
# Kept in sync with tensorrt_edgellm/checkpoint/repacking.py.
_FP4_E2M1_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_FP4_E2M1_BOUNDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]


def _nvfp4_dequantize_eager(weight: torch.Tensor, weight_scale: torch.Tensor,
                            weight_scale_2: torch.Tensor,
                            group_size: int) -> torch.Tensor:
    """Dequantize packed NVFP4 weight to float16 (torch, on-device).

    Torch port of ``repacking.decode_modelopt_nvfp4``: ``weight`` is
    ``[out, in//2]`` int8/uint8 with two E2M1 nibbles per byte (low nibble = even
    index); ``weight_scale`` is ``[out, in//group_size]`` FP8 E4M3 (or an int8/uint8
    view / float cast of it); ``weight_scale_2`` is a fp32 per-tensor scalar.
    """
    device = weight.device
    w = weight.view(torch.uint8) if weight.dtype == torch.int8 else weight
    w = w.to(torch.int64)  # 0..255, so the right-shift below is logical
    out_f, half = w.shape
    nibbles = torch.empty(out_f, half * 2, dtype=torch.int64, device=device)
    nibbles[:, 0::2] = w & 0x0F
    nibbles[:, 1::2] = (w >> 4) & 0x0F
    sign = (nibbles & 0x08) != 0
    magnitude = nibbles & 0x07
    levels = torch.tensor(_FP4_E2M1_LEVELS, dtype=torch.float32, device=device)
    values = torch.where(sign, -levels[magnitude], levels[magnitude])

    if weight_scale.dtype in (torch.int8, torch.uint8):
        ws = weight_scale.view(torch.float8_e4m3fn).to(torch.float32)
    else:
        ws = weight_scale.to(torch.float32)
    ws2 = weight_scale_2.to(torch.float32).reshape(1)
    num_groups = ws.shape[-1]
    in_f = num_groups * group_size
    dense = values.reshape(out_f, num_groups, group_size) * ws.reshape(
        out_f, num_groups, 1) * ws2
    return dense.reshape(out_f, in_f).to(torch.float16)


def _nvfp4_act_qdq_eager(hidden_states: torch.Tensor,
                         global_scale: torch.Tensor,
                         block_size: int = 16) -> torch.Tensor:
    """Dynamic per-block NVFP4 fake-quant of activations (torch, fp16 out).

    Mirrors ``TRT_FP4DynamicQuantize`` + 2x DQ: along the last dim, every
    ``block_size`` elements share a scale derived dynamically from the block
    amax; values are rounded to the E2M1 grid and dequantized back.
    ``global_scale`` is the per-tensor scale-2 (the calibrated ``input_scale``).
    """
    orig_dtype = hidden_states.dtype
    device = hidden_states.device
    x = hidden_states.to(torch.float32)
    *lead, last = x.shape
    nb = last // block_size
    xb = x.reshape(*lead, nb, block_size)

    s2 = global_scale.to(torch.float32).reshape(1)
    per_block_scale = xb.abs().amax(dim=-1, keepdim=True) / 6.0  # [..., nb, 1]
    # Quantize the block scale through FP8 E4M3 (as the kernel does), guard zeros.
    # Saturate to the E4M3 max (448) before the cast: float8_e4m3fn has no inf, so an
    # overflowing value would become NaN instead of clamping like the hardware kernel.
    q_block_scale = (per_block_scale / s2).clamp(max=448.0).to(
        torch.float8_e4m3fn).to(torch.float32)
    q_block_scale = torch.where(per_block_scale == 0,
                                torch.ones_like(q_block_scale), q_block_scale)
    block_scale = (q_block_scale * s2).clamp_min(1e-20)
    # The engine quantizes against the fp32 scale above but dequantizes against an
    # fp16 one: the exported graph casts global_scale to fp16 before the block-scale
    # DQ (see dynamo_translations._nvfp4_act_qdq_translation). Keeping this in fp32
    # leaves ~half the elements off by an ULP of the scale.
    dq_scale = (q_block_scale.to(torch.float16) * s2.to(torch.float16)).to(
        torch.float32).clamp_min(1e-20)

    # Round magnitude to the nearest E2M1 level (midpoints in _FP4_E2M1_BOUNDS).
    bounds = torch.tensor(_FP4_E2M1_BOUNDS, dtype=torch.float32, device=device)
    levels = torch.tensor(_FP4_E2M1_LEVELS, dtype=torch.float32, device=device)
    scaled = xb / block_scale
    magnitude = scaled.abs().contiguous()
    idx = torch.searchsorted(bounds, magnitude)
    # On a midpoint the candidate E2M1 codes are idx and idx+1; searchsorted picks
    # idx (toward zero) while the converter keeps the even code.
    on_midpoint = magnitude == bounds[idx.clamp(max=len(_FP4_E2M1_BOUNDS) - 1)]
    idx = torch.where(on_midpoint & (idx % 2 == 1), idx + 1, idx)
    deq = torch.sign(scaled) * levels[idx] * dq_scale
    return deq.reshape(*lead, last).to(orig_dtype)


# ---------------------------------------------------------------------------
# Custom op: trt::nvfp4_act_qdq  (activation DynQ + 2DQ -> float16)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::nvfp4_act_qdq", mutates_args=())
def nvfp4_act_qdq(
        hidden_states: torch.Tensor,  # float16 activation
        global_scale: torch.Tensor,  # float32 scalar: amax / (6.0 * 448.0)
) -> torch.Tensor:
    """Stub: NVFP4 activation QDQ (DynQ + 2 trt::DQ). Returns float16.

    In the ONNX graph this emits three nodes matching ModelOpt's
    ``export_fp4(onnx_quantizer_type="dynamic")`` pattern::

        TRT_FP4DynamicQuantize(x, scale_f32, axis=-1, block_size=16, scale_type=17)
            -> (x_f4, sx_f8)
        trt::DequantizeLinear(sx_f8, scale_f32)
            -> dq_scale
        trt::DequantizeLinear(x_f4, dq_scale, axis=-1, block_size=16)
            -> x_dq  [float16]

    The eager body computes the real fake-quant (used by the numeric-
    validation golden); ``torch.export`` uses ``register_fake`` for tracing and
    emits the three ONNX nodes above, so the real body does not affect export.
    """
    return _nvfp4_act_qdq_eager(hidden_states, global_scale)


@nvfp4_act_qdq.register_fake
def _(hidden_states, global_scale):
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Custom op: trt::nvfp4_dequantize
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::nvfp4_dequantize", mutates_args=())
def nvfp4_dequantize(
    weight: torch.Tensor,  # uint8 [out, in//2] packed fp4
    weight_scale: torch.Tensor,  # fp8_e4m3fn [out, in//group_size]
    weight_scale_2: torch.Tensor,  # float32 scalar
    group_size: int,
) -> torch.Tensor:
    """Dequantize NVFP4 packed weight to float16.

    Eager body computes the real dequant (used by the numeric-validation
    golden); ``torch.export`` uses ``register_fake`` for tracing and emits the
    two ``trt::DequantizeLinear`` nodes, so the real body does not affect export.
    """
    return _nvfp4_dequantize_eager(weight, weight_scale, weight_scale_2,
                                   group_size)


@nvfp4_dequantize.register_fake
def _(weight, weight_scale, weight_scale_2, group_size):
    # Rank-agnostic: dense linears pass [out, in//2]; MoE passes a stacked
    # [E, out, in//2]. Only the packed last dim is doubled.
    *lead, packed_in = weight.shape
    return torch.empty(*lead,
                       packed_in * 2,
                       dtype=torch.float16,
                       device=weight.device)


# ---------------------------------------------------------------------------
# MXFP8 fake-quant eager helpers (numeric-validation golden)
#
# MXFP8 (OCP microscaling): FP8 E4M3 elements with a per-block (block_size=32)
# power-of-two shared scale stored as E8M0 (uint8, value V -> 2^(V-127)).
# Activation is dynamically quantized at runtime; weights carry a precomputed
# E8M0 scale. As with NVFP4/FP8, the eager bodies were zero stubs (export-only).
# ---------------------------------------------------------------------------

# E8M0 exponent bias and FP8 E4M3 element emax / finite-max (OCP MX spec).
_E8M0_BIAS = 127
_FP8_E4M3_EMAX = 8  # largest binary exponent of an E4M3 normal (448 = 1.75 * 2^8)


def _mxfp8_weight_dq_eager(weight: torch.Tensor, weight_scale: torch.Tensor,
                           block_size: int) -> torch.Tensor:
    """Dequantize an MXFP8 weight (FP8 E4M3 + per-block E8M0 scale) to fp16."""
    out_f, in_f = weight.shape
    nb = in_f // block_size
    w = weight.to(torch.float32).reshape(out_f, nb, block_size)
    scale = torch.exp2(weight_scale.to(torch.float32) - _E8M0_BIAS).reshape(
        out_f, nb, 1)
    return (w * scale).reshape(out_f, in_f).to(torch.float16)


def _mxfp8_act_qdq_eager(hidden_states: torch.Tensor,
                         block_size: int = 32) -> torch.Tensor:
    """Dynamic per-block MXFP8 fake-quant of activations (fp16 out).

    Mirrors ``TRT_MXFP8DynamicQuantize`` + ``DequantizeLinear``: per block of
    ``block_size`` along the last dim, pick an E8M0 (power-of-two) shared scale so
    the block amax lands at the top of E4M3's range, quantize to FP8, dequantize.
    """
    orig_dtype = hidden_states.dtype
    x = hidden_states.to(torch.float32)
    *lead, last = x.shape
    nb = last // block_size
    xb = x.reshape(*lead, nb, block_size)
    amax = xb.abs().amax(dim=-1, keepdim=True)  # [..., nb, 1]
    safe = torch.where(amax > 0, amax, torch.ones_like(amax))
    # E8M0 shared exponent: floor(log2(amax)) - emax(E4M3), clamped to E8M0 range.
    shared_exp = (torch.floor(torch.log2(safe)) - _FP8_E4M3_EMAX).clamp(
        -_E8M0_BIAS, _E8M0_BIAS)
    scale = torch.where(amax > 0, torch.exp2(shared_exp),
                        torch.ones_like(amax))
    q = (xb / scale).clamp(-448.0,
                           448.0).to(torch.float8_e4m3fn).to(torch.float32)
    return (q * scale).reshape(*lead, last).to(orig_dtype)


# ---------------------------------------------------------------------------
# Custom op: trt::mxfp8_act_qdq  (activation DynQ + DQ -> float16)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::mxfp8_act_qdq", mutates_args=())
def mxfp8_act_qdq(
        hidden_states: torch.Tensor,  # float16 activation
) -> torch.Tensor:
    """MXFP8 activation DynQ + DQ -> float16.

    Eager body computes the real fake-quant for the numeric-validation golden; export uses
    register_fake and emits the two ONNX nodes below, so the body is export-safe::

        TRT_MXFP8DynamicQuantize(x, axis=-1, block_size=32, output_dtype=17)
            -> (x_f8, sx_e8m0)
        TRT_MXFP8DequantizeLinear(x_f8, sx_e8m0,
            axis=-1, block_size=32, output_dtype=10)
            -> x_dq  [float16]
    """
    return _mxfp8_act_qdq_eager(hidden_states)


@mxfp8_act_qdq.register_fake
def _(hidden_states):
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Custom op: trt::mxfp8_weight_dq
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::mxfp8_weight_dq", mutates_args=())
def mxfp8_weight_dq(
    weight: torch.Tensor,  # fp8_e4m3fn [out, in]
    weight_scale: torch.Tensor,  # uint8 (E8M0) [out, in // block_size]
    block_size: int,
) -> torch.Tensor:
    """Dequantize MXFP8 weight (FP8E4M3 + per-block E8M0 scale) to float16.

    Eager body computes the real dequant for the numeric-validation golden; export uses
    register_fake and emits the ONNX node, so the body is export-safe::

        TRT_MXFP8DequantizeLinear(weight, weight_scale,
            axis=-1, block_size=block_size, output_dtype=10) -> w_dq [float16]
    """
    return _mxfp8_weight_dq_eager(weight, weight_scale, block_size)


@mxfp8_weight_dq.register_fake
def _(weight, weight_scale, block_size):
    return torch.empty(weight.shape[0],
                       weight.shape[1],
                       dtype=torch.float16,
                       device=weight.device)


# ---------------------------------------------------------------------------
# INT4 groupwise GEMM plugin backend selector
# ---------------------------------------------------------------------------

_INT4_PLUGIN_LOGGED = False

# Export-time backend selection, set once from the export CLI
# (``--int4-gemm-plugin-version``) via :func:`set_int4_gemm_plugin_version`. Defaults to
# ``2`` (cuteDSL) so callers that never set it keep the V2 behavior.
_INT4_GEMM_PLUGIN_VERSION = 2


def set_int4_gemm_plugin_version(version: int) -> None:
    """Select the INT4 groupwise GEMM plugin backend for the current export.

    Called once from the export CLI (``--int4-gemm-plugin-version``) before the weight
    repack / custom-op emission run. ``1`` selects the AWQ
    ``Int4GroupwiseGemmPlugin``; any other value selects the cuteDSL
    ``Int4GroupwiseGemmPluginV2`` (default). See :func:`int4_gemm_plugin_version`.
    """
    global _INT4_GEMM_PLUGIN_VERSION, _INT4_PLUGIN_LOGGED
    _INT4_GEMM_PLUGIN_VERSION = 1 if version == 1 else 2
    _INT4_PLUGIN_LOGGED = False  # re-log the backend after an explicit change


def int4_gemm_plugin_version() -> int:
    """Which INT4 groupwise GEMM plugin backend the export should target.

    Selected at export time via the ``--int4-gemm-plugin-version`` CLI argument (it is
    not a quantize-time property -- the quantized checkpoint is backend-agnostic):

      * ``2`` -> cuteDSL ``Int4GroupwiseGemmPluginV2`` (default; weights repacked
        to the tile-independent fragment layout, ``bN=128``/``bK=64``).
      * ``1`` -> AWQ ``Int4GroupwiseGemmPlugin`` (legacy fallback; weights
        repacked to the AWQ swizzle by ``_pack_intweights``).

    Both the weight repack (``checkpoint/repacking.py``) and the emitted custom op
    (``models/linear.py``) call this so they always agree within one export run.
    """
    version = _INT4_GEMM_PLUGIN_VERSION

    global _INT4_PLUGIN_LOGGED
    if not _INT4_PLUGIN_LOGGED:
        _INT4_PLUGIN_LOGGED = True
        backend = ("Int4GroupwiseGemmPluginV2 (cuteDSL fragment weights)"
                   if version == 2 else
                   "Int4GroupwiseGemmPlugin (AWQ swizzled weights)")
        logger.info("INT4 groupwise GEMM plugin backend: %s", backend)
    return version


# Custom op: trt::int4_groupwise_gemm (AWQ swizzled weights)
# ---------------------------------------------------------------------------
# ``qweight`` is the AWQ swizzle buffer (INT8 [out_features//2, in_features],
# repacked at export time by checkpoint/repacking.py). Emitted when
# int4_gemm_plugin_version() == 1; maps to the Int4GroupwiseGemmPlugin ONNX node.


@torch.library.custom_op("trt::int4_groupwise_gemm", mutates_args=())
def int4_groupwise_gemm(
    hidden_states: torch.Tensor,  # [*, in_features] float16
    qweight: torch.Tensor,  # [out_features//2, in_features] int8 (swizzled)
    scales: torch.Tensor,  # [in_features//group_size, out_features] float16
    gemm_n: int,
    gemm_k: int,
    group_size: int,
) -> torch.Tensor:
    """Stub: INT4 groupwise GEMM - returns zero tensor of correct shape."""
    *leading, _ = hidden_states.shape
    return torch.zeros(*leading,
                       gemm_n,
                       dtype=hidden_states.dtype,
                       device=hidden_states.device)


@int4_groupwise_gemm.register_fake
def _(hidden_states, qweight, scales, gemm_n, gemm_k, group_size):
    *leading, _ = hidden_states.shape
    return torch.empty(*leading,
                       gemm_n,
                       dtype=hidden_states.dtype,
                       device=hidden_states.device)


# Custom op: trt::int4_groupwise_gemm_v2 (cuteDSL fragment-layout weights)
# ---------------------------------------------------------------------------
# Same signature and output as trt::int4_groupwise_gemm, but ``qweight`` is the
# cuteDSL fragment-order buffer (INT8 view of the uint32 [rows, 128] tile) instead
# of the AWQ swizzle. Emitted when int4_gemm_plugin_version() == 2 (default); maps
# to the Int4GroupwiseGemmPluginV2 ONNX node.


@torch.library.custom_op("trt::int4_groupwise_gemm_v2", mutates_args=())
def int4_groupwise_gemm_v2(
    hidden_states: torch.Tensor,  # [*, in_features] float16
    qweight: torch.Tensor,  # cuteDSL fragment buffer, INT8 [rows, 512]
    scales: torch.Tensor,  # [in_features//group_size, out_features] float16
    gemm_n: int,
    gemm_k: int,
    group_size: int,
) -> torch.Tensor:
    """Stub: cuteDSL INT4 groupwise GEMM - returns zero tensor of correct shape."""
    *leading, _ = hidden_states.shape
    return torch.zeros(*leading,
                       gemm_n,
                       dtype=hidden_states.dtype,
                       device=hidden_states.device)


@int4_groupwise_gemm_v2.register_fake
def _(hidden_states, qweight, scales, gemm_n, gemm_k, group_size):
    *leading, _ = hidden_states.shape
    return torch.empty(*leading,
                       gemm_n,
                       dtype=hidden_states.dtype,
                       device=hidden_states.device)


# ---------------------------------------------------------------------------
# Custom op: trt::qkv_concat
# ---------------------------------------------------------------------------

# Q8_0LinearPlugin ``out_dtype`` values (TensorRT DataType).
Q8_0_OUT_DTYPES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 7}


@torch.library.custom_op("trt::q8_0_linear_plugin", mutates_args=())
def q8_0_linear_plugin(hidden: torch.Tensor, packed_weight: torch.Tensor,
                       out_dtype: int) -> torch.Tensor:
    """Reference for llama.cpp Q8_0 x Q8_1 multiplication.

    FP16 or BF16 activations; ``out_dtype`` is a TensorRT DataType value.
    """
    if (hidden.dtype not in (torch.float16, torch.bfloat16)
            or packed_weight.dtype != torch.int8):
        raise TypeError("Q8_0 linear requires FP16 or BF16 activations and "
                        "INT8 packed bytes")
    out_features, blocks, payload = packed_weight.shape
    if payload != 34 or hidden.shape[-1] != blocks * 32:
        raise ValueError("invalid Q8_0 weight geometry")
    hidden = hidden.float()
    result = torch.zeros(*hidden.shape[:-1],
                         out_features,
                         dtype=torch.float32,
                         device=hidden.device)
    for block in range(blocks):
        raw = packed_weight[:, block, :]
        scales = raw[:, :2].contiguous().view(torch.float16).squeeze(-1)
        codes = raw[:, 2:].contiguous().view(torch.int8)
        values = hidden[..., block * 32:(block + 1) * 32]
        peak = values.abs().amax(dim=-1, keepdim=True)
        activation_scale = torch.where(peak > 0, peak / 127, 1)
        quantized = torch.round(values / activation_scale).clamp(-127, 127)
        result += (torch.matmul(quantized,
                                codes.float().T) * activation_scale *
                   scales.float())
    return result.to(_q8_0_torch_dtype(out_dtype))


def _q8_0_torch_dtype(out_dtype: int) -> torch.dtype:
    for dtype, value in Q8_0_OUT_DTYPES.items():
        if value == out_dtype:
            return dtype
    raise ValueError(f"unsupported Q8_0 out_dtype {out_dtype}")


@q8_0_linear_plugin.register_fake
def _(hidden, packed_weight, out_dtype):
    return torch.empty(*hidden.shape[:-1],
                       packed_weight.shape[0],
                       dtype=_q8_0_torch_dtype(out_dtype),
                       device=hidden.device)


@torch.library.custom_op("trt::q8_0_qkv_plugin", mutates_args=())
def q8_0_qkv_plugin(hidden: torch.Tensor, q_weight: torch.Tensor,
                    k_weight: torch.Tensor, v_weight: torch.Tensor,
                    out_dtype: int) -> torch.Tensor:
    """Q8_0 Q, K and V projections of one activation, packed along the last axis."""
    return torch.cat([
        q8_0_linear_plugin(hidden, weight, out_dtype)
        for weight in (q_weight, k_weight, v_weight)
    ],
                     dim=-1)


@q8_0_qkv_plugin.register_fake
def _(hidden, q_weight, k_weight, v_weight, out_dtype):
    columns = q_weight.shape[0] + k_weight.shape[0] + v_weight.shape[0]
    return torch.empty(*hidden.shape[:-1],
                       columns,
                       dtype=_q8_0_torch_dtype(out_dtype),
                       device=hidden.device)


@torch.library.custom_op("trt::w2a16_gemm_plugin", mutates_args=())
def w2a16_gemm_plugin(hidden: torch.Tensor, values: torch.Tensor,
                      bases: torch.Tensor, lookup: torch.Tensor, gemm_n: int,
                      gemm_k: int) -> torch.Tensor:
    """W2A16GemmPlugin placeholder for retiled IQ2_XXS weights and BF16 activations."""
    if hidden.dtype != torch.bfloat16 or hidden.shape[-1] != gemm_k:
        raise ValueError("W2A16 GEMM expects BF16 activations with K columns")
    return torch.empty(*hidden.shape[:-1],
                       gemm_n,
                       dtype=hidden.dtype,
                       device=hidden.device)


@w2a16_gemm_plugin.register_fake
def _(hidden, values, bases, lookup, gemm_n, gemm_k):
    return torch.empty(*hidden.shape[:-1],
                       gemm_n,
                       dtype=hidden.dtype,
                       device=hidden.device)


@torch.library.custom_op("trt::qkv_concat", mutates_args=())
def qkv_concat(q: torch.Tensor, k: torch.Tensor,
               v: torch.Tensor) -> torch.Tensor:
    """Pack contiguous Q/K/V projections without exposing an ONNX Concat."""
    return torch.cat((q, k, v), dim=-1)


@qkv_concat.register_fake
def _(q, k, v):
    if q.shape[:-1] != k.shape[:-1] or q.shape[:-1] != v.shape[:-1]:
        raise ValueError("Q, K, and V must have identical leading shapes")
    return torch.empty(*q.shape[:-1],
                       q.shape[-1] + k.shape[-1] + v.shape[-1],
                       dtype=q.dtype,
                       device=q.device)


# ---------------------------------------------------------------------------
# NVFP4-A16 export target selector (dense GEMM and Nemotron-H routed MoE)
# ---------------------------------------------------------------------------

_NVFP4_A16_BLACKWELL_TARGET_SM = 110
_NVFP4_A16_EXPORT_TARGET_SM: Optional[int] = None
_NVFP4_A16_TARGET_LOGGED = False


def set_nvfp4_a16_export_target_sm(target_sm: Optional[int]) -> None:
    """Select the NVFP4-A16 plugins from an explicit export target.

    ``SM110`` selects ``Nvfp4A16BlackwellGemmPlugin`` (``BLACKWELL_N128_K64_V1``)
    for dense linears and ``Nvfp4A16BlackwellMoePlugin``
    (``BLACKWELL_MOE_N128_K64_V1``) for Nemotron-H routed experts. Every other
    target, including an omitted target, preserves the Marlin
    ``Nvfp4A16GemmPlugin`` / ``Nvfp4A16MoePlugin`` contracts. The selector
    intentionally never probes the export host GPU so an x86 cross-export is
    deterministic.
    """
    if target_sm is not None and (not isinstance(target_sm, int) or isinstance(
            target_sm, bool) or target_sm <= 0):
        raise ValueError("target_sm must be a positive integer or None")
    global _NVFP4_A16_EXPORT_TARGET_SM, _NVFP4_A16_TARGET_LOGGED
    _NVFP4_A16_EXPORT_TARGET_SM = target_sm
    _NVFP4_A16_TARGET_LOGGED = False


def nvfp4_a16_export_target_sm() -> Optional[int]:
    """Return the explicitly configured dense NVFP4-A16 export SM."""
    return _NVFP4_A16_EXPORT_TARGET_SM


def use_blackwell_nvfp4_a16_gemm() -> bool:
    """Whether dense NVFP4-A16 linears should use the SM110 plugin."""
    use_blackwell = (
        _NVFP4_A16_EXPORT_TARGET_SM == _NVFP4_A16_BLACKWELL_TARGET_SM)
    global _NVFP4_A16_TARGET_LOGGED
    if not _NVFP4_A16_TARGET_LOGGED:
        _NVFP4_A16_TARGET_LOGGED = True
        target = (f"SM{_NVFP4_A16_EXPORT_TARGET_SM}" if
                  _NVFP4_A16_EXPORT_TARGET_SM is not None else "unspecified")
        backend = ("Nvfp4A16BlackwellGemmPlugin (BLACKWELL_N128_K64_V1)"
                   if use_blackwell else "Nvfp4A16GemmPlugin (Marlin)")
        logger.info("Dense NVFP4-A16 export target %s: %s", target, backend)
    return use_blackwell


def use_blackwell_nvfp4_a16_moe() -> bool:
    """Whether Nemotron-H routed NVFP4-A16 experts should use the SM110 plugin.

    Reads the same explicit export target as
    :func:`use_blackwell_nvfp4_a16_gemm` so the dense and routed W4A16 paths of
    one export always agree. ``SM110`` selects ``Nvfp4A16BlackwellMoePlugin``
    over ``BLACKWELL_MOE_N128_K64_V1``; every other target, including an
    omitted one, keeps the Marlin ``Nvfp4A16MoePlugin``.
    """
    return _NVFP4_A16_EXPORT_TARGET_SM == _NVFP4_A16_BLACKWELL_TARGET_SM


# ---------------------------------------------------------------------------
# Custom op: trt::nvfp4_a16_gemm  (dense FP16-A / NVFP4-W4 Marlin GEMM)
#
# Inputs are already in Marlin-packed layout (see
# ``repacking.repack_nvfp4_a16_marlin_linear``): ``qweights`` are the INT8 view
# of the Marlin-permuted E2M1 codes, ``block_scales`` are the Marlin-permuted
# raw E4M3 bytes, and ``global_scale`` is the FP16 per-tensor scale pre-scaled
# by 2**7. ``gemm_n`` is the Marlin-padded output width; the caller slices the
# logical width. Export-only (zero eager stub); numeric validation is the
# golden checkpoint test, mirroring ``int4_groupwise_gemm``.
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::nvfp4_a16_gemm", mutates_args=())
def nvfp4_a16_gemm(
    activation: torch.Tensor,  # [*, gemm_k] float16
    qweights: torch.Tensor,  # [1, gemm_k//16, 8*gemm_n] int8
    block_scales: torch.Tensor,  # [1, gemm_k//16, gemm_n] int8
    global_scale: torch.Tensor,  # [1] float16
    gemm_n: int,
    gemm_k: int,
) -> torch.Tensor:
    """Stub: dense NVFP4 W4A16 Marlin GEMM - returns zero tensor of output shape."""
    *leading, _ = activation.shape
    return torch.zeros(*leading,
                       gemm_n,
                       dtype=activation.dtype,
                       device=activation.device)


@nvfp4_a16_gemm.register_fake
def _(activation, qweights, block_scales, global_scale, gemm_n, gemm_k):
    *leading, _ = activation.shape
    return torch.empty(*leading,
                       gemm_n,
                       dtype=activation.dtype,
                       device=activation.device)


# ---------------------------------------------------------------------------
# Custom op: trt::nvfp4_a16_blackwell_gemm
#
# SM110-only dense FP16/BF16-A / NVFP4-W4 GEMM. The checkpoint-provided
# ModelOpt buffers are repacked to the opaque BLACKWELL_N128_K64_V1 ABI:
# qweights [N/128,K/64,128,32], block scales [N/128,K/64,128,4], and the
# unmodified FP32 per-tensor multiplier. The ONNX translation preserves the
# dedicated ``trt_edgellm::Nvfp4A16BlackwellGemmPlugin`` identity.
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::nvfp4_a16_blackwell_gemm", mutates_args=())
def nvfp4_a16_blackwell_gemm(
    activation: torch.Tensor,  # [*, gemm_k] float16/bfloat16
    qweights: torch.Tensor,  # [gemm_n//128, gemm_k//64, 128, 32] int8
    block_scales: torch.Tensor,  # [gemm_n//128, gemm_k//64, 128, 4] int8
    global_scale: torch.Tensor,  # [1] float32
    gemm_n: int,
    gemm_k: int,
) -> torch.Tensor:
    """Stub for the SM110 dense NVFP4-A16 Blackwell plugin."""
    *leading, _ = activation.shape
    return torch.zeros(*leading,
                       gemm_n,
                       dtype=activation.dtype,
                       device=activation.device)


@nvfp4_a16_blackwell_gemm.register_fake
def _(activation, qweights, block_scales, global_scale, gemm_n, gemm_k):
    *leading, _ = activation.shape
    return torch.empty(*leading,
                       gemm_n,
                       dtype=activation.dtype,
                       device=activation.device)


# ---------------------------------------------------------------------------
# INT8 SmoothQuant fake-quant eager helpers (numeric-validation golden)
#
# W8A8: symmetric per-tensor INT8 activation, symmetric per-channel INT8 weight.
# As with the other recipes, the eager bodies were zero stubs (export-only).
# ---------------------------------------------------------------------------

# Symmetric INT8 uses [-127, 127] (not -128), matching ONNX QuantizeLinear sym.
_INT8_SYM_MAX = 127.0


def _int8_sq_act_qdq_eager(hidden_states: torch.Tensor,
                           scale: torch.Tensor) -> torch.Tensor:
    """Symmetric per-tensor INT8 quantize+dequantize of the (smoothed) activation."""
    s = scale.to(torch.float32)
    q = torch.round(hidden_states.to(torch.float32) / s).clamp(
        -_INT8_SYM_MAX, _INT8_SYM_MAX)
    return (q * s).to(torch.float16)


def _int8_sq_weight_dq_eager(weight: torch.Tensor,
                             scale: torch.Tensor) -> torch.Tensor:
    """Per-channel (axis=0) INT8 weight dequantize: value * scale[out] -> fp16."""
    return (weight.to(torch.float32) *
            scale.to(torch.float32).reshape(-1, 1)).to(torch.float16)


# ---------------------------------------------------------------------------
# Custom op: trt::int8_sq_act_qdq  (INT8 SmoothQuant activation QDQ)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::int8_sq_act_qdq", mutates_args=())
def int8_sq_act_qdq(
        hidden_states: torch.Tensor,  # float16 smoothed activation [*, in]
        scale: torch.Tensor,  # float32 per-tensor input scale []
) -> torch.Tensor:
    """Symmetric per-tensor INT8 QuantizeLinear + DequantizeLinear.

    Eager body computes the real fake-quant for the numeric-validation golden; export uses
    register_fake and emits the ONNX nodes below, so the body is export-safe::

        QuantizeLinear(x, scale, output_dtype=INT8) -> q
        DequantizeLinear(q, scale)                  -> dq  [float32]
        Cast(dq, to=FLOAT16)                        -> output
    """
    return _int8_sq_act_qdq_eager(hidden_states, scale)


@int8_sq_act_qdq.register_fake
def _(hidden_states, scale):
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Custom op: trt::int8_sq_weight_dq  (INT8 per-channel weight dequantize)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::int8_sq_weight_dq", mutates_args=())
def int8_sq_weight_dq(
        weight: torch.Tensor,  # int8 [out, in]
        scale: torch.Tensor,  # float32 [out] per-channel scale
) -> torch.Tensor:
    """Per-channel INT8 DequantizeLinear (axis=0), output float16.

    Eager body computes the real dequant for the numeric-validation golden; export uses
    register_fake and emits the ONNX node, so the body is export-safe::

        DequantizeLinear(weight, scale, axis=0) -> dq  [float32]
        Cast(dq, to=FLOAT16)                    -> output
    """
    return _int8_sq_weight_dq_eager(weight, scale)


@int8_sq_weight_dq.register_fake
def _(weight, scale):
    return torch.empty_like(weight, dtype=torch.float16)


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::causal_conv1d
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::causal_conv1d", mutates_args=())
def causal_conv1d(
    hidden_states: torch.Tensor,  # [T_exec, conv_dim]
    weight: torch.Tensor,  # [conv_dim, 1, kernel_size]
    bias: torch.Tensor,  # [conv_dim]
    conv_state: torch.Tensor,  # [resident_rows, conv_dim, conv_kernel]
    query_lengths: torch.Tensor,  # [N] int32
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    query_start_offsets: torch.Tensor,
    state_indices: torch.Tensor,
    execution_phase_marker: torch.Tensor,
    context_sequence_count_carrier: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stub: causal conv1d."""
    return (torch.zeros_like(hidden_states), conv_state.clone(),
            conv_state.clone())


@causal_conv1d.register_fake
def _(hidden_states, weight, bias, conv_state, query_lengths, stride, padding,
      dilation, groups, query_start_offsets, state_indices,
      execution_phase_marker, context_sequence_count_carrier):
    return (torch.empty_like(hidden_states), conv_state.clone(),
            torch.empty_like(conv_state))


@torch.library.custom_op("trt_edgellm::causal_conv1d_with_intermediate",
                         mutates_args=())
def causal_conv1d_with_intermediate(
    hidden_states: torch.Tensor,  # [T_exec, conv_dim]
    weight: torch.Tensor,  # [conv_dim, 1, kernel_size]
    bias: torch.Tensor,  # [conv_dim]
    conv_state: torch.Tensor,  # [resident_rows, conv_dim, conv_kernel]
    query_lengths: torch.Tensor,  # [N] int32
    query_start_offsets: torch.Tensor,
    state_indices: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    execution_phase_marker: torch.Tensor,
    context_sequence_count_carrier: torch.Tensor,
    tree_parent_ids: Optional[torch.Tensor] = None,  # [T_exec] int32
    tree_depths: Optional[torch.Tensor] = None,  # [T_exec] int32
    spec_verify_layout: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stub: causal conv1d with per-token intermediate state output."""
    intermediate_conv_state = torch.zeros(hidden_states.shape[0],
                                          conv_state.shape[1],
                                          conv_state.shape[2],
                                          dtype=conv_state.dtype,
                                          device=conv_state.device)
    return torch.zeros_like(
        hidden_states), conv_state.clone(), intermediate_conv_state


@causal_conv1d_with_intermediate.register_fake
def _(hidden_states,
      weight,
      bias,
      conv_state,
      query_lengths,
      query_start_offsets,
      state_indices,
      stride,
      padding,
      dilation,
      groups,
      execution_phase_marker,
      context_sequence_count_carrier,
      tree_parent_ids=None,
      tree_depths=None,
      spec_verify_layout=None):
    intermediate_conv_state = torch.empty(hidden_states.shape[0],
                                          conv_state.shape[1],
                                          conv_state.shape[2],
                                          dtype=conv_state.dtype,
                                          device=conv_state.device)
    return torch.empty_like(
        hidden_states), conv_state.clone(), intermediate_conv_state


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::update_ssm_state
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::update_ssm_state", mutates_args=())
def update_ssm_state(
    hidden_states: torch.Tensor,  # [T_exec, num_heads, head_dim]
    ssm_a: torch.Tensor,  # [num_heads] float32
    ssm_b: torch.Tensor,  # [T_exec, n_groups, ssm_state_size]
    ssm_c: torch.Tensor,  # [T_exec, n_groups, ssm_state_size]
    ssm_d: torch.Tensor,  # [num_heads] float16
    dt: torch.Tensor,  # [T_exec, num_heads]
    dt_bias: torch.Tensor,  # [num_heads] float16
    state: torch.
    Tensor,  # [resident_rows, num_heads, head_dim, ssm_state_size]
    query_lengths: torch.Tensor,  # [N] int32
    query_start_offsets: torch.Tensor,  # [N + 1] int32
    state_indices: torch.Tensor,  # [N] int32
    execution_phase_marker: torch.Tensor,  # shape carrier with extent 1..8
    context_sequence_count_carrier: torch.Tensor,
    dt_softplus: int,
    ngroups: int,
    chunk_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stub: Mamba SSM state update. Returns zeros for hidden_states and cloned state."""
    return torch.zeros_like(hidden_states), state.clone()


@update_ssm_state.register_fake
def _(hidden_states,
      ssm_a,
      ssm_b,
      ssm_c,
      ssm_d,
      dt,
      dt_bias,
      state,
      query_lengths,
      query_start_offsets,
      state_indices,
      execution_phase_marker,
      context_sequence_count_carrier,
      dt_softplus,
      ngroups,
      chunk_size=0):
    return torch.empty_like(hidden_states), state.clone()


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::update_ssm_state_with_intermediate
#   Mamba2 SSM update that also emits the per-token replay stash for MTP
#   spec-verify. During verify the committed state is left read-only
#   and the recurrent state is reconstructed from the replay stash after accept.
#   Emits four FP32 replay outputs (dA / x / B / dt) instead of a full-state snapshot.
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::update_ssm_state_with_intermediate",
                         mutates_args=())
def update_ssm_state_with_intermediate(
    hidden_states: torch.Tensor,  # [T_exec, num_heads, head_dim]
    ssm_a: torch.Tensor,  # [num_heads] float32
    ssm_b: torch.Tensor,  # [T_exec, n_groups, ssm_state_size]
    ssm_c: torch.Tensor,  # [T_exec, n_groups, ssm_state_size]
    ssm_d: torch.Tensor,  # [num_heads] float16
    dt: torch.Tensor,  # [T_exec, num_heads]
    dt_bias: torch.Tensor,  # [num_heads] float16
    state: torch.
    Tensor,  # [resident_rows, num_heads, head_dim, ssm_state_size]
    query_lengths: torch.Tensor,  # [N] int32
    query_start_offsets: torch.Tensor,  # [N + 1] int32
    state_indices: torch.Tensor,  # [N] int32
    execution_phase_marker: torch.Tensor,  # shape carrier with extent 1..8
    context_sequence_count_carrier: torch.Tensor,
    dt_softplus: int,
    ngroups: int,
    chunk_size: int = 0,
    tree_parent_ids: Optional[torch.Tensor] = None,  # [T_exec] int32
    tree_depths: Optional[torch.Tensor] = None,  # [T_exec] int32
    spec_verify_layout: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Stub: SSM update with the per-token replay stash (dA / x / B / dt, FP32)."""
    t, nh, hd = hidden_states.shape
    ds = state.shape[3]
    replay_da = torch.zeros(t, nh, dtype=torch.float32, device=state.device)
    replay_u = torch.zeros(t, nh, hd, dtype=torch.float32, device=state.device)
    replay_b = torch.zeros(t,
                           ngroups,
                           ds,
                           dtype=torch.float32,
                           device=state.device)
    replay_dt = torch.zeros(t, nh, dtype=torch.float32, device=state.device)
    return (torch.zeros_like(hidden_states), state.clone(), replay_da,
            replay_u, replay_b, replay_dt)


@update_ssm_state_with_intermediate.register_fake
def _(hidden_states,
      ssm_a,
      ssm_b,
      ssm_c,
      ssm_d,
      dt,
      dt_bias,
      state,
      query_lengths,
      query_start_offsets,
      state_indices,
      execution_phase_marker,
      context_sequence_count_carrier,
      dt_softplus,
      ngroups,
      chunk_size=0,
      tree_parent_ids=None,
      tree_depths=None,
      spec_verify_layout=None):
    t, nh, hd = hidden_states.shape
    ds = state.shape[3]
    replay_da = torch.empty(t, nh, dtype=torch.float32, device=state.device)
    replay_u = torch.empty(t, nh, hd, dtype=torch.float32, device=state.device)
    replay_b = torch.empty(t,
                           ngroups,
                           ds,
                           dtype=torch.float32,
                           device=state.device)
    replay_dt = torch.empty(t, nh, dtype=torch.float32, device=state.device)
    return (torch.empty_like(hidden_states), state.clone(), replay_da,
            replay_u, replay_b, replay_dt)


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::int4_moe_plugin  (sparse MoE with INT4 expert GEMMs)
# ---------------------------------------------------------------------------


def _validate_moe_token_layout(router_logits, hidden_states):
    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape [T, E]")
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [T, H]")


@torch.library.custom_op("trt_edgellm::int4_moe_plugin", mutates_args=())
def int4_moe_plugin(
    router_logits: torch.Tensor,  # [T, E] float32 — gate output before softmax
    hidden_states: torch.Tensor,  # [T, H] float16
    fc_gate_up_qweights: torch.Tensor,  # [E, K//16, 2*I] Marlin int8
    fc_gate_up_scales: torch.Tensor,  # [E, num_groups, I] float16
    fc_down_qweights: torch.Tensor,  # [E, K//16, 2*D] Marlin int8
    fc_down_scales: torch.Tensor,  # [E, num_groups, D] float16
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    quantization_group_size: int,
) -> torch.Tensor:
    """Stub: fused sparse MoE (softmax + topk + expert INT4 grouped GEMMs).

    The gate GEMM (Linear) is traced separately as a standard MatMul.
    This op receives the router logits and performs softmax + topk routing,
    then dispatches tokens to experts for gate_up + SiLU + down projections
    using Marlin-packed INT4 weights.

    Mirrors ``trt_edgellm::Int4MoePlugin`` in tensorrt_edgellm.
    """
    _validate_moe_token_layout(router_logits, hidden_states)
    return torch.zeros_like(hidden_states)


@int4_moe_plugin.register_fake
def _(router_logits, hidden_states, fc_gate_up_qweights, fc_gate_up_scales,
      fc_down_qweights, fc_down_scales, num_experts, top_k, hidden_size,
      moe_inter_size, activation_type, quantization_group_size):
    _validate_moe_token_layout(router_logits, hidden_states)
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Custom op: trt::gather_nd  (token selection: GatherND with batch_dims=1)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::gather_nd", mutates_args=())
def gather_nd(
        value: torch.Tensor,  # [batch, seq_len, hidden_size] float16
        indices: torch.Tensor,  # [batch, num_tokens] int64
) -> torch.Tensor:
    """Stub: gather tokens from seq_len dim. Exports as GatherND(batch_dims=1).

    Equivalent to ``value[b, indices[b, t], :]`` for each batch b and
    token position t.  Exports as GatherND(batch_dims=1).
    """
    batch_size, num_tokens = indices.shape
    return torch.zeros(batch_size,
                       num_tokens,
                       value.shape[-1],
                       dtype=value.dtype,
                       device=value.device)


@gather_nd.register_fake
def _(value, indices):
    batch_size, num_tokens = indices.shape
    hidden_size = value.shape[-1]
    return torch.empty(batch_size,
                       num_tokens,
                       hidden_size,
                       dtype=value.dtype,
                       device=value.device)


# ---------------------------------------------------------------------------
# Custom op: trt::rope_onnx  (TRT native RotaryEmbedding)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::rope_onnx", mutates_args=())
def rope_onnx(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Stub for TRT native RotaryEmbedding — returns tensor with same shape as input."""
    return x.clone()


@rope_onnx.register_fake
def _(x, cos, sin, position_ids):
    return torch.empty_like(x)


# Custom op: trt::kv_cache_update_onnx  (TRT native KVCacheUpdate)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::kv_cache_update_onnx", mutates_args=())
def kv_cache_update_onnx(
    cache: torch.Tensor,
    new_kv: torch.Tensor,
    cache_indices: torch.Tensor,
) -> torch.Tensor:
    """Stub for TRT native KVCacheUpdate — returns cache with same shape."""
    return cache.clone()


@kv_cache_update_onnx.register_fake
def _(cache, new_kv, cache_indices):
    return torch.empty_like(cache)


# ---------------------------------------------------------------------------
# Custom op: trt::attention_onnx  (TRT native Attention)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::attention_onnx", mutates_args=())
def attention_onnx(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    scale: float,
) -> torch.Tensor:
    """Stub for TRT native Attention — returns tensor with same shape as query."""
    return query.clone()


@attention_onnx.register_fake
def _(query, key, value, attn_mask, is_causal, scale):
    return torch.empty_like(query)


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::gated_delta_net  (Qwen3.5 GDN linear attention)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::gated_delta_net", mutates_args=())
def gated_delta_net(
    q: torch.Tensor,  # [T_exec, num_k_heads, k_dim]
    k: torch.Tensor,  # [T_exec, num_k_heads, k_dim]
    v: torch.Tensor,  # [T_exec, num_v_heads, v_dim]
    a: torch.Tensor,  # [T_exec, num_v_heads]
    b: torch.Tensor,  # [T_exec, num_v_heads]
    A_log: torch.Tensor,  # [num_v_heads] float32
    dt_bias: torch.Tensor,  # [num_v_heads] float16
    h0_source: torch.
    Tensor,  # [resident_rows, num_v_heads, k_dim, v_dim] float32
    query_lengths: torch.Tensor,  # [N] int32
    k_dim: int,
    v_dim: int,
    query_start_offsets: torch.Tensor,
    state_indices: torch.Tensor,
    execution_phase_marker: torch.Tensor,
    context_sequence_count_carrier: torch.Tensor,
    use_diffusion_state: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stub: GatedDeltaNet."""
    return torch.zeros_like(v), h0_source.clone(), h0_source.clone()


@gated_delta_net.register_fake
def _(q,
      k,
      v,
      a,
      b,
      A_log,
      dt_bias,
      h0_source,
      query_lengths,
      k_dim,
      v_dim,
      query_start_offsets,
      state_indices,
      execution_phase_marker,
      context_sequence_count_carrier,
      use_diffusion_state=False):
    return torch.empty_like(v), h0_source.clone(), torch.empty_like(h0_source)


@torch.library.custom_op("trt_edgellm::gated_delta_net_with_intermediate",
                         mutates_args=())
def gated_delta_net_with_intermediate(
        q: torch.Tensor,  # [T_exec, num_k_heads, k_dim]
        k: torch.Tensor,  # [T_exec, num_k_heads, k_dim]
        v: torch.Tensor,  # [T_exec, num_v_heads, v_dim]
        a: torch.Tensor,  # [T_exec, num_v_heads]
        b: torch.Tensor,  # [T_exec, num_v_heads]
        A_log: torch.Tensor,  # [num_v_heads] float32
        dt_bias: torch.Tensor,  # [num_v_heads] float16
        h0_source: torch.Tensor,  # FP32 resident state [rows, heads, K, V]
        query_lengths: torch.Tensor,  # [N] int32
        query_start_offsets: torch.Tensor,
        state_indices: torch.Tensor,
        k_dim: int,
        v_dim: int,
        execution_phase_marker: torch.Tensor,
        context_sequence_count_carrier: torch.Tensor,
        tree_parent_ids: Optional[torch.Tensor] = None,  # [T_exec] int32
        tree_depths: Optional[torch.Tensor] = None,  # [T_exec] int32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stub: GatedDeltaNet with per-token recurrent state output."""
    num_v_heads = v.shape[1]
    intermediate_recurrent_state = torch.zeros(v.shape[0],
                                               num_v_heads,
                                               k_dim,
                                               v_dim,
                                               dtype=h0_source.dtype,
                                               device=h0_source.device)
    return torch.zeros_like(v), h0_source.clone(), intermediate_recurrent_state


@gated_delta_net_with_intermediate.register_fake
def _(q,
      k,
      v,
      a,
      b,
      A_log,
      dt_bias,
      h0_source,
      query_lengths,
      query_start_offsets,
      state_indices,
      k_dim,
      v_dim,
      execution_phase_marker,
      context_sequence_count_carrier,
      tree_parent_ids=None,
      tree_depths=None):
    num_v_heads = v.shape[1]
    intermediate_recurrent_state = torch.empty(v.shape[0],
                                               num_v_heads,
                                               k_dim,
                                               v_dim,
                                               dtype=h0_source.dtype,
                                               device=h0_source.device)
    return torch.empty_like(v), h0_source.clone(), intermediate_recurrent_state


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::Nvfp4MoePlugin
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::Nvfp4MoePlugin", mutates_args=())
def nvfp4_moe_plugin(
    router_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_qweights: torch.Tensor,
    fc1_blocks_scale: torch.Tensor,
    fc1_alpha: torch.Tensor,
    fc2_qweights: torch.Tensor,
    fc2_blocks_scale: torch.Tensor,
    fc2_alpha: torch.Tensor,
    input_global_scale: torch.Tensor,
    down_input_scale: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    n_group: int,
    topk_group: int,
    norm_topk_prob: int,
    routed_scaling_factor: float,
    routing_mode: int,
    backend: int,
    io_dtype: int,
    max_routed_rows: int,
) -> torch.Tensor:
    _validate_moe_token_layout(router_logits, hidden_states)
    return torch.zeros_like(hidden_states)


@nvfp4_moe_plugin.register_fake
def _(router_logits, hidden_states, fc1_qweights, fc1_blocks_scale, fc1_alpha,
      fc2_qweights, fc2_blocks_scale, fc2_alpha, input_global_scale,
      down_input_scale, e_score_correction_bias, num_experts, top_k,
      hidden_size, moe_inter_size, activation_type, n_group, topk_group,
      norm_topk_prob, routed_scaling_factor, routing_mode, backend, io_dtype,
      max_routed_rows):
    _validate_moe_token_layout(router_logits, hidden_states)
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::Nvfp4A16MoePlugin
#   Marlin FP16-A / NVFP4-W4 MoE (weight-only).
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::Nvfp4A16MoePlugin", mutates_args=())
def nvfp4_a16_moe_plugin(
    router_logits: torch.Tensor,  # [T, num_experts] float32
    hidden_states: torch.Tensor,  # [T, hidden_size] float16
    fc1_qweights: torch.Tensor,  # [E, hidden/16, 8*fc1_out] int8
    fc1_block_scales: torch.Tensor,  # [E, hidden/16, fc1_out] int8
    fc1_global_scales: torch.Tensor,  # [E] or [2, E] float16
    fc2_qweights: torch.Tensor,  # [E, moe_inter/16, 8*hidden] int8
    fc2_block_scales: torch.Tensor,  # [E, moe_inter/16, hidden] int8
    fc2_global_scales: torch.Tensor,  # [E] float16
    e_score_correction_bias: torch.Tensor,  # [E] float32
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    n_group: int,
    topk_group: int,
    norm_topk_prob: int,
    routed_scaling_factor: float,
    routing_mode: int,
    max_routed_rows: int,
) -> torch.Tensor:
    _validate_moe_token_layout(router_logits, hidden_states)
    return torch.zeros_like(hidden_states)


@nvfp4_a16_moe_plugin.register_fake
def _(router_logits, hidden_states, fc1_qweights, fc1_block_scales,
      fc1_global_scales, fc2_qweights, fc2_block_scales, fc2_global_scales,
      e_score_correction_bias, num_experts, top_k, hidden_size, moe_inter_size,
      activation_type, n_group, topk_group, norm_topk_prob,
      routed_scaling_factor, routing_mode, max_routed_rows):
    _validate_moe_token_layout(router_logits, hidden_states)
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::Nvfp4A16BlackwellMoePlugin
#   SM110-only FP16-A / NVFP4-W4 routed MoE (weight-only) over the
#   BLACKWELL_MOE_N128_K64_V1 layout produced by
#   ``repacking.repack_nvfp4_a16_blackwell_moe_experts``. Unlike the Marlin
#   op, ``moe_inter_size`` is the logical intermediate size (FC1 N padding
#   lives inside the layout) and the per-expert global scales stay FP32.
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::Nvfp4A16BlackwellMoePlugin",
                         mutates_args=())
def nvfp4_a16_blackwell_moe_plugin(
    router_logits: torch.Tensor,  # [T, num_experts] float32
    hidden_states: torch.Tensor,  # [T, hidden_size] float16
    fc1_qweights: torch.Tensor,  # [E, I_pad/128, H/64, 128, 32] int8
    fc1_block_scales: torch.Tensor,  # [E, I_pad/128, H/64, 128, 4] int8
    fc1_global_scales: torch.Tensor,  # [E] float32
    fc2_qweights: torch.Tensor,  # [E, H/128, I/64, 128, 32] int8
    fc2_block_scales: torch.Tensor,  # [E, H/128, I/64, 128, 4] int8
    fc2_global_scales: torch.Tensor,  # [E] float32
    e_score_correction_bias: torch.Tensor,  # [E] float32
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,  # logical I (e.g. 1856), not the 128-padded value
    activation_type: int,
    n_group: int,
    topk_group: int,
    norm_topk_prob: int,
    routed_scaling_factor: float,
    routing_mode: int,
    max_routed_rows: int,
    backend: int,
) -> torch.Tensor:
    _validate_moe_token_layout(router_logits, hidden_states)
    return torch.zeros_like(hidden_states)


@nvfp4_a16_blackwell_moe_plugin.register_fake
def _(router_logits, hidden_states, fc1_qweights, fc1_block_scales,
      fc1_global_scales, fc2_qweights, fc2_block_scales, fc2_global_scales,
      e_score_correction_bias, num_experts, top_k, hidden_size, moe_inter_size,
      activation_type, n_group, topk_group, norm_topk_prob,
      routed_scaling_factor, routing_mode, max_routed_rows, backend):
    _validate_moe_token_layout(router_logits, hidden_states)
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::NvFP4MoEPluginGeforce
#   SM12x (consumer Blackwell) fused NVFP4 MoE. Same signature as
#   ``nvfp4_moe_plugin``; FC1 weights must be in the plain ``[up, gate]``
#   concat layout (not the 64-row up/gate interleave) for SwiGLU activations.
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::NvFP4MoEPluginGeforce", mutates_args=())
def nvfp4_moe_plugin_geforce(
    router_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_qweights: torch.Tensor,
    fc1_blocks_scale: torch.Tensor,
    fc1_alpha: torch.Tensor,
    fc2_qweights: torch.Tensor,
    fc2_blocks_scale: torch.Tensor,
    fc2_alpha: torch.Tensor,
    input_global_scale: torch.Tensor,
    down_input_scale: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    n_group: int,
    topk_group: int,
    norm_topk_prob: int,
    routed_scaling_factor: float,
    routing_mode: int,
    backend: int,
    io_dtype: int,
    max_routed_rows: int,
) -> torch.Tensor:
    _validate_moe_token_layout(router_logits, hidden_states)
    return torch.zeros_like(hidden_states)


@nvfp4_moe_plugin_geforce.register_fake
def _(router_logits, hidden_states, fc1_qweights, fc1_blocks_scale, fc1_alpha,
      fc2_qweights, fc2_blocks_scale, fc2_alpha, input_global_scale,
      down_input_scale, e_score_correction_bias, num_experts, top_k,
      hidden_size, moe_inter_size, activation_type, n_group, topk_group,
      norm_topk_prob, routed_scaling_factor, routing_mode, backend, io_dtype,
      max_routed_rows):
    _validate_moe_token_layout(router_logits, hidden_states)
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::Fp16MoePlugin
#   Unquantized FP16 MoE experts via the CuTeDSL FP16 grouped-GEMM
#   plugin. Same softmax+topk routing and 64-row up/gate FC1 interleave as
#   Nvfp4MoePlugin, but plain FP16 weights: no scales or alpha tensors.
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::Fp16MoePlugin", mutates_args=())
def fp16_moe_plugin(
    router_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_weights: torch.Tensor,
    fc2_weights: torch.Tensor,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    norm_topk_prob: int,
    max_routed_rows: int,
) -> torch.Tensor:
    _validate_moe_token_layout(router_logits, hidden_states)
    return torch.zeros_like(hidden_states)


@fp16_moe_plugin.register_fake
def _(router_logits, hidden_states, fc1_weights, fc2_weights, num_experts,
      top_k, hidden_size, moe_inter_size, activation_type, norm_topk_prob,
      max_routed_rows):
    _validate_moe_token_layout(router_logits, hidden_states)
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# TRT-NATIVE (OOTB) MoE: trt::trt_moe_fp16 / trt::trt_moe_w4a8 -> ONNX ``TRT_MoE`` (domain "trt")
#   Parsed by TensorRT's BUILTIN importer (not a plugin) into an IMoELayer, which
#   lowers to Myelin's moe_op. This is the OOTB arm the MoE plugins are compared
#   against, on the same arm-agnostic few-layer golden. Opt-in, default off: see
#   ``set_trt_native_moe`` / ``is_trt_native_moe_enabled`` below.
#
#   Contract (NvInfer.h IMoELayer, parsers/onnx/docs/TRT_custom_ops.md "TRT_MoE"):
#     hidden_states     [B, S, H]  FP16
#     selected_experts  [B, S, K]  INT32
#     scores            [B, S, K]  FP32   <- see below
#     fc_gate_weights   [E, H, I]  FP16   (plain HF weight transposed; NOT the
#     fc_up_weights     [E, H, I]  FP16    fused/interleaved/padded plugin layout)
#     fc_down_weights   [E, I, H]  FP16
#   Biases are all-or-none; Qwen3 has none, so the FP16 arm uses the 6-input form.
#   NON-GATED experts (Nemotron-H: fcDown(relu(fcUp(x))^2)) leave input 3
#   (fc_gate_weights) EMPTY; that selects IMoELayer::setWeights(). Biases are
#   then not supported, and activation_type 2 (kRELU2) is only defined for this
#   form. See trt::trt_moe_fp16_nongated / trt::trt_moe_w4a8_nongated.
#   W4A4 (quantization_mode 2, --trt-native-moe-w4a4): trt::trt_moe_w4a4 /
#   trt::trt_moe_w4a4_nongated take the hidden states through an NVFP4
#   dynamic Q/DQ (trt::nvfp4_act_qdq_moe) and the FP4 weights with separate
#   block and global scales (trt::nvfp4_dequantize_moe_with_global_scale). See
#   set_trt_native_moe.
#   IMoELayer requires RANK 3. The token-major export contract hands the MoE
#   block ``[T, H]`` (see ``_validate_moe_token_layout``), so the model wraps it
#   as one sequence ``[1, T, H]`` / ``[1, T, K]`` and reshapes the output back.
#
#   ``scores`` MUST be FP32 even when hidden_states is FP16. Undocumented; violating
#   it fails deep inside the compiler rather than at the API
#   ("Invalid binary pointwise types: 2 <- 2, 1  op type : mul").
#
#   There is deliberately NO ``swiglu_limit`` parameter/attribute. TRT's TRT_MoE
#   importer treats a PRESENT swiglu_limit as a real clamp and only defaults it
#   to +inf (no clamping) when the attribute is ABSENT. An explicit 0.0 makes
#   Myelin >= 2.21.18 (B200, TRT 11.3.0.89) clamp the up-projection to [-0, 0],
#   so the whole MoE layer returns exact zeros; Myelin 2.21.5 (Thor, TRT
#   11.3.0.7) ignored the limit, which hid the bad attribute there. Qwen3 uses
#   plain SiLU SwiGLU with no clamp, so alpha=1.0 / beta=0.0 are the only knobs.
# ---------------------------------------------------------------------------

_TRT_NATIVE_MOE_ENABLED = False
_TRT_NATIVE_MOE_W4A4 = False
_TRT_NATIVE_MOE_LOGGED = False


def set_trt_native_moe(enabled: bool, w4a4: bool = False) -> None:
    """Select the TRT-native ``TRT_MoE`` arm for MoE experts (export-time switch).

    Driven by ``--trt-native-moe``; off by default (plugin path). FP16 experts
    export the 6-input ``TRT_MoE``, NVFP4 experts the W4A8 form or, with
    ``w4a4`` (``--trt-native-moe-w4a4``), the W4A4 form (quantization_mode 2).
    """
    global _TRT_NATIVE_MOE_ENABLED, _TRT_NATIVE_MOE_W4A4
    global _TRT_NATIVE_MOE_LOGGED
    if not isinstance(enabled, bool) or not isinstance(w4a4, bool):
        raise ValueError("enabled and w4a4 must be bools")
    _TRT_NATIVE_MOE_ENABLED = enabled
    _TRT_NATIVE_MOE_W4A4 = enabled and w4a4
    _TRT_NATIVE_MOE_LOGGED = False


def is_trt_native_moe_enabled() -> bool:
    """True when MoE experts export through TRT_MoE (IMoELayer) instead of a plugin."""
    global _TRT_NATIVE_MOE_LOGGED
    if _TRT_NATIVE_MOE_ENABLED and not _TRT_NATIVE_MOE_LOGGED:
        logger.info("Using TRT-native MoE (TRT_MoE -> IMoELayer)")
        _TRT_NATIVE_MOE_LOGGED = True
    return _TRT_NATIVE_MOE_ENABLED


def is_trt_native_moe_w4a4_enabled() -> bool:
    """True when NVFP4 experts on the native arm export TRT_MoE mode 2 (W4A4)."""
    return _TRT_NATIVE_MOE_W4A4


def _validate_trt_moe_layout(hidden_states: torch.Tensor,
                             selected_experts: torch.Tensor,
                             scores: torch.Tensor) -> None:
    """IMoELayer takes rank-3 tensors: hidden [B, S, H], routing [B, S, K]."""
    if hidden_states.ndim != 3:
        raise ValueError("TRT_MoE hidden_states must have shape [B, S, H]")
    if selected_experts.ndim != 3 or scores.ndim != 3:
        raise ValueError(
            "TRT_MoE selected_experts/scores must have shape [B, S, K]")
    if scores.dtype != torch.float32:
        raise ValueError(
            "TRT_MoE scores must be float32 (Myelin codegen rejects fp16 scores)"
        )


@torch.library.custom_op("trt::trt_moe_fp16", mutates_args=())
def trt_moe_fp16(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    scores: torch.Tensor,
    fc_gate_weights: torch.Tensor,
    fc_up_weights: torch.Tensor,
    fc_down_weights: torch.Tensor,
    activation_type: int,
    quantization_mode: int,
    swiglu_alpha: float,
    swiglu_beta: float,
) -> torch.Tensor:
    _validate_trt_moe_layout(hidden_states, selected_experts, scores)
    return torch.zeros_like(hidden_states)


@trt_moe_fp16.register_fake
def _(hidden_states, selected_experts, scores, fc_gate_weights, fc_up_weights,
      fc_down_weights, activation_type, quantization_mode, swiglu_alpha,
      swiglu_beta):
    _validate_trt_moe_layout(hidden_states, selected_experts, scores)
    return torch.empty_like(hidden_states)


@torch.library.custom_op("trt::nvfp4_dequantize_moe", mutates_args=())
def nvfp4_dequantize_moe(
    weight: torch.Tensor,  # int8 [E, out, in//2] packed NVFP4
    block_scale: torch.Tensor,  # float16 [E, out, in//group_size]
    group_size: int,
) -> torch.Tensor:
    """Block-dequantize stacked NVFP4 MoE expert weights to FP16.

    One ``TRT_BlockDequantize`` with an **fp16** block scale, rather than the
    two-level (fp8 block scale + fp32 global) form the dense linears use: the
    global ``weight_scale_2`` is *per-expert* for MoE, which the per-tensor DQ1
    of :func:`nvfp4_dequantize` cannot express, so it is folded into the block
    scale on the host. fp16 is the scale type the compiler's block dequantize
    accepts for FP4 weights; an fp32 scale fails type inference at build time.

    Trade-off: fp16 block scales are 2x the bytes of fp8 ones. Fine for the
    opt-in arm; a production path should give DQ1 an ``axis=0`` instead.

    Myelin's ``dequantize_moe`` fuses this DQ into the MoE op, which is why the
    block shape must be explicit (``TRT_BlockDequantize``, see the translation).
    """
    return _nvfp4_dequantize_moe_eager(weight, block_scale, group_size)


def _nvfp4_dequantize_moe_eager(weight, block_scale, group_size):
    lo = (weight & 0x0F).to(torch.uint8)
    hi = ((weight >> 4) & 0x0F).to(torch.uint8)
    n = torch.stack([lo, hi], dim=-1).reshape(*weight.shape[:-1],
                                              weight.shape[-1] * 2)
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.],
                       dtype=torch.float32,
                       device=weight.device)
    mag = lut[(n & 0x7).long()]
    val = torch.where((n & 0x8) != 0, -mag, mag)
    sc = block_scale.repeat_interleave(group_size, dim=-1).to(torch.float32)
    return (val * sc).to(torch.float16)


@nvfp4_dequantize_moe.register_fake
def _(weight, block_scale, group_size):
    *lead, packed_in = weight.shape
    return torch.empty(*lead,
                       packed_in * 2,
                       dtype=torch.float16,
                       device=weight.device)


@torch.library.custom_op("trt::nvfp4_dequantize_moe_with_global_scale",
                         mutates_args=())
def nvfp4_dequantize_moe_with_global_scale(
    weight: torch.Tensor,  # int8 [E, out, in//2] packed NVFP4
    block_scale: torch.Tensor,  # float8_e4m3fn [E, out, in//group_size]
    global_scale: torch.Tensor,  # float16 [E] per-expert weight_scale_2
    group_size: int,
) -> torch.Tensor:
    """Block-dequantize stacked NVFP4 MoE expert weights to FP16 (W4A4 form).

    The W4A4 (``TRT_MoE`` quantization_mode 2) counterpart of
    :func:`nvfp4_dequantize_moe`: the checkpoint's E4M3 block scales stay
    verbatim and the per-expert ``weight_scale_2`` stays a separate fp16
    global, the form the mode-2 lowering was validated with (the W4A8 arm
    folds the two into one fp16 scale on the host). In the ONNX graph::

        DequantizeLinear(block_scale_fp8, global_scale_fp16, axis=0)
            -> scales_fp16 [E, out, in // 16]
        TRT_BlockDequantize(weight_fp4, scales_fp16, block_shape=[1, 1, 16],
                            output_dtype=FLOAT16)
            -> w_dq [E, out, in]

    The global is fp16 because the DQ output type follows its scale type and
    the compiler's FP4 block dequantize only type-infers fp16 output from
    fp16 scales; the stacker
    (:func:`~tensorrt_edgellm.checkpoint.repacking.stack_nvfp4_moe_projection_with_global_scale`)
    keeps it in fp16's normal range. The eager body mirrors the graph (block
    scales times the global in fp16, then the FP4 decode) as a reference; the
    export traces the fake kernel.
    """
    return _nvfp4_dequantize_moe_with_global_scale_eager(
        weight, block_scale, global_scale, group_size)


def _nvfp4_dequantize_moe_with_global_scale_eager(weight, block_scale,
                                                  global_scale, group_size):
    if block_scale.dtype in (torch.int8, torch.uint8):
        block_scale = block_scale.view(torch.float8_e4m3fn)
    # The graph's DequantizeLinear(axis=0): fp16 scales = block_scale * global.
    scales = (block_scale.to(torch.float32) *
              global_scale.to(torch.float32).reshape(-1, 1, 1)).to(
                  torch.float16)
    return _nvfp4_dequantize_moe_eager(weight, scales, group_size)


@nvfp4_dequantize_moe_with_global_scale.register_fake
def _(weight, block_scale, global_scale, group_size):
    *lead, packed_in = weight.shape
    return torch.empty(*lead,
                       packed_in * 2,
                       dtype=torch.float16,
                       device=weight.device)


@torch.library.custom_op("trt::nvfp4_act_qdq_moe", mutates_args=())
def nvfp4_act_qdq_moe(
        hidden_states: torch.Tensor,  # float16 [1, T, H] TRT_MoE input
        dblq_scale: torch.Tensor,  # float32 scalar: amax / (6.0 * 448.0)
) -> torch.Tensor:
    """NVFP4 dynamic Q/DQ of the ``TRT_MoE`` hidden states (W4A4 form).

    Same fake-quant as :func:`nvfp4_act_qdq`, but the ONNX chain ends in an
    explicit-block-shape ``TRT_BlockDequantize`` instead of the dense op's
    ``trt::DequantizeLinear(block_size=16)``: Myelin's MoE lowering reads
    the hidden-state block shape off that DQ (as it does for the weights),
    and the standard-style DQ leaves it empty. Emits the reference W4A4
    chain::

        TRT_FP4DynamicQuantize(x, dblq_f32, axis=-1, block_size=16,
                               scale_type=17) -> (x_f4, sx_f8)
        Cast(dblq_f32, FLOAT16) -> dblq_f16
        DequantizeLinear(sx_f8, dblq_f16) -> scale_f16     [B, S, H // 16]
        TRT_BlockDequantize(x_f4, scale_f16, block_shape=[1, 1, 16],
                            output_dtype=FLOAT16) -> x_dq   [B, S, H]

    ``dblq_scale`` is the second-level ("double quant") scale of the block
    scales: ModelOpt's ``input_scale = amax / (6 * 448)`` verbatim, the max
    over the experts because the FC1 input is shared. The host rounds it to
    an fp16-representable value so the fp32 copy the DynQ quantizes with and
    the fp16 copy the DQ dequantizes with are the same number. The eager
    body is :func:`_nvfp4_act_qdq_eager` (fp32 DynQ / fp16 DQ).
    """
    return _nvfp4_act_qdq_eager(hidden_states, dblq_scale)


@nvfp4_act_qdq_moe.register_fake
def _(hidden_states, dblq_scale):
    return torch.empty_like(hidden_states)


@torch.library.custom_op("trt::trt_moe_w4a8", mutates_args=())
def trt_moe_w4a8(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    scores: torch.Tensor,
    fc_gate_weights: torch.Tensor,
    fc_up_weights: torch.Tensor,
    fc_down_weights: torch.Tensor,
    fc_gate_biases: torch.Tensor,
    fc_up_biases: torch.Tensor,
    fc_down_biases: torch.Tensor,
    fc_down_activation_scale: torch.Tensor,
    activation_type: int,
    quantization_mode: int,
    quantization_dtype: int,
    swiglu_alpha: float,
    swiglu_beta: float,
) -> torch.Tensor:
    """W4A8 TRT-native MoE: NVFP4 weights + FP8 down-projection activation.

    Same builtin ``TRT_MoE`` node as :func:`trt_moe_fp16`, with the quantization
    inputs the importer expects:

    * ``fc_down_activation_scale`` must land at input **index 9** -- the importer
      reads it unconditionally there. Indices 6/7/8 are the biases, which are
      all-or-none, so explicit **zero** biases are passed (a numeric no-op).
    * ``quantization_dtype`` is a raw TensorRT ``DataType`` enum (``kFP8`` = 6),
      NOT an ONNX ``TensorProto`` dtype -- the importer ``static_cast``s it.
    * the weights arriving here are the *dequantized* outputs of
      :func:`nvfp4_dequantize_moe`, so the FP4 initializers + block-DQ live in
      the graph and Myelin's ``dequantize_moe`` fuses them back into the MoE op.
    """
    _validate_trt_moe_layout(hidden_states, selected_experts, scores)
    return torch.zeros_like(hidden_states)


@trt_moe_w4a8.register_fake
def _(hidden_states, selected_experts, scores, fc_gate_weights, fc_up_weights,
      fc_down_weights, fc_gate_biases, fc_up_biases, fc_down_biases,
      fc_down_activation_scale, activation_type, quantization_mode,
      quantization_dtype, swiglu_alpha, swiglu_beta):
    _validate_trt_moe_layout(hidden_states, selected_experts, scores)
    return torch.empty_like(hidden_states)


@torch.library.custom_op("trt::trt_moe_w4a4", mutates_args=())
def trt_moe_w4a4(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    scores: torch.Tensor,
    fc_gate_weights: torch.Tensor,
    fc_up_weights: torch.Tensor,
    fc_down_weights: torch.Tensor,
    fc_gate_biases: torch.Tensor,
    fc_up_biases: torch.Tensor,
    fc_down_biases: torch.Tensor,
    fc_down_activation_scale: torch.Tensor,
    activation_type: int,
    quantization_mode: int,
    quantization_dtype: int,
    dyn_q_output_scale_dtype: int,
    swiglu_alpha: float,
    swiglu_beta: float,
) -> torch.Tensor:
    """W4A4 TRT-native MoE: NVFP4 weights + dynamically NVFP4-quantized activations.

    :func:`trt_moe_w4a8` with ``TRT_MoE`` quantization_mode 2 instead of 1.
    Same 10-input layout (explicit zero biases at 6-8 so the scale sits at
    index 9, where the importer reads it for both modes), but the pieces
    around it change:

    * input 0 is the output of :func:`nvfp4_act_qdq_moe` (dynamic NVFP4
      Q/DQ of the hidden states), not a static FP8 Q/DQ pair;
    * the weights are the outputs of :func:`nvfp4_dequantize_moe_with_global_scale`
      (FP8 block scales and per-expert fp16 global kept apart);
    * ``fc_down_activation_scale`` (input 9) is the per-expert fp32
      second-level scale of the FC2 input's dynamically computed FP8 block
      scales: ModelOpt's ``down_proj.input_scale`` verbatim. No ``* 6``;
      that factor turned it into the FP8 per-tensor scale of the W4A8 form;
    * ``quantization_dtype`` is ``kFP4`` (10) and the translation adds
      ``quantization_block_shape=[1, 1, 1, 16]`` (blocks of 16 along
      moe_intermediate_size) and ``dyn_q_output_scale_dtype`` (``kFP8`` =
      6, the type of those block scales). Both dtypes are raw TensorRT
      ``DataType`` values.

    Needs TensorRT 11.6 or newer (``IMoELayer::setQuantizationDynamicDblQ``).
    """
    _validate_trt_moe_layout(hidden_states, selected_experts, scores)
    return torch.zeros_like(hidden_states)


@trt_moe_w4a4.register_fake
def _(hidden_states, selected_experts, scores, fc_gate_weights, fc_up_weights,
      fc_down_weights, fc_gate_biases, fc_up_biases, fc_down_biases,
      fc_down_activation_scale, activation_type, quantization_mode,
      quantization_dtype, dyn_q_output_scale_dtype, swiglu_alpha, swiglu_beta):
    _validate_trt_moe_layout(hidden_states, selected_experts, scores)
    return torch.empty_like(hidden_states)


@torch.library.custom_op("trt::trt_moe_fp16_nongated", mutates_args=())
def trt_moe_fp16_nongated(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    scores: torch.Tensor,
    fc_up_weights: torch.Tensor,
    fc_down_weights: torch.Tensor,
    activation_type: int,
    quantization_mode: int,
) -> torch.Tensor:
    """FP16 TRT-native MoE with NON-GATED experts: ``fcDown(act(fcUp(x)))``.

    Same builtin ``TRT_MoE`` node as :func:`trt_moe_fp16` with input 3
    (``fc_gate_weights``) left EMPTY, which selects ``IMoELayer::setWeights``
    (no gate projection, no mul). ``fc_up_weights`` is ``[E, H, I]`` and
    ``fc_down_weights`` ``[E, I, H]``, exactly as in the gated form.
    ``activation_type`` 2 (``MoEActType::kRELU2`` = ``max(x, 0)^2``) is only
    defined for this form; kNONE and kSILU work too. Biases are not supported
    for non-gated experts, and the SwiGLU attributes are not emitted (alpha /
    beta have no effect on kRELU2, and a present ``swiglu_limit`` would clamp).
    """
    _validate_trt_moe_layout(hidden_states, selected_experts, scores)
    return torch.zeros_like(hidden_states)


@trt_moe_fp16_nongated.register_fake
def _(hidden_states, selected_experts, scores, fc_up_weights, fc_down_weights,
      activation_type, quantization_mode):
    _validate_trt_moe_layout(hidden_states, selected_experts, scores)
    return torch.empty_like(hidden_states)


@torch.library.custom_op("trt::trt_moe_w4a8_nongated", mutates_args=())
def trt_moe_w4a8_nongated(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    scores: torch.Tensor,
    fc_up_weights: torch.Tensor,
    fc_down_weights: torch.Tensor,
    fc_down_activation_scale: torch.Tensor,
    activation_type: int,
    quantization_mode: int,
    quantization_dtype: int,
) -> torch.Tensor:
    """W4A8 TRT-native MoE with NON-GATED experts.

    :func:`trt_moe_w4a8` without the gate: input 3 is empty, inputs 6-8 (the
    biases) are empty too because TRT_MoE rejects biases on non-gated experts,
    and ``fc_down_activation_scale`` still lands at input index 9, where the
    importer reads it unconditionally. The weights are the dequantized outputs
    of :func:`nvfp4_dequantize_moe` (FP4 initializers + block DQ stay in the
    graph for Myelin's ``dequantize_moe`` fusion); the hidden states carry a
    static FP8 Q/DQ pair whose scale is the FC1 activation scale.
    """
    _validate_trt_moe_layout(hidden_states, selected_experts, scores)
    return torch.zeros_like(hidden_states)


@trt_moe_w4a8_nongated.register_fake
def _(hidden_states, selected_experts, scores, fc_up_weights, fc_down_weights,
      fc_down_activation_scale, activation_type, quantization_mode,
      quantization_dtype):
    _validate_trt_moe_layout(hidden_states, selected_experts, scores)
    return torch.empty_like(hidden_states)


@torch.library.custom_op("trt::trt_moe_w4a4_nongated", mutates_args=())
def trt_moe_w4a4_nongated(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    scores: torch.Tensor,
    fc_up_weights: torch.Tensor,
    fc_down_weights: torch.Tensor,
    fc_down_activation_scale: torch.Tensor,
    activation_type: int,
    quantization_mode: int,
    quantization_dtype: int,
    dyn_q_output_scale_dtype: int,
) -> torch.Tensor:
    """W4A4 TRT-native MoE with NON-GATED experts.

    :func:`trt_moe_w4a4` without the gate, laid out like
    :func:`trt_moe_w4a8_nongated` (inputs 3 and 6-8 empty, the scale at
    index 9). ``fc_down_activation_scale`` is the per-expert fp32
    second-level scale of the dynamically block-quantized ReLU^2 output,
    ``down_proj.input_scale`` verbatim (NOT ``* 6``); the hidden states come
    from :func:`nvfp4_act_qdq_moe` and the weights from
    :func:`nvfp4_dequantize_moe_with_global_scale`.
    """
    _validate_trt_moe_layout(hidden_states, selected_experts, scores)
    return torch.zeros_like(hidden_states)


@trt_moe_w4a4_nongated.register_fake
def _(hidden_states, selected_experts, scores, fc_up_weights, fc_down_weights,
      fc_down_activation_scale, activation_type, quantization_mode,
      quantization_dtype, dyn_q_output_scale_dtype):
    _validate_trt_moe_layout(hidden_states, selected_experts, scores)
    return torch.empty_like(hidden_states)


# TensorRT DataType enum values (NvInferRuntimeBase.h) -- raw TRT enums, not ONNX dtypes.
TRT_DATATYPE_FP8 = 6
TRT_DATATYPE_FP4 = 10

# TRT_MoE quantization_mode: 0 = none; 1 = static (the FC2 input quantized to
# kFP8 with one per-expert scale, W4A8); 2 = dynamic double quantization (the
# FC2 input block-quantized to kFP4 with FP8 block scales and a per-expert
# second-level scale, W4A4; IMoELayer::setQuantizationDynamicDblQ, TensorRT
# 11.6 or newer). The static mode supports kFP8 only.
TRT_MOE_QUANT_NONE = 0
TRT_MOE_QUANT_STATIC = 1
TRT_MOE_QUANT_DYNAMIC_DBLQ = 2

# TRT_MoE activation_type: NvInfer.h MoEActType. kRELU2 (max(x, 0)^2) is only
# defined for non-gated experts (the ``*_nongated`` ops above).
TRT_MOE_ACT_NONE = 0
TRT_MOE_ACT_SILU = 1
TRT_MOE_ACT_RELU2 = 2

# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::Fp16MoePluginSigmoid
#   Same FP16 grouped-GEMM plugin (Fp16MoePlugin) as ``fp16_moe_plugin`` but
#   with DeepSeek/Nemotron-H sigmoid-group-topk routing: adds the
#   ``e_score_correction_bias`` input and the n_group / topk_group /
#   routed_scaling_factor attributes.
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::Fp16MoePluginSigmoid", mutates_args=())
def fp16_moe_plugin_sigmoid(
    router_logits: torch.Tensor,  # [T, num_experts] float32
    hidden_states: torch.Tensor,  # [T, hidden_size] float16
    fc1_weights: torch.
    Tensor,  # [E, moe_inter, hidden] float16 (ReLU2, ungated)
    fc2_weights: torch.Tensor,  # [E, hidden, moe_inter] float16
    e_score_correction_bias: torch.Tensor,  # [E] float32
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    n_group: int,
    topk_group: int,
    norm_topk_prob: int,
    routed_scaling_factor: float,
    max_routed_rows: int,
) -> torch.Tensor:
    _validate_moe_token_layout(router_logits, hidden_states)
    return torch.zeros_like(hidden_states)


@fp16_moe_plugin_sigmoid.register_fake
def _(router_logits, hidden_states, fc1_weights, fc2_weights,
      e_score_correction_bias, num_experts, top_k, hidden_size, moe_inter_size,
      activation_type, n_group, topk_group, norm_topk_prob,
      routed_scaling_factor, max_routed_rows):
    _validate_moe_token_layout(router_logits, hidden_states)
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::all_reduce
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::all_reduce", mutates_args=())
def all_reduce(hidden_states: torch.Tensor, tp_size: int) -> torch.Tensor:
    """Trace-time stub for the portable TensorRT ``AllReducePlugin``."""
    return torch.empty_like(hidden_states)


@all_reduce.register_fake
def _(hidden_states, tp_size):
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::fused_nvfp4_gemm_allreduce
#   NVFP4 row-parallel GEMM fused with AllReduce
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::fused_nvfp4_gemm_allreduce",
                         mutates_args=())
def fused_nvfp4_gemm_allreduce(
    hidden_states: torch.Tensor,  # FP16 activation [..., K_per_rank]
    global_scale: torch.Tensor,  # FP32 scalar: amax / (6.0 * 448.0)
    weight_f4: torch.Tensor,  # int8-packed FP4 [N, K_per_rank // 2]
    weight_f8_scale: torch.Tensor,  # FP8E4M3FN [N, K_per_rank // group_size]
    weight_f32_scale: torch.Tensor,  # FP32 scalar
    tp_size: int,
    raw_activation_scale: bool = False,
) -> torch.Tensor:
    """Stub: NVFP4 row-parallel GEMM fused with AllReduce.

    Single op that emits the entire chain feeding
    ``FusedNvfp4GemmAllReducePlugin``::

        TRT_FP4DynamicQuantize(x, global_scale, axis=-1, block_size=16, scale_type=17)
            -> (x_f4, sx_f8)
        trt::DequantizeLinear(sx_f8, global_scale)            -> combined_scale_fp32
        FusedNvfp4GemmAllReducePlugin(x_f4, activation_scale, weight_f4,
                                 weight_f8_scale, weight_f32_scale,
                                 tp_size) -> y_fp16   (already AllReduced across ranks)

    When ``raw_activation_scale`` is true, the fused path passes the raw FP8
    block scale and folds ``global_scale`` into the GEMM alpha to avoid
    dequantizing and requantizing the scale tensor. Otherwise it preserves the
    existing combined-FP32 scale contract.
    """
    out_features = weight_f4.shape[0]
    out_shape = list(hidden_states.shape[:-1]) + [out_features]
    return torch.zeros(*out_shape,
                       dtype=torch.float16,
                       device=hidden_states.device)


@fused_nvfp4_gemm_allreduce.register_fake
def _(hidden_states,
      global_scale,
      weight_f4,
      weight_f8_scale,
      weight_f32_scale,
      tp_size,
      raw_activation_scale=False):
    out_features = weight_f4.shape[0]
    out_shape = list(hidden_states.shape[:-1]) + [out_features]
    return torch.empty(*out_shape,
                       dtype=torch.float16,
                       device=hidden_states.device)


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::gemma4_audio_attention_plugin
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::gemma4_audio_attention_plugin",
                         mutates_args=())
def gemma4_audio_attention_plugin(
    q_raw: torch.Tensor,  # [B, S, H, D]
    k_raw: torch.Tensor,  # [B, S, H, D]
    v: torch.Tensor,  # [B, S, H, D]
    gamma: torch.Tensor,  # [D] float32
    rel_key: torch.Tensor,  # [P, H, D]
    valid: torch.Tensor,  # [B, S] bool
    seq_len_carrier: torch.Tensor,  # [1] int32 (shape carrier)
    chunk_size: int,
    left_horizon: int,
    context_size: int,
    logit_cap: float,
) -> torch.Tensor:
    """Gemma 4 audio chunked local attention (post-QKV, pre-output-proj).

    In eager mode, implements the full attention body using PyTorch ops.
    During dynamo/ONNX tracing the register_fake shape propagation is used.
    """
    import math

    import torch.nn.functional as F

    B, S, H, D = q_raw.shape
    P = rel_key.shape[0]  # num relative positions
    num_chunks = (S + chunk_size - 1) // chunk_size

    # Q/K scaling (matches kernel and HF reference)
    ln2 = math.log(2.0)
    q_scalar = (D**-0.5) / ln2
    k_scale = math.log1p(math.exp(1.0)) / ln2
    softplus_gamma = F.softplus(gamma)  # [D]

    q = q_raw.float() * q_scalar * softplus_gamma.unsqueeze(0).unsqueeze(0)
    k = k_raw.float() * k_scale
    v_f = v.float()

    # Pad sequence to multiple of chunk_size
    pad_len = num_chunks * chunk_size - S
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        v_f = F.pad(v_f, (0, 0, 0, 0, 0, pad_len))

    # Block Q: [B, num_chunks, chunk_size, H, D]
    q_blocks = q.reshape(B, num_chunks, chunk_size, H, D)

    # Extract context windows for K, V (left-pad by one chunk, concat pairs)
    k_padded = F.pad(k, (0, 0, 0, 0, chunk_size, 0))  # left-pad seq dim
    v_padded = F.pad(v_f, (0, 0, 0, 0, chunk_size, 0))
    k_padded = k_padded.reshape(B, num_chunks + 1, chunk_size, H, D)
    v_padded = v_padded.reshape(B, num_chunks + 1, chunk_size, H, D)
    # Concat adjacent pairs: [B, num_chunks, 2*chunk_size=context_size, H, D]
    k_ctx = torch.cat([k_padded[:, :-1], k_padded[:, 1:]], dim=2)
    v_ctx = torch.cat([v_padded[:, :-1], v_padded[:, 1:]], dim=2)

    # Content scores: Q @ K^T
    # queries: [B, H, nB, chunk, D], keys: [B, H, nB, D, ctx]
    queries = q_blocks.permute(0, 3, 1, 2, 4)
    k_c = k_ctx.permute(0, 3, 1, 4, 2)
    matrix_ac = queries @ k_c  # [B, H, nB, chunk, ctx]

    # Relative position bias via _rel_shift
    # rel_key: [P, H, D] -> [H, D, P]
    rel_k_t = rel_key.float().permute(1, 2, 0)
    # queries_flat: [B, H, nB*chunk, D]
    queries_flat = queries.reshape(B, H, -1, D)
    matrix_bd = queries_flat @ rel_k_t  # [B, H, nB*chunk, P]
    matrix_bd = matrix_bd.reshape(B, H, num_chunks, chunk_size, P)
    # Rel shift: pad right, reshape, slice
    matrix_bd = F.pad(matrix_bd, (0, context_size + 1 - P))
    matrix_bd = matrix_bd.reshape(B, H, num_chunks,
                                  chunk_size * (context_size + 1))
    matrix_bd = matrix_bd[..., :chunk_size * context_size]
    matrix_bd = matrix_bd.reshape(B, H, num_chunks, chunk_size, context_size)

    # Combine scores
    scores = matrix_ac + matrix_bd

    # Softcap: cap * tanh(scores / cap)
    scores = logit_cap * torch.tanh(scores / logit_cap)

    # Build attention mask from valid tensor
    # valid: [B, S] -> pad to num_chunks*chunk_size, then build blocked mask
    if pad_len > 0:
        valid_padded = F.pad(valid.float(), (0, pad_len))
    else:
        valid_padded = valid.float()
    # Context validity: left-pad by chunk_size (matching K/V context extraction)
    valid_ctx_flat = F.pad(valid_padded, (chunk_size, 0))
    valid_ctx = valid_ctx_flat.reshape(B, num_chunks + 1, chunk_size)
    valid_ctx = torch.cat([valid_ctx[:, :-1], valid_ctx[:, 1:]],
                          dim=2)  # [B, nB, ctx]
    # Query validity
    valid_q = valid_padded.reshape(B, num_chunks, chunk_size)  # [B, nB, chunk]
    # Mask: both query and key must be valid
    # [B, 1, nB, chunk, 1] * [B, 1, nB, 1, ctx]
    mask = valid_q.unsqueeze(1).unsqueeze(-1) * valid_ctx.unsqueeze(
        1).unsqueeze(3)

    # Apply mask
    scores = scores.masked_fill(mask == 0, -1e4)

    # Softmax and weighted sum
    attn_weights = torch.softmax(scores, dim=-1)
    v_c = v_ctx.permute(0, 3, 1, 2, 4)  # [B, H, nB, ctx, D]
    out = attn_weights @ v_c  # [B, H, nB, chunk, D]

    # Reshape: [B, H, nB, chunk, D] -> [B, nB*chunk, H, D] -> [B, S, H, D]
    out = out.permute(0, 2, 3, 1, 4).reshape(B, num_chunks * chunk_size, H, D)
    out = out[:, :S, :, :]

    return out.to(q_raw.dtype)


@gemma4_audio_attention_plugin.register_fake
def _(q_raw, k_raw, v, gamma, rel_key, valid, seq_len_carrier, chunk_size,
      left_horizon, context_size, logit_cap):
    return torch.empty_like(q_raw)


# ---------------------------------------------------------------------------
# Custom op: trt::qsa_attention_plugin  (Qwen Sparse Attention, prefill + decode)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::qsa_attention_plugin", mutates_args=())
def qsa_attention_plugin(
    qkv: torch.Tensor,
    index_qk: torch.Tensor,
    past_key_value: torch.Tensor,
    query_lengths: torch.Tensor,
    rope_rotary_cos_sin: torch.Tensor,
    past_lengths: torch.Tensor,
    kv_page_table: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    indexer_n_heads: int,
    indexer_head_dim: int,
    indexer_budget: int,
    indexer_compress_ratio: int,
    attention_scale: float,
    rms_norm_eps: float,
    q_norm_gamma: List[float],
    k_norm_gamma: List[float],
    indexer_q_norm_gamma: List[float],
    indexer_k_norm_gamma: List[float],
    query_start_offsets: torch.Tensor,
    attention_sequence_lengths: torch.Tensor,
    execution_phase_marker: torch.Tensor,
    context_sequence_count_carrier: torch.Tensor,
    # Speculative verify state. Defaults so torch.export strips them from
    # vanilla graphs; set together for the base and draft of an MTP pair.
    use_spec_verify_state: int = 0,
    tree_parent_ids: Optional[torch.Tensor] = None,
    tree_depths: Optional[torch.Tensor] = None,
    spec_verify_layout: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stub for ``QsaAttentionPlugin`` (Qwen Sparse Attention) on the
    token-major ragged ABI.

    A weight-free block-compressed indexer selects the top-``indexer_budget``
    KV blocks (of ``indexer_compress_ratio`` tokens each) per query token, and
    a sparse GQA attention attends only the listed tokens. Causality lives in
    the token list: the kernel applies no causal mask.

    Activations are token-major and entry-padded: ``T_exec = N * S_pad`` with
    ``N`` the ``query_lengths`` extent. The execution phase is the shape-only
    ``execution_phase_marker`` extent: 1 = context prefill over the padded
    rows, 2 = chunked prefill, 3 = single-token decode (``T_exec == N``),
    4/5 = speculative proposal/verify (``use_spec_verify_state=1`` with the
    three tree inputs); the diffusion phases are rejected at enqueue.

    Inputs (the first 15 are REQUIRED; 15-17 exist only with
    ``use_spec_verify_state=1``):

    +----+--------------------------------+-------+----------------------------------------+
    | #  | name                           | dtype | shape                                  |
    +====+================================+=======+========================================+
    | 0  | qkv                            | FP16  | [T_exec, (Hq + 2*Hkv) * D] packed      |
    +----+--------------------------------+-------+----------------------------------------+
    | 1  | index_qk                       | FP16  | [T_exec, (indexer_n_heads+1)*idx_dim]  |
    +----+--------------------------------+-------+----------------------------------------+
    | 2  | past_key_value                 | FP16  | [2, num_pages, KV_PAGE_SIZE, Hkv,      |
    |    |                                |       | D + indexer_head_dim] (see Decode)     |
    +----+--------------------------------+-------+----------------------------------------+
    | 3  | query_lengths                  | INT32 | [N] valid query rows per entry         |
    +----+--------------------------------+-------+----------------------------------------+
    | 4  | rope_rotary_cos_sin            | FP32  | [T_exec, 64] token-aligned rows (row   |
    |    |                                |       | t = cos/sin of activation row t;       |
    |    |                                |       | cos [0:32], sin [32:64]); shared by    |
    |    |                                |       | the main partial rope + indexer        |
    +----+--------------------------------+-------+----------------------------------------+
    | 5  | past_lengths                   | INT32 | [N] AttentionPlugin parity; never      |
    |    |                                |       | read (phase 1 has no past)             |
    +----+--------------------------------+-------+----------------------------------------+
    | 6  | kv_page_table                  | INT32 | [N, 2, max_pages_per_seq]              |
    +----+--------------------------------+-------+----------------------------------------+
    | 7  | q_norm_gamma                   | FP16  | [D] Constant engine weights            |
    +----+--------------------------------+-------+----------------------------------------+
    | 8  | k_norm_gamma                   | FP16  | [D] Constant engine weights            |
    +----+--------------------------------+-------+----------------------------------------+
    | 9  | indexer_q_norm_gamma           | FP16  | [128] Constant engine weights          |
    +----+--------------------------------+-------+----------------------------------------+
    | 10 | indexer_k_norm_gamma           | FP16  | [128] Constant engine weights          |
    +----+--------------------------------+-------+----------------------------------------+
    | 11 | query_start_offsets            | INT32 | [N + 1] physical row starts            |
    +----+--------------------------------+-------+----------------------------------------+
    | 12 | attention_sequence_lengths     | INT32 | [N] TOTAL lengths (past + query)       |
    +----+--------------------------------+-------+----------------------------------------+
    | 13 | execution_phase_marker         | INT32 | [phase] shape-only carrier             |
    +----+--------------------------------+-------+----------------------------------------+
    | 14 | context_sequence_count_carrier | INT32 | [N_ctx] shape-only carrier             |
    +----+--------------------------------+-------+----------------------------------------+
    | 15 | tree_parent_ids                | INT32 | [T_exec] verify-tree parent per row    |
    +----+--------------------------------+-------+----------------------------------------+
    | 16 | tree_depths                    | INT32 | [T_exec] verify-tree depth per row     |
    +----+--------------------------------+-------+----------------------------------------+
    | 17 | spec_verify_layout             | INT32 | shape-only verify layout marker        |
    +----+--------------------------------+-------+----------------------------------------+

    Decode (phase extent 3): the leading ``D`` columns of every pool row hold
    roped K / raw V; the tail ``[D, D + indexer_head_dim)`` persists the QSA
    indexer state across steps (block ``g``'s kbar in the V-row tail of token
    ``4g``; the raw index-K of the trailing incomplete block's tokens in their
    K-row tails; the rope row of that block's first token in its V-row tail
    until the block completes). The TOTAL lengths including the token being
    decoded come from ``attention_sequence_lengths`` — ``query_lengths`` is 1
    per decode row.

    Gamma semantics (IMPORTANT):

    * ``q_norm_gamma`` / ``k_norm_gamma`` (inputs 7/8, main-path per-head
      Gemma qk-norm) are passed **PRE-FOLDED as (1 + w)** by the caller. The
      plugin's fused rope/qk-norm kernel computes the llama convention
      ``normalize(x) * gamma`` directly, so the Gemma ``+1`` must be folded
      into the weights at export time.
    * ``indexer_q_norm_gamma`` / ``indexer_k_norm_gamma`` (inputs 9/10) are
      passed **RAW w** — the CUDA indexer kernel computes ``(1 + w)``
      internally in FP32.

    The four gammas are ``List[float]`` stub args; the ONNX translation
    materializes them as ``Cast(Constant(value_floats), FLOAT16)`` so TRT
    bakes them into the engine as weights at build time.

    Outputs:

    * ``attn_output`` FP16 ``[T_exec, Hq, D]`` — the caller reshapes to
      ``[T_exec, Hq*D]``.
    * ``present_key_value`` FP16 — same shape as ``past_key_value`` (the paged
      pool is aliased in-place by the TRT plugin).

    All attributes and the four ragged carriers are REQUIRED with no
    defaults: ``torch.export`` strips default-matching kwargs from the FX
    graph, which would break the positional ONNX translation (same rule as
    ``attention_plugin``). ``attention_scale == 0.0`` selects the default
    ``1 / sqrt(head_size)``. ``rms_norm_eps`` is used by both the main-path
    and indexer Gemma norms.
    """
    if qkv.ndim != 2 or index_qk.ndim != 2:
        raise ValueError(
            "qsa_attention_plugin expects token-major qkv [T_exec, C] and "
            f"index_qk [T_exec, C'], got {tuple(qkv.shape)} and "
            f"{tuple(index_qk.shape)}")
    attn_output = torch.zeros(qkv.shape[0],
                              num_q_heads,
                              head_size,
                              dtype=qkv.dtype,
                              device=qkv.device)
    present_key_value = torch.zeros_like(past_key_value)
    return attn_output, present_key_value


@qsa_attention_plugin.register_fake
def _(
    qkv,
    index_qk,
    past_key_value,
    query_lengths,
    rope_rotary_cos_sin,
    past_lengths,
    kv_page_table,
    num_q_heads,
    num_kv_heads,
    head_size,
    indexer_n_heads,
    indexer_head_dim,
    indexer_budget,
    indexer_compress_ratio,
    attention_scale,
    rms_norm_eps,
    q_norm_gamma,
    k_norm_gamma,
    indexer_q_norm_gamma,
    indexer_k_norm_gamma,
    query_start_offsets,
    attention_sequence_lengths,
    execution_phase_marker,
    context_sequence_count_carrier,
    use_spec_verify_state=0,
    tree_parent_ids=None,
    tree_depths=None,
    spec_verify_layout=None,
):
    return (torch.empty(qkv.shape[0],
                        num_q_heads,
                        head_size,
                        dtype=qkv.dtype,
                        device=qkv.device), torch.empty_like(past_key_value))


@torch.library.custom_op("trt_edgellm::W2A16MoePlugin", mutates_args=())
def w2a16_moe_plugin(
    hidden_states: torch.Tensor,
    fc1_qweights: torch.Tensor,
    fc2_qweights: torch.Tensor,
    fc1_blocks_scale: torch.Tensor,
    fc2_blocks_scale: torch.Tensor,
    unit_scales: torch.Tensor,
    lookup_table: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    moe_inter_size: int,
) -> torch.Tensor:
    if hidden_states.ndim != 2 or hidden_states.shape[1] != 1024:
        raise ValueError("W2A16 MoE expects [tokens, 1024] hidden states")
    return torch.zeros_like(hidden_states)


@w2a16_moe_plugin.register_fake
def _(hidden_states, fc1_qweights, fc2_qweights, fc1_blocks_scale,
      fc2_blocks_scale, unit_scales, lookup_table, topk_ids, topk_weights,
      num_experts, moe_inter_size):
    return torch.empty_like(hidden_states)
