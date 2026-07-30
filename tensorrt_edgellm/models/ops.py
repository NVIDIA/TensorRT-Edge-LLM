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

logger = logging.getLogger(__name__)

#: Tokens per paged-KV page. Must match ``rt::kTOKENS_PER_PAGE``
#: (cpp/runtime/state/pagedKvTypes.h) — the AttentionPlugin's KV cache
#: binding is the pool ``[2, num_pages, KV_PAGE_SIZE, num_kv_heads, head_size]``.
KV_PAGE_SIZE = 128

# ---------------------------------------------------------------------------
# NVFP4 MoE target arch selector
# ---------------------------------------------------------------------------
#
# ``Nvfp4MoePlugin`` (SM100/101/110, split FC1/FC2) and ``NvFP4MoEPluginGeforce``
# (SM12x/Blackwell consumer, fused) share the same 11-input ONNX surface and
# attribute set, but consume **different FC1 weight layouts** for SwiGLU MoE:
#
#   * SM100/101/110 expect FC1 packed as the 64-row up/gate interleave that
#     ``_interleave_qwen3_swiglu_fc1`` produces.
#   * SM12x expects FC1 packed as the plain ``[up_all, gate_all]`` concat
#     that ``_concat_qwen3_swiglu_fc1`` produces.
#
# Repacking and modeling code call :func:`use_geforce_nvfp4_moe` to pick the
# matching plugin op and FC1 layout at export time. Override via env var:
#
#   EDGELLM_NVFP4_MOE_TARGET=sm100  -> Nvfp4MoePlugin
#   EDGELLM_NVFP4_MOE_TARGET=sm110  -> Nvfp4MoePlugin            (default)
#   EDGELLM_NVFP4_MOE_TARGET=sm12x  -> NvFP4MoEPluginGeforce
#
# Accepted aliases for SM12x: ``sm120``, ``sm121``, ``geforce``.

_NVFP4_MOE_TARGET_ENV = "EDGELLM_NVFP4_MOE_TARGET"
_NVFP4_MOE_SM110_ALIASES = frozenset(
    ("sm100", "sm101", "sm110", "blackwell_dc", "thor", ""))
_NVFP4_MOE_SM12X_ALIASES = frozenset(("sm12x", "sm120", "sm121", "geforce"))


def use_geforce_nvfp4_moe() -> bool:
    """Return True iff exporting for ``NvFP4MoEPluginGeforce`` (SM12x).

    The default is the SM100/101/110 path so existing export pipelines keep producing
    the 64-row up/gate interleave layout consumed by ``Nvfp4MoePlugin``.
    """
    val = os.environ.get(_NVFP4_MOE_TARGET_ENV, "sm110").strip().lower()
    if val in _NVFP4_MOE_SM12X_ALIASES:
        return True
    if val in _NVFP4_MOE_SM110_ALIASES:
        return False
    raise ValueError(
        f"{_NVFP4_MOE_TARGET_ENV}={val!r} is not recognized. Use 'sm100'/'sm110' "
        "(Nvfp4MoePlugin) or 'sm12x' (Blackwell consumer/NvFP4MoEPluginGeforce). "
        "Aliases: sm101/blackwell_dc/thor, sm120/sm121/geforce.")


# ---------------------------------------------------------------------------
# Custom op: trt::attention_plugin  (unified: vanilla / FP8-KV / EAGLE tree)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::attention_plugin", mutates_args=())
def attention_plugin(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    past_key_value: torch.Tensor,
    context_lengths: torch.Tensor,
    rope_rotary_cos_sin: torch.Tensor,
    kvcache_start_index: torch.Tensor,
    kv_page_table: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    sliding_window_size: int,
    enable_tree_attention: bool,
    enable_fp8_kv_cache: bool,
    attention_scale: float,
    enable_vision_block_attention: bool,
    attention_mask: Optional[torch.Tensor] = None,
    attention_pos_id: Optional[torch.Tensor] = None,
    qkv_scales: Optional[List[float]] = None,
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
    per-request page table (K page ids then derived V page ids); the runtime
    feeds an identity table by default (bit-equivalent to the non-paged path).

    ``past_key_value`` / ``present_key_value`` are the same paged-pool
    allocation, in-place aliased by the TRT plugin (no growth per call —
    the pool is a fixed-size allocation and writes land at page-table
    positions). ``present_key_value`` therefore always has the exact same
    shape as ``past_key_value``, whatever that is (all models — including
    DFlash's combined draft cache — declare the pool shape ``[2, num_pages,
    kTOKENS_PER_PAGE, num_kv_heads, head_size]``; DFlash recovers its logical
    ``[2, max_batch, cap_padded, num_kv_heads, head_size]`` view at enqueue).

    ``enable_tree_attention``, ``enable_fp8_kv_cache``,
    ``enable_vision_block_attention``, and ``attention_scale`` are
    required (no default) so that ``torch.export`` always includes them
    in the FX graph — default-matching kwargs get stripped, breaking
    ONNX translation.

    Callers must always pass ``qkv_scales=[1.0, 1.0, 1.0]`` explicitly so
    the FX graph contains a valid FLOATS value for the ONNX translation.

    When ``enable_tree_attention=True``, ``attention_mask`` and
    ``attention_pos_id`` must be provided (non-None).

    When ``enable_vision_block_attention=True``, the ``attention_mask`` input
    carries a ``[batch, seq_len]`` INT32 vision-block-ID tensor instead of a
    tree mask.  ``-1`` means causal text/audio; equal non-negative IDs identify
    one contiguous image run whose tokens may attend bidirectionally to
    each other.  This mode is mutually exclusive with tree attention.

    The TRT AttentionPlugin kernel returns a 4-D tensor
    ``[batch, seq_len, num_q_heads, head_size]``.
    The caller (``Attention.forward``) is responsible for reshaping to
    ``[batch, seq_len, num_q_heads * head_size]``.
    """
    batch_size, seq_len, _ = query_states.shape
    attn_output = torch.zeros(batch_size,
                              seq_len,
                              num_q_heads,
                              head_size,
                              dtype=query_states.dtype,
                              device=query_states.device)
    present_key_value = torch.zeros_like(past_key_value)
    return attn_output, present_key_value


@attention_plugin.register_fake
def _(query_states,
      key_states,
      value_states,
      past_key_value,
      context_lengths,
      rope_rotary_cos_sin,
      kvcache_start_index,
      kv_page_table,
      num_q_heads,
      num_kv_heads,
      head_size,
      sliding_window_size,
      enable_tree_attention,
      enable_fp8_kv_cache,
      attention_scale,
      enable_vision_block_attention,
      attention_mask=None,
      attention_pos_id=None,
      qkv_scales=None):
    batch_size, seq_len, _ = query_states.shape
    return (torch.empty(batch_size,
                        seq_len,
                        num_q_heads,
                        head_size,
                        dtype=query_states.dtype,
                        device=query_states.device),
            torch.empty_like(past_key_value))


# ---------------------------------------------------------------------------
# Custom op: trt::vit_attention_plugin  (ViT ragged self-attention)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::vit_attention_plugin", mutates_args=())
def vit_attention_plugin(
    query_states: torch.Tensor,  # [T, num_heads, head_size]
    key_states: torch.Tensor,  # [T, num_heads, head_size]
    value_states: torch.Tensor,  # [T, num_heads, head_size]
    cu_seqlens: torch.Tensor,  # [batch+1] int32
    max_seqlen_carrier: torch.Tensor,  # [] or [1] int32 (scalar)
    num_heads: int,
    head_size: int,
    attention_scale: float,
) -> torch.Tensor:
    """ViT ragged self-attention.

    In eager mode, implements varlen SDPA using cu_seqlens to process each
    sequence segment independently.  During dynamo/ONNX tracing the
    register_fake shape propagation is used and this body is not executed.

    Unlike AttentionPlugin, ViT attention has no KV cache and takes ragged
    input with cu_seqlens instead of context_lengths.  RoPE is applied before
    this call.
    """
    import torch.nn.functional as F
    out = torch.empty_like(query_states)
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
def _(query_states, key_states, value_states, cu_seqlens, max_seqlen_carrier,
      num_heads, head_size, attention_scale):
    return torch.empty_like(query_states)


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
    rope_cos_sin: torch.Tensor,
    delta_start_positions: torch.Tensor,
    delta_lengths: torch.Tensor,
) -> torch.Tensor:
    """Update the draft combined KV cache with target-hidden-derived K/V delta.

    k_delta: [B, L, numKVHeads, headDim] FP16, k_normed, not RoPE-applied.
    v_delta: [B, L, numKVHeads, headDim] FP16.
    past_key_value: [2, num_pages, KV_PAGE_SIZE, numKVHeads, headDim] FP16 —
        the paged KV pool (same contract as the AttentionPlugin kv_cache
        binding). maxBatch/capPadded are recovered at enqueue time from
        num_pages and the builder-configured pages_per_slot attribute (see
        DFlashTargetKVCacheUpdatePlugin::setPagesPerSlot); this cache has no
        page table of its own.
    rope_cos_sin: [ropeBatch, cosSinSeqLen, rotaryDim] FP32, cosSinSeqLen <= capPadded.
    delta_start_positions: [B] INT32, old committed draft target cache length.
    delta_lengths: [B] INT32, per-batch delta lengths.

    Applies RoPE to k_delta and writes k_rope + v_delta into the KV cache at
    positions [delta_start, delta_start + t) for each batch element, where
    t < delta_lengths[b]. Positions beyond delta_lengths[b] are skipped.
    Returns present_key_value (same shape as past_key_value — aliased in TRT).
    """
    return past_key_value.clone()


@dflash_target_kv_cache_update.register_fake
def _(k_delta, v_delta, past_key_value, rope_cos_sin, delta_start_positions,
      delta_lengths):
    return torch.empty_like(past_key_value)


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
    index); ``weight_scale`` is ``[out, in//group_size]`` FP8 E4M3 (or an int8
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

    if weight_scale.dtype == torch.int8:
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
    block_scale = (q_block_scale * s2).clamp_min(
        1e-20)  # effective dequant scale

    # Round magnitude to the nearest E2M1 level (midpoints in _FP4_E2M1_BOUNDS).
    bounds = torch.tensor(_FP4_E2M1_BOUNDS, dtype=torch.float32, device=device)
    levels = torch.tensor(_FP4_E2M1_LEVELS, dtype=torch.float32, device=device)
    scaled = xb / block_scale
    idx = torch.searchsorted(bounds, scaled.abs().contiguous())
    deq = torch.sign(scaled) * levels[idx] * block_scale
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
    out_features, packed_in = weight.shape
    return torch.empty(out_features,
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
# Custom op: trt::int4_groupwise_gemm
# ---------------------------------------------------------------------------


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
    hidden_states: torch.Tensor,  # [batch, seq_len, conv_dim]
    weight: torch.Tensor,  # [conv_dim, 1, kernel_size]
    bias: torch.Tensor,  # [conv_dim]
    conv_state: torch.Tensor,  # [batch, conv_dim, conv_kernel]
    context_lengths: torch.Tensor,  # [batch] int32
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stub: causal conv1d."""
    return (torch.zeros_like(hidden_states), conv_state.clone(),
            conv_state.clone())


@causal_conv1d.register_fake
def _(hidden_states, weight, bias, conv_state, context_lengths, stride,
      padding, dilation, groups):
    return (torch.empty_like(hidden_states), conv_state.clone(),
            torch.empty_like(conv_state))


@torch.library.custom_op("trt_edgellm::causal_conv1d_with_intermediate",
                         mutates_args=())
def causal_conv1d_with_intermediate(
    hidden_states: torch.Tensor,  # [batch, seq_len, conv_dim]
    weight: torch.Tensor,  # [conv_dim, 1, kernel_size]
    bias: torch.Tensor,  # [conv_dim]
    conv_state: torch.Tensor,  # [batch, conv_dim, conv_kernel]
    context_lengths: torch.Tensor,  # [batch] int32
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    spec_verify_phase_marker: torch.Tensor,
    tree_parent_ids: Optional[
        torch.Tensor] = None,  # [batch, verify_seq] int32
    tree_depths: Optional[torch.Tensor] = None,  # [batch, verify_seq] int32
    use_ddtree_state: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stub: causal conv1d with per-token intermediate state output."""
    batch_size, seq_len, _ = hidden_states.shape
    intermediate_conv_state = torch.zeros(batch_size,
                                          seq_len,
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
      context_lengths,
      stride,
      padding,
      dilation,
      groups,
      spec_verify_phase_marker,
      tree_parent_ids=None,
      tree_depths=None,
      use_ddtree_state=False):
    batch_size, seq_len, _ = hidden_states.shape
    intermediate_conv_state = torch.empty(batch_size,
                                          seq_len,
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
    hidden_states: torch.Tensor,  # [batch, seq_len, num_heads, head_dim]
    ssm_a: torch.Tensor,  # [num_heads] float32
    ssm_b: torch.Tensor,  # [batch, seq_len, n_groups, ssm_state_size]
    ssm_c: torch.Tensor,  # [batch, seq_len, n_groups, ssm_state_size]
    ssm_d: torch.Tensor,  # [num_heads] float16
    dt: torch.Tensor,  # [batch, seq_len, num_heads]
    dt_bias: torch.Tensor,  # [num_heads] float16
    state: torch.Tensor,  # [batch, num_heads, head_dim, ssm_state_size]
    context_lengths: torch.Tensor,  # [batch] int32
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
      context_lengths,
      dt_softplus,
      ngroups,
      chunk_size=0):
    return torch.empty_like(hidden_states), state.clone()


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::int4_moe_plugin  (sparse MoE with INT4 expert GEMMs)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::int4_moe_plugin", mutates_args=())
def int4_moe_plugin(
    router_logits: torch.
    Tensor,  # [B*S, E] float32 — gate output before softmax
    hidden_states: torch.Tensor,  # [B, S, H] float16
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
    batch_size, seq_len, _ = hidden_states.shape
    return torch.zeros(batch_size,
                       seq_len,
                       hidden_size,
                       dtype=hidden_states.dtype,
                       device=hidden_states.device)


@int4_moe_plugin.register_fake
def _(router_logits, hidden_states, fc_gate_up_qweights, fc_gate_up_scales,
      fc_down_qweights, fc_down_scales, num_experts, top_k, hidden_size,
      moe_inter_size, activation_type, quantization_group_size):
    batch_size, seq_len, _ = hidden_states.shape
    return torch.empty(batch_size,
                       seq_len,
                       hidden_size,
                       dtype=hidden_states.dtype,
                       device=hidden_states.device)


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

_GDN_CHUNK_MAX_NODES = 64
_GDN_CHUNK_PREP_WORDS = (2 * _GDN_CHUNK_MAX_NODES * _GDN_CHUNK_MAX_NODES +
                         5 * _GDN_CHUNK_MAX_NODES + 4)


def _gdn_chunk_buffer_elements(num_k_heads, num_v_heads, k_dim: int,
                               v_dim: int):
    """Return FP32 storage elements for one compact replay-buffer row."""
    stash_node_bytes = (num_k_heads * k_dim * 4 + num_v_heads * 2 * 4 +
                        num_v_heads * v_dim * 2)
    stash_bytes = _GDN_CHUNK_MAX_NODES * stash_node_bytes
    stash_elements = ((stash_bytes + 255) // 256 * 256) // 4
    ks_qs_elements = (num_v_heads * 2 * _GDN_CHUNK_MAX_NODES * k_dim)
    prep_elements = num_v_heads * _GDN_CHUNK_PREP_WORDS
    return stash_elements + ks_qs_elements + prep_elements


@torch.library.custom_op("trt_edgellm::gated_delta_net", mutates_args=())
def gated_delta_net(
    q: torch.Tensor,  # [batch, seq, num_k_heads, k_dim]
    k: torch.Tensor,  # [batch, seq, num_k_heads, k_dim]
    v: torch.Tensor,  # [batch, seq, num_v_heads, v_dim]
    a: torch.Tensor,  # [batch, seq, num_v_heads]
    b: torch.Tensor,  # [batch, seq, num_v_heads]
    A_log: torch.Tensor,  # [num_v_heads] float32
    dt_bias: torch.Tensor,  # [num_v_heads] float16
    h0_source: torch.Tensor,  # [batch, num_v_heads, k_dim, v_dim] float32
    context_lengths: torch.Tensor,  # [batch] int32
    k_dim: int,
    v_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stub: GatedDeltaNet."""
    return torch.zeros_like(v), h0_source.clone(), h0_source.clone()


@gated_delta_net.register_fake
def _(q, k, v, a, b, A_log, dt_bias, h0_source, context_lengths, k_dim, v_dim):
    return torch.empty_like(v), h0_source.clone(), torch.empty_like(h0_source)


@torch.library.custom_op("trt_edgellm::gated_delta_net_with_intermediate",
                         mutates_args=())
def gated_delta_net_with_intermediate(
    q: torch.Tensor,  # [batch, seq, num_k_heads, k_dim]
    k: torch.Tensor,  # [batch, seq, num_k_heads, k_dim]
    v: torch.Tensor,  # [batch, seq, num_v_heads, v_dim]
    a: torch.Tensor,  # [batch, seq, num_v_heads]
    b: torch.Tensor,  # [batch, seq, num_v_heads]
    A_log: torch.Tensor,  # [num_v_heads] float32
    dt_bias: torch.Tensor,  # [num_v_heads] float16
    h0_source: torch.Tensor,  # [batch, num_v_heads, k_dim, v_dim] float32
    context_lengths: torch.Tensor,  # [batch] int32
    k_dim: int,
    v_dim: int,
    spec_verify_phase_marker: torch.Tensor,
    tree_parent_ids: Optional[
        torch.Tensor] = None,  # [batch, verify_seq] int32
    tree_depths: Optional[torch.Tensor] = None,  # [batch, verify_seq] int32
    use_ddtree_state: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stub: GatedDeltaNet with a compact replay-buffer output."""
    batch_size = v.shape[0]
    buffer_elements = _gdn_chunk_buffer_elements(q.shape[2], v.shape[2], k_dim,
                                                 v_dim)
    intermediate_recurrent_state = torch.zeros(batch_size,
                                               buffer_elements,
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
      context_lengths,
      k_dim,
      v_dim,
      spec_verify_phase_marker,
      tree_parent_ids=None,
      tree_depths=None,
      use_ddtree_state=False):
    batch_size = v.shape[0]
    buffer_elements = _gdn_chunk_buffer_elements(q.shape[2], v.shape[2], k_dim,
                                                 v_dim)
    intermediate_recurrent_state = torch.empty(batch_size,
                                               buffer_elements,
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
    return torch.zeros_like(hidden_states)


@nvfp4_moe_plugin.register_fake
def _(router_logits, hidden_states, fc1_qweights, fc1_blocks_scale, fc1_alpha,
      fc2_qweights, fc2_blocks_scale, fc2_alpha, input_global_scale,
      down_input_scale, e_score_correction_bias, num_experts, top_k,
      hidden_size, moe_inter_size, activation_type, n_group, topk_group,
      norm_topk_prob, routed_scaling_factor, routing_mode, backend, io_dtype,
      max_routed_rows):
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
    return torch.zeros_like(hidden_states)


@nvfp4_moe_plugin_geforce.register_fake
def _(router_logits, hidden_states, fc1_qweights, fc1_blocks_scale, fc1_alpha,
      fc2_qweights, fc2_blocks_scale, fc2_alpha, input_global_scale,
      down_input_scale, e_score_correction_bias, num_experts, top_k,
      hidden_size, moe_inter_size, activation_type, n_group, topk_group,
      norm_topk_prob, routed_scaling_factor, routing_mode, backend, io_dtype,
      max_routed_rows):
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
) -> torch.Tensor:
    """Stub: NVFP4 row-parallel GEMM fused with AllReduce.

    Single op that emits the entire chain feeding
    ``FusedNvfp4GemmAllReducePlugin``::

        TRT_FP4DynamicQuantize(x, global_scale, axis=-1, block_size=16, scale_type=17)
            -> (x_f4, sx_f8)
        trt::DequantizeLinear(sx_f8, global_scale)            -> combined_scale_fp32
        FusedNvfp4GemmAllReducePlugin(x_f4, combined_scale, weight_f4,
                                 weight_f8_scale, weight_f32_scale,
                                 tp_size) -> y_fp16   (already AllReduced across ranks)
    """
    out_features = weight_f4.shape[0]
    out_shape = list(hidden_states.shape[:-1]) + [out_features]
    return torch.zeros(*out_shape,
                       dtype=torch.float16,
                       device=hidden_states.device)


@fused_nvfp4_gemm_allreduce.register_fake
def _(hidden_states, global_scale, weight_f4, weight_f8_scale,
      weight_f32_scale, tp_size):
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
