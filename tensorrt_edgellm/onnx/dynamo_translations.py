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
onnxscript translations for dynamo ONNX export.

Maps ``torch.ops.trt.*`` and ``torch.ops.trt_edgellm.*`` stubs to ONNX graphs.
Implementation files must be importable; ``onnxscript.script`` parses their AST.

Attention plugin
----------------
A single ``_attention_plugin_translation`` covers all feature combinations
(vanilla, FP8-KV, tree attention, FP8-KV + tree attention).  Its signature
matches the full ``trt::attention_plugin`` custom-op schema so positional
alignment with the FX graph (where ``torch.export`` normalizes every kwarg
into a positional arg) is always correct.  ``attention_mask`` /
``attention_pos_id`` are optional ONNX inputs (empty when tree attention is
off); ``qkv_scales`` defaults to ``[1.0, 1.0, 1.0]`` in the op schema so it
is always a valid FLOATS attribute.
"""

from typing import Sequence

import onnx
import onnxscript
import torch
from onnxscript import opset21 as _op21
from onnxscript import script

# Custom ONNX domains
_trt = onnxscript.values.Opset("trt", 1)
_trt_edgellm = onnxscript.values.Opset("trt_edgellm", 1)

# ---------------------------------------------------------------------------
# Attention plugin translation (unified — all feature combinations)
# ---------------------------------------------------------------------------


@script()
def _attention_plugin_translation(
    query_states: onnxscript.FLOAT16,
    key_states: onnxscript.FLOAT16,
    value_states: onnxscript.FLOAT16,
    past_key_value: onnxscript.FLOAT16,
    context_lengths: onnxscript.INT32,
    rope_rotary_cos_sin: onnxscript.FLOAT,
    kvcache_start_index: onnxscript.INT32,
    kv_page_table: onnxscript.INT32,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    sliding_window_size: int,
    enable_tree_attention: int,
    enable_fp8_kv_cache: int,
    attention_scale: float,
    enable_vision_block_attention: int,
    attention_mask: onnxscript.INT32,
    attention_pos_id: onnxscript.INT32,
    qkv_scales: Sequence[float],
) -> tuple[onnxscript.FLOAT16, onnxscript.FLOAT16]:
    """Unified attention plugin covering vanilla, FP8-KV, tree, and tree+FP8-KV.

    The signature matches the full ``trt::attention_plugin`` custom-op schema
    so torch.export's positional-arg normalisation is always aligned.
    ``attention_mask`` / ``attention_pos_id`` are optional ONNX inputs (empty
    when tree attention is off).  ``qkv_scales`` defaults to [1, 1, 1] in the
    op schema and is always a valid FLOATS attribute.
    """
    attn_4d, present_kv = _trt_edgellm.AttentionPlugin(
        query_states,
        key_states,
        value_states,
        past_key_value,
        context_lengths,
        rope_rotary_cos_sin,
        kvcache_start_index,
        kv_page_table,
        attention_mask,
        attention_pos_id,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        enable_tree_attention=enable_tree_attention,
        enable_fp8_kv_cache=enable_fp8_kv_cache,
        enable_vision_block_attention=enable_vision_block_attention,
        sliding_window_size=sliding_window_size,
        qkv_scales=qkv_scales,
        attention_scale=attention_scale,
        _outputs=2,
    )
    return attn_4d, present_kv


# ---------------------------------------------------------------------------
# FP8 ops
# ---------------------------------------------------------------------------


@script()
def _fp8_quantize_translation(
    hidden_states: onnxscript.FLOAT16,
    scale: onnxscript.FLOAT16,
) -> onnxscript.FLOAT8E4M3FN:
    """Standard ONNX QuantizeLinear (see onnx.ai QuantizeLinear)."""
    return _op21.QuantizeLinear(
        hidden_states,
        scale,
        output_dtype=int(onnx.TensorProto.FLOAT8E4M3FN),
    )


@script()
def _fp8_dequantize_translation(
    x: onnxscript.FLOAT8E4M3FN,
    scale: onnxscript.FLOAT16,
) -> onnxscript.FLOAT16:
    """Standard ONNX DequantizeLinear; FP16 scale -> FP16 dequantized output."""
    return _op21.DequantizeLinear(x, scale)


# ---------------------------------------------------------------------------
# NVFP4 ops
# ---------------------------------------------------------------------------


@script()
def _nvfp4_act_qdq_translation(
    hidden_states: onnxscript.FLOAT16,
    global_scale: onnxscript.FLOAT,
) -> onnxscript.FLOAT16:
    """DynQ + 2x trt::DQ for NVFP4 activation quantization.

    Emits the same graph as ModelOpt's ``export_fp4(dynamic)``::

        TRT_FP4DynamicQuantize(x, scale, axis=-1, block_size=16, scale_type=17)
            -> (x_f4, sx_f8)
        trt::DequantizeLinear(sx_f8, scale)
            -> dq_scale
        trt::DequantizeLinear(x_f4, dq_scale, axis=-1, block_size=16)
            -> x_dq
    """
    x_f4, sx_f8 = _trt.TRT_FP4DynamicQuantize(
        hidden_states,
        global_scale,
        axis=-1,
        block_size=16,
        scale_type=17,
        _outputs=2,
    )
    # Cast fp32 global_scale -> fp16 so DQ outputs are fp16 (not fp32)
    global_scale_f16 = _op21.Cast(global_scale, to=10)  # 10 = FLOAT16
    dq_scale = _trt.DequantizeLinear(sx_f8, global_scale_f16)
    x_dq = _trt.DequantizeLinear(x_f4, dq_scale, axis=-1, block_size=16)
    return x_dq


@script()
def _nvfp4_dequantize_translation(
    weight: onnxscript.INT8,
    weight_scale: onnxscript.FLOAT8E4M3FN,
    weight_scale_2: onnxscript.FLOAT,
    group_size: int,
) -> onnxscript.FLOAT16:
    """2xstandard-ONNX DequantizeLinear for NVFP4 weight dequantization.

    Emits the same graph as ModelOpt's ``fp4qdq_to_2dq()``::

        DequantizeLinear(weight_scale_fp8, weight_scale_2_fp32) -> ws
        DequantizeLinear(weight_fp4, ws, axis=-1, block_size=group_size) -> w_dq
    """
    # DQ1 (standard ONNX): fp8 per-block scales -> float32 scales
    ws = _op21.DequantizeLinear(weight_scale, weight_scale_2)
    # DQ2 (standard ONNX): FLOAT4E2M1 weight -> float32 (scale type propagates)
    # Note: weight initializer is rewritten from INT8 to FLOAT4E2M1 post-export.
    w_dq = _op21.DequantizeLinear(weight, ws, axis=-1, block_size=group_size)
    # Cast float32 -> float16 to match activation dtype for MatMul
    return _op21.Cast(w_dq, to=10)  # 10 = ONNX TensorProto.FLOAT16


# ---------------------------------------------------------------------------
# MXFP8 ops
# ---------------------------------------------------------------------------


@script()
def _mxfp8_act_qdq_translation(
    hidden_states: onnxscript.FLOAT16, ) -> onnxscript.FLOAT16:
    """MXFP8 activation: DynQ + DQ.

    Emits::

        TRT_MXFP8DynamicQuantize(x, axis=-1, block_size=32, output_dtype=17)
            -> (x_f8, sx_e8m0)
        TRT_MXFP8DequantizeLinear(x_f8, sx_e8m0,
            axis=-1, block_size=32, output_dtype=10)
            -> x_dq [float16]
    """
    x_f8, sx_e8m0 = _trt.TRT_MXFP8DynamicQuantize(
        hidden_states,
        axis=-1,
        block_size=32,
        output_dtype=17,  # FLOAT8E4M3FN
        _outputs=2,
    )
    x_dq = _trt.TRT_MXFP8DequantizeLinear(
        x_f8,
        sx_e8m0,
        axis=-1,
        block_size=32,
        output_dtype=10,  # FLOAT16
    )
    return x_dq


@script()
def _mxfp8_weight_dq_translation(
    weight: onnxscript.FLOAT8E4M3FN,
    weight_scale: onnxscript.UINT8,
    block_size: int,
) -> onnxscript.FLOAT16:
    """MXFP8 weight dequantize: TRT_MXFP8DequantizeLinear.

    Emits::

        TRT_MXFP8DequantizeLinear(weight, weight_scale,
            axis=-1, block_size=block_size, output_dtype=10) -> w_dq [float16]
    """
    w_dq = _trt.TRT_MXFP8DequantizeLinear(
        weight,
        weight_scale,
        axis=-1,
        block_size=block_size,
        output_dtype=10,  # FLOAT16
    )
    return w_dq


# ---------------------------------------------------------------------------
# AWQ / INT4 op
# ---------------------------------------------------------------------------


@script()
def _int4_groupwise_gemm_translation(
    hidden_states: onnxscript.FLOAT16,
    qweight: onnxscript.INT8,
    scales: onnxscript.FLOAT16,
    gemm_n: int,
    gemm_k: int,
    group_size: int,
) -> onnxscript.FLOAT16:
    return _trt_edgellm.Int4GroupwiseGemmPlugin(
        hidden_states,
        qweight,
        scales,
        gemm_n=gemm_n,
        gemm_k=gemm_k,
        group_size=group_size,
    )


# ---------------------------------------------------------------------------
# INT8 SmoothQuant ops
# ---------------------------------------------------------------------------


@script()
def _int8_sq_act_qdq_translation(
    hidden_states: onnxscript.FLOAT16,
    scale: onnxscript.FLOAT,
) -> onnxscript.FLOAT16:
    """Per-tensor INT8 activation QDQ: QuantizeLinear + DequantizeLinear.

    Emits the standard ONNX QDQ pattern TRT recognises for INT8 GEMM fusion::

        QuantizeLinear(x, scale, output_dtype=INT8) -> q
        DequantizeLinear(q, scale)                  -> dq  [float32]
        Cast(dq, to=FLOAT16)                        -> output
    """
    # output_dtype=3 -> INT8 (symmetric, zero_point=0)
    quantized = _op21.QuantizeLinear(hidden_states, scale, output_dtype=3)
    dq = _op21.DequantizeLinear(quantized, scale)
    return _op21.Cast(dq, to=10)  # 10 = FLOAT16


@script()
def _int8_sq_weight_dq_translation(
    weight: onnxscript.INT8,
    scale: onnxscript.FLOAT,
) -> onnxscript.FLOAT16:
    """Per-channel INT8 weight DequantizeLinear (axis=0).

    Emits::

        DequantizeLinear(weight, scale, axis=0) -> dq  [float32]
        Cast(dq, to=FLOAT16)                    -> output
    """
    dq = _op21.DequantizeLinear(weight, scale, axis=0)
    return _op21.Cast(dq, to=10)  # 10 = FLOAT16


# ---------------------------------------------------------------------------
# Hybrid (Mamba) ops
# ---------------------------------------------------------------------------


@script()
def _causal_conv1d_translation(
    hidden_states: onnxscript.FLOAT16,
    weight: onnxscript.FLOAT16,
    bias: onnxscript.FLOAT16,
    conv_state: onnxscript.FLOAT16,
    context_lengths: onnxscript.INT32,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> tuple[onnxscript.FLOAT16, onnxscript.FLOAT16, onnxscript.FLOAT16]:
    output, conv_state_out = _trt_edgellm.causal_conv1d(
        hidden_states,
        weight,
        bias,
        conv_state,
        context_lengths,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_mtp=0,
        _outputs=2,
    )
    intermediate_conv_state_out = _op21.Identity(conv_state_out)
    return output, conv_state_out, intermediate_conv_state_out


@script()
def _causal_conv1d_with_intermediate_translation(
    hidden_states: onnxscript.FLOAT16,
    weight: onnxscript.FLOAT16,
    bias: onnxscript.FLOAT16,
    conv_state: onnxscript.FLOAT16,
    context_lengths: onnxscript.INT32,
    spec_verify_phase_marker: onnxscript.INT32,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> tuple[onnxscript.FLOAT16, onnxscript.FLOAT16, onnxscript.FLOAT16]:
    output, conv_state_out, intermediate_conv_state_out = _trt_edgellm.causal_conv1d(
        hidden_states,
        weight,
        bias,
        conv_state,
        context_lengths,
        spec_verify_phase_marker,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_mtp=1,
        _outputs=3,
    )
    return output, conv_state_out, intermediate_conv_state_out


@script()
def _causal_conv1d_with_intermediate_tree_translation(
    hidden_states: onnxscript.FLOAT16,
    weight: onnxscript.FLOAT16,
    bias: onnxscript.FLOAT16,
    conv_state: onnxscript.FLOAT16,
    context_lengths: onnxscript.INT32,
    spec_verify_phase_marker: onnxscript.INT32,
    tree_parent_ids: onnxscript.INT32,
    tree_depths: onnxscript.INT32,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> tuple[onnxscript.FLOAT16, onnxscript.FLOAT16, onnxscript.FLOAT16]:
    output, conv_state_out, intermediate_conv_state_out = _trt_edgellm.causal_conv1d(
        hidden_states,
        weight,
        bias,
        conv_state,
        context_lengths,
        spec_verify_phase_marker,
        tree_parent_ids,
        tree_depths,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_ddtree=1,
        _outputs=3,
    )
    return output, conv_state_out, intermediate_conv_state_out


def _causal_conv1d_dispatch(
    hidden_states,
    weight,
    bias,
    conv_state,
    context_lengths,
    stride,
    padding,
    dilation,
    groups,
):
    return _causal_conv1d_translation(hidden_states, weight, bias, conv_state,
                                      context_lengths, stride, padding,
                                      dilation, groups)


def _causal_conv1d_intermediate_dispatch(
    hidden_states,
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
    use_ddtree_state=False,
):
    if use_ddtree_state:
        if tree_parent_ids is None or tree_depths is None:
            raise ValueError(
                "causal_conv1d DDTree state path requires tree_parent_ids and tree_depths"
            )
        return _causal_conv1d_with_intermediate_tree_translation(
            hidden_states, weight, bias, conv_state, context_lengths,
            spec_verify_phase_marker, tree_parent_ids, tree_depths, stride,
            padding, dilation, groups)
    return _causal_conv1d_with_intermediate_translation(
        hidden_states, weight, bias, conv_state, context_lengths,
        spec_verify_phase_marker, stride, padding, dilation, groups)


@script()
def _gated_delta_net_translation(
    q: onnxscript.FLOAT16,
    k: onnxscript.FLOAT16,
    v: onnxscript.FLOAT16,
    a: onnxscript.FLOAT16,
    b: onnxscript.FLOAT16,
    A_log: onnxscript.FLOAT,
    dt_bias: onnxscript.FLOAT16,
    h0_source: onnxscript.FLOAT,
    context_lengths: onnxscript.INT32,
    k_dim: int,
    v_dim: int,
) -> tuple[onnxscript.FLOAT16, onnxscript.FLOAT, onnxscript.FLOAT]:
    output, h0_out = _trt_edgellm.gated_delta_net(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        h0_source,
        context_lengths,
        k_dim=k_dim,
        v_dim=v_dim,
        use_mtp=0,
        _outputs=2,
    )
    intermediate_h0_out = _op21.Identity(h0_out)
    return output, h0_out, intermediate_h0_out


@script()
def _gated_delta_net_with_intermediate_translation(
    q: onnxscript.FLOAT16,
    k: onnxscript.FLOAT16,
    v: onnxscript.FLOAT16,
    a: onnxscript.FLOAT16,
    b: onnxscript.FLOAT16,
    A_log: onnxscript.FLOAT,
    dt_bias: onnxscript.FLOAT16,
    h0_source: onnxscript.FLOAT,
    context_lengths: onnxscript.INT32,
    spec_verify_phase_marker: onnxscript.INT32,
    k_dim: int,
    v_dim: int,
) -> tuple[onnxscript.FLOAT16, onnxscript.FLOAT, onnxscript.FLOAT]:
    output, h0_out, intermediate_h0_out = _trt_edgellm.gated_delta_net(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        h0_source,
        context_lengths,
        spec_verify_phase_marker,
        k_dim=k_dim,
        v_dim=v_dim,
        use_mtp=1,
        _outputs=3,
    )
    return output, h0_out, intermediate_h0_out


@script()
def _gated_delta_net_with_intermediate_tree_translation(
    q: onnxscript.FLOAT16,
    k: onnxscript.FLOAT16,
    v: onnxscript.FLOAT16,
    a: onnxscript.FLOAT16,
    b: onnxscript.FLOAT16,
    A_log: onnxscript.FLOAT,
    dt_bias: onnxscript.FLOAT16,
    h0_source: onnxscript.FLOAT,
    context_lengths: onnxscript.INT32,
    spec_verify_phase_marker: onnxscript.INT32,
    tree_parent_ids: onnxscript.INT32,
    tree_depths: onnxscript.INT32,
    k_dim: int,
    v_dim: int,
) -> tuple[onnxscript.FLOAT16, onnxscript.FLOAT, onnxscript.FLOAT]:
    output, h0_out, intermediate_h0_out = _trt_edgellm.gated_delta_net(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        h0_source,
        context_lengths,
        spec_verify_phase_marker,
        tree_parent_ids,
        tree_depths,
        k_dim=k_dim,
        v_dim=v_dim,
        use_ddtree=1,
        _outputs=3,
    )
    return output, h0_out, intermediate_h0_out


def _gated_delta_net_dispatch(
    q,
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
):
    return _gated_delta_net_translation(q, k, v, a, b, A_log, dt_bias,
                                        h0_source, context_lengths, k_dim,
                                        v_dim)


def _gated_delta_net_intermediate_dispatch(
    q,
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
    use_ddtree_state=False,
):
    if use_ddtree_state:
        if tree_parent_ids is None or tree_depths is None:
            raise ValueError(
                "gated_delta_net DDTree state path requires tree_parent_ids and tree_depths"
            )
        return _gated_delta_net_with_intermediate_tree_translation(
            q, k, v, a, b, A_log, dt_bias, h0_source, context_lengths,
            spec_verify_phase_marker, tree_parent_ids, tree_depths, k_dim,
            v_dim)
    return _gated_delta_net_with_intermediate_translation(
        q, k, v, a, b, A_log, dt_bias, h0_source, context_lengths,
        spec_verify_phase_marker, k_dim, v_dim)


@script()
def _update_ssm_state_translation(
    hidden_states: onnxscript.FLOAT16,
    ssm_a: onnxscript.FLOAT,
    ssm_b: onnxscript.FLOAT16,
    ssm_c: onnxscript.FLOAT16,
    ssm_d: onnxscript.FLOAT16,
    dt: onnxscript.FLOAT16,
    dt_bias: onnxscript.FLOAT16,
    state: onnxscript.FLOAT16,
    context_lengths: onnxscript.INT32,
    dt_softplus: int,
    ngroups: int,
    chunk_size: int = 0,
) -> tuple[onnxscript.FLOAT16, onnxscript.FLOAT16]:
    output, state_out = _trt_edgellm.update_ssm_state(
        hidden_states,
        ssm_a,
        ssm_b,
        ssm_c,
        ssm_d,
        dt,
        dt_bias,
        state,
        context_lengths,
        dt_softplus=dt_softplus,
        ngroups=ngroups,
        chunk_size=chunk_size,
        _outputs=2,
    )
    return output, state_out


# ---------------------------------------------------------------------------
# GatherND op  (token selection)
# ---------------------------------------------------------------------------


@script()
def _gather_nd_translation(
    value: onnxscript.FLOAT16,
    indices: onnxscript.INT64,
) -> onnxscript.FLOAT16:
    """ONNX GatherND with batch_dims=1 for last-token selection.

    Converts [B, S, H] hidden_states + [B, T] indices -> [B, T, H].
    indices is unsqueezed to [B, T, 1] as required by the GatherND spec.

    GatherND (opset 16+), Unsqueeze (opset 13+), and Constant are all
    available at opset 21.
    """
    axes = _op21.Constant(value_ints=[-1])
    indices_3d = _op21.Unsqueeze(indices, axes)
    return _op21.GatherND(value, indices_3d, batch_dims=1)


# ---------------------------------------------------------------------------
# ViT attention op
# ---------------------------------------------------------------------------


@script()
def _vit_attention_plugin_translation(
    query_states: onnxscript.FLOAT16,
    key_states: onnxscript.FLOAT16,
    value_states: onnxscript.FLOAT16,
    cu_seqlens: onnxscript.INT32,
    max_seqlen_carrier: onnxscript.INT32,
    num_heads: int,
    head_size: int,
    attention_scale: float,
) -> onnxscript.FLOAT16:
    """ViT ragged self-attention without KV cache."""
    return _trt_edgellm.ViTAttentionPlugin(
        query_states,
        key_states,
        value_states,
        cu_seqlens,
        max_seqlen_carrier,
        num_heads=num_heads,
        head_size=head_size,
        attention_scale=attention_scale,
    )


# ---------------------------------------------------------------------------
# TRT-native ViT attention (TRT >= 11, packed NHD with Q scaling)
# ---------------------------------------------------------------------------


@script()
def _trt_ragged_attention_inner(
    query_states: onnxscript.FLOAT16,
    key_states: onnxscript.FLOAT16,
    value_states: onnxscript.FLOAT16,
    mask: onnxscript.FLOAT16,
    query_lengths: onnxscript.INT32,
    kv_lengths: onnxscript.INT32,
) -> onnxscript.FLOAT16:
    """Inner onnxscript function with all 6 positional inputs."""
    return _trt.TRT_Attention(
        query_states,
        key_states,
        value_states,
        mask,
        query_lengths,
        kv_lengths,
        query_form="packed_nhd",
        kv_form="packed_nhd",
        causal_kind="none",
        TRT_decomposable=0,
    )


def _trt_ragged_attention_translation(
    query_states,
    key_states,
    value_states,
    query_lengths,
    kv_lengths,
    num_heads,
    head_size,
    attention_scale,
    mask=None,
):
    """Ragged self-attention via TRT-native IAttention (packed NHD).

    Non-identity attention scaling is folded into Q before TRT attention.
    query_lengths and kv_lengths must be separate graph tensors — TRT
    crashes when the same ONNX tensor is wired to both positions.
    """
    if attention_scale != 1.0:
        query_states = _op21.Mul(query_states, attention_scale)
    return _trt_ragged_attention_inner(
        query_states,
        key_states,
        value_states,
        mask,
        query_lengths,
        kv_lengths,
    )


# ---------------------------------------------------------------------------
# TRT native attention ops (RotaryEmbedding, TensorScatter, Attention)
# ---------------------------------------------------------------------------


@script()
def _rope_onnx_translation(
    x: onnxscript.FLOAT16,
    cos: onnxscript.FLOAT16,
    sin: onnxscript.FLOAT16,
    position_ids: onnxscript.INT32,
) -> onnxscript.FLOAT16:
    return _trt.RotaryEmbedding(x, cos, sin, position_ids)


@script()
def _kv_cache_update_onnx_translation(
    cache: onnxscript.FLOAT16,
    new_kv: onnxscript.FLOAT16,
    cache_indices: onnxscript.INT32,
) -> onnxscript.FLOAT16:
    return _trt.TensorScatter(cache, new_kv, cache_indices)


@script()
def _attention_onnx_translation(
    query: onnxscript.FLOAT16,
    key: onnxscript.FLOAT16,
    value: onnxscript.FLOAT16,
    attn_mask: onnxscript.FLOAT16,
    is_causal: int,
    scale: float,
) -> onnxscript.FLOAT16:
    return _trt.Attention(
        query,
        key,
        value,
        attn_mask,
        is_causal=is_causal,
        TRT_decomposable=1,
        scale=scale,
    )


# ---------------------------------------------------------------------------
# INT4 MoE plugin
# ---------------------------------------------------------------------------


@script()
def _int4_moe_plugin_translation(
    router_logits: onnxscript.FLOAT,
    hidden_states: onnxscript.FLOAT16,
    fc_gate_up_qweights: onnxscript.INT8,
    fc_gate_up_scales: onnxscript.FLOAT16,
    fc_down_qweights: onnxscript.INT8,
    fc_down_scales: onnxscript.FLOAT16,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    quantization_group_size: int,
) -> onnxscript.FLOAT16:
    return _trt_edgellm.Int4MoePlugin(
        router_logits,
        hidden_states,
        fc_gate_up_qweights,
        fc_gate_up_scales,
        fc_down_qweights,
        fc_down_scales,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        moe_inter_size=moe_inter_size,
        activation_type=activation_type,
        quantization_group_size=quantization_group_size,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@script()
def _nvfp4_moe_plugin_translation(
    router_logits: onnxscript.FLOAT,
    hidden_states: onnxscript.FLOAT16,
    fc1_qweights: onnxscript.INT8,
    fc1_blocks_scale: onnxscript.INT8,
    fc1_alpha: onnxscript.FLOAT,
    fc2_qweights: onnxscript.INT8,
    fc2_blocks_scale: onnxscript.INT8,
    fc2_alpha: onnxscript.FLOAT,
    input_global_scale: onnxscript.FLOAT,
    down_input_scale: onnxscript.FLOAT,
    e_score_correction_bias: onnxscript.FLOAT,
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
) -> onnxscript.FLOAT16:
    output = _trt_edgellm.Nvfp4MoePlugin(
        router_logits,
        hidden_states,
        fc1_qweights,
        fc1_blocks_scale,
        fc1_alpha,
        fc2_qweights,
        fc2_blocks_scale,
        fc2_alpha,
        input_global_scale,
        down_input_scale,
        e_score_correction_bias,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        moe_inter_size=moe_inter_size,
        activation_type=activation_type,
        n_group=n_group,
        topk_group=topk_group,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
        routing_mode=routing_mode,
        backend=backend,
        io_dtype=io_dtype,
        max_routed_rows=max_routed_rows,
    )
    return output


# ---------------------------------------------------------------------------
# NvFP4MoEPluginGeforce (SM12x fused; same signature, different plugin layer)
# ---------------------------------------------------------------------------


@script()
def _nvfp4_moe_plugin_geforce_translation(
    router_logits: onnxscript.FLOAT,
    hidden_states: onnxscript.FLOAT16,
    fc1_qweights: onnxscript.INT8,
    fc1_blocks_scale: onnxscript.INT8,
    fc1_alpha: onnxscript.FLOAT,
    fc2_qweights: onnxscript.INT8,
    fc2_blocks_scale: onnxscript.INT8,
    fc2_alpha: onnxscript.FLOAT,
    input_global_scale: onnxscript.FLOAT,
    down_input_scale: onnxscript.FLOAT,
    e_score_correction_bias: onnxscript.FLOAT,
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
) -> onnxscript.FLOAT16:
    output = _trt_edgellm.NvFP4MoEPluginGeforce(
        router_logits,
        hidden_states,
        fc1_qweights,
        fc1_blocks_scale,
        fc1_alpha,
        fc2_qweights,
        fc2_blocks_scale,
        fc2_alpha,
        input_global_scale,
        down_input_scale,
        e_score_correction_bias,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        moe_inter_size=moe_inter_size,
        activation_type=activation_type,
        n_group=n_group,
        topk_group=topk_group,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
        routing_mode=routing_mode,
        backend=backend,
        io_dtype=io_dtype,
        max_routed_rows=max_routed_rows,
    )
    return output


# ---------------------------------------------------------------------------
# FusedNvfp4GemmAllReducePlugin (row-parallel NVFP4 GEMM + AllReduce)
# ---------------------------------------------------------------------------


@script()
def _fused_nvfp4_gemm_allreduce_translation(
    hidden_states: onnxscript.FLOAT16,
    global_scale: onnxscript.FLOAT,
    weight_f4: onnxscript.INT8,
    weight_f8_scale: onnxscript.FLOAT8E4M3FN,
    weight_f32_scale: onnxscript.FLOAT,
    tp_size: int,
) -> onnxscript.FLOAT16:
    """Emit the full row-parallel NVFP4 GEMM + AllReduce ONNX subgraph."""
    x_f4, sx_f8 = _trt.TRT_FP4DynamicQuantize(
        hidden_states,
        global_scale,
        axis=-1,
        block_size=16,
        scale_type=17,
        _outputs=2,
    )
    combined_scale = _trt.DequantizeLinear(sx_f8, global_scale)
    return _trt_edgellm.FusedNvfp4GemmAllReducePlugin(
        x_f4,
        combined_scale,
        weight_f4,
        weight_f8_scale,
        weight_f32_scale,
        tp_size=tp_size,
    )


# ---------------------------------------------------------------------------
# DFlash target KV cache update op
# ---------------------------------------------------------------------------


@script()
def _dflash_target_kv_cache_update_translation(
    k_delta: onnxscript.FLOAT16,
    v_delta: onnxscript.FLOAT16,
    past_key_value: onnxscript.FLOAT16,
    rope_cos_sin: onnxscript.FLOAT,
    delta_start_positions: onnxscript.INT32,
    delta_lengths: onnxscript.INT32,
) -> onnxscript.FLOAT16:
    """DFlash target KV cache update: apply RoPE to k_delta, write k+v into cache."""
    present_kv = _trt_edgellm.DFlashTargetKVCacheUpdate(
        k_delta,
        v_delta,
        past_key_value,
        rope_cos_sin,
        delta_start_positions,
        delta_lengths,
    )
    return present_kv


# ---------------------------------------------------------------------------
# Gemma4 Audio Attention Plugin
# ---------------------------------------------------------------------------


@script()
def _gemma4_audio_attention_plugin_translation(
    q_raw: onnxscript.FLOAT16,
    k_raw: onnxscript.FLOAT16,
    v: onnxscript.FLOAT16,
    gamma: onnxscript.FLOAT,
    rel_key: onnxscript.FLOAT16,
    valid: onnxscript.BOOL,
    seq_len_carrier: onnxscript.INT32,
    chunk_size: int,
    left_horizon: int,
    context_size: int,
    logit_cap: float,
) -> onnxscript.FLOAT16:
    """Gemma4 audio chunked local attention plugin."""
    return _trt_edgellm.Gemma4AudioAttentionPlugin(
        q_raw,
        k_raw,
        v,
        gamma,
        rel_key,
        valid,
        seq_len_carrier,
        chunk_size=chunk_size,
        left_horizon=left_horizon,
        context_size=context_size,
        logit_cap=logit_cap,
    )


def build_custom_translation_table() -> dict:
    """Return the ``custom_translation_table`` for ``torch.onnx.export(dynamo=True)``.

    Maps each ``torch.ops.trt.*`` / ``torch.ops.trt_edgellm.*`` op to its
    onnxscript translation function.
    """
    from .onnx_custom_schemas import \
        register_tensorrt_edgellm_onnx_custom_schemas

    register_tensorrt_edgellm_onnx_custom_schemas()

    # Ensure custom ops are registered before accessing torch.ops.trt.*
    from ..models import \
        ops  # noqa: F401 - side-effect: registers all custom_ops

    return {
        torch.ops.trt.attention_plugin.default:
        _attention_plugin_translation,
        torch.ops.trt.fp8_quantize.default:
        _fp8_quantize_translation,
        torch.ops.trt.fp8_dequantize.default:
        _fp8_dequantize_translation,
        torch.ops.trt.nvfp4_act_qdq.default:
        _nvfp4_act_qdq_translation,
        torch.ops.trt.nvfp4_dequantize.default:
        _nvfp4_dequantize_translation,
        torch.ops.trt.mxfp8_act_qdq.default:
        _mxfp8_act_qdq_translation,
        torch.ops.trt.mxfp8_weight_dq.default:
        _mxfp8_weight_dq_translation,
        torch.ops.trt.int4_groupwise_gemm.default:
        _int4_groupwise_gemm_translation,
        torch.ops.trt.int8_sq_act_qdq.default:
        _int8_sq_act_qdq_translation,
        torch.ops.trt.int8_sq_weight_dq.default:
        _int8_sq_weight_dq_translation,
        torch.ops.trt_edgellm.causal_conv1d.default:
        _causal_conv1d_dispatch,
        torch.ops.trt_edgellm.causal_conv1d_with_intermediate.default:
        _causal_conv1d_intermediate_dispatch,
        torch.ops.trt_edgellm.update_ssm_state.default:
        _update_ssm_state_translation,
        torch.ops.trt_edgellm.gated_delta_net.default:
        _gated_delta_net_dispatch,
        torch.ops.trt_edgellm.gated_delta_net_with_intermediate.default:
        _gated_delta_net_intermediate_dispatch,
        torch.ops.trt.vit_attention_plugin.default:
        _vit_attention_plugin_translation,
        torch.ops.trt.trt_ragged_attention.default:
        _trt_ragged_attention_translation,
        torch.ops.trt.gather_nd.default:
        _gather_nd_translation,
        torch.ops.trt_edgellm.int4_moe_plugin.default:
        _int4_moe_plugin_translation,
        torch.ops.trt_edgellm.Nvfp4MoePlugin.default:
        _nvfp4_moe_plugin_translation,
        torch.ops.trt_edgellm.NvFP4MoEPluginGeforce.default:
        _nvfp4_moe_plugin_geforce_translation,
        torch.ops.trt_edgellm.dflash_target_kv_cache_update.default:
        _dflash_target_kv_cache_update_translation,
        # TRT native attention ops (used by Alpamayo)
        torch.ops.trt.rope_onnx.default:
        _rope_onnx_translation,
        torch.ops.trt.kv_cache_update_onnx.default:
        _kv_cache_update_onnx_translation,
        torch.ops.trt.attention_onnx.default:
        _attention_onnx_translation,
        torch.ops.trt_edgellm.fused_nvfp4_gemm_allreduce.default:
        _fused_nvfp4_gemm_allreduce_translation,
        torch.ops.trt_edgellm.gemma4_audio_attention_plugin.default:
        _gemma4_audio_attention_plugin_translation,
    }
