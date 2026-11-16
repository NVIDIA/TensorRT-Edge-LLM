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
into a positional arg) is always correct.  ``context_mask_selector``,
``attention_mask`` and ``attention_pos_id`` are
optional ONNX inputs; ``qkv_scales`` defaults to ``[1.0, 1.0, 1.0]`` in the op
schema so it is always a valid FLOATS attribute.
"""

from typing import Sequence

import onnx
import onnxscript
import torch
from onnxscript import opset21 as _op21
from onnxscript import script
from onnxscript.onnx_types import Union as OnnxUnion

# Custom ONNX domains
_trt = onnxscript.values.Opset("trt", 1)
_trt_edgellm = onnxscript.values.Opset("trt_edgellm", 1)


@script()
def _w2a16_moe_plugin_translation(
    hidden_states: onnxscript.BFLOAT16,
    fc1_qweights: onnxscript.INT32,
    fc2_qweights: onnxscript.INT32,
    fc1_blocks_scale: onnxscript.INT32,
    fc2_blocks_scale: onnxscript.INT32,
    unit_scales: onnxscript.FLOAT,
    lookup_table: onnxscript.INT8,
    topk_ids: onnxscript.INT32,
    topk_weights: onnxscript.FLOAT,
    num_experts: int,
    moe_inter_size: int,
) -> onnxscript.BFLOAT16:
    return _trt_edgellm.W2A16MoePlugin(hidden_states,
                                       fc1_qweights,
                                       fc2_qweights,
                                       fc1_blocks_scale,
                                       fc2_blocks_scale,
                                       unit_scales,
                                       lookup_table,
                                       topk_ids,
                                       topk_weights,
                                       num_experts=num_experts,
                                       moe_inter_size=moe_inter_size)


# ---------------------------------------------------------------------------
# Attention plugin translation (unified — all feature combinations)
# ---------------------------------------------------------------------------


@script()
def _attention_plugin_translation(
    qkv: onnxscript.FLOAT16,
    past_key_value: onnxscript.FLOAT16,
    query_lengths: onnxscript.INT32,
    rope_rotary_cos_sin: onnxscript.FLOAT,
    past_lengths: onnxscript.INT32,
    kv_page_table: onnxscript.INT32,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    sliding_window_size: int,
    enable_tree_attention: int,
    enable_fp8_kv_cache: int,
    attention_scale: float,
    enable_context_mask_selector: int,
    enable_vision_block_attention: int,
    skip_softmax_prefill_input: int,
    context_mask_selector: onnxscript.INT32,
    attention_mask: onnxscript.INT32,
    attention_pos_id: onnxscript.INT32,
    qkv_scales: Sequence[float],
    # Defaults REQUIRED: torch.export strips default-matching kwargs, so callers
    # without these features produce FX nodes lacking the attributes.
    q_norm_gamma: Sequence[float] = (),
    k_norm_gamma: Sequence[float] = (),
    rms_norm_eps: float = 1e-6,
    enable_qk_norm: int = 0,
    qk_norm_post_rope: int = 0,
    enable_kv_shared: int = 0,
    tp_size: int = 1,
    skip_softmax_prefill_scale: onnxscript.INT32 = None,
    skip_softmax_decode_scale: onnxscript.INT32 = None,
    skip_softmax_decode_input: int = 0,
    swa_kv_cache_mode: onnxscript.INT32 = None,
    attention_sinks: Sequence[float] = (),
    enable_attention_sink: int = 0,
    enable_contiguous_query_swa: int = 0,
    query_start_offsets: onnxscript.INT32 = None,
    attention_sequence_lengths: onnxscript.INT32 = None,
    execution_phase_marker: onnxscript.INT32 = None,
    context_sequence_count_carrier: onnxscript.INT32 = None,
    prefix_key_value: onnxscript.FLOAT16 = None,
    prefix_page_table: onnxscript.INT32 = None,
    external_kv_prefix: int = 0,
    fp8_kv_prefix: int = 0,
) -> tuple[onnxscript.FLOAT16, onnxscript.FLOAT16]:
    """Unified attention plugin covering vanilla, FP8-KV, tree, and tree+FP8-KV.

    Feeds the packed ``qkv`` tensor ``[T_exec, (H_q + 2*H_kv) * D]`` to the V3
    ``AttentionPlugin``. The translation always wires the complete optional
    input layout in a stable order: q/k norm gammas, context-mask selector, and
    tree/vision attention mask inputs. The export post-pass compacts disabled
    optional groups so the ONNX node matches the C++ plugin contract.
    """
    # Gammas enter the plugin as FP16 constant INPUTS (engine weights baked at
    # build time); zero-length constants signal "qk_norm disabled".
    q_norm_gamma_fp16 = _op21.Cast(
        _op21.Constant(value_floats=q_norm_gamma),
        to=int(onnx.TensorProto.FLOAT16),
    )
    k_norm_gamma_fp16 = _op21.Cast(
        _op21.Constant(value_floats=k_norm_gamma),
        to=int(onnx.TensorProto.FLOAT16),
    )
    attention_sinks_fp32 = _op21.Constant(value_floats=attention_sinks)
    attn_4d, present_kv = _trt_edgellm.AttentionPlugin(
        qkv,
        past_key_value,
        query_lengths,
        rope_rotary_cos_sin,
        past_lengths,
        kv_page_table,
        q_norm_gamma_fp16,
        k_norm_gamma_fp16,
        context_mask_selector,
        attention_mask,
        attention_pos_id,
        skip_softmax_prefill_scale,
        skip_softmax_decode_scale,
        swa_kv_cache_mode,
        attention_sinks_fp32,
        query_start_offsets,
        attention_sequence_lengths,
        execution_phase_marker,
        context_sequence_count_carrier,
        prefix_key_value,
        prefix_page_table,
        external_kv_prefix=external_kv_prefix,
        fp8_kv_prefix=fp8_kv_prefix,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        enable_tree_attention=enable_tree_attention,
        enable_fp8_kv_cache=enable_fp8_kv_cache,
        enable_context_mask_selector=enable_context_mask_selector,
        enable_vision_block_attention=enable_vision_block_attention,
        sliding_window_size=sliding_window_size,
        qkv_scales=qkv_scales,
        attention_scale=attention_scale,
        skip_softmax_prefill_input=skip_softmax_prefill_input,
        skip_softmax_decode_input=skip_softmax_decode_input,
        rms_norm_eps=rms_norm_eps,
        enable_qk_norm=enable_qk_norm,
        qk_norm_post_rope=qk_norm_post_rope,
        enable_kv_shared=enable_kv_shared,
        tp_size=tp_size,
        enable_attention_sink=enable_attention_sink,
        enable_contiguous_query_swa=enable_contiguous_query_swa,
        plugin_version="1",
        _outputs=2,
    )
    return attn_4d, present_kv


def _attention_plugin_dispatch(
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
    q_norm_gamma=(),
    k_norm_gamma=(),
    rms_norm_eps=1e-6,
    enable_qk_norm=0,
    qk_norm_post_rope=0,
    enable_kv_shared=0,
    tp_size=1,
    skip_softmax_prefill_scale=None,
    skip_softmax_decode_scale=None,
    skip_softmax_decode_input=0,
    swa_kv_cache_mode=None,
    attention_sinks=(),
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
    q_norm_gamma = () if q_norm_gamma is None else q_norm_gamma
    k_norm_gamma = () if k_norm_gamma is None else k_norm_gamma
    # ONNX IR cannot infer the FLOATS attribute type from an empty sequence.
    # Disabled sink inputs are pruned by the export post-pass, so use a typed
    # placeholder until that pruning runs.
    attention_sinks = (0.0, ) if not attention_sinks else attention_sinks
    return _attention_plugin_translation(
        qkv, past_key_value, query_lengths, rope_rotary_cos_sin, past_lengths,
        kv_page_table, num_q_heads, num_kv_heads, head_size,
        sliding_window_size, enable_tree_attention, enable_fp8_kv_cache,
        attention_scale, enable_context_mask_selector,
        enable_vision_block_attention, skip_softmax_prefill_input,
        context_mask_selector, attention_mask, attention_pos_id, qkv_scales,
        q_norm_gamma, k_norm_gamma, rms_norm_eps, enable_qk_norm,
        qk_norm_post_rope, enable_kv_shared, tp_size,
        skip_softmax_prefill_scale, skip_softmax_decode_scale,
        skip_softmax_decode_input, swa_kv_cache_mode, attention_sinks,
        enable_attention_sink, enable_contiguous_query_swa,
        query_start_offsets, attention_sequence_lengths,
        execution_phase_marker, context_sequence_count_carrier,
        prefix_key_value, prefix_page_table, external_kv_prefix, fp8_kv_prefix)


# ---------------------------------------------------------------------------
# QSA attention plugin translation (Qwen Sparse Attention, prefill + decode)
# ---------------------------------------------------------------------------


@script()
def _qsa_attention_plugin_translation(
    qkv: onnxscript.FLOAT16,
    index_qk: onnxscript.FLOAT16,
    past_key_value: onnxscript.FLOAT16,
    query_lengths: onnxscript.INT32,
    rope_rotary_cos_sin: onnxscript.FLOAT,
    past_lengths: onnxscript.INT32,
    kv_page_table: onnxscript.INT32,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    indexer_n_heads: int,
    indexer_head_dim: int,
    indexer_budget: int,
    indexer_compress_ratio: int,
    attention_scale: float,
    rms_norm_eps: float,
    q_norm_gamma: Sequence[float],
    k_norm_gamma: Sequence[float],
    indexer_q_norm_gamma: Sequence[float],
    indexer_k_norm_gamma: Sequence[float],
    query_start_offsets: onnxscript.INT32,
    attention_sequence_lengths: onnxscript.INT32,
    execution_phase_marker: onnxscript.INT32,
    context_sequence_count_carrier: onnxscript.INT32,
) -> tuple[onnxscript.FLOAT16, onnxscript.FLOAT16]:
    """QSA plugin: block-compressed indexer + sparse GQA attention (prefill and decode).

    Signature order matches ``trt::qsa_attention_plugin`` positionally (the
    FX graph normalizes every kwarg into a positional arg). All 15 ONNX
    inputs are required; the node's inputs follow the plugin slot order: the
    seven activation / metadata tensors, the four gamma constants, then the
    ragged-ABI tail ``query_start_offsets, attention_sequence_lengths,
    execution_phase_marker, context_sequence_count_carrier`` (AttentionPlugin
    order, not the GDN ``state_indices`` order).

    Gamma semantics: ``q_norm_gamma`` / ``k_norm_gamma`` arrive PRE-FOLDED as
    (1 + w); ``indexer_q_norm_gamma`` / ``indexer_k_norm_gamma`` arrive RAW w
    (the CUDA indexer kernel adds the 1 internally). All four are FP16
    constant INPUTS (engine weights baked at build time).
    """
    q_norm_gamma_fp16 = _op21.Cast(
        _op21.Constant(value_floats=q_norm_gamma),
        to=int(onnx.TensorProto.FLOAT16),
    )
    k_norm_gamma_fp16 = _op21.Cast(
        _op21.Constant(value_floats=k_norm_gamma),
        to=int(onnx.TensorProto.FLOAT16),
    )
    indexer_q_norm_gamma_fp16 = _op21.Cast(
        _op21.Constant(value_floats=indexer_q_norm_gamma),
        to=int(onnx.TensorProto.FLOAT16),
    )
    indexer_k_norm_gamma_fp16 = _op21.Cast(
        _op21.Constant(value_floats=indexer_k_norm_gamma),
        to=int(onnx.TensorProto.FLOAT16),
    )
    attn_4d, present_kv = _trt_edgellm.QsaAttentionPlugin(
        qkv,
        index_qk,
        past_key_value,
        query_lengths,
        rope_rotary_cos_sin,
        past_lengths,
        kv_page_table,
        q_norm_gamma_fp16,
        k_norm_gamma_fp16,
        indexer_q_norm_gamma_fp16,
        indexer_k_norm_gamma_fp16,
        query_start_offsets,
        attention_sequence_lengths,
        execution_phase_marker,
        context_sequence_count_carrier,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        indexer_n_heads=indexer_n_heads,
        indexer_head_dim=indexer_head_dim,
        indexer_budget=indexer_budget,
        indexer_compress_ratio=indexer_compress_ratio,
        attention_scale=attention_scale,
        rms_norm_eps=rms_norm_eps,
        use_spec_verify_state=0,
        _outputs=2,
    )
    return attn_4d, present_kv


@script()
def _qsa_attention_plugin_spec_translation(
    qkv: onnxscript.FLOAT16,
    index_qk: onnxscript.FLOAT16,
    past_key_value: onnxscript.FLOAT16,
    query_lengths: onnxscript.INT32,
    rope_rotary_cos_sin: onnxscript.FLOAT,
    past_lengths: onnxscript.INT32,
    kv_page_table: onnxscript.INT32,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    indexer_n_heads: int,
    indexer_head_dim: int,
    indexer_budget: int,
    indexer_compress_ratio: int,
    attention_scale: float,
    rms_norm_eps: float,
    q_norm_gamma: Sequence[float],
    k_norm_gamma: Sequence[float],
    indexer_q_norm_gamma: Sequence[float],
    indexer_k_norm_gamma: Sequence[float],
    query_start_offsets: onnxscript.INT32,
    attention_sequence_lengths: onnxscript.INT32,
    execution_phase_marker: onnxscript.INT32,
    context_sequence_count_carrier: onnxscript.INT32,
    tree_parent_ids: onnxscript.INT32,
    tree_depths: onnxscript.INT32,
    spec_verify_layout: onnxscript.INT32,
) -> tuple[onnxscript.FLOAT16, onnxscript.FLOAT16]:
    """QSA plugin with verify state: the 15 inputs above plus the tree inputs."""
    q_norm_gamma_fp16 = _op21.Cast(
        _op21.Constant(value_floats=q_norm_gamma),
        to=int(onnx.TensorProto.FLOAT16),
    )
    k_norm_gamma_fp16 = _op21.Cast(
        _op21.Constant(value_floats=k_norm_gamma),
        to=int(onnx.TensorProto.FLOAT16),
    )
    indexer_q_norm_gamma_fp16 = _op21.Cast(
        _op21.Constant(value_floats=indexer_q_norm_gamma),
        to=int(onnx.TensorProto.FLOAT16),
    )
    indexer_k_norm_gamma_fp16 = _op21.Cast(
        _op21.Constant(value_floats=indexer_k_norm_gamma),
        to=int(onnx.TensorProto.FLOAT16),
    )
    attn_4d, present_kv = _trt_edgellm.QsaAttentionPlugin(
        qkv,
        index_qk,
        past_key_value,
        query_lengths,
        rope_rotary_cos_sin,
        past_lengths,
        kv_page_table,
        q_norm_gamma_fp16,
        k_norm_gamma_fp16,
        indexer_q_norm_gamma_fp16,
        indexer_k_norm_gamma_fp16,
        query_start_offsets,
        attention_sequence_lengths,
        execution_phase_marker,
        context_sequence_count_carrier,
        tree_parent_ids,
        tree_depths,
        spec_verify_layout,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        indexer_n_heads=indexer_n_heads,
        indexer_head_dim=indexer_head_dim,
        indexer_budget=indexer_budget,
        indexer_compress_ratio=indexer_compress_ratio,
        attention_scale=attention_scale,
        rms_norm_eps=rms_norm_eps,
        use_spec_verify_state=1,
        _outputs=2,
    )
    return attn_4d, present_kv


def _qsa_attention_plugin_dispatch(
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
    common = (qkv, index_qk, past_key_value, query_lengths,
              rope_rotary_cos_sin, past_lengths, kv_page_table, num_q_heads,
              num_kv_heads, head_size, indexer_n_heads, indexer_head_dim,
              indexer_budget, indexer_compress_ratio, attention_scale,
              rms_norm_eps, q_norm_gamma, k_norm_gamma, indexer_q_norm_gamma,
              indexer_k_norm_gamma, query_start_offsets,
              attention_sequence_lengths, execution_phase_marker,
              context_sequence_count_carrier)
    if not use_spec_verify_state:
        return _qsa_attention_plugin_translation(*common)
    return _qsa_attention_plugin_spec_translation(*common, tree_parent_ids,
                                                  tree_depths,
                                                  spec_verify_layout)


# ---------------------------------------------------------------------------
# FP8 ops
# ---------------------------------------------------------------------------


@script()
def _ple_pool_lookup_translation(
    pool: OnnxUnion[onnxscript.FLOAT8E4M3FN, onnxscript.FLOAT16,
                    onnxscript.BFLOAT16],
    row_map: onnxscript.INT32,
) -> OnnxUnion[onnxscript.FLOAT8E4M3FN, onnxscript.FLOAT16,
               onnxscript.BFLOAT16]:
    return _trt_edgellm.PleGatherPlugin(pool, row_map, plugin_version="1")


@script()
def _ple_pool_gather_translation(
    pool: onnxscript.FLOAT8E4M3FN,
    row_map: onnxscript.INT32,
    scale: OnnxUnion[onnxscript.FLOAT, onnxscript.FLOAT16,
                     onnxscript.BFLOAT16],
) -> OnnxUnion[onnxscript.FLOAT, onnxscript.FLOAT16, onnxscript.BFLOAT16]:
    """Keep the cache pool compressed; only selected rows enter DQ."""
    rows = _trt_edgellm.PleGatherPlugin(pool, row_map, plugin_version="1")
    embeddings = _op21.DequantizeLinear(rows, scale)
    shape = _op21.Shape(row_map, end=-1)
    output_shape = _op21.Concat(shape, _op21.Constant(value_ints=[-1]), axis=0)
    return _op21.Reshape(embeddings, output_shape)


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
def _nvfp4_act_qdq_moe_translation(
    hidden_states: onnxscript.FLOAT16,
    dblq_scale: onnxscript.FLOAT,
) -> onnxscript.FLOAT16:
    """DynQ + scale DQ + explicit-block-shape DQ for the W4A4 TRT_MoE input.

    The reference mode-2 graph form::

        TRT_FP4DynamicQuantize(x, dblq_f32, axis=-1, block_size=16, scale_type=17)
            -> (x_f4, sx_f8)
        DequantizeLinear(sx_f8, Cast(dblq_f32, fp16))          [standard op]
            -> scale_f16
        TRT_BlockDequantize(x_f4, scale_f16, block_shape=[1, 1, 16],
                            output_dtype=fp16)
            -> x_dq

    Differs from :func:`_nvfp4_act_qdq_translation` in the last node only:
    the MoE lowering takes the hidden-state block shape from an
    ``IDequantizeLayer::setBlockShape`` (``TRT_BlockDequantize``), which the
    standard-style ``trt::DequantizeLinear(block_size)`` never calls, and the
    fp16 output type is what the ``TRT_MoE`` input wants. The fp16 scale copy
    must equal the fp32 one, so the model rounds ``dblq_scale`` to an
    fp16-representable value before registering it.
    """
    x_f4, sx_f8 = _trt.TRT_FP4DynamicQuantize(
        hidden_states,
        dblq_scale,
        axis=-1,
        block_size=16,
        scale_type=17,
        _outputs=2,
    )
    dblq_f16 = _op21.Cast(dblq_scale, to=10)  # 10 = FLOAT16
    scale_f16 = _op21.DequantizeLinear(sx_f8, dblq_f16)
    return _trt.TRT_BlockDequantize(
        x_f4,
        scale_f16,
        block_shape=[1, 1, 16],
        output_dtype=10,  # 10 = ONNX TensorProto.FLOAT16
    )


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


@script()
def _int4_groupwise_gemm_v2_translation(
    hidden_states: onnxscript.FLOAT16,
    qweight: onnxscript.INT8,
    scales: onnxscript.FLOAT16,
    gemm_n: int,
    gemm_k: int,
    group_size: int,
) -> onnxscript.FLOAT16:
    return _trt_edgellm.Int4GroupwiseGemmPluginV2(
        hidden_states,
        qweight,
        scales,
        gemm_n=gemm_n,
        gemm_k=gemm_k,
        group_size=group_size,
    )


# ---------------------------------------------------------------------------
# QKV packing op
# ---------------------------------------------------------------------------


@script()
def _q8_0_linear_translation(
    hidden: OnnxUnion[onnxscript.FLOAT16, onnxscript.BFLOAT16],
    packed_weight: onnxscript.INT8,
    out_dtype: int,
) -> OnnxUnion[onnxscript.FLOAT, onnxscript.FLOAT16, onnxscript.BFLOAT16]:
    return _trt_edgellm.Q8_0LinearPlugin(hidden,
                                         packed_weight,
                                         out_dtype=out_dtype)


@script()
def _q8_0_qkv_translation(
    hidden: OnnxUnion[onnxscript.FLOAT16, onnxscript.BFLOAT16],
    q_weight: onnxscript.INT8,
    k_weight: onnxscript.INT8,
    v_weight: onnxscript.INT8,
    out_dtype: int,
) -> OnnxUnion[onnxscript.FLOAT, onnxscript.FLOAT16, onnxscript.BFLOAT16]:
    return _trt_edgellm.Q8_0LinearPlugin(hidden,
                                         q_weight,
                                         k_weight,
                                         v_weight,
                                         out_dtype=out_dtype)


@script()
def _w2a16_gemm_translation(
    hidden: onnxscript.BFLOAT16,
    values: onnxscript.INT8,
    bases: onnxscript.FLOAT16,
    lookup: onnxscript.BFLOAT16,
    gemm_n: int,
    gemm_k: int,
) -> onnxscript.BFLOAT16:
    return _trt_edgellm.W2A16GemmPlugin(hidden,
                                        values,
                                        bases,
                                        lookup,
                                        gemm_n=gemm_n,
                                        gemm_k=gemm_k)


@script()
def _qkv_concat_translation(
    q: onnxscript.FLOAT16,
    k: onnxscript.FLOAT16,
    v: onnxscript.FLOAT16,
) -> onnxscript.FLOAT16:
    return _trt_edgellm.QkvConcatPlugin(
        q,
        k,
        v,
    )


# ---------------------------------------------------------------------------
# INT8 SmoothQuant ops
# ---------------------------------------------------------------------------


@script()
def _nvfp4_a16_gemm_translation(
    activation: onnxscript.FLOAT16,
    qweights: onnxscript.INT8,
    block_scales: onnxscript.INT8,
    global_scale: onnxscript.FLOAT16,
    gemm_n: int,
    gemm_k: int,
) -> onnxscript.FLOAT16:
    return _trt_edgellm.Nvfp4A16GemmPlugin(
        activation,
        qweights,
        block_scales,
        global_scale,
        gemm_n=gemm_n,
        gemm_k=gemm_k,
        max_m=0,
    )


_NVFP4_A16_BLACKWELL_IO_T = OnnxUnion[onnxscript.FLOAT16, onnxscript.BFLOAT16]


@script()
def _nvfp4_a16_blackwell_gemm_translation(
    activation: _NVFP4_A16_BLACKWELL_IO_T,
    qweights: onnxscript.INT8,
    block_scales: onnxscript.INT8,
    global_scale: onnxscript.FLOAT,
    gemm_n: int,
    gemm_k: int,
) -> _NVFP4_A16_BLACKWELL_IO_T:
    """Emit the dedicated SM110 dense NVFP4-A16 plugin contract."""
    return _trt_edgellm.Nvfp4A16BlackwellGemmPlugin(
        activation,
        qweights,
        block_scales,
        global_scale,
        gemm_n=gemm_n,
        gemm_k=gemm_k,
        max_m=0,
        layout=1,
        backend=0,
    )


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
def _causal_conv1d_ragged_translation(
    hidden_states: onnxscript.FLOAT16,
    weight: onnxscript.FLOAT16,
    bias: onnxscript.FLOAT16,
    conv_state: onnxscript.FLOAT16,
    query_lengths: onnxscript.INT32,
    query_start_offsets: onnxscript.INT32,
    state_indices: onnxscript.INT32,
    execution_phase_marker: onnxscript.INT32,
    context_sequence_count_carrier: onnxscript.INT32,
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
        query_lengths,
        query_start_offsets,
        state_indices,
        execution_phase_marker,
        context_sequence_count_carrier,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_spec_verify_state=0,
        plugin_version="1",
        _outputs=2,
    )
    return output, conv_state_out, _op21.Identity(conv_state_out)


@script()
def _causal_conv1d_with_intermediate_translation(
    hidden_states: onnxscript.FLOAT16,
    weight: onnxscript.FLOAT16,
    bias: onnxscript.FLOAT16,
    conv_state: onnxscript.FLOAT16,
    context_lengths: onnxscript.INT32,
    query_start_offsets: onnxscript.INT32,
    state_indices: onnxscript.INT32,
    execution_phase_marker: onnxscript.INT32,
    context_sequence_count_carrier: onnxscript.INT32,
    tree_parent_ids: onnxscript.INT32,
    tree_depths: onnxscript.INT32,
    spec_verify_layout: onnxscript.INT32,
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
        query_start_offsets,
        state_indices,
        execution_phase_marker,
        context_sequence_count_carrier,
        tree_parent_ids,
        tree_depths,
        spec_verify_layout,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_spec_verify_state=1,
        plugin_version="1",
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
    query_start_offsets,
    state_indices,
    execution_phase_marker,
    context_sequence_count_carrier,
):
    return _causal_conv1d_ragged_translation(
        hidden_states, weight, bias, conv_state, context_lengths,
        query_start_offsets, state_indices, execution_phase_marker,
        context_sequence_count_carrier, stride, padding, dilation, groups)


def _causal_conv1d_intermediate_dispatch(
    hidden_states,
    weight,
    bias,
    conv_state,
    context_lengths,
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
    spec_verify_layout=None,
):
    if tree_parent_ids is None or tree_depths is None or spec_verify_layout is None:
        raise ValueError(
            "Verify state requires parent IDs, depths and spec_verify_layout")
    return _causal_conv1d_with_intermediate_translation(
        hidden_states, weight, bias, conv_state, context_lengths,
        query_start_offsets, state_indices, execution_phase_marker,
        context_sequence_count_carrier, tree_parent_ids, tree_depths,
        spec_verify_layout, stride, padding, dilation, groups)


@script()
def _gated_delta_net_ragged_translation(
    q: onnxscript.FLOAT16,
    k: onnxscript.FLOAT16,
    v: onnxscript.FLOAT16,
    a: onnxscript.FLOAT16,
    b: onnxscript.FLOAT16,
    A_log: onnxscript.FLOAT,
    dt_bias: onnxscript.FLOAT16,
    h0_source: onnxscript.FLOAT,
    query_lengths: onnxscript.INT32,
    query_start_offsets: onnxscript.INT32,
    state_indices: onnxscript.INT32,
    execution_phase_marker: onnxscript.INT32,
    context_sequence_count_carrier: onnxscript.INT32,
    k_dim: int,
    v_dim: int,
    use_diffusion_state: int,
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
        query_lengths,
        query_start_offsets,
        state_indices,
        execution_phase_marker,
        context_sequence_count_carrier,
        k_dim=k_dim,
        v_dim=v_dim,
        use_diffusion_state=use_diffusion_state,
        use_spec_verify_state=0,
        plugin_version="1",
        _outputs=2,
    )
    return output, h0_out, _op21.Identity(h0_out)


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
    query_start_offsets: onnxscript.INT32,
    state_indices: onnxscript.INT32,
    execution_phase_marker: onnxscript.INT32,
    context_sequence_count_carrier: onnxscript.INT32,
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
        query_start_offsets,
        state_indices,
        execution_phase_marker,
        context_sequence_count_carrier,
        tree_parent_ids,
        tree_depths,
        k_dim=k_dim,
        v_dim=v_dim,
        use_spec_verify_state=1,
        plugin_version="1",
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
    query_start_offsets,
    state_indices,
    execution_phase_marker,
    context_sequence_count_carrier,
    use_diffusion_state=False,
):
    return _gated_delta_net_ragged_translation(
        q, k, v, a, b, A_log, dt_bias, h0_source, context_lengths,
        query_start_offsets, state_indices, execution_phase_marker,
        context_sequence_count_carrier, k_dim, v_dim, int(use_diffusion_state))


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
    query_start_offsets,
    state_indices,
    k_dim,
    v_dim,
    execution_phase_marker,
    context_sequence_count_carrier,
    tree_parent_ids=None,
    tree_depths=None,
):
    if tree_parent_ids is None or tree_depths is None:
        raise ValueError("Verify state requires parent IDs and depths")
    return _gated_delta_net_with_intermediate_translation(
        q, k, v, a, b, A_log, dt_bias, h0_source, context_lengths,
        query_start_offsets, state_indices, execution_phase_marker,
        context_sequence_count_carrier, tree_parent_ids, tree_depths, k_dim,
        v_dim)


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
    query_lengths: onnxscript.INT32,
    query_start_offsets: onnxscript.INT32,
    state_indices: onnxscript.INT32,
    execution_phase_marker: onnxscript.INT32,
    context_sequence_count_carrier: onnxscript.INT32,
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
        query_lengths,
        query_start_offsets,
        state_indices,
        execution_phase_marker,
        context_sequence_count_carrier,
        dt_softplus=dt_softplus,
        ngroups=ngroups,
        chunk_size=chunk_size,
        use_spec_verify_state=0,
        plugin_version="1",
        _outputs=2,
    )
    return output, state_out


@script()
def _update_ssm_state_with_intermediate_translation(
    hidden_states: onnxscript.FLOAT16,
    ssm_a: onnxscript.FLOAT,
    ssm_b: onnxscript.FLOAT16,
    ssm_c: onnxscript.FLOAT16,
    ssm_d: onnxscript.FLOAT16,
    dt: onnxscript.FLOAT16,
    dt_bias: onnxscript.FLOAT16,
    state: onnxscript.FLOAT16,
    query_lengths: onnxscript.INT32,
    query_start_offsets: onnxscript.INT32,
    state_indices: onnxscript.INT32,
    execution_phase_marker: onnxscript.INT32,
    context_sequence_count_carrier: onnxscript.INT32,
    tree_parent_ids: onnxscript.INT32,
    tree_depths: onnxscript.INT32,
    spec_verify_layout: onnxscript.INT32,
    dt_softplus: int,
    ngroups: int,
    chunk_size: int = 0,
) -> tuple[onnxscript.FLOAT16, onnxscript.FLOAT16, onnxscript.FLOAT,
           onnxscript.FLOAT, onnxscript.FLOAT, onnxscript.FLOAT]:
    output, state_out, replay_da, replay_u, replay_b, replay_dt = _trt_edgellm.update_ssm_state(
        hidden_states,
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
        tree_parent_ids,
        tree_depths,
        spec_verify_layout,
        dt_softplus=dt_softplus,
        ngroups=ngroups,
        chunk_size=chunk_size,
        use_spec_verify_state=1,
        plugin_version="1",
        _outputs=6,
    )
    return output, state_out, replay_da, replay_u, replay_b, replay_dt


def _update_ssm_state_with_intermediate_dispatch(
    hidden_states,
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
    spec_verify_layout=None,
):
    if tree_parent_ids is None or tree_depths is None or spec_verify_layout is None:
        raise ValueError(
            "Verify state requires parent IDs, depths and spec_verify_layout")
    return _update_ssm_state_with_intermediate_translation(
        hidden_states, ssm_a, ssm_b, ssm_c, ssm_d, dt, dt_bias, state,
        query_lengths, query_start_offsets, state_indices,
        execution_phase_marker, context_sequence_count_carrier,
        tree_parent_ids, tree_depths, spec_verify_layout, dt_softplus, ngroups,
        chunk_size)


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

_QKV_T = OnnxUnion[onnxscript.FLOAT16, onnxscript.FLOAT8E4M3FN]


@script()
def _vit_attention_plugin_translation(
    query_states: _QKV_T,
    key_states: _QKV_T,
    value_states: _QKV_T,
    cu_seqlens: onnxscript.INT32,
    max_seqlen_carrier: onnxscript.INT32,
    num_heads: int,
    head_size: int,
    attention_scale: float,
    qkv_scales: Sequence[float],
) -> onnxscript.FLOAT16:
    """ViT ragged self-attention without KV cache.

    Q/K/V are FLOAT16 (FP16 MHA) or FLOAT8E4M3FN (FP8 MHA via preceding
    QuantizeLinear nodes fused by TRT with upstream RoPE/V-path ops).
    """
    return _trt_edgellm.ViTAttentionPlugin(
        query_states,
        key_states,
        value_states,
        cu_seqlens,
        max_seqlen_carrier,
        num_heads=num_heads,
        head_size=head_size,
        attention_scale=attention_scale,
        qkv_scales=qkv_scales,
    )


def _vit_attention_plugin_dispatch(
    query_states,
    key_states,
    value_states,
    cu_seqlens,
    max_seqlen_carrier,
    num_heads: int,
    head_size: int,
    attention_scale: float,
    qkv_scales=None,
):
    # FP16-MHA callers omit qkv_scales (torch-op default None); the ONNX
    # attribute is FLOATS, so normalize to the identity scales here.
    if qkv_scales is None:
        qkv_scales = [1.0, 1.0, 1.0]
    return _vit_attention_plugin_translation(query_states, key_states,
                                             value_states, cu_seqlens,
                                             max_seqlen_carrier, num_heads,
                                             head_size, attention_scale,
                                             qkv_scales)


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
def _nvfp4_a16_moe_plugin_translation(
    router_logits: onnxscript.FLOAT,
    hidden_states: onnxscript.FLOAT16,
    fc1_qweights: onnxscript.INT8,
    fc1_block_scales: onnxscript.INT8,
    fc1_global_scales: onnxscript.FLOAT16,
    fc2_qweights: onnxscript.INT8,
    fc2_block_scales: onnxscript.INT8,
    fc2_global_scales: onnxscript.FLOAT16,
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
    max_routed_rows: int,
) -> onnxscript.FLOAT16:
    return _trt_edgellm.Nvfp4A16MoePlugin(
        router_logits,
        hidden_states,
        fc1_qweights,
        fc1_block_scales,
        fc1_global_scales,
        fc2_qweights,
        fc2_block_scales,
        fc2_global_scales,
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
        max_routed_rows=max_routed_rows,
    )


@script()
def _nvfp4_a16_blackwell_moe_plugin_translation(
    router_logits: onnxscript.FLOAT,
    hidden_states: onnxscript.FLOAT16,
    fc1_qweights: onnxscript.INT8,
    fc1_block_scales: onnxscript.INT8,
    fc1_global_scales: onnxscript.FLOAT,
    fc2_qweights: onnxscript.INT8,
    fc2_block_scales: onnxscript.INT8,
    fc2_global_scales: onnxscript.FLOAT,
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
    max_routed_rows: int,
    backend: int,
) -> onnxscript.FLOAT16:
    """Emit the dedicated SM110 routed NVFP4-A16 MoE plugin contract."""
    return _trt_edgellm.Nvfp4A16BlackwellMoePlugin(
        router_logits,
        hidden_states,
        fc1_qweights,
        fc1_block_scales,
        fc1_global_scales,
        fc2_qweights,
        fc2_block_scales,
        fc2_global_scales,
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
        max_routed_rows=max_routed_rows,
        layout=1,
        backend=backend,
    )


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


@script()
def _fp16_moe_plugin_translation(
    router_logits: onnxscript.FLOAT,
    hidden_states: onnxscript.FLOAT16,
    fc1_weights: onnxscript.FLOAT16,
    fc2_weights: onnxscript.FLOAT16,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    norm_topk_prob: int,
    max_routed_rows: int,
) -> onnxscript.FLOAT16:
    output = _trt_edgellm.Fp16MoePlugin(
        router_logits,
        hidden_states,
        fc1_weights,
        fc2_weights,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        moe_inter_size=moe_inter_size,
        activation_type=activation_type,
        norm_topk_prob=norm_topk_prob,
        max_routed_rows=max_routed_rows,
    )
    return output


@script()
def _nvfp4_dequantize_moe_translation(
    weight: onnxscript.INT8,
    block_scale: onnxscript.FLOAT16,
    group_size: int,
) -> onnxscript.FLOAT16:
    """Single blocked dequantize for stacked NVFP4 MoE expert weights.

    The weight initializer is rewritten INT8 -> FLOAT4E2M1 post-export by
    ``export.py`` (same fixup the dense NVFP4 path relies on).

    Must be ``TRT_BlockDequantize``, NOT the standard ``DequantizeLinear`` the
    dense linears use. Only TRT_BlockDequantize calls
    ``IDequantizeLayer::setBlockShape``; ``block_size``/``axis`` on the standard
    op leaves the block shape EMPTY and Myelin rejects the MoE ("Weights block
    shape is required for block qdq of weights").
    block_shape is [1, 1, 16]: NVFP4 groups run along the last (K) axis and the
    group size is fixed at 16.
    """
    return _trt.TRT_BlockDequantize(
        weight,
        block_scale,
        block_shape=[1, 1, 16],
        output_dtype=10,  # 10 = ONNX TensorProto.FLOAT16
    )


@script()
def _nvfp4_dequantize_moe_with_global_scale_translation(
    weight: onnxscript.INT8,
    block_scale: onnxscript.FLOAT8E4M3FN,
    global_scale: onnxscript.FLOAT16,
    group_size: int,
) -> onnxscript.FLOAT16:
    """Dequantize stacked NVFP4 MoE expert weights with their global scales.

    ``DequantizeLinear(block_scale_fp8, global_scale_fp16, axis=0)`` applies
    the per-expert global along the leading expert axis of
    ``[E, out, in // 16]`` and outputs fp16 (its scale's type); the
    ``TRT_BlockDequantize`` then unpacks the FP4 codes with those fp16 block
    scales, block_shape ``[1, 1, 16]`` as in
    :func:`_nvfp4_dequantize_moe_translation` (K is the last axis here; the
    model transposes to TRT_MoE's ``[E, in, out]`` afterwards). The weight
    initializer is rewritten INT8 -> FLOAT4E2M1 post-export by name; the fp8
    block-scale and fp16 global initializers are left alone (their names
    contain "scale").
    """
    scales_f16 = _op21.DequantizeLinear(block_scale, global_scale, axis=0)
    return _trt.TRT_BlockDequantize(
        weight,
        scales_f16,
        block_shape=[1, 1, 16],
        output_dtype=10,  # 10 = ONNX TensorProto.FLOAT16
    )


@script()
def _trt_moe_w4a8_translation(
    hidden_states: onnxscript.FLOAT16,
    selected_experts: onnxscript.INT32,
    scores: onnxscript.FLOAT,
    fc_gate_weights: onnxscript.FLOAT16,
    fc_up_weights: onnxscript.FLOAT16,
    fc_down_weights: onnxscript.FLOAT16,
    fc_gate_biases: onnxscript.FLOAT16,
    fc_up_biases: onnxscript.FLOAT16,
    fc_down_biases: onnxscript.FLOAT16,
    fc_down_activation_scale: onnxscript.FLOAT,
    activation_type: int,
    quantization_mode: int,
    quantization_dtype: int,
    swiglu_alpha: float,
    swiglu_beta: float,
) -> onnxscript.FLOAT16:
    """W4A8 TRT-native MoE -> builtin ``TRT_MoE`` with 10 inputs.

    The activation scale has to sit at input index 9, so the three bias inputs
    are passed explicitly (as zeros) rather than as empty optionals.
    No ``swiglu_limit`` attribute on purpose -- see :func:`_trt_moe_fp16_translation`.
    """
    output = _trt.TRT_MoE(
        hidden_states,
        selected_experts,
        scores,
        fc_gate_weights,
        fc_up_weights,
        fc_down_weights,
        fc_gate_biases,
        fc_up_biases,
        fc_down_biases,
        fc_down_activation_scale,
        activation_type=activation_type,
        quantization_mode=quantization_mode,
        quantization_dtype=quantization_dtype,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )
    return output


@script()
def _trt_moe_w4a4_translation(
    hidden_states: onnxscript.FLOAT16,
    selected_experts: onnxscript.INT32,
    scores: onnxscript.FLOAT,
    fc_gate_weights: onnxscript.FLOAT16,
    fc_up_weights: onnxscript.FLOAT16,
    fc_down_weights: onnxscript.FLOAT16,
    fc_gate_biases: onnxscript.FLOAT16,
    fc_up_biases: onnxscript.FLOAT16,
    fc_down_biases: onnxscript.FLOAT16,
    fc_down_activation_scale: onnxscript.FLOAT,
    activation_type: int,
    quantization_mode: int,
    quantization_dtype: int,
    dyn_q_output_scale_dtype: int,
    swiglu_alpha: float,
    swiglu_beta: float,
) -> onnxscript.FLOAT16:
    """W4A4 TRT-native MoE -> builtin ``TRT_MoE`` with 10 inputs, mode 2.

    :func:`_trt_moe_w4a8_translation` plus the two attributes the importer
    requires for quantization_mode 2: ``quantization_block_shape`` (the
    down-projection input is quantized in blocks of 16 along
    moe_intermediate_size, over (batch, seq, topK, moeInterSize)) and
    ``dyn_q_output_scale_dtype`` (the raw TensorRT DataType of the block
    scales, kFP8). Input 9 is the per-expert second-level scale of those
    block scales. Still no ``swiglu_limit`` -- see
    :func:`_trt_moe_fp16_translation`.
    """
    output = _trt.TRT_MoE(
        hidden_states,
        selected_experts,
        scores,
        fc_gate_weights,
        fc_up_weights,
        fc_down_weights,
        fc_gate_biases,
        fc_up_biases,
        fc_down_biases,
        fc_down_activation_scale,
        activation_type=activation_type,
        quantization_mode=quantization_mode,
        quantization_dtype=quantization_dtype,
        quantization_block_shape=[1, 1, 1, 16],
        dyn_q_output_scale_dtype=dyn_q_output_scale_dtype,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )
    return output


@script()
def _trt_moe_fp16_translation(
    hidden_states: onnxscript.FLOAT16,
    selected_experts: onnxscript.INT32,
    scores: onnxscript.FLOAT,
    fc_gate_weights: onnxscript.FLOAT16,
    fc_up_weights: onnxscript.FLOAT16,
    fc_down_weights: onnxscript.FLOAT16,
    activation_type: int,
    quantization_mode: int,
    swiglu_alpha: float,
    swiglu_beta: float,
) -> onnxscript.FLOAT16:
    """FP16 TRT-native MoE -> builtin ``TRT_MoE`` node in domain ``trt``.

    ``scores`` is FLOAT (fp32) on purpose even though the activations are FP16:
    fp16 scores fail inside Myelin codegen. See the note on ``trt::trt_moe_fp16``.

    ``swiglu_limit`` is intentionally NOT emitted. TensorRT's TRT_MoE importer
    honours the attribute as a clamp whenever it is present and only defaults
    it to +inf (no clamp) when absent; emitting 0.0 makes Myelin >= 2.21.18
    clamp the up-projection to [-0, 0] and the layer outputs exact zeros (seen
    on B200 TRT 11.3.0.89; hidden on Thor TRT 11.3.0.7 / Myelin 2.21.5, which
    ignored the limit).
    """
    output = _trt.TRT_MoE(
        hidden_states,
        selected_experts,
        scores,
        fc_gate_weights,
        fc_up_weights,
        fc_down_weights,
        activation_type=activation_type,
        quantization_mode=quantization_mode,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )
    return output


@script()
def _trt_moe_fp16_nongated_translation(
    hidden_states: onnxscript.FLOAT16,
    selected_experts: onnxscript.INT32,
    scores: onnxscript.FLOAT,
    fc_up_weights: onnxscript.FLOAT16,
    fc_down_weights: onnxscript.FLOAT16,
    activation_type: int,
    quantization_mode: int,
) -> onnxscript.FLOAT16:
    """Non-gated FP16 TRT-native MoE -> builtin ``TRT_MoE`` with an EMPTY input 3.

    An empty ``fc_gate_weights`` name selects ``IMoELayer::setWeights`` (no
    gate, no mul): ``expert = fcDown(act(fcUp(x)))``, the form Nemotron-H's
    ReLU^2 experts need (``activation_type`` 2 = kRELU2 is only defined here).
    No biases (rejected for non-gated experts) and no SwiGLU attributes.
    """
    output = _trt.TRT_MoE(
        hidden_states,
        selected_experts,
        scores,
        None,
        fc_up_weights,
        fc_down_weights,
        activation_type=activation_type,
        quantization_mode=quantization_mode,
    )
    return output


@script()
def _trt_moe_w4a8_nongated_translation(
    hidden_states: onnxscript.FLOAT16,
    selected_experts: onnxscript.INT32,
    scores: onnxscript.FLOAT,
    fc_up_weights: onnxscript.FLOAT16,
    fc_down_weights: onnxscript.FLOAT16,
    fc_down_activation_scale: onnxscript.FLOAT,
    activation_type: int,
    quantization_mode: int,
    quantization_dtype: int,
) -> onnxscript.FLOAT16:
    """Non-gated W4A8 TRT-native MoE -> builtin ``TRT_MoE`` with 10 inputs.

    Inputs 3 (gate weights) and 6-8 (biases) are empty: biases are not
    supported for non-gated experts, and the activation scale must still sit
    at input index 9. See :func:`_trt_moe_fp16_nongated_translation`.
    """
    output = _trt.TRT_MoE(
        hidden_states,
        selected_experts,
        scores,
        None,
        fc_up_weights,
        fc_down_weights,
        None,
        None,
        None,
        fc_down_activation_scale,
        activation_type=activation_type,
        quantization_mode=quantization_mode,
        quantization_dtype=quantization_dtype,
    )
    return output


@script()
def _trt_moe_w4a4_nongated_translation(
    hidden_states: onnxscript.FLOAT16,
    selected_experts: onnxscript.INT32,
    scores: onnxscript.FLOAT,
    fc_up_weights: onnxscript.FLOAT16,
    fc_down_weights: onnxscript.FLOAT16,
    fc_down_activation_scale: onnxscript.FLOAT,
    activation_type: int,
    quantization_mode: int,
    quantization_dtype: int,
    dyn_q_output_scale_dtype: int,
) -> onnxscript.FLOAT16:
    """Non-gated W4A4 TRT-native MoE -> builtin ``TRT_MoE`` with 10 inputs.

    :func:`_trt_moe_w4a8_nongated_translation` (inputs 3 and 6-8 empty, the
    scale at index 9) with the mode-2 attributes of
    :func:`_trt_moe_w4a4_translation`; no SwiGLU attributes.
    """
    output = _trt.TRT_MoE(
        hidden_states,
        selected_experts,
        scores,
        None,
        fc_up_weights,
        fc_down_weights,
        None,
        None,
        None,
        fc_down_activation_scale,
        activation_type=activation_type,
        quantization_mode=quantization_mode,
        quantization_dtype=quantization_dtype,
        quantization_block_shape=[1, 1, 1, 16],
        dyn_q_output_scale_dtype=dyn_q_output_scale_dtype,
    )
    return output


@script()
def _fp16_moe_plugin_sigmoid_translation(
    router_logits: onnxscript.FLOAT,
    hidden_states: onnxscript.FLOAT16,
    fc1_weights: onnxscript.FLOAT16,
    fc2_weights: onnxscript.FLOAT16,
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
    max_routed_rows: int,
) -> onnxscript.FLOAT16:
    output = _trt_edgellm.Fp16MoePlugin(
        router_logits,
        hidden_states,
        fc1_weights,
        fc2_weights,
        e_score_correction_bias,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        moe_inter_size=moe_inter_size,
        activation_type=activation_type,
        norm_topk_prob=norm_topk_prob,
        max_routed_rows=max_routed_rows,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
        routing_mode=1,
    )
    return output


# ---------------------------------------------------------------------------
# AllReducePlugin
# ---------------------------------------------------------------------------


@script()
def _all_reduce_translation(
    hidden_states: onnxscript.FLOAT16,
    tp_size: int,
) -> onnxscript.FLOAT16:
    return _trt_edgellm.AllReducePlugin(hidden_states, tp_size=tp_size)


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
    raw_activation_scale: bool = False,
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
    if raw_activation_scale:
        activation_scale = sx_f8
        gemm_scale = _op21.Mul(global_scale, weight_f32_scale)
    else:
        activation_scale = _trt.DequantizeLinear(sx_f8, global_scale)
        gemm_scale = weight_f32_scale
    return _trt_edgellm.FusedNvfp4GemmAllReducePlugin(
        x_f4,
        activation_scale,
        weight_f4,
        weight_f8_scale,
        gemm_scale,
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
    token_aligned_rope_cos_sin: onnxscript.FLOAT,
    delta_positions: onnxscript.INT32,
    delta_token_to_sequence: onnxscript.INT32,
    kv_page_table: onnxscript.INT32,
) -> onnxscript.FLOAT16:
    """DFlash target KV cache update: apply RoPE to k_delta, write k+v into cache."""
    present_kv = _trt_edgellm.DFlashTargetKVCacheUpdate(
        k_delta,
        v_delta,
        past_key_value,
        token_aligned_rope_cos_sin,
        delta_positions,
        delta_token_to_sequence,
        kv_page_table,
    )
    return present_kv


@script()
def _dflash2_grouped_dynamic_conv_translation(
    hidden_states: OnnxUnion[onnxscript.FLOAT16, onnxscript.BFLOAT16],
    delta: OnnxUnion[onnxscript.FLOAT16, onnxscript.BFLOAT16],
    base_kernel: OnnxUnion[onnxscript.FLOAT16, onnxscript.BFLOAT16],
    residual: onnxscript.FLOAT = None,
    block_size: int = 8,
    kernel_size: int = 2,
    group_size: int = 16,
    fuse_residual: int = 0,
) -> OnnxUnion[onnxscript.FLOAT16, onnxscript.BFLOAT16, onnxscript.FLOAT]:
    return _trt_edgellm.DFlash2GroupedDynamicConvPlugin(
        hidden_states,
        delta,
        base_kernel,
        residual,
        block_size=block_size,
        kernel_size=kernel_size,
        group_size=group_size,
        fuse_residual=fuse_residual,
    )


@script()
def _low_rank_candidate_head_translation(
    hidden: onnxscript.FLOAT16,
    down_proj: onnxscript.FLOAT16,
    up_proj: onnxscript.FLOAT16,
    lm_head: onnxscript.FLOAT16,
    num_candidates: int,
) -> tuple[onnxscript.INT32, onnxscript.FLOAT]:
    support_ids, support_log_probs = _trt_edgellm.LowRankCandidateHeadPlugin(
        hidden,
        down_proj,
        up_proj,
        lm_head,
        num_candidates=num_candidates,
        _outputs=2,
    )
    return support_ids, support_log_probs


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
        _attention_plugin_dispatch,
        torch.ops.trt.qsa_attention_plugin.default:
        _qsa_attention_plugin_dispatch,
        torch.ops.trt.ple_pool_lookup.default:
        _ple_pool_lookup_translation,
        torch.ops.trt.ple_pool_gather.default:
        _ple_pool_gather_translation,
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
        torch.ops.trt.int4_groupwise_gemm_v2.default:
        _int4_groupwise_gemm_v2_translation,
        torch.ops.trt.qkv_concat.default:
        _qkv_concat_translation,
        torch.ops.trt.q8_0_linear_plugin.default:
        _q8_0_linear_translation,
        torch.ops.trt.q8_0_qkv_plugin.default:
        _q8_0_qkv_translation,
        torch.ops.trt.w2a16_gemm_plugin.default:
        _w2a16_gemm_translation,
        torch.ops.trt.nvfp4_a16_gemm.default:
        _nvfp4_a16_gemm_translation,
        torch.ops.trt.nvfp4_a16_blackwell_gemm.default:
        _nvfp4_a16_blackwell_gemm_translation,
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
        torch.ops.trt_edgellm.update_ssm_state_with_intermediate.default:
        _update_ssm_state_with_intermediate_dispatch,
        torch.ops.trt_edgellm.gated_delta_net.default:
        _gated_delta_net_dispatch,
        torch.ops.trt_edgellm.gated_delta_net_with_intermediate.default:
        _gated_delta_net_intermediate_dispatch,
        torch.ops.trt.vit_attention_plugin.default:
        _vit_attention_plugin_dispatch,
        torch.ops.trt.trt_ragged_attention.default:
        _trt_ragged_attention_translation,
        torch.ops.trt.gather_nd.default:
        _gather_nd_translation,
        torch.ops.trt_edgellm.int4_moe_plugin.default:
        _int4_moe_plugin_translation,
        torch.ops.trt_edgellm.Nvfp4MoePlugin.default:
        _nvfp4_moe_plugin_translation,
        torch.ops.trt_edgellm.Nvfp4A16MoePlugin.default:
        _nvfp4_a16_moe_plugin_translation,
        torch.ops.trt_edgellm.Nvfp4A16BlackwellMoePlugin.default:
        _nvfp4_a16_blackwell_moe_plugin_translation,
        torch.ops.trt_edgellm.NvFP4MoEPluginGeforce.default:
        _nvfp4_moe_plugin_geforce_translation,
        torch.ops.trt_edgellm.Fp16MoePlugin.default:
        _fp16_moe_plugin_translation,
        torch.ops.trt_edgellm.Fp16MoePluginSigmoid.default:
        _fp16_moe_plugin_sigmoid_translation,
        torch.ops.trt_edgellm.W2A16MoePlugin.default:
        _w2a16_moe_plugin_translation,
        # TRT-NATIVE MoE (OOTB arm; builtin TRT_MoE -> IMoELayer -> Myelin moe_op)
        torch.ops.trt.trt_moe_fp16.default:
        _trt_moe_fp16_translation,
        torch.ops.trt.trt_moe_w4a8.default:
        _trt_moe_w4a8_translation,
        torch.ops.trt.trt_moe_fp16_nongated.default:
        _trt_moe_fp16_nongated_translation,
        torch.ops.trt.trt_moe_w4a8_nongated.default:
        _trt_moe_w4a8_nongated_translation,
        torch.ops.trt.nvfp4_dequantize_moe.default:
        _nvfp4_dequantize_moe_translation,
        # W4A4 form (TRT_MoE quantization_mode 2, --trt-native-moe-w4a4)
        torch.ops.trt.trt_moe_w4a4.default:
        _trt_moe_w4a4_translation,
        torch.ops.trt.trt_moe_w4a4_nongated.default:
        _trt_moe_w4a4_nongated_translation,
        torch.ops.trt.nvfp4_dequantize_moe_with_global_scale.default:
        _nvfp4_dequantize_moe_with_global_scale_translation,
        torch.ops.trt.nvfp4_act_qdq_moe.default:
        _nvfp4_act_qdq_moe_translation,
        torch.ops.trt_edgellm.dflash_target_kv_cache_update.default:
        _dflash_target_kv_cache_update_translation,
        torch.ops.trt_edgellm.dflash2_grouped_dynamic_conv.default:
        _dflash2_grouped_dynamic_conv_translation,
        torch.ops.trt_edgellm.low_rank_candidate_head.default:
        _low_rank_candidate_head_translation,
        # TRT native attention ops (used by Alpamayo)
        torch.ops.trt.rope_onnx.default:
        _rope_onnx_translation,
        torch.ops.trt.kv_cache_update_onnx.default:
        _kv_cache_update_onnx_translation,
        torch.ops.trt.attention_onnx.default:
        _attention_onnx_translation,
        torch.ops.trt_edgellm.all_reduce.default:
        _all_reduce_translation,
        torch.ops.trt_edgellm.fused_nvfp4_gemm_allreduce.default:
        _fused_nvfp4_gemm_allreduce_translation,
        torch.ops.trt_edgellm.gemma4_audio_attention_plugin.default:
        _gemma4_audio_attention_plugin_translation,
    }
