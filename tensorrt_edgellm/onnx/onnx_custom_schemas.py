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
Register ONNX OpSchemas for custom ops used by dynamo export in this package.


Call :func:`register_tensorrt_edgellm_onnx_custom_schemas` before
``torch.onnx.export(..., optimize=True)`` so custom nodes resolve.

Custom op ``since_version`` (ONNX opset) is fixed below; bump if the runtime
expects a newer schema revision. FP8 here uses standard ``QuantizeLinear`` /
``DequantizeLinear``, not custom FP8 schemas.
"""

from __future__ import annotations

import onnx
from onnx.defs import OpSchema

_SCHEMA_SINCE_VERSION = 23


def _safe_register_schema(schema: OpSchema) -> None:
    """Register *schema* if not already present (idempotent across packages)."""
    try:
        onnx.defs.register_schema(schema)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# trt_edgellm::AttentionPlugin, trt_edgellm::ViTAttentionPlugin
# ---------------------------------------------------------------------------

_attention_plugin_schema = OpSchema(
    name="AttentionPlugin",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc="Custom TensorRT attention plugin (packed-QKV contract) with RoPE, "
    "KV cache, and attention computation.  Q/K/V are concatenated on the "
    "last dim by the dynamo translation; the plugin internally splits the "
    "packed tensor and fuses RoPE + KV-cache write in a single kernel.",
    inputs=[
        OpSchema.FormalParameter(
            name="qkv",
            description=
            "Packed QKV tensor [B, S, (H_q + 2*H_kv) * D] (concat on last "
            "dim of separate Q/K/V projections)",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="past_key_value",
            description="KV cache tensor",
            type_str="T_KV",
        ),
        OpSchema.FormalParameter(
            name="context_lengths",
            description="Context length tensor",
            type_str="tensor(int32)",
        ),
        OpSchema.FormalParameter(
            name="rope_rotary_cos_sin",
            description="RoPE rotary embeddings (FP32)",
            type_str="tensor(float)",
        ),
        OpSchema.FormalParameter(
            name="kvcache_start_index",
            description=
            "KV cache start index tensor of shape [kv_cache_start_batch_size]",
            type_str="tensor(int32)",
        ),
        OpSchema.FormalParameter(
            name="kv_page_table",
            description=
            "Per-request page table of shape [batch, 2, max_pages_per_seq]",
            type_str="tensor(int32)",
        ),
        OpSchema.FormalParameter(
            name="q_norm_gamma",
            description=
            "Per-head RMSNorm gamma weights for Q (FP16, 1-D, length == head_size). Fed as a "
            "Constant initializer so TRT bakes it into the engine as weights at build time. "
            "Optional: only wired when enable_qk_norm=1 (models without qk_norm, e.g. Qwen2.5, "
            "omit this input entirely).",
            type_str="T",
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
        OpSchema.FormalParameter(
            name="k_norm_gamma",
            description=
            "Per-head RMSNorm gamma weights for K (FP16, 1-D, length == head_size). Same "
            "conventions as q_norm_gamma.",
            type_str="T",
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
        OpSchema.FormalParameter(
            name="context_mask_selector",
            description="Runtime context-mask selector tensor (optional)",
            type_str="tensor(int32)",
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
        OpSchema.FormalParameter(
            name="attention_mask",
            description="Attention mask tensor (optional)",
            type_str="tensor(int32)",
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
        OpSchema.FormalParameter(
            name="attention_pos_id",
            description="Position IDs tensor (optional)",
            type_str="tensor(int32)",
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
        OpSchema.FormalParameter(
            name="skip_softmax_scale",
            description="Runtime skip-softmax override carrier (optional): 1-D "
            "INT8 dummy whose LENGTH is the integer scale-factor override "
            "(0 = keep the engine default); data never read.",
            type_str="tensor(int8)",
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="attn_output",
            description="Attention output tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="present_key_value",
            description="Updated KV cache tensor",
            type_str="T_KV",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float16)"],
            "Input packed QKV data type.",
        ),
        (
            "T_KV",
            ["tensor(float16)", "tensor(float8e4m3fn)"],
            "KV cache data type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="num_q_heads",
            type=OpSchema.AttrType.INT,
            description="Number of query heads",
            required=True,
        ),
        OpSchema.Attribute(
            name="num_kv_heads",
            type=OpSchema.AttrType.INT,
            description="Number of key-value heads",
            required=True,
        ),
        OpSchema.Attribute(
            name="head_size",
            type=OpSchema.AttrType.INT,
            description="Size of each attention head",
            required=True,
        ),
        OpSchema.Attribute(
            name="enable_tree_attention",
            type=OpSchema.AttrType.INT,
            description="Whether to enable tree attention (0(false), 1(true))",
            required=True,
        ),
        OpSchema.Attribute(
            name="enable_fp8_kv_cache",
            type=OpSchema.AttrType.INT,
            description=
            "Whether to use FP8 KV cache (0(false), 1(true)). Optional.",
            required=False,
        ),
        OpSchema.Attribute(
            name="enable_context_mask_selector",
            type=OpSchema.AttrType.INT,
            description=(
                "Use context_mask_selector as a runtime causal/non-causal "
                "context-mask shape selector (0(false), 1(true)). Optional."),
            required=False,
        ),
        OpSchema.Attribute(
            name="enable_vision_block_attention",
            type=OpSchema.AttrType.INT,
            description=(
                "Use attention_mask as Gemma4 vision block IDs during "
                "prefill (0(false), 1(true)). Optional."),
            required=False,
        ),
        OpSchema.Attribute(
            name="sliding_window_size",
            type=OpSchema.AttrType.INT,
            description=
            "Sliding window size for attention (-1 = none, >0 = window size).",
            required=False,
        ),
        OpSchema.Attribute(
            name="qkv_scales",
            type=OpSchema.AttrType.FLOATS,
            description=
            "QKV dequant scales for FP8 KV cache: [q_scale, k_scale, v_scale]. "
            "Defaults to [1.0, 1.0, 1.0] when the checkpoint has no explicit scales.",
            required=False,
        ),
        OpSchema.Attribute(
            name="attention_scale",
            type=OpSchema.AttrType.FLOAT,
            description="Absolute multiplier applied to QK^T before softmax.",
            required=False,
        ),
        OpSchema.Attribute(
            name="skip_softmax_scale_factor",
            type=OpSchema.AttrType.FLOAT,
            description="Skip-softmax (BLASST) calibrated scale factor S "
            "(0 = disabled); runtime derives lambda = S / context_length "
            "per request for the prefill FMHA.",
            required=False,
        ),
        OpSchema.Attribute(
            name="enable_qk_norm",
            type=OpSchema.AttrType.INT,
            description=
            "Whether the fused per-head q_norm / k_norm RMSNorm is enabled (0(false), 1(true)). "
            "When 1, the q_norm_gamma / k_norm_gamma optional inputs must be wired.",
            required=False,
        ),
        OpSchema.Attribute(
            name="rms_norm_eps",
            type=OpSchema.AttrType.FLOAT,
            description=
            "Epsilon for the q_norm / k_norm RMSNorm formula. Ignored when enable_qk_norm=0.",
            required=False,
        ),
        OpSchema.Attribute(
            name="enable_kv_shared",
            type=OpSchema.AttrType.INT,
            description=
            "Whether this layer reads K/V from a donated (shared) cache; the packed qkv "
            "input then carries Q only [B, S, Hq*D] (0(false), 1(true)).",
            required=False,
        ),
    ],
)

_vit_attention_plugin_schema = OpSchema(
    name="ViTAttentionPlugin",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=
    "Custom TensorRT ViT attention plugin (separate Q/K/V, no KV cache, no RoPE).",
    inputs=[
        OpSchema.FormalParameter(
            name="q",
            description="Query tensor [total_S, H, D]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="k",
            description="Key tensor [total_S, H, D]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="v",
            description="Value tensor [total_S, H, D]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="cu_seqlens",
            description="Prefix sum of sequence lengths (int32, shape [B+1])",
            type_str="tensor(int32)",
        ),
        OpSchema.FormalParameter(
            name="max_seqlen_carrier",
            description="Shape-only carrier for max sequence length hint",
            type_str="tensor(int32)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="attn_output",
            description="Attention output [total_S, H, D]",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float16)"],
            "Input Q/K/V data type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="num_heads",
            type=OpSchema.AttrType.INT,
            description="Number of attention heads",
            required=True,
        ),
        OpSchema.Attribute(
            name="head_size",
            type=OpSchema.AttrType.INT,
            description="Size of each attention head",
            required=True,
        ),
        OpSchema.Attribute(
            name="attention_scale",
            type=OpSchema.AttrType.FLOAT,
            description="Absolute multiplier applied to QK^T before softmax.",
            required=False,
        ),
        OpSchema.Attribute(
            name="qkv_scales",
            type=OpSchema.AttrType.FLOATS,
            description=
            "Per-tensor FP8 dequant scales [q, k, v]; used when Q/K/V arrive "
            "as FLOAT8E4M3FN (the plugin infers FP8 mode from the tensor "
            "dtype), identity otherwise.",
            required=False,
        ),
    ],
)

# ---------------------------------------------------------------------------
# NVFP4 (trt domain)
# ---------------------------------------------------------------------------

_trt_fp4_dynamic_quantize_schema = OpSchema(
    name="TRT_FP4DynamicQuantize",
    domain="trt",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=
    ("Dynamically quantize float16 activations to NVFP4 with per-block scaling."
     ),
    inputs=[
        OpSchema.FormalParameter(
            name="x",
            description="Input activation tensor (float16)",
            type_str="tensor(float16)",
        ),
        OpSchema.FormalParameter(
            name="global_scale",
            description="Global scale constant (float16 scalar)",
            type_str="tensor(float16)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="x_fp4",
            description="Packed NVFP4 tensor",
            type_str="T_fp4",
        ),
        OpSchema.FormalParameter(
            name="x_sf",
            description="Per-block scale factors (packed)",
            type_str="T_fp4",
        ),
    ],
    type_constraints=[
        (
            "T_fp4",
            ["tensor(uint8)", "tensor(int8)"],
            "Packed FP4 tensor representation.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="axis",
            type=OpSchema.AttrType.INT,
            description="Quantization axis (default -1)",
            required=False,
        ),
        OpSchema.Attribute(
            name="block_size",
            type=OpSchema.AttrType.INT,
            description="Elements per block (typically 16)",
            required=True,
        ),
        OpSchema.Attribute(
            name="scale_type",
            type=OpSchema.AttrType.INT,
            description="ONNX elem_type for scale factors (17 = FLOAT8E4M3FN)",
            required=True,
        ),
    ],
)

_trt_dequantize_linear_schema = OpSchema(
    name="DequantizeLinear",
    domain="trt",
    since_version=_SCHEMA_SINCE_VERSION,
    doc="TRT-domain DequantizeLinear for NVFP4 per-block dequantization.",
    inputs=[
        OpSchema.FormalParameter(
            name="x",
            description="Quantized input",
            type_str="T_q",
        ),
        OpSchema.FormalParameter(
            name="x_scale",
            description="Per-block scale factors",
            type_str="T_scale",
        ),
        OpSchema.FormalParameter(
            name="x_zero_point",
            description="Zero-point (optional)",
            type_str="T_q",
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="y",
            description="Dequantized output",
            type_str="T_out",
        ),
    ],
    type_constraints=[
        (
            "T_q",
            ["tensor(uint8)", "tensor(int8)", "tensor(float8e4m3fn)"],
            "Quantized tensor type.",
        ),
        (
            "T_scale",
            ["tensor(float16)", "tensor(float)", "tensor(float8e4m3fn)"],
            "Scale tensor type.",
        ),
        (
            "T_out",
            ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"],
            "Dequantized output type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="axis",
            type=OpSchema.AttrType.INT,
            description="Dequantization axis",
            required=False,
        ),
        OpSchema.Attribute(
            name="block_size",
            type=OpSchema.AttrType.INT,
            description="Block size (NVFP4 uses 16)",
            required=False,
        ),
    ],
)

# ---------------------------------------------------------------------------
# MXFP8 (trt domain)
# ---------------------------------------------------------------------------

_trt_mxfp8_dynamic_quantize_schema = OpSchema(
    name="TRT_MXFP8DynamicQuantize",
    domain="trt",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=("Dynamically quantize float16 activations to MXFP8 with per-block "
         "E8M0 scaling."),
    inputs=[
        OpSchema.FormalParameter(
            name="x",
            description="Input activation tensor (float16)",
            type_str="tensor(float16)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="x_fp8",
            description="FP8E4M3 quantized tensor",
            type_str="T_fp8",
        ),
        OpSchema.FormalParameter(
            name="x_scale",
            description="E8M0 per-block scale factors (uint8)",
            type_str="T_scale",
        ),
    ],
    type_constraints=[
        (
            "T_fp8",
            ["tensor(float8e4m3fn)", "tensor(uint8)"],
            "FP8 quantized tensor.",
        ),
        (
            "T_scale",
            ["tensor(uint8)"],
            "E8M0 scale tensor (UINT8).",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="axis",
            type=OpSchema.AttrType.INT,
            description="Quantization axis (default -1)",
            required=False,
        ),
        OpSchema.Attribute(
            name="block_size",
            type=OpSchema.AttrType.INT,
            description="Elements per block (typically 32)",
            required=True,
        ),
        OpSchema.Attribute(
            name="output_dtype",
            type=OpSchema.AttrType.INT,
            description="ONNX elem_type for output (17 = FLOAT8E4M3FN)",
            required=True,
        ),
    ],
)

_trt_mxfp8_dequantize_linear_schema = OpSchema(
    name="TRT_MXFP8DequantizeLinear",
    domain="trt",
    since_version=_SCHEMA_SINCE_VERSION,
    doc="TRT-domain MXFP8 per-block dequantization with E8M0 scales.",
    inputs=[
        OpSchema.FormalParameter(
            name="x",
            description="Quantized input (FP8E4M3)",
            type_str="T_q",
        ),
        OpSchema.FormalParameter(
            name="x_scale",
            description="E8M0 per-block scale factors (uint8)",
            type_str="T_scale",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="y",
            description="Dequantized output",
            type_str="T_out",
        ),
    ],
    type_constraints=[
        (
            "T_q",
            ["tensor(float8e4m3fn)", "tensor(uint8)"],
            "Quantized tensor type.",
        ),
        (
            "T_scale",
            ["tensor(uint8)"],
            "E8M0 scale tensor type.",
        ),
        (
            "T_out",
            ["tensor(float16)", "tensor(float)"],
            "Dequantized output type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="axis",
            type=OpSchema.AttrType.INT,
            description="Dequantization axis",
            required=False,
        ),
        OpSchema.Attribute(
            name="block_size",
            type=OpSchema.AttrType.INT,
            description="Block size (MXFP8 uses 32)",
            required=False,
        ),
        OpSchema.Attribute(
            name="output_dtype",
            type=OpSchema.AttrType.INT,
            description="ONNX elem_type for output (10 = FLOAT16)",
            required=True,
        ),
    ],
)

# ---------------------------------------------------------------------------
# trt_edgellm::QkvConcatPlugin
# ---------------------------------------------------------------------------

_qkv_concat_schema = OpSchema(
    name="QkvConcatPlugin",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc="Pack three contiguous FP16 Q/K/V tensors along the last dimension.",
    inputs=[
        OpSchema.FormalParameter(
            name=name,
            description=f"{name.upper()} projection",
            type_str="T",
        ) for name in ("q", "k", "v")
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="qkv",
            description="Packed QKV tensor",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float16)"],
            "Q/K/V and output data type.",
        ),
    ],
)

# ---------------------------------------------------------------------------
# trt_edgellm::Int4GroupwiseGemmPlugin (AWQ swizzled weights)
# ---------------------------------------------------------------------------

_int4_groupwise_gemm_schema = OpSchema(
    name="Int4GroupwiseGemmPlugin",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc="TensorRT Int4 groupwise GEMM plugin (AWQ swizzled weights).",
    inputs=[
        OpSchema.FormalParameter(
            name="input",
            description="Input tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="qweight",
            description="Quantized weights, AWQ swizzle layout (int8)",
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="scales",
            description="Scales (float16)",
            type_str="tensor(float16)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            description="Output tensor",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"],
            "Input and output data type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="gemm_n",
            type=OpSchema.AttrType.INT,
            description="Output feature dimension",
            required=True,
        ),
        OpSchema.Attribute(
            name="gemm_k",
            type=OpSchema.AttrType.INT,
            description="Input feature dimension",
            required=True,
        ),
        OpSchema.Attribute(
            name="group_size",
            type=OpSchema.AttrType.INT,
            description="Group size",
            required=True,
        ),
    ],
)

# ---------------------------------------------------------------------------
# trt_edgellm::Int4GroupwiseGemmPluginV2 (cuteDSL fragment-layout weights)
# ---------------------------------------------------------------------------

_int4_groupwise_gemm_v2_schema = OpSchema(
    name="Int4GroupwiseGemmPluginV2",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=
    "TensorRT Int4 groupwise GEMM plugin (cuteDSL fragment-layout weights).",
    inputs=[
        OpSchema.FormalParameter(
            name="input",
            description="Input tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="qweight",
            description="Quantized weights, cuteDSL fragment layout (int8)",
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="scales",
            description="Scales (float16)",
            type_str="tensor(float16)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            description="Output tensor",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"],
            "Input and output data type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="gemm_n",
            type=OpSchema.AttrType.INT,
            description="Output feature dimension",
            required=True,
        ),
        OpSchema.Attribute(
            name="gemm_k",
            type=OpSchema.AttrType.INT,
            description="Input feature dimension",
            required=True,
        ),
        OpSchema.Attribute(
            name="group_size",
            type=OpSchema.AttrType.INT,
            description="Group size",
            required=True,
        ),
    ],
)

# ---------------------------------------------------------------------------
# trt_edgellm::Nvfp4A16GemmPlugin (dense FP16-A / NVFP4-W4 Marlin GEMM)
# ---------------------------------------------------------------------------

_nvfp4_a16_gemm_schema = OpSchema(
    name="Nvfp4A16GemmPlugin",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc="TensorRT dense NVFP4 (W4A16) Marlin GEMM plugin.",
    inputs=[
        OpSchema.FormalParameter(
            name="activation",
            description="FP16 activation [B, S, gemm_k]",
            type_str="tensor(float16)",
        ),
        OpSchema.FormalParameter(
            name="qweights",
            description=
            "Marlin-packed E2M1 codes, INT8 view [1, gemm_k/16, 8*gemm_n]",
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="block_scales",
            description=
            "Marlin-permuted E4M3 block-scale bytes [1, gemm_k/16, gemm_n]",
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="global_scale",
            description="Per-tensor FP16 global scale (pre-scaled by 2**7) [1]",
            type_str="tensor(float16)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            description="FP16 output [B, S, gemm_n]",
            type_str="tensor(float16)",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="gemm_n",
            type=OpSchema.AttrType.INT,
            description="Marlin-padded output feature dimension",
            required=True,
        ),
        OpSchema.Attribute(
            name="gemm_k",
            type=OpSchema.AttrType.INT,
            description="Input feature dimension",
            required=True,
        ),
        OpSchema.Attribute(
            name="max_m",
            type=OpSchema.AttrType.INT,
            description=
            "Profile token capacity for workspace sizing (0 == auto)",
            required=False,
        ),
    ],
)

# ---------------------------------------------------------------------------
# trt_edgellm::causal_conv1d, update_ssm_state
# ---------------------------------------------------------------------------

_causal_conv1d_schema = OpSchema(
    name="causal_conv1d",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc="Causal 1D depthwise convolution with persistent state.",
    inputs=[
        OpSchema.FormalParameter(name="x",
                                 description="Input tensor",
                                 type_str="T"),
        OpSchema.FormalParameter(name="weight",
                                 description="Conv weight",
                                 type_str="T"),
        OpSchema.FormalParameter(name="bias",
                                 description="Conv bias",
                                 type_str="T"),
        OpSchema.FormalParameter(name="conv_state",
                                 description="Conv state",
                                 type_str="T"),
        OpSchema.FormalParameter(name="context_lengths",
                                 description="Context lengths per batch",
                                 type_str="T_CL"),
        OpSchema.FormalParameter(
            name="spec_decode_metadata",
            description=
            "Optional speculative metadata: spec_verify_phase_marker, tree_parent_ids, tree_depths",
            type_str="T_CL",
            param_option=OpSchema.FormalParameterOption.Variadic),
    ],
    outputs=[
        OpSchema.FormalParameter(name="output",
                                 description="Conv output",
                                 type_str="T"),
        OpSchema.FormalParameter(name="conv_state_out",
                                 description="Updated conv state",
                                 type_str="T"),
        OpSchema.FormalParameter(
            name="intermediate_conv_state_out",
            description="Per-token conv states [batch, seq, dim, width]",
            type_str="T",
            param_option=OpSchema.FormalParameterOption.Optional),
    ],
    type_constraints=[
        ("T", ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"], ""),
        ("T_CL", ["tensor(int32)"], ""),
    ],
    attributes=[
        OpSchema.Attribute(name="stride",
                           type=OpSchema.AttrType.INT,
                           description="Stride",
                           required=True),
        OpSchema.Attribute(name="padding",
                           type=OpSchema.AttrType.INT,
                           description="Padding",
                           required=True),
        OpSchema.Attribute(name="dilation",
                           type=OpSchema.AttrType.INT,
                           description="Dilation",
                           required=True),
        OpSchema.Attribute(name="groups",
                           type=OpSchema.AttrType.INT,
                           description="Groups",
                           required=True),
        OpSchema.Attribute(
            name="use_mtp",
            type=OpSchema.AttrType.INT,
            description=
            "Enable linear spec-verify intermediate state output; retained as use_mtp for compatibility with existing MTP ONNX/plugin schema",
            required=False),
        OpSchema.Attribute(
            name="use_ddtree",
            type=OpSchema.AttrType.INT,
            description=
            "Whether tree_parent_ids/tree_depths drive tree-state execution; also enables intermediate state output",
            required=False),
    ],
)

_update_ssm_state_schema = OpSchema(
    name="update_ssm_state",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc="Mamba selective SSM state update.",
    inputs=[
        OpSchema.FormalParameter(name="x", description="Input x",
                                 type_str="T"),
        OpSchema.FormalParameter(name="A",
                                 description="A parameter",
                                 type_str="T_A"),
        OpSchema.FormalParameter(name="B",
                                 description="B parameter",
                                 type_str="T"),
        OpSchema.FormalParameter(name="C",
                                 description="C parameter",
                                 type_str="T"),
        OpSchema.FormalParameter(name="D",
                                 description="D parameter",
                                 type_str="T"),
        OpSchema.FormalParameter(name="dt",
                                 description="dt parameter",
                                 type_str="T"),
        OpSchema.FormalParameter(name="dt_bias",
                                 description="dt_bias parameter",
                                 type_str="T"),
        OpSchema.FormalParameter(name="state",
                                 description="SSM state",
                                 type_str="T"),
        OpSchema.FormalParameter(name="context_lengths",
                                 description="Context lengths per batch",
                                 type_str="T_CL"),
        OpSchema.FormalParameter(
            name="state_start_index",
            description=
            "[0] initial-prefill sentinel or [batch] restored-state marker",
            type_str="T_CL"),
        OpSchema.FormalParameter(
            name="spec_verify_phase_marker",
            description="Optional shape-only INT32 marker (len 0=ordinary, "
            "1=verify) enabling per-token intermediate state capture",
            type_str="T_CL",
            param_option=OpSchema.FormalParameterOption.Optional),
    ],
    outputs=[
        OpSchema.FormalParameter(name="output",
                                 description="SSM output",
                                 type_str="T"),
        OpSchema.FormalParameter(name="state_out",
                                 description="Updated SSM state",
                                 type_str="T"),
        OpSchema.FormalParameter(
            name="replay_da",
            description="Optional spec-verify replay stash: per-token decay "
            "dA [batch, seq, nheads] FP32",
            type_str="T_F32",
            param_option=OpSchema.FormalParameterOption.Optional),
        OpSchema.FormalParameter(
            name="replay_u",
            description="Optional spec-verify replay stash: per-token input "
            "factor u=dt*x [batch, seq, nheads, dim] FP32",
            type_str="T_F32",
            param_option=OpSchema.FormalParameterOption.Optional),
        OpSchema.FormalParameter(
            name="replay_b",
            description="Optional spec-verify replay stash: per-token key "
            "B [batch, seq, ngroups, dstate] FP32",
            type_str="T_F32",
            param_option=OpSchema.FormalParameterOption.Optional),
    ],
    type_constraints=[
        ("T", ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"], ""),
        ("T_A", ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"], ""),
        ("T_CL", ["tensor(int32)"], ""),
        ("T_F32", ["tensor(float)"], ""),
    ],
    attributes=[
        OpSchema.Attribute(name="dt_softplus",
                           type=OpSchema.AttrType.INT,
                           description="Apply softplus to dt",
                           required=True),
        OpSchema.Attribute(name="ngroups",
                           type=OpSchema.AttrType.INT,
                           description="Number of groups",
                           required=True),
        OpSchema.Attribute(
            name="chunk_size",
            type=OpSchema.AttrType.INT,
            description="Mamba2 prefill chunk size (0=sequential)",
            required=False),
        OpSchema.Attribute(name="time_step_limit",
                           type=OpSchema.AttrType.FLOATS,
                           description="Time step clamping range",
                           required=False),
        OpSchema.Attribute(
            name="use_spec_verify_state",
            type=OpSchema.AttrType.INT,
            description="Emit per-token intermediate recurrent states (1) or "
            "not (0)",
            required=False),
    ],
)

# ---------------------------------------------------------------------------
# TRT native attention ops (domain "" — default ONNX domain)
# RotaryEmbedding, TensorScatter, Attention are consumed by TRT >= 10.15
# without any custom plugin; they live in the default ONNX domain.
# ---------------------------------------------------------------------------

_rotary_embedding_schema = OpSchema(
    name="RotaryEmbedding",
    domain="trt",
    since_version=_SCHEMA_SINCE_VERSION,
    doc="TRT native rotary position embedding applied to Q or K tensors.",
    inputs=[
        OpSchema.FormalParameter(
            name="x",
            description="Input tensor [batch, heads, seq, head_dim]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="cos",
            description="Cosine table [max_pos, head_dim//2]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="sin",
            description="Sine table [max_pos, head_dim//2]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="position_ids",
            description="Position indices [batch, seq] (int32)",
            type_str="tensor(int32)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            description="RoPE-embedded tensor, same shape as x",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float16)", "tensor(float)", "tensor(bfloat16)"],
            "Input and output data type.",
        ),
    ],
)

_tensor_scatter_schema = OpSchema(
    name="TensorScatter",
    domain="trt",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=
    "TRT native KV cache scatter update: writes new_kv into cache at cache_indices.",
    inputs=[
        OpSchema.FormalParameter(
            name="cache",
            description="KV cache tensor [batch, kv_heads, capacity, head_dim]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="new_kv",
            description="New KV values [batch, kv_heads, seq, head_dim]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="cache_indices",
            description="Write offset per batch item [batch] (int32)",
            type_str="tensor(int32)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="updated_cache",
            description="Updated KV cache, same shape as cache",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float16)", "tensor(float)", "tensor(bfloat16)"],
            "Cache and new_kv data type.",
        ),
    ],
)

_attention_trt_native_schema = OpSchema(
    name="Attention",
    domain="trt",
    since_version=_SCHEMA_SINCE_VERSION,
    doc="TRT native scaled dot-product attention (TRT_decomposable=1).",
    inputs=[
        OpSchema.FormalParameter(
            name="query",
            description="Query tensor [batch, heads, seq_q, head_dim]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="key",
            description="Key tensor [batch, kv_heads, seq_k, head_dim]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="value",
            description="Value tensor [batch, kv_heads, seq_k, head_dim]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="attn_mask",
            description=
            "Additive attention mask [1, 1, seq_q, seq_k] (optional)",
            type_str="T",
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="attn_output",
            description="Attention output, same shape as query",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float16)", "tensor(float)", "tensor(bfloat16)"],
            "Input and output data type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="is_causal",
            type=OpSchema.AttrType.INT,
            description="Whether to apply a causal mask (0=false, 1=true)",
            required=False,
        ),
        OpSchema.Attribute(
            name="TRT_decomposable",
            type=OpSchema.AttrType.INT,
            description="Allow TRT to decompose this node (always 1)",
            required=False,
        ),
        OpSchema.Attribute(
            name="scale",
            type=OpSchema.AttrType.FLOAT,
            description="Absolute multiplier applied to QK^T before softmax.",
            required=False,
        ),
    ],
)

_trt_ragged_attention_schema = OpSchema(
    name="TRT_Attention",
    domain="trt",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=
    "TRT-native ragged attention (packed NHD, no causal mask, no KV cache).",
    inputs=[
        OpSchema.FormalParameter(
            name="query",
            description="Query tensor [total_S, H, D]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="key",
            description="Key tensor [total_S, H, D]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="value",
            description="Value tensor [total_S, H, D]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="mask",
            description="Optional attention mask",
            type_str="T",
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
        OpSchema.FormalParameter(name="query_lengths",
                                 description="Cumulative query lengths [B+1]",
                                 type_str="tensor(int32)"),
        OpSchema.FormalParameter(name="kv_lengths",
                                 description="Cumulative KV lengths [B+1]",
                                 type_str="tensor(int32)"),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="attn_output",
            description="Attention output [total_S, H, D]",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float16)", "tensor(float)", "tensor(bfloat16)"],
            "Input and output data type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="query_form",
            type=OpSchema.AttrType.STRING,
            description="Query IO form",
            required=True,
        ),
        OpSchema.Attribute(
            name="kv_form",
            type=OpSchema.AttrType.STRING,
            description="Key-value IO form",
            required=True,
        ),
        OpSchema.Attribute(
            name="causal_kind",
            type=OpSchema.AttrType.STRING,
            description="Causal mask kind (none/upper_left/lower_right)",
            required=False,
        ),
        OpSchema.Attribute(
            name="TRT_decomposable",
            type=OpSchema.AttrType.INT,
            description="Whether the attention can be decomposed",
            required=False,
        ),
    ],
)

_gated_delta_net_schema = OpSchema(
    name="gated_delta_net",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc="Qwen3.5 GatedDeltaNet linear attention plugin.",
    inputs=[
        OpSchema.FormalParameter(name="q",
                                 description="Query [n, seq, h, k]",
                                 type_str="T"),
        OpSchema.FormalParameter(name="k",
                                 description="Key [n, seq, h, k]",
                                 type_str="T"),
        OpSchema.FormalParameter(name="v",
                                 description="Value [n, seq, hv, v]",
                                 type_str="T"),
        OpSchema.FormalParameter(name="a",
                                 description="A gating tensor [n, seq, hv]",
                                 type_str="T"),
        OpSchema.FormalParameter(name="b",
                                 description="B gating tensor [n, seq, hv]",
                                 type_str="T"),
        OpSchema.FormalParameter(name="A_log",
                                 description="A_log [hv]",
                                 type_str="T_A"),
        OpSchema.FormalParameter(name="dt_bias",
                                 description="dt_bias [hv]",
                                 type_str="T"),
        OpSchema.FormalParameter(
            name="h0_source",
            description="Recurrent state in [n, hv, k, v]",
            type_str="T_A"),
        OpSchema.FormalParameter(
            name="context_lengths",
            description="Valid token count per batch row [n]",
            type_str="T_CL"),
        OpSchema.FormalParameter(
            name="spec_decode_metadata",
            description=
            "Optional speculative metadata: spec_verify_phase_marker, tree_parent_ids, tree_depths",
            type_str="T_CL",
            param_option=OpSchema.FormalParameterOption.Variadic),
    ],
    outputs=[
        OpSchema.FormalParameter(name="o",
                                 description="Output [n, seq, hv, v]",
                                 type_str="T"),
        OpSchema.FormalParameter(
            name="h0_out",
            description="Recurrent state out [n, hv, k, v]",
            type_str="T_A"),
        OpSchema.FormalParameter(
            name="intermediate_h0_out",
            description="Per-token recurrent states [n, seq, hv, k, v]",
            type_str="T_A",
            param_option=OpSchema.FormalParameterOption.Optional),
    ],
    type_constraints=[
        ("T", ["tensor(float16)"], ""),
        ("T_A", ["tensor(float)"], ""),
        ("T_CL", ["tensor(int32)"], ""),
    ],
    attributes=[
        OpSchema.Attribute(name="k_dim",
                           type=OpSchema.AttrType.INT,
                           description="K head dimension",
                           required=True),
        OpSchema.Attribute(name="v_dim",
                           type=OpSchema.AttrType.INT,
                           description="V head dimension",
                           required=True),
        OpSchema.Attribute(
            name="use_mtp",
            type=OpSchema.AttrType.INT,
            description=
            "Enable linear spec-verify intermediate state output; retained as use_mtp for compatibility with existing MTP ONNX/plugin schema",
            required=False),
        OpSchema.Attribute(
            name="use_ddtree",
            type=OpSchema.AttrType.INT,
            description=
            "Whether tree_parent_ids/tree_depths drive tree-state execution; also enables intermediate state output",
            required=False),
    ],
)

# ---------------------------------------------------------------------------
# trt_edgellm::Int4MoePlugin
# ---------------------------------------------------------------------------

_int4_moe_plugin_schema = OpSchema(
    name="Int4MoePlugin",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=("Fused sparse MoE with INT4 expert GEMMs (Marlin layout).  "
         "Receives router_logits (FP32, from traced gate GEMM) and "
         "hidden_states; performs softmax + topk routing, then per-expert "
         "grouped GEMM (gate_up + SiLU + down) with weighted combine.  "
         "Mirrors trt_edgellm::Int4MoePlugin in tensorrt_edgellm."),
    inputs=[
        OpSchema.FormalParameter(
            name="router_logits",
            description=
            "Router logits (B*S, E) FP32, from gate GEMM + cast, before softmax",
            type_str="tensor(float)",
        ),
        OpSchema.FormalParameter(
            name="hidden_states",
            description="Input hidden states (B, S, D)",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="fc_gate_up_qweights",
            description=
            "Fused gate+up proj Marlin-packed weights (E, K//16, 2*I) int8",
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="fc_gate_up_scales",
            description="Fused gate+up proj scales (E, num_groups, I)",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="fc_down_qweights",
            description="Down proj Marlin-packed weights (E, K//16, 2*D) int8",
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="fc_down_scales",
            description="Down proj scales (E, num_groups, D)",
            type_str="T",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            description="Output tensor (B, S, D)",
            type_str="T",
        ),
    ],
    type_constraints=[
        ("T", ["tensor(float16)"], "FP16 data type."),
    ],
    attributes=[
        OpSchema.Attribute(
            name="num_experts",
            type=OpSchema.AttrType.INT,
            description="Number of experts",
            required=True,
        ),
        OpSchema.Attribute(
            name="top_k",
            type=OpSchema.AttrType.INT,
            description="Top K experts per token",
            required=True,
        ),
        OpSchema.Attribute(
            name="hidden_size",
            type=OpSchema.AttrType.INT,
            description="Hidden size D",
            required=True,
        ),
        OpSchema.Attribute(
            name="moe_inter_size",
            type=OpSchema.AttrType.INT,
            description="MoE intermediate size I",
            required=True,
        ),
        OpSchema.Attribute(
            name="activation_type",
            type=OpSchema.AttrType.INT,
            description="Activation function type (0=SiLU)",
            required=True,
        ),
        OpSchema.Attribute(
            name="quantization_group_size",
            type=OpSchema.AttrType.INT,
            description="Quantization group size G",
            required=True,
        ),
    ],
)

# ---------------------------------------------------------------------------
# trt_edgellm::Nvfp4MoePlugin (SM100/101/110 split FC1/FC2, 64-row up/gate interleave)
# ---------------------------------------------------------------------------

_nvfp4_moe_plugin_schema = OpSchema(
    name="Nvfp4MoePlugin",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=("NVFP4 MoE plugin (CuTeDSL SM100/101/110 split FC1/FC2): FP16 hidden "
         "states, FP4 expert weights with 64-row up/gate interleaved FC1, "
         "and FP8 block scales in 6D MMA layout."),
    inputs=[
        OpSchema.FormalParameter("router_logits", "T_ROUTER",
                                 "Router logits [B*S, E] FP32"),
        OpSchema.FormalParameter("hidden_states", "T_HIDDEN",
                                 "Hidden states [B, S, H] FP16"),
        OpSchema.FormalParameter("fc1_qweights", "T_INT8",
                                 "FC1 weights [E, N1, H/2] INT8"),
        OpSchema.FormalParameter(
            "fc1_blocks_scale", "T_INT8",
            "FC1 block scales [E, m_tiles, k_tiles, 32, 4, 4] INT8"),
        OpSchema.FormalParameter("fc1_alpha", "T_ROUTER",
                                 "FC1 global weight scales [E] FP32"),
        OpSchema.FormalParameter("fc2_qweights", "T_INT8",
                                 "FC2 weights [E, H, I/2] INT8"),
        OpSchema.FormalParameter(
            "fc2_blocks_scale", "T_INT8",
            "FC2 block scales [E, m_tiles, k_tiles, 32, 4, 4] INT8"),
        OpSchema.FormalParameter("fc2_alpha", "T_ROUTER",
                                 "FC2 global weight scales [E] FP32"),
        OpSchema.FormalParameter("input_global_scale", "T_ROUTER",
                                 "FC1 activation scales [E] FP32"),
        OpSchema.FormalParameter("down_input_scale", "T_ROUTER",
                                 "FC2 activation scales [E] FP32"),
        OpSchema.FormalParameter("e_score_correction_bias", "T_ROUTER",
                                 "Router correction bias [E] FP32"),
    ],
    outputs=[
        OpSchema.FormalParameter("output", "T_HIDDEN",
                                 "Output [B, S, H] FP16"),
    ],
    type_constraints=[
        ("T_ROUTER", ["tensor(float)"], "FP32 tensors"),
        ("T_HIDDEN", ["tensor(float16)"], "FP16 tensors"),
        ("T_INT8", ["tensor(int8)"], "INT8 byte tensors"),
    ],
    attributes=[
        OpSchema.Attribute("num_experts", OpSchema.AttrType.INT),
        OpSchema.Attribute("top_k", OpSchema.AttrType.INT),
        OpSchema.Attribute("hidden_size", OpSchema.AttrType.INT),
        OpSchema.Attribute("moe_inter_size", OpSchema.AttrType.INT),
        OpSchema.Attribute("activation_type", OpSchema.AttrType.INT),
        OpSchema.Attribute("n_group", OpSchema.AttrType.INT),
        OpSchema.Attribute("topk_group", OpSchema.AttrType.INT),
        OpSchema.Attribute("norm_topk_prob", OpSchema.AttrType.INT),
        OpSchema.Attribute("routed_scaling_factor", OpSchema.AttrType.FLOAT),
        OpSchema.Attribute("routing_mode", OpSchema.AttrType.INT),
        OpSchema.Attribute("backend", OpSchema.AttrType.INT),
        OpSchema.Attribute("io_dtype", OpSchema.AttrType.INT),
        OpSchema.Attribute("max_routed_rows", OpSchema.AttrType.INT),
    ],
)

# ---------------------------------------------------------------------------
# trt_edgellm::Fp16MoePlugin (unquantized experts, CuTeDSL grouped GEMM)
# ---------------------------------------------------------------------------

_fp16_moe_plugin_schema = OpSchema(
    name="Fp16MoePlugin",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=(
        "FP16 MoE plugin (CuTeDSL grouped GEMM): FP16 hidden states and "
        "plain FP16 expert weights. FC1 is 64-row up/gate interleaved for "
        "SwiGLU or a plain [E, I, H] block for ReLU2. No quantization scales. "
        "Routing is softmax top-k (routing_mode=0, default) or sigmoid "
        "group-topk (routing_mode=1), which additionally consumes the "
        "optional e_score_correction_bias input and the n_group / topk_group "
        "/ routed_scaling_factor attributes."),
    inputs=[
        OpSchema.FormalParameter("router_logits", "T_ROUTER",
                                 "Router logits [B*S, E] FP32"),
        OpSchema.FormalParameter("hidden_states", "T_HIDDEN",
                                 "Hidden states [B, S, H] FP16"),
        OpSchema.FormalParameter("fc1_weights", "T_HIDDEN",
                                 "FC1 weights [E, N1, H] FP16"),
        OpSchema.FormalParameter("fc2_weights", "T_HIDDEN",
                                 "FC2 weights [E, H, I] FP16"),
        OpSchema.FormalParameter(
            "e_score_correction_bias",
            "T_ROUTER",
            "Router correction bias [E] FP32 (sigmoid group-topk only)",
            param_option=OpSchema.FormalParameterOption.Optional),
    ],
    outputs=[
        OpSchema.FormalParameter("output", "T_HIDDEN",
                                 "Output [B, S, H] FP16"),
    ],
    type_constraints=[
        ("T_ROUTER", ["tensor(float)"], "FP32 tensors"),
        ("T_HIDDEN", ["tensor(float16)"], "FP16 tensors"),
    ],
    attributes=[
        OpSchema.Attribute("num_experts", OpSchema.AttrType.INT),
        OpSchema.Attribute("top_k", OpSchema.AttrType.INT),
        OpSchema.Attribute("hidden_size", OpSchema.AttrType.INT),
        OpSchema.Attribute("moe_inter_size", OpSchema.AttrType.INT),
        OpSchema.Attribute("activation_type", OpSchema.AttrType.INT),
        OpSchema.Attribute("norm_topk_prob", OpSchema.AttrType.INT),
        OpSchema.Attribute("max_routed_rows", OpSchema.AttrType.INT),
        # Sigmoid group-topk routing (routing_mode=1) only; optional so the
        # softmax export (Qwen) can omit them.
        OpSchema.Attribute("n_group", OpSchema.AttrType.INT, required=False),
        OpSchema.Attribute("topk_group", OpSchema.AttrType.INT,
                           required=False),
        OpSchema.Attribute("routed_scaling_factor",
                           OpSchema.AttrType.FLOAT,
                           required=False),
        OpSchema.Attribute("routing_mode",
                           OpSchema.AttrType.INT,
                           required=False),
    ],
)

# ---------------------------------------------------------------------------
# trt_edgellm::Nvfp4A16MoePlugin (Marlin FP16-A / NVFP4-W4 weight-only MoE)
# ---------------------------------------------------------------------------

_nvfp4_a16_moe_plugin_schema = OpSchema(
    name="Nvfp4A16MoePlugin",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=("Marlin NVFP4 (W4A16) MoE plugin: FP16 hidden states, Marlin-packed "
         "E2M1 expert weights (INT8 view), raw E4M3 block scales, and FP16 "
         "per-expert global scales pre-scaled by 2**7."),
    inputs=[
        OpSchema.FormalParameter("router_logits", "T_ROUTER",
                                 "Router logits [B*S, E] FP32"),
        OpSchema.FormalParameter("hidden_states", "T_HIDDEN",
                                 "Hidden states [B, S, H] FP16"),
        OpSchema.FormalParameter(
            "fc1_qweights", "T_INT8",
            "FC1 Marlin weights [E, H/16, 8*fc1_out] INT8"),
        OpSchema.FormalParameter(
            "fc1_block_scales", "T_INT8",
            "FC1 E4M3 block scales [E, H/16, fc1_out] INT8"),
        OpSchema.FormalParameter("fc1_global_scales", "T_HIDDEN",
                                 "FC1 per-expert global scales [E] FP16"),
        OpSchema.FormalParameter("fc2_qweights", "T_INT8",
                                 "FC2 Marlin weights [E, I/16, 8*H] INT8"),
        OpSchema.FormalParameter("fc2_block_scales", "T_INT8",
                                 "FC2 E4M3 block scales [E, I/16, H] INT8"),
        OpSchema.FormalParameter("fc2_global_scales", "T_HIDDEN",
                                 "FC2 per-expert global scales [E] FP16"),
        OpSchema.FormalParameter("e_score_correction_bias", "T_ROUTER",
                                 "Router correction bias [E] FP32"),
    ],
    outputs=[
        OpSchema.FormalParameter("output", "T_HIDDEN",
                                 "Output [B, S, H] FP16"),
    ],
    type_constraints=[
        ("T_ROUTER", ["tensor(float)"], "FP32 tensors"),
        ("T_HIDDEN", ["tensor(float16)"], "FP16 tensors"),
        ("T_INT8", ["tensor(int8)"], "INT8 byte tensors"),
    ],
    attributes=[
        OpSchema.Attribute("num_experts", OpSchema.AttrType.INT),
        OpSchema.Attribute("top_k", OpSchema.AttrType.INT),
        OpSchema.Attribute("hidden_size", OpSchema.AttrType.INT),
        OpSchema.Attribute("moe_inter_size", OpSchema.AttrType.INT),
        OpSchema.Attribute("activation_type", OpSchema.AttrType.INT),
        OpSchema.Attribute("n_group", OpSchema.AttrType.INT),
        OpSchema.Attribute("topk_group", OpSchema.AttrType.INT),
        OpSchema.Attribute("norm_topk_prob", OpSchema.AttrType.INT),
        OpSchema.Attribute("routed_scaling_factor", OpSchema.AttrType.FLOAT),
        OpSchema.Attribute("routing_mode", OpSchema.AttrType.INT),
        OpSchema.Attribute("max_routed_rows", OpSchema.AttrType.INT),
    ],
)

# ---------------------------------------------------------------------------
# trt_edgellm::NvFP4MoEPluginGeforce (SM12x fused, plain [up, gate] concat)
# ---------------------------------------------------------------------------

_nvfp4_moe_plugin_geforce_schema = OpSchema(
    name="NvFP4MoEPluginGeforce",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=("NVFP4 MoE plugin (CuTeDSL SM12x fused): FP16 hidden states, FP4 "
         "expert weights with plain [up_all, gate_all] concat FC1 (no 64-row "
         "up/gate interleave), and FP8 block scales in 6D MMA layout. Same "
         "11-input ONNX surface as Nvfp4MoePlugin; FC1 packing convention "
         "and target arch differ."),
    inputs=[
        OpSchema.FormalParameter("router_logits", "T_ROUTER",
                                 "Router logits [B*S, E] FP32"),
        OpSchema.FormalParameter("hidden_states", "T_HIDDEN",
                                 "Hidden states [B, S, H] FP16"),
        OpSchema.FormalParameter(
            "fc1_qweights", "T_INT8",
            "FC1 weights [E, N1, H/2] INT8 (plain [up, gate] concat)"),
        OpSchema.FormalParameter(
            "fc1_blocks_scale", "T_INT8",
            "FC1 block scales [E, m_tiles, k_tiles, 32, 4, 4] INT8"),
        OpSchema.FormalParameter("fc1_alpha", "T_ROUTER",
                                 "FC1 global weight scales [E] FP32"),
        OpSchema.FormalParameter("fc2_qweights", "T_INT8",
                                 "FC2 weights [E, H, I/2] INT8"),
        OpSchema.FormalParameter(
            "fc2_blocks_scale", "T_INT8",
            "FC2 block scales [E, m_tiles, k_tiles, 32, 4, 4] INT8"),
        OpSchema.FormalParameter("fc2_alpha", "T_ROUTER",
                                 "FC2 global weight scales [E] FP32"),
        OpSchema.FormalParameter("input_global_scale", "T_ROUTER",
                                 "FC1 activation scales [E] FP32"),
        OpSchema.FormalParameter("down_input_scale", "T_ROUTER",
                                 "FC2 activation scales [E] FP32"),
        OpSchema.FormalParameter("e_score_correction_bias", "T_ROUTER",
                                 "Router correction bias [E] FP32"),
    ],
    outputs=[
        OpSchema.FormalParameter("output", "T_HIDDEN",
                                 "Output [B, S, H] FP16"),
    ],
    type_constraints=[
        ("T_ROUTER", ["tensor(float)"], "FP32 tensors"),
        ("T_HIDDEN", ["tensor(float16)"], "FP16 tensors"),
        ("T_INT8", ["tensor(int8)"], "INT8 byte tensors"),
    ],
    attributes=[
        OpSchema.Attribute("num_experts", OpSchema.AttrType.INT),
        OpSchema.Attribute("top_k", OpSchema.AttrType.INT),
        OpSchema.Attribute("hidden_size", OpSchema.AttrType.INT),
        OpSchema.Attribute("moe_inter_size", OpSchema.AttrType.INT),
        OpSchema.Attribute("activation_type", OpSchema.AttrType.INT),
        OpSchema.Attribute("n_group", OpSchema.AttrType.INT),
        OpSchema.Attribute("topk_group", OpSchema.AttrType.INT),
        OpSchema.Attribute("norm_topk_prob", OpSchema.AttrType.INT),
        OpSchema.Attribute("routed_scaling_factor", OpSchema.AttrType.FLOAT),
        OpSchema.Attribute("routing_mode", OpSchema.AttrType.INT),
        OpSchema.Attribute("backend", OpSchema.AttrType.INT),
        OpSchema.Attribute("io_dtype", OpSchema.AttrType.INT),
        OpSchema.Attribute("max_routed_rows", OpSchema.AttrType.INT),
    ],
)

# ---------------------------------------------------------------------------
# trt_edgellm::FusedNvfp4GemmAllReducePlugin (row-parallel NVFP4 GEMM)
# ---------------------------------------------------------------------------

_fused_nvfp4_gemm_allreduce_plugin_schema = OpSchema(
    name="FusedNvfp4GemmAllReducePlugin",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=("NVFP4 row-parallel GEMM fused with AllReduce.  Replaces the "
         "dequant→MatMul→AllReduce chain in o_proj / down_proj when "
         "tp_size>1.  Inputs are in raw FP4 form; output is FP16 and "
         "already AllReduced across ranks."),
    inputs=[
        OpSchema.FormalParameter(
            name="fp4_act",
            description="FP4 packed activation [..., K_per_rank // 2]",
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="act_scale",
            description=
            "Combined per-block activation scale [..., K_per_rank // 16] (FP32)",
            type_str="tensor(float)",
        ),
        OpSchema.FormalParameter(
            name="weight_f4",
            description="FP4 packed weight [N, K_per_rank // 2]",
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="weight_f8_scale",
            description=
            "FP8E4M3FN per-block weight scale [N, K_per_rank // group_size]",
            type_str="tensor(float8e4m3fn)",
        ),
        OpSchema.FormalParameter(
            name="weight_f32_scale",
            description="FP32 global weight scale (scalar)",
            type_str="tensor(float)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            description="AllReduced FP16 output [..., N]",
            type_str="tensor(float16)",
        ),
    ],
    type_constraints=[],
    attributes=[
        OpSchema.Attribute(
            name="tp_size",
            type=OpSchema.AttrType.INT,
            description="Tensor parallel world size",
            required=True,
        ),
    ],
)

# ---------------------------------------------------------------------------
# trt_edgellm::DFlashTargetKVCacheUpdate
# ---------------------------------------------------------------------------

_dflash_target_kv_cache_update_schema = OpSchema(
    name="DFlashTargetKVCacheUpdate",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=("DFlash target KV cache update: apply RoPE to k_delta and write "
         "k_rope + v_delta into the combined draft KV cache."),
    inputs=[
        OpSchema.FormalParameter(
            name="k_delta",
            description="K delta [B, L, Hkv, D] after k_norm, no RoPE",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="v_delta",
            description="V delta [B, L, Hkv, D]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="past_key_value",
            description=
            "Combined KV cache, paged pool [2, num_pages, KV_PAGE_SIZE, Hkv, D]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="rope_cos_sin",
            description="RoPE cos/sin [ropeBatch, capacity, rotaryDim] FP32",
            type_str="tensor(float)",
        ),
        OpSchema.FormalParameter(
            name="delta_start_positions",
            description="Start positions for delta write [B] INT32",
            type_str="tensor(int32)",
        ),
        OpSchema.FormalParameter(
            name="delta_lengths",
            description=
            "Per-batch delta lengths [B] INT32 for multi-batch guard",
            type_str="tensor(int32)",
        ),
        OpSchema.FormalParameter(
            name="kv_page_table",
            description=
            ("Canonical KV page table [B, 2, M] INT32: K page IDs are "
             "in [0, num_pages), V page IDs are in [num_pages, 2 * num_pages)"
             ),
            type_str="tensor(int32)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="present_key_value",
            description="Updated KV cache (aliased to past_key_value)",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float16)"],
            "KV cache data type (FP16 for now).",
        ),
    ],
)

# ---------------------------------------------------------------------------
# Gemma4 Audio Attention Plugin
# ---------------------------------------------------------------------------

_gemma4_audio_attention_plugin_schema = OpSchema(
    name="Gemma4AudioAttentionPlugin",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=
    ("Fused Gemma 4 audio chunked local attention (post-QKV, pre-output-proj). "
     "Per-dim Q scaling, fixed K scaling, content + rel-pos scores, "
     "tanh soft-cap, local mask, softmax, value mix."),
    inputs=[
        OpSchema.FormalParameter(
            name="q_raw",
            description="Raw query projections [B, S, H, D]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="k_raw",
            description="Raw key projections [B, S, H, D]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="v",
            description="Value projections [B, S, H, D]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="gamma",
            description="Per-dim learned query scale [D] (float32)",
            type_str="tensor(float)",
        ),
        OpSchema.FormalParameter(
            name="rel_key",
            description="Projected relative-position embeddings [P, H, D]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="valid",
            description="Audio validity mask [B, S] (bool)",
            type_str="tensor(bool)",
        ),
        OpSchema.FormalParameter(
            name="seq_len_carrier",
            description="Shape carrier [1] (int32)",
            type_str="tensor(int32)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="attn_output",
            description="Attention output [B, S, H, D]",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"],
            "Q/K/V/relKey/output data type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="chunk_size",
            type=OpSchema.AttrType.INT,
            description="Query block size (default 12)",
            required=True,
        ),
        OpSchema.Attribute(
            name="left_horizon",
            type=OpSchema.AttrType.INT,
            description="Effective left context (default 12)",
            required=True,
        ),
        OpSchema.Attribute(
            name="context_size",
            type=OpSchema.AttrType.INT,
            description="Gathered K/V context size (default 24)",
            required=True,
        ),
        OpSchema.Attribute(
            name="logit_cap",
            type=OpSchema.AttrType.FLOAT,
            description="Tanh soft-cap on logits (default 50.0)",
            required=True,
        ),
    ],
)

_ALL_CUSTOM_SCHEMAS: tuple[OpSchema, ...] = (
    _attention_plugin_schema,
    _vit_attention_plugin_schema,
    _trt_fp4_dynamic_quantize_schema,
    _trt_dequantize_linear_schema,
    _trt_mxfp8_dynamic_quantize_schema,
    _trt_mxfp8_dequantize_linear_schema,
    _int4_groupwise_gemm_schema,
    _qkv_concat_schema,
    _int4_groupwise_gemm_v2_schema,
    _nvfp4_a16_gemm_schema,
    _causal_conv1d_schema,
    _update_ssm_state_schema,
    _rotary_embedding_schema,
    _tensor_scatter_schema,
    _attention_trt_native_schema,
    _trt_ragged_attention_schema,
    _gated_delta_net_schema,
    _int4_moe_plugin_schema,
    _nvfp4_moe_plugin_schema,
    _nvfp4_a16_moe_plugin_schema,
    _nvfp4_moe_plugin_geforce_schema,
    _fp16_moe_plugin_schema,
    _fused_nvfp4_gemm_allreduce_plugin_schema,
    _dflash_target_kv_cache_update_schema,
    _gemma4_audio_attention_plugin_schema,
)

_registered_tensorrt_edgellm_schemas: bool = False


def register_tensorrt_edgellm_onnx_custom_schemas() -> None:
    """Register custom ONNX ops for :mod:`tensorrt_edgellm.dynamo_translations`. Idempotent."""
    global _registered_tensorrt_edgellm_schemas
    if _registered_tensorrt_edgellm_schemas:
        return
    for s in _ALL_CUSTOM_SCHEMAS:
        _safe_register_schema(s)
    _registered_tensorrt_edgellm_schemas = True
