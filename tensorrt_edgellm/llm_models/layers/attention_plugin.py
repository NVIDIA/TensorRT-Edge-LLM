# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Dummy Attention Plugin for TensorRT Integration

This module provides a custom TensorRT operation for attention computation that can be
exported to ONNX format. It includes RoPE (Rotary Position Embedding) application,
KV cache management, and attention computation in a single fused operation.

The module contains:
- attention_plugin: Dummy TensorRT operation for attention computation, this is not used in the actual inference.
- ONNX export utilities for the custom operation
"""

from typing import Optional, Tuple

import onnx
import torch
from onnx.defs import OpSchema
from torch.onnx import register_custom_op_symbolic, symbolic_helper
from torch.onnx.symbolic_helper import _get_tensor_sizes

from ...common import ONNX_OPSET_VERSION

# Define ONNX OpSchema for AttentionPlugin
attention_plugin_schema = OpSchema(
    name="AttentionPlugin",
    domain="trt",
    since_version=ONNX_OPSET_VERSION,
    doc=
    "Custom TensorRT attention plugin with RoPE, KV cache, and attention computation.",
    inputs=[
        OpSchema.FormalParameter(
            name="qkv",
            description="Concatenated QKV tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="past_key_value",
            description="KV cache tensor",
            type_str="T",
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
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="attn_output",
            description="Attention output tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="present_key_value",
            description=
            "Updated KV cache tensor with dynamic shape [batch_size, 2, num_kv_heads, present_kv_cache_len, head_size]",
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
            description="Whether to enable tree attention (0 or 1)",
            required=True,
        ),
    ],
)
onnx.defs.register_schema(attention_plugin_schema)


@symbolic_helper.parse_args("v", "v", "v", "v", "v", "i", "i", "b", "i", "v",
                            "v")
def symbolic_attention_plugin(
    g: torch.onnx._internal.torchscript_exporter.jit_utils.GraphContext,
    qkv: torch._C.Value,
    past_key_value: torch._C.Value,
    context_lengths: torch._C.Value,
    rope_rotary_cos_sin: torch._C.Value,
    kvcache_start_index: torch._C.Value,
    num_q_heads: torch._C.Value,
    num_kv_heads: torch._C.Value,
    enable_tree_attention: torch._C.Value,
    head_size: torch._C.Value,
    attention_mask: Optional[torch._C.Value] = None,
    position_ids: Optional[torch._C.Value] = None,
):
    """Custom attention plugin operation for ONNX export."""

    # Build inputs list - kvcache_start_index is now always required
    inputs = [
        qkv, past_key_value, context_lengths, rope_rotary_cos_sin,
        kvcache_start_index
    ]
    if enable_tree_attention:
        assert attention_mask is not None and attention_mask.type().kind(
        ) != 'NoneType', "attention_mask should be provided for tree attention"
        assert position_ids is not None and position_ids.type().kind(
        ) != 'NoneType', "position_ids should be provided for tree attention"
        inputs.append(attention_mask)
        inputs.append(position_ids)

    qkv_type = qkv.type()
    past_key_value_type = past_key_value.type()
    attn_output, present_key_value = g.op(
        "trt::AttentionPlugin",
        *inputs,
        num_q_heads_i=num_q_heads,
        num_kv_heads_i=num_kv_heads,
        head_size_i=head_size,
        enable_tree_attention_i=1 if enable_tree_attention else 0,
        outputs=2)

    qkv_sizes = _get_tensor_sizes(qkv)
    attn_output_sizes = qkv_sizes[:-1] + [num_q_heads, head_size]
    attn_output.setType(qkv_type.with_sizes(attn_output_sizes))

    # KV Cache output has the same shape as input past_key_value except for dimension 3 (sequence length)
    # Shape: [batch_size, 2, num_kv_heads, present_kv_cache_len (dynamic), head_size]
    past_kv_sizes = _get_tensor_sizes(past_key_value)
    present_key_value.setType(past_key_value_type.with_sizes(past_kv_sizes))

    return attn_output, present_key_value


@torch.library.custom_op("trt::attention_plugin", mutates_args=())
def attention_plugin(
    qkv: torch.Tensor,
    past_key_value: torch.Tensor,
    context_lengths: torch.Tensor,
    rope_rotary_cos_sin: torch.Tensor,
    kvcache_start_index: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    enable_tree_attention: bool,
    head_size: int,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dummy TensorRT operation for attention computation, this is not used in the actual inference.
    
    This operation wraps the logic after v_proj and before o_proj into a single 
    AttentionPlugin operation during ONNX export. It handles RoPE application,
    KV cache management, and attention computation in a fused manner.
    
    Args:
        qkv: Concatenated QKV tensor of shape (batch_size, seq_len, num_q_heads * head_size + 2 * num_kv_heads * head_size)
        past_key_value: KV cache tensor of shape (batch_size, 2, num_kv_heads, past_len, head_size)
        rope_rotary_cos_sin: RoPE tensor of shape (batch_size, seq_len, rotary_dim) containing cos and sin values
        context_lengths: Context length tensor of shape (batch_size,) indicating current position in cache
        kvcache_start_index: Start index of KV cache of shape (kv_cache_start_batch_size,), required
        num_q_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        enable_tree_attention: Whether to enable tree attention
        head_size: Size of each attention head
        attention_mask: Attention mask of shape (batch_size, seq_len, seq_len + past_len), optional
        position_ids: Position IDs tensor of shape (batch_size, seq_len), optional
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Attention output tensor and updated KV cache
            - Attention output: shape (batch_size, seq_len, num_q_heads * head_size)
            - Updated KV cache: shape (batch_size, 2, num_kv_heads, present_kv_cache_len, head_size) with dynamic shapes
        
    Raises:
        AssertionError: If enable_tree_attention is True but required tensors are missing
    """
    if enable_tree_attention:
        assert attention_mask is not None, "attention_mask should be provided for tree attention"
        assert position_ids is not None, "position_ids should be provided for tree attention"

    batch_size, seq_len, qkv_size = qkv.shape
    assert head_size * (
        num_q_heads + 2 * num_kv_heads
    ) == qkv_size, f"qkv_size {qkv_size} should be equal to head_size * (num_q_heads + 2 * num_kv_heads) {head_size * (num_q_heads + 2 * num_kv_heads)}"
    assert past_key_value.shape[
        0] == batch_size, f"batch_size of kv_cache {past_key_value.shape[0]} should be equal to batch_size of qkv {batch_size}"
    assert past_key_value.shape[
        1] == 2, f"kv_cache {past_key_value.shape[1]} should have 2 tensors"
    assert past_key_value.shape[
        2] == num_kv_heads, f"num_kv_heads of kv_cache {past_key_value.shape[2]} should be equal to num_kv_heads of qkv {num_kv_heads}"
    assert past_key_value.shape[
        4] == head_size, f"head_size of kv_cache {past_key_value.shape[4]} should be equal to head_size of qkv {head_size}"

    assert qkv.dtype == torch.float16, f"qkv {qkv.dtype} should be in float16"
    assert past_key_value.dtype == torch.float16, f"past_key_value {past_key_value.dtype} should be in float16"

    # Dummy implementation for ONNX export, this is not used in the actual inference
    attn_output = torch.zeros(batch_size,
                              seq_len,
                              num_q_heads,
                              head_size,
                              dtype=qkv.dtype,
                              device=qkv.device)

    return attn_output, past_key_value.clone()


def register_attention_plugin_onnx_symbolic_functions() -> None:
    """Register symbolic functions for ONNX export."""

    # Register our custom symbolic functions
    register_custom_op_symbolic("trt::attention_plugin",
                                symbolic_attention_plugin, ONNX_OPSET_VERSION)

    print("Registered ONNX symbolic functions for custom attention plugin")
