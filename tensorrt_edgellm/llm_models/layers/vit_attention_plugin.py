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
Dummy ViT Attention Plugin for TensorRT Integration

This mirrors attention_plugin.py for the LLM AttentionPlugin, but targets the
ViTAttentionPlugin. The Python custom op is a dummy export-time operation: it is
not used for real inference. During ONNX export, the symbolic function emits a
trt-domain ViTAttentionPlugin node that TensorRT can map to the registered C++
plugin.

The module contains:
- vit_attention_plugin: Dummy TensorRT operation for ViT attention.
- ONNX export utilities for the custom operation.
"""

import onnx
import torch
from onnx.defs import OpSchema
from torch.onnx import register_custom_op_symbolic, symbolic_helper
from torch.onnx.symbolic_helper import _get_tensor_sizes

from ...common import ONNX_OPSET_VERSION

# Define ONNX OpSchema for ViTAttentionPlugin.
vit_attention_plugin_schema = OpSchema(
    name="ViTAttentionPlugin",
    domain="trt",
    since_version=ONNX_OPSET_VERSION,
    doc="Custom TensorRT ViT attention plugin.",
    inputs=[
        OpSchema.FormalParameter(
            name="qkv",
            description="Fused QKV tensor [B, S, 3 * num_heads * head_size]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="cos",
            description="RoPE cosine tensor [S, head_size]",
            type_str="T_Rope",
        ),
        OpSchema.FormalParameter(
            name="sin",
            description="RoPE sine tensor [S, head_size]",
            type_str="T_Rope",
        ),
        OpSchema.FormalParameter(
            name="attention_mask",
            description="Additive attention mask [1|B|B*H, S, S]",
            type_str="T",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="attn_output",
            description="Attention output tensor [B, S, num_heads * head_size]",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"],
            "Input and output data type.",
        ),
        (
            "T_Rope",
            ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"],
            "RoPE tensor data type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="num_heads",
            type=OpSchema.AttrType.INT,
            description="Number of attention heads.",
            required=True,
        ),
        OpSchema.Attribute(
            name="head_size",
            type=OpSchema.AttrType.INT,
            description="Size of each attention head.",
            required=True,
        ),
        OpSchema.Attribute(
            name="qkv_fused",
            type=OpSchema.AttrType.INT,
            description="Whether the input projection is fused QKV.",
            required=True,
        ),
    ],
)
onnx.defs.register_schema(vit_attention_plugin_schema)


@symbolic_helper.parse_args("v", "v", "v", "v", "i", "i", "i")
def symbolic_vit_attention_plugin(
    g: torch.onnx._internal.torchscript_exporter.jit_utils.GraphContext,
    qkv: torch._C.Value,
    cos: torch._C.Value,
    sin: torch._C.Value,
    attention_mask: torch._C.Value,
    num_heads: int,
    head_size: int,
    qkv_fused: int,
):
    """Custom ViT attention plugin operation for ONNX export."""
    attn_output = g.op(
        "trt::ViTAttentionPlugin",
        qkv,
        cos,
        sin,
        attention_mask,
        num_heads_i=num_heads,
        head_size_i=head_size,
        qkv_fused_i=qkv_fused,
    )

    qkv_type = qkv.type()
    qkv_sizes = _get_tensor_sizes(qkv)
    if qkv_sizes is not None and len(qkv_sizes) >= 2:
        attn_output.setType(qkv_type.with_sizes(qkv_sizes[:-1] + [num_heads * head_size]))

    return attn_output


@torch.library.custom_op("trt::vit_attention_plugin", mutates_args=())
def vit_attention_plugin(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
    num_heads: int,
    head_size: int,
    qkv_fused: int = 1,
) -> torch.Tensor:
    """
    Dummy TensorRT operation for ViT attention, not used in actual inference.

    This operation wraps the attention math after QKV projection and before the
    output projection into a single ViTAttentionPlugin operation during ONNX
    export.

    Args:
        qkv: Fused QKV tensor of shape [batch_size, seq_len, 3 * num_heads * head_size].
        cos: RoPE cosine tensor of shape [seq_len, head_size].
        sin: RoPE sine tensor of shape [seq_len, head_size].
        attention_mask: Additive attention mask of shape [1|B|B*H, seq_len, seq_len].
        num_heads: Number of attention heads.
        head_size: Size of each attention head.
        qkv_fused: Whether QKV is fused.

    Returns:
        Attention output tensor of shape [batch_size, seq_len, num_heads * head_size].
    """
    batch_size, seq_len, qkv_size = qkv.shape
    assert qkv_fused == 1, "ViTAttentionPlugin currently expects fused QKV input"
    assert (
        qkv_size == 3 * num_heads * head_size
    ), f"qkv_size {qkv_size} should equal 3 * num_heads * head_size {3 * num_heads * head_size}"
    assert qkv.dtype == torch.float16, f"qkv {qkv.dtype} should be in float16"

    return torch.zeros(
        batch_size,
        seq_len,
        num_heads * head_size,
        dtype=qkv.dtype,
        device=qkv.device,
    )


def register_vit_attention_plugin_onnx_symbolic_functions() -> None:
    """Register symbolic functions for ONNX export."""
    register_custom_op_symbolic(
        "trt::vit_attention_plugin",
        symbolic_vit_attention_plugin,
        ONNX_OPSET_VERSION,
    )

    print("Registered ONNX symbolic functions for custom ViT attention plugin")
