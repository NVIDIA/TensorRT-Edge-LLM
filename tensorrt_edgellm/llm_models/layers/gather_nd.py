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
Custom GatherND Plugin for TensorRT Integration

There is no torch op that translates to GatherND. This module provides a custom ONNX operation for gather_nd computation that can be
exported to ONNX format. It implements a gather operation that selects elements from
a tensor based on indices along specified dimensions.

The module contains:
- custom_gather_nd: Custom ONNX operation for gather_nd computation
- symbolic_gather_nd: Symbolic function for ONNX export
- register_gather_nd_onnx_symbolic_functions: Function to register the custom operation with ONNX
"""

import torch
from torch.onnx import symbolic_helper

from ...common import ONNX_OPSET_VERSION


@symbolic_helper.parse_args("v", "v", "i")
def symbolic_gather_nd(
    g: torch.onnx._internal.torchscript_exporter.jit_utils.GraphContext,
    value: torch._C.Value,
    indices: torch._C.Value,
    batch_dims: int,
):
    """
    Symbolic function for ONNX export.
    
    This function defines how to convert to ONNX GatherND.
    For ONNX GatherND with batch_dims=1, indices must have shape:
    [batch_size, num_indices, num_index_dims]
    where num_index_dims is the number of dimensions to index (1 for selecting along seq_len).
    
    Args:
        g: ONNX graph being built
        value: Input tensor
        indices: Indices tensor with dtype int64, shape [batch_size, num_indices]
        batch_dims: Number of batch dimensions (default: 1)
        
    Returns:
        ONNX GatherND operation
    """
    # ONNX GatherND requires indices to have an extra dimension for the number of axes to index
    # indices shape: [batch_size, num_indices] -> [batch_size, num_indices, 1]
    unsqueeze_axes = g.op("Constant",
                          value_t=torch.tensor([-1], dtype=torch.int64))
    indices_expanded = g.op("Unsqueeze", indices, unsqueeze_axes)
    return g.op("GatherND", value, indices_expanded, batch_dims_i=batch_dims)


@torch.library.custom_op("trt::gather_nd", mutates_args=())
def custom_gather_nd(
    value: torch.Tensor,
    indices: torch.Tensor,
    batch_dims: int = 1,
) -> torch.Tensor:
    """
    Custom ONNX operation for gather_nd computation.
    
    This operation implements a gather operation that selects elements from value
    based on indices. It's designed to work with batch_dims=1 for selecting specific
    tokens from a sequence.
    
    Args:
        value: Input tensor
        indices: Indices tensor with dtype int64
        batch_dims: Number of batch dimensions (default: 1)
        
    Returns:
        torch.Tensor: Gathered output tensor
        
    Raises:
        AssertionError: If input shapes or types are invalid
    """
    batch_size, seq_len, hidden_size = value.shape
    indices_batch_size, num_tokens = indices.shape

    # Validate inputs
    assert batch_size == indices_batch_size, f"Batch sizes must match: {batch_size} vs {indices_batch_size}"
    assert indices.dtype == torch.int64, f"indices must be int64, got {indices.dtype}"
    assert batch_dims == 1, f"Only batch_dims=1 is supported, got {batch_dims}"

    # Validate that all indices are within bounds
    assert torch.all(indices >= 0) and torch.all(indices < seq_len), \
        f"All indices must be in range [0, {seq_len})"

    # Use torch.gather for the actual computation
    # For batch_dims=1, we need to gather along dimension 1 (seq_len)
    # Expand last_token_ids to match hidden_states dimensions
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, hidden_size)

    # Gather along dimension 1
    gathered_output = torch.gather(value, 1, indices_expanded)

    return gathered_output


def register_gather_nd_onnx_symbolic_functions() -> None:
    """Register symbolic functions for ONNX export."""
    from torch.onnx import register_custom_op_symbolic

    # Register our custom symbolic functions
    register_custom_op_symbolic("trt::gather_nd", symbolic_gather_nd,
                                ONNX_OPSET_VERSION)

    print("Registered ONNX symbolic functions for custom gather_nd")
