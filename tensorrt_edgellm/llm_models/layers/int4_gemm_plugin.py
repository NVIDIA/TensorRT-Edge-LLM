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
Int4GemmPlugin for TensorRT Integration

This module provides a custom TensorRT operation for Int4 Groupwise GEMM computation that can be
exported to ONNX format. It handles GPTQ (via direct onnx export translation) and AWQ (via onnx_graphsurgeon) quantized weights and translates them to the Int4GroupwiseGemmPlugin
operation during ONNX export.

The module contains:
- Int4GemmPluginModule: Custom module that replaces TorchQuantLinear for ONNX export
- int4_gemm_plugin: Dummy TensorRT operation for Int4 GEMM computation
- ONNX export utilities for the custom operation
"""

import math
from typing import Optional, Tuple

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
import torch.nn as nn
from onnx.defs import OpSchema
from torch.onnx import register_custom_op_symbolic, symbolic_helper
from torch.onnx.symbolic_helper import _get_tensor_sizes

from ...common import ONNX_OPSET_VERSION

# Define ONNX OpSchema for Int4GroupwiseGemmPlugin
int4_gemm_plugin_schema = OpSchema(
    name="Int4GroupwiseGemmPlugin",
    domain="trt",
    since_version=ONNX_OPSET_VERSION,
    doc="Custom TensorRT Int4 Groupwise GEMM plugin for GPTQ quantized models.",
    inputs=[
        OpSchema.FormalParameter(
            name="input",
            description="Input tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="qweight",
            description="Quantized weight tensor (int8)",
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="scales",
            description="Scale tensor (float16)",
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
            description="Group size for groupwise quantization",
            required=True,
        ),
    ],
)
onnx.defs.register_schema(int4_gemm_plugin_schema)


@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i")
def symbolic_int4_gemm_plugin(
    g: torch.onnx._internal.torchscript_exporter.jit_utils.GraphContext,
    input: torch._C.Value,
    qweight: torch._C.Value,
    scales: torch._C.Value,
    gemm_n: int,
    gemm_k: int,
    group_size: int,
):
    """Symbolic function for ONNX export of Int4GroupwiseGemmPlugin."""

    output = g.op(
        "trt::Int4GroupwiseGemmPlugin",
        input,
        qweight,
        scales,
        gemm_n_i=gemm_n,
        gemm_k_i=gemm_k,
        group_size_i=group_size,
    )

    # Set output type based on input type
    input_type = input.type()
    output_sizes = _get_tensor_sizes(input)[:-1] + [gemm_n]
    output.setType(input_type.with_sizes(output_sizes))

    return output


@torch.library.custom_op("trt::int4_gemm_plugin", mutates_args=())
def int4_gemm_plugin(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    gemm_n: int,
    gemm_k: int,
    group_size: int,
) -> torch.Tensor:
    """
    Dummy TensorRT operation for Int4 Groupwise GEMM computation.
    
    This operation is used during ONNX export to replace TorchQuantLinear modules
    with Int4GroupwiseGemmPlugin operations. The actual computation is handled
    by TensorRT at inference time.
    
    Args:
        input: Input tensor of shape (batch_size, seq_len, in_features)
        qweight: Quantized weight tensor of shape (N/2, K) in int8 format
        scales: Scale tensor of shape (K/group_size, N) in float16 format
        gemm_n: Output feature dimension
        gemm_k: Input feature dimension
        group_size: Group size for groupwise quantization
        
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, out_features)
        
    Raises:
        AssertionError: If input shapes or types are invalid
    """
    batch_size, seq_len, in_features = input.shape

    # Validate inputs
    assert in_features == gemm_k, f"Input features {in_features} must match gemm_k {gemm_k}"
    assert qweight.dtype == torch.int8, f"qweight must be int8, got {qweight.dtype}"
    assert scales.dtype == torch.float16, f"scales must be float16, got {scales.dtype}"
    assert qweight.shape[
        0] * 2 == gemm_n, f"qweight shape {qweight.shape} incompatible with gemm_n {gemm_n}"
    assert qweight.shape[
        1] == gemm_k, f"qweight shape {qweight.shape} incompatible with gemm_k {gemm_k}"
    assert scales.shape[
        0] == gemm_k // group_size, f"scales shape {scales.shape} incompatible with group_size {group_size}"
    assert scales.shape[
        1] == gemm_n, f"scales shape {scales.shape} incompatible with gemm_n {gemm_n}"

    # Dummy implementation for ONNX export - returns zeros with correct shape
    # This is not used in actual inference, only for ONNX export
    output = torch.zeros(batch_size,
                         seq_len,
                         gemm_n,
                         dtype=input.dtype,
                         device=input.device)

    return output


def unpack_int4_weights_gptq(qweight: torch.Tensor) -> torch.Tensor:
    """Unpack 4-bit GPTQ packed weights to an int16 matrix.
    
    Parameters
    ----------
    qweight : torch.Tensor
        Packed weight tensor of shape ``(K/8, N)`` and ``dtype=int32``.
        Here ``K`` is the input-feature dimension and ``N`` is the
        output-feature dimension. Each 32-bit integer stores eight
        consecutive 4-bit values column-wise.

    Returns
    -------
    torch.Tensor
        Unpacked weight tensor of shape ``(K, N)`` and ``dtype=int16``
        where every original 4-bit value is expanded to an individual
        int16 element.
    """
    pack_factor = 8
    pack_dtype_bits = 32
    target_dtype = torch.int16
    wf = torch.tensor(list(range(0, pack_dtype_bits, 4)),
                      dtype=torch.int32).unsqueeze(0).to(qweight.device)
    wf_unsqueeze_neg_one = wf.unsqueeze(-1).to(device=qweight.device)
    maxq = 15
    weight = torch.bitwise_and(
        torch.bitwise_right_shift(
            torch.unsqueeze(qweight, 1).expand(-1, pack_factor, -1),
            wf_unsqueeze_neg_one).to(target_dtype), maxq)
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

    return weight


def pack_intweights(unpacked_qweight: np.ndarray) -> np.ndarray:
    """Pack 4-bit GPTQ weights to a 16-bit matrix.
    
    Parameters
    ----------
    unpacked_qweight : np.ndarray
        Unpacked weight tensor of shape ``(N, K)`` and ``dtype=int16``.
        Here ``N`` is the output-feature dimension and ``K`` is the
        input-feature dimension.

    Returns
    -------
    np.ndarray
        Packed weight tensor of shape ``(N/4, K)`` and ``dtype=int16``.
    """
    interleave = 4
    kstride = 64
    N = unpacked_qweight.shape[0]
    K = unpacked_qweight.shape[1]

    Packed_Kernel = unpacked_qweight.reshape(N, K // 32, 32)
    # np.arange(32).reshape(4, 4, 2).transpose(1, 0, 2) => [0, 1, 8, 9, 16, 17, 24, 25, ...]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4,
                                          2).transpose(0, 1, 3, 2, 4)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 32)

    # reorder each 8 weights for fast dequantization
    # [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 8)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4,
                                          2).transpose(0, 1, 2, 4, 3)
    Packed_Kernel = Packed_Kernel.reshape(N, K)

    # interleaving every four rows
    Packed_Kernel = Packed_Kernel.reshape(N // interleave, interleave,
                                          K // kstride, kstride)
    # N // 4, K // 64, 4, 64
    Packed_Kernel = Packed_Kernel.transpose(0, 2, 1, 3)
    Packed_Kernel = Packed_Kernel.reshape(N // interleave, K // kstride,
                                          kstride, interleave)
    # Packing -> (N // 4, K // 64, 64)
    Packed_Kernel = (Packed_Kernel[..., 0]
                     | (Packed_Kernel[..., 1] << 4)
                     | (Packed_Kernel[..., 2] << 8)
                     | (Packed_Kernel[..., 3] << 12))
    # reshape to (N // 4, K), int16 format
    qweight = Packed_Kernel.reshape(N // interleave, K).astype(np.int16)

    return qweight


def gather_rows_by_gidx_order(
        weight: torch.Tensor, g_idx: torch.Tensor,
        group_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gather rows corresponding to positions where g_idx equals 0,1,...,group_size-1 in sequence.
    
    This function is adapted from the int4_gemm_plugin.py file.
    
    Args:
        weight: Tensor, shape (K, N)
        g_idx: Tensor, shape (K,)
        group_size: int
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (new_weight, permute_idx)
            - new_weight: Tensor, concatenated result, shape (K, N)
            - permute_idx: Tensor, permutation indices, shape (K,)
    """
    group_num = int(weight.shape[0] / group_size)
    assert group_num == torch.max(
        g_idx
    ) + 1, f"Group number {group_num} is not equal to max group index {torch.max(g_idx)} + 1"
    indices_list = []
    for i in range(group_num):
        indices = torch.nonzero(g_idx == i, as_tuple=False).squeeze(1)
        indices_list.append(indices)
    permute_idx = torch.cat(indices_list, dim=0)
    new_weight = weight.index_select(0, permute_idx)
    assert new_weight.shape[0] == weight.shape[
        0], f"Result shape {new_weight.shape} is not equal to weight shape {weight.shape}"
    return new_weight, permute_idx


class GatherWrapper(nn.Module):

    def __init__(self, module_to_wrap, permute_idx):
        super().__init__()
        self.module_to_wrap = module_to_wrap
        self.register_buffer('permute_idx', permute_idx)

    def forward(self, x):
        return self.module_to_wrap(x[..., self.permute_idx])


class Int4GemmPluginModule(nn.Module):
    """
    Custom module that replaces TorchQuantLinear in GPTQ quantization for ONNX export.
    
    This module takes the same parameters as TorchQuantLinear but uses the
    Int4GroupwiseGemmPlugin operation during ONNX export instead of the
    standard dequantization and matrix multiplication.
    
    All weights are pre-processed during initialization to avoid processing
    during the forward pass, which is important for ONNX export.
    """

    def __init__(
        self,
        bits: int,
        group_size: int,
        desc_act: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        **kwargs,
    ):
        super().__init__()

        self.bits = bits
        self.group_size = group_size if group_size != -1 else in_features
        self.desc_act = desc_act
        self.in_features = in_features
        self.out_features = out_features
        self.pack_dtype = pack_dtype

        # Register buffers for GPTQ weights
        # qweight: packed weights of shape (in_features // 32 * bits, out_features)
        self.register_buffer(
            "qweight",
            torch.zeros((in_features // 32 * bits, out_features),
                        dtype=pack_dtype),
        )
        # qzeros: zero points for quantization
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (
                    math.ceil(in_features / self.group_size),
                    out_features // 32 * bits,
                ),
                dtype=pack_dtype,
            ),
        )
        # scales: quantization scales
        self.register_buffer(
            "scales",
            torch.zeros(
                (math.ceil(in_features / self.group_size), out_features),
                dtype=torch.float16,
            ),
        )
        # g_idx: group indices for groupwise quantization
        self.register_buffer(
            "g_idx",
            torch.tensor([i // self.group_size for i in range(in_features)],
                         dtype=torch.int32),
        )

        # Pre-processed weights for Int4GroupwiseGemmPlugin
        # Shape: (out_features // 2, in_features) in int8 format
        self.register_buffer(
            "processed_qweight",
            torch.zeros((out_features // 2, in_features), dtype=torch.int8),
        )

        if bias:
            # bias: optional bias term
            self.register_buffer(
                "bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

        # Flag to track if weights have been processed for ONNX export
        self._weights_processed = False

    def _process_weights(self) -> Optional[torch.Tensor]:
        """
        Pre-process weights for Int4GroupwiseGemmPlugin format.
        This is called after loading weights to prepare them for ONNX export.
        """
        if self._weights_processed:
            return None

        permute_idx = None
        with torch.no_grad():
            # Process weights for Int4GroupwiseGemmPlugin format
            # First, unpack the packed GPTQ weights
            unpacked_qweight = unpack_int4_weights_gptq(self.qweight)

            # Handle group-wise quantization if needed
            # Reorder weights according to group indices for non-desc_act models
            if self.desc_act:
                unpacked_qweight, permute_idx = gather_rows_by_gidx_order(
                    unpacked_qweight, self.g_idx, self.group_size)

            # Transpose and pack weights using numpy
            unpacked_qweight_trans = unpacked_qweight.transpose(
                1, 0).contiguous()
            device = unpacked_qweight_trans.device
            np_unpacked_qweight_trans = unpacked_qweight_trans.cpu().numpy()
            # Pack weights to int16 format for efficient storage
            np_weight_int16 = pack_intweights(
                np_unpacked_qweight_trans)  # [N/4, K] int16
            # Convert to int8 format [N/2, K] for the plugin
            np_weight = np_weight_int16.view(np.int8).reshape(
                np_weight_int16.shape[0] * 2, np_weight_int16.shape[1])

            self.processed_qweight = (torch.tensor(
                np_weight.astype("int8")).to(device).contiguous()).to(
                    torch.int8)

        self._weights_processed = True
        return permute_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using Int4GroupwiseGemmPlugin.
        
        During ONNX export, this will be replaced with the Int4GroupwiseGemmPlugin operation.
        During normal PyTorch execution, this performs the standard GPTQ dequantization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, out_features)
        """
        # Ensure weights are processed for ONNX export
        if not self._weights_processed:
            self._process_weights()

        # Use the Int4GroupwiseGemmPlugin operation with pre-processed weights
        output = int4_gemm_plugin(
            input=x,
            qweight=self.processed_qweight,
            scales=self.scales,
            gemm_n=self.out_features,
            gemm_k=self.in_features,
            group_size=self.group_size,
        )

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        return output

    def load_state_dict_from_torch_quant_linear(
            self, torch_quant_linear: nn.Module) -> Optional[torch.Tensor]:
        """
        Load state dict from a TorchQuantLinear module.
        
        This method reuses the original module's data without copying when possible.
        
        Args:
            torch_quant_linear: TorchQuantLinear module to copy weights from
        """
        # Reuse the original module's data directly without copying
        self.qweight = torch_quant_linear.qweight
        self.qzeros = torch_quant_linear.qzeros
        self.scales = torch_quant_linear.scales
        self.g_idx = torch_quant_linear.g_idx
        if self.bias is not None and torch_quant_linear.bias is not None:
            self.bias = torch_quant_linear.bias

        # Process the weights for ONNX export after loading
        return self._process_weights()


def register_int4_gemm_plugin_onnx_symbolic_functions() -> None:
    """Register symbolic functions for ONNX export."""

    # Register our custom symbolic functions
    register_custom_op_symbolic("trt::int4_gemm_plugin",
                                symbolic_int4_gemm_plugin, ONNX_OPSET_VERSION)

    print("Registered ONNX symbolic functions for custom Int4GemmPlugin")


def replace_torch_quant_linear_with_plugin(model: nn.Module) -> nn.Module:
    """
    Replace all TorchQuantLinear modules in a model with Int4GemmPluginModule.
    
    This function reuses the original module's data without copying when possible.
    
    Args:
        model: PyTorch model containing TorchQuantLinear modules
        
    Returns:
        nn.Module: Model with TorchQuantLinear modules replaced by Int4GemmPluginModule
    """
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

    for name, module in model.named_modules():
        if isinstance(module, TorchQuantLinear):
            # Create new Int4GemmPluginModule with same parameters
            new_module = Int4GemmPluginModule(
                bits=module.bits,
                group_size=module.group_size,
                desc_act=module.desc_act,
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                pack_dtype=torch.int8,
            )

            # Load weights from original module (reuses data without copying)
            permute_idx = new_module.load_state_dict_from_torch_quant_linear(
                module)

            final_module = new_module
            if module.desc_act:
                assert permute_idx is not None, "Permute index should not be None for desc_act models"
                final_module = GatherWrapper(new_module, permute_idx)

            # Replace the module in the model
            parent = model
            if '.' in name:
                parent_name, module_name = name.rsplit('.', 1)
                parent = dict(model.named_modules())[parent_name]
            else:
                module_name = name

            setattr(parent, module_name, final_module)

    return model


def int4_dq_gemm_to_plugin(graph: gs.Graph) -> gs.Graph:
    """
    This function starts with ONNX graph containing DequantizeLinear + MatMul patterns
    and replaces them with Int4GemmPlugin operations.

    Parameters:
        graph: gs.Graph
            The ONNX graph containing DequantizeLinear + MatMul patterns

    Returns:
        gs.Graph: Graph with DequantizeLinear + MatMul patterns replaced by Int4GemmPlugin layers.
    """
    from modelopt.onnx.quantization.gs_patching import patch_gs_modules
    patch_gs_modules()

    # Find all DequantizeLinear nodes
    dequantize_nodes = []
    for node in graph.nodes:
        if node.op == "DequantizeLinear":
            dequantize_nodes.append(node)

    for dequantize_node in dequantize_nodes:
        assert len(dequantize_node.outputs
                   ) == 1, "DequantizeLinear should have exactly 1 output"
        dequantize_output = dequantize_node.outputs[0]
        assert len(
            dequantize_output.outputs
        ) == 1, "DequantizeLinear output should have exactly 1 output"
        matmul_node = dequantize_output.outputs[0]
        assert matmul_node.op == "MatMul", "DequantizeLinear output should be a MatMul"
        assert len(
            matmul_node.inputs) == 2, "MatMul should have exactly 2 inputs"

        # Find which input is the dequantized output and which is the other input
        weight_idx = None
        input_idx = None
        for i, input_tensor in enumerate(matmul_node.inputs):
            if input_tensor == dequantize_output:
                weight_idx = i
                input_idx = 1 - i
                break
        assert weight_idx is not None, "Weight index should not be None"

        # Get the other input (this will be the plugin input)
        plugin_input = matmul_node.inputs[input_idx]
        plugin_input.dtype = np.float16

        # Get the output of MatMul
        assert len(
            matmul_node.outputs) == 1, "MatMul should have exactly 1 output"
        matmul_output = matmul_node.outputs[0]
        matmul_output.dtype = np.float16

        # Extract weights and scales from DequantizeLinear
        assert len(dequantize_node.inputs
                   ) == 2, "DequantizeLinear should have exactly 2 inputs"

        # DequantizeLinear inputs: [x, x_scale]
        # We need the weights (x) and scales (x_scale)
        weights_tensor = dequantize_node.inputs[0]
        scales_tensor = dequantize_node.inputs[1]

        if not isinstance(weights_tensor, gs.Constant) or not isinstance(
                scales_tensor, gs.Constant):
            raise ValueError("Weights and scales should be constants")

        # Get the weight and scale data from the DequantizeLinear node
        gemm_k, gemm_n = weights_tensor.shape
        group_size = dequantize_node.attrs['block_size']
        weights_data = weights_tensor.values
        scales_data = scales_tensor.values
        assert scales_data.shape[
            0] == gemm_k // group_size, "Scales should have shape [K/group_size, N]"
        assert scales_data.shape[
            1] == gemm_n, "Scales should have shape [K/group_size, N]"
        assert scales_data.dtype == np.float16, "Scales should be in float16 format"

        # Unpack the weights to int16 format (add 8 to convert from [-8,7] to [0,15] range)
        unpacked_qweight = weights_data.astype(np.int16) + 8
        # Transpose to [gemm_n, gemm_k] for the packing function
        unpacked_qweight = unpacked_qweight.transpose(1, 0)

        # Pack the weights for the plugin format
        packed_weights_int16 = pack_intweights(
            unpacked_qweight)  # [gemm_n//4, gemm_k] int16
        # Convert to int8 format [gemm_n//2, gemm_k] for the plugin
        packed_weights = packed_weights_int16.view(np.int8).reshape(
            packed_weights_int16.shape[0] * 2, packed_weights_int16.shape[1])

        # Create constants for the plugin inputs
        plugin_weights = gs.Constant(f"{matmul_node.name}/processed_qweight",
                                     packed_weights)
        plugin_scales = gs.Constant(f"{matmul_node.name}/scales", scales_data)

        # Plugin attributes for Int4GroupwiseGemmPlugin
        gemm_attrs = {
            "gemm_n": gemm_n,
            "gemm_k": gemm_k,
            "group_size": group_size,
        }

        # Remove the old nodes and their connections
        # Remove connections from input to MatMul
        plugin_input.outputs.remove(matmul_node)
        dequantize_node.outputs[0].outputs.remove(matmul_node)
        matmul_node.outputs[0].inputs.remove(matmul_node)

        # Remove nodes from graph
        graph.nodes.remove(dequantize_node)
        graph.nodes.remove(matmul_node)

        # Create the Int4GroupwiseGemmPlugin node
        graph.layer(name=f"{matmul_node.name}Plugin",
                    op="Int4GroupwiseGemmPlugin",
                    inputs=[plugin_input, plugin_weights, plugin_scales],
                    outputs=[matmul_output],
                    attrs=gemm_attrs)
    # Update Cast nodes around Add/Concat to fp16
    # For inputs: if there's a Cast node, ensure it casts to fp16
    # For the Add/Concat node itself: ensure inputs/outputs are fp16
    # Do not remove any nodes
    for node in [n for n in graph.nodes if n.op in ["Add", "Concat"]]:
        # Update Cast nodes feeding into Add/Concat to cast to fp16
        for inp in node.inputs:
            if not isinstance(inp, gs.Constant):
                inp.dtype = np.float16
                # If input comes from a Cast node, update it to cast to fp16
                if len(inp.inputs) == 1 and inp.inputs[0].op == "Cast":
                    cast_node = inp.inputs[0]
                    cast_node.attrs["to"] = onnx.TensorProto.FLOAT16

        # Update outputs to fp16
        for out in node.outputs:
            out.dtype = np.float16

    # Clean up the graph and ensure topological ordering
    graph.cleanup().toposort().cleanup()
    return graph
