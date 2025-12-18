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

import json
import os
import shutil
import time
from collections import namedtuple
from typing import Tuple

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
from safetensors import safe_open
from safetensors.torch import save_file

GEMMInfo = namedtuple("GEMMInfo", ["input", "output", "name", "weight_shape"])


def _find_matmul_node(quantize_linear_node: gs.Node) -> gs.Node:
    """
    Find the MatMul node after the quantize linear node. Usually it is 2-3 levels deep.
    """
    node = quantize_linear_node
    max_depth = 5
    depth = 0
    while node.op != "MatMul" and depth < max_depth:
        node = node.outputs[0].outputs[0]
        depth += 1
    if depth >= max_depth:
        raise ValueError(
            f"MatMul node not found after {max_depth} levels of quantization for {quantize_linear_node.name}. Please check the ONNX graph."
        )
    return node


def _find_weight_shape(gemm_node: gs.Node) -> gs.Constant:
    """
    Find the weight shape of the GEMM node. The weight shape is not intuitive because of the quantization and transpose nodes.
    """
    # Weights are always on the second of a GEMM node.
    node = gemm_node.inputs[1].inputs[0]
    max_depth = 5
    depth = 0
    num_transpose = 0
    while node.op != "DequantizeLinear" and node.op != "TRT_MXFP8DequantizeLinear" and depth < max_depth:
        if node.op == "Transpose":
            num_transpose += 1
        node = node.inputs[0].inputs[0]
        depth += 1
    if depth >= max_depth:
        raise ValueError(
            f"DequantizeLinear node not found above {max_depth} levels of GEMM for {gemm_node.name}. Please check the ONNX graph."
        )
    weight = node.inputs[0]
    if num_transpose % 2 == 1:
        weight_shape = (weight.shape[1], weight.shape[0])
    else:
        weight_shape = weight.shape
    return weight_shape


def _match_fp8_gemm(graph: gs.Graph):
    """
    Match FP8 GEMM nodes in the graph.
    """
    fp8_gemm_infos = []
    fp8_quantize_linear_nodes = [
        node for node in graph.nodes if node.op == "TRT_FP8QuantizeLinear"
    ]
    for node in fp8_quantize_linear_nodes:
        if node.inputs[0].inputs[0].op == "Cast":
            input_node = node.inputs[0].inputs[0].inputs[0]
        else:
            input_node = node.inputs[0]
        matmul_node = _find_matmul_node(node)
        weight_shape = _find_weight_shape(matmul_node)
        gemm_info = GEMMInfo(input=input_node,
                             output=matmul_node.outputs[0],
                             name=matmul_node.name,
                             weight_shape=weight_shape)
        fp8_gemm_infos.append(gemm_info)
    return fp8_gemm_infos


def _match_nvfp4_gemm(graph: gs.Graph):
    """
    Match NVFP4 GEMM nodes in the graph.
    """
    nvfp4_gemm_infos = []
    nvfp4_quantize_linear_nodes = [
        node for node in graph.nodes if node.op == "TRT_FP4DynamicQuantize"
    ]
    for node in nvfp4_quantize_linear_nodes:
        input_node = node.inputs[0]
        matmul_node = _find_matmul_node(node)
        weight_shape = _find_weight_shape(matmul_node)
        gemm_info = GEMMInfo(input=input_node,
                             output=matmul_node.outputs[0],
                             name=matmul_node.name,
                             weight_shape=weight_shape)
        nvfp4_gemm_infos.append(gemm_info)
    return nvfp4_gemm_infos


def _match_int4_gemm(graph: gs.Graph):
    """
    Match INT4 GEMM nodes in the graph.
    """
    int4_gemm_infos = []
    int4_gemm_nodes = [
        node for node in graph.nodes if node.op == "Int4GroupwiseGemmPlugin"
    ]
    for node in int4_gemm_nodes:
        # For AWQ, the input is smoothed by a Mul and a Cast node.
        if node.inputs[0].inputs[
                0].op == "Cast" and "input_quantizer" in node.inputs[0].inputs[
                    0].inputs[0].name:
            cast_node = node.inputs[0].inputs[0]
            mul_node = cast_node.inputs[0].inputs[0]
            input_node = mul_node.inputs[0]
        # For GPTQ, no smoothing is applied.
        else:
            input_node = node.inputs[0]
        weight_shape = (node.attrs["gemm_k"], node.attrs["gemm_n"])
        gemm_info = GEMMInfo(input=input_node,
                             output=node.outputs[0],
                             name=node.name,
                             weight_shape=weight_shape)
        int4_gemm_infos.append(gemm_info)
    return int4_gemm_infos


def _match_mxfp8_gemm(graph: gs.Graph):
    """
    Match MXFP8 GEMM nodes in the graph.
    """
    mxfp8_gemm_infos = []
    mxfp8_quantize_linear_nodes = [
        node for node in graph.nodes if node.op == "TRT_MXFP8DynamicQuantize"
    ]
    for node in mxfp8_quantize_linear_nodes:
        input_node = node.inputs[0]
        matmul_node = _find_matmul_node(node)
        weight_shape = _find_weight_shape(matmul_node)
        gemm_info = GEMMInfo(input=input_node,
                             output=matmul_node.outputs[0],
                             name=matmul_node.name,
                             weight_shape=weight_shape)
        mxfp8_gemm_infos.append(gemm_info)
    return mxfp8_gemm_infos


def _match_fp16_gemm(graph: gs.Graph):
    """
    Match FP16 GEMM nodes in the graph.
    """
    fp16_gemm_infos = []
    fp16_gemm_nodes = [node for node in graph.nodes if node.op == "MatMul"]
    for node in fp16_gemm_nodes:
        input_node = node.inputs[0]
        if not isinstance(node.inputs[1], gs.Constant):
            continue
        weight_shape = node.inputs[1].shape
        gemm_info = GEMMInfo(input=input_node,
                             output=node.outputs[0],
                             name=node.name,
                             weight_shape=weight_shape)
        fp16_gemm_infos.append(gemm_info)
    return fp16_gemm_infos


def _match_gemm_infos(graph: gs.Graph):
    """
    Match all GEMM nodes in the graph.
    """
    gemm_infos = []
    gemm_infos.extend(_match_fp8_gemm(graph))
    gemm_infos.extend(_match_nvfp4_gemm(graph))
    gemm_infos.extend(_match_int4_gemm(graph))
    gemm_infos.extend(_match_mxfp8_gemm(graph))
    gemm_infos.extend(_match_fp16_gemm(graph))
    return gemm_infos


# Helper functions for LoRA weight processing
def _load_adapter_config(config_path: str) -> Tuple[float, int]:
    """
    Load adapter config and return lora_alpha and r values.
    
    Args:
        config_path (str): Path to adapter_config.json
        
    Returns:
        Tuple[float, int]: (lora_alpha, r)
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['lora_alpha'], config['r']


def _process_tensor_name(key: str) -> str:
    """
    Process tensor name by removing 'base_model.model' prefix and ensuring it starts with 'model'.
    
    Args:
        key (str): Original tensor name
        
    Returns:
        str: Processed tensor name
    """
    if key.startswith('base_model.model.'):
        key = key[len('base_model.model.'):]
    if not key.startswith('model.'):
        key = 'model.' + key
    return key


def _should_keep_tensor(key: str) -> bool:
    """
    Check if tensor should be kept (exclude norm and lm_head tensors).
    
    Args:
        key (str): Tensor name
        
    Returns:
        bool: True if tensor should be kept
    """
    return 'norm' not in key and 'lm_head' not in key


def _process_tensor(tensor: torch.Tensor, key: str, lora_alpha: float,
                    r: int) -> torch.Tensor:
    """
    Process tensor according to requirements:
    1. Convert bf16 to fp16
    2. Multiply lora_B.weight by lora_alpha/r
    3. Ensure correct shapes for lora_A and lora_B
    
    Args:
        tensor (torch.Tensor): Input tensor
        key (str): Tensor name
        lora_alpha (float): LoRA alpha value
        r (int): LoRA rank
        
    Returns:
        torch.Tensor: Processed tensor
    """

    # Handle lora_B.weight multiplication
    if 'lora_B.weight' in key:
        tensor = tensor * (lora_alpha / r)

    # Ensure correct shapes
    if 'lora_A.weight' in key:
        if tensor.shape[-1] != r:
            tensor = tensor.transpose(-2, -1)
    elif 'lora_B.weight' in key:
        if tensor.shape[0] != r:
            tensor = tensor.transpose(-2, -1)

    # Convert to fp16
    tensor = tensor.to(torch.float16).contiguous()

    return tensor


# Main functions for external use
def insert_lora_and_save(onnx_dir: str):
    """
    Insert LoRA patterns into ONNX models.
    
    Args:
        onnx_dir (str): Directory containing the ONNX model (model.onnx and config.json)
        output_dir (str): Directory to save the modified ONNX model
        mode (str): LoRA insertion mode: 'dynamic' (default) or 'static'
        lora_weights_dir (str): Directory containing LoRA weights (required for static mode)
    """
    start_time = time.time()
    # Load ONNX model
    onnx_model_path = os.path.join(onnx_dir, "model.onnx")
    print(f"Loading original ONNX model from {onnx_model_path}...")

    # The LoRA model will share the same data as the base model
    onnx_model = onnx.load(onnx_model_path, load_external_data=False)
    graph = gs.import_onnx(onnx_model)

    # Insert dynamic LoRA patterns
    print("Inserting dynamic LoRA patterns...")
    # Track all GEMM nodes that need LoRA
    gemm_infos = _match_gemm_infos(graph)

    # Insert LoRA patterns for each GEMM
    for gemm_info in gemm_infos:
        input_tensor = gemm_info.input
        output_tensor = gemm_info.output
        gemm_name = gemm_info.name
        weight_shape = gemm_info.weight_shape
        k, n = weight_shape
        if "lm_head" in gemm_name:
            continue

        # Create dynamic input tensors for LoRA weights
        gemm_name_for_lora = gemm_name.replace("/", ".").rsplit(".", 1)[0][1:]

        lora_a = gs.Variable(f"{gemm_name_for_lora}.lora_A.weight",
                             dtype=np.float16,
                             shape=[k, f"{gemm_name_for_lora}.rank"])
        lora_b = gs.Variable(f"{gemm_name_for_lora}.lora_B.weight",
                             dtype=np.float16,
                             shape=[f"{gemm_name_for_lora}.rank", n])
        graph.inputs.extend([lora_a, lora_b])

        # First MatMul: input @ lora_A
        lora_mid = gs.Variable(f"{gemm_name}/lora_mid", dtype=np.float16)
        graph.layer(name=f"{gemm_name}/lora_matmul_A",
                    op="MatMul",
                    inputs=[input_tensor, lora_a],
                    outputs=[lora_mid])

        # Second MatMul: (input @ lora_A) @ lora_B
        lora_out = gs.Variable(f"{gemm_name}/lora_gemm_out", dtype=np.float16)
        graph.layer(name=f"{gemm_name}/lora_matmul_B",
                    op="MatMul",
                    inputs=[lora_mid, lora_b],
                    outputs=[lora_out])

        # Add LoRA output to original output
        final_output = gs.Variable(f"{gemm_name}/lora_add_output",
                                   dtype=np.float16)
        final_output.outputs = output_tensor.outputs.copy()
        graph.layer(name=f"{gemm_name}/lora_add",
                    op="Add",
                    inputs=[output_tensor, lora_out],
                    outputs=[final_output])

        # Update the output connections
        for out_node in final_output.outputs:
            if final_output not in out_node.inputs:
                out_node.inputs.append(final_output)
            if output_tensor in out_node.inputs:
                out_node.inputs.remove(output_tensor)

    graph.cleanup().toposort().fold_constants().cleanup()

    # Save modified ONNX model
    output_model_path = os.path.join(onnx_dir, "lora_model.onnx")
    print(f"Saving modified ONNX model to {output_model_path}...")

    modified_onnx_model = gs.export_onnx(graph)
    onnx.save_model(modified_onnx_model, output_model_path)

    end_time = time.time()
    print(f"LoRA model saved to {output_model_path}")
    print(f"LoRA insertion completed in {end_time - start_time:.2f}s")


def process_lora_weights_and_save(input_dir: str, output_dir: str):
    """
    Process LoRA weights according to specified requirements.
    
    Args:
        input_dir (str): Directory containing input adapter files
        output_dir (str): Directory where processed files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load adapter config
    config_path = os.path.join(input_dir, 'adapter_config.json')
    lora_alpha, r = _load_adapter_config(config_path)

    # Copy config file to output directory
    shutil.copy2(config_path, os.path.join(output_dir, 'config.json'))

    # Load safetensors
    safetensor_path = os.path.join(input_dir, 'adapter_model.safetensors')
    processed_tensors = {}

    try:
        with safe_open(safetensor_path, framework="pt") as f:
            for key in f.keys():
                # Skip unwanted tensors
                if not _should_keep_tensor(key):
                    continue

                # Process tensor name
                new_key = _process_tensor_name(key)

                # Load and process tensor
                tensor = f.get_tensor(key)
                processed_tensor = _process_tensor(tensor, key, lora_alpha, r)

                # Store processed tensor
                processed_tensors[new_key] = processed_tensor

                # Print tensor info
                print(f"\nTensor: {new_key}")
                print(f"Shape: {processed_tensor.shape}")
                print(f"Dtype: {processed_tensor.dtype}")
                print("-" * 50)

        # Save processed tensors
        output_path = os.path.join(output_dir,
                                   'processed_adapter_model.safetensors')
        save_file(processed_tensors, output_path)
        print(f"\nProcessed tensors saved to: {output_path}")
        print(
            f"Config file copied to: {os.path.join(output_dir, 'config.json')}"
        )

    except Exception as e:
        print(f"Error processing safetensor file: {e}")
