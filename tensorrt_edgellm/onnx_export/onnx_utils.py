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

import copy
import os
import time

import onnx
import onnx_graphsurgeon as gs
import torch
import torch.nn as nn
from modelopt.onnx.llm_export_utils.surgeon_utils import fold_fp8_qdq_to_dq
from modelopt.onnx.quantization.qdq_utils import (fp4qdq_to_2dq,
                                                  quantize_weights_to_int4,
                                                  quantize_weights_to_mxfp8)

from ..common import ONNX_OPSET_VERSION
from ..llm_models.layers.int4_gemm_plugin import int4_dq_gemm_to_plugin
from ..llm_models.models.llm_model import EdgeLLMModelForCausalLM


def is_int4_awq_quantized(model: nn.Module) -> bool:
    """Check if the model is quantized in INT4 mode."""
    for _, module in model.named_modules():
        if (hasattr(module, "input_quantizer")
                and hasattr(module, "weight_quantizer")
                and module.weight_quantizer._num_bits == 4
                and module.input_quantizer._disabled):
            return True
    return False


def is_fp4_quantized(model: nn.Module) -> bool:
    """Check if the model is quantized in NVFP4 mode."""
    for _, module in model.named_modules():
        if (hasattr(module, "input_quantizer")
                and module.input_quantizer.block_sizes
                and module.input_quantizer.block_sizes.get("scale_bits",
                                                           None) == (4, 3)):
            return True
    return False


def is_mxfp8_quantized(model: nn.Module) -> bool:
    """Check if the model is quantized in MXFP8 mode."""
    for _, module in model.named_modules():
        if (hasattr(module, "input_quantizer")
                and module.input_quantizer.block_sizes
                and module.input_quantizer.block_sizes.get("scale_bits",
                                                           None) == (8, 0)):
            return True
    return False


def is_fp8_quantized(model: nn.Module) -> bool:
    """Check if the model is quantized in FP8 mode."""
    for _, module in model.named_modules():
        if (hasattr(module, "input_quantizer")
                and module.input_quantizer._num_bits == (4, 3)
                and hasattr(module, "weight_quantizer")
                and module.weight_quantizer._num_bits == (4, 3)):
            return True
    return False


def untie_nvfp4_lm_head_initializer(model: onnx.ModelProto) -> onnx.ModelProto:
    """Untie the weights of the nvFP4 quantized LM head from embed_tokens.weight.
    """
    LM_HEAD_WEIGHT_NAME = "lm_head.weight"
    EMBED_TOKENS_WEIGHT_NAME = "embed_tokens.weight"

    lmhead_weight_quantizer = None
    for node in model.graph.node:
        if node.name == "/lm_head/weight_quantizer/TRT_FP4QDQ":
            lmhead_weight_quantizer = node
            break
    if lmhead_weight_quantizer is None:
        raise ValueError(
            "Target node '/lm_head/weight_quantizer/TRT_FP4QDQ' not found in model.graph.node"
        )

    # If tied to embed, create a duplicate initializer and rewire
    if lmhead_weight_quantizer.input and EMBED_TOKENS_WEIGHT_NAME in lmhead_weight_quantizer.input[
            0]:

        # Find the initializer for embed_tokens.weight
        embed_init = None
        for init in model.graph.initializer:
            if EMBED_TOKENS_WEIGHT_NAME in init.name:
                embed_init = init
                break
        if embed_init is None:
            raise ValueError(
                "Initializer containing 'embed_tokens.weight' not found in model.graph.initializer, cannot untie lm_head weights"
            )

        print(
            f"Untying lm_head weights from {lmhead_weight_quantizer.input[0]}, creating a duplicate initializer {LM_HEAD_WEIGHT_NAME}"
        )
        new_init = copy.deepcopy(embed_init)
        new_init.name = LM_HEAD_WEIGHT_NAME
        model.graph.initializer.append(new_init)
        lmhead_weight_quantizer.input[0] = LM_HEAD_WEIGHT_NAME

    return model


def fix_model_int4_output_dtypes(
        onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """Fix data types for model outputs.
    In modelopt int4 post-processing, some Cast nodes are converted to FP16 instead of FP32 and hidden_states are converted to FP32 instead of FP16, so we need to fix them manually.
    See: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/0.37.0/modelopt/onnx/quantization/qdq_utils.py#L1050
    
    Ensures:
    1. For cast->logits or cast->logsoftmax->logits patterns, both cast and logits are FP32
    2. For hidden_states output, it is FP16
    
    Args:
        onnx_model: The ONNX model to fix
    
    Returns:
        The modified ONNX model
    """
    graph = onnx_model.graph

    # Build a map from output name to producer node
    output_to_node = {}
    for node in graph.node:
        for output in node.output:
            output_to_node[output] = node

    # Build a map from output name to graph output
    graph_outputs = {output.name: output for output in graph.output}

    # Helper to update Cast node's "to" attribute
    def set_cast_dtype(node, dtype):
        for attr in node.attribute:
            if attr.name == "to":
                attr.i = dtype
                return

    # Fix logits output to FP32
    if "logits" in graph_outputs:
        logits = graph_outputs["logits"]
        producer = output_to_node.get(logits.name)

        # Check for cast->logsoftmax->logits
        if producer and producer.op_type == "LogSoftmax":
            cast_node = output_to_node.get(producer.input[0])
            if cast_node and cast_node.op_type == "Cast":
                print("Found cast->logsoftmax->logits, ensuring FP32")
                set_cast_dtype(cast_node, 1)
        # Check for cast->logits
        elif producer and producer.op_type == "Cast":
            print("Found cast->logits, ensuring FP32")
            set_cast_dtype(producer, 1)

        # Set logits output type to FP32
        logits.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

    # Fix hidden_states output to FP16
    if "hidden_states" in graph_outputs:
        hidden_states = graph_outputs["hidden_states"]
        producer = output_to_node.get(hidden_states.name)

        if hidden_states.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            print("hidden_states is already FP16")
        else:
            # If producer is Cast, just update it
            if producer and producer.op_type == "Cast":
                print("Updating existing Cast to FP16 for hidden_states")
                set_cast_dtype(producer, 10)
            else:
                # Insert new Cast node
                print("Inserting Cast to FP16 for hidden_states")
                intermediate = f"{hidden_states.name}_pre_fp16"

                # Rename producer's output
                if producer:
                    for i, out in enumerate(producer.output):
                        if out == hidden_states.name:
                            producer.output[i] = intermediate

                # Add Cast node
                cast = onnx.helper.make_node(
                    "Cast",
                    inputs=[intermediate],
                    outputs=[hidden_states.name],
                    to=10,
                    name=f"{hidden_states.name}_cast_fp16")
                graph.node.append(cast)

            # Set hidden_states output type to FP16
            hidden_states.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16

    return onnx_model


def export_onnx(model, inputs, output_dir, input_names, output_names,
                dynamic_axes):
    '''
    Export the model to ONNX format.
    Args:
        model: The model to export
        inputs: The inputs to the model
        output_dir: The directory to save the ONNX model
        input_names: The names of the input tensors
        output_names: The names of the output tensors
        dynamic_axes: The dynamic axes of the model
    '''
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = f'{output_dir}/model.onnx'
    with torch.inference_mode():
        torch.onnx.export(model,
                          inputs,
                          onnx_path,
                          export_params=True,
                          dynamic_axes=dynamic_axes,
                          input_names=input_names,
                          output_names=output_names,
                          opset_version=ONNX_OPSET_VERSION,
                          do_constant_folding=True,
                          dynamo=False)
    t1 = time.time()
    print(f"ONNX export completed in {t1 - t0}s. Apply post-processing...")
    # Post-processing
    onnx.shape_inference.infer_shapes_path(onnx_path)
    onnx_model = onnx.load(onnx_path)
    graph = None

    if is_int4_awq_quantized(model):
        print(
            "INT4 AWQ quantization detected in the model, compressing some weights to INT4 and inserting int4 gemm plugin"
        )
        onnx_model = quantize_weights_to_int4(onnx_model)
        # Fix the Cast nodes and hidden_states output types for INT4 models
        onnx_model = fix_model_int4_output_dtypes(onnx_model)
        graph = gs.import_onnx(onnx_model)
        graph = int4_dq_gemm_to_plugin(graph)
    if is_fp8_quantized(model):
        print(
            "FP8 quantization detected in the model, compressing some weights to FP8"
        )
        if graph is None:
            graph = gs.import_onnx(onnx_model)
        graph = fold_fp8_qdq_to_dq(graph)
    if graph is not None:
        onnx_model = gs.export_onnx(graph)

    # Since torch.onnx.export deduplicates weights, lm_head and embed_tokens can
    # share the same ONNX initializer. To prevent quantization of lm_head (e.g. NVFP4)
    # from affecting embed_tokens, we manually create a separate initializer.
    # See: https://github.com/pytorch/pytorch/blob/v2.9.0-rc9/torch/csrc/jit/passes/onnx/deduplicate_initializers.cpp#L96
    if isinstance(model, EdgeLLMModelForCausalLM) and is_fp4_quantized(
            model.lm_head):
        onnx_model = untie_nvfp4_lm_head_initializer(onnx_model)
    if is_fp4_quantized(model):
        print(
            "NVFP4 quantization detected in the model, compressing some weights to NVFP4"
        )
        onnx_model = fp4qdq_to_2dq(onnx_model)
    if is_mxfp8_quantized(model):
        print(
            "MXFP8 quantization detected in the model, compressing some weights to MXFP8"
        )
        onnx_model = quantize_weights_to_mxfp8(onnx_model)

    print(
        "Removing all the files in the output directory except for .json files"
    )
    for file in os.listdir(output_dir):
        if file.endswith(".json"):
            continue
        os.remove(os.path.join(output_dir, file))

    # Save the model to the output directory
    onnx.save_model(onnx_model,
                    onnx_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location="onnx_model.data",
                    convert_attribute=True)
    t2 = time.time()
    print(
        f"ONNX post-processing completed in {t2 - t1}s. ONNX file is saved to {output_dir} in {t2 - t0}s."
    )
