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

import json
import logging
import os
import shutil
import time
from collections import namedtuple
from pathlib import Path
from typing import Tuple

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
from safetensors import safe_open

from tensorrt_edgellm._safetensors_io import save_file

from .phi4mm_utils import load_phi4mm_model

logger = logging.getLogger(__name__)

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


_WEIGHT_DQ_OPS = {
    "DequantizeLinear",
    "TRT_MXFP8DequantizeLinear",
}


def _find_weight_shape(gemm_node: gs.Node) -> tuple:
    """
    Find the weight shape of the GEMM node. The weight shape is not intuitive because of the quantization and transpose nodes.
    """
    # Weights are always on the second of a GEMM node.
    node = gemm_node.inputs[1].inputs[0]
    max_depth = 5
    depth = 0
    num_transpose = 0
    while node.op not in _WEIGHT_DQ_OPS and depth < max_depth:
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
    return tuple(weight_shape)


# ONNX FP8 dtype codes: FLOAT8E4M3FN=17, FLOAT8E5M2=18 (TensorProto enum).
_FP8_OUTPUT_DTYPES = {17, 18}


def _is_fp8_quantize_node(node: gs.Node) -> bool:
    """Standard ONNX FP8 producer with ``output_dtype`` set to an FP8 code."""
    if node.op == "QuantizeLinear":
        out_dtype = node.attrs.get("output_dtype") if node.attrs else None
        return isinstance(out_dtype, int) and out_dtype in _FP8_OUTPUT_DTYPES
    return False


_TRANSPARENT_OPS = {"Cast", "Reshape", "Identity"}


def _matmul_consumers_after_dq(quantize_node: gs.Node, max_hops: int = 3):
    """Yield every MatMul reached through Q → DQ → ... → MatMul. The DQ may
    fan out to several MatMul consumers (Q/K/V share one dequantized hidden
    state). Walks through ``max_hops`` levels of transparent ops
    (``Cast``/``Reshape``/``Identity``) between DQ and MatMul so a future
    modelopt emit pattern with intermediate ops keeps binding correctly
    instead of silently dropping the LoRA slot (Greptile P2)."""
    for dq in list(quantize_node.outputs[0].outputs):
        if dq.op != "DequantizeLinear":
            continue
        frontier = list(dq.outputs[0].outputs)
        seen = set()
        for _ in range(max_hops + 1):
            next_frontier = []
            for cons in frontier:
                cons_id = id(cons)
                if cons_id in seen:
                    continue
                seen.add(cons_id)
                if cons.op == "MatMul":
                    yield cons
                elif cons.op in _TRANSPARENT_OPS and cons.outputs:
                    next_frontier.extend(cons.outputs[0].outputs)
            if not next_frontier:
                break
            frontier = next_frontier


def _stem_from_init_name(name: str) -> str:
    """Convert an initializer name like ``_model.<...>.weight`` into the
    module-path stem used by the adapter safetensors (``model.<...>``).
    Returns ``""`` when the name is not in the expected
    ``_model.<...>.weight`` form so the caller refuses to synthesize a
    LoRA binding (silent garbage names produce dummy bindings at runtime)."""
    if not (name.startswith("_model.") and name.endswith(".weight")):
        return ""
    return name[len("_model."):-len(".weight")]


def _synth_gemm_name(stem: str, fallback: str) -> str:
    """Encode ``stem`` as a path so the downstream
    ``gemm_name.replace("/", ".").rsplit(".",1)[0][1:]`` in
    ``insert_lora_and_save`` collapses back to ``stem``. Fall back to the
    raw MatMul/plugin node name when the stem is unrecoverable — the
    runtime will then drop the binding by name mismatch rather than wire
    a silent garbage tensor."""
    return ("/" + stem.replace(".", "/") + "/MatMul") if stem else fallback


def _stem_from_weight_init(matmul_node: gs.Node) -> str:
    """Walk up the MatMul weight side to its originating ``_model.…weight``
    initializer and return its module-path stem. Returns ``""`` when the walk
    does not terminate at an initializer named ``_model.<...>.weight`` — the
    caller must then refuse to synthesize a LoRA binding name (silent garbage
    names produce dummy bindings at runtime)."""
    if not matmul_node.inputs[1].inputs:
        return ""
    node = matmul_node.inputs[1].inputs[0]
    depth = 0
    while node is not None and node.op not in _WEIGHT_DQ_OPS and depth < 5:
        if not node.inputs or not node.inputs[0].inputs:
            return ""
        node = node.inputs[0].inputs[0]
        depth += 1
    if node is None or not node.inputs:
        return ""
    return _stem_from_init_name(getattr(node.inputs[0], "name", "") or "")


def _match_fp8_gemm(graph: gs.Graph):
    """
    Match FP8 GEMM nodes in the graph.

    A single Quantize may fan out to multiple MatMuls through one
    DequantizeLinear; each MatMul becomes its own GEMM, and ``name`` is
    rewritten to a path-style stem derived from the weight initializer so the
    downstream LoRA input names match the adapter safetensors.
    """
    fp8_gemm_infos = []
    seen_matmul_ids = set()
    for node in graph.nodes:
        if not _is_fp8_quantize_node(node):
            continue
        input_node = node.inputs[0]
        matmuls = list(_matmul_consumers_after_dq(node))

        for matmul_node in matmuls:
            if id(matmul_node) in seen_matmul_ids:
                continue
            seen_matmul_ids.add(id(matmul_node))
            stem = _stem_from_weight_init(matmul_node)
            fp8_gemm_infos.append(
                GEMMInfo(input=input_node,
                         output=matmul_node.outputs[0],
                         name=_synth_gemm_name(stem, matmul_node.name),
                         weight_shape=_find_weight_shape(matmul_node)))
    return fp8_gemm_infos


def _match_nvfp4_gemm(graph: gs.Graph):
    """
    Match NVFP4 GEMM nodes in the graph.

    Same dynamo-naming hazard as the FP8 path: the downstream MatMul node
    is named ``node_MatMul_N`` by the dynamo exporter, so derive the GEMM
    name from the weight initializer instead.
    """
    nvfp4_gemm_infos = []
    nvfp4_quantize_linear_nodes = [
        node for node in graph.nodes if node.op == "TRT_FP4DynamicQuantize"
    ]
    for node in nvfp4_quantize_linear_nodes:
        input_node = node.inputs[0]
        matmul_node = _find_matmul_node(node)
        weight_shape = _find_weight_shape(matmul_node)
        stem = _stem_from_weight_init(matmul_node)
        nvfp4_gemm_infos.append(
            GEMMInfo(input=input_node,
                     output=matmul_node.outputs[0],
                     name=_synth_gemm_name(stem, matmul_node.name),
                     weight_shape=weight_shape))
    return nvfp4_gemm_infos


def _match_int4_gemm(graph: gs.Graph):
    """
    Match INT4 GEMM nodes in the graph.

    Both ``Int4GroupwiseGemmPlugin`` (V1) and ``Int4GroupwiseGemmPluginV2``
    (V2, the default backend) carry their weight initializer directly as
    ``node.inputs[1]`` (no DequantizeLinear chain) and share the same
    ``gemm_k``/``gemm_n`` attributes, so the stem can be derived in one step
    for either. V2 must be matched too; otherwise the default INT4 backend's
    GEMMs get no LoRA inputs and adapters are silently dropped at runtime.
    """
    int4_gemm_infos = []
    int4_gemm_nodes = [
        node for node in graph.nodes
        if node.op in ("Int4GroupwiseGemmPlugin", "Int4GroupwiseGemmPluginV2")
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
        weight_init_name = getattr(node.inputs[1], "name", "") or ""
        stem = _stem_from_init_name(weight_init_name)
        int4_gemm_infos.append(
            GEMMInfo(input=input_node,
                     output=node.outputs[0],
                     name=_synth_gemm_name(stem, node.name),
                     weight_shape=weight_shape))
    return int4_gemm_infos


def _match_nvfp4_a16_gemm(graph: gs.Graph):
    """
    Match weight-only NVFP4 (W4A16) GEMM nodes in the graph.

    ``Nvfp4A16GemmPlugin`` (the Marlin weight-only backend) carries its
    packed weight initializer directly as ``node.inputs[1]`` and the GEMM
    shape in its ``gemm_k``/``gemm_n`` attributes, mirroring the INT4
    groupwise plugin (activation stays FP16, so there is no smoothing
    Mul/Cast on the input). This op is distinct from ``TRT_FP4DynamicQuantize``
    (the W4A4 QDQ path); without matching it the weight-only NVFP4 GEMMs get
    no LoRA inputs and adapters are silently dropped at runtime.
    """
    nvfp4_a16_gemm_infos = []
    nvfp4_a16_nodes = [
        node for node in graph.nodes if node.op == "Nvfp4A16GemmPlugin"
    ]
    for node in nvfp4_a16_nodes:
        input_node = node.inputs[0]
        weight_shape = (node.attrs["gemm_k"], node.attrs["gemm_n"])
        weight_init_name = getattr(node.inputs[1], "name", "") or ""
        stem = _stem_from_init_name(weight_init_name)
        nvfp4_a16_gemm_infos.append(
            GEMMInfo(input=input_node,
                     output=node.outputs[0],
                     name=_synth_gemm_name(stem, node.name),
                     weight_shape=weight_shape))
    return nvfp4_a16_gemm_infos


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
    gemm_infos.extend(_match_nvfp4_a16_gemm(graph))
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
    parts = key.split('.')
    is_norm_tensor = any(
        part == 'norm' or part.endswith('_norm') or part.endswith('layernorm')
        for part in parts)
    return not is_norm_tensor and 'lm_head' not in parts


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
        onnx_dir (str): Directory containing model.onnx and config.json.
            The modified graph is written to lora_model.onnx in the same directory.
    """
    start_time = time.time()
    # Load ONNX model
    onnx_model_path = os.path.join(onnx_dir, "model.onnx")
    logger.info("Loading original ONNX model from %s", onnx_model_path)

    # The LoRA model will share the same data as the base model
    onnx_model = onnx.load(onnx_model_path, load_external_data=False)
    graph = gs.import_onnx(onnx_model)

    # Insert dynamic LoRA patterns
    logger.info("Inserting dynamic LoRA patterns")
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
        # Before the Add node: it also consumes output_tensor. Only rewire consumers
        # that existed for the GEMM, or we would replace the Add's input with
        # final_output (the Add output) and create a graph cycle.
        gemm_consumers = list(output_tensor.outputs)
        graph.layer(name=f"{gemm_name}/lora_add",
                    op="Add",
                    inputs=[output_tensor, lora_out],
                    outputs=[final_output])

        # Replace at the same input index; remove+append broke multi-input ops (e.g.
        # Reshape must keep data vs shape tensor order for TensorRT shape inference).
        for out_node in gemm_consumers:
            for idx, inp in enumerate(out_node.inputs):
                if inp is output_tensor:
                    out_node.inputs[idx] = final_output
                    break

    graph.cleanup().toposort().fold_constants().cleanup()

    # Save modified ONNX model
    output_model_path = os.path.join(onnx_dir, "lora_model.onnx")
    logger.info("Saving modified ONNX model to %s", output_model_path)

    modified_onnx_model = gs.export_onnx(graph)
    onnx.save_model(modified_onnx_model, output_model_path)

    end_time = time.time()
    logger.info("LoRA model saved to %s", output_model_path)
    logger.info("LoRA insertion completed in %.2fs", end_time - start_time)


def _model_type_from_config(model_dir: str) -> str:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return ""
    try:
        with open(config_path) as f:
            config = json.load(f)
        return config.get("model_type", "")
    except (OSError, ValueError):
        return ""


def _path_contains(parent: Path, child: Path) -> bool:
    return parent == child or parent in child.parents


def _prepare_merge_output_dir(output_dir: str, model_dir: str,
                              lora_dir: str) -> None:
    output_path = Path(output_dir).expanduser().resolve()
    model_path = Path(model_dir).expanduser().resolve()
    lora_path = Path(lora_dir).expanduser().resolve()
    protected_paths = {Path("/").resolve(), Path.home().resolve()}
    try:
        protected_paths.add(Path.cwd().resolve())
    except OSError:
        pass

    if output_path in protected_paths:
        raise ValueError(f"Refusing to remove protected output_dir: "
                         f"{output_path}")
    if _path_contains(output_path, model_path) or _path_contains(
            output_path, lora_path):
        raise ValueError("Refusing to use an output_dir that contains the "
                         "input model or LoRA adapter directory")

    if output_path.exists():
        if not output_path.is_dir():
            raise ValueError(f"output_dir exists and is not a directory: "
                             f"{output_path}")
        logger.warning("Removing existing LoRA merge output directory: %s",
                       output_path)
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)


def merge_lora_and_save(model_dir: str,
                        lora_dir: str,
                        output_dir: str,
                        device: str = "cuda",
                        torch_dtype: str = "float16") -> None:
    """Merge a PEFT LoRA adapter into a HuggingFace checkpoint."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if torch_dtype not in dtype_map:
        raise ValueError(f"Unsupported torch_dtype={torch_dtype!r}")

    _prepare_merge_output_dir(output_dir, model_dir, lora_dir)

    model_type = _model_type_from_config(model_dir)
    is_phi4mm = model_type in ("phi4mm", "phi4_multimodal")
    if is_phi4mm:
        model = load_phi4mm_model(model_dir,
                                  dtype_map[torch_dtype],
                                  patch_peft_generation=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype_map[torch_dtype],
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    if device:
        model.to(device)

    lora_model = PeftModel.from_pretrained(model, lora_dir)
    merged_model = lora_model.merge_and_unload()
    if is_phi4mm:
        merged_model.config.vision_lora = None
        merged_model.config.speech_lora = None
    merged_model.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    try:
        processor = AutoProcessor.from_pretrained(model_dir,
                                                  trust_remote_code=True)
    except (OSError, ValueError):
        processor = None
    if processor is not None:
        if model_type in ("phi4mm", "phi4_multimodal"):
            for name in ("preprocessor_config.json", "processor_config.json",
                         "processing_phi4mm.py"):
                src = os.path.join(model_dir, name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(output_dir, name))
        else:
            processor.save_pretrained(output_dir)

    logger.info("Merged LoRA adapter %s into %s", lora_dir, output_dir)


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

                logger.info("Processed tensor %s shape=%s dtype=%s", new_key,
                            tuple(processed_tensor.shape),
                            processed_tensor.dtype)

        # Save processed tensors
        output_path = os.path.join(output_dir,
                                   'processed_adapter_model.safetensors')
        save_file(processed_tensors, output_path)
        logger.info("Processed tensors saved to: %s", output_path)
        logger.info("Config file copied to: %s",
                    os.path.join(output_dir, 'config.json'))

    except Exception as e:
        raise RuntimeError(f"Error processing safetensor file: {e}") from e
