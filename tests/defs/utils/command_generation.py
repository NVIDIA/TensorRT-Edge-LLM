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
Centralized command configuration
"""

import os
from typing import Dict, List, Tuple

from ..config import (DEFAULT_SEARCH_DEPTH, ModelType, TestConfig,
                      _find_directory)

# Available LoRA weights mapping
AVAILABLE_LORA_WEIGHTS = {
    "Qwen2.5-0.5B-Instruct": "Jailbreak-Detector-2-XL",
    "Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-Diagrams2SQL-v2",
}


def _generate_quantization_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate quantization commands if needed"""
    commands = []
    if config.llm_precision != "fp16" and config.llm_precision != "int4_gptq":
        torch_model_dir = config.get_torch_model_dir()
        quantized_model_dir = config.get_quantized_model_dir()

        quantize_cmd = [
            "tensorrt-edgellm-quantize-llm", f"--model_dir={torch_model_dir}",
            f"--output_dir={quantized_model_dir}",
            f"--quantization={config.llm_precision}",
            f"--dataset_dir={config.get_cnn_dailymail_dataset_dir()}"
        ]

        if config.lm_head_precision != "fp16":
            quantize_cmd.append(
                f"--lm_head_quantization={config.lm_head_precision}")

        commands.append((quantize_cmd, 1200))

    return commands


def _generate_llm_export_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate LLM export commands"""
    torch_model_dir = config.get_torch_model_dir()

    if config.llm_precision != "fp16" and config.llm_precision != "int4_gptq":
        # Use quantized model for export
        model_dir = config.get_quantized_model_dir()
    else:
        # Use original torch model for fp16 and int4_gptq export
        model_dir = torch_model_dir

    llm_cmd = [
        "tensorrt-edgellm-export-llm", f"--model_dir={model_dir}",
        f"--output_dir={config.get_llm_onnx_dir()}"
    ]

    if config.is_eagle:
        llm_cmd.append("--is_eagle_base")

    # Add custom chat template if specified for this model
    chat_template_path = config.get_chat_template_file()
    if chat_template_path:
        llm_cmd.append(f"--chat_template={chat_template_path}")

    return [(llm_cmd, 1200)]


def _generate_visual_export_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate visual model export commands for VLMs"""
    commands = []
    if config.model_type != ModelType.VLM:
        return commands

    visual_export_cmd = [
        "tensorrt-edgellm-export-visual",
        f"--model_dir={config.get_torch_model_dir()}",
        f"--dtype=fp16",
    ]

    # Always export fp16 visual model regardless of the precision
    fp16_visual_export_cmd = visual_export_cmd.copy()
    fp16_visual_export_cmd.append(
        f"--output_dir={config.get_visual_onnx_dir('fp16')}")
    commands.append((fp16_visual_export_cmd, 1200))

    if config.visual_precision == "fp8":
        fp8_visual_export_cmd = visual_export_cmd.copy()
        fp8_visual_export_cmd.append(f"--quantization=fp8")
        fp8_visual_export_cmd.append(
            f"--output_dir={config.get_visual_onnx_dir('fp8')}")
        fp8_visual_export_cmd.append(
            f"--dataset_dir={config.get_mmmu_dataset_dir()}")
        commands.append((fp8_visual_export_cmd, 1200))
    return commands


def _generate_lora_commands(config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate LoRA processing commands"""
    commands = []
    if not config.lora:
        return commands

    # Insert LoRA command
    lora_cmd = [
        "tensorrt-edgellm-insert-lora",
        f"--onnx_dir={config.get_llm_onnx_dir()}"
    ]
    commands.append((lora_cmd, 120))

    # Process LoRA weights if available
    if config.model_name in AVAILABLE_LORA_WEIGHTS:
        # Get base data directory from environment variable
        edgellm_data_dir = os.environ.get('EDGELLM_DATA_DIR',
                                          '/scratch.edge_llm_cache')

        # Search for the LoRA weights directory
        lora_model_name = AVAILABLE_LORA_WEIGHTS[config.model_name]
        lora_weights_dir = _find_directory(edgellm_data_dir, lora_model_name,
                                           DEFAULT_SEARCH_DEPTH)

        if not lora_weights_dir:
            raise ValueError(
                f"LoRA weights directory '{lora_model_name}' not found under "
                f"'{edgellm_data_dir}' within search depth {DEFAULT_SEARCH_DEPTH}."
            )

        process_lora_cmd = [
            "tensorrt-edgellm-process-lora", f"--input_dir={lora_weights_dir}",
            f"--output_dir={config.get_lora_weights_dir()}"
        ]
        commands.append((process_lora_cmd, 120))
    else:
        raise ValueError(
            f"No LoRA weights available for {config.model_name}. Please add it to AVAILABLE_LORA_WEIGHTS"
        )

    return commands


def _generate_draft_quantization_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate draft model quantization commands for EAGLE"""
    commands = []
    if not config.is_eagle:
        return commands

    if config.draft_llm_precision is None:
        raise ValueError("draft_llm_precision not set for EAGLE mode")

    base_model_dir = config.get_torch_model_dir()
    draft_model_dir = config.get_draft_model_dir()

    # Only quantize if draft model is not fp16
    if config.draft_llm_precision != "fp16" and config.draft_llm_precision != "int4_gptq":
        quantized_draft_dir = config.get_quantized_draft_model_dir()

        quantize_draft_cmd = [
            "tensorrt-edgellm-quantize-draft",
            f"--base_model_dir={base_model_dir}",
            f"--draft_model_dir={draft_model_dir}",
            f"--output_dir={quantized_draft_dir}",
            f"--quantization={config.draft_llm_precision}",
            f"--dataset_dir={config.get_cnn_dailymail_dataset_dir()}"
        ]

        # Add draft lm_head quantization if specified and not fp16
        if config.draft_lm_head_precision and config.draft_lm_head_precision != "fp16":
            quantize_draft_cmd.append(
                f"--lm_head_quantization={config.draft_lm_head_precision}")

        commands.append((quantize_draft_cmd, 900))

    return commands


def _generate_draft_export_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate draft model export commands for EAGLE"""
    commands = []
    if not config.is_eagle:
        return commands

    base_model_dir = config.get_torch_model_dir()
    if config.draft_llm_precision != "fp16":
        draft_model_dir = config.get_quantized_draft_model_dir()
    else:
        draft_model_dir = config.get_draft_model_dir()

    export_draft_cmd = [
        "tensorrt-edgellm-export-draft", f"--base_model_dir={base_model_dir}",
        f"--draft_model_dir={draft_model_dir}",
        f"--output_dir={config.get_draft_onnx_dir()}"
    ]

    if config.model_type == ModelType.VLM:
        export_draft_cmd.append("--use_prompt_tuning")

    commands.append((export_draft_cmd, 600))

    return commands


def generate_export_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate export commands - returns list of (command, timeout) tuples"""
    commands = []

    # Generate commands in order:
    # 1. Quantize base model (if needed)
    # 2. Export base model
    # 3. Export visual model (VLM only)
    # 4. Process LoRA (if needed)
    # 5. Quantize draft model (EAGLE only)
    # 6. Export draft model (EAGLE only)
    commands.extend(_generate_quantization_commands(config))
    commands.extend(_generate_llm_export_commands(config))
    commands.extend(_generate_visual_export_commands(config))
    commands.extend(_generate_lora_commands(config))
    commands.extend(_generate_draft_quantization_commands(config))
    commands.extend(_generate_draft_export_commands(config))

    return commands


def _generate_draft_build_commands(
        config: TestConfig,
        executable_files: Dict[str, str]) -> List[Tuple[List[str], int]]:
    """Generate draft model build commands for EAGLE"""
    commands = []
    if not config.is_eagle:
        return commands

    # Draft model build command
    draft_cmd = [executable_files['llm_build']]
    draft_cmd.extend([
        f"--onnxDir={config.get_draft_onnx_dir()}",
        f"--engineDir={config.get_llm_engine_dir()}",
        f"--maxInputLen={config.max_input_len}",
        f"--maxKVCacheCapacity={config.max_seq_len}",
        f"--maxBatchSize={config.max_batch_size}", "--eagleDraft",
        f"--maxDraftTreeSize={config.max_draft_tree_size}"
    ])

    if config.model_type == ModelType.VLM:
        draft_cmd.append("--vlm")
        draft_cmd.append(f"--minImageTokens={config.min_image_tokens}")
        draft_cmd.append(f"--maxImageTokens={config.max_image_tokens}")

    commands.append((draft_cmd, 1200))
    return commands


def generate_build_commands(
        config: TestConfig,
        executable_files: Dict[str, str]) -> List[Tuple[List[str], int]]:
    """Generate build commands - returns list of (command, timeout) tuples"""
    commands = []

    if config.model_type == ModelType.LLM:
        # LLM build command
        cmd = [executable_files['llm_build']]
        cmd.extend([
            f"--onnxDir={config.get_llm_onnx_dir()}",
            f"--engineDir={config.get_llm_engine_dir()}",
            f"--maxInputLen={config.max_input_len}",
            f"--maxKVCacheCapacity={config.max_seq_len}",
            f"--maxBatchSize={config.max_batch_size}"
        ])

        if config.is_eagle:
            cmd.append("--eagleBase")
            cmd.append(f"--maxVerifyTreeSize={config.max_verify_tree_size}")

        if config.max_lora_rank > 0:
            cmd.append(f"--maxLoraRank={config.max_lora_rank}")

        commands.append((cmd, 1200))

    elif config.model_type == ModelType.VLM:
        # VLM LLM build command
        llm_cmd = [executable_files['llm_build']]
        llm_cmd.extend([
            f"--onnxDir={config.get_llm_onnx_dir()}",
            f"--engineDir={config.get_llm_engine_dir()}",
            f"--maxInputLen={config.max_input_len}",
            f"--maxKVCacheCapacity={config.max_seq_len}", "--vlm",
            f"--maxBatchSize={config.max_batch_size}",
            f"--minImageTokens={config.min_image_tokens}",
            f"--maxImageTokens={config.max_image_tokens}"
        ])

        if config.is_eagle:
            llm_cmd.append("--eagleBase")
            llm_cmd.append(
                f"--maxVerifyTreeSize={config.max_verify_tree_size}")

        if config.max_lora_rank > 0:
            llm_cmd.append(f"--maxLoraRank={config.max_lora_rank}")

        commands.append((llm_cmd, 1200))

        # VLM visual build command
        visual_cmd = [executable_files['visual_build']]
        visual_cmd.extend([
            f"--onnxDir={config.get_visual_onnx_dir(config.visual_precision)}",
            f"--engineDir={config.get_visual_engine_dir()}",
            f"--minImageTokens={config.min_image_tokens}",
            f"--maxImageTokens={config.max_image_tokens}",
            f"--maxImageTokensPerImage={config.max_image_tokens_per_image}"
        ])

        commands.append((visual_cmd, 1200))

    # Add draft model build for EAGLE (must be after base model build)
    commands.extend(_generate_draft_build_commands(config, executable_files))

    return commands


def generate_inference_commands(
        config: TestConfig,
        executable_files: Dict[str, str]) -> List[Tuple[List[str], int]]:
    """Generate inference commands - returns list of (command, timeout) tuples"""
    commands = []

    cmd = [executable_files['llm_inference']]
    cmd.extend([
        f"--engineDir={config.get_llm_engine_dir()}",
        f"--inputFile={config.get_test_case_file()}",
        f"--outputFile={config.get_output_json_file()}", f"--dumpProfile"
    ])

    # Add EAGLE parameters
    if config.is_eagle:
        cmd.append("--eagle")
        cmd.append(f"--eagleDraftTopK={config.eagle_draft_top_k}")
        cmd.append(f"--eagleDraftStep={config.eagle_draft_step}")
        cmd.append(f"--eagleVerifyTreeSize={config.max_verify_tree_size}")

    if config.model_type == ModelType.VLM:
        cmd.append(f"--multimodalEngineDir={config.get_visual_engine_dir()}")

    # Add batch size override if specified
    if config.batch_size is not None:
        cmd.append(f"--batchSize={config.batch_size}")

    commands.append((cmd, 1200))
    return commands


def generate_benchmark_commands(
        config: TestConfig,
        executable_files: Dict[str, str]) -> List[Tuple[List[str], int]]:
    """Generate benchmark commands - returns list of (command, timeout) tuples"""
    commands = []

    if config.model_type == ModelType.LLM:
        cmd = [executable_files['llm_benchmark']]
        cmd.extend([
            f"--engineDir={config.get_llm_engine_dir()}",
            f"--inputLength={config.max_input_len}",
            f"--maxLength={config.output_seq_len + config.max_input_len}",
            "--warmUp=2", "--numRuns=10"
        ])

    elif config.model_type == ModelType.VLM:
        cmd = [executable_files['vlm_benchmark']]
        cmd.extend([
            f"--engineDir={config.get_llm_engine_dir()}",
            f"--visualEngineDir={config.get_visual_engine_dir()}",
            f"--textTokenLength={config.text_token_length}",
            f"--imageTokenLength={config.image_token_length}",
            f"--outputLength={config.output_seq_len}",
            f"--batchSize={config.max_batch_size}", "--warmUp=2",
            "--numRuns=10"
        ])

    commands.append((cmd, 1200))
    return commands
