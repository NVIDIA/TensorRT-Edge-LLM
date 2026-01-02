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
TensorRT Edge-LLM - A Python package for quantizing and exporting LLMs for edge deployment.

This package provides utilities for quantizing large language models using NVIDIA ModelOpt
and preparing them for ONNX export and edge deployment. It supports various quantization
schemes including FP8, INT4 AWQ, and NVFP4 for efficient inference on edge devices.

Key Features:
    - LLM quantization with calibration support
    - Multiple quantization schemes (FP8, INT4 AWQ, NVFP4)
    - Automatic model type detection
    - HuggingFace model compatibility
    - Quantization configuration management
    - ONNX export for LLM and visual models
    - LoRA pattern insertion and weight processing

Example Usage:
    .. code-block:: python

        from tensorrt_edgellm import (
            quantize_and_save_llm,
            quantize_and_save_draft,
            export_llm_model,
            export_draft_model,
            visual_export,
            insert_lora_and_save,
            process_lora_weights_and_save
        )
        
        # Quantize and save a standard LLM model
        quantize_and_save_llm(
            model_dir="path/to/model",
            output_dir="path/to/output",
            quantization="fp8",
            dtype="fp16",
            dataset_dir="cnn_dailymail"
        )

        # Quantize and save an EAGLE draft model
        quantize_and_save_draft(
            base_model_dir="path/to/base_model",
            draft_model_dir="path/to/draft_model",
            output_dir="path/to/output",
            quantization="fp8",
            dtype="fp16",
            dataset_dir="cnn_dailymail"
        )
        
        # Export standard LLM to ONNX
        export_llm_model(
            model_dir="path/to/model",
            output_dir="path/to/output",
            device="cuda"
        )
        
        # Export EAGLE base model to ONNX
        export_llm_model(
            model_dir="path/to/model",
            output_dir="path/to/output",
            is_eagle_base=True
        )
        
        # Export EAGLE draft model to ONNX
        export_draft_model(
            draft_model_dir="path/to/draft_model",
            output_dir="path/to/output",
            base_model_dir="path/to/base_model",
            use_prompt_tuning=False
        )
        
        # Export visual model to ONNX
        visual_export(
            model_dir="path/to/model",
            output_dir="path/to/output",
            dtype="fp16"
        )
        
        # Insert LoRA patterns into ONNX models
        insert_lora_and_save(
            onnx_dir="path/to/onnx_model"
        )
        
        # Process LoRA weights
        process_lora_weights_and_save(
            input_dir="path/to/adapter",
            output_dir="path/to/output"
        )
        # Reduce vocabulary
        reduce_vocab_size(
            model_dir="path/to/model",
            output_dir="path/to/output",
            reduced_vocab_size=30000
        )
"""

from .onnx_export.llm_export import export_draft_model, export_llm_model
from .onnx_export.lora import (insert_lora_and_save,
                               process_lora_weights_and_save)
from .onnx_export.visual_export import visual_export
from .quantization.llm_quantization import (quantize_and_save_draft,
                                            quantize_and_save_llm)
from .vocab_reduction.vocab_reduction import reduce_vocab_size

try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = [
    "quantize_and_save_llm",
    "quantize_and_save_draft",
    "export_draft_model",
    "export_llm_model",
    "visual_export",
    "insert_lora_and_save",
    "process_lora_weights_and_save",
    "reduce_vocab_size",
]
