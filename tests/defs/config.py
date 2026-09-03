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
Test configuration utilities for TensorRT Edge-LLM tests.

This module provides a unified TestConfig class that handles both export and runtime 
test configurations, eliminating redundancy across the test framework.
"""

import enum
import os
from dataclasses import dataclass
from typing import Optional, Set

from conftest import EnvironmentConfig
from valid_precisions import (VALID_AUDIO_PRECISIONS, VALID_LLM_PRECISIONS,
                              VALID_LM_HEAD_PRECISIONS,
                              VALID_VISUAL_PRECISIONS)

# Global configuration constants
DEFAULT_SEARCH_DEPTH = 3

# Models that ship pre-quantized (skip tensorrt-edgellm-quantize step).
# These are exported directly from the HF checkpoint without a quantization step.
PRE_QUANTIZED_MODELS: frozenset = frozenset({
    "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
    "NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "Qwen3-30B-A3B-NVFP4",
    "nvidia-Qwen3-30B-A3B-NVFP4",
    "nvidia-Gemma-4-31B-IT-NVFP4",
    "nvidia-Gemma-4-26B-A4B-NVFP4",
})


def _find_directory(
    root_dir: str,
    target_name: str,
    max_depth: Optional[int] = None,
    require_files: Optional[list] = None,
) -> Optional[str]:
    """Search for a directory with the given name or path.

    Args:
        root_dir: Root directory to start searching from.
        target_name: Target directory name or relative path to find
            (e.g. ``"model"`` or ``"parent/model"``).
        max_depth: Maximum search depth (``None`` for unlimited, ``1`` for
            immediate children only).
        require_files: Optional list of file requirements. Each entry may
            be either a single glob (``"config.json"``) — that pattern
            must match at least one file — or a list of globs
            (``["model*.safetensors", "pytorch_model*.bin"]``) — at least
            one of those globs must match. All outer entries must be
            satisfied. Used to disambiguate HF checkpoint dirs from
            same-named engine cache / ONNX output dirs that happen to be
            deeper in the tree.

    Returns:
        Full path to the first matching directory found, or None if not
        found.
    """
    if root_dir is None or not os.path.exists(root_dir):
        return None

    import glob

    def _is_valid_match(candidate: str) -> bool:
        if not require_files:
            return True
        for pattern in require_files:
            if isinstance(pattern, (list, tuple)):
                # OR: at least one of these globs must match.
                if not any(
                        glob.glob(os.path.join(candidate, p))
                        for p in pattern):
                    return False
            else:
                if not glob.glob(os.path.join(candidate, pattern)):
                    return False
        return True

    def _search(current_dir: str, current_depth: int) -> Optional[str]:
        if max_depth is not None and current_depth > max_depth:
            return None

        try:
            entries = os.listdir(current_dir)
        except PermissionError:
            return None

        candidate_path = os.path.join(current_dir, target_name)
        if os.path.isdir(candidate_path) and _is_valid_match(candidate_path):
            return candidate_path

        if max_depth is None or current_depth < max_depth:
            for entry in entries:
                if entry == '.git':
                    continue
                entry_path = os.path.join(current_dir, entry)

                if os.path.isdir(entry_path):
                    result = _search(entry_path, current_depth + 1)
                    if result:
                        return result

        return None

    return _search(root_dir, 0)


# A real HF checkpoint dir has config.json plus weights or a weight index.
# Used by _find_directory.require_files to distinguish checkpoint dirs from
# engine-cache dirs (no config.json).
_HF_CHECKPOINT_FILES = [
    "config.json",
    ["*.safetensors", "*.bin", "*.safetensors.index.json"],
]

_NVFP4_MOE_TARGET_ENV = "EDGELLM_NVFP4_MOE_TARGET"
_NVFP4_MOE_TARGET_TOKENS = frozenset(
    ("sm100", "sm101", "sm110", "sm12x", "sm120", "sm121", "geforce"))


def _normalize_nvfp4_moe_target(target: str) -> str:
    """Normalize testcase tokens for the NVFP4 MoE export target."""
    target = target.strip().lower()
    if target in ("sm120", "sm121", "geforce"):
        return "sm12x"
    return target


_QUANT_SUFFIX_BY_PRECISION = {
    "int4_awq": "INT4-AWQ",
    "int8_sq": "INT8-SQ",
    "int4_gptq": "GPTQ-Int4",
}

_MODEL_QUANT_SUFFIXES = (
    "-GPTQ-INT4",
    "-INT4_GPTQ",
    "-LMNVFP4",
    "-LMMXFP8",
    "-LMFP8",
    "-INT8-SQ",
    "-INT4-AWQ",
    "-MXFP8",
    "-NVFP4",
    "-VITFP8",
    "-AUDFP8",
    "-FP8-KV",
    "-FP16",
    "-FP8",
)


def canonical_quant_suffix(precision: str) -> str:
    """Canonical directory/ID suffix for a precision token."""
    return _QUANT_SUFFIX_BY_PRECISION.get(precision, precision.upper())


def strip_model_quant_suffixes(model_name: str) -> str:
    """Strip trailing quantization suffixes from model directory names."""
    base = model_name
    while True:
        matched = False
        upper_base = base.upper()
        for suffix in _MODEL_QUANT_SUFFIXES:
            if upper_base.endswith(suffix):
                base = base[:-len(suffix)]
                matched = True
                break
        if not matched:
            break
    return base


class ModelType(enum.Enum):
    """Supported model types"""
    LLM = "llm"
    VLM = "vlm"
    TTS = "tts"
    ASR = "asr"
    OMNI = "omni"
    VLA = "vla"  # Vision-Language-Action (e.g. Alpamayo-R1-10B)


def infer_checkpoint_export_model_type(param_str: str) -> ModelType:
    """Infer model type for pre-quantized checkpoint export tests."""
    parts = param_str.split('-')
    model_parts: list[str] = []
    for i, part in enumerate(parts):
        if part in VALID_LLM_PRECISIONS:
            model_parts = parts[:i]
            break
    model_name = '-'.join(model_parts) if model_parts else param_str
    base = strip_model_quant_suffixes(model_name)

    if base.startswith("Alpamayo"):
        return ModelType.VLA
    if base.startswith("Qwen3-ASR"):
        return ModelType.ASR
    if base.startswith("Qwen3-TTS"):
        return ModelType.TTS
    if "Omni" in base:
        return ModelType.OMNI
    if ("-VL-" in base or base.startswith("InternVL")
            or base.startswith("Cosmos-Reason")
            or base.startswith("Cosmos3-Edge-reasoning")
            or "multimodal" in base.lower()):
        return ModelType.VLM
    return ModelType.LLM


class TaskType(enum.Enum):
    """Supported task types"""
    EXPORT = "export"
    BUILD = "build"
    CHECKPOINT_BUILD = "checkpoint_build"
    E2E_BENCH = "e2e_bench"
    INFERENCE = "inference"
    KERNEL_BENCH = "kernel_bench"
    VLMEVALKIT = "vlmevalkit"


@dataclass
class ParameterSpec:
    """Specification for a parameter with validation rules"""
    name: str
    format_hint: str
    required_for: Set[TaskType]
    valid_for: Set[ModelType]
    is_required: bool = True

    @staticmethod
    def _effective_task_type(task_type: TaskType) -> TaskType:
        if task_type in (TaskType.CHECKPOINT_BUILD, TaskType.VLMEVALKIT):
            return TaskType.INFERENCE
        return task_type

    def is_valid_for_task_and_model(self, task_type: TaskType,
                                    model_type: ModelType) -> bool:
        """Check if this parameter is valid for the given task and model type"""
        task_type = self._effective_task_type(task_type)
        return task_type in self.required_for and model_type in self.valid_for

    def is_required_for_task_and_model(self, task_type: TaskType,
                                       model_type: ModelType) -> bool:
        """Check if this parameter is required for the given task and model type"""
        return self.is_valid_for_task_and_model(
            task_type, model_type) and self.is_required


LLM_MODELS_DIR_MAP = {
    "Qwen2.5-0.5B-Instruct":
    "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct":
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct":
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct":
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-VL-3B-Instruct":
    "Qwen2.5-VL-3B-Instruct",
    "Qwen2.5-VL-7B-Instruct":
    "Qwen2.5-VL-7B-Instruct",
    "Qwen2-VL-2B-Instruct":
    "Qwen2-VL-2B-Instruct",
    "InternVL3-1B":
    "InternVL3-1B-hf",
    "InternVL3-2B":
    "InternVL3-2B-hf",
    "Llama-3.1-8B-Instruct":
    "llama-3.1-model/Llama-3.1-8B-Instruct",
    "Llama-3.2-1B":
    "llama-3.2-models/Llama-3.2-1B",
    "Llama-3.2-3B":
    "llama-3.2-models/Llama-3.2-3B",
    "Qwen3-0.6B":
    "Qwen3/Qwen3-0.6B",
    "Qwen3-1.7B":
    "Qwen3/Qwen3-1.7B",
    "Qwen3-4B": [
        "Qwen/Qwen3-4B",
        "Qwen3/Qwen3-4B",
    ],
    "Qwen3-8B":
    "Qwen3/Qwen3-8B",
    "Qwen3-4B-Instruct-2507":
    "Qwen3/Qwen3-4B-Instruct-2507",
    "Qwen3-VL-2B-Instruct":
    "Qwen3/Qwen3-VL-2B-Instruct",
    "Qwen3-VL-4B-Instruct":
    "Qwen3/Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B-Instruct":
    "Qwen3/Qwen3-VL-8B-Instruct",
    "Qwen3.5-0.8B":
    "Qwen3.5-0.8B",
    "Qwen3.5-2B":
    "Qwen3.5-2B",
    "Qwen3.5-4B":
    "Qwen3.5-4B",
    "Qwen3.5-9B":
    "Qwen3.5-9B",
    "Qwen3.5-27B":
    "Qwen3.5-27B",
    "Qwen3.6-27B":
    "Qwen3.6-27B",
    "Qwen3.8-27B":
    "Qwen3.8-27B",
    # Gemma4 E-models. CI runners may see the public base checkpoint under
    # /scratch.trt_llm_data/llm-models/gemma, while paired MTP assistant
    # checkpoints are usually staged under /scratch.edge_llm_cache/source_models.
    "gemma-4-E2B-it": [
        "gemma/gemma-4-E2B-it",
        "source_models/gemma-4-E2B-it",
        "gemma-4-E2B-it",
    ],
    "gemma-4-E4B-it":
    "gemma/gemma-4-E4B-it",
    "gemma-4-12B-it":
    "gemma/gemma-4-12B-it",
    "gemma-4-31B-it":
    "gemma/gemma-4-31B-it",
    "gemma-4-26B-A4B-it":
    "gemma/gemma-4-26B-A4B-it",
    "Phi-4-multimodal-instruct":
    "Phi-4-multimodal-instruct",
    "Alpamayo-R1-10B":
    "Alpamayo-R1-10B",
    # Pre-quantized models in llm_models_dir
    "Llama-3.2-1B-FP8":
    "llama-3.2-models/Llama-3.2-1B-FP8",
    "Phi-4-FP8":
    "Phi-4-FP8",
    "Phi-4-multimodal-instruct-FP8":
    "Phi-4-multimodal-instruct-FP8",
    # ASR and TTS models
    "Qwen3-ASR-0.6B":
    "Qwen3/Qwen3-ASR-0.6B",
    "Qwen3-TTS-12Hz-0.6B-CustomVoice":
    "Qwen3/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    # Nemotron-H 30B (BF16 base + pre-quantized NVFP4)
    "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16":
    "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    # Pre-quantized NVFP4 model: exported directly without quantization step
    "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4":
    "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
    "NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4":
    "NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "NVIDIA-Nemotron-3-Nano-4B-BF16":
    "NVIDIA-Nemotron-3-Nano-4B-BF16",
    "NVIDIA-Nemotron-3-Nano-4B-FP8":
    "NVIDIA-Nemotron-3-Nano-4B-FP8",
    # Nemotron-Nano 9B v2 family (BF16 base + FP8/NVFP4 pre-quantized
    # variants all live under llm_models_dir/, not edge_llm_cache/).
    "NVIDIA-Nemotron-Nano-9B-v2":
    "NVIDIA-Nemotron-Nano-9B-v2",
    "NVIDIA-Nemotron-Nano-9B-v2-FP8":
    "NVIDIA-Nemotron-Nano-9B-v2-FP8",
    "NVIDIA-Nemotron-Nano-9B-v2-NVFP4":
    "NVIDIA-Nemotron-Nano-9B-v2-NVFP4",
    # Cosmos VLM
    "Cosmos-Reason2-8B":
    "Cosmos-Reason2-8B",
    "Cosmos3-Edge-reasoning":
    "Cosmos3-Edge",
    # Qwen3.5/3.6 35B-A3B (BF16 base; GPTQ-Int4 / NVFP4 variants in GPTQ map)
    "Qwen3.5-35B-A3B":
    "Qwen3.5-35B-A3B",
    "Qwen3.6-35B-A3B":
    "Qwen3.6-35B-A3B",
    # ASR / TTS larger variants (1.7B family)
    "Qwen3-ASR-1.7B":
    "Qwen3/Qwen3-ASR-1.7B",
    "Qwen3-TTS-12Hz-1.7B-CustomVoice":
    "Qwen3/Qwen3-TTS-12Hz-1.7B-CustomVoice",
}

GPTQ_MODELS_DIR_MAP = {
    "Qwen2.5-7B-Instruct-GPTQ-Int4": "Qwen2.5-7B-Instruct-GPTQ-Int4",
    "InternVL3-1B-GPTQ-Int4": "InternVL3-1B-hf-GPTQ-Int4",
    # GPTQ-Int4 large MoE variants
    "Qwen3-30B-A3B-GPTQ-Int4": "Qwen3-30B-A3B-GPTQ-Int4",
    # NVFP4 MoE (pre-quantized, no quantization step needed)
    "Qwen3-30B-A3B-NVFP4": "Qwen3-30B-A3B-NVFP4",
    "nvidia-Qwen3-30B-A3B-NVFP4": "Qwen3/nvidia-Qwen3-30B-A3B-NVFP4",
    "Qwen3.5-35B-A3B-GPTQ-Int4": "Qwen3.5-35B-A3B-GPTQ-Int4",
    "Qwen3.6-35B-A3B-NVFP4": "Qwen3.6-35B-A3B-NVFP4",
    "nvidia-Gemma-4-31B-IT-NVFP4": "nvidia-Gemma-4-31B-IT-NVFP4",
    "nvidia-Gemma-4-26B-A4B-NVFP4": "nvidia-Gemma-4-26B-A4B-NVFP4",
    # Multimodal pre-quantized NVFP4 (LLM + visual + audio).  Test list
    # uses ``Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`` as the
    # canonical name; verify the on-disk dir matches before running.
    "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4":
    "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
    "NVIDIA-Nemotron-3-Nano-4B-NVFP4": "NVIDIA-Nemotron-3-Nano-4B-NVFP4",
    # Pre-quantized unified checkpoints (edge_llm_cache/quantized_models/)
    "Qwen2.5-0.5B-Instruct-FP8": "Qwen2.5-0.5B-Instruct-FP8",
    "Qwen2.5-0.5B-Instruct-FP8-KV": "Qwen2.5-0.5B-Instruct-FP8-KV",
    "Qwen2.5-0.5B-Instruct-NVFP4": "Qwen2.5-0.5B-Instruct-NVFP4",
    "Qwen3-0.6B-FP8": "Qwen3-0.6B-FP8",
    "Qwen3-0.6B-INT8-SQ": "Qwen3-0.6B-INT8-SQ",
    "Qwen3-1.7B-FP8": "Qwen3-1.7B-FP8",
    "Qwen3-1.7B-NVFP4": "Qwen3-1.7B-NVFP4",
    "Qwen3.5-4B-NVFP4": "Qwen3.5-4B-NVFP4",
    "Qwen3-VL-4B-Instruct-NVFP4": "Qwen3-VL-4B-Instruct-NVFP4",
    "Qwen3-VL-2B-Instruct-INT4-AWQ": "Qwen3-VL-2B-Instruct-INT4-AWQ",
}

# Base model + EAGLE ``draft_model_id`` -> draft checkpoint folder name.
# Single source of truth shared by ``_draft_model_dir_name`` (torch dir lookup)
# and ``get_quantized_draft_checkpoint_dir_name`` (hub pre-quant folder name) so
# the two cannot diverge.
MODEL_NAME_TO_DRAFT_MODELS_MAP = {
    "Qwen2.5-VL-7B-Instruct": {
        "v1": "qwen2.5-vl-7b-eagle3-v1",
        "v2": "qwen2.5-vl-7b-eagle3-v2",
        "sgl": "qwen2.5-vl-7b-eagle3-sgl",
    },
    "Llama-3.1-8B-Instruct": {
        "eagle3": "EAGLE3-LLaMA3.1-Instruct-8B",
    },
    "Qwen3-8B": {
        "eagle3": "qwen3_8b_eagle3",
    },
    "Qwen3-4B-Instruct-2507": {
        "v2": "EAGLE3-Qwen3-4B-v2",
        "v2.1": "EAGLE3-Qwen3-4B-v2.1",
    },
    "Qwen3-VL-4B-Instruct": {
        "eagle3": "EAGLE3-Qwen3-VL-4B-v1.1",
    },
    "Qwen3-1.7B": {
        "eagle3": "Qwen3-1.7B_eagle3",
    },
    "Qwen3-1.7B-NVFP4": {
        "eagle3": "Qwen3-1.7B-eagle3-NVFP4",
    },
    "Qwen3-VL-4B-Instruct-NVFP4": {
        "eagle3": "EAGLE3-Qwen3-VL-4B-v1.1-NVFP4",
    },
    "Qwen3-VL-8B-Instruct": {
        "v0": "qwen3-vl-8b-eagle3-v0",
    },
    "gemma-4-12B-it": {
        "ttt7": "eagle3_gemma4_12b_ttt7",
    },
}

# Base model + DFlash ``draft_model_id`` -> draft checkpoint folder name.
MODEL_NAME_TO_DFLASH_DRAFT_MODELS_MAP = {
    "Qwen3-4B-Instruct-2507": {
        "zlab": "Qwen3-4B-DFlash-b16",
    },
    "Qwen3-8B": {
        "zlab": "Qwen3-8B-DFlash-b16",
    },
    "Qwen3.5-4B": {
        "b16": "Qwen3.5-4B-DFlash",
    },
    "Qwen3.5-4B-NVFP4": {
        "b16": "Qwen3.5-4B-DFlash-NVFP4",
    },
    "Qwen3.5-9B": {
        "zlab": "Qwen3.5-9B-DFlash",
    },
    "Qwen3.5-27B": {
        "zlab": "Qwen3.5-27B-DFlash",
    },
    # MoE models: NVFP4 base + FP16 DFlash draft, except
    # Qwen3.5-35B-A3B which is currently supported as GPTQ-Int4 base.
    "Qwen3.5-35B-A3B-GPTQ-Int4": {
        "zlab": "Qwen3.5-35B-A3B-DFlash",
    },
    "Qwen3.6-35B-A3B": {
        "zlab": "Qwen3.6-35B-A3B-DFlash",
    },
    "Qwen3.6-35B-A3B-NVFP4": {
        "zlab": "Qwen3.6-35B-A3B-DFlash",
    },
    "gemma-4-31B-it": {
        "zlab": "gemma-4-31B-it-DFlash",
    },
    "gemma-4-26B-A4B-it": {
        "zlab": "gemma-4-26B-A4B-it-DFlash",
    },
    "gemma-4-12B-it": {
        "block7": "dflash_gemma4_12b_block7",
        "zlab": "gemma4-12B-it-DFlash",
    },
}

# Base model + JetSpec ``draft_model_id`` -> draft checkpoint folder name.
MODEL_NAME_TO_JETSPEC_DRAFT_MODELS_MAP = {
    "Qwen3-8B": {
        "hf": "JetSpec/jetspec-qwen3-8b",
    },
}

# Paired Gemma4 MTP uses a separate assistant checkpoint. The test parameter
# remains ``...-mtp`` so runtime naming is shared with Qwen-style MTP; only
# export needs this model-family-specific assistant lookup.
GEMMA4_MTP_ASSISTANT_MODELS_MAP = {
    "gemma-4-12B-it": [
        "gemma-4-12B-it-assistant",
        "google/gemma-4-12B-it-assistant",
        "gemma/gemma-4-12B-it-assistant",
    ],
    "gemma-4-E2B-it": [
        "source_models/gemma-4-E2B-it-assistant",
        "gemma-4-E2B-it-assistant",
        "gemma/gemma-4-E2B-it-assistant",
    ],
    "nvidia-Gemma-4-26B-A4B": [
        "gemma-4-26B-A4B-it-assistant",
        "google/gemma-4-26B-A4B-it-assistant",
        "gemma/gemma-4-26B-A4B-it-assistant",
    ],
}

# Base model + DSpark ``draft_model_id`` -> draft checkpoint folder name.
# The public DeepSpec block7 model targets Qwen3-4B and uses a linear block
# proposal length of 7, therefore runtime specVerifySize must be 8.
MODEL_NAME_TO_DSPARK_DRAFT_MODELS_MAP = {
    "Qwen3-4B": {
        "block7": "deepseek-ai/dspark_qwen3_4b_block7",
    },
}


@dataclass
class TestConfig:
    """
    Test configuration class that handles both export and runtime test configurations.
    """
    # Core identifiers
    param_str: str
    model_name: str
    model_type: ModelType
    task_type: TaskType

    # Precision settings
    llm_precision: str
    lm_head_precision: Optional[str] = None
    visual_precision: Optional[str] = None
    # ViT MHA precision (Q*K^T / P*V in fp8). Requires visual_precision=fp8:
    # the q/k/v bmm dequant scales are calibrated by the same multimodal
    # calibration loop that visual-tower quantization drives.
    visual_mha_precision: Optional[str] = None
    audio_precision: Optional[str] = None
    # extw_<value> token: ffn / lm / moe / nvfp4_moe / ffn_lm / all
    externalize_weights: Optional[str] = None

    # EAGLE draft model settings
    draft_model_name: Optional[str] = None
    draft_model_id: Optional[str] = None
    draft_llm_precision: Optional[str] = None
    draft_lm_head_precision: Optional[str] = None

    # Speculative decoding flags
    is_eagle: Optional[bool] = None
    is_mtp: Optional[bool] = None
    is_dflash: Optional[bool] = None
    is_dflash_tree: Optional[bool] = None
    is_jetspec: Optional[bool] = None
    is_jetspec_tree: Optional[bool] = None
    is_dspark: Optional[bool] = None

    # Directory paths
    llm_models_dir: Optional[str] = None
    edgellm_data_dir: Optional[str] = None
    vlmevalkit_data_dir: Optional[str] = None
    onnx_dir: Optional[str] = None
    engine_dir: Optional[str] = None
    test_log_dir: Optional[str] = None

    # Export LoRA parameters
    lora: Optional[bool] = None
    merge_lora: Optional[bool] = None

    # Engine build parameters
    max_batch_size: Optional[int] = None
    max_input_len: Optional[int] = None
    max_seq_len: Optional[int] = None
    max_lora_rank: Optional[int] = None

    # EAGLE specific build parameters
    max_verify_tree_size: Optional[int] = None
    max_draft_tree_size: Optional[int] = None

    # EAGLE inference parameters
    eagle_draft_top_k: Optional[int] = None
    eagle_draft_step: Optional[int] = None

    # VLM specific build parameters
    min_image_tokens: Optional[int] = None
    max_image_tokens: Optional[int] = None
    max_image_tokens_per_image: Optional[int] = None

    # ASR specific build parameters
    min_time_steps: Optional[int] = None
    max_time_steps: Optional[int] = None

    max_kv_cache_capacity: Optional[int] = None

    # Context-reuse options
    max_kv_pool_pages: Optional[int] = None
    context_reuse: Optional[bool] = None
    context_cache_recurrent_snapshot_pool_bytes: Optional[int] = None
    context_cache_partial_kv_snapshot_pool_bytes: Optional[int] = None

    # Inference parameters
    test_case: Optional[str] = None

    # Path of a per-config preprocessed copy of the test case JSON. Set by
    # tests/defs/utils/command_execution.py helpers when they need to rewrite
    # audio paths or substitute LoRA placeholders without mutating the
    # version-controlled source under tests/test_cases/. ``get_test_case_file``
    # returns this when populated so downstream command generation
    # transparently picks up the rewritten document.
    _test_case_file_override: Optional[str] = None

    # KV cache options
    fp8_kv_cache: Optional[
        bool] = None  # If true, export ONNX/config with FP8 KV cache enabled

    # Embedding options
    fp8_embedding: Optional[
        bool] = None  # If true, write embedding.safetensors in FP8 E4M3 format

    # Benchmark parameters
    batch_size: Optional[int] = None
    input_seq_len: Optional[int] = None
    output_seq_len: Optional[int] = None

    # VLM-specific parameters (no defaults - must be specified)
    text_token_length: Optional[int] = None
    image_token_length: Optional[int] = None

    # Vocabulary reduction parameters
    reduced_vocab_size: Optional[int] = None
    vocab_reduction_method: Optional[
        str] = None  # "input_aware" or "frequency"
    vocab_reduction_max_samples: Optional[int] = None

    warmup: Optional[int] = None

    # kernel_bench parameters
    bench_mode: Optional[str] = None
    input_len: Optional[int] = None
    past_kv_len: Optional[int] = None

    # Use TRT-native ragged attention (TRT >= 11) instead of plugin
    trt_native_attn: Optional[bool] = None

    # Export NVFP4 MoE graph for a specific plugin target (for example sm12x).
    nvfp4_moe_target: Optional[str] = None

    # Export INT4 GEMM with the legacy AWQ-swizzled Int4GroupwiseGemmPlugin (V1)
    # instead of the default cuteDSL Int4GroupwiseGemmPluginV2. TRT-RTX's ONNX
    # parser only imports V1.
    int4_gemm_plugin_v1: Optional[bool] = None

    # Debug flag for verbose output
    debug: Optional[bool] = None

    # Declarative parameter specifications
    _PARAMETER_SPECS = [
        # Core parameters for engine identification
        ParameterSpec(
            "max_batch_size", "mxbs", {
                TaskType.BUILD, TaskType.E2E_BENCH, TaskType.INFERENCE,
                TaskType.KERNEL_BENCH
            }, {
                ModelType.LLM, ModelType.VLM, ModelType.TTS, ModelType.ASR,
                ModelType.OMNI, ModelType.VLA
            }),
        ParameterSpec(
            "max_input_len", "mxil", {
                TaskType.BUILD, TaskType.E2E_BENCH, TaskType.INFERENCE,
                TaskType.KERNEL_BENCH
            }, {
                ModelType.LLM, ModelType.VLM, ModelType.TTS, ModelType.ASR,
                ModelType.OMNI, ModelType.VLA
            }),
        ParameterSpec(
            "max_seq_len", "mxsl", {
                TaskType.BUILD, TaskType.E2E_BENCH, TaskType.INFERENCE,
                TaskType.KERNEL_BENCH
            }, {
                ModelType.LLM, ModelType.VLM, ModelType.TTS, ModelType.ASR,
                ModelType.OMNI, ModelType.VLA
            }),
        ParameterSpec("max_lora_rank",
                      "mxlr",
                      {TaskType.BUILD, TaskType.E2E_BENCH, TaskType.INFERENCE},
                      {
                          ModelType.LLM, ModelType.VLM, ModelType.TTS,
                          ModelType.ASR, ModelType.OMNI
                      },
                      is_required=False),

        # Export-specific parameters
        ParameterSpec("lora",
                      "", {TaskType.EXPORT}, {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("merge_lora",
                      "", {TaskType.EXPORT}, {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec(
            "fp8_kv_cache",
            "fp8kv", {
                TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                TaskType.INFERENCE
            }, {ModelType.LLM, ModelType.VLM, ModelType.TTS, ModelType.ASR},
            is_required=False),
        ParameterSpec(
            "fp8_embedding",
            "fp8emb",
            {
                TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                TaskType.INFERENCE
            }, {ModelType.LLM, ModelType.VLM, ModelType.ASR, ModelType.OMNI},
            is_required=False),
        ParameterSpec(
            "is_eagle",
            "eagle", {
                TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                TaskType.INFERENCE
            }, {ModelType.LLM, ModelType.VLM, ModelType.TTS, ModelType.ASR},
            is_required=False),
        ParameterSpec("is_mtp",
                      "mtp", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("is_dflash",
                      "dflash", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.LLM},
                      is_required=False),
        ParameterSpec("is_dflash_tree",
                      "ddtree", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.LLM},
                      is_required=False),
        ParameterSpec("is_jetspec",
                      "jetspec", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.LLM},
                      is_required=False),
        ParameterSpec("is_jetspec_tree",
                      "ddtree", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.LLM},
                      is_required=False),
        ParameterSpec("is_dspark",
                      "dspark", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.LLM},
                      is_required=False),
        ParameterSpec("draft_model_id",
                      "", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("draft_llm_precision",
                      "", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("draft_lm_head_precision",
                      "", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("max_verify_tree_size",
                      "mvts",
                      {TaskType.BUILD, TaskType.INFERENCE, TaskType.E2E_BENCH},
                      {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("max_draft_tree_size",
                      "mdts",
                      {TaskType.BUILD, TaskType.INFERENCE, TaskType.E2E_BENCH},
                      {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("eagle_draft_top_k",
                      "edtk", {TaskType.INFERENCE, TaskType.E2E_BENCH},
                      {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("eagle_draft_step",
                      "edst", {TaskType.INFERENCE, TaskType.E2E_BENCH},
                      {ModelType.LLM, ModelType.VLM},
                      is_required=False),

        # ASR-specific parameters
        ParameterSpec("min_time_steps", "mnts",
                      {TaskType.BUILD, TaskType.E2E_BENCH, TaskType.INFERENCE},
                      {ModelType.ASR, ModelType.OMNI}),
        ParameterSpec("max_time_steps", "mxts",
                      {TaskType.BUILD, TaskType.E2E_BENCH, TaskType.INFERENCE},
                      {ModelType.ASR, ModelType.OMNI}),

        # VLM/OMNI/VLA visual-encoder parameters
        ParameterSpec("min_image_tokens", "mnit",
                      {TaskType.BUILD, TaskType.E2E_BENCH, TaskType.INFERENCE},
                      {ModelType.VLM, ModelType.OMNI, ModelType.VLA}),
        ParameterSpec("max_image_tokens", "mxit",
                      {TaskType.BUILD, TaskType.E2E_BENCH, TaskType.INFERENCE},
                      {ModelType.VLM, ModelType.OMNI, ModelType.VLA}),
        # mxpiit is optional — most VLM/OMNI test cases use the same value
        # (512), so set_defaults() falls back to 512 when not in the param
        # string. Override per-test by including ``-mxpiit<N>``.
        ParameterSpec("max_image_tokens_per_image",
                      "mxpiit",
                      {TaskType.BUILD, TaskType.E2E_BENCH, TaskType.INFERENCE},
                      {ModelType.VLM, ModelType.OMNI, ModelType.VLA},
                      is_required=False),
        ParameterSpec("visual_precision",
                      "vit", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.VLM, ModelType.OMNI, ModelType.VLA},
                      is_required=False),
        ParameterSpec("visual_mha_precision",
                      "vitmha", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.VLM},
                      is_required=False),
        ParameterSpec("max_kv_cache_capacity",
                      "mxkvc", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.VLA},
                      is_required=False),
        ParameterSpec("max_kv_pool_pages",
                      "mxkvp",
                      {TaskType.BUILD, TaskType.E2E_BENCH, TaskType.INFERENCE},
                      {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("audio_precision",
                      "aud", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.TTS, ModelType.ASR, ModelType.OMNI},
                      is_required=False),
        ParameterSpec(
            "externalize_weights",
            "extw",
            {TaskType.EXPORT},
            {ModelType.LLM, ModelType.VLM},
            is_required=False,
        ),

        # Inference/Benchmark parameters
        ParameterSpec("test_case", "",
                      {TaskType.INFERENCE, TaskType.E2E_BENCH}, {
                          ModelType.LLM, ModelType.VLM, ModelType.TTS,
                          ModelType.ASR, ModelType.OMNI, ModelType.VLA
                      }),
        ParameterSpec("batch_size",
                      "bs", {TaskType.E2E_BENCH, TaskType.INFERENCE}, {
                          ModelType.LLM, ModelType.VLM, ModelType.TTS,
                          ModelType.ASR, ModelType.OMNI, ModelType.VLA
                      },
                      is_required=False),
        ParameterSpec("context_reuse",
                      "ctxreuse", {TaskType.E2E_BENCH, TaskType.INFERENCE},
                      {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("context_cache_recurrent_snapshot_pool_bytes",
                      "ccrsb", {TaskType.E2E_BENCH, TaskType.INFERENCE},
                      {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("context_cache_partial_kv_snapshot_pool_bytes",
                      "ccpkvsb", {TaskType.E2E_BENCH, TaskType.INFERENCE},
                      {ModelType.LLM, ModelType.VLM},
                      is_required=False),

        # Vocabulary reduction parameters
        ParameterSpec(
            "reduced_vocab_size",
            "rvs", {
                TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                TaskType.INFERENCE
            }, {ModelType.LLM, ModelType.VLM, ModelType.TTS, ModelType.ASR},
            is_required=False),
        ParameterSpec(
            "vocab_reduction_method",
            "vrm", {TaskType.EXPORT},
            {ModelType.LLM, ModelType.VLM, ModelType.TTS, ModelType.ASR},
            is_required=False),
        ParameterSpec(
            "vocab_reduction_max_samples",
            "vrms", {TaskType.EXPORT},
            {ModelType.LLM, ModelType.VLM, ModelType.TTS, ModelType.ASR},
            is_required=False),
        ParameterSpec("trt_native_attn",
                      "trt11", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.LLM, ModelType.VLM, ModelType.OMNI},
                      is_required=False),
        ParameterSpec("nvfp4_moe_target",
                      "sm12x", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.LLM, ModelType.VLM, ModelType.OMNI},
                      is_required=False),
        ParameterSpec("int4_gemm_plugin_v1",
                      "gemmv1", {TaskType.EXPORT},
                      {ModelType.LLM, ModelType.VLM, ModelType.OMNI},
                      is_required=False),
        # kernel_bench parameters
        ParameterSpec("bench_mode",
                      "mode", {TaskType.KERNEL_BENCH}, {ModelType.LLM},
                      is_required=False),
        ParameterSpec("input_len",
                      "il", {TaskType.KERNEL_BENCH}, {ModelType.LLM},
                      is_required=False),
        ParameterSpec("past_kv_len",
                      "pkv", {TaskType.KERNEL_BENCH}, {ModelType.LLM},
                      is_required=False),
    ]

    @classmethod
    def from_param_string(cls,
                          param_str: str,
                          model_type: ModelType,
                          task_type: TaskType,
                          env_config: EnvironmentConfig,
                          *,
                          validate_environment: bool = True) -> 'TestConfig':
        """
        Unified function to parse parameter string and create config with validation.
        
        Handles model names with multiple parts, performs all validation uniformly,
        and constructs the final config object.
        """

        if validate_environment:
            if task_type == TaskType.EXPORT:
                env_config.validate_for_export_tests()
            elif task_type == TaskType.CHECKPOINT_BUILD:
                env_config.validate_for_checkpoint_builder_tests()
            else:
                env_config.validate_for_pipeline_tests()
                if task_type == TaskType.VLMEVALKIT:
                    env_config.validate_for_vlmevalkit_tests()

        # Parse parameter string
        parts = param_str.split('-')

        # Find precision position and extract model name
        model_parts = []
        llm_precision = None
        lm_head_precision = None
        visual_precision = None
        remaining_parts = []

        # Find the first valid precision to determine where model name ends
        for i, part in enumerate(parts):
            if part in VALID_LLM_PRECISIONS:
                llm_precision = part
                model_parts = parts[:
                                    i]  # Everything before precision is model name

                # Check for lm head precision
                if i + 1 < len(parts) and parts[i + 1].startswith('lm'):
                    lm_head_precision = parts[i + 1][2:]
                    if lm_head_precision not in VALID_LM_HEAD_PRECISIONS:
                        raise ValueError(
                            f"Invalid LM head precision: {lm_head_precision}")
                    remaining_parts = parts[i + 2:]
                else:
                    remaining_parts = parts[i + 1:]
                break
        if not lm_head_precision:
            lm_head_precision = "fp16"

        if not llm_precision:
            raise ValueError(f"No valid precision found in: {param_str}")

        if not model_parts:
            raise ValueError(f"No model name found in: {param_str}")

        model_name = '-'.join(model_parts)

        # Parse remaining parameters
        parsed_params = {}

        i = 0
        while i < len(remaining_parts):
            part = remaining_parts[i]

            # For engine identification
            if part.startswith('mxsl'):
                parsed_params['max_seq_len'] = int(part[4:])
            elif part == "lora":
                parsed_params['lora'] = True
            elif part == "merge_lora":
                parsed_params['merge_lora'] = True
            elif part in ("fp8kv", "mhafp8"):
                parsed_params['fp8_kv_cache'] = True
            elif part == "fp8emb":
                parsed_params['fp8_embedding'] = True
            elif part == "mtp":
                parsed_params['is_mtp'] = True
                parsed_params['is_eagle'] = True
            elif part == "dflash":
                parsed_params['is_dflash'] = True
                # Parse dflash-{draft_id}[-{draft_precision}[-lm{draft_lm_head}]].
                if i + 1 >= len(remaining_parts):
                    raise ValueError(
                        f"Missing draft model id after dflash in: {param_str}")
                i += 1
                parsed_params['draft_model_id'] = remaining_parts[i]

                if (i + 1 < len(remaining_parts)
                        and remaining_parts[i + 1] in VALID_LLM_PRECISIONS):
                    i += 1
                    parsed_params['draft_llm_precision'] = remaining_parts[i]

                    if (i + 1 < len(remaining_parts)
                            and remaining_parts[i + 1].startswith('lm')):
                        i += 1
                        draft_lm_precision = remaining_parts[i][2:]
                        if draft_lm_precision not in VALID_LM_HEAD_PRECISIONS:
                            raise ValueError(
                                f"Invalid draft LM head precision: {draft_lm_precision}"
                            )
                        parsed_params[
                            'draft_lm_head_precision'] = draft_lm_precision
                else:
                    parsed_params['draft_llm_precision'] = llm_precision
            elif part == "jetspec":
                parsed_params['is_jetspec'] = True
                # Parse jetspec-{draft_id}[-{draft_precision}[-lm{draft_lm_head}]].
                if i + 1 >= len(remaining_parts):
                    raise ValueError(
                        f"Missing draft model id after jetspec in: {param_str}"
                    )
                i += 1
                parsed_params['draft_model_id'] = remaining_parts[i]

                if (i + 1 < len(remaining_parts)
                        and remaining_parts[i + 1] in VALID_LLM_PRECISIONS):
                    i += 1
                    parsed_params['draft_llm_precision'] = remaining_parts[i]

                    if (i + 1 < len(remaining_parts)
                            and remaining_parts[i + 1].startswith('lm')):
                        i += 1
                        draft_lm_precision = remaining_parts[i][2:]
                        if draft_lm_precision not in VALID_LM_HEAD_PRECISIONS:
                            raise ValueError(
                                f"Invalid draft LM head precision: {draft_lm_precision}"
                            )
                        parsed_params[
                            'draft_lm_head_precision'] = draft_lm_precision
                else:
                    parsed_params['draft_llm_precision'] = llm_precision
            elif part == "dspark":
                parsed_params['is_dspark'] = True
                # Parse dspark-{draft_id}[-{draft_precision}[-lm{draft_lm_head}]].
                if i + 1 >= len(remaining_parts):
                    raise ValueError(
                        f"Missing draft model id after dspark in: {param_str}")
                i += 1
                parsed_params['draft_model_id'] = remaining_parts[i]

                if (i + 1 < len(remaining_parts)
                        and remaining_parts[i + 1] in VALID_LLM_PRECISIONS):
                    i += 1
                    parsed_params['draft_llm_precision'] = remaining_parts[i]

                    if (i + 1 < len(remaining_parts)
                            and remaining_parts[i + 1].startswith('lm')):
                        i += 1
                        draft_lm_precision = remaining_parts[i][2:]
                        if draft_lm_precision not in VALID_LM_HEAD_PRECISIONS:
                            raise ValueError(
                                f"Invalid draft LM head precision: {draft_lm_precision}"
                            )
                        parsed_params[
                            'draft_lm_head_precision'] = draft_lm_precision
                else:
                    parsed_params['draft_llm_precision'] = llm_precision
            elif part == "ddtree":
                parsed_params['is_dflash_tree'] = True
            elif part == "eagle":
                parsed_params['is_eagle'] = True
                # Parse eagle-{draft_id}-{draft_precision}[-lm{draft_lm_head}]
                # Next part should be draft_model_id
                if i + 1 < len(remaining_parts):
                    i += 1
                    parsed_params['draft_model_id'] = remaining_parts[i]

                    # Next part should be draft_llm_precision
                    if i + 1 < len(remaining_parts):
                        i += 1
                        draft_precision = remaining_parts[i]

                        # Check if it starts with 'lm' - this would be draft lm_head
                        if draft_precision.startswith('lm'):
                            raise ValueError(
                                f"Missing draft precision after draft_model_id in: {param_str}"
                            )

                        # Validate draft precision
                        if draft_precision not in VALID_LLM_PRECISIONS:
                            raise ValueError(
                                f"Invalid draft precision: {draft_precision}")
                        parsed_params['draft_llm_precision'] = draft_precision

                        # Check for optional draft lm_head precision
                        if i + 1 < len(remaining_parts) and remaining_parts[
                                i + 1].startswith('lm'):
                            i += 1
                            draft_lm_precision = remaining_parts[i][2:]
                            if draft_lm_precision not in VALID_LM_HEAD_PRECISIONS:
                                raise ValueError(
                                    f"Invalid draft LM head precision: {draft_lm_precision}"
                                )
                            parsed_params[
                                'draft_lm_head_precision'] = draft_lm_precision
            elif part.startswith('mxbs'):
                parsed_params['max_batch_size'] = int(part[4:])
            elif part.startswith('mxil'):
                parsed_params['max_input_len'] = int(part[4:])
            elif part.startswith('mnts'):
                parsed_params['min_time_steps'] = int(part[4:])
            elif part.startswith('mxts'):
                parsed_params['max_time_steps'] = int(part[4:])
            elif part.startswith('mnit'):
                parsed_params['min_image_tokens'] = int(part[4:])
            elif part.startswith('mxit'):
                parsed_params['max_image_tokens'] = int(part[4:])
            elif part.startswith('mxpiit'):
                parsed_params['max_image_tokens_per_image'] = int(part[6:])
            elif part.startswith('mxkvc'):
                parsed_params['max_kv_cache_capacity'] = int(part[5:])
            elif part.startswith('mxkvp'):
                parsed_params['max_kv_pool_pages'] = int(part[5:])
            elif part.startswith('mxlr'):
                parsed_params['max_lora_rank'] = int(part[4:])
            # For benchmark parameters
            elif part.startswith('bs'):
                parsed_params['batch_size'] = int(part[2:])
            elif part.startswith('isl'):
                parsed_params['input_seq_len'] = int(part[3:])
            elif part.startswith('osl'):
                parsed_params['output_seq_len'] = int(part[3:])
            elif part.startswith('ttl'):
                parsed_params['text_token_length'] = int(part[3:])
            elif part.startswith('itl'):
                parsed_params['image_token_length'] = int(part[3:])
            # Exact match must precede the startswith('vit') branch below.
            elif part == 'vitmhafp8':
                parsed_params['visual_mha_precision'] = 'fp8'
            elif part.startswith('vit'):
                visual_precision = part[3:]
                if visual_precision in VALID_VISUAL_PRECISIONS:
                    parsed_params['visual_precision'] = visual_precision
                else:
                    raise ValueError(
                        f"Invalid visual precision: {visual_precision}")
            elif part.startswith('aud'):
                audio_precision = part[3:]
                if audio_precision in VALID_AUDIO_PRECISIONS:
                    parsed_params['audio_precision'] = audio_precision
                else:
                    raise ValueError(
                        f"Invalid audio precision: {audio_precision}")
            # For EAGLE parameters
            elif part.startswith('mvts'):
                parsed_params['max_verify_tree_size'] = int(part[4:])
            elif part.startswith('mdts'):
                parsed_params['max_draft_tree_size'] = int(part[4:])
            elif part.startswith('edtk'):
                parsed_params['eagle_draft_top_k'] = int(part[4:])
            elif part.startswith('edst'):
                parsed_params['eagle_draft_step'] = int(part[4:])
            elif part == 'ctxreuse':
                parsed_params['context_reuse'] = True
            elif part.startswith('ccrsb'):
                parsed_params[
                    'context_cache_recurrent_snapshot_pool_bytes'] = int(
                        part[5:])
            elif part.startswith('ccpkvsb'):
                parsed_params[
                    'context_cache_partial_kv_snapshot_pool_bytes'] = int(
                        part[7:])
            # For vocabulary reduction parameters
            elif part.startswith('extw_'):
                parsed_params['externalize_weights'] = part[len('extw_'):]
            elif part.startswith('rvs'):
                parsed_params['reduced_vocab_size'] = int(part[3:])
            elif part.startswith('vrm'):
                parsed_params['vocab_reduction_method'] = part[3:]
            elif part.startswith('vrms'):
                parsed_params['vocab_reduction_max_samples'] = int(part[4:])
            # For kernel_bench parameters
            elif part.startswith('mode_'):
                parsed_params['bench_mode'] = part[5:]
            elif part.startswith('il') and part[2:].isdigit():
                parsed_params['input_len'] = int(part[2:])
            elif part.startswith('pkv') and part[3:].isdigit():
                parsed_params['past_kv_len'] = int(part[3:])
            elif part == 'trt11':
                parsed_params['trt_native_attn'] = True
            elif part == 'gemmv1':
                parsed_params['int4_gemm_plugin_v1'] = True
            elif part.lower() in _NVFP4_MOE_TARGET_TOKENS:
                parsed_params['nvfp4_moe_target'] = (
                    _normalize_nvfp4_moe_target(part))
            else:
                parsed_params['test_case'] = part

            i += 1

        if not visual_precision and model_type in (ModelType.VLM,
                                                   ModelType.OMNI):
            parsed_params['visual_precision'] = "fp16"
        if (parsed_params.get('visual_mha_precision') == 'fp8'
                and parsed_params.get('visual_precision') != 'fp8'):
            raise ValueError(
                f"vitmhafp8 requires vitfp8 (q/k/v bmm scales are calibrated "
                f"by the visual-tower quantization pass): {param_str}")
        if model_type in (ModelType.TTS, ModelType.ASR, ModelType.OMNI
                          ) and 'audio_precision' not in parsed_params:
            parsed_params['audio_precision'] = "fp16"

        # Create base config object
        config = cls(param_str=param_str,
                     model_name=model_name,
                     model_type=model_type,
                     task_type=task_type,
                     llm_precision=llm_precision,
                     lm_head_precision=lm_head_precision,
                     llm_models_dir=env_config.llm_models_dir if task_type
                     in (TaskType.EXPORT, TaskType.CHECKPOINT_BUILD) else None,
                     edgellm_data_dir=env_config.edgellm_data_dir,
                     vlmevalkit_data_dir=env_config.vlmevalkit_data_dir,
                     onnx_dir=env_config.onnx_dir,
                     engine_dir=env_config.engine_dir
                     if task_type != TaskType.EXPORT else None,
                     test_log_dir=env_config.test_log_dir)

        # Apply validated parameters to config
        for param_name, value in parsed_params.items():
            setattr(config, param_name, value)

        # Validate completeness and set defaults
        config._validate_completeness()

        return config

    @classmethod
    def resolve_quantized_draft_checkpoint_dir_name(
            cls, param_str: str, model_type: ModelType) -> Optional[str]:
        """Hub folder name for a spec draft from test_param only (no real paths)."""
        env_stub = EnvironmentConfig(
            llm_sdk_dir=".",
            llm_models_dir=".",
            edgellm_data_dir=None,
            onnx_dir=".",
            engine_dir=None,
            build_dir="build",
            test_log_dir="logs",
            trt_package_dir=None,
        )
        try:
            config = cls.from_param_string(
                param_str,
                model_type,
                TaskType.EXPORT,
                env_stub,
                validate_environment=False,
            )
        except ValueError:
            return None
        return config.get_quantized_draft_checkpoint_dir_name()

    def _supports_llm_visual_precision(self) -> bool:
        """Allow visual precision for LLM entries that include a visual tower."""
        if self.model_type != ModelType.LLM:
            return False
        return (self.model_name.startswith("Qwen3.5-")
                or self.model_name.startswith("Qwen3.6-")
                or self.model_name.startswith("Qwen3.8-"))

    def _supports_llm_audio_precision(self) -> bool:
        """Allow audio precision for LLM entries that include an audio tower."""
        if self.model_type != ModelType.LLM:
            return False
        return ("-ASR-" in self.model_name or "-TTS-" in self.model_name
                or "Omni" in self.model_name)

    def _validate_completeness(self) -> None:
        """Validate that all required parameters are set for the given task and model type"""

        def set_defaults() -> None:
            """Set default values for optional parameters"""
            if self.task_type == TaskType.EXPORT:
                if self.lora is None:
                    self.lora = False
                if self.merge_lora is None:
                    self.merge_lora = False
                if self.fp8_kv_cache is None:
                    self.fp8_kv_cache = False
                if self.is_eagle is None:
                    self.is_eagle = False
                if self.is_mtp is None:
                    self.is_mtp = False
                if self.is_dflash is None:
                    self.is_dflash = False
                if self.is_dflash_tree is None:
                    self.is_dflash_tree = False
                if self.is_jetspec is None:
                    self.is_jetspec = False
                if self.is_jetspec_tree is None:
                    self.is_jetspec_tree = False
                if self.is_dspark is None:
                    self.is_dspark = False
                if self.draft_llm_precision is not None and self.draft_lm_head_precision is None:
                    self.draft_lm_head_precision = "fp16"
                if self.reduced_vocab_size is not None:
                    if self.vocab_reduction_method is None:
                        self.vocab_reduction_method = "input_aware"
                    if self.vocab_reduction_max_samples is None:
                        self.vocab_reduction_max_samples = 50000
            else:  # Runtime tasks
                if self.max_lora_rank is None:
                    self.max_lora_rank = 0
                if self.lora is None:
                    self.lora = self.max_lora_rank > 0
                if self.fp8_kv_cache is None:
                    self.fp8_kv_cache = False
                if (self.model_type
                        in (ModelType.VLM, ModelType.OMNI, ModelType.VLA
                            )) and (self.max_image_tokens_per_image is None):
                    self.max_image_tokens_per_image = 512
                if (self.model_type == ModelType.VLA
                        and self.max_kv_cache_capacity is None):
                    self.max_kv_cache_capacity = 4096
                if (self.model_type == ModelType.VLA
                        and self.visual_precision is None):
                    self.visual_precision = "fp16"
                if self.is_eagle is None:
                    self.is_eagle = False
                if self.is_mtp is None:
                    self.is_mtp = False
                if self.is_dflash is None:
                    self.is_dflash = False
                if self.is_dflash_tree is None:
                    self.is_dflash_tree = False
                if self.is_jetspec is None:
                    self.is_jetspec = False
                if self.is_jetspec_tree is None:
                    self.is_jetspec_tree = False
                if self.is_dspark is None:
                    self.is_dspark = False
                if self.draft_llm_precision is not None and self.draft_lm_head_precision is None:
                    self.draft_lm_head_precision = "fp16"
                if self.eagle_draft_top_k is None:
                    self.eagle_draft_top_k = 1 if (
                        self.is_mtp or self.is_dflash or self.is_jetspec
                        or self.is_dspark) else 10
                if self.eagle_draft_step is None:
                    self.eagle_draft_step = 1 if (
                        self.is_dflash or self.is_jetspec
                        or self.is_dspark) else (3 if self.is_mtp else 6)
                if self.max_verify_tree_size is None:
                    self.max_verify_tree_size = 8 if self.is_dspark else (
                        16 if (self.is_dflash or self.is_jetspec) else
                        (4 if self.is_mtp else 60))
                if self.max_draft_tree_size is None:
                    self.max_draft_tree_size = 7 if self.is_dspark else (
                        16 if (self.is_dflash or self.is_jetspec) else
                        (4 if self.is_mtp else 60))

        warmup_env = os.environ.get('WARMUP')
        if warmup_env is not None and self.warmup is None:
            self.warmup = int(warmup_env)

        # Read debug flag from environment variable
        debug_env = os.environ.get('DEBUG')
        if debug_env is not None and self.debug is None:
            self.debug = debug_env.lower() in ('1', 'true', 'yes')

        missing_params = []
        invalid_params = []

        # Get valid and required parameters for current task/model combination
        valid_params = set()
        required_params = set()

        for spec in self._PARAMETER_SPECS:
            if spec.is_valid_for_task_and_model(self.task_type,
                                                self.model_type):
                valid_params.add(spec.name)
                if spec.is_required_for_task_and_model(self.task_type,
                                                       self.model_type):
                    required_params.add(spec.name)
        if self._supports_llm_visual_precision():
            valid_params.add("visual_precision")
        if self._supports_llm_audio_precision():
            valid_params.add("audio_precision")

        # Check for invalid parameters (parameters that are set but not allowed)
        for spec in self._PARAMETER_SPECS:
            param_value = getattr(self, spec.name)
            if param_value is not None and spec.name not in valid_params:
                invalid_params.append(spec.name)

        if invalid_params:
            task_desc = f"{self.model_type.value} {self.task_type.value}"
            raise ValueError(
                f"Invalid parameters for {task_desc}: {', '.join(invalid_params)}"
            )

        # Check for missing required parameters
        for spec in self._PARAMETER_SPECS:
            if spec.name in required_params and getattr(self,
                                                        spec.name) is None:
                param_desc = f'{spec.name} ({spec.format_hint})' if spec.format_hint else spec.name
                missing_params.append(param_desc)

        # Raise error if any required parameters are missing
        if missing_params:
            task_desc = f"{self.model_type.value} {self.task_type.value}"
            raise ValueError(
                f"Missing required parameters for {task_desc}: {', '.join(missing_params)}"
            )
        if self.nvfp4_moe_target and self.llm_precision != "nvfp4":
            raise ValueError(
                "nvfp4_moe_target is only valid for nvfp4 LLM precision.")
        if self.max_kv_pool_pages is not None and self.max_kv_pool_pages <= 0:
            raise ValueError("max_kv_pool_pages must be positive when set.")
        snapshot_budgets = (
            self.context_cache_recurrent_snapshot_pool_bytes,
            self.context_cache_partial_kv_snapshot_pool_bytes,
        )
        if any(value is not None and value <= 0 for value in snapshot_budgets):
            raise ValueError(
                "Context-cache snapshot pool byte budgets must be positive when set."
            )
        if (any(value is not None for value in snapshot_budgets)
                and not self.context_reuse):
            raise ValueError(
                "Context-cache snapshot pool budgets require ctxreuse.")

        # Set defaults after validation
        set_defaults()

        if self.is_jetspec and self.is_dflash_tree:
            self.is_jetspec_tree = True
            self.is_dflash_tree = False

        if self.is_dflash_tree and not self.is_dflash:
            raise ValueError("ddtree can only be used with DFlash tests")
        if self.is_jetspec_tree and not self.is_jetspec:
            raise ValueError("ddtree can only be used with JetSpec tests")
        if ((self.is_dflash_tree or self.is_jetspec_tree)
                and self.task_type in (TaskType.CHECKPOINT_BUILD,
                                       TaskType.E2E_BENCH, TaskType.INFERENCE)
                and self.eagle_draft_top_k <= 1):
            raise ValueError(
                "DFlash/JetSpec DDTree runtime tests require edtk > 1; "
                "use linear mode without ddtree for edtk=1")

    def check_trt_native_attn(self) -> None:
        """Skip -trt11 tests when TRT < 11.

        l0_jedha and l0_jedha_trt11 share the same test list but run
        different TRT versions. The CI job sets TRT_VERSION.
        """
        if not self.trt_native_attn:
            return
        trt_ver = os.environ.get('TRT_VERSION', '')
        try:
            major = int(trt_ver.split(".")[0])
        except (ValueError, IndexError):
            major = 0
        if major < 11:
            import pytest
            pytest.skip(f"-trt11 requires TRT >= 11 (TRT_VERSION={trt_ver!r})")

    # Unified path generation methods
    @staticmethod
    def _canonical_quant_suffix(precision: str) -> str:
        return canonical_quant_suffix(precision)

    def get_quantized_model_id(self) -> str:
        """Generate unique quantized model identifier."""
        model_id = self._canonical_quant_suffix(self.llm_precision)
        if self.lm_head_precision != "fp16":
            model_id += (
                f"-LM{self._canonical_quant_suffix(self.lm_head_precision)}")
        if self.visual_precision == "fp8":
            model_id += "-VITFP8"
        if self.visual_mha_precision == "fp8":
            model_id += "-VITMHAFP8"
        if self.audio_precision == "fp8":
            model_id += "-AUDFP8"
        if self.fp8_kv_cache:
            model_id += "-FP8-KV"
        if self.reduced_vocab_size:
            model_id += f"-rvs{self.reduced_vocab_size}"
        if self.trt_native_attn:
            model_id += "-trt11"
        if self.nvfp4_moe_target:
            model_id += f"-{self.nvfp4_moe_target}"
        return model_id

    def get_export_env_vars(self) -> dict:
        """Environment variables that affect ONNX export for this config."""
        if self.nvfp4_moe_target:
            return {_NVFP4_MOE_TARGET_ENV: self.nvfp4_moe_target}
        return {}

    def get_onnx_model_id(self) -> str:
        """Backward-compatible alias for quantized model id."""
        return self.get_quantized_model_id()

    def get_engine_id(self) -> str:
        """Generate unique engine identifier"""
        mxlr = self.max_lora_rank if self.max_lora_rank is not None else 0
        llm_engine_id = f"{self.get_quantized_model_id()}-mxil{self.max_input_len}-mxbs{self.max_batch_size}-mxlr{mxlr}"
        if self.model_type == ModelType.VLM:
            llm_engine_id += f"-mnit{self.min_image_tokens}-mxit{self.max_image_tokens}"
        if self.model_type == ModelType.OMNI:
            llm_engine_id += (
                f"-mnit{self.min_image_tokens}-mxit{self.max_image_tokens}"
                f"-mnts{self.min_time_steps}-mxts{self.max_time_steps}")
        if (self.is_eagle or self.is_mtp or self.is_dflash or self.is_jetspec
                or self.is_dspark):
            if self.max_verify_tree_size is not None:
                llm_engine_id += f"-mvts{self.max_verify_tree_size}"
            if self.max_draft_tree_size is not None:
                llm_engine_id += f"-mdts{self.max_draft_tree_size}"
        if self.max_kv_pool_pages is not None:
            llm_engine_id += f"-mxkvp{self.max_kv_pool_pages}"
        return llm_engine_id

    @staticmethod
    def _strip_model_quant_suffixes(model_name: str) -> str:
        return strip_model_quant_suffixes(model_name)

    @staticmethod
    def _llm_precision_to_quant_suffix(llm_precision: str,
                                       modifier_parts: list) -> Optional[str]:
        """Map export precision modifiers to hub quantization_checkpoint folder suffix."""
        if llm_precision == "fp16":
            suffixes = []
            if any(p.lower() == "vitfp8" for p in modifier_parts):
                suffixes.append("VITFP8")
            if any(p.lower() == "vitmhafp8" for p in modifier_parts):
                suffixes.append("VITMHAFP8")
            if any(p.lower() == "fp8kv" for p in modifier_parts):
                suffixes.append("FP8-KV")
            return "-".join(suffixes) if suffixes else None
        model_id = TestConfig._canonical_quant_suffix(llm_precision)
        for part in modifier_parts:
            lower = part.lower()
            if lower == "lmfp8":
                model_id += "-LMFP8"
            elif lower == "lmmxfp8":
                model_id += "-LMMXFP8"
            elif lower == "lmnvfp4":
                model_id += "-LMNVFP4"
            elif lower == "vitfp8":
                model_id += "-VITFP8"
            elif lower == "vitmhafp8":
                model_id += "-VITMHAFP8"
            elif lower == "fp8kv":
                model_id += "-FP8-KV"
        return model_id

    @staticmethod
    def _append_quant_suffix_once(model_dir_name: str,
                                  quant_suffix: str) -> str:
        """Append only the missing part of a derived quant suffix."""
        full_suffix = f"-{quant_suffix}"
        if model_dir_name.upper().endswith(full_suffix.upper()):
            return model_dir_name

        base_quant, sep, remaining = quant_suffix.partition("-")
        if sep and model_dir_name.upper().endswith(f"-{base_quant.upper()}"):
            return f"{model_dir_name}-{remaining}"
        if not sep and model_dir_name.upper().endswith(
                f"-{base_quant.upper()}"):
            return model_dir_name
        return f"{model_dir_name}{full_suffix}"

    def get_quantized_checkpoint_dir_name(self) -> Optional[str]:
        """Directory name for a pre-quantized checkpoint on the model hub."""
        base_model_name = self._strip_model_quant_suffixes(self.model_name)
        if self.model_name != base_model_name:
            return self.model_name

        parts = self.param_str.split('-')
        spec_idx = -1
        spec_tokens = {"dflash", "dspark", "eagle", "jetspec", "mtp"}
        for i, part in enumerate(parts):
            if part.lower() in spec_tokens:
                spec_idx = i
                break
        scan_end = spec_idx if spec_idx > 0 else len(parts)

        precision_idx = -1
        for i in range(scan_end):
            if parts[i] in VALID_LLM_PRECISIONS:
                precision_idx = i
                break
        if precision_idx < 0:
            return None

        modifier_parts = parts[precision_idx + 1:scan_end]
        llm_prec = parts[precision_idx]
        quant_suffix = self._llm_precision_to_quant_suffix(
            llm_prec, modifier_parts)
        mod_lower = [p.lower() for p in modifier_parts]
        extras = []
        if self.visual_precision == "fp8" and "vitfp8" not in mod_lower:
            extras.append("VITFP8")
        if (self.visual_mha_precision == "fp8"
                and "vitmhafp8" not in mod_lower):
            extras.append("VITMHAFP8")
        if self.audio_precision == "fp8":
            extras.append("AUDFP8")
        if self.fp8_kv_cache and "fp8kv" not in mod_lower:
            extras.append("FP8-KV")

        pieces = []
        if quant_suffix:
            pieces.append(quant_suffix)
        for tag in extras:
            if tag not in pieces and not any(tag in piece for piece in pieces):
                pieces.append(tag)
        if not pieces:
            return None
        return f"{base_model_name}-{'-'.join(pieces)}"

    def get_quantized_draft_checkpoint_dir_name(self) -> Optional[str]:
        """Hub folder name for a pre-quantized EAGLE/DFlash/JetSpec draft checkpoint."""
        if (self.is_mtp or self.draft_llm_precision == "fp16"
                or not (self.is_eagle or self.is_dflash or self.is_jetspec)):
            return None
        if self.draft_model_id is None or self.draft_llm_precision is None:
            return None

        parts = self.param_str.split('-')
        draft_dir_name = None
        draft_modifiers: list = []

        if self.is_dflash or self.is_jetspec:
            spec_token = "jetspec" if self.is_jetspec else "dflash"
            spec_idx = -1
            for i, part in enumerate(parts):
                if part.lower() == spec_token:
                    spec_idx = i
                    break
            if spec_idx < 0:
                return None
            draft_modifiers = parts[spec_idx + 3:]
            draft_models = (self._jetspec_draft_models_for_base()
                            if self.is_jetspec else
                            self._dflash_draft_models_for_base())
            if not draft_models or self.draft_model_id not in draft_models:
                return None
            draft_dir_name = draft_models[self.draft_model_id]
        else:
            eagle_idx = -1
            for i, part in enumerate(parts):
                if part.lower() == "eagle":
                    eagle_idx = i
                    break
            if eagle_idx < 0 or eagle_idx + 2 >= len(parts):
                return None

            draft_precision = parts[eagle_idx + 2]
            if draft_precision not in VALID_LLM_PRECISIONS:
                return None

            draft_modifiers = parts[eagle_idx + 3:]
            base_model_name = self._strip_model_quant_suffixes(self.model_name)
            draft_models = MODEL_NAME_TO_DRAFT_MODELS_MAP.get(base_model_name)
            if not draft_models or self.draft_model_id not in draft_models:
                return None
            draft_dir_name = draft_models[self.draft_model_id]

        quant_suffix = self._llm_precision_to_quant_suffix(
            self.draft_llm_precision, draft_modifiers)
        if not quant_suffix:
            return None
        return self._append_quant_suffix_once(draft_dir_name, quant_suffix)

    def get_torch_model_dir(self) -> str:
        """Resolve torch/hub checkpoint; prefers hub pre-quantized when present."""
        return self._resolve_torch_model_dir(prefer_hub_quant=True)

    def get_base_torch_model_dir(self) -> str:
        """FP16 (or GPTQ) torch checkpoint used as quantization input."""
        return self._resolve_torch_model_dir(prefer_hub_quant=False)

    def _resolve_torch_model_dir(self, *, prefer_hub_quant: bool) -> str:
        """
        Get torch model directory path using dynamic search.
        
        Searches for the model directory under llm_models_dir.
        
        Raises:
            ValueError: If llm_models_dir is not set or model directory is not found
        """

        # Determine search directory and model path. A map entry may be either
        # a single directory name (str) or a list of candidates — useful when
        # the same model ships under multiple folder names (e.g. ``InternVL3-1B``
        # as ``InternVL3-1B-hf`` or ``InternVL3-1B``).
        base_model_name = self._strip_model_quant_suffixes(self.model_name)
        use_torch_base = (not prefer_hub_quant
                          and base_model_name != self.model_name
                          and base_model_name in LLM_MODELS_DIR_MAP)
        use_torch_gptq = (not prefer_hub_quant
                          and self.llm_precision == "int4_gptq"
                          and base_model_name in GPTQ_MODELS_DIR_MAP)

        if use_torch_base:
            search_dir = self.llm_models_dir
            entry = LLM_MODELS_DIR_MAP[base_model_name]
            candidates = [entry] if isinstance(entry, str) else list(entry)
        elif use_torch_gptq:
            search_dir = self.edgellm_data_dir
            entry = GPTQ_MODELS_DIR_MAP[base_model_name]
            candidates = [entry] if isinstance(entry, str) else list(entry)
        elif self.model_name in GPTQ_MODELS_DIR_MAP:
            search_dir = self.edgellm_data_dir
            entry = GPTQ_MODELS_DIR_MAP[self.model_name]
            candidates = [entry] if isinstance(entry, str) else list(entry)
        elif self.model_name in LLM_MODELS_DIR_MAP:
            search_dir = self.llm_models_dir
            entry = LLM_MODELS_DIR_MAP[self.model_name]
            candidates = [entry] if isinstance(entry, str) else list(entry)
        else:
            # Map by base model name, not precision suffixes.
            # For a pre-quantized variant the precision suffix MUST be matched
            # in the dir name — otherwise we'd silently fall back to the
            # unquantized base model and produce an ONNX whose contents
            # contradict its dir name (e.g. ``llm-fp8-lmfp8/`` with FP16
            # weights). Restrict candidates to the exact model_name so a
            # missing pre-quant checkpoint fails loud here instead of being
            # papered over by the fp16 base.
            base_model_name = self._strip_model_quant_suffixes(self.model_name)
            if base_model_name in LLM_MODELS_DIR_MAP:
                search_dir = self.llm_models_dir
                candidates = [self.model_name]
            elif base_model_name in GPTQ_MODELS_DIR_MAP:
                search_dir = self.edgellm_data_dir
                candidates = [self.model_name]
            else:
                all_models = list(LLM_MODELS_DIR_MAP.keys()) + list(
                    GPTQ_MODELS_DIR_MAP.keys())
                raise ValueError(
                    f"Unsupported model name: '{self.model_name}'. "
                    f"Supported models: {', '.join(all_models)}")

        # Keep candidate order but remove duplicates.
        candidates = list(dict.fromkeys(candidates))

        if prefer_hub_quant:
            checkpoint_dir_name = self.get_quantized_checkpoint_dir_name()
            if checkpoint_dir_name:
                if self.model_name == base_model_name:
                    candidates = [checkpoint_dir_name]
                elif checkpoint_dir_name not in candidates:
                    candidates.insert(0, checkpoint_dir_name)

        # Resolve across both model roots to tolerate environment differences
        # (some setups map pre-quantized checkpoints under llm_models_dir, some
        # under edgellm_data_dir).
        search_roots = [search_dir]
        for fallback_root in (self.llm_models_dir, self.edgellm_data_dir):
            if fallback_root and fallback_root not in search_roots:
                search_roots.append(fallback_root)

        for root in search_roots:
            for model_dir_name in candidates:
                model_dir = _find_directory(root,
                                            model_dir_name,
                                            DEFAULT_SEARCH_DEPTH,
                                            require_files=_HF_CHECKPOINT_FILES)
                if model_dir:
                    return model_dir

        raise ValueError(
            f"Model directory not found: none of {candidates} under any of "
            f"{search_roots} (search depth {DEFAULT_SEARCH_DEPTH}, "
            f"requiring config.json + checkpoint weights or index)")

    def is_prequantized(self) -> bool:
        """Model is pre-quantized if its name contains the precision suffix."""
        if self.llm_precision == "fp16":
            return False
        if self.llm_precision == "int4_gptq":
            return True
        if self.model_name in PRE_QUANTIZED_MODELS:
            return True
        return self._canonical_quant_suffix(
            self.llm_precision).upper() in self.model_name.upper()

    def _dflash_draft_models_for_base(self) -> Optional[dict]:
        """Resolve DFlash draft map for fp16 or pre-quant base model names."""
        draft_models = MODEL_NAME_TO_DFLASH_DRAFT_MODELS_MAP.get(
            self.model_name)
        if draft_models is not None:
            return draft_models
        base_name = self._strip_model_quant_suffixes(self.model_name)
        return MODEL_NAME_TO_DFLASH_DRAFT_MODELS_MAP.get(base_name)

    def get_dflash_draft_model_dir(self) -> str:
        """Get DFlash draft checkpoint directory using draft_model_id."""
        draft_models = self._dflash_draft_models_for_base()
        if not draft_models:
            raise ValueError(
                f"Unsupported base model for DFlash: '{self.model_name}'. "
                f"Supported models: {', '.join(MODEL_NAME_TO_DFLASH_DRAFT_MODELS_MAP.keys())}"
            )

        lookup_name = (self.model_name if self.model_name
                       in MODEL_NAME_TO_DFLASH_DRAFT_MODELS_MAP else
                       self._strip_model_quant_suffixes(self.model_name))

        if not self.draft_model_id:
            raise ValueError(
                f"draft_model_id not set. Available DFlash drafts for {lookup_name}: "
                f"{', '.join(draft_models.keys())}")

        if self.draft_model_id not in draft_models:
            raise ValueError(
                f"Unsupported DFlash draft_model_id '{self.draft_model_id}' for {lookup_name}. "
                f"Available: {', '.join(draft_models.keys())}")

        model_dir_name = draft_models[self.draft_model_id]
        model_dir = _find_directory(self.llm_models_dir,
                                    model_dir_name,
                                    5,
                                    require_files=_HF_CHECKPOINT_FILES)
        if not model_dir:
            model_dir = _find_directory(self.edgellm_data_dir,
                                        model_dir_name,
                                        5,
                                        require_files=_HF_CHECKPOINT_FILES)
        if not model_dir:
            raise ValueError(
                f"DFlash draft model directory not found: '{model_dir_name}' under "
                f"{self.llm_models_dir} or {self.edgellm_data_dir} with search depth 5 "
                f"(requiring config.json + checkpoint weights or index)")
        return model_dir

    def _jetspec_draft_models_for_base(self) -> Optional[dict]:
        """Resolve JetSpec draft map for fp16 or pre-quant base model names."""
        draft_models = MODEL_NAME_TO_JETSPEC_DRAFT_MODELS_MAP.get(
            self.model_name)
        if draft_models is not None:
            return draft_models
        base_name = self._strip_model_quant_suffixes(self.model_name)
        return MODEL_NAME_TO_JETSPEC_DRAFT_MODELS_MAP.get(base_name)

    def get_jetspec_draft_model_dir(self) -> str:
        """Resolve the JetSpec draft checkpoint directory using draft_model_id."""
        draft_models = self._jetspec_draft_models_for_base()
        if not draft_models:
            raise ValueError(
                f"Unsupported base model for JetSpec: '{self.model_name}'. "
                f"Supported models: {', '.join(MODEL_NAME_TO_JETSPEC_DRAFT_MODELS_MAP.keys())}"
            )

        lookup_name = (self.model_name if self.model_name
                       in MODEL_NAME_TO_JETSPEC_DRAFT_MODELS_MAP else
                       self._strip_model_quant_suffixes(self.model_name))

        if not self.draft_model_id:
            raise ValueError(
                f"draft_model_id not set. Available JetSpec drafts for {lookup_name}: "
                f"{', '.join(draft_models.keys())}")

        if self.draft_model_id not in draft_models:
            raise ValueError(
                f"Unsupported JetSpec draft_model_id '{self.draft_model_id}' for {lookup_name}. "
                f"Available: {', '.join(draft_models.keys())}")

        model_dir_name = draft_models[self.draft_model_id]
        candidates = list(
            dict.fromkeys([
                f"source_models/{model_dir_name}",
                model_dir_name,
                os.path.basename(model_dir_name),
            ]))
        search_roots = []
        if self.edgellm_data_dir:
            search_roots.append(self.edgellm_data_dir)
        if self.llm_models_dir and self.llm_models_dir not in search_roots:
            search_roots.append(self.llm_models_dir)
        return self._resolve_draft_model_dir(candidates, search_roots)

    def get_dspark_draft_model_dir(self) -> str:
        """Resolve the DSpark draft checkpoint directory using draft_model_id."""
        base_model_name = self._strip_model_quant_suffixes(self.model_name)
        if base_model_name not in MODEL_NAME_TO_DSPARK_DRAFT_MODELS_MAP:
            raise ValueError(
                f"Unsupported base model for DSpark: '{self.model_name}'. "
                f"Supported models: {', '.join(MODEL_NAME_TO_DSPARK_DRAFT_MODELS_MAP.keys())}"
            )

        draft_models = MODEL_NAME_TO_DSPARK_DRAFT_MODELS_MAP[base_model_name]
        if not self.draft_model_id:
            raise ValueError(
                f"draft_model_id not set. Available DSpark drafts for {base_model_name}: "
                f"{', '.join(draft_models.keys())}")
        if self.draft_model_id not in draft_models:
            raise ValueError(
                f"Unsupported DSpark draft_model_id '{self.draft_model_id}' for {base_model_name}. "
                f"Available: {', '.join(draft_models.keys())}")

        model_dir_name = draft_models[self.draft_model_id]
        candidates = list(
            dict.fromkeys([model_dir_name,
                           os.path.basename(model_dir_name)]))
        return self._resolve_draft_model_dir(candidates,
                                             self._draft_torch_search_roots())

    def get_audio_engine_dir(self) -> str:
        suffix = f"audio-{self.audio_precision}"
        if self.min_time_steps is not None and self.max_time_steps is not None:
            suffix += f"-mnts{self.min_time_steps}-mxts{self.max_time_steps}"
        return os.path.join(self.get_engine_base_dir(), suffix)

    def _resolve_draft_model_dir(self, candidates: list[str],
                                 search_roots: list[str]) -> str:
        for root in search_roots:
            for candidate in candidates:
                model_dir = _find_directory(
                    root,
                    candidate,
                    5,
                    require_files=_HF_CHECKPOINT_FILES,
                )
                if model_dir:
                    return model_dir
        raise ValueError(
            f"Draft model directory not found: none of {candidates} under "
            f"{search_roots} (requiring config.json + checkpoint weights or index)"
        )

    def _draft_torch_search_roots(self) -> list[str]:
        roots = []
        if self.llm_models_dir:
            roots.append(self.llm_models_dir)
        if self.edgellm_data_dir and self.edgellm_data_dir not in roots:
            roots.append(self.edgellm_data_dir)
        return roots

    def _draft_hub_search_roots(self) -> list[str]:
        roots = list(self._draft_torch_search_roots())
        if self.onnx_dir and self.onnx_dir not in roots:
            roots.append(self.onnx_dir)
        try:
            onnx_base = self.get_onnx_base_dir()
            if onnx_base not in roots:
                roots.append(onnx_base)
        except ValueError:
            pass
        return roots

    def _draft_model_dir_name(self) -> str:
        """Map base model + draft_model_id to torch/hub folder name."""
        base_model_name = self._strip_model_quant_suffixes(self.model_name)
        if base_model_name not in MODEL_NAME_TO_DRAFT_MODELS_MAP:
            raise ValueError(
                f"Unsupported base model for EAGLE: '{self.model_name}'. "
                f"Supported models: {', '.join(MODEL_NAME_TO_DRAFT_MODELS_MAP.keys())}"
            )

        draft_models = MODEL_NAME_TO_DRAFT_MODELS_MAP[base_model_name]

        if not self.draft_model_id:
            raise ValueError(
                f"draft_model_id not set. Available draft models for {base_model_name}: "
                f"{', '.join(draft_models.keys())}")

        if self.draft_model_id not in draft_models:
            raise ValueError(
                f"Unsupported draft_model_id '{self.draft_model_id}' for {base_model_name}. "
                f"Available: {', '.join(draft_models.keys())}")

        return draft_models[self.draft_model_id]

    def get_draft_torch_model_dir(self) -> str:
        """FP16 torch draft checkpoint (input for draft quantization)."""
        return self._resolve_draft_model_dir(
            [self._draft_model_dir_name()],
            self._draft_torch_search_roots(),
        )

    def get_draft_model_dir(self) -> str:
        """
        Get draft model directory using draft_model_id.
        Prefers hub/local pre-quantized draft when present, else torch draft.
        """
        base_model_name = self._strip_model_quant_suffixes(self.model_name)
        model_dir_name = self._draft_model_dir_name()
        candidates = []
        quant_draft_name = self.get_quantized_draft_checkpoint_dir_name()
        if quant_draft_name:
            candidates.append(quant_draft_name)
        candidates.append(model_dir_name)
        if self.draft_llm_precision and self.draft_llm_precision != "fp16":
            lm_head = self.draft_lm_head_precision or "fp16"
            candidates.append(
                f"quantized-draft/quantized-{self.draft_model_id}-"
                f"{self.draft_llm_precision}-{lm_head}")
            candidates.append(f"{base_model_name}_{self.draft_model_id}-"
                              f"{self.draft_llm_precision.upper()}")
        candidates = list(dict.fromkeys(candidates))
        return self._resolve_draft_model_dir(candidates,
                                             self._draft_hub_search_roots())

    def get_gemma4_mtp_assistant_model_dir(self) -> str:
        """Resolve the paired Gemma4 assistant checkpoint for MTP export."""
        base_model_name = self._strip_model_quant_suffixes(self.model_name)
        if base_model_name not in GEMMA4_MTP_ASSISTANT_MODELS_MAP:
            raise ValueError(
                f"Unsupported Gemma4 MTP model: '{self.model_name}'. "
                f"Supported models: {', '.join(GEMMA4_MTP_ASSISTANT_MODELS_MAP.keys())}"
            )

        entry = GEMMA4_MTP_ASSISTANT_MODELS_MAP[base_model_name]
        candidates = [entry] if isinstance(entry, str) else list(entry)
        return self._resolve_draft_model_dir(
            candidates,
            self._draft_torch_search_roots(),
        )

    def get_onnx_base_dir(self) -> str:
        """Get ONNX model base directory.

        A pre-quantized variant registered under its own name in
        LLM_MODELS_DIR_MAP / GPTQ_MODELS_DIR_MAP keeps that full name
        (precision suffix included) so the ONNX dir matches the registered
        source checkpoint folder and get_engine_base_dir. A variant resolved
        only via the base-model fallback (model_name is not a registered key)
        uses the stripped base name; its precision lives in the llm-<prec>
        subdir.
        """
        if not self.onnx_dir:
            raise ValueError("onnx_dir not set")
        if (self.model_name in LLM_MODELS_DIR_MAP
                or self.model_name in GPTQ_MODELS_DIR_MAP):
            name = self.model_name
        else:
            name = self._strip_model_quant_suffixes(self.model_name)
        return os.path.join(self.onnx_dir, name)

    def get_engine_base_dir(self) -> str:
        """Get engine base directory"""
        if not self.engine_dir:
            raise ValueError("engine_dir not set")
        return os.path.join(self.engine_dir, self.model_name)

    def _get_llm_onnx_model_id(self) -> str:
        """Generate the LLM ONNX subdirectory identifier."""
        onnx_model_id = f"{self.llm_precision.lower()}-{self.lm_head_precision.lower()}"
        if self.fp8_kv_cache:
            onnx_model_id += "-fp8kv"
        if self.reduced_vocab_size:
            onnx_model_id += f"-rvs{self.reduced_vocab_size}"
        if self.nvfp4_moe_target:
            onnx_model_id += f"-{self.nvfp4_moe_target}"
        return onnx_model_id

    def get_llm_onnx_dir(self) -> str:
        """Get LLM ONNX model directory. For TTS this contains talker/ and code_predictor/."""
        if self.is_mtp:
            prefix = "llm-base-mtp"
        elif self.is_dflash:
            mode = "ddtree" if self.is_dflash_tree else "linear"
            prefix = f"llm-base-dflash-{mode}"
        elif self.is_jetspec:
            mode = "ddtree" if self.is_jetspec_tree else "linear"
            prefix = f"llm-base-jetspec-{mode}"
        elif self.is_dspark:
            prefix = "llm-base-dspark"
        elif self.is_eagle:
            prefix = "llm-base"
        else:
            prefix = "llm"
        return os.path.join(self.get_onnx_base_dir(),
                            f"{prefix}-{self._get_llm_onnx_model_id()}")

    def get_tts_tokenizer_dir(self) -> str:
        """
        Get tokenizer directory for TTS benchmark/inference.

        Prefer ONNX export output, where ``tokenizer.json`` is written: the
        tensorrt_edgellm-based export drops it under ``llm-<prec>/talker/`` (since
        the talker submodel is what carries the language modeling head), so
        check there first; older exports placed it directly at
        ``llm-<prec>/``. Fall back to the torch model dir
        (``vocab.json`` + ``tokenizer_config.json``) when neither has it.
        """
        onnx_llm_dir = self.get_llm_onnx_dir()
        candidates = [
            os.path.join(onnx_llm_dir, "talker"),
            onnx_llm_dir,
        ]
        for candidate in candidates:
            if os.path.isfile(os.path.join(candidate, "tokenizer.json")):
                return candidate
        return self.get_torch_model_dir()

    def get_audio_onnx_dir(self, precision: Optional[str] = None) -> str:
        """Audio encoder ONNX directory.

        ``precision=None`` uses ``self.audio_precision``. Pass an explicit
        precision (``"fp16"`` / ``"fp8"``) when the test path needs to refer to
        a specific variant.
        """
        p = precision or self.audio_precision
        return os.path.join(self.get_onnx_base_dir(), f"audio-{p}")

    def get_code2wav_onnx_dir(self, precision: Optional[str] = None) -> str:
        """Code2Wav ONNX directory for TTS models."""
        p = precision or self.audio_precision
        return os.path.join(self.get_onnx_base_dir(), f"code2wav-{p}")

    def get_action_onnx_dir(self) -> str:
        return os.path.join(self.get_onnx_base_dir(), "action-fp16")

    def get_action_engine_dir(self) -> str:
        # action_build appends /action to --engineDir, so passing
        # get_visual_engine_dir() lands action.engine next to visual.engine
        # under the same parent dir, which is what action_inference
        # --multimodalEngineDir expects.
        return os.path.join(self.get_visual_engine_dir(), "action")

    def get_alpamayo_dataset_dir(self) -> str:
        # Resolves $ALPAMAYO_DATASET_DIR to the active test_case's folder.
        return os.path.dirname(self.get_test_case_file())

    def get_draft_onnx_model_id(self) -> str:
        """Generate unique draft model identifier including draft_model_id"""
        if self.draft_model_id is None:
            raise ValueError("draft_model_id not set")
        if self.draft_llm_precision is None:
            raise ValueError("draft_llm_precision not set")
        if self.draft_lm_head_precision is None:
            raise ValueError("draft_lm_head_precision not set")
        draft_id = f"{self.draft_model_id}-{self.draft_llm_precision}-{self.draft_lm_head_precision}"
        return draft_id

    def get_draft_onnx_dir(self) -> str:
        """Get draft model ONNX directory"""
        if self.is_mtp:
            return os.path.join(self.get_onnx_base_dir(),
                                f"mtp-draft-{self._get_llm_onnx_model_id()}")
        if self.is_dflash:
            return os.path.join(
                self.get_onnx_base_dir(),
                f"dflash-draft-{self.get_draft_onnx_model_id()}")
        if self.is_jetspec:
            return os.path.join(
                self.get_onnx_base_dir(),
                f"jetspec-draft-{self.get_draft_onnx_model_id()}")
        if self.is_dspark:
            return os.path.join(
                self.get_onnx_base_dir(),
                f"dspark-draft-{self.get_draft_onnx_model_id()}")
        return os.path.join(self.get_onnx_base_dir(),
                            f"draft-{self.get_draft_onnx_model_id()}")

    def get_quantized_draft_model_dir(self) -> str:
        """Local output dir for a quantized EAGLE/DFlash/JetSpec draft."""
        if self.draft_llm_precision == "fp16":
            if self.is_dflash:
                return self.get_dflash_draft_model_dir()
            if self.is_jetspec:
                return self.get_jetspec_draft_model_dir()
            if self.is_dspark:
                return self.get_dspark_draft_model_dir()
            return self.get_draft_torch_model_dir()
        if not self.onnx_dir:
            raise ValueError("onnx_dir not set")
        hub_name = self.get_quantized_draft_checkpoint_dir_name()
        if hub_name:
            return os.path.join(self.onnx_dir, hub_name)
        if self.draft_model_id is None:
            raise ValueError("draft_model_id not set")
        quantized_name = (
            f"quantized-{self.draft_model_id}-"
            f"{self.draft_llm_precision}-{self.draft_lm_head_precision}")
        return os.path.join(self.get_onnx_base_dir(), "quantized-draft",
                            quantized_name)

    def get_eagle_draft_checkpoint_dir(self) -> str:
        """Resolve EAGLE draft HF checkpoint for checkpoint export (hub/local)."""
        if not self.is_eagle or self.is_mtp:
            raise ValueError(
                "get_eagle_draft_checkpoint_dir requires EAGLE config")
        if (self.draft_llm_precision
                and self.draft_llm_precision not in ("fp16", "int4_gptq")):
            # Quantized draft: require the pre-quant draft, never fall back to
            # fp16. Try both naming conventions for the same draft:
            base = self._strip_model_quant_suffixes(self.model_name)
            candidates = []
            # Quantized-variant map key carries the full hub name (separator
            # may differ from the fp16 draft, e.g. Qwen3-1.7B-eagle3-NVFP4).
            if self.model_name != base:
                full = MODEL_NAME_TO_DRAFT_MODELS_MAP.get(
                    self.model_name, {}).get(self.draft_model_id)
                if full:
                    candidates.append(full)
            # Base draft name + derived quant suffix (e.g. Qwen3-1.7B_eagle3-NVFP4).
            derived = self.get_quantized_draft_checkpoint_dir_name()
            if derived:
                candidates.append(derived)
            candidates = list(dict.fromkeys(candidates))
            if not candidates:
                raise ValueError(
                    f"Quantized EAGLE draft requested "
                    f"({self.draft_llm_precision}) but no pre-quant draft name "
                    f"resolved for {self.model_name}/{self.draft_model_id}")
            return self._resolve_draft_model_dir(
                candidates, self._draft_hub_search_roots())
        return self.get_draft_model_dir()

    def get_visual_onnx_dir(self, precision: str) -> str:
        """Get visual ONNX model directory"""
        name = f"visual-{precision}"
        if self.visual_mha_precision == "fp8":
            name += "-mhafp8"
        if self.trt_native_attn:
            name += "-trt11"
        return os.path.join(self.get_onnx_base_dir(), name)

    def get_llm_engine_dir(self) -> str:
        """Get LLM engine directory"""
        if not self.engine_dir:
            raise ValueError(
                "engine_dir not set - required for LLM engine operations")
        if self.task_type == TaskType.EXPORT:
            raise ValueError(
                "LLM engine directory not available for export tasks")

        if self.is_mtp:
            # MTP shares the base checkpoint; no separate draft_model_id.  Precision
            # info comes from get_engine_id() (already includes llm/lm_head precision).
            prefix = "llm-mtp"
        elif self.is_dflash:
            if self.draft_model_id is None:
                raise ValueError("draft_model_id not set for DFlash engine")
            if self.draft_llm_precision is None:
                raise ValueError(
                    "draft_llm_precision not set for DFlash engine")
            mode = "ddtree" if self.is_dflash_tree else "linear"
            prefix = (
                f"llm-dflash-{mode}-{self.draft_model_id}-{self.draft_llm_precision}"
            )
        elif self.is_jetspec:
            if self.draft_model_id is None:
                raise ValueError("draft_model_id not set for JetSpec engine")
            if self.draft_llm_precision is None:
                raise ValueError(
                    "draft_llm_precision not set for JetSpec engine")
            mode = "ddtree" if self.is_jetspec_tree else "linear"
            prefix = (
                f"llm-jetspec-{mode}-{self.draft_model_id}-{self.draft_llm_precision}"
            )
        elif self.is_dspark:
            if self.draft_model_id is None:
                raise ValueError("draft_model_id not set for DSpark engine")
            if self.draft_llm_precision is None:
                raise ValueError(
                    "draft_llm_precision not set for DSpark engine")
            prefix = f"llm-dspark-{self.draft_model_id}-{self.draft_llm_precision}"
        elif self.is_eagle:
            if self.draft_model_id is None:
                raise ValueError("draft_model_id not set for EAGLE engine")
            if self.draft_llm_precision is None:
                raise ValueError(
                    "draft_llm_precision not set for EAGLE engine")
            prefix = f"llm-eagle-{self.draft_model_id}-{self.draft_llm_precision}"
        else:
            prefix = "llm"

        return os.path.join(self.get_engine_base_dir(),
                            f"{prefix}-{self.get_engine_id()}")

    def get_talker_engine_dir(self) -> str:
        """Get Talker engine directory (TTS only). Subdir under LLM engine dir."""
        return os.path.join(self.get_llm_engine_dir(), "talker")

    def get_code_predictor_engine_dir(self) -> str:
        """Get CodePredictor engine directory (TTS only). Subdir under LLM engine dir."""
        return os.path.join(self.get_llm_engine_dir(), "code_predictor")

    def get_code2wav_engine_dir(self) -> str:
        """Get Code2Wav engine directory (TTS only). Subdir under LLM engine dir."""
        return os.path.join(self.get_llm_engine_dir(), "code2wav")

    def get_visual_engine_dir(self) -> str:
        """Get visual engine directory"""
        name = f"visual-{self.visual_precision}"
        if self.visual_mha_precision == "fp8":
            name += "-mhafp8"
        name += (f"-mnit{self.min_image_tokens}"
                 f"-mxit{self.max_image_tokens}"
                 f"-mxpiit{self.max_image_tokens_per_image}")
        if self.trt_native_attn:
            name += "-trt11"
        return os.path.join(self.get_engine_base_dir(), name)

    def get_multimodal_engine_dir(self) -> str:
        """OMNI multimodal engine parent dir.

        visual_build writes to <this>/visual/ and audio_build writes to
        <this>/audio/, so inference / e2e_bench pass
        --multimodalEngineDir=<this> and runtime auto-locates both encoders.
        """
        suffix = (f"multimodal-mnit{self.min_image_tokens}"
                  f"-mxit{self.max_image_tokens}"
                  f"-mxpiit{self.max_image_tokens_per_image}"
                  f"-mnts{self.min_time_steps}"
                  f"-mxts{self.max_time_steps}")
        return os.path.join(self.get_engine_base_dir(), suffix)

    def get_test_case_file(self) -> str:
        """
        Get test case file path using test case name mapping.

        When ``_test_case_file_override`` is set (by helpers that have
        rewritten the test case JSON for this config, e.g. audio
        preprocessing or LoRA placeholder substitution), that path is
        returned so downstream command generation operates on the
        per-config preprocessed copy and never mutates the
        version-controlled source.

        Returns:
            Full path to the test case JSON file

        Raises:
            ValueError: If test_case is not set or not supported
        """
        if self._test_case_file_override:
            return self._test_case_file_override

        if not self.test_case:
            raise ValueError("test_case not set - required for this operation")

        # Test case name to file path mapping
        # Maps logical test case names to their actual file paths.
        # You can adjust this map to match your test case organization.
        mtbench_dataset = ("mtbench_eagle3.json" if self.is_eagle or "qwen3"
                           in str(getattr(self, "model_name", "")).lower() else
                           "mtbench_dataset.json")

        TEST_CASE_NAME_TO_PATH_MAP = {
            # Add test case mappings here, for example:
            "llm_basic":
            "tests/test_cases/llm_basic.json",
            "ragged_batch":
            "tests/test_cases/ragged_batch.json",
            "llm_logit_bias":
            "tests/test_cases/llm_logit_bias.json",
            "llm_lora":
            "tests/test_cases/llm_lora.json",
            "llm_context_reuse":
            "tests/test_cases/llm_context_reuse.json",
            "llm_spec_prefill_evict":
            "tests/test_cases/llm_spec_prefill_evict.json",
            "asr_basic":
            "tests/test_cases/asr_basic.json",
            "librispeech_clean_test":
            f"{self.edgellm_data_dir}/updated_datasets/librispeech_clean_test/librispeech_clean_test.json",
            "tts_basic":
            "tests/test_cases/tts_basic.json",
            "tts_language":
            "tests/test_cases/tts_language.json",
            "tts_instruct":
            "tests/test_cases/tts_instruct.json",
            "SeedTTS_en_meta":
            f"{self.edgellm_data_dir}/updated_datasets/SeedTTS_en_meta/seedtts_en_meta.json",
            "vlm_basic":
            "tests/test_cases/vlm_basic.json",
            "cosmos3_reasoner":
            "tests/test_cases/vlm_basic.json",
            "vlm_lora":
            "tests/test_cases/vlm_lora.json",
            "vlm_context_reuse":
            "tests/test_cases/vlm_context_reuse.json",
            "alpamayo_action_chat":
            f"{self.edgellm_data_dir}/updated_datasets/alpamayo_action_chat/input.json",
            "alpamayo_action_644":
            f"{self.edgellm_data_dir}/updated_datasets/alpamayo_eval_dataset/input.json",
            "mtbench":
            f"{self.edgellm_data_dir}/updated_datasets/updated_MTBench/{mtbench_dataset}",
            "mmmu":
            f"{self.edgellm_data_dir}/updated_datasets/mmmu/mmmu_dataset.json",
            "mmmu_vlmevalkit":
            f"{self.vlmevalkit_data_dir or self.edgellm_data_dir}/updated_datasets/MMMU_VLMEvalKit/mmmu_dataset.json",
            "mmmu_pro_4":
            f"{self.edgellm_data_dir}/updated_datasets/MMMU_Pro_4/mmmu_pro_4_dataset.json",
            "mmmu_pro_10":
            f"{self.edgellm_data_dir}/updated_datasets/MMMU_Pro_10/mmmu_pro_10_dataset.json",
            "mmmu_pro_vision":
            f"{self.edgellm_data_dir}/updated_datasets/MMMU_Pro_vision/mmmu_pro_vision_dataset.json",
            "coco":
            f"{self.edgellm_data_dir}/updated_datasets/coco/dataset.json",
            "OmniBench":
            f"{self.edgellm_data_dir}/updated_datasets/OmniBench/omnibench_dataset.json",
            "mmlu_0":
            f"{self.edgellm_data_dir}/updated_datasets/MMLU_zero_shot/mmlu_dataset.json",
            "mmlu_5":
            f"{self.edgellm_data_dir}/updated_datasets/MMLU_five_shot/mmlu_dataset.json",
            "mmlu_pro":
            f"{self.edgellm_data_dir}/updated_datasets/MMLU_Pro/mmlu_pro_dataset.json",
            "mmstar":
            f"{self.edgellm_data_dir}/updated_datasets/MMStar/mmstar_reference.json",
            "aime":
            f"{self.edgellm_data_dir}/updated_datasets/AIME/aime_dataset.json",
            "gsm8k":
            f"{self.edgellm_data_dir}/updated_datasets/GSM8K/gsm8k_dataset.json",
            "gsm8k_10":
            f"{self.edgellm_data_dir}/updated_datasets/GSM8K/gsm8k_dataset_10.json",
            "humaneval":
            f"{self.edgellm_data_dir}/updated_datasets/HumanEval/humaneval_dataset.json",
            "math500":
            f"{self.edgellm_data_dir}/updated_datasets/MATH500/math500_dataset.json",
        }

        if self.test_case not in TEST_CASE_NAME_TO_PATH_MAP:
            raise ValueError(
                f"Unsupported test case: '{self.test_case}'. "
                f"Supported test cases: {', '.join(TEST_CASE_NAME_TO_PATH_MAP.keys())}"
            )
        test_case_path = TEST_CASE_NAME_TO_PATH_MAP[self.test_case]
        if not os.path.exists(test_case_path):
            raise ValueError(f"Test case file not found: '{test_case_path}'")

        return test_case_path

    def get_chat_template_file(self) -> Optional[str]:
        """
        Get custom chat template file path for models that require it.

        Returns:
            Path to chat template JSON file, or None if no custom template for this model
        """
        try:
            from tensorrt_edgellm.chat_templates import get_template_path
        except ImportError:
            return None

        MODEL_TO_TEMPLATE = {
            "NVIDIA-Nemotron-Nano-9B-v2": "nemotron_nano_v2",
            "NVIDIA-Nemotron-Nano-9B-v2-FP8": "nemotron_nano_v2",
            "NVIDIA-Nemotron-Nano-9B-v2-NVFP4": "nemotron_nano_v2",
            "Qwen3-TTS-12Hz-0.6B-CustomVoice": "qwen3tts",
            "Qwen3-TTS-12Hz-1.7B-CustomVoice": "qwen3tts",
        }

        template_id = MODEL_TO_TEMPLATE.get(self.model_name)
        if template_id:
            return get_template_path(template_id)
        return None

    def get_output_json_file(self) -> str:
        """
        Get output JSON file path.
        Always stored on host in log directory for subsequent processing.
        """
        return os.path.join(self.test_log_dir, f"{self.param_str}.json")

    def get_profile_json_file(self) -> str:
        """Get the per-configuration JSON profile output path."""
        return os.path.join(self.test_log_dir,
                            f"{self.param_str}_profile.json")

    def get_output_audio_dir(self) -> str:
        """
        Get directory for TTS-generated wav files. Uses a subdir per test case
        (param_str) so that different datasets (e.g. SeedTTS_en_meta vs SeedTTS_zh_meta)
        do not overwrite each other's audio_req*.wav.
        """
        return os.path.join(self.test_log_dir, self.param_str)

    def get_vlmevalkit_tsv_file(self) -> str:
        """Return the VLMEvalKit MMMU TSV generated by dataset preparation."""
        data_root = self.vlmevalkit_data_dir or self.edgellm_data_dir
        return os.path.join(data_root, "updated_datasets", "MMMU_VLMEvalKit",
                            "mmmu_dataset.tsv")

    def get_lora_weights_dir(self) -> str:
        """Get LoRA weights directory"""
        return os.path.join(self.get_onnx_base_dir(), "lora_weights")

    def find_lora_adapter_weights_dir(self,
                                      lora_model_name: str) -> Optional[str]:
        """Find LoRA adapter checkpoint under data or model roots."""
        search_roots: list[str] = []
        for root in (self.edgellm_data_dir, os.environ.get("EDGELLM_DATA_DIR"),
                     self.llm_models_dir):
            if root and root not in search_roots:
                search_roots.append(root)
        if not search_roots:
            search_roots.append("/scratch.edge_llm_cache")

        for root in search_roots:
            found = _find_directory(root, lora_model_name,
                                    DEFAULT_SEARCH_DEPTH)
            if found:
                return found
        return None

    def get_lora_adapter_dir(self) -> str:
        """
        Get LoRA adapter directory for models with embedded LoRA (e.g., Phi-4).
        Returns the path to the vision-lora subdirectory in the torch model dir.
        """
        LORA_ADAPTER_SUBDIRS = {
            "Phi-4-multimodal-instruct": "vision-lora",
        }
        if self.model_name not in LORA_ADAPTER_SUBDIRS:
            raise ValueError(
                f"No LoRA adapter mapping for model: {self.model_name}. "
                f"Supported models: {', '.join(LORA_ADAPTER_SUBDIRS.keys())}")
        subdir = LORA_ADAPTER_SUBDIRS[self.model_name]
        return os.path.join(self.get_base_torch_model_dir(), subdir)

    def get_merged_model_dir(self) -> str:
        """Get merged LoRA model directory (for models requiring LoRA merge before quantization)"""
        return os.path.join(self.get_onnx_base_dir(), "merged-vision")

    def get_quantized_model_dir(self) -> str:
        """Get quantized model output directory (hub-aligned under onnx_dir)."""
        hub_name = self.get_quantized_checkpoint_dir_name()
        if hub_name:
            if not self.onnx_dir:
                raise ValueError("onnx_dir not set")
            return os.path.join(self.onnx_dir, hub_name)
        if self.llm_precision == "fp16":
            return self.get_base_torch_model_dir()
        if (self.is_eagle or self.is_dflash
                or self.is_jetspec) and not self.is_mtp:
            prefix = "quantized-base"
        else:
            prefix = "quantized"
        quantized_name = self.get_quantized_model_id()
        return os.path.join(self.get_onnx_base_dir(), prefix, quantized_name)

    def get_kv_cache_quantized_model_dir(self) -> str:
        """
        Output directory for KV-cache-only quantization (fp16 weights).

        Uses the same hub folder name as checkpoint export when available.
        """
        hub_name = self.get_quantized_checkpoint_dir_name()
        if hub_name and self.llm_precision == "fp16":
            if not self.onnx_dir:
                raise ValueError("onnx_dir not set")
            return os.path.join(self.onnx_dir, hub_name)
        return os.path.join(self.get_onnx_base_dir(), "quantized-kvcache",
                            self.get_quantized_model_id())

    def get_reduced_vocab_dir(self) -> str:
        """Get reduced vocabulary directory (for export)"""
        if not self.reduced_vocab_size:
            raise ValueError("reduced_vocab_size not set")
        vocab_name = f"vocab-{self.reduced_vocab_size}"
        if self.vocab_reduction_method:
            vocab_name += f"-{self.vocab_reduction_method}"
        return os.path.join(self.get_onnx_base_dir(), "reduced_vocab",
                            vocab_name)

    def get_cnn_dailymail_dataset_dir(self) -> str:
        """Get CNN DailyMail dataset directory for LLM quantization calibration"""
        if not self.llm_models_dir:
            raise ValueError("llm_models_dir not set")
        dataset_dir = _find_directory(
            self.llm_models_dir, os.path.join("datasets", "cnn_dailymail"),
            DEFAULT_SEARCH_DEPTH)
        if not dataset_dir:
            raise ValueError(
                f"CNN DailyMail dataset directory not found under {self.llm_models_dir}"
            )
        return dataset_dir

    def get_mmmu_dataset_dir(self) -> str:
        """Get MMMU dataset directory for visual model quantization calibration"""
        if not self.llm_models_dir:
            raise ValueError("llm_models_dir not set")
        dataset_dir = _find_directory(self.llm_models_dir,
                                      os.path.join("datasets", "MMMU"),
                                      DEFAULT_SEARCH_DEPTH)
        if not dataset_dir:
            raise ValueError(
                f"MMMU dataset directory not found under {self.llm_models_dir}"
            )
        return dataset_dir

    def get_reference_json_file(self) -> Optional[str]:
        """
        Get model-specific reference JSON file for ROUGE score calculation.
        
        Lookup order:
        1. {test_case}_{model_name}_{llm_precision}.json
        2. {test_case}_{model_name}_fp16.json (fallback)
        3. None (use original test_case_file as reference)
        
        Returns:
            Path to reference JSON file, or None if no model-specific reference exists
        """
        if not self.test_case or not self.edgellm_data_dir:
            return None

        reference_dir = os.path.join(self.edgellm_data_dir,
                                     "specific_model_reference")
        if not os.path.isdir(reference_dir):
            return None

        # Try exact precision match first
        exact_match = os.path.join(
            reference_dir,
            f"{self.test_case}_{self.model_name}_{self.llm_precision}.json")
        if os.path.exists(exact_match):
            return exact_match

        # Fallback to fp16 if current precision is not fp16
        if self.llm_precision != "fp16":
            fp16_fallback = os.path.join(
                reference_dir, f"{self.test_case}_{self.model_name}_fp16.json")
            if os.path.exists(fp16_fallback):
                return fp16_fallback

        return None
