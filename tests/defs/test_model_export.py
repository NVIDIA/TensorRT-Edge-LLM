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
"""Model quantization test suite for TensorRT Edge-LLM.

Runs pre-export steps only (merge LoRA, vocab reduction, quantization).
ONNX / checkpoint export is covered by ``test_checkpoint_export.py`` and
``onnx_export_test``; do not use ``test_*_model_export`` names here.
"""

import glob
import json
import os

import pytest
from conftest import EnvironmentConfig
from pytest_helpers import run_command, timer_context

from .config import ModelType, TaskType, TestConfig
from .utils.checkpoint_export_helpers import run_command_list
from .utils.command_generation import generate_pre_export_commands

VOCAB_MAP_NAME = "vocab_map.safetensors"
VOCAB_INFO_NAME = "reduced_vocab.json"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _require_hf_checkpoint(dir_path: str,
                           label: str,
                           *,
                           require_quant_config: bool = False) -> None:
    """Require an HF checkpoint with config.json and weight files."""
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"{label} not found: {dir_path}")

    config_json = os.path.join(dir_path, "config.json")
    if not os.path.isfile(config_json):
        raise FileNotFoundError(f"{label} missing config.json: {dir_path}")

    weight_globs = glob.glob(os.path.join(dir_path, "*.safetensors"))
    weight_globs.extend(glob.glob(os.path.join(dir_path, "*.bin")))
    if not weight_globs:
        raise FileNotFoundError(
            f"{label} missing weight files (*.safetensors / *.bin): {dir_path}"
        )

    if require_quant_config:
        quant_cfg = os.path.join(dir_path, "hf_quant_config.json")
        if not os.path.isfile(quant_cfg):
            raise FileNotFoundError(
                f"{label} missing hf_quant_config.json: {dir_path}")


def _require_hf_quant_checkpoint(dir_path: str, label: str) -> None:
    """Require a ModelOpt unified HF checkpoint, not just an empty directory."""
    _require_hf_checkpoint(dir_path, label, require_quant_config=True)


def _require_reduced_vocab_artifacts(config: TestConfig) -> None:
    reduced_dir = config.get_reduced_vocab_dir()
    if not os.path.isdir(reduced_dir):
        raise FileNotFoundError(
            f"Reduced vocab output not found: {reduced_dir}")

    vocab_map = os.path.join(reduced_dir, VOCAB_MAP_NAME)
    if not os.path.isfile(vocab_map):
        raise FileNotFoundError(
            f"Reduced vocab output missing {VOCAB_MAP_NAME}: {reduced_dir}")
    if os.path.getsize(vocab_map) == 0:
        raise ValueError(
            f"Reduced vocab {VOCAB_MAP_NAME} is empty: {vocab_map}")

    vocab_info = os.path.join(reduced_dir, VOCAB_INFO_NAME)
    if not os.path.isfile(vocab_info):
        raise FileNotFoundError(
            f"Reduced vocab output missing {VOCAB_INFO_NAME}: {reduced_dir}")
    with open(vocab_info, "r") as f:
        metadata = json.load(f)
    actual_size = metadata.get("reduced_vocab_size")
    if actual_size != config.reduced_vocab_size:
        raise ValueError(
            f"Reduced vocab metadata size mismatch: expected "
            f"{config.reduced_vocab_size}, got {actual_size} in {vocab_info}")


def validate_quantization_result(config: TestConfig) -> None:
    """Validate quantization artifacts (files), not pre-created empty dirs."""
    # Deferred import: ckpt_layout_validator imports tensorrt_edgellm internals,
    # so keep it out of module scope to avoid pulling the SDK in at collection.
    from .ckpt_layout_validator import validate_ckpt_layout

    if config.merge_lora:
        _require_hf_checkpoint(config.get_merged_model_dir(),
                               "Merged LoRA model")

    if config.reduced_vocab_size:
        _require_reduced_vocab_artifacts(config)

    needs_weight_quant = (config.llm_precision != "fp16"
                          and not config.is_prequantized())
    needs_kv_cache_quant = bool(config.fp8_kv_cache)
    needs_visual_quant = bool(config.visual_precision == "fp8")
    needs_audio_quant = bool(config.audio_precision == "fp8")

    if needs_weight_quant or needs_visual_quant or needs_audio_quant:
        ckpt_dir = config.get_quantized_model_dir()
        _require_hf_quant_checkpoint(ckpt_dir, "Quantized model")
        validate_ckpt_layout(
            ckpt_dir,
            body_precision=config.llm_precision,
            lm_head_precision=config.lm_head_precision,
            visual_precision=config.visual_precision,
            audio_precision=config.audio_precision,
            label="Quantized model",
        )
    elif needs_kv_cache_quant:
        ckpt_dir = config.get_kv_cache_quantized_model_dir()
        _require_hf_quant_checkpoint(ckpt_dir, "KV-cache quantized model")
        # KV-only quantization: body stays fp16; we still verify nothing weird
        # snuck in (e.g. body silently quantized when name doesn't say so).
        validate_ckpt_layout(
            ckpt_dir,
            body_precision="fp16",
            lm_head_precision="fp16",
            visual_precision="fp16",
            audio_precision="fp16",
            label="KV-cache quantized model",
        )

    if ((config.is_eagle or config.is_dflash or config.is_jetspec)
            and not config.is_mtp and config.draft_llm_precision
            and config.draft_llm_precision not in ("fp16", "int4_gptq")):
        draft_dir = config.get_quantized_draft_model_dir()
        _require_hf_quant_checkpoint(draft_dir, "Quantized draft model")
        validate_ckpt_layout(
            draft_dir,
            body_precision=config.draft_llm_precision,
            lm_head_precision=config.draft_lm_head_precision,
            label="Quantized draft model",
        )

    if config.model_type == ModelType.VLA:
        fp16_visual_onnx_dir = config.get_visual_onnx_dir("fp16")
        if not os.path.exists(fp16_visual_onnx_dir):
            raise FileNotFoundError(
                f"VLA visual ONNX not found: {fp16_visual_onnx_dir}")
        action_onnx = os.path.join(config.get_action_onnx_dir(), "model.onnx")
        if not os.path.exists(action_onnx):
            raise FileNotFoundError(
                f"VLA action expert ONNX not found: {action_onnx}")


def _needs_gptqmodel_install(config: TestConfig) -> bool:
    """True only when on-the-fly int4_gptq base quantization needs gptqmodel."""
    if config.llm_precision != "int4_gptq":
        return False
    if config.is_prequantized():
        return False
    if config.is_dflash or config.is_jetspec or config.is_eagle:
        return False
    return True


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestModelExport:
    """Quantization-only tests (YAML: ``test_*_model_quantization``)."""

    def _run_model_quantization(self, test_param: str, test_logger,
                                model_type: ModelType,
                                env_config: EnvironmentConfig):
        """Universal quantization test for LLM, VLM, ASR, TTS, and OMNI."""

        config = TestConfig.from_param_string(test_param, model_type,
                                              TaskType.EXPORT, env_config)

        torch_dir = config.get_base_torch_model_dir()
        if not os.path.exists(torch_dir):
            raise FileNotFoundError(f"Torch model not found: {torch_dir}")

        if config.is_dflash:
            draft_torch_dir = config.get_dflash_draft_model_dir()
            if not os.path.exists(draft_torch_dir):
                raise FileNotFoundError(
                    f"DFlash draft model not found: {draft_torch_dir}")
        elif config.is_jetspec:
            draft_torch_dir = config.get_jetspec_draft_model_dir()
            if not os.path.exists(draft_torch_dir):
                raise FileNotFoundError(
                    f"JetSpec draft model not found: {draft_torch_dir}")
        elif config.is_eagle and not config.is_mtp:
            draft_torch_dir = config.get_draft_torch_model_dir()
            if not os.path.exists(draft_torch_dir):
                raise FileNotFoundError(
                    f"Draft model not found: {draft_torch_dir}")

        if _needs_gptqmodel_install(config):
            install_gptq_cmd = [
                "bash", "-c",
                "BUILD_CUDA_EXT=0 pip install -v gptqmodel==5.7.0 "
                "--no-build-isolation"
            ]
            result = run_command(install_gptq_cmd,
                                 timeout=300,
                                 remote_config=None,
                                 logger=test_logger)
            if not result['success']:
                pytest.fail(f"Failed to install gptqmodel: "
                            f"{result.get('error', 'Unknown error')}")

        if test_logger:
            test_logger.info("Running quantization-only flow for %s",
                             config.model_name)

        pre_commands = generate_pre_export_commands(config)

        with timer_context(
                f"Quantizing {config.model_type.value} {config.model_name} "
                f"to {config.llm_precision} (tensorrt_edgellm)", test_logger):
            run_command_list(pre_commands, "Pre-export", test_logger)
        validate_quantization_result(config)

    def test_llm_model_quantization(self, test_param: str, test_logger,
                                    env_config: EnvironmentConfig):
        self._run_model_quantization(test_param, test_logger, ModelType.LLM,
                                     env_config)

    def test_tts_model_quantization(self, test_param: str, test_logger,
                                    env_config: EnvironmentConfig):
        self._run_model_quantization(test_param, test_logger, ModelType.TTS,
                                     env_config)

    def test_vlm_model_quantization(self, test_param: str, test_logger,
                                    env_config: EnvironmentConfig):
        self._run_model_quantization(test_param, test_logger, ModelType.VLM,
                                     env_config)

    def test_asr_model_quantization(self, test_param: str, test_logger,
                                    env_config: EnvironmentConfig):
        self._run_model_quantization(test_param, test_logger, ModelType.ASR,
                                     env_config)

    def test_omni_model_quantization(self, test_param: str, test_logger,
                                     env_config: EnvironmentConfig):
        self._run_model_quantization(test_param, test_logger, ModelType.OMNI,
                                     env_config)
