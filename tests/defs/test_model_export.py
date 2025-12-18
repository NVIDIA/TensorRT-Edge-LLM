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
"""Model export test suite for TensorRT Edge-LLM"""

import os

import pytest
from conftest import EnvironmentConfig
from pytest_helpers import timer_context

from .config import ModelType, TaskType, TestConfig
from .utils.command_generation import generate_export_commands


def validate_export_result(config: TestConfig) -> None:
    """Simple file validation - fail fast"""
    output_dir = config.get_llm_onnx_dir()

    llm_onnx = os.path.join(output_dir, "model.onnx")
    if not os.path.exists(llm_onnx):
        raise FileNotFoundError(f"LLM ONNX model not found: {llm_onnx}")

    if config.lora:
        lora_onnx = os.path.join(config.get_llm_onnx_dir(), "lora_model.onnx")
        if not os.path.exists(lora_onnx):
            raise FileNotFoundError(f"LoRA ONNX model not found: {lora_onnx}")

    if config.model_type == ModelType.VLM:
        # Visual model should be in separate visual subdirectory
        fp16_visual_onnx_dir = config.get_visual_onnx_dir("fp16")
        if not os.path.exists(fp16_visual_onnx_dir):
            raise FileNotFoundError(
                f"Visual ONNX model not found: {fp16_visual_onnx_dir}")
        if config.visual_precision == "fp8":
            fp8_visual_onnx_dir = config.get_visual_onnx_dir("fp8")
            if not os.path.exists(fp8_visual_onnx_dir):
                raise FileNotFoundError(
                    f"Visual ONNX model not found: {fp8_visual_onnx_dir}")

    if config.is_eagle:
        draft_onnx_dir = config.get_draft_onnx_dir()
        draft_onnx = os.path.join(draft_onnx_dir, "model.onnx")
        if not os.path.exists(draft_onnx):
            raise FileNotFoundError(
                f"Draft ONNX model not found: {draft_onnx}")


class TestModelExport:
    """Unified test suite for model export"""

    def test_model_export(self, test_param: str, test_logger,
                          model_type: ModelType,
                          env_config: EnvironmentConfig):
        """Universal export test - handles both LLM and VLM"""

        config = TestConfig.from_param_string(test_param, model_type,
                                              TaskType.EXPORT, env_config)

        # Validate pre-existing models
        torch_dir = config.get_torch_model_dir()
        if not os.path.exists(torch_dir):
            raise FileNotFoundError(f"Torch model not found: {torch_dir}")

        if config.is_eagle:
            draft_torch_dir = config.get_draft_model_dir()
            if not os.path.exists(draft_torch_dir):
                raise FileNotFoundError(
                    f"Draft model not found: {draft_torch_dir}")

        # Create output directories
        llm_onnx_dir = config.get_llm_onnx_dir()
        print(f"Creating output directory: {llm_onnx_dir}")
        os.makedirs(llm_onnx_dir, exist_ok=True)

        # Create quantized model directory if needed
        if config.llm_precision != "fp16" and config.llm_precision != "int4_gptq":
            quantized_model_dir = config.get_quantized_model_dir()
            os.makedirs(quantized_model_dir, exist_ok=True)

        if config.is_eagle:
            draft_onnx_dir = config.get_draft_onnx_dir()
            os.makedirs(draft_onnx_dir, exist_ok=True)

            if config.draft_llm_precision and config.draft_llm_precision != "fp16":
                quantized_draft_dir = config.get_quantized_draft_model_dir()
                os.makedirs(quantized_draft_dir, exist_ok=True)

        # Install gptqmodel for GPTQ models (required dependencies are in tests/requirements.txt)
        if config.llm_precision == "int4_gptq":
            from pytest_helpers import run_command

            # Install gptqmodel 4.2.5 (stable version that works with GPTQ models)
            install_gptq_cmd = [
                "bash", "-c",
                "BUILD_CUDA_EXT=0 pip install -v gptqmodel==4.2.5 --no-build-isolation"
            ]
            result = run_command(install_gptq_cmd,
                                 timeout=300,
                                 remote_config=None,
                                 logger=test_logger)
            if not result['success']:
                pytest.fail(
                    f"Failed to install gptqmodel: {result.get('error', 'Unknown error')}"
                )

        commands = generate_export_commands(config)

        with timer_context(
                f"Exporting {config.model_type.value} {config.model_name} to {config.llm_precision}",
                test_logger):
            for i, (cmd, timeout) in enumerate(commands):
                task_name = f"Export step {i+1}/{len(commands)}"

                from pytest_helpers import run_command
                result = run_command(cmd,
                                     timeout=timeout,
                                     remote_config=None,
                                     logger=test_logger)
                if not result['success']:
                    pytest.fail(
                        f"{task_name} failed: {result.get('error', 'Unknown error')}"
                    )

        validate_export_result(config)

    def test_llm_model_export(self, test_param: str, test_logger,
                              env_config: EnvironmentConfig):
        """LLM export test entry point"""
        self.test_model_export(test_param, test_logger, ModelType.LLM,
                               env_config)

    def test_vlm_model_export(self, test_param: str, test_logger,
                              env_config: EnvironmentConfig):
        """VLM export test entry point"""
        self.test_model_export(test_param, test_logger, ModelType.VLM,
                               env_config)
