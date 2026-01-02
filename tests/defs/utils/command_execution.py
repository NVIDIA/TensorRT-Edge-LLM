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
Test execution functions for TensorRT Edge-LLM tests.

This module contains the simplified test execution functions for build, inference, 
and benchmark tests. Each function is focused on its specific task without 
unnecessary abstraction layers.
"""

import os
from typing import Any, Dict, Optional

from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import check_file_exists, run_command, run_with_trt_env

from ..config import ModelType, TaskType, TestConfig
from .accuracy import check_accuracy_with_dataset
from .command_generation import (generate_benchmark_commands,
                                 generate_build_commands,
                                 generate_inference_commands)


def execute_build_test(
        config: TestConfig, executable_files: Dict[str, str],
        remote_config: Optional[RemoteConfig], logger,
        env_config: Optional[EnvironmentConfig]) -> Dict[str, Any]:
    """Execute build test for any model type"""

    # Create engine directories
    llm_engine_dir = config.get_llm_engine_dir()
    run_command(['mkdir', '-p', llm_engine_dir], remote_config, 30, logger)

    if config.model_type == ModelType.VLM:
        visual_engine_dir = config.get_visual_engine_dir()
        run_command(['mkdir', '-p', visual_engine_dir], remote_config, 30,
                    logger)

    # Generate all build commands
    commands = generate_build_commands(config, executable_files)

    all_outputs = []

    for i, (cmd, timeout) in enumerate(commands):
        task_name = f"Build step {i+1}/{len(commands)}"
        if logger:
            logger.info(f"Starting {task_name}: {' '.join(cmd)}")

        if i == 0 and check_file_exists(
                os.path.join(config.get_llm_engine_dir(), "llm.engine"),
                remote_config, logger):
            if logger:
                logger.info(
                    "LLM engine already exists. Skipping LLM engine build")
            all_outputs.append("LLM engine already exists - skipped")
            continue

        elif i == 1 and config.model_type == ModelType.VLM and check_file_exists(
                os.path.join(config.get_visual_engine_dir(), f"visual.engine"),
                remote_config, logger):
            if logger:
                logger.info(
                    "Visual engine already exists. Skipping Visual engine build"
                )
            all_outputs.append("Visual engine already exists - skipped")
            continue

        result = run_with_trt_env(cmd, remote_config, timeout, logger,
                                  env_config)
        all_outputs.append(result['output'])

        if not result['success']:
            return {
                'success': False,
                'error':
                f"{task_name} failed: {result.get('error', 'Unknown error')}",
                'output': '\n'.join(all_outputs),
                'test_type': TaskType.BUILD.value
            }

    return {
        'success': True,
        'error': None,
        'output': '\n'.join(all_outputs),
        'test_type': TaskType.BUILD.value
    }


def execute_benchmark_test(
        config: TestConfig, executable_files: Dict[str, str],
        remote_config: Optional[RemoteConfig], logger,
        env_config: Optional[EnvironmentConfig]) -> Dict[str, Any]:
    """Execute benchmark test for any model type"""

    # Generate all benchmark commands
    commands = generate_benchmark_commands(config, executable_files)

    all_outputs = []

    for i, (cmd, timeout) in enumerate(commands):
        task_name = f"Benchmark step {i+1}/{len(commands)}"
        if logger:
            logger.info(f"Starting {task_name}: {' '.join(cmd)}")

        result = run_with_trt_env(cmd, remote_config, timeout, logger,
                                  env_config)
        all_outputs.append(result['output'])

        if not result['success']:
            return {
                'success': False,
                'error':
                f"{task_name} failed: {result.get('error', 'Unknown error')}",
                'output': '\n'.join(all_outputs),
                'test_type': TaskType.BENCHMARK.value
            }

    return {
        'success': True,
        'error': None,
        'output': '\n'.join(all_outputs),
        'test_type': TaskType.BENCHMARK.value
    }


def execute_inference_test(
        config: TestConfig, executable_files: Dict[str, str],
        remote_config: Optional[RemoteConfig], logger,
        env_config: Optional[EnvironmentConfig]) -> Dict[str, Any]:
    """Execute inference test for any model type"""

    # Handle LoRA weights replacement if needed
    if config.max_lora_rank > 0:
        # Edit the test case file to replace $LORA_WEIGHTS_DIR with the lora weights directory
        test_case_file = config.get_test_case_file()
        result = run_command([
            'sed', '-i',
            f's|$LORA_WEIGHTS_DIR|{config.get_lora_weights_dir()}|g',
            test_case_file
        ], remote_config, 300, logger)
        if not result['success']:
            result['test_type'] = TaskType.INFERENCE.value
            return result

    # Generate all inference commands
    commands = generate_inference_commands(config, executable_files)

    all_outputs = []

    for i, (cmd, timeout) in enumerate(commands):
        task_name = f"Inference step {i+1}/{len(commands)}"
        if logger:
            logger.info(f"Starting {task_name}: {' '.join(cmd)}")

        result = run_with_trt_env(cmd, remote_config, timeout, logger,
                                  env_config)
        all_outputs.append(result['output'])

        if not result['success']:
            return {
                'success': False,
                'error':
                f"{task_name} failed: {result.get('error', 'Unknown error')}",
                'output': '\n'.join(all_outputs),
                'test_type': TaskType.INFERENCE.value
            }

    # Calculate metrics based on dataset type
    final_result = {
        'success': True,
        'error': None,
        'output': '\n'.join(all_outputs),
        'test_type': TaskType.INFERENCE.value
    }

    try:
        # Pass file paths directly to the accuracy checker (runs on host only)
        metrics_result = check_accuracy_with_dataset(
            config.get_output_json_file(), config.get_test_case_file(),
            config.test_case, logger)

        # Merge metrics result into final result
        final_result.update(metrics_result)

    except Exception as e:
        final_result['error'] = f"Failed to calculate metrics: {str(e)}"
        final_result['success'] = False

    return final_result
