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
Test suite for LLM pipeline functionality.

Tests engine building, chat inference, benchmarking, and batch inference
using parameterized configurations.
"""
import logging
import os
import re
from typing import Dict, List, Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import run_command, timer_context

from .config import ModelType, TaskType, TestConfig
from .utils.command_execution import (check_result_failures,
                                      execute_build_test,
                                      execute_e2e_bench_test,
                                      execute_inference_test,
                                      execute_kernel_bench_test)

_COMPACT_ENGINE_TEST_FILTER = 'CompactSessionEngineTest.*'


def _compact_engine_test_failures(output: str) -> List[str]:
    """Reasons a gtest run of the real-engine compact session tests did not
    run and pass every selected case."""
    failures = []
    passed = re.search(r'^\[  PASSED  \] (\d+) tests?\.', output, re.MULTILINE)
    if passed is None or int(passed.group(1)) == 0:
        failures.append(
            f"no {_COMPACT_ENGINE_TEST_FILTER} case ran and passed")
    skipped = re.search(r'^\[  SKIPPED \] (\d+) tests?', output, re.MULTILINE)
    if skipped is not None:
        failures.append(f"{skipped.group(1)} {_COMPACT_ENGINE_TEST_FILTER} "
                        "case(s) skipped")
    return failures


class TestLLMPipeline:
    """Test suite for LLM pipeline functionality."""

    def test_engine_build(self, test_param: str, executable_files: Dict[str,
                                                                        str],
                          remote_config: Optional[RemoteConfig],
                          test_logger: logging.Logger,
                          env_config: EnvironmentConfig) -> None:
        """Test TensorRT engine building for LLM models."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.BUILD, env_config)

        with timer_context(f"LLM build for {config.model_name}", test_logger):
            result = execute_build_test(config, executable_files,
                                        remote_config, test_logger, env_config)
            if not result['success']:
                pytest.fail(f"Build failed: {result['error']}")

    def test_e2e_bench(self, test_param: str, executable_files: Dict[str, str],
                       remote_config: Optional[RemoteConfig],
                       test_logger: logging.Logger,
                       env_config: EnvironmentConfig) -> None:
        """Test end-to-end benchmarking for LLM models."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.E2E_BENCH, env_config)

        with timer_context(f"LLM e2e_bench for {config.model_name}",
                           test_logger):
            result = execute_e2e_bench_test(config, executable_files,
                                            remote_config, test_logger,
                                            env_config)
            if not result['success']:
                pytest.fail(f"e2e_bench failed: {result['error']}")
            check_result_failures(result)

    def test_kernel_bench(self, test_param: str, executable_files: Dict[str,
                                                                        str],
                          remote_config: Optional[RemoteConfig],
                          test_logger: logging.Logger,
                          env_config: EnvironmentConfig) -> None:
        """Test request-backed benchmarking with llm_bench."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.KERNEL_BENCH,
                                              env_config)

        with timer_context(f"kernel_bench for {config.model_name}",
                           test_logger):
            result = execute_kernel_bench_test(config, executable_files,
                                               remote_config, test_logger,
                                               env_config)
            if not result['success']:
                pytest.fail(f"kernel_bench failed: {result['error']}")

    def test_compact_engine_tests(self, test_param: str,
                                  executable_files: Dict[str, str],
                                  remote_config: Optional[RemoteConfig],
                                  test_logger: logging.Logger,
                                  env_config: EnvironmentConfig) -> None:
        """Every real-engine compact session test runs and passes on the case
        engine; the C++ unit jobs have no engine, so these cases skip there."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        build_lib = os.path.join(env_config.build_dir, 'lib')
        plugin = os.path.join(build_lib, 'libNvInfer_edgellm_plugin.so')
        library_path = f"{build_lib}:$LD_LIBRARY_PATH"
        if env_config.trt_package_dir:
            library_path += f":{env_config.trt_package_dir}/lib"
        env_vars = {
            'LD_LIBRARY_PATH': library_path,
            'EDGELLM_PLUGIN_LIB': plugin,
            'EDGELLM_PLUGIN_PATH': plugin,
            'EDGELLM_COMPACT_ENGINE_DIR': config.get_llm_engine_dir(),
        }
        cmd = [
            os.path.join(env_config.build_dir, 'bin', 'unitTests',
                         'unitTestRuntime'),
            f'--gtest_filter={_COMPACT_ENGINE_TEST_FILTER}'
        ]
        with timer_context(
                f"Compact session engine tests for {config.model_name}",
                test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=900,
                                 logger=test_logger,
                                 env_vars=env_vars)
        output = result.get('output', '')
        failures = _compact_engine_test_failures(output)
        if not result['success'] or failures:
            pytest.fail("Compact session engine tests failed:\n" +
                        "\n".join(failures) +
                        f"\n{output}{result.get('error', '')}")

    def test_inference(self, test_param: str, executable_files: Dict[str, str],
                       remote_config: Optional[RemoteConfig],
                       test_logger: logging.Logger,
                       env_config: EnvironmentConfig) -> None:
        """Test batch inference for LLM models."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)

        with timer_context(f"LLM inference for {config.model_name}",
                           test_logger):
            result = execute_inference_test(config, executable_files,
                                            remote_config, test_logger,
                                            env_config)
            if not result['success']:
                pytest.fail(f"Inference failed: {result['error']}")
            check_result_failures(result)
