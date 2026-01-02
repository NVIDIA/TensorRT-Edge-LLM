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
"""Common test functions for TensorRT Edge-LLM"""

import logging
import os
from typing import Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import run_command, timer_context

from .utils.device import DeviceConfig


def test_build_project(env_config: EnvironmentConfig,
                       remote_config: Optional[RemoteConfig],
                       test_logger: logging.Logger):
    """Test project build - builds all components"""
    execution_mode = "remote" if remote_config else "local"
    device_config = DeviceConfig.auto_detect(remote_config, test_logger)
    test_logger.info(
        f"Building project for {device_config.target} in {execution_mode} mode"
    )
    build_dir = env_config.build_dir

    # Build cmake command with required components only
    cmake_cmd = ['cmake', '..', '-DBUILD_UNIT_TESTS=ON']

    # Use trt_package_dir from env_config
    if env_config.trt_package_dir:
        cmake_cmd.append(f'-DTRT_PACKAGE_DIR={env_config.trt_package_dir}')
    cmake_cmd.append(f'-DCUDA_VERSION={device_config.cuda_version}')

    if device_config.target in [
            'jetson-orin', 'auto-thor', 'jetson-thor', 'gb10'
    ]:
        cmake_cmd.append(f'-DEMBEDDED_TARGET={device_config.target}')
        cmake_cmd.append(
            '-DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake')

    build_cmd = ' && '.join([
        f'mkdir -p {build_dir}', f'cd {build_dir}', ' '.join(cmake_cmd),
        'make -j16'
    ])

    with timer_context(f"Building ({execution_mode})", test_logger):
        result = run_command(cmd=['bash', '-c', build_cmd],
                             remote_config=remote_config,
                             timeout=600,
                             logger=test_logger)
        success = result['success']

    if not success:
        pytest.fail("Build failed")

    expected_files = [
        'unitTest',
        'examples/llm/llm_build',
        'examples/llm/llm_inference',
        'examples/multimodal/visual_build',
    ]
    for artifact in expected_files:
        artifact_path = os.path.join(build_dir, artifact)

        result = run_command(cmd=['test', '-f', artifact_path],
                             remote_config=remote_config,
                             timeout=60,
                             logger=test_logger)
        if not result['success']:
            pytest.fail(f"Build artifact not found: {artifact_path}")


def test_unit_tests(env_config: EnvironmentConfig,
                    remote_config: Optional[RemoteConfig],
                    test_logger: logging.Logger):
    """Test unit tests execution - model independent"""
    execution_mode = "remote" if remote_config else "local"
    test_logger.info(f"Starting unit tests execution in {execution_mode} mode")

    build_dir = env_config.build_dir
    unit_test_cmd = ['bash', '-c', f'cd {build_dir} && ./unitTest']

    # Unified TensorRT library path handling for both local and remote
    env_vars = None
    if env_config.trt_package_dir:
        trt_lib_path = f"{env_config.trt_package_dir}/lib"
        env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib_path}"}

    result = run_command(cmd=unit_test_cmd,
                         remote_config=remote_config,
                         timeout=600,
                         logger=test_logger,
                         env_vars=env_vars)

    if not result['success']:
        pytest.fail(
            f"Unit tests failed: {result.get('error', 'Unknown error')}")
