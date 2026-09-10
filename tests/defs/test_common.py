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
"""Common test functions for TensorRT Edge-LLM"""

import glob
import logging
import os
from typing import Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import (record_library_size_report, run_command,
                            timer_context)

from .utils.device import DeviceConfig


def _get_trt_env_vars(env_config: EnvironmentConfig) -> Optional[dict]:
    """Get environment variables for TensorRT library path."""
    if env_config.trt_package_dir:
        trt_lib_path = f"{env_config.trt_package_dir}/lib"
        return {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib_path}"}
    return None


def test_build_project(env_config: EnvironmentConfig,
                       remote_config: Optional[RemoteConfig],
                       test_logger: logging.Logger):
    """Test project build - builds all components"""
    _build_project(env_config, remote_config, test_logger, build_pybind=False)


def test_build_project_with_pybind(env_config: EnvironmentConfig,
                                   remote_config: Optional[RemoteConfig],
                                   test_logger: logging.Logger):
    """Project build that also builds + import-checks the pybind runtime
    (the unit-job lists use this; pipeline lists use test_build_project)."""
    _build_project(env_config, remote_config, test_logger, build_pybind=True)


def _build_project(env_config: EnvironmentConfig,
                   remote_config: Optional[RemoteConfig],
                   test_logger: logging.Logger, build_pybind: bool):
    execution_mode = "remote" if remote_config else "local"
    device_config = DeviceConfig.auto_detect(remote_config, test_logger)
    test_logger.info(
        f"Building project for {device_config.target} in {execution_mode} mode"
    )
    build_dir = env_config.build_dir

    # Build cmake command with required components only
    cmake_cmd = [
        'cmake', '..', '-DBUILD_UNIT_TESTS=ON',
        '-DENABLE_CUTEDSL_MODULE_TEST_HOOK=ON'
    ]

    # Opt-in for jobs whose test lists import the pybind runtime (the
    # preprocessing suites); resolved from the pytest interpreter's pybind11.
    if build_pybind:
        import pybind11
        cmake_cmd.append('-DBUILD_PYTHON_BINDINGS=ON')
        cmake_cmd.append(f'-Dpybind11_DIR={pybind11.get_cmake_dir()}')
    else:
        # Explicit OFF: a build/ dir previously configured ON would keep
        # building pybind from the CMake cache.
        cmake_cmd.append('-DBUILD_PYTHON_BINDINGS=OFF')

    # Use trt_package_dir from env_config
    if env_config.trt_package_dir:
        cmake_cmd.append(f'-DTRT_PACKAGE_DIR={env_config.trt_package_dir}')
    cmake_cmd.append(f'-DCUDA_CTK_VERSION={device_config.cuda_version}')

    if device_config.target in [
            'jetson-orin', 'auto-thor', 'jetson-thor', 'gb10'
    ]:
        cmake_cmd.append(f'-DEMBEDDED_TARGET={device_config.target}')
        cmake_cmd.append(
            '-DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake')

    # Enable all available CuTe DSL kernels for aarch64 targets.
    if device_config.target in [
            'jetson-orin', 'auto-thor', 'jetson-thor', 'gb10'
    ]:
        cmake_cmd.append('-DENABLE_CUTE_DSL=ALL')
        test_logger.info("CuTe DSL: using available artifact")

    # Enable CuteDSL kernels for x86 Blackwell (SM120+) targets.
    if (device_config.target == 'x86'
            and device_config.compute_capability is not None
            and device_config.compute_capability >= 120):
        cmake_cmd.append('-DENABLE_CUTE_DSL=ALL')
        test_logger.info("CuTe DSL: x86 Blackwell, using available artifact")

    # Enable CuteDSL kernels on x86 when the job staged a prebuilt tarball for
    # the detected SM. The unified matrix producer downloads tarballs directly
    # into kernelSrcs/cuteDSLPrebuilt so CMake can auto-extract the matching
    # architecture, SM, and CUDA-major artifact. SM86 reuses the SM80 artifact
    # for forward-compatible groups. F16 MoE requires an exact artifact SM, so
    # do not enable that group without a native SM86 artifact.
    x86_cutedsl_selections = {
        80: ('sm_80', 'ALL'),
        86: ('sm_80', r'fmha\;gdn\;gemm\;int4_fp16_gemm\;ssd'),
        100: ('sm_100', 'ALL'),
        120: ('sm_120', 'ALL'),
    }
    x86_selection = x86_cutedsl_selections.get(
        device_config.compute_capability)
    if (device_config.target == 'x86' and x86_selection and glob.glob(
            os.path.join(env_config.llm_sdk_dir, 'kernelSrcs',
                         'cuteDSLPrebuilt',
                         f'cutedsl_x86_64_{x86_selection[0]}_*.tar.gz'))):
        x86_tag, x86_groups = x86_selection
        enable_cutedsl_arg = f'-DENABLE_CUTE_DSL={x86_groups}'
        if enable_cutedsl_arg not in cmake_cmd:
            cmake_cmd.append(enable_cutedsl_arg)
        cmake_cmd.append(f'-DCUTE_DSL_ARTIFACT_TAG={x86_tag}')
        x86_groups_log = x86_groups.replace(r'\;', ';')
        test_logger.info(
            f"CuTe DSL: x86 SM{device_config.compute_capability}, "
            f"using staged prebuilt artifact groups={x86_groups_log}")

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

    # Record how big the library this job just built is. Informational: a
    # failure here must not fail a build that otherwise succeeded.
    size_result = run_command(cmd=[
        'python3', 'scripts/report_library_size.py', '--build-dir', build_dir,
        '--label', device_config.target
    ],
                              remote_config=remote_config,
                              timeout=120,
                              logger=test_logger)
    if size_result['success']:
        record_library_size_report(size_result['output'])
    else:
        test_logger.warning("Library size report failed; continuing")

    expected_files = [
        'unittests/unitTestRuntime',
        'examples/llm/llm_build',
        'examples/llm/llm_inference',
        'examples/multimodal/visual_build',
        'examples/multimodal/audio_build',
    ]
    if build_pybind:
        # The preprocessing suites skip when this module is missing; gate with
        # a real import so a pybind link/ABI regression fails the build step
        # instead of silently skipping every parity test.
        import_check = ('import importlib; importlib.invalidate_caches(); '
                        'import _edgellm_runtime')
        result = run_command(cmd=[
            'bash', '-c', f'PYTHONPATH={build_dir}/pybind '
            f'python3 -c "{import_check}"'
        ],
                             remote_config=remote_config,
                             timeout=120,
                             logger=test_logger)
        if not result['success']:
            pytest.fail('pybind build requested but _edgellm_runtime is not '
                        f'importable from {build_dir}/pybind: '
                        f"{result.get('output', '')[-500:]}")
    # Executables that support --help smoke test
    help_check_files = [
        'examples/llm/llm_build',
        'examples/llm/llm_inference',
        'examples/multimodal/visual_build',
        'examples/multimodal/audio_build',
    ]

    env_vars = _get_trt_env_vars(env_config)

    for artifact in expected_files:
        artifact_path = os.path.join(build_dir, artifact)

        result = run_command(cmd=['test', '-f', artifact_path],
                             remote_config=remote_config,
                             timeout=60,
                             logger=test_logger)
        if not result['success']:
            pytest.fail(f"Build artifact not found: {artifact_path}")

        if artifact not in help_check_files:
            continue

        # Verify executable runs correctly with --help
        result = run_command(cmd=[artifact_path, '--help'],
                             remote_config=remote_config,
                             timeout=180,
                             logger=test_logger,
                             env_vars=env_vars)
        if not result['success']:
            output = result.get('output', '')
            test_logger.error(f"Executable --help output:\n{output}")
            pytest.fail(
                f"Executable --help failed (exit code {result['returncode']}): {artifact_path}"
            )


def test_unit_tests(env_config: EnvironmentConfig,
                    remote_config: Optional[RemoteConfig],
                    test_logger: logging.Logger):
    """Test unit tests execution - model independent"""
    execution_mode = "remote" if remote_config else "local"
    test_logger.info(f"Starting unit tests execution in {execution_mode} mode")

    build_dir = env_config.build_dir
    # The unit tests are several executables, registered with ctest by
    # unittests/CMakeLists.txt. Serial because this also runs against embedded
    # boards, where concurrent groups would contend for device memory.
    unit_test_cmd = [
        'bash', '-c', f'cd {build_dir} && ctest --output-on-failure'
    ]
    env_vars = _get_trt_env_vars(env_config)

    result = run_command(cmd=unit_test_cmd,
                         remote_config=remote_config,
                         timeout=600,
                         logger=test_logger,
                         env_vars=env_vars)

    if not result['success']:
        pytest.fail(
            f"Unit tests failed: {result.get('error', 'Unknown error')}")
