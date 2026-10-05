# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""pi0.5 experimental-runtime build and host contract tests."""
import logging
from typing import Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import run_command, timer_context

from .test_common import _get_trt_env_vars

#: Built explicitly rather than through the default target: the other experimental
#: models have no CI coverage yet, and a failure there would block this job.
PI05_TARGETS = "pi05_policy_build pi05_policy_inference unitTestPi05"


def test_cpp_build_and_unit_tests(env_config: EnvironmentConfig,
                                  remote_config: Optional[RemoteConfig],
                                  test_logger: logging.Logger) -> None:
    """Build the pi0.5 runtime for this device and run its host contract tests.

    Reuses the build directory test_build_project already configured, so the
    toolchain, TensorRT package and embedded target come from the CMake cache.
    """
    build_dir = env_config.build_dir
    env_vars = _get_trt_env_vars(env_config)

    steps = [
        f"cd {build_dir}",
        "cmake .. -DBUILD_EXPERIMENTAL_MODELS=ON -DBUILD_UNIT_TESTS=ON",
        f"cmake --build . --parallel 16 --target {PI05_TARGETS}",
        "ctest -R '^unitTestPi05$' --output-on-failure",
    ]
    with timer_context("pi0.5 build and unit tests", test_logger):
        result = run_command(cmd=["bash", "-c", " && ".join(steps)],
                             remote_config=remote_config,
                             timeout=1800,
                             logger=test_logger,
                             env_vars=env_vars)

    if not result["success"]:
        pytest.fail(
            f"pi0.5 build or unit tests failed: {result.get('error', 'unknown')}"
        )
