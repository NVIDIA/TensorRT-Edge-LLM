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
"""GR00T N1.7 experimental-runtime build and host contract tests."""
import logging
from typing import Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import run_command, timer_context

from .test_common import _get_trt_env_vars

#: Built explicitly rather than through the default target: the other experimental
#: models have no CI coverage yet, and a failure there would block this job.
GR00T_TARGETS = "gr00t_policy_build gr00t_policy_inference unitTestGr00t"


def test_cpp_build_and_unit_tests(env_config: EnvironmentConfig,
                                  remote_config: Optional[RemoteConfig],
                                  test_logger: logging.Logger) -> None:
    """Build the GR00T runtime for this device and run its host contract tests, reusing the
    build directory test_build_project configured so the toolchain, TensorRT package and
    embedded target come from the CMake cache."""
    build_dir = env_config.build_dir
    env_vars = _get_trt_env_vars(env_config)

    steps = [
        f"cd {build_dir}",
        "cmake .. -DBUILD_EXPERIMENTAL_MODELS=ON -DBUILD_UNIT_TESTS=ON",
        f"cmake --build . --parallel 16 --target {GR00T_TARGETS}",
        "ctest -R '^unitTestGr00t$' --output-on-failure",
    ]
    with timer_context("GR00T N1.7 build and unit tests", test_logger):
        result = run_command(cmd=["bash", "-c", " && ".join(steps)],
                             remote_config=remote_config,
                             timeout=1800,
                             logger=test_logger,
                             env_vars=env_vars)

    if not result["success"]:
        pytest.fail("GR00T N1.7 build or unit tests failed: "
                    f"{result.get('error', 'unknown')}")
