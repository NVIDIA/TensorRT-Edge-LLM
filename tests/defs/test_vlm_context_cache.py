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
"""VLM media-aware context cache pipeline test.

Exercises the full export -> build -> inference -> profile check pipeline
with multi-turn VLM requests that verify media-aware KV cache reuse:
same-image hit, different-image miss, position sensitivity, multi-image
ordering, and encoder embedding cache retention.
"""
import logging
from typing import Dict, Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import timer_context

from .config import ModelType, TaskType, TestConfig
from .utils.command_execution import (check_result_failures,
                                      execute_inference_test)


class TestVLMContextCache:

    def test_inference(self, test_param: str, executable_files: Dict[str, str],
                       remote_config: Optional[RemoteConfig],
                       test_logger: logging.Logger,
                       env_config: EnvironmentConfig) -> None:
        config = TestConfig.from_param_string(test_param, ModelType.VLM,
                                              TaskType.INFERENCE, env_config)
        config.check_trt_native_attn()

        with timer_context(f"VLM context cache for {config.model_name}",
                           test_logger):
            result = execute_inference_test(config, executable_files,
                                            remote_config, test_logger,
                                            env_config)
            if not result['success']:
                pytest.fail(f"Inference failed: {result['error']}")
            check_result_failures(result)
