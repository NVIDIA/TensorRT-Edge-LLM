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
"""Behavioral checks for the opt-in checkpoint-direct validation transport."""

import pytest
from conftest import EnvironmentConfig
from defs import config as test_config
from defs import test_builder_pipeline as pipeline


@pytest.fixture
def build_configuration(tmp_path):
    """Return a model configuration without requiring an installed checkpoint."""
    environment = EnvironmentConfig(llm_sdk_dir=".",
                                    llm_models_dir="/models",
                                    edgellm_data_dir="/data",
                                    onnx_dir="/onnx",
                                    engine_dir=str(tmp_path / "engines"),
                                    build_dir="/build",
                                    test_log_dir=str(tmp_path / "logs"),
                                    trt_package_dir="/trt")
    config = test_config.TestConfig.from_param_string(
        "Qwen3-0.6B-fp16-mxsl4096-mxbs1-mxil2048-llm_basic",
        test_config.ModelType.LLM,
        test_config.TaskType.CHECKPOINT_BUILD,
        environment,
        validate_environment=False)
    return config, environment


def test_weightless_command_preserves_baseline_arguments(build_configuration):
    """Weightless selection adds only its opt-in flag to the build command."""
    # Given a baseline build, when weightless validation is selected, then only its opt-in flag is added.
    config, environment = build_configuration
    arguments = (config, "/models/qwen3", "/engines/qwen3", environment)
    baseline = pipeline._build_command(*arguments)
    weightless = pipeline._build_command(*arguments, weightless_engine=True)
    option = "--weightless-engine"

    assert option not in baseline
    assert weightless.count(option) == 1
    assert [argument for argument in weightless
            if argument != option] == baseline


def test_weightless_engine_directory_is_separate(build_configuration):
    """Weightless artifacts cannot overwrite or remove baseline engines."""
    # Given baseline artifacts, when building a weightless variant, then its cleanup cannot remove baseline engines.
    config, _ = build_configuration
    baseline = pipeline._engine_dir(config)

    assert pipeline._engine_dir(config, True) == baseline + "-weightless"
    assert pipeline._engine_dir(config, False) == baseline


@pytest.mark.parametrize("enabled", [False, True])
def test_weightless_validation_requires_opt_in(monkeypatch, enabled):
    """Refit validation runs only when its native feature is opted in."""
    monkeypatch.delenv("ENABLE_EXPERIMENTAL_WEIGHTLESS", raising=False)
    if enabled:
        monkeypatch.setenv("ENABLE_EXPERIMENTAL_WEIGHTLESS", "ON")
    calls = []
    monkeypatch.setattr(pipeline, "_build_and_run",
                        lambda *args, **kwargs: calls.append(kwargs))

    if enabled:
        try:
            pipeline.test_build_and_run_weightless("case", {}, None, None)
        except pytest.skip.Exception as error:
            pytest.fail(
                f"Opted-in refit validation unexpectedly skipped: {error}")
        assert calls == [{"weightless_engine": True}]
    else:
        with pytest.raises(pytest.skip.Exception, match="qualified TensorRT"):
            pipeline.test_build_and_run_weightless("case", {}, None, None)
        assert not calls
