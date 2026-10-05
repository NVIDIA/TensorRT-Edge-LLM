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

import logging

import pytest
from conftest import EnvironmentConfig
from defs import test_server_pipeline
from defs.config import ModelType, TaskType
from defs.test_server_pipeline import TestHLAPI as _TestHLAPI


def _environment(*, model_dir="/checkpoints", engine_dir="/engines"):
    return EnvironmentConfig(
        llm_sdk_dir=".",
        llm_models_dir=model_dir,
        edgellm_data_dir=None,
        onnx_dir=None,
        engine_dir=engine_dir,
        build_dir="build",
        test_log_dir="logs",
        trt_package_dir="/trt",
        vlmevalkit_dir=None,
        vlmevalkit_data_dir=None,
        vlmevalkit_work_dir=None,
    )


def test_hlapi_checkpoint_config_does_not_require_onnx():
    config = _TestHLAPI._checkpoint_config(
        "Qwen2.5-0.5B-Instruct-fp16-mxsl4096-mxbs4-mxil2048-llm_basic",
        ModelType.LLM,
        _environment(),
    )

    assert config.onnx_dir is None
    assert config.task_type == TaskType.INFERENCE


@pytest.mark.parametrize(("model_dir", "engine_dir", "missing_variable"), [
    (None, "/engines", "LLM_MODELS_DIR"),
    ("/checkpoints", None, "ENGINE_DIR"),
])
def test_hlapi_checkpoint_config_requires_direct_build_paths(
        model_dir, engine_dir, missing_variable):
    with pytest.raises(ValueError, match=missing_variable):
        _TestHLAPI._checkpoint_config(
            "Qwen2.5-0.5B-Instruct-fp16-mxsl4096-mxbs4-mxil2048-llm_basic",
            ModelType.LLM,
            _environment(model_dir=model_dir, engine_dir=engine_dir),
        )


def test_hlapi_guided_decoding_does_not_require_onnx(monkeypatch):
    monkeypatch.setattr(
        _TestHLAPI,
        "_llm_init_script",
        staticmethod(lambda *args: "llm = None"),
    )
    monkeypatch.setattr(
        test_server_pipeline,
        "run_command",
        lambda **kwargs: {
            "success": True,
            "output": "HLAPI_GENERATE_WITH_GUIDED_DECODING_PASSED",
        },
    )

    _TestHLAPI().test_hlapi_generate_with_guided_decoding(
        "Qwen2.5-0.5B-Instruct-fp16-mxsl4096-mxbs4-mxil2048-llm_basic",
        {},
        None,
        logging.getLogger(__name__),
        _environment(),
    )
