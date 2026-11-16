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


@pytest.mark.parametrize(("spec_type", "draft_step", "verify_tokens"), [
    ("mtp", 3, 4),
    ("dflash", None, 16),
])
def test_hlapi_spec_geometry_uses_public_contract(monkeypatch, spec_type,
                                                  draft_step, verify_tokens):
    # Given iterative and block speculative configurations.
    # When CI generates public API invocations, then S is iterative-only and
    # P remains owned by the block deployment resolver.
    environment = _environment()
    config = _TestHLAPI._checkpoint_config(
        "Qwen2.5-0.5B-Instruct-fp16-mxsl4096-mxbs4-mxil2048-llm_basic",
        ModelType.LLM,
        environment,
    )
    config.eagle_draft_top_k = 1
    config.eagle_draft_step = draft_step or 1
    config.max_spec_verify_tokens = verify_tokens
    monkeypatch.setattr(
        _TestHLAPI,
        "_checkpoint_model_and_spec",
        staticmethod(lambda *args: ("/checkpoints/model", spec_type, "")),
    )

    command = _TestHLAPI._server_cli_args(config, environment, None,
                                          logging.getLogger(__name__))
    script = _TestHLAPI._llm_init_script(config, environment, None,
                                         logging.getLogger(__name__))

    assert "--spec-verify-tokens" in command
    assert ("--spec-draft-step" in command) == (draft_step is not None)
    assert "--spec-proposal-tokens" not in command
    assert f"draft_step={draft_step!r}" in script
    assert "proposal_tokens=None" in script
    assert f"verify_tokens={verify_tokens!r}" in script
