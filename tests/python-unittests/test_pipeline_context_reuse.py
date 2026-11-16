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

import json

import pytest
from conftest import EnvironmentConfig
from defs.config import ModelType, TaskType, TestConfig
from defs.model_test_config import _parse_model_document
from defs.utils.command_execution import (
    _check_context_reuse_cold_hit_equivalence,
    _check_spec_prefill_evict_equivalence, _read_context_reuse_profile)
from defs.utils.command_generation import (generate_build_commands,
                                           generate_inference_commands)

_BUILD_PARAM = ("Qwen3.5-4B-nvfp4-mxsl1024-mxbs1-mxil1024-mxkvp64")
_INFERENCE_PARAM = f"{_BUILD_PARAM}-ctxreuse-ccms4-llm_context_reuse"
_GEMMA4_MTP_BUILD_PARAM = (
    "gemma-4-E2B-it-fp16-mxsl2048-mtp-mxbs2-mxil1024-mxkvp64")
_KV_PAGE_SIZE = 128


def _environment(tmp_path):
    return EnvironmentConfig(llm_sdk_dir=".",
                             llm_models_dir="/models",
                             edgellm_data_dir="/data",
                             onnx_dir="/onnx",
                             engine_dir="/engine",
                             build_dir="/build",
                             test_log_dir=str(tmp_path),
                             trt_package_dir="/trt")


def _config(param, task_type, tmp_path):
    return TestConfig.from_param_string(param,
                                        ModelType.LLM,
                                        task_type,
                                        _environment(tmp_path),
                                        validate_environment=False)


def test_context_reuse_build_and_inference_commands_share_engine(tmp_path):
    build_config = _config(_BUILD_PARAM, TaskType.BUILD, tmp_path)
    inference_config = _config(_INFERENCE_PARAM, TaskType.INFERENCE, tmp_path)

    build_command = generate_build_commands(build_config,
                                            {"llm_build": "llm_build"})[0][0]
    inference_command = generate_inference_commands(
        inference_config, {"llm_inference": "llm_inference"})[0][0]

    assert build_config.get_llm_engine_dir(
    ) == inference_config.get_llm_engine_dir()
    assert "--maxKVPoolPages=64" in build_command
    assert "--enableContextReuse" in inference_command
    assert "--contextCacheMaxSnapshots=4" in inference_command
    assert any(
        option.startswith("--profileOutputFile=")
        for option in inference_command)


def test_model_manifest_keeps_context_reuse_out_of_build_parameters():
    document = {
        "schema_version":
        1,
        "model_id":
        "google/gemma-4-E2B-it",
        "inputs": ["text"],
        "outputs": ["text"],
        "cases": [{
            "name":
            "context-reuse",
            "mode":
            "onnx",
            "platform":
            "l0_e2e_drive_thor",
            "quantized_model_id_or_dir":
            "/models/gemma-4-E2B-it",
            "requirements": {
                "min_sm": 100,
                "min_gpu_memory_gb": 11,
            },
            "engine": {
                "max_kv_cache_capacity": 4096,
                "max_input_len": 1536,
                "max_batch_size": 1,
                "context_reuse": True,
                "context_cache_max_snapshots": 4,
            },
            "inference": [{
                "name": "context-reuse",
                "scenario": "generate",
                "input_file": "tests/test_cases/llm_context_reuse.json",
                "checks": {
                    "accuracy": {
                        "required": ["rouge1"],
                        "max_regression": 0.3,
                    }
                },
            }],
        }],
    }

    tests = _parse_model_document(document, "test")
    build = next(test for test in tests if test.case_id.endswith("/build"))
    inference = next(test for test in tests
                     if test.case_id.endswith("/context-reuse"))

    for name in ("context_reuse", "context_cache_max_snapshots"):
        assert name not in build.model_parameters
        assert name in inference.model_parameters


def test_spec_build_commands_apply_kv_pool_pages_to_base_and_draft(tmp_path):
    config = _config(_GEMMA4_MTP_BUILD_PARAM, TaskType.BUILD, tmp_path)

    commands = generate_build_commands(config, {"llm_build": "llm_build"})

    assert len(commands) == 2
    assert all("--maxKVPoolPages=64" in command for command, _ in commands)


def test_context_reuse_max_snapshots_require_enable_token(tmp_path):
    invalid_param = f"{_BUILD_PARAM}-ccms4-llm_context_reuse"
    with pytest.raises(ValueError,
                       match="context_cache_max_snapshots requires ctxreuse"):
        _config(invalid_param, TaskType.INFERENCE, tmp_path)


def test_context_reuse_profile_requires_positive_reused_tokens(tmp_path):
    config = _config(_INFERENCE_PARAM, TaskType.INFERENCE, tmp_path)
    with open(config.get_profile_json_file(), "w", encoding="utf-8") as f:
        json.dump({"prefill": {"reused_tokens": 0}}, f)

    with pytest.raises(RuntimeError, match="positive prefill.reused_tokens"):
        _read_context_reuse_profile(config, logger=None)


def test_context_reuse_fixture_has_strict_prefix_seed():
    with open("tests/test_cases/llm_context_reuse.json",
              encoding="utf-8") as f:
        requests = json.load(f)["requests"]

    cold_prompt = requests[0]["messages"][0]["content"]
    seed_prompt = requests[1]["messages"][0]["content"]
    reused_prompt = requests[2]["messages"][0]["content"]
    assert cold_prompt == reused_prompt
    assert cold_prompt.startswith(seed_prompt)
    assert len(seed_prompt) < len(cold_prompt)
    # Qwen emits at least one token per whitespace-delimited field. EAGLE
    # replays the final matched page, so keep at least two pages.
    assert len(seed_prompt.split()) >= 2 * _KV_PAGE_SIZE


def test_context_reuse_cold_and_hit_outputs_are_token_identical(tmp_path):
    config = _config(_INFERENCE_PARAM, TaskType.INFERENCE, tmp_path)
    response = {
        "output_text": "NVIDIA",
        "finish_reason": "length",
        "logprobs": [[{
            "token_id": 123
        }]],
    }
    with open(config.get_output_json_file(), "w", encoding="utf-8") as f:
        json.dump({"responses": [response, {}, response]}, f)

    _check_context_reuse_cold_hit_equivalence(config)


def test_spec_prefill_evict_fixture_and_survivor_equivalence(tmp_path):
    param = f"{_BUILD_PARAM}-ctxreuse-llm_spec_prefill_evict"
    config = _config(param, TaskType.INFERENCE, tmp_path)
    with open(config.get_test_case_file(), encoding="utf-8") as f:
        requests = json.load(f)["requests"]
    assert len(requests) == 4
    assert requests[0]["stop"] == "OK"
    assert requests[1]["messages"] == requests[2]["messages"]
    assert requests[2]["context_cache_lookup_policy"] == "bypass"

    stopped = {
        "output_text": "",
        "finish_reason": "stop-words",
        "logprobs": [[{
            "token_id": 1
        }]],
    }
    survivor = {
        "output_text": "stable",
        "finish_reason": "max-length",
        "logprobs": [[{
            "token_id": 2
        }], [{
            "token_id": 3
        }]],
    }
    with open(config.get_output_json_file(), "w", encoding="utf-8") as f:
        json.dump({"responses": [stopped, survivor, survivor, {}]}, f)
    profile = {
        "prefill": {
            "reused_tokens": 0
        },
        "context_cache": {
            "lookup_bypass_sequences": 1,
            "publications": {
                "attempts": 1,
                "committed": 1,
                "existing": 0,
            },
        },
    }
    with open(config.get_profile_json_file(), "w", encoding="utf-8") as f:
        json.dump(profile, f)

    _check_spec_prefill_evict_equivalence(config)
    assert _read_context_reuse_profile(config, logger=None) == 0

    profile["context_cache"]["publications"]["committed"] = 0
    with open(config.get_profile_json_file(), "w", encoding="utf-8") as f:
        json.dump(profile, f)
    with pytest.raises(RuntimeError,
                       match="at least one committed publication"):
        _read_context_reuse_profile(config, logger=None)


@pytest.mark.parametrize("route", ["inference", "bench", "direct"])
@pytest.mark.parametrize("mode", ["dspark", "dflash", "jetspec"])
def test_block_commands_leave_proposal_depth_to_checkpoint(
        tmp_path, monkeypatch, route, mode):
    # Given a block draft whose one backbone call produces multiple proposals,
    # When constructing any runtime command, Then preserve checkpoint-derived P.
    from defs import test_builder_pipeline
    from defs.utils import command_generation

    config = _config(
        "Qwen3-4B-Instruct-2507-nvfp4-mxsl2048-dspark-block7-fp16-"
        "mxbs1-mxil1024-mvts8-mdts7-llm_basic", TaskType.INFERENCE, tmp_path)
    config.is_dspark = mode == "dspark"
    config.is_dflash = mode == "dflash"
    config.is_jetspec = mode == "jetspec"
    if route == "direct":
        monkeypatch.setattr(config, f"get_{mode}_draft_model_dir",
                            lambda: "/models/draft")
        command = test_builder_pipeline._runtime_command(
            config, "/models/target", "/engine",
            {"llm_inference": "llm_inference"},
            "tests/test_cases/llm_basic.json")
    else:
        generator = (command_generation.generate_inference_commands
                     if route == "inference" else
                     command_generation.generate_e2e_bench_commands)
        command = generator(config, {"llm_inference": "llm_inference"})[0][0]
    assert not any(arg.startswith("--specProposalTokens=") for arg in command)
