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
import logging
import os
import sys

import pytest
from conftest import EnvironmentConfig
from defs.config import ModelType, TaskType, TestConfig
from defs.model_test_config import load_model_tests
from defs.utils.command_execution import (_accuracy_reference_file,
                                          _prepare_alpamayo_action_chat)
from defs.utils.command_generation import (generate_build_commands,
                                           generate_inference_commands)

_CASE_PREFIX = "Alpamayo-R1-10B/fp16-tp2-onnx/"
_CUSTOM_DESCRIPTION = """## Summary

<!-- EDGE_LLM_L0_TESTS -->
```yaml
schema_version: 1
model_id: nvidia/Alpamayo-R1-10B
inputs: [text, image, trajectory]
outputs: [text, action]
cases:
  - name: fp16-tp2-onnx
    mode: onnx
    platform: l0_e2e_super_thor
    quantized_model_id_or_dir: nvidia/Alpamayo-R1-10B
    requirements:
      min_sm: 100
      min_gpu_memory_gb: 24
    engine:
      max_kv_cache_capacity: 4096
      max_input_len: 3424
      max_batch_size: 6
      min_image_tokens: 160
      max_image_tokens: 18432
      max_image_tokens_per_image: 192
      tp_size: 2
    inference:
      - name: action
        scenario: generate
        test_case: alpamayo_action_chat
        checks:
          accuracy:
            required: [minade_6s]
            max_regression: 0.05
```
"""

_COSMOS3_CUSTOM_DESCRIPTION = """## Summary

<!-- EDGE_LLM_L0_TESTS -->
```yaml
schema_version: 1
model_id: nvidia/Cosmos3-Edge-Policy-DROID
inputs: [text, image, trajectory]
outputs: [action]
cases:
  - name: fp16-onnx
    mode: onnx
    pipeline: cosmos3_policy
    platform: l0_e2e_a30
    quantized_model_id_or_dir: nvidia/Cosmos3-Edge-Policy-DROID
    requirements:
      min_sm: 80
      min_gpu_memory_gb: 16
    engine:
      max_batch_size: 2
    inference:
      - name: action_parity
        test_case: cosmos3_droid
```
"""
_COSMOS3_CASE_PREFIX = "Cosmos3-Edge-Policy-DROID/fp16-onnx/"
_FAKE_POLICY_BUILD = """
import argparse, pathlib
parser = argparse.ArgumentParser()
parser.add_argument("--onnx-dir")
parser.add_argument("--engine-dir", type=pathlib.Path)
parser.add_argument("--max-batch-size", default="1")
args = parser.parse_args()
for name in ("und_prefill/und_prefill.engine", "gen/gen.engine",
             "vae_encoder/vae_encoder.engine", "text_tokenizer/tokenizer.json",
             "embed_tokens.safetensors"):
    path = args.engine_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args.max_batch_size)
"""
# Emits actions equal to the batch size the loaded engine was built with.
_FAKE_POLICY_INFERENCE = """
import argparse, json, pathlib
parser = argparse.ArgumentParser()
for name in ("--image", "--prompt", "--view-point", "--seed"):
    parser.add_argument(name)
parser.add_argument("--engine-dir", type=pathlib.Path)
parser.add_argument("--steps", type=int)
parser.add_argument("--output", type=pathlib.Path)
args = parser.parse_args()
batch = float((args.engine_dir / "gen" / "gen.engine").read_text())
args.output.write_text(json.dumps({
    "action": [[batch] * 2] * 3, "shape": [1, 3, 2],
    "domain": "droid_lerobot", "finite": True,
    "num_inference_steps": args.steps}))
"""


def _environment(tmp_path):
    return EnvironmentConfig(llm_sdk_dir=".",
                             llm_models_dir="/models",
                             edgellm_data_dir="/data",
                             onnx_dir="/onnx",
                             engine_dir="/engine",
                             build_dir="/build",
                             test_log_dir=str(tmp_path),
                             trt_package_dir="/trt")


def _alpamayo_test(priority, phase):
    matches = [
        test for test in load_model_tests(priority, _CUSTOM_DESCRIPTION)
        if test.case_id == f"{_CASE_PREFIX}{phase}"
    ]
    assert len(matches) == 1
    return matches[0]


def _cosmos3_test(priority, phase, description=_COSMOS3_CUSTOM_DESCRIPTION):
    matches = [
        test for test in load_model_tests(priority, description)
        if test.case_id == f"{_COSMOS3_CASE_PREFIX}{phase}"
    ]
    assert len(matches) == 1
    return matches[0]


def _write_script(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n{source}")
    path.chmod(0o755)


def test_alpamayo_tp2_is_custom_only():
    """Keep Alpamayo TP2 out of default L0 selections."""
    for priority in ("l0_onnx_export_1", "l0_e2e_super_thor"):
        assert all(not test.case_id.startswith(_CASE_PREFIX)
                   for test in load_model_tests(priority))


def test_custom_alpamayo_tp2_routes_all_phases(tmp_path):
    """Route custom export, build, and action checks to their L0 jobs."""
    export = _alpamayo_test("l0_onnx_export_1", "export")
    build = _alpamayo_test("l0_e2e_super_thor", "build")
    inference = _alpamayo_test("l0_e2e_super_thor", "action")

    assert export.node.endswith("::test_llm_loader_tp_export")
    assert build.node.endswith("TestVLAPipeline::test_engine_build")
    assert inference.node.endswith("TestVLAPipeline::test_inference")
    assert inference.model_parameters["test_case"] == "alpamayo_action_chat"
    assert inference.model_parameters["metric_checks"]["accuracy"][
        "required"] == ["minade_6s"]

    config = TestConfig.from_param_string(export,
                                          ModelType.VLA,
                                          TaskType.EXPORT,
                                          _environment(tmp_path),
                                          validate_environment=False)
    assert config.tp_size == 2


def test_cosmos3_policy_case_runs_every_phase_in_its_own_job():
    """A pipeline case on a job without an ONNX lane exports there before build and inference."""
    phases = [
        (test.node.split("::")[-1], test.case_id)
        for test in load_model_tests("l0_e2e_a30", _COSMOS3_CUSTOM_DESCRIPTION)
        if test.case_id.startswith(_COSMOS3_CASE_PREFIX)
    ]

    assert phases == [
        ("test_checkpoint_export", f"{_COSMOS3_CASE_PREFIX}export"),
        ("test_engine_build", f"{_COSMOS3_CASE_PREFIX}build"),
        ("test_accuracy", f"{_COSMOS3_CASE_PREFIX}action_parity"),
    ]


def test_cosmos3_policy_rejects_engine_settings_it_does_not_build_with():
    """Engine settings the policy build ignores fail validation instead of being dropped."""
    description = _COSMOS3_CUSTOM_DESCRIPTION.replace(
        "      max_batch_size: 2\n",
        "      max_batch_size: 2\n      max_input_len: 2048\n")

    with pytest.raises(ValueError, match="max_input_len"):
        load_model_tests("l0_e2e_a30", description)


def test_cosmos3_policy_parity_uses_engines_built_with_case_batch_size(
        tmp_path, monkeypatch):
    """Action parity runs on the engines the case's build wrote with its max_batch_size."""
    from defs import test_cosmos3_policy

    env = EnvironmentConfig(llm_sdk_dir=".",
                            llm_models_dir=None,
                            edgellm_data_dir=str(tmp_path / "data"),
                            onnx_dir=None,
                            engine_dir=str(tmp_path / "engines"),
                            build_dir=str(tmp_path / "build"),
                            test_log_dir=str(tmp_path / "logs"),
                            trt_package_dir=None)
    _write_script(tmp_path / "build" / "bin" / "cosmos3_policy_build",
                  _FAKE_POLICY_BUILD)
    _write_script(tmp_path / "build" / "bin" / "cosmos3_policy_inference",
                  _FAKE_POLICY_INFERENCE)
    _write_script(tmp_path / "path" / "cmake", "")
    monkeypatch.setenv("PATH",
                       f"{tmp_path / 'path'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("CI_JOB_ID", raising=False)
    dataset = tmp_path / "data" / "updated_datasets" / "cosmos3_droid"
    dataset.mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    sample = {
        "name": "s0",
        "image": "s0.png",
        "prompt": "pick",
        "view_point": "left",
        "seed": 0
    }
    (dataset / "dataset.json").write_text(
        json.dumps({
            "model": "Cosmos3-Edge-Policy-DROID",
            "samples": [sample],
            "num_inference_steps": 4,
            "shape": [1, 3, 2],
        }))
    (dataset / "dataset_reference.json").write_text(
        json.dumps({
            "provenance": "test",
            "samples": {
                "s0": {
                    "actions": [[2.0] * 2] * 3
                }
            },
            "thresholds": {
                "cosine_floor": 0.99,
                "max_abs_ceiling": 0.01
            },
        }))
    logger = logging.getLogger(__name__)

    test_cosmos3_policy.test_engine_build(_cosmos3_test("l0_e2e_a30", "build"),
                                          env, None, logger)
    test_cosmos3_policy.test_accuracy(
        _cosmos3_test("l0_e2e_a30", "action_parity"), env, None, logger)


def test_alpamayo_tp2_commands_shard_only_llm(tmp_path):
    """Shard the LLM build and inference without sharding action or vision."""
    build = _alpamayo_test("l0_e2e_super_thor", "build")
    config = TestConfig.from_param_string(build,
                                          ModelType.VLA,
                                          TaskType.BUILD,
                                          _environment(tmp_path),
                                          validate_environment=False)
    assert config.max_batch_size == 6
    commands = generate_build_commands(
        config, {
            "llm_build": "llm_build",
            "visual_build": "visual_build",
            "action_build": "action_build",
        })

    assert len(commands) == 4
    assert "--tpRank=0" in commands[0][0]
    assert "--tpRank=1" in commands[1][0]
    assert all("--tpSize=2" in commands[index][0] for index in (0, 1))
    assert commands[2][0][0] == "visual_build"
    assert commands[3][0][0] == "action_build"

    inference = _alpamayo_test("l0_e2e_super_thor", "action")
    config = TestConfig.from_param_string(inference,
                                          ModelType.VLA,
                                          TaskType.INFERENCE,
                                          _environment(tmp_path),
                                          validate_environment=False)
    input_file = tmp_path / "input.json"
    input_file.write_text("{}", encoding="utf-8")
    config.input_file = str(input_file)
    commands = generate_inference_commands(
        config, {"action_inference": "action_inference"})

    assert len(commands) == 1
    assert "--tpSize=2" in commands[0][0]


def test_alpamayo_accuracy_uses_dataset_reference_after_preprocessing(
        tmp_path):
    """Score against the original dataset after preparing action inputs."""
    inference = _alpamayo_test("l0_e2e_super_thor", "action")
    config = TestConfig.from_param_string(inference,
                                          ModelType.VLA,
                                          TaskType.INFERENCE,
                                          _environment(tmp_path),
                                          validate_environment=False)
    source = (tmp_path / "updated_datasets" / "alpamayo_action_chat" /
              "input.json")
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    config.edgellm_data_dir = str(tmp_path)
    config._test_case_file_override = str(tmp_path /
                                          "preprocessed_test_case.json")

    assert _accuracy_reference_file(config) == str(source)


def test_alpamayo_action_chat_preprocessing_preserves_source(tmp_path):
    """Prepare batched action requests without editing the shared dataset."""
    inference = _alpamayo_test("l0_e2e_super_thor", "action")
    config = TestConfig.from_param_string(inference,
                                          ModelType.VLA,
                                          TaskType.INFERENCE,
                                          _environment(tmp_path),
                                          validate_environment=False)
    config.edgellm_data_dir = str(tmp_path)
    source = (tmp_path / "updated_datasets" / "alpamayo_action_chat" /
              "input.json")
    source.parent.mkdir(parents=True)
    data = {
        "batch_size":
        1,
        "default_system_prompt":
        "Drive safely",
        "requests": [{
            "id":
            "clip",
            "messages": [{
                "role":
                "user",
                "content": [{
                    "type":
                    "text",
                    "text":
                    "<|traj_history_start|><i3000><|traj_history_end|>Plan a route",
                }],
            }, {
                "role": "assistant",
                "content": "<|cot_start|>",
            }],
        } for _ in range(6)],
    }
    source.write_text(json.dumps(data), encoding="utf-8")
    (source.parent / "gt.json").write_text(json.dumps(
        {"clip": {
            "ego_history_xyz": [[0, 0, 0]] * 16
        }}),
                                           encoding="utf-8")
    substituted = tmp_path / "placeholder_substituted.json"
    substituted.write_text(json.dumps(data), encoding="utf-8")
    config._test_case_file_override = str(substituted)

    assert _prepare_alpamayo_action_chat(config, None)["success"]
    prepared = json.loads(
        (tmp_path / "preprocessed_test_case.json").read_text())
    assert prepared["batch_size"] == 6
    assert len(prepared["requests"]) == 6
    assert prepared["requests"][0]["messages"][0] == {
        "role": "system",
        "content": "Drive safely",
    }
    content = prepared["requests"][0]["messages"][1]["content"]
    assert content == [{
        "type": "trajectory",
        "trajectory": [[0, 0, 0]] * 16,
    }, {
        "type": "text",
        "text": "Plan a route",
    }]
    assert prepared["requests"][0]["messages"][-1] == data["requests"][0][
        "messages"][-1]
    assert prepared["add_generation_prompt"] is False
    assert json.loads(source.read_text()) == data


def _vla_config(param, task_type, tmp_path):
    config = TestConfig.from_param_string(param,
                                          ModelType.VLA,
                                          task_type,
                                          _environment(tmp_path),
                                          validate_environment=False)
    input_file = tmp_path / "input.json"
    input_file.write_text("{}", encoding="utf-8")
    config.input_file = str(input_file)
    return config


_NVFP4_TP2_ENGINE = ("Alpamayo-R1-10B-nvfp4-lmnvfp4-fp8kv-"
                     "mxsl4096-mxbs6-mxil3424-mnit160-mxit17408-mxpiit192")


def test_alpamayo_nvfp4_fp8kv_tp2_case_parses(tmp_path):
    """Accept NVFP4 weights, NVFP4 lm_head and FP8 KV cache on the TP2 action build."""
    config = _vla_config(f"{_NVFP4_TP2_ENGINE}-tp2", TaskType.BUILD, tmp_path)

    assert config.llm_precision == "nvfp4"
    assert config.lm_head_precision == "nvfp4"
    assert config.fp8_kv_cache is True
    assert config.tp_size == 2
    assert config.get_llm_onnx_dir().endswith("llm-nvfp4-nvfp4-fp8kv")


def test_alpamayo_action_artifacts_are_unique_per_tp_kv_config(tmp_path):
    """Action artifacts of one checkpoint never share a path across TP/KV layouts."""
    tail = "mxsl4096-mxbs6-mxil3424-mnit160-mxit17408-mxpiit192"
    params = [
        f"Alpamayo-R1-10B-fp16-{tail}",
        f"Alpamayo-R1-10B-fp16-{tail}-tp2",
        f"Alpamayo-R1-10B-fp16-fp8kv-{tail}-tp2",
        f"{_NVFP4_TP2_ENGINE}",
    ]
    configs = [_vla_config(p, TaskType.BUILD, tmp_path) for p in params]

    onnx_dirs = {c.get_action_onnx_dir() for c in configs}
    engine_dirs = {c.get_action_engine_dir() for c in configs}

    assert len(onnx_dirs) == len(configs)
    assert len(engine_dirs) == len(configs)
    for c in configs:
        assert c.get_action_engine_dir() == os.path.join(
            c.get_visual_engine_dir(), "action")
