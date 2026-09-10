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
"""ONNX-less engine build followed by one C++ runtime execution."""

import json
import os
import shutil
import sys
from typing import Dict, List

import pytest
from conftest import EnvironmentConfig
from pytest_helpers import run_with_trt_env, timer_context

from .config import (ModelType, TaskType, TestConfig,
                     infer_checkpoint_export_model_type)
from .utils.accuracy import check_accuracy_with_dataset
from .utils.command_execution import check_result_failures


def _engine_dir(config: TestConfig) -> str:
    return os.path.join(config.get_engine_base_dir(),
                        f"onnxless-{config.get_engine_id()}")


def _speculative_build_args(config: TestConfig) -> List[str]:
    if config.is_mtp:
        if config.model_name.lower().startswith("gemma-"):
            return [
                "--spec-type", "gemma4_mtp", "--draft-model-dir",
                config.get_gemma4_mtp_assistant_model_dir()
            ]
        return ["--spec-type", "mtp"]
    if config.is_dflash:
        return [
            "--spec-type", "dflash", "--draft-model-dir",
            config.get_dflash_draft_model_dir()
        ]
    if config.is_jetspec:
        return [
            "--spec-type", "jetspec", "--draft-model-dir",
            config.get_jetspec_draft_model_dir()
        ]
    if config.is_eagle:
        return [
            "--spec-type", "eagle3", "--draft-model-dir",
            config.get_eagle_draft_checkpoint_dir()
        ]
    return []


def _build_command(config: TestConfig, model_dir: str, engine_dir: str,
                   env_config: EnvironmentConfig) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "experimental.builder",
        "--model-dir",
        model_dir,
        "--engine-dir",
        engine_dir,
        "--components",
        "all",
        "--plugin-path",
        os.path.join(env_config.build_dir, "libNvInfer_edgellm_plugin.so"),
        "--max-input-len",
        str(config.max_input_len),
        "--max-kv-cache-capacity",
        str(config.max_kv_cache_capacity or config.max_seq_len),
        "--max-batch-size",
        str(config.max_batch_size),
        "--max-lora-rank",
        str(config.max_lora_rank or 0),
        "--max-verify-tree-size",
        str(config.max_verify_tree_size),
        "--max-draft-tree-size",
        str(config.max_draft_tree_size),
        "--min-image-tokens",
        str(config.min_image_tokens or 4),
        "--max-image-tokens",
        str(config.max_image_tokens or 1024),
        "--max-image-tokens-per-image",
        str(config.max_image_tokens_per_image or 512),
        "--min-time-steps",
        str(config.min_time_steps or 100),
        "--max-time-steps",
        str(config.max_time_steps or 6000),
    ]
    command.extend(_speculative_build_args(config))
    if config.fp8_embedding:
        command.append("--fp8-embedding")
    if config.debug:
        command.append("--verbose")
    return command


def _runtime_command(config: TestConfig, model_dir: str, engine_dir: str,
                     executables: Dict[str, str]) -> List[str]:
    common = [
        f"--inputFile={config.get_test_case_file()}",
        f"--outputFile={config.get_output_json_file()}",
        "--dumpProfile",
    ]
    if config.model_type == ModelType.TTS:
        return [
            executables["qwen3_tts_inference"],
            f"--talkerEngineDir={os.path.join(engine_dir, 'talker')}",
            f"--code2wavEngineDir={os.path.join(engine_dir, 'code2wav')}",
            f"--tokenizerDir={os.path.join(engine_dir, 'talker')}",
            f"--checkpointDir={model_dir}",
            f"--outputAudioDir={config.get_output_audio_dir()}",
            *common,
        ]
    if config.model_type == ModelType.VLA:
        return [
            executables["action_inference"],
            f"--engineDir={engine_dir}",
            f"--multimodalEngineDir={engine_dir}",
            f"--checkpointDir={model_dir}",
            *common,
        ]

    command = [
        executables["llm_inference"],
        f"--engineDir={engine_dir}",
        f"--checkpointDir={model_dir}",
        *common,
    ]
    if config.model_type in (ModelType.VLM, ModelType.ASR, ModelType.OMNI):
        command.append(f"--multimodalEngineDir={engine_dir}")
    if config.is_eagle or config.is_mtp or config.is_dflash or config.is_jetspec:
        command.extend([
            "--specDecode",
            f"--specDraftTopK={config.eagle_draft_top_k}",
            f"--specDraftStep={config.eagle_draft_step}",
            f"--specVerifySize={config.max_verify_tree_size}",
        ])
    talker_dir = os.path.join(engine_dir, "talker")
    if (config.model_type == ModelType.OMNI
            and os.path.isfile(os.path.join(talker_dir, "llm.engine"))):
        command.extend([
            "--enableAudioOutput",
            f"--talkerEngineDir={talker_dir}",
            f"--code2wavEngineDir={os.path.join(engine_dir, 'code2wav')}",
            f"--outputAudioDir={config.get_output_audio_dir()}",
        ])
    if config.batch_size is not None:
        command.append(f"--batchSize={config.batch_size}")
    if config.output_seq_len is not None:
        command.append(f"--maxGenerateLength={config.output_seq_len}")
    return command


def _assert_component_engines(model_dir: str, engine_dir: str,
                              config: TestConfig) -> None:
    from experimental.builder.core import contracts
    from experimental.builder.core.config import BundleConfig

    bundle = BundleConfig.from_pretrained(model_dir)
    speculative = bool(config.is_eagle or config.is_mtp or config.is_dflash
                       or config.is_jetspec)
    expected = []
    for component in bundle.components:
        spec = contracts.component_spec(component)
        if component == contracts.Component.LLM and speculative:
            expected.extend((
                spec.output_path(engine_dir, contracts.SpecRole.BASE),
                spec.output_path(engine_dir, contracts.SpecRole.DRAFT),
            ))
        else:
            expected.append(spec.output_path(engine_dir))
    missing = [path for path in expected if not os.path.isfile(path)]
    empty = [
        path for path in expected
        if os.path.isfile(path) and os.path.getsize(path) == 0
    ]
    if missing or empty:
        pytest.fail(f"component engine contract failed; missing={missing}, "
                    f"empty={empty}")


def _has_talker(engine_dir: str) -> bool:
    return os.path.isfile(os.path.join(engine_dir, "talker", "llm.engine"))


def _assert_multimodal_prompt_contract(engine_dir: str,
                                       responses: List[dict]) -> None:
    template_path = os.path.join(engine_dir, "processed_chat_template.json")
    media_types = {"image", "audio", "video"}
    for response in responses:
        present_types = {
            item.get("type")
            for message in response.get("messages", [])
            for item in message.get("content", [])
            if isinstance(item, dict) and item.get("type") in media_types
        }
        if not present_types:
            continue
        if not os.path.isfile(template_path):
            pytest.fail("multimodal runtime has no processed chat template")
        with open(template_path, encoding="utf-8") as stream:
            content_types = json.load(stream).get("content_types", {})
        formatted = response.get("formatted_complete_request", "")
        for media_type in present_types:
            placeholder = content_types.get(media_type, {}).get("format")
            if not placeholder:
                pytest.fail(f"multimodal chat template has no {media_type} "
                            "placeholder")
            if placeholder not in formatted:
                pytest.fail(f"formatted request dropped the {media_type} "
                            f"placeholder {placeholder!r}")


def _assert_runtime_contract(config: TestConfig, engine_dir: str,
                             output: dict) -> None:
    responses = output.get("responses") or []
    if not responses:
        pytest.fail("runtime produced no responses")
    _assert_multimodal_prompt_contract(engine_dir, responses)
    if config.model_type == ModelType.VLA:
        if not all(
                response.get("output_trajectory") for response in responses):
            pytest.fail("VLA runtime produced no action trajectory")
    if config.model_type == ModelType.TTS or _has_talker(engine_dir):
        audio_dir = config.get_output_audio_dir()
        audio_files = [] if not os.path.isdir(audio_dir) else [
            name for name in os.listdir(audio_dir) if name.endswith(".wav")
        ]
        if not audio_files:
            pytest.fail("TTS runtime produced no audio file")


def _run(command: List[str], name: str, timeout: int, env_config,
         test_logger) -> None:
    test_logger.info("Starting %s", name)
    result = run_with_trt_env(command, None, timeout, test_logger, env_config)
    if not result["success"]:
        pytest.fail(
            f"{name} failed: {result.get('error') or result['output']}")


def test_build_and_run(test_param: str, executable_files: Dict[str, str],
                       test_logger, env_config: EnvironmentConfig) -> None:
    """Build every checkpoint component once, then execute its runtime once."""
    model_type = infer_checkpoint_export_model_type(test_param)
    config = TestConfig.from_param_string(test_param, model_type,
                                          TaskType.CHECKPOINT_BUILD,
                                          env_config)
    config.check_trt_native_attn()
    model_dir = config.get_torch_model_dir()
    engine_dir = _engine_dir(config)
    shutil.rmtree(engine_dir, ignore_errors=True)
    if config.model_type in (ModelType.TTS, ModelType.OMNI):
        shutil.rmtree(config.get_output_audio_dir(), ignore_errors=True)
    os.makedirs(os.path.dirname(engine_dir), exist_ok=True)

    with timer_context(f"direct build and runtime for {config.model_name}",
                       test_logger):
        _run(_build_command(config, model_dir, engine_dir,
                            env_config), "single all-component engine build",
             7200, env_config, test_logger)
        _assert_component_engines(model_dir, engine_dir, config)

        _run(_runtime_command(config, model_dir, engine_dir, executable_files),
             "single end-to-end runtime execution", 6000, env_config,
             test_logger)

    output_file = config.get_output_json_file()
    with open(output_file, encoding="utf-8") as stream:
        output = json.load(stream)
    _assert_runtime_contract(config, engine_dir, output)

    reference = config.get_reference_json_file() or config.get_test_case_file()
    accuracy = check_accuracy_with_dataset(output_file, reference,
                                           config.test_case, test_logger)
    check_result_failures(accuracy)
