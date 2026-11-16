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
"""End-to-end coverage for Flax checkpoint conversion."""

import json
import os
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import run_command, timer_context

from experimental.checkpoint_converters.models.gemma4 import \
    schema as gemma4_schema

from .config import ModelType, TaskType, TestConfig
from .utils.checkpoint_export_helpers import run_checkpoint_export
from .utils.command_execution import (check_result_failures,
                                      execute_build_test,
                                      execute_inference_test)

_GEMMA4_E2B_VARIANT = "gemma-4-E2B-it"
_GEMMA4_E2B_EXPORT_PARAM = "gemma-4-E2B-it-fp16"
_GEMMA4_E2B_BUILD_PARAM = ("gemma-4-E2B-it-fp16-mxsl4096-mxbs1-mxil1536")
_GEMMA4_E2B_INFERENCE_PARAM = _GEMMA4_E2B_BUILD_PARAM + \
    "-llm_sliding_window"
_GEMMA4_26B_VARIANT = "gemma-4-26B-A4B-it"
_GEMMA4_26B_EXPORT_PARAM = "gemma-4-26B-A4B-it-fp16"
_GEMMA4_26B_BUILD_PARAM = \
    "gemma-4-26B-A4B-it-fp16-mxsl4096-mxbs1-mxil1536"
_GEMMA4_26B_INFERENCE_PARAM = _GEMMA4_26B_BUILD_PARAM + \
    "-llm_sliding_window"


def _output_texts(path: str) -> List[str]:
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    responses = payload.get("responses")
    if not isinstance(responses, list) or not responses:
        raise AssertionError(f"Inference output has no responses: {path}")
    texts = [response.get("output_text") for response in responses]
    if any(not isinstance(text, str) for text in texts):
        raise AssertionError(
            f"Inference output has invalid output_text: {path}")
    return texts


def _assert_output_text_parity(hf_output: str, converted_output: str) -> None:
    hf_texts = _output_texts(hf_output)
    converted_texts = _output_texts(converted_output)
    assert converted_texts == hf_texts, (
        "Converted Orbax checkpoint output differs from the HF checkpoint: "
        f"HF={hf_texts!r}, converted={converted_texts!r}")


def _convert_and_export(checkpoint_env: str, variant: str, export_param: str,
                        test_logger, env_config: EnvironmentConfig) -> None:
    checkpoint_dir = os.environ.get(checkpoint_env)
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        pytest.fail(f"{checkpoint_env} must name the cached public Gemma4 "
                    "Orbax checkpoint")
    test_cache_dir = os.environ.get("TEST_CACHE_DIR")
    if not test_cache_dir:
        pytest.fail("TEST_CACHE_DIR is required for Gemma4 Orbax conversion")

    config = TestConfig.from_param_string(export_param, ModelType.LLM,
                                          TaskType.EXPORT, env_config)
    tokenizer_checkpoint_dir = Path(config.get_torch_model_dir())
    with tempfile.TemporaryDirectory(prefix="gemma4_orbax_conversion_",
                                     dir=test_cache_dir) as scratch:
        tokenizer_dir = Path(scratch) / "tokenizer"
        tokenizer_dir.mkdir()
        for name in gemma4_schema.TOKENIZER_FILES:
            shutil.copy2(tokenizer_checkpoint_dir / name, tokenizer_dir / name)
        converted_checkpoint = Path(scratch) / "hf-checkpoint"
        command = [
            sys.executable,
            "-m",
            "experimental.checkpoint_converters.cli",
            "--model-family",
            "gemma4",
            "--model-variant",
            variant,
            "--checkpoint-path",
            checkpoint_dir,
            "--tokenizer-path",
            str(tokenizer_dir),
            "--output-path",
            str(converted_checkpoint),
        ]
        with timer_context("Gemma4 Orbax to HF checkpoint conversion",
                           test_logger):
            result = run_command(command,
                                 remote_config=None,
                                 timeout=7200,
                                 logger=test_logger)
        if not result["success"]:
            pytest.fail("Gemma4 Orbax checkpoint conversion failed: " +
                        (result.get("error") or result["output"]))
        run_checkpoint_export(config,
                              test_logger,
                              model_dir=str(converted_checkpoint),
                              timeout=7200,
                              components=("thinker", ))

    onnx_path = os.path.join(config.get_llm_onnx_dir(), "model.onnx")
    if not os.path.isfile(onnx_path):
        pytest.fail(f"Converted Gemma4 export did not produce {onnx_path}")


def _export_hf_reference(onnx_env: str, export_param: str, test_logger,
                         env_config: EnvironmentConfig) -> None:
    onnx_dir = os.environ.get(onnx_env)
    if not onnx_dir:
        pytest.fail(f"{onnx_env} is required for the Gemma4 HF reference")
    reference_env = replace(env_config, onnx_dir=onnx_dir)
    config = TestConfig.from_param_string(export_param, ModelType.LLM,
                                          TaskType.EXPORT, reference_env)
    run_checkpoint_export(config,
                          test_logger,
                          timeout=7200,
                          components=("thinker", ))
    onnx_path = os.path.join(config.get_llm_onnx_dir(), "model.onnx")
    if not os.path.isfile(onnx_path):
        pytest.fail(f"Canonical Gemma4 export did not produce {onnx_path}")


def test_gemma4_orbax_checkpoint_export(test_logger,
                                        env_config: EnvironmentConfig) -> None:
    """Convert the public Gemma4 E2B Orbax checkpoint and export ONNX."""
    # Given the public E2B Orbax checkpoint and pinned HF tokenizer, when
    # conversion and export run, then ONNX is published.
    _convert_and_export("GEMMA4_ORBAX_CHECKPOINT_DIR", _GEMMA4_E2B_VARIANT,
                        _GEMMA4_E2B_EXPORT_PARAM, test_logger, env_config)


def test_gemma4_26b_orbax_checkpoint_export(
        test_logger, env_config: EnvironmentConfig) -> None:
    """Export canonical HF and converted public Gemma4 26B checkpoints."""
    # Given the paired HF and public 26B Orbax checkpoints, when both export,
    # then the canonical and converted ONNX artifacts are published.
    _export_hf_reference("GEMMA4_26B_HF_ONNX_DIR", _GEMMA4_26B_EXPORT_PARAM,
                         test_logger, env_config)
    _convert_and_export("GEMMA4_26B_ORBAX_CHECKPOINT_DIR", _GEMMA4_26B_VARIANT,
                        _GEMMA4_26B_EXPORT_PARAM, test_logger, env_config)


def _build_run_and_compare(
    onnx_env: str,
    engine_directory: str,
    build_param: str,
    inference_param: str,
    executable_files: Dict[str, str],
    remote_config: Optional[RemoteConfig],
    test_logger,
    env_config: EnvironmentConfig,
    monkeypatch: pytest.MonkeyPatch,
    check_quality: bool = True,
) -> None:
    converted_onnx_dir = os.environ.get(onnx_env)
    if not converted_onnx_dir or not os.path.isdir(converted_onnx_dir):
        pytest.fail(f"{onnx_env} must name the converted Gemma4 ONNX artifact")
    if not env_config.engine_dir:
        pytest.fail("ENGINE_DIR is required for converted Gemma4 E2E")

    hf_config = TestConfig.from_param_string(inference_param, ModelType.LLM,
                                             TaskType.INFERENCE, env_config)
    hf_output = hf_config.get_output_json_file()
    if not os.path.isfile(hf_output):
        pytest.fail("The paired Gemma4 HF inference must run before the "
                    f"Orbax comparison; missing {hf_output}")

    converted_env = replace(
        env_config,
        onnx_dir=converted_onnx_dir,
        engine_dir=os.path.join(env_config.engine_dir, engine_directory),
        test_log_dir=os.path.join(env_config.test_log_dir, engine_directory),
    )
    os.makedirs(converted_env.engine_dir, exist_ok=True)
    os.makedirs(converted_env.test_log_dir, exist_ok=True)
    monkeypatch.setenv("EDGE_LLM_ENGINE_CACHE_MODE", "off")

    build_config = TestConfig.from_param_string(build_param, ModelType.LLM,
                                                TaskType.BUILD, converted_env)
    with timer_context("Converted Gemma4 engine build", test_logger):
        build_result = execute_build_test(build_config, executable_files,
                                          remote_config, test_logger,
                                          converted_env)
    if not build_result["success"]:
        pytest.fail("Converted Gemma4 engine build failed: " +
                    (build_result.get("error") or build_result["output"]))

    inference_config = TestConfig.from_param_string(inference_param,
                                                    ModelType.LLM,
                                                    TaskType.INFERENCE,
                                                    converted_env)
    with timer_context("Converted Gemma4 inference", test_logger):
        inference_result = execute_inference_test(inference_config,
                                                  executable_files,
                                                  remote_config, test_logger,
                                                  converted_env)
    if not inference_result["success"]:
        pytest.fail("Converted Gemma4 inference failed: " + (
            inference_result.get("error") or inference_result["output"]))
    if check_quality:
        check_result_failures(inference_result)
    _assert_output_text_parity(hf_output,
                               inference_config.get_output_json_file())


def _build_and_run_hf_reference(
    build_param: str,
    inference_param: str,
    executable_files: Dict[str, str],
    remote_config: Optional[RemoteConfig],
    test_logger,
    env_config: EnvironmentConfig,
) -> None:
    build_config = TestConfig.from_param_string(build_param, ModelType.LLM,
                                                TaskType.BUILD, env_config)
    with timer_context("Canonical Gemma4 engine build", test_logger):
        build_result = execute_build_test(build_config, executable_files,
                                          remote_config, test_logger,
                                          env_config)
    if not build_result["success"]:
        pytest.fail("Canonical Gemma4 engine build failed: " +
                    (build_result.get("error") or build_result["output"]))

    inference_config = TestConfig.from_param_string(inference_param,
                                                    ModelType.LLM,
                                                    TaskType.INFERENCE,
                                                    env_config)
    with timer_context("Canonical Gemma4 inference", test_logger):
        inference_result = execute_inference_test(inference_config,
                                                  executable_files,
                                                  remote_config, test_logger,
                                                  env_config)
    if not inference_result["success"]:
        pytest.fail("Canonical Gemma4 inference failed: " + (
            inference_result.get("error") or inference_result["output"]))


def test_gemma4_orbax_matches_hf(executable_files: Dict[str, str],
                                 remote_config: Optional[RemoteConfig],
                                 test_logger, env_config: EnvironmentConfig,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """Build and run converted Gemma4 E2B, then require HF output parity."""
    # Given the E2B HF run and converted ONNX, when the converted engine runs,
    # then both checkpoints generate the same text.
    _build_run_and_compare("GEMMA4_ORBAX_ONNX_DIR", "gemma4-orbax",
                           _GEMMA4_E2B_BUILD_PARAM,
                           _GEMMA4_E2B_INFERENCE_PARAM, executable_files,
                           remote_config, test_logger, env_config, monkeypatch)


def test_gemma4_26b_orbax_matches_hf(executable_files: Dict[str, str],
                                     remote_config: Optional[RemoteConfig],
                                     test_logger,
                                     env_config: EnvironmentConfig,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """Build and run converted Gemma4 26B, then require HF output parity."""
    # Given the 26B HF run and converted ONNX, when the converted engine runs,
    # then both checkpoints generate the same text.
    _build_and_run_hf_reference(_GEMMA4_26B_BUILD_PARAM,
                                _GEMMA4_26B_INFERENCE_PARAM, executable_files,
                                remote_config, test_logger, env_config)
    _build_run_and_compare("GEMMA4_26B_ORBAX_ONNX_DIR",
                           "gemma4-26b-orbax",
                           _GEMMA4_26B_BUILD_PARAM,
                           _GEMMA4_26B_INFERENCE_PARAM,
                           executable_files,
                           remote_config,
                           test_logger,
                           env_config,
                           monkeypatch,
                           check_quality=False)
