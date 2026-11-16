# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Test suite for OMNI (multimodal LLM + visual + audio) pipeline.

Covers engine_build, e2e_bench (accuracy datasets), inference (chat
smoke tests — llm_basic / vlm_basic / asr_basic), and multistage_inference
(the same chat cases on engines built by tensorrt_edgellm.builder, as text and
as speech).
"""
import json
import logging
import os
import shutil
import subprocess
from typing import Dict, List, Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import check_file_exists, run_with_trt_env, timer_context

from . import test_builder_pipeline
from .config import ModelType, TaskType, TestConfig
from .utils import command_execution, multistage_parity
from .utils.accuracy import check_accuracy_with_dataset
from .utils.command_execution import (check_result_failures,
                                      execute_build_test,
                                      execute_e2e_bench_test,
                                      execute_inference_test)
from .utils.command_generation import generate_inference_commands


def _stem(config: TestConfig, name: str) -> str:
    return os.path.join(config.test_log_dir,
                        f"{config.param_str}-multistage-{name}")


def _has_audio_input(config: TestConfig) -> bool:
    """Whether a request of the test case carries an audio clip."""
    with open(config.get_test_case_file(), encoding="utf-8") as stream:
        requests = json.load(stream)["requests"]
    return any(
        item.get("type") == "audio" for request in requests
        for message in request["messages"]
        if isinstance(message["content"], list) for item in message["content"])


def _direct_reference_command(config: TestConfig, executable_files: Dict[str,
                                                                         str],
                              test_logger) -> List[str]:
    """The llm_inference command whose mirrored flags multistage_inference reuses."""
    command_execution.prepare_spec_decode_input(config, test_logger)
    return generate_inference_commands(config, executable_files)[0][0]


def _sync_remote_dir(path: str, remote_config: Optional[RemoteConfig],
                     test_logger) -> None:
    """Copy a directory the board wrote back to the host, unless the host sees it."""
    if remote_config is None or os.path.isdir(path):
        return
    os.makedirs(path)
    env = os.environ.copy()
    env["SSHPASS"] = remote_config.password
    test_logger.info("Copying remote output directory back: %s", path)
    result = subprocess.run([
        "sshpass", "-e", "rsync", "-a", "-e",
        "ssh -o StrictHostKeyChecking=no",
        f"{remote_config.user}@{remote_config.host}:{path}/", f"{path}/"
    ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            env=env)
    if result.returncode != 0:
        pytest.fail(f"Failed to copy remote directory {path}: {result.stdout}")


def _run_multistage(config: TestConfig,
                    executable_files: Dict[str, str],
                    graph_dir: str,
                    reference_command: List[str],
                    name: str,
                    remote_config: Optional[RemoteConfig],
                    test_logger,
                    env_config: EnvironmentConfig,
                    audio_dir: Optional[str] = None,
                    whole_utterance: bool = False) -> str:
    """Run multistage_inference on the graph; returns its output JSON path.

    The output JSON and the audio directory, if any, are copied back from the
    board unless the host sees them already.
    """
    output_file = f"{_stem(config, name)}.json"
    # A stale output would stand in for one this run failed to write.
    if os.path.exists(output_file):
        os.remove(output_file)
    if audio_dir:
        shutil.rmtree(audio_dir, ignore_errors=True)
        if remote_config:
            _run_step(["rm", "-rf", audio_dir],
                      "stale audio directory removal", 60, remote_config,
                      test_logger, env_config)
    _run_step(
        multistage_parity.command(executable_files["multistage_inference"],
                                  reference_command,
                                  graph_dir,
                                  output_file,
                                  audio_dir,
                                  whole_utterance=whole_utterance),
        f"multistage_inference run ({name})", 6000, remote_config, test_logger,
        env_config)
    if audio_dir:
        _sync_remote_dir(audio_dir, remote_config, test_logger)
    return command_execution._sync_remote_output_file(output_file,
                                                      remote_config,
                                                      test_logger)


def _run_step(command: List[str], name: str, timeout: int,
              remote_config: Optional[RemoteConfig], test_logger,
              env_config: EnvironmentConfig) -> None:
    test_logger.info("Starting %s", name)
    result = run_with_trt_env(command, remote_config, timeout, test_logger,
                              env_config)
    if not result["success"]:
        pytest.fail(
            f"{name} failed: {result.get('error') or result['output']}")


class TestOmniPipeline:
    """Test suite for OMNI pipeline functionality."""

    def test_engine_build(self, test_param: str, executable_files: Dict[str,
                                                                        str],
                          remote_config: Optional[RemoteConfig],
                          test_logger: logging.Logger,
                          env_config: EnvironmentConfig) -> None:
        """Test TensorRT engine building for OMNI models (LLM + visual + audio)."""
        config = TestConfig.from_param_string(test_param, ModelType.OMNI,
                                              TaskType.BUILD, env_config)

        with timer_context(f"OMNI build for {config.model_name}", test_logger):
            result = execute_build_test(config, executable_files,
                                        remote_config, test_logger, env_config)
            if not result['success']:
                pytest.fail(f"Build failed: {result['error']}")

    def test_e2e_bench(self, test_param: str, executable_files: Dict[str, str],
                       remote_config: Optional[RemoteConfig],
                       test_logger: logging.Logger,
                       env_config: EnvironmentConfig) -> None:
        """Test end-to-end accuracy benchmarking for OMNI models."""
        config = TestConfig.from_param_string(test_param, ModelType.OMNI,
                                              TaskType.E2E_BENCH, env_config)

        with timer_context(f"OMNI e2e_bench for {config.model_name}",
                           test_logger):
            result = execute_e2e_bench_test(config, executable_files,
                                            remote_config, test_logger,
                                            env_config)
            if not result['success']:
                pytest.fail(f"e2e_bench failed: {result['error']}")
            check_result_failures(result)

    def test_inference(self, test_param: str, executable_files: Dict[str, str],
                       remote_config: Optional[RemoteConfig],
                       test_logger: logging.Logger,
                       env_config: EnvironmentConfig) -> None:
        """Test batch (chat) inference for OMNI models."""
        config = TestConfig.from_param_string(test_param, ModelType.OMNI,
                                              TaskType.INFERENCE, env_config)

        with timer_context(f"OMNI inference for {config.model_name}",
                           test_logger):
            result = execute_inference_test(config, executable_files,
                                            remote_config, test_logger,
                                            env_config)
            if not result['success']:
                pytest.fail(f"Inference failed: {result['error']}")
            check_result_failures(result)

    def test_inference_multistage(self, test_param: str,
                                  executable_files: Dict[str, str],
                                  remote_config: Optional[RemoteConfig],
                                  test_logger: logging.Logger,
                                  env_config: EnvironmentConfig) -> None:
        """Chat inference for OMNI models through multistage_inference.

        Runs on the engines test_builder_pipeline.py::test_build wrote, and
        scores the text against the dataset reference as test_inference does.
        A test case with an audio clip in its requests, on a bundle with a
        Talker, also speaks the replies:
        the requests run twice, vocoding each reply in one pass and in the
        default windows while the Talker runs. Every reply must carry audio, and
        the two runs must agree on the RVQ codes and the waveform's shape.
        """
        config = TestConfig.from_param_string(test_param,
                                              ModelType.OMNI,
                                              TaskType.INFERENCE,
                                              env_config,
                                              validate_environment=False)
        env_config.validate_for_checkpoint_runtime_tests()
        engine_dir = test_builder_pipeline._engine_dir_on_target(
            config, env_config, remote_config)
        speech = _has_audio_input(config) and check_file_exists(
            os.path.join(engine_dir, "talker", "llm.engine"), remote_config,
            test_logger)
        modes = (("whole", True), ("windowed",
                                   False)) if speech else (("text", False), )
        with timer_context(
                f"OMNI multistage_inference for {config.model_name}",
                test_logger):
            reference_command = _direct_reference_command(
                config, executable_files, test_logger)
            runs = {}
            output_files = {}
            for name, whole_utterance in modes:
                audio_dir = _stem(config, f"audio-{name}") if speech else None
                output_files[name] = _run_multistage(
                    config,
                    executable_files,
                    engine_dir,
                    reference_command,
                    name,
                    remote_config,
                    test_logger,
                    env_config,
                    audio_dir=audio_dir,
                    whole_utterance=whole_utterance)
                with open(output_files[name], encoding="utf-8") as stream:
                    runs[name] = (json.load(stream), audio_dir)

        if speech:
            mismatches = []
            for name, (output, _) in runs.items():
                for response in output["responses"]:
                    if not response.get("audio_samples"):
                        mismatches.append(
                            f"{name} run, request {response['request_idx']}: "
                            "no audio samples")
            mismatches += multistage_parity.windowed_mismatches(
                *runs["whole"], *runs["windowed"])
            if mismatches:
                pytest.fail("multistage_inference speech check failed:\n" +
                            "\n".join(mismatches))

        result = {'success': True, 'error': None, 'output': ''}
        result.update(
            check_accuracy_with_dataset(
                next(iter(output_files.values())),
                command_execution._accuracy_reference_file(config),
                config.test_case, test_logger))
        check_result_failures(result)
