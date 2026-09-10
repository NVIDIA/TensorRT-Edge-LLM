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
"""NeMo Evaluator integration tests."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import run_command, timer_context

from scripts import run_nemo_eval

from .config import ModelType, TaskType, TestConfig

_DEFAULT_CONFIG = "tests/nemo_eval/cases.yml"


def _write_nemo_eval_log(output: str, case_id: str,
                         env_config: EnvironmentConfig) -> str:
    log_path = Path(env_config.test_log_dir) / f"nemo_eval_{case_id}.log"
    log_path.write_text(output +
                        ("\n" if output and not output.endswith("\n") else ""))
    return str(log_path)


def _nemo_eval_summary(output: str) -> str:
    selected_lines = []
    include_next_line = False
    include_metrics = False
    for line in output.splitlines():
        if line in ("Starting Edge-LLM server:", "Running NeMo Evaluator:"):
            selected_lines.append(line)
            include_next_line = True
            include_metrics = False
        elif include_next_line:
            selected_lines.append(line)
            include_next_line = False
        elif line == "NeMo Evaluator metrics:":
            selected_lines.append(line)
            include_metrics = True
        elif include_metrics and line.startswith("  "):
            selected_lines.append(line)
        else:
            include_metrics = False
            if line.startswith("NeMo Evaluator "):
                selected_lines.append(line)
            elif line.startswith("Could not find score key "):
                selected_lines.append(line)
            elif line.startswith("Score below threshold:"):
                selected_lines.append(line)

    return "\n".join(selected_lines)


def _default_server_python(env_config: EnvironmentConfig) -> str:
    server_python = os.path.join(env_config.llm_sdk_dir, "venv", "server",
                                 "bin", "python")
    if os.path.isfile(server_python):
        return server_python
    return sys.executable


def _resolve_repo_path(path: str, env_config: EnvironmentConfig) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(env_config.llm_sdk_dir, path)


def test_nemo_eval(test_param: str, remote_config: Optional[RemoteConfig],
                   test_logger: logging.Logger, env_config: EnvironmentConfig,
                   capfd) -> None:
    """Run NeMo Evaluator through the checkpoint-direct server."""
    if remote_config is not None:
        pytest.skip("NeMo Evaluator CI smoke is local x86-only for now")

    case_id = test_param
    case_config_path = _resolve_repo_path(_DEFAULT_CONFIG, env_config)
    try:
        case_config = run_nemo_eval._load_case_config(case_config_path,
                                                      case_id)
    except (RuntimeError, ValueError) as exc:
        pytest.fail(str(exc))
    model_param = case_config.get("model_param")
    if not isinstance(model_param, str) or not model_param:
        pytest.fail(f"NeMo Evaluator case {case_id} must set model_param")

    config = TestConfig.from_param_string(model_param, ModelType.LLM,
                                          TaskType.CHECKPOINT_BUILD,
                                          env_config)
    model_dir = config.get_torch_model_dir()
    output_dir = os.path.join(env_config.test_log_dir, "nemo-results", case_id)
    cache_dir = os.path.join(env_config.engine_dir, "server-cache")
    evaluator_config = case_config.get("evaluator", {})
    if not isinstance(evaluator_config, dict):
        pytest.fail(
            f"NeMo Evaluator case {case_id} evaluator must be a mapping")
    eval_type = evaluator_config.get("eval_type", "mmlu")

    cmd = [
        _default_server_python(env_config),
        "scripts/run_nemo_eval.py",
        "--config",
        case_config_path,
        "--case",
        case_id,
        "--model",
        model_dir,
        "--cache-dir",
        cache_dir,
        "--output-dir",
        output_dir,
    ]

    env_vars = {
        "PYTHONPATH":
        f"{env_config.llm_sdk_dir}:{os.environ.get('PYTHONPATH', '')}",
    }
    if env_config.trt_package_dir:
        trt_lib = os.path.join(env_config.trt_package_dir, "lib")
        env_vars["LD_LIBRARY_PATH"] = (
            f"{trt_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}")

    timeout = 7200
    with timer_context(f"NeMo Evaluator {eval_type} for {case_id}",
                       test_logger):
        result = run_command(cmd=cmd,
                             remote_config=None,
                             timeout=timeout,
                             logger=test_logger,
                             env_vars=env_vars)

    combined_output = result.get("combined_output", "")
    nemo_log = _write_nemo_eval_log(combined_output, case_id, env_config)
    summary = _nemo_eval_summary(combined_output)
    if summary:
        with capfd.disabled():
            print(f"\nNeMo Evaluator summary for {case_id}:")
            print(summary)
            print(f"NeMo Evaluator log: {nemo_log}")

    if not result["success"]:
        pytest.fail(
            f"NeMo Evaluator failed: {result.get('combined_output', '')}")
