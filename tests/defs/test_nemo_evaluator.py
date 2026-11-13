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

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import run_command, timer_context

from .config import TaskType, TestConfig, infer_checkpoint_export_model_type


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


def test_nemo_eval(test_param: str, remote_config: Optional[RemoteConfig],
                   test_logger: logging.Logger, env_config: EnvironmentConfig,
                   capfd) -> None:
    """Run NeMo Evaluator through the checkpoint-direct server."""
    if remote_config is not None:
        pytest.skip("NeMo Evaluator CI smoke is local x86-only for now")

    inference = getattr(test_param, "inference", None)
    if not inference or inference.get("scenario") != "nemo_evaluator":
        pytest.fail("NeMo Evaluator requires a model-owned YAML case")
    case_id = str(test_param)
    model_type = infer_checkpoint_export_model_type(test_param)
    config = TestConfig.from_param_string(test_param, model_type,
                                          TaskType.CHECKPOINT_BUILD,
                                          env_config)
    model_dir = config.get_torch_model_dir()
    output_dir = os.path.join(env_config.test_log_dir, "nemo-results", case_id)
    cache_dir = os.path.join(env_config.engine_dir, "server-cache")
    evaluator_config = inference["evaluator"]
    score_config = inference["score"]
    eval_type = evaluator_config["eval_type"]

    cmd = [
        _default_server_python(env_config),
        "scripts/run_nemo_eval.py",
        "--model",
        model_dir,
        "--cache-dir",
        cache_dir,
        "--output-dir",
        output_dir,
        "--max-input-len",
        str(config.max_input_len),
        "--max-batch-size",
        str(config.max_batch_size),
        "--max-kv-cache-capacity",
        str(config.max_kv_cache_capacity),
        "--eval-type",
        eval_type,
    ]
    if "model_id" in evaluator_config:
        cmd.extend(["--model-id", evaluator_config["model_id"]])

    overrides = {
        f"config.params.{key}": value
        for key, value in evaluator_config.get("params", {}).items()
    }
    overrides.update(evaluator_config.get("overrides", {}))
    if overrides:
        cmd.extend([
            "--extra-overrides",
            ",".join(f"{key}={json.dumps(value)}"
                     for key, value in overrides.items()),
        ])
    cmd.extend([
        "--score-key",
        str(score_config["key"]),
        "--min-score",
        str(score_config["min"]),
    ])

    env_vars = {
        "PYTHONPATH":
        f"{env_config.llm_sdk_dir}:{os.environ.get('PYTHONPATH', '')}",
    }
    if env_config.trt_package_dir:
        trt_lib = os.path.join(env_config.trt_package_dir, "lib")
        env_vars["LD_LIBRARY_PATH"] = (
            f"{trt_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}")

    timeout = inference.get("timeout_seconds", 7200)
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
