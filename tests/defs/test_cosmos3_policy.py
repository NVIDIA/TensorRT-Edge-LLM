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
"""Cosmos3-Edge-Policy-DROID export, runtime and action-parity tests."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest_helpers

from examples.accuracy.scripts import calculate_action_score

from .config import ModelType, TaskType, TestConfig

MODEL = "Cosmos3-Edge-Policy-DROID"
PROFILE = f"{MODEL}-fp16"
COMPONENTS = ("und_prefill", "gen", "vae_encoder")
_ENGINE = rf"{PROFILE}-mxbs[1-9][0-9]*"


def _binary(env, name):
    return Path(env.build_dir) / "bin" / name


def _param(test_param, *, engine=False, inference=False):
    """Render a model-config case as ``<profile>[-mxbs<N>][-<test_case>]``."""
    parameters = getattr(test_param, "model_parameters", None)
    if parameters is None:
        return test_param
    param = f"{parameters['model_name']}-{parameters['llm_precision']}"
    if engine:
        param += f"-mxbs{parameters.get('max_batch_size', 1)}"
    if inference:
        param += f"-{parameters['test_case']}"
    return param


def _onnx_dir(env):
    # No ONNX_DIR means no export lane: export into this job's engine scratch.
    root = env.onnx_dir or (env.engine_dir and Path(env.engine_dir) / "onnx")
    assert root, "ONNX_DIR or ENGINE_DIR is required"
    return Path(root) / PROFILE


def _runtime(test_param):
    match = re.fullmatch(f"({_ENGINE})-(.+)",
                         _param(test_param, engine=True, inference=True))
    assert match, test_param
    return match.groups()


def _build_binaries(env, remote, logger):
    build = Path(env.build_dir)
    pytest_helpers.run_policy_command(
        ["cmake", build, "-DBUILD_EXPERIMENTAL_MODELS=ON"], env, remote,
        logger)
    pytest_helpers.run_policy_command([
        "cmake", "--build", build, "--parallel", "16", "--target",
        "cosmos3_policy_build", "cosmos3_policy_inference",
        "NvInfer_edgellm_plugin"
    ], env, remote, logger)


def _artifacts(suffix, embedding):
    return [
        f"{component}/{suffix.format(component=component)}"
        for component in COMPONENTS
    ] + ["text_tokenizer/tokenizer.json", embedding]


def test_checkpoint_export(test_param, env_config, test_logger):
    """Export writes non-empty ONNX and a config.json for every Cosmos3 policy component."""
    assert _param(test_param) == PROFILE, test_param
    if isinstance(test_param, str):
        assert env_config.llm_models_dir, "LLM_MODELS_DIR is required"
        checkpoint = Path(env_config.llm_models_dir) / MODEL
    else:
        checkpoint = TestConfig.from_param_string(
            test_param,
            ModelType.VLA,
            TaskType.EXPORT,
            env_config,
            validate_environment=False).get_torch_model_dir()
    output = _onnx_dir(env_config)
    pytest_helpers.run_policy_command([
        sys.executable, "-m", "tensorrt_edgellm.onnx.cli", checkpoint, output,
        "--task", "policy", "--dtype", "float16"
    ], env_config, None, test_logger)
    for name in _artifacts("model.onnx",
                           "und_prefill/embed_tokens.safetensors"):
        assert (output / name).stat().st_size > 0, name
    for component in COMPONENTS:
        assert (output / component / "config.json").is_file()


def test_cpp_build(env_config, remote_config, test_logger):
    """Use the toolchain and TensorRT settings from the project CMake cache."""
    _build_binaries(env_config, remote_config, test_logger)


def test_engine_build(test_param, env_config, remote_config, test_logger):
    """The engine build yields a non-empty engine for every component plus the embedding table."""
    param = _param(test_param, engine=True)
    assert re.fullmatch(_ENGINE, param), param
    assert env_config.engine_dir, "ENGINE_DIR is required"
    _build_binaries(env_config, remote_config, test_logger)
    output = Path(env_config.engine_dir) / param
    pytest_helpers.run_policy_command([
        _binary(env_config, "cosmos3_policy_build"), "--onnx-dir",
        _onnx_dir(env_config), "--engine-dir", output, "--max-batch-size",
        param.rsplit("-mxbs", 1)[1]
    ],
                                      env_config,
                                      remote_config,
                                      test_logger,
                                      timeout=7200)
    for name in _artifacts("{component}.engine", "embed_tokens.safetensors"):
        pytest_helpers.run_policy_command(["test", "-s", output / name],
                                          env_config, remote_config,
                                          test_logger)


def _infer(test_param, env, remote, logger, accuracy):
    engine, options = _runtime(test_param)
    parts = options.split("-")
    assert parts[-1] in ("cosmos3_droid", "cosmos3_droid_lite")
    assert parts[:-1] in ([], ["cudagraph"])
    assert env.edgellm_data_dir, "EDGELLM_DATA_DIR is required"
    dataset_dir = Path(env.edgellm_data_dir) / "updated_datasets" / parts[-1]
    dataset = json.loads((dataset_dir / "dataset.json").read_text())
    assert dataset["model"] == MODEL
    assert dataset["samples"], "The policy dataset must contain samples"
    reference = (json.loads(
        (dataset_dir /
         "dataset_reference.json").read_text()) if accuracy else None)
    output_dir = Path(
        env.test_log_dir) / (f"{engine}-{options}" +
                             ("-accuracy" if accuracy else "-inference"))
    pytest_helpers.run_policy_command(["mkdir", "-p", output_dir], env, remote,
                                      logger)
    rows = []
    for sample in dataset["samples"] if accuracy else dataset["samples"][:1]:
        name = pytest_helpers.safe_sample_name(sample["name"])
        output = output_dir / f"{name}.json"
        pytest_helpers.run_policy_command(["rm", "-f", output], env, remote,
                                          logger)
        command = [
            _binary(env, "cosmos3_policy_inference"), "--engine-dir",
            Path(env.engine_dir) / engine, "--image",
            dataset_dir / sample["image"], "--prompt", sample["prompt"],
            "--view-point", sample["view_point"], "--seed",
            str(sample["seed"]), "--steps",
            str(dataset["num_inference_steps"]), "--output", output
        ]
        if "state" in sample:
            command += ["--state", ",".join(map(str, sample["state"]))]
        if "cudagraph" in parts:
            command += ["--cudagraph", "--warmup", "1", "--iters", "2"]
        pytest_helpers.run_policy_command(command, env, remote, logger)
        result = json.loads(
            pytest_helpers.run_policy_command(["cat", output], env, remote,
                                              logger))
        actions = np.asarray(result["action"], dtype=np.float64)
        assert result["shape"] == dataset["shape"]
        assert list(actions.shape) == dataset["shape"][1:]
        assert result["domain"] == "droid_lerobot"
        assert result["finite"] and np.isfinite(actions).all()
        assert result["num_inference_steps"] == dataset["num_inference_steps"]
        if accuracy:
            metrics = calculate_action_score.compare_actions(
                actions, reference["samples"][name]["actions"])
            rows.append({"name": name, **metrics})
            logger.info(f"Policy accuracy {name}: {metrics}")
    if accuracy:
        report = Path(env.test_log_dir) / f"{engine}-{options}-metrics.json"
        report.write_text(
            json.dumps({
                "reference": reference["provenance"],
                "samples": rows
            },
                       indent=2) + "\n")
        thresholds = reference["thresholds"]
        for row in rows:
            assert row["cosine"] >= thresholds["cosine_floor"], row
            assert row["max_abs_error"] <= thresholds["max_abs_ceiling"], row


def test_inference(test_param, env_config, remote_config, test_logger):
    """One sample yields a finite action chunk matching the dataset shape, domain and step count."""
    _infer(test_param, env_config, remote_config, test_logger, accuracy=False)


def test_accuracy(test_param, env_config, remote_config, test_logger):
    """Every sample matches the PyTorch reference within the dataset cosine and max-abs thresholds."""
    _infer(test_param, env_config, remote_config, test_logger, accuracy=True)
