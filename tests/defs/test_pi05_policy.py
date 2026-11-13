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
"""pi0.5 LIBERO export and action-parity tests; requires the pi05 SDK implementation."""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest_helpers

from examples.accuracy.scripts import calculate_action_score

MODEL = "pi05_libero_base"
PROFILE = MODEL + "-fp16"
ACTION_LIMITS = {
    "cosine_floor": 0.99999,
    "max_abs_ceiling": 0.006,
}


def _binary(env, name):
    return Path(env.build_dir) / "bin" / name


def test_checkpoint_export(test_param, env_config, test_logger):
    """Export publishes every pi0.5 ONNX component and its normalization asset."""
    assert test_param == PROFILE
    env_config.validate_for_export_tests()
    output = Path(env_config.onnx_dir) / PROFILE
    pytest_helpers.run_policy_command([
        sys.executable, "-m", "tensorrt_edgellm.onnx.cli",
        Path(env_config.llm_models_dir) / MODEL, output, "--dtype", "float16",
        "--pi05-policy-config", "pi05_libero"
    ], env_config, None, test_logger)
    policy = json.loads((output / "policy.json").read_text())
    for component in policy["components"]:
        assert (output / component / "model.onnx").stat().st_size > 0
    stats = _dataset_dir(env_config, "pi05_libero_lite") / "norm_stats.json"
    (output / "assets").mkdir(exist_ok=True)
    shutil.copyfile(stats, output / "assets/norm_stats.json")


def _dataset_dir(env, name):
    return Path(env.edgellm_data_dir) / "updated_datasets" / name


def test_cpp_build(env_config, remote_config, test_logger):
    """Build the pi0.5 binaries from this checkout; experimental models are off by default."""
    build = Path(env_config.build_dir)
    pytest_helpers.run_policy_command(
        ["cmake", build, "-DBUILD_EXPERIMENTAL_MODELS=ON"], env_config,
        remote_config, test_logger)
    pytest_helpers.run_policy_command([
        "cmake", "--build", build, "--parallel", "16", "--target",
        "pi05_policy_build", "pi05_policy_inference", "NvInfer_edgellm_plugin"
    ], env_config, remote_config, test_logger)


def test_engine_build(test_param, env_config, remote_config, test_logger):
    """The engine build yields a non-empty engine for every component listed in policy.json."""
    assert test_param == PROFILE + "-mxbs1"
    env_config.validate_for_pipeline_tests()
    onnx_dir = Path(env_config.onnx_dir) / PROFILE
    output = Path(env_config.engine_dir) / PROFILE
    pytest_helpers.run_policy_command([
        _binary(env_config, "pi05_policy_build"), "--onnx-dir", onnx_dir,
        "--engine-dir", output, "--max-batch-size", "1"
    ],
                                      env_config,
                                      remote_config,
                                      test_logger,
                                      timeout=7200)
    policy = json.loads(
        pytest_helpers.run_policy_command(["cat", output / "policy.json"],
                                          env_config, remote_config,
                                          test_logger))
    for component in policy["components"]:
        pytest_helpers.run_policy_command(
            ["test", "-s", output / component / (component + ".engine")],
            env_config, remote_config, test_logger)


def _infer(test_param, env, remote, logger, accuracy):
    prefix = PROFILE + "-mxbs1-"
    assert test_param.startswith(prefix)
    name = test_param.removeprefix(prefix)
    assert name in ("pi05_libero", "pi05_libero_lite")
    dataset_dir = _dataset_dir(env, name)
    dataset = json.loads((dataset_dir / "dataset.json").read_text())
    reference = json.loads(
        (dataset_dir / "dataset_reference.json").read_text())
    assert dataset["model"] == "lerobot/pi05_libero_base" and dataset["samples"]
    output_dir = Path(env.test_log_dir) / test_param
    pytest_helpers.run_policy_command(["mkdir", "-p", output_dir], env, remote,
                                      logger)
    rows = []
    for sample in dataset["samples"] if accuracy else dataset["samples"][:1]:
        files = {
            key: dataset_dir / value
            for key, value in sample["files"].items()
        }
        name = pytest_helpers.safe_sample_name(sample["name"])
        output = output_dir / (name + ".json")
        pytest_helpers.run_policy_command(["rm", "-f", output], env, remote,
                                          logger)
        pytest_helpers.run_policy_command([
            _binary(env, "pi05_policy_inference"), "--engine-dir",
            Path(env.engine_dir) / PROFILE, "--num-views",
            str(sample["num_views"]), "--tokens",
            files["tokens"].read_text().strip(), "--pixelValues",
            files["pixel_values"], "--noise", files["noise"], "--steps",
            str(dataset["denoise_steps"]), "--output", output
        ], env, remote, logger)
        actual = np.asarray(
            json.loads(
                pytest_helpers.run_policy_command(["cat", output], env, remote,
                                                  logger))["actions"])
        assert actual.shape == (sample["horizon"], sample["action_dim"])
        assert np.isfinite(actual).all()
        if accuracy:
            rows.append({
                "name":
                name,
                **calculate_action_score.compare_actions(
                    actual, reference["actions"][name])
            })
    if accuracy:
        (Path(env.test_log_dir) / (test_param + "-metrics.json")).write_text(
            json.dumps(
                {
                    "reference_backend": reference["backend"],
                    "thresholds": ACTION_LIMITS,
                    "samples": rows
                },
                indent=2) + "\n")
        for row in rows:
            assert row["cosine"] >= ACTION_LIMITS["cosine_floor"] and row[
                "max_abs_error"] <= ACTION_LIMITS["max_abs_ceiling"], row


def test_inference(test_param, env_config, remote_config, test_logger):
    """Synthetic and dataset inputs yield a finite action chunk of the declared horizon and dimension."""
    synthetic = re.fullmatch(
        re.escape(PROFILE) + r"-mxbs1-views([23])-dstp(10)-policy_basic",
        test_param)
    if synthetic:
        env_config.validate_for_pipeline_tests()
        output = Path(env_config.test_log_dir) / (test_param + ".json")
        pytest_helpers.run_policy_command(["mkdir", "-p", output.parent],
                                          env_config, remote_config,
                                          test_logger)
        pytest_helpers.run_policy_command(["rm", "-f", output], env_config,
                                          remote_config, test_logger)
        pytest_helpers.run_policy_command([
            _binary(env_config, "pi05_policy_inference"), "--engine-dir",
            Path(env_config.engine_dir) / PROFILE, "--num-views", synthetic[1],
            "--steps", synthetic[2], "--seed", "0", "--output", output
        ], env_config, remote_config, test_logger)
        result = json.loads(
            pytest_helpers.run_policy_command(["cat", output], env_config,
                                              remote_config, test_logger))
        actions = np.asarray(result["actions"])
        assert actions.shape == (result["action_horizon"],
                                 result["action_dim"])
        assert actions.size > 0 and np.isfinite(actions).all()
    else:
        _infer(test_param,
               env_config,
               remote_config,
               test_logger,
               accuracy=False)


def test_accuracy(test_param, env_config, remote_config, test_logger):
    """Every sample matches the reference at cosine >= 0.99999 and max-abs error <= 0.006."""
    _infer(test_param, env_config, remote_config, test_logger, accuracy=True)
