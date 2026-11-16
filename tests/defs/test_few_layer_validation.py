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
"""Run model-owned few-layer PyTorch-to-engine parity cases."""

import logging
import os
import sys

import pytest
from conftest import EnvironmentConfig
from pytest_helpers import run_command, timer_context

from .config import TaskType, TestConfig, infer_checkpoint_export_model_type
from .utils.device import DeviceConfig


def _version_tuple(version: str) -> tuple:
    """"10.16.0.72" -> (10, 16, 0, 72); stops at the first non-numeric token."""
    parts = []
    for token in version.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _draft_checkpoint(config: TestConfig) -> tuple[str, str]:
    """Return the few-layer driver flag and directory for a spec-decode draft."""
    if config.is_dflash:
        return "--dflash-draft-dir", config.get_dflash_draft_model_dir()
    # is_eagle also covers MTP, whose draft ships inside the base checkpoint.
    if config.is_eagle and not config.is_mtp:
        return "--eagle3-draft-dir", config.get_eagle_draft_checkpoint_dir()
    return "", ""


def test_few_layer_validation(request, test_param: str,
                              env_config: EnvironmentConfig, remote_config,
                              test_logger: logging.Logger):
    """Functional few-layer golden-vs-engine numeric smoke."""
    spec = getattr(test_param, "inference", None)
    if not spec or spec.get("scenario") != "few_layer":
        pytest.fail("few-layer validation requires a model-owned YAML case")

    # Host (server) only: the golden runs PyTorch + transformers locally, which
    # the edge boards (remote execution) do not have.
    if remote_config is not None:
        pytest.skip("few-layer validation runs on the host (server) only; "
                    "the PyTorch golden needs torch + transformers")

    # These cases require target-specific CuTe DSL kernels and/or precision.
    device_config = DeviceConfig.auto_detect(remote_config, test_logger)
    cc = device_config.compute_capability
    if cc is None:
        pytest.fail(f"cannot determine target SM for {test_param}")
    minimum_cc = spec.get("minimum_compute_capability", 0)
    if cc < minimum_cc:
        pytest.skip(f"{test_param} requires SM{minimum_cc}+; this runner is "
                    f"compute capability {cc}")

    # Some arms need a newer TensorRT than the runner's (the TRT-native MoE arm
    # needs the TRT_MoE importer, TensorRT >= 11.3). CI selects the runtime
    # through TRT_VERSION (see TestConfig.check_trt_native_attn); fall back to
    # the installed wheel when the variable is absent.
    minimum_trt = spec.get("minimum_trt_version")
    if minimum_trt:
        trt_ver = os.environ.get("TRT_VERSION", "")
        if not trt_ver:
            try:
                import tensorrt
                trt_ver = tensorrt.__version__
            except ImportError:
                trt_ver = ""
        if _version_tuple(trt_ver) < _version_tuple(minimum_trt):
            pytest.skip(f"{test_param} requires TensorRT >= {minimum_trt}; "
                        f"this runner has TRT_VERSION={trt_ver!r}")

    model_type = infer_checkpoint_export_model_type(test_param)
    config = TestConfig.from_param_string(test_param,
                                          model_type,
                                          TaskType.CHECKPOINT_BUILD,
                                          env_config,
                                          validate_environment=False)
    model_dir = config.get_torch_model_dir()
    if not os.path.isdir(model_dir):
        pytest.skip(f"checkpoint not found: {model_dir}")

    # build_dir may be relative ("build"); the script needs an absolute path
    # since it resolves binaries against it independent of its own CWD.
    build_dir = env_config.build_dir
    if not os.path.isabs(build_dir):
        build_dir = os.path.join(env_config.llm_sdk_dir, build_dir)

    script = os.path.join(env_config.llm_sdk_dir, "scripts",
                          "few-layer-validation.sh")
    cmd = [
        "bash",
        script,
        "--model",
        model_dir,
        "--num-layers",
        str(spec["num_layers"]),
        "--build-dir",
        build_dir,
        # The pytest interpreter has torch + transformers + the TRT wheel.
        "--python",
        sys.executable,
        "--cos",
        str(spec["cosine_threshold"]),
        "--target-sm",
        str(cc),
        "--input-file",
        config.get_test_case_file(),
    ]
    # --input-file is resolved against the repo, which is where the script runs from.
    for arg in spec.get("arguments", []):
        cmd.append(
            os.path.join(env_config.llm_sdk_dir, arg) if arg.
            startswith("tests/test_cases/") else arg)

    # A DFlash / EAGLE3 draft is a second checkpoint, untruncated: only the base
    # is cut to --num-layers. Taking it from the case's draft_id_or_dir keeps it
    # on the checkpoint resolver, rather than hard-coding a path into arguments.
    draft_flag, draft_dir = _draft_checkpoint(config)
    if draft_flag:
        if not os.path.isdir(draft_dir):
            pytest.skip(f"draft checkpoint not found: {draft_dir}")
        cmd += [draft_flag, draft_dir]

    with timer_context(f"few-layer-validation [{test_param}]", test_logger):
        result = run_command(cmd=cmd,
                             remote_config=remote_config,
                             timeout=spec.get("timeout_seconds", 1200),
                             logger=test_logger)

    # A passing job otherwise records no number anywhere: the per-test log is not
    # archived, and pytest's fd capture swallows the driver's stdout on success.
    # Thresholds here are supposed to be measured rather than tuned, which needs
    # samples, so put the verdict somewhere a later run can still read it.
    verdict = next((line for line in reversed(result["output"].splitlines())
                    if line.lstrip().startswith("[compare] ")), "")
    reporter = request.config.pluginmanager.getplugin("terminalreporter")
    if verdict and reporter is not None:
        reporter.write_line(f"[few-layer] {test_param} {verdict.strip()}")

    if not result["success"]:
        pytest.fail(f"few-layer validation FAILED for {test_param} "
                    f"(exit {result.get('returncode')}); see log above")
