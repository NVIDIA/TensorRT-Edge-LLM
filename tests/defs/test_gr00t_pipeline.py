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
"""GR00T N1.7 (experimental VLA runtime) export, engine build and inference tests.

GR00T has its own CLIs (``tensorrt-edgellm-export``, ``gr00t_policy_build``,
``gr00t_policy_inference``) and bundle layout, so it does not go through TestConfig /
llm_build. This covers the ONNX route; the checkpoint is resolved from
``GR00T_CHECKPOINT_DIR`` or ``<LLM_MODELS_DIR>/GR00T-N1.7-LIBERO/libero_10``.
"""
import json
import logging
import math
import os
import shlex
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import run_command, timer_context

from .test_common import _get_trt_env_vars

MODEL_NAME = "GR00T-N1.7-LIBERO"
CHECKPOINT_SUBDIR = "libero_10"
CHECKPOINT_REPO = f"nvidia/{MODEL_NAME}"
CHECKPOINT_PATTERNS = (
    f"{CHECKPOINT_SUBDIR}/config.json",
    f"{CHECKPOINT_SUBDIR}/embodiment_id.json",
    f"{CHECKPOINT_SUBDIR}/model*.safetensors*",
    f"{CHECKPOINT_SUBDIR}/processor_config.json",
    f"{CHECKPOINT_SUBDIR}/statistics.json",
)
TOKENIZER_REPO = "Qwen/Qwen3-VL-2B-Instruct"
TOKENIZER_PATTERNS = (
    "chat_template*",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer*",
    "vocab.json",
)
COMPONENTS = ("visual", "vl_backbone", "action")
CASE = f"{MODEL_NAME}-fp16"
BUILD_CASE = f"{CASE}-mxbs1"
#: Fixed instruction and state of the inference smoke request: 7 joint-group values and a
#: two-wide gripper, in the contract's order.
INSTRUCTION = "pick up the black bowl and place it on the plate."
STATE = [-0.39, 0.33, 0.85, 2.24, -1.48, -0.70, 0.012, -0.030]


def _checkpoint_is_ready(path: Path) -> bool:
    required = (
        "config.json",
        "embodiment_id.json",
        "model.safetensors.index.json",
        "statistics.json",
        "tokenizer/preprocessor_config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    )
    return (all((path / name).is_file() for name in required)
            and len(tuple(path.glob("model-*.safetensors"))) == 2)


def _checkpoint_dir(env_config: EnvironmentConfig) -> str:
    explicit = os.environ.get("GR00T_CHECKPOINT_DIR")
    if explicit:
        return explicit
    env_config.validate_for_export_tests()
    checkpoint = Path(
        env_config.llm_models_dir) / MODEL_NAME / CHECKPOINT_SUBDIR
    download_root = os.environ.get("HF_CHECKPOINT_DOWNLOAD_DIR")
    managed = (Path(download_root) / "nvidia" / MODEL_NAME /
               CHECKPOINT_SUBDIR) if download_root else None
    for candidate in (checkpoint, managed):
        if candidate is not None and _checkpoint_is_ready(candidate):
            return str(candidate)

    if (os.environ.get("EDGE_LLM_ALLOW_HF_DOWNLOAD", "1") == "0"
            or managed is None):
        return str(checkpoint)

    from huggingface_hub import snapshot_download

    snapshot_download(CHECKPOINT_REPO,
                      local_dir=managed.parent,
                      allow_patterns=CHECKPOINT_PATTERNS)
    snapshot_download(TOKENIZER_REPO,
                      local_dir=managed / "tokenizer",
                      allow_patterns=TOKENIZER_PATTERNS)
    if not _checkpoint_is_ready(managed):
        pytest.fail(f"downloaded GR00T checkpoint is incomplete: {managed}")
    return str(managed)


def _require(path: str, what: str) -> str:
    if not os.path.exists(path):
        pytest.fail(f"{what} not found: {path}")
    return path


def _workspace(env_config: EnvironmentConfig, kind: str) -> str:
    root = env_config.onnx_dir if kind == "onnx" else env_config.engine_dir
    if not root:
        pytest.fail(f"{'ONNX_DIR' if kind == 'onnx' else 'ENGINE_DIR'} "
                    "environment variable is required for GR00T tests")
    return os.path.join(root, f"{CASE}-{kind}")


def _env(env_config: EnvironmentConfig) -> dict:
    env_vars = dict(_get_trt_env_vars(env_config) or {})
    plugin = os.path.join(env_config.build_dir, "lib",
                          "libNvInfer_edgellm_plugin.so")
    env_vars["EDGELLM_PLUGIN_PATH"] = plugin
    env_vars["PYTHONNOUSERSITE"] = "1"
    return env_vars


def _tail(result: dict, lines: int = 25) -> str:
    return "\n".join(result.get("combined_output", "").splitlines()[-lines:])


def _run(cmd: str, env_config, remote_config, test_logger, timeout: int):
    return run_command(cmd=["bash", "-c", cmd],
                       remote_config=remote_config,
                       timeout=timeout,
                       logger=test_logger,
                       env_vars=_env(env_config))


def _read_text(path: str, remote_config, timeout: int = 30) -> str:
    if remote_config is None:
        return Path(path).read_text()

    result = subprocess.run([
        "sshpass", "-p", remote_config.password, "ssh", "-o",
        "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30", "-o",
        "ServerAliveCountMax=10", "-o", "TCPKeepAlive=yes",
        f"{remote_config.user}@{remote_config.host}",
        f"cat -- {shlex.quote(path)}"
    ],
                            capture_output=True,
                            text=True,
                            timeout=timeout)
    if result.returncode != 0:
        pytest.fail(f"failed to read {path} (rc={result.returncode}):\n"
                    f"{result.stderr.strip()}")
    return result.stdout


def _write_png(path: str, width: int, height: int, shade: int) -> None:
    """A deterministic gradient, so the request needs no image dataset."""
    rows = b"".join(b"\x00" + b"".join(
        bytes((x * 255 // width, y * 255 // height, shade))
        for x in range(width)) for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(
            ">I",
            zlib.crc32(body) & 0xffffffff)

    with open(path, "wb") as stream:
        stream.write(b"\x89PNG\r\n\x1a\n")
        stream.write(
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0,
                                       0)))
        stream.write(chunk(b"IDAT", zlib.compress(rows)))
        stream.write(chunk(b"IEND", b""))


def _read_json(path: str, remote_config) -> dict:
    return json.loads(_read_text(path, remote_config))


def _load_policy(bundle_dir: str, remote_config) -> dict:
    return _read_json(os.path.join(bundle_dir, "action", "policy.json"),
                      remote_config)


def test_checkpoint_export(test_param: str, env_config: EnvironmentConfig,
                           remote_config: Optional[RemoteConfig],
                           test_logger: logging.Logger) -> None:
    """tensorrt-edgellm-export writes the three components plus tokenizer/."""
    assert test_param == CASE
    checkpoint = _require(_checkpoint_dir(env_config), "GR00T checkpoint")
    onnx_dir = _workspace(env_config, "onnx")
    shutil.rmtree(onnx_dir, ignore_errors=True)

    with timer_context("GR00T N1.7 ONNX export", test_logger):
        result = _run(
            f"{shlex.quote(sys.executable)} -m tensorrt_edgellm.onnx.cli "
            f"{shlex.quote(checkpoint)} "
            f"{shlex.quote(onnx_dir)} --dtype float16", env_config,
            remote_config, test_logger, 3600)
    if not result["success"]:
        pytest.fail(f"GR00T export failed (rc={result['returncode']}):\n"
                    f"{_tail(result)}")

    for name in (*COMPONENTS, "tokenizer"):
        _require(os.path.join(onnx_dir, name), f"exported {name}/")
    _require(os.path.join(onnx_dir, "action", "policy.json"), "policy.json")


def test_engine_build(test_param: str, env_config: EnvironmentConfig,
                      remote_config: Optional[RemoteConfig],
                      test_logger: logging.Logger) -> None:
    """gr00t_policy_build compiles the exported components into one bundle whose
    policy.json carries the libero_sim contract."""
    assert test_param == BUILD_CASE
    env_config.validate_for_pipeline_tests()
    onnx_dir = _require(_workspace(env_config, "onnx"),
                        "ONNX export (run test_checkpoint_export first)")
    engine_dir = _workspace(env_config, "engine")
    shutil.rmtree(engine_dir, ignore_errors=True)

    binary = os.path.join(env_config.build_dir, "bin", "gr00t_policy_build")
    with timer_context("GR00T N1.7 engine build", test_logger):
        result = _run(
            f"{shlex.quote(binary)} --onnxDir {shlex.quote(onnx_dir)} "
            f"--engineDir {shlex.quote(engine_dir)}", env_config,
            remote_config, test_logger, 3600)
    if not result["success"]:
        pytest.fail(f"gr00t_policy_build failed (rc={result['returncode']}):\n"
                    f"{_tail(result)}")

    for name in (*COMPONENTS, "tokenizer"):
        _require(os.path.join(engine_dir, name), f"{name}/ in the bundle")
    _require(os.path.join(engine_dir, "embed_tokens.safetensors"),
             "embed_tokens.safetensors")

    policy = _load_policy(engine_dir, remote_config)
    assert policy["embodiment_tag"] == "libero_sim"
    assert policy["action"]["horizon"] == 16
    assert policy["action"]["keys"] == [
        "x", "y", "z", "roll", "pitch", "yaw", "gripper"
    ]
    assert sum(policy["action"]["dims"]) == 7
    assert sum(policy["state"]["dims"]) == 8
    assert policy["video_keys"] == ["image", "wrist_image"]


LITE_CASE = f"{BUILD_CASE}-gr00t_libero_lite"
ROBOT_ACTION_LIMITS = {
    "cosine_floor": 0.999,
    "max_abs_ceiling": 0.03,
}
NORMALIZED_ACTION_LIMITS = {
    "cosine_floor": 0.99999,
    "max_abs_ceiling": 0.1,
}
#: The SDK's own action-chunk scorer; the accuracy case calls it instead of re-implementing the metrics.
COMPARE_ACTIONS = (Path(__file__).resolve().parents[2] /
                   "experimental_models" / "scripts" / "compare_actions.py")


def _lite_dataset(env_config: EnvironmentConfig) -> Path:
    """``<EDGELLM_DATA_DIR>/updated_datasets/gr00t_libero_lite``: dataset.json, dataset_reference.json, samples/."""
    root = env_config.edgellm_data_dir
    if not root:
        pytest.fail(
            "EDGELLM_DATA_DIR is required for the gr00t_libero_lite cases")
    return Path(
        _require(os.path.join(root, "updated_datasets", "gr00t_libero_lite"),
                 "gr00t_libero_lite dataset"))


def _parse_scores(output: str) -> dict:
    """The ``cosine`` and ``max abs`` figures of a compare_actions.py report."""
    scores = {}
    for line in output.splitlines():
        if line.startswith("cosine"):
            scores["cosine"] = float(line.split()[1])
        elif line.startswith("max abs"):
            scores["max_abs"] = float(line.split()[2])
    return scores


def _lite_samples(dataset: Path) -> list:
    samples = sorted(p.name for p in (dataset / "samples").iterdir()
                     if (p / "observation.json").is_file())
    assert samples, f"no samples with observation.json under {dataset / 'samples'}"
    return samples


def _infer_and_check(binary: str,
                     engine_dir: str,
                     request_path: str,
                     output_path: str,
                     env_config,
                     remote_config,
                     test_logger,
                     noise_path: Optional[str] = None) -> dict:
    noise = f"--noise {shlex.quote(noise_path)} " if noise_path else ""
    run_log = output_path + ".log"
    with timer_context("GR00T N1.7 inference", test_logger):
        result = _run(
            f"{shlex.quote(binary)} --engineDir {shlex.quote(engine_dir)} "
            f"--inputFile {shlex.quote(request_path)} {noise}"
            f"--output {shlex.quote(output_path)} > {shlex.quote(run_log)} 2>&1",
            env_config, remote_config, test_logger, 600)
    if not result["success"]:
        details = _read_text(run_log, remote_config)
        details_tail = "\n".join(details.splitlines()[-25:])
        pytest.fail(
            f"gr00t_policy_inference failed (rc={result['returncode']}):\n"
            f"{details_tail}")

    response = _read_json(output_path, remote_config)
    horizon, dim = response["action_horizon"], response["action_dim"]
    assert (horizon, dim) == (16,
                              7), f"expected a 16x7 chunk, got {horizon}x{dim}"
    actions = response["robot_actions"]
    assert len(actions) == horizon * dim, (
        f"robot_actions holds {len(actions)} values, expected {horizon * dim}")
    assert all(math.isfinite(value) for value in actions)
    test_logger.info("%s: %dx%d robot action chunk, %.2f ms e2e",
                     Path(request_path).parent.name, horizon, dim,
                     response["e2e_ms_median"])
    return response


def _infer(test_param: str, env_config: EnvironmentConfig,
           remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
           accuracy: bool) -> None:
    """Runs gr00t_libero_lite from each sample's reference noise ``x0.bin``: the first sample only,
    or with ``accuracy`` every sample, scored against the fp32 PyTorch chunk with the repository's
    measured FP16 parity thresholds and ``experimental_models/scripts/compare_actions.py``."""
    engine_dir = _require(_workspace(env_config, "engine"),
                          "engine bundle (run test_engine_build first)")
    dataset = _lite_dataset(env_config)
    reference = json.loads((dataset / "dataset_reference.json").read_text())
    binary = os.path.join(env_config.build_dir, "bin",
                          "gr00t_policy_inference")
    work = os.path.join(engine_dir, "_lite_accuracy" if accuracy else "_lite")
    mkdir = _run(f"mkdir -p {shlex.quote(work)}", env_config, remote_config,
                 test_logger, 30)
    if not mkdir["success"]:
        pytest.fail(f"failed to create {work}:\n{_tail(mkdir)}")

    samples = _lite_samples(dataset)
    failures, rows = [], []
    for name in samples if accuracy else samples[:1]:
        sample = dataset / "samples" / name
        output = os.path.join(work, f"{name}.json")
        response = _infer_and_check(binary,
                                    engine_dir,
                                    str(sample / "observation.json"),
                                    output,
                                    env_config,
                                    remote_config,
                                    test_logger,
                                    noise_path=str(sample / "x0.bin"))
        if not accuracy:
            continue

        score_output = Path(
            env_config.test_log_dir) / f"{test_param}-{name}.json"
        score_output.write_text(json.dumps(response) + "\n")
        row = {"name": name}
        for field, ref_file, field_limits in (
            ("robot_actions", "reference_robot_actions.npy",
             ROBOT_ACTION_LIMITS),
            ("actions", "reference_actions.npy", NORMALIZED_ACTION_LIMITS),
        ):
            scored = subprocess.run([
                sys.executable,
                str(COMPARE_ACTIONS), "--reference",
                str(sample / ref_file), "--actual",
                str(score_output), "--json-field", field, "--cosine-floor",
                str(field_limits["cosine_floor"]), "--max-abs-ceiling",
                str(field_limits["max_abs_ceiling"])
            ],
                                    capture_output=True,
                                    text=True)
            test_logger.info("%s %s:\n%s%s", name, field, scored.stdout,
                             scored.stderr)
            row[field] = _parse_scores(scored.stdout)
            if scored.returncode != 0:
                failures.append(f"{name} {field}: {scored.stdout.strip()}")
        rows.append(row)

    if accuracy:
        metrics = Path(env_config.test_log_dir) / f"{test_param}-metrics.json"
        metrics.write_text(
            json.dumps(
                {
                    "reference_backend": reference["backend"],
                    "thresholds": {
                        "robot_actions": ROBOT_ACTION_LIMITS,
                        "actions": NORMALIZED_ACTION_LIMITS,
                    },
                    "samples": rows
                },
                indent=2) + "\n")
        assert not failures, "\n".join(failures)


def test_inference(test_param: str, env_config: EnvironmentConfig,
                   remote_config: Optional[RemoteConfig],
                   test_logger: logging.Logger) -> None:
    """Observations in, a finite 16x7 robot-unit action chunk (flattened) out. This is a smoke
    check; the numerical comparison against the PyTorch reference is ``test_accuracy``.
    ``libero_obs`` is one synthetic observation; ``gr00t_libero_lite`` runs the first real LIBERO
    sample of the dataset."""
    assert test_param in (f"{BUILD_CASE}-libero_obs", LITE_CASE)
    if test_param == LITE_CASE:
        _infer(test_param,
               env_config,
               remote_config,
               test_logger,
               accuracy=False)
        return

    engine_dir = _require(_workspace(env_config, "engine"),
                          "engine bundle (run test_engine_build first)")
    binary = os.path.join(env_config.build_dir, "bin",
                          "gr00t_policy_inference")
    work = os.path.join(engine_dir, "_smoke")
    os.makedirs(work, exist_ok=True)

    policy = _load_policy(engine_dir, remote_config)
    for index, name in enumerate(policy["video_keys"]):
        _write_png(os.path.join(work, f"{name}.png"), 256, 256, 64 * index)
    request = {
        "task": INSTRUCTION,
        "state": STATE,
        "cameras": {
            name: f"{name}.png"
            for name in policy["video_keys"]
        },
    }
    request_path = os.path.join(work, "observation.json")
    with open(request_path, "w") as stream:
        json.dump(request, stream)
    _infer_and_check(binary, engine_dir, request_path,
                     os.path.join(work, "action.json"), env_config,
                     remote_config, test_logger)


def test_inference_canonical(test_param: str, env_config: EnvironmentConfig,
                             remote_config: Optional[RemoteConfig],
                             test_logger: logging.Logger) -> None:
    """Canonical mode drives the engines on a seeded synthetic VL context (no vision front
    end) and yields the same finite 16x7 chunk on every run."""
    assert test_param == f"{BUILD_CASE}-canonical"
    engine_dir = _require(_workspace(env_config, "engine"),
                          "engine bundle (run test_engine_build first)")
    work = os.path.join(engine_dir, "_canonical")
    os.makedirs(work, exist_ok=True)
    binary = os.path.join(env_config.build_dir, "bin",
                          "gr00t_policy_inference")

    chunks = []
    for run in range(2):
        output_path = os.path.join(work, f"action_{run}.json")
        result = _run(
            f"{shlex.quote(binary)} --engineDir {shlex.quote(engine_dir)} "
            f"--seed 0 --output {shlex.quote(output_path)}", env_config,
            remote_config, test_logger, 600)
        if not result["success"]:
            pytest.fail(
                f"gr00t_policy_inference failed (rc={result['returncode']}):\n"
                f"{_tail(result)}")
        chunks.append(_read_json(output_path, remote_config))

    first = chunks[0]
    assert (first["action_horizon"], first["action_dim"]) == (16, 7)
    assert len(first["robot_actions"]) == 16 * 7
    assert all(math.isfinite(value) for value in first["robot_actions"])
    assert chunks[0]["actions"] == chunks[1]["actions"], \
        "the same seed must give the same chunk"


def test_accuracy(test_param: str, env_config: EnvironmentConfig,
                  remote_config: Optional[RemoteConfig],
                  test_logger: logging.Logger) -> None:
    """Every gr00t_libero_lite sample matches normalized and robot-unit PyTorch chunks."""
    assert test_param == LITE_CASE
    _infer(test_param, env_config, remote_config, test_logger, accuracy=True)
