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

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from experimental_models.cosmos3.robolab.openpi_adapter import (
    EdgeLLMOpenPIPolicy, backend_result_to_openpi, openpi_obs_to_request)
from experimental_models.cosmos3.robolab.policy_server import \
    Cosmos3PolicyBackend


def test_openpi_observation_preserves_official_composite_image_and_state():
    image = np.zeros((540, 640, 3), np.uint8)
    request = openpi_obs_to_request({
        "observation/image":
        image,
        "prompt":
        "pick the mug",
        "observation/joint_position":
        np.arange(7, dtype=np.float32),
        "observation/gripper_position":
        np.array([0.3], np.float32),
    })

    assert request["image"] is image
    assert request["instruction"] == "pick the mug"
    assert request["domain"] == "droid_lerobot"
    np.testing.assert_allclose(request["state"], [0, 1, 2, 3, 4, 5, 6, 0.7])


def test_openpi_observation_requires_policy_state():
    with pytest.raises(ValueError, match="joint_position"):
        openpi_obs_to_request({
            "observation/image":
            np.zeros((8, 8, 3), np.uint8),
            "prompt":
            "pick the mug",
        })


def test_backend_result_normalizes_action_chunk():
    result = {"action": np.arange(32 * 8, dtype=np.float32).reshape(1, 32, 8)}
    payload = backend_result_to_openpi(result)
    assert payload["action"].shape == (32, 8)


def test_backend_result_rejects_malformed_action_chunk():
    with pytest.raises(ValueError, match="divisible by dim 8"):
        backend_result_to_openpi({"action": np.zeros(255, dtype=np.float32)})


def test_openpi_policy_forwards_request_and_timing():

    class FakeBackend:
        domain = "droid_lerobot"
        action_chunk_size = 32
        steps = 4
        guidance = 3.0
        viewpoint = "concat_view"

        def infer(self, request):
            assert request["instruction"] == "pick the mug"
            return {
                "action": np.zeros((32, 8), np.float32).tolist(),
                "meta": {
                    "server_latency_s": 0.01
                },
            }

    policy = EdgeLLMOpenPIPolicy(FakeBackend())
    output = policy.infer({
        "observation/image":
        np.zeros((540, 640, 3), np.uint8),
        "prompt":
        "pick the mug",
        "observation/joint_position":
        np.zeros(7, np.float32),
        "observation/gripper_position":
        np.zeros(1, np.float32),
    })

    assert output["action"].shape == (32, 8)
    assert output["server_timing"] == {"infer_ms": 10.0}


def test_policy_backend_disables_cli_benchmark_warmup(tmp_path):
    binary = tmp_path / "cosmos3_policy_inference"
    binary.write_text("")
    gen = tmp_path / "engines" / "gen"
    gen.mkdir(parents=True)
    (gen / "config.json").write_text(
        '{"action_chunk_size": 32, "fps": 15.0, "raw_action_dim": 8, '
        '"state_rows": 1, "use_state": true, "action_start_frame_offset": 0}')
    backend = Cosmos3PolicyBackend(str(binary),
                                   str(tmp_path / "engines"),
                                   persistent=False)

    def fake_run(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text('{"action": ' + repr(np.zeros((32, 8)).tolist()) +
                          ', "meta": {}}')
        return mock.Mock(returncode=0, stderr="")

    with mock.patch("subprocess.run", side_effect=fake_run) as run:
        request = {
            "image": np.zeros((8, 8, 3), np.uint8),
            "instruction": "pick the mug",
            "state": [0.0] * 8,
        }
        backend.infer(request)
        backend.infer(request)

    command = run.call_args_list[0].args[0]
    assert command[command.index("--warmup") + 1] == "0"
    assert command[command.index("--iters") + 1] == "1"
    expected_seeds = np.random.default_rng(0).integers(0, 2**31, size=2)
    actual_seeds = []
    for call in run.call_args_list:
        command = call.args[0]
        seed_index = len(command) - 1 - command[::-1].index("--seed")
        actual_seeds.append(int(command[seed_index + 1]))
    np.testing.assert_array_equal(actual_seeds, expected_seeds)


def test_policy_backend_reuses_persistent_worker(tmp_path):
    binary = tmp_path / "cosmos3_policy_inference"
    binary.write_text("""#!/usr/bin/env python3
import json
import sys

print("@@EDGELLM_READY {}", flush=True)
counter = 0
for line in sys.stdin:
    request = json.loads(line)
    counter += 1
    with open(request["output"], "w") as output:
        json.dump({
            "action": [[0.0] * 8 for _ in range(32)],
            "meta": {"worker_request": counter},
        }, output)
    print("@@EDGELLM_RESPONSE " + json.dumps({
        "ok": True,
        "output": request["output"],
    }), flush=True)
""")
    binary.chmod(0o755)
    gen = tmp_path / "engines" / "gen"
    gen.mkdir(parents=True)
    (gen / "config.json").write_text(
        '{"action_chunk_size": 32, "fps": 15.0, "raw_action_dim": 8, '
        '"state_rows": 1, "use_state": true, "action_start_frame_offset": 0}')
    backend = Cosmos3PolicyBackend(str(binary), str(tmp_path / "engines"))
    request = {
        "image": np.zeros((8, 8, 3), np.uint8),
        "instruction": "pick the mug",
        "state": [0.0] * 8,
    }
    try:
        first = backend.infer(request)
        second = backend.infer(request)
    finally:
        backend.close()

    assert first["meta"]["worker_request"] == 1
    assert second["meta"]["worker_request"] == 2


def test_policy_backend_restarts_worker_after_timeout(tmp_path):
    binary = tmp_path / "cosmos3_policy_inference"
    marker = tmp_path / "hung_once"
    binary.write_text("""#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
import time

marker = Path(os.environ["HANG_MARKER"])
print("@@EDGELLM_READY {}", flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if not marker.exists():
        marker.touch()
        time.sleep(2)
        continue
    with open(request["output"], "w") as output:
        json.dump({"action": [[0.0] * 8 for _ in range(32)], "meta": {}}, output)
    print("@@EDGELLM_RESPONSE " + json.dumps({"ok": True}), flush=True)
""")
    binary.chmod(0o755)
    gen = tmp_path / "engines" / "gen"
    gen.mkdir(parents=True)
    (gen / "config.json").write_text(
        '{"action_chunk_size": 32, "fps": 15.0, "raw_action_dim": 8, '
        '"state_rows": 1, "use_state": true, "action_start_frame_offset": 0}')
    backend = Cosmos3PolicyBackend(
        str(binary),
        str(tmp_path / "engines"),
        extra_env={"HANG_MARKER": str(marker)},
        worker_timeout_s=0.1,
    )
    request = {
        "image": np.zeros((8, 8, 3), np.uint8),
        "instruction": "pick the mug",
        "state": [0.0] * 8,
    }
    try:
        with pytest.raises(TimeoutError, match="timed out"):
            backend.infer(request)
        result = backend.infer(request)
    finally:
        backend.close()

    assert np.asarray(result["action"]).shape == (32, 8)
