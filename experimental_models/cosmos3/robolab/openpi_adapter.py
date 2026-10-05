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
"""Map OpenPI / RoboLab observations onto the Edge-LLM HTTP policy backend.

Isaac's ``policies/cosmos3/client.py`` speaks OpenPI websocket+msgpack:
``observation/image``, ``prompt``, ``observation/joint_position``,
``observation/gripper_position``. Policy-DROID forwards the latest 8-D
state (seven joints plus flipped gripper) into the C++ CLI.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .policy_server import RAW_ACTION_DIM, Cosmos3PolicyBackend


def openpi_obs_to_request(
        obs: dict[str, Any],
        default_domain: str = "droid_lerobot") -> dict[str, Any]:
    """Convert one OpenPI observation dict into a backend ``infer`` request."""
    if "observation/image" in obs:
        image = obs["observation/image"]
    elif "image" in obs:
        image = obs["image"]
    else:
        raise KeyError("observation missing 'observation/image'")
    instruction = obs.get("prompt") or obs.get("instruction")
    if not instruction:
        raise ValueError("observation missing 'prompt' / 'instruction'")
    request: dict[str, Any] = {
        "image": image,
        "instruction": instruction,
        "domain": obs.get("domain", default_domain),
    }
    joint_value = obs.get("observation/joint_position")
    gripper_value = obs.get("observation/gripper_position")
    if joint_value is None or gripper_value is None:
        raise ValueError(
            "Policy-DROID requires observation/joint_position and observation/gripper_position"
        )
    joint = np.asarray(joint_value, dtype=np.float32)
    gripper = np.asarray(gripper_value, dtype=np.float32)
    joint = joint if joint.ndim == 1 else joint[-1]
    gripper = gripper if gripper.ndim == 1 else gripper[-1]
    # Match cosmos-framework serving: the model-space state uses an inverted
    # gripper convention. The C++ JSON output flips predicted gripper back.
    request["state"] = np.concatenate(
        (joint.reshape(-1), 1.0 - gripper.reshape(-1))).tolist()
    if obs.get("steps") is not None:
        request["steps"] = obs["steps"]
    if obs.get("seed") is not None:
        request["seed"] = int(obs["seed"])
    return request


def backend_result_to_openpi(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize CLI JSON into the OpenPI ``{"action": (T, 8)}`` payload."""
    action = np.asarray(result["action"], dtype=np.float32)
    if action.size == 0 or action.size % RAW_ACTION_DIM:
        raise ValueError(
            f"expected a non-empty action tensor divisible by dim {RAW_ACTION_DIM}, "
            f"got shape {action.shape}")
    return {"action": action.reshape(-1, RAW_ACTION_DIM)}


class EdgeLLMOpenPIPolicy:
    """``BasePolicy``-shaped wrapper: ``infer(obs) -> {action: ndarray}``."""

    def __init__(self, backend: Cosmos3PolicyBackend) -> None:
        self.backend = backend

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        request = openpi_obs_to_request(obs,
                                        default_domain=self.backend.domain)
        result = self.backend.infer(request)
        payload = backend_result_to_openpi(result)
        meta = result.get("meta") or {}
        payload["server_timing"] = {
            "infer_ms": float(meta.get("server_latency_s", 0.0)) * 1000.0,
        }
        return payload

    def reset(self) -> None:
        return None

    def metadata(self) -> dict[str, Any]:
        return {
            "policy": "edge-cosmos3",
            "action_chunk_size": self.backend.action_chunk_size,
            "raw_action_dim": RAW_ACTION_DIM,
            "action_shape": [self.backend.action_chunk_size, RAW_ACTION_DIM],
            "domain": self.backend.domain,
            "num_inference_steps": self.backend.steps,
            "guidance": self.backend.guidance,
            "viewpoint": self.backend.viewpoint,
        }
