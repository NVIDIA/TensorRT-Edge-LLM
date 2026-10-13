#!/usr/bin/env python3
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
"""RoboLab InferenceClient for the Edge-LLM Cosmos3 policy server.

Mirrors the upstream OpenPI/DreamZero ``Pi0DroidJointposClient`` contract:
``_extract_observation`` / ``_pack_request`` / ``_query_server`` /
``_unpack_response`` plus open-loop action chunking. Unlike the upstream
websocket+msgpack transport, this client speaks **HTTP + JSON** to the
Edge-LLM Cosmos3 policy server (``policy_server.py``), matching the project's
JSON I/O interface convention.

Contract:
    simulator obs -> HTTP/JSON request -> action chunk (32, 8) -> one action / step

The client is dependency-light (stdlib ``urllib`` + numpy) so it can run inside
the Isaac Lab / Isaac Sim launcher on the x86 host.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
import urllib.request
import uuid
from typing import Any

import numpy as np

logger = logging.getLogger("cosmos3.robolab.client")

ACTION_CHUNK_SIZE = 32
RAW_ACTION_DIM = 8
DEFAULT_OPEN_LOOP_HORIZON = 8


def _encode_image_b64(image: np.ndarray) -> str:
    """Encode an HxWx3 uint8 image as a base64 PNG string."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Cosmos3 RoboLab client requires Pillow to encode observation images."
        ) from exc
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Cosmos3JointposClient:
    """DROID-style RoboLab client for the Edge-LLM Cosmos3 policy server.

    Modeled on ``Pi0DroidJointposClient``: it turns one simulator observation
    into one executable 8-D action, calling the server only when the local
    open-loop chunk cache is exhausted.
    """

    def __init__(
        self,
        remote_host: str = "127.0.0.1",
        remote_port: int = 8080,
        path: str = "/infer",
        open_loop_horizon: int = DEFAULT_OPEN_LOOP_HORIZON,
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = f"http://{remote_host}:{remote_port}"
        self.path = path
        self.open_loop_horizon = open_loop_horizon
        self.timeout_s = timeout_s
        # Chunking state.
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk: np.ndarray | None = None
        self.session_id = str(uuid.uuid4())
        # Reporting.
        self.server_calls = 0
        self.last_chunk_latency_s: float | None = None
        self.last_used_server_call = False
        self._server_metadata = self._fetch_metadata()

    # ------------------------------------------------------------------ #
    # Transport helpers (mirror the Pi0DroidJointposClient method names). #
    # ------------------------------------------------------------------ #
    def _fetch_metadata(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/metadata",
                                        timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - server may not expose it
            logger.warning("could not fetch server metadata: %r", exc)
            return {}

    def metadata(self) -> dict[str, Any]:
        return dict(self._server_metadata)

    @staticmethod
    def _extract_observation(obs_dict: dict[str, Any]) -> dict[str, Any]:
        """Extract the (image, instruction, proprio) RoboLab cares about.

        Isaac Lab exposes camera + robot state under ``obs["policy"]``. Cosmos3
        conditions on a single RGB frame + a text instruction, so we pull the
        primary external camera (falling back to any available frame) plus the
        proprioception used for logging.
        """
        policy = obs_dict.get("policy", obs_dict)

        def _to_np(x):
            if x is None:
                return None
            if hasattr(x, "detach"):  # torch tensor
                x = x.detach().cpu().numpy()
            return np.asarray(x)

        # Primary conditioning camera (env 0), with graceful fallbacks.
        image = None
        for key in ("external_cam", "external_cam_2", "wrist_cam", "image",
                    "rgb"):
            if key in policy:
                arr = _to_np(policy[key])
                image = arr[0] if arr.ndim == 4 else arr
                break
        if image is None:
            raise KeyError(
                "observation has no usable camera frame (external_cam/wrist_cam/image)"
            )

        joint_position = _to_np(policy.get("arm_joint_pos"))
        gripper_position = _to_np(policy.get("gripper_pos"))
        return {
            "image": image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }

    def _pack_request(self, curr_obs: dict[str, Any],
                      instruction: str) -> dict[str, Any]:
        """Build the JSON request payload for the Cosmos3 policy server."""
        joint = np.asarray(curr_obs["joint_position"],
                           dtype=np.float32).reshape(-1)
        gripper = np.asarray(curr_obs["gripper_position"],
                             dtype=np.float32).reshape(-1)
        return {
            "image": _encode_image_b64(curr_obs["image"]),
            "instruction": instruction,
            "domain": "droid_lerobot",
            "session_id": self.session_id,
            "state": np.concatenate((joint, 1.0 - gripper)).tolist(),
        }

    def _query_server(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """POST the JSON request and return the parsed JSON response."""
        body = json.dumps(request_data).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{self.path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _unpack_response(response: dict[str, Any]) -> np.ndarray:
        """Decode the server response into an ``(ACTION_CHUNK_SIZE, RAW_ACTION_DIM)`` array."""
        if isinstance(response, dict) and response.get("type") == "error":
            raise RuntimeError(
                f"Error in inference server:\n{response.get('message', response)}"
            )
        action = np.asarray(response["action"], dtype=np.float32)
        # Server emits [1, 32, 8] or [32, 8]; normalize to (32, 8).
        action = action.reshape(ACTION_CHUNK_SIZE, RAW_ACTION_DIM)
        return action

    # ------------------------------------------------------------------ #
    # Public API.                                                         #
    # ------------------------------------------------------------------ #
    def reset(self) -> str:
        """Reset local chunk state and (best-effort) the remote session."""
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.session_id = str(uuid.uuid4())
        self.last_chunk_latency_s = None
        self.last_used_server_call = False
        # The server exposes reset at its own /reset route (stateless no-op).
        try:
            body = json.dumps({"session_id": self.session_id}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/reset",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return str(
                    json.loads(resp.read().decode("utf-8")).get(
                        "status", "reset successful"))
        except Exception:
            return "reset successful"

    def infer(self, obs: dict[str, Any], instruction: str) -> dict[str, Any]:
        """Turn one simulator observation into one executable 8-D action."""
        curr_obs = self._extract_observation(obs)
        self.last_used_server_call = False

        if (self.actions_from_chunk_completed == 0
                or self.actions_from_chunk_completed >= self.open_loop_horizon
                or self.pred_action_chunk is None):
            self.actions_from_chunk_completed = 0
            request_data = self._pack_request(curr_obs, instruction)
            start = time.perf_counter()
            response = self._query_server(request_data)
            self.last_chunk_latency_s = time.perf_counter() - start
            self.last_used_server_call = True
            self.server_calls += 1
            actions = self._unpack_response(response)
            if actions.shape != (ACTION_CHUNK_SIZE, RAW_ACTION_DIM):
                raise AssertionError(
                    f"Expected action shape {(ACTION_CHUNK_SIZE, RAW_ACTION_DIM)}, got {actions.shape}"
                )
            self.pred_action_chunk = actions

        action = np.array(
            self.pred_action_chunk[self.actions_from_chunk_completed],
            copy=True)
        self.actions_from_chunk_completed += 1
        # DROID sim-eval binarizes the gripper command.
        action[-1] = 1.0 if float(action[-1]) > 0.5 else 0.0

        return {
            "action":
            action,
            "joint_position":
            curr_obs["joint_position"],
            "gripper_position":
            curr_obs["gripper_position"],
            "used_server_call":
            self.last_used_server_call,
            "chunk_latency_s":
            self.last_chunk_latency_s if self.last_used_server_call else None,
        }
