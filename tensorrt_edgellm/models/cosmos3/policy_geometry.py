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
"""Framework-independent Cosmos3 policy request geometry."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_ACTION_CHUNK_SIZE = 16
DEFAULT_DOMAIN = "droid_lerobot"
DEFAULT_DOMAIN_ID = 8
DEFAULT_FPS = 5.0
DEFAULT_HEIGHT = 544
DEFAULT_MAX_UND_LEN = 512
DEFAULT_NUM_FRAMES = 17
# Dims >= raw_action_dim of the action head are zeroed padding.
DEFAULT_RAW_ACTION_DIM = 10
DEFAULT_WIDTH = 736
TEMPORAL_COMPRESSION_FACTOR = 4
VAE_PATCH_SIZE = 2
VAE_SPATIAL_DOWNSAMPLE_FACTOR = 16

# Geometry the released checkpoints imply by domain name but omit from metadata.
_DOMAIN_DEFAULTS = {
    DEFAULT_DOMAIN: {
        "content_height": 540,
        "content_width": 640,
        "raw_action_dim": 8,
        "use_state": True,
    },
}
_POLICY_KEYS = {
    "action_chunk_size": "action_chunk_size",
    "conditioning_fps": "fps",
    "domain_id": "domain_id",
    "domain_name": "domain_name",
    "raw_action_dim": "raw_action_dim",
    "use_state": "use_state",
}


def _load_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path) as config_file:
        return json.load(config_file)


def load_policy_metadata(model_dir: str) -> dict | None:
    """Return checkpoint.json's policy object, or None if the checkpoint has none."""
    checkpoint = _load_json(os.path.join(model_dir, "checkpoint.json")) or {}
    policy = checkpoint.get("policy") or None
    if policy is not None and not isinstance(policy, dict):
        raise ValueError("checkpoint.json 'policy' must be an object")
    return policy


@dataclass(frozen=True)
class Cosmos3PolicyGeometry:
    """Static request geometry owned by one Cosmos3 policy engine bundle."""

    action_chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE
    conditioning_frames: int = DEFAULT_NUM_FRAMES
    content_height: int = DEFAULT_HEIGHT
    content_width: int = DEFAULT_WIDTH
    domain_id: int = DEFAULT_DOMAIN_ID
    # None when the checkpoint selects a domain id without naming it.
    domain_name: str | None = DEFAULT_DOMAIN
    fps: float = DEFAULT_FPS
    height: int = DEFAULT_HEIGHT
    max_und_len: int = DEFAULT_MAX_UND_LEN
    num_frames: int = DEFAULT_NUM_FRAMES
    raw_action_dim: int = DEFAULT_RAW_ACTION_DIM
    use_state: bool = False
    width: int = DEFAULT_WIDTH

    @property
    def state_rows(self) -> int:
        return 1 if self.use_state else 0

    @property
    def action_token_count(self) -> int:
        return self.action_chunk_size + self.state_rows

    @property
    def latent_t(self) -> int:
        return (self.num_frames - 1) // TEMPORAL_COMPRESSION_FACTOR + 1

    @property
    def latent_h(self) -> int:
        return self.content_height // VAE_SPATIAL_DOWNSAMPLE_FACTOR

    @property
    def latent_w(self) -> int:
        return self.content_width // VAE_SPATIAL_DOWNSAMPLE_FACTOR

    @classmethod
    def resolve(cls,
                policy: Mapping | None,
                overrides: Mapping | None = None,
                transformer: Mapping | None = None) -> "Cosmos3PolicyGeometry":
        """Resolve each field as override > checkpoint > default; ``policy`` is None without metadata."""
        checkpoint = {}
        if transformer and "domain_id" in transformer:
            checkpoint["domain_id"] = transformer["domain_id"]
        if policy is not None:
            checkpoint.update(
                _DOMAIN_DEFAULTS.get(policy.get("domain_name"), {}))
            checkpoint.update({
                field: policy[key]
                for key, field in _POLICY_KEYS.items() if key in policy
            })
        overrides = {
            name: value
            for name, value in (overrides or {}).items() if value is not None
        }

        def pick(name: str, default):
            return overrides.get(name, checkpoint.get(name, default))

        action_chunk_size = int(
            pick("action_chunk_size", DEFAULT_ACTION_CHUNK_SIZE))
        height = int(pick("height", DEFAULT_HEIGHT))
        width = int(pick("width", DEFAULT_WIDTH))
        domain_id = pick("domain_id", None)
        domain_name = pick("domain_name", None)
        # The default name is only valid alongside the default id.
        if domain_id is None:
            domain_id = DEFAULT_DOMAIN_ID
            if domain_name is None:
                domain_name = DEFAULT_DOMAIN
        if policy is None:
            conditioning_frames = num_frames = DEFAULT_NUM_FRAMES
        else:
            # Policy checkpoints condition on one frame and roll out one frame
            # per action step after it.
            conditioning_frames, num_frames = 1, action_chunk_size + 1
        geometry = cls(
            action_chunk_size=action_chunk_size,
            conditioning_frames=int(
                pick("conditioning_frames", conditioning_frames)),
            content_height=int(pick("content_height", height)),
            content_width=int(pick("content_width", width)),
            domain_id=int(domain_id),
            domain_name=domain_name,
            fps=float(pick("fps", DEFAULT_FPS)),
            height=height,
            max_und_len=int(pick("max_und_len", DEFAULT_MAX_UND_LEN)),
            num_frames=int(pick("num_frames", num_frames)),
            raw_action_dim=int(pick("raw_action_dim", DEFAULT_RAW_ACTION_DIM)),
            use_state=bool(pick("use_state", False)),
            width=width,
        )
        geometry.validate()
        return geometry

    @classmethod
    def from_bundle(cls, bundle, args=None) -> "Cosmos3PolicyGeometry":
        """Resolve geometry for the checkpoint-direct builder."""
        root = bundle.root
        overrides = dict(root.get("edge_llm_builder") or {})
        # A max_und_len pinned in config.json wins over --max-input-len.
        if args is not None:
            overrides.setdefault("max_und_len", args.max_input_len)
        return cls.resolve(root.get("_direct_policy_config"), overrides,
                           root.get("_direct_transformer_config"))

    @classmethod
    def from_checkpoint(
            cls,
            model_dir: str,
            overrides: Mapping | None = None) -> "Cosmos3PolicyGeometry":
        """Resolve geometry for ONNX export from a checkpoint directory."""
        transformer = _load_json(
            os.path.join(model_dir, "transformer", "config.json"))
        return cls.resolve(load_policy_metadata(model_dir), overrides,
                           transformer)

    def validate(self) -> None:
        if self.action_chunk_size <= 0:
            raise ValueError("Cosmos3 action_chunk_size must be positive")
        if self.raw_action_dim <= 0:
            raise ValueError("Cosmos3 raw_action_dim must be positive")
        for name in ("conditioning_frames", "num_frames"):
            frames = getattr(self, name)
            if frames <= 0 or (frames - 1) % TEMPORAL_COMPRESSION_FACTOR:
                raise ValueError(f"Cosmos3 {name} must have the form "
                                 f"{TEMPORAL_COMPRESSION_FACTOR}k+1")
        if self.max_und_len <= 0:
            raise ValueError("Cosmos3 max_und_len must be positive")
        if (self.height <= 0 or self.width <= 0 or self.content_height <= 0
                or self.content_width <= 0):
            raise ValueError("Cosmos3 image dimensions must be positive")
        if (self.content_height > self.height
                or self.content_width > self.width):
            raise ValueError(
                "Cosmos3 content dimensions must fit inside the image canvas")
        if (self.height % VAE_SPATIAL_DOWNSAMPLE_FACTOR
                or self.width % VAE_SPATIAL_DOWNSAMPLE_FACTOR):
            raise ValueError(
                "Cosmos3 image canvas dimensions must be divisible by the "
                f"VAE downsample factor ({VAE_SPATIAL_DOWNSAMPLE_FACTOR})")
        if (self.content_height % VAE_PATCH_SIZE
                or self.content_width % VAE_PATCH_SIZE):
            raise ValueError("Cosmos3 content dimensions must be divisible by "
                             f"the VAE patch size ({VAE_PATCH_SIZE})")
