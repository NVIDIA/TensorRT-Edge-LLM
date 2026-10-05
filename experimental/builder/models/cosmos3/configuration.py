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
"""Cosmos3 policy checkpoint and component configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from ...core import contracts

DEFAULT_ACTION_CHUNK_SIZE = 16
DEFAULT_DOMAIN_ID = 8
DEFAULT_FPS = 5.0
DEFAULT_HEIGHT = 544
DEFAULT_MAX_UND_LEN = 512
DEFAULT_NUM_FRAMES = 17
DEFAULT_WIDTH = 736
DROID_CONTENT_HEIGHT = 540
DROID_CONTENT_WIDTH = 640
VAE_SPATIAL_DOWNSAMPLE_FACTOR = 16


def _component_name(component) -> str:
    return str(getattr(component, "value", component)).replace("-", "_")


def _load_json(model_dir: str, subdirectory: str) -> dict | None:
    path = os.path.join(model_dir, subdirectory, "config.json")
    if not os.path.isfile(path):
        return None
    with open(path) as config_file:
        return json.load(config_file)


def _load_policy_metadata(model_dir: str) -> dict:
    path = os.path.join(model_dir, "checkpoint.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as config_file:
        policy = json.load(config_file).get("policy") or {}
    if not isinstance(policy, dict):
        raise ValueError("checkpoint.json 'policy' must be an object")
    result = dict(policy)
    if result.get("domain_name") == "droid_lerobot":
        result.setdefault("raw_action_dim", 8)
        result.setdefault("use_state", True)
    return result


def prepare_root(model_dir: str, root: dict) -> dict:
    """Attach Diffusers component configs without importing Diffusers."""
    prepared = dict(root)
    transformer = _load_json(model_dir, "transformer")
    vae = _load_json(model_dir, "vae")
    if transformer is not None:
        prepared["_direct_transformer_config"] = transformer
    if vae is not None:
        prepared["_direct_vae_config"] = vae
    policy = _load_policy_metadata(model_dir)
    if policy:
        prepared["_direct_policy_config"] = policy
    return prepared


def available_components(root: dict, registered):
    """Return only components physically present in this Cosmos3 bundle."""
    available = set()
    if (isinstance(root.get("text_config"), dict)
            or root.get("model_type") == "cosmos3_edge_text"):
        available.add(contracts.Component.LLM)
    if isinstance(root.get("vision_config"), dict):
        available.add(contracts.Component.VISUAL)
    if isinstance(root.get("_direct_transformer_config"), dict):
        available.update(
            (contracts.Component.UND_PREFILL, contracts.Component.GEN))
    if isinstance(root.get("_direct_vae_config"), dict):
        available.add(contracts.Component.VAE_ENCODER)
    return frozenset(available).intersection(registered)


def component_config(root: dict, component) -> dict:
    """Select one policy component's checkpoint configuration."""
    name = _component_name(component)
    if name == "llm":
        return dict(root.get("text_config") or root)
    if name == "visual":
        return dict(root.get("vision_config") or root)
    if name in ("gen", "und_prefill"):
        return dict(root["_direct_transformer_config"])
    if name == "vae_encoder":
        return dict(root["_direct_vae_config"])
    raise ValueError(f"Cosmos3 policy has no {name!r} component")


def prepare_text_config(config: dict, root: dict, component,
                        model_dir: str) -> dict:
    """Normalize the provider's flat Cosmos3 reasoning decoder metadata."""
    del root, component, model_dir
    prepared = dict(config)
    prepared.setdefault("hidden_act", "relu2")
    if prepared["hidden_act"] != "relu2":
        raise ValueError(
            "Cosmos3 reasoner text requires the relu2 feed-forward block")
    prepared.setdefault("qk_norm_for_text", False)
    prepared["rotary_dim_override"] = int(
        prepared.get(
            "head_dim",
            prepared["hidden_size"] // prepared["num_attention_heads"]))
    return prepared


def validate_build(args, components) -> None:
    """Cosmos policy components have no speculative-decoding contract."""
    if args.spec_type != "none" or args.spec_role != "none":
        raise ValueError(
            "Cosmos3 policy components do not support speculative decoding")


def setup_profiles(builder, builder_config, network, args, bundle) -> bool:
    """Install the exact dynamic policy-component profiles."""
    policy_components = {
        contracts.Component.UND_PREFILL,
        contracts.Component.GEN,
        contracts.Component.VAE_ENCODER,
    }
    if args.resolved_component not in policy_components:
        return False

    geometry = Cosmos3PolicyGeometry.from_bundle(bundle, args)
    max_batch = int(args.max_batch_size)
    max_und = geometry.max_und_len
    profile = builder.create_optimization_profile()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = tuple(int(dimension) for dimension in tensor.shape)
        if -1 not in shape:
            continue
        minimum = list(shape)
        optimum = list(shape)
        maximum = list(shape)
        for axis, dimension in enumerate(shape):
            if dimension != -1:
                continue
            if axis == 0:
                bounds = (1, max_batch, max_batch)
            elif (axis == 1 and
                  (args.resolved_component == contracts.Component.UND_PREFILL
                   or tensor.name.startswith(("und_k_", "und_v_")))):
                bounds = (1, max(1, max_und // 2), max_und)
            else:
                raise ValueError(
                    f"Cosmos3 {args.component} has unowned dynamic axis "
                    f"{tensor.name}[{axis}]")
            minimum[axis], optimum[axis], maximum[axis] = bounds
        profile.set_shape(tensor.name, tuple(minimum), tuple(optimum),
                          tuple(maximum))
    builder_config.add_optimization_profile(profile)
    return True


@dataclass(frozen=True)
class Cosmos3PolicyGeometry:
    """Static request geometry owned by one Cosmos3 policy engine bundle."""

    action_chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE
    conditioning_frames: int = DEFAULT_NUM_FRAMES
    content_height: int = DEFAULT_HEIGHT
    content_width: int = DEFAULT_WIDTH
    domain_id: int = DEFAULT_DOMAIN_ID
    fps: float = DEFAULT_FPS
    height: int = DEFAULT_HEIGHT
    max_und_len: int = DEFAULT_MAX_UND_LEN
    num_frames: int = DEFAULT_NUM_FRAMES
    raw_action_dim: int = 10
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
        return (self.num_frames - 1) // 4 + 1

    @property
    def latent_h(self) -> int:
        return self.content_height // 16

    @property
    def latent_w(self) -> int:
        return self.content_width // 16

    @classmethod
    def from_bundle(cls, bundle, args=None) -> "Cosmos3PolicyGeometry":
        overrides = bundle.root.get("edge_llm_builder") or {}
        policy = bundle.root.get("_direct_policy_config") or {}

        def value(name: str, default):
            argument = getattr(args, name, None) if args is not None else None
            return overrides.get(name,
                                 default) if argument is None else argument

        action_chunk_size = int(
            value("action_chunk_size",
                  policy.get("action_chunk_size", DEFAULT_ACTION_CHUNK_SIZE)))
        default_num_frames = (action_chunk_size +
                              1 if policy else DEFAULT_NUM_FRAMES)
        domain_name = str(policy.get("domain_name", ""))
        height = int(value("height", DEFAULT_HEIGHT))
        width = int(value("width", DEFAULT_WIDTH))
        geometry = cls(
            action_chunk_size=action_chunk_size,
            conditioning_frames=int(
                value("conditioning_frames",
                      1 if policy else DEFAULT_NUM_FRAMES)),
            content_height=int(
                value(
                    "content_height", DROID_CONTENT_HEIGHT
                    if domain_name == "droid_lerobot" else height)),
            content_width=int(
                value(
                    "content_width", DROID_CONTENT_WIDTH
                    if domain_name == "droid_lerobot" else width)),
            domain_id=int(value("domain_id", DEFAULT_DOMAIN_ID)),
            fps=float(value("fps", policy.get("conditioning_fps",
                                              DEFAULT_FPS))),
            height=height,
            max_und_len=int(
                value("max_und_len",
                      getattr(args, "max_input_len", DEFAULT_MAX_UND_LEN))),
            num_frames=int(value("num_frames", default_num_frames)),
            raw_action_dim=int(
                value(
                    "raw_action_dim",
                    policy.get("raw_action_dim",
                               8 if domain_name == "droid_lerobot" else 10))),
            use_state=bool(
                value("use_state",
                      policy.get("use_state",
                                 domain_name == "droid_lerobot"))),
            width=width,
        )
        geometry.validate()
        return geometry

    def validate(self) -> None:
        if self.action_chunk_size <= 0:
            raise ValueError("Cosmos3 action_chunk_size must be positive")
        if self.raw_action_dim <= 0:
            raise ValueError("Cosmos3 raw_action_dim must be positive")
        if (self.conditioning_frames <= 0
                or (self.conditioning_frames - 1) % 4):
            raise ValueError(
                "Cosmos3 conditioning_frames must have the form 4k+1")
        if self.max_und_len <= 0:
            raise ValueError("Cosmos3 max_und_len must be positive")
        if self.num_frames <= 0 or (self.num_frames - 1) % 4:
            raise ValueError("Cosmos3 num_frames must have the form 4k+1")
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
        if self.content_height % 2 or self.content_width % 2:
            raise ValueError(
                "Cosmos3 content dimensions must be divisible by the VAE patch size"
            )
