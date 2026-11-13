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

from tensorrt_edgellm.models.cosmos3.policy_geometry import (
    Cosmos3PolicyGeometry, load_policy_metadata)

from ...core import contracts


def _component_name(component) -> str:
    return str(getattr(component, "value", component)).replace("-", "_")


def _load_json(model_dir: str, subdirectory: str) -> dict | None:
    path = os.path.join(model_dir, subdirectory, "config.json")
    if not os.path.isfile(path):
        return None
    with open(path) as config_file:
        return json.load(config_file)


def prepare_root(model_dir: str, root: dict) -> dict:
    """Attach Diffusers component configs without importing Diffusers."""
    prepared = dict(root)
    transformer = _load_json(model_dir, "transformer")
    vae = _load_json(model_dir, "vae")
    if transformer is not None:
        prepared["_direct_transformer_config"] = transformer
    if vae is not None:
        prepared["_direct_vae_config"] = vae
    policy = load_policy_metadata(model_dir)
    if policy is not None:
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
