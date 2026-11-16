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
"""GR00T N1.7 runtime component contracts bound to the builder's bundle; the shapes live in
:mod:`tensorrt_edgellm.models.gr00t_n1d7.contracts` so the ONNX export writes the same config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from tensorrt_edgellm.models.gr00t_n1d7 import contracts as shared
from tensorrt_edgellm.models.gr00t_n1d7 import identity
from tensorrt_edgellm.models.gr00t_n1d7.geometry import (DEFAULT_EMBODIMENT_ID,
                                                         DEFAULT_MAX_IMAGE_LEN,
                                                         DEFAULT_MAX_VL_LEN,
                                                         Geometry)

from ...core import contracts

if TYPE_CHECKING:
    from ...core.builder import BuildArgs
    from ...core.bundle import BundleConfig

__all__ = [
    "Geometry", "geometry_from_bundle", "bundle_identity",
    "component_runtime_config"
]


def _component_name(component) -> str:
    return str(getattr(component, "value", component)).replace("-", "_")


def geometry_from_bundle(bundle: BundleConfig,
                         args: Optional[BuildArgs] = None) -> Geometry:
    root = bundle.root
    policy = root.get("_policy") or root
    overrides = policy.get("edge_llm_builder") or {}

    def value(name: str, default: object) -> object:
        argument = getattr(args, name, None) if args is not None else None
        if argument is not None:
            return argument
        return overrides.get(name, default)

    return Geometry.from_policy(
        policy,
        dit_inner_dim=int(root.get("_gr00t_dit_inner_dim", 0)),
        embodiment_id=int(
            value("gr00t_embodiment_id",
                  root.get("_gr00t_embodiment_id", DEFAULT_EMBODIMENT_ID))),
        max_vl_len=int(value("max_input_len", DEFAULT_MAX_VL_LEN)),
        max_image_len=int(value("gr00t_max_image_len", DEFAULT_MAX_IMAGE_LEN)),
    )


def bundle_identity(bundle: BundleConfig, args: Optional[BuildArgs]) -> dict:
    return identity.bundle_identity(bundle.model_dir,
                                    geometry_from_bundle(bundle, args),
                                    "direct")


def _num_deepstack(bundle: BundleConfig) -> int:
    return len(
        bundle.component_dict(
            contracts.Component.VISUAL).get("deepstack_visual_indexes") or [])


def component_runtime_config(bundle: BundleConfig,
                             component: contracts.Component,
                             args: Optional[BuildArgs]) -> dict:
    """Return the runtime config for one GR00T component, stamped with the bundle identity."""
    return identity.stamp(_component_runtime_config(bundle, component, args),
                          bundle_identity(bundle, args))


def _component_runtime_config(bundle: BundleConfig,
                              component: contracts.Component,
                              args: Optional[BuildArgs]) -> dict:
    name = _component_name(component)
    geometry = geometry_from_bundle(bundle, args)
    if name == "visual":
        return shared.visual_runtime_config(
            bundle.component_dict(contracts.Component.VISUAL),
            bundle.component_dict(contracts.Component.VL_BACKBONE),
            geometry,
            min_image_tokens=int(getattr(args, "min_image_tokens", 4) or 4),
            max_image_tokens=int(
                getattr(args, "max_image_tokens", 1024) or 1024),
            max_image_tokens_per_image=int(
                getattr(args, "max_image_tokens_per_image", 512) or 512))
    if name == "vl_backbone":
        return shared.vl_backbone_runtime_config(
            bundle.component_dict(contracts.Component.VL_BACKBONE),
            geometry,
            num_deepstack=_num_deepstack(bundle),
            max_batch=int(args.max_batch_size))
    if name == "action":
        return shared.action_runtime_config(geometry,
                                            max_batch=int(args.max_batch_size))
    raise ValueError(f"GR00T N1.7 has no {name!r} component")


def update_llm_config(config: dict, root: dict, cfg, args) -> None:
    """GR00T exposes no autoregressive LLM component."""
    del config, root, cfg, args
