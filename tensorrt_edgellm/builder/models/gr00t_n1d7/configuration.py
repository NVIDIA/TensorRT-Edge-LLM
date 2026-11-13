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
"""GR00T N1.7 component configuration for the direct builder. The checkpoint facts live in
:mod:`tensorrt_edgellm.models.gr00t_n1d7.configuration`, shared with the ONNX export; this
module binds them to the builder's bundle and profiles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Tuple

from tensorrt_edgellm.models.gr00t_n1d7 import configuration as facts
from tensorrt_edgellm.models.gr00t_n1d7 import identity

from ...core import contracts, weight_policy
from . import runtime_config

if TYPE_CHECKING:
    from tensorrt_edgellm.models.gr00t_n1d7.geometry import Geometry

    from ...core.builder import BuildArgs


def _component_name(component) -> str:
    return str(getattr(component, "value", component)).replace("-", "_")


def prepare_root(model_dir: str, root: dict) -> dict:
    """Attach the backbone configs the GR00T checkpoint does not carry."""
    shapes = facts.checkpoint_shapes(model_dir)
    facts.validate_backbone(shapes)
    prepared = dict(root)
    prepared["text_config"] = facts.text_config(shapes, root)
    prepared["vision_config"] = dict(facts.VISION_CONFIG)
    prepared["image_token_id"] = facts.IMAGE_TOKEN_ID
    prepared["vision_start_token_id"] = facts.VISION_START_TOKEN_ID
    prepared["vision_end_token_id"] = facts.VISION_END_TOKEN_ID
    prepared["_gr00t_dit_inner_dim"] = facts.dit_inner_dim(shapes)
    prepared["_gr00t_embodiment_id"] = facts.resolve_embodiment(model_dir)
    return prepared


def component_weight_policy(
        args: BuildArgs,
        policy: weight_policy.WeightPolicy) -> weight_policy.WeightPolicy:
    """Keep the engines self-contained: the policy runtime binds no weight pages."""
    del args
    return policy.without(weight_policy.EXTERNAL_WEIGHT_KINDS)


def available_components(
    root: dict, registered: Iterable[contracts.Component]
) -> frozenset[contracts.Component]:
    """GR00T always exposes the three policy components."""
    available = {
        contracts.Component.VISUAL,
        contracts.Component.VL_BACKBONE,
        contracts.Component.ACTION,
    }
    return frozenset(available).intersection(registered)


def component_config(root: dict, component) -> dict:
    """Select one component's checkpoint configuration."""
    name = _component_name(component)
    if name == "visual":
        config = dict(root["vision_config"])
        config["image_token_id"] = root["image_token_id"]
        return config
    if name in ("vl_backbone", "action"):
        config = dict(root["text_config"])
        config["_policy"] = {
            key: value
            for key, value in root.items() if not key.startswith("_")
        }
        config["_gr00t_dit_inner_dim"] = root["_gr00t_dit_inner_dim"]
        config["_gr00t_embodiment_id"] = root["_gr00t_embodiment_id"]
        return config
    raise ValueError(f"GR00T N1.7 has no {name!r} component")


def validate_build(args, components: Iterable[contracts.Component]) -> None:
    """No speculative-decoding or tensor-parallel contract, a loadable tokenizer, and no
    build beside components of another checkpoint, option set or route; runs before any
    engine is written."""
    if args.spec_type != "none" or args.spec_role != "none":
        raise ValueError(
            "GR00T N1.7 components do not support speculative decoding")
    if int(getattr(args, "tp_size", 1)) != 1:
        raise ValueError("GR00T N1.7 components are single-rank only")
    facts.require_tokenizer(args.model_dir)
    from ...core.config import BundleConfig
    bundle = BundleConfig.from_pretrained(args.model_dir)
    identity.validate_existing(args.engine_dir,
                               runtime_config.bundle_identity(bundle, args),
                               [_component_name(c) for c in components])


def setup_profiles(builder, builder_config, network, args, bundle) -> bool:
    """Install the dynamic vl_backbone and action profiles. Every dynamic axis is the request
    batch (axis 0) or the VL context length; any other dynamic axis fails the build."""
    owned = {
        contracts.Component.VL_BACKBONE,
        contracts.Component.ACTION,
    }
    if args.resolved_component not in owned:
        return False

    from .runtime_config import geometry_from_bundle
    geometry = geometry_from_bundle(bundle, args)
    max_batch = int(args.max_batch_size)
    profile = builder.create_optimization_profile()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = tuple(int(dimension) for dimension in tensor.shape)
        if -1 not in shape:
            continue
        minimum, optimum, maximum = list(shape), list(shape), list(shape)
        for axis, dimension in enumerate(shape):
            if dimension != -1:
                continue
            bounds = _axis_bounds(axis, max_batch, geometry)
            if bounds is None:
                raise ValueError(f"GR00T {args.component} has unowned dynamic "
                                 f"axis {tensor.name}[{axis}]")
            minimum[axis], optimum[axis], maximum[axis] = bounds
        profile.set_shape(tensor.name, tuple(minimum), tuple(optimum),
                          tuple(maximum))
    builder_config.add_optimization_profile(profile)
    return True


def _axis_bounds(axis: int, max_batch: int,
                 geometry: Geometry) -> Optional[Tuple[int, int, int]]:
    """Axis 0 is the request batch; axis 1 (and 3, on the masks) is the VL context."""
    if axis == 0:
        return (1, max_batch, max_batch)
    if axis in (1, 3):
        span = geometry.max_vl_len
        return (1, max(1, span // 2), span)
    return None
