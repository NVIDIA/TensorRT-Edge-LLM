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
"""Runtime component configs for checkpoint-direct pi0.5 engines."""

from __future__ import annotations

from tensorrt_edgellm.models.pi05 import component_contracts

from ...core import contracts
from .configuration import Pi05Geometry


def component_runtime_config(bundle, component: contracts.Component, args):
    """Return the config consumed by the existing experimental C++ runtime."""
    geometry = Pi05Geometry.from_bundle(bundle, args)
    if component == contracts.Component.VISUAL:
        return component_contracts.make_visual_config(
            {
                "hidden_size": 1152,
                "projection_dim": 2048,
                "image_size": 224,
                "patch_size": 14,
                "num_positions": 256,
            },
            geometry.opt_views,
            geometry.max_views,
            include_onnx=False)
    if component == contracts.Component.PREFIX:
        return component_contracts.make_prefix_config(geometry.prefix,
                                                      geometry.max_prefix_len,
                                                      geometry.max_batch_size,
                                                      include_onnx=False)
    action = {
        **geometry.action,
        "action_dim": geometry.action_dim,
        "action_horizon": geometry.action_horizon,
    }
    if component == contracts.Component.ACTION:
        return component_contracts.make_action_config(
            action,
            geometry.max_prefix_len,
            geometry.max_batch_size,
            geometry.num_denoise_steps,
            hoist_cond=True,
            include_onnx=False)
    if component == contracts.Component.COND:
        return component_contracts.make_cond_config(action,
                                                    geometry.num_denoise_steps,
                                                    include_onnx=False)
    return None
