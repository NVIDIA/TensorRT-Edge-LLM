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
"""Qwen3-VL checkpoint configuration."""

from ...core import contracts


def available_components(root: dict, registered):
    """Honor standalone Qwen3-VL component checkpoints."""
    architectures = set(root.get("architectures") or ())
    if "Qwen3VLVisionModel" in architectures:
        return frozenset((contracts.Component.VISUAL, ))
    return registered


def component_config(root: dict, component: contracts.Component) -> dict:
    if component == contracts.Component.LLM:
        return root
    if component == contracts.Component.VISUAL:
        return root.get("vision_config") or root
    raise ValueError(f"Qwen3-VL has no {component.value} configuration")


def prepare_text_config(config: dict, root: dict,
                        component: contracts.Component,
                        model_dir: str) -> dict:
    config = dict(config)
    visual = root.get("vision_config") or {}
    deepstack = visual.get("deepstack_visual_indexes")
    config.setdefault("num_deepstack_features",
                      len(deepstack) if isinstance(deepstack, list) else 3)
    return config
