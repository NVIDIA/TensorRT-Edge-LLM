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
"""Mistral checkpoint configuration."""

from ...core import contracts


def component_config(root: dict, component: contracts.Component) -> dict:
    if component == contracts.Component.LLM:
        return root
    raise ValueError(f"Mistral has no {component.value} configuration")


def prepare_text_config(config: dict, root: dict,
                        component: contracts.Component,
                        model_dir: str) -> dict:
    config = dict(config)
    if config.get("sliding_window") is not None:
        config["use_sliding_window"] = True
    return config
