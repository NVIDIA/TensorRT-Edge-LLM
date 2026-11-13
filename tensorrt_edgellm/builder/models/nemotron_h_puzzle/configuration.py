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
"""Nemotron-H Puzzle configuration for checkpoint-direct builds."""

from ...core import contracts
from ..nemotron_h.configuration import \
    prepare_text_config as _prepare_text_config
from ..nemotron_omni import configuration as omni_configuration


def available_components(root: dict, registered):
    available = {contracts.Component.LLM}
    if root.get("vision_config"):
        available.add(contracts.Component.VISUAL)
    if root.get("sound_config") or root.get("audio_config"):
        available.add(contracts.Component.AUDIO)
    return available & registered


def component_config(root: dict, component: contracts.Component) -> dict:
    if component != contracts.Component.LLM:
        return omni_configuration.component_config(root, component)
    return root["llm_config"]


def prepare_text_config(config: dict, root: dict,
                        component: contracts.Component,
                        model_dir: str) -> dict:
    return _prepare_text_config(config, root, component, model_dir)


def setup_profiles(builder, builder_config, network, args, bundle):
    return omni_configuration.setup_profiles(builder, builder_config, network,
                                             args, bundle)
