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
"""Phi-4 Multimodal checkpoint configuration."""

from ...core import contracts

_SIGLIP_VISION_DEFAULTS = {
    "hidden_size": 1152,
    "image_size": 448,
    "intermediate_size": 4304,
    "model_type": "siglip_vision_model",
    "num_attention_heads": 16,
    "num_hidden_layers": 27,
    "patch_size": 14,
}


def component_config(root: dict, component: contracts.Component) -> dict:
    if component == contracts.Component.LLM:
        return root
    if component == contracts.Component.VISUAL:
        visual = dict(_SIGLIP_VISION_DEFAULTS)
        visual.update(root.get("vision_config") or {})
        processor = root.get("img_processor")
        visual["feature_layer"] = (processor.get("layer_idx", -2)
                                   if isinstance(processor, dict) else -2)
        return visual
    raise ValueError(f"Phi-4-MM has no {component.value} configuration")


def prepare_text_config(config: dict, root: dict,
                        component: contracts.Component,
                        model_dir: str) -> dict:
    config = dict(config)
    if config.get("sliding_window") is not None:
        config["use_sliding_window"] = True
    return config
