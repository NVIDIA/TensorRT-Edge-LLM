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
"""DiffusionGemma component and text-backbone configuration."""

import math

from ...core import contracts


def available_components(root: dict, registered):
    available = set(registered)
    if not isinstance(root.get("vision_config"), dict):
        available.discard(contracts.Component.VISUAL)
    return frozenset(available)


def component_config(root: dict, component: contracts.Component) -> dict:
    if component == contracts.Component.DLLM:
        return root.get("text_config") or root
    if component == contracts.Component.VISUAL:
        return root.get("vision_config") or root
    raise ValueError(f"DiffusionGemma has no {component.value} configuration")


def prepare_text_config(config: dict, root: dict,
                        component: contracts.Component,
                        model_dir: str) -> dict:
    del component, model_dir
    config = dict(config)
    hidden_size = int(config["hidden_size"])
    config.setdefault("model_type", "diffusion_gemma_text")
    config.setdefault("partial_rotary_factor", 0.25)
    config.setdefault("attention_scaling", 1.0)
    config.setdefault("embedding_scale", math.sqrt(float(hidden_size)))
    config.setdefault("has_value_norm", True)
    config.setdefault("attention_k_eq_v", True)
    config.setdefault("enable_moe_block", True)
    config.setdefault(
        "self_conditioning_size",
        root.get("self_conditioning_size", config.get("intermediate_size", 0)),
    )
    if config.get("sliding_window") is not None:
        config["use_sliding_window"] = True
    return config


def update_device_config(config, root: dict,
                         component: contracts.Component) -> None:
    if component != contracts.Component.DLLM:
        return
    config.model_type = "diffusion_gemma_text"
    config.tie_word_embeddings = True
    config.attention_k_eq_v = True
    config.has_value_norm = True
    config.enable_moe_block = config.num_experts > 0


def validate_build(args, components) -> None:
    del components
    if args.tp_size != 1:
        raise ValueError("DiffusionGemma currently requires --tp-size 1")
    if args.spec_type != "none":
        raise ValueError(
            "DiffusionGemma does not use autoregressive speculative decoding")
