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
"""Alpamayo checkpoint and base-VLM configuration."""

import json
import os

from ...core import contracts, quantization
from ...core.weights import Weights
from . import weights as weight_conversion

DEFAULT_VLM = "Qwen/Qwen3-VL-8B-Instruct"


def prefix_kv_scales(bundle, num_layers: int):
    """Read the LLM prefix precision and per-layer dequantization scales."""
    quant = quantization.parse_quantization(
        bundle.model_dir, bundle.root,
        bundle.component_dict(contracts.Component.LLM), weight_conversion)
    if quant.kv_cache_quant != "fp8":
        return None
    weights = Weights(bundle.model_dir,
                      component="llm",
                      conversion=weight_conversion)
    try:
        return [
            list(weights.qkv_scales(f"model.layers.{index}.self_attn")[1:])
            for index in range(num_layers)
        ]
    finally:
        weights.close()


def vlm_reference(model_dir: str, root: dict) -> str:
    reference = root.get("vlm_name_or_path")
    if isinstance(reference, str) and reference:
        return reference
    local_reference = os.path.join(model_dir, "base_vlm")
    return local_reference if os.path.isdir(local_reference) else DEFAULT_VLM


def prepare_root(model_dir: str, root: dict) -> dict:
    root = dict(root)
    embedded = root.get("vlm_config") or root.get("vlm_cfg")
    if isinstance(embedded, dict):
        root["_direct_vlm_config"] = dict(embedded)
        return root

    candidates = [
        os.path.join(model_dir, "vlm_config.json"),
        os.path.join(model_dir, "config_vlm.json"),
    ]
    reference = vlm_reference(model_dir, root)
    if os.path.isdir(reference):
        candidates.append(os.path.join(reference, "config.json"))
    for path in candidates:
        if os.path.isfile(path):
            with open(path) as config_file:
                root["_direct_vlm_config"] = json.load(config_file)
            return root

    try:
        import huggingface_hub
    except ImportError as error:
        raise RuntimeError(
            "Alpamayo requires its base VLM config; install huggingface_hub "
            "or place vlm_config.json in the checkpoint directory") from error
    path = huggingface_hub.hf_hub_download(reference, "config.json")
    with open(path) as config_file:
        root["_direct_vlm_config"] = json.load(config_file)
    return root


def component_config(root: dict, component: contracts.Component) -> dict:
    supplemental = root.get("_direct_vlm_config") or {}
    if component == contracts.Component.LLM:
        return root
    if component == contracts.Component.VISUAL:
        return supplemental.get("vision_config") or root
    if component == contracts.Component.ACTION:
        return root.get("action_config") or root.get("expert_cfg") or root
    raise ValueError(f"Alpamayo has no {component.value} configuration")


def prepare_text_config(config: dict, root: dict,
                        component: contracts.Component,
                        model_dir: str) -> dict:
    config = dict(config)
    supplemental = root.get("_direct_vlm_config") or {}
    visual = supplemental.get("vision_config") or {}
    deepstack = visual.get("deepstack_visual_indexes")
    config.setdefault("num_deepstack_features",
                      len(deepstack) if isinstance(deepstack, list) else 3)
    return config


def validate_build(args, components) -> None:
    """Reject speculative contracts that Alpamayo's runtime cannot consume."""
    if args.spec_type != "none" or args.spec_role != "none":
        raise ValueError("Alpamayo does not support speculative decoding")
