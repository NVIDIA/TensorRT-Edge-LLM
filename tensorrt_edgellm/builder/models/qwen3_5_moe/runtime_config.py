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
"""Qwen3.5-MoE runtime configuration artifacts."""

from typing import Any, Dict

from ...core import contracts
from ...core.artifacts.runtime_config import normalize_rope_scaling


def component_runtime_config(bundle, component: contracts.Component, args):
    """Return Qwen3.5-MoE component runtime config."""
    if component != contracts.Component.VISUAL:
        return None
    root = bundle.root
    visual = dict(bundle.component_dict(component))
    visual["model_type"] = "qwen3_5"
    text = dict(root.get("text_config") or {})
    result: Dict[str, Any] = {
        "model_type": "qwen3_5",
        "vision_config": visual,
    }
    for key in ("vision_start_token_id", "vision_end_token_id",
                "image_token_id", "video_token_id", "vocab_size",
                "rope_theta"):
        for source in (root, text):
            if key in source:
                result[key] = source[key]
                break

    rope = (text.get("rope_scaling") or text.get("rope_parameters")
            or root.get("rope_scaling") or root.get("rope_parameters"))
    if rope:
        normalized = normalize_rope_scaling(rope)
        result["rope_scaling"] = normalized
        text["rope_scaling"] = normalized
        if "rope_theta" in normalized:
            text.setdefault("rope_theta", normalized["rope_theta"])
    for key in ("vocab_size", "rope_theta"):
        if key in result:
            text.setdefault(key, result[key])
    if text:
        result["text_config"] = text
    result["builder_config"] = {
        "min_image_tokens": args.min_image_tokens,
        "max_image_tokens": args.max_image_tokens,
        "max_image_tokens_per_image": args.max_image_tokens_per_image,
    }
    return result
