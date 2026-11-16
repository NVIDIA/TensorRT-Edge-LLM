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
"""InternVL3 runtime configuration artifacts."""

from typing import Any, Dict

from ...core import contracts
from ...core.artifacts.tokenizer import find_token_id


def update_llm_config(config: Dict[str, Any], root: Dict[str, Any], cfg,
                      args) -> None:
    """Add the InternVL image placeholder contract to the LLM runtime."""
    del cfg
    image_token_id = root.get("image_token_id")
    if not isinstance(image_token_id, int):
        image_token_id = find_token_id(args.model_dir, "<IMG_CONTEXT>")
    if image_token_id is None:
        raise ValueError("InternVL tokenizer does not define <IMG_CONTEXT>")
    config["image_token_id"] = image_token_id


def component_runtime_config(bundle, component: contracts.Component, args):
    """Return InternVL3 visual runtime config."""
    if component != contracts.Component.VISUAL:
        return None
    visual = dict(bundle.component_dict(component))
    text = dict(
        bundle.root.get("text_config") or bundle.root.get("llm_config") or {})
    if not text:
        raise ValueError("InternVL runtime requires text_config or llm_config")
    visual["model_type"] = "internvl"
    result: Dict[str, Any] = {
        "model_type": "internvl",
        "text_config": text,
        "vision_config": visual,
        "builder_config": {
            "min_image_tokens": args.min_image_tokens,
            "max_image_tokens": args.max_image_tokens,
            "max_image_tokens_per_image": args.max_image_tokens_per_image,
        },
    }
    image_token_id = bundle.root.get("image_token_id")
    if not isinstance(image_token_id, int):
        image_token_id = find_token_id(bundle.model_dir, "<IMG_CONTEXT>")
    if image_token_id is not None:
        result["image_token_id"] = image_token_id
    img_start_token_id = find_token_id(bundle.model_dir, "<img>")
    if img_start_token_id is not None:
        result["img_start_token_id"] = img_start_token_id
    img_end_token_id = find_token_id(bundle.model_dir, "</img>")
    if img_end_token_id is not None:
        result["img_end_token_id"] = img_end_token_id
    for key in ("patch_size", "image_size"):
        if isinstance(visual.get(key), int):
            visual[key] = [visual[key], visual[key]]
    return result
