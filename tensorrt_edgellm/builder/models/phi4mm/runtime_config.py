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
"""Phi-4-MM runtime configuration artifacts."""

from typing import Any, Dict

from ...core import contracts
from ...core.artifacts.tokenizer import find_token_id


def update_llm_config(config: Dict[str, Any], root: Dict[str, Any], cfg,
                      args) -> None:
    """Add the Phi-4-MM visual placeholder contract to the LLM runtime."""
    image_token_id = find_token_id(args.model_dir, "<|endoftext10|>")
    if image_token_id is None:
        raise ValueError("Phi-4-MM tokenizer does not define <|endoftext10|>")
    config["image_token_id"] = image_token_id


def component_runtime_config(bundle, component: contracts.Component, args):
    """Return Phi-4-MM visual runtime config."""
    if component != contracts.Component.VISUAL:
        return None
    root = bundle.root
    visual = dict(bundle.component_dict(component))
    runtime_type = "phi4mm"
    visual["model_type"] = runtime_type
    text = root.get("text_config") or {}
    vocab_size = root.get("vocab_size", text.get("vocab_size"))
    if vocab_size is None:
        raise ValueError("Phi-4-MM runtime requires vocab_size")

    embd_layer = dict(root.get("embd_layer") or {})
    image_embd_layer = dict(embd_layer.get("image_embd_layer") or {})
    if "crop_size" not in image_embd_layer:
        crop_size = visual.get("crop_size", visual.get("image_size", 448))
        if isinstance(crop_size, (list, tuple)):
            crop_size = crop_size[0]
        image_embd_layer["crop_size"] = int(crop_size)
    embd_layer["image_embd_layer"] = image_embd_layer

    result: Dict[str, Any] = {
        "model_type": runtime_type,
        "vocab_size": int(vocab_size),
        "embd_layer": embd_layer,
        "vision_config": visual,
        "builder_config": {
            "min_image_tokens": args.min_image_tokens,
            "max_image_tokens": args.max_image_tokens,
            "max_image_tokens_per_image": args.max_image_tokens_per_image,
        },
    }
    return result
