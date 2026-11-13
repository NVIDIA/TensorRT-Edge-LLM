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
"""Phi-4 multimodal checkpoint weight mapping."""

import json
import os
from functools import lru_cache

import numpy as np

_PREFIXES = {
    "llm": ("thinker.", "language_model.", "model.language_model.",
            "vlm.model.language_model.", "vlm."),
    "visual": ("thinker.visual.", "visual.", "vision_tower.", "model.visual."),
}
_WRAPPERS = (
    "model.language_model.",
    "thinker.model.",
    "language_model.",
    "text_model.",
    "llm.",
    "thinker.",
)


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    """Map frontend tensor names to Phi-4 multimodal component checkpoints."""
    del spec_type, spec_role
    prefixes = _PREFIXES.get(component, ())
    names = [name]
    if component == "llm" and name.endswith((
            ".weight",
            ".weight_packed",
            ".weight_scale",
            ".weight_scale_2",
            ".weight_global_scale",
            ".input_scale",
            ".input_global_scale",
            ".bias",
    )):
        stem, suffix = name.rsplit(".", 1)
        if stem.endswith(("qkv_proj", "o_proj", "gate_up_proj", "down_proj")):
            names.append(f"{stem}.base_layer.{suffix}")
    candidates = [
        prefix + candidate for candidate in names for prefix in prefixes
    ]
    if component == "llm" and name.startswith("model."):
        candidates.extend(prefix + candidate[len("model."):]
                          for candidate in names for prefix in prefixes
                          if candidate.startswith("model."))
    candidates.extend(candidate for candidate in names if candidate != name)
    if name == "lm_head.weight" and quant_type == "fp16":
        candidates.extend(("model.embed_tokens.weight",
                           "model.language_model.embed_tokens.weight"))
    return tuple(candidates)


@lru_cache(maxsize=None)
def _vision_lora_scale(model_dir: str) -> float:
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    adapter = config.get("vision_lora") or {}
    rank = int(adapter.get("r", 0))
    if rank <= 0:
        raise ValueError("Phi-4-MM vision_lora.r must be positive")
    return float(adapter.get("lora_alpha", rank)) / rank


def linear_adapter(weights, prefix: str):
    """Return Phi-4-MM's built-in vision adapter for one decoder linear."""
    if weights.component != "llm":
        return None
    adapter_a_name = prefix + ".lora_A.vision.weight"
    adapter_b_name = prefix + ".lora_B.vision.weight"
    if not weights.has(adapter_a_name) or not weights.has(adapter_b_name):
        return None
    return (np.ascontiguousarray(weights.f32(adapter_a_name)),
            np.ascontiguousarray(weights.f32(adapter_b_name)),
            _vision_lora_scale(weights.store.model_dir))


def normalize_checkpoint_name(name: str) -> str:
    """Remove only wrappers used by this model family's checkpoints."""
    for prefix in _WRAPPERS:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name
