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
"""DiffusionGemma runtime configuration artifacts."""

import json
import os

from ...core import contracts, ragged
from ...core.artifacts.tokenizer import find_token_id


def _generation_config(model_dir: str) -> dict:
    path = os.path.join(model_dir, "generation_config.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as config_file:
        return json.load(config_file)


def _diffusion_config(root: dict, generation: dict) -> dict:
    sampler = generation.get("sampler_config") or {}
    sampler_name = str(sampler.get("_cls_name", "EntropyBound"))
    normalized_sampler = sampler_name.lower().replace("_", "")
    sampler_type = ("entropy_bound" if "entropybound" in normalized_sampler
                    else sampler_name.lower())
    return {
        "diffusion_family":
        "uniform_renoise",
        "canvas_length":
        ragged.diffusion_canvas_length(root, generation_config=generation),
        "max_denoising_steps":
        int(generation.get("max_denoising_steps", 48)),
        "t_max":
        float(generation.get("t_max", 0.8)),
        "t_min":
        float(generation.get("t_min", 0.4)),
        "sampler_type":
        sampler_type,
        "entropy_bound":
        float(
            sampler.get("entropy_bound", generation.get("entropy_bound",
                                                        0.1))),
        "entropy_threshold":
        float(
            generation.get("entropy_threshold",
                           generation.get("confidence_threshold", 0.005))),
        "stability_window":
        int(
            generation.get("stability_window",
                           generation.get("stability_threshold", 2))),
        "self_conditioning_enabled":
        True,
        "self_conditioning_repr":
        "embeds",
        "supported_modalities": ["text"],
    }


def update_llm_config(runtime: dict, root: dict, cfg, args) -> None:
    generation = _generation_config(args.model_dir)
    runtime.update({
        "model":
        "diffusion_gemma_text",
        "engine_role":
        "dllm",
        "decoding_strategy":
        "block_diffusion",
        "rms_norm_eps":
        float(cfg.rms_norm_eps),
        "attention_scaling":
        float(cfg.attention_scaling),
        "attention_layer_types":
        list(cfg.attention_layer_types),
        "sliding_window_size":
        int(cfg.sliding_window_size),
        "attention_k_eq_v":
        True,
        "num_global_key_value_heads":
        int(cfg.num_global_key_value_heads),
        "num_experts":
        int(cfg.num_experts),
        "num_experts_per_tok":
        int(cfg.num_experts_per_tok),
        "moe_intermediate_size":
        int(cfg.moe_intermediate_size),
        "enable_moe_block":
        bool(cfg.enable_moe_block),
        "self_conditioning_size":
        int(cfg.self_conditioning_size),
        "diffusion_config":
        _diffusion_config(root, generation),
        "diffusion_unified_conditioning":
        True,
        "context_mask_selector_enabled":
        True,
        "diffusion_engines": {
            "dllm": {
                "path": "dllm.engine",
                "role": "dllm",
            },
        },
    })
    if cfg.final_logit_softcapping is not None:
        runtime["final_logit_softcapping"] = float(cfg.final_logit_softcapping)


def component_runtime_config(bundle, component: contracts.Component, args):
    if component != contracts.Component.VISUAL:
        return None
    root = bundle.root
    visual = dict(bundle.component_dict(component))
    visual["model_type"] = "gemma4_vision"
    result = {
        "model_type": "gemma4_vision",
        "vision_config": visual,
        "builder_config": {
            "min_image_tokens": args.min_image_tokens,
            "max_image_tokens": args.max_image_tokens,
            "max_image_tokens_per_image": args.max_image_tokens_per_image,
        },
    }
    image_token_id = root.get("image_token_id")
    if not isinstance(image_token_id, int):
        image_token_id = find_token_id(bundle.model_dir, "<|image_pad|>")
    if image_token_id is not None:
        result["image_token_id"] = image_token_id
    for key in ("boi_token_id", "eoi_token_id"):
        value = root.get(key)
        if isinstance(value, int):
            result[key] = value
    return result
