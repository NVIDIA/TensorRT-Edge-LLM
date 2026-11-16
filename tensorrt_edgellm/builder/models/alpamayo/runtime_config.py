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
"""Alpamayo runtime configuration artifacts."""

from typing import Any, Dict

from ...core import contracts
from ...core.artifacts.runtime_config import normalize_rope_scaling
from ...core.weights import Weights
from . import weights as weight_conversion
from .configuration import prefix_kv_scales


def update_llm_config(runtime_config: Dict[str, Any], root: Dict[str, Any],
                      cfg, args) -> None:
    """Persist the per-layer K/V dequant scales needed by the action bridge."""
    _ = root
    if cfg.kv_cache_dtype != "fp8":
        return

    weights = Weights(args.model_dir,
                      component=cfg.component,
                      spec_type=args.spec_type,
                      spec_role=args.spec_role,
                      conversion=weight_conversion)
    try:
        runtime_config["kv_cache_scales"] = [
            list(weights.qkv_scales(f"model.layers.{index}.self_attn")[1:])
            for index in range(cfg.num_hidden_layers)
        ]
    finally:
        weights.close()


def _action_config(bundle, args) -> Dict[str, Any]:
    root = bundle.root
    expert = root.get("expert_cfg") or root.get("action_config") or {}
    action_space = root.get("action_space_cfg") or {}
    trajectory = root.get("traj_tokenizer_cfg") or {}
    num_layers = int(expert.get("num_hidden_layers", 0))
    num_kv_heads = int(
        expert.get("num_key_value_heads", expert.get("num_attention_heads",
                                                     0)))
    weights = Weights(bundle.model_dir,
                      component="action",
                      conversion=weight_conversion)
    try:
        layer_indices = {
            int(key.split(".")[2])
            for key in weights.keys() if key.startswith("expert.layers.")
            and len(key.split(".")) > 2 and key.split(".")[2].isdigit()
        }
        if layer_indices:
            num_layers = len(layer_indices)
        head_dim = int(expert.get("head_dim", 128))
        key_name = next(
            (key for key in weights.keys()
             if key.endswith("expert.layers.0.self_attn.k_proj.weight")), None)
        if key_name is not None:
            num_kv_heads = weights.store.shape(key_name)[0] // head_dim
    finally:
        weights.close()
    result = {
        "rope_theta":
        5_000_000.0,
        "rope_scaling": {
            "mrope_section": [24, 20, 20],
            "mrope_interleaved": True,
            "rope_type": "mrope",
            "type": "mrope",
        },
        "num_hidden_layers":
        num_layers,
        "num_attention_heads":
        int(expert.get("num_attention_heads", 0)),
        "num_key_value_heads":
        num_kv_heads,
        "head_dim":
        int(expert.get("head_dim", 128)),
        "hidden_size":
        int(expert.get("hidden_size", 0)),
        "intermediate_size":
        int(expert.get("intermediate_size", 0)),
        "rms_norm_eps":
        float(expert.get("rms_norm_eps", 1e-6)),
        "num_traj_tokens":
        1000,
        "traj_token_start":
        int(root.get("traj_token_start_idx", 0)) +
        int(trajectory.get("num_bins", 0)),
        "n_diffusion_tokens":
        int(action_space.get("n_waypoints", 64)),
        "builder_config": {
            "max_batch_size": args.max_batch_size,
            "max_kv_cache_capacity": args.max_kv_cache_capacity,
        },
    }
    if args.reuse_llm_kv_cache:
        scales = prefix_kv_scales(bundle, num_layers)
        if scales is not None:
            result["kv_cache_scales"] = scales
    return result


def component_runtime_config(bundle, component: contracts.Component, args):
    """Return Alpamayo component runtime config."""
    if component == contracts.Component.ACTION:
        return _action_config(bundle, args)
    if component != contracts.Component.VISUAL:
        return None
    root = bundle.root
    visual = dict(bundle.component_dict(component))
    runtime_type = "qwen3_vl"
    visual["model_type"] = runtime_type
    result: Dict[str, Any] = {
        "model_type": runtime_type,
        "vision_config": visual,
        "builder_config": {
            "min_image_tokens": args.min_image_tokens,
            "max_image_tokens": args.max_image_tokens,
            "max_image_tokens_per_image": args.max_image_tokens_per_image,
        },
    }
    supplemental = root.get("_direct_vlm_config") or {}
    text = dict(
        root.get("text_config") or supplemental.get("text_config")
        or root.get("llm_config") or {})
    rope = (text.get("rope_scaling") or text.get("rope_parameters")
            or root.get("rope_scaling") or root.get("rope_parameters"))
    for key in ("vision_start_token_id", "vision_end_token_id",
                "image_token_id", "video_token_id", "vocab_size",
                "rope_theta"):
        for source in (root, supplemental, text):
            if key in source:
                result[key] = source[key]
                break
    if rope:
        normalized = normalize_rope_scaling(rope)
        result["rope_scaling"] = normalized
        text.setdefault("rope_scaling", normalized)
        if "rope_theta" in normalized:
            text.setdefault("rope_theta", normalized["rope_theta"])
    for key in ("vocab_size", "rope_theta"):
        if key in result:
            text[key] = result[key]
    if text:
        result["text_config"] = text
    return result
