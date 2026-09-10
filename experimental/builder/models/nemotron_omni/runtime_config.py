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
"""Nemotron-Omni runtime configuration artifacts."""

from typing import Any, Dict

from ...core import contracts
from ...core.artifacts.tokenizer import find_token_id


def _video_config(root: Dict[str, Any], key: str, default):
    visual = root.get("vision_config") or {}
    return visual.get(key, root.get(key, default))


def component_runtime_config(bundle, component: contracts.Component, args):
    """Return Nemotron-Omni component runtime config."""
    root = bundle.root
    if component == contracts.Component.AUDIO:
        audio = dict(bundle.component_dict(component))
        result = dict(root)
        result["model_type"] = audio.get("model_type",
                                         "nemotron_omni_audio_encoder")
        result["builder_config"] = {
            "min_time_steps": args.min_time_steps,
            "max_time_steps": args.max_time_steps,
            "min_code_len": args.min_code_len,
            "opt_code_len": args.opt_code_len,
            "max_code_len": args.max_code_len,
        }
        return result
    if component == contracts.Component.VISUAL:
        visual = dict(bundle.component_dict(component))
        visual["model_type"] = "nemotron_omni_vision_encoder"
        result: Dict[str, Any] = {
            "model_type": "nemotron_omni_vision_encoder",
            "vision_config": visual,
            "builder_config": {
                "min_image_tokens": args.min_image_tokens,
                "max_image_tokens": args.max_image_tokens,
                "max_image_tokens_per_image": args.max_image_tokens_per_image,
            },
        }
        llm_config = root.get("llm_config", {})
        if "vocab_size" in llm_config:
            result["llm_config"] = {"vocab_size": llm_config["vocab_size"]}
        token_ids = {
            "img_context_token_id":
            root.get("img_context_token_id")
            or find_token_id(bundle.model_dir, "<image>"),
            "img_start_token_id":
            root.get("img_start_token_id")
            or find_token_id(bundle.model_dir, "<img>"),
            "img_end_token_id":
            root.get("img_end_token_id")
            or find_token_id(bundle.model_dir, "</img>"),
        }
        if not isinstance(token_ids["img_context_token_id"], int):
            raise ValueError("Nemotron-Omni tokenizer does not define <image>")
        result.update({
            key: value
            for key, value in token_ids.items() if isinstance(value, int)
        })
        for key in ("force_image_size", "norm_mean", "norm_std", "patch_size",
                    "downsample_ratio", "vit_hidden_size"):
            if key in root:
                result[key] = root[key]
        result.update({
            "video_temporal_patch_size":
            int(_video_config(root, "video_temporal_patch_size", 2)),
            "video_target_num_patches":
            int(_video_config(root, "video_target_num_patches", 1024)),
            "video_maintain_aspect_ratio":
            bool(_video_config(root, "video_maintain_aspect_ratio", True)),
            "video_pruning_rate":
            float(_video_config(root, "video_pruning_rate", 0.0)),
        })
        return result
    return None
