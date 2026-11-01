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
"""Gemma4 runtime configuration artifacts."""

import json
import os

from ...core import contracts
from ...core.artifacts.tokenizer import find_token_id


def update_llm_config(config: dict, root: dict, cfg, args) -> None:
    """Add Gemma4 multimodal PLE metadata to the runtime config."""
    del args
    if cfg.hidden_size_per_layer_input <= 0:
        return
    has_multimodal_token = any(
        type(config.get(key)) is int or type(root.get(key)) is int
        for key in ("image_token_id", "audio_token_id"))
    if not has_multimodal_token:
        return
    text = root.get("text_config") or {}
    pad_token_id = text.get("pad_token_id", root.get("pad_token_id"))
    if type(pad_token_id) is not int:
        raise ValueError(
            "Gemma4 multimodal PLE requires text_config.pad_token_id")
    config["pad_token_id"] = pad_token_id


def component_runtime_config(bundle, component: contracts.Component, args):
    """Return Gemma4 component runtime config."""
    root = bundle.root
    if component == contracts.Component.AUDIO:
        audio = dict(bundle.component_dict(component))
        if bundle.root_model_type == "gemma4_unified":
            audio["model_type"] = "gemma4_unified_audio"
            result = {
                "model_type": "gemma4_unified_audio",
                "audio_config": audio,
                "text_config": root.get("text_config") or {},
            }
            processor_path = os.path.join(bundle.model_dir,
                                          "processor_config.json")
            if os.path.isfile(processor_path):
                with open(processor_path) as processor_file:
                    feature_extractor = (
                        json.load(processor_file).get("feature_extractor")
                        or {})
                if feature_extractor:
                    result["feature_extractor"] = feature_extractor
                    for key in ("audio_samples_per_token", "sampling_rate",
                                "feature_size", "padding_value"):
                        if key in feature_extractor and key not in audio:
                            audio[key] = feature_extractor[key]
            for key in ("audio_token_id", "image_token_id", "boi_token_id",
                        "eoi_token_id", "boa_token_id", "eoa_token_index"):
                if isinstance(root.get(key), int):
                    result[key] = root[key]
            if "audio_token_id" not in result:
                token_id = find_token_id(bundle.model_dir, "<|audio|>")
                if token_id is not None:
                    result["audio_token_id"] = token_id
            return result
        audio.setdefault("num_mel_bins", 128)
        result = {
            "model_type": "gemma4_audio",
            "audio_config": audio,
            "builder_config": {
                "min_time_steps": args.min_time_steps,
                "max_time_steps": args.max_time_steps,
                "min_code_len": args.min_code_len,
                "opt_code_len": args.opt_code_len,
                "max_code_len": args.max_code_len,
            },
        }
        audio_token_id = root.get("audio_token_id")
        if not isinstance(audio_token_id, int):
            audio_token_id = find_token_id(bundle.model_dir, "<|audio_pad|>")
        if audio_token_id is not None:
            result["audio_token_id"] = audio_token_id
        for key in ("boa_token_id", "eoa_token_id", "eoa_token_index"):
            value = root.get(key)
            if isinstance(value, int):
                result[key] = value
        return result
    if component == contracts.Component.VISUAL:
        visual = dict(bundle.component_dict(component))
        if bundle.root_model_type == "gemma4_unified":
            visual["model_type"] = "gemma4_unified_vision"
            result = {
                "model_type": "gemma4_unified_vision",
                "vision_config": visual,
                "text_config": root.get("text_config") or {},
                "builder_config": {
                    "min_image_tokens":
                    args.min_image_tokens,
                    "max_image_tokens":
                    args.max_image_tokens,
                    "max_image_tokens_per_image":
                    int(
                        visual.get("num_soft_tokens",
                                   args.max_image_tokens_per_image)),
                },
            }
            for key in ("image_token_id", "audio_token_id", "boi_token_id",
                        "eoi_token_id", "boa_token_id", "eoa_token_index"):
                if isinstance(root.get(key), int):
                    result[key] = root[key]
            if "image_token_id" not in result:
                token_id = find_token_id(bundle.model_dir, "<|image|>")
                if token_id is not None:
                    result["image_token_id"] = token_id
            return result
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
    return None
