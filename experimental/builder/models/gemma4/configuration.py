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
"""Gemma4 checkpoint, component, and paired-assistant configuration."""

import json
import math
import os
from typing import List

from ...core import contracts

_ASSISTANT_MODEL_TYPES = frozenset(
    ("gemma4_assistant", "gemma4_unified_assistant"))


def available_components(root: dict, registered):
    """Return only encoder components represented by this checkpoint."""
    available = set(registered)
    if not isinstance(root.get("vision_config"), dict):
        available.discard(contracts.Component.VISUAL)
    if not isinstance(root.get("audio_config"), dict):
        available.discard(contracts.Component.AUDIO)
    return frozenset(available)


def component_config(root: dict, component: contracts.Component) -> dict:
    if component == contracts.Component.LLM:
        return root
    if component == contracts.Component.VISUAL:
        return root.get("vision_config") or root
    if component == contracts.Component.AUDIO:
        return root.get("audio_config") or root
    raise ValueError(f"Gemma4 has no {component.value} configuration")


def prepare_text_config(config: dict, root: dict,
                        component: contracts.Component,
                        model_dir: str) -> dict:
    config = dict(config)
    hidden_size = int(config["hidden_size"])
    config.setdefault("partial_rotary_factor", 0.25)
    config.setdefault("attention_scaling", 1.0)
    config.setdefault("embedding_scale", math.sqrt(float(hidden_size)))
    config.setdefault("has_value_norm", True)
    if config.get("sliding_window") is not None:
        config["use_sliding_window"] = True
    if not config.get("layer_types"):
        num_layers = int(config["num_hidden_layers"])
        pattern = int(config.get("sliding_window_pattern", 6) or 6)
        config["layer_types"] = [
            "full_attention" if
            (index + 1) % pattern == 0 else "sliding_attention"
            for index in range(num_layers)
        ]
        config["layer_types"][-1] = "full_attention"
    if not config.get("rope_parameters"):
        config["rope_parameters"] = {
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
            },
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
            },
        }
    rope_parameters = dict(config["rope_parameters"])
    sliding = dict(rope_parameters["sliding_attention"])
    sliding.setdefault("partial_rotary_factor", 1.0)
    rope_parameters["sliding_attention"] = sliding
    config["rope_parameters"] = rope_parameters
    return config


def update_device_config(config, root: dict,
                         component: contracts.Component) -> None:
    assistant = str(root.get("model_type", "")) in _ASSISTANT_MODEL_TYPES
    config.assistant_hidden_size = config.hidden_size if assistant else 0
    config.shares_target_kv = assistant
    config.has_own_kv_cache = not assistant
    config.constant_draft_positions = assistant
    config.returns_feedback_hidden = assistant


def _token_id_map(model_dir: str):
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.isfile(tokenizer_path):
        raise ValueError(
            "Gemma4 MTP pairing requires tokenizer.json in both checkpoints")
    with open(tokenizer_path) as tokenizer_file:
        tokenizer = json.load(tokenizer_file)
    token_ids = {}
    vocab = tokenizer.get("model", {}).get("vocab", {})
    if isinstance(vocab, dict):
        token_ids.update({
            str(token): int(token_id)
            for token, token_id in vocab.items()
        })
    for entry in tokenizer.get("added_tokens", []):
        content = entry.get("content")
        token_id = entry.get("id")
        if content is not None and token_id is not None:
            token_ids[str(content)] = int(token_id)
    return token_ids


def _kv_sharing_map(target, assistant) -> List[dict]:
    if len(target.attention_layer_types) != target.num_hidden_layers:
        raise ValueError(
            "Gemma4 MTP target layer_types must match num_hidden_layers")
    if len(assistant.attention_layer_types) != assistant.num_hidden_layers:
        raise ValueError(
            "Gemma4 MTP assistant layer_types must match num_hidden_layers")

    first_shared = target.num_hidden_layers - target.num_kv_shared_layers
    if first_shared <= 0 or first_shared > target.num_hidden_layers:
        raise ValueError(
            "Gemma4 MTP target requires a non-empty KV donor prefix")

    sharing_map = []
    for assistant_layer in range(assistant.num_hidden_layers):
        assistant_type = assistant.attention_type(assistant_layer)
        donor = next(
            (target_layer for target_layer in range(first_shared - 1, -1, -1)
             if target.attention_type(target_layer) == assistant_type), None)
        if donor is None:
            raise ValueError(
                "Gemma4 MTP assistant layer "
                f"{assistant_layer} ({assistant_type}) has no compatible "
                "target KV donor")
        if (assistant.layer_num_kv_heads(assistant_layer)
                != target.layer_num_kv_heads(donor)):
            raise ValueError(
                "Gemma4 MTP KV head count mismatch for assistant layer "
                f"{assistant_layer} and target layer {donor}")
        if (assistant.layer_head_dim(assistant_layer)
                != target.layer_head_dim(donor)):
            raise ValueError(
                "Gemma4 MTP KV head dimension mismatch for assistant layer "
                f"{assistant_layer} and target layer {donor}")
        target_attention_layer = sum(
            1 for layer_type in target.layer_types[:donor + 1]
            if layer_type == "attention") - 1
        if target_attention_layer < 0:
            raise ValueError(
                f"Gemma4 MTP target layer {donor} is not an attention layer")
        sharing_map.append({
            "assistant_layer": assistant_layer,
            "target_attention_layer": target_attention_layer,
            "target_layer": donor,
            "target_layer_type": assistant_type,
        })
    return sharing_map


def configure_draft(config, *, paired_target=None, **kwargs) -> None:
    """Validate and configure one Gemma4 paired-assistant draft."""
    if paired_target is None:
        raise ValueError("Gemma4 MTP draft requires a target config")
    target = paired_target
    if target.model_type not in ("gemma4", "gemma4_text"):
        raise ValueError(
            "Gemma4 MTP target must use a standard Gemma4 model type")
    if config.root_model_type not in _ASSISTANT_MODEL_TYPES:
        raise ValueError(
            "Gemma4 MTP draft must use a Gemma4 assistant checkpoint")
    if target.hidden_size != config.backbone_hidden_size:
        raise ValueError(
            "Gemma4 MTP hidden size mismatch: target hidden_size="
            f"{target.hidden_size}, assistant backbone_hidden_size="
            f"{config.backbone_hidden_size}")
    if target.vocab_size != config.vocab_size:
        raise ValueError(
            "Gemma4 MTP target and assistant vocabulary sizes differ")
    if config.num_kv_shared_layers != config.num_hidden_layers:
        raise ValueError(
            "Gemma4 MTP assistant requires every layer to share target KV")
    if target.hidden_size_per_layer_input <= 0:
        raise ValueError("Gemma4 MTP target must have PLE enabled")
    if _token_id_map(target.model_dir) != _token_id_map(config.model_dir):
        raise ValueError(
            "Gemma4 MTP target and assistant tokenizer token IDs differ")
    config.kv_sharing_map = _kv_sharing_map(target, config)
