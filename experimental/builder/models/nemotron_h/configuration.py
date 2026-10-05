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
"""Nemotron-H checkpoint configuration."""

from dataclasses import replace

from ...core import contracts

_PATTERN_TYPES = {
    "M": "mamba",
    "-": "mlp",
    "*": "attention",
    "E": "moe",
}


def component_config(root: dict, component: contracts.Component) -> dict:
    if component == contracts.Component.LLM:
        return root
    raise ValueError(f"Nemotron-H has no {component.value} configuration")


def prepare_text_config(config: dict, root: dict,
                        component: contracts.Component,
                        model_dir: str) -> dict:
    config = dict(config)
    config["rotary_dim_override"] = int(
        config.get("head_dim",
                   config["hidden_size"] // config["num_attention_heads"]))
    config["hybrid_uses_rope"] = False
    raw_types = config.get("layers_block_type") or config.get("layer_types")
    if raw_types:
        config["num_hidden_layers"] = len(raw_types)
        config["layer_types"] = [
            "mamba"
            if str(layer_type).lower() == "linear_attention" else layer_type
            for layer_type in raw_types
        ]
    elif config.get("hybrid_override_pattern"):
        config["layer_types"] = [
            _PATTERN_TYPES[token]
            for token in config["hybrid_override_pattern"]
            if token in _PATTERN_TYPES
        ]
    else:
        config["layer_types"] = ["mamba", "moe", "attention", "mlp"]
    return config


def configure_base(config, *, build_args=None, **kwargs) -> None:
    """Enable the checkpoint-owned MTP feedback contract."""
    del kwargs
    config.mtp_base = True
    config.mtp_tree_base = bool(build_args and build_args.tree_base)


def _mtp_layer_types(raw: dict):
    declared = raw.get("mtp_layers_block_type")
    if declared:
        return [
            "mamba" if str(layer_type).lower() == "linear_attention" else
            "attention" if str(layer_type).lower() == "full_attention" else
            str(layer_type).lower() for layer_type in declared
        ]
    pattern = str(raw.get("mtp_hybrid_override_pattern") or "")
    return [
        _PATTERN_TYPES[token] for token in pattern if token in _PATTERN_TYPES
    ]


def configure_draft(config, **kwargs) -> None:
    """Select the predictor blocks embedded in Nemotron-H checkpoints."""
    del kwargs
    if config.mtp_num_hidden_layers != 1:
        raise ValueError("Nemotron-H MTP requires one next-token predictor")
    layer_types = _mtp_layer_types(config.raw_component)
    if not layer_types:
        raise ValueError("Nemotron-H MTP requires mtp_layers_block_type or "
                         "mtp_hybrid_override_pattern")
    unsupported = sorted(set(layer_types) - {"attention", "moe"})
    if unsupported:
        raise ValueError("unsupported Nemotron-H MTP block types: " +
                         ", ".join(unsupported))

    head_type = config.quant.module_type("lm_head", config.tie_word_embeddings)
    config.num_hidden_layers = len(layer_types)
    config.layer_types = layer_types
    config.attention_layer_types = [
        "full_attention" if layer_type == "attention" else layer_type
        for layer_type in layer_types
    ]
    config.mamba_cfg = None
    config.gdn_cfg = None
    config.mtp_base = False
    config.tie_word_embeddings = False
    config.quant = replace(
        config.quant,
        excluded=(),
        layer_overrides={"lm_head": head_type},
        layer_group_sizes={
            "lm_head": config.quant.module_group_size("lm_head")
        },
        is_mixed_precision=True,
    )
