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
"""Nemotron-Omni checkpoint configuration."""

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
    if component == contracts.Component.VISUAL:
        return root.get("vision_config") or root
    if component == contracts.Component.AUDIO:
        return root.get("audio_config") or root.get("sound_config") or root
    raise ValueError(f"Nemotron-Omni has no {component.value} configuration")


def setup_profiles(builder, builder_config, network, args, bundle) -> bool:
    """Install the dynamic image/video patch profile used by the C++ runner."""
    if args.resolved_component != contracts.Component.VISUAL:
        return False

    root = bundle.root
    image_size = int(root["force_image_size"])
    patch_size = int(root["patch_size"])
    ratio = float(root["downsample_ratio"])
    scale = int(round(1.0 / ratio))
    if (image_size <= 0 or patch_size <= 0 or image_size % patch_size
            or scale <= 0 or not abs(scale * ratio - 1.0) < 1e-6):
        raise ValueError("invalid Nemotron-Omni visual patch geometry")

    patches_per_side = image_size // patch_size
    max_patches = patches_per_side * patches_per_side
    tokens_per_block = (patches_per_side // scale)**2
    if tokens_per_block <= 0:
        raise ValueError("Nemotron-Omni visual tile produces no tokens")
    min_blocks = max(1, args.min_image_tokens // tokens_per_block)
    max_blocks = max(1, args.max_image_tokens // tokens_per_block)
    opt_blocks = (min_blocks + max_blocks) // 2
    hidden_size = int(root["vit_hidden_size"])
    shuffle_width = scale * scale

    inputs = {
        network.get_input(index).name: network.get_input(index)
        for index in range(network.num_inputs)
    }
    if not {"input", "shuffle_indices"}.issubset(inputs):
        raise ValueError("Nemotron-Omni visual network must define input and "
                         "shuffle_indices")
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (min_blocks, shuffle_width, hidden_size),
                      (opt_blocks, max_patches, hidden_size),
                      (max_blocks, max_patches, hidden_size))
    output_patches = max_patches // shuffle_width
    profile.set_shape("shuffle_indices", (1, shuffle_width),
                      (output_patches, shuffle_width),
                      (output_patches, shuffle_width))
    builder_config.add_optimization_profile(profile)
    return True


def prepare_text_config(config: dict, root: dict,
                        component: contracts.Component,
                        model_dir: str) -> dict:
    """Normalize the embedded Nemotron-H decoder configuration."""
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
        raise ValueError("Nemotron Omni requires explicit decoder layer types")
    return config
