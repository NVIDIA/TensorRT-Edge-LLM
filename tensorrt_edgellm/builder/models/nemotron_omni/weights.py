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
"""Nemotron Omni checkpoint weight mapping."""

import numpy as np

from ...weight_packing import nvfp4 as nvfp4_pack

_PREFIXES = {
    "llm": ("thinker.", "language_model.", "model.language_model.",
            "vlm.model.language_model.", "vlm."),
    "visual": ("thinker.visual.", "visual.", "vision_tower.", "model.visual."),
    "audio":
    ("thinker.audio_tower.", "audio_tower.", "audio.", "model.audio."),
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
    """Map frontend tensor names to Nemotron Omni component checkpoints."""
    del spec_type, spec_role
    prefixes = _PREFIXES.get(component, ())
    candidates = [prefix + name for prefix in prefixes]
    if component == "llm" and name == "model.embed_tokens.weight":
        candidates.extend((
            "language_model.backbone.embeddings.weight",
            "model.language_model.backbone.embeddings.weight",
            "thinker.language_model.backbone.embeddings.weight",
            "vlm.model.language_model.backbone.embeddings.weight",
        ))
    if component == "llm" and name.startswith("model."):
        nested_name = name[len("model."):]
        candidates.extend(prefix + nested_name for prefix in prefixes)
    if name == "lm_head.weight" and quant_type == "fp16":
        candidates.extend(("model.embed_tokens.weight",
                           "model.language_model.embed_tokens.weight"))
    return tuple(candidates)


def normalize_checkpoint_name(name: str) -> str:
    """Remove only wrappers used by this model family's checkpoints."""
    for prefix in _WRAPPERS:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def repack_nvfp4_experts(load_expert,
                         num_experts: int,
                         hidden_size: int,
                         intermediate_size: int,
                         group_size: int,
                         hidden_size_alignment: int = 1):
    """Pack this family's padded ReLU2 experts for the Edge-LLM MoE operation."""
    padded_intermediate = ((intermediate_size + 127) // 128) * 128
    if hidden_size_alignment <= 0:
        raise ValueError("hidden_size_alignment must be >= 1")
    padded_hidden = ((hidden_size + hidden_size_alignment - 1) //
                     hidden_size_alignment) * hidden_size_alignment
    if padded_hidden % group_size:
        raise ValueError("padded hidden size must be a multiple of group_size")

    hidden_bytes = padded_hidden // 2
    hidden_scale_groups = padded_hidden // group_size
    fc1_weights, fc1_scales, fc1_alpha = [], [], []
    fc2_weights, fc2_scales, fc2_alpha = [], [], []
    for expert_index in range(num_experts):
        expert = load_expert(expert_index)
        up_weight = np.ascontiguousarray(expert["up_packed"],
                                         np.uint8).view(np.int8)
        down_weight = np.ascontiguousarray(expert["down_packed"],
                                           np.uint8).view(np.int8)
        up_scale = np.ascontiguousarray(expert["up_sf"], np.uint8)
        down_scale = np.ascontiguousarray(expert["down_sf"], np.uint8)

        if (padded_intermediate != intermediate_size
                or padded_hidden != hidden_size):
            padded = np.zeros((padded_intermediate, hidden_bytes),
                              dtype=np.int8)
            padded[:intermediate_size, :hidden_size // 2] = up_weight
            up_weight = padded
            padded = np.zeros((padded_hidden, padded_intermediate // 2),
                              dtype=np.int8)
            padded[:hidden_size, :intermediate_size // 2] = down_weight
            down_weight = padded
            padded_scale = np.zeros((padded_intermediate, hidden_scale_groups),
                                    dtype=np.uint8)
            padded_scale[:intermediate_size, :hidden_size //
                         group_size] = up_scale
            up_scale = padded_scale
            padded_scale = np.zeros(
                (padded_hidden, padded_intermediate // group_size),
                dtype=np.uint8)
            padded_scale[:hidden_size, :intermediate_size //
                         group_size] = down_scale
            down_scale = padded_scale

        fc1_weights.append(up_weight)
        fc1_scales.append(
            nvfp4_pack.swizzle_nvfp4_mma_scales(up_scale, padded_intermediate,
                                                hidden_scale_groups))
        fc1_alpha.append(float(expert["up_alpha"]))
        fc2_weights.append(down_weight)
        fc2_scales.append(
            nvfp4_pack.swizzle_nvfp4_mma_scales(
                down_scale, padded_hidden, padded_intermediate // group_size))
        fc2_alpha.append(float(expert["down_alpha"]))

    return (np.stack(fc1_weights), np.stack(fc1_scales),
            np.asarray(fc1_alpha, np.float32), np.stack(fc2_weights),
            np.stack(fc2_scales), np.asarray(fc2_alpha, np.float32),
            padded_intermediate, padded_hidden)
