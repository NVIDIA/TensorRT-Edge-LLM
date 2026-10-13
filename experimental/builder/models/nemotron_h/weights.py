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
"""Nemotron-H checkpoint weight mapping."""

import numpy as np

from ...core.weights import ParameterSpec
from ...weight_packing import nvfp4 as nvfp4_pack

_MTP_INTERMEDIATE_ALIGNMENT = 128


def quant_type_for_algorithm(algorithm: str, default: str) -> str:
    """Select weight-only NVFP4 for Nemotron W4A16 checkpoints."""
    if default == "nvfp4" and "W4A16" in algorithm:
        return "nvfp4_a16"
    return default


def writes_runtime_embedding(args) -> bool:
    """Native MTP drafts consume the base model's embedding sidecar."""
    return not (args.resolved_spec_role.value == "draft"
                and args.spec_type == "mtp")


def externalizes_runtime_embedding(args) -> bool:
    """Keep cached-draft base embeddings patchable as a runtime sidecar."""
    return not (args.resolved_spec_role.value == "base"
                and args.spec_type in ("dflash", "dspark"))


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    """Map frontend tensor names to Nemotron-H checkpoint aliases."""
    del component
    candidates = []
    if spec_role == "draft" and spec_type == "mtp":
        candidates.append(f"mtp.{name}")
    if name == "model.embed_tokens.weight":
        candidates.append("backbone.embeddings.weight")
    if name == "lm_head.weight" and quant_type == "fp16":
        candidates.append("backbone.embeddings.weight")
    return tuple(candidates)


def prepare_mtp_fp16_experts(weights, experts_prefix: str, num_experts: int,
                             hidden_size: int, intermediate_size: int) -> dict:
    """Pad and stack Nemotron-H's non-gated BF16 MTP experts as FP16."""
    padded_intermediate = (
        (intermediate_size + _MTP_INTERMEDIATE_ALIGNMENT - 1) //
        _MTP_INTERMEDIATE_ALIGNMENT) * _MTP_INTERMEDIATE_ALIGNMENT
    fc1_weights = []
    fc2_weights = []
    for expert in range(num_experts):
        prefix = f"{experts_prefix}.{expert}"
        up = weights.f16(prefix + ".up_proj.weight")
        down = weights.f16(prefix + ".down_proj.weight")
        if up.shape != (intermediate_size, hidden_size):
            raise ValueError(
                f"{prefix}.up_proj has shape {up.shape}, expected "
                f"{(intermediate_size, hidden_size)}")
        if down.shape != (hidden_size, intermediate_size):
            raise ValueError(
                f"{prefix}.down_proj has shape {down.shape}, expected "
                f"{(hidden_size, intermediate_size)}")
        padded_up = np.zeros((padded_intermediate, hidden_size), np.float16)
        padded_up[:intermediate_size] = up
        padded_down = np.zeros((hidden_size, padded_intermediate), np.float16)
        padded_down[:, :intermediate_size] = down
        fc1_weights.append(padded_up)
        fc2_weights.append(padded_down)
    return {
        "fc1_weights": np.stack(fc1_weights),
        "fc2_weights": np.stack(fc2_weights),
        "padded_intermediate": padded_intermediate,
    }


def mtp_fp16_expert_specs(num_experts: int, hidden_size: int,
                          intermediate_size: int) -> dict:
    """Describe the padded non-gated MTP expert buffers."""
    padded_intermediate = (
        (intermediate_size + _MTP_INTERMEDIATE_ALIGNMENT - 1) //
        _MTP_INTERMEDIATE_ALIGNMENT) * _MTP_INTERMEDIATE_ALIGNMENT
    return {
        "fc1_weights":
        ParameterSpec((num_experts, padded_intermediate, hidden_size),
                      np.float16),
        "fc2_weights":
        ParameterSpec((num_experts, hidden_size, padded_intermediate),
                      np.float16),
        "padded_intermediate":
        padded_intermediate,
    }


def mtp_fp16_expert_bindings(weights, experts_prefix: str,
                             num_experts: int) -> dict:
    """Map MTP ReLU2 experts into padded FP16 plugin buffers."""
    fc1_names = []
    fc2_names = []
    for expert in range(num_experts):
        prefix = f"{experts_prefix}.{expert}"
        fc1_names.append(prefix + ".up_proj.weight")
        fc2_names.append(prefix + ".down_proj.weight")
    return {
        "fc1_weights":
        weights.checkpoint_binding(fc1_names,
                                   "fp16",
                                   "fp16_moe_fc1_relu2",
                                   num_experts=num_experts),
        "fc2_weights":
        weights.checkpoint_binding(fc2_names,
                                   "fp16",
                                   "fp16_moe_fc2",
                                   num_experts=num_experts),
    }


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
