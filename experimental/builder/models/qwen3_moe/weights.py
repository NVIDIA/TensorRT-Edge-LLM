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
"""Qwen3 MoE checkpoint weight mapping."""

from dataclasses import replace
from typing import Sequence

import numpy as np

from ...core import quantization
from ...core.weights import LinearWeights, ParameterSpec
from ...weight_packing import nvfp4 as nvfp4_pack


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    """Map frontend tensor names to Qwen3 MoE checkpoint aliases."""
    del component, spec_type, spec_role
    if name == "lm_head.weight" and quant_type == "fp16":
        return ("model.embed_tokens.weight", )
    return ()


def fuse_gptq_qkv(projections: Sequence[LinearWeights],
                  plugin_version: int) -> LinearWeights:
    """Fuse Q/K/V GPTQ descriptors into one family-owned projection.

    Qwen3 checkpoints store three projections, matching Transformers. The
    compiled backend uses one GEMM when their activation permutations agree;
    this also gives attention a physically contiguous packed-QKV tensor.
    """
    if plugin_version not in (1, 2):
        raise ValueError(f"unsupported INT4 plugin version {plugin_version}")
    query, key, value = projections
    for name, projection in zip(("query", "key", "value"), projections):
        if projection.quant_type != quantization.QUANT_INT4_GPTQ:
            raise ValueError(f"{name} projection is not GPTQ INT4")
        if projection.in_features != query.in_features:
            raise ValueError("Qwen3 Q/K/V projections must share input size")
        if projection.group_size != query.group_size:
            raise ValueError("Qwen3 Q/K/V projections must share group size")
        scales = projection.weight_scale
        scale_shape = () if scales is None else scales.shape
        if (len(scale_shape) != 2
                or scale_shape[1] != projection.out_features):
            raise ValueError(f"unsupported Qwen3 GPTQ scale shape for {name}: "
                             f"{None if scales is None else scale_shape}")

    permutation = query.activation_permutation
    if any(not np.array_equal(permutation, projection.activation_permutation)
           for projection in (key, value)):
        raise ValueError("Qwen3 GPTQ Q/K/V activation permutations must match")

    biases = [projection.bias for projection in projections]
    if any(bias is None for bias in biases):
        if not all(bias is None for bias in biases):
            raise ValueError("Qwen3 Q/K/V projections must use uniform bias")
        bias = None
    else:
        bias = np.ascontiguousarray(np.concatenate(biases))

    metadata_only = [
        isinstance(projection.weight, ParameterSpec)
        for projection in projections
    ]
    if any(metadata_only) and not all(metadata_only):
        raise ValueError(
            "Qwen3 GPTQ Q/K/V weights must be uniformly materialized or metadata-only"
        )

    if all(metadata_only):
        weight_recipes = [
            projection.weight_recipe for projection in projections
        ]
        scale_recipes = [projection.scale_recipe for projection in projections]
        if any(recipe is None for recipe in weight_recipes + scale_recipes):
            raise ValueError(
                "metadata-only Qwen3 GPTQ Q/K/V projections require checkpoint recipes"
            )

        def fused_recipe(recipes, assemble):
            source_layout = recipes[0].get("source_layout", "plugin")
            if any(
                    recipe.get("source_layout", "plugin") != source_layout
                    for recipe in recipes):
                raise ValueError(
                    "Qwen3 GPTQ Q/K/V checkpoint layouts must match")
            extras = [dict(recipe.get("extra") or {}) for recipe in recipes]
            common = extras[0]
            if any(extra != common for extra in extras):
                raise ValueError(
                    "Qwen3 GPTQ Q/K/V checkpoint recipe options must match")
            return {
                "checkpoint_keys": [
                    key for recipe in recipes
                    for key in recipe.get("checkpoint_keys", ())
                ],
                "source_layout":
                source_layout,
                "assemble":
                assemble,
                "extra": {
                    **common,
                    "projection_out_features":
                    [projection.out_features for projection in projections],
                },
            }

        weight = ParameterSpec(
            (sum(projection.weight.shape[0]
                 for projection in projections), query.weight.shape[1]),
            np.int8)
        scale = ParameterSpec((query.weight_scale.shape[0],
                               sum(projection.weight_scale.shape[1]
                                   for projection in projections)), np.float16)
        weight_recipe = fused_recipe(weight_recipes, "gptq_qkv_qweight")
        scale_recipe = fused_recipe(scale_recipes, "gptq_qkv_scales")
    else:
        weight = np.ascontiguousarray(
            np.concatenate([projection.weight for projection in projections],
                           axis=0))
        scale = np.ascontiguousarray(
            np.concatenate(
                [projection.weight_scale for projection in projections],
                axis=1))
        weight_recipe = None
        scale_recipe = None

    return replace(
        query,
        weight=weight,
        bias=bias,
        weight_scale=scale,
        weight_recipe=weight_recipe,
        scale_recipe=scale_recipe,
        pre_quant_recipe=None,
        logical_out_features=sum(projection.out_features
                                 for projection in projections),
        logical_in_features=query.in_features,
    )


def repack_nvfp4_experts(load_expert, num_experts: int, hidden_size: int,
                         intermediate_size: int, group_size: int,
                         fc1_layout: str):
    """Arrange Qwen3 provider-packed experts without requantization."""
    return nvfp4_pack.pack_gated_nvfp4_experts(
        load_expert,
        num_experts,
        hidden_size,
        intermediate_size,
        group_size,
        fc1_layout,
    )


def int4_expert_bindings(weights, experts_prefix: str, num_experts: int,
                         group_size: int, zero_point_offset: int) -> dict:
    """Map Qwen3 per-expert GPTQ tensors to Int4MoePlugin inputs."""

    def projection_names(projections, leaves):
        return [
            f"{experts_prefix}.{expert}.{projection}.{leaf}"
            for expert in range(num_experts) for projection in projections
            for leaf in leaves
        ]

    common = {
        "num_experts": num_experts,
        "group_size": group_size,
        "zero_point_offset": zero_point_offset,
    }
    return {
        "fc_gate_up_qweights":
        weights.checkpoint_binding(
            projection_names(("gate_proj", "up_proj"),
                             ("qweight", "qzeros", "g_idx")), "plugin",
            "int4_moe_gate_up", **common),
        "fc_gate_up_scales":
        weights.checkpoint_binding(
            projection_names(("gate_proj", "up_proj"), ("scales", )), "plugin",
            "int4_moe_gate_up_scales", **common),
        "fc_down_qweights":
        weights.checkpoint_binding(
            projection_names(("down_proj", ), ("qweight", "qzeros", "g_idx")),
            "plugin", "int4_moe_down", **common),
        "fc_down_scales":
        weights.checkpoint_binding(
            projection_names(("down_proj", ), ("scales", )), "plugin",
            "int4_moe_down_scales", **common),
    }


def int4_expert_specs(weights, experts_prefix: str, num_experts: int) -> dict:
    """Describe final Marlin MoE buffers without reading expert payloads."""
    gate_qweight = weights.parameter_spec(
        f"{experts_prefix}.0.gate_proj.qweight", np.int32)
    gate_scales = weights.parameter_spec(
        f"{experts_prefix}.0.gate_proj.scales", np.float16)
    down_qweight = weights.parameter_spec(
        f"{experts_prefix}.0.down_proj.qweight", np.int32)
    down_scales = weights.parameter_spec(
        f"{experts_prefix}.0.down_proj.scales", np.float16)
    hidden_size = gate_qweight.shape[0] * 8
    intermediate_size = gate_qweight.shape[1]
    down_input_size = down_qweight.shape[0] * 8
    down_output_size = down_qweight.shape[1]
    return {
        "fc_gate_up_qweights":
        ParameterSpec((num_experts, hidden_size // 16, 16 * intermediate_size),
                      np.int8),
        "fc_gate_up_scales":
        ParameterSpec(
            (num_experts, gate_scales.shape[0], 2 * intermediate_size),
            np.float16),
        "fc_down_qweights":
        ParameterSpec(
            (num_experts, down_input_size // 16, 8 * down_output_size),
            np.int8),
        "fc_down_scales":
        ParameterSpec((num_experts, down_scales.shape[0], down_output_size),
                      np.float16),
    }


def nvfp4_expert_bindings(weights, experts_prefix: str, num_experts: int,
                          sm12x: bool) -> dict:
    """Map Qwen3 provider-packed NVFP4 experts to plugin inputs."""
    records = {}
    for expert in range(num_experts):
        for projection in ("up_proj", "gate_proj", "down_proj"):
            records[expert, projection] = weights.nvfp4_checkpoint_names(
                f"{experts_prefix}.{expert}.{projection}")

    def fields(projections, order):
        return [
            records[expert, projection][field] for expert in range(num_experts)
            for field in order for projection in projections
        ]

    fc1_reciprocal = {
        records[expert, projection][3]
        for expert in range(num_experts)
        for projection in ("up_proj", "gate_proj")
    }
    fc2_reciprocal = {
        records[expert, "down_proj"][3]
        for expert in range(num_experts)
    }
    if len(fc1_reciprocal) != 1 or len(fc2_reciprocal) != 1:
        raise ValueError("Qwen3 NVFP4 experts use inconsistent alpha formats")
    fc1_reciprocal = fc1_reciprocal.pop()
    fc2_reciprocal = fc2_reciprocal.pop()

    common = {
        "num_experts": num_experts,
        "fc1_layout": "concat" if sm12x else "interleave",
    }
    gate_up = ("up_proj", "gate_proj")
    down = ("down_proj", )
    return {
        "fc1_qweights":
        weights.checkpoint_binding(fields(gate_up, (0, 1, 2)),
                                   "nvfp4_qweight",
                                   "nvfp4_gated_fc1_qweight",
                                   reciprocal_alpha=fc1_reciprocal,
                                   **common),
        "fc1_blocks_scale":
        weights.checkpoint_binding(fields(gate_up, (0, 1, 2)),
                                   "nvfp4_scale_linear",
                                   "nvfp4_gated_fc1_scale",
                                   reciprocal_alpha=fc1_reciprocal,
                                   **common),
        "fc1_alpha":
        weights.checkpoint_binding([], "generated", "fill", fill_value=1.0),
        "fc2_qweights":
        weights.checkpoint_binding(fields(down, (0, 1, 2)),
                                   "nvfp4_qweight",
                                   "nvfp4_gated_fc2_qweight",
                                   reciprocal_alpha=fc2_reciprocal,
                                   **common),
        "fc2_blocks_scale":
        weights.checkpoint_binding(fields(down, (0, 1, 2)),
                                   "nvfp4_scale_linear",
                                   "nvfp4_gated_fc2_scale",
                                   reciprocal_alpha=fc2_reciprocal,
                                   **common),
        "fc2_alpha":
        weights.checkpoint_binding([], "generated", "fill", fill_value=1.0),
        "input_global_scale":
        weights.checkpoint_binding([], "generated", "fill", fill_value=1.0),
        "down_input_scale":
        weights.checkpoint_binding([], "generated", "fill", fill_value=1.0),
        "e_score_correction_bias":
        weights.checkpoint_binding([], "generated", "fill", fill_value=0.0),
    }


def nvfp4_expert_specs(weights, experts_prefix: str, num_experts: int) -> dict:
    """Describe architecture-specific NVFP4 MoE buffers from headers only."""
    up_names = weights.nvfp4_checkpoint_names(f"{experts_prefix}.0.up_proj")
    down_names = weights.nvfp4_checkpoint_names(
        f"{experts_prefix}.0.down_proj")
    up_weight = weights.store.shape(up_names[0])
    up_scale = weights.store.shape(up_names[1])
    down_weight = weights.store.shape(down_names[0])
    down_scale = weights.store.shape(down_names[1])
    return {
        "fc1_qweights":
        ParameterSpec((num_experts, 2 * up_weight[0], up_weight[1]), np.int8),
        "fc1_blocks_scale":
        ParameterSpec((num_experts, (2 * up_scale[0] + 127) // 128,
                       (up_scale[1] + 3) // 4, 32, 4, 4), np.int8),
        "fc1_alpha":
        ParameterSpec((num_experts, ), np.float32),
        "fc2_qweights":
        ParameterSpec((num_experts, *down_weight), np.int8),
        "fc2_blocks_scale":
        ParameterSpec((num_experts, (down_scale[0] + 127) // 128,
                       (down_scale[1] + 3) // 4, 32, 4, 4), np.int8),
        "fc2_alpha":
        ParameterSpec((num_experts, ), np.float32),
        "input_global_scale":
        ParameterSpec((num_experts, ), np.float32),
        "down_input_scale":
        ParameterSpec((num_experts, ), np.float32),
        "e_score_correction_bias":
        ParameterSpec((num_experts, ), np.float32),
    }
