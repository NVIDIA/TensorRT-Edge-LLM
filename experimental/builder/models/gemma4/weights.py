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
"""Gemma 4 checkpoint weight mapping."""

import numpy as np

from ...core.weights import ParameterSpec
from ...weight_packing import nvfp4 as nvfp4_pack

_PREFIXES = {
    "llm": ("thinker.", "language_model.", "model.language_model.",
            "vlm.model.language_model.", "vlm."),
    "visual":
    ("thinker.visual.", "visual.", "vision_tower.", "model.visual.", "model."),
    "audio": ("thinker.audio_tower.", "audio_tower.", "audio.", "model.audio.",
              "model."),
}
_WRAPPERS = (
    "model.language_model.",
    "thinker.model.",
    "language_model.",
    "text_model.",
    "llm.",
    "thinker.",
)


def writes_runtime_embedding(args) -> bool:
    """Gemma4 MTP assistants consume the target embedding sidecar."""
    return not (args.resolved_spec_role.value == "draft"
                and args.spec_type == "gemma4_mtp")


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    """Map frontend tensor names to Gemma 4 component checkpoints."""
    del spec_type, spec_role
    prefixes = _PREFIXES.get(component, ())
    candidates = [prefix + name for prefix in prefixes]
    if component == "visual" and name.startswith("vision_embedder."):
        suffix = name[len("vision_embedder."):]
        candidates.extend(
            (f"model.embed_vision.{suffix}", f"embed_vision.{suffix}"))
    if component == "visual" and name.startswith("embed_vision."):
        suffix = name[len("embed_vision."):]
        candidates.extend((f"model.embed_vision.multimodal_embedder.{suffix}",
                           f"embed_vision.multimodal_embedder.{suffix}"))
    if component == "llm" and name.startswith("model."):
        nested_name = name[len("model."):]
        candidates.extend(prefix + nested_name for prefix in prefixes)
    if name == "lm_head.weight" and quant_type == "fp16":
        candidates.extend(("model.embed_tokens.weight",
                           "model.language_model.embed_tokens.weight"))
    if name == "model.embed_tokens_per_layer.weight":
        candidates.append("embed_tokens_per_layer.weight")
    return tuple(candidates)


def normalize_checkpoint_name(name: str) -> str:
    """Remove only wrappers used by this model family's checkpoints."""
    for prefix in _WRAPPERS:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def pack_dense_nvfp4_experts(load_expert,
                             num_experts: int,
                             hidden_size: int,
                             intermediate_size: int,
                             group_size: int,
                             fc1_layout: str,
                             plugin_intermediate_size: int | None = None):
    """Quantize dense Gemma experts for the NVFP4 MoE operation."""
    if fc1_layout not in ("interleave", "concat"):
        raise ValueError(f"unsupported FC1 layout {fc1_layout!r}")

    plugin_intermediate_size = (intermediate_size if plugin_intermediate_size
                                is None else plugin_intermediate_size)
    if plugin_intermediate_size < intermediate_size:
        raise ValueError(
            "plugin intermediate size cannot be smaller than the checkpoint")

    def build_fc1(gate, up):
        padding = plugin_intermediate_size - intermediate_size
        gate = np.pad(gate, ((0, padding), (0, 0)))
        up = np.pad(up, ((0, padding), (0, 0)))
        if fc1_layout == "concat":
            return np.concatenate([up, gate],
                                  axis=0).reshape(2 * plugin_intermediate_size,
                                                  hidden_size)
        rows = 64
        if plugin_intermediate_size % rows:
            raise ValueError("moe_intermediate_size must be a multiple of 64")
        chunks = plugin_intermediate_size // rows
        up_chunks = up.reshape(chunks, rows, hidden_size)
        gate_chunks = gate.reshape(chunks, rows, hidden_size)
        return np.stack([up_chunks, gate_chunks],
                        axis=1).reshape(2 * plugin_intermediate_size,
                                        hidden_size)

    fc1_weights, fc1_scales = [], []
    fc2_weights, fc2_scales = [], []
    for expert_index in range(num_experts):
        expert = load_expert(expert_index)
        fc1 = build_fc1(expert["gate"], expert["up"])
        weight, scale = nvfp4_pack.pack_nvfp4_moe_weight(fc1, group_size)
        fc1_weights.append(weight)
        fc1_scales.append(scale)
        down = np.pad(
            expert["down"],
            ((0, 0), (0, plugin_intermediate_size - intermediate_size)),
        )
        weight, scale = nvfp4_pack.pack_nvfp4_moe_weight(down, group_size)
        fc2_weights.append(weight)
        fc2_scales.append(scale)

    ones = np.ones(num_experts, dtype=np.float32)
    return (np.stack(fc1_weights), np.stack(fc1_scales), ones,
            np.stack(fc2_weights), np.stack(fc2_scales), ones.copy())


def repack_nvfp4_experts(load_expert,
                         num_experts: int,
                         hidden_size: int,
                         intermediate_size: int,
                         group_size: int,
                         fc1_layout: str,
                         plugin_intermediate_size: int | None = None):
    """Arrange provider-packed Gemma experts without requantization."""
    return nvfp4_pack.pack_gated_nvfp4_experts(
        load_expert,
        num_experts,
        hidden_size,
        intermediate_size,
        group_size,
        fc1_layout,
        plugin_intermediate_size,
    )


def nvfp4_expert_specs(weights, experts_prefix: str, num_experts: int,
                       plugin_intermediate_size: int) -> dict:
    """Describe final Gemma NVFP4 MoE buffers from checkpoint headers."""
    up_names = weights.nvfp4_checkpoint_names(f"{experts_prefix}.0.up_proj")
    down_names = weights.nvfp4_checkpoint_names(
        f"{experts_prefix}.0.down_proj")
    up_weight = weights.store.shape(up_names[0])
    up_scale = weights.store.shape(up_names[1])
    down_weight = weights.store.shape(down_names[0])
    down_scale = weights.store.shape(down_names[1])
    return {
        "fc1_qweights":
        ParameterSpec(
            (num_experts, 2 * plugin_intermediate_size, up_weight[1]),
            np.int8),
        "fc1_blocks_scale":
        ParameterSpec(
            (num_experts, (2 * plugin_intermediate_size + 127) // 128,
             (up_scale[1] + 3) // 4, 32, 4, 4), np.int8),
        "fc1_alpha":
        ParameterSpec((num_experts, ), np.float32),
        "fc2_qweights":
        ParameterSpec(
            (num_experts, down_weight[0], plugin_intermediate_size // 2),
            np.int8),
        "fc2_blocks_scale":
        ParameterSpec((num_experts, (down_scale[0] + 127) // 128,
                       (plugin_intermediate_size // 16 + 3) // 4, 32, 4, 4),
                      np.int8),
        "fc2_alpha":
        ParameterSpec((num_experts, ), np.float32),
        "input_global_scale":
        ParameterSpec((num_experts, ), np.float32),
        "down_input_scale":
        ParameterSpec((num_experts, ), np.float32),
        "e_score_correction_bias":
        ParameterSpec((num_experts, ), np.float32),
    }


def nvfp4_expert_bindings(weights, experts_prefix: str, correction_key: str,
                          num_experts: int, sm12x: bool) -> dict:
    """Map provider-packed Gemma experts to their final plugin inputs."""
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
        raise ValueError("Gemma NVFP4 experts use inconsistent alpha formats")
    fc1_reciprocal = fc1_reciprocal.pop()
    fc2_reciprocal = fc2_reciprocal.pop()

    common = {
        "num_experts": num_experts,
        "fc1_layout": "concat" if sm12x else "interleave",
    }
    gate_up = ("up_proj", "gate_proj")
    down = ("down_proj", )
    correction = (weights.checkpoint_binding([correction_key],
                                             assemble="cast_to_fp32") if
                  weights.has(correction_key) else weights.checkpoint_binding(
                      [], "generated", "fill", fill_value=1.0))
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
        correction,
    }
