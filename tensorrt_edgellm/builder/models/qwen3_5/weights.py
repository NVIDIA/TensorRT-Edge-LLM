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
"""Qwen3.5 checkpoint weight mapping."""

import numpy as np

_PREFIXES = {
    "llm": ("thinker.", "language_model.", "model.language_model.",
            "vlm.model.language_model.", "vlm."),
    "visual": ("thinker.visual.", "visual.", "vision_tower.", "model.visual."),
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
    """Qwen3.5 MTP drafts consume the base model's embedding sidecar."""
    return not (args.resolved_spec_role.value == "draft"
                and args.spec_type == "mtp")


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    """Map frontend tensor names to Qwen3.5 component checkpoints."""
    prefixes = _PREFIXES.get(component, ())
    candidates = []
    if spec_role == "draft" and spec_type == "mtp":
        candidates.append(f"mtp.{name}")
    candidates.extend(prefix + name for prefix in prefixes)
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


_GDN_INPUT_PROJECTIONS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")


def expand_quantized_module(name: str):
    """Map fused checkpoint projections to this model's frontend modules."""
    if name.endswith(".self_attn.qkv_proj"):
        prefix = name[:-len("qkv_proj")]
        return tuple(prefix + projection
                     for projection in ("q_proj", "k_proj", "v_proj"))
    if name.endswith(".mlp.gate_up_proj"):
        prefix = name[:-len("gate_up_proj")]
        return (prefix + "gate_proj", prefix + "up_proj")
    return (name, )


def finalize_exclusions(modules):
    """Exclude the fused GDN input when all source projections are plain."""
    result = set(modules)
    grouped = {}
    for module in result:
        for projection in _GDN_INPUT_PROJECTIONS:
            suffix = "." + projection
            if module.endswith(suffix):
                grouped.setdefault(module[:-len(suffix)],
                                   set()).add(projection)
                break
    for prefix, projections in grouped.items():
        if all(projection in projections
               for projection in _GDN_INPUT_PROJECTIONS):
            result.add(prefix + ".in_proj_fused")
    return tuple(result)


def convert_linear_fp16(weights, prefix: str):
    """Split this model's fused QKV or gate/up checkpoint projection."""
    parent, _, projection = prefix.rpartition(".")
    if projection in ("q_proj", "k_proj", "v_proj"):
        fused_prefix = parent + ".qkv_proj"
        if not weights.has(fused_prefix + ".weight"):
            return None
        weight = weights.f16(fused_prefix + ".weight")
        query_size = int(weight.shape[1])
        remaining = int(weight.shape[0]) - query_size
        if remaining <= 0 or remaining % 2:
            raise ValueError(
                f"invalid fused QKV shape {weight.shape} for {prefix!r}")
        key_value_size = remaining // 2
        selected = {
            "q_proj":
            slice(0, query_size),
            "k_proj":
            slice(query_size, query_size + key_value_size),
            "v_proj":
            slice(query_size + key_value_size,
                  query_size + 2 * key_value_size),
        }[projection]
    elif projection in ("gate_proj", "up_proj"):
        fused_prefix = parent + ".gate_up_proj"
        if not weights.has(fused_prefix + ".weight"):
            return None
        weight = weights.f16(fused_prefix + ".weight")
        if int(weight.shape[0]) % 2:
            raise ValueError(
                f"invalid fused gate/up shape {weight.shape} for {prefix!r}")
        half = int(weight.shape[0]) // 2
        selected = (slice(0, half) if projection == "gate_proj" else slice(
            half, 2 * half))
    else:
        return None

    bias = weights.opt_f16(fused_prefix + ".bias")
    return (np.ascontiguousarray(weight[selected]),
            None if bias is None else np.ascontiguousarray(bias[selected]))
