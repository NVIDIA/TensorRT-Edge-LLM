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
"""Cosmos3 reasoning and policy checkpoint tensor aliases."""

from __future__ import annotations

import os


def _component_name(component: str) -> str:
    return str(component).replace("-", "_")


def checkpoint_dir(model_dir: str, component: str) -> str:
    """Resolve Diffusers component shards directly from the checkpoint."""
    name = _component_name(component)
    if name in ("gen", "und_prefill"):
        return os.path.join(model_dir, "transformer")
    if name == "vae_encoder":
        return os.path.join(model_dir, "vae")
    return model_dir


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    """Map readable module aliases onto the native Cosmos3 tensor schema."""
    del spec_type, spec_role, quant_type
    name_component = _component_name(component)
    candidates = []
    if name_component == "llm":
        if name.startswith("model."):
            candidates.append(name[len("model."):])
        attention_aliases = (
            (".self_attn.q_proj.", ".self_attn.to_q."),
            (".self_attn.k_proj.", ".self_attn.to_k."),
            (".self_attn.v_proj.", ".self_attn.to_v."),
            (".self_attn.o_proj.", ".self_attn.to_out."),
        )
        for candidate in tuple(candidates) + (name, ):
            for source, target in attention_aliases:
                if source in candidate:
                    candidates.append(candidate.replace(source, target))
        if name == "lm_head.weight":
            candidates.append("embed_tokens.weight")
    elif name_component == "gen":
        replacements = (
            (".cross_attention.to_q.", ".self_attn.add_q_proj."),
            (".cross_attention.to_k.", ".self_attn.add_k_proj."),
            (".cross_attention.to_v.", ".self_attn.add_v_proj."),
            (".cross_attention.to_out.", ".self_attn.to_add_out."),
            (".cross_attention.norm_q.", ".self_attn.norm_added_q."),
            (".cross_attention.norm_k.", ".self_attn.norm_added_k."),
            (".input_layernorm.", ".input_layernorm_moe_gen."),
            (".post_attention_layernorm.",
             ".post_attention_layernorm_moe_gen."),
            (".mlp.", ".mlp_moe_gen."),
        )
        for source, target in replacements:
            if source in name:
                candidates.append(name.replace(source, target))
    elif name_component == "und_prefill":
        replacements = (
            (".self_attn.q_proj.", ".self_attn.to_q."),
            (".self_attn.k_proj.", ".self_attn.to_k."),
            (".self_attn.v_proj.", ".self_attn.to_v."),
            (".self_attn.o_proj.", ".self_attn.to_out."),
            (".self_attn.q_norm.", ".self_attn.norm_q."),
            (".self_attn.k_norm.", ".self_attn.norm_k."),
        )
        for source, target in replacements:
            if source in name:
                candidates.append(name.replace(source, target))
    return tuple(dict.fromkeys(candidates))


def selected_domain_linear(weights, prefix: str, domain_id: int,
                           in_features: int, out_features: int):
    """Return one domain row as an ordinary ``[out, in]`` linear."""
    raw_weight = weights.f16(prefix + ".fc.weight")
    raw_bias = weights.f16(prefix + ".bias.weight")
    domain_count = int(raw_weight.shape[0])
    if not 0 <= domain_id < domain_count:
        raise ValueError(
            f"{prefix}: domain_id {domain_id} is outside [0, {domain_count})")
    expected = in_features * out_features
    if int(raw_weight.shape[1]) != expected:
        raise ValueError(
            f"{prefix}: flattened domain weight has {raw_weight.shape[1]} "
            f"values, expected {expected}")
    matrix = raw_weight[domain_id].reshape(in_features, out_features).T
    bias = raw_bias[domain_id].reshape(out_features)
    return matrix.copy(), bias.copy()
