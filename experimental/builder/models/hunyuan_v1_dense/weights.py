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
"""HunYuan V1 dense checkpoint weight mapping."""

# HunYuan stores the per-head Q/K norms under these names where the frontend
# modules use the Qwen3 ``q_norm`` / ``k_norm`` spelling.
_KEY_ALIASES = (
    (".self_attn.q_norm.", ".self_attn.query_layernorm."),
    (".self_attn.k_norm.", ".self_attn.key_layernorm."),
)


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    """Map frontend tensor names to HunYuan V1 checkpoint aliases."""
    del component, spec_type, spec_role
    candidates = []
    for frontend, checkpoint in _KEY_ALIASES:
        if frontend in name:
            candidates.append(name.replace(frontend, checkpoint))
    if name == "lm_head.weight" and quant_type == "fp16":
        candidates.append("model.embed_tokens.weight")
    return tuple(candidates)
