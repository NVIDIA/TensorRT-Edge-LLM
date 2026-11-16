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
"""pi0.5 component names mapped onto converted openpi checkpoints."""

from __future__ import annotations

_VISION_PREFIX = "paligemma_with_expert.paligemma.model."
_PREFIX_LM_PREFIX = ("paligemma_with_expert.paligemma.model.language_model.")
_EXPERT_PREFIX = "paligemma_with_expert.gemma_expert.model."
_EMBED_KEYS = (
    _PREFIX_LM_PREFIX + "embed_tokens.weight",
    "paligemma_with_expert.paligemma.lm_head.weight",
)


def _with_wrapper(candidates):
    expanded = []
    for candidate in candidates:
        expanded.extend((candidate, "model." + candidate))
    return tuple(expanded)


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    """Resolve component-local graph names without copying checkpoint tensors."""
    del spec_type, spec_role, quant_type
    component = str(component).replace("-", "_")
    if component == "visual":
        candidates = (_VISION_PREFIX + name, )
    elif component == "prefix":
        if name == "embed_tokens.weight":
            candidates = _EMBED_KEYS
        else:
            candidates = (_PREFIX_LM_PREFIX + name, )
    elif component in ("action", "cond"):
        if name.startswith(("action_in_proj.", "action_out_proj.",
                            "time_mlp_in.", "time_mlp_out.")):
            candidates = (name, )
        else:
            candidates = (_EXPERT_PREFIX + name, )
    else:
        candidates = ()
    return _with_wrapper(candidates)
