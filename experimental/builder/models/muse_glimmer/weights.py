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
"""Muse-Glimmer checkpoint weight mapping.

The base checkpoint nests the text tower under ``model.language_model.`` and the
vision tower under ``model.vision_tower.`` / ``model.vision_adapter.`` /
``model.vision_projection``. ``lm_head`` sits at the top level (untied).
"""

_PREFIXES = {
    "llm": ("model.language_model.", "language_model."),
    "visual": ("model.", ),
}
_WRAPPERS = (
    "model.language_model.",
    "language_model.",
)


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    """Map frontend tensor names to Muse-Glimmer component checkpoints."""
    del spec_type, spec_role, quant_type
    prefixes = _PREFIXES.get(component, ())
    candidates = [prefix + name for prefix in prefixes]
    if component == "llm" and name.startswith("model."):
        nested_name = name[len("model."):]
        candidates.extend(prefix + nested_name for prefix in prefixes)
    # MXFP8 layers (mlp.down_proj / lm_head) store their U8 E8M0 block scale
    # under the compressed-tensors name ``weight_scale_inv`` rather than
    # ModelOpt's ``weight_scale``; offer both so the loader resolves either.
    if name.endswith(".weight_scale"):
        candidates = [
            candidate for base in ([name] + candidates)
            for candidate in (base, base + "_inv")
        ]
    return tuple(candidates)


def normalize_checkpoint_name(name: str) -> str:
    """Strip this family's checkpoint wrappers and canonicalize the gate name.

    ModelOpt records the gated-attention projection under its canonical HF name
    ``self_attn.output_gate_proj`` in ``quantized_layers``, while the saved
    tensor and the modeling graph both call it ``self_attn.gate_proj``. Rewrite
    it so per-layer mixed-precision overrides line up with the graph module.
    """
    for prefix in _WRAPPERS:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if name.endswith(".self_attn.output_gate_proj"):
        name = name[:-len(".output_gate_proj")] + ".gate_proj"
    return name
