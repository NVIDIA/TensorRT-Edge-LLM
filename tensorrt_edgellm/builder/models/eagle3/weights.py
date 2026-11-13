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
"""EAGLE3 draft checkpoint weight mapping."""


def writes_runtime_embedding(args) -> bool:
    """EAGLE3 consumes the base model's embedding sidecar."""
    del args
    return False


def runtime_weight_artifacts(weights, args):
    """Emit the optional draft-to-target vocabulary mapping."""
    del args
    if not weights.has("d2t"):
        return {}
    return {"d2t.safetensors": {"d2t": weights.array("d2t").astype("int32")}}


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    """Map frontend draft names to the provider EAGLE3 checkpoint."""
    del component, spec_type, spec_role, quant_type
    return (
        name.replace("layers.0.", "midlayer."),
        name.replace("layers.0.self_attn.", "midlayer.self_attn.qkv_proj."),
    )
