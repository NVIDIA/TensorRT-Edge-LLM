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
"""Muse-Glimmer DFlash draft checkpoint weight mapping.

The draft is a generic DFlash draft graph, so its decoder layers, ``norm`` and
proposal projections use the frontend names directly. Only the target-feature
fusion projector differs: the checkpoint names it ``encoder.fc`` /
``encoder.output_norm_enc`` where the graph uses ``fc`` / ``hidden_norm``.
The draft shares the base model's embedding and ``lm_head``.
"""

_RENAMES = {
    "fc": "encoder.fc",
    "hidden_norm": "encoder.output_norm_enc",
}


def writes_runtime_embedding(args) -> bool:
    """DFlash reuses the base model's runtime embedding table."""
    del args
    return False


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    """Map the fusion projector to the assistant checkpoint's ``encoder.*``."""
    del component, spec_type, spec_role, quant_type
    for graph_name, checkpoint_name in _RENAMES.items():
        if name == graph_name or name.startswith(graph_name + "."):
            return (checkpoint_name + name[len(graph_name):], )
    return ()
