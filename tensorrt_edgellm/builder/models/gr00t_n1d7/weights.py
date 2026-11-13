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
"""GR00T N1.7 checkpoint key mapping for the direct builder: the backbone lives under
``backbone.model.model.``, the head under ``action_head.``; the vision tower finds its own prefix.
"""

_BACKBONE_LM = "backbone.model.model.language_model."
_ACTION_HEAD = "action_head."


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    """Map frontend tensor names onto GR00T's nested checkpoint keys."""
    del spec_type, spec_role, quant_type
    candidates = []
    if component.replace("_",
                         "-") == "vl-backbone" and name.startswith("model."):
        candidates.append(_BACKBONE_LM + name[len("model."):])
    if component.replace("_", "-") in ("vl-backbone", "action"):
        candidates.append(_ACTION_HEAD + name)
    return tuple(candidates)


def writes_runtime_embedding(args) -> bool:
    """The policy path never detokenizes, so no embedding sidecar is written."""
    del args
    return False
