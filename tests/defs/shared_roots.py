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
"""Named roots for the shared storage exports that model configs reference.

Each export is mounted at a different path on CI than on a developer cluster, so
model configs name the root (``${LLM_MODELS_DIR}/...``) instead of hard-coding
either mount. The environment variable wins when set, which is what lets a user
outside NVIDIA's CI point these at their own storage.
"""

import os
from typing import Optional, Tuple

SHARED_ROOT_CANDIDATES = {
    "LLM_MODELS_DIR": (
        "/scratch.trt_llm_data/llm-models",
        "/home/scratch.trt_llm_data/llm-models",
    ),
    "EDGELLM_DATA_DIR": (
        "/scratch.edge_llm_cache",
        "/home/edge_llm_cache",
        "/home/scratch.edge_llm_cache",
    ),
}


def shared_root_names() -> Tuple[str, ...]:
    return tuple(SHARED_ROOT_CANDIDATES)


def resolve_shared_root(name: str) -> Optional[str]:
    """Resolve one named root, or None when neither the variable nor a mount exists."""
    value = os.environ.get(name)
    if value:
        return value
    for candidate in SHARED_ROOT_CANDIDATES.get(name, ()):
        if os.path.exists(candidate):
            return candidate
    return None
