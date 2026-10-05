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
"""Muse-Glimmer runtime configuration artifacts."""

from ...core import contracts


def component_runtime_config(bundle, component: contracts.Component, args):
    """Return the Muse-Glimmer visual runtime config (no LLM sidecar)."""
    if component != contracts.Component.VISUAL:
        return None
    root = bundle.root
    visual = dict(bundle.component_dict(component))
    visual["model_type"] = "muse_glimmer_vision"
    result = {
        "model_type": "muse_glimmer_vision",
        "vision_config": visual,
        "builder_config": {
            "min_image_tokens": args.min_image_tokens,
            "max_image_tokens": args.max_image_tokens,
            "max_image_tokens_per_image": args.max_image_tokens_per_image,
        },
    }
    for key in ("image_token_id", "video_token_id"):
        if isinstance(root.get(key), int):
            result[key] = root[key]
    return result
