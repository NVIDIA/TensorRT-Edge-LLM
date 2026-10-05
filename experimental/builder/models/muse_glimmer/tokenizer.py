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
"""Muse-Glimmer tokenizer artifacts.

The checkpoint ships its own Jinja ``chat_template`` (the ATEM turn/tool-calling
protocol), so the runtime template is extracted from it rather than from a
packaged fallback. Only the multimodal placeholders are patched in.
"""

from typing import Any, Dict

MUSE_GLIMMER_IMAGE_PLACEHOLDER = "<|patch|>"
MUSE_GLIMMER_VIDEO_PLACEHOLDER = "<|video|>"


def patch_chat_template(template: Dict[str, Any],
                        root_config: Dict[str, Any]) -> None:
    """Patch the provider's image and video placeholders."""
    content_types = template.setdefault("content_types", {})
    content_types.setdefault("image", {}).update(
        {"format": MUSE_GLIMMER_IMAGE_PLACEHOLDER})
    if root_config.get("video_token_id") is not None:
        content_types.setdefault("video", {}).update(
            {"format": MUSE_GLIMMER_VIDEO_PLACEHOLDER})
