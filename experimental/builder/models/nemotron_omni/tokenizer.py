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
"""Nemotron-Omni tokenizer and multimodal prompt contract."""

from typing import Any, Dict

IMAGE_PLACEHOLDER = "<img><image></img>"
VIDEO_PLACEHOLDER = "<image>"
AUDIO_PLACEHOLDER = "<so_start><so_embedding><so_end>"


def patch_chat_template(template: Dict[str, Any],
                        root_config: Dict[str, Any]) -> None:
    """Preserve provider media placeholders for the C++ model runners."""
    content_types = template.setdefault("content_types", {})
    content_types.setdefault("image", {})["format"] = IMAGE_PLACEHOLDER
    content_types.setdefault("video", {})["format"] = VIDEO_PLACEHOLDER
    content_types.setdefault("audio", {})["format"] = AUDIO_PLACEHOLDER
    _ = root_config
