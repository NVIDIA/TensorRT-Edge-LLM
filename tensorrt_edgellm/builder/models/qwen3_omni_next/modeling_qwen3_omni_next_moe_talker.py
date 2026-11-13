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
"""Checkpoint-direct Qwen3-Omni-Next sparse-MoE Talker."""

from .modeling_qwen3_omni_next_moe_text import Qwen3OmniNextSparseMoeBlock
from .modeling_qwen3_omni_next_talker import (Qwen3OmniNextTalker,
                                              Qwen3OmniNextTalkerDecoderLayer,
                                              Qwen3OmniNextTalkerModel)

__all__ = [
    "Qwen3OmniNextMoeTalkerDecoderLayer",
    "Qwen3OmniNextMoeTalkerModel",
    "Qwen3OmniNextMoeTalker",
]


class Qwen3OmniNextMoeTalkerDecoderLayer(Qwen3OmniNextTalkerDecoderLayer):
    """Talker hybrid layer with routed and shared experts."""

    mlp_class = Qwen3OmniNextSparseMoeBlock


class Qwen3OmniNextMoeTalkerModel(Qwen3OmniNextTalkerModel):
    """Sparse-MoE Talker stack."""

    layer_class = Qwen3OmniNextMoeTalkerDecoderLayer


class Qwen3OmniNextMoeTalker(Qwen3OmniNextTalker):
    """Sparse-MoE Talker with its own checkpoint and runtime contract."""

    model_class = Qwen3OmniNextMoeTalkerModel
