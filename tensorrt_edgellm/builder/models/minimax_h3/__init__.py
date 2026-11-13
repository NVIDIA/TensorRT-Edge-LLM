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
"""Checkpoint-direct MiniMax-H3 text-to-video-and-audio components."""

from .modeling_minimax_h3_audio import MiniMaxH3AudioDecoder
from .modeling_minimax_h3_gen import MiniMaxH3GenModel
from .modeling_minimax_h3_text import MiniMaxH3TextEncoder
from .modeling_minimax_h3_vae import MiniMaxH3VaeDecoder

__all__ = [
    "MiniMaxH3AudioDecoder",
    "MiniMaxH3GenModel",
    "MiniMaxH3TextEncoder",
    "MiniMaxH3VaeDecoder",
]
