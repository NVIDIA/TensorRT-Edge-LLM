# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Checkpoint prefixes shared by export, quantization, and validation."""

# Visual submodules across Hugging Face VLM families.
VISUAL_PREFIXES = (
    "visual",
    "vision_tower",
    "vision_model",
    "multi_modal_projector",
    "mlp1",
    "image_embed",
    "embed_vision",
)

# Audio submodules across ASR, Omni, Phi-4mm, and Gemma4 families.
AUDIO_PREFIXES = (
    "audio_tower",
    "audio_embed",
    "embed_audio",
)

# Qwen3-Omni Talker CodePredictor.
CODE_PREDICTOR_PREFIXES = ("code_predictor", )

# VLA action-policy prefixes. Model-specific recipes may exclude these modules
# from quantization.
ACTION_PREFIXES = (
    "expert",
    "action_in_proj",
    "action_out_proj",
    "action_expert",
    "action",
)
