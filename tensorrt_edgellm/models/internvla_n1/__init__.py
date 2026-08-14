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
"""InternVLA-N1-DualVLN: a Qwen2.5-VL planner (System 2) driving a diffusion
trajectory head (System 1), bridged by ``z_latents``."""
from .modeling_internvla_n1_text import (DEFAULT_LATENT_DIM,
                                         InternVLAN1LanguageModel)

__all__ = ["InternVLAN1LanguageModel", "DEFAULT_LATENT_DIM"]
