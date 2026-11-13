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
"""Qwen4-Exp visual encoder.

Architecturally the Qwen3.5 tower: upstream derives ``Qwen4ExpVisionModel``
from ``Qwen3_5MoeVisionModel`` without overriding anything, and
``Qwen4ExpVisionConfig`` changes no field defaults.  This module exists as the
dispatch seam for the family rather than to re-state the architecture -- every
dimension still comes from ``vision_config``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..qwen3_5.modeling_qwen3_5_visual import Qwen3_5VLVisualModel
from ..qwen3_vl.modeling_qwen3_vl_visual import _load_weights

if TYPE_CHECKING:
    from ...config import ModelConfig


class Qwen4ExpVisualModel(Qwen3_5VLVisualModel):
    """Qwen4-Exp visual encoder — Qwen3.5 tower, no deepstack features."""


def build_qwen4_exp_visual(
        config: dict,
        weights: dict,
        model_config: "ModelConfig",
        dtype: torch.dtype = torch.float16) -> Qwen4ExpVisualModel:
    """Instantiate and load weights for a Qwen4-Exp visual encoder.

    Args:
        config:       Parsed ``vision_config`` dict from ``config.json``.
        weights:      Safetensors weight dict (all model weights).
        model_config: Top-level ``ModelConfig`` for quantized Linear dispatch.
        dtype:        Target dtype (default float16).
    """
    model = Qwen4ExpVisualModel(config, model_config=model_config)
    model.to(dtype)
    _load_weights(model, weights, prefix="model.visual")
    model.eval()
    return model


__all__ = [
    "Qwen4ExpVisualModel",
    "build_qwen4_exp_visual",
]
