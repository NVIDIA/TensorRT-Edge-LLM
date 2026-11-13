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
"""Qwen3-Omni embedding artifacts."""

import os
from typing import Any, Dict

from ...core.artifacts.tensors import save_safetensors


def _talker(root: Dict[str, Any]) -> Dict[str, Any]:
    return root.get("talker_config") or root


def _projection_tensors(weights, prefix: str) -> Dict[str, object]:
    tensors = {}
    for layer in ("linear_fc1", "linear_fc2"):
        for suffix in ("weight", "bias"):
            key = f"{prefix}.{layer}.{suffix}"
            if weights.has(key):
                tensors[f"{layer}.{suffix}"] = weights.f16(key)
    return tensors


def write_talker_embeddings(weights, root: Dict[str, Any],
                            output_dir: str) -> None:
    """Write Omni talker projection tensors."""
    for name, prefix in (("text_projection", "talker.text_projection"),
                         ("hidden_projection", "talker.hidden_projection")):
        tensors = _projection_tensors(weights, prefix)
        if tensors:
            save_safetensors(os.path.join(output_dir, f"{name}.safetensors"),
                             tensors)
    _ = root


def write_code_predictor_embeddings(weights, root: Dict[str, Any],
                                    output_dir: str) -> None:
    """Write Omni code-predictor embeddings and heads."""
    talker = _talker(root)
    count = int(talker.get("num_code_groups", 16)) - 1
    embeddings = {}
    heads = {}
    for index in range(count):
        embeddings[f"embedding_{index}"] = weights.f16(
            f"model.codec_embedding.{index}.weight")
        heads[f"lm_head_{index}.weight"] = weights.f16(
            f"lm_head.{index}.weight")
    save_safetensors(os.path.join(output_dir, "codec_embeddings.safetensors"),
                     embeddings)
    save_safetensors(os.path.join(output_dir, "lm_heads.safetensors"), heads)
    projection = {}
    for suffix in ("weight", "bias"):
        key = f"small_to_mtp_projection.{suffix}"
        if weights.has(key):
            projection[suffix] = weights.f16(key)
    if projection:
        save_safetensors(
            os.path.join(output_dir, "small_to_mtp_projection.safetensors"),
            projection)
