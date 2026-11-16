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
"""Qwen3-Omni-Next Talker and CodePredictor sidecars."""

import os
from typing import Any, Dict

from ...core.artifacts.tensors import save_safetensors


def _talker(root: Dict[str, Any]) -> Dict[str, Any]:
    return root.get("talker_config") or root


def write_talker_embeddings(weights, root: Dict[str, Any],
                            output_dir: str) -> None:
    """Write both Talker vocabularies and the single hidden projection."""
    save_safetensors(
        os.path.join(output_dir, "text_embedding.safetensors"), {
            "text_embedding": weights.f16("model.embed_tokens.weight"),
        })
    save_safetensors(
        os.path.join(output_dir, "codec_embedding.safetensors"), {
            "codec_embedding": weights.f16("model.codec_embedding.weight"),
        })
    save_safetensors(
        os.path.join(output_dir, "speaker_codec_embeddings.safetensors"), {
            "speaker_codec_embeddings":
            weights.array("speaker_codec_embeddings"),
        })
    save_safetensors(
        os.path.join(output_dir, "hidden_projection.safetensors"), {
            "weight": weights.f16("hidden_projection.weight"),
            "bias": weights.f16("hidden_projection.bias"),
        })
    _ = root


def write_code_predictor_embeddings(weights, root: Dict[str, Any],
                                    output_dir: str) -> None:
    """Write the stacked codec tables consumed by the device-selected head."""
    count = int(_talker(root).get("num_code_groups", 16)) - 1
    embeddings = {
        f"embedding_{index}":
        weights.f16(f"model.codec_embedding.{index}.weight")
        for index in range(count)
    }
    heads = {
        f"lm_head_{index}.weight": weights.f16(f"lm_head.{index}.weight")
        for index in range(count)
    }
    save_safetensors(os.path.join(output_dir, "codec_embeddings.safetensors"),
                     embeddings)
    save_safetensors(os.path.join(output_dir, "lm_heads.safetensors"), heads)

    projection = {}
    for suffix in ("weight", "bias"):
        key = f"model.talker_projection.{suffix}"
        if weights.has(key):
            projection[suffix] = weights.f16(key)
    if projection:
        save_safetensors(
            os.path.join(output_dir, "small_to_mtp_projection.safetensors"),
            projection)
