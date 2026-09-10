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
"""Nemotron-Omni runtime artifact writing."""

import json
import os

import numpy as np

from ...core import contracts
from ...core.artifacts.runtime_artifacts import (write_component_artifacts,
                                                 write_runtime_artifacts)
from ...core.artifacts.tensors import save_safetensors
from ...core.weights import Weights
from . import runtime_config, tokenizer, weights


def _write_visual_sidecar(bundle, engine_dir: str) -> None:
    output_dir = contracts.component_spec(
        contracts.Component.VISUAL).output_dir(engine_dir)
    checkpoint = Weights(bundle.model_dir,
                         component=contracts.Component.VISUAL.value,
                         conversion=weights)
    try:
        image_key = checkpoint.find_suffix("patch_generator.embedder.weight",
                                           "vision")
        position_key = checkpoint.find_suffix("patch_generator.pos_embed",
                                              "vision")
        video_keys = [
            key for key in checkpoint.keys()
            if key.endswith("patch_generator.video_embedder.weight")
            and "vision" in key
        ]
        if len(video_keys) > 1:
            raise KeyError("multiple Nemotron-Omni video embedders found")
        if ("omni" in bundle.root_model_type.lower() and not video_keys):
            raise KeyError("Nemotron-Omni checkpoint has no video embedder")

        hidden_size = int(bundle.root["vit_hidden_size"])
        patch_size = int(bundle.root["patch_size"])
        image = checkpoint.f16(image_key).reshape(hidden_size, -1)
        position = checkpoint.f16(position_key)
        expected_image_dim = 3 * patch_size * patch_size
        if image.shape[1] != expected_image_dim:
            raise ValueError("Nemotron-Omni image embedder shape mismatch")
        if position.ndim != 3 or position.shape[2] != hidden_size:
            raise ValueError("Nemotron-Omni position embedding shape mismatch")

        tensors = {
            "embedder.weight": np.ascontiguousarray(image),
            "pos_embed": np.ascontiguousarray(position),
        }
        if video_keys:
            video = checkpoint.f16(video_keys[0]).reshape(hidden_size, -1)
            visual = bundle.root.get("vision_config") or {}
            temporal = int(
                visual.get("video_temporal_patch_size",
                           bundle.root.get("video_temporal_patch_size", 2)))
            if video.shape[1] != temporal * expected_image_dim:
                raise ValueError("Nemotron-Omni video embedder shape mismatch")
            tensors["video_embedder.weight"] = np.ascontiguousarray(video)
    finally:
        checkpoint.close()

    save_safetensors(
        os.path.join(output_dir, "nemotron_omni_embedder.safetensors"),
        tensors)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path) as config_file:
        config = json.load(config_file)
    config["supports_video"] = "video_embedder.weight" in tensors
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=2)


def write_artifacts(bundle, config, args, engine_dir: str) -> None:
    if config is not None:
        write_runtime_artifacts(config,
                                args,
                                engine_dir,
                                weight_conversion=weights,
                                runtime_config_module=runtime_config,
                                tokenizer_module=tokenizer)
        return
    write_component_artifacts(bundle,
                              args,
                              engine_dir,
                              runtime_config_module=runtime_config)
    if args.resolved_component == contracts.Component.VISUAL:
        _write_visual_sidecar(bundle, engine_dir)
