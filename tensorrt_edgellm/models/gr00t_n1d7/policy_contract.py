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
"""``policy.json``: the embodiment's observation and action contract in robot units."""

from __future__ import annotations

import json
import os

from .geometry import Geometry

POLICY_CONTRACT_FILENAME = "policy.json"


def embodiment_tag(model_dir: str, embodiment_id: int) -> str:
    """The tag whose id the build baked in; needed to key the statistics."""
    path = os.path.join(model_dir, "embodiment_id.json")
    with open(path) as stream:
        tags = json.load(stream)
    matches = sorted(tag for tag, index in tags.items()
                     if int(index) == embodiment_id)
    if not matches:
        raise ValueError(f"no embodiment tag maps to id {embodiment_id}")
    # Several tags can share an id; the statistics file keys only one of them.
    return matches[0]


def _normalization(statistics: dict, modality: dict, part: str,
                   use_percentiles: bool) -> dict:
    """Flatten one modality's per-key bounds in ``modality_keys`` order, the order the
    processor concatenates joint groups, so the runtime needs two vectors rather than
    the nested statistics file."""
    keys = list(modality["modality_keys"])
    low_name, high_name = ("q01", "q99") if use_percentiles else ("min", "max")
    low, high, dims = [], [], []
    for key in keys:
        stats = statistics[part][key]
        low.extend(float(value) for value in stats[low_name])
        high.extend(float(value) for value in stats[high_name])
        dims.append(len(stats[low_name]))
    return {
        "keys": keys,
        "dims": dims,
        "total_dim": sum(dims),
        "min": low,
        "max": high,
    }


def vision_contract(policy: dict, visual: dict) -> dict:
    """The eval transform's geometry: albumentations' ``SmallestMaxSize`` ->
    ``FractionalCenterCrop`` -> ``SmallestMaxSize``, which the runtime reproduces."""
    if not policy.get("use_albumentations", True):
        raise ValueError("GR00T support assumes the albumentations eval "
                         "transform; this checkpoint disables it")
    if policy.get("letter_box_transform", False):
        raise ValueError("GR00T support assumes no letter-box transform; "
                         "this checkpoint enables it")
    target = list(policy.get("image_target_size") or [256, 256])
    shortest = int(policy.get("shortest_image_edge") or target[0])
    patch = int(visual.get("patch_size", 16))
    merge = int(visual.get("spatial_merge_size", 2))
    unit = patch * merge
    if shortest % unit:
        raise ValueError(
            f"GR00T shortest edge {shortest} is not a multiple of "
            f"{unit}; the merger would drop patches")
    grid = shortest // patch
    return {
        "shortest_edge": shortest,
        "crop_fraction": float(policy.get("crop_fraction", 1.0)),
        "tokens_per_view": grid * grid // (merge**2),
        "formalize_language": bool(policy.get("formalize_language", True)),
    }


def build_policy_contract(model_dir: str, policy: dict, visual: dict,
                          geometry: Geometry) -> dict:
    """The embodiment's bounds in joint-group order, plus the camera and image geometry."""
    tag = embodiment_tag(model_dir, geometry.embodiment_id)
    with open(os.path.join(model_dir, "processor_config.json")) as stream:
        processor = json.load(stream)
    configs = processor["processor_kwargs"]["modality_configs"]
    if tag not in configs:
        raise ValueError(f"processor_config.json has no modality config for "
                         f"embodiment tag {tag!r}")
    with open(os.path.join(model_dir, "statistics.json")) as stream:
        statistics = json.load(stream)
    if tag not in statistics:
        raise ValueError(f"statistics.json has no entry for embodiment tag "
                         f"{tag!r}")

    modality = configs[tag]
    action_config = modality["action"]
    reps = {
        entry["rep"]
        for entry in (action_config.get("action_configs") or [])
    }
    if reps - {"ABSOLUTE"}:
        raise ValueError(
            f"embodiment {tag!r} has non-absolute action representations "
            f"{sorted(reps)}; the runtime implements absolute actions only")

    contract = {
        "embodiment_tag":
        tag,
        "embodiment_id":
        geometry.embodiment_id,
        "use_percentiles":
        geometry.use_percentiles,
        "clip_outliers":
        True,
        "state":
        _normalization(statistics[tag], modality["state"], "state",
                       geometry.use_percentiles),
        "action":
        _normalization(statistics[tag], action_config, "action",
                       geometry.use_percentiles),
    }
    # The policy emits one action per delta index, which is shorter than the
    # model's action horizon; the tail of the chunk is not part of the contract.
    contract["action"]["horizon"] = len(action_config["delta_indices"])
    contract["state"]["history"] = len(modality["state"]["delta_indices"])
    contract["video_keys"] = list(modality["video"]["modality_keys"])
    contract["vision"] = vision_contract(policy, visual)
    return contract


def write_policy_contract(output_dir: str, contract: dict) -> None:
    with open(os.path.join(output_dir, POLICY_CONTRACT_FILENAME),
              "w") as stream:
        json.dump(contract, stream, indent=2)
