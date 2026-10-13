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
"""EAGLE3 target/draft pairing configuration."""

from ...core import contracts
from ...core.bundle import BundleConfig


def _target_layers(values: dict, num_target_layers: int):
    layers = [int(index) for index in values.get("target_layer_ids", ()) or ()]
    if layers:
        return layers
    return [2, num_target_layers // 2, num_target_layers - 3]


def _validate_target_layers(layers, num_target_layers: int) -> None:
    if not layers:
        raise ValueError("EAGLE3 requires at least one target hidden layer")
    if len(set(layers)) != len(layers):
        raise ValueError("EAGLE3 target-layer IDs must be unique")
    invalid = [
        index for index in layers if index < 0 or index >= num_target_layers
    ]
    if invalid:
        raise ValueError(
            f"EAGLE3 target-layer IDs outside the base model: {invalid}")


def configure_base(config, *, paired_draft_dir: str = "", **kwargs) -> None:
    """Enable EAGLE3 feedback on the model-family-owned base graph."""
    if not paired_draft_dir:
        raise ValueError("EAGLE3 base requires a paired draft checkpoint")
    bundle = BundleConfig.from_pretrained(paired_draft_dir)
    draft = bundle.component_dict(contracts.Component.LLM)
    layers = _target_layers(draft, config.num_hidden_layers)
    _validate_target_layers(layers, config.num_hidden_layers)
    config.eagle_base = True
    config.eagle3_target_layer_ids = layers


def configure_draft(config, *, paired_target=None, **kwargs) -> None:
    """Bind draft fusion dimensions to the paired target architecture."""
    if paired_target is None:
        raise ValueError("EAGLE3 draft requires a target config")
    layers = _target_layers(config.raw_component,
                            paired_target.num_hidden_layers)
    _validate_target_layers(layers, paired_target.num_hidden_layers)
    config.eagle3_target_layer_ids = layers
    config.target_hidden_size = paired_target.hidden_size
