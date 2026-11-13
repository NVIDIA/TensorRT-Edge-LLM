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

from tensorrt_edgellm.config import require_dense_eagle_draft

from ...core import contracts
from ...core.bundle import BundleConfig

_CONSUMER = "The checkpoint-direct builder's speculators eagle3 layout"


def component_config(root: dict, component: contracts.Component) -> dict:
    if component != contracts.Component.LLM:
        raise ValueError(f"EAGLE3 has no {component.value} configuration")
    require_dense_eagle_draft(root, _CONSUMER)
    config = dict(root.get("transformer_layer_config") or {})
    for key, value in root.items():
        config.setdefault(key, value)
    config["target_layer_ids"] = list(
        root.get("eagle_aux_hidden_state_layer_ids") or [])
    return config


def _target_layers(values: dict, num_target_layers: int):
    layers = [int(index) for index in values.get("target_layer_ids", ()) or ()]
    if layers:
        return layers
    return [2, num_target_layers // 2, num_target_layers - 3]


def _require_layers_for_clamp(num_decoder_layers: int) -> None:
    # Fewer than 3 decoder layers cannot supply three distinct conditioning layers,
    # so _clamp_unique would fail uniqueness later with a message that does not
    # name the cause.
    if num_decoder_layers < 3:
        raise ValueError(
            "EAGLE3 requires at least 3 decoder layers to provide three "
            f"distinct conditioning layers; got --num-decoder-layer="
            f"{num_decoder_layers}")


def _clamp_unique(layers, num_decoder_layers: int):
    # target_layer_ids index the full base stack, so --num-decoder-layer can push
    # them out of range or collapse them (the default at N=4 gives [2, 2, 0]).
    # validateEagleConfig requires unique conditioning layers, so a plain min()
    # clamp is not enough: move collisions to the nearest free slot.
    used = set()
    result = []
    for index in layers:
        candidate = max(0, min(index, num_decoder_layers - 1))
        while candidate in used and candidate < num_decoder_layers - 1:
            candidate += 1
        while candidate in used and candidate > 0:
            candidate -= 1
        used.add(candidate)
        result.append(candidate)
    return result


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


def configure_base(config,
                   *,
                   paired_draft_dir: str = "",
                   build_args=None,
                   **kwargs) -> None:
    """Enable EAGLE3 feedback on the model-family-owned base graph."""
    if not paired_draft_dir:
        raise ValueError("EAGLE3 base requires a paired draft checkpoint")
    bundle = BundleConfig.from_pretrained(paired_draft_dir)
    draft = bundle.component_dict(contracts.Component.LLM)
    require_dense_eagle_draft(draft, _CONSUMER)
    num_decoder_layers = getattr(build_args, "num_decoder_layers", 0) or 0
    layers = _target_layers(draft, config.num_hidden_layers)
    if num_decoder_layers:
        _require_layers_for_clamp(num_decoder_layers)
        layers = _clamp_unique(layers, num_decoder_layers)
    _validate_target_layers(layers, config.num_hidden_layers)
    config.eagle_base = True
    config.eagle3_target_layer_ids = layers


def configure_draft(config,
                    *,
                    paired_target=None,
                    build_args=None,
                    **kwargs) -> None:
    """Bind draft fusion dimensions to the paired target architecture."""
    require_dense_eagle_draft(config.raw_component, _CONSUMER)
    if paired_target is None:
        raise ValueError("EAGLE3 draft requires a target config")
    num_decoder_layers = getattr(build_args, "num_decoder_layers", 0) or 0
    layers = _target_layers(config.raw_component,
                            paired_target.num_hidden_layers)
    if num_decoder_layers:
        _require_layers_for_clamp(num_decoder_layers)
        layers = _clamp_unique(layers, num_decoder_layers)
    _validate_target_layers(layers, paired_target.num_hidden_layers)
    config.eagle3_target_layer_ids = layers
    config.target_hidden_size = paired_target.hidden_size
