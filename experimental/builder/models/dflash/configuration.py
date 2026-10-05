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
"""DFlash target/draft pairing configuration."""

from ...core import contracts
from ...core.bundle import BundleConfig
from ..gemma4.configuration import normalize_block_draft_config


def _validate_dimensions(draft, target) -> None:
    if target.hidden_size != draft.hidden_size:
        raise ValueError("DFlash base/draft hidden sizes must match: "
                         f"{target.hidden_size} != {draft.hidden_size}")
    if target.vocab_size != draft.vocab_size:
        raise ValueError("DFlash base/draft vocab sizes must match: "
                         f"{target.vocab_size} != {draft.vocab_size}")


def configure_base(config,
                   *,
                   paired_draft_dir: str = "",
                   build_args=None,
                   **kwargs) -> None:
    """Read the draft contract required by a DFlash target graph."""
    config.dflash_base = True
    if not paired_draft_dir:
        raise ValueError("DFlash base requires a paired draft checkpoint")
    bundle = BundleConfig.from_pretrained(paired_draft_dir)
    draft = bundle.component_dict(contracts.Component.LLM)
    dflash = dict(draft.get("dflash_config") or {})
    for key in ("target_layer_ids", "block_size", "mask_token_id",
                "causal_head"):
        if key not in dflash and draft.get(key) is not None:
            dflash[key] = draft[key]
    target_layers = [
        int(index) for index in dflash.get("target_layer_ids", ())
    ]
    if not target_layers:
        raise ValueError(
            "DFlash draft config must provide dflash_config.target_layer_ids")
    if len(set(target_layers)) != len(target_layers):
        raise ValueError("DFlash target-layer IDs must be unique")
    invalid = [
        index for index in target_layers
        if index < 0 or index >= config.num_hidden_layers
    ]
    if invalid:
        raise ValueError(
            f"DFlash target-layer IDs outside base model: {invalid}")
    draft_hidden = int(draft.get("hidden_size", 0))
    if draft_hidden != config.hidden_size:
        raise ValueError("DFlash base/draft hidden sizes must match: "
                         f"{config.hidden_size} != {draft_hidden}")
    draft_vocab = int(draft.get("vocab_size", 0))
    if draft_vocab != config.vocab_size:
        raise ValueError("DFlash base/draft vocab sizes must match: "
                         f"{config.vocab_size} != {draft_vocab}")
    config.dflash_target_layer_ids = target_layers
    config.dflash_block_size = int(
        dflash.get("block_size", draft.get("block_size", 16)))
    config.dflash_mask_token_id = int(dflash.get("mask_token_id", 248070))
    config.dflash_tree_base = True


def configure_draft(config, *, paired_target=None, **kwargs) -> None:
    """Validate that the dedicated DFlash draft matches its target."""
    if paired_target is None:
        raise ValueError("DFlash draft requires a target config")
    _validate_dimensions(config, paired_target)
    normalize_block_draft_config(config)
