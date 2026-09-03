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
"""JetSpec target/draft pairing configuration."""

from ...core import contracts
from ...core.bundle import BundleConfig


def _required_integer(values: dict, draft: dict, name: str) -> int:
    if name in values:
        value = values[name]
    elif name in draft:
        value = draft[name]
    else:
        raise ValueError(f"JetSpec draft config must provide {name}")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"JetSpec {name} must be an integer")
    return value


def _provider_config(draft: dict) -> dict:
    values = draft.get("jetspec_config") or draft.get("dflash_config") or {}
    raw_target_layers = values.get("target_layer_ids", ())
    if not isinstance(raw_target_layers, (list, tuple)):
        raise ValueError("JetSpec target_layer_ids must be an array")
    if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in raw_target_layers):
        raise ValueError("JetSpec target_layer_ids must contain integers")
    target_layers = list(raw_target_layers)
    if not target_layers:
        raise ValueError(
            "JetSpec draft config must provide target_layer_ids in "
            "jetspec_config or dflash_config")
    if len(set(target_layers)) != len(target_layers):
        raise ValueError("JetSpec target-layer IDs must be unique")
    if "causal_head" in values:
        causal_head = values["causal_head"]
    elif "causal_head" in draft:
        causal_head = draft["causal_head"]
    else:
        raise ValueError("JetSpec draft config must provide causal_head")
    if not isinstance(causal_head, bool):
        raise ValueError("JetSpec causal_head must be boolean")
    if not causal_head:
        raise ValueError(
            "JetSpec requires causal_head=true; use DFlash for non-causal drafts"
        )
    result = {
        "target_layer_ids": target_layers,
        "block_size": _required_integer(values, draft, "block_size"),
        "mask_token_id": _required_integer(values, draft, "mask_token_id"),
    }
    if result["block_size"] < 2:
        raise ValueError("JetSpec block_size must be at least 2")
    if result["mask_token_id"] < 0:
        raise ValueError("JetSpec mask_token_id must be non-negative")
    return result


def _validate_dimensions(draft_hidden_size: int, draft_vocab_size: int,
                         target) -> None:
    if target.hidden_size != draft_hidden_size:
        raise ValueError("JetSpec base/draft hidden sizes must match: "
                         f"{target.hidden_size} != {draft_hidden_size}")
    if target.vocab_size != draft_vocab_size:
        raise ValueError("JetSpec base/draft vocab sizes must match: "
                         f"{target.vocab_size} != {draft_vocab_size}")


def _validate_contract(values: dict, target, draft_vocab_size: int) -> None:
    invalid = [
        index for index in values["target_layer_ids"]
        if index < 0 or index >= target.num_hidden_layers
    ]
    if invalid:
        raise ValueError(
            f"JetSpec target-layer IDs outside base model: {invalid}")
    if values["mask_token_id"] >= draft_vocab_size:
        raise ValueError(
            "JetSpec mask_token_id is outside the draft vocabulary")


def configure_base(config,
                   *,
                   paired_draft_dir: str = "",
                   build_args=None,
                   **kwargs) -> None:
    """Read the causal proposal contract required by a JetSpec target."""
    if not paired_draft_dir:
        raise ValueError("JetSpec base requires a paired draft checkpoint")
    draft = BundleConfig.from_pretrained(paired_draft_dir).component_dict(
        contracts.Component.LLM)
    values = _provider_config(draft)
    draft_hidden = _required_integer({}, draft, "hidden_size")
    draft_vocab = _required_integer({}, draft, "vocab_size")
    _validate_dimensions(draft_hidden, draft_vocab, config)
    _validate_contract(values, config, draft_vocab)
    config.dflash_base = True
    config.dflash_target_layer_ids = values["target_layer_ids"]
    config.dflash_block_size = values["block_size"]
    config.dflash_mask_token_id = values["mask_token_id"]
    config.dflash_tree_base = bool(build_args and build_args.tree_base)


def configure_draft(config, *, paired_target=None, **kwargs) -> None:
    """Validate and apply the provider's causal JetSpec draft metadata."""
    if paired_target is None:
        raise ValueError("JetSpec draft requires a target config")
    _validate_dimensions(config.hidden_size, config.vocab_size, paired_target)
    draft = BundleConfig.from_pretrained(config.model_dir).component_dict(
        contracts.Component.LLM)
    values = _provider_config(draft)
    _validate_contract(values, paired_target, config.vocab_size)
    config.dflash_target_layer_ids = values["target_layer_ids"]
    config.dflash_block_size = values["block_size"]
    config.dflash_mask_token_id = values["mask_token_id"]
