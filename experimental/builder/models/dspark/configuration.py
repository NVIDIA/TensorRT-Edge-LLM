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
"""DSpark target/draft pairing configuration."""

from ...core import contracts
from ...core.bundle import BundleConfig
from ...core.dspark_config import resolve_dspark_config
from ..gemma4.configuration import normalize_block_draft_config


def resolve_build_profile(draft: dict, max_draft_tree_size,
                          max_verify_tree_size, tree_base: bool):
    """Resolve DSpark profiles including its optional non-anchor mask slot."""
    values = resolve_dspark_config(draft)
    block_size = int(values["block_size"])
    slot_offset = 0 if values["sample_from_anchor"] else 1
    draft_size = (block_size + slot_offset
                  if max_draft_tree_size is None else int(max_draft_tree_size))
    default_verify_size = (block_size + 1 if tree_base else
                           min(block_size, draft_size - slot_offset) + 1)
    verify_size = (default_verify_size if max_verify_tree_size is None else
                   int(max_verify_tree_size))
    return verify_size, draft_size


def _validate_dimensions(draft, target) -> None:
    if target.hidden_size != draft.hidden_size:
        raise ValueError("DSpark base/draft hidden sizes must match: "
                         f"{target.hidden_size} != {draft.hidden_size}")
    if target.vocab_size != draft.vocab_size:
        raise ValueError("DSpark base/draft vocab sizes must match: "
                         f"{target.vocab_size} != {draft.vocab_size}")


def _validate_runtime_contract(config, build_args,
                               role: contracts.SpecRole) -> None:
    if build_args is None:
        return
    slot_offset = 0 if config.dspark_sample_from_anchor else 1
    max_draft_input = config.dspark_block_size + slot_offset
    if build_args.max_draft_tree_size > max_draft_input:
        raise ValueError(
            "DSpark max_draft_tree_size exceeds the checkpoint input block")
    if role == contracts.SpecRole.BASE and not build_args.tree_base:
        proposal_size = build_args.max_verify_tree_size - 1
        if proposal_size < 1 or proposal_size > config.dspark_block_size:
            raise ValueError(
                "DSpark chain verification size exceeds the checkpoint block")
        if proposal_size + slot_offset > build_args.max_draft_tree_size:
            raise ValueError(
                "DSpark chain draft profile does not include the mask slot")
    if build_args.reduced_vocab_dir or build_args.draft_reduced_vocab_dir:
        raise ValueError("DSpark does not support reduced-vocabulary engines")
    markov_type = config.dspark_markov_head_type or "vanilla"
    if markov_type != "vanilla":
        raise ValueError(
            f"DSpark supports markov_head_type='vanilla', got {markov_type!r}")
    if not 0 <= config.dspark_mask_token_id < config.vocab_size:
        raise ValueError(
            "DSpark mask_token_id is outside the draft vocabulary")


def configure_base(config,
                   *,
                   paired_draft_dir: str = "",
                   build_args=None,
                   **kwargs) -> None:
    """Read the target-hidden and sequential-head contract from the draft."""
    if not paired_draft_dir:
        raise ValueError("DSpark base requires a paired draft checkpoint")
    bundle = BundleConfig.from_pretrained(paired_draft_dir)
    draft = bundle.component_dict(contracts.Component.LLM)
    values = resolve_dspark_config(draft)
    target_layers = [int(index) for index in values["target_layer_ids"]]
    if not target_layers:
        raise ValueError("DSpark draft config must provide target_layer_ids")
    if len(set(target_layers)) != len(target_layers):
        raise ValueError("DSpark target-layer IDs must be unique")
    invalid = [
        index for index in target_layers
        if index < 0 or index >= config.num_hidden_layers
    ]
    if invalid:
        raise ValueError(
            f"DSpark target-layer IDs outside base model: {invalid}")
    draft_hidden = int(draft.get("hidden_size", 0))
    if draft_hidden != config.hidden_size:
        raise ValueError("DSpark base/draft hidden sizes must match: "
                         f"{config.hidden_size} != {draft_hidden}")
    draft_vocab = int(draft.get("vocab_size", 0))
    if draft_vocab != config.vocab_size:
        raise ValueError("DSpark base/draft vocab sizes must match: "
                         f"{config.vocab_size} != {draft_vocab}")

    config.dspark_base = True
    config.dspark_tree_base = bool(build_args and build_args.tree_base)
    config.dspark_target_layer_ids = target_layers
    config.dspark_block_size = int(values["block_size"])
    config.dspark_mask_token_id = int(values["mask_token_id"])
    config.dspark_enable_confidence_head = bool(
        values["enable_confidence_head"])
    config.dspark_confidence_head_with_markov = bool(
        values["confidence_head_with_markov"])
    config.dspark_markov_head_type = str(values["markov_head_type"])
    config.dspark_markov_rank = int(values["markov_rank"])
    config.dspark_causal_proposal = bool(values["causal_head"])
    config.dspark_contiguous_query_swa = bool(values["contiguous_query_swa"])
    config.dspark_sample_from_anchor = bool(values["sample_from_anchor"])
    _validate_runtime_contract(config, build_args, contracts.SpecRole.BASE)


def configure_draft(config,
                    *,
                    paired_target=None,
                    build_args=None,
                    **kwargs) -> None:
    """Validate and normalize one DSpark draft checkpoint."""
    if paired_target is None:
        raise ValueError("DSpark draft requires a target config")
    _validate_dimensions(config, paired_target)
    if not config.dspark_target_layer_ids:
        raise ValueError("DSpark draft config must provide target_layer_ids")
    if config.dspark_markov_rank <= 0:
        raise ValueError("DSpark draft requires markov_rank > 0")
    normalize_block_draft_config(config)
    _validate_runtime_contract(config, build_args, contracts.SpecRole.DRAFT)
