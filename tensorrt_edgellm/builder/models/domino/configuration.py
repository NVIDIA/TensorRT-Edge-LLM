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
"""Domino target/draft pairing configuration.

Domino checkpoints are DFlash V1 checkpoints whose ``dflash_config`` carries
``projector_type: "domino"`` plus the GRU correction-head contract. The shared
``resolve_dflash_contract`` validates that contract; this module pairs it with
the target.
"""

from tensorrt_edgellm.dflash import (DFlashContract, DFlashVersion,
                                     resolve_dflash_contract)

from ...core import contracts
from ...core.bundle import BundleConfig


def _draft_contract(root: dict, draft: dict) -> DFlashContract:
    contract = resolve_dflash_contract(root, draft)
    if not contract.is_domino:
        raise ValueError(
            "Domino draft checkpoint requires dflash_config.projector_type="
            "'domino'")
    return contract


def _validate_dimensions(draft_hidden_size: int, draft_vocab_size: int,
                         target) -> None:
    if target.hidden_size != draft_hidden_size:
        raise ValueError("Domino base/draft hidden sizes must match: "
                         f"{target.hidden_size} != {draft_hidden_size}")
    if target.vocab_size != draft_vocab_size:
        raise ValueError("Domino base/draft vocab sizes must match: "
                         f"{target.vocab_size} != {draft_vocab_size}")


def _clamped_target_layers(contract: DFlashContract, num_base_layers: int,
                           build_args) -> list:
    target_layers = list(contract.target_layer_ids)
    # --num-decoder-layer truncates the base stack, so clamp like DFlash does.
    num_decoder_layers = getattr(build_args, "num_decoder_layers", 0) or 0
    if num_decoder_layers:
        target_layers = [
            min(index, num_decoder_layers - 1) for index in target_layers
        ]
    invalid = [
        index for index in target_layers
        if index < 0 or index >= num_base_layers
    ]
    if invalid:
        raise ValueError(
            f"Domino target-layer IDs outside base model: {invalid}")
    return target_layers


def _apply_contract(config, contract: DFlashContract,
                    target_layers: list) -> None:
    config.dflash_version = DFlashVersion.V1
    config.dflash_projector_type = "domino"
    config.dflash_target_layer_ids = target_layers
    config.dflash_block_size = contract.block_size
    config.dflash_mask_token_id = contract.mask_token_id
    config.domino_gru_hidden_dim = contract.gru_hidden_dim
    config.domino_proj_dim = contract.proj_dim
    config.domino_prefix_len = contract.prefix_len
    config.domino_pool_size = min(contract.pool_size, config.vocab_size)
    config.domino_shift_label = contract.shift_label


def configure_base(config,
                   *,
                   paired_draft_dir: str = "",
                   build_args=None,
                   **kwargs) -> None:
    """Read the Domino draft contract required by the target graph."""
    del kwargs
    if not paired_draft_dir:
        raise ValueError("Domino base requires a paired draft checkpoint")
    bundle = BundleConfig.from_pretrained(paired_draft_dir)
    draft = bundle.component_dict(contracts.Component.LLM)
    contract = _draft_contract(bundle.root, draft)
    _validate_dimensions(int(draft.get("hidden_size", 0)),
                         int(draft.get("vocab_size", 0)), config)
    target_layers = _clamped_target_layers(contract, config.num_hidden_layers,
                                           build_args)
    config.dflash_base = True
    _apply_contract(config, contract, target_layers)


def configure_draft(config,
                    *,
                    paired_target=None,
                    build_args=None,
                    **kwargs) -> None:
    """Validate the Domino draft against its target and apply the contract."""
    del kwargs
    if paired_target is None:
        raise ValueError("Domino draft requires a target config")
    _validate_dimensions(config.hidden_size, config.vocab_size, paired_target)
    if (config.draft_vocab_size is not None
            and config.draft_vocab_size < config.vocab_size):
        raise ValueError(
            "Domino drafts score the candidate pool over the full vocabulary; "
            "reduced draft vocabularies are not supported")
    bundle = BundleConfig.from_pretrained(config.model_dir)
    contract = _draft_contract(bundle.root,
                               bundle.component_dict(contracts.Component.LLM))
    target_layers = _clamped_target_layers(contract,
                                           paired_target.num_hidden_layers,
                                           build_args)
    _apply_contract(config, contract, target_layers)
