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
"""DFlash2 target/draft pairing and checkpoint contract validation."""

from tensorrt_edgellm.dflash import DFlashVersion

from ...core import config as core_config

_EXPECTED_CONTRACT = {
    "dflash2_conv_kernel_size": 2,
    "dflash2_conv_group_size": 16,
    "dflash2_selector_rank": 256,
    "dflash2_selector_top_k": 16,
}
_EXPECTED_DRAFT_LAYERS = 5


def _validate_dimensions(draft, target) -> None:
    if target.hidden_size != draft.hidden_size:
        raise ValueError("DFlash2 base/draft hidden sizes must match: "
                         f"{target.hidden_size} != {draft.hidden_size}")
    if target.vocab_size != draft.vocab_size:
        raise ValueError("DFlash2 base/draft vocab sizes must match: "
                         f"{target.vocab_size} != {draft.vocab_size}")


def _validate_contract(draft) -> None:
    if draft.dflash_version != DFlashVersion.V2:
        raise ValueError(
            "DFlash2 draft checkpoint requires architecture DFlash2DraftModel")
    if draft.dflash2_is_causal:
        raise ValueError("DFlash2 draft checkpoint requires is_causal=false")
    if not draft.dflash2_target_layer_ids:
        raise ValueError(
            "DFlash2 draft requires dflash_config.target_layer_ids")
    if (draft.num_hidden_layers != _EXPECTED_DRAFT_LAYERS
            or len(draft.dflash2_target_layer_ids) != _EXPECTED_DRAFT_LAYERS):
        raise ValueError(
            "DFlash2 production checkpoint requires exactly five draft layers "
            "and exactly five target-layer IDs")
    if len(set(draft.dflash2_target_layer_ids)) != len(
            draft.dflash2_target_layer_ids):
        raise ValueError("DFlash2 target-layer IDs must be unique")
    if draft.dflash2_mask_token_id < 0:
        raise ValueError("DFlash2 draft requires a non-negative mask_token_id")
    if draft.dflash2_block_size < 2 or draft.dflash2_block_size > 16:
        raise ValueError("DFlash2 block_size must be in [2, 16], got "
                         f"{draft.dflash2_block_size}")
    for field, expected in _EXPECTED_CONTRACT.items():
        actual = getattr(draft, field)
        if actual != expected:
            checkpoint_name = field.removeprefix("dflash2_")
            raise ValueError(
                f"DFlash2 {checkpoint_name} must be {expected}, got {actual}")


def configure_base(config,
                   *,
                   paired_draft_dir: str = "",
                   build_args=None,
                   **kwargs) -> None:
    """Read and validate the DFlash2 contract required by the target graph."""
    del build_args, kwargs
    if not paired_draft_dir:
        raise ValueError("DFlash2 base requires a paired draft checkpoint")
    draft = core_config.DeviceConfig.from_pretrained(paired_draft_dir)
    _validate_contract(draft)
    _validate_dimensions(draft, config)
    invalid = [
        index for index in draft.dflash2_target_layer_ids
        if index < 0 or index >= config.num_hidden_layers
    ]
    if invalid:
        raise ValueError(
            f"DFlash2 target-layer IDs outside base model: {invalid}")
    config.dflash_version = DFlashVersion.V2
    config.dflash_tree_base = True
    config.dflash2_target_layer_ids = list(draft.dflash2_target_layer_ids)
    config.dflash2_block_size = draft.dflash2_block_size
    config.dflash2_mask_token_id = draft.dflash2_mask_token_id
    config.dflash2_is_causal = draft.dflash2_is_causal
    config.dflash2_conv_kernel_size = draft.dflash2_conv_kernel_size
    config.dflash2_conv_group_size = draft.dflash2_conv_group_size
    config.dflash2_selector_rank = draft.dflash2_selector_rank
    config.dflash2_selector_top_k = draft.dflash2_selector_top_k
    config.dflash_target_layer_ids = list(draft.dflash2_target_layer_ids)
    config.dflash_block_size = draft.dflash2_block_size
    config.dflash_mask_token_id = draft.dflash2_mask_token_id


def configure_draft(config, *, paired_target=None, **kwargs) -> None:
    """Validate that a dedicated DFlash2 draft matches its target."""
    del kwargs
    if paired_target is None:
        raise ValueError("DFlash2 draft requires a target config")
    _validate_contract(config)
    _validate_dimensions(config, paired_target)
