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
"""DFlash2 target/draft pairing and checkpoint contract validation.

A LiLiCorr draft is a DFlash2 draft whose ``dflash_config.projector_type``
replaces the candidate selector with the candidate-lattice head.
"""

import dataclasses

from tensorrt_edgellm.dflash import DFlashContract, DFlashVersion

from ...core import config as core_config
from ...core import contracts

_EXPECTED_CONTRACT = {
    "dflash2_conv_kernel_size": 2,
    "dflash2_conv_group_size": 16,
    "dflash2_selector_rank": 256,
    "dflash2_selector_top_k": 16,
}
_EXPECTED_DRAFT_LAYERS = 5


def component_config(root: dict, component: contracts.Component) -> dict:
    """Normalize the provider's nested DFlash2 draft configuration."""
    if component != contracts.Component.LLM:
        raise ValueError(f"DFlash2 has no {component.value} configuration")
    config = dict(root.get("transformer_layer_config") or {})
    for key, value in root.items():
        config.setdefault(key, value)
    aux_layers = [
        int(layer) for layer in root.get("aux_hidden_state_layer_ids") or []
    ]
    if any(layer <= 0 for layer in aux_layers):
        raise ValueError(
            "DFlash2 auxiliary hidden-state layer IDs must be positive")
    dflash = dict(root.get("dflash_config") or {})
    if not dflash.get("target_layer_ids"):
        dflash["target_layer_ids"] = [layer - 1 for layer in aux_layers]
    is_causal = bool(
        dflash.get(
            "is_causal",
            root.get("is_causal",
                     not bool(root.get("sliding_window_non_causal", False)))))
    config["is_causal"] = is_causal
    dflash["is_causal"] = is_causal
    for name, default in (("block_size", 8), ("mask_token_id", -1),
                          ("conv_kernel_size", 0), ("conv_group_size", 0),
                          ("selector_rank", 0), ("selector_top_k", 0)):
        dflash.setdefault(name, root.get(name, default))
    config["dflash_config"] = dflash
    return config


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


def _validate_lilicorr_draft(draft, target) -> None:
    _validate_dimensions(draft, target)
    if draft.num_hidden_layers != _EXPECTED_DRAFT_LAYERS:
        raise ValueError(
            "LiLiCorr production checkpoint requires exactly five draft "
            f"layers, got {draft.num_hidden_layers}")
    if (draft.draft_vocab_size is not None
            and draft.draft_vocab_size != draft.vocab_size):
        raise ValueError(
            "LiLiCorr candidates require the full target vocabulary: the "
            f"draft declares draft_vocab_size={draft.draft_vocab_size}, "
            f"vocab_size={draft.vocab_size}")


def _apply_lilicorr_contract(config, contract: DFlashContract) -> None:
    """Copy the LiLiCorr contract onto a base or draft configuration.

    ``DeviceConfig`` reads ``block_size`` and ``is_causal`` from keys the
    LiLiCorr checkpoint does not use (it publishes a top-level ``block_size``
    and ``dflash_config.causal``), so both roles take them from the contract.
    """
    if contract.lilicorr_hidden_size == config.hidden_size:
        raise ValueError(
            "LiLiCorr heads as wide as the draft hidden size use an identity "
            "token_proj, which is not supported")
    config.dflash_version = DFlashVersion.V2
    config.dflash_projector_type = contract.projector_type
    config.dflash_target_layer_ids = list(contract.target_layer_ids)
    config.dflash2_target_layer_ids = list(contract.target_layer_ids)
    config.dflash_block_size = contract.block_size
    config.dflash2_block_size = contract.block_size
    config.dflash_mask_token_id = contract.mask_token_id
    config.dflash2_mask_token_id = contract.mask_token_id
    config.dflash2_is_causal = contract.is_causal
    config.dflash2_conv_kernel_size = contract.conv_kernel_size
    config.dflash2_conv_group_size = contract.conv_group_size
    for field in dataclasses.fields(contract):
        if field.name.startswith("lilicorr_"):
            setattr(config, field.name, getattr(contract, field.name))


def _configure_lilicorr_base(config, draft, build_args) -> None:
    _validate_lilicorr_draft(draft, config)
    contract = core_config.draft_dflash_contract(draft.model_dir)
    invalid = [
        index for index in contract.target_layer_ids
        if index < 0 or index >= config.num_hidden_layers
    ]
    if invalid:
        raise ValueError(
            f"LiLiCorr target-layer IDs outside base model: {invalid}")
    if config.embedding_scale != 1.0:
        raise ValueError(
            "LiLiCorr drafts gather raw rows of the runtime embedding table, "
            f"but this base scales it by {config.embedding_scale}")
    if getattr(build_args, "fp8_embedding", False):
        raise ValueError(
            "--fp8-embedding cannot be combined with a LiLiCorr draft: the "
            "draft engine gathers candidate embeddings from the base's FP16 "
            "embedding table")
    _apply_lilicorr_contract(config, contract)


def configure_base(config,
                   *,
                   paired_draft_dir: str = "",
                   build_args=None,
                   **kwargs) -> None:
    """Read and validate the DFlash2 contract required by the target graph."""
    del kwargs
    if not paired_draft_dir:
        raise ValueError("DFlash2 base requires a paired draft checkpoint")
    draft = core_config.DeviceConfig.from_pretrained(paired_draft_dir)
    if draft.is_lilicorr:
        _configure_lilicorr_base(config, draft, build_args)
        return
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
    if config.is_lilicorr:
        _validate_lilicorr_draft(config, paired_target)
        _apply_lilicorr_contract(
            config, core_config.draft_dflash_contract(config.model_dir))
        # As the ONNX frontend: dflash_config wins over the top-level flag.
        dflash = config.raw_component.get("dflash_config") or {}
        config.attention_sink_bias = bool(
            dflash.get("attention_sink_bias",
                       config.raw_component.get("attention_sink_bias", False)))
        return
    _validate_contract(config)
    _validate_dimensions(config, paired_target)
