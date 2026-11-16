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

from dataclasses import replace

from tensorrt_edgellm.dflash import (dense_last_layer_modules,
                                     resolve_dflash_contract)

from ...core import contracts, quantization
from ...core.bundle import BundleConfig
from ..gemma4.configuration import normalize_block_draft_config


def component_config(root: dict, component: contracts.Component) -> dict:
    if component != contracts.Component.LLM:
        raise ValueError(f"DFlash has no {component.value} configuration")
    config = dict(root.get("transformer_layer_config") or {})
    for key, value in root.items():
        config.setdefault(key, value)
    aux_layers = [
        int(layer) for layer in root.get("aux_hidden_state_layer_ids") or []
    ]
    if any(layer <= 0 for layer in aux_layers):
        raise ValueError(
            "DFlash auxiliary hidden-state layer IDs must be positive")
    dflash = {
        # Speculators/vLLM IDs name the state entering a decoder layer;
        # Edge-LLM captures the equivalent state after the preceding layer.
        "target_layer_ids": [layer - 1 for layer in aux_layers],
        "block_size": root.get("block_size", 8),
    }
    if root.get("mask_token_id") is not None:
        dflash["mask_token_id"] = root["mask_token_id"]
    if (root.get("dflash_config") or {}).get("causal_head") is not None:
        dflash["causal_head"] = root["dflash_config"]["causal_head"]
    config["dflash_config"] = dflash
    return config


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
    resolve_dflash_contract(bundle.root, draft)
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
    # target_layer_ids index the full base stack, so --num-decoder-layer pushes them
    # out of range. Clamp rather than reject: a duplicate only degrades what the
    # draft head can propose, it does not break the pipeline.
    num_decoder_layers = getattr(build_args, "num_decoder_layers", 0) or 0
    if num_decoder_layers:
        target_layers = [
            min(index, num_decoder_layers - 1) for index in target_layers
        ]
    if len(set(target_layers)) != len(
            target_layers) and not num_decoder_layers:
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
    config.dflash_causal_head = bool(dflash.get("causal_head", False))
    if (build_args and build_args.spec_type == "jetspec"
            and not config.dflash_causal_head):
        raise ValueError(
            "JetSpec draft requires dflash_config.causal_head=true")


def configure_draft(config,
                    *,
                    paired_target=None,
                    build_args=None,
                    **kwargs) -> None:
    """Validate that the dedicated DFlash draft matches its target."""
    resolve_dflash_contract(config.raw_root, config.raw_component)
    if paired_target is None:
        raise ValueError("DFlash draft requires a target config")
    _validate_dimensions(config, paired_target)
    # DeepSpec checkpoints publish the DFlash contract at the top level.
    raw = config.raw_component
    if (raw.get("dflash_config") is None
            and raw.get("target_layer_ids") is not None):
        config.dflash_target_layer_ids = [
            int(index) for index in raw["target_layer_ids"]
        ]
        config.dflash_block_size = int(
            raw.get("block_size", config.dflash_block_size))
        config.dflash_mask_token_id = int(
            raw.get("mask_token_id", config.dflash_mask_token_id))
    if not config.dflash_target_layer_ids:
        raise ValueError("DFlash draft requires target-layer IDs")
    if (config.spec_decode_type == "dflash"
            and getattr(build_args, "quantize_activations", True)
            and config.quant.quant_type == quantization.QUANT_NVFP4_A16):
        # The checkpoint's input scales decide: modules without one fall back
        # to W4A16, whatever the declared algorithm says.
        config.quant = replace(config.quant,
                               quant_type=quantization.QUANT_NVFP4)
    if (paired_target.model_type.startswith("gemma4")
            and config.model_type == "qwen3"):
        config.quant = replace(
            config.quant,
            excluded=config.quant.excluded +
            tuple(dense_last_layer_modules(config.num_hidden_layers)))
    config.dflash_block_size = int(
        (config.raw_component.get("dflash_config") or {}).get(
            "block_size",
            config.raw_component.get("block_size", config.dflash_block_size)))
    if (config.spec_decode_type == "jetspec"
            and not config.dflash_causal_head):
        raise ValueError(
            "JetSpec draft requires dflash_config.causal_head=true")
    if (config.draft_vocab_size is not None
            and config.draft_vocab_size < config.vocab_size):
        config.reduced_vocab_size = config.draft_vocab_size
    normalize_block_draft_config(config)
    # The runtime cross-checks the draft engine's own recorded target_layer_ids
    # against the base engine's layer count (validateCachedDraftTargetLayerIds),
    # so this side needs the same --num-decoder-layer clamp as configure_base --
    # otherwise the draft's config.json still carries the untruncated indices.
    num_decoder_layers = getattr(build_args, "num_decoder_layers", 0) or 0
    if num_decoder_layers:
        config.dflash_target_layer_ids = [
            min(index, num_decoder_layers - 1)
            for index in config.dflash_target_layer_ids
        ]
