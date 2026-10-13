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
"""Muse-Glimmer checkpoint and component configuration.

Re-expresses, for the checkpoint-direct builder, the ``muse_glimmer`` block that
``tensorrt_edgellm/config.py`` applies on the ONNX-export path: the
qk_scale_factor softmax scale and the NoPE dual-RoPE synthesis (full-attention
layers use an identity RoPE table).
"""

import json
import math
import os

from ...core import contracts
from ...core.bundle import BundleConfig

_VISUAL_WEIGHT_PREFIXES = (
    "model.vision_tower.",
    "model.vision_adapter.",
    "model.vision_projection",
)


def prepare_root(model_dir: str, root: dict) -> dict:
    """Record whether an indexed checkpoint contains the visual component."""
    root = dict(root)
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return root
    with open(index_path) as index_file:
        weight_names = tuple(json.load(index_file).get("weight_map", ()))
    root["_muse_has_visual_weights"] = all(
        any(name.startswith(prefix) for name in weight_names)
        for prefix in _VISUAL_WEIGHT_PREFIXES)
    return root


def available_components(root: dict, registered):
    """Return only encoder components represented by this checkpoint."""
    available = set(registered)
    if (not isinstance(root.get("vision_config"), dict)
            or root.get("_muse_has_visual_weights") is False):
        available.discard(contracts.Component.VISUAL)
    return frozenset(available)


def component_config(root: dict, component: contracts.Component) -> dict:
    if component == contracts.Component.LLM:
        # ``_promote_llm_dict`` lifts the nested ``text_config`` from the root.
        return root
    if component == contracts.Component.VISUAL:
        return root.get("vision_config") or root
    raise ValueError(f"Muse-Glimmer has no {component.value} configuration")


def _prepare_assistant_config(config: dict) -> dict:
    """Normalize the DFlash draft checkpoint to the builder's contract.

    The assistant is a generic DFlash draft (standard RoPE, learned QK-norm, no
    gated attention or sandwich norm), so it skips the base transforms. It also
    stores the DFlash contract at top level and omits ``vocab_size`` (it shares
    the base tokenizer/``lm_head``): lift the contract into ``dflash_config`` for
    the core parser and surface a placeholder vocab that ``configure_draft``
    replaces with the paired base value.
    """
    config.setdefault("vocab_size", 0)
    if "dflash_config" not in config and "target_layer_ids" in config:
        dflash = {"target_layer_ids": config["target_layer_ids"]}
        if config.get("block_size") is not None:
            dflash["block_size"] = config["block_size"]
        if config.get("mask_token_id") is not None:
            dflash["mask_token_id"] = config["mask_token_id"]
        config["dflash_config"] = dflash
    return config


def prepare_text_config(config: dict, root: dict,
                        component: contracts.Component,
                        model_dir: str) -> dict:
    del root, component, model_dir
    config = dict(config)
    if str(config.get("model_type", "")) == "muse_glimmer_assistant":
        return _prepare_assistant_config(config)
    head_dim = int(
        config.get(
            "head_dim",
            int(config["hidden_size"]) // int(config["num_attention_heads"])))

    # Softmax scale = qk_scale_factor * 1/sqrt(head_dim).
    qk_scale_factor = config.get("qk_scale_factor")
    if qk_scale_factor is not None and "attention_scaling" not in config:
        config["attention_scaling"] = float(qk_scale_factor) / math.sqrt(
            head_dim)

    # Scaleless normed embedding (no sqrt(hidden) scale).
    config.setdefault("embedding_scale", 1.0)

    # NoPE via dual RoPE: sliding-attention layers use the real table, full
    # (NoPE) layers use an identity ("nope") table. Muse stores a single flat
    # ``rope_parameters`` block, so synthesize the per-attention-type pair.
    rope = config.get("rope_parameters")
    if isinstance(rope, dict) and "sliding_attention" not in rope:
        rope_theta = float(
            rope.get("rope_theta", config.get("rope_theta", 10_000.0)))
        config.setdefault("rope_theta", rope_theta)
        config["rope_parameters"] = {
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": rope_theta,
            },
            "full_attention": {
                "rope_type": "nope",
                "rope_theta": rope_theta,
            },
        }
    return config


def configure_base(config,
                   *,
                   paired_draft_dir: str = "",
                   build_args=None,
                   **kwargs) -> None:
    """Read the DFlash draft contract required by the Muse-Glimmer base graph.

    The assistant stores ``target_layer_ids`` / ``block_size`` / ``mask_token_id``
    at top level (not under ``dflash_config``) and omits ``vocab_size``, so read
    the contract directly and skip the vocab check the generic pairing applies.
    """
    del kwargs
    config.dflash_base = True
    if not paired_draft_dir:
        raise ValueError("Muse-Glimmer DFlash base requires a paired draft")
    bundle = BundleConfig.from_pretrained(paired_draft_dir)
    draft = bundle.component_dict(contracts.Component.LLM)
    dflash = draft.get("dflash_config") or draft
    target_layers = [
        int(index) for index in dflash.get("target_layer_ids", ())
    ]
    if not target_layers:
        raise ValueError(
            "Muse-Glimmer DFlash draft must provide target_layer_ids")
    if len(set(target_layers)) != len(target_layers):
        raise ValueError("Muse-Glimmer DFlash target-layer IDs must be unique")
    invalid = [
        index for index in target_layers
        if index < 0 or index >= config.num_hidden_layers
    ]
    if invalid:
        raise ValueError(
            f"Muse-Glimmer DFlash target-layer IDs outside base model: {invalid}"
        )
    draft_hidden = int(draft.get("hidden_size", 0))
    if draft_hidden != config.hidden_size:
        raise ValueError("Muse-Glimmer DFlash base/draft hidden sizes must "
                         f"match: {config.hidden_size} != {draft_hidden}")
    config.dflash_target_layer_ids = target_layers
    config.dflash_block_size = int(
        dflash.get("block_size", draft.get("block_size", 16)))
    mask_token = dflash.get("mask_token_id", draft.get("mask_token_id"))
    if mask_token is not None:
        config.dflash_mask_token_id = int(mask_token)
    config.dflash_tree_base = bool(build_args and build_args.tree_base)


def configure_draft(config, *, paired_target=None, **kwargs) -> None:
    """Validate the DFlash draft and inherit the base vocabulary.

    The assistant shares the base tokenizer, embedding and ``lm_head``, so its
    checkpoint carries no ``vocab_size``; adopt the paired base value.
    """
    del kwargs
    if paired_target is None:
        raise ValueError("Muse-Glimmer DFlash draft requires a target config")
    if config.hidden_size != paired_target.hidden_size:
        raise ValueError("Muse-Glimmer DFlash base/draft hidden sizes must "
                         f"match: {paired_target.hidden_size} != "
                         f"{config.hidden_size}")
    config.vocab_size = paired_target.vocab_size
