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
"""Checkpoint bindings for static TensorRT weight inputs."""

import json
import logging
import os
from collections import Counter
from typing import Sequence

from ..safetensors_np import SafetensorsStore
from .embeddings import (embedding_binding, externalizes_embedding,
                         externalizes_ple, ple_embedding_binding)

logger = logging.getLogger(__name__)

_STORAGE_ALIAS_FIELD = "storage_alias_of"
_CONTENT_IDENTITY_TENSOR_LIMIT = 16


def _materialization_contract(binding: dict) -> dict:
    ignored = {
        "engine_name",
        "role",
        "embedding_scale",
        _STORAGE_ALIAS_FIELD,
    }
    contract = {
        key: value
        for key, value in binding.items() if key not in ignored
    }
    contract.setdefault("checkpoint_source", "component")
    contract.setdefault("source_layout", "plugin")
    return contract


def _share_tied_embedding(bindings: list[dict],
                          tie_word_embeddings: bool) -> None:
    """Alias a runtime embedding to an identical engine weight input."""
    if not tie_word_embeddings:
        return
    candidates = [
        binding for binding in bindings if binding.get("role") not in
        ("embedding", "ple_embedding") and _STORAGE_ALIAS_FIELD not in binding
        and float(binding.get("embedding_scale", 1.0)) == 1.0
    ]
    for embedding in bindings:
        if embedding.get("role") != "embedding":
            continue
        if float(embedding.get("embedding_scale", 1.0)) != 1.0:
            continue
        contract = _materialization_contract(embedding)
        matches = [
            candidate for candidate in candidates
            if _materialization_contract(candidate) == contract
        ]
        if not matches:
            continue
        matches.sort(key=lambda binding: (
            not str(binding["engine_name"]).endswith("lm_head.weight"),
            str(binding["engine_name"]),
        ))
        target = matches[0]["engine_name"]
        embedding[_STORAGE_ALIAS_FIELD] = target
        logger.info("Sharing tied embedding storage with %s", target)


def checkpoint_identity(bindings: Sequence[dict],
                        component_dir: str,
                        target_dir: str = "") -> dict:
    """Describe the checkpoint provider contract enforced by the runtime."""
    keys_by_source = {"component": set(), "target": set()}
    for binding in bindings:
        source = binding.get("checkpoint_source", "component")
        if source not in keys_by_source:
            raise ValueError(f"unsupported checkpoint source {source!r}")
        keys_by_source[source].update(binding.get("checkpoint_keys", ()))

    source_dirs = {
        "component": component_dir,
        "target": target_dir,
    }
    sources = {}
    for source, keys in keys_by_source.items():
        if not keys:
            continue
        source_dir = source_dirs[source]
        if not source_dir:
            raise ValueError(
                f"checkpoint bindings require a {source} checkpoint")
        with SafetensorsStore(source_dir) as store:
            available = [key for key in sorted(keys) if store.has(key)]
            if len(available) <= _CONTENT_IDENTITY_TENSOR_LIMIT:
                sampled = set(available)
            else:
                last = len(available) - 1
                sampled = {
                    available[index * last //
                              (_CONTENT_IDENTITY_TENSOR_LIMIT - 1)]
                    for index in range(_CONTENT_IDENTITY_TENSOR_LIMIT)
                }
                sampled.update(
                    key for binding in bindings
                    if binding.get("checkpoint_source", "component") == source
                    and binding.get("role") == "embedding"
                    for key in binding.get("checkpoint_keys", ())
                    if key in available)
            identities = {
                key:
                store.tensor_identity(
                    key, sample_bytes=(16 if key in sampled else 0))
                for key in available
            }
            files = store.checkpoint_files(available)
        if not identities:
            raise ValueError(
                f"no {source} checkpoint tensors were available to identify")
        sources[source] = {
            "build_source": os.path.realpath(source_dir),
            "files": files,
            "tensors": identities,
        }
    return {
        "version": 1,
        "sources": sources,
    }


def checkpoint_weight_bindings(args, cfg, bindings: Sequence[dict],
                               weights) -> list[dict]:
    """Complete the engine's checkpoint-backed weight bindings.

    The lowering registers one binding per engine input; the embedding table
    is not an engine input, so it is appended here when the runtime should
    read it from the checkpoint instead of a sidecar.
    """
    bindings = list(bindings)
    has_engine_embedding = any(
        binding.get("role") == "embedding" for binding in bindings)
    if (cfg is not None and externalizes_embedding(args, weights.conversion)
            and not has_engine_embedding
            and not weights.is_nvfp4("model.embed_tokens")):
        # NVFP4 embeddings are materialized as FP16 runtime artifacts.
        bindings.append(embedding_binding(weights, cfg))
    if cfg is not None and externalizes_ple(args, cfg):
        bindings.append(ple_embedding_binding(weights, cfg))
    _share_tied_embedding(bindings, bool(cfg and cfg.tie_word_embeddings))
    source_layouts = Counter(
        binding.get("source_layout", "plugin") for binding in bindings)
    destination_types = Counter(binding["dtype"] for binding in bindings)
    layout_summary = ", ".join(
        f"{layout}={count}"
        for layout, count in sorted(source_layouts.items()))
    type_summary = ", ".join(
        f"{dtype}={count}"
        for dtype, count in sorted(destination_types.items()))
    logger.info(
        "Recorded %d runtime checkpoint binding(s) (source layouts: %s; "
        "destination dtypes: %s)", len(bindings), layout_summary, type_summary)
    return bindings


def patch_external_weight_config(config_path: str,
                                 bindings: Sequence[dict],
                                 identity: dict,
                                 checkpoint_dir: str = "") -> None:
    """Publish checkpoint-backed weights in a model-owned runtime config."""
    if not bindings:
        return
    if not identity:
        raise ValueError(
            "checkpoint-backed runtime config requires a checkpoint identity")
    with open(config_path) as config_file:
        config = json.load(config_file)
    config["checkpoint_weight_bindings"] = list(bindings)
    config["checkpoint_identity"] = identity
    if checkpoint_dir:
        config["checkpoint_dir"] = checkpoint_dir
    else:
        config.pop("checkpoint_dir", None)
    config.pop("external_weight_files", None)
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=2)
