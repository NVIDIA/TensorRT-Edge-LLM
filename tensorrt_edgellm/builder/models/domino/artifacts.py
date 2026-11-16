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
"""Domino runtime configuration and correction-head sidecar."""

import os

import numpy as np

from ...core import contracts
from ...core.artifacts.runtime_artifacts import write_runtime_artifacts
from ...core.artifacts.tensors import save_safetensors
from ...core.safetensors_np import SafetensorsStore
from ..dflash import weights

HEADS_FILE_NAME = "domino_heads.safetensors"
_EMBEDDING_SUFFIX = "embed_tokens.weight"
_TABLE_ROW_CHUNK = 8192


def _required_f16(store: SafetensorsStore, name: str, expected_shape: tuple,
                  label: str) -> np.ndarray:
    if not store.has(name):
        raise KeyError(
            f"{label} checkpoint is missing required tensor {name!r}")
    tensor = store.get_f16(name)
    if tuple(tensor.shape) != tuple(expected_shape):
        raise ValueError(f"Domino tensor {name!r} must have shape "
                         f"{tuple(expected_shape)}, got {tuple(tensor.shape)}")
    return tensor


def _target_embedding_key(store: SafetensorsStore) -> str:
    # Dense checkpoints store model.embed_tokens.weight; hybrid families nest
    # it (model.language_model.embed_tokens.weight), so match on the suffix.
    keys = sorted(key for key in store.keys()
                  if key.endswith(_EMBEDDING_SUFFIX))
    if len(keys) != 1:
        raise KeyError(
            "Domino GRU input table needs exactly one embed_tokens.weight in "
            f"the target checkpoint, found {keys}")
    return keys[0]


def gru_input_table(embedding: np.ndarray,
                    weight_ih: np.ndarray) -> np.ndarray:
    """Return ``embedding @ weight_ih.T`` as FP16, accumulated in FP32 by rows.

    The GRU input projection depends only on the token id, so the runtime
    looks the row up instead of streaming the [3G, H] weight every step.
    """
    transposed = weight_ih.astype(np.float32).T
    rows = [(embedding[start:start + _TABLE_ROW_CHUNK].astype(np.float32)
             @ transposed).astype(np.float16)
            for start in range(0, embedding.shape[0], _TABLE_ROW_CHUNK)]
    return np.ascontiguousarray(np.concatenate(rows, axis=0))


def _write_heads_sidecar(config, args, engine_dir: str) -> None:
    output_dir = contracts.component_spec(
        args.resolved_component).output_dir(engine_dir)
    hidden = config.hidden_size
    gru_hidden = config.domino_gru_hidden_dim
    proj_dim = config.domino_proj_dim
    with SafetensorsStore(args.model_dir) as store:
        weight_ih = _required_f16(store, "prefix_gru.weight_ih_l0",
                                  (3 * gru_hidden, hidden), "Domino draft")
        weight_hh = _required_f16(store, "prefix_gru.weight_hh_l0",
                                  (3 * gru_hidden, gru_hidden), "Domino draft")
        embed_proj = _required_f16(store, "embed_proj.0.weight",
                                   (proj_dim, hidden + gru_hidden),
                                   "Domino draft")
    with SafetensorsStore(args.target_model_dir) as target:
        key = _target_embedding_key(target)
        embedding = target.get_f16(key)
    if embedding.ndim != 2 or embedding.shape[1] != hidden:
        raise ValueError(
            f"Target {key!r} must have shape [vocab, {hidden}], got "
            f"{tuple(embedding.shape)}")
    if embedding.shape[0] < config.vocab_size:
        raise ValueError(
            f"Target embedding has {embedding.shape[0]} rows, fewer than the "
            f"Domino vocabulary {config.vocab_size}")

    save_safetensors(
        os.path.join(output_dir, HEADS_FILE_NAME), {
            "gru_input_table": gru_input_table(embedding, weight_ih),
            "gru_weight_hh": weight_hh,
            "proj_state_weight": np.ascontiguousarray(embed_proj[:, hidden:]),
        })


def write_artifacts(bundle, config, args, engine_dir: str) -> None:
    del bundle
    if config is None:
        raise ValueError("Domino artifacts require an LLM configuration")
    write_runtime_artifacts(config,
                            args,
                            engine_dir,
                            weight_conversion=weights)
    if args.resolved_spec_role == contracts.SpecRole.DRAFT:
        _write_heads_sidecar(config, args, engine_dir)
