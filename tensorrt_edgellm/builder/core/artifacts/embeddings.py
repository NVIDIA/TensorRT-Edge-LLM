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
"""Embedding artifact writers."""

import logging
import os
import shutil

import numpy as np

from .. import contracts, numpy_dtypes
from ..config import DeviceConfig
from ..safetensors_np import SafetensorsStore, load_safetensors_tensor
from ..weight_policy import (CHECKPOINT_BINDING_ENGINE_EMBEDDING,
                             CHECKPOINT_BINDING_ROLE_EMBEDDING)
from ..weights import Weights
from .tensors import save_safetensors

EMBEDDING_KEY = "model.embed_tokens.weight"

logger = logging.getLogger(__name__)


def externalizes_embedding(args, weight_conversion) -> bool:
    """Whether the runtime owns this component's embedding artifact.

    Components that pull embeddings from a second checkpoint keep their
    sidecar: the runtime only opens one ``--checkpointDir``.
    """
    if not args.weight_policy.externalizes_embedding:
        return False
    if getattr(weight_conversion, "runtime_embedding_model_dir",
               None) is not None:
        return False
    externalizes = getattr(weight_conversion, "externalizes_runtime_embedding",
                           None)
    if externalizes is not None and not externalizes(args):
        return False
    writes_embedding = getattr(weight_conversion, "writes_runtime_embedding",
                               None)
    return writes_embedding is None or writes_embedding(args)


def write_cached_draft_embedding(output_dir: str, draft_model_dir: str,
                                 mask_token_id: int,
                                 embedding_scale: float) -> None:
    """Fold a cached-draft trained mask-token row into the base embedding."""
    with SafetensorsStore(draft_model_dir) as draft:
        draft_key = next(
            (key
             for key in ("embed_tokens.weight", "model.embed_tokens.weight")
             if draft.has(key)), None)
        if draft_key is None:
            logger.info("Cached draft shares the base embedding table")
            return
        draft_shape = draft.shape(draft_key)
        draft_row = draft.get_f16_row(draft_key, mask_token_id)

    path = os.path.join(output_dir, "embedding.safetensors")
    weight = load_safetensors_tensor(path, "embedding")
    if weight.dtype != np.float16:
        raise TypeError("cached-draft embedding fold requires FP16 embedding")
    if weight.ndim != 2 or not 0 <= mask_token_id < weight.shape[0]:
        raise ValueError(
            f"mask_token_id {mask_token_id} is outside embedding shape "
            f"{weight.shape}")
    if draft_shape != weight.shape:
        raise ValueError(
            f"cached draft embedding shape {draft_shape} does not match "
            f"base embedding shape {weight.shape}")

    weight[mask_token_id] = np.asarray(draft_row * np.float16(embedding_scale),
                                       dtype=np.float16)
    temporary_path = path + ".tmp"
    save_safetensors(temporary_path, {"embedding": weight})
    os.replace(temporary_path, path)


def embedding_binding(weights: Weights, cfg: DeviceConfig) -> dict:
    """Checkpoint binding standing in for ``embedding.safetensors``."""
    key = weights.find(EMBEDDING_KEY)
    binding = {
        "engine_name": CHECKPOINT_BINDING_ENGINE_EMBEDDING,
        "role": CHECKPOINT_BINDING_ROLE_EMBEDDING,
        "checkpoint_keys": [key],
        "source_layout": "fp16",
        # This describes the runtime tensor, not the provider payload. The
        # checkpoint reader converts BF16/F32 embeddings to FP16 while loading.
        "dtype": "F16",
        "shape": [int(dim) for dim in weights.store.shape(key)],
        "embedding_scale": float(np.float16(cfg.embedding_scale)),
    }
    locations = weights.checkpoint_locations([key])
    if locations:
        binding["checkpoint_locations"] = locations
    return binding


def is_modelopt_fp8_embedding(weights: Weights) -> bool:
    """Return whether the canonical input table uses ModelOpt FP8."""
    key = weights.find(EMBEDDING_KEY)
    return weights.store.dtype(key) == "F8_E4M3"


def runtime_embedding_binding(args, cfg, weights):
    """Describe a checkpoint-backed runtime embedding, excluding sidecars."""
    if (cfg is not None and externalizes_embedding(args, weights.conversion)
            and not weights.is_nvfp4("model.embed_tokens")
            and not is_modelopt_fp8_embedding(weights)):
        return embedding_binding(weights, cfg)
    return None


def ple_pool_metadata(weights: Weights, cfg: DeviceConfig) -> dict:
    """Describe PLE checkpoint rows without materializing the table."""
    from tensorrt_edgellm.checkpoint.ple_source import \
        make_gemma_ple_source_metadata

    key = weights.find("model.embed_tokens_per_layer.weight")
    weight_scale = None
    if weights.store.dtype(key) == "F8_E4M3":
        weight_scale = weights.f32(
            "model.embed_tokens_per_layer.weight_scale").item()
    return make_gemma_ple_source_metadata(
        key, weights.store.tensor_identity(key),
        cfg.vocab_size_per_layer_input,
        cfg.num_hidden_layers * cfg.hidden_size_per_layer_input,
        cfg.hidden_size_per_layer_input,
        weights.store.checkpoint_location(key), weight_scale)


def write_embedding(weights: Weights, cfg: DeviceConfig, args,
                    engine_dir: str) -> None:
    path = os.path.join(engine_dir, "embedding.safetensors")
    if is_modelopt_fp8_embedding(weights):
        key = weights.find(EMBEDDING_KEY)
        scale_key = key + "_scale"
        if not weights.store.has(scale_key):
            raise ValueError(f"Missing ModelOpt embedding scales: {scale_key}")
        shape = weights.store.shape(key)
        if (len(shape) != 2 or weights.store.dtype(scale_key) != "F32"
                or int(np.prod(weights.store.shape(scale_key))) != 1):
            raise ValueError("ModelOpt FP8 embedding requires one FP32 scale")
        scales = weights.store.get_f32(scale_key).reshape(1, 1)
        save_safetensors(
            path, {
                "embedding": weights.store.get_numpy(key),
                "embedding_scale":
                scales * float(np.float32(cfg.embedding_scale)),
            }, {"embedding": "F8_E4M3"})
        return
    # Runtime embeddings are FP16, including dequantized NVFP4 checkpoints.
    if weights.is_nvfp4("model.embed_tokens"):
        weight = weights.linear_fp16("model.embed_tokens")[0]
    else:
        weight = weights.f16("model.embed_tokens.weight")
    if args.fp8_embedding:
        scales = np.array([[max(float(np.max(np.abs(weight))), 1e-4) / 448.0]],
                          dtype=np.float32)
        encoded = numpy_dtypes.f32_to_fp8_e4m3_bytes(
            weight.astype(np.float32) / scales)
        save_safetensors(
            path, {
                "embedding":
                encoded,
                "embedding_scale":
                scales.astype(np.float32) *
                float(np.float32(cfg.embedding_scale)),
            }, {"embedding": "F8_E4M3"})
    else:
        weight = np.ascontiguousarray(weight * np.float16(cfg.embedding_scale))
        save_safetensors(path, {"embedding": weight})


def copy_vocab_artifacts(args, engine_dir: str) -> None:
    source_dir = (args.draft_reduced_vocab_dir if args.resolved_spec_role
                  == contracts.SpecRole.DRAFT else args.reduced_vocab_dir)
    if not source_dir:
        return
    draft = args.resolved_spec_role == contracts.SpecRole.DRAFT
    destination_map = ("draft_vocab_map.safetensors"
                       if draft else "vocab_map.safetensors")
    source_map = os.path.join(source_dir, "vocab_map.safetensors")
    vocab_map = load_safetensors_tensor(source_map,
                                        "vocab_map").astype(np.int32)
    save_safetensors(os.path.join(engine_dir, destination_map),
                     {"vocab_map": vocab_map})
    source_info = os.path.join(source_dir, "reduced_vocab.json")
    if os.path.isfile(source_info):
        shutil.copy2(
            source_info,
            os.path.join(
                engine_dir,
                "draft_reduced_vocab.json" if draft else "reduced_vocab.json"))
