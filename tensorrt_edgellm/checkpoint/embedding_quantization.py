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
"""FP8 embedding quantization for tensorrt_edgellm runtime sidecars."""

from __future__ import annotations

import json
import logging
import os
from typing import Tuple

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)

FP8_E4M3_MAX = 448.0


def prepare_ple_export(model,
                       model_dir: str,
                       output_dir: str,
                       fp8_ple: bool = False) -> None:
    """Select the PLE pool dtype and scale before exporting its graph."""
    if model.config.ple_enabled:
        _prepare_gemma_ple(model, model_dir, output_dir, fp8_ple)
    elif model.config.ngram_layers:
        _prepare_ngram_ple(model, model_dir, output_dir, fp8_ple)
    elif fp8_ple:
        raise ValueError("--fp8-ple requires a PLE table")


def _prepare_gemma_ple(model, model_dir: str, output_dir: str,
                       fp8_ple: bool) -> None:
    """Read the packed table's dtype and scale, quantizing it first when asked."""
    from .checkpoint_utils import _gemma_ple_checkpoint_metadata

    metadata = _gemma_ple_checkpoint_metadata(model_dir, model.config)
    if fp8_ple and metadata["dtype"] != "F8_E4M3":
        _quantize_gemma_ple(model, model_dir, output_dir,
                            metadata["source"]["tensor_key"])
        metadata = _gemma_ple_checkpoint_metadata(model._ple_checkpoint_dir,
                                                  model.config)
    model._ple_pool_scale = (metadata["scale"]
                             if metadata["dtype"] == "F8_E4M3" else None)


def _quantize_gemma_ple(model, model_dir: str, output_dir: str,
                        key: str) -> None:
    """Write the packed table as per-tensor FP8 under ``<output_dir>/ple``."""
    from tensorrt_edgellm._safetensors_io import save_file

    from .loader import _build_shard_map

    with safe_open(_build_shard_map(model_dir)[key],
                   framework="pt",
                   device="cpu") as file:
        weight = file.get_tensor(key)
    packed, scale = quantize_embedding_to_fp8(weight)
    ple_dir = os.path.join(output_dir, "ple")
    os.makedirs(ple_dir, exist_ok=True)
    save_file({
        key: packed,
        key + "_scale": scale.reshape(())
    }, os.path.join(ple_dir, "model.safetensors"))
    model._ple_checkpoint_dir = ple_dir


def _prepare_ngram_ple(model, model_dir: str, output_dir: str,
                       fp8_ple: bool) -> None:
    """The n-gram scale is a checkpoint buffer; only FP8 conversion needs preparing."""
    if fp8_ple and _ngram_table_dtype(model_dir) != "F8_E4M3":
        _quantize_ngram_table(model, model_dir, output_dir)


def _ngram_table_dtype(model_dir: str) -> str:
    from . import ple_source
    from .loader import _build_shard_map

    shards = _build_shard_map(model_dir)
    key = ple_source.ngram_table_keys(shards)[1][0]
    with safe_open(shards[key], framework="pt", device="cpu") as file:
        return file.get_slice(key).get_dtype()


def _quantize_ngram_table(model, model_dir: str, output_dir: str) -> None:
    """Write the n-gram table as FP8 shards under one F32 scale, beside its hash constants."""
    from tensorrt_edgellm._safetensors_io import save_file

    from . import ple_source
    from .loader import _build_shard_map

    shards = _build_shard_map(model_dir)
    prefix, keys = ple_source.ngram_table_keys(shards)

    def read(key: str) -> torch.Tensor:
        with safe_open(shards[key], framework="pt", device="cpu") as file:
            return file.get_tensor(key)

    packed, scale = quantize_embedding_to_fp8(
        torch.cat([read(key) for key in keys]))
    tensors = {
        f"{prefix}.shard_{index}.weight": shard
        for index, shard in enumerate(
            packed.chunk(model.config.split_ngram_parts or 1))
    }
    tensors[prefix + ".weight_scale"] = scale.reshape(1)
    for key in ple_source.ngram_hash_constant_keys(model_dir).values():
        tensors[key] = read(key)
    ple_dir = os.path.join(output_dir, "ple")
    os.makedirs(ple_dir, exist_ok=True)
    save_file(tensors, os.path.join(ple_dir, "model.safetensors"))
    for name, buffer in model.named_buffers():
        if name.endswith(".ngram_embedding.weight_scale"):
            buffer.copy_(scale.reshape(1))
    model._ple_checkpoint_dir = ple_dir


def quantize_embedding_to_fp8(
    embedding_weight: torch.Tensor, ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize token or PLE rows with one FP32 scale, without a full FP32 copy."""
    if embedding_weight.dim() != 2:
        raise ValueError(
            f"Embedding must be 2D, got {embedding_weight.dim()}D")

    scale = embedding_weight.abs().amax().float().clamp(
        min=1e-4) / FP8_E4M3_MAX
    packed = torch.empty_like(embedding_weight, dtype=torch.float8_e4m3fn)
    for start in range(0, embedding_weight.shape[0], 4096):
        rows = embedding_weight[start:start + 4096].float() / scale
        packed[start:start + 4096] = rows.clamp(
            -FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return packed, scale.reshape(1, 1)


def write_modelopt_embedding(model_dir: str, embedding: torch.nn.Embedding,
                             embedding_scale: float, output_path: str) -> bool:
    """Write a ModelOpt-quantized input table in the runtime's lookup format.

    The Edge-LLM model does not own embedding scale buffers, so read them from
    the original checkpoint rather than from its loaded embedding parameter.
    """
    from tensorrt_edgellm._safetensors_io import save_file

    from .loader import _build_shard_map

    shards = _build_shard_map(model_dir)
    candidates = [
        key for key in shards if (key == "embed_tokens.weight" or key.endswith(
            ".embed_tokens.weight")) and key + "_scale" in shards
    ]
    if not candidates:
        return False
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one quantized token embedding, found {candidates}")
    weight_key = candidates[0]

    def read(key: str) -> torch.Tensor:
        path = shards[key]
        if not path.endswith(".safetensors"):
            raise ValueError("ModelOpt embedding weights must be safetensors")
        with safe_open(path, framework="pt", device="cpu") as file:
            return file.get_tensor(key)

    weight = read(weight_key).contiguous()
    scales = read(weight_key + "_scale").contiguous()
    vocab, hidden = embedding.num_embeddings, embedding.embedding_dim
    if weight.dtype == torch.float8_e4m3fn:
        if weight.shape != (vocab, hidden):
            raise ValueError("ModelOpt FP8 embedding has an invalid shape")
        if scales.dtype != torch.float32 or scales.numel() != 1:
            raise ValueError(
                "ModelOpt FP8 embedding requires one FP32 tensor scale")
        save_file(
            {
                "embedding": weight,
                "embedding_scale":
                scales.reshape(1, 1).float() * embedding_scale
            }, output_path)
    elif weight.dtype == torch.uint8:
        if hidden % 16 or weight.shape != (vocab, hidden // 2):
            raise ValueError(
                "ModelOpt NVFP4 embedding has invalid packed shape")
        multiplier_key = weight_key + "_scale_2"
        if multiplier_key not in shards:
            raise ValueError(
                "ModelOpt NVFP4 embedding is missing its multiplier scale")
        multiplier = read(multiplier_key)
        if scales.dtype != torch.float8_e4m3fn or scales.shape != (
                vocab, hidden // 16):
            raise ValueError(
                "ModelOpt NVFP4 embedding has invalid block scales")
        if multiplier.dtype != torch.float32 or multiplier.numel() != 1:
            raise ValueError(
                "ModelOpt NVFP4 embedding multiplier must be one FP32 value")
        descriptor = {
            "logical_shape": [vocab, hidden],
            "group_size": 16,
            "model_output_scale": embedding_scale
        }
        save_file(
            {
                "embedding": weight,
                "embedding_block_scale": scales,
                "embedding_multiplier_scale": multiplier.reshape(1)
            },
            output_path,
            metadata={
                "tensorrt_edge_llm.embedding_descriptor":
                json.dumps(descriptor)
            })
    else:
        raise ValueError(
            f"Unsupported ModelOpt embedding dtype: {weight.dtype}")
    logger.info("Wrote ModelOpt %s embedding to %s", weight.dtype, output_path)
    return True
