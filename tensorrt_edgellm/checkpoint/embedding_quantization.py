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
"""Embedding quantization for tensorrt_edgellm runtime sidecars."""

from __future__ import annotations

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)

FP8_E4M3_MAX = 448.0
FP8_EMBEDDING_BLOCK_SIZE = 128
INT8_SYMMETRIC_MAX = 127.0
# Rows are quantized in chunks of this many elements so the FP32 working copy
# stays bounded (64 MiB) for 262k-row vocabularies.
INT8_QUANTIZATION_CHUNK_ELEMENTS = 16 * 1024 * 1024
INT8_SIDECAR_VERSION = "1"
# Token embedding: one scale per row. Gemma 4 PLE: one scale per row and
# per contiguous column group (one group per layer input).
INT8_SIDECAR_FORMATS = {
    "embedding": "int8_symmetric_per_row",
    "ple_embedding": "int8_symmetric_column_groups_per_row",
}


def int8_sidecar_metadata(tensor_role: str) -> dict[str, str]:
    """Return the versioned safetensors metadata for an INT8 sidecar."""
    if tensor_role not in INT8_SIDECAR_FORMATS:
        raise ValueError(
            f"Unsupported INT8 sidecar tensor role: {tensor_role!r}")
    return {
        "trt_edge_llm_quantization": INT8_SIDECAR_FORMATS[tensor_role],
        "trt_edge_llm_quantization_version": INT8_SIDECAR_VERSION,
        "trt_edge_llm_tensor_role": tensor_role,
    }


def quantize_embedding_to_int8(
    embedding_weight: torch.Tensor,
    *,
    num_groups: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize an embedding table to symmetric INT8 with FP32 scales.

    Each row is split into ``num_groups`` contiguous column groups and every
    group gets its own scale, so an outlier in one group does not reduce the
    precision of the others. With one group the scales are ``[vocab]``;
    otherwise ``[vocab, num_groups]``. The payload uses ``[-127, 127]`` so
    zero stays exact and the runtime dequantizes as
    ``fp16(int8_value * scale)``. An all-zero group gets scale 1.0 and an
    all-zero payload.
    """
    if embedding_weight.dim() != 2:
        raise ValueError(
            f"Embedding must be 2D, got {embedding_weight.dim()}D")
    if embedding_weight.shape[0] <= 0 or embedding_weight.shape[1] <= 0:
        raise ValueError(
            f"Embedding dimensions must be positive, got {list(embedding_weight.shape)}"
        )
    if not embedding_weight.is_floating_point():
        raise ValueError(
            f"Embedding must have a floating-point dtype, got {embedding_weight.dtype}"
        )
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}")
    if embedding_weight.shape[1] % num_groups != 0:
        raise ValueError(
            f"Embedding width {embedding_weight.shape[1]} must be divisible "
            f"by num_groups {num_groups}")

    vocab_size, width = embedding_weight.shape
    rows_per_chunk = max(1, INT8_QUANTIZATION_CHUNK_ELEMENTS // width)
    embedding_int8 = torch.empty_like(embedding_weight, dtype=torch.int8)
    scales = torch.empty((vocab_size, num_groups),
                         dtype=torch.float32,
                         device=embedding_weight.device)

    for row_start in range(0, vocab_size, rows_per_chunk):
        row_end = min(vocab_size, row_start + rows_per_chunk)
        weight_fp32 = embedding_weight[row_start:row_end].float()
        if not torch.isfinite(weight_fp32).all():
            raise ValueError("Embedding contains non-finite values")

        grouped = weight_fp32.view(row_end - row_start, num_groups,
                                   width // num_groups)
        group_amax = grouped.abs().amax(dim=2)
        group_scales = torch.where(group_amax > 0,
                                   group_amax / INT8_SYMMETRIC_MAX,
                                   torch.ones_like(group_amax))
        quantized = torch.round(grouped / group_scales.unsqueeze(2))
        embedding_int8[row_start:row_end] = quantized.clamp(
            -INT8_SYMMETRIC_MAX,
            INT8_SYMMETRIC_MAX).to(torch.int8).view(row_end - row_start, width)
        scales[row_start:row_end] = group_scales

    if num_groups == 1:
        scales = scales[:, 0].contiguous()
    logger.info("Quantized embedding to INT8: [%d, %d], scales: %s",
                vocab_size, width, list(scales.shape))
    return embedding_int8, scales


def quantize_embedding_to_fp8(
    embedding_weight: torch.Tensor,
    block_size: int = FP8_EMBEDDING_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize an embedding table to FP8 E4M3 with per-row block scales."""
    if embedding_weight.dim() != 2:
        raise ValueError(
            f"Embedding must be 2D, got {embedding_weight.dim()}D")

    vocab_size, hidden_size = embedding_weight.shape
    if hidden_size % block_size != 0:
        raise ValueError(
            f"Hidden size {hidden_size} must be divisible by block size {block_size}"
        )

    num_groups = hidden_size // block_size
    weight_fp32 = embedding_weight.float()
    weight_reshaped = weight_fp32.view(vocab_size, num_groups, block_size)
    amax = weight_reshaped.abs().amax(dim=-1).clamp(min=1e-4)
    scales = amax / FP8_E4M3_MAX
    quantized = weight_reshaped / scales.unsqueeze(-1)
    quantized = quantized.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    embedding_fp8 = quantized.view(vocab_size,
                                   hidden_size).to(torch.float8_e4m3fn)

    logger.info("Quantized embedding to FP8: [%d, %d], scales: [%d, %d]",
                vocab_size, hidden_size, vocab_size, num_groups)
    return embedding_fp8, scales
