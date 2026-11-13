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
"""NumPy-only conversion of dense 8-bit GPTQ checkpoint weights."""

from typing import Optional, Tuple

import numpy as np


def unpack_gptq(
    qweight: np.ndarray,
    qzeros: np.ndarray,
    *,
    num_groups: int,
    g_idx: Optional[np.ndarray] = None,
    zero_point_offset: int = 1,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return signed ``[N,K]`` weights, K permutation, and group size.

    GPTQ packs four unsigned 8-bit values in each INT32 along K. The returned
    weights represent ``q - 128`` and are ordered by quantizer group so grouped
    scales remain contiguous. Non-midpoint zero points are applied separately
    through :func:`gptq_zero_correction`.
    """
    if qweight.ndim != 2:
        raise ValueError(
            f"GPTQ INT8 qweight must be rank 2, got {qweight.shape}")
    if num_groups <= 0:
        raise ValueError(
            f"GPTQ INT8 requires a positive group count, got {num_groups}")

    qweight = qweight.astype(np.uint32, copy=False)
    in_features = int(qweight.shape[0]) * 4
    out_features = int(qweight.shape[1])
    if in_features % num_groups:
        raise ValueError(
            f"GPTQ INT8 K={in_features} is not divisible by {num_groups} groups"
        )
    group_size = in_features // num_groups

    unpacked = np.empty((in_features, out_features), dtype=np.int16)
    for byte in range(4):
        unpacked[byte::4] = ((qweight >> (8 * byte)) & 0xFF).astype(np.int16)

    if qzeros.size:
        _unpack_zeros(qzeros, num_groups, out_features)

    if g_idx is None:
        groups = np.arange(in_features, dtype=np.int32) // group_size
    else:
        groups = np.asarray(g_idx, dtype=np.int32).reshape(-1)
        if groups.size != in_features:
            raise ValueError(
                f"GPTQ INT8 g_idx must have {in_features} entries, got {groups.size}"
            )
        if np.any(groups < 0) or np.any(groups >= num_groups):
            raise ValueError("GPTQ INT8 g_idx contains an out-of-range group")
        counts = np.bincount(groups, minlength=num_groups)
        if np.any(counts != group_size):
            raise ValueError("GPTQ INT8 g_idx groups do not have uniform size")

    centered = unpacked - 128
    permutation = np.concatenate(
        [np.flatnonzero(groups == group) for group in range(num_groups)])
    weight = np.ascontiguousarray(centered[permutation].T, dtype=np.int8)
    return weight, permutation.astype(np.int64), group_size


def _unpack_zeros(qzeros: np.ndarray, num_groups: int,
                  out_features: int) -> np.ndarray:
    expected = (num_groups, (out_features + 3) // 4)
    if tuple(qzeros.shape) != expected:
        raise ValueError(
            f"GPTQ INT8 qzeros must have shape {expected}, got {qzeros.shape}")
    packed = qzeros.astype(np.uint32, copy=False)
    zeros = np.empty((num_groups, out_features), dtype=np.int16)
    for byte in range(4):
        target = zeros[:, byte::4]
        target[:] = ((packed >> (8 * byte)) & 0xFF)[:, :target.shape[1]]
    return zeros


def gptq_zero_correction(qzeros: np.ndarray,
                         scales: np.ndarray,
                         zero_point_offset: int = 1) -> Optional[np.ndarray]:
    """Return ``[N,G]`` correction weights for asymmetric GPTQ W8."""
    if qzeros.size == 0:
        return None
    scales = np.asarray(scales, dtype=np.float16)
    if scales.ndim != 2:
        raise ValueError(
            f"GPTQ INT8 scales must be rank 2, got {scales.shape}")
    groups, out_features = scales.shape
    zeros = _unpack_zeros(qzeros, groups, out_features)
    offset = 128 - zeros - int(zero_point_offset)
    correction = np.multiply(offset.astype(np.float16),
                             scales,
                             dtype=np.float16)
    if not np.any(correction):
        return None
    return np.ascontiguousarray(correction.T)
