# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared host-side helpers for CuTe DSL GEMM kernels.

These helpers intentionally cover only the repeated Python / CuPy interop and
AOT-export logic. The architecture-specific kernel bodies remain in their own
files because their MMA / copy / pipeline implementations differ substantially.
"""

from __future__ import annotations

import os
from typing import Tuple

import cupy as cp
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cutedsl_utils import aot_placeholders


def create_row_major_2d_tensors(
    m: int,
    n: int,
    k: int,
    *,
    fill_random: bool,
    dtype=cp.float16,
):
    """Create row-major 2D GEMM tensors.

    Returns:
      A[M, K], B[N, K], C[M, N]
    """
    if fill_random:
        a_cp = cp.random.uniform(-1, 1, (m, k)).astype(dtype)
        b_cp = cp.random.uniform(-1, 1, (n, k)).astype(dtype)
    else:
        a_cp = cp.zeros((m, k), dtype=dtype)
        b_cp = cp.zeros((n, k), dtype=dtype)
    c_cp = cp.zeros((m, n), dtype=dtype)
    return a_cp, b_cp, c_cp


def mark_2d_row_major_dynamic(tensor, *, divisibility: int = 8):
    """Mark a 2D row-major tensor as dynamic for AOT export."""
    leading_dim = 1
    stride_order = (0, 1)
    return (
        tensor.mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(mode=0, stride_order=stride_order)
        .mark_compact_shape_dynamic(mode=1, stride_order=stride_order, divisibility=divisibility)
    )


def create_fake_row_major_2d_tensor(
    *,
    dtype=cutlass.Float16,
    mode1_divisibility: int = 8,
):
    """Create a storage-free dynamic row-major tensor for AOT compilation.

    Goes through the same marker chain as the real from_dlpack path so the
    exported descriptor (dynamic shapes/strides and divisibility contracts)
    is identical to a device-tensor export.
    """
    tensor = aot_placeholders.make_compact_tensor(
        dtype, (2, 2), stride_order=(1, 0), assumed_align=16
    )
    return mark_2d_row_major_dynamic(tensor, divisibility=mode1_divisibility)


def create_row_major_3d_tensor(mode0: int, mode1: int, batch: int, *, fill_random: bool, dtype=cp.float16):
    """Create a logical (mode0, mode1, L) row-major tensor.

    Physical storage follows CUTLASS `cutlass.torch.matrix(..., is_mode0_major=False)`:
      physical storage: (L, mode0, mode1) contiguous
      logical tensor:   (mode0, mode1, L) via transpose(1, 2, 0)
    """
    base_shape = (batch, mode0, mode1)
    if fill_random:
        base = cp.random.uniform(-1, 1, size=base_shape).astype(dtype)
    else:
        base = cp.zeros(base_shape, dtype=dtype)
    return cp.transpose(base, (1, 2, 0))


def create_row_major_3d_gemm_tensors(
    m: int,
    n: int,
    k: int,
    *,
    batch: int = 1,
    fill_random: bool,
    dtype=cp.float16,
):
    """Create row-major 3D GEMM tensors.

    Returns logical tensors:
      A[M, K, L], B[N, K, L], C[M, N, L]
    """
    a_cp = create_row_major_3d_tensor(m, k, batch, fill_random=fill_random, dtype=dtype)
    b_cp = create_row_major_3d_tensor(n, k, batch, fill_random=fill_random, dtype=dtype)
    c_cp = create_row_major_3d_tensor(m, n, batch, fill_random=False, dtype=dtype)
    return a_cp, b_cp, c_cp


def mark_3d_row_major_dynamic(tensor):
    """Mark a logical (mode0, mode1, L) tensor as row-major and dynamic."""
    stride_order = (2, 0, 1)
    return (
        tensor.mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=0, stride_order=stride_order)
        .mark_compact_shape_dynamic(mode=1, stride_order=stride_order)
    )


def create_fake_row_major_3d_tensor(*, batch: int = 1, dtype=cutlass.Float16):
    """Create a storage-free logical (mode0, mode1, L) row-major tensor.

    Goes through the same marker chain as the real from_dlpack path, so like
    that path every mode (including L) becomes a runtime-dynamic shape and
    only the mode1 stride stays statically 1.
    """
    tensor = aot_placeholders.make_compact_tensor(
        dtype, (2, 2, batch), stride_order=(1, 0, 2), assumed_align=16
    )
    return mark_3d_row_major_dynamic(tensor)


def to_cute_tensor(cp_arr, *, assumed_align: int = 16):
    return from_dlpack(cp_arr, assumed_align=assumed_align)


def export_compiled_kernel(compiled_kernel, *, output_dir: str, file_name: str, function_prefix: str, tag: str):
    os.makedirs(output_dir, exist_ok=True)
    compiled_kernel.export_to_c(
        file_path=output_dir,
        file_name=file_name,
        function_prefix=function_prefix,
    )
    print(f"{tag} Exported to {output_dir}/{file_name}.h and {file_name}.o")



def create_bias_tensor(N, *, export_only: bool):
    """Create a 1D bias tensor [N] for epilogue fusion, marked dynamic."""
    if export_only:
        # Same marker chain as the real path below, for descriptor parity.
        mBias = aot_placeholders.make_compact_tensor(
            cutlass.Float16, (8,), stride_order=(0,), assumed_align=16
        )
        mBias = mBias.mark_layout_dynamic(leading_dim=0).mark_compact_shape_dynamic(
            mode=0, stride_order=(0,), divisibility=8
        )
        return mBias, None

    bias_cp = cp.random.uniform(-0.1, 0.1, (N,)).astype(cp.float16)
    mBias = from_dlpack(bias_cp, assumed_align=16)
    mBias = mBias.mark_layout_dynamic(leading_dim=0).mark_compact_shape_dynamic(
        mode=0, stride_order=(0,), divisibility=8
    )
    return mBias, bias_cp


def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in s.split(","))
    except ValueError as exc:
        raise ValueError(f"Invalid comma-separated integer list: {s!r}") from exc
