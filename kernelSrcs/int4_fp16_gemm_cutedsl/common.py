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

"""Shared host-side helpers for the CuTe DSL INT4 FP16 (W4A16) GEMM kernel.

These cover the repeated CuPy interop, dynamic-tensor marking, and AOT-export
plumbing.  The kernel body lives in ``int4_fp16_gemm_ampere.py``; the offline
weight repack and reference live in ``int4_reference.py``.

A, scales, C, and the repacked uint32 weight buffer are marked 2D row-major
(mode 1 contiguous) with mode-1 divisibility 8 (the 128-bit copy granularity);
the split-K lock array is marked 1D int32.
"""

from __future__ import annotations

import os
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutedsl_utils import aot_placeholders
from cutlass.cute.runtime import from_dlpack


def mark_row_major_2d(arr, *, assumed_align: int = 16):
    """Mark a 2D row-major tensor dynamic for AOT export.

    mode 0 (rows) divisibility 1, mode 1 (the contiguous axis) divisibility 8 so
    the exported kernel accepts any M/N/K while still permitting the 128-bit
    vectorized copies the mainloop relies on.  Used for A[M,K], scales[G,N],
    C[M,N], the repacked weight buffer QW[rows,128], and the FP32 workspace.
    """
    tensor = from_dlpack(arr, assumed_align=assumed_align)
    return (
        tensor.mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1), divisibility=8)
    )


def mark_lock_1d(arr, *, assumed_align: int = 16):
    """Mark the 1D int32 serial-split-K semaphore array (one entry per tile)."""
    tensor = from_dlpack(arr, assumed_align=assumed_align)
    return tensor.mark_compact_shape_dynamic(mode=0, stride_order=(0,), divisibility=1)


def make_row_major_2d_placeholder(dtype, *, assumed_align: int = 16):
    """Create a storage-free dynamic row-major descriptor for AOT export."""
    tensor = aot_placeholders.make_compact_tensor(
        dtype,
        (cute.sym_int32(), cute.sym_int32(divisibility=8)),
        stride_order=(1, 0),
        assumed_align=assumed_align,
    )
    return tensor


def make_lock_placeholder(*, assumed_align: int = 16):
    """Create a storage-free dynamic split-K semaphore descriptor."""
    return aot_placeholders.make_compact_tensor(
        cutlass.Int32,
        (cute.sym_int32(),),
        stride_order=(0,),
        assumed_align=assumed_align,
    )


def export_compiled_kernel(
    compiled_kernel, *, output_dir: str, file_name: str, function_prefix: str, tag: str
):
    os.makedirs(output_dir, exist_ok=True)
    compiled_kernel.export_to_c(
        file_path=output_dir,
        file_name=file_name,
        function_prefix=function_prefix,
    )
    print(f"{tag} Exported to {output_dir}/{file_name}.h and {file_name}.o")


def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in s.split(","))
    except ValueError as exc:
        raise ValueError(f"Invalid comma-separated integer list: {s!r}") from exc


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def repacked_rows(n: int, k: int, bN: int, bK: int) -> int:
    """Row count of the offline-repacked uint32 weight buffer for a CTA tile.

    Mirrors ``int4_reference.repacked_rows``; duplicated here so the
    ``--export_only`` path can size a zero-filled CuPy weight buffer without a
    Torch dependency.
    """
    k_blocks = bK // 16
    n_pairs = bN // 64
    num_n_blocks = ceil_div(n, bN)
    num_k_tiles = ceil_div(k, bK)
    return num_n_blocks * num_k_tiles * k_blocks * n_pairs
