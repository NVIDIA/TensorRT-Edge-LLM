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

"""AOT export driver for the SM110 blockwise FP8 GEMM kernel."""

import argparse
import os

import blockwise_gemm
import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import cutlass.cute as cute
import cutlass.cute.runtime as cute_runtime

_AOT_M = 128
_AOT_N = 128
_AOT_K = 128
_STRIDE_ORDER_MKL = (2, 0, 1)


def _create_row_major_tensor(
    rows: int,
    columns: int,
    storage_dtype,
    element_type,
    *,
    row_divisibility: int,
    column_divisibility: int,
):
    """Create a logical [rows, columns, 1] tensor with a dynamic row stride."""
    storage = cp.zeros((1, rows, columns), dtype=storage_dtype)
    logical = cp.transpose(storage, (1, 2, 0))
    tensor = cute_runtime.from_dlpack(logical, assumed_align=16)
    tensor.element_type = element_type
    tensor = tensor.mark_layout_dynamic(leading_dim=1)
    tensor = tensor.mark_compact_shape_dynamic(
        mode=0,
        stride_order=_STRIDE_ORDER_MKL,
        divisibility=row_divisibility,
    )
    tensor = tensor.mark_compact_shape_dynamic(
        mode=1,
        stride_order=_STRIDE_ORDER_MKL,
        divisibility=column_divisibility,
    )
    return tensor, storage


def compile_kernel():
    """Compile one dynamic-M/N/K blockwise FP8 GEMM entry point."""
    k_blocks = _AOT_K // 128
    n_blocks = _AOT_N // 128

    xq, _xq_storage = _create_row_major_tensor(
        _AOT_M,
        _AOT_K,
        cp.uint8,
        cutlass.Float8E4M3FN,
        row_divisibility=1,
        column_divisibility=128,
    )
    wq, _wq_storage = _create_row_major_tensor(
        _AOT_N,
        _AOT_K,
        cp.uint8,
        cutlass.Float8E4M3FN,
        row_divisibility=128,
        column_divisibility=128,
    )
    y, _y_storage = _create_row_major_tensor(
        _AOT_M,
        _AOT_N,
        cp.float16,
        cutlass.Float16,
        row_divisibility=1,
        column_divisibility=128,
    )
    sx, _sx_storage = _create_row_major_tensor(
        _AOT_M,
        k_blocks,
        cp.float32,
        cutlass.Float32,
        row_divisibility=1,
        column_divisibility=1,
    )
    sw, _sw_storage = _create_row_major_tensor(
        n_blocks,
        k_blocks,
        cp.float32,
        cutlass.Float32,
        row_divisibility=1,
        column_divisibility=1,
    )

    kernel = blockwise_gemm.BlockwiseGemmKernel(
        acc_dtype=cutlass.Float32,
        mma_tiler_mn=(128, 128),
    )
    max_active_clusters = cutlass.Int32(1)
    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    return cute.compile(
        kernel,
        xq,
        wq,
        y,
        sx,
        sw,
        max_active_clusters,
        stream,
    )


def export_kernel(output_dir: str, file_name: str, function_prefix: str) -> None:
    compiled_kernel = compile_kernel()
    os.makedirs(output_dir, exist_ok=True)
    compiled_kernel.export_to_c(
        file_path=output_dir,
        file_name=file_name,
        function_prefix=function_prefix,
    )
    print(f"[{file_name}] Exported to {output_dir}/{file_name}.h and {file_name}.o")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="./fp8_blockwise_gemm_aot_artifacts")
    parser.add_argument("--file_name", default="fp8_blockwise_gemm")
    parser.add_argument("--function_prefix", default="fp8_blockwise_gemm")
    parser.add_argument(
        "--export_only",
        action="store_true",
        help="Accepted for compatibility; this dedicated driver always exports.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    export_kernel(args.output_dir, args.file_name, args.function_prefix)


if __name__ == "__main__":
    main()
