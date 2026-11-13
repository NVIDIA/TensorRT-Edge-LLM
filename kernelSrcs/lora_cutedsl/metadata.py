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
"""CuTe DSL metadata preparation for packed LoRA grouped GEMM."""

import argparse
import os
import pathlib
import time

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutedsl_utils import aot_placeholders


MAX_NUM_GROUPS = 256
THREADS = 256
RANK_ALIGNMENT = 8
DESCRIPTOR_ALIGNMENT = 16


@cute.kernel
def metadata_kernel(
    input_ptr: cute.Pointer,
    lora_a_ptr: cute.Pointer,
    lora_b_ptr: cute.Pointer,
    workspace: cute.Tensor,
    output_ptr: cute.Pointer,
    segment_offsets: cute.Tensor,
    segment_adapter_ids: cute.Tensor,
    adapter_ranks: cute.Tensor,
    adapter_scales: cute.Tensor,
    problem_shapes: cute.Tensor,
    strides: cute.Tensor,
    addresses: cute.Tensor,
    logical_k_sizes: cute.Tensor,
    group_scales: cute.Tensor,
    total_rows: cutlass.Int32,
    input_size: cutlass.Int32,
    output_size: cutlass.Int32,
    max_rank: cutlass.Int32,
    num_segments: cutlass.Int32,
    num_adapters: cutlass.Int32,
    second_stage: cutlass.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    index = cutlass.Int32(bidx * THREADS + tidx)
    rank_stride = cutlass.Int32(
        ((max_rank + RANK_ALIGNMENT - 1) // RANK_ALIGNMENT) * RANK_ALIGNMENT)

    segment = index
    if segment < num_segments:
        row_begin = segment_offsets[segment]
        row_end = segment_offsets[segment + 1]
        adapter = segment_adapter_ids[segment]
        rank = cutlass.Int32(0)
        if adapter >= 0 and adapter < num_adapters:
            candidate_rank = adapter_ranks[adapter]
            if candidate_rank > 0 and candidate_rank <= max_rank:
                rank = candidate_rank
        aligned_rank = cutlass.Int32(
            ((rank + RANK_ALIGNMENT - 1) // RANK_ALIGNMENT) * RANK_ALIGNMENT)

        problem_shapes[segment, 0] = 0
        if rank > 0:
            problem_shapes[segment, 0] = row_end - row_begin
        problem_shapes[segment, 1] = aligned_rank
        problem_shapes[segment, 2] = input_size
        if second_stage != 0:
            problem_shapes[segment, 1] = output_size
            problem_shapes[segment, 2] = aligned_rank
        problem_shapes[segment, 3] = 1
        logical_k_sizes[segment] = input_size
        if second_stage != 0:
            logical_k_sizes[segment] = rank

        strides[segment, 0, 0] = input_size
        strides[segment, 0, 1] = 1
        strides[segment, 1, 0] = 1
        strides[segment, 1, 1] = rank_stride
        strides[segment, 2, 0] = rank_stride
        strides[segment, 2, 1] = 1
        if second_stage != 0:
            strides[segment, 0, 0] = rank_stride
            strides[segment, 1, 1] = output_size
            strides[segment, 2, 0] = output_size

        row_begin_i64 = cutlass.Int64(row_begin)
        input_size_i64 = cutlass.Int64(input_size)
        output_size_i64 = cutlass.Int64(output_size)
        rank_stride_i64 = cutlass.Int64(rank_stride)
        adapter_i64 = cutlass.Int64(adapter)
        addresses[segment, 0] = (
            input_ptr + row_begin_i64 * input_size_i64).toint()
        addresses[segment, 1] = lora_a_ptr.toint()
        addresses[segment, 2] = (
            workspace.iterator + row_begin_i64 * rank_stride_i64).toint()
        group_scales[segment] = cutlass.Float32(0.0)
        if rank > 0:
            addresses[segment, 1] = (
                lora_a_ptr + adapter_i64 * input_size_i64 *
                rank_stride_i64).toint()

        if second_stage != 0:
            addresses[segment, 0] = (
                workspace.iterator + row_begin_i64 * rank_stride_i64).toint()
            addresses[segment, 1] = lora_b_ptr.toint()
            addresses[segment, 2] = (
                output_ptr + row_begin_i64 * output_size_i64).toint()
            if rank > 0:
                addresses[segment, 1] = (
                    lora_b_ptr + adapter_i64 * rank_stride_i64 *
                    output_size_i64).toint()
                group_scales[segment] = adapter_scales[adapter]


class MetadataExportWrapper:
    """Bind the pointer ABI and launch the metadata preparation kernel."""

    @cute.jit
    def wrapper(
        self,
        input_ptr: cute.Pointer,
        lora_a_ptr: cute.Pointer,
        lora_b_ptr: cute.Pointer,
        workspace_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        segment_offsets_ptr: cute.Pointer,
        segment_adapter_ids_ptr: cute.Pointer,
        adapter_ranks_ptr: cute.Pointer,
        adapter_scales_ptr: cute.Pointer,
        problem_shapes_ptr: cute.Pointer,
        strides_ptr: cute.Pointer,
        addresses_ptr: cute.Pointer,
        logical_k_sizes_ptr: cute.Pointer,
        group_scales_ptr: cute.Pointer,
        total_rows: cutlass.Int32,
        input_size: cutlass.Int32,
        output_size: cutlass.Int32,
        max_rank: cutlass.Int32,
        num_segments: cutlass.Int32,
        num_adapters: cutlass.Int32,
        second_stage: cutlass.Int32,
        work_items: cutlass.Int32,
        stream: cuda.CUstream,
    ) -> cutlass.Int32:
        rank_stride = ((max_rank + RANK_ALIGNMENT - 1) // RANK_ALIGNMENT) * RANK_ALIGNMENT
        workspace = cute.make_tensor(
            workspace_ptr,
            layout=cute.make_layout((total_rows, rank_stride),
                                    stride=(rank_stride, 1)),
        )
        segment_offsets = cute.make_tensor(
            segment_offsets_ptr,
            layout=cute.make_layout((num_segments + 1, )),
        )
        segment_adapter_ids = cute.make_tensor(
            segment_adapter_ids_ptr,
            layout=cute.make_layout((num_segments, )),
        )
        adapter_ranks = cute.make_tensor(
            adapter_ranks_ptr,
            layout=cute.make_layout((num_adapters, )),
        )
        adapter_scales = cute.make_tensor(
            adapter_scales_ptr,
            layout=cute.make_layout((num_adapters, )),
        )
        problem_shapes = cute.make_tensor(
            problem_shapes_ptr,
            layout=cute.make_layout((MAX_NUM_GROUPS, 4), stride=(4, 1)),
        )
        strides = cute.make_tensor(
            strides_ptr,
            layout=cute.make_layout((MAX_NUM_GROUPS, 3, 2), stride=(6, 2, 1)),
        )
        addresses = cute.make_tensor(
            addresses_ptr,
            layout=cute.make_layout((MAX_NUM_GROUPS, 3), stride=(3, 1)),
        )
        logical_k_sizes = cute.make_tensor(
            logical_k_sizes_ptr,
            layout=cute.make_layout((MAX_NUM_GROUPS, )),
        )
        group_scales = cute.make_tensor(
            group_scales_ptr,
            layout=cute.make_layout((MAX_NUM_GROUPS, )),
        )
        metadata_kernel(
            input_ptr,
            lora_a_ptr,
            lora_b_ptr,
            workspace,
            output_ptr,
            segment_offsets,
            segment_adapter_ids,
            adapter_ranks,
            adapter_scales,
            problem_shapes,
            strides,
            addresses,
            logical_k_sizes,
            group_scales,
            total_rows,
            input_size,
            output_size,
            max_rank,
            num_segments,
            num_adapters,
            second_stage,
        ).launch(
            grid=[cute.ceil_div(work_items, THREADS), 1, 1],
            block=[THREADS, 1, 1],
            stream=stream,
        )
        return cutlass.Int32(0)


def _make_pointer(data_type):
    return aot_placeholders.make_ptr(
        data_type, assumed_align=DESCRIPTOR_ALIGNMENT)


def export_metadata(args: argparse.Namespace) -> None:
    pointers = (
        _make_pointer(cutlass.Float16),
        _make_pointer(cutlass.Float16),
        _make_pointer(cutlass.Float16),
        _make_pointer(cutlass.Float16),
        _make_pointer(cutlass.Float16),
        _make_pointer(cutlass.Int32),
        _make_pointer(cutlass.Int32),
        _make_pointer(cutlass.Int32),
        _make_pointer(cutlass.Float32),
        _make_pointer(cutlass.Int32),
        _make_pointer(cutlass.Int32),
        _make_pointer(cutlass.Int64),
        _make_pointer(cutlass.Int32),
        _make_pointer(cutlass.Float32),
    )
    runtime_ints = tuple(aot_placeholders.runtime_int32() for _ in range(8))
    start = time.time()
    compiled = cute.compile(
        MetadataExportWrapper().wrapper,
        *pointers,
        *runtime_ints,
        aot_placeholders.make_stream(),
        options=aot_placeholders.compile_options(),
    )
    print(f"[{args.file_name}] Compilation time: {time.time() - start:.3f}s")
    os.makedirs(args.output_dir, exist_ok=True)
    compiled.export_to_c(args.output_dir, args.file_name,
                         args.function_prefix)
    artifacts = [
        pathlib.Path(args.output_dir) / f"{args.file_name}.h",
        pathlib.Path(args.output_dir) / f"{args.file_name}.o",
    ]
    missing = [
        str(path) for path in artifacts
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            f"CuTe DSL export did not produce nonempty artifacts: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the packed LoRA metadata preparation kernel")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--file_name", required=True)
    parser.add_argument("--function_prefix", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    export_metadata(parse_args())
