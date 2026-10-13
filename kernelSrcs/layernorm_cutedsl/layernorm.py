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

"""AOT-export the CuTe DSL LayerNorm kernels."""

# The kernel and its required helpers are vendored from FlashInfer's
# Apache-2.0 implementation at commit d7f2c64647585b641590e309c75974519dea17db:
# https://github.com/flashinfer-ai/flashinfer/blob/d7f2c64647585b641590e309c75974519dea17db/flashinfer/norm/kernels/layernorm.py
# https://github.com/flashinfer-ai/flashinfer/blob/d7f2c64647585b641590e309c75974519dea17db/flashinfer/norm/utils.py
# Copyright 2025 FlashInfer team.

import argparse
import math
import operator
import os
import sys
from pathlib import Path
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int64

# The launch geometry lives in a sibling module with no CuTe DSL, cupy or CUDA
# imports so it can be unit-tested on a host with no GPU. Kernel scripts are
# executed directly, so add this directory before importing it.
_LAYERNORM_DIR = Path(__file__).resolve().parent
if str(_LAYERNORM_DIR) not in sys.path:
    sys.path.insert(0, str(_LAYERNORM_DIR))
if str(_LAYERNORM_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_LAYERNORM_DIR.parent))

import layernorm_config  # noqa: E402
from cutedsl_utils import aot_placeholders  # noqa: E402

COPY_BITS = 128
AOT_ROWS = 2
SUPPORTED_HIDDEN_SIZES = (4096, 4097, 5120, 7168, 8192)
assert SUPPORTED_HIDDEN_SIZES == layernorm_config.SUPPORTED_HIDDEN_SIZES


@cute.jit
def warp_reduce(val, op, width: cutlass.Constexpr[int] = 32):
    """Reduce across threads in a warp using butterfly shuffle."""
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()

    for i in cutlass.range_constexpr(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def block_reduce(
    val: Float32,
    op: Callable,
    reduction_buffer: cute.Tensor,
    init_val: Float32,
) -> Float32:
    """Reduce one value per warp across the warps covering a single row.

    ``reduction_buffer`` is ``(rows_per_block, warps_per_row)``. Each warp
    contributes only to the row it covers, so a row-block whose trailing rows
    fall past the end of the tensor cannot perturb a valid row's statistics.
    """
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx = warp_idx // warps_per_row
    col_idx = warp_idx % warps_per_row

    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()

    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return warp_reduce(block_reduce_val, op)


@cute.jit
def row_reduce_sum(
    x: cute.TensorSSA,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: cute.Tensor,
) -> Float32:
    """Reduce a row in FP32 using a ``(rows_per_block, warps_per_row)`` buffer."""
    local_val = x.reduce(
        cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
    )
    warp_width = min(threads_per_row, 32)
    warp_val = warp_reduce(local_val, operator.add, width=warp_width)
    warps_per_row = max(threads_per_row // 32, 1)

    if cutlass.const_expr(warps_per_row > 1):
        return block_reduce(warp_val, operator.add, reduction_buffer, Float32(0.0))
    return warp_val


@cute.jit
def predicate_k(tXcX: cute.Tensor, limit: int) -> cute.Tensor:
    """Create a predicate tensor for the fixed hidden-dimension bound."""
    tXpX = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tXcX, mode=[0, 1]),
                cute.size(tXcX, mode=[1]),
                cute.size(tXcX, mode=[2]),
            ),
            stride=(cute.size(tXcX, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
        for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
            tXpX[rest_v, 0, rest_k] = cute.elem_less(
                tXcX[(0, rest_v), 0, rest_k][1], limit
            )
    return tXpX


def make_tv_layout(threads_per_row: int, rows_per_block: int, vec_size: int,
                   num_vec_blocks: int):
    """Create the coalesced thread-value layout used by both schedules.

    Threads are consecutive within a row, so a thread's in-row index is
    ``tidx % threads_per_row`` and the row it covers is
    ``tidx // threads_per_row``. ``rows_per_block == 1`` reduces this to the
    single-row layout vendored from FlashInfer.
    """
    shape = (
        (threads_per_row, rows_per_block),
        (vec_size, num_vec_blocks),
    )
    stride = (
        (vec_size * rows_per_block, 1),
        (rows_per_block, rows_per_block * vec_size * threads_per_row),
    )
    return shape, stride


class LayerNormKernel:
    """Compute ``(x - mean) / sqrt(variance + eps) * gamma + beta``."""

    def __init__(
        self,
        dtype: cutlass.Numeric,
        hidden_size: int,
        target_sm: int,
        schedule: str = None,
    ):
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.target_sm = target_sm

        # Resolved from the *target* SM, never from the local device: the AOT
        # export runs on the build host, which is an x86 GPU for every
        # cross-compiled aarch64 artifact.
        self.config = layernorm_config.layernorm_config(
            target_sm, dtype.width, hidden_size, schedule=schedule
        )
        self.vec_size = self.config.vec_size
        self.copy_bits = self.config.copy_bits
        self.threads_per_row = self.config.threads_per_row
        self.num_threads = self.config.num_threads
        self.num_warps = self.config.warps_per_row
        self.num_vec_blocks = self.config.num_vec_blocks
        self.cols_per_tile = self.config.cols_per_tile
        self.rows_per_block = self.config.rows_per_block
        self.use_async_copy = self.config.use_async_copy
        self.needs_padding = self.config.needs_padding

    def _smem_size_in_bytes(self) -> int:
        # Sum and centered-square reductions each use one FP32 value per warp
        # per row, plus the staged activation tile when the schedule stages it.
        return self.config.smem_bytes

    @cute.jit
    def __call__(
        self,
        output: cute.Tensor,
        x: cute.Tensor,
        gamma: cute.Tensor,
        beta: cute.Tensor,
        rows: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = make_tv_layout(
            self.threads_per_row,
            self.rows_per_block,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)

        self.kernel(
            output,
            x,
            gamma,
            beta,
            rows,
            eps,
            enable_pdl,
            tv_layout,
            tiler_mn,
        ).launch(
            grid=[cute.ceil_div(rows, self.rows_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        output: cute.Tensor,
        x: cute.Tensor,
        gamma: cute.Tensor,
        beta: cute.Tensor,
        rows: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        hidden_size = self.hidden_size
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        num_warps = self.num_warps
        vec_size = self.vec_size
        num_vec_blocks = self.num_vec_blocks

        smem = cutlass.utils.SmemAllocator()
        if cutlass.const_expr(self.use_async_copy):
            # The staged schedule gives each thread up to ten vector blocks.
            # Staging the tile lets each pass re-read the row instead of keeping
            # every activation live across both reductions. Nothing spills
            # either way; what the re-reads buy is registers, and so resident
            # CTAs per SM.
            smem_x = smem.allocate_tensor(
                x.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
        reduction_buffer_sum = smem.allocate_tensor(
            Float32,
            cute.make_layout((rows_per_block, num_warps)),
            byte_alignment=4,
        )
        reduction_buffer_var = smem.allocate_tensor(
            Float32,
            cute.make_layout((rows_per_block, num_warps)),
            byte_alignment=4,
        )

        identity = cute.make_identity_tensor(x.shape)
        g_output = cute.local_tile(output, tiler_mn, (bidx, 0))
        g_x = cute.local_tile(x, tiler_mn, (bidx, 0))
        c_x = cute.local_tile(identity, tiler_mn, (bidx, 0))

        # Broadcast the affine vectors across the rows of the tile so they can
        # be copied with the same vectorized thread-value layout as the
        # activations instead of one indexed scalar load per element.
        row_broadcast = cute.make_layout((rows_per_block,), stride=(0,))
        g_gamma = cute.local_tile(
            cute.make_tensor(gamma.iterator,
                             cute.prepend(gamma.layout, row_broadcast)),
            tiler_mn,
            (0, 0),
        )
        g_beta = cute.local_tile(
            cute.make_tensor(beta.iterator,
                             cute.prepend(beta.layout, row_broadcast)),
            tiler_mn,
            (0, 0),
        )

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            x.element_type,
            num_bits_per_copy=self.copy_bits,
        )
        if cutlass.const_expr(self.use_async_copy):
            copy_atom_load = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                x.element_type,
                num_bits_per_copy=self.copy_bits,
            )
        else:
            copy_atom_load = copy_atom

        tiled_copy_load = cute.make_tiled_copy(copy_atom_load, tv_layout,
                                               tiler_mn)
        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler_mn)
        thread_load = tiled_copy_load.get_slice(tidx)
        thread_copy = tiled_copy.get_slice(tidx)

        thread_g_x = thread_load.partition_S(g_x)
        thread_c_x = thread_load.partition_S(c_x)
        thread_g_output = thread_copy.partition_D(g_output)
        thread_g_gamma = thread_copy.partition_S(g_gamma)
        thread_g_beta = thread_copy.partition_S(g_beta)

        thread_r_x = cute.make_fragment_like(thread_g_x)
        thread_r_gamma = cute.make_fragment_like(thread_g_gamma)
        thread_r_beta = cute.make_fragment_like(thread_g_beta)

        predicate = predicate_k(thread_c_x, limit=hidden_size)
        predicate_affine = predicate_k(thread_copy.partition_S(c_x),
                                       limit=hidden_size)
        # A row-block straddling the end of the tensor still runs every thread;
        # only the rows inside the tensor may touch global memory.
        row_in_bounds = thread_c_x[(0, 0), 0, 0][0] < rows

        if cutlass.const_expr(self.use_async_copy):
            thread_s_x = thread_load.partition_D(smem_x)
            if row_in_bounds:
                cute.copy(copy_atom_load, thread_g_x, thread_s_x,
                          pred=predicate)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.autovec_copy(thread_s_x, thread_r_x)
        else:
            thread_r_x.store(cute.zeros_like(thread_r_x, dtype=x.element_type))
            if row_in_bounds:
                cute.copy(copy_atom, thread_g_x, thread_r_x, pred=predicate)

        x_fp32 = thread_r_x.load().to(Float32)
        sum_x = row_reduce_sum(x_fp32, threads_per_row, reduction_buffer_sum)
        mean = sum_x / Float32(hidden_size)

        # Re-read the row from shared memory rather than keeping the full row
        # fragment live across the reduction. This shortens its live range and
        # reduces register pressure in the staged schedule.
        if cutlass.const_expr(self.use_async_copy):
            cute.autovec_copy(thread_s_x, thread_r_x)
            x_fp32 = thread_r_x.load().to(Float32)

        # Compute the variance as the mean of centered squares in FP32. Padding
        # lanes are explicitly masked because their zero input would otherwise
        # contribute ``mean * mean`` (the H=4097 variant exercises this path).
        diff = x_fp32 - mean
        centered_square = diff * diff
        centered_square_reg = cute.make_rmem_tensor(centered_square.shape, Float32)
        centered_square_reg.store(centered_square)

        if cutlass.const_expr(self.needs_padding):
            in_row_thread = tidx % threads_per_row
            num_elems = vec_size * num_vec_blocks
            for i in cutlass.range_constexpr(num_elems):
                vec_idx = i % vec_size
                block_idx = i // vec_size
                column = (
                    in_row_thread * vec_size
                    + vec_idx
                    + block_idx * vec_size * threads_per_row
                )
                if column >= hidden_size:
                    centered_square_reg[i] = Float32(0.0)

        sum_centered_square = row_reduce_sum(
            centered_square_reg.load(),
            threads_per_row,
            reduction_buffer_var,
        )
        variance = sum_centered_square / Float32(hidden_size)
        reciprocal_stddev = cute.math.rsqrt(variance + eps, fastmath=True)

        cute.arch.barrier()

        if cutlass.const_expr(self.use_async_copy):
            cute.autovec_copy(thread_s_x, thread_r_x)
            x_fp32 = thread_r_x.load().to(Float32)

        # Load the affine vectors only now so neither fragment stays live across
        # the variance reduction. They also stay in storage dtype until the
        # expression below rather than becoming FP32 register tensors early.
        cute.copy(copy_atom, thread_g_gamma, thread_r_gamma,
                  pred=predicate_affine)
        cute.copy(copy_atom, thread_g_beta, thread_r_beta,
                  pred=predicate_affine)

        output_fp32 = (
            (x_fp32 - mean)
            * reciprocal_stddev
            * thread_load.retile(thread_r_gamma).load().to(Float32)
            + thread_load.retile(thread_r_beta).load().to(Float32)
        )
        thread_r_output = cute.make_fragment_like(thread_g_output)
        thread_r_output.store(output_fp32.to(output.element_type))
        if row_in_bounds:
            cute.copy(copy_atom, thread_r_output, thread_g_output,
                      pred=predicate)

        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


def _create_layernorm_jit(dtype, hidden_size, target_sm, schedule=None):
    layernorm_kernel = LayerNormKernel(dtype, hidden_size, target_sm, schedule)

    @cute.jit
    def aot_adapter(
        output: cute.Tensor,
        x: cute.Tensor,
        gamma: cute.Tensor,
        beta: cute.Tensor,
        rows: Int64,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        # Keep the portable artifact independent of PDL support.
        layernorm_kernel(output, x, gamma, beta, rows, eps, False, stream)

    return aot_adapter


def _make_placeholder_tensor(shape, dtype, *, dynamic_rows):
    """Create a storage-free row-major tensor descriptor for the AOT ABI."""
    if dtype not in (cutlass.Float16, cutlass.BFloat16):
        raise ValueError(f"Unsupported LayerNorm dtype: {dtype}")
    # Default stride order: C-contiguous row-major, matching the real path.
    tensor = aot_placeholders.make_compact_tensor(dtype, shape, assumed_align=16)
    if dynamic_rows:
        # The row count is the only dynamic shape component in the AOT ABI.
        tensor = tensor.mark_compact_shape_dynamic(
            mode=0,
            stride_order=(0, 1),
            divisibility=1,
        )
    return tensor


def compile_layernorm(dtype, hidden_size, target_sm, schedule=None):
    output = _make_placeholder_tensor((AOT_ROWS, hidden_size), dtype, dynamic_rows=True)
    x = _make_placeholder_tensor((AOT_ROWS, hidden_size), dtype, dynamic_rows=True)
    gamma = _make_placeholder_tensor((hidden_size,), dtype, dynamic_rows=False)
    beta = _make_placeholder_tensor((hidden_size,), dtype, dynamic_rows=False)
    stream = aot_placeholders.make_stream()

    layernorm = _create_layernorm_jit(dtype, hidden_size, target_sm, schedule)
    return cute.compile(
        layernorm,
        output,
        x,
        gamma,
        beta,
        Int64(AOT_ROWS),
        Float32(1e-6),
        stream,
    )


def export_layernorm(
    dtype,
    hidden_size,
    target_sm,
    schedule,
    output_dir,
    file_name,
    function_prefix,
):
    compiled = compile_layernorm(dtype, hidden_size, target_sm, schedule)
    os.makedirs(output_dir, exist_ok=True)
    compiled.export_to_c(
        file_path=output_dir,
        file_name=file_name,
        function_prefix=function_prefix,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    parser.add_argument(
        "--hidden_size",
        type=int,
        choices=SUPPORTED_HIDDEN_SIZES,
        required=True,
    )
    parser.add_argument(
        "--target_sm",
        type=int,
        required=True,
        help="SM version of the board this artifact will run on (e.g. 110). "
        "Selects the compile-time launch geometry.",
    )
    parser.add_argument(
        "--schedule",
        choices=("auto", "S", "W"),
        default="auto",
        help="Override the SM-derived schedule. For validating both schedules "
        "on one board; leave unset in the build.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--file_name", required=True)
    parser.add_argument("--function_prefix", required=True)
    parser.add_argument("--export_only", action="store_true")
    args = parser.parse_args()
    if args.target_sm not in layernorm_config.SUPPORTED_TARGET_SMS:
        raise SystemExit(
            f"layernorm.py: --target_sm={args.target_sm} is not a supported "
            f"LayerNorm artifact SM "
            f"{list(layernorm_config.SUPPORTED_TARGET_SMS)}. The target SM "
            "must be passed explicitly: CUTE_DSL_ARCH is unset for every "
            "target in kernelSrcs/build_cutedsl_tarballs.sh's matrix, because "
            "cutedsl_compile_wrapper.py only sets it when --host-target is "
            "empty and default_host_target_for_arch returns a non-empty host "
            "target for both x86_64 and cross-built aarch64. A device query "
            "here would therefore resolve the build host's SM, not the "
            "target's."
        )
    return args


def main():
    args = _parse_args()
    if not args.export_only:
        raise ValueError(
            "LayerNorm currently supports AOT export only; pass --export_only."
        )
    dtype = cutlass.Float16 if args.dtype == "fp16" else cutlass.BFloat16
    export_layernorm(
        dtype,
        args.hidden_size,
        args.target_sm,
        None if args.schedule == "auto" else args.schedule,
        args.output_dir,
        args.file_name,
        args.function_prefix,
    )


if __name__ == "__main__":
    main()
