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

"""AOT-export the CuTe DSL RMSNorm kernels."""

# The WBC1 kernel and its required helpers are vendored from FlashInfer's
# Apache-2.0 implementation at commit b065838a4710b527d6c73ab7d95af161e83aee1a:
# https://github.com/flashinfer-ai/flashinfer/blob/b065838a4710b527d6c73ab7d95af161e83aee1a/flashinfer/norm/kernels/rmsnorm.py
# Copyright 2025 FlashInfer team.

import argparse
import math
import operator
import os
import re
from typing import Callable

import cuda.bindings.driver as cuda
import cupy
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

COPY_BITS = 128
NUM_THREADS = 128
WARP_SIZE = 32
NUM_WARPS = NUM_THREADS // WARP_SIZE
LEGACY_SHARED_MEMORY_BYTES = 128
AOT_ROWS = 2
SUPPORTED_HIDDEN_SIZES = (4096, 5120, 7168, 8192)


def _target_sm() -> int:
    """Resolve the requested AOT architecture without depending on Torch."""
    arch = os.environ.get("CUTE_DSL_ARCH", "")
    match = re.search(r"sm_?(\d+)", arch)
    if match is not None:
        return int(match.group(1))
    props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.Device().id)
    return int(props["major"]) * 10 + int(props["minor"])


def _shared_memory_per_block_optin() -> int:
    """Return the active compiler device's opt-in shared-memory limit."""
    props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.Device().id)
    return int(props.get("sharedMemPerBlockOptin", props["sharedMemPerBlock"]))


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> Int32:
    """Map smem pointer to address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_shared_remote(
    val: Float32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store Float32 value to shared memory on a remote CTA in the cluster."""
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    llvm.inline_asm(
        None,
        [remote_smem_ptr_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];",
        "r,f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord, *, loc=None, ip=None) -> cute.Pointer:
    """Get pointer to element at coordinate in tensor."""
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@cute.jit
def warp_reduce(val, op, width: cutlass.Constexpr[int] = 32):
    """Reduce across threads in a warp using butterfly shuffle."""
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
        return val


@cute.jit
def block_reduce_multirow(
    val: Float32,
    op: Callable,
    reduction_buffer: cute.Tensor,
    init_val: Float32,
) -> Float32:
    """Block reduction with 2D buffer (rows_per_block, warps_per_row).

    Each warp writes its partial sum to the row it belongs to, then
    lane 0..warps_per_row-1 read back and do a final warp reduction.
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
def cluster_reduce_multirow(
    val: Float32,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr,
    cluster_n: cutlass.Constexpr[int],
    init_val: Float32,
) -> Float32:
    """Cluster reduction across multiple CTAs using mbarrier.

    reduction_buffer has shape (rows_per_block, (warps_per_row, cluster_n)).
    Each warp sends its partial result to all CTAs in the cluster via
    st.async.shared::cluster, then every CTA reduces the collected values.
    """
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()

    rows_per_block = reduction_buffer.shape[0]
    warps_per_row = reduction_buffer.shape[1][0]

    row_idx = warp_idx // warps_per_row
    col_idx = warp_idx % warps_per_row

    if warp_idx == 0:
        with cute.arch.elect_one():
            num_warps = rows_per_block * warps_per_row
            expected_bytes = num_warps * cluster_n * 4
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, expected_bytes)

    if lane_idx < cluster_n:
        store_shared_remote(
            val,
            elem_pointer(reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))),
            mbar_ptr,
            peer_cta_rank_in_cluster=lane_idx,
        )

    cute.arch.mbarrier_wait(mbar_ptr, phase=0)

    num_total = warps_per_row * cluster_n
    num_iter = cute.ceil_div(num_total, 32)

    block_reduce_val = init_val
    for i in cutlass.range_constexpr(num_iter):
        idx = lane_idx + i * 32
        if idx < num_total:
            block_reduce_val = op(block_reduce_val, reduction_buffer[row_idx, idx])

    return warp_reduce(block_reduce_val, op)


@cute.jit
def row_reduce_sum_multirow(
    x: cute.TensorSSA,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: cute.Tensor,
    mbar_ptr,
    cluster_n: cutlass.Constexpr[int],
) -> Float32:
    """Row reduction for sum with optional cluster support.

    When cluster_n == 1, uses block-level reduction with 2D buffer
    (rows_per_block, warps_per_row). When cluster_n > 1, uses cross-CTA
    cluster reduction with hierarchical buffer
    (rows_per_block, (warps_per_row, cluster_n)).
    """
    local_val = x.reduce(
        cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
    )

    warp_width = min(threads_per_row, 32)
    warp_val = warp_reduce(local_val, operator.add, width=warp_width)

    warps_per_row = max(threads_per_row // 32, 1)

    if cutlass.const_expr(warps_per_row > 1 or cluster_n > 1):
        if cutlass.const_expr(cluster_n == 1):
            return block_reduce_multirow(
                warp_val, operator.add, reduction_buffer, Float32(0.0)
            )
        else:
            return cluster_reduce_multirow(
                warp_val,
                operator.add,
                reduction_buffer,
                mbar_ptr,
                cluster_n,
                Float32(0.0),
            )
    else:
        return warp_val


@cute.jit
def predicate_k(tXcX: cute.Tensor, limit: int) -> cute.Tensor:
    """Create predicate tensor for bounds checking (2D tensors)."""
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


class RMSNormKernel:
    """
    RMSNorm Kernel using CuTe-DSL.

    Computes: output = input / sqrt(mean(input^2) + eps) * (weight + weight_bias)
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        weight_bias: float = 0.0,
        sm_version: int | None = None,
    ):
        self.dtype = dtype
        self.H = H
        self.weight_bias = weight_bias
        self.sm_version = sm_version if sm_version is not None else _target_sm()

        self.cluster_n = self._compute_cluster_n(H, dtype, self.sm_version)
        self.H_per_cta = H // self.cluster_n

        elem_bytes = dtype.width // 8
        max_vec_size = COPY_BITS // 8 // elem_bytes

        h_align = self.H_per_cta & (-self.H_per_cta)
        self.vec_size = min(h_align, max_vec_size)
        self.copy_bits = self.vec_size * dtype.width

        self.threads_per_row = self._compute_threads_per_row(self.H_per_cta)
        self.num_threads = self._compute_num_threads(self.H_per_cta)
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)

        self.num_vec_blocks = max(
            1,
            (self.H_per_cta // self.vec_size + self.threads_per_row - 1)
            // self.threads_per_row,
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row

        if self.copy_bits >= 32:
            tile_bytes = self.rows_per_block * self.cols_per_tile * elem_bytes
            self.use_async_copy = tile_bytes <= _shared_memory_per_block_optin() // 2
        else:
            self.use_async_copy = False

    @staticmethod
    def _compute_cluster_n(H: int, dtype: cutlass.Numeric, sm_version: int) -> int:
        """Compute optimal cluster size based on H and device shared memory."""
        if sm_version < 90:
            return 1

        max_smem_bytes = _shared_memory_per_block_optin()
        elem_size = dtype.width // 8

        for cluster_n in [1, 2, 4, 8, 16]:
            if H % cluster_n != 0:
                continue
            smem_needed = RMSNormKernel._estimate_smem_bytes(H, cluster_n, elem_size)
            if smem_needed <= max_smem_bytes:
                return cluster_n

        return 16

    @staticmethod
    def _estimate_smem_bytes(H: int, cluster_n: int, elem_size: int) -> int:
        """Estimate shared memory bytes for a given cluster configuration."""
        H_per_cta = H // cluster_n
        threads_per_row = RMSNormKernel._compute_threads_per_row(H_per_cta)
        num_threads = RMSNormKernel._compute_num_threads(H_per_cta)
        rows_per_block = num_threads // threads_per_row
        warps_per_row = max(threads_per_row // 32, 1)

        max_vec_size = COPY_BITS // 8 // elem_size
        h_align = H_per_cta & (-H_per_cta)
        vec_size = min(h_align, max_vec_size)
        num_vec_blocks = max(
            1, (H_per_cta // vec_size + threads_per_row - 1) // threads_per_row
        )
        cols_per_tile = vec_size * num_vec_blocks * threads_per_row

        tile_bytes = rows_per_block * cols_per_tile * elem_size

        if cluster_n == 1:
            return tile_bytes + rows_per_block * warps_per_row * 4
        else:
            return (
                tile_bytes
                + rows_per_block * warps_per_row * cluster_n * 4
                + 8  # mbarrier
            )

    @staticmethod
    def _compute_threads_per_row(H: int) -> int:
        if H <= 64:
            return 8
        elif H <= 128:
            return 16
        elif H <= 3072:
            return 32
        elif H <= 6144:
            return 64
        elif H <= 16384:
            return 128
        else:
            return 256

    @staticmethod
    def _compute_num_threads(H: int) -> int:
        return 128 if H <= 16384 else 256

    @staticmethod
    def _make_tv_layout(threads_per_row, rows_per_block, vec_size, num_vec_blocks):
        """Create Thread-Value layout for multi-row coalesced vectorized access."""
        shape = (
            (threads_per_row, rows_per_block),
            (vec_size, num_vec_blocks),
        )
        stride = (
            (vec_size * rows_per_block, 1),
            (rows_per_block, rows_per_block * vec_size * threads_per_row),
        )
        return shape, stride

    def _smem_size_in_bytes(self) -> int:
        if self.use_async_copy:
            tile_bytes = (
                self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)
            )
        else:
            tile_bytes = 0

        if self.cluster_n == 1:
            reduction_bytes = self.rows_per_block * self.warps_per_row * 4
        else:
            reduction_bytes = (
                self.rows_per_block * self.warps_per_row * self.cluster_n * 4
            )

        mbar_bytes = 8 if self.cluster_n > 1 else 0
        return tile_bytes + reduction_bytes + mbar_bytes

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = self._make_tv_layout(
            self.threads_per_row,
            self.rows_per_block,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)

        cluster_n = self.cluster_n

        self.kernel(mX, mW, mY, M, eps, enable_pdl, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), cluster_n, 1],
            block=[self.num_threads, 1, 1],
            cluster=[1, cluster_n, 1] if cutlass.const_expr(cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # PDL: Wait for previous kernel (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        H = self.H
        cluster_n = self.cluster_n
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)

        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        # ===== Allocate shared memory =====
        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        if cutlass.const_expr(cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

        # ===== Initialize cluster =====
        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        # ===== Coordinate tracking and tiling =====
        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gY = cute.local_tile(mY, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        mW_expanded_layout = cute.prepend(
            mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
        )
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        # ===== Create TiledCopy atoms =====
        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mY.element_type,
            num_bits_per_copy=copy_bits,
        )

        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=copy_bits,
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)

        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        # Partition input
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)

        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)

        # Partition weight (sync, separate tiled copy)
        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)

        # Partition output
        tXgO = thr_copy_O.partition_D(gY)
        tXrO = cute.make_fragment_like(tXgO)

        # ===== Bounds checking =====
        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # ===== Pass 1: Load input + compute sum of squares =====
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
            cute.arch.cp_async_commit_group()

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x = tXrX.load().to(Float32)
        x_sq = x * x
        sum_sq = row_reduce_sum_multirow(
            x_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # ===== Pass 2: Normalize and store output =====
        # Re-load x from shared memory to relieve register pressure.
        # Without this, x (up to 128 FP32 values/thread at large H) must
        # survive across the reduction + barrier, causing spills to local mem.
        if cutlass.const_expr(self.use_async_copy):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

        w = tXrW.load().to(Float32)
        y = x * rstd * (w + Float32(weight_bias))

        tXrO.store(y.to(mY.element_type))

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)

        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


def _create_wbc1_rmsnorm_jit(dtype, hidden_size):
    rmsnorm_kernel = RMSNormKernel(
        dtype,
        hidden_size,
        weight_bias=0.0,
        sm_version=_target_sm(),
    )

    @cute.jit
    def aot_adapter(
        x: cute.Tensor,
        gamma: cute.Tensor,
        output: cute.Tensor,
        rows: Int64,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        # Host-only adapter: RMSNormKernel owns the exact upstream launch.
        rmsnorm_kernel(x, gamma, output, rows, eps, False, stream)

    return aot_adapter


def _define_legacy_rmsnorm_kernel(hidden_size):
    """Restore the original one-row WBC0 kernel and arithmetic."""
    values_per_thread = hidden_size // NUM_THREADS

    @cute.kernel
    def rmsnorm_kernel(
        x: cute.Tensor,
        gamma: cute.Tensor,
        output: cute.Tensor,
        eps: Float32,
    ):
        thread_idx, _, _ = cute.arch.thread_idx()
        row_idx, _, _ = cute.arch.block_idx()
        lane_idx = thread_idx % WARP_SIZE
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        partial_sum_layout = cute.make_layout((NUM_WARPS,), stride=(1,))
        smem = cutlass.utils.SmemAllocator()
        partial_sums = smem.allocate_tensor(
            cutlass.Float32,
            partial_sum_layout,
            128,
        )

        sum_of_squares = cutlass.Float32(0.0)
        for value_idx in range(values_per_thread):
            column_idx = thread_idx + value_idx * NUM_THREADS
            value = cutlass.Float32(x[row_idx, column_idx])
            sum_of_squares += value * value

        for offset in (16, 8, 4, 2, 1):
            sum_of_squares += cute.arch.shuffle_sync_bfly(
                sum_of_squares,
                offset=offset,
                mask=-1,
                mask_and_clamp=WARP_SIZE - 1,
            )

        if lane_idx == 0:
            partial_sums[warp_idx] = sum_of_squares
        cute.arch.barrier()

        if warp_idx == 0:
            block_sum = cutlass.Float32(0.0)
            if lane_idx < NUM_WARPS:
                block_sum = partial_sums[lane_idx]
            for offset in (16, 8, 4, 2, 1):
                block_sum += cute.arch.shuffle_sync_bfly(
                    block_sum,
                    offset=offset,
                    mask=-1,
                    mask_and_clamp=WARP_SIZE - 1,
                )
            if lane_idx == 0:
                partial_sums[0] = block_sum
        cute.arch.barrier()

        mean_square = partial_sums[0] / cutlass.Float32(hidden_size)
        reciprocal_rms = cute.rsqrt(mean_square + eps)

        # Preserve the original WBC0 reload and input-type multiply.
        for value_idx in range(values_per_thread):
            column_idx = thread_idx + value_idx * NUM_THREADS
            value = cutlass.Float32(x[row_idx, column_idx])
            normalized = value * reciprocal_rms
            normalized_low_precision = x.element_type(normalized)
            output[row_idx, column_idx] = output.element_type(
                normalized_low_precision * gamma[column_idx]
            )

    return rmsnorm_kernel


def _create_legacy_rmsnorm_jit(hidden_size):
    rmsnorm_kernel = _define_legacy_rmsnorm_kernel(hidden_size)

    @cute.jit
    def run_rmsnorm(
        x: cute.Tensor,
        gamma: cute.Tensor,
        output: cute.Tensor,
        eps: Float32,
        weight_before_cast: Int32,
        stream: cuda.CUstream,
    ):
        # Retain weight_before_cast in the generated ABI; this artifact is WBC0.
        _ = weight_before_cast
        rows = x.layout.shape[0]
        rmsnorm_kernel(x, gamma, output, eps).launch(
            grid=(rows, 1, 1),
            block=(NUM_THREADS, 1, 1),
            smem=LEGACY_SHARED_MEMORY_BYTES,
            stream=stream,
        )

    return run_rmsnorm


def _create_rmsnorm_jit(dtype, hidden_size, weight_before_cast_mode):
    if weight_before_cast_mode == 1:
        return _create_wbc1_rmsnorm_jit(dtype, hidden_size)
    return _create_legacy_rmsnorm_jit(hidden_size)


def _allocate_placeholder(shape, dtype):
    if dtype == cutlass.Float16:
        return cupy.zeros(shape, dtype=cupy.float16)
    if dtype == cutlass.BFloat16:
        # CuPy has no native BF16 storage. CuTe interprets these bits as BF16
        # after the tensor element type is retagged below.
        return cupy.zeros(shape, dtype=cupy.uint16)
    raise ValueError(f"Unsupported RMSNorm dtype: {dtype}")


def _to_cute_tensor(array, dtype, *, dynamic_rows):
    tensor = from_dlpack(array, assumed_align=16)
    tensor.element_type = dtype
    if dynamic_rows:
        # Keep H and the row stride static so the AOT ABI exposes only rows as
        # dynamic and retains the alignment proof for the contiguous axis.
        tensor = tensor.mark_compact_shape_dynamic(
            mode=0,
            stride_order=(0, 1),
            divisibility=1,
        )
    return tensor


def compile_rmsnorm(dtype, hidden_size, weight_before_cast):
    x_storage = _allocate_placeholder((AOT_ROWS, hidden_size), dtype)
    gamma_storage = _allocate_placeholder((hidden_size,), dtype)
    output_storage = _allocate_placeholder((AOT_ROWS, hidden_size), dtype)

    x = _to_cute_tensor(x_storage, dtype, dynamic_rows=True)
    gamma = _to_cute_tensor(gamma_storage, dtype, dynamic_rows=False)
    output = _to_cute_tensor(output_storage, dtype, dynamic_rows=True)
    stream = cuda.CUstream(cupy.cuda.get_current_stream().ptr)

    rmsnorm = _create_rmsnorm_jit(dtype, hidden_size, weight_before_cast)
    if weight_before_cast == 1:
        return cute.compile(
            rmsnorm,
            x,
            gamma,
            output,
            Int64(AOT_ROWS),
            Float32(1e-6),
            stream,
        )
    return cute.compile(
        rmsnorm,
        x,
        gamma,
        output,
        1e-6,
        weight_before_cast,
        stream,
    )


def export_rmsnorm(
    dtype,
    hidden_size,
    weight_before_cast,
    output_dir,
    file_name,
    function_prefix,
):
    compiled = compile_rmsnorm(dtype, hidden_size, weight_before_cast)
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
        "--weight_before_cast",
        type=int,
        choices=(0, 1),
        required=True,
        help="compile-time arithmetic specialization; retained in the runtime ABI",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--file_name", required=True)
    parser.add_argument("--function_prefix", required=True)
    parser.add_argument("--export_only", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    if not args.export_only:
        raise ValueError(
            "RMSNorm currently supports AOT export only; pass --export_only."
        )
    dtype = cutlass.Float16 if args.dtype == "fp16" else cutlass.BFloat16
    export_rmsnorm(
        dtype,
        args.hidden_size,
        args.weight_before_cast,
        args.output_dir,
        args.file_name,
        args.function_prefix,
    )


if __name__ == "__main__":
    main()
