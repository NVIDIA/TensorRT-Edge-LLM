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

"""Split-K GEMM for Ampere: kernel-level K-parallel with FP32 atomicAdd.

Grid = (ceil(M/bM) * ceil(N/bN), split_k, 1).
Each CTA processes a K-slice of one (M,N) tile.
Partial FP32 results are accumulated to global memory via atomicAdd.
A final pass converts FP32 → FP16.

This avoids the multi-stream issue of host-level Split-K while
keeping full FP32 precision across K-splits.

Usage:
  python gemm_ampere_streamk.py --mnk 1,2048,2048 --split_k 4
"""

import argparse
import math
import sys
import time
from typing import Tuple, Type

_parsed_args = None
_saved_argv = None
if __name__ == "__main__":
    _saved_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]

import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
import numpy as np
from cutedsl_utils import aot_placeholders
from common import (
    create_fake_row_major_2d_tensor,
    create_row_major_2d_tensors,
    export_compiled_kernel,
    mark_2d_row_major_dynamic,
    parse_comma_separated_ints,
    to_cute_tensor,
)


class GemmAmpereSplitKFP16:
    """Split-K Ampere GEMM: C[M,N] = A[M,K] @ B[N,K]^T.

    FP16 in, FP32 workspace out. Each CTA processes K/split_k iterations
    for one output tile. Results are atomically added to FP32 workspace.
    Host-side reduction kernel converts FP32 → FP16.
    """

    def __init__(
        self,
        ab_dtype=cutlass.Float16,
        acc_dtype=cutlass.Float32,
        cta_tiler_mnk=(16, 128, 128),
        num_stages=3,
        atom_layout_mnk=(1, 4, 1),
        split_k: int = 4,
    ):
        self.ab_dtype = ab_dtype
        self.acc_dtype = acc_dtype
        self.cta_tiler = cta_tiler_mnk
        self.num_stages = num_stages
        self.atom_layout_mnk = atom_layout_mnk
        self.split_k = split_k
        atom_lay_M, atom_lay_N, atom_lay_K = self.atom_layout_mnk
        self.num_threads = atom_lay_M * atom_lay_N * atom_lay_K * 32
        self.bM, self.bN, self.bK = self.cta_tiler
        self.mma_inst_shape = (16, 8, 16)

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,  # FP16 workspace [split_k * M, N] — each split writes M rows
        stream: cuda.CUstream,
    ):
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)

        ab_copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            mA.element_type, self.a_major_mode, ab_copy_bits,
            (self.bM, self.bK, self.num_stages),
        )
        sB_layout = self._make_smem_layout_AB(
            mB.element_type, self.b_major_mode, ab_copy_bits,
            (self.bN, self.bK, self.num_stages),
        )

        smem_size = (
            cute.size_in_bytes(mA.element_type, sA_layout)
            + cute.size_in_bytes(mB.element_type, sB_layout)
        )

        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mA.element_type, num_bits_per_copy=ab_copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mA.element_type, self.a_major_mode, ab_copy_bits
        )
        tiled_copy_B = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mB.element_type, self.b_major_mode, ab_copy_bits
        )

        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.ab_dtype, self.acc_dtype, self.mma_inst_shape
        )
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tC = cute.make_layout(self.atom_layout_mnk)
        tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)

        self.c_dtype = mC.element_type

        sC_layout = self._make_smem_layout_C(
            mC.element_type, self.c_major_mode, ab_copy_bits,
            (self.bM, self.bN),
        )

        c_copy_bits = 128
        atom_sync_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mC.element_type,
            num_bits_per_copy=c_copy_bits,
        )
        tiled_copy_C = self._make_gmem_tiled_copy_C(
            atom_sync_copy, mC.element_type, self.c_major_mode, c_copy_bits
        )

        # Grid: tiles_m from M (mA dim 0), tiles_n from N (mB dim 0, since B=[N,K]).
        tiles_m = cute.ceil_div(mA.shape[0], self.bM)
        tiles_n = cute.ceil_div(mB.shape[0], self.bN)
        num_tiles = cute.size(tiles_m) * cute.size(tiles_n)
        grid = (num_tiles, self.split_k, 1)

        self.kernel(
            mA, mB, mC,
            sA_layout, sB_layout, sC_layout,
            tiled_copy_A, tiled_copy_B, tiled_copy_C,
            tiled_mma,
            cute.size(tiles_n),
            cute.size(tiles_m),
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tiles_n: cutlass.Int32,
        tiles_m: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        tile_idx, split_idx, _ = cute.arch.block_idx()

        # Decode tile_idx → (tile_m, tile_n).
        tile_m = tile_idx // tiles_n
        tile_n = tile_idx - tile_m * tiles_n
        tiler_coord = (tile_m, tile_n, None)

        # Tile A, B (full K range — split offset computed below).
        gA = cute.local_tile(mA, tiler=self.cta_tiler, coord=tiler_coord, proj=(1, None, 1))
        gB = cute.local_tile(mB, tiler=self.cta_tiler, coord=tiler_coord, proj=(None, 1, 1))

        # Output tile: offset in M by split_idx * tiles_m.
        # mC is [split_k * M, N]; each split writes to a different M-row range.
        out_tile_m = tile_m + split_idx * tiles_m
        out_tiler_coord = (out_tile_m, tile_n, None)
        gC = cute.local_tile(mC, tiler=self.cta_tiler, coord=out_tiler_coord, proj=(1, 1, None))

        # Shared memory.
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
        sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)
        sC = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=self.c_dtype), sC_layout)

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        thr_copy_C = tiled_copy_C.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)
        tCsC_epilogue = thr_copy_C.partition_S(sC)
        tCgC_epilogue = thr_copy_C.partition_D(gC)

        # MMA setup.
        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCsC = thr_mma.partition_C(sC)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        # S2R copy atoms.
        atom_s2r_A = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(
            self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4), mA.element_type)
        atom_s2r_B = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(
            self.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4), mB.element_type)
        tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
        tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)
        thr_s2r_A = tiled_s2r_A.get_slice(tidx)
        thr_s2r_B = tiled_s2r_B.get_slice(tidx)
        tCsA_view = thr_s2r_A.partition_S(sA)
        tCrA_view = thr_s2r_A.retile(tCrA)
        tCsB_view = thr_s2r_B.partition_S(sB)
        tCrB_view = thr_s2r_B.retile(tCrB)

        num_smem_stages = cute.size(tAsA, mode=[3])
        num_k_block = cute.size(tCrA, mode=[2])

        # K-tiles per split: STATIC (split_k is Constexpr, total K-tiles is static).
        total_k_tiles = cute.size(tAgA, mode=[3])
        k_tiles_per_split = total_k_tiles // self.split_k  # Constexpr // Constexpr = Constexpr

        # Runtime offset: which K-tile this split starts from.
        k_tile_index = split_idx * cutlass.Int32(k_tiles_per_split)

        # ---- Prologue: identical to standard kernel, with offset k_tile_index ----
        tAsA.fill(0)
        tBsB.fill(0)
        cute.arch.sync_threads()

        cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, 0])
        cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, 0])
        k_tile_index = k_tile_index + 1
        cute.arch.cp_async_commit_group()

        for s in range(1, num_smem_stages - 1):
            if s < k_tiles_per_split:
                cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, s])
                cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, s])
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

        smem_pipe_read = 0
        smem_pipe_write = num_smem_stages - 1
        tCsA_p = tCsA_view[None, None, None, 0]
        tCsB_p = tCsB_view[None, None, None, 0]
        if num_k_block > 1:
            cute.arch.cp_async_wait_group(num_smem_stages - 2)
            cute.arch.sync_threads()
            cute.copy(tiled_s2r_A, tCsA_p[None, None, 0], tCrA_view[None, None, 0])
            cute.copy(tiled_s2r_B, tCsB_p[None, None, 0], tCrB_view[None, None, 0])

        # ---- Mainloop: range(k_tiles_per_split) is STATIC ----
        k_tile_count = k_tiles_per_split  # static Constexpr
        for k_tile in range(k_tile_count):
            for k_block in cutlass.range(num_k_block, unroll_full=True):
                if k_block == num_k_block - 1:
                    tCsA_p = tCsA_view[None, None, None, smem_pipe_read]
                    tCsB_p = tCsB_view[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(num_smem_stages - 2)
                    cute.arch.sync_threads()

                k_block_next = (k_block + 1) % num_k_block
                cute.copy(tiled_s2r_A, tCsA_p[None, None, k_block_next], tCrA_view[None, None, k_block_next])
                cute.copy(tiled_s2r_B, tCsB_p[None, None, k_block_next], tCrB_view[None, None, k_block_next])

                if k_block == 0:
                    if k_tile + num_smem_stages - 1 < k_tile_count:
                        cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, smem_pipe_write])
                        cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, smem_pipe_write])

                cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block], tCrB[None, None, k_block], tCrC)

                if k_block == 0:
                    k_tile_index = k_tile_index + 1
                    cute.arch.cp_async_commit_group()
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == num_smem_stages:
                        smem_pipe_read = 0

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # ---- Epilogue: standard FP16 smem-staged store ----
        tCrD = cute.make_fragment_like(tCrC, self.c_dtype)
        tCrD[None] = tCrC.load().to(self.c_dtype)
        cute.autovec_copy(tCrD, tCsC)

        tCrC_epilogue = cute.make_fragment_like(tCsC_epilogue)
        cute.arch.sync_threads()
        cute.autovec_copy(tCsC_epilogue, tCrC_epilogue)
        cute.copy(tiled_copy_C, tCrC_epilogue, tCgC_epilogue)

    # --- Layout helpers (same as GemmAmpereFP16) ---
    def _make_smem_layout_AB(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = smem_tiler[1] if major_mode == utils.LayoutEnum.ROW_MAJOR else smem_tiler[0]
        major_mode_size = 64 if major_mode_size >= 64 else major_mode_size
        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)
        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout_atom = cute.make_composed_layout(cute.make_swizzle(swizzle_bits, 3, 3), 0, layout_atom_outer)
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))

    def _make_smem_layout_C(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = smem_tiler[1] if major_mode == utils.LayoutEnum.ROW_MAJOR else smem_tiler[0]
        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)
        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout_atom = cute.make_composed_layout(cute.make_swizzle(swizzle_bits, 3, 4), 0, layout_atom_outer)
        if major_mode == utils.LayoutEnum.COL_MAJOR:
            layout_atom = cute.make_composed_layout(cute.make_swizzle(0, 3, 4), 0, layout_atom_outer)
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1))

    def _make_gmem_tiled_copy_C(self, atom_copy, dtype, major_mode, copy_bits):
        copy_elems = copy_bits // dtype.width
        shape_dim_1 = cute.size(self.bN) // copy_elems
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        if major_mode != utils.LayoutEnum.ROW_MAJOR:
            shape_dim_0 = cute.size(self.bM) // copy_elems
            thread_layout = cute.make_layout(
                (shape_dim_0, self.num_threads // shape_dim_0), stride=(1, shape_dim_0)
            )
        value_layout = (
            cute.make_layout((1, copy_elems)) if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    def _make_gmem_tiled_copy_AB(self, atom_copy, dtype, major_mode, copy_bits):
        copy_elems = copy_bits // dtype.width
        shape_dim_1 = cute.size(self.bK) // copy_elems
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        if major_mode != utils.LayoutEnum.ROW_MAJOR:
            shape_dim_0 = cute.size(self.bM) // copy_elems
            thread_layout = cute.make_layout(
                (shape_dim_0, self.num_threads // shape_dim_0), stride=(1, shape_dim_0)
            )
        value_layout = (
            cute.make_layout((1, copy_elems)) if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def run(mnk=(1, 2048, 2048), split_k=4, cta_tiler_mnk=(16, 128, 128),
        atom_layout_mnk=(1, 4, 1), num_stages=3, num_sms=108,
        export_only=False, output_dir="./artifacts",
        file_name="gemm_splitk", function_prefix="gemm_splitk", gpu_arch=""):
    M, N, K = mnk
    _tag = f"[{file_name}]"
    print(f"{_tag} Split-K GEMM: M={M}, N={N}, K={K}, split_k={split_k}")

    # Workspace: each split needs ceil(M, bM) rows, not just M rows.
    padded_m = ((M + cta_tiler_mnk[0] - 1) // cta_tiler_mnk[0]) * cta_tiler_mnk[0]
    if export_only:
        mA = create_fake_row_major_2d_tensor()
        mB = create_fake_row_major_2d_tensor()
        mC = create_fake_row_major_2d_tensor()
        a_cp = b_cp = c_ws = None
    else:
        a_cp = cp.random.uniform(-1, 1, (M, K)).astype(cp.float16)
        b_cp = cp.random.uniform(-1, 1, (N, K)).astype(cp.float16)
        c_ws = cp.zeros((split_k * padded_m, N), dtype=cp.float16)
        mA = mark_2d_row_major_dynamic(to_cute_tensor(a_cp))
        mB = mark_2d_row_major_dynamic(to_cute_tensor(b_cp))
        mC = mark_2d_row_major_dynamic(to_cute_tensor(c_ws))

    gemm = GemmAmpereSplitKFP16(
        cta_tiler_mnk=cta_tiler_mnk, atom_layout_mnk=atom_layout_mnk,
        num_stages=num_stages, split_k=split_k,
    )
    stream_ptr = (
        aot_placeholders.make_stream()
        if export_only
        else cuda.CUstream(cp.cuda.get_current_stream().ptr)
    )

    print(f"{_tag} Compiling...")
    t0 = time.time()
    compile_opts = aot_placeholders.compile_options(
        ("--gpu-arch " + gpu_arch) if gpu_arch else ""
    ) if export_only else None
    compiled = cute.compile(
        gemm,
        mA,
        mB,
        mC,
        stream_ptr,
        **(dict(options=compile_opts) if export_only else {}),
    )
    print(f"{_tag} Compilation: {time.time() - t0:.2f}s")

    if export_only:
        export_compiled_kernel(compiled, output_dir=output_dir,
                               file_name=file_name, function_prefix=function_prefix, tag=_tag)
        return

    # Run and verify.
    compiled(mA, mB, mC, stream_ptr)
    cp.cuda.Device().synchronize()

    # Reduce across splits: sum FP16 partials (using padded rows) in FP32.
    c_reduced = cp.zeros((M, N), dtype=cp.float32)
    for p in range(split_k):
        c_reduced += c_ws[p * padded_m:p * padded_m + M, :].astype(cp.float32)
    c_fp16 = c_reduced.astype(cp.float16)
    ref = cp.matmul(a_cp.astype(cp.float32), b_cp.astype(cp.float32).T).astype(cp.float16)
    diff = float(cp.max(cp.abs(c_fp16.astype(cp.float32) - ref.astype(cp.float32))))
    print(f"{_tag} max_diff={diff:.6f} ({'PASS' if diff < 0.1 else 'FAIL'})")

    # Benchmark.
    for _ in range(5):
        compiled(mA, mB, mC, stream_ptr)
    cp.cuda.Device().synchronize()
    start = cp.cuda.Event(); end = cp.cuda.Event()
    s = cp.cuda.get_current_stream()
    start.record(s)
    for _ in range(50):
        compiled(mA, mB, mC, stream_ptr)
    end.record(s); end.synchronize()
    us = cp.cuda.get_elapsed_time(start, end) * 1000.0 / 50
    print(f"{_tag} {us:.2f} us (kernel only, excludes reduction)")


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mnk", type=parse_comma_separated_ints, default=(1, 2048, 2048))
    p.add_argument("--split_k", type=int, default=4)
    p.add_argument("--cta_tiler_mnk", type=parse_comma_separated_ints, default=(16, 128, 128))
    p.add_argument("--atom_layout_mnk", type=parse_comma_separated_ints, default=(1, 4, 1))
    p.add_argument("--num_stages", type=int, default=3)
    p.add_argument("--export_only", action="store_true")
    p.add_argument("--output_dir", type=str, default="./artifacts")
    p.add_argument("--file_name", type=str, default="gemm_splitk")
    p.add_argument("--function_prefix", type=str, default="gemm_splitk")
    p.add_argument("--gpu_arch", type=str, default="")
    return p.parse_known_args(args=argv)[0]


if __name__ == "__main__":
    args = _parse_args(_saved_argv)
    run(mnk=args.mnk, split_k=args.split_k, cta_tiler_mnk=args.cta_tiler_mnk,
        atom_layout_mnk=args.atom_layout_mnk, num_stages=args.num_stages,
        export_only=args.export_only, output_dir=args.output_dir,
        file_name=args.file_name, function_prefix=args.function_prefix, gpu_arch=args.gpu_arch)
