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

# Adapted from CUTLASS examples/python/CuTeDSL/ampere/tensorop_gemm.py (BSD-3-Clause).
# Stripped batch dimension, fixed row-major A[M,K] x B[N,K]^T → C[M,N], added AOT export.
# CuPy-only (no PyTorch). Test: python gemm_ampere.py --mnk 256,512,128
# AOT: python gemm_ampere.py --mnk 256,512,128 --export_only --output_dir ./artifacts

import argparse
import math
import os
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
    create_bias_tensor,
    create_fake_row_major_2d_tensor,
    create_row_major_2d_tensors,
    export_compiled_kernel,
    mark_2d_row_major_dynamic,
    parse_comma_separated_ints,
    to_cute_tensor,
)


class GemmAmpereFP16:
    """Ampere tensor-core GEMM: C[M,N] = A[M,K] @ B[N,K]^T, FP16 in/out, FP32 accumulation.

    A is row-major (K contiguous), B is row-major (K contiguous), C is row-major (N contiguous).
    Tile shape 128x128x32, MMA instruction 16x8x16, atom layout 2x2x1, 3-stage pipeline.
    """

    def __init__(
        self,
        ab_dtype: Type[cutlass.Numeric] = cutlass.Float16,
        c_dtype: Type[cutlass.Numeric] = cutlass.Float16,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        cta_tiler_mnk: Tuple[int, int, int] = (16, 128, 128),
        num_stages: int = 3,
        atom_layout_mnk: Tuple[int, int, int] = (1, 4, 1),
    ):
        self.ab_dtype = ab_dtype
        self.c_dtype = c_dtype
        self.acc_dtype = acc_dtype
        self.cta_tiler = cta_tiler_mnk
        self.num_stages = num_stages
        self.atom_layout_mnk = atom_layout_mnk
        atom_lay_M, atom_lay_N, atom_lay_K = self.atom_layout_mnk
        self.num_threads = atom_lay_M * atom_lay_N * atom_lay_K * 32

        self.bM, self.bN, self.bK = self.cta_tiler
        self.mma_inst_shape = (16, 8, 16)
        mmaM, mmaN, mmaK = self.mma_inst_shape

        assert self.bM % (atom_lay_M * mmaM) == 0
        assert self.bN % (atom_lay_N * mmaN) == 0
        assert atom_lay_K == 1
        assert self.bK % mmaK == 0
        assert self.num_stages >= 3

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        stream: cuda.CUstream,
        mBias: cute.Tensor = None,
        use_silu: cutlass.Constexpr = False,
        skip_predication: cutlass.Constexpr = False,
    ):
        # A is (M, K) row-major  → a_major_mode = ROW_MAJOR (K is contiguous)
        # B is (N, K) row-major  → b_major_mode = ROW_MAJOR (K is contiguous)
        # C is (M, N) row-major  → c_major_mode = ROW_MAJOR (N is contiguous)
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)

        ab_copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            mA.element_type,
            self.a_major_mode,
            ab_copy_bits,
            (self.cta_tiler[0], self.cta_tiler[2], self.num_stages),
        )
        sB_layout = self._make_smem_layout_AB(
            mB.element_type,
            self.b_major_mode,
            ab_copy_bits,
            (self.cta_tiler[1], self.cta_tiler[2], self.num_stages),
        )

        sC_layout = self._make_smem_layout_C(
            mC.element_type,
            self.c_major_mode,
            ab_copy_bits,
            (self.cta_tiler[0], self.cta_tiler[1]),
        )

        smem_size = max(
            cute.size_in_bytes(mC.element_type, sC_layout),
            cute.size_in_bytes(mA.element_type, sA_layout)
            + cute.size_in_bytes(mB.element_type, sB_layout),
        )

        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mA.element_type,
            num_bits_per_copy=ab_copy_bits,
        )

        tiled_copy_A = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mA.element_type, self.a_major_mode, ab_copy_bits
        )
        tiled_copy_B = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mB.element_type, self.b_major_mode, ab_copy_bits
        )

        c_copy_bits = 128
        atom_sync_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mC.element_type,
            num_bits_per_copy=c_copy_bits,
        )
        tiled_copy_C = self._make_gmem_tiled_copy_C(
            atom_sync_copy, mC.element_type, self.c_major_mode, c_copy_bits
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
        tiled_mma = cute.make_tiled_mma(
            op,
            tC,
            permutation_mnk=permutation_mnk,
        )

        # 2D grid: ceil(M/bM) x ceil(N/bN), no batch dimension
        grid_dim = cute.ceil_div(mC.shape, (self.bM, self.bN))

        raster_factor = 1
        grid_dim_n = cute.size(grid_dim[1])
        if grid_dim_n > 5:
            raster_factor = 8
        elif grid_dim_n > 2:
            raster_factor = 4
        elif grid_dim_n > 1:
            raster_factor = 2
        rasterization_remap_grid_dim = (
            cute.size(grid_dim[0]) * raster_factor,
            (cute.size(grid_dim[1]) + raster_factor - 1) // raster_factor,
            1,
        )

        self.kernel(
            mA,
            mB,
            mC,
            sA_layout,
            sB_layout,
            sC_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_copy_C,
            tiled_mma,
            raster_factor,
            mBias,
            use_silu,
            skip_predication,
        ).launch(
            grid=rasterization_remap_grid_dim,
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
        rasterization_factor: cutlass.Int32,
        mBias: cute.Tensor = None,
        use_silu: cutlass.Constexpr = False,
        skip_predication: cutlass.Constexpr = False,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        grid_dim = cute.ceil_div(mC.shape, (self.bM, self.bN))
        offset_tile_x, offset_tile_y = self.raster_tile(
            bidx, bidy, rasterization_factor
        )
        if grid_dim[0] <= offset_tile_x or grid_dim[1] <= offset_tile_y:
            pass
        else:
            tiler_coord = (offset_tile_x, offset_tile_y, None)

            # Tile A: (BLK_M, BLK_K, k_tiles) — project out M and K dims
            gA = cute.local_tile(
                mA,
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(1, None, 1),
            )
            # Tile B: (BLK_N, BLK_K, k_tiles) — project out N and K dims
            gB = cute.local_tile(
                mB,
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(None, 1, 1),
            )
            # Tile C: (BLK_M, BLK_N)
            gC = cute.local_tile(
                mC,
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(1, 1, None),
            )

            # Handle K residue: shift first tiles to be the irregular ones
            residual_k = cute.size(mA, mode=[1]) - cutlass.Int32(self.bK) * cute.size(
                gA, mode=[2]
            )
            gA = cute.domain_offset((0, residual_k, 0), gA)
            gB = cute.domain_offset((0, residual_k, 0), gB)
            gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
            gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

            # Identity tensors for predication
            mcA = cute.make_identity_tensor(mA.layout.shape)
            mcB = cute.make_identity_tensor(mB.layout.shape)
            cA = cute.local_tile(
                mcA,
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(1, None, 1),
            )
            cB = cute.local_tile(
                mcB,
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(None, 1, 1),
            )
            cA = cute.domain_offset((0, residual_k, 0), cA)
            cB = cute.domain_offset((0, residual_k, 0), cB)

            # Shared memory buffers
            smem = cutlass.utils.SmemAllocator()
            sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
            sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)
            sC = cute.make_tensor(
                cute.recast_ptr(sA.iterator, dtype=self.c_dtype), sC_layout
            )

            thr_copy_A = tiled_copy_A.get_slice(tidx)
            thr_copy_B = tiled_copy_B.get_slice(tidx)
            thr_copy_C = tiled_copy_C.get_slice(tidx)
            tAgA = thr_copy_A.partition_S(gA)
            tAsA = thr_copy_A.partition_D(sA)
            tBgB = thr_copy_B.partition_S(gB)
            tBsB = thr_copy_B.partition_D(sB)
            tCsC_epilogue = thr_copy_C.partition_S(sC)
            tCgC_epilogue = thr_copy_C.partition_D(gC)

            tAcA = thr_copy_A.partition_S(cA)
            tBcB = thr_copy_B.partition_S(cB)

            # Predicate tensors for M and N boundary checks.
            # When skip_predication=True, all predicates are True (compiler optimizes).
            tApA = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tAgA.shape[0][1],
                        cute.size(tAgA, mode=[1]),
                        cute.size(tAgA, mode=[2]),
                    ),
                    stride=(cute.size(tAgA, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            tBpB = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tBsB.shape[0][1],
                        cute.size(tBsB, mode=[1]),
                        cute.size(tBsB, mode=[2]),
                    ),
                    stride=(cute.size(tBsB, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            if cutlass.const_expr(skip_predication):
                tApA.fill(1)
                tBpB.fill(1)
            else:
                for rest_v in range(tApA.shape[0]):
                    for m in range(tApA.shape[1]):
                        tApA[rest_v, m, 0] = cute.elem_less(
                            tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0]
                        )
                for rest_v in range(tBpB.shape[0]):
                    for n in range(tBpB.shape[1]):
                        tBpB[rest_v, n, 0] = cute.elem_less(
                            tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0]
                        )

            # ---- Prefetch prologue ----
            tAsA.fill(0)
            tBsB.fill(0)
            cute.arch.sync_threads()
            num_smem_stages = cute.size(tAsA, mode=[3])
            k_tile_count = cute.size(tAgA, mode=[3])
            k_tile_index = cutlass.Int32(0)

            for k in range(tApA.shape[2]):
                if cute.elem_less(cutlass.Int32(-1), tAcA[0, 0, k, 0][1]):
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, k, k_tile_index],
                        tAsA[None, None, k, 0],
                        pred=tApA[None, None, k],
                    )
            for k in range(tBpB.shape[2]):
                if cute.elem_less(cutlass.Int32(-1), tBcB[0, 0, k, 0][1]):
                    cute.copy(
                        tiled_copy_B,
                        tBgB[None, None, k, k_tile_index],
                        tBsB[None, None, k, 0],
                        pred=tBpB[None, None, k],
                    )
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

            for k_tile in range(1, num_smem_stages - 1):
                if k_tile == k_tile_count:
                    tApA.fill(0)
                    tBpB.fill(0)
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, k_tile_index],
                    tAsA[None, None, None, k_tile],
                    pred=tApA,
                )
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, k_tile_index],
                    tBsB[None, None, None, k_tile],
                    pred=tBpB,
                )
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()

            # ---- MMA tile partitioning and accumulator ----
            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCsC = thr_mma.partition_C(sC)
            tCgC = thr_mma.partition_C(gC)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrC = tiled_mma.make_fragment_C(tCgC)
            tCrC.fill(0.0)

            # ---- SMEM→Register copy atoms (ldmatrix) ----
            atom_copy_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
                ),
                mA.element_type,
            )
            atom_copy_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    self.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
                ),
                mB.element_type,
            )
            tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
            tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)

            thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tidx)
            thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            smem_pipe_read = 0
            smem_pipe_write = num_smem_stages - 1

            tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
            tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]

            # ---- Register pipeline prefetch ----
            num_k_block = cute.size(tCrA, mode=[2])
            if num_k_block > 1:
                cute.arch.cp_async_wait_group(num_smem_stages - 2)
                cute.arch.sync_threads()
                cute.copy(
                    tiled_copy_s2r_A,
                    tCsA_p[None, None, 0],
                    tCrA_copy_view[None, None, 0],
                )
                cute.copy(
                    tiled_copy_s2r_B,
                    tCsB_p[None, None, 0],
                    tCrB_copy_view[None, None, 0],
                )

            # ---- Mainloop ----
            for k_tile in range(k_tile_count):
                for k_block in cutlass.range(num_k_block, unroll_full=True):
                    if k_block == num_k_block - 1:
                        tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                        tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]
                        cute.arch.cp_async_wait_group(num_smem_stages - 2)
                        cute.arch.sync_threads()

                    k_block_next = (k_block + 1) % num_k_block  # static
                    cute.copy(
                        tiled_copy_s2r_A,
                        tCsA_p[None, None, k_block_next],
                        tCrA_copy_view[None, None, k_block_next],
                    )
                    cute.copy(
                        tiled_copy_s2r_B,
                        tCsB_p[None, None, k_block_next],
                        tCrB_copy_view[None, None, k_block_next],
                    )

                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            cute.copy(
                                tiled_copy_A,
                                tAgA[None, None, None, k_tile_index],
                                tAsA[None, None, None, smem_pipe_write],
                                pred=tApA,
                            )

                    cute.gemm(
                        tiled_mma,
                        tCrC,
                        tCrA[None, None, k_block],
                        tCrB[None, None, k_block],
                        tCrC,
                    )

                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            cute.copy(
                                tiled_copy_B,
                                tBgB[None, None, None, k_tile_index],
                                tBsB[None, None, None, smem_pipe_write],
                                pred=tBpB,
                            )
                        k_tile_index = k_tile_index + 1
                        cute.arch.cp_async_commit_group()
                        smem_pipe_write = smem_pipe_read
                        smem_pipe_read = smem_pipe_read + 1
                        if smem_pipe_read == num_smem_stages:
                            smem_pipe_read = 0

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # ---- Epilogue ----
            # Optional bias fusion: add bias[N] to FP32 accumulator before conversion.
            if cutlass.const_expr(mBias is not None):
                gBias_tile = cute.make_tensor(
                    mBias.iterator + offset_tile_y * cutlass.Int32(self.bN),
                    cute.make_layout((self.bM, self.bN), stride=(0, 1)),
                )
                tCgBias = thr_mma.partition_C(gBias_tile)
                for i in cutlass.range(cute.size(tCrC)):
                    tCrC[i] = tCrC[i] + tCgBias[i].to(self.acc_dtype)

            if cutlass.const_expr(use_silu):
                for i in cutlass.range(cute.size(tCrC)):
                    val = tCrC[i]
                    tCrC[i] = val * cute.arch.rcp_approx(1.0 + cute.exp(-val, fastmath=True))

            tCrD = cute.make_fragment_like(tCrC, self.c_dtype)
            tCrD[None] = tCrC.load().to(self.c_dtype)
            cute.autovec_copy(tCrD, tCsC)

            tCrC_epilogue = cute.make_fragment_like(tCsC_epilogue)
            cute.arch.sync_threads()
            cute.autovec_copy(tCsC_epilogue, tCrC_epilogue)

            if cutlass.const_expr(skip_predication):
                # Unpredicated bulk store — all elements in bounds.
                cute.copy(tiled_copy_C, tCrC_epilogue, tCgC_epilogue)
            else:
                ceilM, ceilN = cute.ceil_div(mC.shape, (self.bM, self.bN))
                mcC = cute.make_identity_tensor(
                    (
                        cute.size(ceilM) * self.cta_tiler[0],
                        cute.size(ceilN) * self.cta_tiler[1],
                    )
                )
                cC = cute.local_tile(mcC, tiler=self.cta_tiler, coord=tiler_coord, proj=(1, 1, None))
                tCcC = thr_copy_C.partition_S(cC)

                tCpC = cute.make_rmem_tensor(
                    cute.make_layout(
                        (tCgC_epilogue.shape[0][1], cute.size(tCgC_epilogue, mode=[1]),
                         cute.size(tCgC_epilogue, mode=[2])),
                        stride=(cute.size(tCgC_epilogue, mode=[1]), 1, 0),
                    ),
                    cutlass.Boolean,
                )
                for rest_v in range(tCpC.shape[0]):
                    for m in range(tCpC.shape[1]):
                        tCpC[rest_v, m, 0] = cute.elem_less(tCcC[(0, rest_v), m, 0][0], mC.shape[0])

                for rest_v in range(tCpC.shape[0]):
                    for n in range(tCpC.shape[2]):
                        if cute.elem_less(tCcC[(0, rest_v), 0, n][1], mC.shape[1]):
                            cute.copy(tiled_copy_C, tCrC_epilogue[None, None, n],
                                      tCgC_epilogue[None, None, n], pred=tCpC[None, None, n])

    def _make_smem_layout_AB(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = (
            smem_tiler[1] if major_mode == utils.LayoutEnum.ROW_MAJOR else smem_tiler[0]
        )
        major_mode_size = 64 if major_mode_size >= 64 else major_mode_size

        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)

        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            layout_atom_outer,
        )
        layout = cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))
        return layout

    def _make_smem_layout_C(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = (
            smem_tiler[1] if major_mode == utils.LayoutEnum.ROW_MAJOR else smem_tiler[0]
        )

        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)

        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 4),
            0,
            layout_atom_outer,
        )

        if major_mode == utils.LayoutEnum.COL_MAJOR:
            layout_atom = cute.make_composed_layout(
                cute.make_swizzle(0, 3, 4), 0, layout_atom_outer
            )
        layout = cute.tile_to_shape(
            layout_atom,
            smem_tiler,
            (0, 1),
        )
        return layout

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
            cute.make_layout((1, copy_elems))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

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
            cute.make_layout((1, copy_elems))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    def raster_tile(self, i, j, f):
        new_i = i // f
        new_j = (i % f) + (j * f)
        return (new_i, new_j)


# ---------------------------------------------------------------------------
# AOT placeholder sizes (arbitrary, only used to seed compilation;
# all dims are marked dynamic so the exported kernel works for any M/N/K).
# ---------------------------------------------------------------------------
AOT_M = 256
AOT_N = 512
AOT_K = 128


def _create_cute_tensors(M, N, K, ab_dtype, c_dtype, export_only):
    """Create CuPy GPU tensors and wrap them as CuTe tensors with dynamic dims."""
    dt = cp.float16

    a_cp, b_cp, c_cp = create_row_major_2d_tensors(
        M, N, K, fill_random=not export_only, dtype=dt
    )

    mA = mark_2d_row_major_dynamic(to_cute_tensor(a_cp))
    mB = mark_2d_row_major_dynamic(to_cute_tensor(b_cp))
    mC = mark_2d_row_major_dynamic(to_cute_tensor(c_cp))

    return mA, mB, mC, a_cp, b_cp, c_cp




def run(
    mnk: Tuple[int, int, int],
    cta_tiler_mnk: Tuple[int, int, int] = (16, 128, 128),
    atom_layout_mnk: Tuple[int, int, int] = (1, 4, 1),
    num_stages: int = 3,
    warmup_iterations: int = 2,
    iterations: int = 100,
    skip_ref_check: bool = False,
    export_only: bool = False,
    output_dir: str = "./gemm_aot_artifacts",
    file_name: str = "gemm_ampere",
    function_prefix: str = "gemm_ampere",
    gpu_arch: str = "",
    use_unpredicated: bool = False,
    fused_epilogue: str = "none",
):
    """Run or export an Ampere GEMM kernel.

    Args:
        fused_epilogue: Epilogue fusion mode.
            "none"      — plain GEMM (C = A @ B^T)
            "bias"      — GEMM + bias add (C = A @ B^T + bias)
            "bias_silu" — GEMM + bias + SiLU (C = SiLU(A @ B^T + bias))
    """
    M, N, K = mnk
    ab_dtype = cutlass.Float16
    c_dtype = cutlass.Float16
    acc_dtype = cutlass.Float32
    _tag = f"[{file_name}]"

    if fused_epilogue not in ("none", "bias", "bias_silu"):
        raise ValueError(f"Unknown fused_epilogue={fused_epilogue!r}")

    if not export_only and cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU is required.")

    epilogue_str = f" +{fused_epilogue}" if fused_epilogue != "none" else ""
    print(f"{_tag} GEMM Ampere FP16: M={M}, N={N}, K={K}{epilogue_str}")
    print(f"{_tag} A[{M},{K}] row-major x B[{N},{K}]^T → C[{M},{N}] row-major")
    print(f"{_tag} Tile {cta_tiler_mnk}, MMA (16,8,16), atoms {atom_layout_mnk}, stages={num_stages}")

    if export_only:
        mA = create_fake_row_major_2d_tensor(dtype=ab_dtype)
        mB = create_fake_row_major_2d_tensor(dtype=ab_dtype)
        mC = create_fake_row_major_2d_tensor(dtype=c_dtype)
        a_cp = b_cp = c_cp = None
    else:
        mA, mB, mC, a_cp, b_cp, c_cp = _create_cute_tensors(
            M, N, K, ab_dtype, c_dtype, False
        )

    # Bias tensor for fused epilogues.
    mBias = None
    bias_cp = None
    if fused_epilogue in ("bias", "bias_silu"):
        mBias, bias_cp = create_bias_tensor(N, export_only=export_only)

    use_silu = (fused_epilogue == "bias_silu")

    gemm = GemmAmpereFP16(
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        cta_tiler_mnk=cta_tiler_mnk,
        num_stages=num_stages,
        atom_layout_mnk=atom_layout_mnk,
    )
    current_stream = (
        aot_placeholders.make_stream()
        if export_only
        else cuda.CUstream(cp.cuda.get_current_stream().ptr)
    )

    compile_opts = aot_placeholders.compile_options(
        ("--gpu-arch " + gpu_arch) if gpu_arch else ""
    ) if export_only else None
    extra_str = " +unpredicated" if use_unpredicated else ""
    print(f"{_tag} Compiling kernel (gpu_arch={gpu_arch or 'default'}){extra_str}{epilogue_str}...")
    t0 = time.time()
    compiled_gemm = cute.compile(
        gemm, mA, mB, mC, current_stream,
        use_silu=use_silu,
        mBias=mBias,
        skip_predication=use_unpredicated,
        **(dict(options=compile_opts) if export_only else {}),
    )
    print(f"{_tag} Compilation time: {time.time() - t0:.4f}s")

    if export_only:
        export_compiled_kernel(
            compiled_gemm,
            output_dir=output_dir,
            file_name=file_name,
            function_prefix=function_prefix,
            tag=_tag,
        )
        return

    # ---- Correctness check ----
    if not skip_ref_check:
        compiled_gemm(mA, mB, mC, current_stream)
        cp.cuda.Device().synchronize()

        a_f32 = a_cp.astype(cp.float32)
        b_f32 = b_cp.astype(cp.float32)
        ref_f32 = cp.matmul(a_f32, b_f32.T)

        if fused_epilogue in ("bias", "bias_silu"):
            ref_f32 = ref_f32 + bias_cp.astype(cp.float32)
        if fused_epilogue == "bias_silu":
            ref_f32 = ref_f32 * (1.0 / (1.0 + cp.exp(-ref_f32)))

        ref_cp = ref_f32.astype(cp.float16)

        c_host = c_cp.get()
        ref_host = ref_cp.get()
        if not np.allclose(c_host, ref_host, atol=1e-2, rtol=1e-2):
            max_diff = np.max(np.abs(c_host.astype(np.float32) - ref_host.astype(np.float32)))
            raise ValueError(
                f"{_tag} Verification FAILED! max_diff={max_diff:.6f}"
            )
        print(f"{_tag} Verification PASSED")

    # ---- Benchmark ----
    def generate_tensors():
        mA_w, mB_w, mC_w, _, _, _ = _create_cute_tensors(
            M, N, K, ab_dtype, c_dtype, False
        )
        return testing.JitArguments(mA_w, mB_w, mC_w, current_stream)

    avg_time_us = testing.benchmark(
        compiled_gemm,
        workspace_generator=generate_tensors,
        workspace_count=1,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        use_cuda_graphs=False,
    )

    tflops = 2.0 * M * N * K / (avg_time_us * 1e-6) / 1e12
    print(f"{_tag} Avg time: {avg_time_us:.2f} us, {tflops:.2f} TFLOPS")
    return avg_time_us


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CuTe DSL Ampere GEMM (FP16): AOT export and test (CuPy only)."
    )
    p.add_argument(
        "--mnk", type=parse_comma_separated_ints, default=(256, 512, 128),
        help="M,N,K dimensions (default: 256,512,128)",
    )
    p.add_argument(
        "--cta_tiler_mnk", type=parse_comma_separated_ints, default=(16, 128, 128),
        help="CTA tile shape M,N,K (default: 16,128,128)",
    )
    p.add_argument(
        "--atom_layout_mnk", type=parse_comma_separated_ints, default=(1, 4, 1),
        help="Atom layout MNK (default: 1,4,1)",
    )
    p.add_argument("--num_stages", type=int, default=3)
    p.add_argument("--warmup_iterations", type=int, default=2)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--skip_ref_check", action="store_true")
    p.add_argument("--export_only", action="store_true")
    p.add_argument("--output_dir", type=str, default="./gemm_aot_artifacts")
    p.add_argument("--file_name", type=str, default="gemm_ampere")
    p.add_argument("--function_prefix", type=str, default="gemm_ampere")
    p.add_argument("--gpu_arch", type=str, default="",
        help="Target GPU arch for export (e.g. sm_87). Empty = current GPU.")
    p.add_argument("--use_unpredicated", action="store_true",
        help="Use unpredicated fast path (requires tile-aligned shapes).")
    p.add_argument("--fused_epilogue", type=str, default="none",
        choices=["none", "bias", "bias_silu"],
        help="Fuse epilogue: none (plain GEMM), bias (GEMM+bias), bias_silu (GEMM+bias+SiLU).")
    return p.parse_known_args(args=argv)[0]


def main():
    args = _parsed_args
    run(
        mnk=args.mnk,
        cta_tiler_mnk=args.cta_tiler_mnk,
        atom_layout_mnk=args.atom_layout_mnk,
        num_stages=args.num_stages,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        skip_ref_check=args.skip_ref_check,
        export_only=args.export_only,
        output_dir=args.output_dir,
        file_name=args.file_name,
        function_prefix=args.function_prefix,
        gpu_arch=args.gpu_arch,
        use_unpredicated=args.use_unpredicated,
        fused_epilogue=args.fused_epilogue,
    )


if __name__ == "__main__":
    _parsed_args = _parse_args(_saved_argv)
    main()
    print("PASS")
