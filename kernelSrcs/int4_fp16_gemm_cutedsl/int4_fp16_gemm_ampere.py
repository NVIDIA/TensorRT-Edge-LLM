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

# Standalone CuPy entry point: run a correctness check against a Torch reference,
# or AOT-export a single (tile, stages, split_k) variant for build_cutedsl.py.
#
#   Test:   python int4_fp16_gemm_ampere.py --mnk 256,512,1024 --split_k 4
#   AOT:    python int4_fp16_gemm_ampere.py --mnk 256,512,1024 --split_k 4 \
#               --export_only --output_dir ./out \
#               --file_name int4_fp16_gemm_16x128x64_s2_sk4 \
#               --function_prefix int4_fp16_gemm_16x128x64_s2_sk4
"""CuTe DSL INT4 (W4A16) GEMM with an Ampere tensor-core mainloop.

Computes the repo-canonical W4A16 contract

    C[M, N] = A[M, K] @ dequant(QW[ceil(N/2), K], scales[ceil(K/G), N])^T

with FP16 A/scales/C, FP32 accumulation, INT4 weights, group size ``G`` (128).
Both A and B are staged through shared memory with cp.async.  B uses the
**repacked** path: an offline fragment-order uint32 weight buffer (the nibble
swizzle is done on the host; see :mod:`int4_reference`) is cp.async-staged as a
coalesced copy and dequantized in the mainloop with one lean
``_dequant_int4_word`` per N-group (two ``lop3`` + ``sub.f16x2`` +
``fma.f16x2``, no per-fragment fixup).  Optional in-kernel serial split-K and a
runtime grouped-M threadblock swizzle.

Constraints: ``K % 64 == 0`` and ``group_size`` a multiple of 16 that is mutually
divisible with ``bK`` (=64) -- validated for 128 and 32 (one ``bK=64`` kernel
serves both via per-group scale staging). N residue is predicated. A baked
``split_k`` factor is correct only when it divides ``ceil(K / 64)``
(``split_k == 1`` always holds); the consumer must pick a valid factor.

Exported ABI (per compiled variant):

    (mA, mQW, mScales, mC, mWorkspace, mLocks, swizzle, stream)

``mWorkspace`` is a placeholder (pass ``mC``) — the serial reduction folds into
``mC`` in place.  ``mLocks`` is a zero-initialized int32 semaphore array with
``ceil(M/bM) * ceil(N/bN)`` entries (a 1-element dummy is fine for
``split_k == 1``).  ``swizzle`` is a runtime Int32 grouped-M width (1 = none).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Tuple, Type

# Reset argv before importing cutlass: the DSL evaluates argparse-like state at
# import and the kernel script owns its own CLI (mirrors gemm_ampere.py).
_parsed_args = None
_saved_argv = None
if __name__ == "__main__":
    _saved_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]

import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float16, Float32, Int32, Uint32
from cutedsl_utils import aot_placeholders

from common import (
    ceil_div,
    export_compiled_kernel,
    make_lock_placeholder,
    make_row_major_2d_placeholder,
    mark_lock_1d,
    mark_row_major_2d,
    parse_comma_separated_ints,
)
from int4_dequant import _dequant_int4_word


class Int4Fp16GemmAmpere:
    """Prefill W4A16 GEMM using FP16 tensor cores and the repacked-B mainloop."""

    def __init__(
        self,
        *,
        ab_dtype: Type[cutlass.Numeric] = cutlass.Float16,
        c_dtype: Type[cutlass.Numeric] = cutlass.Float16,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        cta_tiler_mnk: Tuple[int, int, int] = (16, 128, 64),
        atom_layout_mnk: Tuple[int, int, int] = (1, 4, 1),
        num_stages: int = 2,
        group_size: int = 128,
        split_k: int = 1,
        serial_split_k: bool = True,
    ) -> None:
        self.ab_dtype = ab_dtype
        self.c_dtype = c_dtype
        self.acc_dtype = acc_dtype
        self.cta_tiler = cta_tiler_mnk
        self.atom_layout_mnk = atom_layout_mnk
        self.num_stages = num_stages
        self.group_size = group_size
        # (Grouped-M threadblock swizzle is a *runtime* kernel arg, not a member;
        # see the ``swizzle`` parameter of ``__call__`` / ``kernel``.)
        # Split-K factor: how many ways the K reduction is partitioned across the
        # grid's z dimension.  1 = no split (single CTA owns the full K loop).
        # >1 launches `split_k` CTAs per output tile, each accumulating an equal
        # contiguous block of K-tiles.  The caller must ensure `split_k` divides
        # ceil(K / bK) (every z slice owns an equal, statically-sized block).
        self.split_k = split_k
        # Serial split-K: the z-slices of a given output tile reduce *in-kernel*
        # into the fp16 C buffer under a per-tile semaphore: slice z spin-waits
        # until the lock equals z, adds its partial, and releases the lock to z+1
        # (the last slice writes C and resets the lock to 0).  No extra launch and
        # no split_k x workspace.  Only meaningful when ``split_k > 1``
        # (``split_k == 1`` runs the plain epilogue).
        self.serial_split_k = serial_split_k
        # B is staged as 32-bit fragment-order words (the repacked layout) and
        # dequantized with one lean ``_dequant_int4_word`` per N-group.
        self.b_smem_dtype = Uint32

        atom_lay_m, atom_lay_n, atom_lay_k = self.atom_layout_mnk
        self.num_threads = atom_lay_m * atom_lay_n * atom_lay_k * 32
        self.bM, self.bN, self.bK = self.cta_tiler
        self.mma_inst_shape = (16, 8, 16)

        mma_m, mma_n, mma_k = self.mma_inst_shape
        if self.bM % (atom_lay_m * mma_m) != 0:
            raise ValueError("CTA M tile must be divisible by the MMA M coverage")
        if self.bN % (atom_lay_n * mma_n) != 0:
            raise ValueError("CTA N tile must be divisible by the MMA N coverage")
        if atom_lay_k != 1 or self.bK % mma_k != 0:
            raise ValueError("This SM80 mainloop expects atom_layout_k=1 and bK % 16 == 0")
        if self.num_stages < 2:
            raise ValueError("CUTLASS-style W4A16 pipeline requires at least 2 stages")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")

        # Repacked-B fragment geometry (see ``int4_reference``).  Each k-block is
        # a 16x{bN} slab; each thread owns one 32-bit word per (k-block, N-pair).
        if self.bN % 64 != 0:
            raise ValueError("B path requires bN divisible by 64")
        if self.bK % 16 != 0:
            raise ValueError("B path requires bK divisible by 16")
        self.k_blocks = self.bK // 16
        self.n_pairs = self.bN // 64
        # Scale (quant-group) granularity is decoupled from bK: scales are applied
        # per 16-element k-block, so ``group_size`` must be a multiple of 16, and
        # group boundaries must align with K-tile boundaries (one of bK/group_size
        # divides the other).  ``groups_per_tile`` = how many distinct scale rows a
        # single bK K-tile spans: 1 when group_size >= bK (e.g. 128 with bK=64), >1
        # for fine group sizes (e.g. 2 for group_size=32, bK=64).  Lets one bK=64
        # kernel serve both group_size 128 and 32.
        if self.group_size % 16 != 0:
            raise ValueError("B path requires group_size divisible by 16")
        if self.bK % self.group_size != 0 and self.group_size % self.bK != 0:
            raise ValueError("group_size and bK must be compatible (one divides the other)")
        self.groups_per_tile = max(1, self.bK // self.group_size)
        # The number of K-tiles in the full problem is passed into the mainloop
        # dynamically as ``k_tile_count`` (derived at trace time from A's K
        # extent), so the compiled kernel is shape-agnostic in K.

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mQW: cute.Tensor,
        mScales: cute.Tensor,
        mC: cute.Tensor,
        mWorkspace: cute.Tensor,
        mLocks: cute.Tensor,
        swizzle: Int32,
        stream: cuda.CUstream,
    ) -> None:
        # ``mWorkspace`` is a placeholder (the serial reduction folds into mC in
        # place; the caller passes mC).  ``mLocks`` is the int32 per-output-tile
        # semaphore used by the serial path (a 1-element placeholder for
        # split_k == 1).  Project layout modes once so the helper builders
        # produce the same shared-memory swizzles and ldmatrix variants as the
        # dense Ampere GEMM.
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.ROW_MAJOR
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)

        ab_copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            mA.element_type,
            self.a_major_mode,
            ab_copy_bits,
            (self.bM, self.bK, self.num_stages),
        )
        # B is staged as 32-bit fragment-order words (``_make_smem_layout_B``,
        # word-major); ``sB_fragment_layout`` is only used to derive the MMA
        # B-fragment shape (tCsB/tCrB) -- the data is read via the logical
        # ``sB_bytes`` index in ``_load_b_fragment_from_smem``.
        sB_byte_layout = self._make_smem_layout_B(
            self.b_smem_dtype,
            self.b_major_mode,
            64,
            (self.bN, self.bK, self.num_stages),
        )
        sB_fragment_layout = self._make_smem_layout_AB(
            self.b_smem_dtype,
            self.b_major_mode,
            64,
            (self.bN, self.bK, self.num_stages),
        )
        sScale_layout = cute.make_layout(
            (self.num_stages, self.groups_per_tile, self.bN),
            stride=(self.groups_per_tile * self.bN, self.bN, 1),
        )
        sC_layout = self._make_smem_layout_C(
            mC.element_type,
            self.c_major_mode,
            ab_copy_bits,
            (self.bM, self.bN),
        )

        smem_size = max(
            cute.size_in_bytes(mC.element_type, sC_layout),
            cute.size_in_bytes(mA.element_type, sA_layout)
            + cute.size_in_bytes(self.b_smem_dtype, sB_byte_layout)
            + cute.size_in_bytes(mScales.element_type, sScale_layout),
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
        tiled_mma = cute.make_tiled_mma(
            op,
            cute.make_layout(self.atom_layout_mnk),
            permutation_mnk=permutation_mnk,
        )

        grid_dim = cute.ceil_div(mC.shape, (self.bM, self.bN))
        self.kernel(
            mA,
            mQW,
            mScales,
            mC,
            mWorkspace,
            mLocks,
            sA_layout,
            sB_byte_layout,
            sB_fragment_layout,
            sScale_layout,
            sC_layout,
            tiled_copy_A,
            tiled_copy_C,
            tiled_mma,
            swizzle,
        ).launch(
            grid=(cute.size(grid_dim[0]), cute.size(grid_dim[1]), self.split_k),
            block=[self.num_threads, 1, 1],
            smem=smem_size,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mQW: cute.Tensor,
        mScales: cute.Tensor,
        mC: cute.Tensor,
        mWorkspace: cute.Tensor,
        mLocks: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_byte_layout,
        sB_fragment_layout: cute.ComposedLayout,
        sScale_layout: cute.Layout,
        sC_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        swizzle: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_m_idx, block_n_idx, split_k_idx = cute.arch.block_idx()
        # ``swizzle`` is a *runtime* Int32 (not baked), so one compiled kernel
        # serves every swizzle width.  The remap is pure threadblock-index
        # arithmetic gated on a dynamic predicate -- no smem/MMA/structural
        # dependence -- so making it dynamic costs only a few integer ops at
        # entry (and nothing when swizzle == 1).
        if swizzle > 1:
            # Grouped-M rasterization (Triton/CUTLASS-style): remap the launched
            # 2D grid through a bijection so CTAs are issued M-fastest within a
            # group of `swizzle` M-tiles, keeping one weight N-tile hot in L2
            # across `swizzle` consecutive CTAs.
            grid_m, grid_n, _ = cute.arch.grid_dim()
            group = swizzle
            linear = block_n_idx * grid_m + block_m_idx
            blocks_per_group = group * grid_n
            group_id = linear // blocks_per_group
            first_m = group_id * group
            group_m = cutlass.min(grid_m - first_m, group)
            inner = linear - group_id * blocks_per_group
            block_m_idx = first_m + (inner % group_m)
            block_n_idx = inner // group_m
        tiler_coord = (block_m_idx, block_n_idx, None)

        gA = cute.local_tile(
            mA, tiler=self.cta_tiler, coord=tiler_coord, proj=(1, None, 1)
        )
        gC = cute.local_tile(
            mC, tiler=self.cta_tiler, coord=tiler_coord, proj=(1, 1, None)
        )

        # Keep the dense Ampere GEMM residue convention for A.  The dequant B
        # loader receives the same offset so both operands see identical K tiles.
        residual_k = cute.size(mA, mode=[1]) - cutlass.Int32(self.bK) * cute.size(
            gA, mode=[2]
        )
        gA = cute.domain_offset((0, residual_k, 0), gA)
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)

        mcA = cute.make_identity_tensor(mA.layout.shape)
        cA = cute.local_tile(
            mcA, tiler=self.cta_tiler, coord=tiler_coord, proj=(1, None, 1)
        )
        cA = cute.domain_offset((0, residual_k, 0), cA)

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
        sB_bytes = smem.allocate_tensor(self.b_smem_dtype, sB_byte_layout, 16)
        sB_fragment = cute.make_tensor(sB_bytes.iterator, sB_fragment_layout)
        sScale = smem.allocate_tensor(mScales.element_type, sScale_layout, 16)
        sC = cute.make_tensor(
            cute.recast_ptr(sA.iterator, dtype=self.c_dtype), sC_layout
        )

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_C = tiled_copy_C.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tCsC_epilogue = thr_copy_C.partition_S(sC)
        tCgC_epilogue = thr_copy_C.partition_D(gC)
        tAcA = thr_copy_A.partition_S(cA)

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
        for rest_v in range(tApA.shape[0]):
            for m in range(tApA.shape[1]):
                tApA[rest_v, m, 0] = cute.elem_less(
                    tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0]
                )

        tAsA.fill(0)
        # Pre-zero the scale smem.  The cp.async scale staging (below) skips
        # OOB-N groups (never writes those columns) and skips residue/over-fill
        # K-tiles (never writes that stage), so those slots must read 0; the
        # in-bounds columns are overwritten by the async copy before they are
        # read.  Strided so each thread zeros distinct columns across stages.
        for st in range(self.num_stages):
            for sub in range(self.groups_per_tile):
                for item in range((self.bN + self.num_threads - 1) // self.num_threads):
                    sn = tidx + Int32(item * self.num_threads)
                    if sn < Int32(self.bN):
                        sScale[st, sub, sn] = self.ab_dtype(0.0)
        cute.arch.sync_threads()
        num_smem_stages = cute.size(tAsA, mode=[3])
        # Total K-tiles (static at trace).  Split-K partitions them into
        # `split_k` equal contiguous blocks of `tiles_per_split` each -- the
        # backend guarantees `split_k` divides this count, so the block size is
        # a compile-time constant and the mainloop bound stays static.  Slice
        # `split_k_idx` owns tiles [k_tile_start, k_tile_start + tiles_per_split).
        k_tile_count = cute.size(tAgA, mode=[3])
        tiles_per_split = k_tile_count // self.split_k
        k_tile_start = split_k_idx * cutlass.Int32(tiles_per_split)
        k_tile_index = k_tile_start

        cB = cute.make_identity_tensor((self.bN, self.bK, self.num_stages))

        # Prologue: cp.async A, stage scales, and keep biased INT4 B packed in sB.
        self._load_scale_tile(
            mScales, sScale, tidx, block_n_idx, residual_k, k_tile_index, 0, mA.shape[1], mC.shape[1]
        )
        for k in range(tApA.shape[2]):
            if cute.elem_less(cutlass.Int32(-1), tAcA[0, 0, k, 0][1]):
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, k, k_tile_index],
                    tAsA[None, None, k, 0],
                    pred=tApA[None, None, k],
                )
        self._load_b_tile(
            mQW, sB_bytes, tidx, block_n_idx, residual_k, k_tile_index, 0, mA.shape[1], mC.shape[1], k_tile_count
        )
        k_tile_index = k_tile_index + 1
        cute.arch.cp_async_commit_group()

        for stage in range(1, num_smem_stages - 1):
            if stage == tiles_per_split:
                tApA.fill(0)
            self._load_scale_tile(
                mScales,
                sScale,
                tidx,
                block_n_idx,
                residual_k,
                k_tile_index,
                stage,
                mA.shape[1],
                mC.shape[1],
            )
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, k_tile_index],
                tAsA[None, None, None, stage],
                pred=tApA,
            )
            self._load_b_tile(
                mQW,
                sB_bytes,
                tidx,
                block_n_idx,
                residual_k,
                k_tile_index,
                stage,
                mA.shape[1],
                mC.shape[1],
                k_tile_count,
            )
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB_fragment)
        tCsC = thr_mma.partition_C(sC)
        tCgC = thr_mma.partition_C(gC)
        tCcB = thr_mma.partition_B(cB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        atom_copy_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
            ),
            mA.element_type,
        )
        tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)

        thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tidx)
        tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
        tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)

        smem_pipe_read = 0
        smem_pipe_write = num_smem_stages - 1
        tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]

        num_k_block = cute.size(tCrA, mode=[2])
        if num_k_block > 1:
            cute.arch.cp_async_wait_group(num_smem_stages - 2)
            cute.arch.sync_threads()
            cute.copy(
                tiled_copy_s2r_A,
                tCsA_p[None, None, 0],
                tCrA_copy_view[None, None, 0],
            )
            self._load_b_fragment_from_smem(
                sB_bytes,
                tCrB[None, None, 0],
                tCcB[None, None, 0, smem_pipe_read],
                sScale,
                smem_pipe_read,
            )

        for k_tile in range(tiles_per_split):
            current_smem_stage = smem_pipe_read
            for k_block in cutlass.range(num_k_block, unroll_full=True):
                b_load_stage = current_smem_stage
                if k_block == num_k_block - 1:
                    tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                    b_load_stage = smem_pipe_read
                    cute.arch.cp_async_wait_group(num_smem_stages - 2)
                    cute.arch.sync_threads()

                k_block_next = (k_block + 1) % num_k_block
                cute.copy(
                    tiled_copy_s2r_A,
                    tCsA_p[None, None, k_block_next],
                    tCrA_copy_view[None, None, k_block_next],
                )
                self._load_b_fragment_from_smem(
                    sB_bytes,
                    tCrB[None, None, k_block_next],
                    tCcB[None, None, k_block_next, b_load_stage],
                    sScale,
                    b_load_stage,
                )

                if k_block == 0:
                    if k_tile + num_smem_stages - 1 < tiles_per_split:
                        self._load_scale_tile(
                            mScales,
                            sScale,
                            tidx,
                            block_n_idx,
                            residual_k,
                            k_tile_index,
                            smem_pipe_write,
                            mA.shape[1],
                            mC.shape[1],
                        )
                        cute.copy(
                            tiled_copy_A,
                            tAgA[None, None, None, k_tile_index],
                            tAsA[None, None, None, smem_pipe_write],
                            pred=tApA,
                        )
                        self._load_b_tile(
                            mQW,
                            sB_bytes,
                            tidx,
                            block_n_idx,
                            residual_k,
                            k_tile_index,
                            smem_pipe_write,
                            mA.shape[1],
                            mC.shape[1],
                            k_tile_count,
                        )

                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    tCrC,
                )

                if k_block == 0:
                    k_tile_index = k_tile_index + 1
                    cute.arch.cp_async_commit_group()
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == num_smem_stages:
                        smem_pipe_read = 0

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # Shared coalesced epilogue staging (split_k == 1 AND serial split-K):
        # round the FP32 accumulator to fp16 in registers, bounce it through smem
        # so the per-thread store fragment is in coalesced row-major order, and
        # reload it as ``tCrC_epilogue`` -- this is what makes the global C stores
        # 128-bit-coalesced (the raw MMA-C fragment layout is scattered in N).
        tCrD = cute.make_fragment_like(tCrC, self.c_dtype)
        tCrD[None] = tCrC.load().to(self.c_dtype)
        cute.autovec_copy(tCrD, tCsC)

        tCrC_epilogue = cute.make_fragment_like(tCsC_epilogue)
        cute.arch.sync_threads()
        cute.autovec_copy(tCsC_epilogue, tCrC_epilogue)

        ceil_m, ceil_n = cute.ceil_div(mC.shape, (self.bM, self.bN))
        mcC = cute.make_identity_tensor(
            (
                cute.size(ceil_m) * self.bM,
                cute.size(ceil_n) * self.bN,
            )
        )
        cC = cute.local_tile(
            mcC, tiler=self.cta_tiler, coord=tiler_coord, proj=(1, 1, None)
        )
        tCcC = thr_copy_C.partition_S(cC)
        tCpC = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tCgC_epilogue.shape[0][1],
                    cute.size(tCgC_epilogue, mode=[1]),
                    cute.size(tCgC_epilogue, mode=[2]),
                ),
                stride=(cute.size(tCgC_epilogue, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(tCpC.shape[0]):
            for m in range(tCpC.shape[1]):
                tCpC[rest_v, m, 0] = cute.elem_less(
                    tCcC[(0, rest_v), m, 0][0], mC.shape[0]
                )

        if cutlass.const_expr(self.split_k > 1 and self.serial_split_k):
            # Serial split-K: coalesced in-place reduction into fp16 C.
            # ``tCrC_epilogue`` already holds this slice's
            # fp16 partial in the coalesced store layout.  Under the per-tile
            # semaphore, slice 0 stores it to C and later slices load the running
            # C (same 128-bit coalesced copy), add (in FP32) and store back -- no
            # FP32 ``[M,N]`` workspace, no scattered MMA-layout stores.  The last
            # slice's store is the final result; it also resets the lock to 0.
            grid_m, grid_n, _ = cute.arch.grid_dim()
            tile_id = block_m_idx * grid_n + block_n_idx
            lock_ptr = cute.domain_offset((tile_id,), mLocks).iterator
            last = Int32(self.split_k - 1)

            if tidx == 0:
                old = cute.arch.atomic_add(lock_ptr, Int32(0), sem="relaxed", scope="gpu")
                while old != split_k_idx:
                    old = cute.arch.atomic_add(
                        lock_ptr, Int32(0), sem="relaxed", scope="gpu"
                    )
                cute.arch.fence_acq_rel_gpu()
            cute.arch.sync_threads()

            if split_k_idx != Int32(0):
                # Load the running C (coalesced) and add this slice's partial.
                tCgC_src = thr_copy_C.partition_S(gC)
                prev = cute.make_fragment_like(tCrC_epilogue)
                for rest_v in range(tCpC.shape[0]):
                    for n in range(tCpC.shape[2]):
                        if cute.elem_less(tCcC[(0, rest_v), 0, n][1], mC.shape[1]):
                            cute.copy(
                                tiled_copy_C,
                                tCgC_src[None, None, n],
                                prev[None, None, n],
                                pred=tCpC[None, None, n],
                            )
                tCrC_epilogue[None] = (
                    tCrC_epilogue.load().to(Float32) + prev.load().to(Float32)
                ).to(self.c_dtype)

            for rest_v in range(tCpC.shape[0]):
                for n in range(tCpC.shape[2]):
                    if cute.elem_less(tCcC[(0, rest_v), 0, n][1], mC.shape[1]):
                        cute.copy(
                            tiled_copy_C,
                            tCrC_epilogue[None, None, n],
                            tCgC_epilogue[None, None, n],
                            pred=tCpC[None, None, n],
                        )

            cute.arch.sync_threads()
            if tidx == 0:
                cute.arch.fence_acq_rel_gpu()
                next_lock = Int32(0)
                if split_k_idx != last:
                    next_lock = split_k_idx + Int32(1)
                cute.arch.atomic_exch(lock_ptr, next_lock, sem="release", scope="gpu")
            return

        for rest_v in range(tCpC.shape[0]):
            for n in range(tCpC.shape[2]):
                if cute.elem_less(tCcC[(0, rest_v), 0, n][1], mC.shape[1]):
                    cute.copy(
                        tiled_copy_C,
                        tCrC_epilogue[None, None, n],
                        tCgC_epilogue[None, None, n],
                        pred=tCpC[None, None, n],
                    )

    @cute.jit
    def _load_b_tile(
        self,
        mQW: cute.Tensor,
        sB_bytes: cute.Tensor,
        tidx: Int32,
        block_n_idx: Int32,
        residual_k: Int32,
        k_tile_index: Int32,
        stage: Int32,
        k_extent: Int32,
        n_extent: Int32,
        num_k_tiles: Int32,
    ) -> None:
        # ``mQW`` is the offline-repacked uint32 buffer
        # ``[num_n_blocks * num_k_tiles * kn, 128]`` (see ``int4_reference``):
        # row = (block_n * num_k_tiles + k_tile) * kn + word, column = thread.
        # The ``kn x num_threads`` slab for this (block_n, k_tile) is a coalesced
        # (word, thread) block -- consecutive threads read consecutive columns --
        # so we stage it with cp.async straight into shared memory, exactly as
        # the A operand does (drops the global->register->shared round-trip and
        # overlaps the B load with compute; the async copy joins A's commit
        # group, gated by the mainloop's cp_async_wait_group + sync_threads).
        kn = self.k_blocks * self.n_pairs
        # Split-K can over-fill the pipeline: when tiles_per_split < stages-1 the
        # prologue prefetches past this z-slice.  cp.async is unpredicated here,
        # so clamp the source K-tile to the last valid one (block_n_idx is always
        # < num_n_blocks, so the clamped row is in-buffer).  The over-fill stage
        # stages a valid-but-wrong tile, never consumed by the mainloop MMA.
        eff_k_tile = cutlass.min(k_tile_index, num_k_tiles - Int32(1))
        base_row = (
            block_n_idx * num_k_tiles + eff_k_tile
        ) * Int32(kn)

        # Source: the kn x num_threads (word, thread) slab at ``base_row``,
        # row-major in mQW (stride (num_threads, 1)).  base_row is a multiple of
        # kn and the row width is num_threads (128 uint32 = 512 B), so the slab
        # start is 16 B-aligned; assert it so the 128-bit cp.async atom is
        # accepted (the dynamic offset hides this from static analysis).
        gB_off = cute.domain_offset((base_row, Int32(0)), mQW)
        gB = cute.make_tensor(
            gB_off.iterator.align(16),
            cute.make_layout((kn, self.num_threads), stride=(self.num_threads, 1)),
        )
        # Dest: the same (word, thread) view into this stage of the word-major
        # smem buffer (word stride num_threads, thread stride 1) -- matching the
        # source so the copy is contiguous along the thread axis on both sides.
        sB_off = cute.domain_offset((Int32(0), Int32(0), Int32(0), stage), sB_bytes)
        sB_stage = cute.make_tensor(
            sB_off.iterator.align(16),
            cute.make_layout((kn, self.num_threads), stride=(self.num_threads, 1)),
        )

        tiled_copy_b = self._make_b_async_copy()
        thr = tiled_copy_b.get_slice(tidx)
        cute.copy(
            tiled_copy_b,
            thr.partition_S(gB),
            thr.partition_D(sB_stage),
        )

    def _make_b_async_copy(self) -> cute.TiledCopy:
        # cp.async tiled copy for the (word, thread) B slab, both sides
        # contiguous along the thread (column) axis.  The DSL cp.async atom
        # requires 128-bit transactions, so each copy moves 4 contiguous uint32
        # words along the column axis; consecutive threads -> consecutive 16 B
        # chunks -> coalesced.  Mirrors ``_make_gmem_tiled_copy_AB`` (which
        # vectorizes A's row-major K axis) applied to the column axis here.
        copy_bits = 128
        copy_elems = copy_bits // self.b_smem_dtype.width  # 4 uint32 per copy
        col_groups = self.num_threads // copy_elems        # thread groups on cols
        if self.num_threads % copy_elems != 0:
            raise ValueError("num_threads must be divisible by 4 for 128b cp.async")
        atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            self.b_smem_dtype,
            num_bits_per_copy=copy_bits,
        )
        word_threads = self.num_threads // col_groups
        thread_layout = cute.make_layout(
            (word_threads, col_groups), stride=(col_groups, 1)
        )
        value_layout = cute.make_layout((1, copy_elems))
        return cute.make_tiled_copy_tv(atom, thread_layout, value_layout)

    @cute.jit
    def _load_b_fragment_from_smem(
        self,
        sB_bytes: cute.Tensor,
        tCrB: cute.Tensor,
        tCcB: cute.Tensor,
        sScale: cute.Tensor,
        stage: Int32,
    ) -> None:
        # Plain word load + lean dequant.  Group gN occupies fragment elements
        # [4*gN : 4*gN+4]; the four groups of a k-block come from n_pairs words
        # (two groups per word via the q / q>>8 split).  ``tCcB[base][0]`` gives
        # each group's local_n -> its scale; element 0's local_k gives k-block.
        tidx, _, _ = cute.arch.thread_idx()
        kbl = tCcB[0][1] // Int32(16)
        # Which quant sub-group this k-block falls in (its local-K offset
        # tCcB[0][1] // group_size).  const_expr fast path keeps group_size >= bK
        # at sub=0 (one scale row per tile) with no runtime division.
        if cutlass.const_expr(self.groups_per_tile == 1):
            sub = Int32(0)
        else:
            sub = tCcB[0][1] // Int32(self.group_size)
        tCrB_u32 = cute.recast_tensor(tCrB, Uint32)
        for p in range(self.n_pairs):
            word = sB_bytes[tidx, kbl, Int32(p), stage]
            g_lo = 2 * p
            g_hi = 2 * p + 1
            scale_lo = sScale[stage, sub, tCcB[g_lo * 4][0]]
            scale_hi = sScale[stage, sub, tCcB[g_hi * 4][0]]
            lo0, lo1 = _dequant_int4_word(word, scale_lo)
            hi0, hi1 = _dequant_int4_word(word >> Uint32(8), scale_hi)
            tCrB_u32[g_lo * 2] = lo0
            tCrB_u32[g_lo * 2 + 1] = lo1
            tCrB_u32[g_hi * 2] = hi0
            tCrB_u32[g_hi * 2 + 1] = hi1

    @cute.jit
    def _load_scale_tile(
        self,
        mScales: cute.Tensor,
        sScale: cute.Tensor,
        tidx: Int32,
        block_n_idx: Int32,
        residual_k: Int32,
        k_tile_index: Int32,
        stage: Int32,
        k_extent: Int32,
        n_extent: Int32,
    ) -> None:
        # Stage scales with cp.async, exactly like A and B, instead of a
        # synchronous global->register->shared round-trip.  Scales are fp16
        # (== ab_dtype), so no dtype conversion is needed; the async copy moves
        # native words and the read-side dequant takes fp16.  The DSL cp.async
        # atom requires 128-bit transactions, so each participating thread copies
        # 8 contiguous fp16 scales along N; bN/8 threads participate (bN is a
        # multiple of 64, so 8 divides it and group bounds land on multiples of
        # 8 == the scale granularity).  OOB-N groups are skipped (the slot was
        # pre-zeroed in the prologue and those N are predicated out of the
        # epilogue); an out-of-range K-tile (residue head or split-K over-fill)
        # skips the whole load.  Completion folds into the mainloop's existing
        # cp_async_wait_group + sync_threads -- no scale-only barrier.
        copy_elems = 128 // self.ab_dtype.width
        n_groups = self.bN // copy_elems
        tile_k = residual_k + k_tile_index * Int32(self.bK)
        if tile_k < k_extent and cute.elem_less(Int32(-1), tile_k):
            if tidx < Int32(n_groups):
                local_n0 = tidx * Int32(copy_elems)
                global_n0 = block_n_idx * Int32(self.bN) + local_n0
                if global_n0 < n_extent:
                    atom = cute.make_copy_atom(
                        cute.nvgpu.cpasync.CopyG2SOp(
                            cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
                        ),
                        mScales.element_type,
                        num_bits_per_copy=128,
                    )
                    # One scale row per sub-group the tile spans.  groups_per_tile
                    # == 1 (group_size >= bK) => the original single-row stage;
                    # >1 (e.g. group_size=32, bK=64) stages each group's row into
                    # sScale[stage, sub, :].
                    for sub in range(self.groups_per_tile):
                        scale_row = (
                            tile_k + Int32(sub * self.group_size)
                        ) // Int32(self.group_size)
                        gS = cute.make_tensor(
                            cute.domain_offset(
                                (scale_row, global_n0), mScales
                            ).iterator.align(16),
                            cute.make_layout((copy_elems,), stride=(1,)),
                        )
                        sS = cute.make_tensor(
                            cute.domain_offset(
                                (stage, sub, local_n0), sScale
                            ).iterator.align(16),
                            cute.make_layout((copy_elems,), stride=(1,)),
                        )
                        cute.copy(atom, gS, sS)

    @cute.jit
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
            cute.make_swizzle(swizzle_bits, 3, 3), 0, layout_atom_outer
        )
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))

    def _make_smem_layout_B(self, dtype, major_mode, copy_bits, smem_tiler):
        # Thread-indexed fragment-order words: sB[tidx, kbl, p, stage] is the
        # 32-bit word holding thread ``tidx``'s eight biased nibbles for k-block
        # ``kbl`` and N-pair ``p``.  Total = num_threads * k_blocks * n_pairs
        # words = bN * bK / 8 = 0.5 byte / INT4.
        #
        # Layout is **word-major / thread-minor** (thread stride 1): element
        # [t, kbl, p, st] sits at ``word*num_threads + t + st*num_threads*kn``
        # with ``word = kbl*n_pairs + p``.  This makes the (word, thread) slab
        # contiguous along the thread axis -- the same direction it is
        # contiguous in the global repacked buffer -- so the cp.async staging
        # can use 128-bit (16 B) transactions.  ``_load_b_fragment_from_smem``
        # indexes this tensor logically, so the stride change is transparent.
        t, kb, np_, st = self.num_threads, self.k_blocks, self.n_pairs, self.num_stages
        kn = kb * np_
        return cute.make_layout(
            (t, kb, np_, st),
            stride=(1, np_ * t, t, t * kn),
        )

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
            cute.make_swizzle(swizzle_bits, 3, 4), 0, layout_atom_outer
        )
        if major_mode == utils.LayoutEnum.COL_MAJOR:
            layout_atom = cute.make_composed_layout(
                cute.make_swizzle(0, 3, 4), 0, layout_atom_outer
            )
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1))

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


def run(
    mnk: Tuple[int, int, int],
    cta_tiler_mnk: Tuple[int, int, int] = (16, 128, 64),
    atom_layout_mnk: Tuple[int, int, int] = (1, 4, 1),
    num_stages: int = 2,
    split_k: int = 1,
    swizzle: int = 1,
    group_size: int = 128,
    warmup_iterations: int = 2,
    iterations: int = 100,
    skip_ref_check: bool = False,
    export_only: bool = False,
    output_dir: str = "./int4_fp16_gemm_aot_artifacts",
    file_name: str = "int4_fp16_gemm_ampere",
    function_prefix: str = "int4_fp16_gemm_ampere",
    gpu_arch: str = "",
):
    """Run or export the INT4 (W4A16) FP16 Ampere GEMM kernel.

    A baked ``split_k`` is correct only when it divides ``ceil(K / 64)``
    (``split_k == 1`` always works).
    """
    M, N, K = mnk
    bM, bN, bK = cta_tiler_mnk
    _tag = f"[{file_name}]"

    if N <= 0:
        raise ValueError(f"N must be positive (got {N})")
    if K % 64 != 0:
        raise ValueError(f"K must be a multiple of 64 (got {K})")
    k_tile_count = ceil_div(K, bK)
    if split_k > 1 and k_tile_count % split_k != 0:
        raise ValueError(
            f"split_k={split_k} must divide ceil(K/{bK})={k_tile_count} "
            f"(K={K}); split_k=1 always works."
        )
    if not export_only and cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU is required.")

    print(f"{_tag} INT4 W4A16 FP16: M={M}, N={N}, K={K}, group_size={group_size}")
    print(f"{_tag} Tile {cta_tiler_mnk}, MMA (16,8,16), atoms {atom_layout_mnk}, "
          f"stages={num_stages}, split_k={split_k} (serial), swizzle={swizzle}")

    kernel = Int4Fp16GemmAmpere(
        ab_dtype=cutlass.Float16,
        c_dtype=cutlass.Float16,
        acc_dtype=cutlass.Float32,
        cta_tiler_mnk=cta_tiler_mnk,
        atom_layout_mnk=atom_layout_mnk,
        num_stages=num_stages,
        group_size=group_size,
        split_k=split_k,
        serial_split_k=True,
    )
    current_stream = (
        aot_placeholders.make_stream()
        if export_only
        else cuda.CUstream(cp.cuda.get_current_stream().ptr)
    )
    swizzle_i32 = cutlass.Int32(swizzle)

    if export_only:
        mA = make_row_major_2d_placeholder(cutlass.Float16)
        mQW = make_row_major_2d_placeholder(cutlass.Uint32)
        mScales = make_row_major_2d_placeholder(cutlass.Float16)
        mC = make_row_major_2d_placeholder(cutlass.Float16)
        mLocks = make_lock_placeholder()

        compile_opts = aot_placeholders.compile_options(
            ("--gpu-arch " + gpu_arch) if gpu_arch else ""
        )
        print(f"{_tag} Compiling kernel (gpu_arch={gpu_arch or 'default'})...")
        t0 = time.time()
        compiled = cute.compile(
            kernel, mA, mQW, mScales, mC, mC, mLocks, swizzle_i32, current_stream,
            options=compile_opts,
        )
        print(f"{_tag} Compilation time: {time.time() - t0:.4f}s")
        export_compiled_kernel(
            compiled,
            output_dir=output_dir,
            file_name=file_name,
            function_prefix=function_prefix,
            tag=_tag,
        )
        return

    # ---- Reference correctness path (uses Torch via int4_reference) ----
    import torch  # noqa: PLC0415 -- only needed for the reference check

    from int4_reference import (
        int4_fp16_reference,
        make_int4_inputs,
        repack_b_for_tile,
    )

    act, qweight, scales = make_int4_inputs(M, N, K, group_size=group_size, device="cuda")
    ref = int4_fp16_reference(act, qweight, scales, group_size=group_size)
    c_t = torch.zeros((M, N), device="cuda", dtype=torch.float16)
    rep = repack_b_for_tile(qweight, N, K, bN, bK).to("cuda")

    a_cp = cp.from_dlpack(act.contiguous())
    scales_cp = cp.from_dlpack(scales.contiguous())
    c_cp = cp.from_dlpack(c_t)
    qw_cp = cp.from_dlpack(rep.contiguous()).view(cp.uint32)
    n_tiles = max(ceil_div(M, bM) * ceil_div(N, bN), 1)
    locks_t = torch.zeros(
        n_tiles if split_k > 1 else 1, device="cuda", dtype=torch.int32
    )
    locks_cp = cp.from_dlpack(locks_t)

    mA = mark_row_major_2d(a_cp)
    mQW = mark_row_major_2d(qw_cp)
    mScales = mark_row_major_2d(scales_cp)
    mC = mark_row_major_2d(c_cp)
    mLocks = mark_lock_1d(locks_cp)

    print(f"{_tag} Compiling kernel...")
    t0 = time.time()
    compiled = cute.compile(
        kernel, mA, mQW, mScales, mC, mC, mLocks, swizzle_i32, current_stream
    )
    print(f"{_tag} Compilation time: {time.time() - t0:.4f}s")

    if not skip_ref_check:
        compiled(mA, mQW, mScales, mC, mC, mLocks, swizzle_i32, current_stream)
        torch.cuda.synchronize()
        diff = (c_t.to(torch.float32) - ref.to(torch.float32)).abs()
        max_abs = diff.max().item()
        denom = max(ref.abs().max().item(), 1e-6)
        rel = max_abs / denom
        # W4A16 FP16-accumulate (split-K adds a little more FP32 rounding); a few
        # percent relative error is expected vs the FP32-accumulate reference.
        if rel > 0.03:
            raise ValueError(
                f"{_tag} Verification FAILED! max_abs_diff={max_abs:.6f} "
                f"rel={rel:.4f} (denom={denom:.4f})"
            )
        print(f"{_tag} Verification PASSED (max_abs_diff={max_abs:.6f}, rel={rel:.4f})")

    # ---- Benchmark ----
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(max(warmup_iterations, 0)):
        compiled(mA, mQW, mScales, mC, mC, mLocks, swizzle_i32, current_stream)
    torch.cuda.synchronize()
    start.record()
    for _ in range(max(iterations, 1)):
        compiled(mA, mQW, mScales, mC, mC, mLocks, swizzle_i32, current_stream)
    end.record()
    torch.cuda.synchronize()
    avg_time_us = start.elapsed_time(end) / max(iterations, 1) * 1e3
    tflops = 2.0 * M * N * K / (avg_time_us * 1e-6) / 1e12
    print(f"{_tag} Avg time: {avg_time_us:.2f} us, {tflops:.2f} TFLOPS")
    return avg_time_us


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CuTe DSL INT4 (W4A16) FP16 Ampere GEMM: AOT export and test."
    )
    p.add_argument(
        "--mnk", type=parse_comma_separated_ints, default=(256, 512, 1024),
        help="M,N,K dimensions (default: 256,512,1024)",
    )
    p.add_argument(
        "--cta_tiler_mnk", type=parse_comma_separated_ints, default=(16, 128, 64),
        help="CTA tile shape M,N,K (default: 16,128,64)",
    )
    p.add_argument(
        "--atom_layout_mnk", type=parse_comma_separated_ints, default=(1, 4, 1),
        help="Atom layout MNK (default: 1,4,1)",
    )
    p.add_argument("--num_stages", type=int, default=2)
    p.add_argument(
        "--split_k", type=int, default=1,
        help="Serial split-K factor (must divide ceil(K/64); 1 = no split).",
    )
    p.add_argument(
        "--swizzle", type=int, default=1,
        help="Grouped-M threadblock swizzle width (runtime arg; 1 = none).",
    )
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--warmup_iterations", type=int, default=2)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--skip_ref_check", action="store_true")
    p.add_argument("--export_only", action="store_true")
    p.add_argument("--output_dir", type=str, default="./int4_fp16_gemm_aot_artifacts")
    p.add_argument("--file_name", type=str, default="int4_fp16_gemm_ampere")
    p.add_argument("--function_prefix", type=str, default="int4_fp16_gemm_ampere")
    p.add_argument(
        "--gpu_arch", type=str, default="",
        help="Target GPU arch for export (e.g. sm_87). Empty = current GPU.",
    )
    return p.parse_known_args(args=argv)[0]


def main():
    args = _parsed_args
    run(
        mnk=args.mnk,
        cta_tiler_mnk=args.cta_tiler_mnk,
        atom_layout_mnk=args.atom_layout_mnk,
        num_stages=args.num_stages,
        split_k=args.split_k,
        swizzle=args.swizzle,
        group_size=args.group_size,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        skip_ref_check=args.skip_ref_check,
        export_only=args.export_only,
        output_dir=args.output_dir,
        file_name=args.file_name,
        function_prefix=args.function_prefix,
        gpu_arch=args.gpu_arch,
    )


if __name__ == "__main__":
    _parsed_args = _parse_args(_saved_argv)
    main()
    print("PASS")
