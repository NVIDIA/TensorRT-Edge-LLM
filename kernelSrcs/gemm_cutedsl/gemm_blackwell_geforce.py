# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Blackwell GeForce (SM 120/121) FP16 GEMM kernel for Qwen3-Omni Talker MLP.

Computes C = A @ B^T where:
  A is [M, K] row-major FP16
  B is [N, K] row-major FP16 (stored as weight, transposed during GEMM)
  C is [M, N] row-major FP16
  Accumulation in FP32

Uses CuTe DSL with:
  - MmaF16BF16Op (16, 8, 16) — Ampere-class MMA with TMA for memory
  - TMA (Tensor Memory Accelerator) for G2S and S2G copies
  - LdMatrix for S2R copies
  - Warp-specialized pipeline (producer DMA / consumer MMA)
  - Tile shape (128, 128, 64), cluster (1, 1, 1)

Usage:
  # JIT run with reference check
  python gemm_blackwell_geforce.py --mnk 4096,4096,4096

  # AOT export only (produce .o + .h)
  python gemm_blackwell_geforce.py --mnk 4096,4096,4096 \\
      --export_only --output_dir ./gemm_aot --file_name gemm_fp16 \\
      --function_prefix gemm_fp16
"""

import argparse
import os
import sys
import time
from typing import Tuple, Type

import cuda.bindings.driver as cuda
import cupy as cp

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import numpy as np
from cutedsl_utils import aot_placeholders
from common import (
    create_bias_tensor,
    create_fake_row_major_3d_tensor,
    create_row_major_3d_gemm_tensors,
    export_compiled_kernel,
    mark_3d_row_major_dynamic,
    parse_comma_separated_ints,
    to_cute_tensor,
)


class GemmBlackwellGeforceFP16:
    """Blackwell GeForce (SM 120/121) warp-specialized GEMM.

    C[M,N] = A[M,K] @ B[N,K]^T   (FP16 in/out, FP32 accumulation)

    The kernel follows the same structure as the upstream CUTLASS Blackwell
    GeForce dense_gemm example: TMA loads A/B into SMEM, LdMatrix moves
    data from SMEM to registers, MmaF16BF16Op computes, StMatrix + TMA
    stores the result back to GMEM.
    """

    def __init__(
        self,
        acc_dtype,
        tile_shape_mnk: Tuple[int, int, int] = (128, 128, 64),
    ):
        self.acc_dtype = acc_dtype
        self.cluster_shape_mnk = (1, 1, 1)
        self.tile_shape_mnk = tuple(tile_shape_mnk)

        self.tiled_mma = None
        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False

        self.occupancy = 1
        self.atom_layout = (2, 2, 1)
        self.num_mma_warps = (
            self.atom_layout[0] * self.atom_layout[1] * self.atom_layout[2]
        )
        self.num_threads_per_warp = 32
        self.threads_per_cta = (
            self.num_mma_warps + 1  # 1 warp for DMA
        ) * self.num_threads_per_warp
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232

    def _setup_attributes(self):
        self.mma_inst_mnk = (16, 8, 16)
        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.a_dtype,
            self.acc_dtype,
            self.mma_inst_mnk,
        )
        tC = cute.make_layout(self.atom_layout)
        permutation_mnk = (
            self.atom_layout[0] * self.mma_inst_mnk[0],
            self.atom_layout[1] * self.mma_inst_mnk[1] * 2,
            self.atom_layout[2] * self.mma_inst_mnk[2],
        )
        self.tiled_mma = cute.make_tiled_mma(
            op,
            tC,
            permutation_mnk=permutation_mnk,
        )

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        self.num_mcast_ctas_a = self.cluster_shape_mnk[1]
        self.num_mcast_ctas_b = self.cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=False
        )

        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        if self.ab_stage == 0:
            raise RuntimeError(
                f"ab_stage == 0: not enough shared memory for tile {self.tile_shape_mnk}. "
                "Try a smaller tile shape or fewer pipeline stages."
            )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )

    # =========================================================================
    # Host-side __call__: set up TMA descriptors, SMEM layouts, launch kernel
    # =========================================================================
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
        use_silu: cutlass.Constexpr = False,
        mBias: cute.Tensor = None,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(
            self.a_dtype.width == 16 and self.a_dtype != self.b_dtype
        ):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype.width != self.b_dtype.width):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype.width != 16):
            raise TypeError("a_dtype must be float16")
        if cutlass.const_expr(self.b_dtype.width != 16):
            raise TypeError("b_dtype must be float16")

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )

        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        tile_sched_params, grid = self._compute_grid(
            c,
            self.tile_shape_mnk,
            max_active_clusters,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
            use_silu,
            mBias,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )
        return

    # =========================================================================
    # GPU device kernel
    # =========================================================================
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        use_silu: cutlass.Constexpr = False,
        mBias: cute.Tensor = None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        # Multicast masks (no-ops for cluster (1,1,1))
        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )
        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        # Allocate shared storage and pipeline barriers
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * self.num_mma_warps
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        # SMEM tensors
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        # Global tensor partitioning
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        # MMA thread partition
        thr_mma = tiled_mma.get_slice(tidx)

        # TMA partitions for A
        a_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (0, None, 0)).shape
        )
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        # TMA partitions for B
        b_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (None, 0, 0)).shape
        )
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        # MMA fragments
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])

        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # Cluster/CTA sync after barrier init
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            pipeline.sync(barrier_id=1)

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        # Persistent tile scheduler
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        # TMA-store pipeline must be created outside dynamic control flow.
        # CuTe DSL does not allow values first defined inside `if warp_idx < ...`
        # to be referenced later in nested regions.
        tma_store_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_mma_warps * self.num_threads_per_warp,
        )
        tma_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=tma_store_producer_group,
        )

        # =====================================================================
        # MMA warp group
        # =====================================================================
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            num_k_blocks = cute.size(tCrA, mode=[2])

            # LdMatrix copy atoms for S2R
            atom_copy_ldmatrix_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    self.a_layout.is_m_major_a(), 4
                ),
                self.a_dtype,
            )
            atom_copy_ldmatrix_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    self.b_layout.is_n_major_b(), 4
                ),
                self.b_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(
                atom_copy_ldmatrix_A, tiled_mma
            )
            smem_tiled_copy_B = cute.make_tiled_copy_B(
                atom_copy_ldmatrix_B, tiled_mma
            )

            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                accumulators.fill(0.0)

                # =============================================================
                # Pipelined mainloop
                # =============================================================
                mainloop_consumer_state.reset_count()

                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_state, peek_ab_full_status
                )
                tCsA_p = tCsA_copy_view[
                    None, None, None, mainloop_consumer_state.index
                ]
                tCsB_p = tCsB_copy_view[
                    None, None, None, mainloop_consumer_state.index
                ]
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA_p[None, None, 0],
                    tCrA_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_p[None, None, 0],
                    tCrB_copy_view[None, None, 0],
                )

                for k_tile in range(0, k_tile_cnt - 1, 1, unroll=1):
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = (
                            0
                            if k_block_idx + 1 == num_k_blocks
                            else k_block_idx + 1
                        )

                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(
                                mainloop_consumer_state
                            )
                            mainloop_consumer_state.advance()

                            peek_ab_full_status = cutlass.Boolean(1)
                            peek_ab_full_status = (
                                mainloop_pipeline.consumer_try_wait(
                                    mainloop_consumer_state
                                )
                            )

                            tCsA_p = tCsA_copy_view[
                                None,
                                None,
                                None,
                                mainloop_consumer_state.index,
                            ]
                            tCsB_p = tCsB_copy_view[
                                None,
                                None,
                                None,
                                mainloop_consumer_state.index,
                            ]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )

                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[None, None, k_block_idx],
                            tCrB[None, None, k_block_idx],
                            accumulators,
                        )

                # Last k_tile (hoisted out of loop)
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = (
                        0
                        if k_block_idx + 1 == num_k_blocks
                        else k_block_idx + 1
                    )

                    if k_block_idx == num_k_blocks - 1:
                        mainloop_pipeline.consumer_release(
                            mainloop_consumer_state
                        )
                        mainloop_consumer_state.advance()

                    if k_block_next > 0:
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        tCrA[None, None, k_block_idx],
                        tCrB[None, None, k_block_idx],
                        accumulators,
                    )

                # =============================================================
                # Fused epilogue operators applied directly to the MMA
                # accumulators (before retile/R2S), so the element-wise index
                # iteration is aligned with the MMA's thread partitioning.
                # Bias uses broadcast stride (0, 1) so mBias[n] is correctly
                # selected for every (m, n) element the thread owns.
                # =============================================================
                if cutlass.const_expr(mBias is not None):
                    tile_n_offset = (
                        tile_coord_mnl[1] * cutlass.Int32(self.tile_shape_mnk[1])
                    )
                    gBias_tile = cute.make_tensor(
                        mBias.iterator + tile_n_offset,
                        cute.make_layout(
                            (self.tile_shape_mnk[0], self.tile_shape_mnk[1]),
                            stride=(0, 1),
                        ),
                    )
                    tCgBias = thr_mma.partition_C(gBias_tile)
                    for i in cutlass.range(cute.size(accumulators)):
                        accumulators[i] = (
                            accumulators[i] + tCgBias[i].to(self.acc_dtype)
                        )

                if cutlass.const_expr(use_silu):
                    for i in cutlass.range(cute.size(accumulators)):
                        val = accumulators[i]
                        accumulators[i] = val * cute.arch.rcp_approx(
                            1.0 + cute.exp(-val, fastmath=True)
                        )

                # =============================================================
                # Epilogue: R2S via StMatrix, then S2G via TMA
                # =============================================================
                copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                    self.c_layout,
                    elem_ty_d=self.c_dtype,
                    elem_ty_acc=self.acc_dtype,
                )

                copy_atom_C = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(
                        self.c_layout.is_m_major_c(),
                        4,
                    ),
                    self.c_dtype,
                )

                tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(
                    copy_atom_C, tiled_mma
                )

                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s,
                    tiled_copy_C_Atom,
                )

                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                tRS_rAcc = tiled_copy_r2s.retile(accumulators)

                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(
                    tRS_rD_layout.shape, self.acc_dtype
                )
                size_tRS_rD = cute.size(tRS_rD)

                sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
                tcgc_for_tma_partition = cute.zipped_divide(
                    gC_mnl_slice, self.epi_tile
                )

                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    sepi_for_tma_partition,
                    tcgc_for_tma_partition,
                )

                epi_tile_num = cute.size(tcgc_for_tma_partition, mode=[1])
                epi_tile_shape = tcgc_for_tma_partition.shape[1]
                epi_tile_layout = cute.make_layout(
                    epi_tile_shape, stride=(1, epi_tile_shape[0])
                )

                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[
                            epi_idx * size_tRS_rD + epi_v
                        ]

                    tRS_rD_out = cute.make_rmem_tensor(
                        tRS_rD_layout.shape, self.c_dtype
                    )
                    tRS_rD_out.store(tRS_rD.load().to(self.c_dtype))

                    epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rD_out,
                        tRS_sD[(None, None, None, epi_buffer)],
                    )

                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    self.epilog_sync_barrier.arrive_and_wait()

                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    if warp_idx == 0:
                        cute.copy(
                            tma_atom_c,
                            bSG_sD[(None, epi_buffer)],
                            bSG_gD[(None, gmem_coord)],
                        )
                    tma_store_pipeline.producer_commit()
                    tma_store_pipeline.producer_acquire()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            tma_store_pipeline.producer_tail()

        # =====================================================================
        # DMA warp group
        # =====================================================================
        elif warp_idx == self.num_mma_warps:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[
                    (None, tile_coord_mnl[0], None, tile_coord_mnl[2])
                ]
                tBgB_nkl = tBgB[
                    (None, tile_coord_mnl[1], None, tile_coord_mnl[2])
                ]

                mainloop_producer_state.reset_count()

                for k_tile in range(0, k_tile_cnt, 1, unroll=1):
                    mainloop_pipeline.producer_acquire(
                        mainloop_producer_state
                    )

                    tAgA_k = tAgA_mkl[
                        (None, mainloop_producer_state.count)
                    ]
                    tAsA_pipe = tAsA[
                        (None, mainloop_producer_state.index)
                    ]

                    tBgB_k = tBgB_nkl[
                        (None, mainloop_producer_state.count)
                    ]
                    tBsB_pipe = tBsB[
                        (None, mainloop_producer_state.index)
                    ]

                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                        mcast_mask=b_mcast_mask,
                    )
                    mainloop_pipeline.producer_commit(
                        mainloop_producer_state
                    )
                    mainloop_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mainloop_pipeline.producer_tail(mainloop_producer_state)
        return

    # =========================================================================
    # Static helpers
    # =========================================================================
    @staticmethod
    def _compute_stages(
        tile_shape_mnk,
        a_dtype,
        b_dtype,
        epi_tile,
        c_dtype,
        smem_capacity,
        occupancy,
    ):
        epi_stage = 8
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy
            - mbar_helpers_bytes
            - epi_bytes
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk,
        epi_tile,
        a_dtype,
        a_layout,
        b_dtype,
        b_layout,
        ab_stage,
        c_dtype,
        c_layout,
        epi_stage,
    ):
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout, tile_shape_mnk, a_dtype, ab_stage
        )
        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout, tile_shape_mnk, b_dtype, ab_stage
        )
        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            c_dtype, c_layout, epi_tile, epi_stage
        )
        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    @staticmethod
    def _compute_grid(c, tile_shape_mnk, max_active_clusters):
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(tensor_c, epi_smem_layout_staged, epi_tile):
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )
        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(tensor, smem_layout_staged, smem_tile, mcast_dim):
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor


# =============================================================================
# Epilogue helpers
# =============================================================================



# =============================================================================
# Host run function (CuPy-only, no PyTorch)
# =============================================================================
def run(
    mnk: Tuple[int, int, int],
    tile_shape_mnk: Tuple[int, int, int],
    tolerance: float,
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    export_only: bool = False,
    output_dir: str = "./gemm_aot_artifacts",
    file_name: str = "gemm_fp16",
    function_prefix: str = "gemm_fp16",
    fused_epilogue: str = "none",
):
    m, n, k = mnk
    batch = 1

    # ------------------------------------------------------------------
    # Epilogue selection
    # ------------------------------------------------------------------
    _valid_epilogues = ("none", "silu", "bias", "bias_silu")
    if fused_epilogue not in _valid_epilogues:
        raise ValueError(
            f"fused_epilogue must be one of {_valid_epilogues}, got '{fused_epilogue}'"
        )

    # For BW GeForce we fuse the activation into the GEMM epilogue.
    # Bias addition is NOT fused into the CUDA kernel here — the C++
    # caller is responsible for the bias add when mBias is None.
    if fused_epilogue in ("silu", "bias_silu"):
        _use_silu = True
    else:
        _use_silu = False

    _tag = f"[{file_name}]"
    print(f"{_tag} Blackwell GeForce FP16 GEMM: M={m}, N={n}, K={k}")
    print(f"{_tag} Tile shape: {tile_shape_mnk}")
    print(f"{_tag} export_only={export_only}")
    print(f"{_tag} fused_epilogue={fused_epilogue}")

    if not export_only and cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU is required to run this example!")

    if export_only:
        a_tensor = create_fake_row_major_3d_tensor(batch=batch)
        b_tensor = create_fake_row_major_3d_tensor(batch=batch)
        c_tensor = create_fake_row_major_3d_tensor(batch=batch)
        a_cp = b_cp = c_cp = None
    else:
        cp.random.seed(42)
        a_cp, b_cp, c_cp = create_row_major_3d_gemm_tensors(
            m, n, k, batch=batch, fill_random=True, dtype=cp.float16
        )
        a_tensor = mark_3d_row_major_dynamic(to_cute_tensor(a_cp))
        b_tensor = mark_3d_row_major_dynamic(to_cute_tensor(b_cp))
        c_tensor = mark_3d_row_major_dynamic(to_cute_tensor(c_cp))

    # Bias tensor for fused epilogues.
    mBias = None
    bias_cp = None
    if fused_epilogue in ("bias", "bias_silu"):
        mBias, bias_cp = create_bias_tensor(n, export_only=export_only)

    gemm = GemmBlackwellGeforceFP16(
        acc_dtype=cutlass.Float32,
        tile_shape_mnk=tile_shape_mnk,
    )

    # max_active_clusters is a RUNTIME kernel argument (persistent grid
    # sizing): the deployed caller passes the target GPU's SM count at launch
    # time, so the AOT artifact is correct on every SKU instead of baking the
    # build GPU's occupancy. Export mode traces with a placeholder; run/verify
    # mode probes the local GPU as before.
    if export_only:
        max_active_clusters = aot_placeholders.runtime_int32()
    else:
        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = cutlass.Int32(hardware_info.get_max_active_clusters(1))

    current_stream = (
        aot_placeholders.make_stream()
        if export_only
        else cuda.CUstream(cp.cuda.get_current_stream().ptr)
    )

    start_time = time.time()
    compiled_gemm = cute.compile(
        gemm,
        a_tensor,
        b_tensor,
        c_tensor,
        max_active_clusters,
        current_stream,
        use_silu=_use_silu,
        mBias=mBias,
        **(
            dict(options=aot_placeholders.compile_options())
            if export_only
            else {}
        ),
    )
    compilation_time = time.time() - start_time
    print(f"{_tag} Compilation time: {compilation_time:.4f}s")

    if export_only:
        export_compiled_kernel(
            compiled_gemm,
            output_dir=output_dir,
            file_name=file_name,
            function_prefix=function_prefix,
            tag=_tag,
        )
        return None

    # Run the kernel
    compiled_gemm(a_tensor, b_tensor, c_tensor, max_active_clusters, current_stream)
    cp.cuda.Device().synchronize()

    if not skip_ref_check:
        print(f"{_tag} Reference checking ...")
        # C = A @ B^T  =>  for batch L:  C[:,:,l] = A[:,:,l] @ B[:,:,l].T
        a_np = cp.asnumpy(a_cp[:, :, 0]).astype(np.float32)
        b_np = cp.asnumpy(b_cp[:, :, 0]).astype(np.float32)
        ref_f32 = a_np @ b_np.T

        if fused_epilogue in ("bias", "bias_silu"):
            bias_f32 = cp.asnumpy(bias_cp).astype(np.float32)
            ref_f32 = ref_f32 + bias_f32
        if fused_epilogue == "bias_silu":
            # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
            ref_f32 = ref_f32 * (1.0 / (1.0 + np.exp(-ref_f32)))

        ref_np = ref_f32.astype(np.float16)
        result_np = cp.asnumpy(c_cp[:, :, 0])

        max_abs_err = np.max(np.abs(result_np.astype(np.float32) - ref_np.astype(np.float32)))
        print(f"{_tag} Max absolute error: {max_abs_err:.6f} (tolerance={tolerance})")
        if max_abs_err > tolerance:
            raise AssertionError(
                f"GEMM result mismatch: max_abs_err={max_abs_err} > tolerance={tolerance}"
            )
        print(f"{_tag} Reference check PASSED")

    # Benchmark
    if iterations > 0:
        for _ in range(warmup_iterations):
            compiled_gemm(a_tensor, b_tensor, c_tensor, max_active_clusters, current_stream)
        cp.cuda.Device().synchronize()

        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        start_event.record()
        for _ in range(iterations):
            compiled_gemm(a_tensor, b_tensor, c_tensor, max_active_clusters, current_stream)
        end_event.record()
        end_event.synchronize()
        elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
        avg_us = (elapsed_ms / iterations) * 1000.0
        print(f"{_tag} Avg latency: {avg_us:.2f} us over {iterations} iterations")

        flops = 2.0 * m * n * k
        tflops = (flops / (avg_us * 1e-6)) / 1e12
        print(f"{_tag} Throughput: {tflops:.2f} TFLOPS")

        return avg_us

    return None


# =============================================================================
# CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Blackwell GeForce (SM 120/121) FP16 GEMM — CuTe DSL kernel"
    )

    parser.add_argument(
        "--mnk",
        type=parse_comma_separated_ints,
        default=(4096, 4096, 4096),
        help="M,N,K dimensions (comma-separated)",
    )
    parser.add_argument(
        "--tile_shape_mnk",
        type=parse_comma_separated_ints,
        default=(128, 128, 64),
        help="CTA tile shape M,N,K (comma-separated)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-01,
        help="Tolerance for reference validation",
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=0,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--skip_ref_check",
        action="store_true",
        default=False,
        help="Skip reference checking against NumPy",
    )
    parser.add_argument(
        "--export_only",
        action="store_true",
        help="Compile and export .o + .h via export_to_c; skip execution",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gemm_aot_artifacts",
        help="Output directory for AOT compiled artifacts",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="gemm_fp16",
        help="Base file name for exported artifacts (e.g., gemm_fp16 -> gemm_fp16.h, gemm_fp16.o)",
    )
    parser.add_argument(
        "--function_prefix",
        type=str,
        default="gemm_fp16",
        help="Function prefix for exported C symbols",
    )
    parser.add_argument(
        "--fused_epilogue",
        type=str,
        default="none",
        choices=["none", "silu", "bias", "bias_silu"],
        help=(
            "Fused epilogue type. "
            "'none': plain GEMM (identity epilogue). "
            "'silu': fuse SiLU activation into the GEMM epilogue (saves one kernel launch). "
            "'bias': plain GEMM; bias is applied by the C++ caller. "
            "'bias_silu': fuse SiLU into the GEMM epilogue; bias applied by C++ caller."
        ),
    )

    args = parser.parse_args()

    if len(args.mnk) != 3:
        parser.error("--mnk must contain exactly 3 values (M,N,K)")

    latency = run(
        mnk=args.mnk,
        tile_shape_mnk=args.tile_shape_mnk,
        tolerance=args.tolerance,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        skip_ref_check=args.skip_ref_check,
        export_only=args.export_only,
        output_dir=args.output_dir,
        file_name=args.file_name,
        function_prefix=args.function_prefix,
        fused_epilogue=args.fused_epilogue,
    )

    if latency is not None:
        print("PASS")
