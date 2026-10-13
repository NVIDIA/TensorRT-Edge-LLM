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

"""CuTe DSL Blackwell GEMM kernel: C = A @ B^T, FP16 in / FP16 out / FP32 accumulation.

Replaces cuBLAS cublasGemmEx for the Qwen3-Omni Talker MLP linear layers.

  A: [M, K]  row-major  (input activation)
  B: [N, K]  row-major  (weight, transposed for B^T)
  C: [M, N]  row-major  (output)

Uses tcgen05.mma (UMMA) instructions on SM 100/101/103/110 (Blackwell datacenter).
Does NOT support SM 120/121 (Blackwell Ultra) which lacks tcgen05.

Usage:
  # Test on GPU:
  python gemm_blackwell.py --mnk 128,3584,896

  # AOT export (.o + .h):
  python gemm_blackwell.py --mnk 128,3584,896 --export_only \\
      --output_dir ./gemm_artifacts --file_name gemm_fp16 --function_prefix gemm_fp16
"""

import argparse
import os
import sys
import time
from typing import Optional, Type, Tuple, Union

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
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import numpy as np
from cutedsl_utils import aot_placeholders
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from common import (
    create_bias_tensor,
    create_fake_row_major_3d_tensor,
    create_row_major_3d_gemm_tensors,
    create_row_major_3d_tensor,
    export_compiled_kernel,
    mark_3d_row_major_dynamic,
    parse_comma_separated_ints,
    to_cute_tensor,
)


class GemmBlackwellFP16:
    """Blackwell SM100 dense GEMM: C[M,N] = A[M,K] @ B[N,K]^T.

    FP16 inputs/outputs with FP32 accumulation via tcgen05.mma (UMMA).
    TMA store epilogue (cp.bulk.tensor.s2g) with auto box-bounds.

    Configurable per construction:
      mma_tiler_mn:     per-CTA tile (default (64, 128); use (128, 128) for
                        large M >= 1024)
      cluster_shape_mn: cluster size for TMA multicast (default (1, 2)
                        multicasts B across two CTAs in N)
      use_2cta:         pair two CTAs along M for a 2x M tile per UMMA op
                        (requires cluster[0] >= 2; not used by current AOT)
      mma_inst_tile_k:  K-tile multiplier per MMA instruction (default 4)

    Designed for the Qwen3-Omni Talker MLP where M is the token count
    (dynamic) and N, K are weight dimensions (also dynamic for AOT).
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        mma_tiler_mn: Tuple[int, int] = (64, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 2),
        use_2cta: bool = False,
        mma_inst_tile_k: int = 4,
    ):
        self.acc_dtype = acc_dtype
        # 2-CTA MMA pairs two CTAs along the M dim and issues a single
        # tcgen05.mma instruction that covers a 2x larger M tile per cta_group.
        # Requires cluster_shape[0] >= 2 so the leader CTA can drive the peer.
        self.use_2cta_instrs = use_2cta
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        # K-tile multiplier per MMA instruction K-shape. Larger -> more K
        # accumulated per MMA op (longer K iter span, fewer pipeline syncs).
        self.mma_inst_tile_k = mma_inst_tile_k
        # TMA store: TMEM -> regs -> SMEM -> gmem via cp.bulk.tensor.s2g.
        # The TMA descriptor handles partial-tile boundaries automatically,
        # so no manual M/N predication is needed in the epilogue.
        self.use_tma_store = True

        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta else tcgen05.CtaGroup.ONE
        self.occupancy = 1
        self.threads_per_cta = 128

    def _setup_attributes(self):
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * self.mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        if cutlass.const_expr(self.use_tma_store):
            self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
                self.cta_tile_shape_mnk,
                self.use_2cta_instrs,
                self.c_layout,
                self.c_dtype,
            )
        else:
            self.epi_tile = self.cta_tile_shape_mnk[:2]

        # cuTe DSL 4.5.2 does not reliably auto-detect Thor's Blackwell family
        # SMs (e.g. SM101/SM110) in get_smem_capacity_in_bytes(). Like the FMHA
        # Blackwell path, request the Blackwell-family capacity explicitly.
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.smem_capacity,
            self.occupancy,
            self.use_tma_store,
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage,
        )
        self.c_smem_layout_staged = (
            sm100_utils.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage,
            )
            if self.use_tma_store else None
        )

        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: cuda.CUstream,
        use_silu: cutlass.Constexpr = False,
        mBias: cute.Tensor = None,
    ):
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"A/B dtype mismatch: {self.a_dtype} vs {self.b_dtype}")

        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # TMA load A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # TMA store atom for C
        tma_atom_c = None
        tma_tensor_c = None
        if cutlass.const_expr(self.use_tma_store):
            epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                c,
                epi_smem_layout,
                self.epi_tile,
            )

        grid = self._compute_grid(c, self.cta_tile_shape_mnk, self.cluster_shape_mn)

        self.kernel(
            tiled_mma,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_c, tma_tensor_c if self.use_tma_store else c,
            mBias,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            use_silu,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        mBias: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        use_silu: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(self.use_tma_store):
                cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        cta_coord = (bidx, bidy, bidz)
        mma_tile_coord_mnl = (
            cta_coord[0] // cute.size(tiled_mma.thr_id.shape),
            cta_coord[1],
            cta_coord[2],
        )
        tidx, _, _ = cute.arch.thread_idx()

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.threads_per_cta
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_acc_stage
        )
        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_acc_stage
        )

        # Barrier id 0 is reserved by other driver APIs (e.g. sync_threads()).
        # Use a non-zero named barrier to avoid runtime warnings and potential
        # synchronization conflicts.
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3, num_threads=self.threads_per_cta
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # SMEM tensors. sC is only allocated for TMA store (R2S staging).
        sC = None
        if cutlass.const_expr(self.use_tma_store):
            sC = smem.allocate_tensor(
                element_type=self.c_dtype,
                layout=c_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=c_smem_layout_staged.inner,
            )
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        # Identity tensor mirroring tCgC so the epilogue can derive per-element
        # absolute (M, N) coordinates that physically align with tCgC.
        mcC_id = cute.make_identity_tensor(mC_mnl.layout.shape)
        gC_id_mnl = cute.local_tile(
            mcC_id, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        tCcC = thr_mma.partition_C(gC_id_mnl)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        tmem.allocate(self.num_tmem_alloc_cols)
        tmem.wait_for_alloc()

        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

        # ---- Mainloop: pipelined TMA load + tcgen05 MMA ----
        prefetch_k_tile_cnt = cutlass.min(self.num_ab_stage - 2, k_tile_cnt)
        if warp_idx == 0:
            for k_tile_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):
                producer_handle = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, k_tile_idx)],
                    tAsA[(None, producer_handle.index)],
                    tma_bar_ptr=producer_handle.barrier,
                    mcast_mask=a_full_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, k_tile_idx)],
                    tBsB[(None, producer_handle.index)],
                    tma_bar_ptr=producer_handle.barrier,
                    mcast_mask=b_full_mcast_mask,
                )

            peek_ab_full_status = cutlass.Boolean(False)
            if is_leader_cta:
                peek_ab_full_status = ab_consumer.try_wait()

            peek_ab_empty_status = ab_producer.try_acquire()

            for k_tile_idx in cutlass.range(k_tile_cnt):
                if k_tile_idx < k_tile_cnt - prefetch_k_tile_cnt:
                    producer_handle = ab_producer.acquire_and_advance(
                        peek_ab_empty_status
                    )
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, producer_handle.count)],
                        tAsA[(None, producer_handle.index)],
                        tma_bar_ptr=producer_handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, producer_handle.count)],
                        tBsB[(None, producer_handle.index)],
                        tma_bar_ptr=producer_handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )

                if is_leader_cta:
                    consumer_handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                    num_kblks = cute.size(tCrA, mode=[2])
                    for kblk_idx in cutlass.range(num_kblks, unroll_full=True):
                        kblk_crd = (None, None, kblk_idx, consumer_handle.index)
                        cute.gemm(
                            tiled_mma, tCtAcc, tCrA[kblk_crd], tCrB[kblk_crd], tCtAcc
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    consumer_handle.release()

                if k_tile_idx + 1 < k_tile_cnt - prefetch_k_tile_cnt:
                    peek_ab_empty_status = ab_producer.try_acquire()

                if k_tile_idx + 1 < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

            # Commit accumulator buffer full ONCE after all k-tiles are MMA'd.
            # Committing inside the loop would let the consumer wake on a
            # partial K accumulation and read garbage from TMEM.
            if is_leader_cta:
                acc_pipeline.producer_commit(acc_producer_state)

        # ---- Epilogue: TMEM -> registers (-> SMEM -> gmem via TMA) ----
        tmem.relinquish_alloc_permit()
        acc_pipeline.consumer_wait(acc_consumer_state)

        if cutlass.const_expr(self.use_tma_store):
            self.epilogue_tma_store(
                tidx, warp_idx, mma_tile_coord_mnl, tma_atom_c,
                tCtAcc, sC, tCgC, tCcC, mC_mnl, epi_tile, mBias, use_silu,
            )
        else:
            self.epilogue(
                tidx, mma_tile_coord_mnl, tCtAcc, tCgC, tCcC,
                mC_mnl, epi_tile, mBias, use_silu,
            )

        pipeline.sync(barrier_id=1)
        tmem.free(tmem_ptr)

        if warp_idx == 0:
            ab_producer.tail()
        return

    @cute.jit
    def epilogue_tma_store(
        self,
        epi_tidx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        mma_tile_coord_mnl: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        tma_atom_c: cute.CopyAtom,
        tCtAcc: cute.Tensor,
        sC: cute.Tensor,
        tCgC: cute.Tensor,
        tCcC: cute.Tensor,
        mC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        mBias: cute.Tensor,
        use_silu: cutlass.Constexpr,
    ) -> None:
        """TMA-store epilogue: TMEM -> regs -> SMEM -> gmem via cp.bulk.tensor.

        Adapted from cutlass/examples/python/CuTeDSL/blackwell/dense_gemm.py
        with the bias-add and SiLU activation hooks added in the register
        stage (between T2R and R2S).
        """
        # T2R: TMEM -> registers (per-thread fragment).
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk, self.c_layout, self.c_dtype, self.acc_dtype,
            epi_tile, self.use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

        # tCgC partition for absolute coords (used for bias N indexing).
        tCgC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tTR_gC = thr_copy_t2r.partition_D(tCgC_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)

        # Mirror tCgC's partition path on the identity coord tensor so we can
        # gate bias loads by the absolute N coord.
        tCcC_epi = cute.flat_divide(
            tCcC[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tTR_cC = thr_copy_t2r.partition_D(tCcC_epi)
        tTR_cC = tTR_cC[(None, None, None, None, None, *mma_tile_coord_mnl)]
        tTR_cC = cute.group_modes(tTR_cC, 3, cute.rank(tTR_cC))

        actual_N = mC_mnl.layout.shape[1]

        # Bias broadcast tile (gmem). Same construction as the simt path.
        if cutlass.const_expr(mBias is not None):
            n_tile_offset = mma_tile_coord_mnl[1] * cutlass.Int32(self.mma_tiler[1])
            gBias_tile = cute.make_tensor(
                mBias.iterator + n_tile_offset,
                cute.make_layout(
                    (self.mma_tiler[0], self.mma_tiler[1]), stride=(0, 1)
                ),
            )
            tCgBias_epi = cute.flat_divide(gBias_tile, epi_tile)
            tTR_gBias = thr_copy_t2r.partition_D(tCgBias_epi)
            tTR_gBias = cute.group_modes(tTR_gBias, 3, cute.rank(tTR_gBias))

        # R2S: registers -> SMEM (staging buffer for TMA store).
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(epi_tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)

        # TMA store partition: SMEM -> gmem. The TMA descriptor itself
        # handles partial-tile boundaries — no manual M/N predication needed.
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(tCgC_epi, 0, 2),
        )
        bSG_gC = bSG_gC[(None, None, None, *mma_tile_coord_mnl)]
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

        c_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.threads_per_cta
        )
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.num_c_stage, producer_group=c_producer_group
        )

        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        for subtile_idx in cutlass.range(subtile_cnt):
            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

            coord_slice = tTR_cC[(None, None, None, subtile_idx)]

            # Bias add (broadcast over M, indexed by N), gated by actual_N.
            if cutlass.const_expr(mBias is not None):
                bias_slice = tTR_gBias[(None, None, None, subtile_idx)]
                for i in cutlass.range(cute.size(tTR_rAcc)):
                    if coord_slice[i][1] < actual_N:
                        tTR_rAcc[i] = tTR_rAcc[i] + bias_slice[i].to(self.acc_dtype)
            if cutlass.const_expr(use_silu):
                for si in cutlass.range(cute.size(tTR_rAcc)):
                    val = tTR_rAcc[si]
                    tTR_rAcc[si] = val * cute.arch.rcp_approx(
                        1.0 + cute.exp(-val, fastmath=True)
                    )

            # Convert acc -> c_dtype in registers.
            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
            tRS_rC.store(acc_vec.to(self.c_dtype))

            # R2S into the c-stage SMEM buffer.
            c_buffer = subtile_idx % self.num_c_stage
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
            cute.arch.fence_proxy("async.shared", space="cta")
            pipeline.sync(barrier_id=1)

            # TMA S2G — only warp 0 issues the store.
            if warp_idx == 0:
                cute.copy(
                    tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, subtile_idx)]
                )
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            pipeline.sync(barrier_id=1)

        c_pipeline.producer_tail()

    @cute.jit
    def epilogue(
        self,
        epi_tidx: cutlass.Int32,
        mma_tile_coord_mnl: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        tCtAcc: cute.Tensor,
        tCgC: cute.Tensor,
        tCcC: cute.Tensor,
        mC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        mBias: cute.Tensor,
        use_silu: cutlass.Constexpr,
    ) -> None:
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            self.use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

        tCgC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tTR_gC = thr_copy_t2r.partition_D(tCgC_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        tTR_rC = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.c_dtype
        )
        simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)

        tTR_gC = tTR_gC[(None, None, None, None, None, *mma_tile_coord_mnl)]
        tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))

        # Bias tile: build from full mBias via local_tile so iterator offsets
        # are derived by cute, not by manual pointer arithmetic. mBias is 1D
        # (N,); we replicate it across M with a stride-(0,1) layout.
        if cutlass.const_expr(mBias is not None):
            n_tile_offset = mma_tile_coord_mnl[1] * cutlass.Int32(self.mma_tiler[1])
            gBias_tile = cute.make_tensor(
                mBias.iterator + n_tile_offset,
                cute.make_layout(
                    (self.mma_tiler[0], self.mma_tiler[1]), stride=(0, 1)
                ),
            )
            tCgBias_epi = cute.flat_divide(gBias_tile, epi_tile)
            tTR_gBias = thr_copy_t2r.partition_D(tCgBias_epi)
            tTR_gBias = cute.group_modes(tTR_gBias, 3, cute.rank(tTR_gBias))

        # Boundary predication coords. tCcC mirrors tCgC's partition path, so
        # after the same flat_divide / partition_D / index / group_modes chain,
        # tTR_cC[i] holds the absolute (M, N) coord of the element tTR_gC[i]
        # writes to. (M, N) values past mC_mnl's actual shape are out-of-bounds.
        actual_M = mC_mnl.layout.shape[0]
        actual_N = mC_mnl.layout.shape[1]

        tCcC_epi = cute.flat_divide(
            tCcC[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tTR_cC = thr_copy_t2r.partition_D(tCcC_epi)
        tTR_cC = tTR_cC[(None, None, None, None, None, *mma_tile_coord_mnl)]
        tTR_cC = cute.group_modes(tTR_cC, 3, cute.rank(tTR_cC))

        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        for subtile_idx in cutlass.range(subtile_cnt):
            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

            coord_slice = tTR_cC[(None, None, None, subtile_idx)]

            # Bias add (broadcast over M, indexed by N). Predicated by the N
            # coord so we never load past actual_N when the bias tile spans an
            # N partial-tile boundary.
            if cutlass.const_expr(mBias is not None):
                bias_slice = tTR_gBias[(None, None, None, subtile_idx)]
                for i in cutlass.range(cute.size(tTR_rAcc)):
                    if coord_slice[i][1] < actual_N:
                        tTR_rAcc[i] = tTR_rAcc[i] + bias_slice[i].to(self.acc_dtype)
            if cutlass.const_expr(use_silu):
                for si in cutlass.range(cute.size(tTR_rAcc)):
                    val = tTR_rAcc[si]
                    tTR_rAcc[si] = val * cute.arch.rcp_approx(1.0 + cute.exp(-val, fastmath=True))

            tTR_rC.store(tTR_rAcc.load().to(self.c_dtype))

            # Predicated store: skip elements past the actual M/N boundary so
            # partial tiles do not write past the output buffer. Used by the
            # use_tma_store=False fallback only — the TMA store path's
            # descriptor enforces box-bounds for free.
            dst = tTR_gC[(None, None, None, subtile_idx)]
            for i in cutlass.range(cute.size(tTR_rC)):
                if (coord_slice[i][0] < actual_M
                        and coord_slice[i][1] < actual_N):
                    dst[i] = tTR_rC[i]

    @staticmethod
    def _compute_stages(
        tiled_mma, mma_tiler_mnk, a_dtype, b_dtype,
        epi_tile, c_dtype, c_layout, smem_capacity, occupancy, use_tma_store,
    ) -> Tuple[int, int, int]:
        num_acc_stage = 1
        num_c_stage = 2 if use_tma_store else 0

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, a_dtype, 1,
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, b_dtype, 1,
        )
        c_smem_layout_staged_one = (
            sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)
            if use_tma_store else None
        )
        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = (
            cute.size_in_bytes(c_dtype, c_smem_layout_staged_one) if use_tma_store else 0
        )
        c_bytes = c_bytes_per_stage * num_c_stage

        num_ab_stage = (
            smem_capacity - (occupancy + 1) * (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        if use_tma_store:
            num_c_stage += (
                smem_capacity
                - ab_bytes_per_stage * num_ab_stage
                - (occupancy + 1) * (mbar_helpers_bytes + c_bytes)
            ) // ((occupancy + 1) * c_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_c_stage

    @staticmethod
    def _compute_grid(c, cta_tile_shape_mnk, cluster_shape_mn):
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        grid = cute.round_up(
            (
                cute.ceil_div(c.layout.shape[0], cta_tile_shape_mnk[0]),
                cute.ceil_div(c.layout.shape[1], cta_tile_shape_mnk[1]),
                c.layout.shape[2],
            ),
            cluster_shape_mnl,
        )
        return grid

    @staticmethod
    def _compute_num_tmem_alloc_cols(tiled_mma, mma_tiler):
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        return utils.get_num_tmem_alloc_cols(tCtAcc_fake)


# ---------------------------------------------------------------------------
# Epilogue helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Run / Export
# ---------------------------------------------------------------------------

def run(
    mnk: Tuple[int, int, int],
    export_only: bool = False,
    output_dir: str = "./gemm_aot_artifacts",
    file_name: str = "gemm_fp16",
    function_prefix: str = "gemm_fp16",
    skip_ref_check: bool = False,
    tolerance: float = 0.05,
    warmup_iterations: int = 3,
    iterations: int = 10,
    fused_epilogue: str = "none",
    mma_tiler_mn: Tuple[int, int] = (64, 128),
    cluster_shape_mn: Tuple[int, int] = (1, 2),
    use_2cta: bool = False,
    c_dtype: str = "float16",
):
    m, n, k = mnk
    _tag = f"[{file_name}]"

    if fused_epilogue not in ("none", "bias", "bias_silu"):
        raise ValueError(f"Unknown fused_epilogue={fused_epilogue!r}")
    if c_dtype not in ("float16", "float32"):
        raise ValueError(f"Unknown c_dtype={c_dtype!r} (expected float16/float32)")
    c_cp_dtype = cp.float16 if c_dtype == "float16" else cp.float32

    epilogue_str = f" +{fused_epilogue}" if fused_epilogue != "none" else ""

    if export_only:
        print(f"{_tag} AOT compile: M={m}, N={n}, K={k}{epilogue_str} "
              f"tile={mma_tiler_mn} cluster={cluster_shape_mn} c_dtype={c_dtype}")
    else:
        print(f"{_tag} Running Blackwell GEMM (C = A @ B^T) test:")
        print(f"{_tag}   M={m}, N={n}, K={k}{epilogue_str}")
        print(f"{_tag}   FP16 in, {c_dtype} out, FP32 accumulation")
        print(f"{_tag}   mma_tiler={mma_tiler_mn}, cluster={cluster_shape_mn}, TMA store")

    if not export_only and cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU is required!")

    if not export_only:
        cp.random.seed(1111)
    np.random.seed(1111)

    l = 1  # batch dim required by DenseGemmKernel tensor layout

    # A: (M, K, L) row-major  => a_major="k"
    # B: (N, K, L) row-major  => b_major="k"  (kernel computes C = A @ B^T)
    # C: (M, N, L) row-major  => c_major="n"
    # A and B are FP16 tensor-core operands; C is allocated separately because
    # its dtype is configurable (float16 or float32). C's row-major
    # (L, mode0, mode1) layout is the same for either dtype, so the dynamic
    # layout markup below applies unchanged.
    if export_only:
        c_type = cutlass.Float16 if c_dtype == "float16" else cutlass.Float32
        a_dyn = create_fake_row_major_3d_tensor(batch=l)
        b_dyn = create_fake_row_major_3d_tensor(batch=l)
        c_dyn = create_fake_row_major_3d_tensor(batch=l, dtype=c_type)
        a_cp = b_cp = c_cp = None
    else:
        a_cp, b_cp, _ = create_row_major_3d_gemm_tensors(
            m, n, k, batch=l, fill_random=True, dtype=cp.float16
        )
        c_cp = create_row_major_3d_tensor(
            m, n, l, fill_random=False, dtype=c_cp_dtype
        )
        a_dyn = mark_3d_row_major_dynamic(to_cute_tensor(a_cp))
        b_dyn = mark_3d_row_major_dynamic(to_cute_tensor(b_cp))
        c_dyn = mark_3d_row_major_dynamic(to_cute_tensor(c_cp))

    # Bias tensor for fused epilogues.
    mBias = None
    bias_cp = None
    if fused_epilogue in ("bias", "bias_silu"):
        mBias, bias_cp = create_bias_tensor(n, export_only=export_only)

    use_silu = (fused_epilogue == "bias_silu")

    current_stream = (
        aot_placeholders.make_stream()
        if export_only
        else cuda.CUstream(cp.cuda.get_current_stream().ptr)
    )

    gemm = GemmBlackwellFP16(
        acc_dtype=cutlass.Float32,
        mma_tiler_mn=tuple(mma_tiler_mn),
        cluster_shape_mn=tuple(cluster_shape_mn),
        use_2cta=use_2cta,
    )

    start_time = time.time()
    compiled_gemm = cute.compile(
        gemm, a_dyn, b_dyn, c_dyn, current_stream,
        use_silu=use_silu,
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

    # ---- Functional test ----
    if not skip_ref_check:
        compiled_gemm(a_dyn, b_dyn, c_dyn, current_stream)
        cp.cuda.get_current_stream().synchronize()

        # Reference: C_ref = A @ B^T, squeezing the L=1 batch dim. Cast the
        # reference to the C output tensor's dtype so both sides are compared at
        # the same precision.
        _np_c = c_cp.dtype
        a_f32 = cp.asnumpy(a_cp[:, :, 0]).astype(np.float32)
        b_f32 = cp.asnumpy(b_cp[:, :, 0]).astype(np.float32)
        c_ref = (a_f32 @ b_f32.T).astype(_np_c)

        if fused_epilogue in ("bias", "bias_silu"):
            bias_f32 = cp.asnumpy(bias_cp).astype(np.float32)
            c_ref = (c_ref.astype(np.float32) + bias_f32).astype(_np_c)

        if fused_epilogue == "bias_silu":
            c_ref_f32 = c_ref.astype(np.float32)
            c_ref = (c_ref_f32 * (1.0 / (1.0 + np.exp(-c_ref_f32)))).astype(_np_c)

        c_result = cp.asnumpy(c_cp[:, :, 0])
        max_err = np.max(np.abs(c_result.astype(np.float32) - c_ref.astype(np.float32)))
        print(f"{_tag} Max abs error: {max_err:.6f} (tolerance: {tolerance})")
        np.testing.assert_allclose(
            c_result.astype(np.float32), c_ref.astype(np.float32),
            atol=tolerance, rtol=1e-2,
        )
        print(f"{_tag} Reference check PASSED")

    # ---- Benchmark ----
    for _ in range(warmup_iterations):
        compiled_gemm(a_dyn, b_dyn, c_dyn, current_stream)
    cp.cuda.get_current_stream().synchronize()

    t0 = time.perf_counter()
    for _ in range(iterations):
        compiled_gemm(a_dyn, b_dyn, c_dyn, current_stream)
    cp.cuda.get_current_stream().synchronize()
    us = (time.perf_counter() - t0) * 1e6 / iterations
    print(f"{_tag} Latency: {us:.2f} us (M={m}, N={n}, K={k})")

    return us


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CuTe DSL Blackwell GEMM (C = A @ B^T, FP16/FP32acc) for Qwen3-Omni Talker MLP."
    )
    p.add_argument(
        "--mnk", type=parse_comma_separated_ints, default=(256, 256, 512),
        help="M,N,K dimensions (comma-separated)",
    )
    p.add_argument("--export_only", action="store_true",
                   help="Compile and export .o + .h; skip test and benchmark")
    p.add_argument("--output_dir", type=str, default="./gemm_aot_artifacts",
                   help="Output directory for AOT artifacts")
    p.add_argument("--file_name", type=str, default="gemm_fp16",
                   help="Base file name for exported .h/.o")
    p.add_argument("--function_prefix", type=str, default="gemm_fp16",
                   help="C function prefix (avoids symbol conflicts)")
    p.add_argument("--skip_ref_check", action="store_true",
                   help="Skip reference correctness check")
    p.add_argument("--tolerance", type=float, default=0.05,
                   help="Tolerance for reference check")
    p.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    p.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    p.add_argument("--fused_epilogue", type=str, default="none",
        choices=["none", "bias", "bias_silu"],
        help="Fuse epilogue: none (plain GEMM), bias (GEMM+bias), bias_silu (GEMM+bias+SiLU).")
    p.add_argument("--mma_tiler_mn", type=parse_comma_separated_ints, default=(64, 128),
        help="MMA tile (M_per_cta, N_per_cta) — (64, 128) optimal for small M, "
             "(128, 128) better for M >= 1024.")
    p.add_argument("--cluster_shape_mn", type=parse_comma_separated_ints, default=(1, 2),
        help="Cluster shape (M_clusters, N_clusters) for TMA multicast.")
    p.add_argument("--c_dtype", type=str, default="float16",
        choices=["float16", "float32"],
        help="Output (C) dtype. A/B stay FP16 (tensor-core inputs); float32 emits "
             "an FP16-in/FP32-out kernel (tcgen05 already accumulates in FP32). "
             "Default float16 = unchanged behavior.")
    p.add_argument("--use_2cta", action="store_true",
        help="Pair two CTAs along M for a 2x M tile per tcgen05.mma op. "
             "Requires cluster_shape[0] >= 2 so the leader CTA can drive the "
             "peer. Wins on low-SM-count GPUs (e.g. Thor 20 SMs) for M >= 384 "
             "where larger arithmetic intensity per MMA matters more than "
             "CTA count.")
    return p.parse_known_args(args=argv)[0]


if __name__ == "__main__":
    _parsed_args = _parse_args(_saved_argv)
    args = _parsed_args

    if len(args.mnk) != 3:
        raise ValueError("--mnk must contain exactly 3 values (M,N,K)")

    run(
        mnk=args.mnk,
        export_only=args.export_only,
        output_dir=args.output_dir,
        file_name=args.file_name,
        function_prefix=args.function_prefix,
        skip_ref_check=args.skip_ref_check,
        tolerance=args.tolerance,
        warmup_iterations=args.warmup,
        iterations=args.iterations,
        fused_epilogue=args.fused_epilogue,
        mma_tiler_mn=tuple(args.mma_tiler_mn),
        cluster_shape_mn=tuple(args.cluster_shape_mn),
        use_2cta=args.use_2cta,
        c_dtype=args.c_dtype,
    )
    if not args.export_only:
        print("PASS")
