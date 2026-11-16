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

"""CuTe DSL Blackwell dual GEMM (SwiGLU gate/up fusion) for the LLM MLP.

    H = silu(A @ Wg^T) * (A @ Wu^T)

FP16 in / FP16 out / FP32 accumulation, tcgen05.mma (UMMA) + TMA on SM 100/101/103/110 (Blackwell datacenter /
Thor). Replaces the two gate/up GEMMs plus the SiLU*Mul epilogue that TensorRT/Myelin fuses with
`-peep:match_dual_gemm` (disabled on SM>=100 by Edge-LLM).

  A:  [M, K]   row-major activation (K contiguous)
  W:  [2N, K]  the PACKED gate/up weight (see below)
  H:  [M, N]   row-major output, feeds down_proj

Packed weight format (plugin-owned, `pack_gate_up_weights`): for every 128-wide output tile nt and every 64-wide
K tile kt, the 128 rows of Wg then the 128 rows of Wu, each tile row-major:
    W_packed[(nt, kt, half, r), kk] = (half == 0 ? Wg : Wu)[nt*128 + r, kt*64 + kk]
so one TMA box is one contiguous 16 KB (Wg or Wu tile) or 32 KB ([Wg | Wu]) block instead of 128 row segments
2K bytes apart (decode is a pure weight stream). The kernels read it through nested CuTe layouts
(`plugin_weight_view`); the math is bit-identical to reading row-major Wg / Wu.

Two kernel structures share one pipeline (TMA -> UMMA, TMEM accumulators, TMA-store epilogue with the SwiGLU in
the register stage):
  * two accumulators (default): per k-step two 128-wide MMAs (gate, up) share the A tile; 64x128 tiles for decode,
    128x128 with the grouped raster for M >= 4096 (A no longer fits L2);
  * `merged_n`: gate|up stacked along N -> one (M, 256) MMA per k-step into a single 256-column accumulator
    (A read from SMEM once, half the MMA instructions); the epilogue pairs accumulator column j with 128 + j.
    Shipped as 256x128 2-CTA with a 2x1 cluster (B multicast) for 64 < M <= 2048.

The exported ABI is 3D with batch L=1, like gemm_blackwell.py: A[M,K,1], W[2N,K,1], H[M,N,1].

Usage:
  # Test + benchmark on a Blackwell GPU (Thor):
  python dual_gemm_blackwell.py --mnk 1024,12288,4096 --merged_n --mma_tiler_mn 256,128 --cluster_shape_mn 2,1 --use_2cta

  # AOT export (.o + .h) for the plugin on a GPU-less host:
  CUTE_DSL_ARCH=sm_110a python dual_gemm_blackwell.py --mnk 8,12288,4096 --mma_tiler_mn 64,128 \\
      --export_only --output_dir ./aot \\
      --file_name dual_gemm_blackwell_decode_fp16 --function_prefix dual_gemm_blackwell_decode_fp16
"""

import argparse
import sys
import time
from typing import Optional, Tuple, Type, Union

_parsed_args = None
_saved_argv = None
if __name__ == "__main__":
    _saved_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import numpy as np
from cutedsl_utils import aot_placeholders
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from common import (
    create_fake_row_major_3d_tensor,
    export_compiled_kernel,
    mark_3d_row_major_dynamic,
    parse_comma_separated_ints,
)


class DualGemmBlackwellFP16:
    """Blackwell SM100 dual GEMM: H[M,N] = silu(A @ Wg^T) * (A @ Wu^T).

    FP16 inputs/outputs with FP32 accumulation via tcgen05.mma (UMMA).
    TMA store epilogue (cp.bulk.tensor.s2g) with auto box-bounds.

    Configurable per construction:
      mma_tiler_mn:     per-CTA tile (M_per_cta, N_per_cta). (128, 128) for
                        prefill; (64, 128) for small M (decode).
      cluster_shape_mn: cluster size for TMA multicast (default (1, 1)).
      use_2cta:         pair two CTAs along M (requires cluster[0] >= 2).
      mma_inst_tile_k:  K-tile multiplier per MMA instruction (default 4 ->
                        K-tile 64 for FP16).
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
        use_2cta: bool = False,
        raster_group_m: int = 0,
        merged_n: bool = False,
    ):
        if mma_tiler_mn[1] != PLUGIN_WEIGHT_TILE_N:
            raise ValueError(
                f"the packed weight format fixes the output N tile at {PLUGIN_WEIGHT_TILE_N}, got {mma_tiler_mn[1]}"
            )
        # Grouped tile raster order (Triton/CUTLASS "group_m" swizzle): consecutive CTAs cover a
        # raster_group_m x N_tiles stripe so the ~20 co-resident CTAs on Thor share A row-tiles and
        # B column-tiles through L2 instead of streaming all of A once per N tile. 0 = plain order.
        if raster_group_m > 0 and (tuple(cluster_shape_mn) != (1, 1) or use_2cta):
            raise ValueError("raster_group_m requires cluster_shape_mn=(1,1) and use_2cta=False")
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = use_2cta
        self.cluster_shape_mn = cluster_shape_mn
        self.raster_group_m = raster_group_m
        # merged_n: gate|up stacked along N -> MMA tile (M, 2*N_out), a single accumulator whose column halves are
        # gate | up, one UMMA per k-step. Otherwise two accumulators (gate, up) of N_out columns each.
        self.merged_n = merged_n
        self.n_out = mma_tiler_mn[1]
        mma_n = 2 * self.n_out if merged_n else self.n_out
        self.mma_tiler_mn = (mma_tiler_mn[0], mma_n)
        self.mma_tiler = (mma_tiler_mn[0], mma_n, 1)
        # fp16 UMMA K = 16 per instruction; the K tile must equal the packed block's K (64).
        self.mma_inst_tile_k = PLUGIN_WEIGHT_TILE_K // 16
        self.use_tma_store = True
        self.num_acc = 1 if merged_n else 2
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
        # Output (H) tile per CTA: N_out wide. Equal to the MMA tile unless merged_n (MMA N = 2*N_out).
        self.out_tile_shape_mnk = (self.cta_tile_shape_mnk[0], self.n_out, self.mma_tiler[2])

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.out_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        # CuTe DSL does not reliably auto-detect Thor's Blackwell family SMs
        # (SM101/SM110) in get_smem_capacity_in_bytes(); request the Blackwell
        # family capacity explicitly (same as gemm_blackwell.py / FMHA).
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
            1 if self.merged_n else 2,
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage,
        )
        # Wg and Wu share one B smem layout (same dtype / tile); each gets its
        # own staged buffer.
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage,
        )

        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        w: cute.Tensor,
        c: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """H (M, N, L) = silu(A @ Wg^T) * (A @ Wu^T); `w` is the packed (2N, K, L) gate/up weight."""
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = w.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(w).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"A/W dtype mismatch: {self.a_dtype} vs {self.b_dtype}")

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
        # Trivial tiled MMA over the OUTPUT tile (N_out): partitions C and sizes the epilogue. Same as `tiled_mma`
        # unless merged_n (MMA N = 2*N_out). Must be created here and passed into the kernel as an argument.
        tiled_mma_out = tiled_mma
        if cutlass.const_expr(self.merged_n):
            tiled_mma_out = sm100_utils.make_trivial_tiled_mma(
                self.a_dtype, self.a_major_mode, self.b_major_mode, self.acc_dtype, self.cta_group,
                (self.mma_tiler[0], self.n_out),
            )

        # TMA load A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load Wg / Wu (two descriptors, one shared smem layout)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        # B operands are views of the packed weight: the whole 256-row [Wg | Wu] block per MMA tile (merged_n)
        # or its 128-row Wg / Wu halves (two accumulators).
        if cutlass.const_expr(self.merged_n):
            b_gate = plugin_weight_view(w, None)
            b_up = b_gate
        else:
            b_gate = plugin_weight_view(w, 0)
            b_up = plugin_weight_view(w, 1)
        tma_atom_bg, tma_tensor_bg = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b_gate, b_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        if cutlass.const_expr(self.merged_n):
            # merged_n: one B operand; the second descriptor is the same one passed through.
            tma_atom_bu, tma_tensor_bu = tma_atom_bg, tma_tensor_bg
        else:
            tma_atom_bu, tma_tensor_bu = cute.nvgpu.make_tiled_tma_atom_B(
                b_op, b_up, b_smem_layout, self.mma_tiler, tiled_mma,
                self.cluster_layout_vmnk.shape,
            )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        # One pipeline stage = A tile + Wg tile + Wu tile (merged_n: A + one [Wg|Wu] tile).
        num_b_tiles = 1 if self.merged_n else 2
        self.num_tma_load_bytes = (a_copy_size + num_b_tiles * b_copy_size) * atom_thr_size

        # TMA store atom for H
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c,
            epi_smem_layout,
            self.epi_tile,
        )

        grid = self._compute_grid(c, self.out_tile_shape_mnk, self.cluster_shape_mn)

        self.kernel(
            tiled_mma,
            tiled_mma_out,
            tma_atom_a, tma_tensor_a,
            tma_atom_bg, tma_tensor_bg,
            tma_atom_bu, tma_tensor_bu,
            tma_atom_c, tma_tensor_c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
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
        tiled_mma_out: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_bg: cute.CopyAtom,
        mBg_nkl: cute.Tensor,
        tma_atom_bu: cute.CopyAtom,
        mBu_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_bg)
            cpasync.prefetch_descriptor(tma_atom_bu)
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

        # Barrier id 0 is reserved (sync_threads()); use a named barrier.
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

        # SMEM tensors: C staging (TMA store), A, Wg, Wu.
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
        sBg = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        if cutlass.const_expr(self.merged_n):
            sBu = sBg   # merged_n: one [Wg|Wu] tile per stage; no second buffer
        else:
            sBu = smem.allocate_tensor(
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
        # (bN, bK, RestN, RestK, RestL) for Wg and Wu
        gBg_nkl = cute.local_tile(
            mBg_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gBu_nkl = cute.local_tile(
            mBu_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN_out, RestM, RestN, RestL) -- the OUTPUT tile (N_out wide; = the MMA tile unless merged_n)
        gC_mnl = cute.local_tile(
            mC_mnl, (self.mma_tiler[0], self.n_out), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        if cutlass.const_expr(self.raster_group_m > 0):
            # Remap the linear CTA id onto (tile_m, tile_n) in grouped order.
            m_tiles = cute.size(gC_mnl, mode=[2])
            n_tiles = cute.size(gC_mnl, mode=[3])
            lin = bidy * m_tiles + bidx
            num_in_group = self.raster_group_m * n_tiles
            group_id = lin // num_in_group
            first_m = group_id * self.raster_group_m
            group_size_m = cutlass.min(m_tiles - first_m, self.raster_group_m)
            in_group = lin % num_in_group
            mma_tile_coord_mnl = (
                first_m + in_group % group_size_m,
                in_group // group_size_m,
                bidz,
            )

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgBg = thr_mma.partition_B(gBg_nkl)
        tCgBu = thr_mma.partition_B(gBu_nkl)
        # C is partitioned with the OUTPUT-tile MMA (N_out); same object as tiled_mma unless merged_n.
        tCgC = tiled_mma_out.get_slice(mma_tile_coord_v).partition_C(gC_mnl)

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
        tBsBg, tBgBg = cpasync.tma_partition(
            tma_atom_bg,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sBg, 0, 3),
            cute.group_modes(tCgBg, 0, 3),
        )
        tBsBu, tBgBu = cpasync.tma_partition(
            tma_atom_bu,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sBu, 0, 3),
            cute.group_modes(tCgBu, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrBg = tiled_mma.make_fragment_B(sBg)
        tCrBu = tiled_mma.make_fragment_B(sBu)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # Two accumulators stacked along an extra mode: [..., 0] = gate, [..., 1] = up
        # (merged_n: a single (M, 2*N_out) accumulator whose column halves are gate | up).
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc))

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        tmem.allocate(self.num_tmem_alloc_cols)
        tmem.wait_for_alloc()

        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        tCtAccG = tCtAcc_base[(None, None, None, 0)]
        if cutlass.const_expr(self.merged_n):
            tCtAccU = tCtAccG      # same accumulator; the epilogue splits its columns into gate | up halves
        else:
            tCtAccU = tCtAcc_base[(None, None, None, 1)]

        tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        tBgBg = tBgBg[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
        tBgBu = tBgBu[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

        # ---- Mainloop: pipelined TMA load (A, Wg, Wu) + 2x tcgen05 MMA ----
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
                    tma_atom_bg,
                    tBgBg[(None, k_tile_idx)],
                    tBsBg[(None, producer_handle.index)],
                    tma_bar_ptr=producer_handle.barrier,
                    mcast_mask=b_full_mcast_mask,
                )
                if cutlass.const_expr(not self.merged_n):
                    cute.copy(
                        tma_atom_bu,
                        tBgBu[(None, k_tile_idx)],
                        tBsBu[(None, producer_handle.index)],
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
                        tma_atom_bg,
                        tBgBg[(None, producer_handle.count)],
                        tBsBg[(None, producer_handle.index)],
                        tma_bar_ptr=producer_handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )
                    if cutlass.const_expr(not self.merged_n):
                        cute.copy(
                            tma_atom_bu,
                            tBgBu[(None, producer_handle.count)],
                            tBsBu[(None, producer_handle.index)],
                            tma_bar_ptr=producer_handle.barrier,
                            mcast_mask=b_full_mcast_mask,
                        )

                if is_leader_cta:
                    consumer_handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                    num_kblks = cute.size(tCrA, mode=[2])
                    for kblk_idx in cutlass.range(num_kblks, unroll_full=True):
                        kblk_crd = (None, None, kblk_idx, consumer_handle.index)
                        if cutlass.const_expr(self.merged_n):
                            # one MMA: acc[:, 0:N_out] += A @ Wg^T, acc[:, N_out:2N_out] += A @ Wu^T (A read once)
                            cute.gemm(
                                tiled_mma, tCtAccG, tCrA[kblk_crd], tCrBg[kblk_crd], tCtAccG
                            )
                        else:
                            # gate: acc_g += A_tile @ Wg_tile^T
                            cute.gemm(
                                tiled_mma, tCtAccG, tCrA[kblk_crd], tCrBg[kblk_crd], tCtAccG
                            )
                            # up:   acc_u += A_tile @ Wu_tile^T (same A fragment)
                            cute.gemm(
                                tiled_mma, tCtAccU, tCrA[kblk_crd], tCrBu[kblk_crd], tCtAccU
                            )
                        # Both accumulators saw their first MMA with ACCUMULATE=0;
                        # every later MMA accumulates.
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    consumer_handle.release()

                if k_tile_idx + 1 < k_tile_cnt - prefetch_k_tile_cnt:
                    peek_ab_empty_status = ab_producer.try_acquire()

                if k_tile_idx + 1 < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

            # Commit accumulators full ONCE after all k-tiles are MMA'd.
            if is_leader_cta:
                acc_pipeline.producer_commit(acc_producer_state)

        # ---- Epilogue: TMEM(g,u) -> regs -> silu(g)*u -> SMEM -> gmem via TMA ----
        tmem.relinquish_alloc_permit()
        acc_pipeline.consumer_wait(acc_consumer_state)

        # (M, N) accumulator views handed to the epilogue: two accumulators, or the two column halves of the
        # merged (M, 2*N_out) accumulator (gate = columns [0, N_out), up = [N_out, 2*N_out)).
        if cutlass.const_expr(self.merged_n):
            acc_mn = tCtAccG[((None, None), 0, 0)]
            acc_halves = cute.flat_divide(acc_mn, (self.out_tile_shape_mnk[0], self.n_out))
            accG_mn = acc_halves[(None, None, 0, 0)]
            accU_mn = acc_halves[(None, None, 0, 1)]
        else:
            accG_mn = tCtAccG[((None, None), 0, 0)]
            accU_mn = tCtAccU[((None, None), 0, 0)]
        self.epilogue_tma_store(
            tidx, warp_idx, mma_tile_coord_mnl, tma_atom_c,
            accG_mn, accU_mn, sC, tCgC, epi_tile,
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
        tCtAccG: cute.Tensor,
        tCtAccU: cute.Tensor,
        sC: cute.Tensor,
        tCgC: cute.Tensor,
        epi_tile: cute.Tile,
    ) -> None:
        """TMA-store epilogue with the SwiGLU fusion in the register stage.

        Per epilogue subtile: T2R(acc_g), T2R(acc_u), h = silu(g) * u (FP32),
        convert to c_dtype, R2S, TMA S2G. The TMA descriptor clips partial
        M/N tiles, so no manual predication is needed.
        """
        # T2R: TMEM -> registers (per-thread fragment). Both accumulators
        # have the same TMEM layout, so one tiled copy serves both.
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.out_tile_shape_mnk, self.c_layout, self.c_dtype, self.acc_dtype,
            epi_tile, self.use_2cta_instrs,
        )
        # tCtAccG / tCtAccU arrive as (M, N_out) TMEM views (see kernel()).
        tAccG_epi = cute.flat_divide(tCtAccG, epi_tile)
        tAccU_epi = cute.flat_divide(tCtAccU, epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAccG_epi[(None, None, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
        tTR_tAccG = thr_copy_t2r.partition_S(tAccG_epi)
        tTR_tAccG = cute.group_modes(tTR_tAccG, 3, cute.rank(tTR_tAccG))
        tTR_tAccU = thr_copy_t2r.partition_S(tAccU_epi)
        tTR_tAccU = cute.group_modes(tTR_tAccU, 3, cute.rank(tTR_tAccU))

        tCgC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tTR_gC = thr_copy_t2r.partition_D(tCgC_epi)
        frag_shape = tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape
        tTR_rAccG = cute.make_rmem_tensor(frag_shape, self.acc_dtype)
        tTR_rAccU = cute.make_rmem_tensor(frag_shape, self.acc_dtype)
        tTR_rC = cute.make_rmem_tensor(frag_shape, self.c_dtype)

        # R2S: registers -> SMEM (staging buffer for TMA store).
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(epi_tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)

        # TMA store partition: SMEM -> gmem.
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

        subtile_cnt = cute.size(tTR_tAccG.shape, mode=[3])
        for subtile_idx in cutlass.range(subtile_cnt):
            cute.copy(tiled_copy_t2r, tTR_tAccG[(None, None, None, subtile_idx)], tTR_rAccG)
            cute.copy(tiled_copy_t2r, tTR_tAccU[(None, None, None, subtile_idx)], tTR_rAccU)

            # h = silu(g) * u = g * sigmoid(g) * u, FP32. Fully unrolled so the
            # fragment stays in registers (a dynamic index would spill to local).
            for i in cutlass.range(cute.size(tTR_rAccG), unroll_full=True):
                g = tTR_rAccG[i]
                u = tTR_rAccU[i]
                sig = cute.arch.rcp_approx(1.0 + cute.exp(-g, fastmath=True))
                tTR_rAccG[i] = g * sig * u

            # Convert acc -> c_dtype in registers.
            acc_vec = tiled_copy_r2s.retile(tTR_rAccG).load()
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

    @staticmethod
    def _compute_stages(
        tiled_mma, mma_tiler_mnk, a_dtype, b_dtype,
        epi_tile, c_dtype, c_layout, smem_capacity, occupancy, num_b_tiles=2,
    ) -> Tuple[int, int, int]:
        num_acc_stage = 1
        num_c_stage = 2

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, a_dtype, 1,
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, b_dtype, 1,
        )
        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype, c_layout, epi_tile, 1
        )
        # One stage holds A + Wg + Wu tiles (merged_n: A + one [Wg|Wu] tile).
        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one
        ) + num_b_tiles * cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        num_ab_stage = (
            smem_capacity - (occupancy + 1) * (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

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
    def _compute_num_tmem_alloc_cols(tiled_mma, mma_tiler, num_acc):
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc))
        return utils.get_num_tmem_alloc_cols(tCtAcc_fake)


# ---------------------------------------------------------------------------
# Packed gate/up weight format (plugin-owned)
# ---------------------------------------------------------------------------

PLUGIN_WEIGHT_TILE_N = 128   # rows of Wg (and of Wu) per packed block = the kernels' output N tile
PLUGIN_WEIGHT_TILE_K = 64    # K per packed block = the kernels' K tile (128 B rows)


def plugin_weight_view(w: cute.Tensor, half: Optional[int]) -> cute.Tensor:
    """JIT helper: re-interpret the packed (2N, K, L) weight `[N/128][K/64][Wg 128x64 | Wu 128x64]`.

    half None -> the (2N, K, L) tensor with the nested layout ((256, N/128), (64, K/64), L): every MMA B tile
                 of 256 rows ([Wg | Wu] of one output tile) is one contiguous 32 KB block (merged_n).
    half 0 / 1 -> the (N, K, L) view of Wg / Wu: ((128, N/128), (64, K/64), L) selecting the first / second
                 16 KB block (two accumulators).
    """
    n2, k, l = w.shape[0], w.shape[1], w.shape[2]
    bn, bk = PLUGIN_WEIGHT_TILE_N, PLUGIN_WEIGHT_TILE_K
    blk = 2 * bn                       # rows per (n-tile, k-tile) block
    n_tiles = n2 // blk
    k_tiles = k // bk
    if half is None:
        layout = cute.make_layout(
            ((blk, n_tiles), (bk, k_tiles), l),
            stride=((bk, blk * k), (1, blk * bk), n2 * k),
        )
        return cute.make_tensor(w.iterator, layout)
    layout = cute.make_layout(
        ((bn, n_tiles), (bk, k_tiles), l, 2),
        stride=((bk, blk * k), (1, blk * bk), n2 * k, bn * bk),
    )
    return cute.make_tensor(w.iterator, layout)[(None, None, None, half)]


def pack_gate_up_weights(wg, wu):
    """Host (numpy or cupy): Wg, Wu (N, K) row-major -> the packed (2N, K) weight consumed by the kernels."""
    xp = __import__(type(wg).__module__.split(".")[0])
    n, k = int(wg.shape[0]), int(wg.shape[1])
    bn, bk = PLUGIN_WEIGHT_TILE_N, PLUGIN_WEIGHT_TILE_K
    if wu.shape != wg.shape or n % bn or k % bk:
        raise ValueError(f"packed gate/up weights need Wg.shape == Wu.shape, N % {bn} == 0, K % {bk} == 0 (got {wg.shape}, {wu.shape})")
    g = wg.reshape(n // bn, bn, k // bk, bk).transpose(0, 2, 1, 3)   # (nt, kt, r, kk)
    u = wu.reshape(n // bn, bn, k // bk, bk).transpose(0, 2, 1, 3)
    packed = xp.stack([g, u], axis=2)                                 # (nt, kt, half, r, kk)
    return xp.ascontiguousarray(packed.reshape(2 * n, k))


# ---------------------------------------------------------------------------
# Host helpers (numpy / cupy)
# ---------------------------------------------------------------------------


def _logical_3d(xp, mode0: int, mode1: int, batch: int, dtype, fill=None):
    """(mode0, mode1, L) logical tensor over (L, mode0, mode1) contiguous storage."""
    if fill is None:
        base = xp.zeros((batch, mode0, mode1), dtype=dtype)
    else:
        base = fill((batch, mode0, mode1)).astype(dtype)
    return xp.transpose(base, (1, 2, 0))


def silu_ref_np(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def check_close(out: np.ndarray, ref: np.ndarray, rel_tol: float = 0.02):
    """Scale-aware correctness gate for FP16 outputs of FP32-accumulated GEMMs.

    Passes when every value is finite, the relative L2 error is < rel_tol/4 and
    the max abs error is < rel_tol * max|ref| (+ 2 FP16 ulps at that scale).
    An all-zero or swapped-operand output fails regardless of output scale.
    """
    out = out.astype(np.float32)
    ref = ref.astype(np.float32)
    finite = bool(np.isfinite(out).all())
    ref_max = float(np.abs(ref).max()) if ref.size else 0.0
    max_err = float(np.abs(out - ref).max()) if ref.size else 0.0
    rel_l2 = float(np.linalg.norm(out - ref) / max(np.linalg.norm(ref), 1e-30))
    ulp = float(np.spacing(np.float16(ref_max))) if ref_max > 0 else 1e-6
    max_tol = rel_tol * ref_max + 2.0 * ulp
    ok = finite and rel_l2 < rel_tol / 4.0 and max_err <= max_tol
    msg = (f"max_abs_err={max_err:.4g} (tol {max_tol:.4g}, max|ref|={ref_max:.4g}) "
           f"rel_l2={rel_l2:.3g} (tol {rel_tol / 4.0:.3g}) finite={finite} -> {'OK' if ok else 'FAIL'}")
    return ok, msg


# ---------------------------------------------------------------------------
# Run / Export
# ---------------------------------------------------------------------------

def run(
    mnk: Tuple[int, int, int],
    export_only: bool = False,
    no_gpu: bool = False,
    output_dir: str = "./dual_gemm_aot_artifacts",
    file_name: str = "dual_gemm_blackwell_fp16",
    function_prefix: str = "dual_gemm_blackwell_fp16",
    skip_ref_check: bool = False,
    tolerance: float = 0.02,
    warmup_iterations: int = 5,
    iterations: int = 20,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    cluster_shape_mn: Tuple[int, int] = (1, 1),
    use_2cta: bool = False,
    raster_group_m: int = 0,
    merged_n: bool = False,
    weight_scale: Optional[float] = None,
    seed: int = 1111,
):
    m, n, k = mnk
    _tag = f"[{file_name}]"
    l = 1

    print(f"{_tag} Dual GEMM H = silu(A@Wg^T) * (A@Wu^T): M={m}, N={n}, K={k} "
          f"tile={mma_tiler_mn} cluster={cluster_shape_mn} 2cta={use_2cta} raster_group_m={raster_group_m} "
          f"merged_n={merged_n} {'AOT-export' if export_only else 'run'}")

    if no_gpu and not export_only:
        raise ValueError("--no_gpu is only valid together with --export_only")

    if export_only:
        a_dyn = create_fake_row_major_3d_tensor(batch=l)
        w_dyn = create_fake_row_major_3d_tensor(batch=l)
        c_dyn = create_fake_row_major_3d_tensor(batch=l)
        stream = aot_placeholders.make_stream()
    else:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError("GPU is required!")
        cp.random.seed(seed)
        scale = weight_scale if weight_scale is not None else (1.0 / np.sqrt(k))
        a_x = _logical_3d(cp, m, k, l, cp.float16, lambda s: cp.random.uniform(-1, 1, size=s))
        wg = cp.random.uniform(-scale, scale, size=(n, k)).astype(cp.float16)   # row-major reference weights
        wu = cp.random.uniform(-scale, scale, size=(n, k)).astype(cp.float16)
        w_x = _logical_3d(cp, 2 * n, k, l, cp.float16)
        w_x[:, :, 0] = pack_gate_up_weights(wg, wu)
        c_x = _logical_3d(cp, m, n, l, cp.float16)
        stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)

        a_dyn = mark_3d_row_major_dynamic(from_dlpack(a_x, assumed_align=16))
        w_dyn = mark_3d_row_major_dynamic(from_dlpack(w_x, assumed_align=16))
        c_dyn = mark_3d_row_major_dynamic(from_dlpack(c_x, assumed_align=16))

    dual = DualGemmBlackwellFP16(
        acc_dtype=cutlass.Float32,
        mma_tiler_mn=tuple(mma_tiler_mn),
        cluster_shape_mn=tuple(cluster_shape_mn),
        use_2cta=use_2cta,
        raster_group_m=raster_group_m,
        merged_n=merged_n,
    )

    t0 = time.time()
    compiled = cute.compile(
        dual, a_dyn, w_dyn, c_dyn, stream,
        **(dict(options=aot_placeholders.compile_options()) if export_only else {}),
    )
    print(f"{_tag} Compilation time: {time.time() - t0:.2f}s "
          f"(ab_stages={dual.num_ab_stage} c_stages={dual.num_c_stage} "
          f"tmem_cols={dual.num_tmem_alloc_cols} k_tile={dual.mma_tiler[2]})")

    if export_only:
        export_compiled_kernel(
            compiled, output_dir=output_dir, file_name=file_name,
            function_prefix=function_prefix, tag=_tag,
        )
        return None

    import cupy as cp  # GPU path only

    # ---- Functional test ----
    if not skip_ref_check:
        compiled(a_dyn, w_dyn, c_dyn, stream)
        cp.cuda.get_current_stream().synchronize()
        a_f32 = cp.asnumpy(a_x[:, :, 0]).astype(np.float32)
        g_ref = a_f32 @ cp.asnumpy(wg).astype(np.float32).T
        u_ref = a_f32 @ cp.asnumpy(wu).astype(np.float32).T
        h_ref = (silu_ref_np(g_ref) * u_ref).astype(np.float16).astype(np.float32)
        h_out = cp.asnumpy(c_x[:, :, 0]).astype(np.float32)
        ok, msg = check_close(h_out, h_ref, rel_tol=tolerance)
        print(f"{_tag} {msg}")
        if not ok:
            raise AssertionError(f"reference check FAILED: {msg}")
        print(f"{_tag} Reference check PASSED")

    # ---- Benchmark (CUDA events) ----
    for _ in range(warmup_iterations):
        compiled(a_dyn, w_dyn, c_dyn, stream)
    cp.cuda.get_current_stream().synchronize()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(iterations):
        compiled(a_dyn, w_dyn, c_dyn, stream)
    end.record()
    end.synchronize()
    us = cp.cuda.get_elapsed_time(start, end) * 1e3 / iterations
    flops = 2.0 * 2.0 * m * n * k
    bytes_moved = 2.0 * (m * k + 2 * n * k + m * n)
    print(f"{_tag} Latency: {us:.2f} us  |  {flops / us / 1e6:.2f} TFLOP/s  |  "
          f"{bytes_moved / us / 1e3:.1f} GB/s effective  (M={m}, N={n}, K={k})")
    return us


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CuTe DSL Blackwell dual GEMM: H = silu(A@Wg^T) * (A@Wu^T), FP16/FP32acc, packed gate/up weight."
    )
    p.add_argument("--mnk", type=parse_comma_separated_ints, default=(256, 512, 512),
                   help="M,N,K (M tokens, N intermediate size, K hidden size)")
    p.add_argument("--export_only", action="store_true",
                   help="Compile and export .o + .h; skip test and benchmark")
    p.add_argument("--no_gpu", action="store_true",
                   help="Compatibility flag; --export_only always uses GPU-free placeholders")
    p.add_argument("--output_dir", type=str, default="./dual_gemm_aot_artifacts")
    p.add_argument("--file_name", type=str, default="dual_gemm_blackwell_fp16")
    p.add_argument("--function_prefix", type=str, default="dual_gemm_blackwell_fp16")
    p.add_argument("--skip_ref_check", action="store_true")
    p.add_argument("--tolerance", type=float, default=0.02,
                   help="relative tolerance: max_abs_err < tol*max|ref|, rel_L2 < tol/4")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--mma_tiler_mn", type=parse_comma_separated_ints, default=(128, 128),
                   help="MMA tile (M_per_cta, N_out): N_out must be 128; (64,128) decode, (128,128) large prefill, "
                        "(256,128) with --use_2cta --merged_n for 64 < M <= 2048")
    p.add_argument("--cluster_shape_mn", type=parse_comma_separated_ints, default=(1, 1))
    p.add_argument("--use_2cta", action="store_true")
    p.add_argument("--raster_group_m", type=int, default=0,
                   help="grouped CTA raster order (group of M tiles per stripe); 0 = plain order")
    p.add_argument("--merged_n", action="store_true",
                   help="gate|up stacked along N: MMA tile (M, 256), one accumulator, one UMMA per k-step")
    p.add_argument("--weight_scale", type=float, default=None,
                   help="Uniform(-s, s) weight init; default 1/sqrt(K)")
    p.add_argument("--seed", type=int, default=1111)
    return p.parse_known_args(args=argv)[0]


if __name__ == "__main__":
    _parsed_args = _parse_args(_saved_argv)
    args = _parsed_args
    if len(args.mnk) != 3:
        raise ValueError("--mnk must contain exactly 3 values (M,N,K)")
    run(
        mnk=args.mnk,
        export_only=args.export_only,
        no_gpu=args.no_gpu,
        output_dir=args.output_dir,
        file_name=args.file_name,
        function_prefix=args.function_prefix,
        skip_ref_check=args.skip_ref_check,
        tolerance=args.tolerance,
        warmup_iterations=args.warmup,
        iterations=args.iterations,
        mma_tiler_mn=tuple(args.mma_tiler_mn),
        cluster_shape_mn=tuple(args.cluster_shape_mn),
        use_2cta=args.use_2cta,
        raster_group_m=args.raster_group_m,
        merged_n=args.merged_n,
        weight_scale=args.weight_scale,
        seed=args.seed,
    )
    if not args.export_only:
        print("PASS")
