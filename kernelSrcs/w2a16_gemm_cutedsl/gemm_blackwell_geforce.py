# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# IQ2_XXS specialization of b12x 2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88,
# derived from the CUTLASS dense block-scaled GEMM example.

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Int32, Int64, Uint32
from cutlass.cute.nvgpu import cpasync
from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo
from w2a16_gemm_cutedsl.intrinsics import (cp_async4_shared_global,
                                           cp_async_bulk_g2s_mbar,
                                           elem_pointer, get_ptr_as_int64,
                                           iq2_xs_descriptor_pair_to_bf16x2x2,
                                           iq2_xxs_descriptor_pair,
                                           shared_ptr_to_u32)


def _convert_layout_acc_mn(acc_layout: cute.Layout,
                           transpose: bool = False) -> cute.Layout:
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    shape = ((acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
             (acc_layout_col_major.shape[0][0],
              *acc_layout_col_major.shape[0][2:],
              acc_layout_col_major.shape[2]), *acc_layout_col_major.shape[3:])
    stride = ((acc_layout_col_major.stride[0][1],
               acc_layout_col_major.stride[1]),
              (acc_layout_col_major.stride[0][0],
               *acc_layout_col_major.stride[0][2:],
               acc_layout_col_major.stride[2]),
              *acc_layout_col_major.stride[3:])
    if cutlass.const_expr(transpose):
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    return cute.composition(acc_layout, cute.make_layout(shape, stride=stride))


def _reshape_acc_to_mn(acc: cute.Tensor,
                       transpose: bool = False) -> cute.Tensor:
    return cute.make_tensor(
        acc.iterator, _convert_layout_acc_mn(acc.layout, transpose=transpose))


class DenseGemmKernel:

    def __init__(self, tile_m, tile_n, tile_k, slices, descriptor_tile_n):
        self.tile_shape_mnk = (tile_m, tile_n, tile_k)
        self.swap_ab = tile_m == 8
        self.a16_block_transposed = self.swap_ab
        self.mma_tile_shape_mnk = (
            tile_n, tile_m, tile_k) if self.swap_ab else self.tile_shape_mnk
        self.atom_shape = (tile_n // 16, 1, 1) if self.swap_ab else (1, 8, 1)
        self.num_mma_warps = tile_n // 16 if self.swap_ab else 8
        self.descriptor_tile_n = descriptor_tile_n
        self.acc_dtype = cutlass.Float32
        self.split_k_slices = slices
        self.cluster_shape_mnk = (1, 1, 1)
        self.epi_tile = (tile_m, tile_n)
        self.sfa_tiles_per_block = max(128, tile_m) // tile_m
        self.sfb_tiles_per_block = max(128, tile_n) // tile_n
        self.num_threads_per_warp = 32
        self.tma_load_warp_id = self.num_mma_warps
        self.threads_per_cta = (self.num_mma_warps + 1) * 32
        self.smem_capacity = utils.get_smem_capacity_in_bytes('sm_120')
        self.occupancy = 1
        self.buffer_align_bytes = 1024
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.num_mma_warps * 32)
        self.load_register_requirement = 40
        self.mma_register_requirement = 232

    def _setup_attributes(self):
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.BFloat16, self.acc_dtype,
                                              (16, 8, 16))
        atom_shape = self.atom_shape
        atom_layout = cute.make_layout(atom_shape)
        permutation_mnk = self.mma_tile_shape_mnk
        self.tiled_mma = cute.make_tiled_mma(mma_op,
                                             atom_layout,
                                             permutation_mnk=permutation_mnk)
        self.mma_atom = cute.make_mma_atom(mma_op)
        mma_m, mma_n, mma_k = (16, 8, 16)
        self.num_m_tiles = self.mma_tile_shape_mnk[0] // (mma_m *
                                                          atom_shape[0])
        self.num_n_tiles = self.mma_tile_shape_mnk[1] // (mma_n *
                                                          atom_shape[1])
        self.num_k_blocks = self.mma_tile_shape_mnk[2] // mma_k
        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        self.a16_scale_bytes = 256
        sfa_smem_layout_per_stage = cute.make_layout((1, 1))
        sfb_smem_layout_per_stage = cute.make_layout((self.a16_scale_bytes, 1))

        def _probe_a16_stages():
            return self._compute_stages(self.tile_shape_mnk,
                                        self.a_dtype,
                                        cutlass.Uint8,
                                        self.sf_dtype,
                                        sfa_smem_layout_per_stage,
                                        sfb_smem_layout_per_stage,
                                        self.epi_tile,
                                        self.c_dtype,
                                        self.smem_capacity -
                                        4096 * self.occupancy,
                                        self.occupancy,
                                        epi_stage_cap=1,
                                        minimum_ab_stage=0,
                                        b_storage_bits=8 // 4)

        self.ab_stage, self.epi_stage = _probe_a16_stages()
        while self.ab_stage == 0:
            epi_m, epi_n = self.epi_tile
            if epi_n // 2 >= 8 * self.atom_shape[1]:
                self.epi_tile = (epi_m, epi_n // 2)
            elif epi_m // 2 >= 16 * self.atom_shape[0]:
                self.epi_tile = (epi_m // 2, epi_n)
            else:
                raise ValueError('A16 tile cannot fit one shared-memory stage')
            self.ab_stage, self.epi_stage = _probe_a16_stages()
        if self.epi_stage <= 0:
            raise ValueError('A16 epilogue cannot fit one shared-memory stage')
        self.a_smem_layout_staged, self.b_smem_layout_staged, self.sfa_smem_layout_staged, self.sfb_smem_layout_staged, self.epi_smem_layout_staged = self._make_smem_layouts(
            self.tile_shape_mnk, self.epi_tile, self.a_dtype, self.a_layout,
            self.b_dtype, self.b_layout, self.ab_stage, self.c_dtype,
            self.c_layout, self.epi_stage, 32, self.tiled_mma, True)
        self.sfb_smem_layout_staged = cute.make_layout(
            (self.a16_scale_bytes, 1, self.ab_stage))
        self.a16_packed_k = self.tile_shape_mnk[2] // 4
        self.a16_b_smem_layout = cute.make_layout(
            (self.tile_shape_mnk[1], self.a16_packed_k, self.ab_stage),
            stride=(self.a16_packed_k, 1,
                    self.tile_shape_mnk[1] * self.a16_packed_k))

    @cute.jit
    def __call__(self,
                 a: cute.Tensor,
                 b: cute.Tensor,
                 sfa: cute.Tensor,
                 sfb: cute.Tensor,
                 c: cute.Tensor,
                 max_active_clusters: cutlass.Constexpr,
                 stream: cuda.CUstream,
                 epilogue_op: cutlass.Constexpr = lambda x: x):
        self.a_dtype = a.element_type
        self.b_dtype = cutlass.BFloat16
        self.c_dtype = c.element_type
        self.sf_dtype = sfa.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.ROW_MAJOR
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f'Type mismatch: {self.a_dtype} != {self.b_dtype}')
        self._setup_attributes()
        sfa_tensor = sfa
        sfb_tensor = sfb
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b, self.a16_b_smem_layout,
            (self.tile_shape_mnk[1], self.a16_packed_k), 1)
        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a, self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]), 1)
        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c, self.epi_smem_layout_staged, self.epi_tile)
        tile_sched_params, grid = self._compute_grid(c, self.tile_shape_mnk,
                                                     max_active_clusters,
                                                     self.split_k_slices,
                                                     False, True)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64,
                                                              self.ab_stage *
                                                              2]
            sA: cute.struct.Align[cute.struct.MemRange[
                self.a_dtype,
                cute.cosize(self.a_smem_layout_staged)],
                                  self.buffer_align_bytes]
            sB: cute.struct.Align[cute.struct.MemRange[
                cutlass.Uint8,
                cute.cosize(self.a16_b_smem_layout)], self.buffer_align_bytes]
            sSFA: cute.struct.Align[cute.struct.MemRange[
                self.sf_dtype,
                cute.cosize(self.sfa_smem_layout_staged)],
                                    self.buffer_align_bytes]
            sSFB: cute.struct.Align[cute.struct.MemRange[
                self.sf_dtype,
                cute.cosize(self.sfb_smem_layout_staged)],
                                    self.buffer_align_bytes]
            sC: cute.struct.Align[cute.struct.MemRange[
                self.c_dtype,
                cute.cosize(self.epi_smem_layout_staged)],
                                  self.buffer_align_bytes]

        self.shared_storage = SharedStorage
        b_packed_smem_layout_arg = self.a16_b_smem_layout
        self.kernel(tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, b,
                    sfa_tensor, sfb_tensor, tma_atom_c, tma_tensor_c, c,
                    self.tiled_mma, self.cta_layout_mnk,
                    self.a_smem_layout_staged, self.b_smem_layout_staged,
                    b_packed_smem_layout_arg, self.sfa_smem_layout_staged,
                    self.sfb_smem_layout_staged, self.epi_smem_layout_staged,
                    tile_sched_params,
                    epilogue_op).launch(grid=grid,
                                        block=[self.threads_per_cta, 1, 1],
                                        cluster=[1, 1, 1],
                                        stream=stream)
        return

    @cute.jit
    def _load_a16_b_fragment(self, fragment: cute.Tensor,
                             coordinates: cute.Tensor, packed: cute.Tensor,
                             scales: cute.Tensor, lut: cute.Tensor,
                             stage: Int32, n_tile: Int32, k_tile: Int32):
        pairs = cute.recast_tensor(fragment, Uint32)
        cute.recast_tensor(packed, cutlass.Uint16)
        packed_u32 = cute.recast_tensor(packed, Uint32)
        bases = cute.recast_tensor(scales, cutlass.Uint16)
        rows_per_fragment = 2 if self.a16_block_transposed else 1
        for i in cutlass.range_constexpr(
                cute.size(fragment) // (4 * rows_per_fragment)):
            for row in cutlass.range_constexpr(rows_per_fragment):
                local_n, local_k = coordinates[i * 4 * rows_per_fragment +
                                               row * 2]
                scale_n = local_n + n_tile % self.sfb_tiles_per_block * self.tile_shape_mnk[
                    1]
                (k_tile * self.tile_shape_mnk[2] + local_k) % 256
                grids = Uint32(packed_u32[local_n, local_k // 32 * 2, stage])
                signs_scale = Uint32(packed_u32[local_n, local_k // 32 * 2 + 1,
                                                stage])
                descriptors = iq2_xxs_descriptor_pair(grids, signs_scale,
                                                      Int32(local_k // 16 % 2))
                subscale = signs_scale >> 28
                p0, p1 = iq2_xs_descriptor_pair_to_bf16x2x2(
                    descriptors, Uint32(bases[scale_n, 0, stage]), subscale,
                    Int64(shared_ptr_to_u32(lut.iterator)), Int32(local_k % 8))
                pairs[i * 2 * rows_per_fragment + row] = p0
                pairs[i * 2 * rows_per_fragment + row + rows_per_fragment] = p1

    @cute.kernel
    def kernel(self, tma_atom_a: cute.CopyAtom, mA_mkl: cute.Tensor,
               tma_atom_b: cute.CopyAtom, mB_nkl: cute.Tensor,
               directB_nkl: cute.Tensor, directSFA_mkl: cute.Tensor,
               directSFB_nkl: cute.Tensor, tma_atom_c: cute.CopyAtom,
               mC_mnl: cute.Tensor, directC_mnl: cute.Tensor,
               tiled_mma: cute.TiledMma, cta_layout_mnk: cute.Layout,
               a_smem_layout_staged: cute.ComposedLayout,
               b_smem_layout_staged: cute.ComposedLayout,
               b_packed_smem_layout_staged,
               sfa_smem_layout_staged: cute.Layout,
               sfb_smem_layout_staged: cute.Layout,
               epi_smem_layout_staged: cute.ComposedLayout,
               tile_sched_params: utils.PersistentTileSchedulerParams,
               epilogue_op: cutlass.Constexpr):
        alpha_value = cutlass.Float32(1.0)
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster())
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        cute.slice_(b_smem_layout_staged, (None, None, 0))
        cute.slice_(sfa_smem_layout_staged, (None, None, 0))
        cute.slice_(sfb_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        tma_copy_bytes += self.tile_shape_mnk[
            1] * self.a16_packed_k + self.a16_scale_bytes
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        a16_lut = directSFA_mkl
        a16_lut = smem.allocate_tensor(cutlass.Uint8,
                                       cute.make_layout((4096, )),
                                       byte_alignment=16)
        lut_address = shared_ptr_to_u32(a16_lut.iterator)
        for chunk_base in cutlass.range_constexpr(
            (4096 // 16 + self.threads_per_cta - 1) // self.threads_per_cta):
            chunk = Int32(tidx) + chunk_base * self.threads_per_cta
            if chunk < 4096 // 16:
                cp_async4_shared_global(
                    lut_address + chunk * 16,
                    get_ptr_as_int64(directSFA_mkl,
                                     Int64(chunk) * 16))
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr(
        )
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread)
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_warps)
        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk)
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer,
                                   swizzle=a_smem_layout_staged.inner)
        sBPacked = storage.sB.get_tensor(b_packed_smem_layout_staged)
        sB = cute.make_tensor(
            cute.recast_ptr(storage.sB.data_ptr(), dtype=cutlass.BFloat16),
            b_smem_layout_staged)
        sC = storage.sC.get_tensor(epi_smem_layout_staged.outer,
                                   swizzle=epi_smem_layout_staged.inner)
        storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None))
        gB_nkl = cute.local_tile(mB_nkl,
                                 (self.tile_shape_mnk[1], self.a16_packed_k),
                                 (None, None, None))
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None))
        thr_mma = tiled_mma.get_slice(tidx)
        a_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cpasync.tma_partition(tma_atom_a, a_cta_crd, a_cta_layout,
                                           cute.group_modes(sA, 0, 2),
                                           cute.group_modes(gA_mkl, 0, 2))
        b_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cpasync.tma_partition(tma_atom_b, b_cta_crd, b_cta_layout,
                                           cute.group_modes(sBPacked, 0, 2),
                                           cute.group_modes(gB_nkl, 0, 2))
        if cutlass.const_expr(self.swap_ab):
            tCsA = thr_mma.partition_A(sB)
            tCsB = thr_mma.partition_B(sA)
        else:
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        if cutlass.const_expr(self.swap_ab):
            tCgC = thr_mma.partition_C(
                cute.make_identity_tensor(
                    (self.tile_shape_mnk[1], self.tile_shape_mnk[0])))
            a16_weight_fragments = tCrA
            a16_b_coordinates = thr_mma.partition_A(
                cute.make_identity_tensor(
                    (self.tile_shape_mnk[1], self.tile_shape_mnk[2])))
        else:
            tCgC = thr_mma.partition_C(gC_mnl)
            a16_weight_fragments = tCrB
            a16_b_coordinates = thr_mma.partition_B(
                cute.make_identity_tensor(
                    (self.tile_shape_mnk[1], self.tile_shape_mnk[2])))
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        if cutlass.const_expr(self.swap_ab):
            swap_ab_acc_mn = _reshape_acc_to_mn(accumulators, transpose=True)
            swap_ab_c_identity = cute.make_identity_tensor(
                (self.tile_shape_mnk[1], self.tile_shape_mnk[0]))
            swap_ab_coord_mn = _reshape_acc_to_mn(
                thr_mma.partition_C(swap_ab_c_identity), transpose=True)
        if cutlass.const_expr(self.split_k_slices > 1):
            split_k_acc_mn = _reshape_acc_to_mn(accumulators)
            split_k_c_identity = cute.make_identity_tensor(
                cute.slice_(self.tile_shape_mnk, (None, None, 0)))
            split_k_coord_mn = _reshape_acc_to_mn(
                thr_mma.partition_C(split_k_c_identity))
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()
        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        block_idx = cute.arch.block_idx()
        k_tile_start = Int32(0)
        k_tile_iter_cnt = k_tile_cnt
        if cutlass.const_expr(self.split_k_slices > 1):
            k_tile_start = Int32(
                block_idx[1]) * k_tile_cnt // self.split_k_slices
            k_tile_iter_cnt = (
                Int32(block_idx[1]) +
                1) * k_tile_cnt // self.split_k_slices - k_tile_start
        direct_tile_valid = Int32(block_idx[2]) < Int32(
            tile_sched_params.problem_shape_ntile_mnl[1])
        work_tile = WorkTileInfo(
            (Int32(block_idx[0]), Int32(block_idx[2]), Int32(0)),
            direct_tile_valid)
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage)
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage)
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            num_k_blocks = cute.size(tCrA, mode=[2])
            if cutlass.const_expr(self.swap_ab):
                atom_copy_ldmatrix_A = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(
                        self.b_layout.is_n_major_b(), 4), self.b_dtype)
                atom_copy_ldmatrix_B = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(
                        self.a_layout.is_m_major_a(), 2), self.a_dtype)
            else:
                atom_copy_ldmatrix_A = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(
                        self.a_layout.is_m_major_a(), 4), self.a_dtype)
                atom_copy_ldmatrix_B = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(
                        self.b_layout.is_n_major_b(), 4), self.b_dtype)
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A,
                                                       tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B,
                                                       tiled_mma)
            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(
                sB if cutlass.const_expr(self.swap_ab) else sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(
                sA if cutlass.const_expr(self.swap_ab) else sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)
            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[None, None, *tile_coord_mnl]
                tile_coord_mnl[0] % self.sfa_tiles_per_block
                tile_coord_mnl[1] % self.sfb_tiles_per_block
                accumulators.fill(0.0)
                mainloop_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_iter_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state)
                mainloop_pipeline.consumer_wait(mainloop_consumer_state,
                                                peek_ab_full_status)
                tCsA_p = tCsA_copy_view[None, None, None,
                                        mainloop_consumer_state.index]
                tCsB_p = tCsB_copy_view[None, None, None,
                                        mainloop_consumer_state.index]
                if cutlass.const_expr(self.a16_block_transposed):
                    cute.copy(smem_tiled_copy_B, tCsB_p[None, None, 0],
                              tCrB_copy_view[None, None, 0])
                else:
                    cute.copy(smem_tiled_copy_A, tCsA_p[None, None, 0],
                              tCrA_copy_view[None, None, 0])
                self._load_a16_b_fragment(a16_weight_fragments[None, None, 0],
                                          a16_b_coordinates[None, None, 0],
                                          sBPacked, sSFB, a16_lut,
                                          mainloop_consumer_state.index,
                                          tile_coord_mnl[1], k_tile_start)
                for k_tile in range(0, k_tile_iter_cnt - 1, 1, unroll=2):
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(
                                mainloop_consumer_state)
                            mainloop_consumer_state.advance()
                            peek_ab_full_status = cutlass.Boolean(1)
                            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                mainloop_consumer_state)
                            tCsA_p = tCsA_copy_view[
                                None, None, None,
                                mainloop_consumer_state.index]
                            tCsB_p = tCsB_copy_view[
                                None, None, None,
                                mainloop_consumer_state.index]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status)
                        cute.gemm(tiled_mma, accumulators, tCrA[None, None,
                                                                k_block_idx],
                                  tCrB[None, None, k_block_idx], accumulators)
                        if cutlass.const_expr(self.a16_block_transposed):
                            cute.copy(smem_tiled_copy_B, tCsB_p[None, None,
                                                                k_block_next],
                                      tCrB_copy_view[None, None, k_block_next])
                        else:
                            cute.copy(smem_tiled_copy_A, tCsA_p[None, None,
                                                                k_block_next],
                                      tCrA_copy_view[None, None, k_block_next])
                        self._load_a16_b_fragment(
                            a16_weight_fragments[None, None, k_block_next],
                            a16_b_coordinates[None, None,
                                              k_block_next], sBPacked, sSFB,
                            a16_lut, mainloop_consumer_state.index,
                            tile_coord_mnl[1], k_tile_start + Int32(k_tile) +
                            (1 if k_block_next == 0 else 0))
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    if k_block_idx == num_k_blocks - 1:
                        mainloop_pipeline.consumer_release(
                            mainloop_consumer_state)
                        mainloop_consumer_state.advance()
                    if k_block_next > 0:
                        if cutlass.const_expr(self.a16_block_transposed):
                            cute.copy(smem_tiled_copy_B, tCsB_p[None, None,
                                                                k_block_next],
                                      tCrB_copy_view[None, None, k_block_next])
                        else:
                            cute.copy(smem_tiled_copy_A, tCsA_p[None, None,
                                                                k_block_next],
                                      tCrA_copy_view[None, None, k_block_next])
                        self._load_a16_b_fragment(
                            a16_weight_fragments[None, None, k_block_next],
                            a16_b_coordinates[None, None, k_block_next],
                            sBPacked, sSFB, a16_lut,
                            mainloop_consumer_state.index, tile_coord_mnl[1],
                            k_tile_start + Int32(k_tile_iter_cnt - 1))
                    cute.gemm(tiled_mma, accumulators, tCrA[None, None,
                                                            k_block_idx],
                              tCrB[None, None, k_block_idx], accumulators)
                if cutlass.const_expr(self.swap_ab):
                    for acc_m in cutlass.range_constexpr(
                            cute.size(swap_ab_acc_mn.shape[0])):
                        for acc_n in cutlass.range_constexpr(
                                cute.size(swap_ab_acc_mn.shape[1])):
                            coord = swap_ab_coord_mn[acc_m, acc_n]
                            m_coord = tile_coord_mnl[0] * Int32(
                                self.tile_shape_mnk[0]) + coord[1]
                            n_coord = tile_coord_mnl[1] * Int32(
                                self.tile_shape_mnk[1]) + coord[0]
                            if m_coord < Int32(
                                    directC_mnl.shape[0]) and n_coord < Int32(
                                        directC_mnl.shape[1]):
                                directC_mnl[m_coord, n_coord,
                                            tile_coord_mnl[2]] = epilogue_op(
                                                (alpha_value *
                                                 swap_ab_acc_mn[acc_m, acc_n]
                                                 ).to(self.c_dtype))
                    work_tile = WorkTileInfo(work_tile.tile_idx,
                                             cutlass.Boolean(0))
                if cutlass.const_expr(not self.swap_ab):
                    _is_m_major = self.c_layout.is_m_major_c()
                    if cutlass.const_expr(self.c_dtype.width == 16):
                        copy_atom_r2s = cute.make_copy_atom(
                            cute.nvgpu.warp.StMatrix8x8x16bOp(_is_m_major, 2),
                            self.c_dtype)
                    else:
                        copy_atom_r2s = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(), self.c_dtype)
                    if cutlass.const_expr(self.c_dtype.width == 16):
                        copy_atom_C = cute.make_copy_atom(
                            cute.nvgpu.warp.StMatrix8x8x16bOp(
                                self.c_layout.is_m_major_c(), 2), self.c_dtype)
                    else:
                        copy_atom_C = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(), self.c_dtype)
                    tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(
                        copy_atom_C, tiled_mma)
                    tiled_copy_r2s = cute.make_tiled_copy_S(
                        copy_atom_r2s, tiled_copy_C_Atom)
                    thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                    tRS_sD = thr_copy_r2s.partition_D(sC)
                    tRS_rAcc = tiled_copy_r2s.retile(accumulators)
                    rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                    tRS_rD_layout = cute.make_layout(rD_shape[:3])
                    tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape,
                                                   self.acc_dtype)
                    sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
                    tcgc_for_tma_partition = cute.zipped_divide(
                        gC_mnl_slice, self.epi_tile)
                    bSG_sD, bSG_gD = cpasync.tma_partition(
                        tma_atom_c, 0, cute.make_layout(1),
                        sepi_for_tma_partition, tcgc_for_tma_partition)
                    epi_rest_m = bSG_gD.shape[1][0]
                    epi_rest_n = bSG_gD.shape[1][1]
                    epi_tile_m = self.epi_tile[0]
                    epi_tile_n = self.epi_tile[1]
                    mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rAcc,
                                                                     mode=[1])
                    mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rAcc,
                                                                     mode=[2])
                    has_multi_epi_store = cutlass.const_expr(not (
                        self.epi_stage == 1 and epi_rest_m == 1 and
                        (epi_rest_n == 1)))
                    tma_store_producer_group = pipeline.CooperativeGroup(
                        pipeline.Agent.Thread,
                        self.num_mma_warps * self.num_threads_per_warp)
                    tma_store_pipeline = pipeline.PipelineTmaStore.create(
                        num_stages=self.epi_stage,
                        producer_group=tma_store_producer_group)
                    for epi_m in cutlass.range_constexpr(epi_rest_m):
                        for epi_n in cutlass.range_constexpr(epi_rest_n):
                            MmaMPerEpiM = epi_tile_m // mma_tile_m
                            MmaNPerEpiN = epi_tile_n // mma_tile_n
                            for mma_n_in_epi in cutlass.range_constexpr(
                                    MmaNPerEpiN):
                                for mma_m_in_epi in cutlass.range_constexpr(
                                        MmaMPerEpiM):
                                    mma_n = epi_n * MmaNPerEpiN + mma_n_in_epi
                                    mma_m = epi_m * MmaMPerEpiM + mma_m_in_epi
                                    tRS_rD_slice = tRS_rD[None, mma_m_in_epi,
                                                          mma_n_in_epi]
                                    tRS_rAcc_slice = tRS_rAcc[None, mma_m,
                                                              mma_n]
                                    for elem_idx in cutlass.range_constexpr(
                                            cute.size(tRS_rD_slice)):
                                        tRS_rD_slice[
                                            elem_idx] = tRS_rAcc_slice[
                                                elem_idx]
                            gmem_coord = (epi_m, epi_n)
                            if cutlass.const_expr(self.split_k_slices > 1):
                                split_idx = Int32(block_idx[1])
                                for acc_m in cutlass.range_constexpr(
                                        cute.size(split_k_acc_mn.shape[0])):
                                    for acc_n in cutlass.range_constexpr(
                                            cute.size(
                                                split_k_acc_mn.shape[1])):
                                        coord = split_k_coord_mn[acc_m, acc_n]
                                        m_coord = tile_coord_mnl[0] * Int32(
                                            self.tile_shape_mnk[0]) + coord[0]
                                        n_coord = tile_coord_mnl[1] * Int32(
                                            self.tile_shape_mnk[1]) + coord[1]
                                        if m_coord < Int32(
                                                directC_mnl.shape[0]
                                        ) and n_coord < Int32(
                                                directC_mnl.shape[1]):
                                            directC_mnl[
                                                m_coord, n_coord,
                                                split_idx] = alpha_value * split_k_acc_mn[
                                                    acc_m, acc_n]
                            else:
                                tRS_rD_out = cute.make_rmem_tensor(
                                    tRS_rD_layout.shape, self.c_dtype)
                                acc_vec = tRS_rD.load()
                                acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                                tRS_rD_out.store(acc_vec)
                                epi_buffer = (epi_m * epi_rest_n +
                                              epi_n) % cute.size(tRS_sD,
                                                                 mode=[3])
                                if has_multi_epi_store:
                                    self.epilog_sync_barrier.arrive_and_wait()
                                cute.copy(tiled_copy_r2s, tRS_rD_out,
                                          tRS_sD[None, None, None, epi_buffer])
                                cute.arch.fence_proxy('async.shared',
                                                      space='cta')
                                self.epilog_sync_barrier.arrive_and_wait()
                                if warp_idx == 0:
                                    cute.copy(tma_atom_c, bSG_sD[None,
                                                                 epi_buffer],
                                              bSG_gD[None, gmem_coord])
                                    if has_multi_epi_store:
                                        tma_store_pipeline.producer_commit()
                                        tma_store_pipeline.producer_acquire()
                    work_tile = WorkTileInfo(work_tile.tile_idx,
                                             cutlass.Boolean(0))
                    if has_multi_epi_store and cutlass.const_expr(
                            self.split_k_slices == 1):
                        tma_store_pipeline.producer_tail()
        elif warp_idx == self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[None, tile_coord_mnl[0], None,
                                tile_coord_mnl[2]]
                tBgB_nkl = tBgB[None, tile_coord_mnl[1], None,
                                tile_coord_mnl[2]]
                mainloop_producer_state.reset_count()
                for _k_tile in range(0, k_tile_iter_cnt, 1, unroll=2):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    k_tile_global = k_tile_start + mainloop_producer_state.count
                    tBgB_k = tBgB_nkl[None, k_tile_global]
                    tBsB_pipe = tBsB[None, mainloop_producer_state.index]
                    tAgA_k = tAgA_mkl[None, k_tile_global]
                    tAsA_pipe = tAsA[None, mainloop_producer_state.index]
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state))
                    if Int32(tidx % self.num_threads_per_warp) == 0:
                        logical_k = cute.size(directB_nkl, mode=[1]) * 4
                        size_n = cute.size(directB_nkl, mode=[0])
                        base_n = Int64(tile_coord_mnl[1]
                                       ) * self.tile_shape_mnk[1] // 128 * 128
                        block_k = 256
                        barrier = shared_ptr_to_u32(
                            mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state))
                        for unit in cutlass.range_constexpr(
                                self.a16_scale_bytes // 256):
                            base_k = Int64(
                                k_tile_global
                            ) * self.tile_shape_mnk[2] // block_k + unit
                            if base_k >= logical_k // block_k:
                                base_k = Int64(logical_k // block_k - 1)
                            dst = shared_ptr_to_u32(
                                elem_pointer(sSFB,
                                             (unit * 256, 0,
                                              mainloop_producer_state.index)))
                            base_chunk = 0
                            base_row = Int64(0)
                            if base_n + 128 <= size_n:
                                cp_async_bulk_g2s_mbar(
                                    dst,
                                    get_ptr_as_int64(
                                        directSFB_nkl,
                                        (base_k * size_n + base_n) * 2),
                                    Int32(256), barrier)
                            else:
                                for base_chunk in cutlass.range_constexpr(16):
                                    base_row = base_n + Int64(base_chunk * 8)
                                    if base_row >= size_n:
                                        base_row = Int64(0)
                                    cp_async_bulk_g2s_mbar(
                                        dst + Int32(base_chunk * 16),
                                        get_ptr_as_int64(
                                            directSFB_nkl,
                                            (base_k * size_n + base_row) * 2),
                                        Int32(16), barrier)
                    if cutlass.const_expr(
                            self.tile_shape_mnk[2] == 256
                            and self.tile_shape_mnk[1] in (64, 128)
                            and (self.descriptor_tile_n == 128)):
                        if Int32(tidx % self.num_threads_per_warp) == 0:
                            size_n = cute.size(directB_nkl, mode=[0])
                            storage_k = cute.size(directB_nkl, mode=[1])
                            n_start = Int64(
                                tile_coord_mnl[1]) * self.tile_shape_mnk[1]
                            descriptor_offset = Int64(
                                tile_coord_mnl[2]) * size_n * storage_k + (
                                    n_start // 128 *
                                    (storage_k // 64) + Int64(k_tile_global)
                                ) * 8192 + n_start % 128 * 64
                            cp_async_bulk_g2s_mbar(
                                shared_ptr_to_u32(
                                    elem_pointer(
                                        sBPacked,
                                        (0, 0,
                                         mainloop_producer_state.index))),
                                get_ptr_as_int64(directB_nkl,
                                                 descriptor_offset),
                                Int32(self.tile_shape_mnk[1] * 64),
                                shared_ptr_to_u32(
                                    mainloop_pipeline.producer_get_barrier(
                                        mainloop_producer_state)))
                    elif cutlass.const_expr(self.occupancy > 1):
                        cute.copy(
                            tma_atom_b,
                            tBgB_k,
                            tBsB_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state),
                            cache_policy=Int64(1364590687093260288))
                    else:
                        cute.copy(
                            tma_atom_b,
                            tBgB_k,
                            tBsB_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state))
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()
                work_tile = WorkTileInfo(work_tile.tile_idx,
                                         cutlass.Boolean(0))
            mainloop_pipeline.producer_tail(mainloop_producer_state)
        return

    @staticmethod
    def _compute_stages(tile_shape_mnk: tuple,
                        a_dtype,
                        b_dtype,
                        sf_dtype,
                        sfa_smem_layout,
                        sfb_smem_layout,
                        epi_tile: tuple,
                        c_dtype,
                        smem_capacity: int,
                        occupancy: int,
                        b_packed: bool = False,
                        epi_stage_cap: int = 0,
                        decode_stage3: bool = False,
                        b_storage_bits: int | None = None,
                        minimum_ab_stage: int = 1) -> tuple:
        epi_stage_max = tile_shape_mnk[1] // epi_tile[1] * (
            tile_shape_mnk[0] // epi_tile[0])
        epi_stage = min(epi_stage_max, 4)
        if epi_stage_cap:
            epi_stage = max(1, min(epi_stage, epi_stage_cap))
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage
        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = cute.size(
            a_shape) * a_dtype.width // 8 + cute.size(b_shape) * (
                b_dtype.width
                if b_storage_bits is None else b_storage_bits) // 8
        sf_bytes_per_stage = cute.size(
            cute.filter_zeros(sfa_smem_layout).shape
        ) * sf_dtype.width // 8 + cute.size(
            cute.filter_zeros(sfb_smem_layout).shape) * sf_dtype.width // 8
        mbar_helpers_bytes = 1024
        raw_ab_stage = ((smem_capacity - occupancy * 1024) // occupancy -
                        mbar_helpers_bytes -
                        epi_bytes) // (ab_bytes_per_stage + sf_bytes_per_stage)
        ab_stage = max(minimum_ab_stage, min(raw_ab_stage, 4))
        if tile_shape_mnk[0] in (16, 64) and tile_shape_mnk[1] == 128:
            ab_stage = max(minimum_ab_stage, min(raw_ab_stage, 5))
        if b_packed:
            ab_stage = max(minimum_ab_stage, min(raw_ab_stage, 5))
        if decode_stage3 and occupancy >= 2 and (tile_shape_mnk[0] <= 16):
            ab_stage = max(minimum_ab_stage, min(raw_ab_stage, 3))
        return (ab_stage, epi_stage)

    @staticmethod
    def _make_smem_layouts(tile_shape_mnk: tuple,
                           epi_tile: tuple,
                           a_dtype,
                           a_layout,
                           b_dtype,
                           b_layout,
                           ab_stage: int,
                           c_dtype,
                           c_layout,
                           epi_stage: int,
                           sf_vec_size: int,
                           tiled_mma,
                           block_fp8: bool = False) -> tuple:
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        a_is_k_major = a_layout.is_k_major_a()
        b_is_k_major = b_layout.is_k_major_b()
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(a_layout, a_dtype,
                                            a_major_mode_size), a_dtype)
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2))
        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(b_layout, b_dtype,
                                            b_major_mode_size), b_dtype)
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2))
        sfa_smem_layout_staged = cute.make_layout((1, 1, ab_stage))
        sfb_smem_layout_staged = cute.make_layout((1, 1, ab_stage))
        c_smem_shape = epi_tile
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c(
        ) else epi_tile[0]
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(c_layout, c_dtype,
                                            c_major_mode_size), c_dtype)
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(c_smem_shape, epi_stage),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2))
        return (a_smem_layout_staged, b_smem_layout_staged,
                sfa_smem_layout_staged, sfb_smem_layout_staged,
                epi_smem_layout_staged)

    @staticmethod
    def _compute_grid(c,
                      tile_shape_mnk: tuple,
                      max_active_clusters,
                      split_k_slices: int,
                      large_m_unroll: bool,
                      split_k_all_m: bool = False) -> tuple:
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[0, (None, None, None)].shape
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl,
            cluster_shape_mnl,
            swizzle_size=16 if tile_shape_mnk == (128, 128, 64) and
            (not large_m_unroll) else 1)
        if cutlass.const_expr(split_k_slices > 1 or split_k_all_m):
            grid = (num_ctas_mnl[0] if split_k_all_m else 1, split_k_slices,
                    num_ctas_mnl[1])
        else:
            grid = utils.StaticPersistentTileScheduler.get_grid_shape(
                tile_sched_params, max_active_clusters)
        return (tile_sched_params, grid)

    @staticmethod
    def _make_tma_store_atoms_and_tensors(tensor_c, epi_smem_layout_staged,
                                          epi_tile: tuple) -> tuple:
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), tensor_c, epi_smem_layout,
            epi_tile)
        return (tma_atom_c, tma_tensor_c)

    @staticmethod
    def _make_tma_atoms_and_tensors(tensor,
                                    smem_layout_staged,
                                    smem_tile: tuple,
                                    mcast_dim: int,
                                    internal_type=None) -> tuple:
        op = cpasync.CopyBulkTensorTileG2SOp(
        ) if mcast_dim == 1 else cpasync.CopyBulkTensorTileG2SMulticastOp()
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
            internal_type=internal_type)
        return (tma_atom, tma_tensor)
