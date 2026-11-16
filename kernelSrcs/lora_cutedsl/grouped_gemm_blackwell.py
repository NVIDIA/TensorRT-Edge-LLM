# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Adapted from NVIDIA CUTLASS v4.4.2
# examples/python/CuTeDSL/blackwell/grouped_gemm.py.
"""Blackwell datacenter FP16 grouped GEMM kernel for LoRA.

The device-only workload is D_i = A_i @ B_i.T for a runtime group count. The
optional LoRA epilogue computes D_i = C_i + scale_i * FP16(A_i @ B_i.T).
Problem shapes, element strides, A/B/D addresses, and per-group scales remain
resident on the device; empty groups are skipped by the persistent grouped-tile
scheduler. The kernel uses TMA, tcgen05 MMA with FP32 accumulation, and TMEM on
SM100/101/103/110.
"""

import argparse
from inspect import isclass
import os
import pathlib
import sys
import time
from typing import Type, Union

_saved_argv = None
if __name__ == "__main__":
    _saved_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]

_KERNEL_SRCS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_KERNEL_SRCS_DIR))

import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import cutlass.cute as cute
import cutlass.cute.runtime as cute_runtime
import cutlass.cute.testing as testing
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutedsl_utils import aot_placeholders
import numpy as np


MAX_NUM_GROUPS = 256
DESCRIPTOR_ALIGNMENT = 16
TENSORMAP_ALIGNMENT = 128
BYTES_PER_TENSORMAP = 128
TENSORMAPS_PER_BLOCK = 3


class LoraGroupedGemmBlackwell:
    """Persistent FP16 grouped GEMM for Blackwell datacenter GPUs.

    Version one intentionally fixes the UMMA tile to 64x128, uses one-CTA
    instructions and a (1, 1) cluster. The only runtime-varying dimensions and
    addresses are read from the device metadata tensors.
    """

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        use_2cta_instrs: bool = False,
        mma_tiler_mn: tuple[int, int] = (64, 128),
        cluster_shape_mn: tuple[int, int] = (1, 1),
        expected_b_layout: utils.LayoutEnum = utils.LayoutEnum.COL_MAJOR,
        tensormap_update_mode: utils.TensorMapUpdateMode = utils.
        TensorMapUpdateMode.SMEM,
    ):
        """Initializes the configuration for a Blackwell grouped GEMM kernel.

        Besides configurations for dense persistent GEMM, there is an extra config specific to grouped GEMM:

        Tensormap Update Mode:
        - tensormap_update_mode: Specifies whether the tensormap is
            updated in global memory(GMEM) or shared memory(SMEM).
           The 2 modes are functionally equivalent and the difference are:
            - We buffer three 128-byte tensor maps in SMEM for A, B, and D
              when tensor-map updates are performed in SMEM.
            - Performance varies between modes depending on problem size; optimal choice differs across workloads.

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param use_2cta_instrs: Boolean, True to use cta_group=2 MMA variant.
        :type use_2cta_instrs: bool
        :param mma_tiler_mn: tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: tuple[int, int]
        :param cluster_shape_mn: tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: tuple[int, int]
        :param expected_b_layout: Required layout for matrix B, defaults to column-major.
        :type expected_b_layout: utils.LayoutEnum, optional
        :param tensormap_update_mode: Mode for updating the tensormap (GMEM or SMEM), defaults to SMEM.
        :type tensormap_update_mode: utils.TensorMapUpdateMode, optional
        """
        if acc_dtype != cutlass.Float32:
            raise ValueError(
                "Blackwell grouped GEMM requires FP32 accumulation")
        if use_2cta_instrs:
            raise ValueError("Blackwell grouped GEMM requires one-CTA UMMA")
        if mma_tiler_mn != (64, 128):
            raise ValueError(
                "Blackwell grouped GEMM requires mma_tiler_mn=(64, 128)")
        if cluster_shape_mn != (1, 1):
            raise ValueError(
                "Blackwell grouped GEMM requires cluster_shape_mn=(1, 1)")
        if expected_b_layout not in (utils.LayoutEnum.ROW_MAJOR,
                                     utils.LayoutEnum.COL_MAJOR):
            raise ValueError("Blackwell grouped GEMM requires a matrix B layout")
        if tensormap_update_mode != utils.TensorMapUpdateMode.SMEM:
            raise ValueError(
                "Blackwell grouped GEMM requires SMEM tensor-map updates")

        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.expected_b_layout = expected_b_layout
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.cta_group = (tcgen05.CtaGroup.TWO
                          if use_2cta_instrs else tcgen05.CtaGroup.ONE)

        self.tensormap_update_mode = tensormap_update_mode
        # Delegate tensormap ab initialization to MMA warp when SMEM mode is used for better latency hiding
        self.delegate_tensormap_ab_init = (
            tensormap_update_mode == utils.TensorMapUpdateMode.SMEM)

        self.num_mcast_ctas_a = 1
        self.num_mcast_ctas_b = 1
        self.is_a_mcast = False
        self.is_b_mcast = False

        self.occupancy = 1
        # Set specialized warp ids
        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id))
        # Set barrier for epilog sync, tmem ptr sync and tensormap update sync
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        # Barrier used by MMA/TMA warps to signal A/B tensormap initialization completion
        self.tensormap_ab_init_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * (len(self.epilog_warp_id) + 1),
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tma_load_bytes = 0

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        Most of the implementation follows standard dense GEMM patterns,
        with the key difference being additional consideration for SMEM
        buffer needed for tensormap updates.
        """
        # Configure tiled mma
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_tile_shape_mnk = tuple(
            x * y
            for x, y in zip(self.cta_tile_shape_mnk, (*self.cluster_shape_mn,
                                                      1)))

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape, ),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        self.epi_tile = utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_epi_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.smem_capacity,
            self.occupancy,
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.epi_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_epi_stage,
        )

        mbar_smem_bytes = self._get_mbar_smem_bytes(
            num_acc_stage=self.num_acc_stage,
            num_ab_stage=self.num_ab_stage,
            num_epi_stage=self.num_epi_stage,
        )
        tensormap_smem_bytes = self._get_tensormap_smem_bytes(
            self.tensormap_update_mode)
        if (mbar_smem_bytes + tensormap_smem_bytes +
                LoraGroupedGemmBlackwell.tensor_memory_management_bytes
                > self.reserved_smem_bytes):
            raise ValueError(
                f"smem consumption for mbar and tensormap {mbar_smem_bytes + tensormap_smem_bytes} exceeds the "
                f"reserved smem bytes {self.reserved_smem_bytes}")

        # Compute the number of tensor memory allocation columns
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage)

    @cute.jit
    def __call__(
        self,
        initial_a: cute.Tensor,
        initial_b: cute.Tensor,
        initial_d: cute.Tensor,
        problem_shapes: cute.Tensor,
        strides: cute.Tensor,
        addresses: cute.Tensor,
        logical_k_sizes: cute.Tensor,
        group_scales: cute.Tensor,
        tensormap_scratch: cute.Tensor,
        group_count: cutlass.Int32,
        max_active_clusters: cutlass.Int32,
        fuse_lora_epilogue: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        For grouped GEMM, tensor shapes, tensor strides, and tensor address are all provided
        by different tensors in global memory. The "initial" tensors only carry data type and
        majorness information.

        :param initial_a: Initial tensor A, used for data type and majorness information.
        :type initial_a: cute.Tensor
        :param initial_b: Initial tensor B, used for data type and majorness information.
        :type initial_b: cute.Tensor
        :param initial_d: Initial tensor D, used for data type and majorness information.
        :type initial_d: cute.Tensor
        :param group_count: Runtime ABI value: the actual group count. The
            cubin is runtime-polymorphic in this dimension; callers bound it
            by export_common.MAX_NUM_EXPERTS (descriptor trace sizing).
        :type group_count: cutlass.Int32
        :param problem_shapes: Device tensor ``[MAX_NUM_EXPERTS, 4]`` containing (M, N, K, L).
        :type problem_shapes: cute.Tensor
        :param strides: Device tensor ``[MAX_NUM_EXPERTS, 3, 2]`` containing A/B/D strides.
        :type strides: cute.Tensor
        :param addresses: Device tensor ``[MAX_NUM_EXPERTS, 3]`` containing A/B/D addresses.
        :type addresses: cute.Tensor
        :param logical_k_sizes: Device tensor ``[MAX_NUM_EXPERTS]`` containing the
            valid K extent used to mask aligned padding in TMA loads.
        :type logical_k_sizes: cute.Tensor
        :param group_scales: Device tensor ``[MAX_NUM_EXPERTS]`` containing the
            FP32 scale applied by the optional LoRA epilogue.
        :type group_scales: cute.Tensor
        :param tensormap_scratch: Device tensor ``[max_active_clusters, 3, 128]``
            of UInt8 values used for per-CTA tensor-map descriptors.
        :type tensormap_scratch: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Int32
        :param fuse_lora_epilogue: Nonzero to round the GEMM result to FP16,
            scale it, and accumulate it into the existing output in FP32.
        :type fuse_lora_epilogue: cutlass.Int32
        :param stream: CUDA stream for asynchronous execution.
        :type stream: cuda.CUstream
        :raises TypeError: If A and B data types do not match.
        """

        self.a_dtype = initial_a.element_type
        self.b_dtype = initial_b.element_type
        self.c_dtype = initial_d.element_type

        a_layout = utils.LayoutEnum.from_tensor(initial_a)
        b_layout = utils.LayoutEnum.from_tensor(initial_b)
        self.c_layout = utils.LayoutEnum.from_tensor(initial_d)
        self.a_major_mode = a_layout.mma_major_mode()
        self.b_major_mode = b_layout.mma_major_mode()
        if cutlass.const_expr(self.a_dtype != cutlass.Float16):
            raise TypeError("Blackwell grouped GEMM requires FP16 A")
        if cutlass.const_expr(self.b_dtype != cutlass.Float16):
            raise TypeError("Blackwell grouped GEMM requires FP16 B")
        if cutlass.const_expr(self.c_dtype != cutlass.Float16):
            raise TypeError("Blackwell grouped GEMM requires FP16 D")
        if cutlass.const_expr(problem_shapes.element_type != cutlass.Int32):
            raise TypeError("problem_shapes must contain Int32 values")
        if cutlass.const_expr(strides.element_type != cutlass.Int32):
            raise TypeError("strides must contain Int32 values")
        if cutlass.const_expr(addresses.element_type != cutlass.Int64):
            raise TypeError("addresses must contain Int64 values")
        if cutlass.const_expr(logical_k_sizes.element_type != cutlass.Int32):
            raise TypeError("logical_k_sizes must contain Int32 values")
        if cutlass.const_expr(group_scales.element_type != cutlass.Float32):
            raise TypeError("group_scales must contain Float32 values")
        if cutlass.const_expr(tensormap_scratch.element_type not in
                              (cutlass.Int8, cutlass.Uint8)):
            raise TypeError(
                "tensormap_scratch must contain byte values, got "
                f"{tensormap_scratch.element_type}")
        if cutlass.const_expr(a_layout != utils.LayoutEnum.ROW_MAJOR):
            raise ValueError("Blackwell grouped GEMM requires row-major A")
        if cutlass.const_expr(b_layout != self.expected_b_layout):
            raise ValueError("Blackwell grouped GEMM received an unexpected B layout")
        if cutlass.const_expr(self.c_layout != utils.LayoutEnum.ROW_MAJOR):
            raise ValueError("Blackwell grouped GEMM requires row-major D")

        # Setup attributes that dependent on gemm inputs
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

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn,
                                                       tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged,
                                    (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            initial_a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn,
                                                       tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged,
                                    (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            initial_b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # Setup TMA store for C
        tma_atom_c = None
        tma_tensor_c = None
        epi_smem_layout = cute.slice_(self.epi_smem_layout_staged,
                                      (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            initial_d,
            epi_smem_layout,
            self.epi_tile,
        )

        self.tile_sched_params, grid = self._compute_grid(
            max_active_clusters, self.cluster_shape_mn, max_active_clusters)

        self.buffer_align_bytes = 1024
        self.size_tensormap_in_i64 = (
            0 if self.tensormap_update_mode == utils.TensorMapUpdateMode.GMEM
            else LoraGroupedGemmBlackwell.num_tensormaps *
            LoraGroupedGemmBlackwell.bytes_per_tensormap // 8)

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            tensormap_buffer: cute.struct.MemRange[cutlass.Int64,
                                                   self.size_tensormap_in_i64]
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                   self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                    self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                    self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                     self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.epi_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype,
                    cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype,
                    cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            group_count,
            problem_shapes,
            strides,
            addresses,
            logical_k_sizes,
            group_scales,
            tensormap_scratch,
            fuse_lora_epilogue,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        group_count: cutlass.Int32,
        problem_sizes_mnkl: cute.Tensor,
        strides: cute.Tensor,
        addresses: cute.Tensor,
        logical_k_sizes: cute.Tensor,
        group_scales: cute.Tensor,
        tensormaps: cute.Tensor,
        fuse_lora_epilogue: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the grouped GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coord inside cluster
        bid = cute.arch.block_idx()
        mma_tile_coord_v = bid[0] % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster)
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: tensormap buffer, a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tensormap_a_smem_ptr = None
        tensormap_b_smem_ptr = None
        tensormap_c_smem_ptr = None
        if cutlass.const_expr(
                self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM):
            tensormap_smem_ptr = storage.tensormap_buffer.data_ptr()
            tensormap_a_smem_ptr = tensormap_smem_ptr
            tensormap_b_smem_ptr = (
                tensormap_a_smem_ptr +
                LoraGroupedGemmBlackwell.bytes_per_tensormap // 8)
            tensormap_c_smem_ptr = (
                tensormap_b_smem_ptr +
                LoraGroupedGemmBlackwell.bytes_per_tensormap // 8)

        #  init barrier for loading A, B with TMA
        ab_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer)
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # Accumulator barrier init
        acc_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread)
        num_acc_consumer_threads = len(
            self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn,
                             is_relaxed=True)

        #
        # Setup smem tensor A/B/C
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(epi_smem_layout_staged.outer,
                                   swizzle=epi_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer,
                                   swizzle=a_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer,
                                   swizzle=b_smem_layout_staged.inner)

        #
        # Compute multicast mask for A/B buffer full and empty
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        ab_empty_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast
                              or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)
            ab_empty_mcast_mask = a_full_mcast_mask | b_full_mcast_mask
        if cutlass.const_expr(use_2cta_instrs):
            block_in_cluster_coord_vmnk_peer = (
                block_in_cluster_coord_vmnk[0] ^ 1,
                *block_in_cluster_coord_vmnk[1:],
            )
            a_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk,
                block_in_cluster_coord_vmnk_peer,
                mcast_mode=2)
            b_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk,
                block_in_cluster_coord_vmnk_peer,
                mcast_mode=1)
            ab_empty_mcast_mask = (
                a_full_mcast_mask_peer
                | b_full_mcast_mask_peer
                | cutlass.Int16(0 if ab_empty_mcast_mask is
                                None else ab_empty_mcast_mask))

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(mA_mkl,
                                 cute.slice_(self.mma_tiler, (None, 0, None)),
                                 (None, None, None))
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(mB_nkl,
                                 cute.slice_(self.mma_tiler, (0, None, None)),
                                 (None, None, None))
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(mC_mnl,
                                 cute.slice_(self.mma_tiler, (None, None, 0)),
                                 (None, None, None))

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        if cutlass.const_expr(fuse_lora_epilogue):
            cC_mnl = cute.make_identity_tensor(mC_mnl.layout.shape)
            gC_id_mnl = cute.local_tile(
                cC_mnl,
                cute.slice_(self.mma_tiler, (None, None, 0)),
                (None, None, None),
            )
            tCcC = thr_mma.partition_C(gC_id_mnl)

        #
        # Partition global/shared tensor for load A, B with TMA
        #
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage))

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # Get tensormap buffer address
        #
        grid_dim = cute.arch.grid_dim()
        tensormap_workspace_idx = (bid[2] * grid_dim[1] * grid_dim[0] +
                                   bid[1] * grid_dim[0] + bid[0])

        tensormap_manager = utils.TensorMapManager(
            self.tensormap_update_mode,
            LoraGroupedGemmBlackwell.bytes_per_tensormap,
        )
        tensormap_a_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 0, None)].iterator)
        tensormap_b_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 1, None)].iterator)
        tensormap_c_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 2, None)].iterator)
        # Setup tensormap initialization pointer based on the mode
        if cutlass.const_expr(
                self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM):
            tensormap_a_init_ptr = tensormap_a_smem_ptr
            tensormap_b_init_ptr = tensormap_b_smem_ptr
            tensormap_c_init_ptr = tensormap_c_smem_ptr
        else:
            tensormap_a_init_ptr = tensormap_a_ptr
            tensormap_b_init_ptr = tensormap_b_ptr
            tensormap_c_init_ptr = tensormap_c_ptr

        #
        # Persistent tile scheduling loop
        #
        # When the problem shapes are on device, we launch one CTA per SM.
        # The if condition later prevents the warps from extra CTAs from doing any work.
        tile_sched = utils.StaticPersistentGroupTileScheduler.create(
            tile_sched_params,
            bid,
            grid_dim,
            self.cluster_tile_shape_mnk,
            utils.create_initial_search_state(),
            group_count,
            problem_sizes_mnkl,
        )
        initial_work_tile_info = tile_sched.initial_work_tile_info()

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id and initial_work_tile_info.is_valid_tile:
            # Initialize tensormaps for A, B
            if cutlass.const_expr(not self.delegate_tensormap_ab_init):
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_a, tensormap_a_init_ptr, self.tma_warp_id)
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_b, tensormap_b_init_ptr, self.tma_warp_id)

            tensormap_init_done = cutlass.Boolean(False)
            # group index of last tile
            last_group_idx = cutlass.Int32(-1)

            work_tile = initial_work_tile_info
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage)

            while work_tile.is_valid_tile:
                grouped_gemm_cta_tile_info = work_tile.group_search_result

                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                is_k_tile_cnt_zero = cur_k_tile_cnt == 0
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                # Do not load any data if cur_k_tile_cnt is 0
                if not is_k_tile_cnt_zero:
                    is_group_changed = cur_group_idx != last_group_idx
                    # skip tensormap update if we're working on the same group
                    if is_group_changed:
                        logical_k = logical_k_sizes[cur_group_idx]
                        tensor_problem_shape = (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            logical_k,
                        )
                        real_tensor_a = self.make_tensor_for_tensormap_update(
                            cur_group_idx,
                            self.a_dtype,
                            tensor_problem_shape,
                            strides,
                            addresses,
                            0,  # 0 for tensor A
                        )
                        real_tensor_b = self.make_tensor_for_tensormap_update(
                            cur_group_idx,
                            self.b_dtype,
                            tensor_problem_shape,
                            strides,
                            addresses,
                            1,  # 1 for tensor B
                        )
                        # wait tensormap initialization complete before update
                        if not tensormap_init_done:
                            if cutlass.const_expr(
                                    self.delegate_tensormap_ab_init):
                                self.tensormap_ab_init_barrier.arrive_and_wait(
                                )
                            tensormap_manager.fence_tensormap_initialization()
                            tensormap_init_done = True

                        tensormap_manager.update_tensormap(
                            (real_tensor_a, real_tensor_b),
                            (tma_atom_a, tma_atom_b),
                            (tensormap_a_ptr, tensormap_b_ptr),
                            self.tma_warp_id,
                            (tensormap_a_smem_ptr, tensormap_b_smem_ptr),
                        )

                    mma_tile_coord_mnl = (
                        grouped_gemm_cta_tile_info.cta_tile_idx_m //
                        cute.size(tiled_mma.thr_id.shape),
                        grouped_gemm_cta_tile_info.cta_tile_idx_n,
                        0,
                    )

                    #
                    # Slice to per mma tile index
                    #
                    # ((atom_v, rest_v), RestK)
                    tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None,
                                       mma_tile_coord_mnl[2])]
                    # ((atom_v, rest_v), RestK)
                    tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None,
                                       mma_tile_coord_mnl[2])]

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                    ab_producer_state.reset_count()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < cur_k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state)
                    # ensure the update to tensormap has completed before using it
                    if is_group_changed:
                        tensormap_manager.fence_tensormap_update(
                            tensormap_a_ptr)
                        tensormap_manager.fence_tensormap_update(
                            tensormap_b_ptr)
                        #
                        # Tma load loop
                        #
                    for k_tile in cutlass.range(0, cur_k_tile_cnt, 1,
                                                unroll=1):
                        # Wait for AB buffer empty
                        ab_pipeline.producer_acquire(ab_producer_state,
                                                     peek_ab_empty_status)

                        # Load A/B with TMA
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, ab_producer_state.count)],
                            tAsA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                ab_producer_state),
                            mcast_mask=a_full_mcast_mask,
                            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                tensormap_a_ptr,
                                cute.AddressSpace.generic,
                            ),
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_slice[(None, ab_producer_state.count)],
                            tBsB[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                ab_producer_state),
                            mcast_mask=b_full_mcast_mask,
                            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                tensormap_b_ptr,
                                cute.AddressSpace.generic,
                            ),
                        )

                        # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                        ab_producer_state.advance()
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if ab_producer_state.count < cur_k_tile_cnt:
                            peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                ab_producer_state)
                else:
                    # If tensormap initialization is not done, wait for it to complete
                    if not tensormap_init_done:
                        if cutlass.const_expr(self.delegate_tensormap_ab_init):
                            self.tensormap_ab_init_barrier.arrive_and_wait()
                        tensormap_manager.fence_tensormap_initialization()
                        tensormap_init_done = True
                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx

            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id and initial_work_tile_info.is_valid_tile:
            #  Bar sync for retrieve tmem ptr from shared mem
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop
            #
            work_tile = initial_work_tile_info
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage)

            # tile count we have searched
            while work_tile.is_valid_tile:
                cur_group_idx = work_tile.group_search_result.group_idx
                problem_shape_k = work_tile.group_search_result.problem_shape_k

                # MMA warp is only interested in number of tiles along K dimension
                cur_k_tile_cnt = (problem_shape_k +
                                  self.cluster_tile_shape_mnk[2] -
                                  1) // self.cluster_tile_shape_mnk[2]
                is_k_tile_cnt_zero = cur_k_tile_cnt == 0

                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None,
                                      acc_producer_state.index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    if ab_consumer_state.count < cur_k_tile_cnt:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(
                            ab_consumer_state)

                    #
                    # Wait for accumulator buffer empty
                    #
                    if not is_k_tile_cnt_zero:
                        acc_pipeline.producer_acquire(acc_producer_state)

                    #
                    # Reset the ACCUMULATE field for each tile
                    #
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    #
                    # Mma mainloop
                    #
                    for k_tile in cutlass.range(0, cur_k_tile_cnt, 1,
                                                unroll=1):
                        # Wait for AB buffer full
                        ab_pipeline.consumer_wait(ab_consumer_state,
                                                  peek_ab_full_status)
                        # tCtAcc += tCrA * tCrB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks,
                                                        unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                        # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                        ab_consumer_state.advance()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if ab_consumer_state.count < cur_k_tile_cnt:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state)

                    #
                    # Async arrive accumulator buffer full
                    #
                    if not is_k_tile_cnt_zero:
                        acc_pipeline.producer_commit(acc_producer_state)
                        acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id and initial_work_tile_info.is_valid_tile:
            # initialize tensormap A, B for TMA warp
            if cutlass.const_expr(self.delegate_tensormap_ab_init):
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_a, tensormap_a_init_ptr, self.epilog_warp_id[0])
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_b, tensormap_b_init_ptr, self.epilog_warp_id[0])
                # signal tensormap initialization has finished
                self.tensormap_ab_init_barrier.arrive_and_wait()
            # initialize tensorap for C
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_c,
                tensormap_c_init_ptr,
                self.epilog_warp_id[0],
            )
            # Alloc tensor memory buffer
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            epi_tidx = tidx
            #
            # Partition for epilogue
            #
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(epi_tidx, tCtAcc_base,
                                                    tCgC, epi_tile,
                                                    use_2cta_instrs)

            if cutlass.const_expr(fuse_lora_epilogue):
                tCcC_epi = cute.flat_divide(
                    tCcC[((None, None), 0, 0, None, None, None)], epi_tile)
                tTR_cC_partitioned = tiled_copy_t2r.get_slice(
                    epi_tidx).partition_D(tCcC_epi)

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC)
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(tma_atom_c, tCgC, epi_tile,
                                                    sC)

            #
            # Persistent tile scheduling loop
            #

            work_tile = initial_work_tile_info

            # wait tensormap initialization complete before update
            tensormap_manager.fence_tensormap_initialization()
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage)
            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_epi_stage,
                producer_group=c_producer_group,
            )
            # group index of last tile
            last_group_idx = cutlass.Int32(-1)
            while work_tile.is_valid_tile:
                grouped_gemm_cta_tile_info = work_tile.group_search_result
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                is_k_tile_cnt_zero = cur_k_tile_cnt == 0
                is_group_changed = cur_group_idx != last_group_idx
                # We still need to store 0s when k_tile_cnt is 0
                if cutlass.const_expr(fuse_lora_epilogue):
                    real_tensor_c = self.make_tensor_for_tensormap_update(
                        cur_group_idx,
                        self.c_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        strides,
                        addresses,
                        2,
                    )
                    if is_group_changed:
                        tensormap_manager.update_tensormap(
                            ((real_tensor_c), ),
                            ((tma_atom_c), ),
                            ((tensormap_c_ptr), ),
                            self.epilog_warp_id[0],
                            (tensormap_c_smem_ptr, ),
                        )
                else:
                    if is_group_changed:
                        real_tensor_c = self.make_tensor_for_tensormap_update(
                            cur_group_idx,
                            self.c_dtype,
                            (
                                grouped_gemm_cta_tile_info.problem_shape_m,
                                grouped_gemm_cta_tile_info.problem_shape_n,
                                grouped_gemm_cta_tile_info.problem_shape_k,
                            ),
                            strides,
                            addresses,
                            2,
                        )
                        tensormap_manager.update_tensormap(
                            ((real_tensor_c), ),
                            ((tma_atom_c), ),
                            ((tensormap_c_ptr), ),
                            self.epilog_warp_id[0],
                            (tensormap_c_smem_ptr, ),
                        )

                mma_tile_coord_mnl = (
                    grouped_gemm_cta_tile_info.cta_tile_idx_m //
                    cute.size(tiled_mma.thr_id.shape),
                    grouped_gemm_cta_tile_info.cta_tile_idx_n,
                    0,
                )

                #
                # Slice to per mma tile index
                #
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[(
                    None,
                    None,
                    None,
                    *mma_tile_coord_mnl,
                )]
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None,
                                          acc_consumer_state.index)]
                #
                # Wait for accumulator buffer full
                #
                # Not waiting for accumulator buffer full when k_tile_cnt is 0
                if not is_k_tile_cnt_zero:
                    acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                if cutlass.const_expr(fuse_lora_epilogue):
                    real_gC_mnl = cute.local_tile(
                        real_tensor_c,
                        cute.slice_(self.mma_tiler, (None, None, 0)),
                        (None, None, None),
                    )
                    real_tCgC = thr_mma.partition_C(real_gC_mnl)
                    real_tCgC_epi = cute.flat_divide(
                        real_tCgC[((None, None), 0, 0, None, None, None)],
                        epi_tile,
                    )
                    tTR_gC = tiled_copy_t2r.get_slice(
                        epi_tidx).partition_D(real_tCgC_epi)
                    tTR_gC = tTR_gC[(None, None, None, None, None,
                                     *mma_tile_coord_mnl)]
                    tTR_gC = cute.group_modes(tTR_gC, 3,
                                              cute.rank(tTR_gC))
                    tTR_cC = tTR_cC_partitioned[
                        (None, None, None, None, None, *mma_tile_coord_mnl)]
                    tTR_cC = cute.group_modes(tTR_cC, 3,
                                              cute.rank(tTR_cC))
                    actual_m = grouped_gemm_cta_tile_info.problem_shape_m
                    actual_n = grouped_gemm_cta_tile_info.problem_shape_n
                    group_scale = cutlass.Float32(
                        group_scales[cur_group_idx])
                # ensure the update to tensormap has completed before using it
                if is_group_changed:
                    if warp_idx == self.epilog_warp_id[0]:
                        tensormap_manager.fence_tensormap_update(
                            tensormap_c_ptr)
                #
                # Store accumulator to global memory in subtiles
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                for subtile_idx in range(subtile_cnt):
                    #
                    # Store C to shared memory
                    #
                    epi_buffer = (num_prev_subtiles +
                                  subtile_idx) % self.num_epi_stage
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    if not is_k_tile_cnt_zero:
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                        if cutlass.const_expr(fuse_lora_epilogue):
                            output_slice = tTR_gC[(None, None, None,
                                                  subtile_idx)]
                            coord_slice = tTR_cC[(None, None, None,
                                                 subtile_idx)]
                            for i in cutlass.range(cute.size(tTR_rAcc)):
                                if (coord_slice[i][0] < actual_m
                                        and coord_slice[i][1] < actual_n):
                                    delta = cutlass.Float16(tTR_rAcc[i])
                                    tTR_rAcc[i] = (
                                        cutlass.Float32(output_slice[i]) +
                                        group_scale * cutlass.Float32(delta))

                        #
                        # Convert to output type
                        #
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        tRS_rC.store(acc_vec.to(self.c_dtype))
                    else:
                        tRS_rC.fill(0)
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, epi_buffer)],
                    )
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    self.epilog_sync_barrier.arrive_and_wait()
                    #
                    # store C to global memory with TMA
                    #
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, epi_buffer)],
                            bSG_gC[(None, subtile_idx)],
                            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                tensormap_c_ptr,
                                cute.AddressSpace.generic,
                            ),
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()
                #
                # Async arrive accumulator buffer empty
                #
                if not is_k_tile_cnt_zero:
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

            #
            # Wait for C store complete
            #
            c_pipeline.producer_tail()

    @cute.jit
    def make_tensor_for_tensormap_update(
        self,
        group_idx: cutlass.Int32,
        dtype: Type[cutlass.Numeric],
        problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        strides: cute.Tensor,
        addresses: cute.Tensor,
        tensor_index: int,
    ):
        """Extract stride and tensor address for a given group and construct a global tensor.

        This function is used within the kernel to dynamically create a CUTE tensor
        representing A, B, or D for the current group being processed, using the
        group-specific address, shape, and stride information.

        :param group_idx: The index of the current group within the grouped GEMM.
        :type group_idx: cutlass.Int32
        :param dtype: The data type of the tensor elements (e.g., cutlass.Float16).
        :type dtype: Type[cutlass.Numeric]
        :param problem_shape_mnk: The (M, N, K) problem shape for the current group.
        :type problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
        :param strides: Tensor containing strides for A, B, D for all groups.
            Layout: (group_count, 3, 2).
        :type strides: cute.Tensor
        :param addresses: Tensor containing global addresses for A, B, D.
            Layout: (group_count, 3).
        :type addresses: cute.Tensor
        :param tensor_index: Specifies which tensor to create: 0 for A, 1 for B, 2 for D.
        :type tensor_index: int
        :return: A CUTE tensor representing the requested global-memory tensor.
        :rtype: cute.Tensor
        :raises TypeError: If the provided dtype is not a subclass of cutlass.Numeric.
        """
        ptr_i64 = addresses[(group_idx, tensor_index)]
        if cutlass.const_expr(not isclass(dtype)
                              or not issubclass(dtype, cutlass.Numeric)):
            raise TypeError(
                f"dtype must be a type of cutlass.Numeric, got {type(dtype)}")
        tensor_gmem_ptr = cute.make_ptr(dtype,
                                        ptr_i64,
                                        cute.AddressSpace.gmem,
                                        assumed_align=16)

        strides_tensor_gmem = strides[(group_idx, tensor_index, None)]
        strides_tensor_reg = cute.make_rmem_tensor(
            cute.make_layout(2),
            strides.element_type,
        )
        cute.autovec_copy(strides_tensor_gmem, strides_tensor_reg)
        stride_mn = strides_tensor_reg[0]
        stride_k = strides_tensor_reg[1]
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)

        if cutlass.const_expr(tensor_index == 0):  # tensor A
            m = problem_shape_mnk[0]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, k, c1), stride=(stride_mn, stride_k, c0)),
            )
        elif cutlass.const_expr(tensor_index == 1):  # tensor B
            n = problem_shape_mnk[1]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((n, k, c1), stride=(stride_mn, stride_k, c0)),
            )
        else:  # tensor D
            m = problem_shape_mnk[0]
            n = problem_shape_mnk[1]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, n, c1), stride=(stride_mn, stride_k, c0)),
            )

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make a tiled copy for tensor-memory loads, then partition tensor
        memory (source) and the register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load(t2r)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make a tiled copy for shared-memory stores, then partition the
        register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rC: The partitioned tensor C (register source)
            - tRS_sC: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(self.c_layout,
                                                      self.c_dtype,
                                                      self.acc_dtype,
                                                      tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tma_atom_c: cute.CopyAtom,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to partition
        shared memory (source) and global memory (destination) for TMA store version.

        :param tma_atom_c: The TMA copy atom configured for storing tensor C.
        :type tma_atom_c: cute.CopyAtom
        :param gC_mnl: The global memory tensor C.
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler defining the granularity of the operation.
        :type epi_tile: cute.Tile
        :param sC: The shared memory epilogue buffer tensor.
        :type sC: cute.Tensor
        :return: A tuple containing:
                 - tma_atom_c: The input TMA copy atom (passed through).
                 - bSG_sC: The source shared memory tensor partitioned for the TMA operation.
                 - tCgC: The destination global memory tensor partitioned for the TMA operation.
        :rtype: tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int, int]:
        """Computes the number of stages for accumulator, A/B operands, and epilogue based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum of operand C in global memory.
        :type c_layout: utils.LayoutEnum
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (accumulator stages, A/B operand stages, epilogue stages)
        :rtype: tuple[int, int, int]
        """
        # Default accumulator and epilogue stages
        num_acc_stage = 2
        num_epi_stage = 2

        # Calculate smem layout and size for one stage of A, B, and Epilogue
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # stage=1
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # stage=1
        )
        epi_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,  # stage=1
        )
        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one) + cute.size_in_bytes(
                b_dtype, b_smem_layout_staged_one)

        epi_bytes_per_stage = cute.size_in_bytes(c_dtype,
                                                 epi_smem_layout_staged_one)
        epi_bytes = epi_bytes_per_stage * num_epi_stage

        # Calculate A/B stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial epilogue bytes
        # Divide remaining by bytes needed per A/B stage
        num_ab_stage = (smem_capacity // occupancy -
                        LoraGroupedGemmBlackwell.reserved_smem_bytes -
                        epi_bytes) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        remaining_smem = (
            smem_capacity - occupancy * ab_bytes_per_stage * num_ab_stage -
            occupancy *
            (LoraGroupedGemmBlackwell.reserved_smem_bytes + epi_bytes))
        num_epi_stage += remaining_smem // (occupancy * epi_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_epi_stage

    @staticmethod
    def _compute_grid(
        total_num_clusters: cutlass.Int32,
        cluster_shape_mn: tuple[int, int],
        max_active_clusters: cutlass.Int32,
    ) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
        """Compute tile scheduler parameters and grid shape for grouped GEMM operations.

        :param total_num_clusters: Total number of clusters to process across all groups.
        :type total_num_clusters: cutlass.Int32
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Int32

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: tuple[utils.PersistentTileSchedulerParams, tuple[int, ...]]
        """
        # Create problem shape with M, N dimensions from cluster shape
        # and L dimension representing the total number of clusters.
        problem_shape_ntile_mnl = (
            cluster_shape_mn[0],
            cluster_shape_mn[1],
            cutlass.Int32(total_num_clusters),
        )

        tile_sched_params = utils.PersistentTileSchedulerParams(
            problem_shape_ntile_mnl, (*cluster_shape_mn, 1))

        grid = utils.StaticPersistentGroupTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters)

        return tile_sched_params, grid

    @staticmethod
    def _get_mbar_smem_bytes(**kwargs_stages: int) -> int:
        """Calculate shared memory consumption for memory barriers based on provided stages.

        Each stage requires 2 barriers, and each barrier consumes 8 bytes of shared memory.
        The total consumption is the sum across all provided stages. This function calculates the total
        shared memory needed for these barriers.

        :param kwargs_stages: Variable keyword arguments where each key is a stage name
                              (e.g., num_acc_stage, num_ab_stage) and each value is the
                              number of stages of that type.
        :type kwargs_stages: int
        :return: Total shared memory bytes required for all memory barriers.
        :rtype: int
        """
        num_barriers_per_stage = 2
        num_bytes_per_barrier = 8
        mbar_smem_consumption = sum([
            num_barriers_per_stage * num_bytes_per_barrier * stage
            for stage in kwargs_stages.values()
        ])
        return mbar_smem_consumption

    @staticmethod
    def _get_tensormap_smem_bytes(
        tensormap_update_mode: utils.TensorMapUpdateMode, ) -> int:
        """Get the SMEM consumption for the tensormap buffer based on the update mode.

        :param tensormap_update_mode: Specifies whether tensormaps are updated in GMEM or SMEM.
        :type tensormap_update_mode: utils.TensorMapUpdateMode
        :return: The shared memory bytes required for the tensormap buffer. Returns 0 if mode is GMEM.
        :rtype: int
        :raises ValueError: If an invalid tensormap update mode is provided.
        """
        if tensormap_update_mode == utils.TensorMapUpdateMode.GMEM:
            return 0
        elif tensormap_update_mode == utils.TensorMapUpdateMode.SMEM:
            return (LoraGroupedGemmBlackwell.bytes_per_tensormap *
                    LoraGroupedGemmBlackwell.num_tensormaps)
        else:
            raise ValueError(
                f"Invalid tensormap update mode: {tensormap_update_mode}")

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: tuple[int, int, int],
        num_acc_stage: int,
    ) -> int:
        """
        Compute the number of tensor memory allocation columns.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler: The shape (M, N, K) of the MMA tile.
        :type mma_tiler: tuple[int, int, int]
        :param acc_stage: The stage of the accumulator tensor.
        :type acc_stage: int

        :return: The number of tensor memory allocation columns.
        :rtype: int
        """
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        return num_tmem_alloc_cols

    # Size of smem we reserved for mbarrier, tensor memory management and tensormap update
    reserved_smem_bytes = 1024
    bytes_per_tensormap = 128
    num_tensormaps = 3
    # size of smem used for tensor memory management
    tensor_memory_management_bytes = 12


class LoraGroupedGemmExportWrapper:

    def __init__(self, fuse_lora_epilogue=False):
        self.kernel = LoraGroupedGemmBlackwell()
        self.fuse_lora_epilogue = fuse_lora_epilogue

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        d_ptr: cute.Pointer,
        problem_shapes_ptr: cute.Pointer,
        strides_ptr: cute.Pointer,
        addresses_ptr: cute.Pointer,
        logical_k_sizes_ptr: cute.Pointer,
        group_scales_ptr: cute.Pointer,
        scratch_ptr: cute.Pointer,
        max_m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        group_count: cutlass.Int32,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ) -> cutlass.Int32:
        initial_a = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout((max_m, k, 1), order=(1, 0, 2)),
        )
        initial_b = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout((n, k, 1), order=(0, 1, 2)),
        )
        initial_d = cute.make_tensor(
            d_ptr,
            layout=cute.make_ordered_layout((max_m, n, 1), order=(1, 0, 2)),
        )
        problem_shapes = cute.make_tensor(
            problem_shapes_ptr,
            layout=cute.make_layout((MAX_NUM_GROUPS, 4), stride=(4, 1)),
        )
        strides = cute.make_tensor(
            strides_ptr,
            layout=cute.make_layout((MAX_NUM_GROUPS, 3, 2),
                                    stride=(6, 2, 1)),
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
        scratch = cute.make_tensor(
            cute.recast_ptr(scratch_ptr, dtype=cutlass.Uint8),
            layout=cute.make_layout(
                (MAX_NUM_GROUPS, TENSORMAPS_PER_BLOCK, BYTES_PER_TENSORMAP),
                stride=(
                    TENSORMAPS_PER_BLOCK * BYTES_PER_TENSORMAP,
                    BYTES_PER_TENSORMAP,
                    1,
                ),
            ),
        )
        self.kernel(
            initial_a,
            initial_b,
            initial_d,
            problem_shapes,
            strides,
            addresses,
            logical_k_sizes,
            group_scales,
            scratch,
            group_count,
            max_active_clusters,
            self.fuse_lora_epilogue,
            stream,
        )
        return cutlass.Int32(0)


def _make_pointer(data_type, address: int, assumed_align: int = DESCRIPTOR_ALIGNMENT):
    return cute_runtime.make_ptr(
        data_type, address, cute.AddressSpace.gmem, assumed_align=assumed_align)


def _make_export_pointer(data_type, assumed_align: int = DESCRIPTOR_ALIGNMENT):
    return aot_placeholders.make_ptr(data_type, assumed_align=assumed_align)


def _export(args: argparse.Namespace) -> None:
    pointers = (
        _make_export_pointer(cutlass.Float16),
        _make_export_pointer(cutlass.Float16),
        _make_export_pointer(cutlass.Float16),
        _make_export_pointer(cutlass.Int32),
        _make_export_pointer(cutlass.Int32),
        _make_export_pointer(cutlass.Int64),
        _make_export_pointer(cutlass.Int32),
        _make_export_pointer(cutlass.Float32),
        _make_export_pointer(cutlass.Uint8, TENSORMAP_ALIGNMENT),
    )
    start = time.time()
    compiled = cute.compile(
        LoraGroupedGemmExportWrapper(args.fuse_lora_epilogue).wrapper,
        *pointers,
        128,
        128,
        128,
        MAX_NUM_GROUPS,
        aot_placeholders.runtime_int32(),
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
    missing = [str(path) for path in artifacts if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(
            f"CuTeDSL export did not produce nonempty artifacts: {missing}")
    print(f"[{args.file_name}] Exported {artifacts[0]} and {artifacts[1]}")


def _create_runtime_problem(problem_sizes, max_active_clusters, stream):
    group_count = len(problem_sizes)
    shapes_host = np.zeros((MAX_NUM_GROUPS, 4), dtype=np.int32)
    strides_host = np.zeros((MAX_NUM_GROUPS, 3, 2), dtype=np.int32)
    addresses_host = np.zeros((MAX_NUM_GROUPS, 3), dtype=np.int64)
    logical_k_sizes_host = np.zeros((MAX_NUM_GROUPS, ), dtype=np.int32)
    group_scales_host = np.ones((MAX_NUM_GROUPS, ), dtype=np.float32)
    groups = []
    for index, (m, n, k) in enumerate(problem_sizes):
        a = cp.random.uniform(-0.5, 0.5, (m, k)).astype(cp.float16)
        b = cp.random.uniform(-0.5, 0.5, (k, n)).astype(cp.float16)
        d = cp.zeros((m, n), dtype=cp.float16)
        groups.append((a, b, d))
        shapes_host[index] = (m, n, k, 1)
        logical_k_sizes_host[index] = k
        strides_host[index] = ((k, 1), (1, n), (n, 1))
        addresses_host[index] = (a.data.ptr, b.data.ptr, d.data.ptr)

    shapes = cp.asarray(shapes_host)
    strides = cp.asarray(strides_host)
    addresses = cp.asarray(addresses_host)
    logical_k_sizes = cp.asarray(logical_k_sizes_host)
    group_scales = cp.asarray(group_scales_host)
    scratch = cp.empty(
        (MAX_NUM_GROUPS, TENSORMAPS_PER_BLOCK, BYTES_PER_TENSORMAP),
        dtype=cp.uint8,
    )
    first_a, first_b, first_d = groups[0]
    launch_args = (
        _make_pointer(cutlass.Float16, first_a.data.ptr),
        _make_pointer(cutlass.Float16, first_b.data.ptr),
        _make_pointer(cutlass.Float16, first_d.data.ptr),
        _make_pointer(cutlass.Int32, shapes.data.ptr),
        _make_pointer(cutlass.Int32, strides.data.ptr),
        _make_pointer(cutlass.Int64, addresses.data.ptr),
        _make_pointer(cutlass.Int32, logical_k_sizes.data.ptr),
        _make_pointer(cutlass.Float32, group_scales.data.ptr),
        _make_pointer(cutlass.Uint8, scratch.data.ptr, TENSORMAP_ALIGNMENT),
        cutlass.Int32(max(m for m, _, _ in problem_sizes)),
        cutlass.Int32(max(n for _, n, _ in problem_sizes)),
        cutlass.Int32(max(k for _, _, k in problem_sizes)),
        cutlass.Int32(group_count),
        cutlass.Int32(max_active_clusters),
        stream,
    )
    backing = [
        shapes, strides, addresses, logical_k_sizes, group_scales, scratch,
        groups
    ]
    return launch_args, backing, groups


def _run(args: argparse.Namespace) -> None:
    if args.export_only:
        _export(args)
        return

    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU is required unless --export_only is set")
    problem_sizes = args.problem_sizes_mnk
    if not problem_sizes or len(problem_sizes) > MAX_NUM_GROUPS:
        raise ValueError(f"problem count must be in [1, {MAX_NUM_GROUPS}]")
    if any(m <= 0 or n <= 0 or k <= 0 for m, n, k in problem_sizes):
        raise ValueError("all grouped GEMM dimensions must be positive")
    if any(n % 8 != 0 or k % 8 != 0 for _, n, k in problem_sizes):
        raise ValueError("FP16 grouped GEMM N and K dimensions must be multiples of 8")

    tag = f"[{args.file_name}]"
    print(f"{tag} Running {len(problem_sizes)} FP16 LoRA grouped GEMMs")
    for index, shape in enumerate(problem_sizes):
        print(f"{tag}   group {index}: M,N,K={shape}")
    print(f"{tag}   warmup={args.warmup_iterations}, iterations={args.iterations}, "
          f"skip_ref_check={args.skip_ref_check}, cold_l2={args.use_cold_l2}")

    cp.random.seed(args.seed)
    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(1)
    launch_args, backing, groups = _create_runtime_problem(
        problem_sizes, max_active_clusters, stream)
    start = time.time()
    compiled = cute.compile(LoraGroupedGemmExportWrapper(False).wrapper,
                            *launch_args)
    print(f"{tag} Compilation time: {time.time() - start:.3f}s")

    if not args.skip_ref_check:
        compiled(*launch_args)
        cp.cuda.Device().synchronize()
        for index, (a, b, d) in enumerate(groups):
            expected = cp.matmul(a.astype(cp.float32),
                                 b.astype(cp.float32)).astype(cp.float16)
            np.testing.assert_allclose(
                d.get(), expected.get(), atol=args.tolerance, rtol=1.0e-3,
                err_msg=f"group {index}")
        print(f"{tag} Reference check PASSED")

    def generate_arguments():
        runtime_args, runtime_backing, _ = _create_runtime_problem(
            problem_sizes, max_active_clusters, stream)
        jit_args = testing.JitArguments(*runtime_args)
        jit_args.add_to_scope(runtime_backing)
        return jit_args

    workspace_count = 1
    if args.use_cold_l2:
        one_workspace_bytes = sum(array.nbytes for array in backing[:-1])
        one_workspace_bytes += sum(array.nbytes for group in groups for array in group)
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, args.warmup_iterations, args.iterations)
    average_us = testing.benchmark(
        compiled,
        workspace_generator=generate_arguments,
        workspace_count=workspace_count,
        stream=stream,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
    )
    flops = 2 * sum(m * n * k for m, n, k in problem_sizes)
    tflops = flops / (average_us * 1.0e-6) / 1.0e12
    print(f"{tag} avg_time_us: {average_us:.2f} | {tflops:.3f} TFLOPS")


def _parse_problem_sizes(value: str):
    try:
        shapes = tuple(
            tuple(int(dimension) for dimension in group.split("x"))
            for group in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated MxNxK shapes") from error
    if any(len(shape) != 3 for shape in shapes):
        raise argparse.ArgumentTypeError(
            "expected comma-separated MxNxK shapes")
    return shapes


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blackwell FP16 LoRA grouped GEMM: test, benchmark, or AOT export.")
    parser.add_argument(
        "--problem_sizes_mnk", type=_parse_problem_sizes,
        default=((1, 8, 4096), (2, 16, 4096), (4, 32, 4096)),
        help="Comma-separated MxNxK groups used for standalone testing.")
    parser.add_argument("--tolerance", type=float, default=2.0e-2)
    parser.add_argument("--warmup_iterations", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--use_cold_l2", action="store_true")
    parser.add_argument("--export_only", action="store_true")
    parser.add_argument("--fuse_lora_epilogue", action="store_true")
    parser.add_argument("--output_dir", default="./lora_grouped_gemm_aot")
    parser.add_argument("--file_name", default="lora_blackwell_grouped_fp16")
    parser.add_argument("--function_prefix", default="lora_blackwell_grouped_fp16")
    return parser.parse_known_args(args=argv)[0]


def main() -> int:
    _run(_parsed_args)
    print("PASS")
    return 0


if __name__ == "__main__":
    _parsed_args = _parse_args(_saved_argv)
    sys.exit(main())
