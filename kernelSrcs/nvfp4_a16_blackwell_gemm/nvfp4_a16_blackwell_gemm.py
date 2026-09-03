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

"""SM110 dense NVFP4-weight FP16/BF16 GEMM using an RMEM-to-TMEM transform."""

import argparse
from math import ceil, log2
import os
from typing import Optional, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Int32, Uint8, Uint16, Uint32
from cutlass._mlir.dialects import arith, llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.runtime import make_ptr
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.mixed_input_helpers as mixed_input_utils
from cutlass.utils.mixed_input_helpers import TransformMode
from cutlass.cute.nvgpu import cpasync, tcgen05


@dsl_user_op
def cvt_e2m1x2_to_bf16x2_u32(packed_e2m1, *, loc=None, ip=None):
    """Convert low/high E2M1 nibbles to low/high BF16 halfwords."""
    src_i8 = Uint8(packed_e2m1).ir_value(loc=loc, ip=ip)
    src_i16 = arith.extui(Uint16.mlir_type, src_i8, loc=loc, ip=ip)
    return Uint32(
        llvm.inline_asm(
            Int32.mlir_type,
            [src_i16],
            """{
                .reg .b8 src;
                mov.b16 {src, _}, $1;
                cvt.rn.bf16x2.e2m1x2 $0, src;
            }""",
            "=r,h",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class Nvfp4A16BlackwellGemm:
    """
    Mixed-input GEMM kernel for NVIDIA Blackwell SM110 architecture.

    This kernel supports GEMM operations where input tensors A and B have different
    data types, with tensor A being transformed to the precision of tensor B before
    matrix multiplication.

    :param scale_granularity_m: Number of elements sharing the same scale factor along the M mode
    :type scale_granularity_m: int
    :param scale_granularity_k: Number of elements sharing the same scale factor along the K mode
    :type scale_granularity_k: int
    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param use_2cta_instrs: Whether to use CTA group 2 for advanced thread cooperation
    :type use_2cta_instrs: bool
    :param mma_tiler_mnk: Shape of the Matrix Multiply-Accumulate (MMA) tile (M, N, K)
    :type mma_tiler_mnk: tuple[int, int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: tuple[int, int]
    :param use_tma_store: Whether to use Tensor Memory Access (TMA) for storing results
    :type use_tma_store: bool
    :param shuffle_a: Whether to use shuffle intrinsic for int4-to-bf16 conversion
    :type shuffle_a: bool
    """

    def __init__(
        self,
        scale_granularity_m: int,
        scale_granularity_k: int,
        acc_dtype: type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        use_tma_store: bool,
        shuffle_a: bool,
    ):
        """
        Initializes the mixed-input GEMM kernel with a specified configuration.
        """
        # Scale granularity defines how many elements share the same scale factor
        # along the M and K modes.
        self.scale_granularity_m = scale_granularity_m
        self.scale_granularity_k = scale_granularity_k
        # Set transform mode
        if cutlass.const_expr(
            self.scale_granularity_m == 0 and self.scale_granularity_k == 0
        ):
            self.scale_mode = TransformMode.ConvertOnly
        else:
            self.scale_mode = TransformMode.ConvertScale
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = mma_tiler_mnk
        self.use_tma_store = use_tma_store
        self.shuffle_a = shuffle_a
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        # Set specialized warp ids
        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.scale_tma_warp_id = 6
        self.idle_warp_id = 7
        # 4 warps to do the transformation
        self.transform_warp_id = (
            8,
            9,
            10,
            11,
        )
        self.num_regs_epilogue_warps = 192
        self.num_regs_mma_warp = 96
        self.num_regs_tma_warps = 96
        self.num_regs_transform_warps = 208
        self.num_regs_idle_warp = 24
        self.threads_per_cta = 32 * (
            max(
                (
                    self.mma_warp_id,
                    self.tma_warp_id,
                    self.scale_tma_warp_id,
                    *self.epilog_warp_id,
                    *self.transform_warp_id,
                )
            )
            + 1
        )

        # Set barrier id for epilogue sync, tmem ptr sync, and transform sync
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            1, 32 * len(self.epilog_warp_id)
        )
        self.tmem_ptr_sync_barrier = pipeline.NamedBarrier(2, self.threads_per_cta)
        self.transform_sync_barrier = pipeline.NamedBarrier(
            3, 32 * len(self.transform_warp_id)
        )

        self.smem_buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Deduce where the transformed A tensor is stored
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue sub-tile
        - Setting up A/scale/B/C stage counts in shared memory
        - Setting up transformed A stage count in shared memory or tensor memory
        - Computing A/transformed A/scale/B/C memory layout
        - Computing tensor memory allocation columns
        """
        # Deduce where the transformed A tensor is stored, shared memory(SMEM) or tensor memory(TMEM)
        self.transform_a_source = mixed_input_utils.get_transform_a_source(
            self.a_major_mode
        )
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.mma_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
            self.transform_a_source,
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

        # Compute tensor memory(TMEM) columns and stages for each pipeline
        (
            self.num_load2trans_stage,
            self.num_scale_load2trans_stage,
            self.num_trans2mma_stage,
            self.num_acc_stage,
            self.num_c_stage,
            self.num_acc_tmem_cols,
            self.num_a_tmem_cols,
        ) = self._compute_stages_and_tmem_cols(
            tiled_mma,
            self.mma_tiler,
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_scale_dtype,
            self.b_dtype,
            self.c_dtype,
            self.c_layout,
            self.transform_a_source,
            self.scale_granularity_m,
            self.scale_granularity_k,
            self.smem_buffer_align_bytes,
            self.use_tma_store,
            self.scale_mode,
        )

        # Align TMEM columns for allocation
        # TMEM allocation requires power-of-2 column alignment
        # and must meet minimum allocation requirements
        self.num_tmem_alloc_cols = cute.round_up(
            self.num_acc_tmem_cols + self.num_a_tmem_cols,
            cute.arch.get_min_tmem_alloc_cols("sm_100"),
        )
        self.num_tmem_alloc_cols = 2 ** (ceil(log2(self.num_tmem_alloc_cols)))
        # Get smem layout for C tensor when TMA store is enabled
        self.c_smem_layout_staged = (
            sm100_utils.make_smem_layout_epi(
                self.c_dtype,
                self.c_layout,
                self.epi_tile,
                self.num_c_stage,
            )
            if self.use_tma_store
            else None
        )
        # Get smem layout for A, transformed A, and B
        (
            self.smem_layout_a,
            self.smem_layout_a_transform,
            self.smem_layout_b,
        ) = mixed_input_utils.compute_smem_layout(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.num_load2trans_stage,
            self.num_trans2mma_stage,
        )
        # Get smem layout for scale tensor
        self.smem_layout_scale_per_stage = None
        self.smem_layout_scale = None
        self.smem_layout_scale_words_per_stage = None
        self.smem_layout_scale_words = None
        if cutlass.const_expr(self.scale_mode == TransformMode.ConvertScale):
            self.scale_tile_shape = (
                (
                    cute.size(self.mma_tiler[0]) // 2
                    if self.use_2cta_instrs
                    else cute.size(self.mma_tiler[0])
                ),
                cute.size(self.mma_tiler[2]),
            )
            scale_rows = self.scale_tile_shape[0] // self.scale_granularity_m
            scale_groups = self.scale_tile_shape[1] // self.scale_granularity_k

            # The physical Blackwell tile stores four raw E4M3 bytes for each row.
            # Keep the M1/K16 broadcast modes for the transform-side partition.
            scale_outer = cute.make_layout(
                (
                    (self.scale_granularity_m, scale_rows),
                    (self.scale_granularity_k, scale_groups),
                ),
                stride=((0, scale_groups), (0, 1)),
            )
            self.smem_layout_scale_per_stage = cute.make_composed_layout(
                cute.make_swizzle(0, 4, 3), 0, scale_outer
            )
            self.smem_layout_scale = cute.append(
                self.smem_layout_scale_per_stage,
                cute.make_layout(
                    self.num_scale_load2trans_stage,
                    stride=cute.cosize(self.smem_layout_scale_per_stage.outer),
                ),
            )

            # TMA treats each physical [row, 4 scales] record as one opaque
            # 32-bit word.  This produces a naturally aligned 512-byte 1-D
            # transfer and avoids asking the tensor map to encode the K16
            # zero-stride broadcast.  Transform warps reinterpret the same SMEM
            # bytes through smem_layout_scale above.
            scale_words_outer = cute.make_layout(
                (scale_rows, 1), stride=(1, 0)
            )
            self.smem_layout_scale_words_per_stage = cute.make_composed_layout(
                cute.make_swizzle(0, 4, 3), 0, scale_words_outer
            )
            self.smem_layout_scale_words = cute.append(
                self.smem_layout_scale_words_per_stage,
                cute.make_layout(
                    self.num_scale_load2trans_stage,
                    stride=cute.cosize(
                        self.smem_layout_scale_words_per_stage.outer
                    ),
                ),
            )
            assert (
                cute.cosize(self.smem_layout_scale_per_stage.outer) == 512
                and cute.cosize(self.smem_layout_scale_words_per_stage.outer)
                * cutlass.Uint32.width
                // 8
                == 512
            ), "scale byte and word aliases must cover the same 512 bytes"

    def _validate_inputs(
        self,
        a: cute.Tensor,
        a_scale: Optional[cute.Tensor],
        b: cute.Tensor,
        c: cute.Tensor,
    ) -> None:
        """
        Validates input tensors and their properties.

        :param a: Input tensor A.
        :type a: cute.Tensor
        :param a_scale: Scale tensor for tensor A (None for ConvertOnly mode).
        :type a_scale: Optional[cute.Tensor]
        :param b: Input tensor B.
        :type b: cute.Tensor
        :param c: Output tensor C.
        :type c: cute.Tensor
        :raises ValueError: If inputs don't meet kernel requirements.
        """
        if cutlass.const_expr(self.scale_mode != TransformMode.ConvertScale):
            raise ValueError("NVFP4 weights require E4M3 block scales")
        if cutlass.const_expr(a.element_type is not cutlass.Float4E2M1FN):
            raise ValueError("weights must use packed Float4E2M1FN elements")
        if cutlass.const_expr(
            a_scale is None or a_scale.element_type is not cutlass.Float8E4M3FN
        ):
            raise ValueError("block scales must use raw Float8E4M3FN elements")
        if cutlass.const_expr(
            b.element_type not in (cutlass.Float16, cutlass.BFloat16)
            or c.element_type is not b.element_type
        ):
            raise ValueError("activation and output must have the same FP16/BF16 dtype")
        if cutlass.const_expr(
            self.scale_granularity_m != 1
            or self.scale_granularity_k != 16
            or self.mma_tiler[2] != 64
        ):
            raise ValueError("Blackwell opaque layout requires M1/K16 scales and MMA K64")

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        a_scale: Optional[cute.Tensor],  # None for ConvertOnly mode
        b: cute.Tensor,
        c: cute.Tensor,
        global_scale: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """
        Executes the Mixed Input GEMM operation.

        This method sets up the kernel parameters, computes the grid size,
        defines the shared storage, and launches the kernel.

        The execution steps are as follows:
        - Setup static attributes before smem/grid/tma computation.
        - Setup TMA load/store atoms and tensors.
        - Compute grid size with regard to hardware constraints.
        - Define shared storage for kernel.
        - Launch the kernel synchronously.

        :param a: Input tensor A.
        :type a: cute.Tensor
        :param a_scale: Scale tensor for tensor A (None for ConvertOnly mode).
        :type a_scale: Optional[cute.Tensor]
        :param b: Input tensor B.
        :type b: cute.Tensor
        :param c: Output tensor C.
        :type c: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters to launch.
        :type max_active_clusters: cutlass.Int32
        :param stream: CUDA stream to launch the kernel on.
        :type stream: cuda.CUstream
        """
        self.a_dtype: type[cutlass.Numeric] = a.element_type
        self.a_scale_dtype: type[cutlass.Numeric] = (
            a_scale.element_type
            if self.scale_mode is TransformMode.ConvertScale
            else None
        )
        self.b_dtype: type[cutlass.Numeric] = b.element_type
        self.c_dtype: type[cutlass.Numeric] = c.element_type
        self.mma_dtype = self.b_dtype

        self.a_major_mode = cute.nvgpu.OperandMajorMode.K
        self.scale_major_mode = cute.nvgpu.OperandMajorMode.MN
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        if cutlass.const_expr(self.scale_mode == TransformMode.ConvertScale):
            self.gmem_layout_scale = a_scale.layout

        # Validate inputs
        self._validate_inputs(a, a_scale, b, c)

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.mma_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
            self.transform_a_source,
        )
        # Set up gmem copy atoms for A, scale, and B
        a_op = mixed_input_utils.get_tma_atom_kind(
            self.is_a_mcast, self.use_2cta_instrs, False
        )
        b_op = mixed_input_utils.get_tma_atom_kind(
            self.is_b_mcast, self.use_2cta_instrs, True
        )
        a_scale_op = a_op
        # Deduce TMA copy atom and TMA tensor for A, scale, and B
        smem_layout_a_per_stage = cute.slice_(self.smem_layout_a, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            smem_layout_a_per_stage,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if a.element_type is cutlass.Float32 else None
            ),
        )

        tma_atom_scale, tma_tensor_scale = None, None
        if cutlass.const_expr(self.scale_mode == TransformMode.ConvertScale):
            scale_n_tiles = cute.size(a_scale.layout.shape[0][1])
            scale_k_tiles = cute.size(a_scale.layout.shape[1][1])
            scale_words = cute.make_tensor(
                cute.recast_ptr(a_scale.iterator, dtype=cutlass.Uint32),
                cute.make_layout(
                    (
                        (self.scale_tile_shape[0], scale_n_tiles),
                        scale_k_tiles,
                        1,
                    ),
                    stride=(
                        (1, scale_k_tiles * self.scale_tile_shape[0]),
                        self.scale_tile_shape[0],
                        scale_n_tiles
                        * scale_k_tiles
                        * self.scale_tile_shape[0],
                    ),
                ),
            )
            tma_atom_scale, tma_tensor_scale = cpasync.make_tiled_tma_atom(
                a_scale_op,
                scale_words,
                self.smem_layout_scale_words_per_stage,
                (self.scale_tile_shape[0], 1),
            )

        smem_layout_b_per_stage = cute.slice_(self.smem_layout_b, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            smem_layout_b_per_stage,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if b.element_type is cutlass.Float32 else None
            ),
        )

        # Calculate copy size for tensor A, B, and scale
        a_copy_size = cute.size_in_bytes(self.a_dtype, smem_layout_a_per_stage)
        b_copy_size = cute.size_in_bytes(self.b_dtype, smem_layout_b_per_stage)
        a_scale_copy_size = (
            cute.size_in_bytes(
                cutlass.Uint32, self.smem_layout_scale_words_per_stage
            )
            if self.scale_mode is TransformMode.ConvertScale
            else 0
        )

        self.num_tma_load_bytes_a = a_copy_size
        self.num_tma_load_bytes_b = b_copy_size * cute.size(tiled_mma.thr_id.shape)
        self.num_tma_load_bytes_scale = a_scale_copy_size
        self.tile_sched_params, grid = self._compute_grid(
            c,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        tma_atom_c = None
        tma_tensor_c = None
        c_smem_size = 0
        if cutlass.const_expr(self.use_tma_store):
            epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                c,
                epi_smem_layout,
                self.epi_tile,
            )
            c_smem_size = cute.cosize(self.c_smem_layout_staged.outer)

        # Shared memory structure
        a_smem_size = cute.cosize(self.smem_layout_a.outer)
        b_smem_size = cute.cosize(self.smem_layout_b.outer)
        a_transform_smem_size = (
            cute.cosize(self.smem_layout_a_transform.outer)
            if self.transform_a_source == tcgen05.OperandSource.SMEM
            else 0
        )
        a_scale_smem_size = (
            cute.cosize(self.smem_layout_scale.outer)
            if self.scale_mode is TransformMode.ConvertScale
            else 0
        )

        @cute.struct
        class SharedStorage:
            a_load2trans_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_load2trans_stage
            ]
            a_load2trans_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_load2trans_stage
            ]
            a_scale_load2trans_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_scale_load2trans_stage
            ]
            a_scale_load2trans_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_scale_load2trans_stage
            ]
            a_trans2mma_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_trans2mma_stage
            ]
            a_trans2mma_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_trans2mma_stage
            ]
            b_load2mma_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_load2trans_stage
            ]
            b_load2mma_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_load2trans_stage
            ]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # Tensor buffers
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            smem_C: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, c_smem_size],
                self.smem_buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            smem_A: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, a_smem_size],
                self.smem_buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            smem_B: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, b_smem_size],
                self.smem_buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            smem_A_transform: cute.struct.Align[
                cute.struct.MemRange[self.mma_dtype, a_transform_smem_size],
                self.smem_buffer_align_bytes,
            ]
            # (MMA, MMA_M_SCALE, MMA_K_SCALE, STAGE)
            smem_A_scale: cute.struct.Align[
                cute.struct.MemRange[self.a_scale_dtype, a_scale_smem_size],
                self.smem_buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch kernel
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_scale,
            tma_tensor_scale,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c if self.use_tma_store else c,
            global_scale,
            self.cluster_layout_vmnk,
            self.smem_layout_a,
            self.smem_layout_scale,
            self.smem_layout_scale_words,
            self.smem_layout_a_transform,
            self.smem_layout_b,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_s: Optional[cute.CopyAtom],
        mS_mkl: Optional[cute.Tensor],
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        global_scale: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        scale_smem_layout: cute.ComposedLayout,
        scale_smem_layout_words: cute.ComposedLayout,
        a_smem_layout_transform: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        """
        GPU device kernel performing the Persistent Mixed-Input GEMM computation.
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        # Prefetch TMA descriptors
        if warp_idx == self.epilog_warp_id[0]:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(self.scale_mode == TransformMode.ConvertScale):
                cpasync.prefetch_descriptor(tma_atom_s)
            if cutlass.const_expr(self.use_tma_store):
                cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        bidx, bidy, bidz = cute.arch.block_idx()
        # Compute how many k_tiles share the same scale
        num_k_tiles_per_scale = max(
            1, self.scale_granularity_k // self.cta_tile_shape_mnk[2]
        )

        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Initialize load2transform pipeline, which tracks the dependencies between TMA's loading
        # of A and B, and the transformation of A and MMA's consumption
        transform_thread_idx = (
            tidx - 32 * self.transform_warp_id[0]
            if tidx >= 32 * self.transform_warp_id[0]
            else tidx
        )
        a_load2trans_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.a_load2trans_full_mbar_ptr.data_ptr(),
            num_stages=self.num_load2trans_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mcast_ctas_a * len(self.transform_warp_id),
            ),
            tx_count=self.num_tma_load_bytes_a,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=transform_thread_idx,
            mcast_mode_mn=(1, 0),  # multicast for A will only happen on the M-mode
            defer_sync=True,
        )
        # Initialize scale_load2trans pipeline, which tracks the dependencies between TMA's loading
        # of scale, and the transformation of A
        scale_load2trans_pipeline = None
        if cutlass.const_expr(self.scale_mode == TransformMode.ConvertScale):
            num_producers_a_scale = self.num_mcast_ctas_a
            scale_load2trans_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.a_scale_load2trans_full_mbar_ptr.data_ptr(),
                num_stages=self.num_scale_load2trans_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    num_producers_a_scale
                    * len(self.transform_warp_id)
                    * num_k_tiles_per_scale,
                ),
                tx_count=self.num_tma_load_bytes_scale,
                cta_layout_vmnk=cluster_layout_vmnk,
                tidx=transform_thread_idx,
                mcast_mode_mn=(
                    1,
                    0,
                ),  # multicast for scale_a will only happen on the M-mode
                defer_sync=True,
            )
        # Initialize transform2mma pipeline, which tracks the dependencies between the transformation
        # of A and MMA's consumption of transformed A
        cta_v_size = cute.size(cluster_layout_vmnk, mode=[0])
        trans2mma_pipeline = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.a_trans2mma_full_mbar_ptr.data_ptr(),
            num_stages=self.num_trans2mma_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.transform_warp_id) * cta_v_size,
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # Initialize pipeline for tensor B load to MMA
        # MMA warp informs TMA warp to proceed to load next tile of B tensor
        b_load2mma_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_load2mma_full_mbar_ptr.data_ptr(),
            num_stages=self.num_load2trans_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mcast_ctas_b
            ),
            tx_count=self.num_tma_load_bytes_b,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=(0, 1),  # multicast for B will only happen on the N-mode
            defer_sync=True,
        )
        # Initialize accumulator pipeline, which tracks the dependencies between
        # MMA's computation of accumulators and epilogue warps' consumption of accumulators
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, cta_v_size * len(self.epilog_warp_id)
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_ptr_sync_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # Setup smem tensor A/scale/B/C
        sC = (
            storage.smem_C.get_tensor(
                c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
            )
            if self.use_tma_store
            else None
        )
        sA_input = storage.smem_A.get_tensor(
            a_smem_layout.outer, swizzle=a_smem_layout.inner
        )
        sS_transform = (
            storage.smem_A_scale.get_tensor(
                scale_smem_layout.outer, swizzle=scale_smem_layout.inner
            )
            if self.scale_mode is TransformMode.ConvertScale
            else None
        )
        sS_words = (
            cute.make_tensor(
                cute.recast_ptr(sS_transform.iterator, dtype=cutlass.Uint32),
                scale_smem_layout_words,
            )
            if self.scale_mode is TransformMode.ConvertScale
            else None
        )
        sB_input = storage.smem_B.get_tensor(
            b_smem_layout.outer, swizzle=b_smem_layout.inner
        )
        sA_transform = None
        # Get smem tensor for transformed A when transform_a_source is SMEM
        if cutlass.const_expr(self.transform_a_source == tcgen05.OperandSource.SMEM):
            sA_transform = storage.smem_A_transform.get_tensor(
                a_smem_layout_transform.outer, swizzle=a_smem_layout_transform.inner
            )

        # Compute multicast mask for A/B buffer full
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        s_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            # scale tensor share the same multicast mask with A tensor
            s_full_mcast_mask = a_full_mcast_mask
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        # local_tile partition global tensors
        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (scale-row, 1, loopM, loopK, loopL)
        gS_mkl = (
            cute.local_tile(
                mS_mkl,
                (self.scale_tile_shape[0], 1),
                (None, None, None),
            )
            if self.scale_mode is TransformMode.ConvertScale
            else None
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, loopM, loopN, loopL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        # Partition global tensor for TiledMMA_A/B/C
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, loopM, loopK, loopL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        tCgC = thr_mma.partition_C(gC_mnl)

        # Setup copy atom to load A from shared memory for further transformation
        copy_atom_a_input = (
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.a_dtype, num_bits_per_copy=32
            )
            if self.scale_mode is TransformMode.ConvertScale
            else None
        )
        a_smem_shape = tiled_mma.partition_shape_A(
            cute.dice(self.mma_tiler, (1, None, 1))
        )
        # Setup copy atom to store transformed A into tensor memory or shared memory
        copy_atom_a_transform = mixed_input_utils.get_copy_atom_a_transform(
            self.mma_dtype,
            self.use_2cta_instrs,
            self.transform_a_source,
            a_smem_shape,
            self.a_dtype,
        )

        # Partition global/shared tensor for TMA load A/B
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA_input, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )

        tCsS = None
        tSsS = None
        tSgS = None
        if cutlass.const_expr(self.scale_mode == TransformMode.ConvertScale):
            thr_mma_leader_cta = tiled_mma.get_slice(0)
            # (MMA, MMA_M, MMA_K, STAGE)
            tCsS = thr_mma_leader_cta.partition_A(sS_transform)
            # (atom_v, STAGE), (atom_v, loopM, loopK, loopL)
            tSsS, tSgS = cpasync.tma_partition(
                tma_atom_s,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                cute.group_modes(sS_words, 0, 2),
                cute.group_modes(gS_mkl, 0, 2),
            )

        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB_input, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB_input)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        # Cluster wait before TMEM alloc and ensure pipelines are ready
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # TMEM allocation
        tmem.allocate(self.num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        # Get the pointer to the TMEM buffer
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        accumulators = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        tCrA = None
        if cutlass.const_expr(self.transform_a_source == tcgen05.OperandSource.TMEM):
            tmem_ptr_transform = cute.recast_ptr(
                accumulators.iterator + self.num_acc_tmem_cols, dtype=self.mma_dtype
            )
            tCrA = cute.make_tensor(
                tmem_ptr_transform,
                tiled_mma.make_fragment_A(a_smem_layout_transform.outer).layout,
            )
        else:
            tCrA = tiled_mma.make_fragment_A(sA_transform)

        # Specialized TMA load warp for A/B tensor
        if warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_tma_warps)
            # Persistent tile scheduling loop
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, (bidx, bidy, bidz), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            a_load2trans_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_load2trans_stage
            )
            b_load2mma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_load2trans_stage
            )
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                a_load2trans_producer_state.reset_count()
                peek_load2trans_empty_status = cutlass.Boolean(1)
                if a_load2trans_producer_state.count < k_tile_cnt:
                    peek_load2trans_empty_status = (
                        a_load2trans_pipeline.producer_try_acquire(
                            a_load2trans_producer_state
                        )
                    )
                b_load2mma_producer_state.reset_count()
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    a_load2trans_pipeline.producer_acquire(
                        a_load2trans_producer_state, peek_load2trans_empty_status
                    )
                    b_load2mma_pipeline.producer_acquire(b_load2mma_producer_state)
                    # TMA load A/B
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, a_load2trans_producer_state.count)],
                        tAsA[(None, a_load2trans_producer_state.index)],
                        tma_bar_ptr=a_load2trans_pipeline.producer_get_barrier(
                            a_load2trans_producer_state
                        ),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, b_load2mma_producer_state.count)],
                        tBsB[(None, b_load2mma_producer_state.index)],
                        tma_bar_ptr=b_load2mma_pipeline.producer_get_barrier(
                            b_load2mma_producer_state
                        ),
                        mcast_mask=b_full_mcast_mask,
                    )
                    a_load2trans_pipeline.producer_commit(a_load2trans_producer_state)
                    b_load2mma_pipeline.producer_commit(b_load2mma_producer_state)
                    a_load2trans_producer_state.advance()
                    b_load2mma_producer_state.advance()
                    if a_load2trans_producer_state.count < k_tile_cnt:
                        peek_load2trans_empty_status = (
                            a_load2trans_pipeline.producer_try_acquire(
                                a_load2trans_producer_state
                            )
                        )
                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # Wait A/B buffer empty
            a_load2trans_pipeline.producer_tail(a_load2trans_producer_state)
            b_load2mma_pipeline.producer_tail(b_load2mma_producer_state)

        # Specialized TMA load for scale tensor
        if warp_idx == self.scale_tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_tma_warps)
            if cutlass.const_expr(self.scale_mode == TransformMode.ConvertScale):
                # Persistent tile scheduling loop
                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, (bidx, bidy, bidz), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()
                scale_load2trans_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_scale_load2trans_stage
                )
                scale_k_tile_cnt = cute.size(gS_mkl, mode=[3])

                while work_tile.is_valid_tile:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_mnl = (
                        cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                        cur_tile_coord[1],
                        cur_tile_coord[2],
                    )
                    # (atom_v, RestK)
                    tSgS_slice = tSgS[
                        (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                    ]

                    scale_load2trans_producer_state.reset_count()
                    peek_scale_load2trans_empty_status = cutlass.Boolean(1)
                    if scale_load2trans_producer_state.count < scale_k_tile_cnt:
                        peek_scale_load2trans_empty_status = (
                            scale_load2trans_pipeline.producer_try_acquire(
                                scale_load2trans_producer_state
                            )
                        )
                    for k_tile in cutlass.range(0, scale_k_tile_cnt, 1, unroll=1):
                        scale_load2trans_pipeline.producer_acquire(
                            scale_load2trans_producer_state,
                            peek_scale_load2trans_empty_status,
                        )
                        # TMA load scale
                        cute.copy(
                            tma_atom_s,
                            tSgS_slice[
                                (None, scale_load2trans_producer_state.count)
                            ],
                            tSsS[(None, scale_load2trans_producer_state.index)],
                            tma_bar_ptr=scale_load2trans_pipeline.producer_get_barrier(
                                scale_load2trans_producer_state
                            ),
                            mcast_mask=s_full_mcast_mask,
                        )

                        scale_load2trans_producer_state.advance()
                        peek_scale_load2trans_empty_status = cutlass.Boolean(1)
                        if scale_load2trans_producer_state.count < scale_k_tile_cnt:
                            peek_scale_load2trans_empty_status = (
                                scale_load2trans_pipeline.producer_try_acquire(
                                    scale_load2trans_producer_state
                                )
                            )
                    # Advance to next tile
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                # Wait scale buffer empty
                scale_load2trans_pipeline.producer_tail(scale_load2trans_producer_state)

        # Specialized transform warps
        if warp_idx >= self.transform_warp_id[0]:
            cute.arch.setmaxregister_increase(self.num_regs_transform_warps)
            transform_local_tidx = tidx - 32 * self.transform_warp_id[0]
            # Partition tensors for transform input and output and set up the copy atom
            # used for loading and storing transformed A tensor
            (
                src_copy_a,
                dst_copy_a,
                tAsA_input,
                tAsA_transform,
            ) = mixed_input_utils.transform_partition(
                self.transform_a_source,
                self.scale_mode,
                copy_atom_a_input,
                copy_atom_a_transform,
                sA_input,
                (
                    tCrA
                    if self.transform_a_source == tcgen05.OperandSource.TMEM
                    else sA_transform
                ),
                transform_local_tidx,
            )
            # make rmem tensor for input A and transformed A
            tArA = cute.make_rmem_tensor(
                tAsA_input[(None, None, None, None, 0)].shape, tAsA_input.element_type
            )
            tArA_transform = cute.make_rmem_tensor(
                tAsA_input[(None, None, None, None, 0)].shape, self.mma_dtype
            )
            # Partition scale tensor
            smem_thr_copy_S = None
            tSsS_trans = None
            tSrS_copy = None
            tSrS = None
            if cutlass.const_expr(self.scale_mode == TransformMode.ConvertScale):
                smem_thr_copy_S, tSsS_trans, tSrS_copy, tSrS = (
                    mixed_input_utils.scale_partition(
                        src_copy_a, tCsS, transform_local_tidx, self.a_scale_dtype
                    )
                )
                assert cute.size(tSrS, mode=[0]) == cute.size(tArA, mode=[0]), (
                    "tSrS and tArA have different leading dimension"
                )
                assert cute.size(tSrS) == cute.size(tArA), (
                    "tSrS and tArA have different shape"
                )
            # Deduce a sub-tile size and tile tensors
            transform_tiler_size = min(
                cute.size(cute.coalesce(tAsA_input.layout), mode=[0]), 64
            )
            transform_tiler = cute.make_layout(transform_tiler_size)
            tArA_load = cute.flat_divide(tArA, transform_tiler)
            tArA_load = cute.group_modes(tArA_load, 1, cute.rank(tArA_load))
            tSrS_load = (
                cute.flat_divide(tSrS, transform_tiler)
                if self.scale_mode is TransformMode.ConvertScale
                else None
            )
            tSrS_load = (
                cute.group_modes(tSrS_load, 1, cute.rank(tSrS_load))
                if self.scale_mode is TransformMode.ConvertScale
                else None
            )
            if cutlass.const_expr(self.scale_mode == TransformMode.ConvertScale):
                assert cute.size(tSrS_load) == cute.size(tArA_load), (
                    "scale broadcast must cover every transformed FP4 element"
                )
                assert cute.size(tArA_load, mode=[1]) == 1, (
                    "Blackwell K64 transform must fit in one per-thread vector"
                )
            tArA_transform_store = cute.flat_divide(tArA_transform, transform_tiler)
            tArA_transform_store = cute.group_modes(
                tArA_transform_store, 1, cute.rank(tArA_transform_store)
            )

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, (bidx, bidy, bidz), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            a_load2trans_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_load2trans_stage,
            )
            scale_load2trans_consumer_state = (
                pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    self.num_scale_load2trans_stage,
                )
                if self.scale_mode is TransformMode.ConvertScale
                else None
            )
            trans2mma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_trans2mma_stage,
            )
            while work_tile.is_valid_tile:
                a_load2trans_consumer_state.reset_count()
                peek_load2trans_full_status = cutlass.Boolean(1)
                if a_load2trans_consumer_state.count < k_tile_cnt:
                    peek_load2trans_full_status = (
                        a_load2trans_pipeline.consumer_try_wait(
                            a_load2trans_consumer_state
                        )
                    )
                peek_scale_load2trans_full_status = cutlass.Boolean(1)
                if cutlass.const_expr(self.scale_mode == TransformMode.ConvertScale):
                    scale_load2trans_consumer_state.reset_count()
                    peek_scale_load2trans_full_status = (
                        scale_load2trans_pipeline.consumer_try_wait(
                            scale_load2trans_consumer_state
                        )
                    )
                trans2mma_producer_state.reset_count()
                peek_trans2mma_empty_status = cutlass.Boolean(1)
                if trans2mma_producer_state.count < k_tile_cnt:
                    peek_trans2mma_empty_status = (
                        trans2mma_pipeline.producer_try_acquire(
                            trans2mma_producer_state
                        )
                    )

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    a_load2trans_pipeline.consumer_wait(
                        a_load2trans_consumer_state, peek_load2trans_full_status
                    )
                    # Load A from shared memory
                    tAsA_input_slice = tAsA_input[
                        (None, None, None, None, a_load2trans_consumer_state.index)
                    ]
                    tAsA_input_slice = cute.flat_divide(
                        tAsA_input_slice, transform_tiler
                    )
                    tAsA_input_slice = cute.group_modes(
                        tAsA_input_slice, 1, cute.rank(tAsA_input_slice)
                    )
                    if cutlass.const_expr(
                        self.scale_mode == TransformMode.ConvertScale
                    ):
                        scale_load2trans_pipeline.consumer_wait(
                            scale_load2trans_consumer_state,
                            peek_scale_load2trans_full_status,
                        )
                    trans2mma_pipeline.producer_acquire(
                        trans2mma_producer_state, peek_trans2mma_empty_status
                    )
                    # load scale tensor when needed
                    if cutlass.const_expr(
                        self.scale_mode == TransformMode.ConvertScale
                    ):
                        if k_tile % num_k_tiles_per_scale == 0:
                            tSsS_slice = tSsS_trans[
                                (
                                    None,
                                    None,
                                    None,
                                    None,
                                    scale_load2trans_consumer_state.index,
                                )
                            ]
                            tSsS_slice_filtered = cute.make_tensor(
                                tSsS_slice.iterator,
                                cute.filter_zeros(tSsS_slice.layout),
                            )
                            cute.autovec_copy(tSsS_slice_filtered, tSrS_copy)
                        cur_scale_load2trans_consumer_state = (
                            scale_load2trans_consumer_state.clone()
                        )
                        if (k_tile + 1) % num_k_tiles_per_scale == 0:
                            scale_load2trans_consumer_state.advance()

                    cur_a_load2trans_consumer_state = (
                        a_load2trans_consumer_state.clone()
                    )
                    for idx in cutlass.range_constexpr(cute.size(tArA_load, mode=[1])):
                        # Load A from shared memory
                        cute.autovec_copy(
                            tAsA_input_slice[(None, idx)],
                            tArA_load[(None, idx)],
                        )
                        if cutlass.const_expr(
                            idx == cute.size(tArA_load, mode=[1]) - 1
                        ):
                            a_load2trans_consumer_state.advance()
                            if a_load2trans_consumer_state.count < k_tile_cnt:
                                peek_load2trans_full_status = (
                                    a_load2trans_pipeline.consumer_try_wait(
                                        a_load2trans_consumer_state
                                    )
                                )
                                if cutlass.const_expr(
                                    self.scale_mode == TransformMode.ConvertScale
                                ):
                                    peek_scale_load2trans_full_status = (
                                        scale_load2trans_pipeline.consumer_try_wait(
                                            scale_load2trans_consumer_state
                                        )
                                    )
                        # Blackwell has a direct packed E2M1-to-BF16x2 conversion.
                        # It preserves the layout's low-even/high-odd nibble
                        # order and avoids FP4->FP16->FP32->BF16 expansion.
                        if cutlass.const_expr(
                            self.mma_dtype is BFloat16
                            and self.a_dtype is cutlass.Float4E2M1FN
                        ):
                            source_fragment = tArA_load[(None, idx)]
                            packed_source = cute.recast_tensor(source_fragment, Uint8)
                            transformed_fragment = cute.make_rmem_tensor(
                                source_fragment.shape, BFloat16
                            )
                            packed_transformed = cute.recast_tensor(
                                transformed_fragment, Uint32
                            )
                            for packed_idx in cutlass.range_constexpr(
                                cute.size(packed_source.shape)
                            ):
                                packed_transformed[packed_idx] = (
                                    cvt_e2m1x2_to_bf16x2_u32(
                                        packed_source[packed_idx]
                                    )
                                )
                            tensor_transformed = transformed_fragment.load()
                        else:
                            tensor_transformed = mixed_input_utils.cvt_tensor_a(
                                tArA_load[(None, idx)],
                                self.mma_dtype,
                                self.shuffle_a,
                            )
                        if cutlass.const_expr(
                            self.scale_mode == TransformMode.ConvertScale
                        ):
                            expanded_scale = cute.make_rmem_tensor(
                                tensor_transformed.shape, self.mma_dtype
                            )
                            for element_idx in cutlass.range_constexpr(
                                cute.size(tensor_transformed.shape)
                            ):
                                # Preserve the scale tensor's logical zero-stride
                                # broadcast instead of assuming a thread-local
                                # fragment order. This selects the K16 scale for
                                # the exact A element even when M/K are interleaved.
                                expanded_scale[element_idx] = tSrS_load[
                                    cute.idx2crd(element_idx, tSrS_load.shape)
                                ].to(self.mma_dtype)
                            global_scale_value = global_scale[0].to(self.mma_dtype)
                            tensor_transformed = (
                                tensor_transformed
                                * expanded_scale.load()
                                * global_scale_value
                            )
                        tArA_transform_store[(None, idx)].store(tensor_transformed)
                    # Store transformed A to tensor memory or shared memory
                    mixed_input_utils.store_transformed_a(
                        tArA_transform,
                        tAsA_transform[
                            (None, None, None, None, trans2mma_producer_state.index)
                        ],
                        dst_copy_a,
                    )
                    # Ensure all transform threads have finished the copy and reached the fence
                    self.transform_sync_barrier.arrive_and_wait()
                    if cutlass.const_expr(
                        self.transform_a_source == tcgen05.OperandSource.TMEM
                    ):
                        cute.arch.fence_view_async_tmem_store()
                    else:
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                    # Signal the completion of transformation
                    if cutlass.const_expr(
                        self.scale_mode == TransformMode.ConvertScale
                    ):
                        scale_load2trans_pipeline.consumer_release(
                            cur_scale_load2trans_consumer_state
                        )
                    a_load2trans_pipeline.consumer_release(
                        cur_a_load2trans_consumer_state
                    )
                    # Signal the completion of transformation
                    trans2mma_pipeline.producer_commit(trans2mma_producer_state)
                    trans2mma_producer_state.advance()
                    if trans2mma_producer_state.count < k_tile_cnt:
                        peek_trans2mma_empty_status = (
                            trans2mma_pipeline.producer_try_acquire(
                                trans2mma_producer_state
                            )
                        )
                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # Wait a_transform buffer empty
            trans2mma_pipeline.producer_tail(trans2mma_producer_state)

        # Specialized MMA warp
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma_warp)
            tCtAcc_base = accumulators
            # Persistent tile scheduling loop
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, (bidx, bidy, bidz), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            trans2mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_trans2mma_stage
            )
            b_load2mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_load2trans_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                b_load2mma_consumer_state.reset_count()
                trans2mma_consumer_state.reset_count()
                peek_trans2mma_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    if trans2mma_consumer_state.count < k_tile_cnt:
                        peek_trans2mma_full_status = (
                            trans2mma_pipeline.consumer_try_wait(
                                trans2mma_consumer_state
                            )
                        )
                    acc_pipeline.producer_acquire(acc_producer_state)

                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    # Mma mainloop
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        trans2mma_pipeline.consumer_wait(
                            trans2mma_consumer_state, peek_trans2mma_full_status
                        )
                        b_load2mma_pipeline.consumer_wait(b_load2mma_consumer_state)
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord_a = (
                                None,
                                None,
                                kblock_idx,
                                trans2mma_consumer_state.index,
                            )
                            kblock_coord_b = (
                                None,
                                None,
                                kblock_idx,
                                b_load2mma_consumer_state.index,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord_a],
                                tCrB[kblock_coord_b],
                                tCtAcc,
                            )
                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        trans2mma_pipeline.consumer_release(trans2mma_consumer_state)
                        b_load2mma_pipeline.consumer_release(b_load2mma_consumer_state)
                        trans2mma_consumer_state.advance()
                        b_load2mma_consumer_state.advance()
                        peek_trans2mma_full_status = cutlass.Boolean(1)
                        if trans2mma_consumer_state.count < k_tile_cnt:
                            peek_trans2mma_full_status = (
                                trans2mma_pipeline.consumer_try_wait(
                                    trans2mma_consumer_state
                                )
                            )
                    # Async arrive accumulator buffer full
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # Wait for accumulator buffer empty
            acc_pipeline.producer_tail(acc_producer_state)

        # Specialized epilogue warps
        if warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_epilogue_warps)
            epi_tidx = tidx
            tCtAcc_base = accumulators
            # Partition for epilogue
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = mixed_input_utils.epilog_tmem_copy_and_partition(
                self.cta_tile_shape_mnk,
                self.c_layout,
                self.c_dtype,
                self.acc_dtype,
                epi_tidx,
                tCtAcc_base,
                tCgC,
                epi_tile,
                self.use_2cta_instrs,
            )

            tTR_rC = None
            tiled_copy_r2s = None
            simt_atom = None
            tRS_rC = None
            tRS_sC = None
            bSG_sC = None
            bSG_gC_partitioned = None
            tTR_gC_partitioned = None
            if cutlass.const_expr(self.use_tma_store):
                tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
                tiled_copy_r2s, tRS_rC, tRS_sC = (
                    mixed_input_utils.epilog_smem_copy_and_partition(
                        self.c_layout,
                        self.c_dtype,
                        self.acc_dtype,
                        tiled_copy_t2r,
                        tTR_rC,
                        epi_tidx,
                        sC,
                    )
                )
                (
                    tma_atom_c,
                    bSG_sC,
                    bSG_gC_partitioned,
                ) = self.epilog_gmem_copy_and_partition(
                    epi_tidx, tma_atom_c, tCgC, epi_tile, sC
                )
            else:
                (
                    simt_atom,
                    tTR_rC,
                    tTR_gC_partitioned,
                ) = self.epilog_gmem_copy_and_partition(
                    epi_tidx, tiled_copy_t2r, tCgC, epi_tile, sC
                )
            # Persistent tile scheduling loop
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, (bidx, bidy, bidz), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            c_pipeline = None
            if cutlass.const_expr(self.use_tma_store):
                c_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    32 * len(self.epilog_warp_id),
                )
                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_c_stage,
                    producer_group=c_producer_group,
                )

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                bSG_gC = None
                tTR_gC = None
                if cutlass.const_expr(self.use_tma_store):
                    bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                else:
                    tTR_gC = tTR_gC_partitioned[
                        (None, None, None, None, None, *mma_tile_coord_mnl)
                    ]

                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]
                # Wait for accumulator buffer full
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                if cutlass.const_expr(self.use_tma_store):
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                else:
                    tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))

                # Store accumulator to global memory in subtiles
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                for subtile_idx in cutlass.range(subtile_cnt):
                    # Load accumulator from tensor memory buffer to register
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                    if cutlass.const_expr(self.use_tma_store):
                        # Convert to C type
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        acc_vec = acc_vec.to(self.c_dtype)
                        tRS_rC.store(acc_vec)
                        c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                        # Store C to shared memory
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, c_buffer)],
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        self.epilog_sync_barrier.arrive_and_wait()
                        # TMA store C to global memory
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, subtile_idx)],
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()
                    else:
                        # Convert to C type
                        acc_vec = tTR_rAcc.load()
                        acc_vec = acc_vec.to(self.c_dtype)
                        tTR_rC.store(acc_vec)
                        # Store C to global memory
                        cute.autovec_copy(
                            tTR_rC, tTR_gC[(None, None, None, subtile_idx)]
                        )
                # Async arrive accumulator buffer empty
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()
                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            if cutlass.const_expr(self.use_tma_store):
                c_pipeline.producer_tail()

        # Idle warp
        if warp_idx == self.idle_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_idle_warp)

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """
        Partitions source and destination tensors for a TMA store or SIMT store.
        """
        if self.use_tma_store:
            tma_atom_c, bSG_sC, bSG_gC, _, _ = (
                mixed_input_utils.epilog_gmem_copy_and_partition(
                    self.c_dtype, tidx, atom, None, gC_mnl, None, epi_tile, sC
                )
            )
            return tma_atom_c, bSG_sC, bSG_gC
        else:
            _, _, _, simt_atom, tTR_gC = (
                mixed_input_utils.epilog_gmem_copy_and_partition(
                    self.c_dtype, tidx, None, atom, None, gC_mnl, epi_tile, sC
                )
            )
            # (T2R, T2R_M, T2R_N)
            tTR_rC = cute.make_rmem_tensor(
                tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.c_dtype
            )
            simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)
            return simt_atom, tTR_rC, tTR_gC

    @staticmethod
    def _compute_stages_and_tmem_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: tuple[int, int, int],
        cta_tile_shape_mnk: tuple[int, int, int],
        epi_tile: cute.Tile,
        a_dtype: type[cutlass.Numeric],
        a_scale_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        transform_a_source: tcgen05.OperandSource,
        scale_granularity_m: int,
        scale_granularity_k: int,
        smem_buffer_align_bytes: int,
        use_tma_store: bool,
        scale_mode: TransformMode,
    ) -> tuple[int, int, int, int, int, int, int]:
        """
        Compute pipeline stages and TMEM column allocation configurations.

        This method calculates the number of pipeline stages for different operations
        (load2trans, trans2mma, accumulator, etc.) and determines TMEM column allocation
        based on available memory resources and tile configuration.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param a_scale_dtype: Data type of operand A block scales.
        :type a_scale_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param c_dtype: Data type of operand C.
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum of operand C.
        :type c_layout: utils.LayoutEnum
        :param transform_a_source: The source of the transformed A tensor.
        :type transform_a_source: tcgen05.OperandSource
        :param scale_granularity_m: The granularity of the scale tensor along the M mode.
        :type scale_granularity_m: int
        :param scale_granularity_k: The granularity of the scale tensor along the K mode.
        :type scale_granularity_k: int
        :param smem_buffer_align_bytes: The alignment of the shared memory buffer.
        :type smem_buffer_align_bytes: int
        :param use_tma_store: Whether TMA store is enabled.
        :type use_tma_store: bool
        :param scale_mode: The transform mode.
        :type scale_mode: TransformMode

        :return: A tuple containing the number of stages for:
                 (load2trans, scale_load2trans, transform2mma, accumulator, c, tmem_acc_cols, tmem_a_cols)
        :rtype: tuple[int, int, int, int, int, int, int]
        - num_load2trans_stage: Stages for load-to-transform A and B tensors pipeline
        - num_scale_load2trans_stage: Stages for scale load-to-transform A tensor pipeline
        - num_trans2mma_stage: Stages for transform-to-MMA pipeline
        - num_acc_stage: Stages for accumulator-to-epilogue pipeline
        - num_c_stage: Stages for epilogue-to-output C pipeline
        - num_acc_tmem_cols: TMEM columns for accumulator
        - num_a_tmem_cols: TMEM columns for transformed A tensor
        """
        # Compute tmem columns required for accumulator
        acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
        tCtAcc_stage1 = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
        num_tmem_acc_col_per_stage = utils.get_num_tmem_alloc_cols(tCtAcc_stage1, True)
        # Heuristic to decide the number of stages for accumulator
        sm100_tmem_columns = cute.arch.get_max_tmem_alloc_cols("sm_100")
        accumulator_stage_count = sm100_tmem_columns // num_tmem_acc_col_per_stage
        if transform_a_source == tcgen05.OperandSource.TMEM:
            if num_tmem_acc_col_per_stage < 128:
                accumulator_stage_count = 3
            elif num_tmem_acc_col_per_stage < 256:
                accumulator_stage_count = 2
            else:
                accumulator_stage_count = 1
        # transformed A in 16bit, thus 1 tmem column could hold 2 elements
        num_elts_per_tmem_col = 32 // tiled_mma.op.a_dtype.width
        num_tmem_cols_a_per_stage = cute.round_up(
            (
                cta_tile_shape_mnk[2] // num_elts_per_tmem_col
                if transform_a_source == tcgen05.OperandSource.TMEM
                else 0
            ),
            4,
        )

        c_stage_count = 2 if use_tma_store else 0
        c_smem_layout_staged_one = (
            sm100_utils.make_smem_layout_epi(
                c_dtype,
                c_layout,
                epi_tile,
                1,
            )
            if use_tma_store
            else None
        )
        c_bytes_per_stage = (
            cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
            if use_tma_store
            else 0
        )
        c_bytes = c_bytes_per_stage * c_stage_count

        smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        bytes_per_pipeline_stage = 16
        if scale_mode == TransformMode.ConvertOnly:
            scale_load2trans_stage_count = 0
            a_scale_bytes_per_stage = 0
        else:
            # Ensure we have 2 buffers for scale tiles needed for 1 CTA tile
            a_scale_k_mode = max(cta_tile_shape_mnk[2] // scale_granularity_k, 1)
            a_scale_m_mode = max(cta_tile_shape_mnk[0] // scale_granularity_m, 1)
            scale_load2trans_stage_count = 4
            a_scale_bytes_per_stage = cute.round_up(
                cute.size_in_bytes(
                    a_scale_dtype,
                    cute.make_layout((a_scale_m_mode, a_scale_k_mode)),
                ),
                smem_buffer_align_bytes,
            )
        a_scale_bytes = (
            a_scale_bytes_per_stage + bytes_per_pipeline_stage
        ) * scale_load2trans_stage_count
        carveout_smem_bytes = (
            bytes_per_pipeline_stage * accumulator_stage_count + a_scale_bytes + c_bytes
        )

        # Compute transform stages if A is in TMEM
        num_tmem_acc_cols = cute.round_up(
            accumulator_stage_count * num_tmem_acc_col_per_stage, 4
        )

        transform2mma_stage_count_a_source_tmem_potential = (
            (sm100_tmem_columns - num_tmem_acc_cols) // num_tmem_cols_a_per_stage
            if transform_a_source == tcgen05.OperandSource.TMEM
            else -1
        )
        if (
            transform_a_source == tcgen05.OperandSource.TMEM
            and transform2mma_stage_count_a_source_tmem_potential <= 0
        ):
            raise ValueError("Not enough TMEM capacity for selected tile size")
        a_load_bytes_per_stage = cute.round_up(
            cute.size_in_bytes(
                a_dtype,
                cute.make_layout((cta_tile_shape_mnk[0], cta_tile_shape_mnk[2])),
            ),
            smem_buffer_align_bytes,
        )
        b_load_bytes_per_stage = cute.round_up(
            cute.size_in_bytes(
                b_dtype,
                cute.make_layout(
                    (
                        cta_tile_shape_mnk[1] // cute.size(tiled_mma.thr_id),
                        cta_tile_shape_mnk[2],
                    )
                ),
            ),
            smem_buffer_align_bytes,
        )
        ab_load_bytes_per_stage = (
            a_load_bytes_per_stage
            + b_load_bytes_per_stage
            + 2 * bytes_per_pipeline_stage
        )
        a_transform_bytes_per_stage = (
            cute.round_up(
                cute.size_in_bytes(
                    tiled_mma.op.a_dtype,
                    cute.make_layout((cta_tile_shape_mnk[0], cta_tile_shape_mnk[2])),
                ),
                smem_buffer_align_bytes,
            )
            if transform_a_source == tcgen05.OperandSource.SMEM
            else 0
        )

        a_transform_bytes_per_stage = (
            a_transform_bytes_per_stage + bytes_per_pipeline_stage
        )
        transform2mma_stage_count_a_source_smem_potential = (
            smem_capacity - carveout_smem_bytes
        ) // (ab_load_bytes_per_stage + a_transform_bytes_per_stage)
        transform2mma_stage_count = (
            min(
                transform2mma_stage_count_a_source_tmem_potential,
                transform2mma_stage_count_a_source_smem_potential,
            )
            if transform_a_source == tcgen05.OperandSource.TMEM
            else transform2mma_stage_count_a_source_smem_potential
        )
        load2transform_stage_count = (
            smem_capacity
            - carveout_smem_bytes
            - (transform2mma_stage_count * a_transform_bytes_per_stage)
        ) // ab_load_bytes_per_stage
        if (
            load2transform_stage_count < 2
            or transform2mma_stage_count < 2
            or accumulator_stage_count < 1
        ):
            raise ValueError("Not enough SMEM or TMEM capacity for selected tile size")
        num_tmem_a_cols = transform2mma_stage_count * num_tmem_cols_a_per_stage
        # Check if we can increase c_stage_count with leftover smem
        if use_tma_store:
            c_stage_count += (
                smem_capacity
                - load2transform_stage_count * ab_load_bytes_per_stage
                - transform2mma_stage_count * a_transform_bytes_per_stage
                - scale_load2trans_stage_count * a_scale_bytes_per_stage
                - c_bytes
            ) // c_bytes_per_stage

        return (
            load2transform_stage_count,
            scale_load2trans_stage_count,
            transform2mma_stage_count,
            accumulator_stage_count,
            c_stage_count,
            num_tmem_acc_cols,
            num_tmem_a_cols,
        )

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        max_active_clusters: cutlass.Int32,
    ) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
        """
        Use persistent tile scheduler to compute the grid size for the output tensor C.
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    def is_valid_epilog_store_option(
        m: int,
        n: int,
        mma_tiler_mn: tuple[int, int],
        use_tma_store: bool,
        use_2cta_instrs: bool,
    ) -> bool:
        """
        Check if the epilogue store option is valid for the given problem size.
        """
        cta_tile_shape_mn = (
            mma_tiler_mn[0] // (2 if use_2cta_instrs else 1),
            mma_tiler_mn[1],
        )
        # No OOB tile support when TMA store is disabled
        if not use_tma_store:
            if not (m % cta_tile_shape_mn[0] == 0 and n % cta_tile_shape_mn[1] == 0):
                return False
        return True


_TOKEN_TILES = (8, 16, 32, 64, 128, 256)
_N_TILE = 128
_K_TILE = 64
_SCALE_GROUP = 16


class Nvfp4A16BlackwellGemmLaunch:
    """Pointer ABI for one token-tile and activation-dtype variant."""

    def __init__(
        self,
        io_dtype: type[cutlass.Numeric],
        token_tile: int,
    ):
        self.io_dtype = io_dtype
        self.token_tile = token_tile
        self.kernel = Nvfp4A16BlackwellGemm(
            scale_granularity_m=1,
            scale_granularity_k=_SCALE_GROUP,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=False,
            mma_tiler_mnk=(_N_TILE, token_tile, _K_TILE),
            cluster_shape_mn=(1, 1),
            use_tma_store=True,
            shuffle_a=False,
        )

    @cute.jit
    def __call__(
        self,
        activation_ptr: cute.Pointer,
        qweight_ptr: cute.Pointer,
        block_scales_ptr: cute.Pointer,
        global_scale_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        num_tokens: cutlass.Int32,
        out_features: cutlass.Int32,
        in_features: cutlass.Int32,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        # Keep the public wrapper ABI at int32, but promote every extent and
        # stride calculation before multiplying runtime dimensions.  Large M
        # profiles must not wrap while CuTe constructs the TMA tensor maps.
        num_tokens_64 = cutlass.Int64(num_tokens)
        out_features_64 = cutlass.Int64(out_features)
        in_features_64 = cutlass.Int64(in_features)
        n_tiles = out_features_64 // _N_TILE
        k_tiles = in_features_64 // _K_TILE

        qweight = cute.make_tensor(
            qweight_ptr,
            cute.make_layout(
                ((_N_TILE, n_tiles), (_K_TILE, k_tiles), 1),
                stride=(
                    (_K_TILE, k_tiles * _N_TILE * _K_TILE),
                    (1, _N_TILE * _K_TILE),
                    out_features_64 * in_features_64,
                ),
            ),
        )
        block_scales = cute.make_tensor(
            block_scales_ptr,
            cute.make_layout(
                (
                    (_N_TILE, n_tiles),
                    ((_SCALE_GROUP, _K_TILE // _SCALE_GROUP), k_tiles),
                    1,
                ),
                stride=(
                    (_K_TILE // _SCALE_GROUP, k_tiles * _N_TILE * (_K_TILE // _SCALE_GROUP)),
                    ((0, 1), _N_TILE * (_K_TILE // _SCALE_GROUP)),
                    (out_features_64 * in_features_64) // _SCALE_GROUP,
                ),
            ),
        )
        activation = cute.make_tensor(
            activation_ptr,
            cute.make_layout(
                (num_tokens_64, in_features_64, 1),
                stride=(in_features_64, 1, num_tokens_64 * in_features_64),
            ),
        )
        output = cute.make_tensor(
            output_ptr,
            cute.make_layout(
                (out_features_64, num_tokens_64, 1),
                stride=(1, out_features_64, num_tokens_64 * out_features_64),
            ),
        )
        global_scale = cute.make_tensor(
            global_scale_ptr, cute.make_layout((1,), stride=(1,))
        )
        self.kernel(
            qweight,
            block_scales,
            activation,
            output,
            global_scale,
            max_active_clusters,
            stream,
        )


def _dtype_from_name(name: str) -> type[cutlass.Numeric]:
    if name == "fp16":
        return cutlass.Float16
    if name == "bf16":
        return cutlass.BFloat16
    raise ValueError(f"unsupported io dtype: {name}")


def export_variant(args) -> None:
    if args.token_tile not in _TOKEN_TILES:
        raise ValueError(f"token tile must be one of {_TOKEN_TILES}")
    io_dtype = _dtype_from_name(args.io_dtype)
    launch = Nvfp4A16BlackwellGemmLaunch(io_dtype, args.token_tile)
    activation_ptr = make_ptr(
        io_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    qweight_ptr = make_ptr(
        cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    block_scales_ptr = make_ptr(
        cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    global_scale_ptr = make_ptr(
        cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4
    )
    output_ptr = make_ptr(
        io_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    compiled = cute.compile(
        launch,
        activation_ptr,
        qweight_ptr,
        block_scales_ptr,
        global_scale_ptr,
        output_ptr,
        cutlass.Int32(args.token_tile),
        cutlass.Int32(_N_TILE),
        cutlass.Int32(_K_TILE),
        cutlass.Int32(1),
        cuda.CUstream(0),
    )
    os.makedirs(args.output_dir, exist_ok=True)
    compiled.export_to_c(args.output_dir, args.file_name, args.function_prefix)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--io_dtype", choices=("fp16", "bf16"), required=True)
    parser.add_argument("--token_tile", type=int, required=True)
    parser.add_argument("--export_only", action="store_true")
    parser.add_argument("--output_dir", default="./nvfp4_a16_blackwell_gemm_artifacts")
    parser.add_argument("--file_name", default="nvfp4_a16_blackwell_gemm")
    parser.add_argument("--function_prefix", default="nvfp4_a16_blackwell_gemm")
    return parser.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    if not _args.export_only:
        raise ValueError("This SM110 AOT kernel currently supports --export_only")
    export_variant(_args)
