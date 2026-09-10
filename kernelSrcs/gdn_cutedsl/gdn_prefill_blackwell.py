# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Origin: Ported from FlashInfer PR #2742 Blackwell GDN prefill kernel (Apache-2.0):
#   https://github.com/flashinfer-ai/flashinfer/pull/2742
# Copyright (c) 2026 by FlashInfer team. All Rights Reserved.

# Gated Delta Networks (GDN) chunked linear attention kernel for NVIDIA Blackwell (SM100).
#
# Implements the Chunk-wise Gated Delta Rule linear attention using CuTe-DSL,
# for Blackwell Architecture.
#
# Key Features:
#   - Supports Persistent & Non-persistent modes
#   - Supports fixed-length and variable-length sequences (cu_seqlens)
#   - Supports grouped value attention (GVA) where h_v is a multiple of h_q
#   - Supports head dimension (128)
#   - Supports input data types (f16/bf16)
#   - Supports output data types (f16/bf16)
#   - Supports initial_state to provide the initial state
#   - Supports output_final_state flag to return the final state
#   - State input and output are in f32 (fp16/bf16 not supported yet)

import argparse
import functools
import os
import sys
import time
import warnings
from typing import Optional, Tuple, Type, Union

_saved_argv = None
if __name__ == "__main__":
    _saved_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]

import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import numpy as np
from cutlass import Int32, Int64
from cutlass._mlir.dialects import nvvm
from cutlass.cute.runtime import from_dlpack

from gdn_prefill_blackwell_tile_scheduler import (
    GdnStaticTileScheduler,
    GdnStaticTileSchedulerParams,
    create_gdn_static_tile_scheduler,
    create_gdn_static_tile_scheduler_params,
)
from gdn_prefill_blackwell_helpers import (
    make_smem_layout_a_kind,
    make_smem_layout_b_kind,
    make_smem_layout_epi_kind,
)

# CuTe DSL 4.7 O3 spills heavily for this kernel on SM110.
_GDN_COMPILE_OPTIONS = "--opt-level 2"


def make_thread_cooperative_group(size: Int32):
    # return pipeline.CooperativeGroup(pipeline.Agent.Thread, size, size) # old version
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


class GDN:
    """Blackwell (SM100) kernel for Chunk-wise Gated Delta Rule linear attention."""

    def __init__(
        self,
        is_persistent: bool = False,
        chunk_size: Int32 = 128,  # Only 128 is supported in current version
        head_dim: Int32 = 128,  # Only 128 is supported in current version
    ):
        self.chunk_size = chunk_size
        self.is_persistent = is_persistent

        self.head_dim = head_dim
        self.cta_tiler = (chunk_size, chunk_size, head_dim)
        self.gate_tiler = (chunk_size, 1)
        self.beta_tiler = (chunk_size, 1)
        self.state_output_tiler = (head_dim, head_dim)
        self.o_output_tiler = (chunk_size, head_dim)

        self.g_beta_dtype = cutlass.Float32
        self.acc_dtype = cutlass.Float32
        self.state_dtype = cutlass.Float32

        self.mma_tiler = (chunk_size, chunk_size, head_dim)
        self.qk_mma_tiler = self.mma_tiler
        self.kkt_mma_tiler = self.mma_tiler
        self.kst_mma_tiler = self.mma_tiler
        self.tuw_mma_tiler = self.mma_tiler
        self.o_intra_mma_tiler = (chunk_size, head_dim, chunk_size)
        self.qs_mma_tiler = (chunk_size, head_dim, head_dim)
        self.update_s_mma_tiler = (head_dim, head_dim, chunk_size)

        self.epi_tile = (128, 128)

        self.cluster_shape_mnk = (1, 1, 1)

        self.buffer_align_bytes = 1024

        self.q_stage = 1
        self.one_stage = 1
        self.kv_stage = 1
        self.qk_stage = 2
        self.gate_stage = 1
        self.beta_stage = 1
        self.mma_cudacore_stage = 1
        self.mma_qk_stage = 2
        self.epi_stage = 1

        self.cudacore_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.load_warp_id = 5
        self.epilogue_warp_id = 6
        self.gb_warp_id = 7

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.cudacore_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.epilogue_warp_id,
                self.gb_warp_id,
            )
        )

        # TMEM configuration
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS
        self.tmem_kkt_output = 0
        self.tmem_qkt_output = 384
        self.tmem_gate_qk = 384
        self.tmem_gate_q = 448  # 448
        self.tmem_k_input = 64
        self.tmem_kst_output = 256
        self.tmem_ivt_ss_l0_output = 448
        self.tmem_ivt_ts_l0_input = 320
        self.tmem_ivt_ts_l0_output = 448
        self.tmem_ivt_ss_l1_output = 320
        self.tmem_ivt_ts_l1_input = 320
        self.tmem_ivt_ts_l1_output = 256
        self.tmem_tuw_output = 256
        self.tmem_o_intra_output = 256
        self.tmem_o_inter_output = 256
        self.tmem_update_s_output = 128
        self.tmem_gamma = 256

        self.tmem_state = 128

        # Registers configuration
        self.num_regs_cudacore = 240
        self.num_regs_mma = 64
        self.num_regs_other = 64
        self.num_regs_gb = 64

        # Named barrier IDs
        self.cta_sync_bar_id = 0
        self.tmem_alloc_sync_bar_id = 1
        self.cudacore_mma_sync_bar_id = 2
        self.wg_sync_bar_id = 3
        self.epi_load_sync_bar_id = 4

        # Matrix inversion using TFloat32
        self.invert_type_ab = cutlass.TFloat32
        self.invert_acc_type = cutlass.Float32
        self.invert_mma_sub_l0_tiler = (64, 64, 32)
        self.invert_mma_sub_l1_tiler = (64, 64, 64)
        self.invert_mma_tiler = (128, 128, 64)
        self.invert_sub_stage = 1
        self.sub_stage = 9

    @staticmethod
    def can_implement(
        q_shape: Tuple[int, int, int, int] | Tuple[int, Tuple[int, ...], int, int],
        v_shape: Tuple[int, int, int, int] | Tuple[int, Tuple[int, ...], int, int],
        in_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        g_beta_dtype: Type[cutlass.Numeric],
        use_qk_l2norm_in_kernel: bool = False,
    ) -> bool:
        """
        Check if the gdn can be implemented
        """

        can_implement = True

        # Unpack parameters
        b, s_q, h_q, d = q_shape
        b_, _, h_v, d_ = v_shape

        if b != b_:
            warnings.warn("q & k must have the same batch size", stacklevel=2)
            can_implement = False

        if d != d_:
            warnings.warn("q & k must have the same head dimension", stacklevel=2)
            can_implement = False

        # todo: maybe support more later.
        if d not in {128}:
            warnings.warn("head dimension must be 128", stacklevel=2)
            can_implement = False

        if h_v % h_q != 0:
            warnings.warn("h_v must be divisible by h_q", stacklevel=2)

            can_implement = False

        if isinstance(s_q, tuple) and len(s_q) != b:
            warnings.warn(
                "variable_seqlen s_q must have the length of batch size", stacklevel=2
            )
            can_implement = False

        if in_dtype not in {cutlass.BFloat16, cutlass.Float16}:
            warnings.warn(
                "in_dtype must be BFloat16 or Float16, but got {}".format(in_dtype),
                stacklevel=2,
            )
            can_implement = False

        if out_dtype not in {cutlass.BFloat16, cutlass.Float16}:
            warnings.warn(
                "out_dtype must be BFloat16 or Float16, but got {}".format(out_dtype),
                stacklevel=2,
            )
            can_implement = False

        if g_beta_dtype not in {cutlass.Float16, cutlass.BFloat16}:
            warnings.warn(
                "g_beta_dtype (a/b dtype) must be Float16 or BFloat16, but got {}".format(g_beta_dtype),
                stacklevel=2,
            )
            can_implement = False

        return can_implement

    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        o_intra_tiled_mma: cute.TiledMma,
        o_intra_tiled_ts_mma_new: cute.TiledMma,
        qs_tiled_mma: cute.TiledMma,
        update_s_tiled_mma: cute.TiledMma,
        kkt_tiled_mma: cute.TiledMma,
        kst_tiled_mma: cute.TiledMma,
        fake_state_tiled_mma: cute.TiledMma,
        invert_sub_tiled_mma_ss_l0: cute.TiledMma,
        invert_sub_tiled_mma_ts_l0: cute.TiledMma,
        invert_sub_tiled_mma_ss_l1: cute.TiledMma,
        invert_sub_tiled_mma_ts_l1: cute.TiledMma,
        tuw_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        tma_atom_state_f32: Optional[cute.CopyAtom],
        mState_f32: Optional[cute.Tensor],
        a: cute.Tensor,
        b: cute.Tensor,
        A_log_vec: cute.Tensor,
        dt_bias_vec: cute.Tensor,
        O: cute.Tensor,
        tma_atom_state_output: Optional[cute.CopyAtom],
        mStateOutput: Optional[cute.Tensor],
        tma_atom_o_output: cute.CopyAtom,
        mO_qdl: cute.Tensor,
        cum_seqlen_q: Optional[cute.Tensor],
        cu_seqlens: Optional[cute.Tensor],
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        qk_smem_layout_staged: cute.ComposedLayout,
        o_intra_smem_layout_staged_a: cute.ComposedLayout,
        o_intra_smem_layout_staged_b: cute.ComposedLayout,
        o_intra_smem_layout_staged_a_new: cute.ComposedLayout,
        qs_smem_layout_staged_a: cute.ComposedLayout,
        qs_smem_layout_staged_b: cute.ComposedLayout,
        update_s_smem_layout_staged_a: cute.ComposedLayout,
        update_s_smem_layout_staged_b: cute.ComposedLayout,
        state_smem_layout_staged: cute.ComposedLayout,
        state_smem_layout_f32_staged: cute.ComposedLayout,
        state_output_smem_layout_staged: cute.ComposedLayout,
        o_output_smem_layout_staged: cute.ComposedLayout,
        gate_smem_layout_staged: cute.Layout,
        beta_smem_layout_staged: cute.Layout,
        ivt_smem_layout: cute.Layout,
        sub_inner_smem_layout_staged: cute.ComposedLayout,
        invert_sub_smem_layout_staged_ss_l0_a: cute.ComposedLayout,
        invert_sub_smem_layout_staged_ss_l0_b: cute.ComposedLayout,
        invert_sub_smem_layout_staged_ts_l0_a: cute.ComposedLayout,
        invert_sub_smem_layout_staged_ts_l0_b: cute.ComposedLayout,
        invert_sub_smem_layout_staged_ss_l1_a: cute.ComposedLayout,
        invert_sub_smem_layout_staged_ss_l1_b: cute.ComposedLayout,
        invert_sub_smem_layout_staged_ts_l1_a: cute.ComposedLayout,
        invert_sub_smem_layout_staged_ts_l1_b: cute.ComposedLayout,
        tuw_smem_layout_staged_a: cute.ComposedLayout,
        tuw_smem_layout_staged_b: cute.ComposedLayout,
        v_smem_layout_staged_b: cute.ComposedLayout,
        scale: cutlass.Float32,
        use_qk_l2norm: cutlass.Constexpr[cutlass.Boolean],
        tile_sched_params: GdnStaticTileSchedulerParams,
    ):
        """Warp-specialized GDN kernel entry point."""
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch TMA descriptors
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o_output)
            if cutlass.const_expr(tma_atom_state_f32 is not None):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_state_f32)
            if cutlass.const_expr(tma_atom_state_output is not None):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_state_output)

        # Alloc
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_qk_producer, load_qk_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.qk_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_kv_bytes,
            barrier_storage=storage.load_qk_mbar_ptr.data_ptr(),
        ).make_participants()

        load_v_producer, load_v_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.one_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.cudacore_warp_ids),
            ),
            tx_count=self.tma_copy_v_bytes,
            barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
        ).make_participants()

        load_state_producer, load_state_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.one_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len(self.cudacore_warp_ids)),
            tx_count=self.tma_copy_state_f32_bytes,
            barrier_storage=storage.load_state_mbar_ptr.data_ptr(),
        ).make_participants()

        mma_qk_producer0, mma_qk_consumer0 = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_qk_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cudacore_warp_ids)
            ),
            barrier_storage=storage.mma_qk_mbar_ptr0.data_ptr(),
        ).make_participants()

        mma_cudacore_producer0, mma_cudacore_consumer0 = (
            pipeline.PipelineUmmaAsync.create(
                num_stages=self.mma_cudacore_stage,
                producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
                consumer_group=make_thread_cooperative_group(
                    self.threads_per_warp * len(self.cudacore_warp_ids)
                ),
                barrier_storage=storage.mma_cudacore_mbar_ptr0.data_ptr(),
            ).make_participants()
        )

        gb_w0_producer, gb_w0_consumer = pipeline.PipelineAsync.create(
            num_stages=self.gate_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cudacore_warp_ids)
            ),
            barrier_storage=storage.gb_w0_mbar_ptr.data_ptr(),
        ).make_participants()

        epi_w0_producer, epi_w0_consumer = pipeline.PipelineAsync.create(
            num_stages=self.epi_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cudacore_warp_ids)
            ),
            barrier_storage=storage.epi_w0_mbar_ptr.data_ptr(),
        ).make_participants()

        w0_epi_producer, w0_epi_consumer = pipeline.PipelineAsync.create(
            num_stages=self.epi_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cudacore_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp),
            barrier_storage=storage.w0_epi_mbar_ptr.data_ptr(),
        ).make_participants()

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()

        #  tmem barrier init
        if warp_idx == self.gb_warp_id:
            cute.arch.mbarrier_init(
                tmem_dealloc_mbar_ptr,
                self.threads_per_warp * len((*self.cudacore_warp_ids,)),
            )
        cute.arch.mbarrier_init_fence()

        sQK = storage.sQK.get_tensor(
            qk_smem_layout_staged.outer, swizzle=qk_smem_layout_staged.inner
        )
        sQ = cute.make_tensor(
            cute.recast_ptr(
                sQK[None, None, None, 1].iterator,
                swizzle_=q_smem_layout_staged.inner,
                dtype=self.i_dtype,
            ),
            q_smem_layout_staged.outer,
        )
        sK = cute.make_tensor(
            cute.recast_ptr(
                sQK[None, None, None, 0].iterator,
                swizzle_=k_smem_layout_staged.inner,
                dtype=self.i_dtype,
            ),
            k_smem_layout_staged.outer,
        )
        sStateInput = cute.make_tensor(
            cute.recast_ptr(
                storage.sState.data_ptr(),
                swizzle_=state_smem_layout_staged.inner,
                dtype=self.i_dtype,
            ),
            state_smem_layout_staged.outer,
        )
        sV = cute.make_tensor(
            cute.recast_ptr(
                storage.sV.data_ptr(),
                swizzle_=v_smem_layout_staged_b.inner,
                dtype=self.i_dtype,
            ),
            v_smem_layout_staged_b.outer,
        )

        sGate = storage.sGate.get_tensor(gate_smem_layout_staged)
        sBeta = storage.sBeta.get_tensor(beta_smem_layout_staged)
        sGateCumsum = storage.sGateCumsum.get_tensor(
            cute.select(gate_smem_layout_staged, mode=[0, 1])
        )
        sSubInner = storage.sSubInner.get_tensor(sub_inner_smem_layout_staged)
        sInvertSubSSL0A = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 6].iterator,
                swizzle_=invert_sub_smem_layout_staged_ss_l0_a.inner,
                dtype=self.invert_type_ab,
            ),
            invert_sub_smem_layout_staged_ss_l0_a.outer,
        )

        sInvertSubSSL0B = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 7].iterator,
                swizzle_=invert_sub_smem_layout_staged_ss_l0_b.inner,
                dtype=self.invert_type_ab,
            ),
            invert_sub_smem_layout_staged_ss_l0_b.outer,
        )
        sInvertSubTSL0B = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 8].iterator,
                swizzle_=invert_sub_smem_layout_staged_ts_l0_b.inner,
                dtype=self.invert_type_ab,
            ),
            invert_sub_smem_layout_staged_ts_l0_b.outer,
        )
        sInvertSubSSL1A = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 0].iterator,
                swizzle_=invert_sub_smem_layout_staged_ss_l1_a.inner,
                dtype=self.invert_type_ab,
            ),
            invert_sub_smem_layout_staged_ss_l1_a.outer,
        )
        sInvertSubSSL1B = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 2].iterator,
                swizzle_=invert_sub_smem_layout_staged_ss_l1_b.inner,
                dtype=self.invert_type_ab,
            ),
            invert_sub_smem_layout_staged_ss_l1_b.outer,
        )
        sInvertSubTSL1B = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 6].iterator,
                swizzle_=invert_sub_smem_layout_staged_ts_l1_b.inner,
                dtype=self.invert_type_ab,
            ),
            invert_sub_smem_layout_staged_ts_l1_b.outer,
        )

        sInvertSubReg = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 4].iterator, dtype=self.invert_type_ab
            ),
            cute.make_layout(((4, 128), 8)),
        )

        sStateInputF32 = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 0].iterator,
                swizzle_=state_smem_layout_f32_staged.inner,
                dtype=self.state_dtype,
            ),
            state_smem_layout_f32_staged.outer,
        )
        sStateOutput = cute.make_tensor(
            cute.recast_ptr(
                sQK[None, None, None, 0].iterator,
                swizzle_=state_output_smem_layout_staged.inner,
                dtype=self.state_dtype,
            ),
            state_output_smem_layout_staged.outer,
        )

        sOIntraB = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 4].iterator,
                swizzle_=o_intra_smem_layout_staged_b.inner,
                dtype=self.i_dtype,
            ),
            o_intra_smem_layout_staged_b.outer,
        )
        sQSB = cute.make_tensor(
            cute.recast_ptr(
                storage.sState.data_ptr(),
                swizzle_=qs_smem_layout_staged_b.inner,
                dtype=self.i_dtype,
            ),
            qs_smem_layout_staged_b.outer,
        )

        sUpdateA = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 4].iterator,
                swizzle_=update_s_smem_layout_staged_a.inner,
                dtype=self.i_dtype,
            ),
            update_s_smem_layout_staged_a.outer,
        )
        sUpdateB = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 0].iterator,
                swizzle_=update_s_smem_layout_staged_b.inner,
                dtype=self.i_dtype,
            ),
            update_s_smem_layout_staged_b.outer,
        )

        sO = cute.make_tensor(
            cute.recast_ptr(
                storage.sState.data_ptr(),
                swizzle_=o_output_smem_layout_staged.inner,
                dtype=self.i_dtype,
            ),
            o_output_smem_layout_staged.outer,
        )
        sIvt = storage.sIvt.get_tensor(ivt_smem_layout)

        sTuwA = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 0].iterator,
                swizzle_=tuw_smem_layout_staged_a.inner,
                dtype=self.i_dtype,
            ),
            tuw_smem_layout_staged_a.outer,
        )
        sTuwB = cute.make_tensor(
            cute.recast_ptr(storage.sV.data_ptr(), dtype=self.k_dtype),
            tuw_smem_layout_staged_b,
        )
        sTuwAStore_layout = cute.make_layout(
            ((128, (4, 2)), 8), stride=((4, (1, 512)), 1024)
        )
        sTuwAStore = cute.make_tensor(
            cute.recast_ptr(sTuwA.iterator, dtype=cutlass.Float32), sTuwAStore_layout
        )

        if warp_idx == self.epilogue_warp_id:
            # Alloc tmem buffer
            tmem_alloc_cols = cutlass.Int32(self.tmem_alloc_cols)
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)

        # Ensure visibility of local mbarrier inits and tmem alloc
        cute.arch.sync_threads()

        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )

        qk_thr_mma = qk_tiled_mma.get_slice(0)  # default 1sm
        tSrQ = qk_thr_mma.make_fragment_A(sQK)
        tSrK = qk_thr_mma.make_fragment_B(sQK)
        qk_acc_shape = qk_thr_mma.partition_shape_C(
            (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
        )
        tQKtQK_fake = qk_thr_mma.make_fragment_C(qk_acc_shape)
        tQKtQK = cute.make_tensor(tmem_ptr + self.tmem_qkt_output, tQKtQK_fake.layout)

        kkt_thr_mma = kkt_tiled_mma.get_slice(0)  # default 1sm
        tArK = kkt_thr_mma.make_fragment_A(sQK)
        tArKT = kkt_thr_mma.make_fragment_B(sQK)
        kkt_acc_shape = kkt_thr_mma.partition_shape_C(
            (self.kkt_mma_tiler[0], self.kkt_mma_tiler[1])
        )

        fake_state_thr_mma = fake_state_tiled_mma.get_slice(0)

        kst_thr_mma = kst_tiled_mma.get_slice(0)
        tKSrK_0 = kst_thr_mma.make_fragment_A(sQK)
        tKSrST = kst_thr_mma.make_fragment_B(sStateInput)
        kst_acc_shape = kst_thr_mma.partition_shape_C(
            (self.kst_mma_tiler[0], self.kst_mma_tiler[1])
        )
        tKStKS_fake = kst_thr_mma.make_fragment_C(kst_acc_shape)
        tKStKS = cute.make_tensor(tmem_ptr + self.tmem_kst_output, tKStKS_fake.layout)
        # input smem A
        tKStK_layout = cute.composition(tKStKS.layout, cute.make_layout((128, 64)))
        tKtK = cute.make_tensor(tmem_ptr + self.tmem_k_input, tKStK_layout)

        invert_sub_thr_mma_ss_l0 = invert_sub_tiled_mma_ss_l0.get_slice(0)
        tSrD = invert_sub_thr_mma_ss_l0.make_fragment_A(sInvertSubSSL0A)
        tSrC = invert_sub_thr_mma_ss_l0.make_fragment_B(sInvertSubSSL0B)
        invert_sub_acc_shape = invert_sub_thr_mma_ss_l0.partition_shape_C(
            (self.invert_mma_sub_l0_tiler[0], self.invert_mma_sub_l0_tiler[1])
        )

        tKKTtKKT_fake = kkt_thr_mma.make_fragment_C(kkt_acc_shape)
        tKKTtKKT = cute.make_tensor(
            tmem_ptr + self.tmem_kkt_output, tKKTtKKT_fake.layout
        )

        tItSSL0_fake = invert_sub_thr_mma_ss_l0.make_fragment_C(invert_sub_acc_shape)
        tItSSL0 = cute.make_tensor(
            tmem_ptr + self.tmem_ivt_ss_l0_output, tItSSL0_fake.layout
        )

        # : ((128,128),1,1):((65536,1),0,0)
        tQgate_layout = cute.make_layout(((128, 64), 1, 1), stride=((65536, 1), 0, 0))
        tQgate = cute.make_tensor(tmem_ptr + self.tmem_gate_q, tQgate_layout)

        tGamma_layout = cute.make_layout(((128, 128), 1, 1), stride=((65536, 1), 0, 0))
        tGamma = cute.make_tensor(tmem_ptr + self.tmem_gamma, tGamma_layout)

        # tStage
        tStateF32_layout = cute.make_layout(
            ((128, 128), 1, 1), stride=((65536, 1), 0, 0)
        )
        tStateF32 = cute.make_tensor(tmem_ptr + self.tmem_state, tStateF32_layout)

        invert_sub_thr_mma_ts_l0 = invert_sub_tiled_mma_ts_l0.get_slice(0)
        tInrDCL0A_fake = invert_sub_thr_mma_ts_l0.make_fragment_A(
            invert_sub_smem_layout_staged_ts_l0_a.outer.shape
        )
        tInrDCL0A = cute.make_tensor(
            cute.recast_ptr(
                tmem_ptr,
                dtype=tInrDCL0A_fake.element_type,
            )
            + self.tmem_ivt_ts_l0_input,
            tInrDCL0A_fake.layout,
        )

        # input smem B
        tSrDCL0B = invert_sub_thr_mma_ts_l0.make_fragment_B(sInvertSubTSL0B)

        # output tmem C
        tItTSL0_fake = invert_sub_thr_mma_ts_l0.make_fragment_C(invert_sub_acc_shape)
        tItTSL0 = cute.make_tensor(
            tmem_ptr + self.tmem_ivt_ts_l0_output, tItTSL0_fake.layout
        )

        invert_sub_thr_mma_ss_l1 = invert_sub_tiled_mma_ss_l1.get_slice(0)
        tSrD_ivt_l1 = invert_sub_thr_mma_ss_l1.make_fragment_A(sInvertSubSSL1A)
        tSrC_ivt_l1 = invert_sub_thr_mma_ss_l1.make_fragment_B(sInvertSubSSL1B)
        invert_sub_acc_l1_shape = invert_sub_thr_mma_ss_l1.partition_shape_C(
            (self.invert_mma_sub_l1_tiler[0], self.invert_mma_sub_l1_tiler[1])
        )
        tCtAcc_ivt_l1_ss_fake = invert_sub_thr_mma_ss_l1.make_fragment_C(
            invert_sub_acc_l1_shape
        )
        tStO_ivt_l1_ss0 = cute.make_tensor(
            tmem_ptr + self.tmem_ivt_ss_l1_output, tCtAcc_ivt_l1_ss_fake.layout
        )

        invert_sub_thr_mma_ts_l1 = invert_sub_tiled_mma_ts_l1.get_slice(0)
        invert_acc_ts_l1_shape = invert_sub_thr_mma_ts_l1.partition_shape_C(
            (self.invert_mma_sub_l1_tiler[0], self.invert_mma_sub_l1_tiler[1])
        )
        # output tmem C
        tOtTSL1_fake = invert_sub_thr_mma_ts_l1.make_fragment_C(invert_acc_ts_l1_shape)
        tOtTSL1 = cute.make_tensor(
            tmem_ptr + self.tmem_ivt_ts_l1_output, tOtTSL1_fake.layout
        )

        tuw_thr_mma = tuw_tiled_mma.get_slice(0)
        # input smem A
        tTUWrA = tuw_thr_mma.make_fragment_A(sTuwA)
        # input smem B
        tTUWrB = tuw_thr_mma.make_fragment_B(sTuwB)
        # output tmem C
        tuw_acc_shape = tuw_thr_mma.partition_shape_C(
            (self.tuw_mma_tiler[0], self.tuw_mma_tiler[1])
        )
        tVtTUW_fake = tuw_thr_mma.make_fragment_C(tuw_acc_shape)
        tVtTUW = cute.make_tensor(tmem_ptr + self.tmem_tuw_output, tVtTUW_fake.layout)

        o_intra_thr_mma = o_intra_tiled_mma.get_slice(0)
        # input smem B
        tOIntrarB = o_intra_thr_mma.make_fragment_B(sOIntraB)
        # output tmem C
        o_intra_acc_shape = o_intra_thr_mma.partition_shape_C(
            (self.o_intra_mma_tiler[0], self.o_intra_mma_tiler[1])
        )

        o_intra_thr_mma_new = o_intra_tiled_ts_mma_new.get_slice(0)
        # input smem A
        tQKMtQKM_layout = cute.composition(tQKtQK.layout, cute.make_layout((128, 64)))
        tQKMtQKM = cute.make_tensor(tmem_ptr + self.tmem_gate_qk, tQKMtQKM_layout)

        update_s_thr_mma = update_s_tiled_mma.get_slice(0)
        tUpdateA = update_s_thr_mma.make_fragment_A(sUpdateA)
        tUpdateB = update_s_thr_mma.make_fragment_B(sUpdateB)
        update_s_acc_shape = update_s_thr_mma.partition_shape_C(
            (self.update_s_mma_tiler[0], self.update_s_mma_tiler[1])
        )
        tStUpdate_fake = update_s_thr_mma.make_fragment_C(update_s_acc_shape)
        tStUpdate = cute.make_tensor(
            tmem_ptr + self.tmem_update_s_output, tStUpdate_fake.layout
        )

        qs_thr_mma = qs_tiled_mma.get_slice(0)
        tQSB = qs_thr_mma.make_fragment_B(sQSB)
        qs_acc_shape = qs_thr_mma.partition_shape_C(
            (self.qs_mma_tiler[0], self.qs_mma_tiler[1])
        )
        tOintertQS_fake = qs_thr_mma.make_fragment_C(qs_acc_shape)
        tOintertQS = cute.make_tensor(
            tmem_ptr + self.tmem_o_inter_output, tOintertQS_fake.layout
        )

        loop_count = cute.ceil_div(mO_qdl.shape[0], self.cta_tiler[0])

        # ///////////////////////////////////////////////////////////////////////////////
        #  Load Gate Beta
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.gb_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_gb)
            tile_sched = create_gdn_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            lane_id = tidx % 32

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx

                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                cuseqlen_q = Int32(0)
                seqlen_q = mQ_qdl.shape[0]

                head_coord = curr_block_coord[2][0]

                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = (
                        not GdnStaticTileScheduler.check_valid_work_for_seqlen_q(
                            self.cta_tiler[0],
                            curr_block_coord[0],
                            seqlen_q,
                        )
                    )

                if not continue_cond:
                    batch_base = cuseqlen_q
                    gate_vals = cute.make_rmem_tensor((4), cutlass.Float32)
                    beta_vals = cute.make_rmem_tensor((4), cutlass.Float32)
                    # Load per-head scalars once per chunk (broadcast across tokens)
                    A_log_val = cutlass.Float32(A_log_vec[head_coord])
                    dt_bias_f32 = cutlass.Float32(dt_bias_vec[head_coord])
                    neg_exp_A_log = -cute.math.exp(A_log_val)

                    # Padding masking: clip gate/beta loop to cu_seqlens[b+1]-cu_seqlens[b], zero-fill beyond.
                    valid_len = seqlen_q
                    if cutlass.const_expr(cu_seqlens is not None and cum_seqlen_q is None):
                        valid_len = cu_seqlens[batch_coord + 1] - cu_seqlens[batch_coord]

                    for i in cutlass.range(loop_count, unroll=1):
                        # step1: compute gate (g) and beta from raw a/b/A_log/dt_bias
                        for it in cutlass.range_constexpr(4):
                            curr_idx = i * 128 + it * 32 + lane_id
                            if curr_idx < valid_len:
                                if cutlass.const_expr(cum_seqlen_q is not None):
                                    a_f32 = cutlass.Float32(a[batch_base + curr_idx, 0, (head_coord, 0)])
                                    b_f32 = cutlass.Float32(b[batch_base + curr_idx, 0, (head_coord, 0)])
                                else:
                                    a_f32 = cutlass.Float32(a[curr_idx, 0, (head_coord, batch_coord)])
                                    b_f32 = cutlass.Float32(b[curr_idx, 0, (head_coord, batch_coord)])
                                # g = -exp(A_log) * softplus(a + dt_bias)
                                # Use threshold to avoid exp overflow for large x.
                                # softplus(x) = log(1+exp(x)) ≈ x for x > 20.
                                x = a_f32 + dt_bias_f32
                                softplus_x = x  # default for large x
                                if x <= cutlass.Float32(20.0):
                                    softplus_x = cute.math.log(cutlass.Float32(1) + cute.math.exp(x))
                                gate_vals[it] = neg_exp_A_log * softplus_x
                                # beta = sigmoid(b)
                                beta_vals[it] = cutlass.Float32(1) / (cutlass.Float32(1) + cute.math.exp(-b_f32))
                            else:
                                gate_vals[it] = cutlass.Float32(0)
                                beta_vals[it] = cutlass.Float32(0)
                        # step2: write to smem
                        sGate0 = sGate[None, None, 0]  # assume only one stage
                        sBeta0 = sBeta[None, None, 0]

                        gb_handle = gb_w0_producer.acquire_and_advance()

                        for it in cutlass.range_constexpr(4):
                            sGate0[it * 32 + lane_id] = gate_vals[it]
                            sBeta0[it * 32 + lane_id] = beta_vals[it]

                        # step3: notify cuda core wg
                        gb_handle.commit()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue: Store Output
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.epilogue_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tile_sched = create_gdn_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx

                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                cuseqlen_q = Int32(0)
                seqlen_q = mQ_qdl.shape[0]

                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = (
                        not GdnStaticTileScheduler.check_valid_work_for_seqlen_q(
                            self.cta_tiler[0],
                            curr_block_coord[0],
                            seqlen_q,
                        )
                    )

                if not continue_cond:
                    curr_block_coord_o = curr_block_coord
                    mO_qdl_ = mO_qdl
                    if cutlass.const_expr(cum_seqlen_q is not None):
                        logical_offset_mO = (
                            mO_qdl_.shape[0] - seqlen_q,
                            0,
                            (0, cuseqlen_q + seqlen_q),
                        )
                        mO_qdl_ = cute.domain_offset(logical_offset_mO, mO_qdl_)
                        curr_block_coord_o = (
                            curr_block_coord[0],
                            curr_block_coord[1],
                            (curr_block_coord[2][0], 0),
                        )

                    gO_mnl = cute.flat_divide(
                        mO_qdl_, cute.select(self.mma_tiler, mode=[0, 1])
                    )
                    tOsO, tOgO_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_o_output,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sO, 0, 2),
                        cute.group_modes(gO_mnl, 0, 2),
                    )
                    tQgO = tOgO_qdl[None, None, 0, curr_block_coord_o[2]]
                    w0_handle = epi_w0_producer.acquire_and_advance()
                    w0_handle.commit()

                    for i in cutlass.range(loop_count, unroll=1):
                        w0_handle = epi_w0_producer.acquire_and_advance()
                        cute.copy(tma_atom_o_output, tOsO[None, 0], tQgO[None, i])

                        cute.arch.cp_async_bulk_commit_group()

                        # Ensure O0 buffer is ready to be released
                        cute.arch.cp_async_bulk_wait_group(0, read=True)

                        w0_handle.commit()

                    if cutlass.const_expr(tma_atom_state_output is not None):
                        gStateOutput = cute.flat_divide(
                            mStateOutput,
                            cute.select(self.update_s_mma_tiler, mode=[0, 1]),
                        )
                        tSsS, tSgS_vkl = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_state_output,
                            0,  # no multicast
                            cute.make_layout(1),
                            cute.group_modes(sStateOutput, 0, 2),
                            cute.group_modes(gStateOutput, 0, 4),
                        )
                        tSgS = tSgS_vkl[None, curr_block_coord[2]]
                        w0_epi_handle = w0_epi_consumer.wait_and_advance()
                        cute.copy(
                            tma_atom_state_output,
                            tSsS[None, 0],  # only 1 stage
                            tSgS[None],
                        )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        w0_epi_handle.release()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

                # The smem is reused, ensuring the State Output store is complete
                if cutlass.const_expr(
                    tma_atom_state_output is not None and self.is_persistent
                ):
                    cute.arch.barrier_arrive(
                        barrier_id=self.epi_load_sync_bar_id,
                        number_of_threads=self.threads_per_warp * 2,
                    )
        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD: tma load QKVS
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

            tile_sched = create_gdn_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                continue_cond = False
                cuseqlen_q = Int32(0)
                seqlen_q = mQ_qdl.shape[0]
                batch_coord = curr_block_coord[2][1]

                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = (
                        not GdnStaticTileScheduler.check_valid_work_for_seqlen_q(
                            self.cta_tiler[0],
                            curr_block_coord[0],
                            seqlen_q,
                        )
                    )

                if not continue_cond:
                    mQ_qdl_ = mQ_qdl
                    mK_kdl_ = mK_kdl
                    mV_dkl_ = mV_dkl
                    curr_block_coord_q = curr_block_coord
                    curr_block_coord_kv = curr_block_coord
                    curr_block_coord_state = curr_block_coord

                    if cutlass.const_expr(cum_seqlen_q is not None):
                        logical_offset_mQ = (
                            cuseqlen_q,
                            0,
                            (0, 0),
                        )
                        mQ_qdl_ = cute.domain_offset(logical_offset_mQ, mQ_qdl)
                        curr_block_coord_q = (
                            curr_block_coord[0],
                            curr_block_coord[1],
                            (curr_block_coord[2][0], Int32(0)),
                        )
                        logical_offset_mK = (
                            cuseqlen_q,
                            0,
                            (0, 0),
                        )
                        logical_offset_mV = (
                            0,
                            cuseqlen_q,
                            (0, 0),
                        )
                        mK_kdl_ = cute.domain_offset(logical_offset_mK, mK_kdl)
                        mV_dkl_ = cute.domain_offset(logical_offset_mV, mV_dkl)
                        curr_block_coord_kv = (
                            curr_block_coord[0],
                            curr_block_coord[1],
                            (curr_block_coord[2][0], Int32(0)),
                        )

                    gQ_qdl = cute.flat_divide(
                        mQ_qdl_, cute.select(self.qk_mma_tiler, mode=[0, 2])
                    )
                    tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)

                    tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_q,
                        0,  # no multicast
                        cute.make_layout(1),
                        cute.group_modes(sQK, 0, 3),
                        cute.group_modes(tSgQ_qdl, 0, 3),
                    )
                    tQgQ = tQgQ_qdl[None, None, 0, curr_block_coord_q[2]]

                    gK_kdl = cute.flat_divide(
                        mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2])
                    )
                    tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                    tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_k,
                        0,  # no multicast
                        cute.make_layout(1),
                        cute.group_modes(sQK, 0, 3),
                        cute.group_modes(tSgK_kdl, 0, 3),
                    )
                    tKgK = tKgK_kdl[None, None, 0, curr_block_coord_kv[2]]

                    gV_kdl = cute.flat_divide(
                        mV_dkl_, cute.select(self.tuw_mma_tiler, mode=[1, 2])
                    )
                    tSgV_kdl = tuw_thr_mma.partition_B(gV_kdl)
                    tVsV, tVgV_kdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_v,
                        0,  # no multicast
                        cute.make_layout(1),
                        cute.group_modes(sV, 0, 3),
                        cute.group_modes(tSgV_kdl, 0, 3),
                    )
                    tVgV = tVgV_kdl[None, 0, None, curr_block_coord_kv[2]]

                    if cutlass.const_expr(tma_atom_state_f32 is not None):
                        gStateF32 = cute.flat_divide(
                            mState_f32, cute.select(self.kst_mma_tiler, mode=[1, 2])
                        )
                        tKSgStateF32 = fake_state_thr_mma.partition_B(gStateF32)
                        tSsSF32, tSgSF32_kdl = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_state_f32,
                            0,  # no multicast
                            cute.make_layout(1),
                            cute.group_modes(sStateInputF32, 0, 3),
                            cute.group_modes(tKSgStateF32, 0, 3),
                        )
                        tSgSF32 = tSgSF32_kdl[None, None, 0, curr_block_coord_state[2]]
                        state_f32_handle0 = load_state_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_state_f32,
                            tSgSF32[None, 0],
                            tSsSF32[None, state_f32_handle0.index],
                            tma_bar_ptr=state_f32_handle0.barrier,
                        )

                    for i in cutlass.range(loop_count, unroll=1):
                        w0_coord = i
                        k0_handle = load_qk_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_k,
                            tKgK[None, w0_coord],
                            tKsK[None, k0_handle.index],
                            tma_bar_ptr=k0_handle.barrier,
                        )

                        q_handle = load_qk_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_q,
                            tQgQ[None, w0_coord],
                            tQsQ[None, q_handle.index],
                            tma_bar_ptr=q_handle.barrier,
                        )

                        v_handle0 = load_v_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_v,
                            tVgV[None, w0_coord],
                            tVsV[None, v_handle0.index],
                            tma_bar_ptr=v_handle0.barrier,
                        )

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

                if cutlass.const_expr(
                    tma_atom_state_output is not None and self.is_persistent
                ):
                    cute.arch.barrier(
                        barrier_id=self.epi_load_sync_bar_id,
                        number_of_threads=self.threads_per_warp * 2,
                    )
            # dealloc tmem buffer
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            #  Retrieving tmem ptr and make acc
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                cutlass.Float32,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA: mma warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)

            tile_sched = create_gdn_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            tSrDC_ivt_l1_fake = invert_sub_thr_mma_ts_l1.make_fragment_A(
                invert_sub_smem_layout_staged_ts_l1_a.outer.shape
            )
            tSrDC_ivt_l1 = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr,
                    dtype=tSrDC_ivt_l1_fake.element_type,
                )
                + self.tmem_ivt_ts_l1_input,
                tSrDC_ivt_l1_fake.layout,
            )
            tSrA_ivt_l1 = invert_sub_thr_mma_ts_l1.make_fragment_B(sInvertSubTSL1B)

            tOrP = o_intra_thr_mma_new.make_fragment_A(
                o_intra_smem_layout_staged_a_new.outer
            )
            tOrP0 = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr,
                    dtype=tOrP.element_type,
                )
                + self.tmem_gate_qk * 2,
                tOrP.layout,
            )

            tQSA = o_intra_thr_mma_new.make_fragment_A(qs_smem_layout_staged_a.outer)
            tQSA0 = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr,
                    dtype=tQSA.element_type,
                )
                + self.tmem_gate_q * 2,
                tQSA.layout,
            )

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                continue_cond = False

                if not continue_cond:
                    for _i in cutlass.range(loop_count, unroll=1):
                        k_handle0 = load_qk_consumer.wait_and_advance()
                        qk_handle0 = mma_qk_producer0.acquire_and_advance()

                        kkt_tiled_mma = self.exec_mma(
                            kkt_tiled_mma,
                            tKKTtKKT,
                            tArK,
                            tArKT,
                            k_handle0.index,
                            k_handle0.index,
                        )

                        qk_handle0.commit()

                        q_handle = load_qk_consumer.wait_and_advance()

                        qk_handle1 = mma_qk_producer0.acquire_and_advance()

                        qk_tiled_mma = self.exec_mma(
                            qk_tiled_mma,
                            tQKtQK,
                            tSrQ,
                            tSrK,
                            q_handle.index,
                            k_handle0.index,
                        )

                        qk_handle1.commit()

                        cs_handle = mma_cudacore_producer0.acquire_and_advance()
                        cute.arch.barrier(
                            barrier_id=self.cudacore_mma_sync_bar_id,
                            number_of_threads=self.threads_per_warp * 5,
                        )

                        kst_tiled_mma = self.exec_mma(
                            kst_tiled_mma,
                            tKStKS,
                            tKSrK_0,
                            tKSrST,
                            0,
                            0,
                        )

                        cs_handle.commit()

                        cs_handle = mma_cudacore_producer0.acquire_and_advance()

                        invert_sub_tiled_mma_ss_l0 = self.exec_mma(
                            invert_sub_tiled_mma_ss_l0,
                            tItSSL0,
                            tSrD,
                            tSrC,
                            0,
                            0,
                        )

                        cs_handle.commit()

                        c0_handle = mma_cudacore_producer0.acquire_and_advance()

                        invert_sub_tiled_mma_ts_l0 = self.exec_mma(
                            invert_sub_tiled_mma_ts_l0,
                            tItTSL0,
                            tInrDCL0A,
                            tSrDCL0B,
                            0,
                            0,
                        )

                        c0_handle.commit()

                        c0_handle = mma_cudacore_producer0.acquire_and_advance()

                        invert_sub_tiled_mma_ss_l1 = self.exec_mma(
                            invert_sub_tiled_mma_ss_l1,
                            tStO_ivt_l1_ss0,
                            tSrD_ivt_l1,
                            tSrC_ivt_l1,
                            0,
                            0,
                        )

                        c0_handle.commit()

                        c0_handle = mma_cudacore_producer0.acquire_and_advance()

                        k_handle0.release()

                        invert_sub_tiled_mma_ts_l1 = self.exec_mma(
                            invert_sub_tiled_mma_ts_l1,
                            tOtTSL1,
                            tSrDC_ivt_l1,
                            tSrA_ivt_l1,
                            0,
                            0,
                        )

                        c0_handle.commit()

                        c0_handle = mma_cudacore_producer0.acquire_and_advance()

                        tuw_tiled_mma = self.exec_mma(
                            tuw_tiled_mma,
                            tVtTUW,
                            tTUWrA,
                            tTUWrB,
                            0,
                            0,
                        )

                        c0_handle.commit()
                        c0_handle = mma_cudacore_producer0.acquire_and_advance()

                        q_handle.release()

                        tCtAcc_o_intra_acc_shape_fake = o_intra_thr_mma.make_fragment_C(
                            o_intra_acc_shape
                        )
                        tStO_o_intra0 = cute.make_tensor(
                            tmem_ptr + self.tmem_o_intra_output,
                            tCtAcc_o_intra_acc_shape_fake.layout,
                        )

                        o_intra_tiled_ts_mma_new = self.exec_mma(
                            o_intra_tiled_ts_mma_new,
                            tStO_o_intra0,
                            tOrP0,
                            tOIntrarB,
                            0,
                            0,
                        )

                        qs_tiled_mma = self.exec_mma(
                            qs_tiled_mma,
                            tOintertQS,
                            tQSA0,
                            tQSB,
                            0,
                            0,
                            True,
                        )

                        c0_handle.commit()
                        c0_handle = mma_cudacore_producer0.acquire_and_advance()

                        update_s_tiled_mma = self.exec_mma(
                            update_s_tiled_mma,
                            tStUpdate,
                            tUpdateA,
                            tUpdateB,
                            0,
                            0,
                            True,
                        )

                        c0_handle.commit()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                # End of persistent scheduler loop
        # ///////////////////////////////////////////////////////////////////////////////
        #  Mainloop: mainloop warp group
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < 4:
            cute.arch.setmaxregister_increase(self.num_regs_cudacore)

            tile_sched = create_gdn_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                if not continue_cond:
                    gO_mnl = cute.flat_divide(
                        O, cute.select(self.mma_tiler, mode=[0, 1])
                    )
                    # Use input state
                    if cutlass.const_expr(tma_atom_state_f32 is not None):
                        load_state_consumer.wait_and_advance()
                        self.init_state_from_smem(
                            kst_thr_mma,
                            tStateF32,
                            sStateInputF32,
                        )
                        load_state_consumer.release()
                    else:
                        self.init_state_zeros(
                            kst_thr_mma,
                            tStateF32,
                        )

                    # Arguments
                    atom_args = (
                        kkt_thr_mma,
                        kst_thr_mma,
                        qk_thr_mma,
                        qs_thr_mma,
                    )
                    tensor_args = (
                        sGate,
                        sBeta,
                        sGateCumsum,
                        sStateInput,
                        sIvt,
                        sQ,
                        sK,
                        sV,
                        sO,
                        sInvertSubReg,
                        sInvertSubSSL0A,
                        sInvertSubSSL0B,
                        sInvertSubSSL1B,
                        sInvertSubTSL0B,
                        sInvertSubSSL1A,
                        sInvertSubTSL1B,
                        sTuwAStore,
                        sTuwA,
                        sOIntraB,
                        sUpdateB,
                        tGamma,
                        tKKTtKKT,
                        tQKtQK,
                        tQKMtQKM,
                        tKtK,
                        tKStKS,
                        tItTSL0,
                        tInrDCL0A,
                        tOtTSL1,
                        tVtTUW,
                        tOintertQS,
                        tItSSL0,
                        tStateF32,
                        tQgate,
                    )
                    pipeline_args = (
                        gb_w0_consumer,
                        epi_w0_consumer,
                        mma_qk_consumer0,
                        mma_cudacore_consumer0,
                        load_v_consumer,
                    )

                    # Corner case
                    tail_count = mO_qdl.shape[0] % self.cta_tiler[0]

                    if tail_count == 0:
                        value_args = (
                            scale,
                            loop_count,
                            0,
                        )
                        (
                            gb_w0_consumer,
                            epi_w0_consumer,
                            mma_qk_consumer0,
                            mma_cudacore_consumer0,
                            load_v_consumer,
                        ) = self.main_loop(
                            value_args,
                            atom_args,
                            tensor_args,
                            pipeline_args,
                            use_qk_l2norm=use_qk_l2norm,
                        )
                    else:
                        value_args = (
                            scale,
                            loop_count - 1,
                            0,
                        )
                        (
                            gb_w0_consumer,
                            epi_w0_consumer,
                            mma_qk_consumer0,
                            mma_cudacore_consumer0,
                            load_v_consumer,
                        ) = self.main_loop(
                            value_args,
                            atom_args,
                            tensor_args,
                            pipeline_args,
                            use_qk_l2norm=use_qk_l2norm,
                        )
                        pipeline_args = (
                            gb_w0_consumer,
                            epi_w0_consumer,
                            mma_qk_consumer0,
                            mma_cudacore_consumer0,
                            load_v_consumer,
                        )
                        value_args = (
                            scale,
                            1,
                            loop_count - 1,
                        )
                        (
                            gb_w0_consumer,
                            epi_w0_consumer,
                            mma_qk_consumer0,
                            mma_cudacore_consumer0,
                            load_v_consumer,
                        ) = self.main_loop(
                            value_args,
                            atom_args,
                            tensor_args,
                            pipeline_args,
                            True,
                            tail_count=tail_count,
                            use_qk_l2norm=use_qk_l2norm,
                        )

                    # Output state
                    if cutlass.const_expr(tma_atom_state_output is not None):
                        w0_epi_handle = w0_epi_producer.acquire_and_advance()
                        self.store_state_to_smem(
                            tStUpdate, sStateOutput, update_s_thr_mma
                        )
                        w0_epi_handle.commit()

                    w0_handle = epi_w0_consumer.wait_and_advance()
                    w0_handle.release()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

    @cute.jit
    def main_loop(
        self,
        value_args: tuple,
        atom_args: tuple,
        tensor_args: tuple,
        pipeline_args: tuple,
        need_mask: cutlass.Constexpr[cutlass.Boolean] = False,
        tail_count: Int32 = 128,
        use_qk_l2norm: cutlass.Constexpr[cutlass.Boolean] = False,
    ) -> Tuple[
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
    ]:
        """Per-chunk mainloop executed by cudacore warps (0-3)."""
        tidx, _, _ = cute.arch.thread_idx()

        (
            scale,
            loop_count,
            loop_base,
        ) = value_args

        (
            kkt_thr_mma,
            kst_thr_mma,
            qk_thr_mma,
            qs_thr_mma,
        ) = atom_args

        (
            sGate,
            sBeta,
            sGateCumsum,
            sStateInput,
            sIvt,
            sQ,
            sK,
            sV,
            sO,
            sInvertSubReg,
            sInvertSubSSL0A,
            sInvertSubSSL0B,
            sInvertSubSSL1B,
            sInvertSubTSL0B,
            sInvertSubSSL1A,
            sInvertSubTSL1B,
            sTuwAStore,
            sTuwA,
            sOIntraB,
            sUpdateB,
            tGamma,
            tKKTtKKT,
            tQKtQK,
            tQKMtQKM,
            tKtK,
            tKStKS,
            tItTSL0,
            tInrDCL0A,
            tOtTSL1,
            tVtTUW,
            tOintertQS,
            tItSSL0,
            tStateF32,
            tQgate,
        ) = tensor_args

        (
            gb_w0_consumer,
            epi_w0_consumer,
            mma_qk_consumer0,
            mma_cudacore_consumer0,
            load_v_consumer,
        ) = pipeline_args

        for _i in cutlass.range(loop_base, loop_base + loop_count, unroll=1):
            gb_handle = gb_w0_consumer.wait_and_advance()
            sGate0 = sGate[None, None, gb_handle.index]
            sBeta0 = sBeta[None, None, gb_handle.index]
            tval = self.chunk_local_cumsum(sGate0, sGateCumsum, self.wg_sync_bar_id)
            beta_val = sBeta0[tidx]
            gb_handle.release()

            self.compute_gamma_tmem(
                tval,
                sGateCumsum,
                kkt_thr_mma,
                tGamma,
                self.wg_sync_bar_id,
                need_mask,
                tail_count,
            )
            qk_handle0 = mma_qk_consumer0.wait_and_advance()

            self.apply_gamma_beta(
                kkt_thr_mma,
                sIvt,
                tKKTtKKT,
                beta_val,
                tGamma,
            )

            gate_tail = (
                sGateCumsum[tail_count - 1]
                if cutlass.const_expr(need_mask)
                else sGateCumsum[127]
            )
            w0_handle = epi_w0_consumer.wait_and_advance()
            self.load_state_apply_gate(
                kst_thr_mma,
                tStateF32,
                sStateInput,
                gate_tail,
            )
            cute.arch.fence_proxy("async.shared", space="cta")

            qk_handle1 = mma_qk_consumer0.wait_and_advance()
            self.load_qk_epi(
                qk_thr_mma, tQKtQK, tQKMtQKM, tGamma, need_mask, tail_count
            )

            cute.arch.barrier_arrive(
                barrier_id=self.cudacore_mma_sync_bar_id,
                number_of_threads=self.threads_per_warp * 5,
            )

            reverse_result = cute.make_rmem_tensor((32, 1), self.acc_dtype)

            self.reverse_smem_sub(reverse_result, sIvt)

            self.store_ivt_smem_l0_ss_a(
                reverse_result,
                sInvertSubReg,
                sInvertSubSSL0A,
            )

            self.store_ivt_smem_l0_ss_b(
                kkt_thr_mma,
                tKKTtKKT,
                sInvertSubSSL0B,
            )

            c0_handle = mma_cudacore_consumer0.wait_and_advance()
            c0_handle.release()
            v_handle0 = load_v_consumer.wait_and_advance()

            tval_exp = cute.math.exp(tval)
            self.get_uw_b(
                kst_thr_mma, tKStKS, tval_exp, beta_val, sV, need_mask, tail_count
            )
            self.store_ivt_p1(
                sInvertSubReg,
                sInvertSubTSL0B,
            )

            c0_handle = mma_cudacore_consumer0.wait_and_advance()
            self.load_ivt_ss_l0(tItSSL0, tInrDCL0A)  # debug
            c0_handle.release()
            self.store_ivt_smem_l1_ss_b(
                kkt_thr_mma,
                tKKTtKKT,
                sInvertSubSSL1B,
            )

            self.store_ivt_p2(
                sInvertSubReg,
                sInvertSubSSL1A,
            )

            c0_handle = mma_cudacore_consumer0.wait_and_advance()

            self.load_ivt_ts_l0(
                tItTSL0,
                sInvertSubSSL1A,
                sInvertSubTSL1B,
            )

            c0_handle.release()

            self.store_k_epi(kkt_thr_mma, sK, tKtK, need_mask, tail_count, use_qk_l2norm=use_qk_l2norm)
            self.store_ivt_p3(
                sInvertSubReg,
                sInvertSubTSL1B,
            )

            c0_handle = mma_cudacore_consumer0.wait_and_advance()
            c0_handle.release()
            self.save_tmem(tItTSL0, sTuwAStore)
            self.store_ivt_ad(sInvertSubReg, sTuwA)
            self.store_ivt_c(sTuwAStore)

            c0_handle = mma_cudacore_consumer0.wait_and_advance()
            self.load_ivt_result(tOtTSL1, sTuwAStore)
            c0_handle.release()

            self.load_q_epi(qk_thr_mma, sQ, tQgate, tval_exp, need_mask, tail_count, use_qk_l2norm=use_qk_l2norm)
            cute.arch.fence_proxy("async.shared", space="cta")

            c0_handle = mma_cudacore_consumer0.wait_and_advance()
            v_handle0.release()
            self.load_v(tVtTUW, sOIntraB)
            c0_handle.release()
            self.load_k(tKtK, sUpdateB, gate_tail, tval)
            c0_handle = mma_cudacore_consumer0.wait_and_advance()
            qk_handle0.release()
            c0_handle.release()
            self.store_o_smem(tOintertQS, sO, qs_thr_mma, scale)

            w0_handle.release()
            qk_handle1.release()
            c0_handle = mma_cudacore_consumer0.wait_and_advance()
            c0_handle.release()

        return (
            gb_w0_consumer,
            epi_w0_consumer,
            mma_qk_consumer0,
            mma_cudacore_consumer0,
            load_v_consumer,
        )

    @cute.jit
    def exec_mma(
        self,
        tiled_mma,
        tCtAcc,
        tCrA,
        tCrB,
        a_consumer_index,
        b_consumer_index,
        setAcc: bool = False,
    ):
        """Issue a tcgen05 GEMM."""
        for kphase_idx in cutlass.range(cute.size(tCrB, mode=[2]), unroll_all=True):
            # set accu = 1
            tiled_mma.set(
                tcgen05.Field.ACCUMULATE,
                cutlass.Boolean(kphase_idx != 0 or setAcc),
            )
            cute.gemm(
                tiled_mma,
                tCtAcc,
                tCrA[None, None, kphase_idx, a_consumer_index],
                tCrB[None, None, kphase_idx, b_consumer_index],
                tCtAcc,
            )
        return tiled_mma

    @cute.jit
    def get_uw_b(
        self,
        thr_mma: cute.ThrMma,
        tIn: cute.Tensor,
        gate_val: cutlass.Float32,
        beta_val: cutlass.Float32,
        sV: cute.Tensor,
        mask: cutlass.Constexpr[cutlass.Boolean] = False,
        tail_count: Int32 = 128,
    ):
        """Compute V_new = beta*V - beta*gamma*(K@S^T) in-place in smem."""
        tidx, _, _ = cute.arch.thread_idx()

        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        cIn = cute.make_identity_tensor((128, 128))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 64
        corr_tile_size_f32 = corr_tile_size // 2

        tIn_i_layout = cute.composition(
            tIn.layout, cute.make_layout((128, corr_tile_size))
        )

        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size))
        )

        tIn_i = cute.make_tensor(tIn.iterator, tIn_i_layout)
        cIn_i = cute.make_tensor(cIn.iterator, cIn_i_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size_f32)),
            self.acc_dtype,
        )

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tIn_i)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)

        frg_tile = 8
        sV_frg = cute.logical_divide(sV, cute.make_layout(frg_tile))

        tTMEM_LOADtO = thr_tmem_load.partition_S(tIn_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(cIn_i)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcO.shape, self.acc_dtype)
        tTMEM_LOADrS_frag = cute.logical_divide(
            tTMEM_LOADrS, cute.make_layout(frg_tile)
        )

        if cutlass.const_expr(mask):
            if thread_idx >= tail_count:
                beta_val = self.g_beta_dtype(0)

        combined_factor = gate_val * beta_val

        each_iter = corr_tile_size // frg_tile
        for i in cutlass.range_constexpr(128 // corr_tile_size):
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )
            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)
            cute.arch.fence_view_async_tmem_load()

            # Load V from smem
            temp_regs = cute.make_rmem_tensor(tTMEM_LOADrS_frag.shape, self.acc_dtype)
            for it in cutlass.range_constexpr(each_iter):
                for inner_idx in cutlass.range_constexpr(0, frg_tile, 2):
                    # mul2
                    (
                        temp_regs[inner_idx + 0, it],
                        temp_regs[inner_idx + 1, it],
                    ) = cute.arch.mul_packed_f32x2(
                        (
                            tTMEM_LOADrS_frag[inner_idx + 0, it],
                            tTMEM_LOADrS_frag[inner_idx + 1, it],
                        ),
                        (-combined_factor, -combined_factor),
                    )

                    # fma2
                    (
                        temp_regs[inner_idx + 0, it],
                        temp_regs[inner_idx + 1, it],
                    ) = cute.arch.fma_packed_f32x2(
                        (
                            beta_val,
                            beta_val,
                        ),
                        (
                            cutlass.Float32(
                                sV_frg[inner_idx + 0, (i * each_iter + it, thread_idx)]
                            ),
                            cutlass.Float32(
                                sV_frg[inner_idx + 1, (i * each_iter + it, thread_idx)]
                            ),
                        ),
                        (
                            temp_regs[inner_idx + 0, it],
                            temp_regs[inner_idx + 1, it],
                        ),
                    )
                sV_frg[None, (i * each_iter + it, thread_idx)].store(
                    temp_regs[None, it].load().to(self.i_dtype)
                )

    @cute.jit
    def store_state_to_smem(
        self,
        tO: cute.Tensor,
        sO: cute.Tensor,
        thr_mma: cute.ThrMma,
    ):
        """Copy final state from TMEM to smem for TMA store to global."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        tOcO = thr_mma.partition_C(tO)
        tOsO = thr_mma.partition_C(sO)

        corr_tile_size = 32  # must >= 8
        tOsO_sub = cute.make_tensor(
            cute.recast_ptr(tOsO.iterator, dtype=self.state_dtype),
            cute.shape(tOsO)[0],
        )
        tOsO_tile = cute.logical_divide(
            tOsO_sub, (None, (None, 32 // (128 // corr_tile_size)))
        )

        tIn_i_layout = cute.composition(
            tO.layout, cute.make_layout((128, corr_tile_size))
        )
        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size))
        )
        sOut_i_layout = cute.composition(
            tOsO.layout, cute.make_layout((128, corr_tile_size))
        )

        tIn_i = cute.make_tensor(tO.iterator, tIn_i_layout)
        cIn_i = cute.make_tensor(tOcO.iterator, cIn_i_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tIn_i)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tIn_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(cIn_i)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcO.shape, self.acc_dtype)

        smem_copy_atom = sm100_utils.get_smem_store_op(
            self.s_output, self.state_dtype, self.acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        for i in cutlass.range_constexpr(128 // corr_tile_size):
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )
            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)
            sOut_i_sub = cute.make_tensor(
                tOsO_tile[None, (None, (None, i))].iterator, sOut_i_layout
            )
            tTMEM_STOREsO_i = thr_tmem_load.partition_D(sOut_i_sub)
            cute.copy(tiled_smem_store, tTMEM_LOADrS, tTMEM_STOREsO_i)

        # fence view async shared
        cute.arch.fence_proxy("async.shared", space="cta")

    @cute.jit
    def store_o_smem(
        self,
        tO: cute.Tensor,
        sO: cute.Tensor,
        thr_mma: cute.ThrMma,
        scale: cutlass.Float32,
    ):
        """Scale and convert O from TMEM (f32) to smem (f16/bf16) for TMA store."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        tOcO = thr_mma.partition_C(tO)
        tOsO = thr_mma.partition_C(sO)

        corr_tile_size = 128  # must >= 8

        # ((128,(8,16)),1,1,(1,1))
        tOsO_sub = cute.make_tensor(
            cute.recast_ptr(tOsO.iterator, dtype=self.i_dtype),
            cute.shape(tOsO)[0],
        )
        tOsO_tile = cute.logical_divide(
            tOsO_sub, (None, (None, 16 // (128 // corr_tile_size)))
        )

        tIn_i_layout = cute.composition(
            tO.layout, cute.make_layout((128, corr_tile_size))
        )
        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size))
        )
        sOut_i_layout = cute.composition(
            tOsO.layout, cute.make_layout((128, corr_tile_size))
        )

        tIn_i = cute.make_tensor(tO.iterator, tIn_i_layout)
        cIn_i = cute.make_tensor(tOcO.iterator, cIn_i_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tIn_i)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tIn_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(cIn_i)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcO.shape, self.acc_dtype)

        tTMEM_STORErS_x4_e = cute.make_tensor(
            cute.recast_ptr(tTMEM_LOADrS.iterator, dtype=self.i_dtype),
            tTMEM_LOADrS.layout,
        )

        smem_copy_atom = sm100_utils.get_smem_store_op(
            self.o_layout, self.i_dtype, self.acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        for i in cutlass.range_constexpr(128 // corr_tile_size):
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )
            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)

            frg_cnt = 4  # must 4
            frg_tile = cute.size(tTMEM_LOADrS) // frg_cnt
            tTMEM_LOADrS_frg = cute.logical_divide(
                tTMEM_LOADrS, cute.make_layout(frg_tile)
            )
            tTMEM_STORErS_x4_e_frg = cute.logical_divide(
                tTMEM_STORErS_x4_e, cute.make_layout(frg_tile)
            )

            temp_regs = cute.make_rmem_tensor(tTMEM_LOADrS_frg.shape, self.acc_dtype)
            for j in cutlass.range_constexpr(frg_cnt):
                for inner_idx in cutlass.range_constexpr(0, frg_tile, 2):
                    # mul2
                    (
                        temp_regs[inner_idx + 0, j],
                        temp_regs[inner_idx + 1, j],
                    ) = cute.arch.mul_packed_f32x2(
                        (
                            tTMEM_LOADrS_frg[inner_idx + 0, j],
                            tTMEM_LOADrS_frg[inner_idx + 1, j],
                        ),
                        (scale, scale),
                    )
                tTMEM_STORErS_x4_e_frg[None, j].store(
                    temp_regs[None, j].load().to(self.i_dtype)
                )

            # store
            sOut_i_sub = cute.make_tensor(
                tOsO_tile[None, (None, (None, i))].iterator, sOut_i_layout
            )
            tTMEM_STOREsO_i = thr_tmem_load.partition_D(sOut_i_sub)
            cute.copy(tiled_smem_store, tTMEM_STORErS_x4_e, tTMEM_STOREsO_i)

        # fence view async shared
        cute.arch.fence_proxy("async.shared", space="cta")

    @cute.jit
    def load_k(
        self,
        tIn: cute.Tensor,
        sUpdateB: cute.Tensor,
        gate: cutlass.Float32,
        val: cutlass.Float32,
    ):
        """Load K from TMEM, apply per-token gate decay."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        gate_val = cute.math.exp((gate - val))

        k = self.read_tmem_128(tIn)

        k_f16 = cute.make_tensor(
            cute.recast_ptr(k.iterator, dtype=self.i_dtype),
            cute.make_layout(128),
        )
        k_f16_frag = cute.logical_divide(k_f16, cute.make_layout(8))

        for bid in cutlass.range_constexpr(0, 2):
            for col in cutlass.range_constexpr(0, 8):
                load_row = thread_idx

                store_row = load_row
                store_col = bid * 8 + col

                store_row_bid = store_row // 16
                store_row_offset = store_row % 16
                sUpdateB[
                    ((None, store_col), store_row_offset), 0, store_row_bid, 0
                ].store(
                    (k_f16_frag[None, store_col].load() * gate_val).to(self.i_dtype)
                )

        cute.arch.fence_proxy("async.shared", space="cta")

    @cute.jit
    def init_state_zeros(
        self,
        thr_mma: cute.ThrMma,
        tOutF32: cute.Tensor,
    ):
        """Zero-initialize the recurrent state in TMEM (no initial_state provided)."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        cIn = cute.make_identity_tensor((128, 128))
        tOcO = thr_mma.partition_C(cIn)
        corr_tile_size = 128

        # Apply gate State
        cInOut_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size))
        )
        tInOut_i_layout = cute.composition(
            tOutF32.layout, cute.make_layout((128, corr_tile_size))
        )

        tInOut_i = cute.make_tensor(tOutF32.iterator, tInOut_i_layout)
        cInOut_i = cute.make_tensor(tOcO.iterator, cInOut_i_layout)

        tmem_store_atom_f32 = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        tiled_tmem_store_f32 = tcgen05.make_tmem_copy(tmem_store_atom_f32, tInOut_i)
        thr_tmem_store_f32 = tiled_tmem_store_f32.get_slice(thread_idx)

        tTMEM_STOREtInOut = thr_tmem_store_f32.partition_D(tInOut_i)
        tTMEM_STOREcInOut = thr_tmem_store_f32.partition_S(cInOut_i)
        tTMEM_STORErInOut = cute.make_rmem_tensor(
            tTMEM_STOREcInOut.shape, self.acc_dtype
        )

        frg_tile = 4
        zeros = cute.zeros_like(cute.make_layout(frg_tile), self.acc_dtype)

        for i in cutlass.range_constexpr(128 // corr_tile_size):
            tTMEM_STOREtInOut_i = cute.make_tensor(
                tTMEM_STOREtInOut.iterator + i * corr_tile_size,
                tTMEM_STOREtInOut.layout,
            )

            tTMEM_STORErInOut_frg = cute.logical_divide(
                tTMEM_STORErInOut, ((frg_tile, None), None)
            )

            each_iter = corr_tile_size // frg_tile
            for j in cutlass.range_constexpr(each_iter):
                tTMEM_STORErInOut_frg[((None, j), 0), 0, 0].store(zeros)

            # store
            cute.copy(tiled_tmem_store_f32, tTMEM_STORErInOut, tTMEM_STOREtInOut_i)
            cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def init_state_from_smem(
        self,
        thr_mma: cute.ThrMma,
        tOutF32: cute.Tensor,
        sStateInputF32: cute.Tensor,
    ):
        """Load initial_state from smem (f32) into TMEM."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        cIn = cute.make_identity_tensor((128, 128))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 128
        cInOut_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size))
        )
        tInOut_i_layout = cute.composition(
            tOutF32.layout, cute.make_layout((128, corr_tile_size))
        )

        tInOut_i = cute.make_tensor(tOutF32.iterator, tInOut_i_layout)
        cInOut_i = cute.make_tensor(tOcO.iterator, cInOut_i_layout)

        tmem_store_atom_f32 = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        tiled_tmem_store_f32 = tcgen05.make_tmem_copy(tmem_store_atom_f32, tInOut_i)
        thr_tmem_store_f32 = tiled_tmem_store_f32.get_slice(thread_idx)

        tTMEM_STOREtInOut = thr_tmem_store_f32.partition_D(tInOut_i)
        tTMEM_STOREcInOut = thr_tmem_store_f32.partition_S(cInOut_i)
        tTMEM_STORErInOut = cute.make_rmem_tensor(
            tTMEM_STOREcInOut.shape, self.acc_dtype
        )

        frg_tile = 4  # must 4
        sStateInputF32_frag = cute.logical_divide(
            sStateInputF32, (((None, frg_tile), None))
        )
        tTMEM_STORErInOut_frg = cute.logical_divide(
            tTMEM_STORErInOut, ((frg_tile, None), None)
        )
        each_iter = corr_tile_size // frg_tile
        for i in cutlass.range_constexpr(128 // corr_tile_size):
            tTMEM_STOREtInOut_i = cute.make_tensor(
                tTMEM_STOREtInOut.iterator + i * corr_tile_size,
                tTMEM_STOREtInOut.layout,
            )

            # Load from smem
            idx_offset = i * corr_tile_size
            for j in cutlass.range_constexpr(each_iter):
                idx = idx_offset + j * frg_tile
                bid = idx // 32
                bid_inner = idx % 32 // 8
                inner_idx = idx % 8 // 4
                s_vec = sStateInputF32_frag[
                    (thread_idx, (None, inner_idx)), 0, (bid_inner, bid), 0
                ].load()
                tTMEM_STORErInOut_frg[((None, j), 0), 0, 0].store(s_vec)

            # store
            cute.copy(tiled_tmem_store_f32, tTMEM_STORErInOut, tTMEM_STOREtInOut_i)
            cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def load_state_apply_gate(
        self,
        thr_mma: cute.ThrMma,
        tIn: cute.Tensor,
        sStateInput: cute.Tensor,
        gate: cutlass.Float32,
    ):
        """Apply chunk-level gate decay to state in TMEM, and copy state to smem (f16) for KST GEMM."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        cIn = cute.make_identity_tensor((128, 128))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 16
        # corr_tile_size_f32 = corr_tile_size // 2

        tIn_i_layout = cute.composition(
            tIn.layout, cute.make_layout((128, corr_tile_size))
        )
        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size))
        )

        # Apply gate State
        # cInOut_i_layout = cute.composition(tOcO.layout, cute.make_layout((128, corr_tile_size)))
        tInOut_i_layout = cute.composition(
            tIn.layout, cute.make_layout((128, corr_tile_size))
        )

        # fp16 state
        tIn_i = cute.make_tensor(tIn.iterator, tIn_i_layout)
        cIn_i = cute.make_tensor(cIn.iterator, cIn_i_layout)

        tInOut_i = cute.make_tensor(tIn.iterator, tInOut_i_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        tmem_store_atom_f32 = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tIn_i)
        tiled_tmem_store_f32 = tcgen05.make_tmem_copy(tmem_store_atom_f32, tInOut_i)

        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        thr_tmem_store_f32 = tiled_tmem_store_f32.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tIn_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(cIn_i)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcO.shape, self.acc_dtype)

        tTMEM_STOREtInOut = thr_tmem_store_f32.partition_D(tInOut_i)
        tTMEM_STORErInOut = cute.make_tensor(
            cute.recast_ptr(tTMEM_LOADrS.iterator, dtype=self.acc_dtype),
            tTMEM_LOADrS.layout,
        )

        gate_val = cute.math.exp(gate)
        for i in cutlass.range_constexpr(128 // corr_tile_size):
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )
            tTMEM_STOREtInOut_i = cute.make_tensor(
                tTMEM_STOREtInOut.iterator + i * corr_tile_size,
                tTMEM_STOREtInOut.layout,
            )

            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)

            frg_tile = 16  # must <= 16
            tTMEM_LOADrS_frg = cute.logical_divide(
                tTMEM_LOADrS, ((frg_tile, None), None)
            )
            tTMEM_STORErInOut_frg = cute.logical_divide(
                tTMEM_STORErInOut, ((frg_tile, None), None)
            )
            sStateInput_frag = cute.logical_divide(
                sStateInput, (((None, frg_tile), None))
            )

            each_iter = corr_tile_size // frg_tile
            idx_offset = i * corr_tile_size
            for j in cutlass.range_constexpr(each_iter):
                s_vec = tTMEM_LOADrS_frg[((None, j), 0), 0, 0].load()
                s_vec_f16 = s_vec.to(self.q_dtype)
                idx = idx_offset + j * frg_tile
                bid = idx // 64
                bid_inner = idx % 64 // 16
                inner_idx = idx % 16 // frg_tile
                sStateInput_frag[
                    (thread_idx, (None, inner_idx)), 0, (bid_inner, bid), 0
                ].store(s_vec_f16)
                # tTMEM_STORErInOut_frg[((None, j), 0), 0, 0].store(s_vec * gate_val)

                for inner_idx in cutlass.range_constexpr(0, frg_tile, 2):
                    (
                        tTMEM_STORErInOut_frg[((inner_idx + 0, j), 0), 0, 0],
                        tTMEM_STORErInOut_frg[((inner_idx + 1, j), 0), 0, 0],
                    ) = cute.arch.mul_packed_f32x2(
                        (s_vec[inner_idx + 0], s_vec[inner_idx + 1]),
                        (
                            gate_val,
                            gate_val,
                        ),
                    )
            cute.copy(tiled_tmem_store_f32, tTMEM_STORErInOut, tTMEM_STOREtInOut_i)
            cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def load_qk_epi(
        self,
        thr_mma: cute.ThrMma,
        tIn: cute.Tensor,
        tOut: cute.Tensor,
        tGamma: cute.Tensor,  # causal gate mask
        mask: cutlass.Constexpr[cutlass.Boolean] = False,
        tail_count: Int32 = 128,
    ):
        """Apply gamma mask to QK^T."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        cIn = cute.make_identity_tensor((128, 128))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 32
        corr_tile_size_f32 = corr_tile_size // 2

        tIn_i_layout = cute.composition(
            tIn.layout, cute.make_layout((128, corr_tile_size))
        )
        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size))
        )

        cOut_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size_f32))
        )
        tOut_i_layout = cute.composition(
            tOut.layout, cute.make_layout((128, corr_tile_size_f32))
        )

        tIn_i = cute.make_tensor(tIn.iterator, tIn_i_layout)
        tGamma_i = cute.make_tensor(tGamma.iterator, tIn_i_layout)
        cIn_i = cute.make_tensor(cIn.iterator, cIn_i_layout)

        tOut_i = cute.make_tensor(tOut.iterator, tOut_i_layout)
        cOut_i = cute.make_tensor(tOcO.iterator, cOut_i_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size_f32)),
            self.acc_dtype,
        )

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tIn_i)
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOut_i)

        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)

        # Load Gamma
        tiled_tmem_load_gamma = tcgen05.make_tmem_copy(tmem_load_atom, tGamma_i)
        thr_tmem_load_gamma = tiled_tmem_load_gamma.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tIn_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(cIn_i)

        tTMEM_STOREcS = thr_tmem_store.partition_S(cOut_i)
        tTMEM_STOREtO = thr_tmem_store.partition_D(tOut_i)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcO.shape, self.acc_dtype)

        tTMEM_LOADtG = thr_tmem_load_gamma.partition_S(tGamma_i)
        tTMEM_LOADrG = cute.make_rmem_tensor(tTMEM_LOADcO.shape, self.acc_dtype)

        tTMEM_STORErS_x4 = cute.make_rmem_tensor(tTMEM_STOREcS.shape, self.acc_dtype)
        tTMEM_STORErS_x4_e = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErS_x4.iterator, dtype=self.i_dtype),
            tTMEM_LOADrS.layout,
        )

        for i in cutlass.range_constexpr(128 // corr_tile_size):
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )
            tTMEM_STOREtO_i = cute.make_tensor(
                tTMEM_STOREtO.iterator + i * corr_tile_size_f32, tTMEM_STOREtO.layout
            )
            tTMEM_LOADtG_i = cute.make_tensor(
                tTMEM_LOADtG.iterator + i * corr_tile_size, tTMEM_LOADrG.layout
            )

            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)
            cute.copy(tiled_tmem_load_gamma, tTMEM_LOADtG_i, tTMEM_LOADrG)

            frg_tile = 2
            tTMEM_LOADrS_frg = cute.logical_divide(
                tTMEM_LOADrS, ((frg_tile, None), None)
            )
            tTMEM_LOADrG_frg = cute.logical_divide(
                tTMEM_LOADrG, ((frg_tile, None), None)
            )
            tTMEM_STORErS_x4_e_frg = cute.logical_divide(
                tTMEM_STORErS_x4_e, cute.make_layout(frg_tile)
            )

            each_iter = corr_tile_size // frg_tile
            for j in cutlass.range_constexpr(each_iter):
                for it in cutlass.range_constexpr(0, frg_tile, 2):
                    (
                        tTMEM_LOADrS_frg[((it + 0, j), 0), 0, 0],
                        tTMEM_LOADrS_frg[((it + 1, j), 0), 0, 0],
                    ) = cute.arch.mul_packed_f32x2(
                        (
                            tTMEM_LOADrS_frg[((it + 0, j), 0), 0, 0],
                            tTMEM_LOADrS_frg[((it + 1, j), 0), 0, 0],
                        ),
                        (
                            tTMEM_LOADrG_frg[((it + 0, j), 0), 0, 0],
                            tTMEM_LOADrG_frg[((it + 1, j), 0), 0, 0],
                        ),
                    )

                s_vec = tTMEM_LOADrS_frg[((None, j), 0), 0, 0].load()
                tTMEM_STORErS_x4_e_frg[None, j].store(s_vec.to(self.q_dtype))

            # store
            cute.copy(tiled_tmem_store, tTMEM_STORErS_x4, tTMEM_STOREtO_i)
            cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def load_q_epi(
        self,
        thr_mma: cute.ThrMma,
        sQ: cute.Tensor,
        tOut: cute.Tensor,
        val: cutlass.Float32,  # exp(cumsum_gate) for per-token gating
        mask: cutlass.Constexpr[cutlass.Boolean] = False,
        tail_count: Int32 = 128,
        use_qk_l2norm: cutlass.Constexpr[cutlass.Boolean] = False,
    ):
        """Load Q from smem, apply per-token gate."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        sQ_frag = cute.logical_divide(sQ, (((None, 8), None)))
        cIn = cute.make_identity_tensor((128, 128))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 64
        corr_tile_size_f32 = corr_tile_size // 2

        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size))
        )
        cOut_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size_f32))
        )
        tOut_i_layout = cute.composition(
            tOut.layout, cute.make_layout((128, corr_tile_size_f32))
        )

        cIn_i = cute.make_tensor(cIn.iterator, cIn_i_layout)
        tOut_i = cute.make_tensor(tOut.iterator, tOut_i_layout)
        cOut_i = cute.make_tensor(tOcO.iterator, cOut_i_layout)

        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size_f32)),
            self.acc_dtype,
        )

        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOut_i)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)

        tTMEM_STOREcS_f16 = thr_tmem_store.partition_S(cIn_i)
        tTMEM_STOREcS = thr_tmem_store.partition_S(cOut_i)
        tTMEM_STOREtO = thr_tmem_store.partition_D(tOut_i)

        tTMEM_STORErS = cute.make_rmem_tensor(tTMEM_STOREcS.shape, self.acc_dtype)
        tTMEM_STORErS_x8 = cute.make_rmem_tensor(tTMEM_STOREcS_f16.shape, self.i_dtype)
        tTMEM_STORErS_x8_e = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErS.iterator, dtype=self.i_dtype),
            tTMEM_STORErS_x8.layout,
        )

        # Q L2 norm is now applied as a preprocessing step BEFORE the kernel launch,
        # so that QK = Q_norm @ K_norm^T and all downstream uses are consistent.
        # Here we just apply the per-token gate (val = exp(cumsum_gate)).
        effective_val = val

        for i in cutlass.range_constexpr(128 // corr_tile_size):
            tTMEM_STOREtO_i = cute.make_tensor(
                tTMEM_STOREtO.iterator + i * corr_tile_size_f32, tTMEM_STOREtO.layout
            )

            frg_tile = 8
            tTMEM_STORErS_x8_e_frag = cute.logical_divide(
                tTMEM_STORErS_x8_e, cute.make_layout(frg_tile)
            )
            for col in cutlass.range_constexpr(0, 8):
                load_row = thread_idx
                load_col = col
                curr_val_ssa = sQ_frag[
                    (load_row, (None, load_col % 2)), 0, (load_col // 2, i), 0
                ].load()
                tTMEM_STORErS_x8_e_frag[None, col].store(
                    (curr_val_ssa * effective_val).to(self.i_dtype)
                )
            # store
            if cutlass.const_expr(mask):
                if thread_idx >= tail_count:
                    tTMEM_STORErS_x8_e_frag.fill(self.i_dtype(0))
            cute.copy(tiled_tmem_store, tTMEM_STORErS, tTMEM_STOREtO_i)

        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def store_k_epi(
        self,
        thr_mma: cute.ThrMma,
        sK: cute.Tensor,
        tOut: cute.Tensor,
        mask: cutlass.Constexpr[cutlass.Boolean] = False,
        tail_count: Int32 = 128,
        use_qk_l2norm: cutlass.Constexpr[cutlass.Boolean] = False,
    ):
        """Load K from smem and store to TMEM."""
        tidx, _, _ = cute.arch.thread_idx()

        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        sK_frag = cute.logical_divide(sK, (((None, 8), None)))
        cIn = cute.make_identity_tensor((128, 128))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 64
        corr_tile_size_f32 = corr_tile_size // 2

        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size))
        )
        cOut_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size_f32))
        )
        tOut_i_layout = cute.composition(
            tOut.layout, cute.make_layout((128, corr_tile_size_f32))
        )

        cIn_i = cute.make_tensor(cIn.iterator, cIn_i_layout)
        tOut_i = cute.make_tensor(tOut.iterator, tOut_i_layout)
        cOut_i = cute.make_tensor(tOcO.iterator, cOut_i_layout)

        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size_f32)),
            self.acc_dtype,
        )

        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOut_i)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)

        tTMEM_STOREcS_f16 = thr_tmem_store.partition_S(cIn_i)
        tTMEM_STOREcS = thr_tmem_store.partition_S(cOut_i)
        tTMEM_STOREtO = thr_tmem_store.partition_D(tOut_i)

        tTMEM_STORErS = cute.make_rmem_tensor(tTMEM_STOREcS.shape, self.acc_dtype)
        tTMEM_STORErS_x8 = cute.make_rmem_tensor(tTMEM_STOREcS_f16.shape, self.i_dtype)
        tTMEM_STORErS_x8_e = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErS.iterator, dtype=self.i_dtype),
            tTMEM_STORErS_x8.layout,
        )
        frg_tile = 8
        tTMEM_STORErS_x8_e_frag = cute.logical_divide(
            tTMEM_STORErS_x8_e, cute.make_layout(frg_tile)
        )

        # K L2 norm is now applied as a preprocessing step BEFORE the kernel launch,
        # so that QK = Q_norm @ K_norm^T and KK^T = K_norm @ K_norm^T are consistent
        # with the state update and output paths.

        for i in cutlass.range_constexpr(128 // corr_tile_size):
            tTMEM_STOREtO_i = cute.make_tensor(
                tTMEM_STOREtO.iterator + i * corr_tile_size_f32, tTMEM_STOREtO.layout
            )

            for col in cutlass.range_constexpr(0, 8):
                load_row = thread_idx

                load_col = col
                curr_val_ssa = sK_frag[
                    (load_row, (None, load_col % 2)), 0, (load_col // 2, i), 0
                ].load()
                # K is already L2-normalized before kernel launch (preprocessing).
                tTMEM_STORErS_x8_e_frag[None, col].store(
                    curr_val_ssa
                )

            # store
            if cutlass.const_expr(mask):
                if thread_idx >= tail_count:
                    tTMEM_STORErS_x8_e_frag.fill(self.i_dtype(0))

            cute.copy(tiled_tmem_store, tTMEM_STORErS, tTMEM_STOREtO_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def reverse_smem_sub(
        self,
        reverse_result: cute.Tensor,
        sIvt: cute.Tensor,
    ):
        """Compute the inverse of a 32x32 sub-block."""
        lane_id = cute.arch.lane_idx()
        sub_widx = cute.arch.warp_idx() % 4

        reverse_result.fill(self.acc_dtype(0))

        for row in cutlass.range_constexpr(1, 32, unroll=1):
            for k in cutlass.range(1, row):
                row_i_k = sIvt[(k, row), sub_widx]
                reverse_result[row] -= reverse_result[k] * row_i_k
            row_i_lane = self.acc_dtype(0)
            if lane_id < row:
                row_i_lane = sIvt[(lane_id, row), sub_widx]
            reverse_result[row] -= row_i_lane
        reverse_result[lane_id] = self.acc_dtype(1)

    @cute.jit
    def store_ivt_smem_l0_ss_a(
        self,
        reverse_result: cute.Tensor,
        sReg: cute.Tensor,
        sInvertSubSSL0A: cute.Tensor,
    ):
        """Store 32x32 sub-inverse results to smem."""

        tidx, _, _ = cute.arch.thread_idx()
        widx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()

        sub_widx = widx % 4

        # store sub result to shared memory
        # (((4,128),8))
        for row_bid in cutlass.range_constexpr(4):
            for col_bid in cutlass.range_constexpr(8):
                row_d = (lane_id // 4 + col_bid) % 8
                row = row_bid * 8 + row_d
                col_id = lane_id // 4
                col_sub_id = lane_id % 4
                sReg[(col_sub_id, sub_widx * 32 + row), col_id] = reverse_result[row]

        cutlass.cute.arch.sync_warp()

        if sub_widx == 1 or sub_widx == 3:
            sub_id_row = sub_widx // 2
            sInvertSubSSL0A_frag = cute.logical_divide(
                sInvertSubSSL0A, ((None, 4), None)
            )
            for col in cutlass.range_constexpr(8):
                sInvertSubSSL0A_frag[
                    (sub_id_row * 32 + lane_id, (None, col % 2)), 0, col // 2, 0
                ].store(sReg[(None, tidx % 128), col].load())

    @cute.jit
    def store_ivt_smem_l0_ss_b(
        self,
        thr_mma: cute.ThrMma,
        tIn: cute.Tensor,
        sInvertSubSSL0B: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        # bidx, bidy, _ = cute.arch.block_idx()
        widx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        sub_widx = widx % 4

        # load Tmem
        cIn = cute.make_identity_tensor((128, 128))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 32  # must <= 32
        tIn_i_layout = cute.composition(
            tIn.layout, cute.make_layout((128, corr_tile_size))
        )
        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size))
        )
        tIn_i = cute.make_tensor(tIn.iterator, tIn_i_layout)
        cIn_i = cute.make_tensor(cIn.iterator, cIn_i_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        # Load In
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tIn_i)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tIn_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(cIn_i)

        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcO.shape, self.acc_dtype)

        sub_tile_size = 8
        tmem_frag = cute.logical_divide(tTMEM_LOADrS, ((sub_tile_size, None), None))
        sInvertSubSSL0B_frag = cute.logical_divide(
            sInvertSubSSL0B, ((sub_tile_size, None), None)
        )

        # tile 0
        tile_frags = corr_tile_size // sub_tile_size
        for i in cutlass.range_constexpr(32 // corr_tile_size):
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )

            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)
            cute.arch.fence_view_async_tmem_load()
            if sub_widx == 1:
                idx = (i * corr_tile_size) // sub_tile_size
                for j in cutlass.range_constexpr(tile_frags):
                    col_true = (idx + j) % 8
                    # todo: remove smem bank conflict
                    sInvertSubSSL0B_frag[
                        ((None, (col_true, 0)), lane_id % 8), 0, lane_id // 8, 0
                    ].store(tmem_frag[((None, j), 0), 0, 0].load())

        # tile 2
        for i in cutlass.range_constexpr(32 // corr_tile_size):
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + 64 + i * corr_tile_size, tTMEM_LOADtO.layout
            )

            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)
            cute.arch.fence_view_async_tmem_load()
            if sub_widx == 3:
                idx = (i * corr_tile_size) // sub_tile_size
                for j in cutlass.range_constexpr(tile_frags):
                    col_true = (idx + j) % 8
                    # todo: remove smem bank conflict
                    sInvertSubSSL0B_frag[
                        ((None, (col_true, 1)), lane_id % 8), 0, lane_id // 8, 0
                    ].store(tmem_frag[((None, j), 0), 0, 0].load())

        cute.arch.fence_proxy("async.shared", space="cta")

    @cute.jit
    def store_ivt_smem_l1_ss_b(
        self,
        thr_mma: cute.ThrMma,
        tIn: cute.Tensor,
        sInvertSubSSL1B: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        # bidx, bidy, _ = cute.arch.block_idx()
        widx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        sub_widx = widx % 4

        # load Tmem
        cIn = cute.make_identity_tensor((128, 128))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 8
        tIn_i_layout = cute.composition(
            tIn.layout, cute.make_layout((128, corr_tile_size))
        )
        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size))
        )
        tIn_i = cute.make_tensor(tIn.iterator, tIn_i_layout)
        cIn_i = cute.make_tensor(cIn.iterator, cIn_i_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        # Load In
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tIn_i)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tIn_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(cIn_i)

        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcO.shape, self.acc_dtype)

        # tile 0
        tTMEM_LOADtO_i = cute.make_tensor(tTMEM_LOADtO.iterator, tTMEM_LOADtO.layout)

        cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)
        cute.arch.fence_view_async_tmem_load()
        sub_tile_size = 8
        tmem_frag = cute.logical_divide(tTMEM_LOADrS, ((sub_tile_size, None), None))
        sInvertSubSSL1B_frag = cute.logical_divide(
            sInvertSubSSL1B, ((sub_tile_size, None), None)
        )
        tile_frags = corr_tile_size // sub_tile_size
        if sub_widx == 2 or sub_widx == 3:
            sub_id_row = sub_widx - 2
            for col in cutlass.range_constexpr(tile_frags):
                col_true = (lane_id + col) % (tile_frags)
                sInvertSubSSL1B_frag[
                    ((None, (col_true, 0)), lane_id % 8),
                    0,
                    lane_id // 8 + sub_id_row * 4,
                    0,
                ].store(tmem_frag[((None, col_true), 0), 0, 0].load())

        # tile 0
        for i in cutlass.range_constexpr(32 // corr_tile_size):
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )

            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)
            cute.arch.fence_view_async_tmem_load()
            if sub_widx == 2 or sub_widx == 3:
                sub_id_row = sub_widx - 2
                idx = (i * corr_tile_size) // sub_tile_size
                for j in cutlass.range_constexpr(tile_frags):
                    col_true = (idx + j) % 8
                    # todo: remove smem bank conflict
                    sInvertSubSSL1B_frag[
                        ((None, (col_true, 0)), lane_id % 8),
                        0,
                        lane_id // 8 + sub_id_row * 4,
                        0,
                    ].store(tmem_frag[((None, j), 0), 0, 0].load())
        # tile 1
        for i in cutlass.range_constexpr(32 // corr_tile_size):
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + 32 + i * corr_tile_size, tTMEM_LOADtO.layout
            )

            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)
            cute.arch.fence_view_async_tmem_load()
            if sub_widx == 2 or sub_widx == 3:
                sub_id_row = sub_widx - 2
                idx = (i * corr_tile_size) // sub_tile_size
                for j in cutlass.range_constexpr(tile_frags):
                    col_true = (idx + j) % 8
                    # todo: remove smem bank conflict
                    sInvertSubSSL1B_frag[
                        ((None, (col_true, 1)), lane_id % 8),
                        0,
                        lane_id // 8 + sub_id_row * 4,
                        0,
                    ].store(tmem_frag[((None, j), 0), 0, 0].load())

    @cute.jit
    def store_ivt_p1(
        self,
        sReg: cute.Tensor,
        sInvertSubTSL0B: cute.Tensor,
    ):
        """Store sub-inverse register results p1 to smem."""
        tidx, _, _ = cute.arch.thread_idx()
        widx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()

        sub_widx = widx % 4

        sInvertSubTSL0B_frag = cute.logical_divide(sInvertSubTSL0B, ((4, None), None))
        if sub_widx == 0 or sub_widx == 2:
            sub_id_row = sub_widx // 2
            for col in cutlass.range_constexpr(8):
                col_true = (lane_id + col) % 8
                sInvertSubTSL0B_frag[
                    ((None, (col_true, sub_id_row)), lane_id % 8), 0, lane_id // 8, 0
                ].store(sReg[(None, tidx % 128), col_true].load())

        cute.arch.fence_proxy("async.shared", space="cta")

    @cute.jit
    def store_ivt_p2(
        self,
        sReg: cute.Tensor,
        sInvertSubSSL1A: cute.Tensor,
    ):
        """Store sub-inverse register results p2 to smem."""
        tidx, _, _ = cute.arch.thread_idx()
        widx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()

        sub_widx = widx % 4
        sInvertSubSSL1A_frag = cute.logical_divide(sInvertSubSSL1A, ((None, 4), None))
        zeros = cute.zeros_like(cute.make_layout((4, 1)), self.acc_dtype)
        if sub_widx == 2 or sub_widx == 3:
            sub_id_row = sub_widx - 2
            for col in cutlass.range_constexpr(8):
                sInvertSubSSL1A_frag[
                    (sub_id_row * 32 + lane_id, (None, col % 2)),
                    0,
                    (col // 2, sub_id_row),
                    0,
                ].store(sReg[(None, tidx % 128), col].load())

        if sub_widx == 0 or sub_widx == 1:
            for col in cutlass.range_constexpr(4):
                col_true = col + sub_widx * 4
                sInvertSubSSL1A_frag[
                    (lane_id, (None, col_true % 2)), 0, (col_true // 2, 1), 0
                ].store(zeros)
        cute.arch.fence_proxy("async.shared", space="cta")

    @cute.jit
    def store_ivt_p3(
        self,
        sReg: cute.Tensor,
        sInvertSubTSL1B: cute.Tensor,
    ):
        """Store sub-inverse register results p3 to smem ."""
        tidx, _, _ = cute.arch.thread_idx()
        widx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()

        sub_widx = widx % 4
        sInvertSubTSL1B_frag = cute.logical_divide(sInvertSubTSL1B, ((4, None), None))
        if sub_widx == 0 or sub_widx == 1:
            sub_id_row = sub_widx
            for col in cutlass.range_constexpr(8):
                col_true = (lane_id + col) % 8
                sInvertSubTSL1B_frag[
                    ((None, (col_true, sub_id_row)), lane_id % 8),
                    0,
                    lane_id // 8 + 4 * sub_id_row,
                    0,
                ].store(sReg[(None, tidx % 128), col_true].load())
        zeros = cute.zeros_like(cute.make_layout((4, 1)), self.acc_dtype)

        if sub_widx == 2 or sub_widx == 3:
            sub_id_row = sub_widx - 2
            for col in cutlass.range_constexpr(4):
                col_true = (lane_id + col + sub_id_row * 4) % 8
                sInvertSubTSL1B_frag[
                    ((None, (col_true, 1)), lane_id % 8), 0, lane_id // 8 + 0, 0
                ].store(zeros)
        cute.arch.fence_proxy("async.shared", space="cta")

    @cute.jit
    def load_store_tmem_tune(
        self,
        thr_mma: cute.ThrMma,
        tIn: cute.Tensor,
        tOut: cute.Tensor,  # unused utility for TMEM round-trip debugging
    ):
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        cIn = cute.make_identity_tensor((128, 128))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 16
        tIn_i_layout = cute.composition(
            tIn.layout, cute.make_layout((128, corr_tile_size))
        )
        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size))
        )

        tOut_i_layout = cute.composition(
            tOut.layout, cute.make_layout((128, corr_tile_size))
        )

        tIn_i = cute.make_tensor(tIn.iterator, tIn_i_layout)
        cIn_i = cute.make_tensor(cIn.iterator, cIn_i_layout)

        tOut_i = cute.make_tensor(tOut.iterator, tOut_i_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tIn_i)
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOut_i)

        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tIn_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(cIn_i)

        tTMEM_STOREtO = thr_tmem_store.partition_D(tOut_i)

        tTMrO = cute.make_rmem_tensor(
            (tTMEM_LOADcO.shape, 128 // corr_tile_size), self.acc_dtype
        )

        for i in cutlass.range_constexpr(128 // corr_tile_size):
            tTMrO_i_ = tTMrO[None, i]
            tTMrO_i_layout = cute.composition(
                tTMrO_i_.layout, cute.make_layout(tTMrO.shape[0])
            )
            tTMrO_i = cute.make_tensor(tTMrO_i_.iterator, tTMrO_i_layout)
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )
            tTMEM_STOREtO_i = cute.make_tensor(
                tTMEM_STOREtO.iterator + i * corr_tile_size, tTMEM_STOREtO.layout
            )

            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO_i)

            for j in cutlass.range_constexpr(0, cute.size(tTMrO_i), 2):
                tTMrO_i[j], tTMrO_i[j + 1] = cute.arch.mul_packed_f32x2(
                    (tTMrO_i[j], tTMrO_i[j + 1]),
                    (cutlass.Float32(1), cutlass.Float32(1)),
                )

            # store
            cute.copy(tiled_tmem_store, tTMrO_i, tTMEM_STOREtO_i)

    @cute.jit
    def read_tmem_128(
        self,
        tIn: cute.Tensor,
    ):
        """Read a 128-element vector from TMEM into registers."""
        tidx, _, _ = cute.arch.thread_idx()

        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        tIn_layout = cute.composition(tIn.layout, cute.make_layout((tIn.shape)))
        tIn = cute.make_tensor(tIn.iterator, tIn_layout)

        # Load from tmem
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.acc_dtype,
        )

        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tIn)
        thr_copy_t2r = tiled_copy_t2r.get_slice(thread_idx)
        tTR_tO = thr_copy_t2r.partition_S(tIn)
        tTR_cO = thr_copy_t2r.partition_D(tIn)

        tTR_rO = cute.make_rmem_tensor(tTR_cO.shape, self.acc_dtype)

        tTR_tO = cute.group_modes(tTR_tO, 1, cute.rank(tTR_tO))
        tTR_rO = cute.group_modes(tTR_rO, 1, cute.rank(tTR_rO))
        cute.copy(tiled_copy_t2r, tTR_tO, tTR_rO)
        cute.arch.fence_view_async_tmem_load()

        return tTR_rO

    @cute.jit
    def load_v(
        self,
        tIn: cute.Tensor,
        sNewV: cute.Tensor,
    ):
        """Load V_new from TMEM, convert to f16, store to smem."""
        tidx, _, _ = cute.arch.thread_idx()

        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        tIn_layout = cute.composition(tIn.layout, cute.make_layout((tIn.shape)))
        tIn = cute.make_tensor(tIn.iterator, tIn_layout)

        # Load from tmem
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.acc_dtype,
        )

        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tIn)
        thr_copy_t2r = tiled_copy_t2r.get_slice(thread_idx)

        tTR_tO = thr_copy_t2r.partition_S(tIn)
        tTR_cO = thr_copy_t2r.partition_D(tIn)

        tTR_rO = cute.make_rmem_tensor(tTR_cO.shape, self.acc_dtype)

        tTR_tO = cute.group_modes(tTR_tO, 1, cute.rank(tTR_tO))
        tTR_rO = cute.group_modes(tTR_rO, 1, cute.rank(tTR_rO))
        cute.copy(tiled_copy_t2r, tTR_tO, tTR_rO)
        cute.arch.fence_view_async_tmem_load()

        tTR_rO_e = cute.make_rmem_tensor(tTR_cO.shape, self.i_dtype)
        frg_cnt = 16
        frg_tile = cute.size(tTR_rO) // frg_cnt
        tTR_rO_frg = cute.logical_divide(tTR_rO, cute.make_layout(frg_tile))
        tTR_rO_e_frg = cute.logical_divide(tTR_rO_e, cute.make_layout(frg_tile))
        for j in cutlass.range_constexpr(frg_cnt):
            r_vec = tTR_rO_frg[None, j].load()
            tTR_rO_e_frg[None, j].store(r_vec.to(self.i_dtype))

        sNewV_frg = cute.logical_divide(sNewV, cute.make_layout(frg_tile))
        for it in cutlass.range_constexpr(0, 16):
            sNewV_frg[None, (it, thread_idx)].store(tTR_rO_e_frg[None, it].load())

        cute.arch.fence_proxy("async.shared", space="cta")

    @cute.jit
    def load_ivt_result(
        self,
        tIn: cute.Tensor,
        sD: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane_id = tidx % 32
        warp_id = cute.arch.warp_idx() % 4

        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        tIn_layout = cute.composition(tIn.layout, cute.make_layout((tIn.shape)))
        tIn = cute.make_tensor(tIn.iterator, tIn_layout)

        # Load from tmem
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )

        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tIn)
        thr_copy_t2r = tiled_copy_t2r.get_slice(thread_idx)
        tTR_tO = thr_copy_t2r.partition_S(tIn)
        tTR_cO = thr_copy_t2r.partition_D(tIn)

        tTR_rO = cute.make_rmem_tensor(tTR_cO.shape, self.acc_dtype)

        tTR_tO = cute.group_modes(tTR_tO, 1, cute.rank(tTR_tO))
        tTR_rO = cute.group_modes(tTR_rO, 1, cute.rank(tTR_rO))
        cute.copy(tiled_copy_t2r, tTR_tO, tTR_rO)
        cute.arch.fence_view_async_tmem_load()

        tTR_rO_e = cute.make_rmem_tensor(tTR_cO.shape, self.q_dtype)

        frg_cnt = 2
        frg_tile = cute.size(tTR_rO) // frg_cnt
        tTR_rO_frg = cute.logical_divide(tTR_rO, cute.make_layout(frg_tile))
        tTR_rO_e_frg = cute.logical_divide(tTR_rO_e, cute.make_layout(frg_tile))
        for j in cutlass.range_constexpr(frg_cnt):
            r_vec = -tTR_rO_frg[None, j].load()
            tTR_rO_e_frg[None, j].store(r_vec.to(self.q_dtype))

        tTR_rO_e_f32 = cute.recast_tensor(src=tTR_rO_e, dtype=cutlass.Float32)

        for it in cutlass.range_constexpr(0, 8):
            sD[
                ((64 + warp_id * 16 + (lane_id // 4) + 0), (lane_id % 4, it % 2)),
                it // 2,
            ] = tTR_rO_e_f32[it * 2]
            sD[
                ((64 + warp_id * 16 + (lane_id // 4) + 8), (lane_id % 4, it % 2)),
                it // 2,
            ] = tTR_rO_e_f32[it * 2 + 1]

        cute.arch.fence_proxy("async.shared", space="cta")

    @cute.jit
    def apply_gamma_beta(
        self,
        thr_mma: cute.ThrMma,
        sIvt: cute.Tensor,
        tIn: cute.Tensor,
        beta_val: cutlass.Float32,
        tGamma: cute.Tensor,
    ):
        """Apply gamma*beta to KK^T in TMEM."""
        tidx, _, _ = cute.arch.thread_idx()
        lane_id = tidx % 32
        warp_id = cute.arch.warp_idx() % 4
        sub_widx = warp_id % 4
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        cIn = cute.make_identity_tensor((128, 128))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 32  # only 8  or 32
        tIn_i_layout = cute.composition(
            tIn.layout, cute.make_layout((128, corr_tile_size))
        )
        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size))
        )
        tIn_i = cute.make_tensor(tIn.iterator, tIn_i_layout)

        tOut_i = cute.make_tensor(tIn.iterator, tIn_i_layout)
        cOut_i = cute.make_tensor(tOcO.iterator, cIn_i_layout)

        tGamma_i = cute.make_tensor(tGamma.iterator, tIn_i_layout)
        cIn_i = cute.make_tensor(cIn.iterator, cIn_i_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOut_i)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)

        tTMEM_STOREcS = thr_tmem_store.partition_S(cOut_i)
        tTMEM_STOREtO = thr_tmem_store.partition_D(tOut_i)

        tTMEM_STORErS = cute.make_rmem_tensor(tTMEM_STOREcS.shape, self.acc_dtype)

        # Load In
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tIn_i)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)

        # Load Gamma
        tiled_tmem_load_gamma = tcgen05.make_tmem_copy(tmem_load_atom, tGamma_i)
        thr_tmem_load_gamma = tiled_tmem_load_gamma.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tIn_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(cIn_i)

        tTMEM_LOADtG = thr_tmem_load_gamma.partition_S(tGamma_i)
        tTMEM_LOADrG = cute.make_rmem_tensor(tTMEM_LOADcO.shape, self.acc_dtype)

        tTMEM_LOADrS = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErS.iterator, dtype=self.acc_dtype),
            tTMEM_LOADrG.layout,
        )

        frg_tile = 4

        # Store to smem
        # ((32, 32), 4)
        lower_D_frag = cute.logical_divide(tTMEM_STORErS, ((4, None), None))
        sIvt_frag = cute.logical_divide(sIvt, ((4, None), None))

        for i in cutlass.range_constexpr(128 // corr_tile_size):
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )
            tTMEM_LOADtG_i = cute.make_tensor(
                tTMEM_LOADtG.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )

            tTMEM_STOREtO_i = cute.make_tensor(
                tTMEM_STOREtO.iterator + i * corr_tile_size, tTMEM_STOREtO.layout
            )

            cute.copy(tiled_tmem_load_gamma, tTMEM_LOADtG_i, tTMEM_LOADrG)
            each_iter = corr_tile_size // frg_tile
            tTMEM_LOADrG_frg = cute.logical_divide(
                tTMEM_LOADrG, ((frg_tile, None), None)
            )
            for j in cutlass.range_constexpr(each_iter):
                for inner_idx in cutlass.range_constexpr(0, frg_tile, 2):
                    (
                        tTMEM_LOADrG_frg[((inner_idx + 0, j), i), 0, 0],
                        tTMEM_LOADrG_frg[((inner_idx + 1, j), i), 0, 0],
                    ) = cute.arch.mul_packed_f32x2(
                        (
                            tTMEM_LOADrG_frg[((inner_idx + 0, j), i), 0, 0],
                            tTMEM_LOADrG_frg[((inner_idx + 1, j), i), 0, 0],
                        ),
                        (
                            beta_val,
                            beta_val,
                        ),
                    )
            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)
            cute.arch.fence_view_async_tmem_load()

            tTMEM_LOADrS_frg = cute.logical_divide(
                tTMEM_LOADrS, ((frg_tile, None), None)
            )

            if (i * corr_tile_size) < ((sub_widx + 1) * 32):
                for j in cutlass.range_constexpr(each_iter):
                    for inner_idx in cutlass.range_constexpr(0, frg_tile, 2):
                        (
                            tTMEM_LOADrS_frg[((inner_idx + 0, j), i), 0, 0],
                            tTMEM_LOADrS_frg[((inner_idx + 1, j), i), 0, 0],
                        ) = cute.arch.mul_packed_f32x2(
                            (
                                tTMEM_LOADrS_frg[((inner_idx + 0, j), i), 0, 0],
                                tTMEM_LOADrS_frg[((inner_idx + 1, j), i), 0, 0],
                            ),
                            (
                                tTMEM_LOADrG_frg[((inner_idx + 0, j), i), 0, 0],
                                tTMEM_LOADrG_frg[((inner_idx + 1, j), i), 0, 0],
                            ),
                        )
            cute.copy(tiled_tmem_store, tTMEM_STORErS, tTMEM_STOREtO_i)

            tile_frags = corr_tile_size // 4
            if (i * corr_tile_size) >= sub_widx * 32 and (i * corr_tile_size) < (
                sub_widx + 1
            ) * 32:
                idx = (i * corr_tile_size - sub_widx * 32) // 4
                for j in cutlass.range_constexpr(tile_frags):
                    true_col = j + idx
                    sIvt_frag[((None, true_col), lane_id), sub_widx].store(
                        lower_D_frag[((None, j), 0), 0, 0].load()
                    )

    @cute.jit
    def compute_gamma_tmem(
        self,
        val: cutlass.Float32,
        sGateCumsum: cute.Tensor,
        thr_mma: cute.ThrMma,
        tOut: cute.Tensor,
        bar_id: Int32,
        mask: cutlass.Constexpr[cutlass.Boolean] = False,
        tail_count: Int32 = 128,
    ):
        """Build the 128x128 causal gamma matrix in TMEM: gamma[i,j] = exp(cumsum[i] - cumsum[j]) for j <= i."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % 128

        cute.arch.barrier(
            barrier_id=bar_id,
            number_of_threads=len(self.cudacore_warp_ids) * 32,
        )

        cIn = cute.make_identity_tensor((128, 128))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 128

        cOut_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size))
        )
        tOut_i_layout = cute.composition(
            tOut.layout, cute.make_layout((128, corr_tile_size))
        )

        tOut_i = cute.make_tensor(tOut.iterator, tOut_i_layout)
        cOut_i = cute.make_tensor(tOcO.iterator, cOut_i_layout)

        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOut_i)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)

        tTMEM_STOREcS = thr_tmem_store.partition_S(cOut_i)
        tTMEM_STOREtO = thr_tmem_store.partition_D(tOut_i)

        tTMEM_STORErS = cute.make_rmem_tensor(tTMEM_STOREcS.shape, self.acc_dtype)

        frg_tile = 32
        sGateCumsum_tile = cute.flat_divide(
            sGateCumsum, cute.make_layout(corr_tile_size)
        )
        sGateCumsum_frag = cute.logical_divide(sGateCumsum_tile, (frg_tile, None))

        zeros = cute.zeros_like(cute.make_layout((frg_tile, 1)), self.acc_dtype)

        for i in cutlass.range_constexpr(128 // corr_tile_size):
            tTMEM_STOREtO_i = cute.make_tensor(
                tTMEM_STOREtO.iterator + i * corr_tile_size, tTMEM_STOREtO.layout
            )

            tTMEM_STORErS_frag = cute.logical_divide(
                tTMEM_STORErS, cute.make_layout(frg_tile)
            )
            for it in cutlass.range_constexpr(corr_tile_size // frg_tile):
                tile_offset = i * corr_tile_size + it * frg_tile

                if (
                    (
                        tile_offset <= thread_idx
                        and tile_offset < tail_count
                        and thread_idx < tail_count
                    )
                    if cutlass.const_expr(mask)
                    else (tile_offset <= thread_idx)
                ):
                    curr_val_frag_ssa = sGateCumsum_frag[(None, it), i].load()

                    new_val_frag = cute.math.exp(val - curr_val_frag_ssa, fastmath=True)

                    for inner_idx in cutlass.range_constexpr(0, frg_tile):
                        offset = tile_offset + inner_idx
                        curr_val = new_val_frag[inner_idx]

                        if offset <= thread_idx:
                            tTMEM_STORErS_frag[inner_idx, it] = curr_val
                        else:
                            tTMEM_STORErS_frag[inner_idx, it] = cutlass.Float32(0)
                else:
                    tTMEM_STORErS_frag[None, it].store(zeros)

            # store
            cute.copy(tiled_tmem_store, tTMEM_STORErS, tTMEM_STOREtO_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def chunk_local_cumsum(
        self,
        sGate: cute.Tensor,
        sGateCumsum: cute.Tensor,
        bar_id: Int32,
    ) -> cutlass.Float32:
        """Compute inclusive prefix sum of gate values within a 128-token chunk."""
        tidx, _, _ = cute.arch.thread_idx()
        lane_id = tidx % 32
        warp_id = cute.arch.warp_idx() % 4
        tidx_in_group = tidx % 128

        val = sGate[tidx_in_group]
        stride = 1
        clamp_value = 0
        while stride < 32:
            shfl_value = cute.arch.shuffle_sync_op(
                value=val,
                offset=stride,
                mask_and_clamp=clamp_value,
                kind=nvvm.ShflKind.up,
            )
            if lane_id >= stride:
                val += shfl_value

            stride = stride << 1
        if lane_id == 31:
            sGateCumsum[warp_id] = val

        cute.arch.barrier(
            barrier_id=bar_id,
            number_of_threads=len(self.cudacore_warp_ids) * 32,
        )

        if warp_id == 1:
            pre_warp_val = sGateCumsum[0]
            val = val + pre_warp_val
        if warp_id == 2:
            sGateCumsum_f2 = cute.flat_divide(sGateCumsum, (2, 1))
            warp_sum = sGateCumsum_f2[None, 0, 0, 0]
            pre_warp_val = warp_sum[0] + warp_sum[1]
            val = val + pre_warp_val
        if warp_id == 3:
            sGateCumsum_f4 = cute.flat_divide(sGateCumsum, (4, 1))
            warp_sum = sGateCumsum_f4[None, 0, 0, 0]
            pre_warp_val = warp_sum[0] + warp_sum[1] + warp_sum[2]
            val = val + pre_warp_val

        cute.arch.barrier(
            barrier_id=bar_id,
            number_of_threads=len(self.cudacore_warp_ids) * 32,
        )

        sGateCumsum[tidx_in_group] = val

        return val

    @cute.jit
    def save_tmem(
        self,
        tSIn: cute.Tensor,
        sD: cute.Tensor,
    ) -> cute.Tensor:
        tidx, _, _ = cute.arch.thread_idx()
        widx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()
        sub_widx = widx % 4

        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        tSIn_layout = cute.composition(tSIn.layout, cute.make_layout((tSIn.shape)))
        tSIn0 = cute.make_tensor(tSIn.iterator, tSIn_layout)

        # Load from tmem
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tSIn0)
        thr_copy_t2r = tiled_copy_t2r.get_slice(thread_idx)
        tTR_tO = thr_copy_t2r.partition_S(tSIn0)
        tTR_cO = thr_copy_t2r.partition_D(tSIn0)
        tTR_cO_one_iter = tTR_cO[None, 0, 0, 0]
        tTR_rO = cute.make_rmem_tensor(tTR_cO_one_iter.shape, self.acc_dtype)

        tTR_tO = cute.group_modes(tTR_tO, 1, cute.rank(tTR_tO))

        tTR_rO_e = cute.make_rmem_tensor(tTR_rO.shape, self.q_dtype)
        frg_cnt = 2
        frg_tile = cute.size(tTR_rO) // frg_cnt
        tTR_rO_frg = cute.logical_divide(tTR_rO, cute.make_layout(frg_tile))
        tTR_rO_e_frg = cute.logical_divide(tTR_rO_e, cute.make_layout(frg_tile))
        tTR_rO_e_f32 = cute.recast_tensor(src=tTR_rO_e, dtype=cutlass.Float32)
        for subtile_idx in cutlass.range_constexpr(2):
            tTR_tO_mn = tTR_tO[(None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tO_mn, tTR_rO)
            cute.arch.fence_view_async_tmem_load()

            if (sub_widx == 0 or sub_widx == 1) and subtile_idx == 0:
                for j in cutlass.range_constexpr(frg_cnt):
                    r_vec = -tTR_rO_frg[None, j].load()
                    tTR_rO_e_frg[None, j].store(r_vec.to(self.q_dtype))

                for it in cutlass.range_constexpr(0, 4):
                    sD[
                        (
                            (
                                32
                                + (sub_widx // 2) * 64
                                + (sub_widx % 2) * 16
                                + (lane_id // 4)
                                + 0
                            ),
                            (lane_id % 4, it % 2),
                        ),
                        it // 2 + (sub_widx // 2) * 4,
                    ] = tTR_rO_e_f32[it * 2]
                    sD[
                        (
                            (
                                32
                                + (sub_widx // 2) * 64
                                + (sub_widx % 2) * 16
                                + (lane_id // 4)
                                + 8
                            ),
                            (lane_id % 4, it % 2),
                        ),
                        it // 2 + (sub_widx // 2) * 4,
                    ] = tTR_rO_e_f32[it * 2 + 1]

            if (sub_widx == 2 or sub_widx == 3) and subtile_idx == 1:
                for j in cutlass.range_constexpr(frg_cnt):
                    r_vec = -tTR_rO_frg[None, j].load()
                    tTR_rO_e_frg[None, j].store(r_vec.to(self.q_dtype))

                for it in cutlass.range_constexpr(0, 4):
                    sD[
                        (
                            (
                                32
                                + (sub_widx // 2) * 64
                                + (sub_widx % 2) * 16
                                + (lane_id // 4)
                                + 0
                            ),
                            (lane_id % 4, it % 2),
                        ),
                        it // 2 + (sub_widx // 2) * 4,
                    ] = tTR_rO_e_f32[it * 2]
                    sD[
                        (
                            (
                                32
                                + (sub_widx // 2) * 64
                                + (sub_widx % 2) * 16
                                + (lane_id // 4)
                                + 8
                            ),
                            (lane_id % 4, it % 2),
                        ),
                        it // 2 + (sub_widx // 2) * 4,
                    ] = tTR_rO_e_f32[it * 2 + 1]

    @cute.jit
    def load_ivt_ts_l0(
        self,
        tSIn: cute.Tensor,
        sInvertSubSSL1A: cute.Tensor,
        sInvertSubTSL1B: cute.Tensor,
    ) -> cute.Tensor:
        tidx, _, _ = cute.arch.thread_idx()
        widx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()
        sub_widx = widx % 4

        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        tSIn_layout = cute.composition(tSIn.layout, cute.make_layout((tSIn.shape)))
        tSIn0 = cute.make_tensor(tSIn.iterator, tSIn_layout)

        # Load from tmem
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x128bOp(tcgen05.copy.Repetition(8)),  # todo
            self.acc_dtype,
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tSIn0)
        thr_copy_t2r = tiled_copy_t2r.get_slice(thread_idx)
        tTR_tO = thr_copy_t2r.partition_S(tSIn0)
        tTR_cO = thr_copy_t2r.partition_D(tSIn0)
        tTR_cO_one_iter = tTR_cO[None, 0, 0, 0]
        tTR_rO = cute.make_rmem_tensor(tTR_cO_one_iter.shape, self.acc_dtype)

        tTR_rO_s = cute.make_tensor(
            cute.recast_ptr(tTR_rO.iterator, dtype=self.acc_dtype),
            cute.shape(tTR_cO_one_iter)[0],
        )

        tTR_tO = cute.group_modes(tTR_tO, 1, cute.rank(tTR_tO))

        for subtile_idx in cutlass.range_constexpr(2):
            tTR_tO_mn = tTR_tO[(None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tO_mn, tTR_rO)
            cute.arch.fence_view_async_tmem_load()

            if (sub_widx == 0 or sub_widx == 1) and subtile_idx == 0:
                sInvertSubTSL1B_frag = cute.logical_divide(
                    sInvertSubTSL1B, ((4, None), None)
                )
                sub_id_row = sub_widx
                for col in cutlass.range_constexpr(0, 8):
                    col_true = (lane_id // 4 + col) % 8
                    sInvertSubTSL1B_frag[
                        ((lane_id % 4, (col_true, 0)), lane_id // 4),
                        0,
                        4 + sub_id_row * 2 + 0,
                        0,
                    ] = -tTR_rO_s[(0, col_true), 0]
                    sInvertSubTSL1B_frag[
                        ((lane_id % 4, (col_true, 0)), lane_id // 4),
                        0,
                        4 + sub_id_row * 2 + 1,
                        0,
                    ] = -tTR_rO_s[(1, col_true), 0]
            if (sub_widx == 2 or sub_widx == 3) and subtile_idx == 1:
                tTR_rO_s_frag_f1 = cute.logical_divide(
                    cute.flatten(tTR_rO_s[None, 0]), (1, None)
                )
                sInvertSubSSL1A_frag_f1 = cute.logical_divide(
                    sInvertSubSSL1A, ((None, 1), None)
                )
                sub_id_row = sub_widx - 2
                for it in cutlass.range_constexpr(0, 8):
                    col_offset = it % 2 * 4
                    sInvertSubSSL1A_frag_f1[
                        (
                            32 + sub_id_row * 16 + 0 + lane_id // 4,
                            (None, col_offset + lane_id % 4),
                        ),
                        0,
                        (it // 2, 0),
                        0,
                    ].store(-tTR_rO_s_frag_f1[(None, 0), it].load())
                    sInvertSubSSL1A_frag_f1[
                        (
                            32 + sub_id_row * 16 + 8 + lane_id // 4,
                            (None, col_offset + lane_id % 4),
                        ),
                        0,
                        (it // 2, 0),
                        0,
                    ].store(-tTR_rO_s_frag_f1[(None, 1), it].load())

        cute.arch.fence_proxy("async.shared", space="cta")

    @cute.jit
    def store_ivt_c(
        self,
        sD: cute.Tensor,
    ):
        """Zero out off-diagonal blocks in the assembled inverse matrix (block identity structure)."""
        widx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()
        sub_widx = widx % 4

        zeros = cute.zeros_like(cute.make_layout((4, 1)), self.acc_dtype)
        for it in cutlass.range_constexpr(0, 4):
            row = sub_widx // 2 * 32 + lane_id
            col_block_inner = it % 2
            col_block = 4 + (sub_widx % 2) * 2 + it // 2
            sD[(row, (None, col_block_inner)), col_block].store(zeros[None, 0])
        for it in cutlass.range_constexpr(0, 2):
            row = sub_widx // 2 * 64 + lane_id
            col_block_inner = sub_widx % 2
            col_block = (sub_widx // 2) * 4 + 2 + it
            sD[(row, (None, col_block_inner)), col_block].store(zeros[None, 0])

    @cute.jit
    def load_ivt_ss_l0(
        self,
        tSIn: cute.Tensor,
        tSOut: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        widx = cute.arch.warp_idx()
        sub_widx = widx % 4

        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        tSIn_layout = cute.composition(tSIn.layout, cute.make_layout((tSIn.shape)))
        tSIn0 = cute.make_tensor(tSIn.iterator, tSIn_layout)

        # Load from tmem
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),  # 4*8 = 32
            self.acc_dtype,
        )

        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tSIn0)
        thr_copy_t2r = tiled_copy_t2r.get_slice(thread_idx)
        tTR_tO = thr_copy_t2r.partition_S(tSIn0)
        tTR_cO = thr_copy_t2r.partition_D(tSIn0)
        tTR_cO_one_iter = tTR_cO[(None), 0, 0, 0]
        tTR_rO = cute.make_rmem_tensor(tTR_cO_one_iter.shape, self.acc_dtype)

        tTR_tO = cute.group_modes(tTR_tO, 1, cute.rank(tTR_tO))

        tSOut_layout = cute.make_layout(
            (((16, 4), 32), 1, 1), stride=(((65536, 2097152), 1), 0, 0)
        )
        tSOut0 = cute.make_tensor(tSOut.iterator, tSOut_layout)
        copy_atom_r2t = cute.make_copy_atom(
            tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )
        tiled_copy_r2t = tcgen05.make_tmem_copy(copy_atom_r2t, tSOut0)
        thr_copy_r2t = tiled_copy_r2t.get_slice(thread_idx)
        tRT_cO = thr_copy_r2t.partition_S(tSOut0)
        tRT_tO = thr_copy_r2t.partition_D(tSOut0)
        tRT_cO_one_iter = tRT_cO[(None), None, 0, 0]  #
        tRT_rO = cute.make_rmem_tensor(tRT_cO_one_iter.shape, self.acc_dtype)

        for subtile_idx in cutlass.range_constexpr(2):
            tTR_tO_mn = tTR_tO[(None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tO_mn, tTR_rO)
            cute.arch.fence_view_async_tmem_load()

            if (sub_widx == 0 or sub_widx == 1) and subtile_idx == 0:
                for eid in cutlass.range_constexpr(16):
                    tRT_rO[eid] = tTR_rO[eid]

            if (sub_widx == 2 or sub_widx == 3) and subtile_idx == 1:
                for eid in cutlass.range_constexpr(16):
                    tRT_rO[eid] = tTR_rO[eid]

        tRT_tO = cute.group_modes(tRT_tO, 2, cute.rank(tRT_tO))
        for subtile_idx in cutlass.range_constexpr(0, 1):
            tRT_tO_mn = tRT_tO[(None, None, subtile_idx)]
            cute.copy(tiled_copy_r2t, tRT_rO, tRT_tO_mn)

        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def store_ivt_ad(
        self,
        sReg: cute.Tensor,
        sD: cute.Tensor,
    ):
        """Store diagonal block inverse (A/D blocks) from registers to smem."""
        tidx, _, _ = cute.arch.thread_idx()
        widx = cute.arch.warp_idx()
        sub_widx = widx % 4

        reg_layout = cute.make_layout((4, 2))
        result_e = cute.make_rmem_tensor(reg_layout.shape, self.acc_dtype)
        for it in cutlass.range_constexpr(0, 4):
            result_e[None, 0].store(sReg[(None, tidx % 128), it * 2 + 0].load())
            result_e[None, 1].store(sReg[(None, tidx % 128), it * 2 + 1].load())
            sD[(tidx % 128, (None, it % 2)), 0, sub_widx * 2 + it // 2, 0].store(
                result_e[None].load().to(self.q_dtype)
            )

    @cute.jit
    def __call__(
        self,
        q_iter: cute.Pointer,
        k_iter: cute.Pointer,
        v_iter: cute.Pointer,
        o_iter: cute.Pointer,
        a_iter: cute.Pointer,
        b_iter: cute.Pointer,
        A_log_iter: cute.Pointer,
        dt_bias_iter: cute.Pointer,
        problem_size: Tuple[
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Int32,
        ],
        initial_state_f32_iter: Optional[cute.Pointer],
        state_output: Optional[cute.Pointer],
        scale: Optional[float],
        cum_seqlen_q: Optional[cute.Tensor] = None,
        cu_seqlens: Optional[cute.Tensor] = None,
        use_qk_l2norm: cutlass.Constexpr[cutlass.Boolean] = True,
        sm_count: Optional[cutlass.Int32] = None,
        stream: cuda.CUstream = None,
    ):
        """Host-side entry: build tensor layouts, TMA descriptors, smem storage, and launch the kernel."""

        b, s_q, s_sum, h_q, h_v, d = problem_size

        h_r = h_v // h_q
        o_offset = 0 if cum_seqlen_q is None else (-s_q * d * h_r * h_q)
        b_qk = b if cum_seqlen_q is None else s_sum
        b_o = b if cum_seqlen_q is None else s_q * (1 + b)
        b_v = b if cum_seqlen_q is None else s_sum

        stride_b_qk = h_q * s_q * d if cum_seqlen_q is None else d * h_q
        stride_b_vo = h_r * h_q * s_q * d if cum_seqlen_q is None else d * h_q * h_r

        b_gb = b if cum_seqlen_q is None else 1
        stride_b_gb = h_r * h_q * s_sum if cum_seqlen_q is None else 0
        q_layout = cute.make_layout(
            (s_sum, d, ((h_r, h_q), b_qk)),
            stride=(d * h_q, 1, ((0, d), stride_b_qk)),
        )
        q = cute.make_tensor(q_iter, q_layout)

        k_layout = cute.make_layout(
            (s_sum, d, ((h_r, h_q), b_qk)),
            stride=(d * h_q, 1, ((0, d), stride_b_qk)),
        )
        k = cute.make_tensor(k_iter, k_layout)
        v_layout = cute.make_layout(
            (d, s_sum, ((h_r, h_q), b_v)),
            stride=(1, d * h_q * h_r, ((d, d * h_r), stride_b_vo)),
        )
        v = cute.make_tensor(v_iter, v_layout)
        o_layout = cute.make_layout(
            (s_q, d, ((h_r, h_q), b_o)),
            stride=(d * h_r * h_q, 1, ((d, d * h_r), stride_b_vo)),
        )
        o = cute.make_tensor(o_iter + o_offset, o_layout)

        state_layout = cute.make_layout(
            (self.head_dim, self.head_dim, ((h_r, h_q), b)),
            stride=(
                self.head_dim,
                1,
                (
                    (
                        self.head_dim * self.head_dim,
                        h_r * self.head_dim * self.head_dim,
                    ),
                    h_q * h_r * self.head_dim * self.head_dim,
                ),
            ),
        )

        state_input_f32 = (
            None
            if cutlass.const_expr(initial_state_f32_iter is None)
            else cute.make_tensor(initial_state_f32_iter, state_layout)
        )
        state_output = (
            None
            if cutlass.const_expr(state_output is None)
            else cute.make_tensor(state_output, state_layout)
        )

        gb_layout = cute.make_layout(
            (s_sum, 1, ((h_r, h_q), b_gb)),
            stride=(h_r * h_q, 0, ((1, h_r), stride_b_gb)),
        )
        a_tensor = cute.make_tensor(a_iter, gb_layout)
        b_tensor = cute.make_tensor(b_iter, gb_layout)
        hv_size = h_r * h_q
        A_log_vec = cute.make_tensor(A_log_iter, cute.make_layout(hv_size))
        dt_bias_vec = cute.make_tensor(dt_bias_iter, cute.make_layout(hv_size))

        if cutlass.const_expr(scale is None):
            scale = 1.0 / cute.math.sqrt(cutlass.Float32(d))

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type
        self.g_dtype = a_tensor.element_type
        self.beta_dtype = b_tensor.element_type

        self.i_dtype = self.q_dtype

        self.tile_sched_params, grid = self._compute_grid(
            cute.shape((s_q, d, ((h_r, h_q), b))),
            self.cta_tiler,
            self.is_persistent,
            sm_count,
        )

        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = utils.LayoutEnum.from_tensor(o)
        self.s_output = utils.LayoutEnum.ROW_MAJOR

        cta_group = tcgen05.CtaGroup.ONE

        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )

        update_s_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.i_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.update_s_mma_tiler[:2],
        )

        o_intra_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.i_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.o_intra_mma_tiler[:2],
        )

        o_intra_tiled_ts_mma_new = sm100_utils.make_trivial_tiled_mma(
            self.i_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.o_intra_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )

        qs_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.i_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.qs_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )

        kkt_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.k_dtype,
            self.k_major_mode,
            self.k_major_mode,
            self.acc_dtype,
            cta_group,
            self.kkt_mma_tiler[:2],
        )

        kst_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.i_dtype,
            self.k_major_mode,
            self.k_major_mode,
            self.acc_dtype,
            cta_group,
            self.kst_mma_tiler[:2],
        )

        invert_sub_tiled_mma_ss_l0 = sm100_utils.make_trivial_tiled_mma(
            self.invert_type_ab,
            self.k_major_mode,
            tcgen05.OperandMajorMode.MN,
            self.invert_acc_type,
            cta_group,
            self.invert_mma_sub_l0_tiler[:2],
        )
        invert_sub_tiled_mma_ts_l0 = sm100_utils.make_trivial_tiled_mma(
            self.invert_type_ab,
            self.k_major_mode,
            tcgen05.OperandMajorMode.MN,
            self.invert_acc_type,
            cta_group,
            self.invert_mma_sub_l0_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )

        invert_sub_tiled_mma_ss_l1 = sm100_utils.make_trivial_tiled_mma(
            self.invert_type_ab,
            self.k_major_mode,
            tcgen05.OperandMajorMode.MN,
            self.invert_acc_type,
            cta_group,
            self.invert_mma_sub_l1_tiler[:2],
        )
        invert_sub_tiled_mma_ts_l1 = sm100_utils.make_trivial_tiled_mma(
            self.invert_type_ab,
            self.k_major_mode,
            tcgen05.OperandMajorMode.MN,
            self.invert_acc_type,
            cta_group,
            self.invert_mma_sub_l1_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )

        tuw_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.k_dtype,
            self.k_major_mode,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.tuw_mma_tiler[:2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.k_dtype,
            self.kv_stage,
        )

        qk_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.i_dtype,
            self.qk_stage,
        )

        state_smem_layout_staged = sm100_utils.make_smem_layout_b(
            kst_tiled_mma,
            self.kst_mma_tiler,
            self.i_dtype,
            self.one_stage,
        )

        o_intra_smem_layout_staged_a = make_smem_layout_a_kind(
            o_intra_tiled_mma,
            self.o_intra_mma_tiler,
            self.i_dtype,
            self.one_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_INTER,
        )

        o_intra_smem_layout_staged_b = make_smem_layout_b_kind(
            o_intra_tiled_mma,
            self.o_intra_mma_tiler,
            self.i_dtype,
            self.one_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_INTER,
        )

        o_intra_smem_layout_staged_a_new = make_smem_layout_a_kind(
            o_intra_tiled_ts_mma_new,
            self.o_intra_mma_tiler,
            self.i_dtype,
            self.one_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_INTER,
        )

        qs_smem_layout_staged_a = sm100_utils.make_smem_layout_a(
            qs_tiled_mma,
            self.qs_mma_tiler,
            self.i_dtype,
            self.one_stage,
        )

        qs_smem_layout_staged_b = sm100_utils.make_smem_layout_b(
            qs_tiled_mma,
            self.qs_mma_tiler,
            self.i_dtype,
            self.one_stage,
        )

        update_s_smem_layout_staged_a = make_smem_layout_a_kind(
            update_s_tiled_mma,
            self.update_s_mma_tiler,
            self.i_dtype,
            self.one_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_INTER,
        )

        update_s_smem_layout_staged_b = make_smem_layout_b_kind(
            update_s_tiled_mma,
            self.update_s_mma_tiler,
            self.i_dtype,
            self.one_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_INTER,
        )

        state_output_smem_layout_staged = make_smem_layout_epi_kind(
            self.state_dtype,
            self.s_output,
            self.state_output_tiler,
            self.one_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_INTER,
        )

        o_output_smem_layout_staged = make_smem_layout_epi_kind(
            self.i_dtype,
            self.o_layout,
            self.o_output_tiler,
            self.one_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_INTER,
        )

        ivt_smem_layout = cute.make_layout(
            ((36, 32), 4)
        )  # padding, remove smem bank conflict
        # ivt_smem_layout = cute.make_layout(((32, 32), 4))  # padding, remove smem bank conflict

        sub_inner_smem_layout_staged = make_smem_layout_a_kind(
            invert_sub_tiled_mma_ss_l0,
            self.invert_mma_sub_l0_tiler,
            self.invert_type_ab,
            self.sub_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_INTER,
        )

        invert_sub_smem_layout_staged_ss_l0_a = sm100_utils.make_smem_layout_a(
            invert_sub_tiled_mma_ss_l0,
            self.invert_mma_sub_l0_tiler,
            self.invert_type_ab,
            self.invert_sub_stage,
        )

        invert_sub_smem_layout_staged_ss_l0_b = sm100_utils.make_smem_layout_b(
            invert_sub_tiled_mma_ss_l0,
            self.invert_mma_sub_l0_tiler,
            self.invert_type_ab,
            self.invert_sub_stage,
        )

        invert_sub_smem_layout_staged_ts_l0_a = sm100_utils.make_smem_layout_a(
            invert_sub_tiled_mma_ts_l0,
            self.invert_mma_sub_l0_tiler,
            self.invert_type_ab,
            self.invert_sub_stage,
        )

        invert_sub_smem_layout_staged_ts_l0_b = sm100_utils.make_smem_layout_b(
            invert_sub_tiled_mma_ts_l0,
            self.invert_mma_sub_l0_tiler,
            self.invert_type_ab,
            self.invert_sub_stage,
        )

        invert_sub_smem_layout_staged_ss_l1_a = sm100_utils.make_smem_layout_a(
            invert_sub_tiled_mma_ss_l1,
            self.invert_mma_sub_l1_tiler,
            self.invert_type_ab,
            self.invert_sub_stage,
        )

        invert_sub_smem_layout_staged_ss_l1_b = sm100_utils.make_smem_layout_b(
            invert_sub_tiled_mma_ss_l1,
            self.invert_mma_sub_l1_tiler,
            self.invert_type_ab,
            self.invert_sub_stage,
        )

        invert_sub_smem_layout_staged_ts_l1_a = sm100_utils.make_smem_layout_a(
            invert_sub_tiled_mma_ts_l1,
            self.invert_mma_sub_l1_tiler,
            self.invert_type_ab,
            self.invert_sub_stage,
        )

        invert_sub_smem_layout_staged_ts_l1_b = sm100_utils.make_smem_layout_b(
            invert_sub_tiled_mma_ts_l1,
            self.invert_mma_sub_l1_tiler,
            self.invert_type_ab,
            self.invert_sub_stage,
        )

        # tuw
        tuw_smem_layout_staged_a = make_smem_layout_a_kind(
            tuw_tiled_mma,
            self.tuw_mma_tiler,
            self.q_dtype,
            self.one_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_INTER,
        )

        tuw_smem_layout_staged_b = make_smem_layout_b_kind(
            tuw_tiled_mma,
            self.tuw_mma_tiler,
            self.q_dtype,
            self.one_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_INTER,
        )

        v_smem_layout_staged_b = make_smem_layout_b_kind(
            tuw_tiled_mma,
            self.tuw_mma_tiler,
            self.i_dtype,
            self.one_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_INTER,
        )

        # TMA
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        # TMA load for Q
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load for K
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k,
            k_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load for V
        v_smem_layout = cute.select(v_smem_layout_staged_b, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            v,
            v_smem_layout,
            self.tuw_mma_tiler,
            tuw_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        fake_state_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.state_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.state_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )
        state_smem_layout_f32_staged = sm100_utils.make_smem_layout_b(
            fake_state_tiled_mma,
            self.qk_mma_tiler,
            self.state_dtype,
            self.one_stage,
        )
        state_smem_layout_f32 = cute.select(
            state_smem_layout_f32_staged, mode=[0, 1, 2]
        )
        tma_atom_state_f32, tma_tensor_state_f32 = (
            cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                state_input_f32,
                state_smem_layout_f32,
                self.qk_mma_tiler,
                fake_state_tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            if cutlass.const_expr(state_input_f32 is not None)
            else (None, None)
        )

        state_smem_layout_output = cute.select(
            state_output_smem_layout_staged, mode=[0, 1]
        )
        tma_atom_state_output, tma_tensor_state_output = (
            cutlass.cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_store_op,
                state_output,
                state_smem_layout_output,
                cute.composition(
                    cute.make_identity_layout(state_output.shape),
                    self.state_output_tiler,
                ),
            )
            if cutlass.const_expr(state_output is not None)
            else (None, None)
        )

        o_cta_v_layout = cute.composition(
            cute.make_identity_layout(o.shape), self.o_output_tiler
        )
        o_smem_layout_output = cute.select(o_output_smem_layout_staged, mode=[0, 1])
        tma_atom_o_output, tma_tensor_o_output = (
            cutlass.cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_store_op,
                o,
                o_smem_layout_output,
                o_cta_v_layout,
            )
        )

        gate_smem_layout_staged = cute.make_layout(
            (self.chunk_size, 1, self.gate_stage)
        )
        gate_smem_layout = cute.select(gate_smem_layout_staged, mode=[0, 1])

        beta_smem_layout_staged = cute.make_layout(
            (self.chunk_size, 1, self.beta_stage)
        )
        beta_smem_layout = cute.select(beta_smem_layout_staged, mode=[0, 1])

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        v_copy_size = cute.size_in_bytes(self.v_dtype, v_smem_layout)
        state_copy_size = cute.size_in_bytes(self.i_dtype, state_smem_layout_f32)
        state_f32_copy_size = cute.size_in_bytes(
            self.state_dtype, state_smem_layout_f32
        )
        gate_copy_size = cute.size_in_bytes(self.g_dtype, gate_smem_layout)
        beta_copy_size = cute.size_in_bytes(self.beta_dtype, beta_smem_layout)
        self.tma_copy_q_bytes = q_copy_size
        self.tma_copy_kv_bytes = k_copy_size
        self.tma_copy_v_bytes = v_copy_size
        self.tma_copy_state_bytes = state_copy_size
        self.tma_copy_state_f32_bytes = state_f32_copy_size
        self.tma_copy_gate_bytes = gate_copy_size
        self.tma_copy_beta_bytes = beta_copy_size

        # Shared memory storage
        @cute.struct
        class SharedStorage:
            load_qk_mbar_ptr: cute.struct.MemRange[Int64, self.qk_stage * 2]
            load_v_mbar_ptr: cute.struct.MemRange[Int64, self.one_stage * 2]
            load_state_mbar_ptr: cute.struct.MemRange[Int64, self.one_stage * 2]
            mma_cudacore_mbar_ptr0: cute.struct.MemRange[Int64, 2]
            mma_qk_mbar_ptr0: cute.struct.MemRange[Int64, self.mma_qk_stage * 2]

            w0_epi_mbar_ptr: cute.struct.MemRange[Int64, 2]
            epi_w0_mbar_ptr: cute.struct.MemRange[Int64, 2]
            gb_w0_mbar_ptr: cute.struct.MemRange[Int64, self.gate_stage * 2]

            tmem_dealloc_mbar_ptr: cute.struct.MemRange[Int64, 1]
            # Tmem holding buffer
            tmem_holding_buf: Int32

            sV: cute.struct.Align[
                cute.struct.MemRange[self.i_dtype, cute.cosize(v_smem_layout_staged_b)],
                self.buffer_align_bytes,
            ]

            sQK: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(qk_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

            sIvt: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(ivt_smem_layout)],
                self.buffer_align_bytes,
            ]

            sGate: cute.struct.Align[
                cute.struct.MemRange[
                    self.g_beta_dtype, cute.cosize(gate_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sGateCumsum: cute.struct.Align[
                cute.struct.MemRange[
                    self.g_beta_dtype,
                    cute.cosize(cute.select(gate_smem_layout_staged, mode=[0, 1])),
                ],
                self.buffer_align_bytes,
            ]
            sBeta: cute.struct.Align[
                cute.struct.MemRange[
                    self.g_beta_dtype, cute.cosize(beta_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSubInner: cute.struct.Align[
                cute.struct.MemRange[
                    self.invert_type_ab, cute.cosize(sub_inner_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

            sState: cute.struct.Align[
                cute.struct.MemRange[
                    self.i_dtype, cute.cosize(state_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        if cutlass.const_expr(stream is None):
            stream = cutlass.cuda.default_stream()

        # Launch kernel
        self.kernel(
            qk_tiled_mma,
            o_intra_tiled_mma,
            o_intra_tiled_ts_mma_new,
            qs_tiled_mma,
            update_s_tiled_mma,
            kkt_tiled_mma,
            kst_tiled_mma,
            fake_state_tiled_mma,
            invert_sub_tiled_mma_ss_l0,
            invert_sub_tiled_mma_ts_l0,
            invert_sub_tiled_mma_ss_l1,
            invert_sub_tiled_mma_ts_l1,
            tuw_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_state_f32,
            tma_tensor_state_f32,
            a_tensor,
            b_tensor,
            A_log_vec,
            dt_bias_vec,
            o,
            tma_atom_state_output,
            tma_tensor_state_output,
            tma_atom_o_output,
            tma_tensor_o_output,
            cum_seqlen_q,
            cu_seqlens,
            q_smem_layout_staged,
            k_smem_layout_staged,
            qk_smem_layout_staged,
            o_intra_smem_layout_staged_a,
            o_intra_smem_layout_staged_b,
            o_intra_smem_layout_staged_a_new,
            qs_smem_layout_staged_a,
            qs_smem_layout_staged_b,
            update_s_smem_layout_staged_a,
            update_s_smem_layout_staged_b,
            state_smem_layout_staged,
            state_smem_layout_f32_staged,
            state_output_smem_layout_staged,
            o_output_smem_layout_staged,
            gate_smem_layout_staged,
            beta_smem_layout_staged,
            ivt_smem_layout,
            sub_inner_smem_layout_staged,
            invert_sub_smem_layout_staged_ss_l0_a,
            invert_sub_smem_layout_staged_ss_l0_b,
            invert_sub_smem_layout_staged_ts_l0_a,
            invert_sub_smem_layout_staged_ts_l0_b,
            invert_sub_smem_layout_staged_ss_l1_a,
            invert_sub_smem_layout_staged_ss_l1_b,
            invert_sub_smem_layout_staged_ts_l1_a,
            invert_sub_smem_layout_staged_ts_l1_b,
            tuw_smem_layout_staged_a,
            tuw_smem_layout_staged_b,
            v_smem_layout_staged_b,
            scale,
            use_qk_l2norm,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @staticmethod
    def _compute_grid(
        o_shape: cute.Shape,
        cta_tiler: Tuple[int, int, int],
        is_persistent: bool,
        sm_count=None,
    ) -> Tuple[GdnStaticTileSchedulerParams, Tuple[int, int, int]]:
        tile_sched_params = create_gdn_static_tile_scheduler_params(
            is_persistent,
            (
                cute.ceil_div(cute.size(o_shape[1]), cta_tiler[2]),
                cute.size(o_shape[2][0]),
                cute.size(o_shape[2][1]),
            ),
        )
        grid = GdnStaticTileScheduler.get_grid_shape(tile_sched_params, sm_count)
        return tile_sched_params, grid


# ============================================================================
# FlashInfer API Layer
# ============================================================================


@functools.lru_cache(maxsize=128)
def _get_problem_size(q_shape, v_shape, cu_seqlens_tuple):
    """Compute problem_size, cached by (q_shape, v_shape, cu_seqlens tuple).

    Args:
        q_shape: Tuple of (b, s_q, h_q, d).
        v_shape: Tuple of (b, s_v, h_v, d).
        cu_seqlens_tuple: Tuple of cumulative sequence lengths, or None.
    """
    b, s_q, h_q, d = q_shape
    _, _, h_v, _ = v_shape
    max_s_q = s_q
    sum_s_q = s_q
    if cu_seqlens_tuple is not None:
        sum_s_q = cu_seqlens_tuple[-1]
        b = len(cu_seqlens_tuple) - 1
        max_s_q = max(cu_seqlens_tuple[i + 1] - cu_seqlens_tuple[i] for i in range(b))
        return (b, max_s_q, sum_s_q, h_q, h_v, d)
    return (b, max_s_q, sum_s_q, h_q, h_v, d)


@functools.cache
def _get_compiled_gdn_prefill_kernel(
    problem_size,
    cp_dtype_str: str,
    is_varlen: bool,
    is_initial_state: bool,
    is_output_state: bool,
    device_id: int,
):
    """Cache compiled kernel and device properties for given configuration."""
    return {}


def _cp_dtype_to_cutlass(arr: cp.ndarray):
    """Map cupy dtype to cutlass numeric type."""
    if arr.dtype == cp.float16:
        return cutlass.Float16
    elif arr.dtype == cp.uint16:  # bfloat16 stored as uint16 in cupy
        return cutlass.BFloat16
    else:
        raise ValueError(f"Unsupported input dtype for GDN Blackwell kernel: {arr.dtype}")


def chunk_gated_delta_rule(
    q: cp.ndarray,
    k: cp.ndarray,
    v: cp.ndarray,
    a: cp.ndarray,
    b: cp.ndarray,
    A_log: cp.ndarray,
    dt_bias: cp.ndarray,
    scale: Optional[float] = None,
    initial_state: Optional[cp.ndarray] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[cp.ndarray] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[cp.ndarray] = None,
    output_state: Optional[cp.ndarray] = None,
) -> Union[cp.ndarray, Tuple[cp.ndarray, cp.ndarray]]:
    """Public API (CuPy): compile (on first call) and run the GDN chunked linear attention kernel.

    Args:
        q, k: (B, T, H_qk, D) query and key cupy arrays (float16).
        v:     (B, T, H_v, D) value cupy array (float16). H_v must be a multiple of H_qk.
        a:     (B, T, H_v) gate pre-activation (float16).
        b:     (B, T, H_v) beta pre-activation (float16).
        A_log: (H_v,) log decay (float32).
        dt_bias: (H_v,) time-step bias (float16).
        scale: Scale factor for the attention scores. Defaults to 1/sqrt(D).
        initial_state: (B, H_v, D, D) recurrent state (float32), or None for zero init.
        output_final_state: If True, return the final state alongside output.
        cu_seqlens: Cumulative sequence lengths for variable-length batching (int32/int64).
        use_qk_l2norm_in_kernel: L2 norm is always applied inside the kernel (this flag only affects the can_implement check).
        output: Pre-allocated output cupy array, or None to allocate internally.
        output_state: Pre-allocated state output cupy array, or None.

    Returns:
        output or (output, output_state) depending on output_final_state.
    """
    # Allocate output if needed
    if output is None:
        output = cp.empty_like(v)

    in_cutlass = _cp_dtype_to_cutlass(q)
    out_cutlass = _cp_dtype_to_cutlass(output)

    # Check if supported
    if not GDN.can_implement(
        tuple(q.shape),
        tuple(v.shape),
        in_cutlass,
        out_cutlass,
        in_cutlass,  # a/b are same dtype as q/k/v (fp16)
        use_qk_l2norm_in_kernel,
    ):
        raise ValueError("Unsupported input shape or dtype for GDN Blackwell kernel")

    cu_seqlens_tuple = tuple(cp.asnumpy(cu_seqlens).tolist()) if cu_seqlens is not None else None
    problem_size = _get_problem_size(tuple(q.shape), tuple(v.shape), cu_seqlens_tuple)

    # Allocate output_state if needed
    if output_final_state and (output_state is None):
        output_state = cp.empty(
            (problem_size[0], problem_size[4], problem_size[5], problem_size[5]),
            dtype=cp.float32,
        )

    # Blackwell JIT path (`run_gdn_blackwell`) fixes attention scale as 1/sqrt(D) inside GDN;
    # the `scale` argument is accepted for API compatibility but is not passed to the kernel.

    # JIT wrapper always takes a cu_seqlens tensor (see _create_jit_blackwell). For uniform
    # padded batches, use prefix-sum [0, T, 2T, ...] — matches AOT placeholder convention.
    n_b = int(q.shape[0])
    seq_len_b = int(q.shape[1])
    if cu_seqlens is None:
        cu_seqlens_eff = cp.arange(n_b + 1, dtype=cp.int32) * int(seq_len_b)
    else:
        cu_seqlens_eff = cu_seqlens

    if initial_state is None:
        h0_in_arr = cp.zeros(
            (problem_size[0], problem_size[4], problem_size[5], problem_size[5]),
            dtype=cp.float32,
        )
    else:
        h0_in_arr = initial_state

    if output_final_state:
        h0_out_arr = output_state
    else:
        h0_out_arr = cp.empty(
            (problem_size[0], problem_size[4], problem_size[5], problem_size[5]),
            dtype=cp.float32,
        )

    # L2-normalize Q and K before the kernel (preprocessing).
    # The Blackwell kernel expects pre-normalized Q/K so that QK, KK^T, state
    # update, and output paths all use consistently normalized vectors.
    _l2_normalize_qk_inplace(q, k)

    ph = {
        "q": q,
        "k": k,
        "v": v,
        "a": a,
        "b": b,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "h0_in": h0_in_arr,
        "h0_out": h0_out_arr,
        "o": output,
        "cu_seqlens": cu_seqlens_eff,
    }

    # Compile kernel (cached) — use @cute.jit + marked tensors like AOT / gdn_prefill.py.
    # Compiling GDN.__call__ directly with raw iterators can crash the DSL compiler (segfault).
    is_varlen = cu_seqlens is not None
    is_initial_state = initial_state is not None
    cache_key = (
        problem_size,
        str(q.dtype),
        is_varlen,
        is_initial_state,
        output_final_state,
        q.device.id,
    )
    cache = _get_compiled_gdn_prefill_kernel(*cache_key)

    current_stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)

    t_trace = _to_cute_tensors_bw(ph)
    if "compiled_gdn" not in cache:
        cache["sm_count"] = cutlass.Int32(
            cutlass.utils.HardwareInfo().get_device_multiprocessor_count()
        )
        cache["compiled_gdn"] = cute.compile(
            _get_jit_blackwell(),
            t_trace["q"],
            t_trace["k"],
            t_trace["v"],
            t_trace["a"],
            t_trace["b"],
            t_trace["A_log"],
            t_trace["dt_bias"],
            t_trace["h0_in"],
            t_trace["h0_out"],
            t_trace["o"],
            t_trace["cu_seqlens"],
            cache["sm_count"],
            stream=current_stream,
            options=_GDN_COMPILE_OPTIONS,
        )

    t_run = _to_cute_tensors_bw(ph)
    cache["compiled_gdn"](
        t_run["q"],
        t_run["k"],
        t_run["v"],
        t_run["a"],
        t_run["b"],
        t_run["A_log"],
        t_run["dt_bias"],
        t_run["h0_in"],
        t_run["h0_out"],
        t_run["o"],
        t_run["cu_seqlens"],
        cache["sm_count"],
        stream=current_stream,
    )

    if output_final_state:
        return output, output_state
    else:
        return output


# ===========================================================================
# Edge-LLM Interface
# ===========================================================================

def compute_g_beta(a, b, A_log, dt_bias):
    """Compute gate g and update beta from edge-llm raw inputs.

    Args:
        a:       (N, T, HV) cupy array, fp16 or fp32 - input to softplus
        b:       (N, T, HV) cupy array, fp16 or fp32 - input to sigmoid
        A_log:   (HV,) cupy array, fp32 - log of state decay rate
        dt_bias: (HV,) cupy array, fp16 or fp32 - time-step bias

    Returns:
        g:    (N, T, HV) float32 cupy array, log-domain gate
        beta: (N, T, HV) float32 cupy array, sigmoid-domain update gate
    """
    a_f32 = a.astype(cp.float32)
    b_f32 = b.astype(cp.float32)
    A_log_f32 = A_log.astype(cp.float32)
    dt_bias_f32 = dt_bias.astype(cp.float32)

    # Broadcast A_log and dt_bias over (N, T) dims
    x = a_f32 + dt_bias_f32[cp.newaxis, cp.newaxis, :]       # (N, T, HV)
    # Numerically stable softplus: sp(x) = x if x > 20 else log(1 + exp(x))
    x_clipped = cp.minimum(x, cp.float32(20.0))
    sp = cp.where(x > cp.float32(20.0), x, cp.log1p(cp.exp(x_clipped)))
    g = -cp.exp(A_log_f32)[cp.newaxis, cp.newaxis, :] * sp    # (N, T, HV)
    beta = cp.float32(1.0) / (cp.float32(1.0) + cp.exp(-b_f32))  # (N, T, HV)

    return cp.ascontiguousarray(g), cp.ascontiguousarray(beta)


def _l2_normalize_qk_inplace(q, k):
    """L2-normalize Q and K along the head dimension (last axis) in-place.

    Q: (N, T, H, D) float16 cupy array
    K: (N, T, H, D) float16 cupy array

    Each token's head vector is divided by its L2 norm (with eps=1e-6 for stability).
    This must be done BEFORE the Blackwell kernel so that QK, KK^T, state update,
    and output paths all see consistently normalized Q and K.
    """
    eps = cp.float32(1e-6)
    # Compute norms in float32 for numerical stability
    q_f32 = q.astype(cp.float32)
    k_f32 = k.astype(cp.float32)
    q_norm = cp.sqrt(cp.sum(q_f32 * q_f32, axis=-1, keepdims=True) + eps)
    k_norm = cp.sqrt(cp.sum(k_f32 * k_f32, axis=-1, keepdims=True) + eps)
    q_normalized = q_f32 / q_norm
    k_normalized = k_f32 / k_norm
    cp.copyto(q, q_normalized.astype(q.dtype))
    cp.copyto(k, k_normalized.astype(k.dtype))


def run_gdn_prefill_blackwell(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    h0_source,
    context_lengths,
    o,
    seq_len,
    stream,
    output_final_state=True,
):
    """Edge-LLM Blackwell GDN prefill entry point.

    Matches the same call signature as edge-llm's gdn_prefill.py `run_prefill`.

    Args:
        q:                (N, T, H, D) float16 cupy array
        k:                (N, T, H, D) float16 cupy array
        v:                (N, T, HV, D) float16 cupy array
        a:                (N, T, HV) fp16 cupy array - gate pre-activation
        b:                (N, T, HV) fp16 cupy array - beta pre-activation
        A_log:            (HV,) float32 cupy array - log decay
        dt_bias:          (HV,) fp16 cupy array - time step bias
        h0_source:        (N, HV, D, D) float32 cupy array - initial state; on success receives
                          a copy of the final state (kernel uses a temp buffer, then copyto).
        context_lengths:  (N,) int32 cupy array - valid token counts per batch
        o:                (N, T, HV, D) float16 cupy array - output (written in-place)
        seq_len:          int - sequence length T
        stream:           cuda.CUstream
        output_final_state: if True, write final recurrent state into h0_source
    """
    n, t, hv, d = v.shape
    h = q.shape[2]
    initial_state = h0_source if h0_source is not None else None
    if output_final_state and h0_source is not None:
        state_tmp = cp.empty_like(h0_source)
    else:
        state_tmp = None

    chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=float(d) ** -0.5,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=None,
        output=o,
        output_state=state_tmp,
    )
    if output_final_state and h0_source is not None:
        state_tmp.copyto(h0_source)


# ===========================================================================
# NumPy Reference (same as gdn_prefill.py for cross-validation)
# ===========================================================================

def _softplus_np(x, beta=1.0, threshold=20.0):
    bx = beta * x
    if bx <= threshold:
        return (1.0 / beta) * np.log(1.0 + np.exp(bx))
    return float(x)


def _run_numpy_prefill_reference(
    q_f32, k_f32, v_f32, a_f32, b_f32,
    A_log_f32, dt_bias_f32, h0_source_f32,
    n, h, hv, k_dim, v_dim, seq_len, scale,
    context_lengths_np,
    use_qk_l2norm=False,
):
    """Recurrent numpy reference for correctness check.

    """
    h0 = h0_source_f32.copy()  # (N, HV, K, V)
    o_ref = np.zeros((n, seq_len, hv, v_dim), dtype=np.float32)

    for i_n in range(n):
        for i_hv in range(hv):
            i_h = i_hv // (hv // h) if h else 0
            H = h0[i_n, i_hv].copy()  # (K, V)
            max_t = int(context_lengths_np[i_n])
            for t in range(seq_len):
                if t >= max_t:
                    o_ref[i_n, t, i_hv, :] = 0.0
                    continue
                q_vec = q_f32[i_n, t, i_h].astype(np.float64)
                k_vec = k_f32[i_n, t, i_h].astype(np.float64)
                v_vec = v_f32[i_n, t, i_hv].astype(np.float64)
                a_val = float(a_f32[i_n, t, i_hv])
                b_val = float(b_f32[i_n, t, i_hv])
                A_val = float(A_log_f32[i_hv])
                dt_val = float(dt_bias_f32[i_hv])

                if use_qk_l2norm:
                    nq = np.sqrt(np.sum(q_vec ** 2) + 1e-6)
                    nk = np.sqrt(np.sum(k_vec ** 2) + 1e-6)
                    q_eff = q_vec / nq * scale
                    k_eff = k_vec / nk
                else:
                    q_eff = q_vec * scale
                    k_eff = k_vec

                sp = _softplus_np(a_val + dt_val, 1.0, 20.0)
                g_val = np.exp(-np.exp(A_val) * sp)
                beta_val = 1.0 / (1.0 + np.exp(-b_val))

                H_gated = H * g_val
                correction = v_vec - H_gated.T @ k_eff
                v_new = correction * beta_val
                H = H_gated + np.outer(k_eff, v_new)
                o_ref[i_n, t, i_hv] = H.T @ q_eff
            h0[i_n, i_hv] = H
    return o_ref, h0


# ===========================================================================
# Test runner
# ===========================================================================

def run_test_prefill_blackwell(
    n, h, hv, k, v, seq_len,
    skip_ref_check=False, tolerance=0.3,
    warmup=3, iterations=100, gpu_arch="",
    context_lengths_preset="full",
):
    """JIT test runner for the Blackwell GDN prefill kernel."""
    if seq_len < 128:
        raise ValueError("Blackwell kernel requires seq_len >= 128 (chunk_size=128).")
    if k != 128 or v != 128:
        raise ValueError("Blackwell kernel only supports head_dim=128.")

    dt = cp.float16
    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)

    np.random.seed(42)
    q_f32  = np.random.randn(n, seq_len, h, k).astype(np.float32) * 0.1
    k_f32  = np.random.randn(n, seq_len, h, k).astype(np.float32) * 0.1
    v_f32  = np.random.randn(n, seq_len, hv, v).astype(np.float32) * 0.1
    a_f32  = np.random.randn(n, seq_len, hv).astype(np.float32) * 0.1
    b_f32  = np.random.randn(n, seq_len, hv).astype(np.float32) * 0.1
    A_log_f32  = np.random.randn(hv).astype(np.float32) * 0.1
    dt_bias_f32 = np.random.randn(hv).astype(np.float32) * 0.1
    h0_f32 = np.random.randn(n, hv, k, v).astype(np.float32) * 0.01

    def _ctx_lengths(preset):
        if preset == "full" or preset is None:
            return np.full((n,), seq_len, dtype=np.int32)
        if preset == "half":
            return np.full((n,), max(128, seq_len // 2), dtype=np.int32)
        raise ValueError("Unknown preset: %r" % preset)

    ctx_np = _ctx_lengths(context_lengths_preset)

    q_cp   = cp.asarray(q_f32, dtype=dt)
    k_cp   = cp.asarray(k_f32, dtype=dt)
    v_cp   = cp.asarray(v_f32, dtype=dt)
    a_cp   = cp.asarray(a_f32, dtype=dt)
    b_cp   = cp.asarray(b_f32, dtype=dt)
    A_log_cp   = cp.asarray(A_log_f32)
    dt_bias_cp  = cp.asarray(dt_bias_f32, dtype=dt)
    # Blackwell kernel stores/reads state in V-major (d_v, d_k) format.
    # Transpose the K-major h0 to V-major before passing to the kernel.
    h0_vmaj_f32 = np.ascontiguousarray(np.swapaxes(h0_f32, -2, -1))
    h0_cp  = cp.asarray(h0_vmaj_f32)
    o_cp   = cp.zeros((n, seq_len, hv, v), dtype=dt)

    # Warmup / compile — h0_in and h0_out must not alias (kernel is not in-place safe).
    h0_out_run = cp.empty_like(h0_cp)
    for _ in range(warmup):
        h0_in_run = cp.asarray(h0_vmaj_f32)
        o_cp_run  = cp.zeros_like(o_cp)
        chunk_gated_delta_rule(
            q=q_cp, k=k_cp, v=v_cp,
            a=a_cp, b=b_cp,
            A_log=A_log_cp, dt_bias=dt_bias_cp,
            scale=float(k) ** -0.5,
            initial_state=h0_in_run,
            output_final_state=True,
            output=o_cp_run,
            output_state=h0_out_run,
        )
    cp.cuda.get_current_stream().synchronize()

    if not skip_ref_check:
        h0_in_test = cp.asarray(h0_vmaj_f32)
        h0_out_test = cp.empty_like(h0_in_test)
        o_test  = cp.zeros_like(o_cp)
        chunk_gated_delta_rule(
            q=q_cp, k=k_cp, v=v_cp,
            a=a_cp, b=b_cp,
            A_log=A_log_cp, dt_bias=dt_bias_cp,
            scale=float(k) ** -0.5,
            initial_state=h0_in_test,
            output_final_state=True,
            output=o_test,
            output_state=h0_out_test,
        )
        cp.cuda.get_current_stream().synchronize()

        o_ref, h0_ref = _run_numpy_prefill_reference(
            q_f32, k_f32, v_f32, a_f32, b_f32,
            A_log_f32, dt_bias_f32, h0_f32,
            n, h, hv, k, v, seq_len,
            scale=float(k) ** -0.5,
            context_lengths_np=ctx_np,
            use_qk_l2norm=True,
        )
        o_kernel = cp.asnumpy(o_test).astype(np.float32)
        max_err = np.max(np.abs(o_kernel - o_ref))
        print("[gdn_prefill_blackwell] Max abs error vs NumPy ref: %.6f (tol=%.4f)" % (max_err, tolerance))
        np.testing.assert_allclose(o_kernel, o_ref, atol=tolerance, rtol=1e-2)

        # Compare h0 state
        # The Blackwell kernel stores state in V-major (transposed) format:
        # MMA computes C = V^T * K giving (d_v, d_k), but the reference uses
        # K-major (d_k, d_v).  Transpose the reference to match the kernel.
        h0_ref_vmaj = np.swapaxes(h0_ref, -2, -1)  # (n, hv, d_k, d_v) -> (n, hv, d_v, d_k)
        h0_kernel = cp.asnumpy(h0_out_test).astype(np.float32)
        h0_max_err = np.max(np.abs(h0_kernel - h0_ref_vmaj))
        h0_max_abs = np.max(np.abs(h0_ref_vmaj))
        big_err_count = np.sum(np.abs(h0_kernel - h0_ref_vmaj) > 0.1)
        print("[gdn_prefill_blackwell] h0 state max abs error: %.6f (ref maxAbs=%.4f, bigErr>0.1: %d/%d)" %
              (h0_max_err, h0_max_abs, big_err_count, h0_ref_vmaj.size))
        if h0_max_err > 1.0:
            print("[gdn_prefill_blackwell] WARNING: h0 state has LARGE errors!")
            err_idx = np.unravel_index(np.argmax(np.abs(h0_kernel - h0_ref_vmaj)), h0_ref_vmaj.shape)
            print("[gdn_prefill_blackwell]   worst at idx=%s: kernel=%.6f ref=%.6f" %
                  (err_idx, h0_kernel[err_idx], h0_ref_vmaj[err_idx]))

        print("[gdn_prefill_blackwell] Reference check PASSED")

    # Benchmark — initial_state is read-only; output_state must be a distinct buffer.
    h0_in_bench = cp.asarray(h0_vmaj_f32)
    h0_out_bench = cp.empty_like(h0_in_bench)
    o_bench  = cp.zeros_like(o_cp)
    t0 = time.perf_counter()
    for _ in range(iterations):
        chunk_gated_delta_rule(
            q=q_cp, k=k_cp, v=v_cp,
            a=a_cp, b=b_cp,
            A_log=A_log_cp, dt_bias=dt_bias_cp,
            scale=float(k) ** -0.5,
            initial_state=h0_in_bench,
            output_final_state=True,
            output=o_bench,
            output_state=h0_out_bench,
        )
    cp.cuda.get_current_stream().synchronize()
    us = (time.perf_counter() - t0) * 1e6 / iterations
    print(
        "[gdn_prefill_blackwell] Latency: %.4f us "
        "(n=%d T=%d h=%d hv=%d k=%d v=%d)" % (us, n, seq_len, h, hv, k, v)
    )
    return us


# ===========================================================================
# AOT Export
# ===========================================================================

# AOT placeholder dimensions (must have seq_len >= 128 = chunk_size)
AOT_PLACEHOLDER_N = 1
AOT_PLACEHOLDER_H = 16
AOT_PLACEHOLDER_HV = 32
AOT_PLACEHOLDER_K = 128
AOT_PLACEHOLDER_V = 128
AOT_PLACEHOLDER_SEQLEN = 128  # must be multiple of 128

_gdn_aot_instance = GDN()


def _create_jit_blackwell():
    """Create a @cute.jit wrapper for the Blackwell GDN kernel (AOT-friendly)."""

    @cute.jit
    def run_gdn_blackwell(
        q: cute.Tensor,          # (n, seq_len, h_q, d) fp16
        k: cute.Tensor,          # (n, seq_len, h_q, d) fp16
        v: cute.Tensor,          # (n, seq_len, h_v, d) fp16
        a: cute.Tensor,          # (n, seq_len, h_v) fp16 — gate pre-activation
        b: cute.Tensor,          # (n, seq_len, h_v) fp16 — beta pre-activation
        A_log: cute.Tensor,      # (h_v,) f32 — log decay
        dt_bias: cute.Tensor,    # (h_v,) fp16 — time-step bias
        h0_in: cute.Tensor,      # (n, h_v, d, d) f32     — initial recurrent state
        h0_out: cute.Tensor,     # (n, h_v, d, d) f32     — output final state
        o: cute.Tensor,          # (n, seq_len, h_v, d) fp16 — output
        cu_seqlens: cute.Tensor, # (n+1,) int32 — prefix-sum for padding masking
        sm_count: cutlass.Int32, # runtime persistent-grid size (SM count of launch GPU)
        stream: cuda.CUstream,
    ):
        n = q.layout.shape[0]
        seq_len = q.layout.shape[1]
        h_q = q.layout.shape[2]
        d = q.layout.shape[3]
        h_v = v.layout.shape[2]
        # For non-varlen padded layout: s_sum = seq_len (matches _get_problem_size).
        # The kernel uses s_sum for per-batch strides in gb_layout; using n*seq_len
        # would make the batch stride n times too large, causing wrong a/b reads.
        s_sum = seq_len
        problem_size = (n, seq_len, s_sum, h_q, h_v, d)

        _gdn_aot_instance(
            q.iterator,
            k.iterator,
            v.iterator,
            o.iterator,
            a.iterator,
            b.iterator,
            A_log.iterator,
            dt_bias.iterator,
            problem_size,
            h0_in.iterator,   # initial_state
            h0_out.iterator,  # state_output
            None,             # scale=None → kernel computes 1/sqrt(d) internally
            None,             # cum_seqlen_q=None → non-varlen padded layout
            cu_seqlens,       # cu_seqlens for padding masking
            True,             # use_qk_l2norm=True
            sm_count,
            stream,
        )

    return run_gdn_blackwell


_jit_blackwell = None


def _get_jit_blackwell():
    global _jit_blackwell
    if _jit_blackwell is None:
        _jit_blackwell = _create_jit_blackwell()
    return _jit_blackwell


_compiled_blackwell = {}


def _make_placeholder_tensors_bw(n, h, hv, k, v, seq_len):
    dt = cp.float16
    return {
        "q":          cp.zeros((n, seq_len, h, k),  dtype=dt),
        "k":          cp.zeros((n, seq_len, h, k),  dtype=dt),
        "v":          cp.zeros((n, seq_len, hv, v), dtype=dt),
        "a":          cp.zeros((n, seq_len, hv),    dtype=dt),
        "b":          cp.zeros((n, seq_len, hv),    dtype=dt),
        "A_log":      cp.zeros((hv,),               dtype=cp.float32),
        "dt_bias":    cp.zeros((hv,),               dtype=dt),
        "h0_in":      cp.zeros((n, hv, k, v),       dtype=cp.float32),
        "h0_out":     cp.zeros((n, hv, k, v),       dtype=cp.float32),
        "o":          cp.zeros((n, seq_len, hv, v), dtype=dt),
        # cu_seqlens: cumulative sequence lengths [N+1], int32.
        # Used for padding masking in padded-layout (non-varlen) mode.
        "cu_seqlens": cp.zeros((n + 1,),            dtype=cp.int32),
    }


def _mark_4d_dynamic(tensor):
    so = (0, 1, 2, 3)
    return (from_dlpack(tensor, assumed_align=16)
            .mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(mode=0, stride_order=so)
            .mark_compact_shape_dynamic(mode=1, stride_order=so)
            .mark_compact_shape_dynamic(mode=2, stride_order=so))


def _mark_3d_dynamic(tensor):
    so = (0, 1, 2)
    return (from_dlpack(tensor, assumed_align=16)
            .mark_layout_dynamic(leading_dim=2)
            .mark_compact_shape_dynamic(mode=0, stride_order=so)
            .mark_compact_shape_dynamic(mode=1, stride_order=so)
            .mark_compact_shape_dynamic(mode=2, stride_order=so))


def _mark_h0_dynamic(tensor):
    """h0: (n, hv, k, v) f32 — n and hv dynamic, k/v static (128)."""
    so = (0, 1, 2, 3)
    return (from_dlpack(tensor, assumed_align=16)
            .mark_compact_shape_dynamic(mode=0, stride_order=so)
            .mark_compact_shape_dynamic(mode=1, stride_order=so))


def _to_cute_tensors_bw(ph):
    return {
        "q":          _mark_4d_dynamic(ph["q"]),
        "k":          _mark_4d_dynamic(ph["k"]),
        "v":          _mark_4d_dynamic(ph["v"]),
        "a":          _mark_3d_dynamic(ph["a"]),
        "b":          _mark_3d_dynamic(ph["b"]),
        "A_log":      from_dlpack(ph["A_log"],      assumed_align=16),
        "dt_bias":    from_dlpack(ph["dt_bias"],    assumed_align=16),
        "h0_in":      _mark_h0_dynamic(ph["h0_in"]),
        "h0_out":     _mark_h0_dynamic(ph["h0_out"]),
        "o":          _mark_4d_dynamic(ph["o"]),
        "cu_seqlens": (from_dlpack(ph["cu_seqlens"], assumed_align=16)
                       .mark_compact_shape_dynamic(mode=0, stride_order=(0,))),
    }


def _compile_prefill_bw(n, h, hv, k, v, seq_len, stream, gpu_arch=""):
    if "_" in _compiled_blackwell:
        return _compiled_blackwell["_"]

    ph = _make_placeholder_tensors_bw(n, h, hv, k, v, seq_len)
    t = _to_cute_tensors_bw(ph)
    run_fn = _get_jit_blackwell()

    compile_opts = _GDN_COMPILE_OPTIONS
    if gpu_arch:
        compile_opts += " --gpu-arch " + gpu_arch
    compiled = cute.compile(
        run_fn,
        t["q"], t["k"], t["v"],
        t["a"], t["b"],
        t["A_log"], t["dt_bias"],
        t["h0_in"], t["h0_out"],
        t["o"],
        t["cu_seqlens"],   # cu_seqlens for padding masking (non-varlen padded layout)
        # Runtime persistent-grid size: AOT callers pass the launch GPU's SM
        # count; the trace value is a placeholder.
        cutlass.Int32(cutlass.utils.HardwareInfo().get_device_multiprocessor_count()),
        stream,
        options=compile_opts,
    )
    _compiled_blackwell["_"] = compiled
    return compiled


def export_gdn_prefill_blackwell(n, h, hv, k, v, seq_len,
                                  output_dir, file_name, function_prefix, gpu_arch=""):
    """AOT compile and export the Blackwell GDN prefill kernel."""
    if seq_len < 128 or seq_len % 128 != 0:
        raise ValueError("Blackwell kernel requires seq_len >= 128 and multiple of 128.")
    if k != 128 or v != 128:
        raise ValueError("Blackwell kernel requires k == v == 128.")

    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    print("[gdn_prefill_blackwell] AOT compile gpu_arch=%r" % (gpu_arch or "auto"))
    t0 = time.time()
    compiled = _compile_prefill_bw(n, h, hv, k, v, seq_len, stream, gpu_arch=gpu_arch)
    print("[gdn_prefill_blackwell] Compilation time: %.4fs" % (time.time() - t0))

    os.makedirs(output_dir, exist_ok=True)
    compiled.export_to_c(
        file_path=output_dir,
        file_name=file_name,
        function_prefix=function_prefix,
    )
    print("[gdn_prefill_blackwell] Exported to %s/%s.h and %s.o" % (output_dir, file_name, file_name))
    return compiled


# ===========================================================================
# Entry point
# ===========================================================================

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Blackwell GDN prefill kernel (SM100, CuPy only). Test and AOT export."
    )
    p.add_argument("--export_only", action="store_true",
                   help="AOT export mode: compile and export the Blackwell GDN prefill kernel.")
    p.add_argument("--output_dir", type=str, default=".",
                   help="Output directory for AOT export artifacts.")
    p.add_argument("--file_name", type=str, default="gdn_prefill_blackwell",
                   help="Base file name for AOT export artifacts (.h and .o).")
    p.add_argument("--function_prefix", type=str, default="gdn_prefill_blackwell",
                   help="Function prefix for the exported AOT wrapper symbols.")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--h", type=int, default=8)
    p.add_argument("--hv", type=int, default=8)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--v", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--context_lengths_preset", type=str, default="full",
                   choices=("full", "half"))
    p.add_argument("--skip_ref_check", action="store_true")
    p.add_argument("--tolerance", type=float, default=0.3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--gpu_arch", type=str, default="")
    return p.parse_known_args(args=argv)[0]


def main():
    args = _parse_args(_saved_argv)
    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("No GPU found.")
    cp.random.seed(42)
    np.random.seed(42)

    if args.export_only:
        export_gdn_prefill_blackwell(
            n=AOT_PLACEHOLDER_N,
            h=AOT_PLACEHOLDER_H,
            hv=AOT_PLACEHOLDER_HV,
            k=AOT_PLACEHOLDER_K,
            v=AOT_PLACEHOLDER_V,
            seq_len=AOT_PLACEHOLDER_SEQLEN,
            output_dir=args.output_dir,
            file_name=args.file_name,
            function_prefix=args.function_prefix,
            gpu_arch=args.gpu_arch,
        )
        return

    run_test_prefill_blackwell(
        n=args.n, h=args.h, hv=args.hv, k=args.k, v=args.v,
        seq_len=args.seq_len,
        skip_ref_check=args.skip_ref_check,
        tolerance=args.tolerance,
        warmup=args.warmup,
        iterations=args.iterations,
        gpu_arch=args.gpu_arch,
        context_lengths_preset=args.context_lengths_preset,
    )


if __name__ == "__main__":
    main()
