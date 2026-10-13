# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 FlashInfer team.
#
# Adapted from FlashInfer commit d020372b068f335e2fe427372e134977a2235c49
# for TensorRT Edge-LLM Blackwell GeForce GDN prefill.

import argparse
import os
import sys
import time
from dataclasses import dataclass
from enum import IntEnum

_saved_argv = None
if __name__ == "__main__":
    _saved_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]

import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import numpy as np
from cutlass.cute.nvgpu import warp, warpgroup, cpasync
from cutlass.cute.runtime import from_dlpack
from gdn_prefill_sm12x_helpers import (
    TENSOR_MAP_DESCRIPTOR_BYTES,
    CollectiveStoreTma,
    SM80,
    round_down,
)
from gdn_prefill_sm12x_inverse import CollectiveInverse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cutedsl_utils import aot_placeholders  # isort: skip


@dataclass
class WorkDesc:
    seq_idx: cutlass.Int32
    state_idx: cutlass.Int32
    private_q_head_idx: cutlass.Int32
    private_v_head_idx: cutlass.Int32
    tok_offset: cutlass.Int32
    seq_len: cutlass.Int32
    tile_idx: cutlass.Int32

    def q_head_idx(self):
        return self.private_q_head_idx

    @staticmethod
    @cute.jit
    def is_gva(num_q_heads: cutlass.Int32, num_v_heads: cutlass.Int32):
        return num_v_heads > num_q_heads

    @cute.jit
    def k_head_idx(self, num_q_heads: cutlass.Int32, num_v_heads: cutlass.Int32):
        k_head_idx = self.private_v_head_idx
        if WorkDesc.is_gva(num_q_heads, num_v_heads):
            k_head_idx = self.private_q_head_idx
        return k_head_idx

    def v_head_idx(self):
        return self.private_v_head_idx

    @cute.jit
    def o_head_idx(self, num_q_heads: cutlass.Int32, num_v_heads: cutlass.Int32):
        o_head_idx = self.private_q_head_idx
        if WorkDesc.is_gva(num_q_heads, num_v_heads):
            o_head_idx = self.private_v_head_idx
        return o_head_idx


class AlphaProcessor:
    CUMSUM_LOG = 0
    CUMPROD = 1
    CUMPROD_SCALE = 2
    CUMPROD_NEG_END_RCP = 2
    NUM_CHANNELS = 3

    @cute.jit
    def run(
        self,
        vecs: cute.Tensor,
        scale: cutlass.Float32,
        channel2_neg_end_rcp: cutlass.Constexpr = False,
    ):
        warp_size = 32
        blk_q = cute.size(vecs.shape[0])
        num_iters = blk_q // warp_size
        lane_id = cute.arch.lane_idx()
        vecs_32 = cute.flat_divide(vecs, (warp_size,))
        frag = cute.make_rmem_tensor(num_iters, cutlass.Float32)
        for i in cutlass.range_constexpr(num_iters):
            # LOAD_ALPHA writes gate directly in log2 space.
            frag[i] = cutlass.Float32(
                vecs_32[lane_id, i, AlphaProcessor.CUMSUM_LOG]
            )
        for log_off in cutlass.range_constexpr(5):
            off = 1 << log_off
            for i in cutlass.range_constexpr(num_iters):
                value = cute.arch.shuffle_sync_up(frag[i], off, mask_and_clamp=0)
                if lane_id >= off:
                    frag[i] = frag[i] + value
        for i in cutlass.range_constexpr(1, num_iters):
            frag[i] = frag[i] + cute.arch.shuffle_sync(frag[i - 1], 31)
        if cutlass.const_expr(channel2_neg_end_rcp):
            end_log = cute.arch.shuffle_sync(frag[num_iters - 1], 31)
        for i in cutlass.range_constexpr(num_iters):
            vecs_32[lane_id, i, AlphaProcessor.CUMSUM_LOG] = frag[i]
            cumprod = cute.math.exp2(frag[i], fastmath=True)
            vecs_32[lane_id, i, AlphaProcessor.CUMPROD] = cumprod
            if cutlass.const_expr(channel2_neg_end_rcp):
                vecs_32[
                    lane_id, i, AlphaProcessor.CUMPROD_NEG_END_RCP
                ] = -cute.math.exp2(end_log - frag[i], fastmath=True)
            else:
                vecs_32[lane_id, i, AlphaProcessor.CUMPROD_SCALE] = cumprod * scale


# ─── Named-barrier IDs used by the compute kernel ────────────────────────────
# Must not conflict with each other or with pipeline barrier storage.


class NamedBarrier(IntEnum):
    MATH_WG0 = 4  # OrderedMathBarriers: StreamkBarrier0
    MATH_WG1 = 5  # OrderedMathBarriers: StreamkBarrier1
    KK_SYNC = 13  # sync all 128 WG0 threads before collective_inverse


class WarpGroupRole(IntEnum):
    LDST = 0
    MATH_KK = 1
    MATH_QK = 2


class LoadStoreWarpRole(IntEnum):
    LOAD_QKV = 0
    STORE_O = 1
    LOAD_BETA = 2
    LOAD_ALPHA = 3


class MathWarpGroupRole(IntEnum):
    KK = 0
    QK = 1


# ─── Warp-specialized delta-rule kernel ───────────────────────────────────────
# Grid: (num_seqs * num_sab_heads, 1, 1)
# Block: 384 threads → WG0=[0,127], WG1=[128,255], WG2=[256,383]
#
# needs_alpha / needs_beta are class attributes set in __init__.
# The JIT compiler specialises per instance, so they are compile-time booleans
# inside the kernel without any parameter-passing trickery.


class FullyFusedGdnPrefillBlackwellGeforce:
    @staticmethod
    def get_register_requirements(
        max_threads_per_block: int,
        min_blocks_per_multiprocessor: int,
        num_mma_warp_groups: int,
        threads_per_warp_group: int,
    ) -> tuple[int, int]:
        reg_alloc_granularity = 8
        load_registers = 40 - 2 * reg_alloc_granularity
        total_registers = (
            round_down(
                64 * 1024 // min_blocks_per_multiprocessor,
                max_threads_per_block * reg_alloc_granularity,
            )
            // threads_per_warp_group
        )
        mma_registers = round_down(
            (total_registers - load_registers) // num_mma_warp_groups,
            reg_alloc_granularity,
        )
        return min(248, load_registers), min(248, mma_registers)

    def __init__(
        self,
        dtype: type[cutlass.Numeric] = cutlass.Float16,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
    ):
        self.needs_alpha = True
        self.needs_beta = True
        self.dtype = dtype
        self.acc_dtype = acc_dtype
        self.inverse_dtype = cutlass.Float16
        self.BLK_Q = 64
        self.BLK_KV = 64
        self.D = 128
        self.q_stage = 1
        self.k_stage = 2
        self.v_stage = 1
        self.o_stage = 1
        self.alpha_beta_stage = 2

    def get_next_work(
        self,
        context_lengths: cute.Tensor,
        state_indices: cute.Tensor,
        use_state_indices: cutlass.Int32,
        physical_seq_len: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
    ) -> WorkDesc:
        bx, _, _ = cute.arch.block_idx()
        seq_idx = bx // num_sab_heads
        o_head_idx = bx % num_sab_heads
        q_head_idx = o_head_idx * num_q_heads // num_sab_heads
        v_head_idx = o_head_idx * num_v_heads // num_sab_heads
        tok_start = seq_idx * physical_seq_len
        seq_len = cutlass.Int32(context_lengths[seq_idx])
        indexed_state_idx = cutlass.Int32(state_indices[seq_idx])
        state_idx = cutlass.Int32(
            seq_idx + use_state_indices * (indexed_state_idx - seq_idx)
        )

        return WorkDesc(
            seq_idx=seq_idx,
            state_idx=state_idx,
            private_q_head_idx=q_head_idx,
            private_v_head_idx=v_head_idx,
            tok_offset=tok_start,
            seq_len=seq_len,
            tile_idx=cutlass.Int32(0),
        )

    # ─── Ordered 2-WG math barriers ───────────────────────────────────────────
    # Translates flat::OrderedNamedBarriers<UseReservedNB, NB0, NB1>.
    # wg_idx: MathWarpGroupRole.KK or MathWarpGroupRole.QK.

    @cute.jit
    def _math_order_init(self, wg_idx: cutlass.Int32):
        """Pre-arrive at WG0's barrier so WG0 is unblocked on the first wait."""
        if wg_idx == MathWarpGroupRole.QK:
            cute.arch.barrier_arrive(
                barrier_id=NamedBarrier.MATH_WG0, number_of_threads=256
            )

    @cute.jit
    def _math_order_wait(self, wg_idx: cutlass.Int32):
        """Arrive+wait on this WG's own ordered barrier."""
        if wg_idx == MathWarpGroupRole.KK:
            cute.arch.barrier(barrier_id=NamedBarrier.MATH_WG0, number_of_threads=256)
        else:
            cute.arch.barrier(barrier_id=NamedBarrier.MATH_WG1, number_of_threads=256)

    @cute.jit
    def _math_order_notify(self, wg_idx: cutlass.Int32):
        """Arrive at the other WG's barrier to unblock it."""
        if wg_idx == MathWarpGroupRole.KK:
            cute.arch.barrier_arrive(
                barrier_id=NamedBarrier.MATH_WG1, number_of_threads=256
            )
        else:
            cute.arch.barrier_arrive(
                barrier_id=NamedBarrier.MATH_WG0, number_of_threads=256
            )

    # ─── kk_store_and_inv ─────────────────────────────────────────────────────

    @cute.jit
    def _kk_store_and_inv(
        self,
        tKKrKK: cute.Tensor,  # fp32 KK accumulator (from 128-thread kk_tiled_mma)
        kk_tiled_mma,
        kk_thread_idx: cutlass.Int32,
        sKK_inv: cute.Tensor,  # (BlkKV, BlkKV)
        sKK_opd: cute.Tensor,  # sKK_inv storage recast as Element for MMA operand
        sBeta: cute.Tensor,  # (BlkKV, StagesBeta) - used when needs_beta
        beta_pipe_idx: cutlass.Int32,
        tKKcMkk: cute.Tensor,  # coordinate mapping for KK fragment
    ):
        """Store tKKrKK → sKK_inv, Inverse, optionally reload+beta."""
        stsm_atom = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), self.inverse_dtype
        )
        tiled_store = cute.make_tiled_copy_C(stsm_atom, kk_tiled_mma)
        thr_store = tiled_store.get_slice(kk_thread_idx)
        tKKsKK = thr_store.partition_D(sKK_inv)
        tKKrKK_inv = cute.make_fragment_like(tKKrKK, self.inverse_dtype)
        tKKrKK_cv = thr_store.retile(tKKrKK_inv)
        for i in cutlass.range_constexpr(cute.size(tKKrKK)):
            tKKrKK_inv[i] = self.inverse_dtype(tKKrKK[i])
        cute.copy(tiled_store, tKKrKK_cv, tKKsKK)

        cute.arch.barrier(barrier_id=NamedBarrier.KK_SYNC, number_of_threads=128)
        CollectiveInverse().run(sKK_inv, NamedBarrier.KK_SYNC)

        if cutlass.const_expr(self.needs_beta or self.dtype != self.inverse_dtype):
            cute.arch.barrier(barrier_id=NamedBarrier.KK_SYNC, number_of_threads=128)
            ldsm_atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.inverse_dtype,
            )
            tiled_load = cute.make_tiled_copy_C(ldsm_atom, kk_tiled_mma)
            thr_load = tiled_load.get_slice(kk_thread_idx)
            tKKrKK_cpy = cute.make_fragment_like(tKKrKK_inv)
            tKKrKK_cvt = cute.make_fragment_like(tKKrKK_inv, self.dtype)
            tKKrKK_cv2 = thr_load.retile(tKKrKK_cpy)
            cute.copy(tiled_load, thr_load.partition_S(sKK_inv), tKKrKK_cv2)
            tKKcMkk_cv = thr_load.retile(tKKcMkk)

            for i in cutlass.range_constexpr(cute.size(tKKrKK_cpy)):
                if cutlass.const_expr(self.needs_beta):
                    _, t = tKKcMkk_cv[i]
                    tKKrKK_cvt[i] = self.dtype(
                        cutlass.Float32(tKKrKK_cpy[i])
                        * cutlass.Float32(sBeta[t, beta_pipe_idx])
                    )
                else:
                    tKKrKK_cvt[i] = self.dtype(tKKrKK_cpy[i])

            tKKsKK2 = thr_store.partition_D(sKK_opd)
            tKKrKK_cv3 = thr_store.retile(tKKrKK_cvt)
            cute.copy(tiled_store, tKKrKK_cv3, tKKsKK2)

    # ─── kk_epi ───────────────────────────────────────────────────────────────

    @cute.jit
    def kk_epi(
        self,
        tKKrKK: cute.Tensor,
        tKKcMkk: cute.Tensor,
        sAlpha: cute.Tensor,
        sBeta: cute.Tensor,
        alpha_stage: cutlass.Int32,
        beta_stage: cutlass.Int32,
    ):
        if cutlass.const_expr(self.needs_alpha):
            alpha_cumlog = sAlpha[None, AlphaProcessor.CUMSUM_LOG, alpha_stage]
            for i in cutlass.range_constexpr(cute.size(tKKrKK)):
                s, t = tKKcMkk[i]
                tKKrKK[i] = tKKrKK[i] * cute.math.exp2(
                    cutlass.Float32(alpha_cumlog[s]) - cutlass.Float32(alpha_cumlog[t]),
                    fastmath=True,
                )
        if cutlass.const_expr(self.needs_beta):
            beta_row = sBeta[None, beta_stage]
            for i in cutlass.range_constexpr(cute.size(tKKrKK)):
                s, _ = tKKcMkk[i]
                tKKrKK[i] = tKKrKK[i] * cutlass.Float32(beta_row[s])

    # ─── qk_or_kk_mask ────────────────────────────────────────────────────────

    @cute.jit
    def qk_or_kk_mask(
        self,
        frag: cute.Tensor,
        coord_tensor: cute.Tensor,
        is_final_block: bool,
        B: cutlass.Int32,
    ):
        for i in cutlass.range_constexpr(cute.size(frag)):
            s, t = coord_tensor[i]
            pred = s >= t
            if cutlass.const_expr(is_final_block):
                pred = pred and (s < B and t < B)
            if not pred:
                frag[i] = cutlass.Float32(0.0)

    # ─── qk_epi ───────────────────────────────────────────────────────────────

    @cute.jit
    def qk_epi(
        self,
        tQKrQK: cute.Tensor,
        tQKcMqk: cute.Tensor,
        sAlpha: cute.Tensor,
        alpha_stage: cutlass.Int32,
        scale: cutlass.Float32,
    ):
        if cutlass.const_expr(self.needs_alpha):
            alpha_cumlog = sAlpha[None, AlphaProcessor.CUMSUM_LOG, alpha_stage]
            for i in cutlass.range_constexpr(cute.size(tQKrQK)):
                s, t = tQKcMqk[i]
                tQKrQK[i] = (
                    tQKrQK[i]
                    * cute.math.exp2(
                        cutlass.Float32(alpha_cumlog[s])
                        - cutlass.Float32(alpha_cumlog[t]),
                        fastmath=True,
                    )
                    * scale
                )
        else:
            for i in cutlass.range_constexpr(cute.size(tQKrQK)):
                tQKrQK[i] = tQKrQK[i] * scale

    # ─── qk_store ─────────────────────────────────────────────────────────────

    @cute.jit
    def qk_store(
        self,
        tQKrQK: cute.Tensor,
        sQK: cute.Tensor,
        qk_tiled_mma,
        qk_thread_idx: cutlass.Int32,
    ):
        stsm_atom = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        qk_tiled_copy = cute.make_tiled_copy_C(stsm_atom, qk_tiled_mma)
        qk_thr_copy = qk_tiled_copy.get_slice(qk_thread_idx)
        tQKsQK = qk_thr_copy.partition_D(sQK)
        tQKrQK_cvt = cute.make_fragment_like(tQKrQK, self.dtype)
        tQKrQK_cvt_cv = qk_thr_copy.retile(tQKrQK_cvt)
        for i in cutlass.range_constexpr(cute.size(tQKrQK)):
            tQKrQK_cvt[i] = self.dtype(tQKrQK[i])
        cute.copy(qk_tiled_copy, tQKrQK_cvt_cv, tQKsQK)

    # ─── o1_epi ───────────────────────────────────────────────────────────────

    @cute.jit
    def o1_epi(
        self,
        tOrO: cute.Tensor,
        tOcO: cute.Tensor,
        sAlpha: cute.Tensor,
        alpha_stage: cutlass.Int32,
        scale: cutlass.Float32,
    ):
        if cutlass.const_expr(self.needs_alpha):
            alpha_cpscale = sAlpha[None, AlphaProcessor.CUMPROD_SCALE, alpha_stage]
            for i in cutlass.range_constexpr(cute.size(tOrO)):
                _, tok_q = tOcO[i]
                tOrO[i] = cutlass.Float32(alpha_cpscale[tok_q]) * tOrO[i]
        else:
            for i in cutlass.range_constexpr(cute.size(tOrO)):
                tOrO[i] = scale * tOrO[i]

    # ─── sk_epi ───────────────────────────────────────────────────────────────

    @cute.jit
    def sk_epi(
        self,
        tSKrSK: cute.Tensor,
        tSKcSK: cute.Tensor,
        sAlpha: cute.Tensor,
        alpha_stage: cutlass.Int32,
    ):
        if cutlass.const_expr(self.needs_alpha):
            alpha_cp = sAlpha[None, AlphaProcessor.CUMPROD, alpha_stage]
            for i in cutlass.range_constexpr(cute.size(tSKrSK)):
                _, tok_kv = tSKcSK[i]
                tSKrSK[i] = tSKrSK[i] * cutlass.Float32(alpha_cp[tok_kv])

    # ─── sk_load_v ────────────────────────────────────────────────────────────

    @cute.jit
    def sk_load_v(
        self,
        tSKrSK: cute.Tensor,
        sV_DS: cute.Tensor,
        sk_tiled_copy_C,
        sk_thr_copy_C,
        v_stage: cutlass.Int32,
    ) -> cute.Tensor:
        tSKrV = cute.make_fragment_like(tSKrSK, self.dtype)
        tSKrV_cv = sk_thr_copy_C.retile(tSKrV)
        tSKsV = sk_thr_copy_C.partition_S(sV_DS)
        cute.copy(sk_tiled_copy_C, tSKsV[None, None, None, v_stage], tSKrV_cv)
        return tSKrV

    # ─── kv_decay_v ───────────────────────────────────────────────────────────

    @cute.jit
    def kv_decay_v(
        self,
        tKVrV: cute.Tensor,
        tKVcV: cute.Tensor,
        sAlpha: cute.Tensor,
        alpha_stage: cutlass.Int32,
        is_final_block: bool,
        B: cutlass.Int32,
    ):
        if cutlass.const_expr(self.needs_alpha):
            alpha_cumlog = sAlpha[None, AlphaProcessor.CUMSUM_LOG, alpha_stage]
            block_log = cutlass.Float32(alpha_cumlog[B - cutlass.Int32(1)])
            for i in cutlass.range_constexpr(cute.size(tKVrV)):
                _, tok = tKVcV[i]
                coeff = cute.math.exp2(
                    block_log - cutlass.Float32(alpha_cumlog[tok]), fastmath=True
                )
                if cutlass.const_expr(is_final_block):
                    if tok >= B:
                        coeff = cutlass.Float32(0.0)
                tKVrV[i] = self.dtype(cutlass.Float32(tKVrV[i]) * coeff)
        else:
            for i in cutlass.range_constexpr(cute.size(tKVrV)):
                _, tok = tKVcV[i]
                if cutlass.const_expr(is_final_block):
                    if tok >= B:
                        tKVrV[i] = self.dtype(0.0)

    # ─── o_store ──────────────────────────────────────────────────────────────

    @cute.jit
    def o_store(
        self,
        tOrO: cute.Tensor,
        tOsO: cute.Tensor,
        o_tiled_copy_r2s,
        o_thr_copy_r2s,
    ):
        tOrO_f16 = cute.make_fragment_like(tOrO, self.dtype)
        for i in cutlass.range_constexpr(cute.size(tOrO)):
            tOrO_f16[i] = self.dtype(tOrO[i])
        tOrO_cv = o_thr_copy_r2s.retile(tOrO_f16)
        cute.arch.fence_view_async_shared()
        cute.copy(o_tiled_copy_r2s, tOrO_cv, tOsO)
        cute.arch.fence_view_async_shared()

    @cute.jit
    def zero_output_padding(
        self,
        g_o: cute.Tensor,
        work_desc: WorkDesc,
        physical_seq_len: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
    ):
        lane_idx = cute.arch.lane_idx()
        o_head_idx = work_desc.o_head_idx(num_q_heads, num_v_heads)
        num_tail_elements = (
            physical_seq_len - work_desc.seq_len
        ) * cutlass.Int32(self.D)
        for linear_idx in cutlass.range(
            lane_idx,
            num_tail_elements,
            cutlass.Int32(32),
            unroll=1,
        ):
            value_idx = linear_idx % cutlass.Int32(self.D)
            token_idx = (
                work_desc.tok_offset
                + work_desc.seq_len
                + linear_idx // cutlass.Int32(self.D)
            )
            g_o[value_idx, token_idx, o_head_idx] = self.dtype(0.0)

    # ─── TMA load helpers ────────────────────────────────────────────────────

    @cute.jit
    def load_qkv_tma(
        self,
        sQ_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        q_pipeline,
        q_producer_state,
        k_pipeline,
        k_producer_state,
        v_pipeline,
        v_producer_state,
        blk: cutlass.Int32,
        tok_start: cutlass.Int32,
        q_head_idx: cutlass.Int32,
        k_head_idx: cutlass.Int32,
        v_head_idx: cutlass.Int32,
    ):
        blk_tok = tok_start + blk * cutlass.Int32(self.BLK_KV)

        sK = sK_DS[None, None, k_producer_state.index]
        mK = cute.domain_offset(
            (cutlass.Int32(0), blk_tok),
            tma_tensor_k[None, None, k_head_idx],
        )
        gK = cute.zipped_divide(mK, (self.D, self.BLK_KV))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        tKsK, tKgK = cpasync.tma_partition(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        k_pipeline.producer_acquire(k_producer_state)
        cute.copy(
            tma_atom_k,
            tKgK,
            tKsK,
            tma_bar_ptr=k_pipeline.producer_get_barrier(k_producer_state),
        )
        k_pipeline.producer_commit(k_producer_state)
        k_producer_state.advance()

        sQ = sQ_SD[None, None, q_producer_state.index]
        mQ = cute.domain_offset(
            (blk_tok, cutlass.Int32(0)),
            tma_tensor_q[None, None, q_head_idx],
        )
        gQ = cute.zipped_divide(mQ, (self.BLK_Q, self.D))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_atom_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 2),
            cute.group_modes(gQ, 0, 2),
        )
        q_pipeline.producer_acquire(q_producer_state)
        cute.copy(
            tma_atom_q,
            tQgQ,
            tQsQ,
            tma_bar_ptr=q_pipeline.producer_get_barrier(q_producer_state),
        )
        q_pipeline.producer_commit(q_producer_state)
        q_producer_state.advance()

        sV = sV_DS[None, None, v_producer_state.index]
        mV = cute.domain_offset(
            (cutlass.Int32(0), blk_tok),
            tma_tensor_v[None, None, v_head_idx],
        )
        gV = cute.zipped_divide(mV, (self.D, self.BLK_KV))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        tVsV, tVgV = cpasync.tma_partition(
            tma_atom_v,
            0,
            cute.make_layout(1),
            cute.group_modes(sV, 0, 2),
            cute.group_modes(gV, 0, 2),
        )
        v_pipeline.producer_acquire(v_producer_state)
        cute.copy(
            tma_atom_v,
            tVgV,
            tVsV,
            tma_bar_ptr=v_pipeline.producer_get_barrier(v_producer_state),
        )
        v_pipeline.producer_commit(v_producer_state)
        v_producer_state.advance()
        return q_producer_state, k_producer_state, v_producer_state

    # ─── load_alpha ───────────────────────────────────────────────────────────
    # Translates FlatMainloopTmaWarpSpecializedDeltaRule::load_alpha (scalar load).
    # Caller must sync before calling AlphaProcessor on the loaded data.

    @cute.jit
    def load_alpha(
        self,
        sAlpha: cute.Tensor,
        g_a: cute.Tensor,
        g_A_log: cute.Tensor,
        g_dt_bias: cute.Tensor,
        blk_tok: cutlass.Int32,
        tok_end: cutlass.Int32,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        alpha_stage: cutlass.Int32,
    ):
        lane_id = cute.arch.lane_idx()
        sAlpha_k = sAlpha[None, None, alpha_stage]
        num_iters = self.BLK_Q // 32
        for i in cutlass.range_constexpr(num_iters):
            row = cutlass.Int32(i * 32) + lane_id
            tok = blk_tok + row
            if tok < tok_end:
                x = (
                    cutlass.Float32(g_a[tok * num_sab_heads + sab_head_idx])
                    + cutlass.Float32(g_dt_bias[sab_head_idx])
                )
                softplus_x = x
                if x <= cutlass.Float32(20.0):
                    softplus_x = cute.log(
                        cutlass.Float32(1.0) + cute.exp(x)
                    )
                gate = -cute.exp(cutlass.Float32(g_A_log[sab_head_idx])) * softplus_x
                sAlpha_k[row, AlphaProcessor.CUMSUM_LOG] = (
                    gate * cutlass.Float32(1.4426950408889634)
                )
            else:
                sAlpha_k[row, AlphaProcessor.CUMSUM_LOG] = cutlass.Float32(0.0)

    # ─── load_beta ────────────────────────────────────────────────────────────
    # Translates FlatMainloopTmaWarpSpecializedDeltaRule::load_beta.

    @cute.jit
    def load_beta(
        self,
        sBeta: cute.Tensor,
        g_b: cute.Tensor,
        blk_tok: cutlass.Int32,
        tok_end: cutlass.Int32,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        beta_stage: cutlass.Int32,
    ):
        lane_id = cute.arch.lane_idx()
        sBeta_k = sBeta[None, beta_stage]
        num_iters = self.BLK_KV // 32
        for i in cutlass.range_constexpr(num_iters):
            row = cutlass.Int32(i * 32) + lane_id
            tok = blk_tok + row
            if tok < tok_end:
                b = cutlass.Float32(g_b[tok * num_sab_heads + sab_head_idx])
                sBeta_k[row] = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-b)
                )
            else:
                sBeta_k[row] = cutlass.Float32(0.0)

    # ─── kv_load / kv_store ───────────────────────────────────────────────────

    @cute.jit
    def kv_load(
        self,
        tKVrKV: cute.Tensor,
        gKV: cute.Tensor,
        kv_thr_mma,
    ):
        c_kv = cute.make_identity_tensor((self.D, self.D))
        tKVcKV = kv_thr_mma.partition_C(c_kv)
        for i in cutlass.range(cute.size(tKVrKV), unroll_full=True):
            v_idx, k_idx = tKVcKV[i]
            tKVrKV[i] = gKV[k_idx, v_idx]

    @cute.jit
    def kv_store(
        self,
        tKVrKV: cute.Tensor,
        gKV: cute.Tensor,
        kv_thr_mma,
    ):
        c_kv = cute.make_identity_tensor((self.D, self.D))
        tKVcKV = kv_thr_mma.partition_C(c_kv)
        for i in cutlass.range(cute.size(tKVrKV), unroll_full=True):
            v_idx, k_idx = tKVcKV[i]
            gKV[k_idx, v_idx] = tKVrKV[i]

    # ─── compute_loop_body ───────────────────────────────────────────────────
    # Translates the C++ compute_loop_body lambda captured inside compute().
    # Called by Math WGs (tidx >= 128) for one block iteration.

    @cute.jit
    def compute_loop_body(
        self,
        # Smem tensors (staged; caller indexes the active stage)
        sQ_SD: cute.Tensor,  # (BlkQ, D, StagesQ)  – row-major atom, swizzled
        sK_SD: cute.Tensor,  # (BlkKV, D, StagesK) – same atom
        sK_DS: cute.Tensor,  # (D, BlkKV, StagesK) – K transposed
        sV_DS: cute.Tensor,  # (D, BlkKV, StagesV) – V transposed
        sQK: cute.Tensor,  # (BlkQ, BlkKV)
        sKK_inv: cute.Tensor,  # (BlkKV, BlkKV)
        sKK_opd: cute.Tensor,  # sKK_inv storage recast as Element
        sO: cute.Tensor,  # O output smem (staged)
        sAlpha: cute.Tensor,  # (BlkQ, AlphaProcessor.NUM_CHANNELS, StagesAlpha) or zero-shaped
        sBeta: cute.Tensor,  # (BlkKV, StagesBeta) or zero-shaped
        kv_tiled_mma,
        # Mainloop pipelines and active read states
        q_pipeline,
        q_consumer_state,
        k_pipeline,
        k_consumer_state,
        v_pipeline,
        v_consumer_state,
        o_pipeline,
        o_producer_state,
        alpha_pipeline,
        alpha_consumer_state,
        beta_pipeline,
        beta_consumer_state,
        # Compile-time flags
        is_first_block: bool,
        is_final_block: bool,
        # Valid token count for masking on final block
        B: cutlass.Int32,
        # Running KV state (D×D fp32, in registers across all blocks)
        tKVrKV: cute.Tensor,
        # Scale factor
        scale: cutlass.Float32,
        # WG role: MathWarpGroupRole.KK or MathWarpGroupRole.QK.
        wg_idx: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx - cutlass.Int32(128)  # relative to compute threads
        kk_thread_idx = thread_idx % cutlass.Int32(128)
        qk_thread_idx = thread_idx % cutlass.Int32(128)

        # ── TiledMMAs ─────────────────────────────────────────────────────────
        blk_q = cute.size(sQ_SD, mode=[0])
        blk_kv = cute.size(sK_SD, mode=[0])
        d = cute.size(sQ_SD, mode=[1])
        tile_shape_qk = (blk_q, blk_kv, d)
        tile_shape_kk = tile_shape_qk
        tile_shape_o1 = (d, blk_q, d)
        tile_shape_o2 = (d, blk_q, blk_kv)
        tile_shape_sk = (d, blk_kv, d)
        tile_shape_newv = (d, blk_kv, blk_kv)
        k_stage = k_consumer_state.index
        q_stage = q_consumer_state.index
        v_stage = v_consumer_state.index
        o_stage = cutlass.Int32(0)
        alpha_stage = alpha_consumer_state.index
        beta_stage = beta_consumer_state.index

        mma_atom_4w = warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16))
        mma_atom_8w = warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16))

        # QK/KK: 4 warps × 16M = 64M  (1 warpgroup, 128 threads)
        qk_tiled_mma = cute.make_tiled_mma(
            mma_atom_4w, cute.make_layout((4, 1, 1)), permutation_mnk=tile_shape_qk
        )
        kk_tiled_mma = cute.make_tiled_mma(
            mma_atom_4w, cute.make_layout((4, 1, 1)), permutation_mnk=tile_shape_kk
        )

        # O1/O2/SK/NewV: 8 warps × 16M = 128M (both warpgroups, 256 threads)
        o1_tiled_mma = cute.make_tiled_mma(
            mma_atom_8w, cute.make_layout((8, 1, 1)), permutation_mnk=tile_shape_o1
        )
        o2_tiled_mma = cute.make_tiled_mma(
            mma_atom_8w, cute.make_layout((8, 1, 1)), permutation_mnk=tile_shape_o2
        )
        sk_tiled_mma = cute.make_tiled_mma(
            mma_atom_8w, cute.make_layout((8, 1, 1)), permutation_mnk=tile_shape_sk
        )
        newv_tiled_mma = cute.make_tiled_mma(
            mma_atom_8w, cute.make_layout((8, 1, 1)), permutation_mnk=tile_shape_newv
        )

        # ── Thread slices ─────────────────────────────────────────────────────
        qk_thr_mma = qk_tiled_mma.get_slice(qk_thread_idx)
        kk_thr_mma = kk_tiled_mma.get_slice(kk_thread_idx)
        sk_thr_mma = sk_tiled_mma.get_slice(thread_idx)
        newv_thr_mma = newv_tiled_mma.get_slice(thread_idx)
        o1_thr_mma = o1_tiled_mma.get_slice(thread_idx)
        o2_thr_mma = o2_tiled_mma.get_slice(thread_idx)
        kv_thr_mma = kv_tiled_mma.get_slice(thread_idx)

        # ── Copy atoms ────────────────────────────────────────────────────────
        ldsm_n4 = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        ldsm_t4 = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )

        # ── Active smem slices (extract 2D from staged tensors) ───────────────
        sQ_k = sQ_SD[None, None, q_stage]  # (BlkQ, D)
        sK_SD_k = sK_SD[None, None, k_stage]  # (BlkKV, D)
        sK_DS_k = sK_DS[None, None, k_stage]  # (D, BlkKV)

        # ── QK copies ─────────────────────────────────────────────────────────
        qk_tiled_copy_A = cute.make_tiled_copy_A(ldsm_n4, qk_tiled_mma)
        qk_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, qk_tiled_mma)
        qk_thr_copy_A = qk_tiled_copy_A.get_slice(qk_thread_idx)
        qk_thr_copy_B = qk_tiled_copy_B.get_slice(qk_thread_idx)

        tQKrQ = qk_thr_mma.make_fragment_A(qk_thr_mma.partition_A(sQ_k))
        tQKrQ_cv = qk_thr_copy_A.retile(tQKrQ)
        tQKsQ = qk_thr_copy_A.partition_S(sQ_SD)
        tQKrK = qk_thr_mma.make_fragment_B(qk_thr_mma.partition_B(sK_SD_k))
        tQKrK_cv = qk_thr_copy_B.retile(tQKrK)
        tQKsK = qk_thr_copy_B.partition_S(sK_SD)

        # ── KK copies (same atom as QK) ───────────────────────────────────────
        kk_tiled_copy_A = cute.make_tiled_copy_A(ldsm_n4, kk_tiled_mma)
        kk_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, kk_tiled_mma)
        kk_thr_copy_A = kk_tiled_copy_A.get_slice(kk_thread_idx)
        kk_thr_copy_B = kk_tiled_copy_B.get_slice(kk_thread_idx)

        tKKrA = kk_thr_mma.make_fragment_A(kk_thr_mma.partition_A(sK_SD_k))
        tKKrA_cv = kk_thr_copy_A.retile(tKKrA)
        tKKsA = kk_thr_copy_A.partition_S(sK_SD)
        tKKrB = kk_thr_mma.make_fragment_B(kk_thr_mma.partition_B(sK_SD_k))
        tKKrB_cv = kk_thr_copy_B.retile(tKKrB)
        tKKsB = kk_thr_copy_B.partition_S(sK_SD)

        # ── SK copies ─────────────────────────────────────────────────────────
        # SK B: K loaded from sK_SD (row-major BlkKV×D) with LDSM_N — matches C++ SK B-operand
        # SK C: V loaded from sV_DS (col-major D×BlkKV) with LDSM_T
        sk_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, sk_tiled_mma)
        sk_tiled_copy_C = cute.make_tiled_copy_C(ldsm_t4, sk_tiled_mma)
        sk_thr_copy_B = sk_tiled_copy_B.get_slice(thread_idx)
        sk_thr_copy_C = sk_tiled_copy_C.get_slice(thread_idx)

        # Work around DSL make_fragment_B not accepting partition_shape_B output directly.
        tSKrK = cute.make_rmem_tensor(
            sk_thr_mma.partition_shape_B(cute.slice_(tile_shape_sk, (0, None, None))),
            self.dtype,
        )
        tSKrK_cv = sk_thr_copy_B.retile(tSKrK)
        tSKsK = sk_thr_copy_B.partition_S(sK_SD)

        # ── NewV copies ───────────────────────────────────────────────────────
        newv_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, newv_tiled_mma)
        newv_thr_copy_B = newv_tiled_copy_B.get_slice(thread_idx)
        tNewVrB = newv_thr_mma.make_fragment_B(newv_thr_mma.partition_B(sKK_opd))
        tNewVrB_cv = newv_thr_copy_B.retile(tNewVrB)
        tNewVsB = newv_thr_copy_B.partition_S(sKK_opd)

        # ── KV copies ─────────────────────────────────────────────────────────
        kv_tiled_copy_B = cute.make_tiled_copy_B(ldsm_t4, kv_tiled_mma)
        kv_thr_copy_B = kv_tiled_copy_B.get_slice(thread_idx)
        tKVrK = kv_thr_mma.make_fragment_B(kv_thr_mma.partition_B(sK_DS_k))
        tKVrK_cv = kv_thr_copy_B.retile(tKVrK)
        tKVsK = kv_thr_copy_B.partition_S(sK_DS)

        # ── O1/O2 copies ──────────────────────────────────────────────────────
        o1_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, o1_tiled_mma)
        o2_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, o2_tiled_mma)
        o1_thr_copy_B = o1_tiled_copy_B.get_slice(thread_idx)
        o2_thr_copy_B = o2_tiled_copy_B.get_slice(thread_idx)

        # Direct partition_B(sQ_k) preserves the swizzled Q layout here and produces
        # a non-C++ B fragment shape; derive the fragment from TileShapeO1 instead.
        tOrQ = cute.make_rmem_tensor(
            o1_thr_mma.partition_shape_B(cute.slice_(tile_shape_o1, (0, None, None))),
            self.dtype,
        )
        tOrQ_cv = o1_thr_copy_B.retile(tOrQ)
        tOsQ = o1_thr_copy_B.partition_S(sQ_SD)
        tOrQK = o2_thr_mma.make_fragment_B(o2_thr_mma.partition_B(sQK))
        tOrQK_cv = o2_thr_copy_B.retile(tOrQK)
        tOsQK = o2_thr_copy_B.partition_S(sQK)

        # ── O store (R→S STSM) ────────────────────────────────────────────────
        o_stsm = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )
        o_tiled_copy_r2s = cute.make_tiled_copy_C(o_stsm, o1_tiled_mma)
        o_thr_copy_r2s = o_tiled_copy_r2s.get_slice(thread_idx)
        tOsO = o_thr_copy_r2s.partition_D(sO)

        # ── Coordinate tensors for masking / alpha/beta indexing ──────────────
        cMqk = cute.make_identity_tensor((blk_q, blk_kv))
        tQKcMqk = qk_thr_mma.partition_C(cMqk)
        cMkk = cMqk  # same shape (BlkKV == BlkQ == 64)
        tKKcMkk = kk_thr_mma.partition_C(cMkk)
        cO = cute.make_identity_tensor((d, blk_q))
        tOcO = o1_thr_mma.partition_C(cO)
        cSK = cute.make_identity_tensor((d, blk_kv))
        tSKcSK = sk_thr_mma.partition_C(cSK)
        cV = cute.make_identity_tensor((d, blk_kv))
        tKVcV = kv_thr_mma.partition_A(cV)

        # ── KK GEMM (WG0 only) ────────────────────────────────────────────────
        k_pipeline.consumer_wait(k_consumer_state)
        if cutlass.const_expr(self.needs_alpha):
            alpha_pipeline.consumer_wait(alpha_consumer_state)
            cute.arch.fence_view_async_shared()
        if cutlass.const_expr(self.needs_beta):
            beta_pipeline.consumer_wait(beta_consumer_state)
            cute.arch.fence_view_async_shared()
        # Match the C++ reject-non-role-first shape; ptxas keeps BRA.U around
        # the role body instead of predicating the HMMA/LDSM/STSM sequence.
        if wg_idx != MathWarpGroupRole.KK:
            cute.arch.sync_warp()
        else:
            cute.copy(kk_tiled_copy_A, tKKsA[None, None, None, k_stage], tKKrA_cv)
            cute.copy(kk_tiled_copy_B, tKKsB[None, None, None, k_stage], tKKrB_cv)
            tKKrKK = cute.make_rmem_tensor(
                kk_thr_mma.partition_shape_C((blk_kv, blk_kv)), self.acc_dtype
            )
            tKKrKK.fill(self.acc_dtype(0.0))
            cute.gemm(kk_tiled_mma, tKKrKK, tKKrA, tKKrB, tKKrKK)
            self.kk_epi(tKKrKK, tKKcMkk, sAlpha, sBeta, alpha_stage, beta_stage)
            self.qk_or_kk_mask(tKKrKK, tKKcMkk, is_final_block, B)
            self._kk_store_and_inv(
                tKKrKK,
                kk_tiled_mma,
                kk_thread_idx,
                sKK_inv,
                sKK_opd,
                sBeta,
                beta_stage,
                tKKcMkk,
            )
        if cutlass.const_expr(self.needs_beta):
            beta_pipeline.consumer_release(beta_consumer_state)
            beta_consumer_state.advance()

        # ── QK GEMM (WG1 only) ────────────────────────────────────────────────
        q_pipeline.consumer_wait(q_consumer_state)
        if wg_idx != MathWarpGroupRole.QK:
            cute.arch.sync_warp()
        else:
            cute.copy(qk_tiled_copy_A, tQKsQ[None, None, None, q_stage], tQKrQ_cv)
            cute.copy(qk_tiled_copy_B, tQKsK[None, None, None, k_stage], tQKrK_cv)
            tQKrQK = cute.make_rmem_tensor(
                qk_thr_mma.partition_shape_C((blk_q, blk_kv)), self.acc_dtype
            )
            tQKrQK.fill(self.acc_dtype(0.0))
            cute.gemm(qk_tiled_mma, tQKrQK, tQKrQ, tQKrK, tQKrQK)
            self.qk_epi(tQKrQK, tQKcMqk, sAlpha, alpha_stage, scale)
            self.qk_or_kk_mask(tQKrQK, tQKcMqk, is_final_block, B)
            self.qk_store(tQKrQK, sQK, qk_tiled_mma, qk_thread_idx)

        # ── O1: KV_state @ Q (both WGs, skip on first block) ─────────────────
        tOrO = cute.make_rmem_tensor(
            o1_thr_mma.partition_shape_C((d, blk_q)), self.acc_dtype
        )
        tOrO.fill(self.acc_dtype(0.0))
        if cutlass.const_expr(not is_first_block):
            cute.copy(o1_tiled_copy_B, tOsQ[None, None, None, q_stage], tOrQ_cv)
            tOrKV = SM80.make_acc_into_op(tKVrKV, o1_tiled_mma, self.dtype)
            cute.gemm(o1_tiled_mma, tOrO, tOrKV, tOrQ, tOrO)
            self.o1_epi(tOrO, tOcO, sAlpha, alpha_stage, scale)
        q_pipeline.consumer_release(q_consumer_state)
        q_consumer_state.advance()

        # ── SK: KV_state @ K^T (result negated below via V - SK) ─────────────
        tSKrSK = cute.make_rmem_tensor(
            sk_thr_mma.partition_shape_C((d, blk_kv)), self.acc_dtype
        )
        tSKrSK.fill(self.acc_dtype(0.0))
        if cutlass.const_expr(not is_first_block):
            tSKrS = SM80.make_acc_into_op(tKVrKV, sk_tiled_mma, self.dtype)
            cute.copy(sk_tiled_copy_B, tSKsK[None, None, None, k_stage], tSKrK_cv)
            cute.gemm(sk_tiled_mma, tSKrSK, tSKrS, tSKrK, tSKrSK)

        # ── Load V from smem ──────────────────────────────────────────────────
        v_pipeline.consumer_wait(v_consumer_state)
        tSKrV = self.sk_load_v(tSKrSK, sV_DS, sk_tiled_copy_C, sk_thr_copy_C, v_stage)

        # sk_epi + V - SK  (SK=0 on first block, so V - SK = V)
        if cutlass.const_expr(not is_first_block):
            self.sk_epi(tSKrSK, tSKcSK, sAlpha, alpha_stage)
            for i in cutlass.range_constexpr(cute.size(tSKrV)):
                tSKrV[i] = tSKrV[i] - self.dtype(tSKrSK[i])

        # ── NewV = (V - SK) @ T^T  (ordered: WG0 first) ──────────────────────
        tNewVrA = SM80.make_acc_into_op(tSKrV, newv_tiled_mma, self.dtype)
        tNewVrC = cute.make_rmem_tensor(
            newv_thr_mma.partition_shape_C((d, blk_kv)), self.acc_dtype
        )
        self._math_order_wait(wg_idx)
        cute.copy(newv_tiled_copy_B, tNewVsB, tNewVrB_cv)
        tNewVrC.fill(self.acc_dtype(0.0))
        cute.gemm(newv_tiled_mma, tNewVrC, tNewVrA, tNewVrB, tNewVrC)
        self._math_order_notify(wg_idx)
        v_pipeline.consumer_release(v_consumer_state)
        v_consumer_state.advance()

        # ── O2 = O1 + NewV @ QK  (ordered: WG0 first) ────────────────────────
        tOrNewV = SM80.make_acc_into_op(tNewVrC, o2_tiled_mma, self.dtype)
        self._math_order_wait(wg_idx)
        cute.copy(o2_tiled_copy_B, tOsQK, tOrQK_cv)
        cute.gemm(o2_tiled_mma, tOrO, tOrNewV, tOrQK, tOrO)
        self._math_order_notify(wg_idx)

        # ── O store to smem ───────────────────────────────────────────────────
        o_pipeline.producer_acquire(o_producer_state)
        self.o_store(
            tOrO,
            tOsO[None, None, None, o_stage],
            o_tiled_copy_r2s,
            o_thr_copy_r2s,
        )
        o_pipeline.producer_commit(o_producer_state)
        o_producer_state.advance()

        # ── KV state update ───────────────────────────────────────────────────
        block_coeff = cutlass.Float32(1.0)
        if cutlass.const_expr(self.needs_alpha):
            block_coeff = cutlass.Float32(
                sAlpha[B - cutlass.Int32(1), AlphaProcessor.CUMPROD, alpha_stage]
            )

        for i in cutlass.range(cute.size(tKVrKV), unroll_full=True):
            tKVrKV[i] = block_coeff * tKVrKV[i]

        self.kv_decay_v(tOrNewV, tKVcV, sAlpha, alpha_stage, is_final_block, B)

        # KV += NewV @ K
        cute.copy(kv_tiled_copy_B, tKVsK[None, None, None, k_stage], tKVrK_cv)
        cute.gemm(kv_tiled_mma, tKVrKV, tOrNewV, tKVrK, tKVrKV)
        k_pipeline.consumer_release(k_consumer_state)
        k_consumer_state.advance()
        if cutlass.const_expr(self.needs_alpha):
            alpha_pipeline.consumer_release(alpha_consumer_state)
            alpha_consumer_state.advance()
        return (
            q_consumer_state,
            k_consumer_state,
            v_consumer_state,
            o_producer_state,
            alpha_consumer_state,
            beta_consumer_state,
        )

    # ─── Warp role entry points ──────────────────────────────────────────────
    # The current DSL bridge still uses CTA-wide sync epochs, but each role owns
    # its own loop, matching the C++ warp-specialized dispatch shape.

    @cute.jit
    def run_load_qkv_role(
        self,
        sQ_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        q_pipeline,
        k_pipeline,
        v_pipeline,
        num_blocks: cutlass.Int32,
        tok_start: cutlass.Int32,
        q_head_idx: cutlass.Int32,
        k_head_idx: cutlass.Int32,
        v_head_idx: cutlass.Int32,
    ):
        q_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.q_stage
        )
        k_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.k_stage
        )
        v_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.v_stage
        )
        for blk in cutlass.range(num_blocks, unroll=1):
            (
                q_producer_state,
                k_producer_state,
                v_producer_state,
            ) = self.load_qkv_tma(
                sQ_SD,
                sK_DS,
                sV_DS,
                tma_atom_q,
                tma_tensor_q,
                tma_atom_k,
                tma_tensor_k,
                tma_atom_v,
                tma_tensor_v,
                q_pipeline,
                q_producer_state,
                k_pipeline,
                k_producer_state,
                v_pipeline,
                v_producer_state,
                blk,
                tok_start,
                q_head_idx,
                k_head_idx,
                v_head_idx,
            )

    @cute.jit
    def run_load_alpha_role(
        self,
        sAlpha: cute.Tensor,
        g_a: cute.Tensor,
        g_A_log: cute.Tensor,
        g_dt_bias: cute.Tensor,
        alpha_pipeline,
        scale: cutlass.Float32,
        num_blocks: cutlass.Int32,
        tok_start: cutlass.Int32,
        tok_end: cutlass.Int32,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
    ):
        alpha_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.alpha_beta_stage
        )
        for blk in cutlass.range(num_blocks, unroll=1):
            blk_tok = tok_start + blk * cutlass.Int32(self.BLK_Q)
            if cutlass.const_expr(self.needs_alpha):
                alpha_pipeline.producer_acquire(alpha_producer_state)
                cute.arch.fence_view_async_shared()
                self.load_alpha(
                    sAlpha,
                    g_a,
                    g_A_log,
                    g_dt_bias,
                    blk_tok,
                    tok_end,
                    sab_head_idx,
                    num_sab_heads,
                    alpha_producer_state.index,
                )
                AlphaProcessor().run(
                    sAlpha[None, None, alpha_producer_state.index], scale
                )
                cute.arch.fence_view_async_shared()
                alpha_pipeline.producer_commit(alpha_producer_state)
                alpha_producer_state.advance()

    @cute.jit
    def run_load_beta_role(
        self,
        sBeta: cute.Tensor,
        g_b: cute.Tensor,
        beta_pipeline,
        num_blocks: cutlass.Int32,
        tok_start: cutlass.Int32,
        tok_end: cutlass.Int32,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
    ):
        beta_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.alpha_beta_stage
        )
        for blk in cutlass.range(num_blocks, unroll=1):
            blk_tok = tok_start + blk * cutlass.Int32(self.BLK_KV)
            if cutlass.const_expr(self.needs_beta):
                beta_pipeline.producer_acquire(beta_producer_state)
                cute.arch.fence_view_async_shared()
                self.load_beta(
                    sBeta,
                    g_b,
                    blk_tok,
                    tok_end,
                    sab_head_idx,
                    num_sab_heads,
                    beta_producer_state.index,
                )
                cute.arch.fence_view_async_shared()
                beta_pipeline.producer_commit(beta_producer_state)
                beta_producer_state.advance()

    @cute.jit
    def run_math_role(
        self,
        sQ_SD: cute.Tensor,
        sK_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,
        sQK: cute.Tensor,
        sKK_inv: cute.Tensor,
        sKK_opd: cute.Tensor,
        sO: cute.Tensor,
        sAlpha: cute.Tensor,
        sBeta: cute.Tensor,
        q_pipeline,
        k_pipeline,
        v_pipeline,
        o_pipeline,
        alpha_pipeline,
        beta_pipeline,
        g_state: cute.Tensor,
        g_init_state: cute.Tensor,
        work_desc: WorkDesc,
        scale: cutlass.Float32,
        wg_idx: cutlass.Int32,
        math_tidx: cutlass.Int32,
        num_blocks: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        state_pool_rows: cutlass.Int32,
    ):
        self._math_order_init(wg_idx)
        q_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.q_stage
        )
        k_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.k_stage
        )
        v_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.v_stage
        )
        o_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.o_stage
        )
        alpha_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_beta_stage
        )
        beta_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_beta_stage
        )

        kv_tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16)),
            cute.make_layout((8, 1, 1)),
            permutation_mnk=(self.D, self.D, self.BLK_KV),
        )
        kv_thr_mma = kv_tiled_mma.get_slice(math_tidx)
        tKVrKV = cute.make_rmem_tensor(
            kv_thr_mma.partition_shape_C((self.D, self.D)), self.acc_dtype
        )
        tKVrKV.fill(self.acc_dtype(0.0))

        # Plugin storage is row-major [resident, Hv, K, V], so V must be the
        # innermost mode in this reordered (K, V, Hv, resident) view. FlashInfer's
        # original state tensor uses the opposite D-major orientation.
        state_layout = cute.make_ordered_layout(
            (self.D, self.D, num_sab_heads, state_pool_rows), order=(1, 0, 2, 3)
        )
        o_head_idx = work_desc.o_head_idx(num_q_heads, num_v_heads)
        mState = cute.make_tensor(g_state.iterator, state_layout)
        mInitState = cute.make_tensor(g_init_state.iterator, state_layout)
        # Invalid standalone indices use a zero initial state and never update the pool.
        if cute.elem_less(cutlass.Int32(-1), work_desc.state_idx):
            if cute.elem_less(work_desc.state_idx, state_pool_rows):
                gInitKV = mInitState[None, None, o_head_idx, work_desc.state_idx]
                self.kv_load(tKVrKV, gInitKV, kv_thr_mma)

        first_B = work_desc.seq_len
        if first_B > cutlass.Int32(self.BLK_KV):
            first_B = cutlass.Int32(self.BLK_KV)
        (
            q_consumer_state,
            k_consumer_state,
            v_consumer_state,
            o_producer_state,
            alpha_consumer_state,
            beta_consumer_state,
        ) = self.compute_loop_body(
            sQ_SD,
            sK_SD,
            sK_DS,
            sV_DS,
            sQK,
            sKK_inv,
            sKK_opd,
            sO,
            sAlpha,
            sBeta,
            kv_tiled_mma,
            q_pipeline,
            q_consumer_state,
            k_pipeline,
            k_consumer_state,
            v_pipeline,
            v_consumer_state,
            o_pipeline,
            o_producer_state,
            alpha_pipeline,
            alpha_consumer_state,
            beta_pipeline,
            beta_consumer_state,
            False,
            True,
            first_B,
            tKVrKV,
            scale,
            wg_idx,
        )
        for blk in cutlass.range(
            cutlass.Int32(1), num_blocks - cutlass.Int32(1), cutlass.Int32(1), unroll=1
        ):
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                o_producer_state,
                alpha_consumer_state,
                beta_consumer_state,
            ) = self.compute_loop_body(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                sBeta,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                o_pipeline,
                o_producer_state,
                alpha_pipeline,
                alpha_consumer_state,
                beta_pipeline,
                beta_consumer_state,
                False,
                False,
                cutlass.Int32(self.BLK_KV),
                tKVrKV,
                scale,
                wg_idx,
            )
        if num_blocks != cutlass.Int32(1):
            last_blk = num_blocks - cutlass.Int32(1)
            last_B = work_desc.seq_len - last_blk * cutlass.Int32(self.BLK_KV)
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                o_producer_state,
                alpha_consumer_state,
                beta_consumer_state,
            ) = self.compute_loop_body(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                sBeta,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                o_pipeline,
                o_producer_state,
                alpha_pipeline,
                alpha_consumer_state,
                beta_pipeline,
                beta_consumer_state,
                False,
                True,
                last_B,
                tKVrKV,
                scale,
                wg_idx,
        )
        if cute.elem_less(cutlass.Int32(-1), work_desc.state_idx):
            if cute.elem_less(work_desc.state_idx, state_pool_rows):
                gStateKV = mState[None, None, o_head_idx, work_desc.state_idx]
                self.kv_store(tKVrKV, gStateKV, kv_thr_mma)

    # ─── Kernel entry point ───────────────────────────────────────────────────

    @cute.jit
    def __call__(
        self,
        g_q: cute.Tensor,
        g_k: cute.Tensor,
        g_v: cute.Tensor,
        g_o: cute.Tensor,
        g_a: cute.Tensor,
        g_b: cute.Tensor,
        g_A_log: cute.Tensor,
        g_dt_bias: cute.Tensor,
        g_state: cute.Tensor,
        g_init_state: cute.Tensor,
        g_tensormaps: cute.Tensor,
        context_lengths: cute.Tensor,
        state_indices: cute.Tensor,
        use_state_indices: cutlass.Int32,
        state_pool_rows: cutlass.Int32,
        physical_seq_len: cutlass.Int32,
        scale: cutlass.Float32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        num_seqs: cutlass.Int32,
        enable_pdl: cutlass.Int32,
        grid_x: int,
        stream,
    ):
        qkv_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128,
            self.dtype,
        )
        q_storage_layout = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_Q, self.D, self.q_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        q_smem_layout = cute.slice_(q_storage_layout, (None, None, 0))
        k_storage_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.k_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        k_storage_layout_ds = cute.select(k_storage_layout_sd, [1, 0, 2])
        v_storage_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.v_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        v_storage_layout_ds = cute.select(v_storage_layout_sd, [1, 0, 2])
        k_smem_layout = cute.slice_(k_storage_layout_ds, (None, None, 0))
        v_smem_layout = cute.slice_(v_storage_layout_ds, (None, None, 0))
        o_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.MN_SW32,
            self.dtype,
        )
        o_storage_layout = cute.tile_to_shape(
            o_smem_layout_atom,
            (self.D, self.BLK_Q, self.o_stage),
            order=(1, 0, 2),
        )
        o_smem_layout = cute.slice_(o_storage_layout, (None, None, 0))

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
            tma_load_op, g_q, q_smem_layout, (self.BLK_Q, self.D)
        )
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            tma_load_op, g_k, k_smem_layout, (self.D, self.BLK_KV)
        )
        tma_atom_v, tma_tensor_v = cpasync.make_tiled_tma_atom(
            tma_load_op, g_v, v_smem_layout, (self.D, self.BLK_KV)
        )

        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_o, tma_tensor_o = cpasync.make_tiled_tma_atom(
            tma_store_op, g_o, o_smem_layout, (self.D, self.BLK_Q)
        )

        dtype_bytes = self.dtype.width // 8
        self.tma_load_q_bytes = cute.size(q_smem_layout) * dtype_bytes
        self.tma_load_k_bytes = cute.size(k_smem_layout) * dtype_bytes
        self.tma_load_v_bytes = cute.size(v_smem_layout) * dtype_bytes

        qk_layout_atom = cute.make_layout((8, 8), stride=(8, 1))
        qk_storage_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_Q, self.BLK_KV), order=(1, 0)
        )
        kk_storage_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_KV, self.BLK_KV), order=(1, 0)
        )
        alpha_storage_layout = cute.make_layout(
            (self.BLK_Q, AlphaProcessor.NUM_CHANNELS, self.alpha_beta_stage)
        )
        beta_storage_layout = cute.make_layout((self.BLK_KV, self.alpha_beta_stage))

        @cute.struct
        class SharedStorage:
            q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.k_stage * 2]
            v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.v_stage * 2]
            o_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.o_stage * 2]
            alpha_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.alpha_beta_stage * 2
            ]
            beta_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.alpha_beta_stage * 2
            ]

            smem_q: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(q_storage_layout)],
                128,
            ]
            smem_k: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(k_storage_layout_sd)],
                128,
            ]
            smem_v: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(v_storage_layout_sd)],
                128,
            ]
            smem_qk: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(qk_storage_layout)],
                16,
            ]
            smem_kk: cute.struct.Align[
                cute.struct.MemRange[
                    self.inverse_dtype, cute.cosize(kk_storage_layout)
                ],
                16,
            ]
            smem_o: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(o_storage_layout)],
                128,
            ]
            smem_alpha: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(alpha_storage_layout)
                ],
                16,
            ]
            smem_beta: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(beta_storage_layout)],
                16,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            g_a,
            g_b,
            g_A_log,
            g_dt_bias,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_o,
            tma_tensor_o,
            g_o,
            g_state,
            g_init_state,
            g_tensormaps,
            context_lengths,
            state_indices,
            use_state_indices,
            state_pool_rows,
            physical_seq_len,
            scale,
            num_q_heads,
            num_v_heads,
            num_sab_heads,
            num_seqs,
            enable_pdl,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(384, 1, 1),
            max_number_threads=(384, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        g_a: cute.Tensor,
        g_b: cute.Tensor,
        g_A_log: cute.Tensor,
        g_dt_bias: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        g_o: cute.Tensor,
        g_state: cute.Tensor,
        g_init_state: cute.Tensor,
        g_tensormaps: cute.Tensor,
        context_lengths: cute.Tensor,
        state_indices: cute.Tensor,
        use_state_indices: cutlass.Int32,
        state_pool_rows: cutlass.Int32,
        physical_seq_len: cutlass.Int32,
        scale: cutlass.Float32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        num_seqs: cutlass.Int32,
        enable_pdl: cutlass.Int32,
    ):
        NUM_LOAD_WARP_GROUPS = 1
        NUM_MMA_WARP_GROUPS = 2
        THREADS_PER_WARP_GROUP = 128
        WARPS_PER_WARP_GROUP = 4
        MIN_BLOCKS_PER_MP = 1
        MAX_THREADS_PER_BLOCK = (
            NUM_LOAD_WARP_GROUPS + NUM_MMA_WARP_GROUPS
        ) * THREADS_PER_WARP_GROUP
        load_registers, mma_registers = self.get_register_requirements(
            MAX_THREADS_PER_BLOCK,
            MIN_BLOCKS_PER_MP,
            NUM_MMA_WARP_GROUPS,
            THREADS_PER_WARP_GROUP,
        )

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // cutlass.Int32(THREADS_PER_WARP_GROUP)
        )
        ldst_warp_role = cute.arch.make_warp_uniform(
            warp_idx % cutlass.Int32(WARPS_PER_WARP_GROUP)
        )

        if warp_idx == LoadStoreWarpRole.LOAD_QKV:
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)
            cpasync.prefetch_descriptor(tma_atom_o)

        work_desc = self.get_next_work(
            context_lengths,
            state_indices,
            use_state_indices,
            physical_seq_len,
            num_q_heads,
            num_v_heads,
            num_sab_heads,
        )
        tok_end = work_desc.tok_offset + work_desc.seq_len
        num_blocks = (
            work_desc.seq_len + cutlass.Int32(self.BLK_KV) - cutlass.Int32(1)
        ) // cutlass.Int32(self.BLK_KV)

        # math_tidx / wg_idx: valid for Math WG threads; LdSt WG gets negative values (unused)
        math_tidx = tidx - cutlass.Int32(THREADS_PER_WARP_GROUP)
        wg_idx = math_tidx // cutlass.Int32(THREADS_PER_WARP_GROUP)

        # ── Smem allocation ───────────────────────────────────────────────────
        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)

        qkv_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128,
            self.dtype,
        )
        q_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_Q, self.D, self.q_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        sQ_SD = storage.smem_q.get_tensor(q_layout_sd.outer, swizzle=q_layout_sd.inner)

        k_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.k_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        k_layout_ds = cute.select(k_layout_sd, [1, 0, 2])
        sK_SD = storage.smem_k.get_tensor(k_layout_sd.outer, swizzle=k_layout_sd.inner)
        sK_DS = storage.smem_k.get_tensor(k_layout_ds.outer, swizzle=k_layout_ds.inner)

        v_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.v_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        v_layout_ds = cute.select(v_layout_sd, [1, 0, 2])
        sV_DS = storage.smem_v.get_tensor(v_layout_ds.outer, swizzle=v_layout_ds.inner)

        qk_layout_atom = cute.make_layout((8, 8), stride=(8, 1))
        qk_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_Q, self.BLK_KV), order=(1, 0)
        )
        sQK = storage.smem_qk.get_tensor(qk_layout)

        kk_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_KV, self.BLK_KV), order=(1, 0)
        )
        sKK_inv = storage.smem_kk.get_tensor(kk_layout)
        kk_opd_ptr = cute.recast_ptr(storage.smem_kk.data_ptr(), dtype=self.dtype)
        sKK_opd = cute.make_tensor(kk_opd_ptr, kk_layout)

        o_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.MN_SW32,
            self.dtype,
        )
        o_layout = cute.tile_to_shape(
            o_smem_layout_atom,
            (self.D, self.BLK_Q, self.o_stage),
            order=(1, 0, 2),
        )
        sO = storage.smem_o.get_tensor(o_layout.outer, swizzle=o_layout.inner)
        alpha_layout = cute.make_layout(
            (self.BLK_Q, AlphaProcessor.NUM_CHANNELS, self.alpha_beta_stage)
        )
        sAlpha = storage.smem_alpha.get_tensor(alpha_layout)

        beta_layout = cute.make_layout((self.BLK_KV, self.alpha_beta_stage))
        sBeta = storage.smem_beta.get_tensor(beta_layout)

        load_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        load_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 8)
        vector_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        vector_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, NUM_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP
        )
        o_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, NUM_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP
        )
        o_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar_ptr.data_ptr(),
            num_stages=self.q_stage,
            producer_group=load_producer_group,
            consumer_group=load_consumer_group,
            tx_count=self.tma_load_q_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        k_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.k_mbar_ptr.data_ptr(),
            num_stages=self.k_stage,
            producer_group=load_producer_group,
            consumer_group=load_consumer_group,
            tx_count=self.tma_load_k_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        v_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.v_mbar_ptr.data_ptr(),
            num_stages=self.v_stage,
            producer_group=load_producer_group,
            consumer_group=load_consumer_group,
            tx_count=self.tma_load_v_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        o_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.o_mbar_ptr.data_ptr(),
            num_stages=self.o_stage,
            producer_group=o_producer_group,
            consumer_group=o_consumer_group,
        )
        alpha_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.alpha_mbar_ptr.data_ptr(),
            num_stages=self.alpha_beta_stage,
            producer_group=vector_producer_group,
            consumer_group=vector_consumer_group,
        )
        beta_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.beta_mbar_ptr.data_ptr(),
            num_stages=self.alpha_beta_stage,
            producer_group=vector_producer_group,
            consumer_group=vector_consumer_group,
        )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # The Q/K descriptors, work selection, and shared-memory pipelines are
        # independent of the preceding L2-normalization kernel. Defer only the
        # first payload access until every producer CTA has completed its stores.
        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        if (
            work_desc.seq_len != cutlass.Int32(0)
            and warp_group_idx == WarpGroupRole.LDST
        ):
            cute.arch.setmaxregister_decrease(load_registers)
            if ldst_warp_role == LoadStoreWarpRole.LOAD_QKV:
                self.run_load_qkv_role(
                    sQ_SD,
                    sK_DS,
                    sV_DS,
                    tma_atom_q,
                    tma_tensor_q,
                    tma_atom_k,
                    tma_tensor_k,
                    tma_atom_v,
                    tma_tensor_v,
                    q_pipeline,
                    k_pipeline,
                    v_pipeline,
                    num_blocks,
                    work_desc.tok_offset,
                    work_desc.q_head_idx(),
                    work_desc.k_head_idx(num_q_heads, num_v_heads),
                    work_desc.v_head_idx(),
                )
            elif ldst_warp_role == LoadStoreWarpRole.STORE_O:
                CollectiveStoreTma(self.BLK_Q, self.D).run(
                    sO,
                    tma_atom_o,
                    tma_tensor_o,
                    g_tensormaps,
                    o_pipeline,
                    num_blocks,
                    work_desc,
                    self.o_stage,
                    num_q_heads,
                    num_v_heads,
                )
                self.zero_output_padding(
                    g_o,
                    work_desc,
                    physical_seq_len,
                    num_q_heads,
                    num_v_heads,
                )
                if enable_pdl:
                    # Release after output TMA while the state-store tail drains.
                    with cute.arch.elect_one():
                        cute.arch.griddepcontrol_launch_dependents()
            elif ldst_warp_role == LoadStoreWarpRole.LOAD_BETA:
                self.run_load_beta_role(
                    sBeta,
                    g_b,
                    beta_pipeline,
                    num_blocks,
                    work_desc.tok_offset,
                    tok_end,
                    work_desc.o_head_idx(num_q_heads, num_v_heads),
                    num_sab_heads,
                )
            elif ldst_warp_role == LoadStoreWarpRole.LOAD_ALPHA:
                self.run_load_alpha_role(
                    sAlpha,
                    g_a,
                    g_A_log,
                    g_dt_bias,
                    alpha_pipeline,
                    scale,
                    num_blocks,
                    work_desc.tok_offset,
                    tok_end,
                    work_desc.o_head_idx(num_q_heads, num_v_heads),
                    num_sab_heads,
                )
        elif work_desc.seq_len != cutlass.Int32(0):
            cute.arch.setmaxregister_increase(mma_registers)

            self.run_math_role(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                sBeta,
                q_pipeline,
                k_pipeline,
                v_pipeline,
                o_pipeline,
                alpha_pipeline,
                beta_pipeline,
                g_state,
                g_init_state,
                work_desc,
                scale,
                wg_idx,
                math_tidx,
                num_blocks,
                num_q_heads,
                num_v_heads,
                num_sab_heads,
                state_pool_rows,
            )


# ─── EdgeLLM AOT interface ───────────────────────────────────────────────────

AOT_PLACEHOLDER_N = 1
AOT_PLACEHOLDER_T = 128
AOT_PLACEHOLDER_HQK = 8
AOT_PLACEHOLDER_HV = 8
AOT_HEAD_DIM = 128
AOT_TENSORMAP_BYTES = TENSOR_MAP_DESCRIPTOR_BYTES

_kernel_instance = FullyFusedGdnPrefillBlackwellGeforce(dtype=cutlass.Float16)


@cute.jit
def run_gdn_prefill_blackwell_geforce(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    a: cute.Tensor,
    b: cute.Tensor,
    A_log: cute.Tensor,
    dt_bias: cute.Tensor,
    h0_in: cute.Tensor,
    h0_out: cute.Tensor,
    state_indices: cute.Tensor,
    use_state_indices: cutlass.Int32,
    context_lengths: cute.Tensor,
    o: cute.Tensor,
    tensormap_scratch: cute.Tensor,
    enable_pdl: cutlass.Int32,
    stream: cuda.CUstream,
):
    n = q.layout.shape[0]
    physical_seq_len = q.layout.shape[1]
    num_q_heads = q.layout.shape[2]
    d = q.layout.shape[3]
    num_v_heads = v.layout.shape[2]
    state_pool_rows = h0_out.layout.shape[0]
    total_tokens = n * physical_seq_len

    q_tma = cute.make_tensor(
        q.iterator,
        cute.make_layout(
            (total_tokens, d, num_q_heads),
            stride=(num_q_heads * d, 1, d),
        ),
    )
    k_tma = cute.make_tensor(
        k.iterator,
        cute.make_layout(
            (d, total_tokens, num_q_heads),
            stride=(1, num_q_heads * d, d),
        ),
    )
    v_tma = cute.make_tensor(
        v.iterator,
        cute.make_layout(
            (d, total_tokens, num_v_heads),
            stride=(1, num_v_heads * d, d),
        ),
    )
    o_tma = cute.make_tensor(
        o.iterator,
        cute.make_layout(
            (d, total_tokens, num_v_heads),
            stride=(1, num_v_heads * d, d),
        ),
    )
    ab_layout = cute.make_layout((total_tokens * num_v_heads,), stride=(1,))
    a_flat = cute.make_tensor(a.iterator, ab_layout)
    b_flat = cute.make_tensor(b.iterator, ab_layout)

    _kernel_instance(
        q_tma,
        k_tma,
        v_tma,
        o_tma,
        a_flat,
        b_flat,
        A_log,
        dt_bias,
        h0_out,
        h0_in,
        tensormap_scratch,
        context_lengths,
        state_indices,
        use_state_indices,
        cutlass.Int32(state_pool_rows),
        cutlass.Int32(physical_seq_len),
        cutlass.Float32(AOT_HEAD_DIM ** -0.5),
        cutlass.Int32(num_q_heads),
        cutlass.Int32(num_v_heads),
        cutlass.Int32(num_v_heads),
        cutlass.Int32(n),
        enable_pdl,
        n * num_v_heads,
        stream,
    )


def _make_placeholders(n, t, hqk, hv):
    fp16 = cp.float16
    return {
        "q": cp.zeros((n, t, hqk, AOT_HEAD_DIM), dtype=fp16),
        "k": cp.zeros((n, t, hqk, AOT_HEAD_DIM), dtype=fp16),
        "v": cp.zeros((n, t, hv, AOT_HEAD_DIM), dtype=fp16),
        "a": cp.zeros((n, t, hv), dtype=fp16),
        "b": cp.zeros((n, t, hv), dtype=fp16),
        "A_log": cp.zeros((hv,), dtype=cp.float32),
        "dt_bias": cp.zeros((hv,), dtype=fp16),
        "h0_in": cp.zeros((n, hv, AOT_HEAD_DIM, AOT_HEAD_DIM), dtype=cp.float32),
        "h0_out": cp.zeros((n, hv, AOT_HEAD_DIM, AOT_HEAD_DIM), dtype=cp.float32),
        "state_indices": cp.arange(n, dtype=cp.int32),
        "context_lengths": cp.full((n,), t, dtype=cp.int32),
        "o": cp.zeros((n, t, hv, AOT_HEAD_DIM), dtype=fp16),
        "tensormap_scratch": cp.zeros((AOT_TENSORMAP_BYTES,), dtype=cp.uint8),
    }


def _mark_4d_dynamic(array):
    order = (0, 1, 2, 3)
    return (
        from_dlpack(array, assumed_align=16)
        .mark_layout_dynamic(leading_dim=3)
        .mark_compact_shape_dynamic(mode=0, stride_order=order)
        .mark_compact_shape_dynamic(mode=1, stride_order=order)
        .mark_compact_shape_dynamic(mode=2, stride_order=order)
    )


def _mark_3d_dynamic(array):
    order = (0, 1, 2)
    return (
        from_dlpack(array, assumed_align=16)
        .mark_layout_dynamic(leading_dim=2)
        .mark_compact_shape_dynamic(mode=0, stride_order=order)
        .mark_compact_shape_dynamic(mode=1, stride_order=order)
        .mark_compact_shape_dynamic(mode=2, stride_order=order)
    )


def _mark_state_dynamic(array):
    order = (0, 1, 2, 3)
    return (
        from_dlpack(array, assumed_align=16)
        .mark_compact_shape_dynamic(mode=0, stride_order=order)
        .mark_compact_shape_dynamic(mode=1, stride_order=order)
    )


def _mark_1d_dynamic(array, assumed_align=16):
    return from_dlpack(array, assumed_align=assumed_align).mark_compact_shape_dynamic(
        mode=0, stride_order=(0,)
    )


def _to_cute_tensors(placeholders):
    return {
        "q": _mark_4d_dynamic(placeholders["q"]),
        "k": _mark_4d_dynamic(placeholders["k"]),
        "v": _mark_4d_dynamic(placeholders["v"]),
        "a": _mark_3d_dynamic(placeholders["a"]),
        "b": _mark_3d_dynamic(placeholders["b"]),
        "A_log": _mark_1d_dynamic(placeholders["A_log"]),
        "dt_bias": _mark_1d_dynamic(placeholders["dt_bias"]),
        "h0_in": _mark_state_dynamic(placeholders["h0_in"]),
        "h0_out": _mark_state_dynamic(placeholders["h0_out"]),
        "state_indices": _mark_1d_dynamic(placeholders["state_indices"]),
        "context_lengths": _mark_1d_dynamic(placeholders["context_lengths"]),
        "o": _mark_4d_dynamic(placeholders["o"]),
        "tensormap_scratch": _mark_1d_dynamic(
            placeholders["tensormap_scratch"], assumed_align=128
        ),
    }


def _make_aot_cute_tensors(n, t, hqk, hv):
    """Storage-free trace tensors mirroring _make_placeholders/_to_cute_tensors."""

    def _compact(dtype, shape, assumed_align=16):
        # Default stride order: C-contiguous, matching the cupy placeholders.
        return aot_placeholders.make_compact_tensor(
            dtype, shape, assumed_align=assumed_align,
        )

    def _mark_4d(dtype, shape):
        order = (0, 1, 2, 3)
        return (
            _compact(dtype, shape)
            .mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(mode=0, stride_order=order)
            .mark_compact_shape_dynamic(mode=1, stride_order=order)
            .mark_compact_shape_dynamic(mode=2, stride_order=order)
        )

    def _mark_3d(dtype, shape):
        order = (0, 1, 2)
        return (
            _compact(dtype, shape)
            .mark_layout_dynamic(leading_dim=2)
            .mark_compact_shape_dynamic(mode=0, stride_order=order)
            .mark_compact_shape_dynamic(mode=1, stride_order=order)
            .mark_compact_shape_dynamic(mode=2, stride_order=order)
        )

    def _mark_state(dtype, shape):
        order = (0, 1, 2, 3)
        return (
            _compact(dtype, shape)
            .mark_compact_shape_dynamic(mode=0, stride_order=order)
            .mark_compact_shape_dynamic(mode=1, stride_order=order)
        )

    def _mark_1d(dtype, shape, assumed_align=16):
        return _compact(dtype, shape, assumed_align=assumed_align) \
            .mark_compact_shape_dynamic(mode=0, stride_order=(0,))

    d = AOT_HEAD_DIM
    return {
        "q": _mark_4d(cutlass.Float16, (n, t, hqk, d)),
        "k": _mark_4d(cutlass.Float16, (n, t, hqk, d)),
        "v": _mark_4d(cutlass.Float16, (n, t, hv, d)),
        "a": _mark_3d(cutlass.Float16, (n, t, hv)),
        "b": _mark_3d(cutlass.Float16, (n, t, hv)),
        "A_log": _mark_1d(cutlass.Float32, (hv,)),
        "dt_bias": _mark_1d(cutlass.Float16, (hv,)),
        "h0_in": _mark_state(cutlass.Float32, (n, hv, d, d)),
        "h0_out": _mark_state(cutlass.Float32, (n, hv, d, d)),
        "state_indices": _mark_1d(cutlass.Int32, (n,)),
        "context_lengths": _mark_1d(cutlass.Int32, (n,)),
        "o": _mark_4d(cutlass.Float16, (n, t, hv, d)),
        "tensormap_scratch": _mark_1d(
            cutlass.Uint8, (AOT_TENSORMAP_BYTES,), assumed_align=128
        ),
    }


def compile_gdn_prefill_blackwell_geforce(gpu_arch="", export_only=False):
    if export_only:
        tensors = _make_aot_cute_tensors(
            AOT_PLACEHOLDER_N,
            AOT_PLACEHOLDER_T,
            AOT_PLACEHOLDER_HQK,
            AOT_PLACEHOLDER_HV,
        )
        stream = aot_placeholders.make_stream()
    else:
        placeholders = _make_placeholders(
            AOT_PLACEHOLDER_N,
            AOT_PLACEHOLDER_T,
            AOT_PLACEHOLDER_HQK,
            AOT_PLACEHOLDER_HV,
        )
        tensors = _to_cute_tensors(placeholders)
        stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    # Only the export path may pin a foreign target arch in the compile
    # options; a native JIT run must compile for the local GPU.
    options = aot_placeholders.compile_options(
        ("--gpu-arch " + gpu_arch) if gpu_arch else ""
    ) if export_only else None
    args = (
        tensors["q"], tensors["k"], tensors["v"],
        tensors["a"], tensors["b"],
        tensors["A_log"], tensors["dt_bias"],
        tensors["h0_in"], tensors["h0_out"],
        tensors["state_indices"], cutlass.Int32(1),
        tensors["context_lengths"], tensors["o"],
        tensors["tensormap_scratch"], cutlass.Int32(1), stream,
    )
    return cute.compile(
        run_gdn_prefill_blackwell_geforce,
        *args,
        **({"options": options} if options else {}),
    )


def export_gdn_prefill_blackwell_geforce(
    output_dir, file_name, function_prefix, gpu_arch=""
):
    print(
        "[gdn_prefill_blackwell_geforce] AOT compile gpu_arch=%r"
        % (gpu_arch or "auto")
    )
    start = time.time()
    compiled = compile_gdn_prefill_blackwell_geforce(
        gpu_arch=gpu_arch, export_only=True
    )
    print(
        "[gdn_prefill_blackwell_geforce] Compilation time: %.4fs"
        % (time.time() - start)
    )
    os.makedirs(output_dir, exist_ok=True)
    compiled.export_to_c(
        file_path=output_dir,
        file_name=file_name,
        function_prefix=function_prefix,
    )
    return compiled


def run_accuracy_test(n, h, hv, k, v, seq_len, tolerance, gpu_arch=""):
    if k != AOT_HEAD_DIM or v != AOT_HEAD_DIM:
        raise ValueError("Blackwell GeForce prefill requires k=v=128")
    if n <= 0 or seq_len <= 1 or h <= 0 or hv <= 0 or hv % h != 0:
        raise ValueError("Require n>0, seq_len>1, h>0, hv>0, and hv % h == 0")

    from gdn_prefill import _run_numpy_prefill_reference

    rng = np.random.default_rng(2026)
    q_host = (rng.standard_normal((n, seq_len, h, k), dtype=np.float32) * 0.1)
    k_host = (rng.standard_normal((n, seq_len, h, k), dtype=np.float32) * 0.1)
    v_host = (rng.standard_normal((n, seq_len, hv, v), dtype=np.float32) * 0.1)
    a_host = (rng.standard_normal((n, seq_len, hv), dtype=np.float32) * 0.1)
    b_host = (rng.standard_normal((n, seq_len, hv), dtype=np.float32) * 0.1)
    A_log_host = rng.standard_normal(hv, dtype=np.float32) * 0.1
    dt_bias_host = rng.standard_normal(hv, dtype=np.float32) * 0.1
    state_pool_rows = n + 2
    resident_state_host = (
        rng.standard_normal((state_pool_rows, hv, k, v), dtype=np.float32) * 0.01
    )
    context_lengths_host = np.maximum(
        1, seq_len - np.arange(n, dtype=np.int32)
    )

    q = cp.asarray(q_host, dtype=cp.float16)
    k_tensor = cp.asarray(k_host, dtype=cp.float16)
    v_tensor = cp.asarray(v_host, dtype=cp.float16)
    a = cp.asarray(a_host, dtype=cp.float16)
    b = cp.asarray(b_host, dtype=cp.float16)
    A_log = cp.asarray(A_log_host, dtype=cp.float32)
    dt_bias = cp.asarray(dt_bias_host, dtype=cp.float16)
    context_lengths = cp.asarray(context_lengths_host)

    # Match CuteDslGDNRunner: normalization is external to the Blackwell GeForce AOT kernel.
    q_fp32 = q.astype(cp.float32)
    k_fp32 = k_tensor.astype(cp.float32)
    q[...] = (
        q_fp32 / cp.sqrt(cp.sum(q_fp32 * q_fp32, axis=-1, keepdims=True) + 1e-6)
    ).astype(cp.float16)
    k_tensor[...] = (
        k_fp32 / cp.sqrt(cp.sum(k_fp32 * k_fp32, axis=-1, keepdims=True) + 1e-6)
    ).astype(cp.float16)

    multiprocessor_count = cp.cuda.Device().attributes["MultiProcessorCount"]
    tensormap_scratch = cp.zeros(
        (multiprocessor_count * AOT_TENSORMAP_BYTES,), dtype=cp.uint8
    )
    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)

    print(
        "[gdn_prefill_blackwell_geforce] JIT compile gpu_arch=%r"
        % (gpu_arch or "auto")
    )
    start = time.time()
    compiled = compile_gdn_prefill_blackwell_geforce(gpu_arch=gpu_arch)
    print(
        "[gdn_prefill_blackwell_geforce] Compilation time: %.4fs"
        % (time.time() - start)
    )
    q_ref = cp.asnumpy(q).astype(np.float32)
    k_ref = cp.asnumpy(k_tensor).astype(np.float32)
    v_ref = cp.asnumpy(v_tensor).astype(np.float32)
    a_ref = cp.asnumpy(a).astype(np.float32)
    b_ref = cp.asnumpy(b).astype(np.float32)
    dt_bias_ref = cp.asnumpy(dt_bias).astype(np.float32)
    indexed_identity = np.arange(n, dtype=np.int32)
    indexed_sparse = np.arange(n, 0, -1, dtype=np.int32)
    invalid_indices = np.where(
        np.arange(n, dtype=np.int32) % 2 == 0, -1, state_pool_rows
    ).astype(np.int32)
    mixed_indices = indexed_sparse.copy()
    if n > 1:
        mixed_indices[1] = -1
    if n > 2:
        mixed_indices[2] = state_pool_rows
    cases = (
        ("identity", indexed_identity, True),
        ("permuted_sparse", indexed_sparse, True),
        ("invalid", invalid_indices, True),
        ("mixed_valid_invalid", mixed_indices, True),
        ("identity_flag", invalid_indices, False),
    )
    for case_name, state_indices_host, use_state_indices in cases:
        resident_state = cp.asarray(resident_state_host, dtype=cp.float32)
        output = cp.empty((n, seq_len, hv, v), dtype=cp.float16)
        state_indices = cp.asarray(state_indices_host)
        arrays = {
            "q": q,
            "k": k_tensor,
            "v": v_tensor,
            "a": a,
            "b": b,
            "A_log": A_log,
            "dt_bias": dt_bias,
            "h0_in": resident_state,
            "h0_out": resident_state,
            "state_indices": state_indices,
            "context_lengths": context_lengths,
            "o": output,
            "tensormap_scratch": tensormap_scratch,
        }
        tensors = _to_cute_tensors(arrays)
        compiled(
            tensors["q"], tensors["k"], tensors["v"],
            tensors["a"], tensors["b"],
            tensors["A_log"], tensors["dt_bias"],
            tensors["h0_in"], tensors["h0_out"],
            tensors["state_indices"], cutlass.Int32(use_state_indices),
            tensors["context_lengths"], tensors["o"],
            tensors["tensormap_scratch"], cutlass.Int32(0), stream,
        )
        cp.cuda.get_current_stream().synchronize()

        active_state = np.zeros((n, hv, k, v), dtype=np.float32)
        expected_state = resident_state_host.copy()
        for row in range(n):
            state_slot = int(state_indices_host[row]) if use_state_indices else row
            if 0 <= state_slot < state_pool_rows:
                active_state[row] = resident_state_host[state_slot]
        output_ref, state_ref = _run_numpy_prefill_reference(
            q_ref, k_ref, v_ref, a_ref, b_ref,
            A_log_host, dt_bias_ref, active_state,
            n, h, hv, k, v, seq_len, scale=k ** -0.5,
            context_lengths_np=context_lengths_host,
            use_qk_l2norm=False,
        )
        for row in range(n):
            state_slot = int(state_indices_host[row]) if use_state_indices else row
            if 0 <= state_slot < state_pool_rows:
                expected_state[state_slot] = state_ref[row]
        output_actual = cp.asnumpy(output).astype(np.float32)
        state_actual = cp.asnumpy(resident_state).astype(np.float32)
        output_error = float(np.max(np.abs(output_actual - output_ref)))
        state_error = float(np.max(np.abs(state_actual - expected_state)))
        print(
            "[gdn_prefill_blackwell_geforce] %s output max abs error: %.8f; "
            "final state max abs error: %.8f; tolerance: %.1e"
            % (case_name, output_error, state_error, tolerance)
        )
        np.testing.assert_allclose(
            output_actual, output_ref, atol=tolerance, rtol=tolerance
        )
        np.testing.assert_allclose(
            state_actual, expected_state, atol=tolerance, rtol=tolerance
        )
    print("[gdn_prefill_blackwell_geforce] Standalone accuracy check PASSED")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Blackwell GeForce CuTe DSL GDN prefill"
    )
    parser.add_argument("--export_only", action="store_true")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--file_name", default="gdn_prefill_blackwell_geforce")
    parser.add_argument("--function_prefix", default="gdn_prefill_blackwell_geforce")
    parser.add_argument("--gpu_arch", default="")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--h", type=int, default=16)
    parser.add_argument("--hv", type=int, default=32)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--v", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    return parser.parse_known_args(args=argv)[0]


def main():
    args = _parse_args(_saved_argv)
    if args.export_only:
        export_gdn_prefill_blackwell_geforce(
            output_dir=args.output_dir,
            file_name=args.file_name,
            function_prefix=args.function_prefix,
            gpu_arch=args.gpu_arch,
        )
    else:
        run_accuracy_test(
            args.n, args.h, args.hv, args.k, args.v, args.seq_len,
            args.tolerance, gpu_arch=args.gpu_arch,
        )


if __name__ == "__main__":
    main()
