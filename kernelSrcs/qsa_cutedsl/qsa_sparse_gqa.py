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

# Origin: Structural fork of kernelSrcs/fmha_v2_cutedsl/fmha.py (FMHA-v2
# Ampere-floor forward kernel), which is itself adapted from
# xlite-dev/ffpa-attn (Apache-2.0).  The MMA/softmax core copies that file;
# the sparse index-gather traversal is this variant's own.

import argparse
import math
import os
import sys
import time
from types import SimpleNamespace
from typing import Callable, Type

_parsed_args = None
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
import numpy as np
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cutedsl_utils import aot_placeholders  # isort: skip


"""QSA sparse-GQA prefill and split-K decode kernels (CuTe DSL), AOT-export build.

Qwen Sparse Attention (Qwen3.8-Flash-Next): every query token carries an
int32 list of selected KV token indices produced by the QSA indexer
(top-k blocks of 4 tokens, expanded, plus an always-attended tail;
``-1``-padded on the right, unsorted, distinct, all ``< t + 1``).  The
kernel attends ONLY to the listed tokens — there is no causal or
positional mask in-kernel; causality lives entirely in the index list.

Layout: Q / K / V / O are dense padded ``BSND = (batch, seq, num_head,
head_dim)``, fp16 or bf16.  ``indices`` is ``(B, S, topk)`` Int32 and
``context_lengths`` is ``(B,)`` Int32 (live length per padded batch row).

Per-CTA tile: grid ``(S, H_kv, B)`` — one CTA per (query token, KV head).
The M tile is the token's GQA head group (``group = H_q / H_kv`` rows,
zero-padded to ``m_block_size``); the index list is walked in
``n_block_size`` chunks with a per-row cp.async gather (one warp-wave per
K/V row).  FA2-style online softmax (exp2, scale folded at exponent time —
i.e. applied to the FP32 scores, never pre-scaled into the fp16/bf16 Q).
``QSASparseGQADecode`` is the single-token split-K variant over the paged
KV pool; its class docstring documents the decode layout and grid.

Guarantees:
  * ``-1`` entries: K/V smem rows zero-filled AND scores forced to -inf
    (NaN hardening — masked columns stay inert even if memory holds NaN).
  * padding query rows (``t >= context_lengths[b]``): zero chunks run and
    the epilogue stores exact zeros.
  * all-invalid rows: ``row_sum == 0`` guard in normalize keeps zeros.
  * head-padding rows (``row >= group``): never loaded, never stored.

Variant axes baked at compile time: ``head_dim`` and the
``(Br, Bc, threads)`` tuning.  Runtime-dynamic: batch, seq, H_q, H_kv
(GQA group size), topk, strides, softmax scale.
"""

QSA_DEFAULT_M_BLOCK = 16
QSA_DEFAULT_N_BLOCK = 16
QSA_DEFAULT_THREADS = 32
# Software-pipeline depth: number of (sK, sV) smem stages kept resident.  The
# single warp is gather-latency bound (No-Eligible ~66%); a DEPTH-stage
# cp.async pipeline issues DEPTH-1 chunks' gathers ahead so the warp always has
# outstanding memory ops to overlap with compute.  DEPTH costs DEPTH*(sK+sV)
# smem; occupancy is not the binding resource here.  DEPTH=2 (prefetch one
# chunk of BOTH K and V ahead) measured fastest on Thor/SM110: DEPTH>=3 pushes
# smem past ~48KB, stealing the L1 cache the scattered gather relies on
# (97% L2 hit) and dropping occupancy, which outweighs the extra prefetch.
QSA_DEFAULT_PIPE_DEPTH = 2


class QSASparseGQAPrefill:

    def __init__(
        self,
        head_dim: int,
        m_block_size: int = QSA_DEFAULT_M_BLOCK,
        n_block_size: int = QSA_DEFAULT_N_BLOCK,
        num_threads: int = QSA_DEFAULT_THREADS,
        pipe_depth: int = QSA_DEFAULT_PIPE_DEPTH,
    ):
        """Initialize the QSA sparse-GQA prefill kernel.

        ``head_dim`` must be a multiple of 8 (16-byte alignment of the
        contiguous mode).  ``m_block_size`` bounds the GQA group size
        (``H_q / H_kv <= m_block_size``); the production shape is 12 -> 16.
        ``pipe_depth`` is the number of gathered (K, V) chunk stages kept in
        smem for the cp.async software pipeline (>= 2 to overlap gather with
        compute; DEPTH-1 chunks are prefetched ahead).
        """
        self._head_dim = head_dim
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        self._head_dim_padded = (head_dim + 31) // 32 * 32
        self._num_threads = num_threads
        self._pipe_depth = pipe_depth
        # cp.async row gather: one warp-wave per K/V row, 128-bit per lane.
        self._async_load_cache_mode = cpasync.LoadCacheMode.GLOBAL

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    @staticmethod
    def can_implement(
        dtype, head_dim, m_block_size, n_block_size, num_threads,
        pipe_depth=QSA_DEFAULT_PIPE_DEPTH,
    ) -> bool:
        """Check whether the (dtype, tile, threads) combo is implementable.

        Each warp owns 16 query rows in the MMA layout, so the warp count
        must tile ``m_block_size`` exactly.  The row-gather copy uses one
        full warp per K/V row (32 lanes x 8 elems = 256), so ``head_dim``
        must be a multiple of ``32 * 8 = 256 / (16 / dtype_bytes)`` — for
        the 16-bit dtypes this means ``head_dim % 256 == 0`` when the whole
        row is one wave; smaller head dims would need a different value
        layout, deferred until a variant needs it.
        """
        if dtype != cutlass.Float16 and dtype != cutlass.BFloat16:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if (m_block_size * 2) % num_threads != 0:
            return False
        # One warp-wave covers num_threads lanes * 8 halves; the per-row
        # gather and the row-wise Q/O copies require exactly one wave per
        # 16-bit row, so the CTA width is pinned to the head dim.
        if head_dim != num_threads * 8:
            return False
        if pipe_depth < 2:
            return False
        # The coalesced index fetch holds a whole chunk's indices in one
        # per-lane register (32 lanes) and distributes them with warp shuffle.
        if n_block_size > 32:
            return False

        head_dim_padded = (head_dim + 31) // 32 * 32
        # sQ (1 copy) + pipe_depth stages of (sK + sV).
        smem_usage = (
            m_block_size * head_dim_padded
            + pipe_depth * n_block_size * head_dim_padded * 2
        ) * 2
        smem_capacity = utils.get_smem_capacity_in_bytes("sm_80")
        if smem_usage > smem_capacity:
            return False

        return True

    @cute.jit
    def __call__(
        self,
        q_tensor: cute.Tensor,
        k_tensor: cute.Tensor,
        v_tensor: cute.Tensor,
        o_tensor: cute.Tensor,
        indices: cute.Tensor,
        context_lengths: cute.Tensor,
        attention_scale: cutlass.Float32,
        sm_count: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Configure SMEM / tiled-copy / tiled-mma and launch the kernel.

        ``q_tensor``/``o_tensor`` are ``(B, S, H_q, D)``; ``k_tensor``/
        ``v_tensor`` are ``(B, S, H_kv, D)``; all share dtype (fp16/bf16)
        with contiguous ``D``-innermost packing.  ``indices`` is
        ``(B, S, topk)`` Int32, ``context_lengths`` is ``(B,)`` Int32.
        ``H_q % H_kv == 0`` and ``H_q / H_kv <= m_block_size`` are caller
        contracts.  ``sm_count`` is ABI symmetry with the FMHA-v2 wrappers
        and is unused.
        """
        if cutlass.const_expr(
            not (
                q_tensor.element_type
                == k_tensor.element_type
                == v_tensor.element_type
                == o_tensor.element_type
            )
        ):
            raise TypeError("Q, K, V, and O must have the same data type")
        if cutlass.const_expr(
            not (
                q_tensor.element_type == cutlass.Float16
                or q_tensor.element_type == cutlass.BFloat16
            )
        ):
            raise TypeError("Only Float16 or BFloat16 is supported")
        self._dtype: Type[cutlass.Numeric] = q_tensor.element_type

        # ///////////////////////////////////////////////////////////////////
        # Shared memory layouts (same swizzled atoms as FMHA-v2).
        # ///////////////////////////////////////////////////////////////////
        smem_k_block_size = (
            64
            if self._head_dim_padded % 64 == 0
            else 32
            if self._head_dim_padded % 32 == 0
            else 16
        )
        swizzle_bits = (
            3 if smem_k_block_size == 64 else 2 if smem_k_block_size == 32 else 1
        )
        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        sQ_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self._m_block_size, self._head_dim_padded),
            (0, 1),
        )
        # Multi-stage K/V smem: append a pipe_depth stage mode (stride =
        # per-stage cosize) to the swizzled (n_block, head_dim) atom, exactly
        # like the gemm-ampere num_stages layout.
        sKV_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self._n_block_size, self._head_dim_padded, self._pipe_depth),
            (0, 1, 2),
        )
        sV_layout = sKV_layout
        sO_layout = sQ_layout

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sV_layout)], 1024
            ]
            # Per-stage staging of each chunk's token indices: written by the
            # gather, read by the sparse score mask.
            sIdx: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int32, self._pipe_depth * self._n_block_size
                ],
                16,
            ]

        # ///////////////////////////////////////////////////////////////////
        # GMEM copies.  The row gather uses a (1, 32) x (1, 8) tiled copy:
        # one warp-wave moves one whole 256-element K/V/Q row (32 lanes x
        # 128 bits), so a gathered row is a single fully-coalesced wave.
        # ///////////////////////////////////////////////////////////////////
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self._dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=self._async_load_cache_mode),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        row_thr_layout = cute.make_layout(
            (1, self._num_threads), stride=(self._num_threads, 1)
        )
        row_val_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_row = cute.make_tiled_copy_tv(
            atom_async_copy, row_thr_layout, row_val_layout
        )
        gmem_tiled_copy_row_O = cute.make_tiled_copy_tv(
            atom_universal_copy, row_thr_layout, row_val_layout
        )

        # ///////////////////////////////////////////////////////////////////
        # Tiled mma (same m16n8k16 arrangement as FMHA-v2).
        # ///////////////////////////////////////////////////////////////////
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
        )

        # grid: (query token, KV head, batch) — adjacent CTAs share L2 for
        # overlapping gathered rows of neighbouring tokens.
        grid_dim = (
            cute.size(q_tensor.shape[1]),
            cute.size(k_tensor.shape[2]),
            cute.size(q_tensor.shape[0]),
        )
        LOG2_E = 1.4426950408889634074
        softmax_scale_log2 = attention_scale * LOG2_E
        _ = sm_count
        self.kernel(
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            indices,
            context_lengths,
            softmax_scale_log2,
            sQ_layout,
            sKV_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_row,
            gmem_tiled_copy_row_O,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self._num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mIdx: cute.Tensor,
        mCtxLen: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_row: cute.TiledCopy,
        gmem_tiled_copy_row_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        """Sparse FA2 body: prologue cp.async Q rows + gathered K chunk 0,
        chunk loop of ``compute_one_chunk`` (BMM1 -> sparse mask + online
        softmax -> BMM2), epilogue rmem -> smem (aliased over sQ) -> gmem
        for the ``row < group`` rows only.
        """
        tidx, _, _ = cute.arch.thread_idx()
        token, kv_head, batch = cute.arch.block_idx()

        num_q_heads = mQ.shape[2]
        num_kv_heads = mK.shape[2]
        group_size = num_q_heads // num_kv_heads
        head_base = kv_head * group_size

        ctx_len = mCtxLen[batch]
        topk = mIdx.shape[2]

        # Chunk bound: never more useful entries than (token + 1) causal
        # candidates; correctness never depends on this (entries past the
        # valid prefix are -1 and masked), only wasted work does.
        row_limit = cutlass.min(topk, token + 1)
        n_chunks = cute.ceil_div(row_limit, self._n_block_size)
        # Padding query rows: run zero chunks; acc_O keeps its zero fill and
        # the row_sum == 0 guard in normalize_softmax stores exact zeros.
        n_chunks = 0 if token >= ctx_len else n_chunks

        # ///////////////////////////////////////////////////////////////////
        # Strides recomputed from shapes (not read from the layout) so the
        # divby facts survive dynamic-shape marking — same trick as the
        # FMHA-v2 packed-varlen path.
        # ///////////////////////////////////////////////////////////////////
        q_seq_stride = num_q_heads * self._head_dim
        q_seq_stride = cute.assume(q_seq_stride, divby=8)
        q_batch_stride = mQ.shape[1] * q_seq_stride
        kv_seq_stride = num_kv_heads * self._head_dim
        kv_seq_stride = cute.assume(kv_seq_stride, divby=8)
        kv_batch_stride = mK.shape[1] * kv_seq_stride

        q_row_base = (
            batch * q_batch_stride
            + token * q_seq_stride
            + head_base * self._head_dim
        )
        kv_row_base = batch * kv_batch_stride + kv_head * self._head_dim

        # ///////////////////////////////////////////////////////////////////
        # Shared memory.
        # ///////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sKV_layout)
        sV = storage.sV.get_tensor(sV_layout)
        sIdx = storage.sIdx.get_tensor(
            cute.make_layout(
                (self._pipe_depth, self._n_block_size),
                stride=(self._n_block_size, 1),
            )
        )

        gmem_thr_copy_row = gmem_tiled_copy_row.get_slice(tidx)
        gmem_thr_copy_row_O = gmem_tiled_copy_row_O.get_slice(tidx)
        # Row-tiled destination partitions: mode 1 walks the tile rows, mode 3
        # walks the pipeline stage.
        tQsQ = gmem_thr_copy_row.partition_D(sQ)
        tKsK = gmem_thr_copy_row.partition_D(sK)
        tVsV = gmem_thr_copy_row.partition_D(sV)

        # ///////////////////////////////////////////////////////////////////
        # MMA partitions and accumulators.  Stage-0 slices only fix the
        # (stage-independent) fragment shapes; the actual stage is selected
        # per chunk inside compute_one_chunk.
        # ///////////////////////////////////////////////////////////////////
        thr_mma = tiled_mma.get_slice(tidx)
        sK0 = sK[None, None, 0]
        sVt0 = cute.composition(
            sV[None, None, 0],
            cute.make_layout(
                (self._head_dim_padded, self._n_block_size),
                stride=(self._n_block_size, 1),
            ),
        )
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK0))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt0))
        acc_shape_O = thr_mma.partition_shape_C(
            (self._m_block_size, self._head_dim_padded)
        )
        acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)

        smem_copy_atom_Q = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )
        smem_copy_atom_K = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self._dtype,
        )
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)

        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
        tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
        tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

        # ///////////////////////////////////////////////////////////////////
        # Online-softmax state.
        # ///////////////////////////////////////////////////////////////////
        row_max = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_sum = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        mma_params = SimpleNamespace(
            thr_mma=thr_mma,
            tiled_mma=tiled_mma,
            tSrQ=tSrQ,
            tSrK=tSrK,
            tOrVt=tOrVt,
            acc_O=acc_O,
        )
        smem_copy_params = SimpleNamespace(
            smem_tiled_copy_Q=smem_tiled_copy_Q,
            smem_tiled_copy_K=smem_tiled_copy_K,
            smem_tiled_copy_V=smem_tiled_copy_V,
            smem_thr_copy_K=smem_thr_copy_K,
            smem_thr_copy_V=smem_thr_copy_V,
            sK=sK,
            sV=sV,
            tSsQ=tSsQ,
            tSrQ_copy_view=tSrQ_copy_view,
            tSrK_copy_view=tSrK_copy_view,
            tOrVt_copy_view=tOrVt_copy_view,
        )
        softmax_params = SimpleNamespace(
            row_max=row_max,
            row_sum=row_sum,
            softmax_scale_log2=softmax_scale_log2,
        )

        # ///////////////////////////////////////////////////////////////////
        # Prologue: Q rows (row < group only; head-padding rows zero-filled and
        # never read from gmem) + gather the first ``pipe_depth`` K/V chunk
        # stages.  Padding tokens (n_chunks == 0) issue NO cp.async at all —
        # the mainloop (the only waiter) never runs for them, so an in-flight Q
        # load could otherwise clobber sQ after the epilogue reuses it as sO.
        # ///////////////////////////////////////////////////////////////////
        if n_chunks > 0:
            for row in cutlass.range_constexpr(self._m_block_size):
                if row < group_size:
                    gRow = self._gmem_row_view(mQ, q_row_base + row * self._head_dim)
                    tRow = gmem_thr_copy_row.partition_S(gRow)
                    tRowAligned = cute.make_tensor(tRow.iterator.align(16), tRow.layout)
                    cute.copy(
                        gmem_tiled_copy_row,
                        tRowAligned[None, 0, 0],
                        tQsQ[None, row, 0],
                    )
                else:
                    tQsQ[None, row, 0].fill(0)
        # Index prefetch runs ONE pipeline step ahead of the K/V gather: the
        # coalesced index load for chunk c is issued a full iteration before
        # the gather that consumes it, so its gmem round-trip hides under a
        # chunk of compute instead of sitting on the cp.async address path.
        idx_reg = cutlass.Int32(-1)
        if n_chunks > 0:
            idx_reg = self._load_chunk_indices(
                mIdx, batch, token, cutlass.Int32(0), topk
            )
        for s in cutlass.range_constexpr(self._pipe_depth):
            if s < n_chunks:
                self._gather_stage(
                    mK, mV, sIdx, tKsK, tVsV,
                    gmem_tiled_copy_row, gmem_thr_copy_row,
                    kv_row_base, kv_seq_stride,
                    idx_reg, cutlass.Int32(s),
                )
                idx_reg = self._load_chunk_indices(
                    mIdx, batch, token, cutlass.Int32(s + 1), topk
                )
            cute.arch.cp_async_commit_group()

        # ///////////////////////////////////////////////////////////////////
        # Software-pipelined mainloop: wait for chunk i (keeping pipe_depth-1
        # gathers in flight), compute it, then refill its stage with chunk
        # i+pipe_depth.  Stage of chunk i+pipe_depth == stage of chunk i.
        # ///////////////////////////////////////////////////////////////////
        for i in range(n_chunks):
            stage = i % self._pipe_depth
            cute.arch.cp_async_wait_group(self._pipe_depth - 1)
            self.cta_sync_barrier.arrive_and_wait()
            self.compute_one_chunk(
                mma_params, smem_copy_params, softmax_params, sIdx, stage,
            )
            next_chunk = i + self._pipe_depth
            if next_chunk < n_chunks:
                self._gather_stage(
                    mK, mV, sIdx, tKsK, tVsV,
                    gmem_tiled_copy_row, gmem_thr_copy_row,
                    kv_row_base, kv_seq_stride,
                    idx_reg, stage,
                )
                idx_reg = self._load_chunk_indices(
                    mIdx, batch, token, next_chunk + 1, topk
                )
            cute.arch.cp_async_commit_group()

        # ///////////////////////////////////////////////////////////////////
        # Epilogue: normalize, rmem -> smem (aliased over sQ) -> gmem for
        # the live group rows only.
        # ///////////////////////////////////////////////////////////////////
        self.normalize_softmax(acc_O, row_sum)
        rO = cute.make_fragment_like(acc_O, self._dtype)
        rO.store(acc_O.load().to(self._dtype))
        sO = cute.make_tensor(sQ.iterator, sO_layout)

        smem_copy_atom_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self._dtype
        )
        smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
        smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(
            smem_copy_atom_O,
            taccOrO,
            taccOsO,
        )
        self.cta_sync_barrier.arrive_and_wait()

        o_seq_stride = mO.shape[2] * self._head_dim
        o_seq_stride = cute.assume(o_seq_stride, divby=8)
        o_batch_stride = mO.shape[1] * o_seq_stride
        o_row_base = (
            batch * o_batch_stride
            + token * o_seq_stride
            + head_base * self._head_dim
        )
        tOsO = gmem_thr_copy_row_O.partition_S(sO)
        for row in cutlass.range_constexpr(self._m_block_size):
            if row < group_size:
                gRow = self._gmem_row_view(mO, o_row_base + row * self._head_dim)
                tRowO = gmem_thr_copy_row_O.partition_D(gRow)
                tRowOAligned = cute.make_tensor(tRowO.iterator.align(16), tRowO.layout)
                cute.copy(
                    gmem_tiled_copy_row_O,
                    tOsO[None, row, 0],
                    tRowOAligned[None, 0, 0],
                )

    @cute.jit
    def _gmem_row_view(self, mT: cute.Tensor, element_offset: cutlass.Int32):
        """A ``(1, head_dim)`` gmem row view at a dynamic element offset.

        Every contributing offset term is a multiple of 8 elements (16
        bytes at 16-bit dtypes) — batch/seq strides carry the static
        ``head_dim`` factor and the head offset is ``row * head_dim`` —
        so the 128-bit cp.async alignment is provable.
        """
        raw_ptr = mT.iterator + element_offset
        aligned_ptr = cute.make_ptr(
            mT.element_type,
            raw_ptr.toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        return cute.make_tensor(
            aligned_ptr,
            cute.make_layout((1, self._head_dim), stride=(self._head_dim, 1)),
        )

    @cute.jit
    def _load_chunk_indices(
        self,
        mIdx: cute.Tensor,
        batch: cutlass.Int32,
        token: cutlass.Int32,
        chunk: cutlass.Int32,
        topk: cutlass.Int32,
    ) -> cutlass.Int32:
        """ONE coalesced index load for a whole chunk.

        Lane ``l < n_block_size`` loads ``indices[chunk * n_block + l]`` — a
        single 64B transaction — instead of ``n_block`` warp-uniform 4B scalar
        loads, each of which exposed a full gmem round-trip on the cp.async
        address path (long_scoreboard stalls, 12.5% bytes/sector signature).
        Out-of-range lanes/positions return ``-1`` (masked).  Callers issue
        this one pipeline step AHEAD of the consuming gather so the load's
        latency hides under a chunk of compute.
        """
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        pos_l = chunk * self._n_block_size + lane
        idx_l = cutlass.Int32(-1)
        if lane < self._n_block_size and pos_l < topk:
            idx_l = mIdx[batch, token, pos_l]
        return idx_l

    @cute.jit
    def _gather_stage(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sIdx: cute.Tensor,
        dstK,
        dstV,
        gmem_tiled_copy_row: cute.TiledCopy,
        gmem_thr_copy_row,
        kv_row_base: cutlass.Int32,
        kv_seq_stride: cutlass.Int32,
        idx_l: cutlass.Int32,
        stage: cutlass.Int32,
    ):
        """Gather one ``n_block_size``-row chunk's K AND V into pipeline
        ``stage`` and stage its token indices into ``sIdx[stage]``.

        One warp-wave per row: the whole warp cp.async-copies one 256-element
        row (32 lanes x 128 bits, fully coalesced).  Invalid (``-1``) rows are
        zero-filled so masked columns stay inert in BMM2 even against
        NaN-poisoned memory.  K and V share the same index, so a single pass
        issues both — the whole chunk becomes one cp.async group.

        ``idx_l`` is the chunk's per-lane index register prefetched by
        ``_load_chunk_indices``; the row loop broadcasts row ``w``'s index
        from lane ``w`` with ``shfl.sync.idx`` — zero memory traffic per row,
        and the broadcast value keeps the per-row branches warp-uniform.

        The lane-local ``sIdx[stage, lane]`` store replaces the old
        all-lanes-store-everything pattern; readers (the softmax mask) only
        run after the mainloop's CTA barrier for this chunk, which publishes
        the stores.
        """
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        if lane < self._n_block_size:
            sIdx[stage, lane] = idx_l
        for w in cutlass.range_constexpr(self._n_block_size):
            # Broadcast row w's index from lane w (warp-uniform result).
            idx = cutlass.Int32(cute.arch.shuffle_sync(idx_l, w))
            self._gather_row(
                mK, mV, dstK, dstV, gmem_tiled_copy_row,
                gmem_thr_copy_row, kv_row_base, kv_seq_stride,
                idx, w, stage,
            )

    @cute.jit
    def _gather_row(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        dstK,
        dstV,
        gmem_tiled_copy_row: cute.TiledCopy,
        gmem_thr_copy_row,
        kv_row_base: cutlass.Int32,
        kv_seq_stride: cutlass.Int32,
        idx: cutlass.Int32,
        w: cutlass.Constexpr,
        stage: cutlass.Int32,
    ):
        """cp.async one token's K and V rows into ``dstK/dstV[w, stage]``;
        invalid (``idx < 0``) rows are zero-filled."""
        if idx >= 0:
            row_off = kv_row_base + idx * kv_seq_stride
            gRowK = self._gmem_row_view(mK, row_off)
            tRowK = gmem_thr_copy_row.partition_S(gRowK)
            tRowKAligned = cute.make_tensor(tRowK.iterator.align(16), tRowK.layout)
            cute.copy(
                gmem_tiled_copy_row,
                tRowKAligned[None, 0, 0],
                dstK[None, w, 0, stage],
            )
            gRowV = self._gmem_row_view(mV, row_off)
            tRowV = gmem_thr_copy_row.partition_S(gRowV)
            tRowVAligned = cute.make_tensor(tRowV.iterator.align(16), tRowV.layout)
            cute.copy(
                gmem_tiled_copy_row,
                tRowVAligned[None, 0, 0],
                dstV[None, w, 0, stage],
            )
        else:
            dstK[None, w, 0, stage].fill(0)
            dstV[None, w, 0, stage].fill(0)

    @cute.jit
    def compute_one_chunk(
        self,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        sIdx: cute.Tensor,
        stage: cutlass.Int32,
    ):
        """One index chunk (K/V already gathered into ``stage`` by the
        pipeline): BMM1 (Q @ gathered-K^T) -> sparse mask + online softmax ->
        BMM2 (P @ gathered-V).  No gather/wait here — the mainloop owns the
        cp.async pipeline.
        """
        # Select this chunk's K/V smem stage and re-derive the ldmatrix source
        # partitions (cheap: same layout, stage-offset pointer).
        sK_s = smem_copy_params.sK[None, None, stage]
        sVt_s = cute.composition(
            smem_copy_params.sV[None, None, stage],
            cute.make_layout(
                (self._head_dim_padded, self._n_block_size),
                stride=(self._n_block_size, 1),
            ),
        )
        tSsK = smem_copy_params.smem_thr_copy_K.partition_S(sK_s)
        tOsVt = smem_copy_params.smem_thr_copy_V.partition_S(sVt_s)

        acc_shape_S = mma_params.thr_mma.partition_shape_C(
            (self._m_block_size, self._n_block_size)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
        acc_S.fill(0.0)

        # ///////////////////////////////////////////////////////////////////
        # S = Q @ K^T  (BMM1), ldmatrix double-buffered over the k mode.
        # ///////////////////////////////////////////////////////////////////
        cute.copy(
            smem_copy_params.smem_tiled_copy_Q,
            smem_copy_params.tSsQ[None, None, 0],
            smem_copy_params.tSrQ_copy_view[None, None, 0],
        )
        cute.copy(
            smem_copy_params.smem_tiled_copy_K,
            tSsK[None, None, 0],
            smem_copy_params.tSrK_copy_view[None, None, 0],
        )
        for k in cutlass.range_constexpr(cute.size(smem_copy_params.tSsQ.shape[2])):
            k_next = (k + 1) % cute.size(smem_copy_params.tSsQ.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_Q,
                smem_copy_params.tSsQ[None, None, k_next],
                smem_copy_params.tSrQ_copy_view[None, None, k_next],
            )
            cute.copy(
                smem_copy_params.smem_tiled_copy_K,
                tSsK[None, None, k_next],
                smem_copy_params.tSrK_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                acc_S,
                mma_params.tSrQ[None, None, k],
                mma_params.tSrK[None, None, k],
                acc_S,
            )

        # ///////////////////////////////////////////////////////////////////
        # Sparse mask + online softmax.
        # ///////////////////////////////////////////////////////////////////
        self.softmax_rescale_O(
            mma_params,
            softmax_params,
            sIdx,
            stage,
            acc_S,
        )

        rP = cute.make_fragment_like(acc_S, self._dtype)
        rP.store(acc_S.load().to(self._dtype))
        # ///////////////////////////////////////////////////////////////////
        # O += P @ V  (BMM2)
        # ///////////////////////////////////////////////////////////////////
        rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
        rP_mma_view = cute.make_layout(
            (
                (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                rP_layout_divided.shape[1],
                rP_layout_divided.shape[2][1],
            ),
            stride=(
                (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                rP_layout_divided.stride[1],
                rP_layout_divided.stride[2][1],
            ),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

        cute.copy(
            smem_copy_params.smem_tiled_copy_V,
            tOsVt[None, None, 0],
            smem_copy_params.tOrVt_copy_view[None, None, 0],
        )
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            k_next = (k + 1) % cute.size(tOrS.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_V,
                tOsVt[None, None, k_next],
                smem_copy_params.tOrVt_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                mma_params.acc_O,
                tOrS[None, None, k],
                mma_params.tOrVt[None, None, k],
                mma_params.acc_O,
            )

    @cute.jit
    def softmax_rescale_O(
        self,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        sIdx: cute.Tensor,
        stage: cutlass.Int32,
        acc_S: cute.Tensor,
    ):
        """Apply the sparse-index mask and online softmax to ``acc_S``.

        The only mask is index validity: staged ``sIdx[stage]`` entries < 0
        force the column to -inf for every row.  There is no positional
        masking — causality is baked into the index list by the indexer.  The
        scale is applied to the FP32 scores at exp2 time (row_max tracked on
        raw scores), never pre-folded into the low-precision Q.

        The general online-softmax rescale path is used for every chunk: on
        the first chunk ``row_max`` is still ``-inf`` and ``row_sum`` is 0, so
        the running-max correction ``exp2(-inf) = 0`` correctly leaves the
        zero-initialised accumulator untouched — no ``is_first`` peel needed.
        """
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        acc_O_mn = self._make_acc_tensor_mn_view(mma_params.acc_O)
        row_max_prev = cute.make_fragment_like(
            softmax_params.row_max, cutlass.Float32
        )
        cute.basic_copy(softmax_params.row_max, row_max_prev)

        mcS = cute.make_identity_tensor(
            (self._m_block_size, self._n_block_size)
        )
        tScS = mma_params.thr_mma.partition_C(mcS)
        tScS_mn = self._make_acc_tensor_mn_view(tScS)
        # Per-thread column validity, hoisted out of the row loop (each
        # thread sees the same columns for both of its accumulator rows).
        col_valid = cute.make_rmem_tensor(
            cute.make_layout(cute.size(tScS_mn.shape[1])), cutlass.Boolean
        )
        for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
            col_idx = tScS_mn[0, c][1]
            col_valid[c] = sIdx[stage, col_idx] >= 0

        for r in cutlass.range_constexpr(cute.size(softmax_params.row_max)):
            for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                if not col_valid[c]:
                    acc_S_mn[r, c] = -cutlass.Float32.inf

            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur_row = acc_S_row.reduce(
                cute.ReductionOp.MAX, -cutlass.Float32.inf, 0
            )
            row_max_cur_row = self._threadquad_reduce_max(row_max_cur_row)
            row_max_prev_row = row_max_prev[r]
            row_max_cur_row = cute.arch.fmax(row_max_prev_row, row_max_cur_row)
            # Keep -inf in the running max until a valid score arrives, but
            # use a finite max for the exponent arithmetic so fully-masked
            # rows stay at exp2(-inf) = 0 instead of NaN.
            row_max_safe_row = (
                0.0 if row_max_cur_row == -cutlass.Float32.inf else row_max_cur_row
            )

            acc_S_row_exp = cute.math.exp2(
                acc_S_row * softmax_params.softmax_scale_log2
                - row_max_safe_row * softmax_params.softmax_scale_log2,
                fastmath=True,
            )
            acc_S_row_sum = acc_S_row_exp.reduce(
                cute.ReductionOp.ADD, cutlass.Float32.zero, 0
            )
            prev_minus_cur_exp = cute.math.exp2(
                row_max_prev_row * softmax_params.softmax_scale_log2
                - row_max_safe_row * softmax_params.softmax_scale_log2,
                fastmath=True,
            )
            acc_S_row_sum = (
                acc_S_row_sum + softmax_params.row_sum[r] * prev_minus_cur_exp
            )
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * prev_minus_cur_exp
            softmax_params.row_max[r] = row_max_cur_row
            softmax_params.row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None] = acc_S_row_exp

    @cute.jit
    def normalize_softmax(
        self,
        acc_O: cute.Tensor,
        row_sum: cute.Tensor,
    ):
        """Final softmax normalisation with the zero-row guard.

        ``row_sum == 0`` (padding token rows and all-invalid index rows)
        keeps scale = 1 so the zero-initialised accumulator stores exact
        zeros — never NaN.
        """
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]

            scale = (
                1.0 if acc_O_mn_row_is_zero_or_nan else cute.arch.rcp_approx(row_sum[r])
            )

            acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale

    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        """Reinterpret a ``(MMA, MMA_M, MMA_N)`` accumulator as ``(M, N)``."""
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        acc_layout_mn = cute.make_layout(
            (
                (
                    acc_layout_col_major.shape[0][1],
                    acc_layout_col_major.shape[1],
                ),
                (
                    acc_layout_col_major.shape[0][0],
                    acc_layout_col_major.shape[2],
                ),
            ),
            stride=(
                (
                    acc_layout_col_major.stride[0][1],
                    acc_layout_col_major.stride[1],
                ),
                (
                    acc_layout_col_major.stride[0][0],
                    acc_layout_col_major.stride[2],
                ),
            ),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)

    def _threadquad_reduce(self, val: cutlass.Float32, op: Callable) -> cutlass.Float32:
        """Reduce across the four threads holding the same column of an MMA fragment."""
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31),
        )
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31),
        )
        return val

    def _threadquad_reduce_max(self, val: cutlass.Float32) -> cutlass.Float32:
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val: cutlass.Float32) -> cutlass.Float32:
        return self._threadquad_reduce(val, lambda x, y: x + y)


# ---------------------------------------------------------------------------
# PTX intrinsics for the decode split-K last-arriver merge.  Same
# llvm.inline_asm dsl_user_op idiom as kernelSrcs/nvfp4_fused_moe_cutedsl/
# fp4_common.py (atomic_add_global_i32 :1280, membar :245) — proven on this
# CuTe-DSL version and target family.
# ---------------------------------------------------------------------------


@dsl_user_op
def _gmem_addr_i64(tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None) -> Int64:
    """Address of ``tensor[offset]`` as Int64 (global-memory tensors only)."""
    elem_ptr = tensor.iterator + Int32(offset)
    return Int64(llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def _atomic_add_global_i32(addr: Int64, val: Int32, *, loc=None, ip=None) -> Int32:
    """Relaxed global int32 atomic add; returns the old value."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Int32(val).ir_value(loc=loc, ip=ip),
            ],
            "atom.global.add.s32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _st_global_release_i32(addr: Int64, val: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "st.global.release.gpu.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _threadfence_gl(*, loc=None, ip=None):
    """Device-scope memory fence (membar.gl) — the classic threadfence
    reduction ordering: producers fence before the counter atomic, the merge
    winner fences after it and before reading the partials."""
    llvm.inline_asm(
        None,
        [],
        "membar.gl;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# Decode CTA geometry.  The row wave stays one warp (32 lanes x 8 halves =
# head_dim 256), but the decode CTA holds several such warps: at B=1 the
# grid is MAX_SPLITS * H_kv = 16 CTAs on 20 SMs, so one warp per CTA left
# each SM with a single gather in flight (ncu: 1.1 warps/SM, 80%
# no-eligible).  4 independent warps per CTA multiply the in-flight
# gathers per SM; they only meet for the final merge.  One cp.async stage
# per warp keeps the CTA at 72 KB smem (2-3 CTAs/SM): the Thor sweep of
# threads x depth x n_block measured a second stage (136 KB, 1 CTA/SM) at
# 18.5/55/122 us vs 16.6/37/102 us for B=1/4/8, i.e. cross-warp overlap
# beats intra-warp prefetch once the CTA carries 4 warps.
QSA_DECODE_DEFAULT_THREADS = 128
QSA_DECODE_DEFAULT_PIPE_DEPTH = 1


class QSASparseGQADecode(QSASparseGQAPrefill):
    """Single-launch split-K decode variant of the QSA sparse-GQA kernel.

    Decode shape: one query token per sequence (``q [B, 1, H_q, D]``), K/V
    read from the Edge-LLM paged pool ``[2*numPages, 128, H_kv, poolHeadDim]``
    through the page table ``[B, 2, maxPagesPerSeq]`` (V page ids pre-offset
    by ``+numPages``; ``poolHeadDim >= head_dim`` — only columns
    ``[0, head_dim)`` of each row are read, the tail carries the QSA indexer
    state and is provably untouched because the row wave copies exactly
    ``head_dim`` elements).

    Split-K, KDA-style single launch: fixed grid ``(MAX_SPLITS, H_kv, B)``;
    every CTA independently derives the SAME active-split count from
    ``context_lengths`` (``n_active = clamp(ceil(n_tiles / 4), 1, MAX_SPLITS)``)
    and its tile slice.  Inside an active CTA, warp ``w`` owns every
    ``num_warps``-th chunk of the slice and runs the single-warp
    (16 x n_block) MMA pipeline over them with its own cp.async stages,
    index/page prefetch and online-softmax state — the warps never
    synchronize during the mainloop.  They meet once at the end: each warp
    stages its unnormalized fp32 accumulator and per-row (max, sum) in smem
    (aliasing its own, now idle, K and V stages), then all threads merge the
    warps row-wise with 128-bit accesses.  ``n_active == 1`` stores the
    normalized output directly from that merge; otherwise the merged split
    partial + stats go to the fp32 workspace, every CTA fences and bumps a
    per-(b, kv_head) counter (ALL ``MAX_SPLITS`` CTAs arrive, inactive ones
    skip compute, so the last-arriver test is against the static
    denominator), and the winner merges the split partials with the same
    row-wise mapping, stores O, and release-stores the counter back to 0
    (callers must ALSO zero counters before the first launch on a fresh
    workspace — workspace contents are not persistent).
    """

    def __init__(
        self,
        head_dim: int,
        m_block_size: int = QSA_DEFAULT_M_BLOCK,
        n_block_size: int = QSA_DEFAULT_N_BLOCK,
        num_threads: int = QSA_DECODE_DEFAULT_THREADS,
        pipe_depth: int = QSA_DECODE_DEFAULT_PIPE_DEPTH,
        max_splits: int = 8,
    ):
        super().__init__(head_dim, m_block_size, n_block_size, num_threads, pipe_depth)
        self._num_warps = num_threads // 32
        self._max_splits = max_splits

    @staticmethod
    def can_implement_decode(
        dtype, head_dim, m_block_size, n_block_size, num_threads,
        pipe_depth=QSA_DECODE_DEFAULT_PIPE_DEPTH, max_splits=8,
    ) -> bool:
        """Decode implementability (independent of the prefill check: here
        the CTA width is decoupled from the 32-lane row wave).

        One 32-lane wave per 16-bit row pins ``head_dim == 256``; one MMA M
        tile per warp pins ``m_block_size == 16``; ``n_block_size`` is a
        multiple of the MMA K (16) and at most 32 (a chunk's indices live
        one per lane); the warp count must tile the merge rows; and a warp's
        fp32 merge staging (``m_block x head_dim x 4 B``) must fit its own
        K + V stages (``2 x pipe_depth x n_block x head_dim x 2 B``).
        """
        if dtype != cutlass.Float16 and dtype != cutlass.BFloat16:
            return False
        if head_dim != 32 * 8:
            return False
        if m_block_size != 16:
            return False
        if num_threads < 32 or num_threads % 32 != 0:
            return False
        num_warps = num_threads // 32
        if m_block_size % num_warps != 0:
            return False
        if n_block_size % 16 != 0 or n_block_size > 32:
            return False
        if pipe_depth < 1:
            return False
        if pipe_depth * n_block_size < m_block_size:
            return False
        if max_splits < 1 or max_splits > 64:
            return False
        head_dim_padded = (head_dim + 31) // 32 * 32
        num_stages = num_warps * pipe_depth
        smem_usage = (
            m_block_size * head_dim_padded * 2  # sQ
            + 2 * num_stages * n_block_size * head_dim_padded * 2  # sK + sV
            + num_stages * n_block_size * 4  # sIdx
            + num_warps * 2 * m_block_size * 4  # sStats
            + 16  # sFlag
        )
        return smem_usage <= utils.get_smem_capacity_in_bytes("sm_100")

    @cute.jit
    def __call__(
        self,
        q_tensor: cute.Tensor,
        kv_pool: cute.Tensor,
        page_table: cute.Tensor,
        indices: cute.Tensor,
        context_lengths: cute.Tensor,
        o_tensor: cute.Tensor,
        partial_o: cute.Tensor,
        partial_stats: cute.Tensor,
        counters: cute.Tensor,
        attention_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        """Launch the decode kernel.

        No ``sm_count``: unlike the prefill wrapper (FMHA-v2 ABI symmetry),
        nothing here is FMHA-v2-shaped.
        ``q_tensor``/``o_tensor``: ``(B, 1, H_q, D)`` fp16/bf16.
        ``kv_pool``: ``(2*numPages, 128, H_kv, poolHeadDim)`` same dtype.
        ``page_table``: ``(B, 2, maxPagesPerSeq)`` Int32 (V ids pre-offset).
        ``indices``: ``(B, 1, topk)`` Int32 (-1 padded).
        ``context_lengths``: ``(B,)`` Int32 = TOTAL length incl. the new token.
        ``partial_o``: fp32 ``(B*H_kv*MAX_SPLITS, m_block, head_dim)``.
        ``partial_stats``: fp32 ``(B*H_kv*MAX_SPLITS, 2, m_block)``
        (mode 1: 0 = row max (raw scores), 1 = row sum).
        ``counters``: Int32 ``(B*H_kv,)`` — MUST be zero on entry.
        """
        if cutlass.const_expr(
            not (
                q_tensor.element_type
                == kv_pool.element_type
                == o_tensor.element_type
            )
        ):
            raise TypeError("Q, KV pool, and O must have the same data type")
        if cutlass.const_expr(
            not (
                q_tensor.element_type == cutlass.Float16
                or q_tensor.element_type == cutlass.BFloat16
            )
        ):
            raise TypeError("Only Float16 or BFloat16 is supported")
        self._dtype: Type[cutlass.Numeric] = q_tensor.element_type

        smem_k_block_size = (
            64
            if self._head_dim_padded % 64 == 0
            else 32
            if self._head_dim_padded % 32 == 0
            else 16
        )
        swizzle_bits = (
            3 if smem_k_block_size == 64 else 2 if smem_k_block_size == 32 else 1
        )
        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        sQ_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self._m_block_size, self._head_dim_padded),
            (0, 1),
        )
        # Stage mode = warp-major (warp w owns stages [w*depth, (w+1)*depth)).
        num_stages = self._num_warps * self._pipe_depth
        sKV_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self._n_block_size, self._head_dim_padded, num_stages),
            (0, 1, 2),
        )

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            # All K stages then all V stages in ONE range: a warp's fp32 merge
            # staging aliases its own K stages (low output columns) and V
            # stages (high columns) at a static element offset.
            sKV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, 2 * cute.cosize(sKV_layout)],
                1024,
            ]
            sIdx: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, num_stages * self._n_block_size],
                16,
            ]
            sStats: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, self._num_warps * 2 * self._m_block_size
                ],
                16,
            ]
            sFlag: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]

        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self._dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=self._async_load_cache_mode),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # One warp per row wave, regardless of the CTA width.
        row_thr_layout = cute.make_layout((1, 32), stride=(32, 1))
        row_val_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_row = cute.make_tiled_copy_tv(
            atom_async_copy, row_thr_layout, row_val_layout
        )
        gmem_tiled_copy_row_sync = cute.make_tiled_copy_tv(
            atom_universal_copy, row_thr_layout, row_val_layout
        )

        # Single-warp MMA: every warp owns the whole (16 x n_block) tile of
        # its own chunk.
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (1, 1, 1),
            permutation_mnk=(16, 16, 16),
        )

        # grid: (split, KV head, batch) — B is fixed per captured CUDA graph,
        # so the whole grid is capture-static; per-CTA work is bounded by
        # device tensors only.
        grid_dim = (
            self._max_splits,
            cute.size(kv_pool.shape[2]),
            cute.size(q_tensor.shape[0]),
        )
        LOG2_E = 1.4426950408889634074
        softmax_scale_log2 = attention_scale * LOG2_E
        self.kernel_decode(
            q_tensor,
            kv_pool,
            page_table,
            indices,
            context_lengths,
            o_tensor,
            partial_o,
            partial_stats,
            counters,
            softmax_scale_log2,
            sQ_layout,
            sKV_layout,
            gmem_tiled_copy_row,
            gmem_tiled_copy_row_sync,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self._num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel_decode(
        self,
        mQ: cute.Tensor,
        mPool: cute.Tensor,
        mPT: cute.Tensor,
        mIdx: cute.Tensor,
        mCtxLen: cute.Tensor,
        mO: cute.Tensor,
        mPartialO: cute.Tensor,
        mPartialStats: cute.Tensor,
        mCounters: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        gmem_tiled_copy_row: cute.TiledCopy,
        gmem_tiled_copy_row_sync: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        """Split-K decode body; see ``__call__`` for the launch contract."""
        tidx, _, _ = cute.arch.thread_idx()
        sid, kv_head, batch = cute.arch.block_idx()
        lane = tidx % 32
        warp_id = tidx // 32
        num_warps = self._num_warps
        depth = self._pipe_depth

        num_q_heads = mQ.shape[2]
        num_kv_heads = mPool.shape[2]
        group_size = num_q_heads // num_kv_heads
        head_base = kv_head * group_size

        ctx_len = mCtxLen[batch]
        topk = mIdx.shape[2]

        # Deterministic per-(batch) split policy — every CTA computes the
        # same values, so the last-arriver denominator can stay the static
        # MAX_SPLITS while inactive splits contribute zero tiles.
        row_limit = cutlass.min(topk, ctx_len)
        n_tiles = cute.ceil_div(row_limit, self._n_block_size)
        n_active = cutlass.min(
            cutlass.max(cute.ceil_div(n_tiles, 4), 1), self._max_splits
        )
        tile_lo = cutlass.min(n_tiles, (n_tiles * sid) // n_active)
        tile_hi = cutlass.min(n_tiles, (n_tiles * (sid + 1)) // n_active)
        my_tiles = tile_hi - tile_lo
        # Warp w owns chunks tile_lo + w, + num_warps, ... of the slice and
        # stages [w*depth, (w+1)*depth) of sK/sV/sIdx.
        n_my = (my_tiles - warp_id + num_warps - 1) // num_warps
        chunk_base = tile_lo + warp_id
        stage_base = warp_id * depth

        # Strides recomputed from shapes so divby facts survive dynamic
        # marking (prefill convention).
        q_seq_stride = num_q_heads * self._head_dim
        q_seq_stride = cute.assume(q_seq_stride, divby=8)
        q_batch_stride = mQ.shape[1] * q_seq_stride
        q_row_base = batch * q_batch_stride + head_base * self._head_dim

        pool_head_dim = cute.assume(mPool.shape[3], divby=8)
        pool_row_stride = num_kv_heads * pool_head_dim
        pool_page_stride = 128 * pool_row_stride
        pool_head_off = kv_head * pool_head_dim

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sKV.get_tensor(sKV_layout)
        kv_half_elems = cute.cosize(sKV_layout)
        sV = cute.make_tensor(
            cute.make_ptr(
                self._dtype,
                (sK.iterator + kv_half_elems).toint(),
                cute.AddressSpace.smem,
                assumed_align=1024,
            ),
            sKV_layout,
        )
        num_stages = num_warps * depth
        sIdx = storage.sIdx.get_tensor(
            cute.make_layout(
                (num_stages, self._n_block_size),
                stride=(self._n_block_size, 1),
            )
        )
        sStats = storage.sStats.get_tensor(
            cute.make_layout(
                (num_warps, 2, self._m_block_size),
                stride=(2 * self._m_block_size, self._m_block_size, 1),
            )
        )
        sFlag = storage.sFlag.get_tensor(cute.make_layout(4))

        gmem_thr_copy_row = gmem_tiled_copy_row.get_slice(lane)
        gmem_thr_copy_row_sync = gmem_tiled_copy_row_sync.get_slice(lane)
        tQsQ = gmem_thr_copy_row_sync.partition_D(sQ)
        tKsK = gmem_thr_copy_row.partition_D(sK)
        tVsV = gmem_thr_copy_row.partition_D(sV)

        thr_mma = tiled_mma.get_slice(lane)
        sK0 = sK[None, None, 0]
        sVt0 = cute.composition(
            sV[None, None, 0],
            cute.make_layout(
                (self._head_dim_padded, self._n_block_size),
                stride=(self._n_block_size, 1),
            ),
        )
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK0))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt0))
        acc_shape_O = thr_mma.partition_shape_C(
            (self._m_block_size, self._head_dim_padded)
        )
        acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)

        smem_copy_atom_Q = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
        )
        smem_copy_atom_K = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._dtype
        )
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)
        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(lane)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(lane)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(lane)
        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
        tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
        tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

        row_max = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_sum = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        mma_params = SimpleNamespace(
            thr_mma=thr_mma, tiled_mma=tiled_mma, tSrQ=tSrQ, tSrK=tSrK,
            tOrVt=tOrVt, acc_O=acc_O,
        )
        smem_copy_params = SimpleNamespace(
            smem_tiled_copy_Q=smem_tiled_copy_Q,
            smem_tiled_copy_K=smem_tiled_copy_K,
            smem_tiled_copy_V=smem_tiled_copy_V,
            smem_thr_copy_K=smem_thr_copy_K,
            smem_thr_copy_V=smem_thr_copy_V,
            sK=sK, sV=sV, tSsQ=tSsQ,
            tSrQ_copy_view=tSrQ_copy_view,
            tSrK_copy_view=tSrK_copy_view,
            tOrVt_copy_view=tOrVt_copy_view,
        )
        softmax_params = SimpleNamespace(
            row_max=row_max, row_sum=row_sum,
            softmax_scale_log2=softmax_scale_log2,
        )

        o_seq_stride = mO.shape[2] * self._head_dim
        o_seq_stride = cute.assume(o_seq_stride, divby=8)
        o_batch_stride = mO.shape[1] * o_seq_stride
        o_row_base = batch * o_batch_stride + head_base * self._head_dim

        counter_slot = batch * num_kv_heads + kv_head
        p_base = counter_slot * self._max_splits

        # Row-wise merge mapping shared by the intra-CTA and the split
        # merge: warp w owns rows w, w+num_warps, ...; a lane owns 4
        # consecutive columns of each half row -> 128-bit fp32 accesses.
        half_cols = self._head_dim // 2
        rows_per_warp = self._m_block_size // num_warps
        col = lane * 4
        k_stage_f32 = self._n_block_size * self._head_dim_padded // 2
        kv_half_f32 = kv_half_elems // 2
        f_kv = cute.recast_ptr(sK.iterator, dtype=cutlass.Float32)

        if my_tiles > 0:
            # ///////////////////////////////////////////////////////////////
            # Prologue: this warp's first `depth` chunk gathers.  Every
            # stage's index + page-id loads are issued before any gather so
            # the dependent round trips of the stages overlap.
            # ///////////////////////////////////////////////////////////////
            metas = []
            for s in cutlass.range_constexpr(depth):
                metas.append(
                    self._load_chunk_meta(
                        mIdx, mPT, batch, ctx_len, chunk_base + s * num_warps, topk
                    )
                )
            for s in cutlass.range_constexpr(depth):
                if s < n_my:
                    idx_s, pgk_s, pgv_s = metas[s]
                    self._gather_stage_paged(
                        mPool, sIdx, tKsK, tVsV,
                        gmem_tiled_copy_row, gmem_thr_copy_row,
                        pool_page_stride, pool_row_stride, pool_head_off,
                        idx_s, pgk_s, pgv_s, stage_base + s,
                    )
                cute.arch.cp_async_commit_group()
            # Index/page metadata runs TWO chunks ahead of the gather that
            # consumes it; one compute iteration does not hide the dependent
            # index -> page-table round trip.
            idx_a, pgk_a, pgv_a = self._load_chunk_meta(
                mIdx, mPT, batch, ctx_len, chunk_base + depth * num_warps, topk
            )
            idx_b, pgk_b, pgv_b = self._load_chunk_meta(
                mIdx, mPT, batch, ctx_len, chunk_base + (depth + 1) * num_warps, topk
            )

            # Q rows (shared by all warps): warp-strided synchronous 128-bit
            # copies, issued behind the in-flight gathers.
            for row in cutlass.range_constexpr(self._m_block_size):
                if row % num_warps == warp_id:
                    if row < group_size:
                        gRow = self._gmem_row_view(
                            mQ, q_row_base + row * self._head_dim
                        )
                        tRow = gmem_thr_copy_row_sync.partition_S(gRow)
                        tRowAligned = cute.make_tensor(
                            tRow.iterator.align(16), tRow.layout
                        )
                        cute.copy(
                            gmem_tiled_copy_row_sync,
                            tRowAligned[None, 0, 0],
                            tQsQ[None, row, 0],
                        )
                    else:
                        tQsQ[None, row, 0].fill(0)
            cute.arch.sync_threads()

            # ///////////////////////////////////////////////////////////////
            # Per-warp pipelined mainloop over this warp's chunks.
            # ///////////////////////////////////////////////////////////////
            for j in range(n_my):
                stage = stage_base + j % depth
                cute.arch.cp_async_wait_group(depth - 1)
                cute.arch.sync_warp()
                self.compute_one_chunk(
                    mma_params, smem_copy_params, softmax_params, sIdx, stage,
                )
                # All lanes' ldmatrix reads of `stage` retire before refill.
                cute.arch.sync_warp()
                next_j = j + depth
                if next_j < n_my:
                    self._gather_stage_paged(
                        mPool, sIdx, tKsK, tVsV,
                        gmem_tiled_copy_row, gmem_thr_copy_row,
                        pool_page_stride, pool_row_stride, pool_head_off,
                        idx_a, pgk_a, pgv_a, stage,
                    )
                    idx_a = idx_b
                    pgk_a = pgk_b
                    pgv_a = pgv_b
                    idx_b, pgk_b, pgv_b = self._load_chunk_meta(
                        mIdx, mPT, batch, ctx_len,
                        chunk_base + (next_j + 2) * num_warps, topk,
                    )
                cute.arch.cp_async_commit_group()

            # ///////////////////////////////////////////////////////////////
            # Stage this warp's accumulator + stats in smem.  The fp32 tile
            # aliases the warp's OWN K stages (columns [0, D/2)) and V stages
            # (columns [D/2, D)); no other warp ever touched them.
            # ///////////////////////////////////////////////////////////////
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_warp()
            for r in cutlass.range_constexpr(cute.size(row_sum)):
                row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
            mcO = cute.make_identity_tensor(
                (self._m_block_size, self._head_dim_padded)
            )
            tScO = thr_mma.partition_C(mcO)
            tScO_mn = self._make_acc_tensor_mn_view(tScO)
            for r in cutlass.range_constexpr(cute.size(tScO_mn.shape[0])):
                row_g = tScO_mn[r, 0][0]
                # Lanes of a quad hold identical reduced stats; one writes.
                if lane % 4 == 0:
                    sStats[warp_id, 0, row_g] = row_max[r]
                    sStats[warp_id, 1, row_g] = row_sum[r]
            sAcc = cute.make_tensor(
                self._smem_f32_ptr(f_kv, warp_id * (depth * k_stage_f32), 16),
                cute.make_layout(
                    (self._m_block_size, (half_cols, 2)),
                    stride=(half_cols, (1, kv_half_f32)),
                ),
            )
            atom_f32 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Float32
            )
            tiled_copy_acc = cute.make_tiled_copy_C(atom_f32, tiled_mma)
            thr_copy_acc = tiled_copy_acc.get_slice(lane)
            cute.copy(
                atom_f32,
                thr_copy_acc.retile(acc_O),
                thr_copy_acc.partition_D(sAcc),
            )
            cute.arch.sync_threads()

            # ///////////////////////////////////////////////////////////////
            # Cooperative merge of the warps (standard split-K rescale) ->
            # direct normalized store (n_active == 1) or the split partial.
            # ///////////////////////////////////////////////////////////////
            for rr in cutlass.range_constexpr(rows_per_warp):
                row = warp_id + rr * num_warps
                if row < group_size:
                    m_all = -cutlass.Float32.inf
                    for v in cutlass.range_constexpr(num_warps):
                        m_all = cute.arch.fmax(m_all, sStats[v, 0, row])
                    m_safe = 0.0 if m_all == -cutlass.Float32.inf else m_all
                    denom = cutlass.Float32(0.0)
                    num_lo = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
                    num_hi = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
                    num_lo.fill(0.0)
                    num_hi.fill(0.0)
                    for v in cutlass.range_constexpr(num_warps):
                        w_v = cute.math.exp2(
                            sStats[v, 0, row] * softmax_scale_log2
                            - m_safe * softmax_scale_log2,
                            fastmath=True,
                        )
                        denom = denom + sStats[v, 1, row] * w_v
                        off = v * (depth * k_stage_f32) + row * half_cols + col
                        num_lo.store(
                            num_lo.load() + self._smem_f32_vec4(f_kv, off).load() * w_v
                        )
                        num_hi.store(
                            num_hi.load()
                            + self._smem_f32_vec4(f_kv, off + kv_half_f32).load() * w_v
                        )
                    if n_active == 1:
                        zero_or_nan = denom == 0.0 or denom != denom
                        inv = 0.0 if zero_or_nan else 1.0 / denom
                        o_off = o_row_base + row * self._head_dim + col
                        self._gmem_vec4(mO, o_off, self._dtype, 8).store(
                            (num_lo.load() * inv).to(self._dtype)
                        )
                        self._gmem_vec4(mO, o_off + half_cols, self._dtype, 8).store(
                            (num_hi.load() * inv).to(self._dtype)
                        )
                    else:
                        p_slot = p_base + sid
                        po_off = cute.crd2idx((p_slot, row, col), mPartialO.layout)
                        self._gmem_vec4(mPartialO, po_off, cutlass.Float32, 16).store(
                            num_lo.load()
                        )
                        self._gmem_vec4(
                            mPartialO, po_off + half_cols, cutlass.Float32, 16
                        ).store(num_hi.load())
                        if lane == 0:
                            mPartialStats[p_slot, 0, row] = m_all
                            mPartialStats[p_slot, 1, row] = denom

        # ///////////////////////////////////////////////////////////////////
        # Last-arriver merge (threadfence reduction pattern): every CTA of the
        # (batch, kv_head) family arrives — the static MAX_SPLITS denominator.
        # ///////////////////////////////////////////////////////////////////
        _threadfence_gl()
        cute.arch.sync_threads()
        if tidx == 0:
            old = _atomic_add_global_i32(
                _gmem_addr_i64(mCounters, counter_slot), cutlass.Int32(1)
            )
            sFlag[0] = old
        cute.arch.sync_threads()
        arrived = sFlag[0]
        if arrived == self._max_splits - 1:
            # Self-reset for the next step (callers still zero on fresh
            # workspaces); release pairs with the next launch's reads.
            if tidx == 0:
                _st_global_release_i32(
                    _gmem_addr_i64(mCounters, counter_slot), cutlass.Int32(0)
                )
            if n_active > 1:
                _threadfence_gl()
                for rr in cutlass.range_constexpr(rows_per_warp):
                    row = warp_id + rr * num_warps
                    if row < group_size:
                        # Branch-free over the static MAX_SPLITS: inactive
                        # slots re-read the last active split with weight 0,
                        # so all loads issue back to back.
                        m_all = -cutlass.Float32.inf
                        for s in cutlass.range_constexpr(self._max_splits):
                            slot = p_base + cutlass.min(s, n_active - 1)
                            m_s = mPartialStats[slot, 0, row]
                            if s > 0:
                                m_s = -cutlass.Float32.inf if s >= n_active else m_s
                            m_all = cute.arch.fmax(m_all, m_s)
                        m_safe = 0.0 if m_all == -cutlass.Float32.inf else m_all
                        denom = cutlass.Float32(0.0)
                        num_lo = cute.make_rmem_tensor(
                            cute.make_layout(4), cutlass.Float32
                        )
                        num_hi = cute.make_rmem_tensor(
                            cute.make_layout(4), cutlass.Float32
                        )
                        num_lo.fill(0.0)
                        num_hi.fill(0.0)
                        for s in cutlass.range_constexpr(self._max_splits):
                            slot = p_base + cutlass.min(s, n_active - 1)
                            w_s = cute.math.exp2(
                                mPartialStats[slot, 0, row] * softmax_scale_log2
                                - m_safe * softmax_scale_log2,
                                fastmath=True,
                            )
                            if s > 0:
                                w_s = 0.0 if s >= n_active else w_s
                            denom = denom + mPartialStats[slot, 1, row] * w_s
                            po_off = cute.crd2idx((slot, row, col), mPartialO.layout)
                            num_lo.store(
                                num_lo.load()
                                + self._gmem_vec4(
                                    mPartialO, po_off, cutlass.Float32, 16
                                ).load()
                                * w_s
                            )
                            num_hi.store(
                                num_hi.load()
                                + self._gmem_vec4(
                                    mPartialO, po_off + half_cols, cutlass.Float32, 16
                                ).load()
                                * w_s
                            )
                        zero_or_nan = denom == 0.0 or denom != denom
                        inv = 0.0 if zero_or_nan else 1.0 / denom
                        o_off = o_row_base + row * self._head_dim + col
                        self._gmem_vec4(mO, o_off, self._dtype, 8).store(
                            (num_lo.load() * inv).to(self._dtype)
                        )
                        self._gmem_vec4(mO, o_off + half_cols, self._dtype, 8).store(
                            (num_hi.load() * inv).to(self._dtype)
                        )

    def _smem_f32_ptr(self, base: cute.Pointer, elem_off, align: int) -> cute.Pointer:
        """fp32 smem pointer ``base + elem_off`` with an asserted alignment
        (all offsets used here are multiples of 4 elements)."""
        return cute.make_ptr(
            cutlass.Float32,
            (base + elem_off).toint(),
            cute.AddressSpace.smem,
            assumed_align=align,
        )

    def _smem_f32_vec4(self, base: cute.Pointer, elem_off) -> cute.Tensor:
        """A 4-element fp32 smem view (one 128-bit access)."""
        return cute.make_tensor(
            self._smem_f32_ptr(base, elem_off, 16), cute.make_layout(4)
        )

    def _gmem_vec4(self, mT: cute.Tensor, elem_off, dtype, align: int) -> cute.Tensor:
        """A 4-element gmem view of ``mT`` at ``elem_off`` (one 64/128-bit
        access; the caller guarantees ``elem_off % 4 == 0`` on a 16-byte
        aligned base)."""
        ptr = cute.make_ptr(
            dtype,
            (mT.iterator + elem_off).toint(),
            cute.AddressSpace.gmem,
            assumed_align=align,
        )
        return cute.make_tensor(ptr, cute.make_layout(4))

    @cute.jit
    def _load_chunk_meta(
        self,
        mIdx: cute.Tensor,
        mPT: cute.Tensor,
        batch: cutlass.Int32,
        ctx_len: cutlass.Int32,
        chunk: cutlass.Int32,
        topk: cutlass.Int32,
    ):
        """Coalesced per-lane prefetch of a chunk's token indices AND their
        K/V page ids (page = idx / 128 via the page table; V ids pre-offset
        by +numPages by the caller).  Branch-free so the compiler can issue
        the loads of several chunks together: out-of-range lanes read a
        clamped in-bounds index and are then forced to ``-1``; the page
        lookup clamps to ``[0, ctx_len - 1]`` (always a mapped page) and
        invalid rows are still skipped by the gather's ``idx >= 0`` test."""
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        pos_l = chunk * self._n_block_size + lane
        in_range = lane < self._n_block_size and pos_l < topk
        pos_c = cutlass.min(pos_l, topk - 1)
        idx_raw = mIdx[batch, 0, pos_c]
        idx_l = idx_raw if in_range else cutlass.Int32(-1)
        page = cutlass.min(cutlass.max(idx_l, 0), ctx_len - 1) // 128
        pgk_l = mPT[batch, 0, page]
        pgv_l = mPT[batch, 1, page]
        return idx_l, pgk_l, pgv_l

    @cute.jit
    def _gather_stage_paged(
        self,
        mPool: cute.Tensor,
        sIdx: cute.Tensor,
        dstK,
        dstV,
        gmem_tiled_copy_row: cute.TiledCopy,
        gmem_thr_copy_row,
        pool_page_stride: cutlass.Int32,
        pool_row_stride: cutlass.Int32,
        pool_head_off: cutlass.Int32,
        idx_l: cutlass.Int32,
        pgk_l: cutlass.Int32,
        pgv_l: cutlass.Int32,
        stage: cutlass.Int32,
    ):
        """Paged variant of ``_gather_stage``: one warp-wave per K/V row,
        addressed pool-page-wise.  Only columns [0, head_dim) of each pool
        row are copied — the indexer-state tail is never read."""
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        if lane < self._n_block_size:
            sIdx[stage, lane] = idx_l
        for w in cutlass.range_constexpr(self._n_block_size):
            idx = cutlass.Int32(cute.arch.shuffle_sync(idx_l, w))
            pgk = cutlass.Int32(cute.arch.shuffle_sync(pgk_l, w))
            pgv = cutlass.Int32(cute.arch.shuffle_sync(pgv_l, w))
            if idx >= 0:
                in_page_off = (idx % 128) * pool_row_stride + pool_head_off
                gRowK = self._gmem_row_view(
                    mPool, pgk * pool_page_stride + in_page_off
                )
                tRowK = gmem_thr_copy_row.partition_S(gRowK)
                tRowKAligned = cute.make_tensor(tRowK.iterator.align(16), tRowK.layout)
                cute.copy(
                    gmem_tiled_copy_row,
                    tRowKAligned[None, 0, 0],
                    dstK[None, w, 0, stage],
                )
                gRowV = self._gmem_row_view(
                    mPool, pgv * pool_page_stride + in_page_off
                )
                tRowV = gmem_thr_copy_row.partition_S(gRowV)
                tRowVAligned = cute.make_tensor(tRowV.iterator.align(16), tRowV.layout)
                cute.copy(
                    gmem_tiled_copy_row,
                    tRowVAligned[None, 0, 0],
                    dstV[None, w, 0, stage],
                )
            else:
                dstK[None, w, 0, stage].fill(0)
                dstV[None, w, 0, stage].fill(0)


# ---------------------------------------------------------------------------
# Host-side helpers (CuPy/NumPy; no torch dependency)
# ---------------------------------------------------------------------------


def _cutlass_to_cupy_dtype(cutlass_dtype):
    if cutlass_dtype == cutlass.Float16:
        return cp.float16
    if cutlass_dtype == cutlass.BFloat16:
        # CuPy lacks a native bf16 dtype; store raw uint16 bytes — `from_dlpack`
        # plus `element_type = cutlass.BFloat16` interpret it correctly.
        return cp.uint16
    raise ValueError(f"Unsupported cutlass dtype for CuPy: {cutlass_dtype}")


def _to_float32(arr: cp.ndarray, dtype) -> cp.ndarray:
    """View a stored fp16/bf16 CuPy array as fp32 values."""
    if dtype == cutlass.Float16:
        return arr.astype(cp.float32)
    return (arr.astype(cp.uint32) << 16).view(cp.float32).reshape(arr.shape)


def _nan_pattern(dtype):
    if dtype == cutlass.Float16:
        return cp.float16(cp.nan)
    return cp.uint16(0x7FC0)  # bf16 quiet NaN


def _create_bsnd_tensor(
    b: int,
    s: int,
    h: int,
    d: int,
    dtype: Type[cutlass.Numeric],
    *,
    fill_random: bool,
    storage_free: bool = False,
):
    """Allocate a BSND CuPy tensor and return the ``cute.Tensor`` wrapper.

    B / S / H are runtime-dynamic; D is compile-time-known.  We deliberately
    do NOT call ``mark_layout_dynamic`` — see the FMHA-v2 comment: it would
    strip the static-D alignment fact the IR verifier needs for the 128-bit
    cp.async source pointers.
    """
    shape = (b, s, h, d)
    if storage_free:
        if fill_random:
            raise ValueError("storage_free tensors carry no data to randomize")
        arr = None
        t = aot_placeholders.make_compact_tensor(
            dtype, shape, stride_order=(3, 2, 1, 0), assumed_align=16
        )
    else:
        cp_dtype = _cutlass_to_cupy_dtype(dtype)
        if fill_random:
            if dtype == cutlass.Float16:
                arr = cp.random.uniform(-1.0, 1.0, shape).astype(cp_dtype)
            else:
                f32 = cp.random.uniform(-1.0, 1.0, shape).astype(cp.float32)
                arr = cp.ascontiguousarray(
                    (f32.view(cp.uint32) >> 16).astype(cp.uint16)
                )
        else:
            arr = cp.zeros(shape, dtype=cp_dtype)

        t = from_dlpack(arr, assumed_align=16)
        t.element_type = dtype
    so = (0, 1, 2, 3)
    t = (
        t.mark_compact_shape_dynamic(mode=0, stride_order=so)
        .mark_compact_shape_dynamic(mode=1, stride_order=so)
        .mark_compact_shape_dynamic(mode=2, stride_order=so)
    )
    return t, arr


def _wrap_indices_tensor(arr: cp.ndarray):
    """Wrap a contiguous ``(B, S, topk)`` Int32 index tensor (all modes dynamic)."""
    t = from_dlpack(arr, assumed_align=16)
    return _mark_indices_dynamic(t)


def _make_indices_placeholder(batch_size: int, seqlen: int, topk: int):
    """Storage-free ``(B, S, topk)`` Int32 index-tensor descriptor."""
    t = aot_placeholders.make_compact_tensor(
        cutlass.Int32, (batch_size, seqlen, topk), stride_order=(2, 1, 0),
        assumed_align=16,
    )
    return _mark_indices_dynamic(t)


def _mark_indices_dynamic(t):
    so = (0, 1, 2)
    return (
        t.mark_layout_dynamic(leading_dim=2)
        .mark_compact_shape_dynamic(mode=0, stride_order=so)
        .mark_compact_shape_dynamic(mode=1, stride_order=so)
        .mark_compact_shape_dynamic(mode=2, stride_order=so)
    )


def _wrap_ctx_lengths_tensor(arr: cp.ndarray):
    """Wrap a ``(B,)`` Int32 context-lengths tensor."""
    t = from_dlpack(arr, assumed_align=16)
    return t.mark_layout_dynamic(leading_dim=0).mark_compact_shape_dynamic(
        mode=0, stride_order=(0,)
    )


def _make_ctx_lengths_placeholder(batch_size: int):
    """Storage-free ``(B,)`` Int32 context-lengths descriptor."""
    t = aot_placeholders.make_compact_tensor(
        cutlass.Int32, (batch_size,), stride_order=(0,), assumed_align=16
    )
    return t.mark_layout_dynamic(leading_dim=0).mark_compact_shape_dynamic(
        mode=0, stride_order=(0,)
    )


def _generate_qsa_indices(
    batch_size: int,
    seqlen: int,
    topk: int,
    ctx_lengths,
    *,
    compress_ratio: int = 4,
    block_topk: int = 512,
    seed: int = 20260829,
) -> np.ndarray:
    """Generate index lists matching the QSA indexer output contract.

    Per valid row t: ``min(block_topk, (t+1)//ratio)`` distinct blocks
    (random subset standing in for the score top-k), expanded to runs of
    ``ratio`` consecutive token ids and packed to the front in shuffled
    block order, followed by the ``(t+1) % ratio`` tail tokens (always
    present), ``-1``-padded to ``topk``.  Padding rows are all ``-1``.
    """
    rng = np.random.default_rng(seed)
    idx = np.full((batch_size, seqlen, topk), -1, dtype=np.int32)
    for b in range(batch_size):
        L = int(ctx_lengths[b])
        for t in range(L):
            n_vis = (t + 1) // compress_ratio
            k_sel = min(block_topk, n_vis)
            if k_sel > 0:
                blocks = rng.choice(n_vis, size=k_sel, replace=False)
                toks = (
                    blocks[:, None] * compress_ratio
                    + np.arange(compress_ratio)[None, :]
                ).reshape(-1)
            else:
                toks = np.empty(0, dtype=np.int64)
            tail = np.arange(n_vis * compress_ratio, t + 1)
            row = np.concatenate([toks, tail]).astype(np.int32)
            assert len(row) <= topk, (t, len(row), topk)
            idx[b, t, : len(row)] = row
    return idx


def _qsa_sparse_reference(
    q: cp.ndarray,
    k: cp.ndarray,
    v: cp.ndarray,
    indices: np.ndarray,
    ctx_lengths,
    softmax_scale: float,
    dtype,
) -> cp.ndarray:
    """FP32 CuPy oracle: gather-by-index softmax attention per query row.

    Rows with no valid index (and padding rows) produce exact zeros.
    """
    batch_size, seqlen, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    group_size = num_q_heads // num_kv_heads
    reference = cp.zeros(q.shape, dtype=cp.float32)
    q_f32 = _to_float32(q, dtype)
    k_f32 = _to_float32(k, dtype)
    v_f32 = _to_float32(v, dtype)
    idx_gpu = cp.asarray(indices)
    for b in range(batch_size):
        L = int(ctx_lengths[b])
        for t in range(L):
            row = idx_gpu[b, t]
            valid = row[row >= 0]
            if valid.size == 0:
                continue
            for g in range(num_kv_heads):
                k_sel = k_f32[b, valid, g, :]  # (n, D)
                v_sel = v_f32[b, valid, g, :]
                q_grp = q_f32[b, t, g * group_size : (g + 1) * group_size, :]
                scores = q_grp @ k_sel.T * softmax_scale
                scores -= cp.max(scores, axis=1, keepdims=True)
                p = cp.exp(scores)
                p /= cp.sum(p, axis=1, keepdims=True)
                reference[b, t, g * group_size : (g + 1) * group_size, :] = (
                    p @ v_sel
                )
    return reference


def _report_reference_error(tag: str, actual: cp.ndarray, reference: cp.ndarray):
    """Print and enforce the standalone low-precision-vs-FP32 gate."""
    error = cp.abs(actual - reference)
    max_abs = float(cp.max(error).get()) if error.size else 0.0
    close_rate = float(cp.mean(error <= 5.0e-2).get()) if error.size else 1.0
    print(f"{tag} reference max_abs={max_abs:.6f}, close_rate@0.05={close_rate:.6f}")
    if max_abs > 1.0e-1 or close_rate < 0.999:
        raise RuntimeError(
            f"{tag} reference check failed: max_abs={max_abs:.6f}, "
            f"close_rate@0.05={close_rate:.6f}"
        )


# ---------------------------------------------------------------------------
# run(): test + AOT export entry point
# ---------------------------------------------------------------------------


def run(
    dtype: Type[cutlass.Numeric],
    batch_size: int,
    seqlen: int,
    num_head: int,
    head_dim: int,
    kv_group_size: int = 12,
    topk: int = 2051,
    softmax_scale: float = 0.0,
    m_block_size: int = QSA_DEFAULT_M_BLOCK,
    n_block_size: int = QSA_DEFAULT_N_BLOCK,
    num_threads: int = QSA_DEFAULT_THREADS,
    pipe_depth: int = QSA_DEFAULT_PIPE_DEPTH,
    ragged: bool = False,
    warmup_iterations: int = 3,
    iterations: int = 10,
    skip_ref_check: bool = False,
    export_only: bool = False,
    output_dir: str = "./qsa_sparse_aot_artifacts",
    file_name: str = "qsa_sparse",
    function_prefix: str = "qsa_sparse",
    **kwargs,
):
    """Compile, test, benchmark, or export the QSA sparse-GQA kernel.

    AOT export uses dummy placeholder shapes; only ``head_dim`` and the
    ``(Br, Bc, threads)`` tuning are baked at compile time — batch, seq,
    H_q, H_kv, topk, strides and the softmax scale stay runtime-dynamic.
    """
    _tag = f"[{file_name}]"

    if not QSASparseGQAPrefill.can_implement(
        dtype, head_dim, m_block_size, n_block_size, num_threads, pipe_depth
    ):
        raise ValueError(
            f"{_tag} Unsupported config: dtype={dtype}, head_dim={head_dim}, "
            f"Br={m_block_size}, Bc={n_block_size}, threads={num_threads}, "
            f"pipe_depth={pipe_depth}"
        )
    if num_head % kv_group_size != 0:
        raise ValueError(
            f"{_tag} num_head ({num_head}) must be divisible by "
            f"kv_group_size ({kv_group_size})"
        )
    if kv_group_size > m_block_size:
        raise ValueError(
            f"{_tag} kv_group_size ({kv_group_size}) must be <= "
            f"m_block_size ({m_block_size})"
        )

    if not export_only and cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU is required to run this kernel.")

    if softmax_scale <= 0.0:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    h_q = num_head
    h_kv = h_q // kv_group_size

    if export_only:
        print(
            f"{_tag} Compiling QSA sparse-GQA CuTe DSL: dtype={dtype}, "
            f"head_dim={head_dim}, Br={m_block_size}, Bc={n_block_size}, "
            f"threads={num_threads}"
        )
    else:
        print(f"{_tag} Running QSA sparse-GQA CuTe DSL forward with:")
        print(f"{_tag}   dtype={dtype}, head_dim={head_dim}, topk={topk}")
        print(
            f"{_tag}   B={batch_size}, S={seqlen}, H_q={h_q}, H_kv={h_kv}, "
            f"group={kv_group_size}, ragged={ragged}"
        )
        print(f"{_tag}   softmax_scale={softmax_scale}")
        print(f"{_tag}   Br={m_block_size}, Bc={n_block_size}, threads={num_threads}, "
              f"pipe_depth={pipe_depth}")
        cp.random.seed(20260829)
        print(f"{_tag}   CuPy random seed=20260829")

    q_dyn, q_arr = _create_bsnd_tensor(
        batch_size, seqlen, h_q, head_dim, dtype,
        fill_random=not export_only, storage_free=export_only,
    )
    k_dyn, k_arr = _create_bsnd_tensor(
        batch_size, seqlen, h_kv, head_dim, dtype,
        fill_random=not export_only, storage_free=export_only,
    )
    v_dyn, v_arr = _create_bsnd_tensor(
        batch_size, seqlen, h_kv, head_dim, dtype,
        fill_random=not export_only, storage_free=export_only,
    )
    o_dyn, o_arr = _create_bsnd_tensor(
        batch_size, seqlen, h_q, head_dim, dtype,
        fill_random=False, storage_free=export_only,
    )

    if export_only:
        ctx_host = None
        idx_host = None
        ctx_dyn = _make_ctx_lengths_placeholder(batch_size)
        idx_dyn = _make_indices_placeholder(batch_size, seqlen, topk)
    else:
        if ragged and batch_size > 1:
            rng = np.random.default_rng(20260829)
            ctx_host = np.sort(
                rng.integers(1, seqlen + 1, size=batch_size).astype(np.int32)
            )[::-1].copy()
            ctx_host[0] = seqlen
        else:
            ctx_host = np.full(batch_size, seqlen, dtype=np.int32)
        ctx_arr = cp.asarray(ctx_host)
        ctx_dyn = _wrap_ctx_lengths_tensor(ctx_arr)

        idx_host = _generate_qsa_indices(batch_size, seqlen, topk, ctx_host)
        idx_arr = cp.asarray(idx_host)
        idx_dyn = _wrap_indices_tensor(idx_arr)

    if not export_only:
        # Poison everything the kernel must never read: K/V rows at or past
        # the live length (NaN), Q rows of padding tokens (NaN), and the
        # output buffer (sentinel — padding rows must come back as zeros).
        nan_val = _nan_pattern(dtype)
        for b in range(batch_size):
            L = int(ctx_host[b])
            if L < seqlen:
                k_arr[b, L:] = nan_val
                v_arr[b, L:] = nan_val
                q_arr[b, L:] = nan_val
        if dtype == cutlass.Float16:
            o_arr.fill(cp.float16(123.0))
        else:
            o_arr.fill(cp.uint16(0x42F6))  # bf16 123.0

    qsa_fwd = QSASparseGQAPrefill(
        head_dim=head_dim,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        num_threads=num_threads,
        pipe_depth=pipe_depth,
    )

    current_stream = (
        aot_placeholders.make_stream()
        if export_only
        else cuda.CUstream(cp.cuda.get_current_stream().ptr)
    )
    # Runtime persistent-grid size (sm_count kernel argument): AOT callers pass
    # the deployment GPU's multiprocessor count at launch; only the non-export
    # smoke path seeds it from the local device.
    _sm_count = (
        aot_placeholders.runtime_int32()
        if export_only
        else cutlass.Int32(utils.HardwareInfo().get_device_multiprocessor_count())
    )

    _ptx_parts = []
    if os.getenv("QSA_PTXAS_VERBOSE"):
        _ptx_parts.append("-v")
    _extra_ptx = os.getenv("QSA_PTXAS_OPTS", "")
    if _extra_ptx:
        _ptx_parts.append(_extra_ptx)
    compile_options = (
        {"options": f"--ptxas-options={','.join(_ptx_parts)}"} if _ptx_parts else {}
    )

    print(f"{_tag} Compiling kernel...")
    t0 = time.time()
    compiled_qsa = cute.compile(
        qsa_fwd,
        q_dyn,
        k_dyn,
        v_dyn,
        o_dyn,
        idx_dyn,
        ctx_dyn,
        cutlass.Float32(softmax_scale),
        _sm_count,
        current_stream,
        **compile_options,
    )
    print(f"{_tag} Compilation time: {time.time() - t0:.4f}s")

    if export_only:
        os.makedirs(output_dir, exist_ok=True)
        compiled_qsa.export_to_c(
            file_path=output_dir,
            file_name=file_name,
            function_prefix=function_prefix,
        )
        print(f"{_tag} Exported to {output_dir}/{file_name}.h and {file_name}.o")
        return None

    compiled_qsa(
        q_dyn,
        k_dyn,
        v_dyn,
        o_dyn,
        idx_dyn,
        ctx_dyn,
        cutlass.Float32(softmax_scale),
        cutlass.Int32(utils.HardwareInfo().get_device_multiprocessor_count()),
        current_stream,
    )
    cp.cuda.Device().synchronize()

    if not skip_ref_check:
        reference = _qsa_sparse_reference(
            q_arr, k_arr, v_arr, idx_host, ctx_host, softmax_scale, dtype
        )
        o_f32 = _to_float32(o_arr, dtype)
        # Padding token rows must be EXACT zeros (sentinel-poisoned output
        # buffer proves the kernel actually stored them).
        for b in range(batch_size):
            L = int(ctx_host[b])
            if L < seqlen:
                pad = o_f32[b, L:]
                if float(cp.max(cp.abs(pad)).get()) != 0.0:
                    raise RuntimeError(
                        f"{_tag} padding rows of batch {b} are not exact zeros"
                    )
        # Gate only the live rows: padded rows are exactly zero on both sides
        # and would otherwise inflate close_rate on ragged runs.
        valid_actual = cp.concatenate(
            [o_f32[b, : int(ctx_host[b])].ravel() for b in range(batch_size)]
        )
        valid_reference = cp.concatenate(
            [reference[b, : int(ctx_host[b])].ravel() for b in range(batch_size)]
        )
        _report_reference_error(_tag, valid_actual, valid_reference)
        print(f"{_tag} SPARSE_PASS: matched the FP32 gather-softmax reference")

    def _bench():
        t_start = time.time()
        for _ in range(iterations):
            compiled_qsa(
                q_dyn,
                k_dyn,
                v_dyn,
                o_dyn,
                idx_dyn,
                ctx_dyn,
                cutlass.Float32(softmax_scale),
                cutlass.Int32(
                    utils.HardwareInfo().get_device_multiprocessor_count()
                ),
                current_stream,
            )
        cp.cuda.Device().synchronize()
        return (time.time() - t_start) / iterations * 1e6

    for _ in range(warmup_iterations):
        compiled_qsa(
            q_dyn,
            k_dyn,
            v_dyn,
            o_dyn,
            idx_dyn,
            ctx_dyn,
            cutlass.Float32(softmax_scale),
            cutlass.Int32(utils.HardwareInfo().get_device_multiprocessor_count()),
            current_stream,
        )
    cp.cuda.Device().synchronize()
    avg_time_us = _bench()

    # Sparse FMHA FLOPs: 4 * H_q * D * sum over live rows of min(topk, t+1).
    flops = 0.0
    for b in range(batch_size):
        live = int(ctx_host[b])
        active = np.minimum(np.arange(1, live + 1), topk)
        flops += 4.0 * h_q * float(active.sum()) * head_dim
    tflops = flops / (avg_time_us * 1e-6) / 1e12
    print(f"{_tag} avg_time_us: {avg_time_us:.2f}  |  {tflops:.2f} TFLOPS ({dtype})")
    return avg_time_us


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _wrap_bsnd(arr: cp.ndarray, dtype):
    """Wrap an EXISTING contiguous BSND CuPy array (same marking as
    ``_create_bsnd_tensor``)."""
    t = from_dlpack(arr, assumed_align=16)
    t.element_type = dtype
    so = (0, 1, 2, 3)
    return (
        t.mark_compact_shape_dynamic(mode=0, stride_order=so)
        .mark_compact_shape_dynamic(mode=1, stride_order=so)
        .mark_compact_shape_dynamic(mode=2, stride_order=so)
    )


def _wrap_pool_tensor(arr: cp.ndarray, dtype):
    """Wrap the paged pool ``(numFlatPages, 128, Hkv, poolHeadDim)`` with all
    modes dynamic (the kernel addresses it through raw-pointer views only)."""
    t = from_dlpack(arr, assumed_align=16)
    t.element_type = dtype
    so = (0, 1, 2, 3)
    return (
        t.mark_layout_dynamic(leading_dim=3)
        .mark_compact_shape_dynamic(mode=0, stride_order=so)
        .mark_compact_shape_dynamic(mode=1, stride_order=so)
        .mark_compact_shape_dynamic(mode=2, stride_order=so)
        .mark_compact_shape_dynamic(mode=3, stride_order=so)
    )


def _wrap_page_table_tensor(arr: cp.ndarray):
    """Wrap a contiguous ``(B, 2, maxPagesPerSeq)`` Int32 page table."""
    t = from_dlpack(arr, assumed_align=16)
    so = (0, 1, 2)
    return (
        t.mark_layout_dynamic(leading_dim=2)
        .mark_compact_shape_dynamic(mode=0, stride_order=so)
        .mark_compact_shape_dynamic(mode=2, stride_order=so)
    )


def _wrap_partials_tensor(arr: cp.ndarray):
    """Wrap fp32 partial workspaces ``(P, a, b)`` with dynamic P."""
    t = from_dlpack(arr, assumed_align=16)
    so = (0, 1, 2)
    return t.mark_layout_dynamic(leading_dim=2).mark_compact_shape_dynamic(
        mode=0, stride_order=so
    )


def _make_decode_aot_tensors(
    dtype,
    *,
    batch_size: int,
    num_head: int,
    num_kv_head: int,
    head_dim: int,
    pool_head_dim: int,
    num_flat_pages: int,
    max_pages: int,
    topk: int,
    max_splits: int,
    m_block_size: int,
):
    """Create storage-free descriptors matching the decode runtime layouts."""
    q, _ = _create_bsnd_tensor(
        batch_size,
        1,
        num_head,
        head_dim,
        dtype,
        fill_random=False,
        storage_free=True,
    )
    output, _ = _create_bsnd_tensor(
        batch_size,
        1,
        num_head,
        head_dim,
        dtype,
        fill_random=False,
        storage_free=True,
    )

    pool = aot_placeholders.make_compact_tensor(
        dtype,
        (num_flat_pages, 128, num_kv_head, pool_head_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    pool_stride_order = (0, 1, 2, 3)
    pool = (
        pool.mark_layout_dynamic(leading_dim=3)
        .mark_compact_shape_dynamic(mode=0, stride_order=pool_stride_order)
        .mark_compact_shape_dynamic(mode=1, stride_order=pool_stride_order)
        .mark_compact_shape_dynamic(mode=2, stride_order=pool_stride_order)
        .mark_compact_shape_dynamic(mode=3, stride_order=pool_stride_order)
    )

    page_table = aot_placeholders.make_compact_tensor(
        cutlass.Int32,
        (batch_size, 2, max_pages),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    page_table_stride_order = (0, 1, 2)
    page_table = (
        page_table.mark_layout_dynamic(leading_dim=2)
        .mark_compact_shape_dynamic(mode=0, stride_order=page_table_stride_order)
        .mark_compact_shape_dynamic(mode=2, stride_order=page_table_stride_order)
    )

    num_partials = batch_size * num_kv_head * max_splits

    def _make_partials(shape):
        tensor = aot_placeholders.make_compact_tensor(
            cutlass.Float32,
            shape,
            stride_order=(2, 1, 0),
            assumed_align=16,
        )
        return tensor.mark_layout_dynamic(
            leading_dim=2
        ).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))

    return {
        "q": q,
        "pool": pool,
        "page_table": page_table,
        "indices": _make_indices_placeholder(batch_size, 1, topk),
        "context_lengths": _make_ctx_lengths_placeholder(batch_size),
        "output": output,
        "partial_output": _make_partials(
            (num_partials, m_block_size, head_dim)
        ),
        "partial_stats": _make_partials(
            (num_partials, 2, m_block_size)
        ),
        "counters": _make_ctx_lengths_placeholder(batch_size * num_kv_head),
    }


def _generate_decode_indices(
    batch_size: int,
    ctx_lengths,
    topk: int,
    *,
    compress_ratio: int = 4,
    block_topk: int = 512,
    seed: int = 20260830,
) -> np.ndarray:
    """Decode-step index rows: the QSA contract at query position ctx-1."""
    rng = np.random.default_rng(seed)
    idx = np.full((batch_size, 1, topk), -1, dtype=np.int32)
    for b in range(batch_size):
        t = int(ctx_lengths[b]) - 1
        n_vis = (t + 1) // compress_ratio
        k_sel = min(block_topk, n_vis)
        if k_sel > 0:
            blocks = rng.choice(n_vis, size=k_sel, replace=False)
            toks = (
                blocks[:, None] * compress_ratio
                + np.arange(compress_ratio)[None, :]
            ).reshape(-1)
        else:
            toks = np.empty(0, dtype=np.int64)
        tail = np.arange(n_vis * compress_ratio, t + 1)
        row = np.concatenate([toks, tail]).astype(np.int32)
        idx[b, 0, : len(row)] = row
    return idx


def _qsa_decode_reference(
    q: cp.ndarray,
    k_dense: cp.ndarray,
    v_dense: cp.ndarray,
    indices: np.ndarray,
    ctx_lengths,
    softmax_scale: float,
    dtype,
) -> cp.ndarray:
    """FP32 oracle for one decode step (one query row per request)."""
    batch_size, _, num_q_heads, head_dim = q.shape
    num_kv_heads = k_dense.shape[2]
    group_size = num_q_heads // num_kv_heads
    reference = cp.zeros((batch_size, 1, num_q_heads, head_dim), dtype=cp.float32)
    q_f32 = _to_float32(q, dtype)
    k_f32 = _to_float32(k_dense, dtype)
    v_f32 = _to_float32(v_dense, dtype)
    idx_gpu = cp.asarray(indices)
    for b in range(batch_size):
        row = idx_gpu[b, 0]
        valid = row[row >= 0]
        if valid.size == 0:
            continue
        for g in range(num_kv_heads):
            k_sel = k_f32[b, valid, g, :]
            v_sel = v_f32[b, valid, g, :]
            q_grp = q_f32[b, 0, g * group_size : (g + 1) * group_size, :]
            scores = q_grp @ k_sel.T * softmax_scale
            scores -= cp.max(scores, axis=1, keepdims=True)
            p = cp.exp(scores)
            p /= cp.sum(p, axis=1, keepdims=True)
            reference[b, 0, g * group_size : (g + 1) * group_size, :] = p @ v_sel
    return reference


def run_decode(
    dtype,
    batch_size: int,
    seqlen: int,
    num_head: int,
    head_dim: int,
    kv_group_size: int = 12,
    topk: int = 2051,
    pool_head_dim: int = 384,
    max_splits: int = 8,
    softmax_scale: float = 0.0,
    m_block_size: int = QSA_DEFAULT_M_BLOCK,
    n_block_size: int = QSA_DEFAULT_N_BLOCK,
    num_threads: int = QSA_DEFAULT_THREADS,
    pipe_depth: int = QSA_DEFAULT_PIPE_DEPTH,
    ragged: bool = False,
    decode_steps: int = 1,
    warmup_iterations: int = 3,
    iterations: int = 10,
    skip_ref_check: bool = False,
    export_only: bool = False,
    output_dir: str = "./qsa_sparse_aot_artifacts",
    file_name: str = "qsa_sparse_decode",
    function_prefix: str = "qsa_sparse_decode",
    **kwargs,
):
    """Compile, test (multi-step, paged, NaN-poisoned), or export the decode
    split-K kernel.

    ``seqlen`` is the TOTAL context of the first decode step (past tokens +
    the current one); ``--decode_steps`` appends one token per step.  The
    pool is fully NaN-poisoned with a permuted non-identity page table and
    ``pool_head_dim``-wide rows whose ``[head_dim, pool_head_dim)`` tails stay
    NaN — proving the kernel never reads the indexer-state tail.  Each step
    runs the kernel TWICE and requires bitwise-identical outputs (validates
    the counter self-reset)."""
    _tag = f"[{file_name}]"

    if not QSASparseGQADecode.can_implement_decode(
        dtype, head_dim, m_block_size, n_block_size, num_threads, pipe_depth,
        max_splits,
    ):
        raise ValueError(f"{_tag} Unsupported decode config")
    if num_head % kv_group_size != 0:
        raise ValueError(f"{_tag} num_head must be divisible by kv_group_size")
    if pool_head_dim < head_dim or pool_head_dim % 8 != 0:
        raise ValueError(f"{_tag} pool_head_dim must be >= head_dim and /8")
    if not export_only and cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU is required to run this kernel.")
    if softmax_scale <= 0.0:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    h_q = num_head
    h_kv = h_q // kv_group_size
    cap = seqlen + decode_steps  # token capacity incl. appended steps
    max_pages = (cap + 127) // 128
    num_pages = batch_size * max_pages + 1  # +1 poison page
    num_flat_pages = 2 * num_pages

    if export_only:
        tensors = _make_decode_aot_tensors(
            dtype,
            batch_size=batch_size,
            num_head=h_q,
            num_kv_head=h_kv,
            head_dim=head_dim,
            pool_head_dim=pool_head_dim,
            num_flat_pages=num_flat_pages,
            max_pages=max_pages,
            topk=topk,
            max_splits=max_splits,
            m_block_size=m_block_size,
        )
        dec = QSASparseGQADecode(
            head_dim=head_dim,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            num_threads=num_threads,
            pipe_depth=pipe_depth,
            max_splits=max_splits,
        )
        current_stream = aot_placeholders.make_stream()
        compiled = cute.compile(
            dec,
            tensors["q"],
            tensors["pool"],
            tensors["page_table"],
            tensors["indices"],
            tensors["context_lengths"],
            tensors["output"],
            tensors["partial_output"],
            tensors["partial_stats"],
            tensors["counters"],
            cutlass.Float32(softmax_scale),
            current_stream,
        )
        os.makedirs(output_dir, exist_ok=True)
        compiled.export_to_c(
            file_path=output_dir,
            file_name=file_name,
            function_prefix=function_prefix,
        )
        print(f"{_tag} Exported to {output_dir}/{file_name}.h and .o")
        return None

    cp_dtype = _cutlass_to_cupy_dtype(dtype)
    nan_val = _nan_pattern(dtype)

    print(
        f"{_tag} decode: dtype={dtype}, B={batch_size}, ctx0={seqlen}, "
        f"steps={decode_steps}, Hq={h_q}/Hkv={h_kv}, D={head_dim}, "
        f"poolD={pool_head_dim}, topk={topk}, max_splits={max_splits}, "
        f"Br={m_block_size}, Bc={n_block_size}, threads={num_threads}, "
        f"depth={pipe_depth}"
    )
    cp.random.seed(20260830)

    # Pool: fully NaN-poisoned; only mapped K/V rows' [0:head_dim) get data.
    pool_arr = cp.full(
        (num_flat_pages, 128, h_kv, pool_head_dim), nan_val, dtype=cp_dtype
    )
    # Permuted, non-identity page table; V ids pre-offset by +num_pages.
    # Physical page 0 of each plane stays unmapped (poison stays NaN there).
    pt_host = np.empty((batch_size, 2, max_pages), dtype=np.int32)
    page_rng = np.random.default_rng(20260830)
    for b in range(batch_size):
        base = b * max_pages
        pt_host[b, 0] = 1 + base + page_rng.permutation(max_pages)
        pt_host[b, 1] = num_pages + 1 + base + page_rng.permutation(max_pages)
    pt_arr = cp.asarray(pt_host)

    if ragged and batch_size > 1:
        rng = np.random.default_rng(20260830)
        ctx_host = np.sort(
            rng.integers(4, seqlen + 1, size=batch_size).astype(np.int32)
        )[::-1].copy()
        ctx_host[0] = seqlen
    else:
        ctx_host = np.full(batch_size, seqlen, dtype=np.int32)

    # Dense mirrors (reference source of truth), NaN outside the live region.
    k_dense = cp.full((batch_size, cap, h_kv, head_dim), nan_val, dtype=cp_dtype)
    v_dense = cp.full((batch_size, cap, h_kv, head_dim), nan_val, dtype=cp_dtype)

    def _rand(shape):
        if dtype == cutlass.Float16:
            return cp.random.uniform(-1.0, 1.0, shape).astype(cp.float16)
        f32 = cp.random.uniform(-1.0, 1.0, shape).astype(cp.float32)
        return cp.ascontiguousarray((f32.view(cp.uint32) >> 16).astype(cp.uint16))

    def _write_tokens(b, t0, t1):
        """Fill dense mirrors + pool rows for tokens [t0, t1) of request b."""
        if t1 <= t0:
            return
        k_new = _rand((t1 - t0, h_kv, head_dim))
        v_new = _rand((t1 - t0, h_kv, head_dim))
        k_dense[b, t0:t1] = k_new
        v_dense[b, t0:t1] = v_new
        for t in range(t0, t1):
            pg = t // 128
            row = t % 128
            pool_arr[int(pt_host[b, 0, pg]), row, :, :head_dim] = k_dense[b, t]
            pool_arr[int(pt_host[b, 1, pg]), row, :, :head_dim] = v_dense[b, t]

    for b in range(batch_size):
        _write_tokens(b, 0, int(ctx_host[b]))

    q_arr = _rand((batch_size, 1, h_q, head_dim))
    o_arr = cp.zeros((batch_size, 1, h_q, head_dim), dtype=cp_dtype)
    idx_host = _generate_decode_indices(batch_size, ctx_host, topk)
    idx_arr = cp.asarray(idx_host)
    ctx_arr = cp.asarray(ctx_host)
    partial_o = cp.empty(
        (batch_size * h_kv * max_splits, m_block_size, head_dim), dtype=cp.float32
    )
    partial_stats = cp.empty(
        (batch_size * h_kv * max_splits, 2, m_block_size), dtype=cp.float32
    )
    counters = cp.zeros(batch_size * h_kv, dtype=cp.int32)

    q_dyn = _wrap_bsnd(q_arr, dtype)
    o_dyn = _wrap_bsnd(o_arr, dtype)
    pool_dyn = _wrap_pool_tensor(pool_arr, dtype)
    pt_dyn = _wrap_page_table_tensor(pt_arr)
    idx_dyn = _wrap_indices_tensor(idx_arr)
    ctx_dyn = _wrap_ctx_lengths_tensor(ctx_arr)
    po_dyn = _wrap_partials_tensor(partial_o)
    ps_dyn = _wrap_partials_tensor(partial_stats)
    cnt_dyn = _wrap_ctx_lengths_tensor(counters)

    dec = QSASparseGQADecode(
        head_dim=head_dim,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        num_threads=num_threads,
        pipe_depth=pipe_depth,
        max_splits=max_splits,
    )
    current_stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)

    print(f"{_tag} Compiling decode kernel...")
    t0 = time.time()
    compiled = cute.compile(
        dec,
        q_dyn, pool_dyn, pt_dyn, idx_dyn, ctx_dyn, o_dyn,
        po_dyn, ps_dyn, cnt_dyn,
        cutlass.Float32(softmax_scale),
        current_stream,
    )
    print(f"{_tag} Compilation time: {time.time() - t0:.4f}s")

    def _launch():
        compiled(
            q_dyn, pool_dyn, pt_dyn, idx_dyn, ctx_dyn, o_dyn,
            po_dyn, ps_dyn, cnt_dyn,
            cutlass.Float32(softmax_scale),
            current_stream,
        )

    for step in range(decode_steps):
        counters.fill(0)  # contract: zero on entry (kernel self-resets after)
        _launch()
        cp.cuda.Device().synchronize()
        out1 = o_arr.copy()
        _launch()  # counter self-reset check: identical result
        cp.cuda.Device().synchronize()
        if not bool(cp.all(out1 == o_arr).get()):
            raise RuntimeError(f"{_tag} step {step}: rerun mismatch (counters?)")
        if not skip_ref_check:
            reference = _qsa_decode_reference(
                q_arr, k_dense, v_dense, idx_host, ctx_host, softmax_scale, dtype
            )
            _report_reference_error(
                f"{_tag} step{step}", _to_float32(o_arr, dtype), reference
            )
        # Append one token per request and rebuild the step inputs.
        if step + 1 < decode_steps:
            for b in range(batch_size):
                t_new = int(ctx_host[b])
                _write_tokens(b, t_new, t_new + 1)
                ctx_host[b] += 1
            ctx_arr[...] = cp.asarray(ctx_host)
            q_arr[...] = _rand((batch_size, 1, h_q, head_dim))
            idx_host = _generate_decode_indices(
                batch_size, ctx_host, topk, seed=20260830 + step + 1
            )
            idx_arr[...] = cp.asarray(idx_host)
    print(f"{_tag} DECODE_PASS: {decode_steps} step(s) matched the FP32 oracle")

    for _ in range(warmup_iterations):
        counters.fill(0)
        _launch()
    cp.cuda.Device().synchronize()
    t_start = time.time()
    for _ in range(iterations):
        _launch()
    cp.cuda.Device().synchronize()
    avg_us = (time.time() - t_start) / iterations * 1e6
    print(f"{_tag} avg_time_us: {avg_us:.2f} per decode step (B={batch_size})")
    return avg_us


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="QSA sparse-GQA CuTe DSL prefill kernel: test and AOT export."
    )
    p.add_argument("--dtype", type=cutlass.dtype, default=cutlass.Float16,
                   help="Input/output dtype: Float16 or BFloat16 (default: Float16).")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seqlen", type=int, default=384)
    p.add_argument("--num_head", type=int, default=24,
                   help="Number of Q heads (H_q). Must be divisible by kv_group_size.")
    p.add_argument("--head_dim", type=int, default=256)
    p.add_argument("--kv_group_size", type=int, default=12,
                   help="GQA group size H_q / H_kv (default 12 = Qwen3.8-Flash-Next).")
    p.add_argument("--topk", type=int, default=2051,
                   help="Index-list width (default 2051 = 512*4 + 3 tail).")
    p.add_argument("--softmax_scale", type=float, default=0.0,
                   help="Softmax scale; 0 (default) => 1 / sqrt(head_dim).")
    p.add_argument("--m_block_size", type=int, default=QSA_DEFAULT_M_BLOCK)
    p.add_argument("--n_block_size", type=int, default=QSA_DEFAULT_N_BLOCK)
    p.add_argument("--num_threads", type=int, default=None,
                   help="CTA threads (default: 32 prefill, 128 decode)")
    p.add_argument("--pipe_depth", type=int, default=None,
                   help="cp.async (K, V) stages per warp "
                        "(default: 2 prefill, 2 decode)")
    p.add_argument("--ragged", action="store_true",
                   help="Use random per-batch context lengths (< seqlen).")
    p.add_argument("--decode", action="store_true",
                   help="Run/export the split-K DECODE kernel instead of prefill. "
                        "seqlen = total context of the first step.")
    p.add_argument("--decode_steps", type=int, default=1,
                   help="Decode steps to run+check (each appends one token).")
    p.add_argument("--max_splits", type=int, default=8,
                   help="Decode split-K MAX_SPLITS (compile-time grid dim).")
    p.add_argument("--pool_head_dim", type=int, default=384,
                   help="Decode paged-pool row width (>= head_dim; the tail "
                        "[head_dim, pool_head_dim) carries indexer state and "
                        "is never read — verified via NaN poisoning).")
    p.add_argument("--warmup_iterations", type=int, default=3)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--skip_ref_check", action="store_true")
    p.add_argument("--export_only", action="store_true",
                   help="Compile and export only; skip reference check and benchmark.")
    p.add_argument("--output_dir", type=str, default="./qsa_sparse_aot_artifacts",
                   help="Output directory for AOT artifacts (<file_name>.{h,o}).")
    p.add_argument("--file_name", type=str, default="qsa_sparse",
                   help="Base file name for exported artifacts.")
    p.add_argument("--function_prefix", type=str, default="qsa_sparse",
                   help="Function prefix for exported C symbols.")
    return p.parse_known_args(args=argv)[0]


def main():
    args = _parsed_args
    if args.decode:
        num_threads = (QSA_DECODE_DEFAULT_THREADS if args.num_threads is None
                       else args.num_threads)
        pipe_depth = (QSA_DECODE_DEFAULT_PIPE_DEPTH if args.pipe_depth is None
                      else args.pipe_depth)
        run_decode(
            dtype=args.dtype,
            batch_size=args.batch_size,
            seqlen=args.seqlen,
            num_head=args.num_head,
            head_dim=args.head_dim,
            kv_group_size=args.kv_group_size,
            topk=args.topk,
            pool_head_dim=args.pool_head_dim,
            max_splits=args.max_splits,
            softmax_scale=args.softmax_scale,
            m_block_size=args.m_block_size,
            n_block_size=args.n_block_size,
            num_threads=num_threads,
            pipe_depth=pipe_depth,
            ragged=args.ragged,
            decode_steps=args.decode_steps,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            skip_ref_check=args.skip_ref_check,
            export_only=args.export_only,
            output_dir=args.output_dir,
            file_name=args.file_name,
            function_prefix=args.function_prefix,
        )
        return
    run(
        dtype=args.dtype,
        batch_size=args.batch_size,
        seqlen=args.seqlen,
        num_head=args.num_head,
        head_dim=args.head_dim,
        kv_group_size=args.kv_group_size,
        topk=args.topk,
        softmax_scale=args.softmax_scale,
        m_block_size=args.m_block_size,
        n_block_size=args.n_block_size,
        num_threads=(QSA_DEFAULT_THREADS if args.num_threads is None
                     else args.num_threads),
        pipe_depth=(QSA_DEFAULT_PIPE_DEPTH if args.pipe_depth is None
                    else args.pipe_depth),
        ragged=args.ragged,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        skip_ref_check=args.skip_ref_check,
        export_only=args.export_only,
        output_dir=args.output_dir,
        file_name=args.file_name,
        function_prefix=args.function_prefix,
    )


if __name__ == "__main__":
    _parsed_args = _parse_args(_saved_argv)
    main()
    print("PASS")
