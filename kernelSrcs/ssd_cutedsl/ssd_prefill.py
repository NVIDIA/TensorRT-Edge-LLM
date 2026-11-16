# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Origin: Translated from Mamba SSD Triton chunk scan kernels (Apache-2.0):
#   https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/
#   Files: ssd_chunk_state.py, ssd_chunk_scan.py, ssd_state_passing.py, ssd_bmm.py
# Copyright (c) 2024 Tri Dao, Albert Gu
# SPDX-License-Identifier: Apache-2.0
#
# Modifications by NVIDIA:
# - Line-by-line translation from Triton to CuTe DSL
# - tl.dot replaced with shared memory tiled matmul (scalar FMA from smem)
# - Triton autotuned block sizes replaced with fixed tile sizes
# - Adapted for Ampere+ (SM80+), no TMA/TMEM
# - Tensor layouts adapted for TensorRT Edge-LLM

"""SSD chunk-scan: 5 CuteDSL kernels translated from Triton with tiled matmul.

Triton source: kernelSrcs/triton_reference/

Pipeline:
  1. cumsum        (ssd_chunk_state.py L40-86)   → dA_cumsum, dt_processed
  2. chunk_state   (ssd_chunk_state.py L192-268) → per-chunk states [b,c,h,dim,dstate]
  3. state_passing (ssd_state_passing.py L30-87)  → prev_states (state before each chunk)
  4. bmm           (ssd_bmm.py L37-103)          → CB = C @ B^T [b,c,g,L,L]
  5. chunk_scan    (ssd_chunk_scan.py L49-180)    → output [b,seq,h,dim]

Usage:
  python ssd_chunk_scan.py --n 1 --nheads 8 --seq_len 1024
"""

import argparse
import os
import sys
import time

_saved_argv = None
# DIM / DSTATE pinned at module-import time -- overridable via --dim/--dstate CLI
# when run as __main__ (AOT build path) or via SSD_DIM/SSD_DSTATE env vars when
# imported by an out-of-tree harness.
_early_dim = int(os.environ.get("SSD_DIM", "128"))
_early_dstate = int(os.environ.get("SSD_DSTATE", "128"))
if __name__ == "__main__":
    _saved_argv = list(sys.argv)
    for i, a in enumerate(_saved_argv):
        if a == "--dim" and i + 1 < len(_saved_argv):
            _early_dim = int(_saved_argv[i + 1])
        if a == "--dstate" and i + 1 < len(_saved_argv):
            _early_dstate = int(_saved_argv[i + 1])
    sys.argv = [sys.argv[0]]

import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import cutlass.cute as cute
import numpy as np
from cutedsl_utils import aot_placeholders
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32

# ============================================================================
# Tile sizes — chosen from Triton autotune configs
# ============================================================================
CHUNK_SIZE = 128
DIM = _early_dim       # headdim (compile-time, set via --dim)
DSTATE = _early_dstate # state dim (compile-time, set via --dstate)

# Matmul tile sizes — always 128; kernels have bounds checks for DIM < 128
BM = 128         # output tile M
BN = 128         # output tile N
BK = 32          # reduction tile K (scalar FMA path)
BK_MMA = 16      # reduction tile K for MMA path (atom K=16)
THREADS = 256    # 8 warps

# Thread-to-output mapping for 4×4 sub-tile per thread (scalar FMA path):
# BM*BN = 4096, 256 threads → 16 elements/thread = 4×4
# tid → m_base = (tid // 16) * 4, n_base = (tid % 16) * 4
SUB_M = 4        # sub-tile rows per thread
SUB_N = 4        # sub-tile cols per thread
THREADS_N = BN // SUB_N   # = 16 threads in N direction

WARP_SIZE = 32


# ============================================================================
# Tensor wrapping
# ============================================================================
def wrap_nd(arr):
    ct = from_dlpack(arr, assumed_align=16)
    ndim = len(arr.shape)
    ct = ct.mark_layout_dynamic(leading_dim=ndim - 1)
    so = tuple(range(ndim))
    for mode in range(ndim - 1):
        ct = ct.mark_compact_shape_dynamic(mode=mode, stride_order=so)
    return ct


def wrap_1d(arr):
    ct = from_dlpack(arr, assumed_align=16)
    return (ct.mark_layout_dynamic(leading_dim=0)
            .mark_compact_shape_dynamic(mode=0, stride_order=(0,)))


def wrap_nd_cpasync(arr):
    """Wrap for cp.async: compact_shape_dynamic only, preserves alignment."""
    ct = from_dlpack(arr, assumed_align=32)
    ndim = len(arr.shape)
    so = tuple(range(ndim))
    for mode in range(ndim - 1):
        ct = ct.mark_compact_shape_dynamic(mode=mode, stride_order=so)
    return ct


# ============================================================================
# Kernel 1: CUMSUM — prefix sum of A*dt per chunk
# ============================================================================
# Grid: (batch, nheads, nchunks). Block: CHUNK_SIZE threads -- one per chunk position.
# Hillis-Steele in-place prefix scan over CHUNK_SIZE values (128 -> 7 doubling steps).

@cute.kernel
def cumsum_kernel(
    dt_in: cute.Tensor,       # [batch, seq_len, nheads] fp16
    A: cute.Tensor,           # [nheads] fp32
    dt_bias: cute.Tensor,     # [nheads] fp16 (matches plugin kIN_DT_BIAS_IDX type)
    dA_cumsum: cute.Tensor,   # [batch, nheads, nchunks, CHUNK_SIZE] fp32
    dt_out: cute.Tensor,      # [batch, nheads, nchunks, CHUNK_SIZE] fp32
    context_lengths: cute.Tensor,  # [batch] int32 -- per-batch effective seq_len (uniform=seq_len)
    dt_softplus: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    b, h, c = cute.arch.block_idx()
    a_val = cutlass.Float32(A[h])
    bias = cutlass.Float32(dt_bias[h])
    # Per-batch effective seq_len: tokens at t >= cl[b] are padding -> zero contribution.
    eff_seq_len = cutlass.Int32(context_lengths[b])

    # Step 1: each thread computes its dt_val (softplus(dt + bias) with cl-mask).
    t = cutlass.Int32(c) * CHUNK_SIZE + tidx
    dt_val = cutlass.Float32(0.0)
    if t < eff_seq_len:
        dt_val = cutlass.Float32(dt_in[b, t, h]) + bias
        if dt_softplus:
            if dt_val <= cutlass.Float32(20.0):
                dt_val = cute.log(cutlass.Float32(1.0) + cute.exp(dt_val))
    dt_out[b, h, c, tidx] = dt_val

    # Step 2: parallel inclusive prefix scan over a_val * dt_val across the 128 threads.
    # Hillis-Steele single-buffer with explicit barrier between read and write phases.
    smem = cutlass.utils.SmemAllocator()
    sScan = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout(CHUNK_SIZE), 16)
    sScan[tidx] = a_val * dt_val
    cute.arch.barrier()

    # 7 doubling steps cover CHUNK_SIZE = 128.
    for log_offset in cutlass.range_constexpr(7):
        offset = 1 << log_offset
        partial = cutlass.Float32(0.0)
        if tidx >= offset:
            partial = sScan[tidx - offset]
        cute.arch.barrier()
        if tidx >= offset:
            sScan[tidx] = sScan[tidx] + partial
        cute.arch.barrier()

    dA_cumsum[b, h, c, tidx] = sScan[tidx]


@cute.jit
def jit_cumsum(
    dt_in: cute.Tensor, A: cute.Tensor, dt_bias: cute.Tensor,
    dA_cumsum: cute.Tensor, dt_out: cute.Tensor,
    context_lengths: cute.Tensor,
    n_batch: Int32, nheads: Int32, nchunks: Int32,
    dt_softplus: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    cumsum_kernel(dt_in, A, dt_bias, dA_cumsum, dt_out, context_lengths, dt_softplus).launch(
        grid=(n_batch, nheads, nchunks), block=[CHUNK_SIZE, 1, 1],
        smem=cutlass.Int32(CHUNK_SIZE * 4 + 256), stream=stream)


# ============================================================================
# Kernel 2: CHUNK_STATE — B^T @ (decay * dt * X) with tiled matmul
# ============================================================================
# Translated from ssd_chunk_state.py L192-268
# Triton: acc += tl.dot(x_tile[BM,BK], scaled_B_tile[BK,BN])
# Grid: (cdiv(DIM,BM)*cdiv(DSTATE,BN), batch*nchunks, nheads)
# Block: 256 threads, each computes 4×4 output sub-tile

@cute.kernel
def chunk_state_kernel(
    x: cute.Tensor,            # [batch, seq_len, nheads, dim] fp16
    B: cute.Tensor,            # [batch, seq_len, ngroups, dstate] fp16
    dt_proc: cute.Tensor,      # [batch, nheads, nchunks, CHUNK_SIZE] fp32
    dA_cumsum: cute.Tensor,    # [batch, nheads, nchunks, CHUNK_SIZE] fp32
    states: cute.Tensor,       # [batch, nchunks, nheads, dim, dstate] fp32
    context_lengths: cute.Tensor,  # [batch] int32 -- per-batch effective seq_len
    nchunks: Int32,
    nheads_ngroups_ratio: Int32,
    tiled_mma: cute.TiledMma,
    smem_copy_A: cute.TiledCopy,
    smem_copy_B: cute.TiledCopy,
    r2s_copy_C: cute.TiledCopy,
):
    tidx, _, _ = cute.arch.thread_idx()
    tile_idx, bc, h = cute.arch.block_idx()
    b = bc // nchunks
    c = bc - b * nchunks

    num_pid_n = cutlass.Int32(((DSTATE + BN - 1) // BN))
    pid_m = tile_idx // num_pid_n
    pid_n = tile_idx - pid_m * num_pid_n
    g = h // nheads_ngroups_ratio

    dA_last = dA_cumsum[b, h, c, CHUNK_SIZE - 1]

    eff_seq_len = cutlass.Int32(context_lengths[b])
    chunk_size_limit = eff_seq_len - cutlass.Int32(c) * CHUNK_SIZE
    if chunk_size_limit > CHUNK_SIZE:
        chunk_size_limit = cutlass.Int32(CHUNK_SIZE)
    if chunk_size_limit < 0:
        chunk_size_limit = cutlass.Int32(0)

    # Shared memory: X[BM, BK_MMA] fp16, B_scaled[BN, BK_MMA] fp16, out[BM, BN] fp32
    smem = cutlass.utils.SmemAllocator()
    sX_layout = cute.make_layout((BM, BK_MMA), stride=(BK_MMA, 1))
    sB_layout = cute.make_layout((BN, BK_MMA), stride=(BK_MMA, 1))
    sOut_layout = cute.make_layout((BM, BN), stride=(BN, 1))
    sX = smem.allocate_tensor(cutlass.Float16, sX_layout, 128)
    sB_s = smem.allocate_tensor(cutlass.Float16, sB_layout, 128)
    sOut = smem.allocate_tensor(cutlass.Float32, sOut_layout, 128)

    # MMA setup
    thr_mma = tiled_mma.get_slice(tidx)
    thr_copy_A = smem_copy_A.get_slice(tidx)
    thr_copy_B = smem_copy_B.get_slice(tidx)

    tArA_mma = thr_mma.partition_A(sX)
    tBrB_mma = thr_mma.partition_B(sB_s)
    tCrC_mma = thr_mma.partition_C(sOut)

    tCrA = cute.make_fragment_like(tArA_mma)
    tCrB = cute.make_fragment_like(tBrB_mma)
    accum = cute.make_fragment_like(tCrC_mma, dtype=cutlass.Float32)
    accum.fill(cutlass.Float32(0.0))

    tAsA_src = thr_copy_A.partition_S(sX)
    tAsA_dst = smem_copy_A.retile(tCrA)
    tBsB_src = thr_copy_B.partition_S(sB_s)
    tBsB_dst = smem_copy_B.retile(tCrB)

    # K loop over chunk_size in steps of BK_MMA=16 — scalar loads with bounds checks
    for k_base in range(0, CHUNK_SIZE, BK_MMA):
        # Load X[BM, BK_MMA] as fp16
        for idx in range(tidx, BM * BK_MMA, THREADS):
            row = idx // BK_MMA
            col = idx - row * BK_MMA
            k_pos = k_base + col
            m_global = pid_m * BM + row
            t_global = cutlass.Int32(c) * CHUNK_SIZE + k_pos
            val = cutlass.Float16(0.0)
            if k_pos < chunk_size_limit and m_global < DIM:
                val = x[b, t_global, h, m_global]
            sX[(row, col)] = val

        # Load B[BN, BK_MMA] scaled by decay*dt, stored as fp16
        for idx in range(tidx, BN * BK_MMA, THREADS):
            row = idx // BK_MMA
            col = idx - row * BK_MMA
            k_pos = k_base + col
            n_global = pid_n * BN + row
            val = cutlass.Float16(0.0)
            if k_pos < chunk_size_limit and n_global < DSTATE:
                t_global = cutlass.Int32(c) * CHUNK_SIZE + k_pos
                dA_k = dA_cumsum[b, h, c, k_pos]
                diff = dA_last - dA_k
                if diff > cutlass.Float32(0.0):
                    diff = cutlass.Float32(0.0)
                scale = cute.exp(diff) * dt_proc[b, h, c, k_pos]
                val = cutlass.Float16(cutlass.Float32(B[b, t_global, g, n_global]) * scale)
            sB_s[(row, col)] = val

        cute.arch.barrier()

        cute.copy(smem_copy_A, tAsA_src, tAsA_dst)
        cute.copy(smem_copy_B, tBsB_src, tBsB_dst)
        cute.gemm(tiled_mma, accum, tCrA, tCrB, accum)

        cute.arch.barrier()

    # Epilogue: register -> smem -> global
    thr_r2s = r2s_copy_C.get_slice(tidx)
    r2s_src = r2s_copy_C.retile(accum)
    r2s_dst = thr_r2s.partition_D(sOut)
    cute.copy(r2s_copy_C, r2s_src, r2s_dst)

    cute.arch.barrier()

    for idx in range(tidx, BM * BN, THREADS):
        row = idx // BN
        col = idx - row * BN
        gm = pid_m * BM + row
        gn = pid_n * BN + col
        if gm < DIM and gn < DSTATE:
            states[b, c, h, gm, gn] = sOut[(row, col)]


@cute.jit
def jit_chunk_state(
    x: cute.Tensor, B: cute.Tensor, dt_proc: cute.Tensor, dA_cumsum: cute.Tensor,
    states: cute.Tensor, context_lengths: cute.Tensor,
    nchunks: Int32, nheads: Int32, nheads_ngroups_ratio: Int32,
    n_batch: Int32,
    stream: cuda.CUstream,
):
    from cutlass.cute.nvgpu.warp import MmaF16BF16Op
    mma_op = MmaF16BF16Op(ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32, shape_mnk=(16, 8, 16))
    tiled_mma = cute.make_tiled_mma(mma_op, atom_layout_mnk=(2, 4, 1))
    s2r_A = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
    smem_copy_A = cute.make_tiled_copy_A(s2r_A, tiled_mma)
    s2r_B = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
    smem_copy_B = cute.make_tiled_copy_B(s2r_B, tiled_mma)
    r2s = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)
    r2s_copy_C = cute.make_tiled_copy_C(r2s, tiled_mma)

    n_tiles = cutlass.Int32(((DIM + BM - 1) // BM) * ((DSTATE + BN - 1) // BN))
    smem_bytes = cutlass.Int32((BM * BK_MMA + BN * BK_MMA) * 2 + BM * BN * 4 + 256)
    chunk_state_kernel(
        x, B, dt_proc, dA_cumsum, states, context_lengths, nchunks, nheads_ngroups_ratio,
        tiled_mma, smem_copy_A, smem_copy_B, r2s_copy_C,
    ).launch(
        grid=(n_tiles, n_batch * nchunks, nheads),
        block=[THREADS, 1, 1], smem=smem_bytes, stream=stream)


# ============================================================================
# Kernel 3: STATE_PASSING — sequential scan over chunks
# ============================================================================
# Translated from ssd_state_passing.py L30-87
# Grid: (cdiv(dim, BLOCK), batch, nheads), Block: 256
# Input states: [batch, nchunks, nheads, flat_dim] (flat_dim = dim*dstate)
# Output prev_states: [batch, nchunks, nheads, flat_dim]
#   prev_states[c] = accumulated state BEFORE chunk c
# Output final_states: [batch, nheads, flat_dim]

@cute.kernel
def state_passing_kernel(
    states_in: cute.Tensor,      # [batch, nchunks, nheads, dim, dstate] fp32
    prev_states: cute.Tensor,    # [batch, nchunks, nheads, dim, dstate] fp32 — output
    final_states: cute.Tensor,   # [batch, nheads, dim, dstate] fp16 -- output (matches plugin kIN_STATE_IDX type)
    dA_cumsum: cute.Tensor,      # [batch, nheads, nchunks, CHUNK_SIZE] fp32 (read last element)
    init_states_in: cute.Tensor, # [batch, nheads, dim, dstate] fp16 -- initial SSM state (zeros for fresh request); aliased to final_states by runner
    nchunks: Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    tile_idx, b, h = cute.arch.block_idx()

    # Decode (d, ds) from tile_idx and tidx
    # Each block covers THREADS elements of the flattened dim*dstate space
    elem_idx = tile_idx * THREADS + tidx
    d = elem_idx // DSTATE
    ds = elem_idx - d * DSTATE

    if d < DIM and ds < DSTATE:
        # Seed prev_states[0] and the running accumulator from the caller-provided init state.
        # When the runner aliases init_states_in to the same buffer holding the input state, this
        # implements has_init_states for chunked prefill / continuous batching. When init_states_in
        # is zeroed (fresh request), this matches the original "start from 0" behaviour.
        init_val = cutlass.Float32(init_states_in[b, h, d, ds])
        prev_states[b, 0, h, d, ds] = init_val
        running = init_val

        for c_idx in range(nchunks):
            new_state = cutlass.Float32(states_in[b, c_idx, h, d, ds])
            # Read last cumsum value directly: dA_cumsum[b, h, c, CHUNK_SIZE-1]
            decay = cute.exp(dA_cumsum[b, h, c_idx, CHUNK_SIZE - 1])
            running = decay * running + new_state
            if c_idx < nchunks - 1:
                prev_states[b, c_idx + 1, h, d, ds] = running
            else:
                final_states[b, h, d, ds] = cutlass.Float16(running)


@cute.jit
def jit_state_passing(
    states_in: cute.Tensor, prev_states: cute.Tensor, final_states: cute.Tensor,
    dA_cumsum: cute.Tensor, init_states_in: cute.Tensor,
    nchunks: Int32, n_batch: Int32, nheads: Int32,
    stream: cuda.CUstream,
):
    flat_dim = cutlass.Int32(DIM * DSTATE)  # 128*128 = 16384
    n_blocks_m = (flat_dim + THREADS - 1) // THREADS
    state_passing_kernel(
        states_in, prev_states, final_states, dA_cumsum, init_states_in, nchunks,
    ).launch(
        grid=(n_blocks_m, n_batch, nheads), block=[THREADS, 1, 1], smem=0, stream=stream)


# ============================================================================
# Kernel 4: BMM — CB[i,j] = C[i,:] · B[j,:] (C @ B^T) with tiled matmul
# ============================================================================
# Translated from ssd_bmm.py L37-103
# Grid: (cdiv(L,BM)*cdiv(L,BN), batch*nchunks, ngroups)
# IS_CAUSAL: skip tile if pid_n * BN >= (pid_m + 1) * BM

@cute.kernel
def bmm_kernel(
    C_in: cute.Tensor,  # [batch, seq_len, ngroups, dstate] fp16
    B_in: cute.Tensor,  # [batch, seq_len, ngroups, dstate] fp16
    CB: cute.Tensor,    # [batch, nchunks, ngroups, CHUNK_SIZE, CHUNK_SIZE] fp32
    context_lengths: cute.Tensor,  # [batch] int32
    nchunks: Int32,
    tiled_mma: cute.TiledMma,
    smem_copy_A: cute.TiledCopy,
    smem_copy_B: cute.TiledCopy,
    r2s_copy_C: cute.TiledCopy,
    tiled_copy_g2s: cute.TiledCopy,  # cp.async G2S copy
):
    tidx, _, _ = cute.arch.thread_idx()
    tile_idx, bc, g = cute.arch.block_idx()
    b = bc // nchunks
    c = bc - b * nchunks

    num_pid_n = cutlass.Int32(CHUNK_SIZE // BN)
    pid_m = tile_idx // num_pid_n
    pid_n = tile_idx - pid_m * num_pid_n

    eff_seq_len = cutlass.Int32(context_lengths[b])
    chunk_size_limit = eff_seq_len - cutlass.Int32(c) * CHUNK_SIZE
    if chunk_size_limit > CHUNK_SIZE:
        chunk_size_limit = cutlass.Int32(CHUNK_SIZE)
    if chunk_size_limit < 0:
        chunk_size_limit = cutlass.Int32(0)

    # Shared memory: C_tile[BM, BK_MMA] fp16, B_tile[BN, BK_MMA] fp16, out[BM, BN] fp32
    smem = cutlass.utils.SmemAllocator()
    sA_layout = cute.make_layout((BM, BK_MMA), stride=(BK_MMA, 1))
    sB_layout = cute.make_layout((BN, BK_MMA), stride=(BK_MMA, 1))
    sOut_layout = cute.make_layout((BM, BN), stride=(BN, 1))
    sA = smem.allocate_tensor(cutlass.Float16, sA_layout, 128)
    sB = smem.allocate_tensor(cutlass.Float16, sB_layout, 128)
    sOut = smem.allocate_tensor(cutlass.Float32, sOut_layout, 128)

    # MMA partitions
    thr_mma = tiled_mma.get_slice(tidx)
    thr_copy_A = smem_copy_A.get_slice(tidx)
    thr_copy_B = smem_copy_B.get_slice(tidx)

    tArA_mma = thr_mma.partition_A(sA)
    tBrB_mma = thr_mma.partition_B(sB)
    tCrC_mma = thr_mma.partition_C(sOut)

    tCrA = cute.make_fragment_like(tArA_mma)
    tCrB = cute.make_fragment_like(tBrB_mma)
    accum = cute.make_fragment_like(tCrC_mma, dtype=cutlass.Float32)
    accum.fill(cutlass.Float32(0.0))

    tAsA_src = thr_copy_A.partition_S(sA)
    tAsA_dst = smem_copy_A.retile(tCrA)
    tBsB_src = thr_copy_B.partition_S(sB)
    tBsB_dst = smem_copy_B.retile(tCrB)

    # cp.async setup: 2D views for vectorized global→smem loads
    thr_cp = tiled_copy_g2s.get_slice(tidx)
    C_2d = C_in[(b, None, g, None)]  # [seq_len, dstate]
    B_2d = B_in[(b, None, g, None)]  # [seq_len, dstate]
    thr_sA_dst = thr_cp.partition_D(sA)
    thr_sB_dst = thr_cp.partition_D(sB)

    # K loop over dstate in steps of BK_MMA=16
    for k_base in range(0, DSTATE, BK_MMA):
        k_tile = k_base // BK_MMA

        if chunk_size_limit >= CHUNK_SIZE:
            # cp.async path: full chunk, all elements in bounds
            seq_tile_C = c * cutlass.Int32(CHUNK_SIZE // BM) + pid_m
            seq_tile_B = c * cutlass.Int32(CHUNK_SIZE // BN) + pid_n
            gC = cute.local_tile(C_2d, (BM, BK_MMA), (seq_tile_C, k_tile))
            gB = cute.local_tile(B_2d, (BN, BK_MMA), (seq_tile_B, k_tile))
            cute.copy(tiled_copy_g2s, thr_cp.partition_S(gC), thr_sA_dst)
            cute.copy(tiled_copy_g2s, thr_cp.partition_S(gB), thr_sB_dst)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
        else:
            # Scalar path: last partial chunk, needs bounds checking
            for idx in range(tidx, BM * BK_MMA, THREADS):
                row = idx // BK_MMA
                col = idx - row * BK_MMA
                m_seq = cutlass.Int32(c) * CHUNK_SIZE + pid_m * BM + row
                ds = k_base + col
                val = cutlass.Float16(0.0)
                if pid_m * BM + row < chunk_size_limit and ds < DSTATE:
                    val = C_in[b, m_seq, g, ds]
                sA[(row, col)] = val

            for idx in range(tidx, BN * BK_MMA, THREADS):
                row = idx // BK_MMA
                col = idx - row * BK_MMA
                n_seq = cutlass.Int32(c) * CHUNK_SIZE + pid_n * BN + row
                ds = k_base + col
                val = cutlass.Float16(0.0)
                if pid_n * BN + row < chunk_size_limit and ds < DSTATE:
                    val = B_in[b, n_seq, g, ds]
                sB[(row, col)] = val

        cute.arch.barrier()

        # smem -> register
        cute.copy(smem_copy_A, tAsA_src, tAsA_dst)
        cute.copy(smem_copy_B, tBsB_src, tBsB_dst)

        # Tensor Core MMA: accum += C_tile @ B_tile^T
        cute.gemm(tiled_mma, accum, tCrA, tCrB, accum)

        cute.arch.barrier()

    # Epilogue: register -> smem -> global
    thr_r2s = r2s_copy_C.get_slice(tidx)
    r2s_src = r2s_copy_C.retile(accum)
    r2s_dst = thr_r2s.partition_D(sOut)
    cute.copy(r2s_copy_C, r2s_src, r2s_dst)

    cute.arch.barrier()

    # Store from smem to global
    for idx in range(tidx, BM * BN, THREADS):
        row = idx // BN
        col = idx - row * BN
        gm = pid_m * BM + row
        gn = pid_n * BN + col
        if gm < CHUNK_SIZE and gn < CHUNK_SIZE:
            CB[b, c, g, gm, gn] = sOut[(row, col)]


@cute.jit
def jit_bmm(
    C_in: cute.Tensor, B_in: cute.Tensor, CB: cute.Tensor,
    context_lengths: cute.Tensor,
    nchunks: Int32, n_batch: Int32, ngroups: Int32,
    stream: cuda.CUstream,
):
    from cutlass.cute.nvgpu.warp import MmaF16BF16Op
    mma_op = MmaF16BF16Op(ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32, shape_mnk=(16, 8, 16))
    tiled_mma = cute.make_tiled_mma(mma_op, atom_layout_mnk=(2, 4, 1))

    s2r_A = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
    smem_copy_A = cute.make_tiled_copy_A(s2r_A, tiled_mma)
    s2r_B = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
    smem_copy_B = cute.make_tiled_copy_B(s2r_B, tiled_mma)
    r2s = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)
    r2s_copy_C = cute.make_tiled_copy_C(r2s, tiled_mma)

    # cp.async G2S copy for [BM, BK_MMA] = [128, 16] tiles
    # 256 threads x 8 fp16 per 128-bit copy = 2048 elements = 128 x 16
    copy_atom_g2s = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float16, num_bits_per_copy=128)
    thread_layout_g2s = cute.make_layout((128, 2), stride=(2, 1))
    val_layout_g2s = cute.make_layout((1, 8))
    tiled_copy_g2s = cute.make_tiled_copy_tv(
        copy_atom_g2s, thread_layout_g2s, val_layout_g2s)

    n_tiles = cutlass.Int32((CHUNK_SIZE // BM) * (CHUNK_SIZE // BN))
    smem_bytes = cutlass.Int32((BM * BK_MMA + BN * BK_MMA) * 2 + BM * BN * 4 + 256)
    bmm_kernel(C_in, B_in, CB, context_lengths, nchunks,
               tiled_mma, smem_copy_A, smem_copy_B, r2s_copy_C,
               tiled_copy_g2s).launch(
        grid=(n_tiles, n_batch * nchunks, ngroups),
        block=[THREADS, 1, 1], smem=smem_bytes, stream=stream)


# ============================================================================
# Kernel 5: CHUNK_SCAN — output from CB, prev_states, x
# ============================================================================
# Translated from ssd_chunk_scan.py L49-180
# Part A (inter-chunk): acc = tl.dot(C[BM,BD], prev_states[BD,BN]) * exp(dA_m)
# Part B (intra-chunk): acc += tl.dot(cb_weighted[BM,BK], x[BK,BN])
# Grid: (cdiv(L,BM)*cdiv(DIM,BN), batch*nchunks, nheads)

@cute.kernel
def chunk_scan_kernel(
    CB: cute.Tensor,           # [batch, nchunks, ngroups, CHUNK_SIZE, CHUNK_SIZE] fp32
    x: cute.Tensor,            # [batch, seq_len, nheads, dim] fp16
    dt_proc: cute.Tensor,      # [batch, nheads, nchunks, CHUNK_SIZE] fp32
    dA_cumsum: cute.Tensor,    # [batch, nheads, nchunks, CHUNK_SIZE] fp32
    C_in: cute.Tensor,         # [batch, seq_len, ngroups, dstate] fp16
    prev_states: cute.Tensor,  # [batch, nchunks, nheads, dim, dstate] fp32
    D: cute.Tensor,            # [nheads] fp16 (matches plugin kIN_D_IDX type; cast to fp32 on read)
    z: cute.Tensor,            # [batch, seq_len, nheads, dim] fp16 (or dummy)
    output: cute.Tensor,       # [batch, seq_len, nheads, dim] fp16
    context_lengths: cute.Tensor,  # [batch] int32 -- per-batch effective seq_len
    nchunks: Int32,
    nheads_ngroups_ratio: Int32,
    has_D: cutlass.Constexpr[bool],
    has_z: cutlass.Constexpr[bool],
    tiled_mma: cute.TiledMma,
    smem_copy_A: cute.TiledCopy,
    smem_copy_B: cute.TiledCopy,
    r2s_copy_C: cute.TiledCopy,
):
    tidx, _, _ = cute.arch.thread_idx()
    tile_idx, bc, h = cute.arch.block_idx()
    b = bc // nchunks
    c = bc - b * nchunks
    g = h // nheads_ngroups_ratio

    num_pid_n = cutlass.Int32(((DIM + BN - 1) // BN))
    pid_m = tile_idx // num_pid_n
    pid_n = tile_idx - pid_m * num_pid_n

    eff_seq_len = cutlass.Int32(context_lengths[b])
    chunk_size_limit = eff_seq_len - cutlass.Int32(c) * CHUNK_SIZE
    if chunk_size_limit > CHUNK_SIZE:
        chunk_size_limit = cutlass.Int32(CHUNK_SIZE)
    if chunk_size_limit < 0:
        chunk_size_limit = cutlass.Int32(0)

    # Shared memory: A[BM, BK_MMA] fp16, B[BN, BK_MMA] fp16, Out[BM, BN] fp32
    smem = cutlass.utils.SmemAllocator()
    sA_layout = cute.make_layout((BM, BK_MMA), stride=(BK_MMA, 1))
    sB_layout = cute.make_layout((BN, BK_MMA), stride=(BK_MMA, 1))
    sOut_layout = cute.make_layout((BM, BN), stride=(BN, 1))
    sA = smem.allocate_tensor(cutlass.Float16, sA_layout, 128)
    sB_s = smem.allocate_tensor(cutlass.Float16, sB_layout, 128)
    sOut = smem.allocate_tensor(cutlass.Float32, sOut_layout, 128)

    # MMA setup
    thr_mma = tiled_mma.get_slice(tidx)
    thr_copy_A = smem_copy_A.get_slice(tidx)
    thr_copy_B = smem_copy_B.get_slice(tidx)

    tArA_mma = thr_mma.partition_A(sA)
    tBrB_mma = thr_mma.partition_B(sB_s)
    tCrC_mma = thr_mma.partition_C(sOut)

    tCrA = cute.make_fragment_like(tArA_mma)
    tCrB = cute.make_fragment_like(tBrB_mma)
    accum = cute.make_fragment_like(tCrC_mma, dtype=cutlass.Float32)
    accum.fill(cutlass.Float32(0.0))

    tAsA_src = thr_copy_A.partition_S(sA)
    tAsA_dst = smem_copy_A.retile(tCrA)
    tBsB_src = thr_copy_B.partition_S(sB_s)
    tBsB_dst = smem_copy_B.retile(tCrB)

    # ---- Part A: Inter-chunk — C_scaled[BM,dstate] @ prev_states[dstate,BN] ----
    # Fuse exp(dA_m) scaling into C_in load: C_scaled[m,ds] = C[m,ds] * exp(dA_m)
    # This avoids needing to post-scale the MMA accumulator.
    for k_base in range(0, DSTATE, BK_MMA):
        # Load C_in * exp(dA) as fp16
        for idx in range(tidx, BM * BK_MMA, THREADS):
            row = idx // BK_MMA
            col = idx - row * BK_MMA
            m_global = pid_m * BM + row
            t_seq = cutlass.Int32(c) * CHUNK_SIZE + m_global
            ds = k_base + col
            val = cutlass.Float16(0.0)
            if m_global < chunk_size_limit and ds < DSTATE:
                scale = cute.exp(dA_cumsum[b, h, c, m_global])
                val = cutlass.Float16(cutlass.Float32(C_in[b, t_seq, g, ds]) * scale)
            sA[(row, col)] = val

        # Load prev_states as fp16
        for idx in range(tidx, BN * BK_MMA, THREADS):
            row = idx // BK_MMA
            col = idx - row * BK_MMA
            n_global = pid_n * BN + row
            ds = k_base + col
            val = cutlass.Float16(0.0)
            if ds < DSTATE and n_global < DIM:
                val = cutlass.Float16(prev_states[b, c, h, n_global, ds])
            sB_s[(row, col)] = val

        cute.arch.barrier()
        cute.copy(smem_copy_A, tAsA_src, tAsA_dst)
        cute.copy(smem_copy_B, tBsB_src, tBsB_dst)
        cute.gemm(tiled_mma, accum, tCrA, tCrB, accum)
        cute.arch.barrier()

    # ---- Part B: Intra-chunk — CB_weighted[BM,BK] @ X[BK,BN] ----
    # accum already contains Part A result; Part B adds to it.
    k_max = (pid_m + 1) * BM
    if k_max > chunk_size_limit:
        k_max = chunk_size_limit

    for k_base in range(0, k_max, BK_MMA):
        # Load CB_weighted[BM, BK_MMA] as fp16 with decay+causal mask
        for idx in range(tidx, BM * BK_MMA, THREADS):
            row = idx // BK_MMA
            col = idx - row * BK_MMA
            m_pos = pid_m * BM + row
            k_pos = k_base + col
            val = cutlass.Float16(0.0)
            if m_pos < CHUNK_SIZE and k_pos < CHUNK_SIZE and k_pos < chunk_size_limit and m_pos >= k_pos:
                cb_val = CB[b, c, g, m_pos, k_pos]
                dA_m = dA_cumsum[b, h, c, m_pos]
                dA_k = dA_cumsum[b, h, c, k_pos]  # k varies per iteration, can't cache
                diff = dA_m - dA_k
                if diff > cutlass.Float32(0.0):
                    diff = cutlass.Float32(0.0)
                val = cutlass.Float16(cb_val * cute.exp(diff) * dt_proc[b, h, c, k_pos])
            sA[(row, col)] = val

        # Load X as fp16 into sB_s[BN, BK_MMA]
        for idx in range(tidx, BN * BK_MMA, THREADS):
            row = idx // BK_MMA
            col = idx - row * BK_MMA
            k_pos = k_base + col
            n_global = pid_n * BN + row
            val = cutlass.Float16(0.0)
            k_seq = cutlass.Int32(c) * CHUNK_SIZE + k_pos
            if k_pos < chunk_size_limit and n_global < DIM:
                val = x[b, k_seq, h, n_global]
            sB_s[(row, col)] = val

        cute.arch.barrier()
        cute.copy(smem_copy_A, tAsA_src, tAsA_dst)
        cute.copy(smem_copy_B, tBsB_src, tBsB_dst)
        cute.gemm(tiled_mma, accum, tCrA, tCrB, accum)
        cute.arch.barrier()

    # Store accum (Part A + Part B) to sOut
    thr_r2s = r2s_copy_C.get_slice(tidx)
    r2s_src = r2s_copy_C.retile(accum)
    r2s_dst = thr_r2s.partition_D(sOut)
    cute.copy(r2s_copy_C, r2s_src, r2s_dst)
    cute.arch.barrier()

    # ---- Part C + D + Store: operate on sOut[BM, BN] in smem ----
    # D residual, z-gating, and store to global — all work from sOut
    for idx in range(tidx, BM * BN, THREADS):
        row = idx // BN
        col = idx - row * BN
        gm = pid_m * BM + row
        gn = pid_n * BN + col
        if gm < chunk_size_limit and gn < DIM:
            val = sOut[(row, col)]
            # D residual
            if has_D:
                m_seq = cutlass.Int32(c) * CHUNK_SIZE + gm
                val = val + cutlass.Float32(D[h]) * cutlass.Float32(x[b, m_seq, h, gn])
            # z-gating (SiLU)
            if has_z:
                m_seq_z = cutlass.Int32(c) * CHUNK_SIZE + gm
                z_val = cutlass.Float32(z[b, m_seq_z, h, gn])
                sig_z = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(cutlass.Float32(0.0) - z_val))
                val = val * z_val * sig_z
            # Store to global
            output[b, cutlass.Int32(c) * CHUNK_SIZE + gm, h, gn] = cutlass.Float16(val)
        elif gn < DIM and cutlass.Int32(c) * CHUNK_SIZE + gm < output.shape[1]:
            output[b, cutlass.Int32(c) * CHUNK_SIZE + gm, h, gn] = cutlass.Float16(0.0)


@cute.jit
def jit_chunk_scan(
    CB: cute.Tensor, x: cute.Tensor, dt_proc: cute.Tensor, dA_cumsum: cute.Tensor,
    C_in: cute.Tensor, prev_states: cute.Tensor, D: cute.Tensor, z: cute.Tensor,
    output: cute.Tensor, context_lengths: cute.Tensor,
    nchunks: Int32, nheads: Int32, nheads_ngroups_ratio: Int32,
    n_batch: Int32,
    has_D: cutlass.Constexpr[bool],
    has_z: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    from cutlass.cute.nvgpu.warp import MmaF16BF16Op
    mma_op = MmaF16BF16Op(ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32, shape_mnk=(16, 8, 16))
    tiled_mma = cute.make_tiled_mma(mma_op, atom_layout_mnk=(2, 4, 1))
    s2r_A = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
    smem_copy_A = cute.make_tiled_copy_A(s2r_A, tiled_mma)
    s2r_B = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
    smem_copy_B = cute.make_tiled_copy_B(s2r_B, tiled_mma)
    r2s = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)
    r2s_copy_C = cute.make_tiled_copy_C(r2s, tiled_mma)

    n_tiles = cutlass.Int32(((CHUNK_SIZE + BM - 1) // BM) * ((DIM + BN - 1) // BN))
    smem_bytes = cutlass.Int32((BM * BK_MMA + BN * BK_MMA) * 2 + BM * BN * 4 + 256)
    chunk_scan_kernel(
        CB, x, dt_proc, dA_cumsum, C_in, prev_states, D, z, output,
        context_lengths, nchunks, nheads_ngroups_ratio, has_D, has_z,
        tiled_mma, smem_copy_A, smem_copy_B, r2s_copy_C,
    ).launch(
        grid=(n_tiles, n_batch * nchunks, nheads),
        block=[THREADS, 1, 1], smem=smem_bytes, stream=stream)


# ============================================================================
# NumPy reference (serial scan)
# ============================================================================
def _ssd_reference(x, dt, A, B, C, D, dt_bias, state, dt_softplus=True):
    batch, seq_len, nheads, dim = x.shape
    _, _, ngroups, dstate = B.shape
    hpg = nheads // ngroups
    output = np.zeros_like(x, dtype=np.float32)
    state = state.copy().astype(np.float32)
    for b_ in range(batch):
        for t_ in range(seq_len):
            for h_ in range(nheads):
                g_ = h_ // hpg
                dtv = float(dt[b_, t_, h_]) + float(dt_bias[h_])
                if dt_softplus and dtv <= 20.0:
                    dtv = np.log(1.0 + np.exp(dtv))
                da = np.exp(float(A[h_]) * dtv)
                for d_ in range(dim):
                    xv = float(x[b_, t_, h_, d_])
                    yv = float(D[h_]) * xv
                    for ds_ in range(dstate):
                        state[b_, h_, d_, ds_] = da * state[b_, h_, d_, ds_] + dtv * float(B[b_, t_, g_, ds_]) * xv
                        yv += float(C[b_, t_, g_, ds_]) * state[b_, h_, d_, ds_]
                    output[b_, t_, h_, d_] = yv
    return output, state


# ============================================================================
# Pipeline
# ============================================================================
def run_pipeline(n, nheads, dim, dstate, ngroups, seq_len, warmup=3, iterations=10):
    cp.cuda.Device(0).use()
    nchunks = (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
    nheads_ngroups_ratio = nheads // ngroups
    flat_dim = dim * dstate
    print(f"[ssd_chunk_scan] n={n} h={nheads} d={dim} ds={dstate} g={ngroups} "
          f"s={seq_len} chunks={nchunks}")

    np.random.seed(42)
    x_np = np.random.randn(n, seq_len, nheads, dim).astype(np.float16)
    dt_np = (np.random.rand(n, seq_len, nheads) * 0.5 + 0.1).astype(np.float16)
    A_np = -(np.random.rand(nheads).astype(np.float32) * 0.5 + 0.5)
    B_np = np.random.randn(n, seq_len, ngroups, dstate).astype(np.float16)
    C_np = np.random.randn(n, seq_len, ngroups, dstate).astype(np.float16)
    D_np = (np.random.randn(nheads) * 0.1).astype(np.float16)  # plugin contract: fp16
    dt_bias_np = (np.random.randn(nheads) * 0.1).astype(np.float16)  # plugin contract: fp16
    state_np = np.zeros((n, nheads, dim, dstate), dtype=np.float16)  # plugin contract: fp16

    print("  NumPy reference...")
    ref_out, _ = _ssd_reference(x_np, dt_np, A_np, B_np, C_np, D_np, dt_bias_np, state_np)

    # GPU arrays
    x_g = cp.array(x_np); dt_g = cp.array(dt_np); A_g = cp.array(A_np)
    B_g = cp.array(B_np); C_g = cp.array(C_np); D_g = cp.array(D_np)
    dtb_g = cp.array(dt_bias_np); state_g = cp.array(state_np)
    out_g = cp.zeros((n, seq_len, nheads, dim), dtype=cp.float16)

    # Intermediates
    dA_g = cp.zeros((n, nheads, nchunks, CHUNK_SIZE), dtype=cp.float32)
    dtp_g = cp.zeros((n, nheads, nchunks, CHUNK_SIZE), dtype=cp.float32)
    sts_g = cp.zeros((n, nchunks, nheads, dim, dstate), dtype=cp.float32)
    prev_states_g = cp.zeros((n, nchunks, nheads, dim, dstate), dtype=cp.float32)
    final_states_g = cp.zeros((n, nheads, dim, dstate), dtype=cp.float16)  # plugin contract: fp16
    CB_g = cp.zeros((n, nchunks, ngroups, CHUNK_SIZE, CHUNK_SIZE), dtype=cp.float32)

    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)

    print("  Compiling 5 kernels...")
    t0 = time.time()

    # Uniform context_lengths placeholder (cl[b] = seq_len for all b in non-varlen test runs).
    cl_g = cp.full((n,), seq_len, dtype=cp.int32)

    c1 = cute.compile(jit_cumsum,
        wrap_nd(dt_g), wrap_1d(A_g), wrap_1d(dtb_g), wrap_nd(dA_g), wrap_nd(dtp_g),
        wrap_1d(cl_g), n, nheads, nchunks, dt_softplus=True, stream=stream)

    c2 = cute.compile(jit_chunk_state,
        wrap_nd(x_g), wrap_nd(B_g), wrap_nd(dtp_g), wrap_nd(dA_g),
        wrap_nd(sts_g), wrap_1d(cl_g), nchunks, nheads, nheads_ngroups_ratio, n,
        stream=stream)

    c3 = cute.compile(jit_state_passing,
        wrap_nd(sts_g), wrap_nd(prev_states_g), wrap_nd(final_states_g), wrap_nd(dA_g),
        wrap_nd(state_g),
        nchunks, n, nheads, stream=stream)

    c4 = cute.compile(jit_bmm,
        wrap_nd_cpasync(C_g), wrap_nd_cpasync(B_g), wrap_nd(CB_g),
        wrap_1d(cl_g), nchunks, n, ngroups, stream=stream)

    # z tensor (dummy zeros if not used)
    z_g = cp.zeros((n, seq_len, nheads, dim), dtype=cp.float16)

    c5 = cute.compile(jit_chunk_scan,
        wrap_nd(CB_g), wrap_nd(x_g), wrap_nd(dtp_g), wrap_nd(dA_g),
        wrap_nd(C_g), wrap_nd(prev_states_g), wrap_1d(D_g), wrap_nd(z_g), wrap_nd(out_g),
        wrap_1d(cl_g), nchunks, nheads, nheads_ngroups_ratio, n,
        has_D=True, has_z=False, stream=stream)

    print(f"  Compilation: {time.time() - t0:.2f}s")

    # Pre-wrap tensors once (avoid Python overhead in benchmark loop)
    w_dt = wrap_nd(dt_g); w_A = wrap_1d(A_g); w_dtb = wrap_1d(dtb_g)
    w_dA = wrap_nd(dA_g); w_dtp = wrap_nd(dtp_g)
    w_x = wrap_nd(x_g); w_B = wrap_nd(B_g); w_C = wrap_nd(C_g)
    w_sts = wrap_nd(sts_g); w_prev = wrap_nd(prev_states_g)
    w_final = wrap_nd(final_states_g); w_CB = wrap_nd(CB_g)
    w_state_init = wrap_nd(state_g)
    w_D = wrap_1d(D_g); w_z = wrap_nd(z_g); w_out = wrap_nd(out_g)
    w_x_cp = wrap_nd_cpasync(x_g)
    w_C_cp = wrap_nd_cpasync(C_g); w_B_cp = wrap_nd_cpasync(B_g)
    w_cl = wrap_1d(cl_g)

    def run():
        # Reset intermediates
        dA_g.fill(0); dtp_g.fill(0); sts_g.fill(0); CB_g.fill(0); out_g.fill(0)
        prev_states_g.fill(0); final_states_g.fill(0)

        c1(w_dt, w_A, w_dtb, w_dA, w_dtp, w_cl, n, nheads, nchunks, stream)
        c2(w_x, w_B, w_dtp, w_dA, w_sts, w_cl, nchunks, nheads, nheads_ngroups_ratio, n, stream)
        c3(w_sts, w_prev, w_final, w_dA, w_state_init, nchunks, n, nheads, stream)
        c4(w_C_cp, w_B_cp, w_CB, w_cl, nchunks, n, ngroups, stream)
        c5(w_CB, w_x, w_dtp, w_dA, w_C, w_prev, w_D, w_z, w_out,
           w_cl, nchunks, nheads, nheads_ngroups_ratio, n, stream)

    print("  Running...")
    run()
    cp.cuda.get_current_stream().synchronize()

    gpu_out = out_g.get().astype(np.float32)
    max_diff = np.max(np.abs(gpu_out - ref_out))
    ref_max = np.max(np.abs(ref_out))
    rel_err = max_diff / (ref_max + 1e-8)
    print(f"  Max diff: {max_diff:.6f}, Ref max: {ref_max:.6f}, Rel err: {rel_err:.6f}")
    print(f"  {'PASSED' if rel_err < 0.05 else 'FAILED'}")

    if iterations > 0:
        for _ in range(warmup): run()
        cp.cuda.get_current_stream().synchronize()
        t0 = time.time()
        for _ in range(iterations): run()
        cp.cuda.get_current_stream().synchronize()
        ms = (time.time() - t0) / iterations * 1000
        print(f"  Avg latency: {ms:.3f} ms")


# ============================================================================
# Combined JIT — single function that launches all 5 kernels (for AOT export)
# ============================================================================
_combined_jit = None


def _create_combined_jit():
    from cutlass.cute.nvgpu import cpasync as _cpasync
    from cutlass.cute.nvgpu.warp import MmaF16BF16Op as _MmaOp

    k_cumsum = cumsum_kernel
    k_chunk_state = chunk_state_kernel
    k_state_passing = state_passing_kernel
    k_bmm = bmm_kernel
    k_chunk_scan = chunk_scan_kernel

    @cute.jit
    def ssd_combined(
        # Primary inputs
        x: cute.Tensor,            # [batch, seq_len, nheads, dim] fp16
        dt_in: cute.Tensor,        # [batch, seq_len, nheads] fp16
        A: cute.Tensor,            # [nheads] fp32
        B: cute.Tensor,            # [batch, seq_len, ngroups, dstate] fp16
        C: cute.Tensor,            # [batch, seq_len, ngroups, dstate] fp16
        D: cute.Tensor,            # [nheads] fp16 (plugin contract)
        dt_bias: cute.Tensor,      # [nheads] fp16 (plugin contract)
        z: cute.Tensor,            # [batch, seq_len, nheads, dim] fp16 (dummy if has_z=False)
        output: cute.Tensor,       # [batch, seq_len, nheads, dim] fp16
        final_states: cute.Tensor, # [batch, nheads, dim, dstate] fp16 (plugin contract)
        init_states: cute.Tensor,  # [batch, nheads, dim, dstate] fp16 -- initial SSM state
                                   # (runner aliases this to final_states buffer to enable
                                   # has_init_states for chunked prefill / continuous batching)
        # Intermediate buffers (pre-allocated by caller)
        dA_cumsum: cute.Tensor,    # [batch, nheads, nchunks, CHUNK_SIZE] fp32
        dt_proc: cute.Tensor,      # [batch, nheads, nchunks, CHUNK_SIZE] fp32
        states: cute.Tensor,       # [batch, nchunks, nheads, dim, dstate] fp32
        prev_states: cute.Tensor,  # [batch, nchunks, nheads, dim, dstate] fp32
        CB: cute.Tensor,           # [batch, nchunks, ngroups, CHUNK_SIZE, CHUNK_SIZE] fp32
        # Varlen metadata (always present; uniform when no varlen)
        context_lengths: cute.Tensor,  # [batch] int32 -- per-batch effective seq_len
        # Scalar params (seq_len removed; per-batch length is read from context_lengths inside kernels)
        nchunks: Int32,
        n_batch: Int32,
        nheads_val: Int32,
        nheads_ngroups_ratio: Int32,
        ngroups: Int32,
        # Constexpr
        dt_softplus: cutlass.Constexpr[bool],
        has_D: cutlass.Constexpr[bool],
        has_z: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        # Step 1: cumsum (uses cl[b] for per-batch effective seq_len -> padding tokens contribute 0)
        k_cumsum(dt_in, A, dt_bias, dA_cumsum, dt_proc, context_lengths, dt_softplus).launch(
            grid=(n_batch, nheads_val, nchunks), block=[CHUNK_SIZE, 1, 1],
            smem=cutlass.Int32(CHUNK_SIZE * 4 + 256), stream=stream)

        # Step 2: chunk_state (MMA + cp.async)
        cs_mma_op = _MmaOp(ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32, shape_mnk=(16, 8, 16))
        cs_tiled_mma = cute.make_tiled_mma(cs_mma_op, atom_layout_mnk=(2, 4, 1))
        cs_s2r_A = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
        cs_copy_A = cute.make_tiled_copy_A(cs_s2r_A, cs_tiled_mma)
        cs_s2r_B = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
        cs_copy_B = cute.make_tiled_copy_B(cs_s2r_B, cs_tiled_mma)
        cs_r2s = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)
        cs_r2s_C = cute.make_tiled_copy_C(cs_r2s, cs_tiled_mma)
        n_tiles_cs = cutlass.Int32(((DIM + BM - 1) // BM) * ((DSTATE + BN - 1) // BN))
        smem_cs = cutlass.Int32((BM * BK_MMA + BN * BK_MMA) * 2 + BM * BN * 4 + 256)
        k_chunk_state(x, B, dt_proc, dA_cumsum, states, context_lengths, nchunks, nheads_ngroups_ratio,
                      cs_tiled_mma, cs_copy_A, cs_copy_B, cs_r2s_C).launch(
            grid=(n_tiles_cs, n_batch * nchunks, nheads_val),
            block=[THREADS, 1, 1], smem=smem_cs, stream=stream)

        # Step 3: state_passing (reads dA_cumsum[:,:,:,-1] directly)
        flat_dim = cutlass.Int32(DIM * DSTATE)
        n_blocks_sp = (flat_dim + THREADS - 1) // THREADS
        k_state_passing(states, prev_states, final_states, dA_cumsum, init_states, nchunks).launch(
            grid=(n_blocks_sp, n_batch, nheads_val), block=[THREADS, 1, 1], smem=0, stream=stream)

        # Step 4: bmm (MMA + cp.async)
        bmm_mma_op = _MmaOp(ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32, shape_mnk=(16, 8, 16))
        bmm_tiled_mma = cute.make_tiled_mma(bmm_mma_op, atom_layout_mnk=(2, 4, 1))
        bmm_s2r_A = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
        bmm_copy_A = cute.make_tiled_copy_A(bmm_s2r_A, bmm_tiled_mma)
        bmm_s2r_B = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
        bmm_copy_B = cute.make_tiled_copy_B(bmm_s2r_B, bmm_tiled_mma)
        bmm_r2s = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)
        bmm_r2s_C = cute.make_tiled_copy_C(bmm_r2s, bmm_tiled_mma)
        bmm_g2s_atom = cute.make_copy_atom(
            _cpasync.CopyG2SOp(cache_mode=_cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float16, num_bits_per_copy=128)
        bmm_g2s_thr = cute.make_layout((128, 2), stride=(2, 1))
        bmm_g2s_val = cute.make_layout((1, 8))
        bmm_tiled_copy_g2s = cute.make_tiled_copy_tv(
            bmm_g2s_atom, bmm_g2s_thr, bmm_g2s_val)
        n_tiles_bmm = cutlass.Int32((CHUNK_SIZE // BM) * (CHUNK_SIZE // BN))
        smem_bmm = cutlass.Int32((BM * BK_MMA + BN * BK_MMA) * 2 + BM * BN * 4 + 256)
        k_bmm(C, B, CB, context_lengths, nchunks,
              bmm_tiled_mma, bmm_copy_A, bmm_copy_B, bmm_r2s_C,
              bmm_tiled_copy_g2s).launch(
            grid=(n_tiles_bmm, n_batch * nchunks, ngroups),
            block=[THREADS, 1, 1], smem=smem_bmm, stream=stream)

        # Step 5: chunk_scan (MMA)
        scan_mma_op = _MmaOp(ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32, shape_mnk=(16, 8, 16))
        scan_tiled_mma = cute.make_tiled_mma(scan_mma_op, atom_layout_mnk=(2, 4, 1))
        scan_s2r_A = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
        scan_copy_A = cute.make_tiled_copy_A(scan_s2r_A, scan_tiled_mma)
        scan_s2r_B = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
        scan_copy_B = cute.make_tiled_copy_B(scan_s2r_B, scan_tiled_mma)
        scan_r2s = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)
        scan_r2s_C = cute.make_tiled_copy_C(scan_r2s, scan_tiled_mma)
        n_tiles_scan = cutlass.Int32(((CHUNK_SIZE + BM - 1) // BM) * ((DIM + BN - 1) // BN))
        smem_scan = cutlass.Int32((BM * BK_MMA + BN * BK_MMA) * 2 + BM * BN * 4 + 256)
        k_chunk_scan(CB, x, dt_proc, dA_cumsum, C, prev_states, D, z, output,
                     context_lengths, nchunks, nheads_ngroups_ratio, has_D, has_z,
                     scan_tiled_mma, scan_copy_A, scan_copy_B, scan_r2s_C).launch(
            grid=(n_tiles_scan, n_batch * nchunks, nheads_val),
            block=[THREADS, 1, 1], smem=smem_scan, stream=stream)

    return ssd_combined


def _get_combined_jit():
    global _combined_jit
    if _combined_jit is None:
        _combined_jit = _create_combined_jit()
    return _combined_jit


# ============================================================================
# AOT Export — single combined kernel
# ============================================================================
# AOT placeholders (minimal shapes for compilation)
AOT_N = 1
AOT_NHEADS = 8
AOT_DIM = DIM       # uses the --dim setting
AOT_DSTATE = DSTATE # uses the --dstate setting
AOT_NGROUPS = 1
AOT_SEQLEN = 128


def _make_aot_placeholders(n, nheads, dim, dstate, ngroups, seq_len):
    """Create storage-free descriptors matching the pipeline's intermediate shapes."""
    nchunks = (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE

    def make_nd(dtype, shape, *, layout_dynamic=True, assumed_align=16):
        tensor = aot_placeholders.make_compact_tensor(dtype, shape, assumed_align=assumed_align)
        ndim = len(shape)
        if layout_dynamic:
            tensor = tensor.mark_layout_dynamic(leading_dim=ndim - 1)
        stride_order = tuple(range(ndim))
        for mode in range(ndim - 1):
            tensor = tensor.mark_compact_shape_dynamic(mode=mode, stride_order=stride_order)
        return tensor

    def make_1d(dtype, shape):
        tensor = aot_placeholders.make_compact_tensor(dtype, shape, assumed_align=16)
        return (tensor.mark_layout_dynamic(leading_dim=0)
                .mark_compact_shape_dynamic(mode=0, stride_order=(0,)))

    return {
        "x": make_nd(cutlass.Float16, (n, seq_len, nheads, dim)),
        "dt_in": make_nd(cutlass.Float16, (n, seq_len, nheads)),
        "A": make_1d(cutlass.Float32, (nheads,)),
        "dt_bias": make_1d(cutlass.Float16, (nheads,)),
        "B": make_nd(cutlass.Float16, (n, seq_len, ngroups, dstate), layout_dynamic=False,
                     assumed_align=32),
        "C": make_nd(cutlass.Float16, (n, seq_len, ngroups, dstate), layout_dynamic=False,
                     assumed_align=32),
        "D": make_1d(cutlass.Float16, (nheads,)),
        "z": make_nd(cutlass.Float16, (n, seq_len, nheads, dim)),
        "output": make_nd(cutlass.Float16, (n, seq_len, nheads, dim)),
        "dA_cumsum": make_nd(cutlass.Float32, (n, nheads, nchunks, CHUNK_SIZE)),
        "dt_proc": make_nd(cutlass.Float32, (n, nheads, nchunks, CHUNK_SIZE)),
        "states": make_nd(cutlass.Float32, (n, nchunks, nheads, dim, dstate)),
        "CB": make_nd(cutlass.Float32, (n, nchunks, ngroups, CHUNK_SIZE, CHUNK_SIZE)),
        "prev_states": make_nd(cutlass.Float32, (n, nchunks, nheads, dim, dstate)),
        "state": make_nd(cutlass.Float16, (n, nheads, dim, dstate)),
        # Initial SSM state input -- runner aliases this to the same buffer as `state`
        # (which is staged with the input state via cudaMemcpyAsync in the plugin).
        "init_states": make_nd(cutlass.Float16, (n, nheads, dim, dstate)),
        # Varlen metadata (always present, uniform when no varlen)
        "context_lengths": make_1d(cutlass.Int32, (n,)),
    }


def export_ssd_chunk_scan(n, nheads, dim, dstate, ngroups, seq_len,
                          output_dir, file_name, function_prefix, gpu_arch=""):
    """AOT compile the combined SSD chunk scan pipeline and export to a single .o/.h.

    The combined JIT launches all 5 kernels internally.
    C++ caller must pre-allocate intermediate buffers and pass them in.
    """
    nchunks = (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
    nheads_ngroups_ratio = nheads // ngroups

    stream = aot_placeholders.make_stream()
    ph = _make_aot_placeholders(n, nheads, dim, dstate, ngroups, seq_len)
    compile_opts = aot_placeholders.compile_options(
        f"--gpu-arch={gpu_arch}" if gpu_arch else "")

    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"[ssd_chunk_scan] AOT compile gpu_arch={gpu_arch or 'default'}")
    t0 = time.time()

    combined_fn = _get_combined_jit()
    compiled = cute.compile(
        combined_fn,
        # Primary inputs
        ph["x"], ph["dt_in"], ph["A"],
        ph["B"], ph["C"], ph["D"],
        ph["dt_bias"], ph["z"],
        ph["output"], ph["state"],
        # Initial state input (aliased by runner to the state buffer for has_init_states)
        ph["init_states"],
        # Intermediate buffers
        ph["dA_cumsum"], ph["dt_proc"],
        ph["states"], ph["prev_states"], ph["CB"],
        # Varlen metadata (per-batch effective seq_len)
        ph["context_lengths"],
        # Scalars (seq_len removed from kernel signature; baked into context_lengths)
        nchunks, n, nheads, nheads_ngroups_ratio, ngroups,
        # Constexpr
        dt_softplus=True, has_D=True, has_z=False,
        stream=stream, options=compile_opts,
    )

    compiled.export_to_c(
        file_path=output_dir,
        file_name=file_name,
        function_prefix=function_prefix,
    )

    elapsed = time.time() - t0
    print(f"[ssd_chunk_scan] Exported to {output_dir}/{file_name}.h and .o in {elapsed:.2f}s")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="SSD chunk-scan CuTe DSL kernels (SM80+): test and AOT export."
    )
    p.add_argument("--export_only", action="store_true",
                   help="AOT compile and export kernels, skip test")
    p.add_argument("--output_dir", type=str, default="./ssd_aot_artifacts")
    _default_name = f"ssd_prefill_d{DIM}_n{DSTATE}"
    p.add_argument("--file_name", type=str, default=_default_name)
    p.add_argument("--function_prefix", type=str, default=_default_name)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--dstate", type=int, default=128)
    p.add_argument("--ngroups", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--gpu_arch", type=str, default="",
                   help="Target GPU arch for export (e.g. sm_87). Empty = current GPU.")
    args = p.parse_args(_saved_argv[1:] if _saved_argv else [])

    if args.export_only:
        export_ssd_chunk_scan(
            n=AOT_N, nheads=AOT_NHEADS, dim=AOT_DIM, dstate=AOT_DSTATE,
            ngroups=AOT_NGROUPS, seq_len=AOT_SEQLEN,
            output_dir=args.output_dir, file_name=args.file_name,
            function_prefix=args.function_prefix,
            gpu_arch=args.gpu_arch)
    else:
        run_pipeline(args.n, args.nheads, args.dim, args.dstate, args.ngroups, args.seq_len,
                     args.warmup, args.iterations)
