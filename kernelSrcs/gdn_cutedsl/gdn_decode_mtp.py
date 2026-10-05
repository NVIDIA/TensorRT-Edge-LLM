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

# CuTe DSL GDN MTP (Multi-Token Processing) decode kernel.
#
# Used for speculative-decoding verification: the draft model produces T tokens
# per batch item; this kernel evolves the GDN recurrent state over all T steps
# and produces T output tokens per batch item.
#
# Key differences vs gdn_decode (seq_len=1):
#   1. seq_len = T >= 1; outer loop over T time steps.
#   2. Intermediate h-states are cached after each step for rollback support
#      (compile-time opt-in via cache_intermediate_states Constexpr).
#   3. Two-phase warp-specialized approach: Phase 1 precomputes sQ/sK/sG/sBeta
#      for all T steps into SMEM; Phase 2 processes V-tiles with state evolution,
#      reading per-step q/k/g/beta from SMEM (avoids T repeated global loads).
#
# Reference: FlashInfer gdn_decode_mtp.py (Apache-2.0) — adapted for edge-llm
#   AOT export pipeline (CuPy only, no PyTorch).
#   Source: https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gdn_kernels/gdn_decode_mtp.py
#   SGLang integration: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py
#
# Test:  python gdn_decode_mtp.py --n 4 --h 8 --hv 8 --seq_len 4
# AOT:   python gdn_decode_mtp.py --export_only [--cache] [--gpu_arch sm_87]

import argparse
import os
import sys
import time
from typing import Dict, Tuple

_parsed_args = None
_saved_argv = None
if __name__ == "__main__":
    _saved_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]

import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import cutlass.cute as cute
import numpy as np
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cutedsl_utils import aot_placeholders  # isort: skip

# ---------------------------------------------------------------------------
# Compile-time kernel constants (shared with gdn_decode.py for compatibility)
# ---------------------------------------------------------------------------
TILE_K = 128
TILE_V = 32
TILE_V_PADDED = 36
NUM_STAGES = 2
NUM_THREADS = 256       # 8 warps — same as gdn_decode large_batch
NUM_WARPS = 8
V_PER_WARP = 4          # TILE_V // NUM_WARPS
ROWS_PER_ITER = 8       # k-rows swept per inner k_iter step
NUM_K_ITERS = TILE_K // ROWS_PER_ITER   # = 16

# Max T steps baked into the Phase-1 SMEM layout. AOT export uses this value.
# At runtime, seq_len can be 1..T_MAX_AOT.
T_MAX_AOT = 16

# AOT placeholder shapes — only N/H/HV are dynamic shape markers.
AOT_PLACEHOLDER_N   = 1
AOT_PLACEHOLDER_H   = 14
AOT_PLACEHOLDER_HV  = 14
AOT_PLACEHOLDER_K   = 128
AOT_PLACEHOLDER_V   = 128
AOT_PLACEHOLDER_T   = 8     # representative T for compilation


# ---------------------------------------------------------------------------
# Kernel definition
# ---------------------------------------------------------------------------

def _define_mtp_kernels():
    """Return (gdn_mtp_kernel_warpspec,) — one JIT launcher per Constexpr config."""

    @cute.kernel
    def gdn_mtp_kernel_warpspec(
        tiled_copy_load: cute.TiledCopy,
        h0_source: cute.Tensor,           # [n, hv, K, V] FP32  (in-place updated)
        smem_h_layout_staged: cute.Layout,
        num_v_tiles: Int32,
        q: cute.Tensor,                   # [n, seq_len, h, K] FP16
        k: cute.Tensor,                   # [n, seq_len, h, K] FP16
        v: cute.Tensor,                   # [n, seq_len, hv, V] FP16
        a: cute.Tensor,                   # [n, seq_len, hv] FP16
        b: cute.Tensor,                   # [n, seq_len, hv] FP16
        A_log: cute.Tensor,               # [hv] FP32
        dt_bias: cute.Tensor,             # [hv] FP16
        o: cute.Tensor,                   # [n, seq_len, hv, V] FP16 (output)
        intermediate_states: cute.Tensor, # [n, seq_len, hv, K, V] FP32 (state cache)
        seq_len: Int32,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        use_qk_l2norm: cutlass.Constexpr[bool],
        cache_intermediate_states: cutlass.Constexpr[bool],  # write intermediate_states
        T_MAX: cutlass.Constexpr[int],    # SMEM prealloc upper bound (== T_MAX_AOT)
    ):
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        block_idx, _, _ = cute.arch.block_idx()

        H  = q.layout.shape[2]
        HV = v.layout.shape[2]
        i_n  = block_idx // HV
        i_hv = block_idx % HV
        i_h  = i_hv // (HV // H)

        k_local = in_warp_tid // V_PER_WARP
        v_local = in_warp_tid % V_PER_WARP
        v_base  = warp_idx * V_PER_WARP
        v_idx   = v_base + v_local

        # All batch items process the same number of T steps.
        num_valid_t = seq_len

        # ---------------------------------------------------------------
        # Shared-memory allocations
        # ---------------------------------------------------------------
        smem = cutlass.utils.SmemAllocator()

        # h-state tiles: [TILE_K, TILE_V, NUM_STAGES]  (same as gdn_decode)
        sData = smem.allocate_tensor(cutlass.Float32, smem_h_layout_staged, 128)

        # Per-step precomputed Q/K vectors: [T_MAX, TILE_K]
        smem_qk_t_layout = cute.make_layout((T_MAX, TILE_K), stride=(TILE_K, 1))
        sQ_all = smem.allocate_tensor(cutlass.Float32, smem_qk_t_layout, 128)
        sK_all = smem.allocate_tensor(cutlass.Float32, smem_qk_t_layout, 128)

        # Per-step scalars: g, beta stored interleaved [T_MAX, 2]
        smem_gb_layout = cute.make_layout((T_MAX, 2), stride=(2, 1))
        sGBeta = smem.allocate_tensor(cutlass.Float32, smem_gb_layout, 128)

        # Temp buffer for cross-warp norm reduction [NUM_WARPS * 2]
        smem_norm_layout = cute.make_layout((NUM_WARPS * 2,), stride=(1,))
        sNorm = smem.allocate_tensor(cutlass.Float32, smem_norm_layout, 128)

        # ---------------------------------------------------------------
        # Phase 1: Precompute sQ/sK/sG/sBeta for all T_max steps.
        #   All 256 threads cooperate so the TILE_K=128 loads finish in
        #   a single pass (threads 0..127 each load one element).
        # ---------------------------------------------------------------
        for t in range(seq_len):

            # Load raw q[i_n, t, i_h, :] and k[i_n, t, i_h, :] into SMEM (threads 0..127)
            if tidx < TILE_K:
                sQ_all[(t, tidx)] = cutlass.Float32(q[i_n, t, i_h, tidx])
                sK_all[(t, tidx)] = cutlass.Float32(k[i_n, t, i_h, tidx])

            # Warp-0, lane-0 computes g and beta for this step.
            # Only lane-0 needs to load the scalar inputs; all other threads
            # receive the result via shuffle + SMEM broadcast below.
            r_g    = 0.0
            r_beta = 0.0
            if in_warp_tid == 0:
                r_A_log   = cutlass.Float32(A_log[i_hv])
                r_dt_bias = cutlass.Float32(dt_bias[i_hv])
                r_a = cutlass.Float32(a[i_n, t, i_hv])
                r_b = cutlass.Float32(b[i_n, t, i_hv])
                x          = r_a + r_dt_bias
                beta_x     = softplus_beta * x
                softplus_x = 0.0
                if beta_x <= softplus_threshold:
                    exp_beta_x = cute.exp(beta_x)
                    log_input  = cutlass.Float32(1.0 + exp_beta_x)
                    log_result = cutlass.Float32(cute.log(log_input))
                    softplus_x = cutlass.Float32((cutlass.Float32(1.0) / softplus_beta) * log_result)
                else:
                    softplus_x = x
                r_g_value = -cute.exp(r_A_log) * softplus_x
                r_beta    = 1.0 / (1.0 + cute.exp(-r_b))
                r_g       = cute.exp(r_g_value)

            # Broadcast g, beta from warp-0 lane-0 to all warps.
            r_g    = cute.arch.shuffle_sync(r_g,    0)
            r_beta = cute.arch.shuffle_sync(r_beta, 0)
            # Cross-warp broadcast via SMEM (warp_idx==0 writes, others read)
            if warp_idx == 0 and in_warp_tid == 0:
                sGBeta[(t, 0)] = r_g
                sGBeta[(t, 1)] = r_beta
            cute.arch.barrier()
            r_g    = sGBeta[(t, 0)]
            r_beta = sGBeta[(t, 1)]

            # ---- L2-normalise sQ_all[t] and sK_all[t] ----
            if use_qk_l2norm:
                sum_q_partial = 0.0
                sum_k_partial = 0.0
                if tidx < TILE_K:
                    qv = sQ_all[(t, tidx)]
                    kv = sK_all[(t, tidx)]
                    sum_q_partial = qv * qv
                    sum_k_partial = kv * kv

                for offset in [16, 8, 4, 2, 1]:
                    sum_q_partial += cute.arch.shuffle_sync_bfly(
                        sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31)
                    sum_k_partial += cute.arch.shuffle_sync_bfly(
                        sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31)

                if in_warp_tid == 0:
                    sNorm[warp_idx]               = sum_q_partial
                    sNorm[warp_idx + NUM_WARPS]   = sum_k_partial
                cute.arch.barrier()

                inv_norm_q = 0.0
                inv_norm_k = 0.0
                if warp_idx == 0:
                    local_sum_q = 0.0
                    local_sum_k = 0.0
                    if in_warp_tid < NUM_WARPS:
                        local_sum_q = sNorm[in_warp_tid]
                        local_sum_k = sNorm[in_warp_tid + NUM_WARPS]
                    for offset in [4, 2, 1]:
                        local_sum_q += cute.arch.shuffle_sync_bfly(
                            local_sum_q, offset=offset, mask=-1, mask_and_clamp=31)
                        local_sum_k += cute.arch.shuffle_sync_bfly(
                            local_sum_k, offset=offset, mask=-1, mask_and_clamp=31)
                    if in_warp_tid == 0:
                        sNorm[0] = cute.rsqrt(local_sum_q + 1e-6)
                        sNorm[1] = cute.rsqrt(local_sum_k + 1e-6)
                cute.arch.barrier()

                inv_norm_q = sNorm[0]
                inv_norm_k = sNorm[1]

                if tidx < TILE_K:
                    sQ_all[(t, tidx)] = sQ_all[(t, tidx)] * scale * inv_norm_q
                    sK_all[(t, tidx)] = sK_all[(t, tidx)] * inv_norm_k
                cute.arch.barrier()
            else:
                if tidx < TILE_K:
                    sQ_all[(t, tidx)] = sQ_all[(t, tidx)] * scale
                cute.arch.barrier()

        # ---------------------------------------------------------------
        # Phase 2: Process V-tiles for each valid T step.
        #   Outer loop: t in range(num_valid_t)
        #   Inner loop: v_tile in range(num_v_tiles)  (CP-ASYNC pipelined)
        # ---------------------------------------------------------------
        if num_valid_t > 0:
            gSrc_batch = h0_source[(i_n, i_hv, None, None)]
            gSrc = cute.local_tile(gSrc_batch, (TILE_K, TILE_V), (0, None))
            thr_copy_load = tiled_copy_load.get_slice(tidx)

            for t in range(num_valid_t):
                # Reload g, beta from SMEM for this step.
                r_g    = sGBeta[(t, 0)]
                r_beta = sGBeta[(t, 1)]
                cute.arch.barrier()

                # Prefetch first NUM_STAGES-1 h-tiles from h0_source.
                prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles)
                for v_tile_pf in range(prefetch_count):
                    stage_pf  = v_tile_pf % NUM_STAGES
                    # h0_source may have been updated by the previous t step;
                    # always reload from global (not from prior SMEM stage).
                    gSrc_tile = gSrc[(None, None, v_tile_pf)]
                    sData_stage = sData[(None, None, stage_pf)]
                    thr_gSrc  = thr_copy_load.partition_S(gSrc_tile)
                    thr_sData = thr_copy_load.partition_D(sData_stage)
                    cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                    cute.arch.cp_async_commit_group()

                for v_tile in range(num_v_tiles):
                    stage = v_tile % NUM_STAGES

                    cute.arch.cp_async_wait_group(0)
                    cute.arch.barrier()

                    # Issue prefetch for v_tile + prefetch_count
                    next_v_tile = v_tile + prefetch_count
                    if next_v_tile < num_v_tiles:
                        next_stage = next_v_tile % NUM_STAGES
                        gSrc_next  = gSrc[(None, None, next_v_tile)]
                        sData_next = sData[(None, None, next_stage)]
                        thr_gSrc_n  = thr_copy_load.partition_S(gSrc_next)
                        thr_sData_n = thr_copy_load.partition_D(sData_next)
                        cute.copy(tiled_copy_load, thr_gSrc_n, thr_sData_n)
                        cute.arch.cp_async_commit_group()

                    # Load v for this (t, v_tile)
                    v_global = v_tile * TILE_V + v_idx
                    r_v = cutlass.Float32(v[i_n, t, i_hv, v_global])

                    # Gate h-state once into SMEM, compute h @ k simultaneously.
                    # This avoids the redundant double-read and double-multiply by r_g
                    # that the unfused version had.
                    sum_hk = 0.0
                    for k_iter in range(NUM_K_ITERS):
                        k_base  = k_iter * ROWS_PER_ITER
                        k_idx   = k_base + k_local
                        h_gated = sData[(k_idx, v_idx, stage)] * r_g
                        sData[(k_idx, v_idx, stage)] = h_gated
                        r_k_val = sK_all[(t, k_idx)]
                        sum_hk += h_gated * r_k_val

                    for offset in [4, 2, 1]:
                        sum_hk += cute.arch.shuffle_sync_bfly(
                            sum_hk, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31)

                    # Compute delta-rule correction and broadcast within warp.
                    v_new = (r_v - sum_hk) * r_beta
                    v_new = cute.arch.shuffle_sync(v_new, v_local)

                    # Update h-state (already gated) and compute output h @ q.
                    sum_hq = 0.0
                    for k_iter in range(NUM_K_ITERS):
                        k_base  = k_iter * ROWS_PER_ITER
                        k_idx   = k_base + k_local
                        h_gated = sData[(k_idx, v_idx, stage)]
                        r_k_val = sK_all[(t, k_idx)]
                        r_q_val = sQ_all[(t, k_idx)]
                        h_new   = h_gated + r_k_val * v_new
                        sData[(k_idx, v_idx, stage)] = h_new
                        sum_hq += h_new * r_q_val

                    for offset in [4, 2, 1]:
                        sum_hq += cute.arch.shuffle_sync_bfly(
                            sum_hq, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31)

                    # Write output for position t.
                    if k_local == 0:
                        v_global_out = v_tile * TILE_V + v_idx
                        o[i_n, t, i_hv, v_global_out] = cutlass.Float16(sum_hq)

                    cute.arch.barrier()

                    # Write updated h-tile back to h0_source (in-place).
                    for k_iter in range(NUM_K_ITERS):
                        flat_idx  = tidx + k_iter * NUM_THREADS
                        k_write   = flat_idx // TILE_V
                        v_write   = flat_idx % TILE_V
                        if k_write < TILE_K:
                            h_val         = sData[(k_write, v_write, stage)]
                            v_global_write = v_tile * TILE_V + v_write
                            h0_source[(i_n, i_hv, k_write, v_global_write)] = h_val

                            # Optionally cache per-step intermediate state for rollback.
                            if cache_intermediate_states:
                                intermediate_states[
                                    i_n, t, i_hv, k_write, v_global_write
                                ] = h_val

                    cute.arch.barrier()

        # Note: In MTP mode all batch items process exactly seq_len steps
        # (num_valid_t == seq_len), so there are no padding positions to zero.
        # The zero-padding loop has been removed as dead code.

    return (gdn_mtp_kernel_warpspec,)


_jit_mtp = None


def _get_jit_mtp():
    global _jit_mtp
    if _jit_mtp is None:
        _jit_mtp = _define_mtp_kernels()
    return _jit_mtp


# ---------------------------------------------------------------------------
# JIT wrapper functions  (copy_atom / layouts created inside @cute.jit context)
# ---------------------------------------------------------------------------

_jit_mtp_wrappers = None


def _create_jit_mtp_wrappers():
    (gdn_mtp_kernel_warpspec,) = _get_jit_mtp()

    @cute.jit
    def run_mtp(
        h0_source: cute.Tensor,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        o: cute.Tensor,
        intermediate_states: cute.Tensor,
        seq_len: Int32,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        use_qk_l2norm: cutlass.Constexpr[bool],
        cache_intermediate_states: cutlass.Constexpr[bool],
        T_MAX: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        n_batch = h0_source.layout.shape[0]
        hv_dim = v.layout.shape[2]
        v_dim = v.layout.shape[3]
        batch_size = n_batch * hv_dim

        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32,
            num_bits_per_copy=128,
        )
        num_v_tiles = cute.ceil_div(v_dim, TILE_V)
        smem_h_layout = cute.make_layout(
            (TILE_K, TILE_V, NUM_STAGES),
            stride=(TILE_V_PADDED, 1, TILE_K * TILE_V_PADDED),
        )
        thread_layout = cute.make_layout((32, 8), stride=(8, 1))
        val_layout = cute.make_layout((1, 4))
        tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)

        smem_bytes = (
            4 * TILE_K * TILE_V_PADDED * NUM_STAGES
            + 4 * T_MAX * TILE_K * 2       # sQ_all + sK_all
            + 4 * T_MAX * 2                 # sGBeta
            + 4 * NUM_WARPS * 2             # sNorm
            + 128                           # alignment padding
        )

        gdn_mtp_kernel_warpspec(
            tiled_copy_load,
            h0_source,
            smem_h_layout,
            num_v_tiles,
            q, k, v, a, b,
            A_log, dt_bias,
            o,
            intermediate_states,
            seq_len,
            softplus_beta,
            softplus_threshold,
            scale,
            use_qk_l2norm,
            cache_intermediate_states,
            T_MAX,
        ).launch(
            grid=(batch_size, 1, 1),
            block=[NUM_THREADS, 1, 1],
            smem=smem_bytes,
            stream=stream,
        )

    return run_mtp


def _get_jit_mtp_wrapper():
    global _jit_mtp_wrappers
    if _jit_mtp_wrappers is None:
        _jit_mtp_wrappers = _create_jit_mtp_wrappers()
    return _jit_mtp_wrappers


_compiled_mtp: Dict[Tuple, object] = {}


# ---------------------------------------------------------------------------
# Placeholder / tensor helpers  (mirrors gdn_decode.py patterns)
# ---------------------------------------------------------------------------

def _cp_dtype_fp16():
    return cp.float16


def _make_mtp_placeholder_tensors(n, h, hv, k, v, seq_len, with_cache):
    dt = _cp_dtype_fp16()
    ph = {
        "q":             cp.zeros((n, seq_len, h, k),    dtype=dt),
        "k":             cp.zeros((n, seq_len, h, k),    dtype=dt),
        "v":             cp.zeros((n, seq_len, hv, v),   dtype=dt),
        "a":             cp.zeros((n, seq_len, hv),      dtype=dt),
        "b":             cp.zeros((n, seq_len, hv),      dtype=dt),
        "A_log":         cp.zeros(hv,                    dtype=cp.float32),
        "dt_bias":       cp.zeros(hv,                    dtype=dt),
        "h0_source":     cp.zeros((n, hv, k, v),         dtype=cp.float32),
        "o":             cp.zeros((n, seq_len, hv, v),   dtype=dt),
    }
    # Always allocate a full 5D tensor so MLIR type matches 5D store coordinates
    # even in the no-cache path (the store is guarded by Constexpr but the
    # MLIR verifier checks type congruence before dead-code elimination).
    ph["intermediate_states"] = cp.zeros((n, seq_len, hv, k, v), dtype=cp.float32)
    return ph


def _mark_dynamic_4d(tensor):
    so = (0, 1, 2, 3)
    return (tensor.mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(mode=0, stride_order=so)
            .mark_compact_shape_dynamic(mode=1, stride_order=so)
            .mark_compact_shape_dynamic(mode=2, stride_order=so)
              .mark_compact_shape_dynamic(mode=3, stride_order=so))


def _mark_dynamic_3d(tensor):
    so = (0, 1, 2)
    return (tensor.mark_layout_dynamic(leading_dim=2)
            .mark_compact_shape_dynamic(mode=0, stride_order=so)
            .mark_compact_shape_dynamic(mode=1, stride_order=so))


def _mark_h0_source_dynamic(tensor):
    so = (0, 1, 2, 3)
    return (tensor.mark_compact_shape_dynamic(mode=0, stride_order=so)
            .mark_compact_shape_dynamic(mode=1, stride_order=so))


def _mark_intermediate_states_dynamic(tensor, with_cache):
    # Both variants use a full 5D tensor now; mark identically.
    so = (0, 1, 2, 3, 4)
    ct = from_dlpack(tensor, assumed_align=32)
    return (ct.mark_layout_dynamic(leading_dim=4)
              .mark_compact_shape_dynamic(mode=0, stride_order=so)
              .mark_compact_shape_dynamic(mode=1, stride_order=so)
              .mark_compact_shape_dynamic(mode=2, stride_order=so)
              .mark_compact_shape_dynamic(mode=3, stride_order=so))


def _to_mtp_cute_tensors(ph, with_cache):
    def wrap(arr, leading_dim=None):
        ct = from_dlpack(arr, assumed_align=16)
        ld = (len(arr.shape) - 1) if leading_dim is None else leading_dim
        return ct.mark_layout_dynamic(leading_dim=ld)

    q_ct  = wrap(ph["q"])
    v_ct  = wrap(ph["v"])
    return {
        "q":          _mark_dynamic_4d(q_ct),
        "k":          wrap(ph["k"]),
        "v":          _mark_dynamic_4d(v_ct),
        "a":          _mark_dynamic_3d(wrap(ph["a"])),
        "b":          _mark_dynamic_3d(wrap(ph["b"])),
        "A_log":      wrap(ph["A_log"], leading_dim=0),
        "dt_bias":    wrap(ph["dt_bias"], leading_dim=0),
        "h0_source":  _mark_h0_source_dynamic(from_dlpack(ph["h0_source"], assumed_align=32)),
        "o":          wrap(ph["o"]),
        "intermediate_states": _mark_intermediate_states_dynamic(ph["intermediate_states"], with_cache),
    }


def _make_aot_mtp_cute_tensors(n, h, hv, k, v, seq_len):
    def compact(dtype, shape, assumed_align=16):
        return aot_placeholders.make_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=assumed_align,
        )

    q = compact(cutlass.Float16, (n, seq_len, h, k))
    v_tensor = compact(cutlass.Float16, (n, seq_len, hv, v))
    a = compact(cutlass.Float16, (n, seq_len, hv))
    b = compact(cutlass.Float16, (n, seq_len, hv))
    h0_source = compact(cutlass.Float32, (n, hv, k, v), assumed_align=32)
    intermediate_states = compact(
        cutlass.Float32, (n, seq_len, hv, k, v), assumed_align=32
    )
    return {
        "q": _mark_dynamic_4d(q.mark_layout_dynamic(leading_dim=3)),
        "k": compact(cutlass.Float16, (n, seq_len, h, k)).mark_layout_dynamic(leading_dim=3),
        "v": _mark_dynamic_4d(v_tensor.mark_layout_dynamic(leading_dim=3)),
        "a": _mark_dynamic_3d(a.mark_layout_dynamic(leading_dim=2)),
        "b": _mark_dynamic_3d(b.mark_layout_dynamic(leading_dim=2)),
        "A_log": compact(cutlass.Float32, (hv,)).mark_layout_dynamic(leading_dim=0),
        "dt_bias": compact(cutlass.Float16, (hv,)).mark_layout_dynamic(leading_dim=0),
        "h0_source": _mark_h0_source_dynamic(h0_source),
        "o": compact(cutlass.Float16, (n, seq_len, hv, v)).mark_layout_dynamic(leading_dim=3),
        "intermediate_states": (
            intermediate_states.mark_layout_dynamic(leading_dim=4)
            .mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
            .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1, 2, 3, 4))
            .mark_compact_shape_dynamic(mode=2, stride_order=(0, 1, 2, 3, 4))
            .mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 2, 3, 4))
        ),
    }


def _compile_mtp(
    n, h, hv, k, v, seq_len, with_cache, stream, gpu_arch="", export_only=False
):
    key = (with_cache,)
    if key in _compiled_mtp:
        return _compiled_mtp[key]

    run_mtp = _get_jit_mtp_wrapper()

    if export_only:
        t = _make_aot_mtp_cute_tensors(n, h, hv, k, v, seq_len)
    else:
        ph = _make_mtp_placeholder_tensors(n, h, hv, k, v, seq_len, with_cache)
        t = _to_mtp_cute_tensors(ph, with_cache)

    # Only the export path may pin a foreign target arch in the compile
    # options; a native JIT run must compile for the local GPU (see the
    # native-vs-cross note in cutedsl_utils/cutedsl_compile_wrapper.py).
    compile_opts = aot_placeholders.compile_options(
        f"--gpu-arch={gpu_arch}" if gpu_arch else ""
    ) if export_only else None
    compiled = cute.compile(
        run_mtp,
        t["h0_source"],
        t["q"], t["k"], t["v"], t["a"], t["b"],
        t["A_log"], t["dt_bias"],
        t["o"],
        t["intermediate_states"],
        seq_len,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        scale=k ** -0.5,
        use_qk_l2norm=True,
        cache_intermediate_states=with_cache,
        T_MAX=T_MAX_AOT,
        stream=stream,
        **(dict(options=compile_opts) if compile_opts else {}),
    )
    _compiled_mtp[key] = compiled
    return compiled


# ---------------------------------------------------------------------------
# AOT export
# ---------------------------------------------------------------------------

def export_gdn_decode_mtp(
    n, h, hv, k, v,
    output_dir, file_name, function_prefix,
    gpu_arch="",
    cache_only=False,
):
    """AOT-compile and export gdn_decode_mtp kernel (cache variant only).

    Exports into output_dir:
      <file_name>_cache.h/.o   — cache variant (writes intermediate_states per step)

    The cache variant is used for speculative-decoding rollback.
    """
    stream = aot_placeholders.make_stream()
    os.makedirs(output_dir, exist_ok=True)

    fn = file_name + "_cache"
    fp = function_prefix + "_cache"

    print("[gdn_decode_mtp] AOT compile cache variant gpu_arch=%r" % (gpu_arch or "default"))
    t0 = time.time()
    compiled = _compile_mtp(
        n, h, hv, k, v,
        seq_len=AOT_PLACEHOLDER_T,
        with_cache=True,
        stream=stream,
        gpu_arch=gpu_arch,
        export_only=True,
    )
    print("[gdn_decode_mtp] Compilation time: %.4fs" % (time.time() - t0))

    compiled.export_to_c(
        file_path=output_dir,
        file_name=fn,
        function_prefix=fp,
    )
    print("[gdn_decode_mtp] Exported %s/%s.h and %s.o" % (output_dir, fn, fn))


# ---------------------------------------------------------------------------
# NumPy reference implementation
# ---------------------------------------------------------------------------

def _softplus_np(x, beta=1.0, threshold=20.0):
    bx = beta * x
    if bx <= threshold:
        return (1.0 / beta) * np.log(1.0 + np.exp(bx))
    return float(x)


def _run_numpy_mtp_reference(
    q_f32,         # [n, T, h, k]
    k_f32,         # [n, T, h, k]
    v_f32,         # [n, T, hv, v]
    a_f32,         # [n, T, hv]
    b_f32,         # [n, T, hv]
    A_log_f32,     # [hv]
    dt_bias_f32,   # [hv]
    h0_f32,        # [n, hv, k, v]  — updated in-place
    n, h, hv, k_dim, v_dim, seq_len,
    scale,
    use_qk_l2norm=True,
):
    """CPU reference: evolves GDN state over seq_len (T) time steps per batch item.

    Returns:
        o_ref   : [n, T, hv, v] float32
        h0_out  : [n, hv, k, v] float32  (equals h0_f32 after in-place update)
        interm  : [n, T, hv, k, v] float32  (per-step intermediate states)
    """
    h0 = h0_f32.copy()   # don't modify caller's array
    o_ref  = np.zeros((n, seq_len, hv, v_dim), dtype=np.float32)
    interm = np.zeros((n, seq_len, hv, k_dim, v_dim), dtype=np.float32)

    for i_n in range(n):
        for i_hv in range(hv):
            i_h = i_hv // (hv // h) if h else 0
            H = h0[i_n, i_hv].copy()   # [k, v]

            for t in range(seq_len):
                q_vec = q_f32[i_n, t, i_h].astype(np.float64)
                k_vec = k_f32[i_n, t, i_h].astype(np.float64)
                v_vec = v_f32[i_n, t, i_hv].astype(np.float64)
                a_val = float(a_f32[i_n, t, i_hv])
                b_val = float(b_f32[i_n, t, i_hv])

                if use_qk_l2norm:
                    nq = np.sqrt(np.sum(q_vec ** 2) + 1e-6)
                    nk = np.sqrt(np.sum(k_vec ** 2) + 1e-6)
                    q_eff = q_vec / nq * scale
                    k_eff = k_vec / nk
                else:
                    q_eff = q_vec * scale
                    k_eff = k_vec

                A_val  = float(A_log_f32[i_hv])
                dt_val = float(dt_bias_f32[i_hv])
                sp     = _softplus_np(a_val + dt_val, 1.0, 20.0)
                g      = np.exp(-np.exp(A_val) * sp)
                beta   = 1.0 / (1.0 + np.exp(-b_val))

                H_gated    = H * g
                correction = (v_vec - H_gated.T @ k_eff) * beta
                H_new      = H_gated + np.outer(k_eff, correction)
                o_ref[i_n, t, i_hv, :] = (H_new.T @ q_eff).astype(np.float32)

                H = H_new
                interm[i_n, t, i_hv] = H_new.astype(np.float32)

            # Padded steps: o = 0 (already zeroed), state unchanged
            h0[i_n, i_hv] = H.astype(np.float32)

    return o_ref, h0, interm


# ---------------------------------------------------------------------------
# JIT launcher helper
# ---------------------------------------------------------------------------

def _run_mtp_decode_jit(
    q_cp, k_cp, v_cp, a_cp, b_cp,
    A_log_cp, dt_bias_cp,
    h0_cp,
    o_cp, intermediate_states_cp,
    seq_len, with_cache, stream, gpu_arch="",
):
    n   = h0_cp.shape[0]
    hv  = h0_cp.shape[1]
    k   = h0_cp.shape[2]
    v   = h0_cp.shape[3]
    h   = q_cp.shape[2]

    compiled = _compile_mtp(n, h, hv, k, v, seq_len, with_cache, stream, gpu_arch=gpu_arch)

    def wrap(arr, assumed_align=16):
        return from_dlpack(arr, assumed_align=assumed_align)

    compiled(
        wrap(h0_cp, 32),
        wrap(q_cp), wrap(k_cp), wrap(v_cp),
        wrap(a_cp), wrap(b_cp),
        wrap(A_log_cp), wrap(dt_bias_cp),
        wrap(o_cp),
        wrap(intermediate_states_cp, 32),
        seq_len,
        stream,
    )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def run_test_mtp(
    n=4, h=8, hv=8, k=128, v=128, seq_len=4,
    with_cache=False,
    skip_ref_check=False,
    tolerance=0.1,
    warmup=3,
    iterations=100,
    gpu_arch="",
):
    dt     = _cp_dtype_fp16()
    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    scale  = k ** -0.5

    # Generate random FP32 data; convert to FP16 for GPU inputs.
    q_f32  = np.random.randn(n, seq_len, h, k).astype(np.float32) * 0.1
    k_f32  = np.random.randn(n, seq_len, h, k).astype(np.float32) * 0.1
    v_f32  = np.random.randn(n, seq_len, hv, v).astype(np.float32) * 0.1
    a_f32  = np.random.randn(n, seq_len, hv).astype(np.float32) * 0.1
    b_f32  = np.random.randn(n, seq_len, hv).astype(np.float32) * 0.1
    A_log_f32   = np.random.randn(hv).astype(np.float32) * 0.1
    dt_bias_f32 = np.random.randn(hv).astype(np.float32) * 0.1
    h0_f32      = np.random.randn(n, hv, k, v).astype(np.float32) * 0.01

    # GPU arrays
    q_cp  = cp.asarray(q_f32.astype(np.float16))
    k_cp  = cp.asarray(k_f32.astype(np.float16))
    v_cp  = cp.asarray(v_f32.astype(np.float16))
    a_cp  = cp.asarray(a_f32.astype(np.float16))
    b_cp  = cp.asarray(b_f32.astype(np.float16))
    A_log_cp    = cp.asarray(A_log_f32)
    dt_bias_cp  = cp.asarray(dt_bias_f32.astype(np.float16))
    h0_cp       = cp.asarray(h0_f32)
    o_cp = cp.zeros((n, seq_len, hv, v), dtype=dt)

    if with_cache:
        interm_cp = cp.zeros((n, seq_len, hv, k, v), dtype=cp.float32)
    else:
        interm_cp = cp.zeros((1,), dtype=cp.float32)  # dummy

    # Warmup
    for _ in range(warmup):
        h0_warm = cp.asarray(h0_f32)
        _run_mtp_decode_jit(
            q_cp, k_cp, v_cp, a_cp, b_cp,
            A_log_cp, dt_bias_cp,
            h0_warm,
            o_cp, interm_cp, seq_len, with_cache, stream, gpu_arch,
        )
    cp.cuda.get_current_stream().synchronize()

    if not skip_ref_check:
        h0_kernel_cp = cp.asarray(h0_f32)
        o_cp_check   = cp.zeros((n, seq_len, hv, v), dtype=dt)
        if with_cache:
            interm_check = cp.zeros((n, seq_len, hv, k, v), dtype=cp.float32)
        else:
            interm_check = cp.zeros((1,), dtype=cp.float32)

        _run_mtp_decode_jit(
            q_cp, k_cp, v_cp, a_cp, b_cp,
            A_log_cp, dt_bias_cp,
            h0_kernel_cp,
            o_cp_check, interm_check, seq_len, with_cache, stream, gpu_arch,
        )
        cp.cuda.get_current_stream().synchronize()

        o_ref, h0_ref, interm_ref = _run_numpy_mtp_reference(
            q_f32, k_f32, v_f32, a_f32, b_f32,
            A_log_f32, dt_bias_f32,
            h0_f32, n, h, hv, k, v, seq_len, scale,
        )

        o_kernel  = cp.asnumpy(o_cp_check).astype(np.float32)
        h0_kernel = cp.asnumpy(h0_kernel_cp)

        np.testing.assert_allclose(o_kernel, o_ref, atol=tolerance, rtol=1e-2)
        print("[gdn_decode_mtp] Output reference check PASSED.")
        np.testing.assert_allclose(h0_kernel, h0_ref, atol=tolerance, rtol=1e-2)
        print("[gdn_decode_mtp] h0 reference check PASSED.")

        if with_cache:
            interm_kernel = cp.asnumpy(interm_check)
            np.testing.assert_allclose(interm_kernel, interm_ref, atol=tolerance, rtol=1e-2)
            print("[gdn_decode_mtp] Intermediate states reference check PASSED.")

        label = "uniform" + ("+cache" if with_cache else "")
        print("[gdn_decode_mtp] Reference check PASSED (%s)." % label)

    # Benchmark
    h0_bench = cp.asarray(h0_f32)
    t0 = time.perf_counter()
    for _ in range(iterations):
        _run_mtp_decode_jit(
            q_cp, k_cp, v_cp, a_cp, b_cp,
            A_log_cp, dt_bias_cp,
            h0_bench,
            o_cp, interm_cp, seq_len, with_cache, stream, gpu_arch,
        )
    cp.cuda.get_current_stream().synchronize()
    us = (time.perf_counter() - t0) * 1e6 / iterations
    print("[gdn_decode_mtp] Latency: %.4f us (n=%d h=%d hv=%d k=%d v=%d T=%d)" % (
        us, n, h, hv, k, v, seq_len))
    return us


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = _parsed_args
    np.random.seed(42)

    if args.export_only:
        export_gdn_decode_mtp(
            n=AOT_PLACEHOLDER_N,
            h=AOT_PLACEHOLDER_H,
            hv=AOT_PLACEHOLDER_HV,
            k=AOT_PLACEHOLDER_K,
            v=AOT_PLACEHOLDER_V,
            output_dir=args.output_dir,
            file_name=args.file_name,
            function_prefix=args.function_prefix,
            gpu_arch=args.gpu_arch,
            cache_only=args.cache_only,
        )
        return

    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU required.")
    cp.random.seed(42)

    run_test_mtp(
        n=args.n,
        h=args.h,
        hv=args.hv,
        k=args.k,
        v=args.v,
        seq_len=args.seq_len,
        with_cache=args.cache,
        skip_ref_check=args.skip_ref_check,
        tolerance=args.tolerance,
        warmup=args.warmup,
        iterations=args.iterations,
        gpu_arch=args.gpu_arch,
    )


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CuTe DSL GDN MTP decode: AOT export and test (Ampere+)."
    )
    p.add_argument("--export_only",   action="store_true",
                   help="AOT export only (no test run)")
    p.add_argument("--output_dir",    type=str,  default="./gdn_mtp_artifacts")
    p.add_argument("--file_name",     type=str,  default="gdn_decode_mtp")
    p.add_argument("--function_prefix", type=str, default="gdn_decode_mtp")
    p.add_argument("--n",             type=int,  default=4)
    p.add_argument("--h",             type=int,  default=8)
    p.add_argument("--hv",            type=int,  default=8)
    p.add_argument("--k",             type=int,  default=128)
    p.add_argument("--v",             type=int,  default=128)
    p.add_argument("--seq_len",       type=int,  default=4,
                   help="Number of MTP draft tokens T (1 .. T_MAX=%d)" % T_MAX_AOT)
    p.add_argument("--cache",         action="store_true",
                   help="Enable intermediate-state caching in test mode")
    p.add_argument("--cache_only",    action="store_true",
                   help="AOT export: only export the cache variant (skip base)")
    p.add_argument("--skip_ref_check", action="store_true")
    p.add_argument("--tolerance",     type=float, default=0.1)
    p.add_argument("--warmup",        type=int,   default=3)
    p.add_argument("--iterations",    type=int,   default=100)
    p.add_argument("--gpu_arch",      type=str,   default="",
                   help="Target GPU arch for export (e.g. sm_87). Empty = current GPU.")
    return p.parse_known_args(args=argv)[0]


if __name__ == "__main__":
    _parsed_args = _parse_args(_saved_argv)
    main()
