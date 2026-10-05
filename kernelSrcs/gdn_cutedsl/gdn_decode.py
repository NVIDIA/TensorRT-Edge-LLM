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

# Origin: Adapted from SGLang CuTe DSL GDN decode kernel (Apache-2.0):
#   https://github.com/sgl-project/sglang/blob/v0.5.9/python/sglang/jit_kernel/cutedsl_gdn.py
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang Team. All Rights Reserved.

# This version uses CuPy only (no PyTorch) and is built for edge-llm AOT export and C++ plugin 
# integration.
# CuTe DSL GDN decode (seq_len=1). Test: python gdn_decode.py --n 64 ... ; AOT: --export_only ...

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

TILE_K = 128
TILE_V = 32
TILE_V_PADDED = 36
TILE_V_SMALL = 16
TILE_V_SMALL_PADDED = 20
NUM_STAGES = 2
NUM_THREADS = 128
NUM_BLOCKS_PER_STATE_SMALL = 8
NUM_THREADS_LARGE = 256
NUM_WARPS_LARGE = 8
V_PER_WARP = 4
ROWS_PER_ITER = 8
NUM_K_ITERS = TILE_K // ROWS_PER_ITER
SMALL_BATCH_THRESHOLD = 32

# AOT placeholders.
AOT_PLACEHOLDER_N = 1
AOT_PLACEHOLDER_H = 14
AOT_PLACEHOLDER_HV = 14
AOT_PLACEHOLDER_K = 128
AOT_PLACEHOLDER_V = 128


def _define_kernels():
    """Define CuTe DSL kernels for normal and varlen decode modes."""

    NUM_WARPS_SMALL = 4
    V_PER_WARP_SMALL = TILE_V_SMALL // NUM_WARPS_SMALL
    ROWS_PER_ITER_SMALL = 32 // V_PER_WARP_SMALL
    NUM_K_ITERS_SMALL = TILE_K // ROWS_PER_ITER_SMALL

    @cute.kernel
    def gdn_kernel_small_batch(
        tiled_copy_load: cute.TiledCopy,
        h0_source: cute.Tensor,
        state_indices: cute.Tensor,
        use_state_indices: Int32,
        smem_layout_staged: cute.Layout,
        num_v_tiles: Int32,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        context_lengths: cute.Tensor,
        o: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        use_qk_l2norm: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        block_idx, _, _ = cute.arch.block_idx()
        H = q.layout.shape[2]
        HV = v.layout.shape[2]

        batch_idx = block_idx // NUM_BLOCKS_PER_STATE_SMALL
        batch_inner = block_idx % NUM_BLOCKS_PER_STATE_SMALL
        num_v_tiles_per_block = num_v_tiles // NUM_BLOCKS_PER_STATE_SMALL
        start_v_tile = batch_inner * num_v_tiles_per_block

        i_n = batch_idx // HV
        i_hv = batch_idx % HV
        i_h = i_hv // (HV // H)
        state_slot = cutlass.Int32(i_n)
        if use_state_indices != 0:
            state_slot = cutlass.Int32(state_indices[i_n])

        k_local = in_warp_tid // V_PER_WARP_SMALL
        v_local = in_warp_tid % V_PER_WARP_SMALL
        v_base = warp_idx * V_PER_WARP_SMALL
        v_idx = v_base + v_local

        ctx_len = cutlass.Int32(context_lengths[i_n])

        if ctx_len > 0:
            smem = cutlass.utils.SmemAllocator()
            sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
            smem_o_layout = cute.make_layout((TILE_V_SMALL,), stride=(1,))
            smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
            smem_k_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_q_layout = cute.make_layout((TILE_K,), stride=(1,))
            sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
            sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)

            if tidx < TILE_K:
                sK[tidx] = cutlass.Float32(k[i_n, 0, i_h, tidx])
                sQ[tidx] = cutlass.Float32(q[i_n, 0, i_h, tidx])

            gSrc_batch = h0_source[(state_slot, i_hv, None, None)]
            gSrc = cute.local_tile(gSrc_batch, (TILE_K, TILE_V_SMALL), (0, None))
            thr_copy_load = tiled_copy_load.get_slice(tidx)

            prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles_per_block)
            for v_tile_offset in range(prefetch_count):
                v_tile = start_v_tile + v_tile_offset
                stage = v_tile_offset % NUM_STAGES
                gSrc_tile = gSrc[(None, None, v_tile)]
                sData_stage = sData[(None, None, stage)]
                thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
                thr_sData = thr_copy_load.partition_D(sData_stage)
                cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                cute.arch.cp_async_commit_group()

            r_A_log = cutlass.Float32(A_log[i_hv])
            r_dt_bias = cutlass.Float32(dt_bias[i_hv])
            r_a = cutlass.Float32(a[i_n, 0, i_hv])
            r_b = cutlass.Float32(b[i_n, 0, i_hv])

            r_g = 0.0
            r_beta = 0.0
            if in_warp_tid == 0:
                x = r_a + r_dt_bias
                beta_x = softplus_beta * x
                softplus_x = 0.0
                if beta_x <= softplus_threshold:
                    exp_beta_x = cute.exp(beta_x)
                    log_input = cutlass.Float32(1.0 + exp_beta_x)
                    log_result = cutlass.Float32(cute.log(log_input))
                    softplus_x = cutlass.Float32(
                        (cutlass.Float32(1.0) / softplus_beta) * log_result
                    )
                else:
                    softplus_x = x
                r_g_value = -cute.exp(r_A_log) * softplus_x
                r_beta = 1.0 / (1.0 + cute.exp(-r_b))
                r_g = cute.exp(r_g_value)

            r_g = cute.arch.shuffle_sync(r_g, 0)
            r_beta = cute.arch.shuffle_sync(r_beta, 0)

            cute.arch.barrier()

            if use_qk_l2norm:
                sum_q_partial = 0.0
                sum_k_partial = 0.0
                if tidx < TILE_K:
                    q_val = sQ[tidx]
                    k_val = sK[tidx]
                    sum_q_partial = q_val * q_val
                    sum_k_partial = k_val * k_val

                for offset in [16, 8, 4, 2, 1]:
                    sum_q_partial += cute.arch.shuffle_sync_bfly(
                        sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    sum_k_partial += cute.arch.shuffle_sync_bfly(
                        sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31
                    )

                if in_warp_tid == 0:
                    smem_o[warp_idx] = sum_q_partial
                    smem_o[warp_idx + 4] = sum_k_partial
                cute.arch.barrier()

                inv_norm_q = 0.0
                inv_norm_k = 0.0
                if warp_idx == 0:
                    local_sum_q = 0.0
                    local_sum_k = 0.0
                    if in_warp_tid < NUM_WARPS_SMALL:
                        local_sum_q = smem_o[in_warp_tid]
                        local_sum_k = smem_o[in_warp_tid + 4]
                    for offset in [2, 1]:
                        local_sum_q += cute.arch.shuffle_sync_bfly(
                            local_sum_q, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        local_sum_k += cute.arch.shuffle_sync_bfly(
                            local_sum_k, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    if in_warp_tid == 0:
                        smem_o[0] = cute.rsqrt(local_sum_q + 1e-6)
                        smem_o[1] = cute.rsqrt(local_sum_k + 1e-6)
                cute.arch.barrier()

                inv_norm_q = smem_o[0]
                inv_norm_k = smem_o[1]

                if tidx < TILE_K:
                    sK[tidx] = sK[tidx] * inv_norm_k
                    sQ[tidx] = sQ[tidx] * scale * inv_norm_q
                cute.arch.barrier()
            else:
                if tidx < TILE_K:
                    sQ[tidx] = sQ[tidx] * scale
                cute.arch.barrier()

            for v_tile_offset in range(num_v_tiles_per_block):
                v_tile = start_v_tile + v_tile_offset
                stage = v_tile_offset % NUM_STAGES

                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                next_v_tile_offset = v_tile_offset + prefetch_count
                if next_v_tile_offset < num_v_tiles_per_block:
                    next_v_tile = start_v_tile + next_v_tile_offset
                    next_stage = next_v_tile_offset % NUM_STAGES
                    gSrc_next = gSrc[(None, None, next_v_tile)]
                    sData_next = sData[(None, None, next_stage)]
                    thr_gSrc = thr_copy_load.partition_S(gSrc_next)
                    thr_sData = thr_copy_load.partition_D(sData_next)
                    cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                    cute.arch.cp_async_commit_group()

                v_global = v_tile * TILE_V_SMALL + v_idx
                r_v = cutlass.Float32(v[i_n, 0, i_hv, v_global])

                sum_hk = 0.0
                for k_iter in range(NUM_K_ITERS_SMALL):
                    k_base = k_iter * ROWS_PER_ITER_SMALL
                    k_idx = k_base + k_local
                    h_val = sData[(k_idx, v_idx, stage)] * r_g
                    r_k_val = sK[k_idx]
                    sum_hk += h_val * r_k_val

                for offset in [4, 2, 1]:
                    sum_hk += cute.arch.shuffle_sync_bfly(
                        sum_hk,
                        offset=offset * V_PER_WARP_SMALL,
                        mask=-1,
                        mask_and_clamp=31,
                    )

                v_new = (r_v - sum_hk) * r_beta
                v_new = cute.arch.shuffle_sync(v_new, v_local)

                sum_hq = 0.0
                for k_iter in range(NUM_K_ITERS_SMALL):
                    k_base = k_iter * ROWS_PER_ITER_SMALL
                    k_idx = k_base + k_local
                    h_old = sData[(k_idx, v_idx, stage)] * r_g
                    r_k_val = sK[k_idx]
                    r_q_val = sQ[k_idx]
                    h_new = h_old + r_k_val * v_new
                    sData[(k_idx, v_idx, stage)] = h_new
                    sum_hq += h_new * r_q_val

                for offset in [4, 2, 1]:
                    sum_hq += cute.arch.shuffle_sync_bfly(
                        sum_hq,
                        offset=offset * V_PER_WARP_SMALL,
                        mask=-1,
                        mask_and_clamp=31,
                    )

                if k_local == 0:
                    v_global_out = v_tile * TILE_V_SMALL + v_idx
                    o[(i_n, 0, i_hv, v_global_out)] = cutlass.Float16(sum_hq)

                cute.arch.barrier()

                for k_iter in range(NUM_K_ITERS_SMALL):
                    flat_idx = tidx + k_iter * 128
                    k_write = flat_idx // TILE_V_SMALL
                    v_write = flat_idx % TILE_V_SMALL
                    if k_write < TILE_K:
                        h_val = sData[(k_write, v_write, stage)]
                        v_global_write = v_tile * TILE_V_SMALL + v_write
                        h0_source[(state_slot, i_hv, k_write, v_global_write)] = h_val

                cute.arch.barrier()
        else:
            for v_tile_offset in range(num_v_tiles_per_block):
                v_tile = start_v_tile + v_tile_offset
                v_global = v_tile * TILE_V_SMALL + v_idx
                if k_local == 0:
                    o[(i_n, 0, i_hv, v_global)] = cutlass.Float16(0.0)

    @cute.kernel
    def gdn_kernel_small_batch_varlen(
        tiled_copy_load: cute.TiledCopy,
        h0_source: cute.Tensor,
        state_indices: cute.Tensor,
        use_state_indices: Int32,
        smem_layout_staged: cute.Layout,
        num_v_tiles: Int32,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        context_lengths: cute.Tensor,
        o: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        use_qk_l2norm: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        block_idx, _, _ = cute.arch.block_idx()
        H = q.layout.shape[2]
        HV = v.layout.shape[2]

        batch_idx = block_idx // NUM_BLOCKS_PER_STATE_SMALL
        batch_inner = block_idx % NUM_BLOCKS_PER_STATE_SMALL
        num_v_tiles_per_block = num_v_tiles // NUM_BLOCKS_PER_STATE_SMALL
        start_v_tile = batch_inner * num_v_tiles_per_block

        i_n = batch_idx // HV
        i_hv = batch_idx % HV
        i_h = i_hv // (HV // H)
        state_slot = cutlass.Int32(i_n)
        if use_state_indices != 0:
            state_slot = cutlass.Int32(state_indices[i_n])

        k_local = in_warp_tid // V_PER_WARP_SMALL
        v_local = in_warp_tid % V_PER_WARP_SMALL
        v_base = warp_idx * V_PER_WARP_SMALL
        v_idx = v_base + v_local

        ctx_len = cutlass.Int32(context_lengths[i_n])

        if ctx_len > 0:
            smem = cutlass.utils.SmemAllocator()
            sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
            smem_o_layout = cute.make_layout((TILE_V_SMALL,), stride=(1,))
            smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
            smem_k_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_q_layout = cute.make_layout((TILE_K,), stride=(1,))
            sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
            sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)

            if tidx < TILE_K:
                sK[tidx] = cutlass.Float32(k[0, i_n, i_h, tidx])
                sQ[tidx] = cutlass.Float32(q[0, i_n, i_h, tidx])

            gSrc_batch = h0_source[(state_slot, i_hv, None, None)]
            gSrc = cute.local_tile(gSrc_batch, (TILE_K, TILE_V_SMALL), (0, None))
            thr_copy_load = tiled_copy_load.get_slice(tidx)

            prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles_per_block)
            for v_tile_offset in range(prefetch_count):
                v_tile = start_v_tile + v_tile_offset
                stage = v_tile_offset % NUM_STAGES
                gSrc_tile = gSrc[(None, None, v_tile)]
                sData_stage = sData[(None, None, stage)]
                thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
                thr_sData = thr_copy_load.partition_D(sData_stage)
                cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                cute.arch.cp_async_commit_group()

            r_A_log = cutlass.Float32(A_log[i_hv])
            r_dt_bias = cutlass.Float32(dt_bias[i_hv])
            r_a = cutlass.Float32(a[i_n, i_hv])
            r_b = cutlass.Float32(b[i_n, i_hv])

            r_g = 0.0
            r_beta = 0.0
            if in_warp_tid == 0:
                x = r_a + r_dt_bias
                beta_x = softplus_beta * x
                softplus_x = 0.0
                if beta_x <= softplus_threshold:
                    exp_beta_x = cute.exp(beta_x)
                    log_input = cutlass.Float32(1.0 + exp_beta_x)
                    log_result = cutlass.Float32(cute.log(log_input))
                    softplus_x = cutlass.Float32(
                        (cutlass.Float32(1.0) / softplus_beta) * log_result
                    )
                else:
                    softplus_x = x
                r_g_value = -cute.exp(r_A_log) * softplus_x
                r_beta = 1.0 / (1.0 + cute.exp(-r_b))
                r_g = cute.exp(r_g_value)

            r_g = cute.arch.shuffle_sync(r_g, 0)
            r_beta = cute.arch.shuffle_sync(r_beta, 0)

            cute.arch.barrier()

            if use_qk_l2norm:
                sum_q_partial = 0.0
                sum_k_partial = 0.0
                if tidx < TILE_K:
                    q_val = sQ[tidx]
                    k_val = sK[tidx]
                    sum_q_partial = q_val * q_val
                    sum_k_partial = k_val * k_val

                for offset in [16, 8, 4, 2, 1]:
                    sum_q_partial += cute.arch.shuffle_sync_bfly(
                        sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    sum_k_partial += cute.arch.shuffle_sync_bfly(
                        sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31
                    )

                if in_warp_tid == 0:
                    smem_o[warp_idx] = sum_q_partial
                    smem_o[warp_idx + 4] = sum_k_partial
                cute.arch.barrier()

                inv_norm_q = 0.0
                inv_norm_k = 0.0
                if warp_idx == 0:
                    local_sum_q = 0.0
                    local_sum_k = 0.0
                    if in_warp_tid < NUM_WARPS_SMALL:
                        local_sum_q = smem_o[in_warp_tid]
                        local_sum_k = smem_o[in_warp_tid + 4]
                    for offset in [2, 1]:
                        local_sum_q += cute.arch.shuffle_sync_bfly(
                            local_sum_q, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        local_sum_k += cute.arch.shuffle_sync_bfly(
                            local_sum_k, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    if in_warp_tid == 0:
                        smem_o[0] = cute.rsqrt(local_sum_q + 1e-6)
                        smem_o[1] = cute.rsqrt(local_sum_k + 1e-6)
                cute.arch.barrier()

                inv_norm_q = smem_o[0]
                inv_norm_k = smem_o[1]

                if tidx < TILE_K:
                    sK[tidx] = sK[tidx] * inv_norm_k
                    sQ[tidx] = sQ[tidx] * scale * inv_norm_q
                cute.arch.barrier()
            else:
                if tidx < TILE_K:
                    sQ[tidx] = sQ[tidx] * scale
                cute.arch.barrier()

            for v_tile_offset in range(num_v_tiles_per_block):
                v_tile = start_v_tile + v_tile_offset
                stage = v_tile_offset % NUM_STAGES

                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                next_v_tile_offset = v_tile_offset + prefetch_count
                if next_v_tile_offset < num_v_tiles_per_block:
                    next_v_tile = start_v_tile + next_v_tile_offset
                    next_stage = next_v_tile_offset % NUM_STAGES
                    gSrc_next = gSrc[(None, None, next_v_tile)]
                    sData_next = sData[(None, None, next_stage)]
                    thr_gSrc = thr_copy_load.partition_S(gSrc_next)
                    thr_sData = thr_copy_load.partition_D(sData_next)
                    cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                    cute.arch.cp_async_commit_group()

                v_global = v_tile * TILE_V_SMALL + v_idx
                r_v = cutlass.Float32(v[0, i_n, i_hv, v_global])

                sum_hk = 0.0
                for k_iter in range(NUM_K_ITERS_SMALL):
                    k_base = k_iter * ROWS_PER_ITER_SMALL
                    k_idx = k_base + k_local
                    h_val = sData[(k_idx, v_idx, stage)] * r_g
                    r_k_val = sK[k_idx]
                    sum_hk += h_val * r_k_val

                for offset in [4, 2, 1]:
                    sum_hk += cute.arch.shuffle_sync_bfly(
                        sum_hk,
                        offset=offset * V_PER_WARP_SMALL,
                        mask=-1,
                        mask_and_clamp=31,
                    )

                v_new = (r_v - sum_hk) * r_beta
                v_new = cute.arch.shuffle_sync(v_new, v_local)

                sum_hq = 0.0
                for k_iter in range(NUM_K_ITERS_SMALL):
                    k_base = k_iter * ROWS_PER_ITER_SMALL
                    k_idx = k_base + k_local
                    h_old = sData[(k_idx, v_idx, stage)] * r_g
                    r_k_val = sK[k_idx]
                    r_q_val = sQ[k_idx]
                    h_new = h_old + r_k_val * v_new
                    sData[(k_idx, v_idx, stage)] = h_new
                    sum_hq += h_new * r_q_val

                for offset in [4, 2, 1]:
                    sum_hq += cute.arch.shuffle_sync_bfly(
                        sum_hq,
                        offset=offset * V_PER_WARP_SMALL,
                        mask=-1,
                        mask_and_clamp=31,
                    )

                if k_local == 0:
                    v_global_out = v_tile * TILE_V_SMALL + v_idx
                    o[(0, i_n, i_hv, v_global_out)] = cutlass.Float16(sum_hq)

                cute.arch.barrier()

                for k_iter in range(NUM_K_ITERS_SMALL):
                    flat_idx = tidx + k_iter * 128
                    k_write = flat_idx // TILE_V_SMALL
                    v_write = flat_idx % TILE_V_SMALL
                    if k_write < TILE_K:
                        h_val = sData[(k_write, v_write, stage)]
                        v_global_write = v_tile * TILE_V_SMALL + v_write
                        h0_source[(state_slot, i_hv, k_write, v_global_write)] = h_val

                cute.arch.barrier()
        else:
            for v_tile_offset in range(num_v_tiles_per_block):
                v_tile = start_v_tile + v_tile_offset
                v_global = v_tile * TILE_V_SMALL + v_idx
                if k_local == 0:
                    o[(0, i_n, i_hv, v_global)] = cutlass.Float16(0.0)

    @cute.kernel
    def gdn_kernel_large_batch(
        tiled_copy_load: cute.TiledCopy,
        h0_source: cute.Tensor,
        state_indices: cute.Tensor,
        use_state_indices: Int32,
        smem_layout_staged: cute.Layout,
        num_v_tiles: Int32,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        context_lengths: cute.Tensor,
        o: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        use_qk_l2norm: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        batch_idx, _, _ = cute.arch.block_idx()
        H = q.layout.shape[2]
        HV = v.layout.shape[2]
        i_n = batch_idx // HV
        i_hv = batch_idx % HV
        i_h = i_hv // (HV // H)
        state_slot = cutlass.Int32(i_n)
        if use_state_indices != 0:
            state_slot = cutlass.Int32(state_indices[i_n])

        k_local = in_warp_tid // V_PER_WARP
        v_local = in_warp_tid % V_PER_WARP
        v_base = warp_idx * V_PER_WARP
        v_idx = v_base + v_local

        ctx_len = cutlass.Int32(context_lengths[i_n])

        if ctx_len > 0:
            smem = cutlass.utils.SmemAllocator()
            sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
            smem_o_layout = cute.make_layout((TILE_V,), stride=(1,))
            smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
            smem_k_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_q_layout = cute.make_layout((TILE_K,), stride=(1,))
            sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
            sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)

            if tidx < TILE_K:
                sK[tidx] = cutlass.Float32(k[i_n, 0, i_h, tidx])
                sQ[tidx] = cutlass.Float32(q[i_n, 0, i_h, tidx])

            gSrc_batch = h0_source[(state_slot, i_hv, None, None)]
            gSrc = cute.local_tile(gSrc_batch, (TILE_K, TILE_V), (0, None))
            thr_copy_load = tiled_copy_load.get_slice(tidx)

            prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles)
            for v_tile in range(prefetch_count):
                stage = v_tile % NUM_STAGES
                gSrc_tile = gSrc[(None, None, v_tile)]
                sData_stage = sData[(None, None, stage)]
                thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
                thr_sData = thr_copy_load.partition_D(sData_stage)
                cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                cute.arch.cp_async_commit_group()

            r_A_log = cutlass.Float32(A_log[i_hv])
            r_dt_bias = cutlass.Float32(dt_bias[i_hv])
            r_a = cutlass.Float32(a[i_n, 0, i_hv])
            r_b = cutlass.Float32(b[i_n, 0, i_hv])

            r_g = 0.0
            r_beta = 0.0
            if in_warp_tid == 0:
                x = r_a + r_dt_bias
                beta_x = softplus_beta * x
                softplus_x = 0.0
                if beta_x <= softplus_threshold:
                    exp_beta_x = cute.exp(beta_x)
                    log_input = cutlass.Float32(1.0 + exp_beta_x)
                    log_result = cutlass.Float32(cute.log(log_input))
                    softplus_x = cutlass.Float32(
                        (cutlass.Float32(1.0) / softplus_beta) * log_result
                    )
                else:
                    softplus_x = x
                r_g_value = -cute.exp(r_A_log) * softplus_x
                r_beta = 1.0 / (1.0 + cute.exp(-r_b))
                r_g = cute.exp(r_g_value)

            r_g = cute.arch.shuffle_sync(r_g, 0)
            r_beta = cute.arch.shuffle_sync(r_beta, 0)

            cute.arch.barrier()

            if use_qk_l2norm:
                sum_q_partial = 0.0
                sum_k_partial = 0.0
                if tidx < TILE_K:
                    q_val = sQ[tidx]
                    k_val = sK[tidx]
                    sum_q_partial = q_val * q_val
                    sum_k_partial = k_val * k_val

                for offset in [16, 8, 4, 2, 1]:
                    sum_q_partial += cute.arch.shuffle_sync_bfly(
                        sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    sum_k_partial += cute.arch.shuffle_sync_bfly(
                        sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31
                    )

                if in_warp_tid == 0:
                    smem_o[warp_idx] = sum_q_partial
                    smem_o[warp_idx + 8] = sum_k_partial
                cute.arch.barrier()

                inv_norm_q = 0.0
                inv_norm_k = 0.0
                if warp_idx == 0:
                    local_sum_q = 0.0
                    local_sum_k = 0.0
                    if in_warp_tid < NUM_WARPS_LARGE:
                        local_sum_q = smem_o[in_warp_tid]
                        local_sum_k = smem_o[in_warp_tid + 8]
                    for offset in [4, 2, 1]:
                        local_sum_q += cute.arch.shuffle_sync_bfly(
                            local_sum_q, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        local_sum_k += cute.arch.shuffle_sync_bfly(
                            local_sum_k, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    if in_warp_tid == 0:
                        smem_o[0] = cute.rsqrt(local_sum_q + 1e-6)
                        smem_o[1] = cute.rsqrt(local_sum_k + 1e-6)
                cute.arch.barrier()

                inv_norm_q = smem_o[0]
                inv_norm_k = smem_o[1]

                if tidx < TILE_K:
                    sK[tidx] = sK[tidx] * inv_norm_k
                    sQ[tidx] = sQ[tidx] * scale * inv_norm_q
                cute.arch.barrier()
            else:
                if tidx < TILE_K:
                    sQ[tidx] = sQ[tidx] * scale
                cute.arch.barrier()

            for v_tile in range(num_v_tiles):
                stage = v_tile % NUM_STAGES

                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                next_v_tile = v_tile + prefetch_count
                if next_v_tile < num_v_tiles:
                    next_stage = next_v_tile % NUM_STAGES
                    gSrc_next = gSrc[(None, None, next_v_tile)]
                    sData_next = sData[(None, None, next_stage)]
                    thr_gSrc = thr_copy_load.partition_S(gSrc_next)
                    thr_sData = thr_copy_load.partition_D(sData_next)
                    cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                    cute.arch.cp_async_commit_group()

                v_global = v_tile * TILE_V + v_idx
                r_v = cutlass.Float32(v[i_n, 0, i_hv, v_global])

                sum_hk = 0.0
                for k_iter in range(NUM_K_ITERS):
                    k_base = k_iter * ROWS_PER_ITER
                    k_idx = k_base + k_local
                    h_val = sData[(k_idx, v_idx, stage)] * r_g
                    r_k_val = sK[k_idx]
                    sum_hk += h_val * r_k_val

                for offset in [4, 2, 1]:
                    sum_hk += cute.arch.shuffle_sync_bfly(
                        sum_hk, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31
                    )

                v_new = (r_v - sum_hk) * r_beta
                v_new = cute.arch.shuffle_sync(v_new, v_local)

                sum_hq = 0.0
                for k_iter in range(NUM_K_ITERS):
                    k_base = k_iter * ROWS_PER_ITER
                    k_idx = k_base + k_local
                    h_old = sData[(k_idx, v_idx, stage)] * r_g
                    r_k_val = sK[k_idx]
                    r_q_val = sQ[k_idx]
                    h_new = h_old + r_k_val * v_new
                    sData[(k_idx, v_idx, stage)] = h_new
                    sum_hq += h_new * r_q_val

                for offset in [4, 2, 1]:
                    sum_hq += cute.arch.shuffle_sync_bfly(
                        sum_hq, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31
                    )

                if k_local == 0:
                    v_global_out = v_tile * TILE_V + v_idx
                    o[(i_n, 0, i_hv, v_global_out)] = cutlass.Float16(sum_hq)

                cute.arch.barrier()

                for k_iter in range(NUM_K_ITERS):
                    flat_idx = tidx + k_iter * 256
                    k_write = flat_idx // TILE_V
                    v_write = flat_idx % TILE_V
                    if k_write < TILE_K:
                        h_val = sData[(k_write, v_write, stage)]
                        v_global_write = v_tile * TILE_V + v_write
                        h0_source[(state_slot, i_hv, k_write, v_global_write)] = h_val

                cute.arch.barrier()
        else:
            for v_tile in range(num_v_tiles):
                v_global = v_tile * TILE_V + v_idx
                if k_local == 0:
                    o[(i_n, 0, i_hv, v_global)] = cutlass.Float16(0.0)

    @cute.kernel
    def gdn_kernel_large_batch_varlen(
        tiled_copy_load: cute.TiledCopy,
        h0_source: cute.Tensor,
        state_indices: cute.Tensor,
        use_state_indices: Int32,
        smem_layout_staged: cute.Layout,
        num_v_tiles: Int32,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        context_lengths: cute.Tensor,
        o: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        use_qk_l2norm: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        batch_idx, _, _ = cute.arch.block_idx()
        H = q.layout.shape[2]
        HV = v.layout.shape[2]
        i_n = batch_idx // HV
        i_hv = batch_idx % HV
        i_h = i_hv // (HV // H)
        state_slot = cutlass.Int32(i_n)
        if use_state_indices != 0:
            state_slot = cutlass.Int32(state_indices[i_n])

        k_local = in_warp_tid // V_PER_WARP
        v_local = in_warp_tid % V_PER_WARP
        v_base = warp_idx * V_PER_WARP
        v_idx = v_base + v_local

        ctx_len = cutlass.Int32(context_lengths[i_n])

        if ctx_len > 0:
            smem = cutlass.utils.SmemAllocator()
            sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
            smem_o_layout = cute.make_layout((TILE_V,), stride=(1,))
            smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
            smem_k_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_q_layout = cute.make_layout((TILE_K,), stride=(1,))
            sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
            sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)

            if tidx < TILE_K:
                sK[tidx] = cutlass.Float32(k[0, i_n, i_h, tidx])
                sQ[tidx] = cutlass.Float32(q[0, i_n, i_h, tidx])

            gSrc_batch = h0_source[(state_slot, i_hv, None, None)]
            gSrc = cute.local_tile(gSrc_batch, (TILE_K, TILE_V), (0, None))
            thr_copy_load = tiled_copy_load.get_slice(tidx)

            prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles)
            for v_tile in range(prefetch_count):
                stage = v_tile % NUM_STAGES
                gSrc_tile = gSrc[(None, None, v_tile)]
                sData_stage = sData[(None, None, stage)]
                thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
                thr_sData = thr_copy_load.partition_D(sData_stage)
                cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                cute.arch.cp_async_commit_group()

            r_A_log = cutlass.Float32(A_log[i_hv])
            r_dt_bias = cutlass.Float32(dt_bias[i_hv])
            r_a = cutlass.Float32(a[i_n, i_hv])
            r_b = cutlass.Float32(b[i_n, i_hv])

            r_g = 0.0
            r_beta = 0.0
            if in_warp_tid == 0:
                x = r_a + r_dt_bias
                beta_x = softplus_beta * x
                softplus_x = 0.0
                if beta_x <= softplus_threshold:
                    exp_beta_x = cute.exp(beta_x)
                    log_input = cutlass.Float32(1.0 + exp_beta_x)
                    log_result = cutlass.Float32(cute.log(log_input))
                    softplus_x = cutlass.Float32(
                        (cutlass.Float32(1.0) / softplus_beta) * log_result
                    )
                else:
                    softplus_x = x
                r_g_value = -cute.exp(r_A_log) * softplus_x
                r_beta = 1.0 / (1.0 + cute.exp(-r_b))
                r_g = cute.exp(r_g_value)

            r_g = cute.arch.shuffle_sync(r_g, 0)
            r_beta = cute.arch.shuffle_sync(r_beta, 0)

            cute.arch.barrier()

            if use_qk_l2norm:
                sum_q_partial = 0.0
                sum_k_partial = 0.0
                if tidx < TILE_K:
                    q_val = sQ[tidx]
                    k_val = sK[tidx]
                    sum_q_partial = q_val * q_val
                    sum_k_partial = k_val * k_val

                for offset in [16, 8, 4, 2, 1]:
                    sum_q_partial += cute.arch.shuffle_sync_bfly(
                        sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    sum_k_partial += cute.arch.shuffle_sync_bfly(
                        sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31
                    )

                if in_warp_tid == 0:
                    smem_o[warp_idx] = sum_q_partial
                    smem_o[warp_idx + 8] = sum_k_partial
                cute.arch.barrier()

                inv_norm_q = 0.0
                inv_norm_k = 0.0
                if warp_idx == 0:
                    local_sum_q = 0.0
                    local_sum_k = 0.0
                    if in_warp_tid < NUM_WARPS_LARGE:
                        local_sum_q = smem_o[in_warp_tid]
                        local_sum_k = smem_o[in_warp_tid + 8]
                    for offset in [4, 2, 1]:
                        local_sum_q += cute.arch.shuffle_sync_bfly(
                            local_sum_q, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        local_sum_k += cute.arch.shuffle_sync_bfly(
                            local_sum_k, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    if in_warp_tid == 0:
                        smem_o[0] = cute.rsqrt(local_sum_q + 1e-6)
                        smem_o[1] = cute.rsqrt(local_sum_k + 1e-6)
                cute.arch.barrier()

                inv_norm_q = smem_o[0]
                inv_norm_k = smem_o[1]

                if tidx < TILE_K:
                    sK[tidx] = sK[tidx] * inv_norm_k
                    sQ[tidx] = sQ[tidx] * scale * inv_norm_q
                cute.arch.barrier()
            else:
                if tidx < TILE_K:
                    sQ[tidx] = sQ[tidx] * scale
                cute.arch.barrier()

            for v_tile in range(num_v_tiles):
                stage = v_tile % NUM_STAGES

                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                next_v_tile = v_tile + prefetch_count
                if next_v_tile < num_v_tiles:
                    next_stage = next_v_tile % NUM_STAGES
                    gSrc_next = gSrc[(None, None, next_v_tile)]
                    sData_next = sData[(None, None, next_stage)]
                    thr_gSrc = thr_copy_load.partition_S(gSrc_next)
                    thr_sData = thr_copy_load.partition_D(sData_next)
                    cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                    cute.arch.cp_async_commit_group()

                v_global = v_tile * TILE_V + v_idx
                r_v = cutlass.Float32(v[0, i_n, i_hv, v_global])

                sum_hk = 0.0
                for k_iter in range(NUM_K_ITERS):
                    k_base = k_iter * ROWS_PER_ITER
                    k_idx = k_base + k_local
                    h_val = sData[(k_idx, v_idx, stage)] * r_g
                    r_k_val = sK[k_idx]
                    sum_hk += h_val * r_k_val

                for offset in [4, 2, 1]:
                    sum_hk += cute.arch.shuffle_sync_bfly(
                        sum_hk, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31
                    )

                v_new = (r_v - sum_hk) * r_beta
                v_new = cute.arch.shuffle_sync(v_new, v_local)

                sum_hq = 0.0
                for k_iter in range(NUM_K_ITERS):
                    k_base = k_iter * ROWS_PER_ITER
                    k_idx = k_base + k_local
                    h_old = sData[(k_idx, v_idx, stage)] * r_g
                    r_k_val = sK[k_idx]
                    r_q_val = sQ[k_idx]
                    h_new = h_old + r_k_val * v_new
                    sData[(k_idx, v_idx, stage)] = h_new
                    sum_hq += h_new * r_q_val

                for offset in [4, 2, 1]:
                    sum_hq += cute.arch.shuffle_sync_bfly(
                        sum_hq, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31
                    )

                if k_local == 0:
                    v_global_out = v_tile * TILE_V + v_idx
                    o[(0, i_n, i_hv, v_global_out)] = cutlass.Float16(sum_hq)

                cute.arch.barrier()

                for k_iter in range(NUM_K_ITERS):
                    flat_idx = tidx + k_iter * 256
                    k_write = flat_idx // TILE_V
                    v_write = flat_idx % TILE_V
                    if k_write < TILE_K:
                        h_val = sData[(k_write, v_write, stage)]
                        v_global_write = v_tile * TILE_V + v_write
                        h0_source[(state_slot, i_hv, k_write, v_global_write)] = h_val

                cute.arch.barrier()
        else:
            for v_tile in range(num_v_tiles):
                v_global = v_tile * TILE_V + v_idx
                if k_local == 0:
                    o[(0, i_n, i_hv, v_global)] = cutlass.Float16(0.0)

    return (
        gdn_kernel_small_batch,
        gdn_kernel_small_batch_varlen,
        gdn_kernel_large_batch,
        gdn_kernel_large_batch_varlen,
    )


def _create_jit_functions():
    """Create JIT-compiled launcher functions for all kernel variants."""

    gdn_small, gdn_small_varlen, gdn_large, gdn_large_varlen = _define_kernels()

    @cute.jit
    def run_small_batch(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        h0_source: cute.Tensor,
        state_indices: cute.Tensor,
        use_state_indices: Int32,
        context_lengths: cute.Tensor,
        o: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        use_initial_state: cutlass.Constexpr[bool],
        use_qk_l2norm: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        n_batch = q.layout.shape[0]
        hv_dim = v.layout.shape[2]
        v_dim = v.layout.shape[3]
        batch_size = n_batch * hv_dim

        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32,
            num_bits_per_copy=128,
        )
        num_v_tiles_small = cute.ceil_div(v_dim, TILE_V_SMALL)
        smem_layout_small = cute.make_layout(
            (TILE_K, TILE_V_SMALL, NUM_STAGES),
            stride=(TILE_V_SMALL_PADDED, 1, TILE_K * TILE_V_SMALL_PADDED),
        )
        thread_layout_small = cute.make_layout((32, 4), stride=(4, 1))
        val_layout_small = cute.make_layout((1, 4))
        tiled_copy_load_small = cute.make_tiled_copy_tv(
            copy_atom, thread_layout_small, val_layout_small
        )
        smem_bytes_small = (
            4 * TILE_K * TILE_V_SMALL_PADDED * NUM_STAGES
            + 4 * TILE_V_SMALL
            + 4 * TILE_K * 2
            + 64
        )

        gdn_small(
            tiled_copy_load_small,
            h0_source,
            state_indices,
            use_state_indices,
            smem_layout_small,
            num_v_tiles_small,
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            context_lengths,
            o,
            softplus_beta,
            softplus_threshold,
            scale,
            use_qk_l2norm,
        ).launch(
            grid=(batch_size * NUM_BLOCKS_PER_STATE_SMALL, 1, 1),
            block=[NUM_THREADS, 1, 1],
            smem=smem_bytes_small,
            stream=stream,
        )

    @cute.jit
    def run_small_batch_varlen(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        h0_source: cute.Tensor,
        state_indices: cute.Tensor,
        use_state_indices: Int32,
        context_lengths: cute.Tensor,
        o: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        use_initial_state: cutlass.Constexpr[bool],
        use_qk_l2norm: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        n_batch = q.layout.shape[0]
        hv_dim = v.layout.shape[2]
        v_dim = v.layout.shape[3]
        batch_size = n_batch * hv_dim

        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32,
            num_bits_per_copy=128,
        )
        num_v_tiles_small = cute.ceil_div(v_dim, TILE_V_SMALL)
        smem_layout_small = cute.make_layout(
            (TILE_K, TILE_V_SMALL, NUM_STAGES),
            stride=(TILE_V_SMALL_PADDED, 1, TILE_K * TILE_V_SMALL_PADDED),
        )
        thread_layout_small = cute.make_layout((32, 4), stride=(4, 1))
        val_layout_small = cute.make_layout((1, 4))
        tiled_copy_load_small = cute.make_tiled_copy_tv(
            copy_atom, thread_layout_small, val_layout_small
        )
        smem_bytes_small = (
            4 * TILE_K * TILE_V_SMALL_PADDED * NUM_STAGES
            + 4 * TILE_V_SMALL
            + 4 * TILE_K * 2
            + 64
        )

        gdn_small_varlen(
            tiled_copy_load_small,
            h0_source,
            state_indices,
            use_state_indices,
            smem_layout_small,
            num_v_tiles_small,
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            context_lengths,
            o,
            softplus_beta,
            softplus_threshold,
            scale,
            use_qk_l2norm,
        ).launch(
            grid=(batch_size * NUM_BLOCKS_PER_STATE_SMALL, 1, 1),
            block=[NUM_THREADS, 1, 1],
            smem=smem_bytes_small,
            stream=stream,
        )

    @cute.jit
    def run_large_batch(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        h0_source: cute.Tensor,
        state_indices: cute.Tensor,
        use_state_indices: Int32,
        context_lengths: cute.Tensor,
        o: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        use_initial_state: cutlass.Constexpr[bool],
        use_qk_l2norm: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        n_batch = q.layout.shape[0]
        hv_dim = v.layout.shape[2]
        v_dim = v.layout.shape[3]
        batch_size = n_batch * hv_dim

        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32,
            num_bits_per_copy=128,
        )
        num_v_tiles = cute.ceil_div(v_dim, TILE_V)
        base_smem_layout = cute.make_layout(
            (TILE_K, TILE_V, NUM_STAGES),
            stride=(TILE_V_PADDED, 1, TILE_K * TILE_V_PADDED),
        )
        thread_layout = cute.make_layout((32, 8), stride=(8, 1))
        val_layout = cute.make_layout((1, 4))
        tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)
        smem_bytes = (
            4 * TILE_K * TILE_V_PADDED * NUM_STAGES + 4 * TILE_V + 4 * TILE_K * 2 + 64
        )

        gdn_large(
            tiled_copy_load,
            h0_source,
            state_indices,
            use_state_indices,
            base_smem_layout,
            num_v_tiles,
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            context_lengths,
            o,
            softplus_beta,
            softplus_threshold,
            scale,
            use_qk_l2norm,
        ).launch(
            grid=(batch_size, 1, 1),
            block=[NUM_THREADS_LARGE, 1, 1],
            smem=smem_bytes,
            stream=stream,
        )

    @cute.jit
    def run_large_batch_varlen(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        h0_source: cute.Tensor,
        state_indices: cute.Tensor,
        use_state_indices: Int32,
        context_lengths: cute.Tensor,
        o: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        use_initial_state: cutlass.Constexpr[bool],
        use_qk_l2norm: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        n_batch = q.layout.shape[0]
        hv_dim = v.layout.shape[2]
        v_dim = v.layout.shape[3]
        batch_size = n_batch * hv_dim

        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32,
            num_bits_per_copy=128,
        )
        num_v_tiles = cute.ceil_div(v_dim, TILE_V)
        base_smem_layout = cute.make_layout(
            (TILE_K, TILE_V, NUM_STAGES),
            stride=(TILE_V_PADDED, 1, TILE_K * TILE_V_PADDED),
        )
        thread_layout = cute.make_layout((32, 8), stride=(8, 1))
        val_layout = cute.make_layout((1, 4))
        tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)
        smem_bytes = (
            4 * TILE_K * TILE_V_PADDED * NUM_STAGES + 4 * TILE_V + 4 * TILE_K * 2 + 64
        )

        gdn_large_varlen(
            tiled_copy_load,
            h0_source,
            state_indices,
            use_state_indices,
            base_smem_layout,
            num_v_tiles,
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            context_lengths,
            o,
            softplus_beta,
            softplus_threshold,
            scale,
            use_qk_l2norm,
        ).launch(
            grid=(batch_size, 1, 1),
            block=[NUM_THREADS_LARGE, 1, 1],
            smem=smem_bytes,
            stream=stream,
        )

    return (
        run_small_batch,
        run_small_batch_varlen,
        run_large_batch,
        run_large_batch_varlen,
    )


_jit_functions = None


def _get_jit_functions():
    global _jit_functions
    if _jit_functions is None:
        _jit_functions = _create_jit_functions()
    return _jit_functions


_compiled_kernels: Dict[Tuple, object] = {}


def _cp_dtype_fp16():
    return cp.float16


def _get_leading_dim(arr):
    for i, s in enumerate(arr.strides):
        if s == arr.dtype.itemsize:
            return i
    return len(arr.shape) - 1


def _make_placeholder_tensors(n, h, hv, k, v, varlen):
    dt = _cp_dtype_fp16()
    if varlen:
        q  = cp.zeros((1, n, h, k), dtype=dt)
        kt = cp.zeros((1, n, h, k), dtype=dt)
        vt = cp.zeros((1, n, hv, v), dtype=dt)
        a  = cp.zeros((n, hv), dtype=dt)
        b  = cp.zeros((n, hv), dtype=dt)
        o  = cp.zeros((1, n, hv, v), dtype=dt)
    else:
        q  = cp.zeros((n, 1, h, k), dtype=dt)
        kt = cp.zeros((n, 1, h, k), dtype=dt)
        vt = cp.zeros((n, 1, hv, v), dtype=dt)
        a  = cp.zeros((n, 1, hv), dtype=dt)
        b  = cp.zeros((n, 1, hv), dtype=dt)
        o  = cp.zeros((n, 1, hv, v), dtype=dt)
    return {
        "q": q, "k": kt, "v": vt, "a": a, "b": b,
        "A_log":      cp.zeros(hv, dtype=cp.float32),
        "dt_bias":    cp.zeros(hv, dtype=dt),
        "h0_source":  cp.zeros((n, hv, k, v), dtype=cp.float32),
        "state_indices": cp.arange(n, dtype=cp.int32),
        "context_lengths": cp.ones(n, dtype=cp.int32),
        "o": o,
    }


def _mark_gdn_qv_dynamic(tensor):
    so = (0, 1, 2, 3)
    return (tensor.mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(mode=0, stride_order=so)
            .mark_compact_shape_dynamic(mode=2, stride_order=so))

def _mark_gdn_1d_dynamic(tensor):
    return (tensor.mark_layout_dynamic(leading_dim=0)
            .mark_compact_shape_dynamic(mode=0, stride_order=(0,)))

def _mark_h0_source_dynamic(tensor):
    """Only pool/hv dims dynamic; stride static."""
    so = (0, 1, 2, 3)
    return tensor.mark_compact_shape_dynamic(mode=0, stride_order=so).mark_compact_shape_dynamic(
        mode=1, stride_order=so
    )

def _to_cute_tensors(ph):
    def wrap(arr, leading_dim=None, skip_dynamic=False):
        ct = from_dlpack(arr, assumed_align=16)
        if skip_dynamic:
            return ct
        ld = _get_leading_dim(arr) if leading_dim is None else leading_dim
        return ct.mark_layout_dynamic(leading_dim=ld)

    q = wrap(ph["q"])
    v = wrap(ph["v"])
    h0_src = _mark_h0_source_dynamic(from_dlpack(ph["h0_source"], assumed_align=32))
    state_indices = _mark_gdn_1d_dynamic(from_dlpack(ph["state_indices"], assumed_align=16))
    ctx = from_dlpack(ph["context_lengths"], assumed_align=16)
    ctx = ctx.mark_layout_dynamic(leading_dim=0).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
    return {
        "q":          _mark_gdn_qv_dynamic(q),
        "k":          wrap(ph["k"]),
        "v":          _mark_gdn_qv_dynamic(v),
        "a":          wrap(ph["a"]),
        "b":          wrap(ph["b"]),
        "A_log":      wrap(ph["A_log"], leading_dim=0),
        "dt_bias":    wrap(ph["dt_bias"], leading_dim=0),
        "h0_source":  h0_src,
        "state_indices": state_indices,
        "context_lengths": ctx,
        "o":          wrap(ph["o"]),
    }


def _make_aot_cute_tensors(n, h, hv, k, v, varlen):
    if varlen:
        shapes = {
            "q": (1, n, h, k),
            "k": (1, n, h, k),
            "v": (1, n, hv, v),
            "a": (n, hv),
            "b": (n, hv),
            "o": (1, n, hv, v),
        }
    else:
        shapes = {
            "q": (n, 1, h, k),
            "k": (n, 1, h, k),
            "v": (n, 1, hv, v),
            "a": (n, 1, hv),
            "b": (n, 1, hv),
            "o": (n, 1, hv, v),
        }

    def compact(dtype, name, assumed_align=16):
        shape = shapes[name]
        return aot_placeholders.make_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=assumed_align,
        )

    q = compact(cutlass.Float16, "q")
    v_tensor = compact(cutlass.Float16, "v")
    h0_source = aot_placeholders.make_compact_tensor(
        cutlass.Float32, (n, hv, k, v), stride_order=(3, 2, 1, 0), assumed_align=32
    )
    context_lengths = aot_placeholders.make_compact_tensor(
        cutlass.Int32, (n,), stride_order=(0,), assumed_align=16
    )
    state_indices = aot_placeholders.make_compact_tensor(
        cutlass.Int32, (n,), stride_order=(0,), assumed_align=16
    )
    return {
        "q": _mark_gdn_qv_dynamic(q),
        "k": compact(cutlass.Float16, "k").mark_layout_dynamic(leading_dim=3),
        "v": _mark_gdn_qv_dynamic(v_tensor),
        "a": compact(cutlass.Float16, "a").mark_layout_dynamic(leading_dim=len(shapes["a"]) - 1),
        "b": compact(cutlass.Float16, "b").mark_layout_dynamic(leading_dim=len(shapes["b"]) - 1),
        "A_log": aot_placeholders.make_compact_tensor(
            cutlass.Float32, (hv,), stride_order=(0,), assumed_align=16
        ).mark_layout_dynamic(leading_dim=0),
        "dt_bias": aot_placeholders.make_compact_tensor(
            cutlass.Float16, (hv,), stride_order=(0,), assumed_align=16
        ).mark_layout_dynamic(leading_dim=0),
        "h0_source": _mark_h0_source_dynamic(h0_source),
        "state_indices": _mark_gdn_1d_dynamic(state_indices),
        "context_lengths": _mark_gdn_1d_dynamic(context_lengths),
        "o": compact(cutlass.Float16, "o").mark_layout_dynamic(leading_dim=3),
    }


def _select_kernel(use_small_batch, varlen):
    run_small, run_small_varlen, run_large, run_large_varlen = _get_jit_functions()
    if use_small_batch:
        return run_small_varlen if varlen else run_small
    return run_large_varlen if varlen else run_large


def _compile_decode(n, h, hv, k, v, use_small_batch, varlen, stream, gpu_arch="", export_only=False):
    key = (use_small_batch, varlen)
    if key in _compiled_kernels:
        return _compiled_kernels[key]

    if export_only:
        t = _make_aot_cute_tensors(n, h, hv, k, v, varlen)
    else:
        ph = _make_placeholder_tensors(n, h, hv, k, v, varlen)
        t = _to_cute_tensors(ph)
    kernel_func = _select_kernel(use_small_batch, varlen)

    # Only the export path may pin a foreign target arch in the compile
    # options; a native JIT run must compile for the local GPU (see the
    # native-vs-cross note in cutedsl_utils/cutedsl_compile_wrapper.py).
    compile_opts = aot_placeholders.compile_options(
        f"--gpu-arch={gpu_arch}" if gpu_arch else ""
    ) if export_only else None
    compiled = cute.compile(
        kernel_func,
        t["q"], t["k"], t["v"], t["a"], t["b"],
        t["A_log"], t["dt_bias"], t["h0_source"], t["state_indices"], 1, t["context_lengths"], t["o"],
        softplus_beta=1.0,
        softplus_threshold=20.0,
        scale=k ** -0.5,
        use_initial_state=True,
        use_qk_l2norm=True,
        stream=stream,
        **(dict(options=compile_opts) if compile_opts else {}),
    )
    _compiled_kernels[key] = compiled
    return compiled


def export_gdn_decode(n, h, hv, k, v,
                      output_dir, file_name, function_prefix,
                      varlen=False, use_small_batch=False, gpu_arch=""):
    stream = aot_placeholders.make_stream()
    print("[gdn_decode] AOT compile varlen=%s small_batch=%s gpu_arch=%r" % (varlen, use_small_batch, gpu_arch or "default"))
    t0 = time.time()
    compiled = _compile_decode(
        n, h, hv, k, v, use_small_batch, varlen, stream, gpu_arch=gpu_arch, export_only=True
    )
    print("[gdn_decode] Compilation time: %.4fs" % (time.time() - t0))

    os.makedirs(output_dir, exist_ok=True)
    compiled.export_to_c(
        file_path=output_dir,
        file_name=file_name,
        function_prefix=function_prefix,
    )
    print("[gdn_decode] Exported to %s/%s.h and %s.o" % (output_dir, file_name, file_name))
    return compiled


def _softplus_np(x, beta=1.0, threshold=20.0):
    bx = beta * x
    if bx <= threshold:
        return (1.0 / beta) * np.log(1.0 + np.exp(bx))
    return float(x)


def _run_numpy_decode_reference(
    q_f32, k_f32, v_f32, a_f32, b_f32,
    A_log_f32, dt_bias_f32, h0_source_f32,
    n, h, hv, k_dim, v_dim, varlen, scale,
    context_lengths_np,
    use_qk_l2norm=True,
):
    h0 = h0_source_f32.copy()  # (n, hv, k, v) batch-dense

    if varlen:
        o_ref = np.zeros((1, n, hv, v_dim), dtype=np.float32)
    else:
        o_ref = np.zeros((n, 1, hv, v_dim), dtype=np.float32)

    for i_n in range(n):
        for i_hv in range(hv):
            i_h = i_hv // (hv // h) if h else 0
            H = h0[i_n, i_hv].copy()  # (K, V)

            if int(context_lengths_np[i_n]) <= 0:
                if varlen:
                    o_ref[0, i_n, i_hv, :] = 0.0
                else:
                    o_ref[i_n, 0, i_hv, :] = 0.0
                continue

            if varlen:
                q_vec = q_f32[0, i_n, i_h].astype(np.float64)
                k_vec = k_f32[0, i_n, i_h].astype(np.float64)
                v_vec = v_f32[0, i_n, i_hv].astype(np.float64)
                a_val = float(a_f32[i_n, i_hv])
                b_val = float(b_f32[i_n, i_hv])
            else:
                q_vec = q_f32[i_n, 0, i_h].astype(np.float64)
                k_vec = k_f32[i_n, 0, i_h].astype(np.float64)
                v_vec = v_f32[i_n, 0, i_hv].astype(np.float64)
                a_val = float(a_f32[i_n, 0, i_hv])
                b_val = float(b_f32[i_n, 0, i_hv])

            if use_qk_l2norm:
                nq = np.sqrt(np.sum(q_vec ** 2) + 1e-6)
                nk = np.sqrt(np.sum(k_vec ** 2) + 1e-6)
                q_eff = q_vec / nq * scale
                k_eff = k_vec / nk
            else:
                q_eff = q_vec * scale
                k_eff = k_vec

            A_val = float(A_log_f32[i_hv])
            dt_val = float(dt_bias_f32[i_hv])
            sp = _softplus_np(a_val + dt_val, 1.0, 20.0)
            g = np.exp(-np.exp(A_val) * sp)
            beta = 1.0 / (1.0 + np.exp(-b_val))

            H_gated = H * g
            correction = v_vec - H_gated.T @ k_eff
            v_new = correction * beta
            H_new = H_gated + np.outer(k_eff, v_new)
            out = H_new.T @ q_eff

            if varlen:
                o_ref[0, i_n, i_hv] = out
            else:
                o_ref[i_n, 0, i_hv] = out
            h0[i_n, i_hv] = H_new
    return o_ref, h0


def _decode_context_lengths_array(n, preset):
    """INT32 [n] for decode: >0 runs step, <=0 skips (o=0, h0 unchanged)."""
    if preset == "all_ones" or preset is None:
        return np.ones(n, dtype=np.int32)
    if preset == "first_half_active":
        mid = (n + 1) // 2
        return np.array([1 if i < mid else 0 for i in range(n)], dtype=np.int32)
    raise ValueError("Unknown context_lengths preset: %r" % (preset,))


def run_test_decode(n, h, hv, k, v, varlen=False,
                    skip_ref_check=False, tolerance=0.1,
                    warmup=3, iterations=100, gpu_arch="",
                    context_lengths_preset="all_ones"):
    dt = _cp_dtype_fp16()
    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    use_small_batch = n < SMALL_BATCH_THRESHOLD

    # Generate float32 data (used for both kernel and reference). h0 is batch-dense [n, hv, k, v].
    if varlen:
        q_f32  = np.random.randn(1, n, h, k).astype(np.float32) * 0.1
        k_f32  = np.random.randn(1, n, h, k).astype(np.float32) * 0.1
        v_f32  = np.random.randn(1, n, hv, v).astype(np.float32) * 0.1
        a_f32  = np.random.randn(n, hv).astype(np.float32) * 0.1
        b_f32  = np.random.randn(n, hv).astype(np.float32) * 0.1
    else:
        q_f32  = np.random.randn(n, 1, h, k).astype(np.float32) * 0.1
        k_f32  = np.random.randn(n, 1, h, k).astype(np.float32) * 0.1
        v_f32  = np.random.randn(n, 1, hv, v).astype(np.float32) * 0.1
        a_f32  = np.random.randn(n, 1, hv).astype(np.float32) * 0.1
        b_f32  = np.random.randn(n, 1, hv).astype(np.float32) * 0.1
    A_log_f32  = np.random.randn(hv).astype(np.float32) * 0.1
    dt_bias_f32 = np.random.randn(hv).astype(np.float32) * 0.1
    h0_f32 = np.random.randn(n, hv, k, v).astype(np.float32) * 0.01

    ctx_np = _decode_context_lengths_array(n, context_lengths_preset)

    ph = {
        "q":          cp.asarray(q_f32).astype(dt),
        "k":          cp.asarray(k_f32).astype(dt),
        "v":          cp.asarray(v_f32).astype(dt),
        "a":          cp.asarray(a_f32).astype(dt),
        "b":          cp.asarray(b_f32).astype(dt),
        "A_log":      cp.asarray(A_log_f32),
        "dt_bias":    cp.asarray(dt_bias_f32).astype(dt),
        "h0_source":  cp.asarray(h0_f32),
        "context_lengths": cp.asarray(ctx_np),
        "o":          cp.zeros((1, n, hv, v) if varlen else (n, 1, hv, v), dtype=dt),
    }

    compiled = _compile_decode(n, h, hv, k, v, use_small_batch, varlen, stream, gpu_arch=gpu_arch)
    t = _to_cute_tensors(ph)
    args = (
        t["q"], t["k"], t["v"], t["a"], t["b"],
        t["A_log"], t["dt_bias"], t["h0_source"], t["state_indices"], 1, t["context_lengths"], t["o"],
        stream,
    )

    for _ in range(warmup):
        compiled(*args)
    cp.cuda.get_current_stream().synchronize()

    if not skip_ref_check:
        ph["h0_source"] = cp.asarray(h0_f32)  # kernel mutates h0
        t = _to_cute_tensors(ph)
        args = (
            t["q"], t["k"], t["v"], t["a"], t["b"],
            t["A_log"], t["dt_bias"], t["h0_source"], t["state_indices"], 1, t["context_lengths"], t["o"],
            stream,
        )
        compiled(*args)
        cp.cuda.get_current_stream().synchronize()

        o_ref, h0_ref = _run_numpy_decode_reference(
            q_f32, k_f32, v_f32, a_f32, b_f32,
            A_log_f32, dt_bias_f32, h0_f32,
            n, h, hv, k, v, varlen, scale=k ** -0.5,
            context_lengths_np=ctx_np,
            use_qk_l2norm=True,
        )
        o_kernel = cp.asnumpy(ph["o"]).astype(np.float32)
        max_err = np.max(np.abs(o_kernel - o_ref))
        print("[gdn_decode] Max abs error vs NumPy ref: %.6f (tol=%.4f)" % (max_err, tolerance))
        np.testing.assert_allclose(o_kernel, o_ref, atol=tolerance, rtol=1e-2)
        h0_kernel = cp.asnumpy(ph["h0_source"]).astype(np.float32)
        max_err_h0 = np.max(np.abs(h0_kernel - h0_ref))
        print("[gdn_decode] h0 max abs error vs NumPy ref: %.6f (tol=%.4f)" % (max_err_h0, tolerance))
        np.testing.assert_allclose(h0_kernel, h0_ref, atol=tolerance, rtol=1e-2)
        print("[gdn_decode] Reference check PASSED (o + h0). context_lengths preset=%r" % (context_lengths_preset,))

    # Benchmark
    ph["h0_source"] = cp.asarray(h0_f32)
    t = _to_cute_tensors(ph)
    args = (
        t["q"], t["k"], t["v"], t["a"], t["b"],
        t["A_log"], t["dt_bias"], t["h0_source"], t["state_indices"], 1, t["context_lengths"], t["o"],
        stream,
    )
    t0 = time.perf_counter()
    for _ in range(iterations):
        compiled(*args)
    cp.cuda.get_current_stream().synchronize()
    us = (time.perf_counter() - t0) * 1e6 / iterations
    print(
        "[gdn_decode] Latency: %.4f us "
        "(n=%d h=%d hv=%d k=%d v=%d varlen=%s)" % (us, n, h, hv, k, v, varlen)
    )
    return us


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = _parsed_args
    np.random.seed(42)

    if args.export_only:
        export_gdn_decode(
            n=AOT_PLACEHOLDER_N,
            h=AOT_PLACEHOLDER_H,
            hv=AOT_PLACEHOLDER_HV,
            k=AOT_PLACEHOLDER_K,
            v=AOT_PLACEHOLDER_V,
            output_dir=args.output_dir,
            file_name=args.file_name,
            function_prefix=args.function_prefix,
            varlen=args.varlen,
            use_small_batch=args.small_batch,
            gpu_arch=args.gpu_arch,
        )
        return

    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU required.")
    cp.random.seed(42)

    run_test_decode(
        n=args.n, h=args.h, hv=args.hv, k=args.k, v=args.v,
        varlen=args.varlen,
        skip_ref_check=args.skip_ref_check,
        tolerance=args.tolerance,
        warmup=args.warmup,
        iterations=args.iterations,
        gpu_arch=args.gpu_arch,
        context_lengths_preset=args.context_lengths_preset,
    )


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CuTe DSL GDN decode: AOT export and test (CuPy only, Ampere+)."
    )
    p.add_argument("--export_only", action="store_true")
    p.add_argument("--output_dir", type=str, default="./gdn_aot_artifacts")
    p.add_argument("--file_name", type=str, default="gdn_decode")
    p.add_argument("--function_prefix", type=str, default="gdn_decode")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--h", type=int, default=14)
    p.add_argument("--hv", type=int, default=14)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--v", type=int, default=128)
    p.add_argument("--varlen", action="store_true",
                   help="Use varlen layout (1,N,...) instead of (N,1,...)")
    p.add_argument(
        "--context_lengths_preset",
        type=str,
        default="all_ones",
        choices=("all_ones", "first_half_active"),
        help="Decode mask test: all_ones=every batch active; "
        "first_half_active=first ceil(n/2) rows active, rest skipped (o=0, h0 unchanged).",
    )
    p.add_argument("--small_batch", action="store_true",
                   help="Use small-batch kernel (multi-block-per-state)")
    p.add_argument("--skip_ref_check", action="store_true")
    p.add_argument("--tolerance", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--gpu_arch", type=str, default="",
                   help="Target GPU arch for export (e.g. sm_87 for Orin). Empty = current GPU.")
    return p.parse_known_args(args=argv)[0]


if __name__ == "__main__":
    _parsed_args = _parse_args(_saved_argv)
    main()
