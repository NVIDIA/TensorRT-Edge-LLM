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

# This version add support for prefill(seq_len T > 1) stage. Uses CuPy only (no PyTorch) and
# is built for edge-llm AOT export and C++ plugin integration.
# CuTe DSL GDN prefill (seq_len T > 1). Test: python gdn_prefill.py --n 16 ... ; AOT: --export_only ...

import argparse
import os
import sys
import time

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

TILE_K = 128        # state head dim K
TILE_V = 32         # V columns per smem tile
TILE_V_PADDED = 36  # padded stride to avoid smem bank conflicts
NUM_THREADS = 256   # 8 warps per block
NUM_WARPS = 8
V_PER_WARP = TILE_V // NUM_WARPS    # 4 V cols per warp
ROWS_PER_ITER = 32 // V_PER_WARP    # 8 K rows covered per warp-iter
NUM_K_ITERS = TILE_K // ROWS_PER_ITER  # 16 iters to cover full K

# AOT placeholders; h0_source batch-dense [n, hv, k, v], assumed_align=32 for cp.async.
AOT_PLACEHOLDER_N = 1
AOT_PLACEHOLDER_H = 8
AOT_PLACEHOLDER_HV = 8
AOT_PLACEHOLDER_K = 128
AOT_PLACEHOLDER_V = 128
AOT_PLACEHOLDER_SEQLEN = 2


def _define_prefill_kernel():
    @cute.kernel
    def gdn_kernel_prefill(
        tiled_copy_load: cute.TiledCopy,
        h0_source: cute.Tensor,
        state_indices: cute.Tensor,
        use_state_indices: Int32,
        smem_layout: cute.Layout,
        num_v_tiles: Int32,
        seq_len: Int32,
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
        smem = cutlass.utils.SmemAllocator()
        sH = smem.allocate_tensor(cutlass.Float32, smem_layout, 128)
        smem_qk_layout = cute.make_layout((TILE_K,), stride=(1,))
        sK = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 128)
        sQ = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 128)
        smem_norm_layout = cute.make_layout((TILE_V,), stride=(1,))
        sNorm = smem.allocate_tensor(cutlass.Float32, smem_norm_layout, 128)
        gSrc_batch = h0_source[(state_slot, i_hv, None, None)]
        gSrc = cute.local_tile(gSrc_batch, (TILE_K, TILE_V), (0, None))
        thr_copy_load = tiled_copy_load.get_slice(tidx)

        # ---- Outer loop: v_tile ------------------------------------------------
        for v_tile in range(num_v_tiles):
            v_global_base = v_tile * TILE_V

            # Load H[:, v_tile] from h0_source into sH (async, then wait).
            gSrc_tile = gSrc[(None, None, v_tile)]
            thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
            thr_sH = thr_copy_load.partition_D(sH)
            cute.copy(tiled_copy_load, thr_gSrc, thr_sH)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            # ---- Inner loop: t ----
            for t in range(seq_len):
                row_valid = cutlass.Int32(t) < cutlass.Int32(context_lengths[i_n])

                # Load q[t] and k[t] into smem.
                if tidx < TILE_K:
                    sK[tidx] = cutlass.Float32(k[i_n, t, i_h, tidx])
                    sQ[tidx] = cutlass.Float32(q[i_n, t, i_h, tidx])

                # Compute gate g and update gate beta (only lane 0 of each warp).
                r_A_log = cutlass.Float32(A_log[i_hv])
                r_dt_bias = cutlass.Float32(dt_bias[i_hv])
                r_a = cutlass.Float32(a[i_n, t, i_hv])
                r_b = cutlass.Float32(b[i_n, t, i_hv])

                r_g = 0.0
                r_beta = 0.0
                if in_warp_tid == 0:
                    x = r_a + r_dt_bias
                    beta_x = softplus_beta * x
                    softplus_x = x  # default (linear); overwritten when below threshold
                    if beta_x <= softplus_threshold:
                        exp_bx = cute.exp(beta_x)
                        softplus_x = cutlass.Float32(
                            (cutlass.Float32(1.0) / softplus_beta)
                            * cutlass.Float32(cute.log(cutlass.Float32(1.0 + exp_bx)))
                        )
                    r_g = cute.exp(-cute.exp(r_A_log) * softplus_x)
                    r_beta = 1.0 / (1.0 + cute.exp(-r_b))

                r_g = cute.arch.shuffle_sync(r_g, 0)
                r_beta = cute.arch.shuffle_sync(r_beta, 0)
                cute.arch.barrier()  # sK, sQ, r_g, r_beta ready

                # Optional L2-norm of q and k (QK norm variant).
                if use_qk_l2norm:
                    sum_q_partial = 0.0
                    sum_k_partial = 0.0
                    if tidx < TILE_K:
                        q_val = sQ[tidx]
                        k_val = sK[tidx]
                        sum_q_partial = q_val * q_val
                        sum_k_partial = k_val * k_val
                    # Warp-level reduce (full 32 lanes, TILE_K=128 → 4 warps each cover 32 elems).
                    for offset in [16, 8, 4, 2, 1]:
                        sum_q_partial += cute.arch.shuffle_sync_bfly(
                            sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        sum_k_partial += cute.arch.shuffle_sync_bfly(
                            sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    # Lane 0 of each warp writes partial sums; warp 0 does final reduction.
                    if in_warp_tid == 0:
                        sNorm[warp_idx] = sum_q_partial
                        sNorm[warp_idx + NUM_WARPS] = sum_k_partial
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
                                local_sum_q, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            local_sum_k += cute.arch.shuffle_sync_bfly(
                                local_sum_k, offset=offset, mask=-1, mask_and_clamp=31
                            )
                        if in_warp_tid == 0:
                            sNorm[0] = cute.rsqrt(local_sum_q + 1e-6)
                            sNorm[1] = cute.rsqrt(local_sum_k + 1e-6)
                    cute.arch.barrier()
                    inv_norm_q = sNorm[0]
                    inv_norm_k = sNorm[1]
                    if tidx < TILE_K:
                        sK[tidx] = sK[tidx] * inv_norm_k
                        sQ[tidx] = sQ[tidx] * scale * inv_norm_q
                    cute.arch.barrier()
                else:
                    if tidx < TILE_K:
                        sQ[tidx] = sQ[tidx] * scale
                    cute.arch.barrier()

                # Read v and recurrent update only for valid sequence positions (t < context_lengths[i_n]).
                v_global = v_global_base + v_idx
                if row_valid:
                    r_v = cutlass.Float32(v[i_n, t, i_hv, v_global])

                    # Step 1: compute correction = v - (g * H).T @ k.
                    sum_hk = 0.0
                    for k_iter in range(NUM_K_ITERS):
                        k_idx = k_iter * ROWS_PER_ITER + k_local
                        h_val = sH[(k_idx, v_idx)] * r_g
                        sum_hk += h_val * sK[k_idx]
                    for offset in [4, 2, 1]:
                        sum_hk += cute.arch.shuffle_sync_bfly(
                            sum_hk, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31
                        )
                    v_new = (r_v - sum_hk) * r_beta
                    v_new = cute.arch.shuffle_sync(v_new, v_local)

                    # Step 2: update H and compute output o = H_new.T @ q.
                    sum_hq = 0.0
                    for k_iter in range(NUM_K_ITERS):
                        k_idx = k_iter * ROWS_PER_ITER + k_local
                        h_old = sH[(k_idx, v_idx)] * r_g
                        h_new = h_old + sK[k_idx] * v_new
                        sH[(k_idx, v_idx)] = h_new
                        sum_hq += h_new * sQ[k_idx]
                    for offset in [4, 2, 1]:
                        sum_hq += cute.arch.shuffle_sync_bfly(
                            sum_hq, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31
                        )

                    if k_local == 0:
                        o[(i_n, t, i_hv, v_global)] = cutlass.Float16(sum_hq)
                else:
                    if k_local == 0:
                        o[(i_n, t, i_hv, v_global)] = cutlass.Float16(0.0)

                cute.arch.barrier()
            # ---- end t loop ----------------------------------------------------

            # Write final H[:, v_tile] from sH back to h0_source (batch-dense [n, hv, k, v]).
            # Each thread writes its portion: (k_idx, v_idx) pairs.
            for k_iter in range(NUM_K_ITERS):
                k_write = k_iter * ROWS_PER_ITER + k_local
                h_val = sH[(k_write, v_idx)]
                h0_source[(state_slot, i_hv, k_write, v_global_base + v_idx)] = h_val
            cute.arch.barrier()
        # ---- end v_tile loop ---------------------------------------------------

    return gdn_kernel_prefill


_jit_function_prefill = None


def _create_jit_function_prefill():
    gdn_prefill_kernel = _define_prefill_kernel()

    @cute.jit
    def run_prefill(
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
        seq_len: Int32,
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
        num_blocks = n_batch * hv_dim

        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32,
            num_bits_per_copy=128,
        )
        num_v_tiles = cute.ceil_div(v_dim, TILE_V)
        smem_layout = cute.make_layout(
            (TILE_K, TILE_V),
            stride=(TILE_V_PADDED, 1),
        )
        thread_layout = cute.make_layout((32, 8), stride=(8, 1))
        val_layout = cute.make_layout((1, 4))
        tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)
        smem_bytes = (
            4 * TILE_K * TILE_V_PADDED + 4 * TILE_K * 2 + 4 * TILE_V + 128
        )

        gdn_prefill_kernel(
            tiled_copy_load,
            h0_source,
            state_indices,
            use_state_indices,
            smem_layout,
            num_v_tiles,
            seq_len,
            q, k, v, a, b,
            A_log, dt_bias, context_lengths, o,
            softplus_beta, softplus_threshold, scale,
            use_qk_l2norm,
        ).launch(
            grid=(num_blocks, 1, 1),
            block=[NUM_THREADS, 1, 1],
            smem=smem_bytes,
            stream=stream,
        )

    return run_prefill


def _get_jit_function_prefill():
    global _jit_function_prefill
    if _jit_function_prefill is None:
        _jit_function_prefill = _create_jit_function_prefill()
    return _jit_function_prefill


_compiled_kernels_prefill = {}


def _cp_dtype_fp16():
    return cp.float16


def _get_leading_dim(arr):
    for i, s in enumerate(arr.strides):
        if s == arr.dtype.itemsize:
            return i
    return len(arr.shape) - 1


def _make_placeholder_tensors(n, h, hv, k, v, seq_len):
    dt = _cp_dtype_fp16()
    return {
        "q":          cp.zeros((n, seq_len, h, k), dtype=dt),
        "k":          cp.zeros((n, seq_len, h, k), dtype=dt),
        "v":          cp.zeros((n, seq_len, hv, v), dtype=dt),
        "a":          cp.zeros((n, seq_len, hv), dtype=dt),
        "b":          cp.zeros((n, seq_len, hv), dtype=dt),
        "A_log":      cp.zeros(hv, dtype=cp.float32),
        "dt_bias":    cp.zeros(hv, dtype=dt),
        "h0_source":  cp.zeros((n, hv, k, v), dtype=cp.float32),
        "state_indices": cp.arange(n, dtype=cp.int32),
        "context_lengths": cp.full((n,), seq_len, dtype=cp.int32),
        "o":          cp.zeros((n, seq_len, hv, v), dtype=dt),
    }


def _mark_gdn_prefill_qv_dynamic(tensor):
    so = (0, 1, 2, 3)
    return (tensor.mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(mode=0, stride_order=so)
            .mark_compact_shape_dynamic(mode=1, stride_order=so)
            .mark_compact_shape_dynamic(mode=2, stride_order=so))

def _mark_gdn_1d_dynamic(tensor):
    return (tensor.mark_layout_dynamic(leading_dim=0)
            .mark_compact_shape_dynamic(mode=0, stride_order=(0,)))

def _mark_h0_source_dynamic(tensor):
    """Batch (n) and hv dims dynamic; stride stays static."""
    so = (0, 1, 2, 3)
    return tensor.mark_compact_shape_dynamic(mode=0, stride_order=so).mark_compact_shape_dynamic(
        mode=1, stride_order=so
    )

def _to_cute_tensors(ph):
    def wrap(arr, leading_dim=None, skip_dynamic=False, assumed_align=16):
        ct = from_dlpack(arr, assumed_align=assumed_align)
        if skip_dynamic:
            return ct
        ld = _get_leading_dim(arr) if leading_dim is None else leading_dim
        return ct.mark_layout_dynamic(leading_dim=ld)

    q = wrap(ph["q"])
    v = wrap(ph["v"])
    h0_src = _mark_h0_source_dynamic(from_dlpack(ph["h0_source"], assumed_align=32))
    state_indices = from_dlpack(ph["state_indices"], assumed_align=16)
    state_indices = _mark_gdn_1d_dynamic(state_indices)
    ctx = from_dlpack(ph["context_lengths"], assumed_align=16)
    ctx = ctx.mark_layout_dynamic(leading_dim=0).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
    return {
        "q":          _mark_gdn_prefill_qv_dynamic(q),
        "k":          wrap(ph["k"]),
        "v":          _mark_gdn_prefill_qv_dynamic(v),
        "a":          wrap(ph["a"]),
        "b":          wrap(ph["b"]),
        "A_log":      wrap(ph["A_log"], leading_dim=0),
        "dt_bias":    wrap(ph["dt_bias"], leading_dim=0),
        "h0_source":  h0_src,
        "state_indices": state_indices,
        "context_lengths": ctx,
        "o":          wrap(ph["o"]),
    }


def _make_aot_cute_tensors(n, h, hv, k, v, seq_len):
    def compact(dtype, shape, assumed_align=16):
        return aot_placeholders.make_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=assumed_align,
        )

    q = compact(cutlass.Float16, (n, seq_len, h, k))
    v_tensor = compact(cutlass.Float16, (n, seq_len, hv, v))
    h0_source = compact(cutlass.Float32, (n, hv, k, v), assumed_align=32)
    state_indices = compact(cutlass.Int32, (n,))
    context_lengths = compact(cutlass.Int32, (n,))
    return {
        "q": _mark_gdn_prefill_qv_dynamic(q.mark_layout_dynamic(leading_dim=3)),
        "k": compact(cutlass.Float16, (n, seq_len, h, k)).mark_layout_dynamic(leading_dim=3),
        "v": _mark_gdn_prefill_qv_dynamic(v_tensor.mark_layout_dynamic(leading_dim=3)),
        "a": compact(cutlass.Float16, (n, seq_len, hv)).mark_layout_dynamic(leading_dim=2),
        "b": compact(cutlass.Float16, (n, seq_len, hv)).mark_layout_dynamic(leading_dim=2),
        "A_log": compact(cutlass.Float32, (hv,)).mark_layout_dynamic(leading_dim=0),
        "dt_bias": compact(cutlass.Float16, (hv,)).mark_layout_dynamic(leading_dim=0),
        "h0_source": _mark_h0_source_dynamic(h0_source),
        "state_indices": _mark_gdn_1d_dynamic(state_indices),
        "context_lengths": _mark_gdn_1d_dynamic(context_lengths),
        "o": compact(cutlass.Float16, (n, seq_len, hv, v)).mark_layout_dynamic(leading_dim=3),
    }


def _compile_prefill(n, h, hv, k, v, seq_len, stream, gpu_arch="", export_only=False):
    if "_" in _compiled_kernels_prefill:
        return _compiled_kernels_prefill["_"]

    if export_only:
        t = _make_aot_cute_tensors(n, h, hv, k, v, seq_len)
    else:
        ph = _make_placeholder_tensors(n, h, hv, k, v, seq_len)
        t = _to_cute_tensors(ph)
    run_prefill = _get_jit_function_prefill()
    # Only the export path may pin a foreign target arch in the compile
    # options; a native JIT run must compile for the local GPU (see the
    # native-vs-cross note in cutedsl_utils/cutedsl_compile_wrapper.py).
    compile_opts = aot_placeholders.compile_options(
        f"--gpu-arch={gpu_arch}" if gpu_arch else ""
    ) if export_only else None
    compiled = cute.compile(
        run_prefill,
        t["q"], t["k"], t["v"], t["a"], t["b"],
        t["A_log"], t["dt_bias"], t["h0_source"], t["state_indices"], 1, t["context_lengths"], t["o"],
        seq_len,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        scale=k ** -0.5,
        use_initial_state=True,
        use_qk_l2norm=True,
        stream=stream,
        **(dict(options=compile_opts) if compile_opts else {}),
    )
    _compiled_kernels_prefill["_"] = compiled
    return compiled


def export_gdn_prefill(n, h, hv, k, v, seq_len,
                       output_dir, file_name, function_prefix, gpu_arch=""):
    if seq_len < 2:
        raise ValueError("Prefill requires seq_len >= 2.")
    stream = aot_placeholders.make_stream()
    print("[gdn_prefill] AOT compile gpu_arch=%r" % (gpu_arch or "default"))
    t0 = time.time()
    compiled = _compile_prefill(
        n, h, hv, k, v, seq_len, stream, gpu_arch=gpu_arch, export_only=True
    )
    print("[gdn_prefill] Compilation time: %.4fs" % (time.time() - t0))

    os.makedirs(output_dir, exist_ok=True)
    compiled.export_to_c(
        file_path=output_dir,
        file_name=file_name,
        function_prefix=function_prefix,
    )
    print("[gdn_prefill] Exported to %s/%s.h and %s.o" % (output_dir, file_name, file_name))
    return compiled


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
    use_qk_l2norm=True,
):
    h0 = h0_source_f32.copy()  # (n, HV, K, V) batch-dense
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
                g = np.exp(-np.exp(A_val) * sp)
                beta = 1.0 / (1.0 + np.exp(-b_val))

                H_gated = H * g                           # (K, V)
                correction = v_vec - H_gated.T @ k_eff    # (V,)
                v_new = correction * beta                  # (V,)
                H = H_gated + np.outer(k_eff, v_new)      # (K, V)
                o_ref[i_n, t, i_hv] = H.T @ q_eff
            h0[i_n, i_hv] = H
    return o_ref, h0


def _prefill_context_lengths_array(n, seq_len, preset):
    """INT32 [n]: valid token count per row; must be in [0, seq_len]."""
    if preset == "full" or preset is None:
        return np.full((n,), seq_len, dtype=np.int32)
    if preset == "half":
        t = max(1, seq_len // 2)
        return np.full((n,), t, dtype=np.int32)
    if preset == "staggered":
        out = np.empty((n,), dtype=np.int32)
        span = max(1, seq_len - 1)
        for i in range(n):
            out[i] = max(1, seq_len - (i % span))
        return out
    raise ValueError("Unknown context_lengths preset: %r" % (preset,))


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test_prefill(n, h, hv, k, v, seq_len,
                     skip_ref_check=False, tolerance=0.2,
                     warmup=3, iterations=100, gpu_arch="",
                     context_lengths_preset="full"):
    if seq_len < 2:
        raise ValueError("Prefill requires seq_len >= 2.")

    dt = _cp_dtype_fp16()
    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)

    # Generate float32 data, cast to fp16 for kernel input. h0 batch-dense [n, hv, k, v].
    q_f32  = np.random.randn(n, seq_len, h, k).astype(np.float32) * 0.1
    k_f32  = np.random.randn(n, seq_len, h, k).astype(np.float32) * 0.1
    v_f32  = np.random.randn(n, seq_len, hv, v).astype(np.float32) * 0.1
    a_f32  = np.random.randn(n, seq_len, hv).astype(np.float32) * 0.1
    b_f32  = np.random.randn(n, seq_len, hv).astype(np.float32) * 0.1
    A_log_f32  = np.random.randn(hv).astype(np.float32) * 0.1
    dt_bias_f32 = np.random.randn(hv).astype(np.float32) * 0.1
    h0_f32 = np.random.randn(n, hv, k, v).astype(np.float32) * 0.01

    ctx_np = _prefill_context_lengths_array(n, seq_len, context_lengths_preset)
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
        "o":          cp.zeros((n, seq_len, hv, v), dtype=dt),
    }

    compiled = _compile_prefill(n, h, hv, k, v, seq_len, stream, gpu_arch=gpu_arch)
    t = _to_cute_tensors(ph)
    args = (
        t["q"], t["k"], t["v"], t["a"], t["b"],
        t["A_log"], t["dt_bias"], t["h0_source"], t["state_indices"], 1, t["context_lengths"], t["o"],
        seq_len,
        stream,
    )

    for _ in range(warmup):
        compiled(*args)
    cp.cuda.get_current_stream().synchronize()

    if not skip_ref_check:
        # Reset h0_source to original (kernel mutates it).
        ph["h0_source"] = cp.asarray(h0_f32)
        t = _to_cute_tensors(ph)
        args = (
            t["q"], t["k"], t["v"], t["a"], t["b"],
            t["A_log"], t["dt_bias"], t["h0_source"], t["state_indices"], 1, t["context_lengths"], t["o"],
            seq_len,
            stream,
        )
        compiled(*args)
        cp.cuda.get_current_stream().synchronize()

        o_ref, h0_ref = _run_numpy_prefill_reference(
            q_f32, k_f32, v_f32, a_f32, b_f32,
            A_log_f32, dt_bias_f32, h0_f32,
            n, h, hv, k, v, seq_len, scale=k ** -0.5,
            context_lengths_np=ctx_np,
            use_qk_l2norm=True,
        )
        o_kernel = cp.asnumpy(ph["o"]).astype(np.float32)
        max_err = np.max(np.abs(o_kernel - o_ref))
        print("[gdn_prefill] Max abs error vs NumPy ref: %.6f (tol=%.4f)" % (max_err, tolerance))
        np.testing.assert_allclose(o_kernel, o_ref, atol=tolerance, rtol=1e-2)
        h0_kernel = cp.asnumpy(ph["h0_source"]).astype(np.float32)
        max_err_h0 = np.max(np.abs(h0_kernel - h0_ref))
        print("[gdn_prefill] h0 max abs error vs NumPy ref: %.6f (tol=%.4f)" % (max_err_h0, tolerance))
        np.testing.assert_allclose(h0_kernel, h0_ref, atol=tolerance, rtol=1e-2)
        print("[gdn_prefill] Reference check PASSED (o + h0). context_lengths preset=%r" % (context_lengths_preset,))

    # Benchmark
    ph["h0_source"] = cp.asarray(h0_f32)
    t = _to_cute_tensors(ph)
    args = (
        t["q"], t["k"], t["v"], t["a"], t["b"],
        t["A_log"], t["dt_bias"], t["h0_source"], t["state_indices"], 1, t["context_lengths"], t["o"],
        seq_len,
        stream,
    )
    t0 = time.perf_counter()
    for _ in range(iterations):
        compiled(*args)
    cp.cuda.get_current_stream().synchronize()
    us = (time.perf_counter() - t0) * 1e6 / iterations
    print(
        "[gdn_prefill] Latency: %.4f us "
        "(n=%d T=%d h=%d hv=%d k=%d v=%d)" % (us, n, seq_len, h, hv, k, v)
    )
    return us


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = _parsed_args
    np.random.seed(42)

    if args.export_only:
        export_gdn_prefill(
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

    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU required.")
    cp.random.seed(42)

    run_test_prefill(
        n=args.n, h=args.h, hv=args.hv, k=args.k, v=args.v,
        seq_len=args.seq_len,
        skip_ref_check=args.skip_ref_check,
        tolerance=args.tolerance,
        warmup=args.warmup,
        iterations=args.iterations,
        gpu_arch=args.gpu_arch,
        context_lengths_preset=args.context_lengths_preset,
    )


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CuTe DSL GDN prefill: AOT export and test (CuPy only, Ampere+)."
    )
    p.add_argument("--export_only", action="store_true")
    p.add_argument("--output_dir", type=str, default="./gdn_aot_artifacts")
    p.add_argument("--file_name", type=str, default="gdn_prefill")
    p.add_argument("--function_prefix", type=str, default="gdn_prefill")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--h", type=int, default=8)
    p.add_argument("--hv", type=int, default=8)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--v", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument(
        "--context_lengths_preset",
        type=str,
        default="full",
        choices=("full", "half", "staggered"),
        help="Prefill mask test: full=T for all rows; half=all rows use max(1,T//2) valid tokens; "
        "staggered=each row i uses max(1, T - i%%(T-1)) valid tokens (variable-length within batch).",
    )
    p.add_argument("--skip_ref_check", action="store_true")
    p.add_argument("--tolerance", type=float, default=0.2)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--gpu_arch", type=str, default="",
                   help="Target GPU arch for export (e.g. sm_87 for Orin). Empty = current GPU.")
    return p.parse_known_args(args=argv)[0]


if __name__ == "__main__":
    _parsed_args = _parse_args(_saved_argv)
    main()
