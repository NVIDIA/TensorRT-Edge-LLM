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
"""On-device oracle and micro-benchmark for the SM110 grouped W4A16 MoE GEMM.

Runs the ``nvfp4_a16_blackwell_moe_gemm`` FC1 (relu2_store) and FC2
(scatter_add) variants through the CuTe DSL JIT on a Blackwell SM110 board and
checks them against an fp32 NumPy reference built from the dequantized
NVFP4 weights.  Only ``numpy``, ``cupy`` and ``nvidia-cutlass-dsl`` are needed
(no torch), so it runs inside the board's CuTe DSL venv:

    python moe_gemm_oracle.py --tokens 128 --token_tile 128
    python moe_gemm_oracle.py --tokens 2048 --token_tile 128 --bench --iters 50

Default problem is the Nemotron 3.5 Lightning routed MoE: E=128, top_k=6,
H=2688, I=1856 (FC1 N padded to 1920), ReLU^2.  Routing and the permuted,
tile-padded activation buffer are built on the host exactly the way
``buildLayoutGpu`` + the gather kernel build them in the plugin.
"""

import argparse
import os
import sys
import time

import cupy as cp
import numpy as np

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvfp4_a16_blackwell_moe_gemm import (  # noqa: E402
    FUSION_RELU2_STORE, FUSION_SCATTER_ADD, make_launch)

_TILE_N = 128
_TILE_K = 64
_GROUP = 16
_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Weights: random NVFP4 checkpoint tensors, dequant reference, layout repack
# --------------------------------------------------------------------------- #
def e4m3_to_f32(codes_u8: np.ndarray) -> np.ndarray:
    """Decode E4M3 (fn) bytes to fp32 (no NaN codes expected)."""
    c = codes_u8.astype(np.uint32)
    sign = np.where(c & 0x80, -1.0, 1.0).astype(np.float32)
    exp = (c >> 3) & 0xF
    man = c & 0x7
    normal = exp > 0
    val = np.where(normal,
                   (1.0 + man / 8.0) * np.exp2(exp.astype(np.float32) - 7.0),
                   (man / 8.0) * np.exp2(-6.0)).astype(np.float32)
    return sign * val


def random_expert(n: int, k: int, rng: np.random.Generator):
    """Packed codes [n, k/2] u8, E4M3 scales [n, k/16] u8, fp32 ws2."""
    packed = rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)
    # E4M3 scales in [0.25, 2): exponent 5..7 (0x28..0x3F), realistic spread.
    scales = rng.integers(0x28, 0x40, size=(n, k // _GROUP), dtype=np.uint8)
    ws2 = np.float32(rng.uniform(0.004, 0.012))
    return packed, scales, ws2


def dequant(packed: np.ndarray, scales: np.ndarray,
            ws2: np.float32) -> np.ndarray:
    n, half = packed.shape
    nib = np.empty((n, half * 2), dtype=np.uint8)
    nib[:, 0::2] = packed & 0x0F
    nib[:, 1::2] = (packed >> 4) & 0x0F
    vals = _E2M1[nib & 0x7] * np.where(nib & 0x8, -1.0, 1.0).astype(np.float32)
    sc = e4m3_to_f32(scales)
    vals = vals.reshape(n, -1, _GROUP) * sc[:, :, None]
    return (vals.reshape(n, half * 2) * ws2).astype(np.float32)


def repack_blackwell(packed: np.ndarray, scales: np.ndarray):
    """BLACKWELL_MOE_N128_K64_V1 plane: [N_pad/128, K/64, 128, 32] + [.., 128, 4]."""
    n, half = packed.shape
    k = half * 2
    n_pad = ((n + _TILE_N - 1) // _TILE_N) * _TILE_N
    wp = np.zeros((n_pad, half), dtype=np.uint8)
    wp[:n] = packed
    ws = np.zeros((n_pad, k // _GROUP), dtype=np.uint8)
    ws[:n] = scales
    q = np.ascontiguousarray(
        wp.reshape(n_pad // _TILE_N, _TILE_N, k // _TILE_K,
                   _TILE_K // 2).transpose(0, 2, 1, 3))
    # TMA SWIZZLE_32B image: rows 4-7 of every 8 swap their 16-byte halves.
    t = q.reshape(n_pad // _TILE_N, k // _TILE_K, 16, 8, 2, 16).copy()
    t[:, :, :, 4:8, :, :] = t[:, :, :, 4:8, ::-1, :]
    q = np.ascontiguousarray(t.reshape(q.shape))
    s = np.ascontiguousarray(
        ws.reshape(n_pad // _TILE_N, _TILE_N, k // _TILE_K,
                   _TILE_K // _GROUP).transpose(0, 2, 1, 3))
    return q, s, n_pad


# --------------------------------------------------------------------------- #
# Routing -> permuted, tile-padded layout (mirrors buildLayoutGpu + gather)
# --------------------------------------------------------------------------- #
def build_routing(num_tokens, num_experts, top_k, rng, hot_expert=None):
    """Random distinct top-k experts per token; optional hot expert skew."""
    idx = np.empty((num_tokens, top_k), dtype=np.int32)
    for t in range(num_tokens):
        if hot_expert is not None and rng.uniform() < 0.5:
            others = rng.choice(
                [e for e in range(num_experts) if e != hot_expert],
                size=top_k - 1,
                replace=False)
            idx[t] = np.concatenate([[hot_expert], others])
        else:
            idx[t] = rng.choice(num_experts, size=top_k, replace=False)
    w = rng.uniform(0.05, 1.0, size=(num_tokens, top_k)).astype(np.float32)
    w = (w / w.sum(axis=1, keepdims=True) * 2.5).astype(
        np.float32)  # renorm * scale
    return idx, w


def build_layout(topk_idx, num_experts, tile, max_rows_padded):
    """tile_group_idx, permuted_idx (-1 pad), num_valid_tiles, per-expert offsets."""
    num_tokens, top_k = topk_idx.shape
    counts = np.bincount(topk_idx.reshape(-1), minlength=num_experts)
    padded = ((counts + tile - 1) // tile) * tile
    offsets = np.concatenate([[0], np.cumsum(padded)[:-1]])
    total_rows = int(padded.sum())
    assert total_rows <= max_rows_padded, (total_rows, max_rows_padded)
    num_tiles = total_rows // tile
    tile_group_idx = np.full(max_rows_padded // tile, -1, dtype=np.int32)
    t = 0
    for e in range(num_experts):
        for _ in range(padded[e] // tile):
            tile_group_idx[t] = e
            t += 1
    assert t == num_tiles
    permuted_idx = np.full(max_rows_padded, -1, dtype=np.int32)
    cursor = offsets.copy()
    for tok in range(num_tokens):
        for s in range(top_k):
            e = topk_idx[tok, s]
            permuted_idx[cursor[e]] = tok * top_k + s
            cursor[e] += 1
    return tile_group_idx, permuted_idx, np.array([num_tiles], dtype=np.int32)


# --------------------------------------------------------------------------- #
# Launch helpers
# --------------------------------------------------------------------------- #
def dptr(arr: cp.ndarray, dtype, align=16):
    return make_ptr(dtype,
                    arr.data.ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=align)


def compile_variant(io_dtype, token_tile, fusion, tensors, ints, stream):
    launch = make_launch(io_dtype, token_tile, fusion)
    args = list(tensors) + [cutlass.Int32(v) for v in ints] + [stream]
    t0 = time.time()
    compiled = cute.compile(launch, *args)
    print(f"  compiled {fusion} tn{token_tile} in {time.time() - t0:.1f}s")
    return compiled, args


def run(compiled, args):
    compiled(*args)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--token_tile", type=int, default=128)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=2688)
    ap.add_argument("--inter", type=int, default=1856)
    ap.add_argument("--io_dtype", choices=("fp16", "bf16"), default="fp16")
    ap.add_argument(
        "--hot_expert",
        type=int,
        default=None,
        help="route ~50%% of tokens through this expert (ragged tiles)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--skip_check", action="store_true")
    ap.add_argument("--pdl", type=int, choices=(0, 1), default=0,
                    help="launch with programmatic stream serialization "
                    "(the kernels always carry the griddepcontrol wait/trigger)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    E, K1, I, T, topk, tile = (args.experts, args.hidden, args.inter,
                               args.tokens, args.top_k, args.token_tile)
    io_dtype = cutlass.Float16 if args.io_dtype == "fp16" else cutlass.BFloat16
    np_io = np.float16 if args.io_dtype == "fp16" else None
    if np_io is None:
        raise SystemExit("bf16 host reference not implemented in this driver")
    n1_pad = ((I + _TILE_N - 1) // _TILE_N) * _TILE_N  # FC1 N (1920)

    print(
        f"E={E} topK={topk} H={K1} I={I} (FC1 N_pad={n1_pad}) T={T} tile={tile}"
    )
    # ---- weights
    fc1_q = np.empty((E, n1_pad // _TILE_N, K1 // _TILE_K, _TILE_N, 32),
                     np.uint8)
    fc1_s = np.empty((E, n1_pad // _TILE_N, K1 // _TILE_K, _TILE_N, 4),
                     np.uint8)
    fc2_q = np.empty((E, K1 // _TILE_N, I // _TILE_K, _TILE_N, 32), np.uint8)
    fc2_s = np.empty((E, K1 // _TILE_N, I // _TILE_K, _TILE_N, 4), np.uint8)
    fc1_g = np.empty((E, ), np.float32)
    fc2_g = np.empty((E, ), np.float32)
    w1 = []  # dequantized [I, H] per expert (reference; alpha excluded)
    w2 = []  # dequantized [H, I]
    for e in range(E):
        p, s, g = random_expert(I, K1, rng)
        fc1_q[e], fc1_s[e], _ = repack_blackwell(p, s)
        fc1_g[e] = g
        w1.append(dequant(p, s, np.float32(1.0)))
        p, s, g = random_expert(K1, I, rng)
        fc2_q[e], fc2_s[e], _ = repack_blackwell(p, s)
        fc2_g[e] = g
        w2.append(dequant(p, s, np.float32(1.0)))
    # ---- routing + permuted activations
    topk_idx, topk_w = build_routing(T, E, topk, rng, args.hot_expert)
    max_rows = ((T * topk + E * (tile - 1)) + tile - 1) // tile * tile
    tile_group_idx, permuted_idx, num_valid = build_layout(
        topk_idx, E, tile, max_rows)
    x = rng.standard_normal((T, K1)).astype(np_io)
    x_perm = np.zeros((max_rows, K1), dtype=np_io)
    valid = permuted_idx >= 0
    x_perm[valid] = x[permuted_idx[valid] // topk]
    print(
        f"  routed rows={T * topk}, padded rows={int(num_valid[0]) * tile}, "
        f"alloc rows={max_rows}, tiles={int(num_valid[0])}/{max_rows // tile}")

    # ---- device buffers
    d = {}
    d["fc1_q"] = cp.asarray(fc1_q)
    d["fc1_s"] = cp.asarray(fc1_s)
    d["fc1_g"] = cp.asarray(fc1_g)
    d["fc2_q"] = cp.asarray(fc2_q)
    d["fc2_s"] = cp.asarray(fc2_s)
    d["fc2_g"] = cp.asarray(fc2_g)
    d["x_perm"] = cp.asarray(x_perm)
    d["fc1_out"] = cp.zeros((max_rows, n1_pad), dtype=np_io)
    d["out"] = cp.zeros((T, K1), dtype=np_io)
    d["tile_group_idx"] = cp.asarray(tile_group_idx)
    d["permuted_idx"] = cp.asarray(permuted_idx)
    d["num_valid"] = cp.asarray(num_valid)
    d["topk_w"] = cp.asarray(topk_w)
    cp.cuda.Device().synchronize()
    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    sm_count = cp.cuda.Device().attributes["MultiProcessorCount"]

    common = dict(
        tile_group_idx=dptr(d["tile_group_idx"], cutlass.Int32, 4),
        num_valid=dptr(d["num_valid"], cutlass.Int32, 4),
        permuted_idx=dptr(d["permuted_idx"], cutlass.Int32, 4),
        topk_w=dptr(d["topk_w"], cutlass.Float32, 4),
    )
    fc1_tensors = (
        dptr(d["x_perm"], io_dtype),
        dptr(d["fc1_q"], cutlass.Float4E2M1FN),
        dptr(d["fc1_s"], cutlass.Float8E4M3FN),
        dptr(d["fc1_g"], cutlass.Float32, 4),
        dptr(d["fc1_out"], io_dtype),
        common["tile_group_idx"],
        common["num_valid"],
        common["permuted_idx"],
        common["topk_w"],
    )
    #        rows, act_ld, out_features, in_features, E, T, topk, max_active_clusters, enable_pdl
    fc1_ints = (max_rows, K1, n1_pad, K1, E, T, topk, sm_count, args.pdl)
    fc2_tensors = (
        dptr(d["fc1_out"], io_dtype),
        dptr(d["fc2_q"], cutlass.Float4E2M1FN),
        dptr(d["fc2_s"], cutlass.Float8E4M3FN),
        dptr(d["fc2_g"], cutlass.Float32, 4),
        dptr(d["out"], io_dtype),
        common["tile_group_idx"],
        common["num_valid"],
        common["permuted_idx"],
        common["topk_w"],
    )
    fc2_ints = (max_rows, n1_pad, K1, I, E, T, topk, sm_count, args.pdl)

    fc1, fc1_args = compile_variant(io_dtype, tile, FUSION_RELU2_STORE,
                                    fc1_tensors, fc1_ints, stream)
    fc2, fc2_args = compile_variant(io_dtype, tile, FUSION_SCATTER_ADD,
                                    fc2_tensors, fc2_ints, stream)

    run(fc1, fc1_args)
    run(fc2, fc2_args)
    cp.cuda.Device().synchronize()

    if not args.skip_check:
        # ---- reference
        fc1_out = cp.asnumpy(d["fc1_out"]).astype(np.float32)
        out = cp.asnumpy(d["out"]).astype(np.float32)
        ref1 = np.zeros((max_rows, n1_pad), dtype=np.float32)
        ref_out = np.zeros((T, K1), dtype=np.float32)
        rows_of = {}
        for r in np.nonzero(valid)[0]:
            e = topk_idx[permuted_idx[r] // topk, permuted_idx[r] % topk]
            rows_of.setdefault(int(e), []).append(int(r))
        for e, rows in rows_of.items():
            rows = np.array(rows)
            h = x_perm[rows].astype(np.float32)
            a1 = fc1_g[e] * (h @ w1[e].T)  # [rows, I]
            a1 = np.square(np.maximum(a1, 0.0)).astype(np.float16).astype(
                np.float32)
            ref1[rows, :I] = a1
            a2 = fc2_g[e] * (a1 @ w2[e].T)  # [rows, H]
            for r, contrib in zip(rows, a2):
                tok, slot = divmod(int(permuted_idx[r]), topk)
                ref_out[tok] += topk_w[tok, slot] * contrib
        # FC1 check (valid rows only; pad rows are relu2(0)=0)
        v1 = fc1_out[valid]
        r1 = ref1[valid]
        err1 = np.abs(v1 - r1)
        tol1 = 0.02 * np.abs(r1) + 0.05
        print(
            f"FC1: cos={cosine(v1, r1):.6f} max_abs_err={err1.max():.4g} "
            f"max_ref={np.abs(r1).max():.4g} viol={(err1 > tol1).mean():.2e}")
        pad_rows = fc1_out[~valid]
        print(
            f"FC1: pad rows max |value| = {np.abs(pad_rows).max() if pad_rows.size else 0:.3g}"
        )
        err2 = np.abs(out - ref_out)
        tol2 = 0.02 * np.abs(ref_out) + 0.05
        print(
            f"FC2: cos={cosine(out, ref_out):.6f} max_abs_err={err2.max():.4g} "
            f"max_ref={np.abs(ref_out).max():.4g} viol={(err2 > tol2).mean():.2e}"
        )
        ok = cosine(v1, r1) > 0.999 and cosine(out, ref_out) > 0.999
        print("ORACLE", "PASS" if ok else "FAIL")

    if args.bench:

        def timeit(fn, label, bytes_moved, flops):
            for _ in range(args.warmup):
                fn()
            cp.cuda.Device().synchronize()
            ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
            times = []
            for _ in range(args.iters):
                ev0.record()
                fn()
                ev1.record()
                ev1.synchronize()
                times.append(cp.cuda.get_elapsed_time(ev0, ev1) * 1e3)  # us
            t = float(np.median(times))
            print(
                f"BENCH {label}: median {t:8.1f} us  p90 {np.percentile(times, 90):8.1f} us"
                f"  {bytes_moved / t / 1e3:7.1f} GB/s  {flops / t / 1e6:6.1f} TFLOPS"
            )
            return t

        active = np.unique(topk_idx).size
        b1 = active * (n1_pad * K1 // 2 + n1_pad * K1 // 16)
        b2 = active * (K1 * I // 2 + K1 * I // 16)
        rows = T * topk
        f1 = 2.0 * rows * n1_pad * K1
        f2 = 2.0 * rows * K1 * I
        t1 = timeit(lambda: run(fc1, fc1_args), f"FC1 tn{tile} T={T}", b1, f1)
        d["out"].fill(0)
        t2 = timeit(lambda: run(fc2, fc2_args), f"FC2 tn{tile} T={T}", b2, f2)
        print(
            f"BENCH FC1+FC2 tn{tile} T={T}: {t1 + t2:8.1f} us "
            f"(weights {(b1 + b2) / 1e6:.1f} MB over {active} active experts)")


if __name__ == "__main__":
    main()
