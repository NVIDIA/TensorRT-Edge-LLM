# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""CuTe DSL CUDA-core W4A16 (INT4-weight FP16) decode GEMV: narrow-CTA mapping.

Computes ``C[M,N] = A[M,K] @ dequant(QW_frag, scales)^T`` for the decode regime
(small M), reading the SAME offline fragment-order uint32 weight buffer as the
INT4 FP16 GEMM kernel (``int4_fp16_gemm_ampere.py``; bN=128/bK=64, built by
``int4_reference.repack_b_for_tile``).  One weight copy therefore serves both
prefill (GEMM, tensor-core) and decode (this GEMV, CUDA-core) -- no repack, no
duplicate.  At small M the tensor-core MMA (m16n8k16) fills only a fraction of
its M rows, so a CUDA-core GEMV is the right compute path for decode.

Narrow-CTA mapping (16 output channels per CTA):
- CTA = (nb, cg, pblk): 8 Int128 columns ``[8cg, 8cg+8)`` of N-block ``nb``, one
  N-pair ``pblk`` -> **16 output channels**;  grid = (ceil(N/128)*4, 2) = N/16 CTAs.
- Lane l of a warp: column ``cp = l%8`` (n_base = 8cg+cp), k-block ``r = l//8``.
  One Int128 weight load per lane per K-tile; lanes 0-7 read 128 contiguous bytes
  (row r*2+pblk), lanes 8-15 the kbl=1 row, etc -- 4 fully-used 128 B lines/warp.
  Coalescing is judged per 128 B line, so this 8-column x 4-row tile reads at
  full efficiency; the narrow CTA is not a penalty.
- Warp w strides K-tiles by W  =>  per-channel K-split = 4*W (32 at W=8), so the
  fp16 accumulation chain is short (~K/64 per accumulator half) and **psum is
  fp16x2 only** -- ~2M registers, roughly M-independent: no FP32 flush, no
  register cliff, activation staging always on, no M gate.
- Reduction: convert psum to FP32, ``shfl.bfly`` xor 8 / xor 16 folds the 4
  same-channel lanes of a warp; warp 0 sums the W per-warp partials via smem
  ``[W, 8, M, 2]`` and writes fp16.  Single launch, single fixed config (W=8).
"""

from __future__ import annotations

import argparse
import sys
import time

# Reset argv before importing cutlass: the DSL evaluates argparse-like state at
# import and the kernel script owns its own CLI (mirrors int4_fp16_gemm_ampere.py).
_parsed_args = None
_saved_argv = None
if __name__ == "__main__":
    _saved_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]

import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float16, Float32, Int32, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import dsl_user_op
from cutedsl_utils import aot_placeholders

from common import (
    ceil_div,
    export_compiled_kernel,
    make_row_major_2d_placeholder,
    mark_row_major_2d,
    parse_comma_separated_ints,
)

# Shared INT4 fragment-word dequant (the same path the prefill GEMM uses; both
# consume the identical offline fragment weight buffer, so one dequant serves
# both and keeps them bit-identical).
from int4_dequant import _dequant_int4_word


# fp16x2 fma / horizontal-sum device helpers for the CUDA-core accumulation.
@dsl_user_op
def _fma2(acc, w, a, *, loc=None, ip=None):
    """``acc = fma.rn.f16x2(w, a, acc)`` on packed fp16x2 words (all Uint32)."""
    i32 = Int32.mlir_type
    r = llvm.inline_asm(
        i32,
        [Uint32(w).ir_value(loc=loc, ip=ip), Uint32(a).ir_value(loc=loc, ip=ip),
         Uint32(acc).ir_value(loc=loc, ip=ip)],
        "fma.rn.f16x2 $0, $1, $2, $3;", "=r,r,r,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Uint32(r)


@dsl_user_op
def _hsum2(acc, *, loc=None, ip=None):
    """Horizontal add of a packed fp16x2 word's two lanes -> Float32."""
    vec_t = ir.VectorType.get([2], Float16.mlir_type, loc=loc)
    v = llvm.bitcast(vec_t, Uint32(acc).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    lo = vector.extract(v, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    hi = vector.extract(v, dynamic_position=[], static_position=[1], loc=loc, ip=ip)
    return Float32(lo) + Float32(hi)


# One-tile building blocks for the UNROLL2 mainloop.  Module-level on purpose:
# the DSL forbids closures that capture dynamic values (or ``self``) inside a
# dynamic loop, so every operand is passed explicitly.  BK=64 is baked in.
def _u2_dequant_tile(kt, wr, G, r, s_lo, s_hi, mScales):
    """Load this tile's two scales and dequant its 4 c-words -> 4x(4 half2)."""
    g = (kt * Int32(64) + Int32(16) * r) // G
    slo = mScales[g, s_lo]
    shi = mScales[g, s_hi]
    dqc = []
    for c in range(4):
        wc = Uint32(wr[c])
        lo_lo, lo_hi = _dequant_int4_word(wc, slo)
        hi_lo, hi_hi = _dequant_int4_word(wc >> Uint32(8), shi)
        dqc.append((lo_lo, lo_hi, hi_lo, hi_hi))
    return dqc


def _u2_fma_tile(kt, dqc, r, mA128, psum, M):
    """Per-row: two Int128 act loads + 16 fp16x2 FMAs into psum."""
    ac = kt * Int32(8) + Int32(2) * r          # Int128 act col base (BK//8 == 8)
    for m in range(M):
        astage = cute.make_rmem_tensor(cute.make_layout(8), Uint32)
        astage128 = cute.recast_tensor(astage, cutlass.Int128)
        astage128[0] = mA128[m, ac]
        astage128[1] = mA128[m, ac + Int32(1)]
        for c in range(4):
            lo_lo, lo_hi, hi_lo, hi_hi = dqc[c]
            a01 = astage[c]
            a89 = astage[c + 4]
            psum[m, 0] = _fma2(psum[m, 0], lo_lo, a01)
            psum[m, 0] = _fma2(psum[m, 0], lo_hi, a89)
            psum[m, 1] = _fma2(psum[m, 1], hi_lo, a01)
            psum[m, 1] = _fma2(psum[m, 1], hi_hi, a89)


class Int4Fp16GemvAmpere:
    """Narrow-CTA (16-channel) CUDA-core GEMV over the INT4 fragment layout."""

    BN = 128   # channels per fragment N-block
    BK = 64    # K per tile
    KN = 8     # uint32 word-rows per (N-block, K-tile): 4 k-blocks x 2 N-pairs
    COLS = 8   # Int128 columns per CTA  -> 16 channels with one N-pair

    def __init__(self, *, m: int, group_size: int = 128, warp_groups: int = 8,
                 prefetch: bool = True, unroll2: bool = False,
                 u2_eager: bool = False, min_blocks: int = 0) -> None:
        if m <= 0:
            raise ValueError("m must be positive")
        if group_size % 16 != 0:
            raise ValueError("group_size must be a multiple of 16")
        if not 1 <= warp_groups <= 32:
            raise ValueError("warp_groups must be in [1, 32]")
        self.M = int(m)
        self.group_size = int(group_size)
        self.W = int(warp_groups)
        self.num_threads = 32 * self.W
        # Software-pipelined weight prefetch (see mainloop): prefetch tile kt+W
        # while the current tile computes.  Off by default (measured net-neutral
        # or negative -- the mainloop waits on weights, scales, and activations,
        # so pipelining only the weights relocates the stall rather than removing
        # it).  Kept for experiments.
        self.PREFETCH = bool(prefetch)
        # Unroll-by-2 of the tile loop (see mainloop); takes precedence over
        # PREFETCH (the pair body already double-buffers the weight loads).  It
        # issues both tiles' Int128 weight loads before either dequant chain,
        # widening the scheduling window on tall-K shapes.
        self.UNROLL2 = bool(unroll2)
        # Dequant-B placement inside the pair body: eager (both dequant sets
        # before the FMAs, larger scheduling window / more register pressure) vs
        # deferred (dequant B after tile A's FMAs, lighter live range).
        self.U2_EAGER = bool(u2_eager)
        # __launch_bounds__(256, MINB) equivalent: emits .minnctapersm so ptxas
        # schedules under an explicit register budget (85 at MINB=3).  This is
        # the ingredient -maxrregcount CANNOT provide: ptxas silently ignores
        # the flag for functions carrying launch bounds, and the DSL always
        # emits .reqntid.  0 = off (ptxas picks).
        self.MINB = int(min_blocks)

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mQW: cute.Tensor,
        mScales: cute.Tensor,
        mOut: cute.Tensor,          # [M, N] fp16, written directly
        stream: cuda.CUstream,
    ) -> None:
        num_n_blocks = cute.ceil_div(mOut.shape[1], self.BN)
        # grid.x packs (N-block, column-group): bx = nb*4 + cg; grid.y = N-pair.
        if cutlass.const_expr(self.MINB > 0):
            # min_blocks_per_mp requires smem to be set; passing it explicitly
            # overrides the inferred size, so it must be the allocator's real
            # footprint (sRed [W, 8, M, 2] Float32).
            smem_bytes = self.W * 8 * self.M * 2 * 4
            self.kernel(mA, mQW, mScales, mOut).launch(
                grid=(cute.size(num_n_blocks) * 4, 2, 1),
                block=[self.num_threads, 1, 1],
                stream=stream,
                min_blocks_per_mp=self.MINB,
                smem=smem_bytes,
            )
        else:
            self.kernel(mA, mQW, mScales, mOut).launch(
                grid=(cute.size(num_n_blocks) * 4, 2, 1),
                block=[self.num_threads, 1, 1],
                stream=stream,
            )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mQW: cute.Tensor,
        mScales: cute.Tensor,
        mOut: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bx, pblk, _ = cute.arch.block_idx()
        nb = bx // Int32(4)
        cg = bx % Int32(4)

        M = self.M
        G = Int32(self.group_size)
        N = mOut.shape[1]
        K = mA.shape[1]
        num_k_tiles = K // Int32(self.BK)

        grp = tidx // Int32(32)        # warp: strides K-tiles by W
        lane = tidx % Int32(32)
        cp = lane % Int32(8)           # column within the CTA's 8-column group
        r = lane // Int32(8)           # k-block [0, 4) within each K-tile

        n_base = Int32(8) * cg + cp    # Int128 column in [0, 32)
        # This lane's two output channels (lo/hi nibbles of its words).
        c_lo = nb * Int32(self.BN) + n_base + Int32(64) * pblk
        c_hi = c_lo + Int32(32)
        n_last = N - Int32(1)
        s_lo = cutlass.min(c_lo, n_last)   # clamped scale index (OOB weights are 0)
        s_hi = cutlass.min(c_hi, n_last)

        # fp16x2 psum, accumulated over the lane's whole K share with NO FP32
        # flush: per-half chain ~= K/64 (matches awq's; accuracy parity measured).
        psum = cute.make_rmem_tensor(cute.make_layout((M, 2)), Uint32)
        psum.fill(0)

        mQW128 = cute.recast_tensor(mQW, cutlass.Int128)   # [rows, 32]
        wreg = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
        wreg128 = cute.recast_tensor(wreg, cutlass.Int128)
        mA128 = cute.recast_tensor(mA, cutlass.Int128)     # [M, K/8]

        if cutlass.const_expr(self.UNROLL2):
            # Unroll the tile loop by 2: both tiles' Int128 weight loads issue
            # before either dequant chain -- 2x loads in flight, 2x the ptxas
            # scheduling window, half the loop overhead, at the cost of ~20
            # registers (one fewer resident block).  Helps latency-exposed tall-K
            # shapes.  The odd tile (per-warp tile count = ceil((nkt-grp)/W)) is
            # handled by a single-tile tail.
            wregB = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            wregB128 = cute.recast_tensor(wregB, cutlass.Int128)

            stop_pair = cutlass.max(num_k_tiles - Int32(self.W), Int32(0))
            for kt in cutlass.range(grp, stop_pair, 2 * self.W):
                rowA = (nb * num_k_tiles + kt) * Int32(self.KN) + Int32(2) * r + pblk
                wreg128[0] = mQW128[rowA, n_base]                       # tile kt
                wregB128[0] = mQW128[rowA + Int32(self.W * self.KN), n_base]  # tile kt+W
                # Dequant-B placement: eager keeps both tiles' 16-reg dequant sets
                # live through tile A's FMAs (bigger scheduling window, more
                # pressure); deferred sinks dequant B after tile A's FMAs (lighter
                # live range).  Which wins is M-dependent; either way the two
                # weight loads above stay hoisted, which is what buys the win.
                if cutlass.const_expr(self.U2_EAGER):
                    dqcA = _u2_dequant_tile(kt, wreg, G, r, s_lo, s_hi, mScales)
                    dqcB = _u2_dequant_tile(kt + Int32(self.W), wregB, G, r, s_lo, s_hi, mScales)
                    _u2_fma_tile(kt, dqcA, r, mA128, psum, M)
                    _u2_fma_tile(kt + Int32(self.W), dqcB, r, mA128, psum, M)
                else:
                    dqcA = _u2_dequant_tile(kt, wreg, G, r, s_lo, s_hi, mScales)
                    _u2_fma_tile(kt, dqcA, r, mA128, psum, M)
                    dqcB = _u2_dequant_tile(kt + Int32(self.W), wregB, G, r, s_lo, s_hi, mScales)
                    _u2_fma_tile(kt + Int32(self.W), dqcB, r, mA128, psum, M)
            if grp < num_k_tiles:
                ntl = (num_k_tiles - grp + Int32(self.W) - 1) // Int32(self.W)
                if ntl % Int32(2) == Int32(1):                           # odd tile count
                    kt_t = grp + (ntl - Int32(1)) * Int32(self.W)
                    row_t = (nb * num_k_tiles + kt_t) * Int32(self.KN) + Int32(2) * r + pblk
                    wreg128[0] = mQW128[row_t, n_base]
                    dqc_t = _u2_dequant_tile(kt_t, wreg, G, r, s_lo, s_hi, mScales)
                    _u2_fma_tile(kt_t, dqc_t, r, mA128, psum, M)
        else:
            # Software-pipelined weight load: the dynamic tile loop otherwise
            # serializes each tile's (DRAM-latency) weight load behind the previous
            # tile's compute.  Prefetch tile kt+W into the wreg buffer while the
            # current tile computes from loop-carried SSA copies (w0..w3); +4
            # registers, M-independent.  Off by default (see __init__).
            if cutlass.const_expr(self.PREFETCH):
                row0 = (nb * num_k_tiles + grp) * Int32(self.KN) + Int32(2) * r + pblk
                if grp < num_k_tiles:                      # warp has >= 1 tile
                    wreg128[0] = mQW128[row0, n_base]
            w0 = Uint32(wreg[0])
            w1 = Uint32(wreg[1])
            w2 = Uint32(wreg[2])
            w3 = Uint32(wreg[3])
            # NOTE: scale prefetch on top of the weight prefetch was tried and
            # dropped -- it only relocates the stall to the scale load, and the
            # extra carried values + clamp math cost more than the removed wait.
            for kt in cutlass.range(grp, num_k_tiles, self.W):
                if cutlass.const_expr(self.PREFETCH):
                    kt_next = kt + Int32(self.W)
                    if kt_next < num_k_tiles:              # prefetch next tile's weights
                        row_n = (nb * num_k_tiles + kt_next) * Int32(self.KN) + Int32(2) * r + pblk
                        wreg128[0] = mQW128[row_n, n_base]
                    w = [w0, w1, w2, w3]
                else:
                    row = (nb * num_k_tiles + kt) * Int32(self.KN) + Int32(2) * r + pblk
                    wreg128[0] = mQW128[row, n_base]
                    w = [Uint32(wreg[c]) for c in range(4)]
                g = (kt * Int32(self.BK) + Int32(16) * r) // G
                slo = mScales[g, s_lo]
                shi = mScales[g, s_hi]
                ac = kt * Int32(self.BK // 8) + Int32(2) * r   # Int128 act col base
                # Dequant the 4 words once (M-independent registers), reused across
                # rows.
                # NOTE: inlining the dequant per-row at M>=5 to free its 16
                # registers was tried and dropped -- ptxas kept the same block
                # limit regardless, so the ~6x dequant recompute was pure loss.
                dqc = []
                for c in cutlass.range_constexpr(4):
                    lo_lo, lo_hi = _dequant_int4_word(w[c], slo)
                    hi_lo, hi_hi = _dequant_int4_word(w[c] >> Uint32(8), shi)
                    dqc.append((lo_lo, lo_hi, hi_lo, hi_hi))
                # NOTE: explicitly batching all rows' act loads before the FMAs at
                # small M was tried and dropped -- no effect (the small-M stall is
                # split across weight+scale load latency that the small 32-FMA tile
                # body cannot cover).  Fresh per-m staging tensors remain (ptxas
                # pipelines them).
                for m in cutlass.range_constexpr(M):
                    astage = cute.make_rmem_tensor(cute.make_layout(8), Uint32)
                    astage128 = cute.recast_tensor(astage, cutlass.Int128)
                    astage128[0] = mA128[m, ac]            # this row's 16 acts: 2 loads
                    astage128[1] = mA128[m, ac + Int32(1)]
                    for c in cutlass.range_constexpr(4):
                        lo_lo, lo_hi, hi_lo, hi_hi = dqc[c]
                        a01 = astage[c]
                        a89 = astage[c + 4]
                        psum[m, 0] = _fma2(psum[m, 0], lo_lo, a01)
                        psum[m, 0] = _fma2(psum[m, 0], lo_hi, a89)
                        psum[m, 1] = _fma2(psum[m, 1], hi_lo, a01)
                        psum[m, 1] = _fma2(psum[m, 1], hi_hi, a89)
                if cutlass.const_expr(self.PREFETCH):
                    # Hand the prefetched words to the next iteration (loop-carried
                    # SSA).
                    w0 = Uint32(wreg[0])
                    w1 = Uint32(wreg[1])
                    w2 = Uint32(wreg[2])
                    w3 = Uint32(wreg[3])

        # In-warp reduce: fold the 4 same-channel lanes {l, l^8, l^16, l^24}
        # (same cp, all four k-block slots) in FP32; every lane of the quad ends
        # with the sum (bfly is symmetric), lanes r==0 carry it onward.
        accf = cute.make_rmem_tensor(cute.make_layout((M, 2)), Float32)
        for ch in cutlass.range_constexpr(2):
            for m in cutlass.range_constexpr(M):
                v = _hsum2(psum[m, ch])
                v = v + cute.arch.shuffle_sync_bfly(v, 8)
                v = v + cute.arch.shuffle_sync_bfly(v, 16)
                accf[m, ch] = v

        # Cross-warp reduce via smem [W, 8, M, 2]; warp 0's lanes 0..7 sum the W
        # partials for their column and write fp16.  One writer per channel.
        sRed = utils.SmemAllocator().allocate_tensor(
            Float32, cute.make_layout((self.W, 8, M, 2)), 16
        )
        if r == Int32(0):
            for ch in cutlass.range_constexpr(2):
                for m in cutlass.range_constexpr(M):
                    sRed[grp, cp, m, ch] = accf[m, ch]
        cute.arch.sync_threads()
        if grp == Int32(0) and r == Int32(0):
            for ch in cutlass.range_constexpr(2):
                cc = c_lo if ch == 0 else c_hi
                if cc < N:
                    for m in cutlass.range_constexpr(M):
                        acc = sRed[0, cp, m, ch]
                        for gg in cutlass.range_constexpr(1, self.W):
                            acc = acc + sRed[gg, cp, m, ch]
                        mOut[m, cc] = acc.to(Float16)


# ---------------------------------------------------------------------------
# Standalone test + AOT export harness (CuPy-only export path; the reference
# check additionally uses Torch via int4_reference).  ABI per compiled variant:
#     (mA, mQW, mScales, mOut, stream)
# mA: [M,K] fp16; mQW: fragment-order uint32 [rows,128] (shared with the prefill
# GEMM); mScales: [ceil(K/G),N] fp16; mOut: [M,N] fp16 written directly.  Single
# launch, no workspace/locks.  M is baked (one exported function per M in [1, 8]).
# ---------------------------------------------------------------------------
_BN = 128
_BK = 64


def _gemv_defaults(m: int):
    """Per-M tuning defaults ``(unroll2, u2_eager, min_blocks)`` -- arch-independent.

    ONE table, a function of M only (no compute-capability key), so the baked AOT
    config for a given M is identical on every arch and the build never depends on
    the host device.  The tuning targets the tall-K latency exposure that grows
    with M: M1 base, M2-4 unroll2-deferred + minb3, M5 unroll2-eager, M>=6 minb3
    alone (M1 stays base -- the unroll/budget knobs only start paying off once the
    per-CTA body is large enough to hide their extra register pressure).
    ``prefetch`` stays off everywhere.
    """
    if m == 1:
        return False, False, 0
    if m <= 4:
        return True, False, 3
    if m == 5:
        return True, True, 0
    return False, False, 3


def run(
    mnk: "tuple[int, int, int]",
    group_size: int = 128,
    warp_groups: int = 8,
    prefetch: bool = False,
    unroll2: "bool | None" = None,
    u2_eager: "bool | None" = None,
    min_blocks: "int | None" = None,
    warmup_iterations: int = 2,
    iterations: int = 100,
    skip_ref_check: bool = False,
    export_only: bool = False,
    output_dir: str = "./int4_fp16_gemv_aot_artifacts",
    file_name: str = "int4_fp16_gemv_ampere",
    function_prefix: str = "int4_fp16_gemv_ampere",
    gpu_arch: str = "",
):
    """Run or export the narrow-CTA W4A16 decode GEMV (M<=8)."""
    M, N, K = mnk
    _tag = f"[{file_name}]"

    if not 1 <= M <= 8:
        raise ValueError(f"decode GEMV is M in [1, 8] (got M={M})")
    if K % 64 != 0:
        raise ValueError(f"K must be a multiple of 64 (got {K})")
    if group_size % 16 != 0:
        raise ValueError(f"group_size must be a multiple of 16 (got {group_size})")
    if not export_only and cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU is required.")

    # Arch-independent per-M tuning defaults (no device-SM detection), so the AOT
    # build is identical on every arch.  Explicit args override for experiments.
    d_unroll2, d_eager, d_minb = _gemv_defaults(M)
    unroll2 = d_unroll2 if unroll2 is None else unroll2
    u2_eager = d_eager if u2_eager is None else u2_eager
    min_blocks = d_minb if min_blocks is None else min_blocks

    print(f"{_tag} INT4 W4A16 FP16 GEMV: M={M}, N={N}, K={K}, group_size={group_size}")
    print(f"{_tag} W={warp_groups}, prefetch={prefetch}, unroll2={unroll2}, "
          f"u2_eager={u2_eager}, min_blocks={min_blocks}")

    kernel = Int4Fp16GemvAmpere(
        m=M, group_size=group_size, warp_groups=warp_groups,
        prefetch=prefetch, unroll2=unroll2, u2_eager=u2_eager, min_blocks=min_blocks,
    )
    current_stream = (
        aot_placeholders.make_stream()
        if export_only
        else cuda.CUstream(cp.cuda.get_current_stream().ptr)
    )

    if export_only:
        mA = make_row_major_2d_placeholder(cutlass.Float16)
        mQW = make_row_major_2d_placeholder(cutlass.Uint32)
        mScales = make_row_major_2d_placeholder(cutlass.Float16)
        mOut = make_row_major_2d_placeholder(cutlass.Float16)

        compile_opts = aot_placeholders.compile_options(
            ("--gpu-arch " + gpu_arch) if gpu_arch else ""
        )
        print(f"{_tag} Compiling kernel (gpu_arch={gpu_arch or 'default'})...")
        t0 = time.time()
        compiled = cute.compile(
            kernel, mA, mQW, mScales, mOut, current_stream,
            options=compile_opts,
        )
        print(f"{_tag} Compilation time: {time.time() - t0:.4f}s")
        export_compiled_kernel(
            compiled,
            output_dir=output_dir,
            file_name=file_name,
            function_prefix=function_prefix,
            tag=_tag,
        )
        return

    # ---- Reference correctness path (uses Torch via int4_reference) ----
    import torch  # noqa: PLC0415 -- only needed for the reference check

    from int4_reference import (
        int4_fp16_reference,
        make_int4_inputs,
        repack_b_for_tile,
    )

    act, qweight, scales = make_int4_inputs(M, N, K, group_size=group_size, device="cuda")
    ref = int4_fp16_reference(act, qweight, scales, group_size=group_size)
    out_t = torch.zeros((M, N), device="cuda", dtype=torch.float16)
    rep = repack_b_for_tile(qweight, N, K, _BN, _BK).to("cuda")

    a_cp = cp.from_dlpack(act.contiguous())
    scales_cp = cp.from_dlpack(scales.contiguous())
    out_cp = cp.from_dlpack(out_t)
    qw_cp = cp.from_dlpack(rep.contiguous()).view(cp.uint32)

    mA = mark_row_major_2d(a_cp)
    mQW = mark_row_major_2d(qw_cp)
    mScales = mark_row_major_2d(scales_cp)
    mOut = mark_row_major_2d(out_cp)

    print(f"{_tag} Compiling kernel...")
    t0 = time.time()
    compiled = cute.compile(kernel, mA, mQW, mScales, mOut, current_stream)
    print(f"{_tag} Compilation time: {time.time() - t0:.4f}s")

    if not skip_ref_check:
        compiled(mA, mQW, mScales, mOut, current_stream)
        torch.cuda.synchronize()
        diff = (out_t.to(torch.float32) - ref.to(torch.float32)).abs()
        max_abs = diff.max().item()
        denom = max(ref.abs().max().item(), 1e-6)
        rel = max_abs / denom
        # fp16x2 psum with a ~K/64 chain (awq-parity accuracy); a few percent
        # relative error vs the FP32-accumulate reference is expected.
        if rel > 0.05:
            raise ValueError(
                f"{_tag} Verification FAILED! max_abs_diff={max_abs:.6f} "
                f"rel={rel:.4f} (denom={denom:.4f})"
            )
        print(f"{_tag} Verification PASSED (max_abs_diff={max_abs:.6f}, rel={rel:.4f})")

    # ---- Benchmark ----
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(max(warmup_iterations, 0)):
        compiled(mA, mQW, mScales, mOut, current_stream)
    torch.cuda.synchronize()
    start.record()
    for _ in range(max(iterations, 1)):
        compiled(mA, mQW, mScales, mOut, current_stream)
    end.record()
    torch.cuda.synchronize()
    avg_time_us = start.elapsed_time(end) / max(iterations, 1) * 1e3
    print(f"{_tag} Avg time: {avg_time_us:.2f} us")
    return avg_time_us


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CuTe DSL INT4 (W4A16) FP16 decode GEMV: AOT export and test."
    )
    p.add_argument(
        "--mnk", type=parse_comma_separated_ints, default=(1, 512, 1024),
        help="M,N,K dimensions (M in [1,8]; default: 1,512,1024)",
    )
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument(
        "--warp_groups", type=int, default=8,
        help="Warp groups W (block = 32*W threads; baked; default 8).",
    )
    p.add_argument(
        "--prefetch", action="store_true",
        help="Software-pipelined weight prefetch (default off everywhere).",
    )
    # Tuning knobs default to the arch-independent per-M table (_gemv_defaults);
    # pass these only to override for experiments (--no-unroll2 forces off).
    p.add_argument(
        "--unroll2", action=argparse.BooleanOptionalAction, default=None,
        help="Override unroll-by-2 tile loop (default: per-M table).",
    )
    p.add_argument(
        "--u2_eager", action=argparse.BooleanOptionalAction, default=None,
        help="Override UNROLL2 dequant-B placement eager/deferred (default: per-M table).",
    )
    p.add_argument(
        "--min_blocks", type=int, default=None,
        help="Override __launch_bounds__ min_blocks_per_mp (default: per-M table).",
    )
    p.add_argument("--warmup_iterations", type=int, default=2)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--skip_ref_check", action="store_true")
    p.add_argument("--export_only", action="store_true")
    p.add_argument("--output_dir", type=str, default="./int4_fp16_gemv_aot_artifacts")
    p.add_argument("--file_name", type=str, default="int4_fp16_gemv_ampere")
    p.add_argument("--function_prefix", type=str, default="int4_fp16_gemv_ampere")
    p.add_argument(
        "--gpu_arch", type=str, default="",
        help="Target GPU arch for export (e.g. sm_87). Empty = current GPU.",
    )
    return p.parse_known_args(args=argv)[0]


def main():
    args = _parsed_args
    run(
        mnk=args.mnk,
        group_size=args.group_size,
        warp_groups=args.warp_groups,
        prefetch=args.prefetch,
        unroll2=args.unroll2,
        u2_eager=args.u2_eager,
        min_blocks=args.min_blocks,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        skip_ref_check=args.skip_ref_check,
        export_only=args.export_only,
        output_dir=args.output_dir,
        file_name=args.file_name,
        function_prefix=args.function_prefix,
        gpu_arch=args.gpu_arch,
    )


if __name__ == "__main__":
    _parsed_args = _parse_args(_saved_argv)
    main()
    print("PASS")
