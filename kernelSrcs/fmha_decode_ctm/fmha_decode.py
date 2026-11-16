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
"""Paged GQA decode attention for Blackwell SM100 / SM110 (XQA decode semantics) in
CTM (`cutlass.experimental.primitives` on CuTe DSL): ONE module holding the two
decode kernel bodies, `build(D, kv_dtype, spec_dec)` and one host `run()`.  The
single-token body (`spec_dec=False`, q_len == 1, q/out `[B, HQ, D]`) and the
spec-decode tree-attention body (`spec_dec=True`, q/out `[B, S, HQ, D]` with a
`[B, S, MW]` int32 tree mask) share the imports, the SM100 arch probe, the warp
reductions, the pointer helpers, the CTA budget, the ceil-div magic table and the
fixed page geometry; everything that differs (MMA tiles, stage depths, TMEM
layout, barrier ids, split plan and join constants) lives inside `_build_single`
/ `_build_tree`, so the two bodies never see each other's names.  The head dim
(32 / 64 / 128 / 256 / 512) and the KV pool element type ("fp16" or "e4m3";
e4m3 tree needs D >= 64) are the only baked axes: ONE `cute.compile` per
(D, kv_dtype, spec_dec) serves every B / HQ / HKV / GQA ratio / MAXP / pool
size / window / sink flag / split plan.  The KV pool is one flat array
`[2*NPAGES, 128, HKV, D]`, the page table `[B, 2, MAXP]` (K row 0, V row 1
already offset by NPAGES), q/out fp16, fp32 accumulate.  Launch contracts (the
AOT `export_to_c` wrapper is `module` + these, in this order):
single-token, 24 args: pQ, pKV, pPT, pSL, pSNK, pO, pWS, pWH, pSEM, B, HKV,
NP2, MAXP, gqa, splits, csplit, window, has_sink, hk_shift, qk_scale, v_scale,
stream, pSKT, skt_len (9 x void*, 10 x int32, 2 x float, cudaStream_t, then the
skip-softmax scale table: int32* of flattened (seqlen_bin, S) pairs and its
element count, 0 = dense); tree, 29 args: pQ,
pKV, pPT, pSL, pSNK, pO, pWS, pWH, pSEM, pTM, B, HKV, NP2, MAXP, gqa, splits,
W, CQ, SNK, S, MW, RB, DSH, RBM, GQM, SPLM, qk_scale, v_scale, stream (10 x
void*, 16 x int32, 2 x float, cudaStream_t).  `pWS` / `pWH` are the same
split-K workspace viewed as fp32 / fp16; `pSEM` the join counters; RBM / GQM /
SPLM are ceil(2^16 / RB), ceil(2^16 / gqa), ceil(2^16 / splits).
"""

import math
from functools import partial

# torch backs only the host-side `run()` helpers; the AOT export (`__main__`,
# --export_only) traces storage-free placeholders and runs where torch is absent.
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda_drv
import cutlass.experimental.cuda as ccuda
from cutlass.cute.runtime import make_ptr
from cutlass.cute.typing import Float32, Float16, Int32
from cutlass.experimental import primitives as prims

e4m3 = cutlass.Float8E4M3FN     # e4m3 KV pool / staged Q / staged P element type
e4m3_max = 448.0

# Constants shared by both bodies
D_DEFAULT = 128
KV_DTYPES = ("fp16", "e4m3")
KV_DTYPE_DEFAULT = "fp16"
tokens_per_page = 128         # tokens per page (fixed)
tokens_per_k_stage = 128         # tokens per K smem stage (single: M of BMM1; tree: 2 S tiles)
tokens_per_v_stage = 128         # tokens per V smem stage (= a whole page); equal to
                   # tokens_per_k_stage so K and V share one TMA descriptor
warp_threads = 32
warpgroup_threads = 128
page_chunk = warp_threads     # page ids held one-per-lane; reloaded every 32 pages
min_pages_per_cta = 4         # sequences at or under this never split

log2_e = math.log2(math.e)

exp2 = partial(cute.math.exp2, fastmath=True)
ffma2 = partial(prims.fma_packed_f32x2, ftz=False, rnd="rn")

# Split-plan constants the HOST plan shares with a kernel body.  Each builder
# aliases them back to the body's own name; the values are the source of truth
# for both sides.
single_mma_n = 8              # grouped-head MMA tile (N of BMM1); covers GQA 1..8
single_mma_n_wide = 16        # e4m3 ratio-16 head tile: ONE n=16 MMA chain
single_max_join_splits = 64   # host cap on the split plan
single_narrow_join_splits = 32   # e4m3: ...and on engines whose capacity is <= 64 pages
single_wide_engine_pages = 64    # e4m3: page-table capacity above which the wide cap pays
single_sem_words = 128        # e4m3 rendezvous: group stride in int32 words (512 B)
tree_narrow_splits = 8        # deals wider than this need `narrow_depth` pages per CTA
tree_clu_ext2 = 16            # non-portable cluster extent (SM100); wide D = 256 split cap
tree_sem_stride = 32          # one split-K arrival counter per 128 B line


def _target_has_f32_redux():
    """`redux.sync.max.f32` is accepted by ptxas for sm_100a and sm_103a only (rejected on sm_110a / sm_120a).
    The AOT build passes the target as `--target_sm <sm>` (build_cutedsl.py `wants_target_sm`; CUTE_DSL_ARCH is unset
    on cross builds); a JIT run falls back to the local device."""
    import sys
    if "--target_sm" in sys.argv:
        return int(sys.argv[sys.argv.index("--target_sm") + 1]) in (100, 103)
    return torch is not None and torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))


_USE_REDUX = _target_has_f32_redux()


def warp_fmax(x):
    # Compile-time arch flag: sm_100 / sm_103 collapse the five-step shfl.bfly+fmax
    # ladder to ONE hardware reduction; the other targets keep the ladder.
    if _USE_REDUX:
        return cute.arch.warp_redux_sync(x, kind="fmax", nan=True)
    for off in (16, 8, 4, 2, 1):
        peer = prims.shfl_sync(
            thread_mask=0xFFFFFFFF, val=x, offset=off, mask_and_clamp=0x1F,
            kind=prims.Shfl.BFLY,
        )
        x = cute.arch.fmax(x, peer)
    return x


def warp_fold16(vals, lane, is_max):
    # Halving-butterfly warp reduction of a 16-wide per-head array: each
    # exchange sends the half the PEER keeps, so the array shrinks
    # 16 -> 8 -> 4 -> 2 -> 1 in four rounds (16 shfl, not 80) and every lane
    # returns the warp fmax/sum of head h = (lane >> 1) & 15.  This helper is
    # NOT AST-preprocessed (plain range / no runtime ifs, like warp_fmax); the
    # keep/send pick is branchless FFMA on the lane bit, exact for m in {0, 1}.
    hb = cutlass.Array(Float32, 16)
    for n in range(16):
        hb[n] = vals[n]
    for rd in range(4):
        m = ((lane >> (4 - rd)) & 1).to(Float32)
        for i in range(8 >> rd):
            a = hb[i]
            b = hb[i + (8 >> rd)]
            snd = b + m * (a - b)
            kp = a + m * (b - a)
            rcv = prims.shfl_sync(
                thread_mask=0xFFFFFFFF, val=snd, offset=(16 >> rd),
                mask_and_clamp=0x1F, kind=prims.Shfl.BFLY,
                return_value_and_is_valid=False,
            )
            if is_max:
                hb[i] = cute.arch.fmax(kp, rcv)
            else:
                hb[i] = kp + rcv
    last = prims.shfl_sync(
        thread_mask=0xFFFFFFFF, val=hb[0], offset=1, mask_and_clamp=0x1F,
        kind=prims.Shfl.BFLY, return_value_and_is_valid=False,
    )
    if is_max:
        return cute.arch.fmax(hb[0], last)
    return hb[0] + last


def _magic(d):
    """ceil(2^16 / d): IMAD + SHF instead of a ~40-instruction signed IDIV."""
    return ((1 << 16) + d - 1) // d


# ceil(2^16 / d) for every divisor the launch path needs a magic for (row
# blocks, GQA ratio, split count).  A table lookup keeps the launch path free of
# Python integer division, which the small CI rows measure directly.
_MAGIC = [0] + [_magic(d) for d in range(1, 1025)]


# Single-token body
def _mc(D):
    """e4m3 cluster-mailbox width per head dim (host and kernel key on the SAME
    value): a slot is `mma_n * D` fp16, so the 8-slot D=128 mailbox shrinks to
    4 slots at D=256 (unchanged 16 KiB) and to 0 at D=512, where the smem is
    worth more as K/V stages and every multi-split row joins globally.
    Ratio-16 rows never mailbox (the slot would double); both sides also key
    the cluster launch on `gqa <= mma_n`."""
    return 8 if D <= 128 else (4 if D == 256 else 1)


def _build_single(D=D_DEFAULT, kv_dtype="fp16"):
    """Single-token paged GQA decode: one launcher per (head dim, KV dtype); nothing else is baked.

    Both dtypes share one kernel body; the trace-time differences: fp16 runs kind::f16 MMA on 128 x 64 K/V tiles
    and serves GQA ratio 9..16 as two adjacent 8-head CTAs per KV head (`hk_shift`); e4m3 runs kind::f8f6f4 MMA
    on 128 x 128 tiles with Q and P quantised to e4m3 on device and serves ratio 9..16 as one n = 16 MMA chain
    (`wide`).  The split-K join is a GPU-scope arrival counter (fp16) or a CTA-cluster mailbox / device-memory
    generation-word rendezvous (e4m3, see `_mc`).
    """
    global _USE_REDUX
    _USE_REDUX = _target_has_f32_redux()
    # Single-token body constants
    mma_n = single_mma_n             # grouped-head MMA tile (N of BMM1); covers GQA 1..8
    mma_n_wide = single_mma_n_wide   # e4m3 ratio-16 head tile: ONE n=16 MMA chain
    p_lanes = 16       # e4m3 N of BMM2: an MN-major fp8 B operand needs >= 16 B per
                       # token row, so the 8-head tile is padded to 16 lanes

    # The GQA ratio is runtime, so the softmax path always walks the full mma_n
    # tile and the Q staging zero-fills the rows above the ratio: a zero query row
    # scores 0 against every key, which is finite, and the epilogue simply never
    # stores it.
    softmax_heads = mma_n

    # Pipeline depths are the one thing that MUST stay compile-time (they size smem
    # stage buffers).  They are cut for the worst case; a CTA with fewer live pages
    # than stages simply never wraps its pipeline.  Per-dtype values live in
    # build(); these are the fp16 defaults the D-dependent trims start from.
    k_stages = 7       # fp16 K smem stages (16 KiB each, 2 per page: the d-halves)
    v_stages = 3       # fp16 V smem stages (32 KiB each: a whole page)
    p_stages = 4       # fp16 SMEM P buffers (D = 512 keeps 3)
    max_join_splits = single_max_join_splits          # host cap on the split plan
    narrow_join_splits = single_narrow_join_splits    # e4m3: ...and on engines whose capacity is <= 64 pages
    wide_engine_pages = single_wide_engine_pages      # e4m3: page-table capacity above which the wide cap pays
    # e4m3 rendezvous flag line: one int32 word per split, per (b, kv-head) group.
    flag_words = 64               # >= max_join_splits: one rendezvous flag word per split
    gen_word = flag_words         # per-group launch generation (device-side rendezvous token)
    sem_words = single_sem_words  # group stride in int32 words (512 B)
    sreg = max_join_splits // warp_threads   # (m, l) registers per fold lane: 32 splits each
    max_headroom = 2.0            # e4m3: log2 units the running max jumps past the tile that raised it

    # barrier ids
    softmax_sync_barrier = 1                           # softmax warpgroup internal sync
    join_barrier = 3                           # CTA-wide release into the split-K fold
    kv_ready_barrier = 6                       # fp16: warp 0 -> warp 3: K/V mbarriers are built

    # Skip-softmax page-vote ring (softmax warpgroup -> V producer / BMM2).  The
    # softmax publishes page s's vote before its P and runs at most pstg pages
    # ahead of BMM2, so a ring deeper than pstg (<= 5) makes slot s & 7 free
    # again before page s + 8 reuses it; the parity is (s >> 3) & 1.
    vote_ring = 8
    vote_ring_log2 = 3

    # fp16 global join: splits whose partial-O runs are pulled as one batch of
    # independent 16 B loads.
    join_blk = 16

    # e4m3 cluster mailbox geometry (the splits <= max_cluster path): slot s holds
    # split s's normalised fp16 partial [mma_n, D] plus its per-head (m, l) fp32
    # pair.  Slot 0 is the folder's own, staged locally; slots 1.. arrive over the
    # cluster interconnect.
    slot_ml_elems = 2 * mma_n           # m[8] then l[8]
    slot_ml_bytes = slot_ml_elems * 4   # 64 B

    if kv_dtype not in KV_DTYPES:
        raise ValueError(f"kv_dtype must be one of {KV_DTYPES}, got {kv_dtype!r}")
    fp8 = kv_dtype == "e4m3"
    kv_t = e4m3 if fp8 else Float16
    kv_bytes = 1 if fp8 else 2

    # Tile geometry (per dtype)
    if fp8:
        # A KV page is `D // 128` 16 KiB e4m3 d-slices: every smem tile is the
        # SAME 128-token x 128 B s128b tile at every D, so the TMA box, swizzle
        # and MMA smem descriptors never change with D.  D < 128 keeps the
        # 128-elem box (the tail bytes are the next head's rows / TMA zero
        # fill); BMM1 only walks the first `bmm1_kblks` 32-elem granules and
        # BMM2's garbage rows land in O lanes >= D, which no store emits.
        narrow_d = False
        mma_k = 128
        mma_kblk = 32
        k_slices = max(1, D // mma_k)
        v_slices = max(1, D // mma_k)
        o_blocks = max(1, D // mma_k)
        v_boxes = 1
        bmm1_kblks = min(D, mma_k) // mma_kblk       # 1 / 2 / 4 / 4 / 4
        sw_code = 2
        sw_tma = ccuda.TensorMapSwizzle.s128b
        k_stage_elems = tokens_per_k_stage * mma_k
        v_stage_elems = tokens_per_v_stage * mma_k
        q_tile_elems = mma_n_wide * mma_k            # 16-row Q tile: heads 8..15 real only on ratio-16 rows
        p_row = p_lanes                              # P lanes per token row (16 B)
        p_stage_elems = tokens_per_k_stage * p_row
        k_stage_bytes = k_stage_elems
        v_tx_bytes = v_stage_elems
        k_sbo = 8 * mma_k                            # 8 rows of the 128 B atom
        q_sbo = 8 * mma_k                            # rows 8..15 are the second core matrix
        v_lbo = v_stage_elems                        # MN-major V: leading offset = one d-slice (M = 128 never steps it)
        o_blk_d = mma_k                              # d-elems per O block
        # Descriptor steps in 16-byte units.
        kdesc_kstep = mma_kblk // 16                 # 2   (K and Q, K-major)
        vdesc_kstep = (mma_kblk * mma_k) // 16       # 256 (V, MN-major)
        pdesc_kstep = (mma_kblk * p_row) // 16       # 32  (P, MN-major)
        bmm2_ksteps = tokens_per_v_stage // mma_kblk # 4
        bmm2_n = p_lanes
        mma_kind = prims.Tcgen05MMAKind.F8F6F4
        # NCU: latency-bound at 1 CTA/SM; steady state is bounded by
        # `period >= tma_latency / stages`, so the deeper rings keep more bytes
        # in flight out of the smem the fp16 tiles would have used.
        kstg = 6
        vstg = 6
        sstg = 6                                     # TMEM S buffers (6*16 = 96 cols)
        pstg = 5
        # BMM2-drain mbarriers, one per page slot: `try_wait.parity` tests ONE
        # bit, so `p_stages` bounds how far the softmax can run ahead and
        # 8 > 5 makes (slot, phase) unambiguous.
        o_ring = 8
        o_ring_log2 = 3
        s_slot = mma_n_wide                          # TMEM cols per S ring slot (8-wide rows use the low half)
        o_blk_cols = p_lanes                         # TMEM cols per O block
        tmem_cols = 128 if D <= 256 else 256         # S 96 + O 16*o_blocks, PO2-rounded
        mx_heads = mma_n_wide                        # sMx / sLx width
        pad_n = mma_n_wide                           # workspace head padding
        headroom = max_headroom
        gqa_arm = False
        spec_page0 = True
        lookahead = False
        warm_first_page = False
        dg_guess = False
        min_pages = min_pages_per_cta
        trim_plan = False
        spread_fold = False
        d_quads = D // 4               # 16 B fold granules per head (8 .. 128)
        # Longest split run ONE lane of the split-parallel fold walks
        # (`L >= 16 * live / d_quads`, so `ceil(live / L) <= d_quads / 16`).
        nsp_max = max(2, d_quads // 16)
        n_qit = max(1, D // 64)        # 16-elem Q chunks per staging lane
        max_cluster_d = _mc(D)
        slot_o_elems = mma_n * D
        slot_o_bytes = slot_o_elems * 2
        slot_tx_bytes = slot_o_bytes + slot_ml_bytes
        # fp16-only knobs, bound so every name the kernel closes over exists.
        v_slice_d = mma_k
        warm_slices = 0
        warm_v_slices = 0
    else:
        # D = 32 is HALF a 128 B swizzle row, so its K/V/Q tiles are 32 elems
        # wide and ride a 64 B swizzle instead: the stage extents, both MMA
        # descriptors' stride-byte-offset and their layout code follow the TILE
        # width, not the head dim.
        narrow_d = D == 32
        mma_k = 32 if narrow_d else 64
        k_stage_elems = tokens_per_k_stage * mma_k
        v_subtile_elems = tokens_per_v_stage * mma_k
        q_tile_elems = mma_n * mma_k
        k_stage_bytes = k_stage_elems * 2
        sw_code = 4 if narrow_d else 2   # Tcgen05SmemSwizzle SWIZZLE_64B / _128B
        sw_tma = (ccuda.TensorMapSwizzle.s64b if narrow_d
                  else ccuda.TensorMapSwizzle.s128b)
        k_sbo = 8 * mma_k * 2            # stride-byte-offset: 8 rows of the atom
        q_sbo = 0
        # BMM2 keeps M = 128 at every head dim (an M = 64 tcgen05 MMA lays its
        # accumulator out differently across the 128 TMEM lanes and measured
        # INCORRECT_NUMERICAL at D = 64).  D = 512/256 fill it with two real
        # 64-elem d-atoms per V stage, D = 64 leaves the upper atom stale, and
        # D = 32 aliases FOUR 32-elem atoms over the same 8 KiB, so O TMEM lanes
        # past `d_lanes` are duplicates the store predicate drops.
        k_slices = D // mma_k               # K d-slices per page: 1 / 2 / 4 / 8
        v_slices = max(D // (2 * mma_k), 1)  # V stages per page:  1 / 1 / 2 / 4
        v_boxes = 2 if D >= 2 * mma_k else 1  # 64-elem TMA boxes per V stage
        v_tx_bytes = v_boxes * v_subtile_elems * 2
        # Where a V stage carries ONE box (D = 32 / 64) the upper atoms have no
        # data behind them: a leading offset of 0 aliases them onto the real
        # subtile so the V stage shrinks to exactly what the TMA fills.
        v_lbo = v_subtile_elems * 2 if v_boxes == 2 else 0
        v_stage_elems = v_boxes * v_subtile_elems
        # Stage depths are per head dim.  At D = 512 a page is EIGHT 16 KiB K
        # slices and FOUR 32 KiB V slices, and 7/3 fits neither; moving 32 KiB
        # from K to V puts the WHOLE V page in the first round.  Where the V
        # stage shrank to one box the 32 KiB goes to the K ring, not the V one:
        # K heads the whole dependency chain (BMM1 -> softmax -> P -> BMM2).
        kstg = 5 if D >= 8 * mma_k else (k_stages if v_boxes == 2 else 10)
        vstg = 4 if D >= 8 * mma_k else v_stages
        sstg = 4
        v_slice_d = 2 * mma_k               # d-elems covered by one V stage
        o_blocks = max(D // (2 * mma_k), 1)  # 8-column O blocks in TMEM
        o_blk_d = v_slice_d
        spread_fold = o_blocks > 2          # split the join sweep across arrivals
        # D = 512 already spends 8 KiB on Q and 112 KiB on K stages; the fourth
        # P buffer measured as a LOSS there while paying for itself at D <= 256.
        pstg = p_stages if D < 8 * mma_k else 3
        o_ring = 4
        o_ring_log2 = 2
        s_slot = mma_n
        o_blk_cols = mma_n
        tmem_cols = 64                      # S: sstg*8 = 32, O: 8*o_blocks -> 64
        mx_heads = softmax_heads
        pad_n = mma_n
        p_row = mma_n
        p_stage_elems = tokens_per_k_stage * p_row
        bmm1_kblks = mma_k // 16
        kdesc_kstep = 2
        vdesc_kstep = 2 * mma_k             # 16 tokens x mma_k halves / 16 B
        pdesc_kstep = 16                    # 16 tokens x 8 halves / 16 B
        bmm2_ksteps = tokens_per_v_stage // 16
        bmm2_n = mma_n
        mma_kind = prims.Tcgen05MMAKind.F16
        headroom = 1.0
        # The half-width softmax arm (GQA <= 4) is traced at every head dim
        # below 512: at D = 512 o_blocks = 4 makes the duplicated
        # correction/epilogue the widest in the kernel while every D = 512 row
        # walks one or two pages.
        gqa_arm = D <= 256
        # Cycle-zero L2 warm-up of this split's guessed first page, capped at
        # what BMM1 chains first: two K d-slices and one V stage, at every D
        # (an ungated D = 512 ramp pushed 256 KiB per CTA into DRAM before a
        # single real box issued and measured as pure start-up contention).
        warm_slices = k_slices if k_slices < 2 else 2
        warm_v_slices = v_slices if v_slices < 1 else 1
        warm_first_page = True
        dg_guess = True
        # Split 0 prefetches page 0 ahead of the plan only when every box fits
        # a fresh pipeline stage (all D but 512); heavier pages also split sooner.
        spec_page0 = (k_slices <= kstg) and (v_slices <= vstg)
        # Paced L2 lookahead (D <= 64 only): page j+1's bytes start their DRAM
        # trip while page j's boxes are still issuing.  D = 256 measured ~+1%
        # (probe traffic on the critical K stream), D = 512 +3..8% -- declined.
        lookahead = k_slices == 1
        min_pages = min_pages_per_cta if D <= 256 else 1
        trim_plan = D >= 8 * mma_k          # only D = 512 wants the narrower plan
        # e4m3-only knobs, bound so every name the kernel closes over exists.
        max_cluster_d = 1
        d_quads = D // 4
        nsp_max = 2
        n_qit = 1
        slot_o_elems = 0
        slot_o_bytes = 0
        slot_tx_bytes = 0
    p_t = kv_t                              # P is staged in the KV element type
    d_lanes = D if D < 128 else 128         # TMEM lanes holding real O rows
    o_col_offset = sstg * s_slot
    bmm1_m = tokens_per_k_stage
    bmm2_m = 128
    # Softmax head-tile arms.  The GQA <= 4 half-width arm (fp16, D <= 256)
    # halves the per-page serial chain; the body below is written ONCE and
    # traced per arm with `NH` a Python constant.  Where only the 8-wide arm
    # exists its select predicate is a trace-time constant and folds away.
    arms = (4, 8) if gqa_arm else (8,)

    @cute.kernel
    def decode(
        mQ: cute.Tensor,
        mPT: cute.Tensor,
        mSL: cute.Tensor,
        mSNK: cute.Tensor,
        mO: cute.Tensor,
        mWS: cute.Tensor,
        mWH: cute.Tensor,
        mSEM: cute.Tensor,
        mSKT: cute.Tensor,
        tma_k: cutlass.GridConstant[ccuda.TensorMap],
        tma_v: cutlass.GridConstant[ccuda.TensorMap],
        scale: Float32,
        v_scale: Float32,
        MAXP: Int32,
        gqa_ratio: Int32,
        window: Int32,
        has_sink: Int32,
        hk_shift: Int32,
    ):
        h_k_idx, split_idx, b_idx = cute.arch.block_idx()
        # fp16 serves GQA ratio 9..16 as TWO adjacent mma_n = 8 head groups of
        # the SAME KV head: the launch widens the fast grid mode to HKV << 1 and
        # `hk_shift` maps a group back onto its KV head, so the two groups walk
        # the same pages side by side and DRAM sees the bytes once.  e4m3 widens
        # the MMA chain to n = 16 instead (`wide`, below) and never shifts.
        if cutlass.const_expr(fp8):
            h_kv = h_k_idx
        else:
            h_kv = h_k_idx >> hk_shift
        # B / HKV / SPLITS ride in for free as grid dimensions instead of as
        # extra dispatch scalars.  HKV is the FAST grid mode: the KV-head CTAs
        # of one (b, split) walk the very same page ids, each taking a
        # different slice of the same token row, so dispatching them adjacently
        # puts their requests on the same DRAM rows at the same time.
        HKV, SPLITS, b_dim = cute.arch.grid_dim()
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        wg_widx = warp_idx % 4                       # position inside warpgroup
        row = wg_widx * warp_threads + lane                  # TMEM row this thread owns

        HQ = HKV * gqa_ratio
        groups = b_dim * HKV
        coord_hb = b_idx * HKV + h_k_idx
        if cutlass.const_expr(fp8):
            # The mailbox join replaces the global one exactly when the host put
            # the splits of this group in one cluster; both sides key on the
            # same predicate.  Ratio-16 rows never cluster (their slot would be
            # 2x) and the host launches them unclustered.
            wide = gqa_ratio > mma_n
            clustered = SPLITS <= max_cluster_d
            if wide:
                clustered = cutlass.Boolean(False)

        # Smem
        tmem_holder = cutlass.Array(
            cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4
        )
        kbar = cutlass.Array(
            cutlass.Int64, kstg * 2, space=cutlass.AddressSpace.smem, alignment=8
        )
        vbar = cutlass.Array(
            cutlass.Int64, vstg * 2, space=cutlass.AddressSpace.smem, alignment=8
        )
        sbar = cutlass.Array(
            cutlass.Int64, sstg * 2, space=cutlass.AddressSpace.smem, alignment=8
        )
        pbar = cutlass.Array(
            cutlass.Int64, pstg * 2, space=cutlass.AddressSpace.smem, alignment=8
        )
        # BMM2's per-page drain commits rotate over a ring of `o_ring`
        # barriers: the softmax only waits here on the rare growing-max page,
        # so between two waits the phase can flip several times -- the ring
        # keeps the parity test unambiguous for any run-ahead the P ring allows.
        obar = cutlass.Array(
            cutlass.Int64, o_ring, space=cutlass.AddressSpace.smem, alignment=8
        )
        sMx = cutlass.Array(
            Float32, 2 * mx_heads * 4, space=cutlass.AddressSpace.smem, alignment=16
        )
        sLx = cutlass.Array(
            Float32, mx_heads * 4, space=cutlass.AddressSpace.smem, alignment=16
        )
        sFlag = cutlass.Array(
            cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16
        )
        # Skip-softmax page votes (1 = the page's V load and P.V are skipped).
        sVote = cutlass.Array(
            cutlass.Int32, vote_ring, space=cutlass.AddressSpace.smem, alignment=16
        )
        vtbar = cutlass.Array(
            cutlass.Int64, vote_ring, space=cutlass.AddressSpace.smem, alignment=8
        )
        if cutlass.const_expr(fp8):
            # Split-K mailbox (splits <= max_cluster rows): split 0's copy is
            # the live one; every CTA allocates it because a cluster has no
            # shared malloc -- writers reach split 0's instance of the SAME
            # offset through prims.mapa.  jbar completes the (live_splits - 1)
            # inbound bulk copies (receiver-armed).
            jbar = cutlass.Array(
                cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
            )
            sJO = cutlass.Array(
                Float16, max_cluster_d * slot_o_elems,
                space=cutlass.AddressSpace.smem, alignment=16,
            )
            sJML = cutlass.Array(
                Float32, max_cluster_d * slot_ml_elems,
                space=cutlass.AddressSpace.smem, alignment=16,
            )
            # Per-head Q dequant scale: Q-staging warp -> softmax warpgroup.
            sQS = cutlass.Array(
                Float32, mma_n_wide, space=cutlass.AddressSpace.smem, alignment=64
            )
        sK = cutlass.Array(
            kv_t, k_stage_elems * kstg, space=cutlass.AddressSpace.smem, alignment=1024
        )
        sV = cutlass.Array(
            kv_t, v_stage_elems * vstg, space=cutlass.AddressSpace.smem, alignment=1024
        )
        sQ = cutlass.Array(
            kv_t, q_tile_elems * k_slices, space=cutlass.AddressSpace.smem, alignment=1024
        )
        sP = cutlass.Array(
            p_t, p_stage_elems * pstg, space=cutlass.AddressSpace.smem, alignment=1024
        )

        desc_k = tma_k.get_ptr()
        desc_v = tma_v.get_ptr()

        # `seq_lens` is issued HERE, at cycle zero, but first consumed only
        # after the whole seq_lens-independent prologue below: the scoreboard
        # stall lands on `live_pages`, so the mbarrier builds, the TMEM alloc,
        # the Q staging and the speculative page-table window all execute in
        # the shadow of this load instead of behind it.
        seq_len = mSL.iterator[b_idx]

        if cutlass.const_expr(dg_guess):
            # `dg` arms the capacity-guessed per-split probe in the producer
            # warps below; the guess also names THIS split's FIRST page, which
            # is what the cycle-zero L2 warm-up prefetches.  Two integer
            # divisions on the two producer warps, in the shadow of `seq_lens`.
            dg = Int32(1)

        # Speculative page-id fetch
        # The page ids sit behind a TWO-deep dependent global chain
        # (`seq_lens` -> live_pages -> pages_per_cta -> p0 -> page id); the
        # second load is pure serialisation.  But the split walk deals every
        # split a window inside [0, live_pages), and live_pages is at most one
        # page_chunk on most rows -- so the row's FIRST 32 page ids, an address
        # known at cycle zero, cover every split's window whatever the plan
        # turns out to be.  Warp 0 fetches that window immediately; only the
        # wide-window rows still pay the dependent load.  Lanes past the live
        # window hold a page id that is never shuffled out, so the clamp is
        # only there to keep every probe inside this batch's own table row.
        kp_g = Int32(0)
        vp_g = Int32(0)
        pbase_g = Int32(0)
        if cutlass.const_expr(dg_guess):
            kp_gg = Int32(0)
            fp_g = Int32(0)
            p0_gg = Int32(0)
        if warp_idx == 0:
            prims.prefetch_tensormap(desc_k)
            if cutlass.const_expr(v_boxes == 2):
                prims.prefetch_tensormap(desc_v)
            gpt = mPT.iterator + (b_idx * 2) * MAXP
            # The shared probe window is the FIRST 32 pages of the row: one
            # `shfl.idx` reaches any lane of it.  Windowed rows probe AFTER the
            # mbarrier builds instead (their base needs seq_len).
            if window == 0:
                gi = lane
                if gi > MAXP - 1:
                    gi = MAXP - 1
                kp_g = gpt[gi]
                vp_g = gpt[MAXP + gi]
            if cutlass.const_expr(dg_guess):
                # Capacity-guessed probe: on every long row the engine's table
                # is FULL (ceil(KV/128) == MAXP) and windowed rows are
                # page-aligned to it, so a plan computed from the GUESS
                # seq_len == MAXP * 128 needs nothing from memory -- this
                # split's own window's ids go out here, at cycle zero, instead
                # of behind the seq_lens -> plan -> page-id dependent chain.
                # The walk validates the guess against the real plan before
                # using them, so the guess has to reproduce the real plan
                # EXACTLY (including the `hk_shift == 0` gate on the D = 512
                # halving).
                if dg != 0:
                    if window > 0:
                        lo_gg = (MAXP << 7) - window
                        if lo_gg > 0:
                            fp_g = lo_gg >> 7
                    span_gg = MAXP - fp_g
                    p0_gg = fp_g
                    if span_gg > min_pages:
                        live_gg = SPLITS
                        if span_gg < SPLITS:
                            live_gg = span_gg
                        if cutlass.const_expr(trim_plan):
                            if span_gg <= SPLITS:
                                if groups != 1:
                                    if span_gg >= 8:
                                        if hk_shift == 0:
                                            live_gg = (span_gg + 1) >> 1
                        qg = span_gg // live_gg
                        rg = span_gg - qg * live_gg
                        p0_gg = fp_g + split_idx * qg + rg
                        if split_idx < rg:
                            p0_gg = fp_g + split_idx * (qg + 1)
                        if split_idx >= live_gg:
                            p0_gg = fp_g
                    gig = p0_gg + lane
                    if gig > MAXP - 1:
                        gig = MAXP - 1
                    kp_gg = gpt[gig]
            else:
                # Warm the WHOLE page-table row in L2 while the probe is in
                # flight: one 128 B line per lane covers `[2, MAXP]` int32 in
                # ceil(MAXP/16) lanes, so the long rows' per-chunk id refetches
                # become L2 hits instead of dependent DRAM misses in front of a
                # TMA issue.
                if lane < ((MAXP + Int32(15)) >> 4):
                    prims.prefetch_l2(gpt + (lane << 5))
            # The KV mbarriers and the CTA-wide arrive carry no dependency on
            # `seq_lens` either, so they are built here rather than after the
            # split plan (or the windowed rows' probe).
            if prims.elect_sync():
                for i in cutlass.range_constexpr(kstg):
                    prims.mbarrier_init(kbar.subview(i).data_ptr(), 1)
                    prims.mbarrier_init(kbar.subview(kstg + i).data_ptr(), 1)
                for i in cutlass.range_constexpr(vstg):
                    prims.mbarrier_init(vbar.subview(i).data_ptr(), 1)
                    prims.mbarrier_init(vbar.subview(vstg + i).data_ptr(), 1)
                prims.fence_mbarrier_init()
            prims.bar_warp_sync(0xFFFFFFFF)
            if cutlass.const_expr(fp8):
                # Warp 3 stages Q here, so the whole CTA meets on barrier 0.
                prims.barrier_cta_arrive(0, 256)
            else:
                # Two releases, not one.  The CTA-wide barrier publishes the
                # TMEM base and the S/P/O mbarriers, so it cannot retire before
                # the Q staging; the V producer needs NEITHER -- only the K/V
                # mbarriers this warp just built -- so it leaves on a private
                # 64-thread barrier and stops paying for the Q tile.
                prims.barrier_cta_arrive(kv_ready_barrier, 64)
                prims.barrier_cta_arrive(0, 224)
            # Cycle-zero L2 warm-up of this split's GUESSED first K page: one
            # tensor prefetch moves it DRAM -> L2 while the plan is still
            # resolving.  A prefetch writes nothing, so an invalid guess or a
            # dead split only warms one wasted page.  Gated to MAXP <= 128: the
            # ramp is the walk's whole cost only on shallow walks.
            if cutlass.const_expr(warm_first_page):
              if dg != 0:
                if MAXP <= 128:
                  kpg0 = prims.shfl_sync(
                      thread_mask=0xFFFFFFFF,
                      val=kp_gg,
                      offset=0,
                      mask_and_clamp=0x1F,
                      kind=prims.Shfl.IDX,
                      return_value_and_is_valid=False,
                  )
                  if prims.elect_sync():
                      for dk in cutlass.range_constexpr(warm_slices):
                          prims.cp_async_bulk_tensor_prefetch(
                              desc_k,
                              [Int32(dk * mma_k), h_kv, Int32(0), kpg0],
                              [],
                          )
            # A sliding window moves the first live page: only the windowed
            # rows wait on seq_lens here, and they do it AFTER the mbarrier
            # builds and the CTA-wide arrive above.
            if window > 0:
                lo2 = seq_len - window
                if lo2 > 0:
                    pbase_g = lo2 >> 7
                gi = pbase_g + lane
                if gi > MAXP - 1:
                    gi = MAXP - 1
                kp_g = gpt[gi]
                vp_g = gpt[MAXP + gi]
            # Split 0's window starts at page 0 no matter what the plan says,
            # and live_pages >= 1 guarantees page 0 is always inside its window
            # -- so its first page's K and V boxes are issued HERE, gated only
            # on the speculative page-table window landing, never on seq_lens
            # or the plan ALU.  The walk below starts this CTA at page 1 with
            # the pipeline indices pre-advanced; the release-barrier waits are
            # skipped because a first-ever use of a fresh stage never blocks.
            if cutlass.const_expr(spec_page0):
              if split_idx == 0:
                kp0 = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=kp_g,
                    offset=0,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                    return_value_and_is_valid=False,
                )
                vp0 = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=vp_g,
                    offset=0,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                    return_value_and_is_valid=False,
                )
                if prims.elect_sync():
                    for dk in cutlass.range_constexpr(k_slices):
                        prims.mbarrier_arrive_expect_tx(
                            kbar.subview(dk).data_ptr(), k_stage_bytes
                        )
                        if cutlass.const_expr(fp8):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sK.subview(dk * k_stage_elems).data_ptr(),
                                desc_k,
                                (h_k_idx * D + dk * mma_k, kp0 * tokens_per_page),
                                kbar.subview(dk).data_ptr(),
                            )
                        else:
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sK.subview(dk * k_stage_elems).data_ptr(),
                                desc_k,
                                (Int32(dk * mma_k), h_kv, Int32(0), kp0),
                                kbar.subview(dk).data_ptr(),
                            )
                    for sk in cutlass.range_constexpr(v_slices):
                        prims.mbarrier_arrive_expect_tx(
                            vbar.subview(sk).data_ptr(), v_tx_bytes
                        )
                        if cutlass.const_expr(fp8):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sV.subview(sk * v_stage_elems).data_ptr(),
                                desc_k,
                                (h_k_idx * D + sk * mma_k, vp0 * tokens_per_page),
                                vbar.subview(sk).data_ptr(),
                            )
                        elif cutlass.const_expr(v_boxes == 2):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sV.subview(sk * v_stage_elems).data_ptr(),
                                desc_v,
                                (Int32(0), Int32(0), h_kv, Int32(sk * 2), vp0),
                                vbar.subview(sk).data_ptr(),
                            )
                        else:
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sV.subview(sk * v_stage_elems).data_ptr(),
                                desc_k,
                                (Int32(sk * v_slice_d), h_kv, Int32(0), vp0),
                                vbar.subview(sk).data_ptr(),
                            )

        # Init warps
        if warp_idx == 1:
            prims.tcgen05_alloc(tmem_holder, Int32(tmem_cols))
            prims.tcgen05_relinquish_alloc_permit()
            if cutlass.const_expr(fp8):
                # Warm the split-K rendezvous lines while the TMEM allocator
                # runs -- only the wide (global-join) plans touch them.  None of
                # these addresses depends on `seq_lens`; the epilogue touches
                # them ~5 us later, by which time a cold DRAM fetch has landed.
                if SPLITS > max_cluster_d:
                    hq_pad_w = b_dim * HQ + mma_n
                    max_off_w = hq_pad_w * SPLITS * D
                    hq0_w = b_idx * HQ + h_k_idx * gqa_ratio
                    if lane == 0:
                        prims.prefetch_l2(mSEM.iterator + coord_hb * sem_words)
                    if lane < softmax_heads:
                        mbase_w = mWS.iterator + max_off_w + (hq0_w + lane) * SPLITS
                        prims.prefetch_l2(mbase_w)
                        prims.prefetch_l2(mbase_w + hq_pad_w * SPLITS)
        if warp_idx == 2:
            if prims.elect_sync():
                # The sink logits are consumed at the very END of the kernel as
                # a dependent gmem load on the terminal critical path, and the
                # KV stream has long since evicted the line.  Warm it at cycle
                # zero; `has_sink == 0` never touches the tensor.
                if has_sink == 1:
                    prims.prefetch_l2(mSNK.iterator + h_k_idx * gqa_ratio)
                for i in cutlass.range_constexpr(sstg):
                    prims.mbarrier_init(sbar.subview(i).data_ptr(), 1)
                    prims.mbarrier_init(sbar.subview(sstg + i).data_ptr(), warpgroup_threads)
                for i in cutlass.range_constexpr(pstg):
                    prims.mbarrier_init(pbar.subview(i).data_ptr(), warpgroup_threads)
                    prims.mbarrier_init(pbar.subview(pstg + i).data_ptr(), 1)
                for i in cutlass.range_constexpr(o_ring):
                    prims.mbarrier_init(obar.subview(i).data_ptr(), 1)
                for i in cutlass.range_constexpr(vote_ring):
                    prims.mbarrier_init(vtbar.subview(i).data_ptr(), 1)
                if cutlass.const_expr(fp8):
                    # The mailbox barrier: ONE arrival (the folder arms it with
                    # the inbound byte count once the plan is known), plus the
                    # engine's complete_tx for every writer's two bulk copies.
                    prims.mbarrier_init(jbar.data_ptr(), 1)
                prims.fence_mbarrier_init()
            # Q stage (warp 2)
            # fp16 only: a one-slice Q tile (D = 32 / 64) is at most two 16 B
            # chunks per lane, so a wider crew buys nothing.  Layout reproduced:
            # 128 B swizzle over an 8-row atom, i.e. the physical 16 B chunk
            # index is `chunk ^ row`.  Rows at or past the runtime GQA ratio
            # are zero-filled.
            if cutlass.const_expr(narrow_d):
                # D = 32: the whole Q tile is 8 rows x 32 elems under a 64 B
                # swizzle, i.e. FOUR 16 B chunks per row -- exactly one chunk
                # per lane.  The swizzle XOR takes the row's 64 B-pair index
                # (Swizzle<2,4,3>) where the 128 B tile takes the row itself.
                qr = lane >> 2                      # Q row inside the mma_n tile
                qc = lane & 3                       # 16 B chunk along d
                qbase = (b_idx * HQ + h_k_idx * gqa_ratio) * D
                sof = qr * mma_k + ((qc ^ ((qr >> 1) & 3)) * 8)
                gof = qbase + qr * D + qc * 8
                if qr < gqa_ratio:
                    prims.store_ext(
                        prims.load_ext(mQ.iterator + gof, dtype=Float32, count=4),
                        sQ.subview(sof).data_ptr(),
                    )
                else:
                    prims.store_ext(
                        cutlass.vector.full((4,), 0, Float32),
                        sQ.subview(sof).data_ptr(),
                    )
                cute.arch.fence_view_async_shared()
            if cutlass.const_expr((not fp8) and k_slices == 1 and not narrow_d):
                qh = lane >> 3                      # row within a 4-row half
                qc = lane & 7                       # 16 B chunk along d
                qbase = (b_idx * HQ + h_k_idx * gqa_ratio) * D
                # The two 16 B chunks a lane owns are rows `qh` and `qh + 4` of
                # the SAME d-slice; `qh + 4 < gqa_ratio` implies
                # `qh < gqa_ratio`, so three nested cases cover the walk and the
                # full-ratio one issues both loads together (one DRAM round).
                sof0 = qh * mma_k + ((qc ^ qh) * 8)
                sof1 = (4 + qh) * mma_k + ((qc ^ (4 + qh)) * 8)
                gof0 = qbase + qh * D + qc * 8
                gof1 = qbase + (4 + qh) * D + qc * 8
                if 4 + qh < gqa_ratio:
                    q0 = prims.load_ext(mQ.iterator + gof0, dtype=Float32, count=4)
                    q1 = prims.load_ext(mQ.iterator + gof1, dtype=Float32, count=4)
                    prims.store_ext(q0, sQ.subview(sof0).data_ptr())
                    prims.store_ext(q1, sQ.subview(sof1).data_ptr())
                else:
                    if qh < gqa_ratio:
                        prims.store_ext(
                            prims.load_ext(mQ.iterator + gof0, dtype=Float32, count=4),
                            sQ.subview(sof0).data_ptr(),
                        )
                    else:
                        prims.store_ext(
                            cutlass.vector.full((4,), 0, Float32),
                            sQ.subview(sof0).data_ptr(),
                        )
                    prims.store_ext(
                        cutlass.vector.full((4,), 0, Float32),
                        sQ.subview(sof1).data_ptr(),
                    )
                cute.arch.fence_view_async_shared()
        # Q stage (softmax wg)
        # fp16, k_slices >= 2: the Q tile is `8 rows x D` halves = 64 * k_slices
        # chunks of 16 B.  The softmax warpgroup has nothing to do until the
        # first S tile lands, and it is 128 threads wide, so it covers TWO
        # d-slices per round.  Every load is issued before the first store so
        # the chain collapses to ONE DRAM round.
        if cutlass.const_expr((not fp8) and k_slices >= 2):
            if warp_idx >= 4:
                wt = tidx - warpgroup_threads
                qr = (wt >> 3) & 7                  # Q row inside the mma_n tile
                qc = wt & 7                         # 16 B chunk along d
                dsel = wt >> 6                      # which of the round's two slices
                qbase = (b_idx * HQ + h_k_idx * gqa_ratio) * D
                if qr < gqa_ratio:
                    qv = []
                    for it in cutlass.range_constexpr(k_slices // 2):
                        dk = it * 2 + dsel
                        gof = qbase + qr * D + dk * mma_k + qc * 8
                        qv.append(
                            prims.load_ext(mQ.iterator + gof, dtype=Float32, count=4)
                        )
                    for it in cutlass.range_constexpr(k_slices // 2):
                        dk = it * 2 + dsel
                        sof = dk * q_tile_elems + qr * mma_k + ((qc ^ qr) * 8)
                        prims.store_ext(qv[it], sQ.subview(sof).data_ptr())
                else:
                    for it in cutlass.range_constexpr(k_slices // 2):
                        dk = it * 2 + dsel
                        sof = dk * q_tile_elems + qr * mma_k + ((qc ^ qr) * 8)
                        prims.store_ext(
                            cutlass.vector.full((4,), 0, Float32),
                            sQ.subview(sof).data_ptr(),
                        )
                cute.arch.fence_view_async_shared()
        # V-producer id window (warp 3)
        vp_g3 = Int32(0)
        if cutlass.const_expr(dg_guess):
            pbase3 = Int32(0)
            vp_gg3 = Int32(0)
            fp_g3 = Int32(0)
            p0_gg3 = Int32(0)
        if warp_idx == 3:
            if cutlass.const_expr(fp8):
                # Warp 3 owns the V half of the producer, so it pulls its own
                # cycle-zero page-id window here -- ahead of the Q staging
                # block, whose dependent chain then runs in the load's shadow.
                # The line is already in L1 from warp 0's identical probe.
                gi3 = lane
                gcap3 = MAXP - 1
                if gi3 > gcap3:
                    gi3 = gcap3
                vp_g3 = (mPT.iterator + (b_idx * 2) * MAXP + MAXP)[gi3]
                # -------- Q stage + quantise to e4m3: hand-swizzled copy
                # (chunk ^ row), rows past the GQA ratio zero-filled, PER-HEAD
                # absmax over the WHOLE head (a clipping Q quantisation fails
                # the long rows).  Every chunk is issued BEFORE the absmax tree
                # runs and the quantise pass reuses the SAME registers, so the
                # whole staging costs ONE memory round trip at every head dim.
                # The GQA-padding rows alias onto head 0 so the loads need no
                # predicate -- `qmul` carries the mask that zeroes them.
                qr = lane >> 2                          # head row of the mma_n tile
                qc = lane & 3                           # 16 B chunk pair of a 128 B row
                qbase = (b_idx * HQ + h_k_idx * gqa_ratio) * D
                qrs = qr
                if qr >= gqa_ratio:
                    qrs = Int32(0)
                qvs = []
                for it in cutlass.range_constexpr(n_qit):
                    for h in cutlass.range_constexpr(2):
                        coff = (qc + it * 4) * 16 + h * 8
                        if cutlass.const_expr(D == 32):
                            # A 32-elem head row is two 16-elem chunks; lanes
                            # qc 2/3 re-read the row head (same absmax, and
                            # their sQ chunks sit past bmm1_kblks so the MMA
                            # never reads them) instead of running past the
                            # tensor.
                            if coff >= D:
                                coff = Int32(h * 8)
                        gof = qbase + qrs * D + coff
                        qvs.append(
                            prims.load_ext(
                                mQ.iterator + gof, dtype=cutlass.Int16, count=8
                            )
                        )
                qq = cutlass.Array(e4m3, 16, alignment=16)
                amx = Float32(0.0)
                for it in cutlass.range_constexpr(n_qit):
                    for h in cutlass.range_constexpr(2):
                        vv = qvs[it * 2 + h]
                        for e in cutlass.range_constexpr(8):
                            x = vv[e].bitcast(Float16).to(Float32)
                            amx = cute.arch.fmax(amx, cute.arch.fmax(x, -x))
                for off in cutlass.range_constexpr(2):
                    amx = cute.arch.fmax(
                        amx,
                        prims.shfl_sync(
                            thread_mask=0xFFFFFFFF, val=amx, offset=(2 >> off),
                            mask_and_clamp=0x1F, kind=prims.Shfl.BFLY,
                            return_value_and_is_valid=False,
                        ),
                    )
                # An all-zero head (GQA padding) would divide by nothing.
                amx = cute.arch.fmax(amx, Float32(1.0e-30))
                qmul = cute.arch.rcp_approx(amx) * Float32(e4m3_max)
                if qr >= gqa_ratio:
                    qmul = Float32(0.0)
                for it in cutlass.range_constexpr(n_qit):
                    for h in cutlass.range_constexpr(2):
                        vv = qvs[it * 2 + h]
                        for e in cutlass.range_constexpr(8):
                            x = vv[e].bitcast(Float16).to(Float32)
                            qq[h * 8 + e] = (x * qmul).to(e4m3)
                    # chunk index over the whole D-wide row routes to its
                    # 16 KiB d-slice: slice = chunk >> 3, swizzle within it.
                    ch = qc + it * 4
                    sof = (ch >> 3) * q_tile_elems + qr * mma_k + (((ch & 7) ^ qr) * 16)
                    sQ.subview(sof).data_ptr().store(
                        qq.data_ptr().load(count=16, alignment=16), alignment=16
                    )
                if qc == 0:
                    sQS[qr] = scale * amx * Float32(1.0 / e4m3_max)
                # Ratio-16: heads 8..15 land in rows 8..15 of the SAME Q tile
                # (the second 8-row core matrix of the K-major SW128 layout,
                # which the n=8 descriptor never reads on narrow rows).
                if gqa_ratio > mma_n:
                    qr2 = qr + mma_n
                    qrs2 = qr2
                    if qr2 >= gqa_ratio:
                        qrs2 = Int32(0)
                    qvs2 = []
                    for it in cutlass.range_constexpr(n_qit):
                        for h in cutlass.range_constexpr(2):
                            gof2 = qbase + qrs2 * D + (qc + it * 4) * 16 + h * 8
                            qvs2.append(
                                prims.load_ext(
                                    mQ.iterator + gof2, dtype=cutlass.Int16, count=8
                                )
                            )
                    amx2 = Float32(0.0)
                    for it in cutlass.range_constexpr(n_qit):
                        for h in cutlass.range_constexpr(2):
                            vv2 = qvs2[it * 2 + h]
                            for e in cutlass.range_constexpr(8):
                                x2 = vv2[e].bitcast(Float16).to(Float32)
                                amx2 = cute.arch.fmax(amx2, cute.arch.fmax(x2, -x2))
                    for off in cutlass.range_constexpr(2):
                        amx2 = cute.arch.fmax(
                            amx2,
                            prims.shfl_sync(
                                thread_mask=0xFFFFFFFF, val=amx2, offset=(2 >> off),
                                mask_and_clamp=0x1F, kind=prims.Shfl.BFLY,
                                return_value_and_is_valid=False,
                            ),
                        )
                    amx2 = cute.arch.fmax(amx2, Float32(1.0e-30))
                    qmul2 = cute.arch.rcp_approx(amx2) * Float32(e4m3_max)
                    if qr2 >= gqa_ratio:
                        qmul2 = Float32(0.0)
                    for it in cutlass.range_constexpr(n_qit):
                        for h in cutlass.range_constexpr(2):
                            vv2 = qvs2[it * 2 + h]
                            for e in cutlass.range_constexpr(8):
                                x2 = vv2[e].bitcast(Float16).to(Float32)
                                qq[h * 8 + e] = (x2 * qmul2).to(e4m3)
                        ch2 = qc + it * 4
                        sof2 = (
                            (ch2 >> 3) * q_tile_elems + (mma_n + qr) * mma_k
                            + (((ch2 & 7) ^ qr) * 16)
                        )
                        sQ.subview(sof2).data_ptr().store(
                            qq.data_ptr().load(count=16, alignment=16), alignment=16
                        )
                    if qc == 0:
                        sQS[mma_n + qr] = scale * amx2 * Float32(1.0 / e4m3_max)
                cute.arch.fence_view_async_shared()
            else:
                gpt3 = mPT.iterator + (b_idx * 2) * MAXP
                if window == 0:
                    gi3 = lane
                    if gi3 > MAXP - 1:
                        gi3 = MAXP - 1
                    vp_g3 = gpt3[MAXP + gi3]
                # Capacity-guessed V probe: cycle zero, no seq_lens (see warp 0).
                if dg != 0:
                    if window > 0:
                        lo_g3 = (MAXP << 7) - window
                        if lo_g3 > 0:
                            fp_g3 = lo_g3 >> 7
                    span_g3 = MAXP - fp_g3
                    p0_gg3 = fp_g3
                    if span_g3 > min_pages:
                        live_g3 = SPLITS
                        if span_g3 < SPLITS:
                            live_g3 = span_g3
                        if cutlass.const_expr(trim_plan):
                            if span_g3 <= SPLITS:
                                if groups != 1:
                                    if span_g3 >= 8:
                                        if hk_shift == 0:
                                            live_g3 = (span_g3 + 1) >> 1
                        qg3 = span_g3 // live_g3
                        rg3 = span_g3 - qg3 * live_g3
                        p0_gg3 = fp_g3 + split_idx * qg3 + rg3
                        if split_idx < rg3:
                            p0_gg3 = fp_g3 + split_idx * (qg3 + 1)
                        if split_idx >= live_g3:
                            p0_gg3 = fp_g3
                    gig3 = p0_gg3 + lane
                    if gig3 > MAXP - 1:
                        gig3 = MAXP - 1
                    vp_gg3 = gpt3[MAXP + gig3]
                    # Cycle-zero L2 warm-up of the guessed first V page (see
                    # the K-side warm-up on warp 0).  Warp 3 has slack until
                    # the plan resolves, so the wait on the guessed probe costs
                    # nothing.
                    if cutlass.const_expr(warm_first_page):
                      if MAXP <= 128:
                          vpg0 = prims.shfl_sync(
                              thread_mask=0xFFFFFFFF,
                              val=vp_gg3,
                              offset=0,
                              mask_and_clamp=0x1F,
                              kind=prims.Shfl.IDX,
                              return_value_and_is_valid=False,
                          )
                          if prims.elect_sync():
                              for sk in cutlass.range_constexpr(warm_v_slices):
                                  prims.cp_async_bulk_tensor_prefetch(
                                      desc_k,
                                      [Int32(sk * v_slice_d), h_kv, Int32(0), vpg0],
                                      [],
                                  )
                                  if cutlass.const_expr(v_boxes == 2):
                                      prims.cp_async_bulk_tensor_prefetch(
                                          desc_k,
                                          [Int32(sk * v_slice_d + mma_k), h_kv, Int32(0), vpg0],
                                          [],
                                      )
                if window > 0:
                    lo23 = seq_len - window
                    if lo23 > 0:
                        pbase3 = lo23 >> 7
                    gi3 = pbase3 + lane
                    if gi3 > MAXP - 1:
                        gi3 = MAXP - 1
                    vp_g3 = gpt3[MAXP + gi3]

        # Cost-proportional split plan
        # The page window is cut from the LIVE sequence length, not from the
        # engine capacity MAXP.  `seq_lens` is a device value, so the plan is
        # derived here rather than on the host; every CTA of the group computes
        # the same `live_splits`, which keeps the join exact.
        live_pages = (seq_len + Int32(tokens_per_page - 1)) >> 7
        if live_pages < 1:
            live_pages = Int32(1)
        # Sliding window: pages fully below `lo_tok` are skipped, not masked.
        lo_tok = Int32(0)
        if window > 0:
            lo2 = seq_len - window
            if lo2 > 0:
                lo_tok = lo2
        first_page = lo_tok >> 7
        span = live_pages - first_page
        # The window is cut as thin as the launched split count allows and
        # every page of slack is handed to an idle SM; sequences that already
        # fit one window keep the fused epilogue (their join costs more than
        # the two or three pages a split could save).
        p0 = first_page
        iters = span
        live_splits = Int32(1)
        if span > min_pages:
            # Remainder-dealt windows, not a uniform ceil: `q + (s < r)` pages
            # for split s.  This form leaves NO split empty, so `live_splits`
            # is `min(SPLITS, span)`.
            live = SPLITS
            if span < SPLITS:
                live = span
            if cutlass.const_expr(trim_plan):
                # fp16 D = 512 only: when the span fits one window per launched
                # split, halve the plan instead of parking one-page CTAs.  A
                # ratio-16 KV head is served by TWO 8-head groups, so it keeps
                # every window; `hk_shift` is the discriminator that survives
                # the shift (`gqa_ratio` is the post-shift ratio, always <= 8).
                if span <= SPLITS:
                    if groups != 1:
                        if span >= 8:
                            if hk_shift == 0:
                                live = (span + 1) >> 1
            q = span // live
            r = span - q * live
            iters = q
            p0 = first_page + split_idx * q + r
            if split_idx < r:
                iters = q + 1
                p0 = first_page + split_idx * (q + 1)
            if split_idx >= live:
                # Windows past the trimmed plan take no pages AND no page-table
                # probe: their `p0` would run off this batch's table row.
                iters = Int32(0)
                p0 = first_page
            live_splits = live
        elif split_idx != 0:
            iters = Int32(0)
        # Splits whose window is entirely past the live length clamp to zero
        # iterations: their pipelines never turn, and they skip the epilogue so
        # the join expects exactly `live_splits` arrivals.
        active = iters > 0
        # Every page id the walk needs is already in the cycle-zero register
        # window whenever the whole live span fits one page_chunk, whatever
        # this split's window is; only longer spans reload.
        spec_win = span <= page_chunk

        # Skip-softmax (BLASST) threshold, once per CTA: S is the floor-bin
        # lookup of this row's KV length in the flattened [bin, S, ...] table
        # and lambda = S / seq_len.  An empty table, a row below the first bin
        # or S >= seq_len (lambda >= 1 would pass every page) leaves the CTA on
        # the dense path: no vote handshake, V issued as usual.  A page is
        # skipped when EVERY live head has exp2(tile_max - running_max) < lambda,
        # i.e. tile_max - running_max < thr_log2 in the log2 domain the softmax
        # already works in (the e4m3 -8 offset cancels between the two terms).
        skt_rows = cute.size(mSKT) // 2
        skt_s = Int32(0)
        for r in cutlass.range(skt_rows, unroll=1):
            if mSKT[2 * r] <= seq_len:
                skt_s = mSKT[2 * r + 1]
        skip_on = Int32(0)
        thr_log2 = Float32(-2.0**100)
        if skt_s > 0:
            if skt_s < seq_len:
                skip_on = Int32(1)
                thr_log2 = cute.math.log2(Float32(skt_s), fastmath=True) - cute.math.log2(
                    Float32(seq_len), fastmath=True)
        if cutlass.const_expr(fp8):
            # The 16-wide ratio-16 arm does not vote.
            if wide:
                skip_on = Int32(0)
        # Split seed: every split but the row's first prepends the first live page (the
        # attention sink) to its K walk -- scored for its tile max only, never accumulated,
        # no V -- so its running max starts near the global one instead of at -inf.  Cold
        # per-split chains otherwise dilute the vote (18 splits skip ~9% of the pages one
        # chain would skip 65% of).  Costs one K page per split.
        seed = Int32(0)
        if skip_on != 0:
            if p0 != first_page:
                if iters > 0:
                    seed = Int32(1)

        if cutlass.const_expr(fp8):
            # Receiver-armed cluster mailbox: split 0 raises jbar's expected-tx
            # count to the exact inbound byte total the plan implies and spends
            # the barrier's single arrival credit doing it.  Warp 2 issued the
            # init, so program order sequences init -> arm.  A writer's
            # complete_tx can only race this arm by finishing an entire page
            # pipeline before this CTA's one seq_lens load resolves --
            # structurally >2 us of slack.
            if clustered and live_splits > 1:
                if warp_idx == 2 and split_idx == 0:
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(
                            jbar.data_ptr(), (live_splits - 1) * slot_tx_bytes
                        )
            if warp_idx != 0:
                prims.barrier_cta_sync(0, thread_count=256)
        else:
            if warp_idx == 3:
                prims.barrier_cta_sync(kv_ready_barrier, thread_count=64)
            elif warp_idx != 0:
                prims.barrier_cta_sync(0, thread_count=224)

        tmem_base = tmem_holder.load()

        # ==== K TMA warp (V on warp 3 so V stalls never block K) ====
        kp_r = Int32(0)
        if warp_idx == 0:
            # One page id per lane; the walk refills the register window every
            # 32 pages, because a shfl.idx past lane 31 clamps instead of
            # wrapping.  `sh0` is where this split's window sits inside the
            # register window the lanes hold: 0 once the ids were pulled for
            # this split specifically, `p0` when the cycle-zero probe of the
            # row's first 32 pages already covers it.
            sh0 = Int32(0)
            if cutlass.const_expr(dg_guess):
                # The capacity guess validated iff it reproduced the real
                # plan's inputs exactly: same live page count, same first page.
                g_ok = Int32(0)
                if dg != 0:
                    if live_pages == MAXP:
                        if first_page == fp_g:
                            g_ok = Int32(1)
                if g_ok != 0:
                    # The guessed per-split probe was exact: the ids for pages
                    # [p0, p0 + 32) already sit one per lane.
                    kp_r = kp_gg
                else:
                    if spec_win:
                        kp_r = kp_g
                        sh0 = p0 - pbase_g
                    else:
                        li = lane
                        if li >= iters:
                            li = iters - 1
                        if li < 0:
                            li = Int32(0)
                        pt0 = mPT.iterator + (b_idx * 2) * MAXP + p0
                        kp_r = pt0[li]
            else:
                if spec_win:
                    # The speculative window issued at cycle zero COVERS this
                    # window, whatever the plan dealt this split.
                    kp_r = kp_g
                    sh0 = p0 - pbase_g
                else:
                    li = lane
                    if li >= iters:
                        li = iters - 1
                    if li < 0:
                        li = Int32(0)
                    pt0 = mPT.iterator + (b_idx * 2) * MAXP + p0
                    kp_r = pt0[li]
            # Split 0's page 0 was already issued speculatively above: its
            # pipeline indices start pre-advanced and chunk 0 of the walk
            # starts at page 1.
            k_idx = Int32(0)
            j00 = Int32(0)
            if cutlass.const_expr(spec_page0):
                if split_idx == 0:
                    k_idx = Int32(k_slices)
                    j00 = Int32(1)
            k_ph = Int32(1)
            pt = mPT.iterator + (b_idx * 2) * MAXP + p0
            if seed != 0:
                kps = mPT.iterator[(b_idx * 2) * MAXP + first_page]
                for dk in cutlass.range_constexpr(k_slices):
                    while not prims.mbarrier_try_wait_parity(
                        kbar.subview(kstg + k_idx).data_ptr(), k_ph
                    ):
                        pass
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(
                            kbar.subview(k_idx).data_ptr(), k_stage_bytes
                        )
                        if cutlass.const_expr(fp8):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sK.subview(k_idx * k_stage_elems).data_ptr(),
                                desc_k,
                                (h_k_idx * D + dk * mma_k, kps * tokens_per_page),
                                kbar.subview(k_idx).data_ptr(),
                            )
                        else:
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sK.subview(k_idx * k_stage_elems).data_ptr(),
                                desc_k,
                                (Int32(dk * mma_k), h_kv, Int32(0), kps),
                                kbar.subview(k_idx).data_ptr(),
                            )
                    k_idx += 1
                    if k_idx == kstg:
                        k_idx = Int32(0)
                        k_ph ^= 1
            # The refill is hoisted into an outer chunk loop rather than
            # predicated inside the page walk: a conditional multi-instruction
            # body inside a staged loop is predicated, not branched, so an
            # in-loop refill would issue its dependent global load on every page.
            n_chunks = (iters + Int32(page_chunk - 1)) >> 5
            for c in cutlass.range(n_chunks):
              cbase = c << 5
              if c != 0:
                li = cbase + lane
                if li >= iters:
                    li = iters - 1
                kp_r = pt[li]
              n_in_chunk = iters - cbase
              if n_in_chunk > page_chunk:
                  n_in_chunk = Int32(page_chunk)
              js = Int32(0)
              if c == 0:
                  js = j00
              for j in cutlass.range(n_in_chunk - js):
                jj = sh0 + js + j
                kp = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=kp_r,
                    offset=jj,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                    return_value_and_is_valid=False,
                )
                # A page is always 128 physically-valid tokens, so the whole
                # box is fetched even on the tail page: the rows past seq_len
                # are masked to -inf in the softmax (their P is exactly 0).
                for dk in cutlass.range_constexpr(k_slices):
                    while not prims.mbarrier_try_wait_parity(
                        kbar.subview(kstg + k_idx).data_ptr(), k_ph
                    ):
                        pass
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(
                            kbar.subview(k_idx).data_ptr(), k_stage_bytes
                        )
                        if cutlass.const_expr(fp8):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sK.subview(k_idx * k_stage_elems).data_ptr(),
                                desc_k,
                                (h_k_idx * D + dk * mma_k, kp * tokens_per_page),
                                kbar.subview(k_idx).data_ptr(),
                            )
                        else:
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sK.subview(k_idx * k_stage_elems).data_ptr(),
                                desc_k,
                                (Int32(dk * mma_k), h_kv, Int32(0), kp),
                                kbar.subview(k_idx).data_ptr(),
                            )
                    k_idx += 1
                    if k_idx == kstg:
                        k_idx = Int32(0)
                        k_ph ^= 1
                # Paced L2 lookahead (fp16 D <= 64 only): page j+1's bytes
                # start their DRAM trip while page j's boxes are still issuing,
                # paced by the same slot waits, no smem, no slot, no barrier.
                if cutlass.const_expr(lookahead):
                  if js + j + 1 < n_in_chunk:
                      jl = jj + 1
                      if jl < 32:
                          kpl = prims.shfl_sync(
                              thread_mask=0xFFFFFFFF,
                              val=kp_r,
                              offset=jl,
                              mask_and_clamp=0x1F,
                              kind=prims.Shfl.IDX,
                              return_value_and_is_valid=False,
                          )
                          if prims.elect_sync():
                              for dk in cutlass.range_constexpr(k_slices):
                                  prims.cp_async_bulk_tensor_prefetch(
                                      desc_k,
                                      [Int32(dk * mma_k), h_kv, Int32(0), kpl],
                                      [],
                                  )
        # =================================================== V producer (warp 3)
        # A V smem stage is held from the moment its TMA is ISSUED until BMM2
        # has drained it -- across the whole K-arrival -> BMM1 -> softmax ->
        # BMM2 chain -- while a K stage is released as soon as BMM1 has read
        # it.  Splitting the two producers lets warp 0 keep kstg of K in flight
        # against BMM1 alone; warp 3 re-issues a V stage the instant BMM2
        # frees it, so V only has to cover the DRAM round trip.
        elif warp_idx == 3:
            vp_r3 = Int32(0)
            sh03 = Int32(0)
            if cutlass.const_expr(dg_guess):
                g_ok3 = Int32(0)
                if dg != 0:
                    if live_pages == MAXP:
                        if first_page == fp_g3:
                            g_ok3 = Int32(1)
                if g_ok3 != 0:
                    vp_r3 = vp_gg3
                else:
                    if spec_win:
                        vp_r3 = vp_g3
                        sh03 = p0 - pbase3
                    else:
                        li3 = lane
                        if li3 >= iters:
                            li3 = iters - 1
                        if li3 < 0:
                            li3 = Int32(0)
                        vp_r3 = (mPT.iterator + (b_idx * 2) * MAXP + p0)[MAXP + li3]
            else:
                if spec_win:
                    vp_r3 = vp_g3
                    sh03 = p0 - first_page
                    # The cycle-zero probe covered pages [0, 32); a windowed
                    # row's live span starts at `first_page` instead, so reload
                    # the id window there (the plan already stalled on seq_len,
                    # and warp 0's identical windowed probe has the line in L2).
                    if window > 0:
                        li3 = first_page + lane
                        if li3 > MAXP - 1:
                            li3 = MAXP - 1
                        vp_r3 = (mPT.iterator + (b_idx * 2) * MAXP + MAXP)[li3]
                else:
                    li3 = lane
                    if li3 >= iters:
                        li3 = iters - 1
                    if li3 < 0:
                        li3 = Int32(0)
                    vp_r3 = (mPT.iterator + (b_idx * 2) * MAXP + MAXP + p0)[li3]
            v_idx = Int32(0)
            j003 = Int32(0)
            if cutlass.const_expr(spec_page0):
                if split_idx == 0:
                    v_idx = Int32(v_slices)
                    j003 = Int32(1)
            v_ph = Int32(1)
            pt3 = mPT.iterator + (b_idx * 2) * MAXP + p0
            n_chunks3 = (iters + Int32(page_chunk - 1)) >> 5
            for c in cutlass.range(n_chunks3):
              cbase = c << 5
              if c != 0:
                li3 = cbase + lane
                if li3 >= iters:
                    li3 = iters - 1
                vp_r3 = pt3[MAXP + li3]
              n_in_chunk = iters - cbase
              if n_in_chunk > page_chunk:
                  n_in_chunk = Int32(page_chunk)
              js = Int32(0)
              if c == 0:
                  js = j003
              for j in cutlass.range(n_in_chunk - js):
                jj = sh03 + js + j
                vp = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=vp_r3,
                    offset=jj,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                    return_value_and_is_valid=False,
                )
                # Skip-softmax: V for page `sp` is issued only after the softmax
                # has voted on it.  A skipped page still turns its V stage --
                # the plain arrive completes the fill barrier with no bytes, so
                # BMM2's wait, its release commit and both ring indices stay in
                # lockstep with the dense walk.  The K ring keeps DRAM busy
                # meanwhile; V only has to land before the P ring runs dry.
                sp = cbase + js + j
                v_skip = Int32(0)
                if skip_on != 0:
                    while not prims.mbarrier_try_wait_parity(
                        vtbar.subview(sp & (vote_ring - 1)).data_ptr(),
                        (sp >> vote_ring_log2) & 1,
                    ):
                        pass
                    v_skip = sVote[sp & (vote_ring - 1)]
                for sk in cutlass.range_constexpr(v_slices):
                    while not prims.mbarrier_try_wait_parity(
                        vbar.subview(vstg + v_idx).data_ptr(), v_ph
                    ):
                        pass
                    if prims.elect_sync():
                        if v_skip != 0:
                            prims.mbarrier_arrive(vbar.subview(v_idx).data_ptr())
                        else:
                            prims.mbarrier_arrive_expect_tx(
                                vbar.subview(v_idx).data_ptr(), v_tx_bytes
                            )
                            if cutlass.const_expr(fp8):
                                prims.cp_async_bulk_tensor_shared_cta_global(
                                    sV.subview(v_idx * v_stage_elems).data_ptr(),
                                    desc_k,
                                    (h_k_idx * D + sk * mma_k,
                                     vp * tokens_per_page),
                                    vbar.subview(v_idx).data_ptr(),
                                )
                            elif cutlass.const_expr(v_boxes == 2):
                                # V rides ONE box per stage where a stage carries
                                # two d-subtiles (see the 5-mode tma_v in launch).
                                prims.cp_async_bulk_tensor_shared_cta_global(
                                    sV.subview(v_idx * v_stage_elems).data_ptr(),
                                    desc_v,
                                    (Int32(0), Int32(0), h_kv, Int32(sk * 2), vp),
                                    vbar.subview(v_idx).data_ptr(),
                                )
                            else:
                                prims.cp_async_bulk_tensor_shared_cta_global(
                                    sV.subview(v_idx * v_stage_elems).data_ptr(),
                                    desc_k,
                                    (Int32(sk * v_slice_d), h_kv, Int32(0), vp),
                                    vbar.subview(v_idx).data_ptr(),
                                )
                    v_idx += 1
                    if v_idx == vstg:
                        v_idx = Int32(0)
                        v_ph ^= 1
                # Paced L2 lookahead for V (fp16 D <= 64 only), same gate as
                # the K side; the V page id rides the pre-offset second table
                # row, so the warm covers exactly the next page.  Off under
                # skip-softmax: a warmed page the vote then drops is a wasted
                # DRAM trip.
                if cutlass.const_expr(lookahead):
                  if skip_on == 0:
                   if js + j + 1 < n_in_chunk:
                      jl3 = jj + 1
                      if jl3 < 32:
                          vpl = prims.shfl_sync(
                              thread_mask=0xFFFFFFFF,
                              val=vp_r3,
                              offset=jl3,
                              mask_and_clamp=0x1F,
                              kind=prims.Shfl.IDX,
                              return_value_and_is_valid=False,
                          )
                          if prims.elect_sync():
                              for sk in cutlass.range_constexpr(v_slices):
                                  prims.cp_async_bulk_tensor_prefetch(
                                      desc_k,
                                      [Int32(sk * v_slice_d), h_kv, Int32(0), vpl],
                                      [],
                                  )
                                  if cutlass.const_expr(v_boxes == 2):
                                      prims.cp_async_bulk_tensor_prefetch(
                                          desc_k,
                                          [Int32(sk * v_slice_d + mma_k), h_kv, Int32(0), vpl],
                                          [],
                                      )
        # ======================================================== BMM1 warp
        elif warp_idx == 1:
            idesc = prims.Tcgen05InstrDesc.build(
                c_dtype=Float32,
                a_dtype=kv_t,
                b_dtype=kv_t,
                n_dim=mma_n,
                m_dim=bmm1_m,
            )
            if cutlass.const_expr(fp8):
                # Ratio-16 rows widen the ONE chain to n=16 (S slot is 16 cols
                # either way; P and BMM2 are 16-lane already), picked once here.
                idesc_w = prims.Tcgen05InstrDesc.build(
                    c_dtype=Float32,
                    a_dtype=kv_t,
                    b_dtype=kv_t,
                    n_dim=mma_n_wide,
                    m_dim=bmm1_m,
                )
                if wide:
                    idesc = idesc_w
            s_idx = Int32(0)
            s_ph = Int32(1)
            k_idx = Int32(0)
            k_ph = Int32(0)
            for s in cutlass.range(iters + seed):
                while not prims.mbarrier_try_wait_parity(
                    sbar.subview(sstg + s_idx).data_ptr(), s_ph
                ):
                    pass
                acc = prims.make_tmem_ptr(tmem_base + s_idx * s_slot, Float32)
                scale_d = False
                for dk in cutlass.range_constexpr(k_slices):
                    while not prims.mbarrier_try_wait_parity(
                        kbar.subview(k_idx).data_ptr(), k_ph
                    ):
                        pass
                    dK = cutlass.Int64(
                        prims.Tcgen05SmemDesc.build(
                            start_address=(sK.data_ptr() + k_idx * k_stage_elems).toint(Int32),
                            leading_byte_offset=16,
                            stride_byte_offset=k_sbo,
                            base_offset=cutlass.Int8(0),
                            layout=cutlass.Int8(sw_code),
                            leading_dim_mode=cutlass.Boolean(0),
                        )
                    )
                    dQ = cutlass.Int64(
                        prims.Tcgen05SmemDesc.build(
                            start_address=(sQ.data_ptr() + dk * q_tile_elems).toint(Int32),
                            leading_byte_offset=16,
                            # e4m3: rows 8..15 (heads 8..15 of a ratio-16 row)
                            # are the second 8-row core matrix, one
                            # K-stage-row stride away; n=8 never dereferences
                            # it.  fp16's 8-row tile has no second matrix.
                            stride_byte_offset=q_sbo,
                            base_offset=cutlass.Int8(0),
                            layout=cutlass.Int8(sw_code),
                            leading_dim_mode=cutlass.Boolean(0),
                        )
                    )
                    for mk in cutlass.range_constexpr(bmm1_kblks):
                        if prims.elect_sync():
                            prims.tcgen05_mma(
                                mma_kind,
                                prims.CTAGroup.CTA_1,
                                acc,
                                dK + mk * kdesc_kstep,
                                dQ + mk * kdesc_kstep,
                                idesc,
                                cutlass.Boolean(scale_d),
                            )
                        scale_d = True
                    if prims.elect_sync():
                        prims.tcgen05_commit(
                            kbar.subview(kstg + k_idx).data_ptr(),
                            group=prims.CTAGroup.CTA_1,
                        )
                    k_idx += 1
                    if k_idx == kstg:
                        k_idx = Int32(0)
                        k_ph ^= 1
                if prims.elect_sync():
                    prims.tcgen05_commit(
                        sbar.subview(s_idx).data_ptr(), group=prims.CTAGroup.CTA_1
                    )
                s_idx += 1
                if s_idx == sstg:
                    s_idx = Int32(0)
                    s_ph ^= 1

        # ======================================================== BMM2 warp
        elif warp_idx == 2:
            idesc = prims.Tcgen05InstrDesc.build(
                c_dtype=Float32,
                a_dtype=kv_t,
                b_dtype=p_t,
                a_major=1,
                b_major=1,
                n_dim=bmm2_n,
                m_dim=bmm2_m,
            )
            # One O block per d-slice of a V stage; M = 128 at every head dim
            # (rows past `d_lanes` are duplicates/garbage nothing stores).
            # No O-zero handshake: the FIRST V-MMA of the CTA overwrites the
            # accumulator (enable-input-d is a runtime predicate).  No "O is
            # rescaled" handshake back either: the softmax rescales page s-1's
            # O BEFORE it publishes P_s, and this loop waits `pbar` before
            # touching page s, so `pbar` already orders the two.
            p_idx = Int32(0)
            p_ph = Int32(0)
            v_idx = Int32(0)
            v_ph = Int32(0)
            o_init = Int32(0)
            for s in cutlass.range(iters):
                while not prims.mbarrier_try_wait_parity(
                    pbar.subview(p_idx).data_ptr(), p_ph
                ):
                    pass
                # every V stage is a full 128-token d-slice of the page, so the
                # k-chunk loop is fully constexpr; only the accumulate flag of
                # the first kept page's first chunk is a runtime predicate (the
                # first V-MMA overwrites the accumulator).  P is shared by every
                # d-slice of the page.
                accum0 = o_init != 0
                # Skip-softmax: the vote for page s was stored before P(s) was
                # published, so the pbar wait above already orders it.  A
                # skipped page's V stage completed with no bytes and its P was
                # never written: no MMA, but the same commits, so the rings
                # keep turning.
                b_skip = Int32(0)
                if skip_on != 0:
                    b_skip = sVote[s & (vote_ring - 1)]
                for sk in cutlass.range_constexpr(v_slices):
                    while not prims.mbarrier_try_wait_parity(
                        vbar.subview(v_idx).data_ptr(), v_ph
                    ):
                        pass
                    accO = prims.make_tmem_ptr(
                        tmem_base + o_col_offset + sk * o_blk_cols, Float32
                    )
                    dV = cutlass.Int64(
                        prims.Tcgen05SmemDesc.build(
                            start_address=(sV.data_ptr() + v_idx * v_stage_elems).toint(Int32),
                            leading_byte_offset=v_lbo,
                            stride_byte_offset=k_sbo,
                            base_offset=cutlass.Int8(0),
                            layout=cutlass.Int8(sw_code),
                            leading_dim_mode=cutlass.Boolean(0),
                        )
                    )
                    dP = cutlass.Int64(
                        prims.Tcgen05SmemDesc.build(
                            start_address=(
                                sP.data_ptr() + p_idx * p_stage_elems
                            ).toint(Int32),
                            leading_byte_offset=128,
                            stride_byte_offset=0,
                            base_offset=cutlass.Int8(0),
                            layout=cutlass.Int8(0),
                            leading_dim_mode=cutlass.Boolean(0),
                        )
                    )
                    if b_skip == 0:
                        for mk in cutlass.range_constexpr(bmm2_ksteps):
                            if prims.elect_sync():
                                prims.tcgen05_mma(
                                    mma_kind,
                                    prims.CTAGroup.CTA_1,
                                    accO,
                                    dV + mk * vdesc_kstep,
                                    dP + mk * pdesc_kstep,
                                    idesc,
                                    accum0 if mk == 0 else cutlass.Boolean(True),
                                )
                    if prims.elect_sync():
                        prims.tcgen05_commit(
                            vbar.subview(vstg + v_idx).data_ptr(),
                            group=prims.CTAGroup.CTA_1,
                        )
                    v_idx += 1
                    if v_idx == vstg:
                        v_idx = Int32(0)
                        v_ph ^= 1
                if b_skip == 0:
                    o_init = Int32(1)
                if prims.elect_sync():
                    prims.tcgen05_commit(
                        pbar.subview(pstg + p_idx).data_ptr(), group=prims.CTAGroup.CTA_1
                    )
                p_idx += 1
                if p_idx == pstg:
                    p_idx = Int32(0)
                    p_ph ^= 1
                if prims.elect_sync():
                    prims.tcgen05_commit(
                        obar.subview(s & (o_ring - 1)).data_ptr(),
                        group=prims.CTAGroup.CTA_1,
                    )

        # ============================================ softmax / correction warpgroup
        elif warp_idx >= 4:
            trow = wg_widx * warp_threads
            # M = 128 accumulators put D-matrix row m on TMEM lane m, so a
            # thread's token (S) / d-lane (O) is just its TMEM row.

            if cutlass.const_expr(fp8):
                # Per-head Q dequant scale, folded with k_scale and log2(e)/sqrt(D).
                qsc = cutlass.Array(Float32, softmax_heads)
                qa = sQS.data_ptr().load(count=4, alignment=16)
                for n in cutlass.range_constexpr(4):
                    qsc[n] = qa[n]
                if cutlass.const_expr(softmax_heads > 4):
                    qb = (sQS.data_ptr() + 4).load(count=4, alignment=16)
                    for n in cutlass.range_constexpr(4):
                        qsc[4 + n] = qb[n]
                qi = lane
                if qi >= softmax_heads:
                    qi = Int32(0)
                qsl = sQS[qi]

            # e4m3 ratio-16 rows zero-trip the 8-wide loop (and skip its
            # epilogue) and run the 16-wide copy below instead; narrow rows do
            # the reverse.  fp16 never widens (ratio 16 is two CTAs).
            iters_n = iters
            ep_ok = active
            if cutlass.const_expr(fp8):
                if wide:
                    iters_n = Int32(0)
                    ep_ok = cutlass.Boolean(False)

            # The head-tile arm.  GQA <= 4 rows (fp16, D <= 256) take a
            # half-width softmax: the per-page serial chain -- the warp max
            # ladder, the shfl broadcasts, the exp2s, the S / O TMEM loads and
            # the epilogue colsum tree -- all halve, and the padded P rows
            # become exact zeros instead of exp2(-m) garbage.  The branch is
            # warpgroup-uniform (gqa_ratio is CTA-uniform).  ONE body serves
            # every arm: `NH` is a Python constant per trace and, where only
            # the 8-wide arm exists, `sel` is a constant that folds away.
            for NH in arms:
                sel = cutlass.Boolean(True)
                if cutlass.const_expr(gqa_arm):
                    if cutlass.const_expr(NH == 4):
                        sel = gqa_ratio <= 4
                    else:
                        sel = gqa_ratio > 4
                if sel:
                    # No O zeroing: BMM2's first V-MMA overwrites the accumulator, so
                    # the softmax warpgroup starts straight into the S pipeline.
                    rM = cutlass.Array(Float32, NH)
                    rL = cutlass.Array(Float32, NH)
                    # rX is the EXACT running max the skip vote compares against;
                    # rM carries the headroom and is only an upper bound.
                    rX = cutlass.Array(Float32, NH)
                    for n in cutlass.range_constexpr(NH):
                        rM[n] = Float32(-1.0e30)
                        rL[n] = Float32(0.0)
                        rX[n] = Float32(-1.0e30)
                    if cutlass.const_expr(fp8):
                        # P is e4m3; the upper bytes are BMM2's MN-major 16 B padding.
                        rPq = cutlass.Array(e4m3, p_lanes, alignment=16)
                        for n in cutlass.range_constexpr(NH, p_lanes):
                            rPq[n] = Float32(0.0).to(e4m3)

                    s_idx = Int32(0)
                    s_ph = Int32(0)
                    p_idx = Int32(0)
                    p_ph = Int32(1)
                    for s in cutlass.range(iters_n + seed):
                        mx_ph = s & 1
                        sr = s - seed     # real page ordinal; -1 = the seed page
                        while not prims.mbarrier_try_wait_parity(
                            sbar.subview(s_idx).data_ptr(), s_ph
                        ):
                            pass
                        sptr = prims.make_tmem_ptr(
                            (trow << 16) | (tmem_base + s_idx * s_slot), Float32
                        )
                        ld = prims.tcgen05_ld("32x32b", sptr, num=NH)
                        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                        rS = cutlass.Array(Float32, NH)
                        for n in cutlass.range_constexpr(NH):
                            rS[n] = ld[n]
                        prims.mbarrier_arrive(sbar.subview(sstg + s_idx).data_ptr())
                        s_idx += 1
                        if s_idx == sstg:
                            s_idx = Int32(0)
                            s_ph ^= 1

                        # mask the sequence tail and the pre-window head tokens: one
                        # predicate covers both ends of the live span (the left mask
                        # only ever bites on the first touched page of a windowed row).
                        pg = p0 + sr
                        if sr < 0:
                            pg = first_page
                        tok = pg * tokens_per_page + row
                        msk = tok >= seq_len
                        if tok < lo_tok:
                            msk = cutlass.Boolean(True)
                        if msk:
                            for n in cutlass.range_constexpr(NH):
                                rS[n] = Float32(-1.0e30)

                        # per-head tile max: warp reduce, then across the 4 warps
                        mlane = Float32(0.0)
                        for n in cutlass.range_constexpr(NH):
                            v = warp_fmax(rS[n])
                            if n == lane:
                                mlane = v
                        if lane < NH:
                            sMx[mx_ph * (softmax_heads * 4) + lane * 4 + wg_widx] = mlane
                        prims.barrier_cta_sync(softmax_sync_barrier, thread_count=warpgroup_threads)
                        # The 4-warp fold is warpgroup-uniform, so lane n folds head n
                        # once and the results are broadcast with shfl.idx.
                        # e4m3 carries the max -8: P clears e4m3's subnormal floor
                        # and the offset cancels against the colsum.
                        mbase = sMx.data_ptr() + mx_ph * (softmax_heads * 4)
                        tmv = Float32(-1.0e30)
                        if lane < NH:
                            q4 = (mbase + lane * 4).load(count=4, alignment=16)
                            if cutlass.const_expr(fp8):
                                tmv = cute.arch.fmax(
                                    cute.arch.fmax(q4[0], q4[1]), cute.arch.fmax(q4[2], q4[3])
                                ) * qsl - Float32(8.0)
                            else:
                                tmv = cute.arch.fmax(
                                    cute.arch.fmax(q4[0], q4[1]), cute.arch.fmax(q4[2], q4[3])
                                ) * scale
                        # The correction factors are exactly 1.0 on every page whose
                        # max did not grow, and the branch is warpgroup-uniform -- so
                        # the MUFU exp2s (and the multiplies in the colsum update
                        # below) only run on the rare growing-max page.  FFMA(a,1,b)
                        # and FADD(a,b) round identically, so the common path is
                        # bit-exact with the unconditional form.
                        corr = cutlass.Array(Float32, NH)
                        mold_a = cutlass.Array(Float32, NH)
                        chg = Int32(0)
                        # Skip vote: the page is dropped only when no live head
                        # (n < gqa_ratio; the zero-filled padding heads score 0
                        # and would veto every page) reaches the threshold.  An
                        # unseeded split's first page compares against -inf and
                        # is kept; the seed page only advances rX / rM.
                        keep = Int32(0)
                        for n in cutlass.range_constexpr(NH):
                            tm = prims.shfl_sync(
                                thread_mask=0xFFFFFFFF,
                                val=tmv,
                                offset=n,
                                mask_and_clamp=0x1F,
                                kind=prims.Shfl.IDX,
                                return_value_and_is_valid=False,
                            )
                            xold = rX[n]
                            if n < gqa_ratio:
                                if tm - xold >= thr_log2:
                                    keep = Int32(1)
                            if tm > xold:
                                rX[n] = tm
                            mold = rM[n]
                            mnew = mold
                            if tm > mold:
                                # HEADROOM max, not the exact running max.  The softmax
                                # reference only has to be an UPPER bound on the scores
                                # seen so far -- any m >= max keeps exp2 in (0, 1] and
                                # cancels exactly against the colsum -- so when the tile
                                # max overtakes it, the running max jumps `headroom`
                                # log2 units PAST it instead of landing on it, and a page
                                # that overtakes the bound by less than that leaves it
                                # alone.  That is what the correction path costs: the
                                # wait for BMM2 to drain page s-1 before the O
                                # accumulator can be rescaled.  m is still monotone,
                                # still >= every score, still per head.
                                mnew = tm + Float32(headroom)
                                chg = Int32(1)
                            rM[n] = mnew
                            mold_a[n] = mold
                        if chg != 0:
                            for n in cutlass.range_constexpr(NH):
                                corr[n] = exp2(mold_a[n] - rM[n])

                        if sr >= 0:
                            # Publish the vote (warpgroup-uniform) the moment the tile
                            # max is known: the V producer is waiting on it.  A skipped
                            # page has tile_max < running max on every live head, so
                            # chg == 0 and O needs no correction; its P is never
                            # written and its mass never enters the colsum.
                            pskip = Int32(0)
                            if skip_on != 0:
                                if keep == 0:
                                    pskip = Int32(1)
                                if tidx == warpgroup_threads:
                                    sVote[sr & (vote_ring - 1)] = pskip
                                    prims.mbarrier_arrive(
                                        vtbar.subview(sr & (vote_ring - 1)).data_ptr()
                                    )

                            # P = exp2(S*scale - m) and the running colsum.  rP is kept at
                            # the full MMA-N width so the smem publish stays one 16 B store;
                            # the padding lanes are explicit zeros, not stale registers.
                            # e4m3's scale is per head (the Q dequant scalar), so its
                            # packed FFMA takes a pair, not a broadcast.
                            rP = cutlass.Array(Float32, mma_n)
                            if pskip == 0:
                                for n in cutlass.range_constexpr(NH, mma_n):
                                    rP[n] = Float32(0.0)
                                if cutlass.const_expr(NH >= 2):
                                    for n in cutlass.range_constexpr(0, NH, 2):
                                        if cutlass.const_expr(fp8):
                                            v2 = ffma2(
                                                (qsc[n], qsc[n + 1]),
                                                (rS[n], rS[n + 1]),
                                                (-rM[n], -rM[n + 1]),
                                            )
                                        else:
                                            v2 = ffma2(
                                                (scale, scale),
                                                (rS[n], rS[n + 1]),
                                                (-rM[n], -rM[n + 1]),
                                            )
                                        rP[n] = exp2(v2[0])
                                        rP[n + 1] = exp2(v2[1])
                                else:
                                    if cutlass.const_expr(fp8):
                                        rP[0] = exp2(rS[0] * qsc[0] - rM[0])
                                    else:
                                        rP[0] = exp2(rS[0] * scale - rM[0])
                                if chg != 0:
                                    for n in cutlass.range_constexpr(NH):
                                        rL[n] = rL[n] * corr[n] + rP[n]
                                else:
                                    for n in cutlass.range_constexpr(NH):
                                        rL[n] = rL[n] + rP[n]

                                # Rescale the O accumulator produced by the previous tiles.
                                # The running max is warpgroup-uniform, so this branch is
                                # uniform, and on the (rare) growing-max page it waits for
                                # BMM2's page s-1 drain on the commit ring before the TMEM
                                # round trip.  Every other page skips BOTH the wait and the
                                # round trip, so the softmax free-runs at pbar depth: P(s) is
                                # published only after the correction's STORE retired, which
                                # already orders it before BMM2's page-s MMAs.
                                if sr > 0:
                                    if chg != 0:
                                        while not prims.mbarrier_try_wait_parity(
                                            obar.subview((sr - 1) & (o_ring - 1)).data_ptr(),
                                            ((sr - 1) >> o_ring_log2) & 1,
                                        ):
                                            pass
                                        cv = corr.data_ptr().load(count=NH, alignment=4)
                                        # The O blocks live in disjoint TMEM columns, so their
                                        # loads are independent and only the stores need the
                                        # data back: one wait for the whole correction.
                                        optrs = []
                                        los = []
                                        for blk in cutlass.range_constexpr(o_blocks):
                                            optr = prims.make_tmem_ptr(
                                                (trow << 16)
                                                | (tmem_base + o_col_offset + blk * o_blk_cols),
                                                Float32,
                                            )
                                            optrs.append(optr)
                                            los.append(
                                                prims.tcgen05_ld("32x32b", optr, num=NH)
                                            )
                                        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                                        for blk in cutlass.range_constexpr(o_blocks):
                                            prims.tcgen05_st("32x32b", optrs[blk], los[blk] * cv)
                                        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)

                            # publish P to smem for BMM2 (a skipped page only turns the
                            # ring: BMM2 reads the vote and issues no MMA on it)
                            while not prims.mbarrier_try_wait_parity(
                                pbar.subview(pstg + p_idx).data_ptr(), p_ph
                            ):
                                pass
                            if pskip == 0:
                                if cutlass.const_expr(fp8):
                                    for n in cutlass.range_constexpr(NH):
                                        rPq[n] = rP[n].to(e4m3)
                                    sP.subview(p_idx * p_stage_elems + row * p_row).data_ptr().store(
                                        rPq.data_ptr().load(count=p_lanes, alignment=16),
                                        alignment=16,
                                    )
                                else:
                                    sP.subview(p_idx * p_stage_elems + row * p_row).data_ptr().store(
                                        rP.data_ptr().load(count=mma_n, alignment=4).to(Float16),
                                        alignment=16,
                                    )
                                cute.arch.fence_view_async_shared()
                            prims.mbarrier_arrive(pbar.subview(p_idx).data_ptr())
                            p_idx += 1
                            if p_idx == pstg:
                                p_idx = Int32(0)
                                p_ph ^= 1

                    # Epilogue
                    # The colsum only depends on rL, which is final once the last
                    # softmax tile is done -- run it (and the m/l partial stores) while
                    # BMM2 is still draining the last V tile, so the TMEM read of O is
                    # the only thing left on the critical path.  The whole epilogue
                    # sits under `active`: a split whose window is past the live length
                    # has nothing to publish.
                    if ep_ok:
                        for off in cutlass.range_constexpr(5):
                            for n in cutlass.range_constexpr(NH):
                                rL[n] = Float32(
                                    rL[n]
                                    + prims.shfl_sync(
                                        thread_mask=0xFFFFFFFF,
                                        val=rL[n],
                                        offset=(16 >> off),
                                        mask_and_clamp=0x1F,
                                        kind=prims.Shfl.BFLY,
                                        return_value_and_is_valid=False,
                                    )
                                )
                        llane = Float32(0.0)
                        for n in cutlass.range_constexpr(NH):
                            if n == lane:
                                llane = rL[n]
                        if lane < NH:
                            sLx[lane * 4 + wg_widx] = llane
                        prims.barrier_cta_sync(softmax_sync_barrier, thread_count=warpgroup_threads)
                        inv = cutlass.Array(Float32, NH)
                        tots = cutlass.Array(Float32, NH)
                        for n in cutlass.range_constexpr(NH):
                            q4 = (sLx.data_ptr() + n * 4).load(count=4, alignment=16)
                            tots[n] = (q4[0] + q4[1]) + (q4[2] + q4[3])
                            # 1e-2 tolerance: MUFU.RCP (2^-23 relative) instead of the
                            # ~15-instruction IEEE divide sequence.  e4m3's `v_scale`
                            # rides in HERE: every path passes this once.
                            if cutlass.const_expr(fp8):
                                inv[n] = cute.arch.rcp_approx(tots[n]) * v_scale
                            else:
                                inv[n] = cute.arch.rcp_approx(tots[n])
                            # A seeded split may have skipped every page: publish a zero
                            # partial with m = -inf so the join weighs it out.
                            if tots[n] == Float32(0.0):
                                inv[n] = Float32(0.0)
                                rM[n] = Float32(-1.0e30)
                        # Attention sink, single-split path: the per-head logit joins the
                        # row max and the denominator and contributes no V --
                        # m' = max(m, s), l' = l*exp2(m-m') + exp2(s-m'), O' = O*exp2(m-m'),
                        # all folded into `inv` so O itself is untouched.  Sinks are
                        # natural-log logits while rM/tots live in the log2 domain,
                        # hence the log2_e conversion (e4m3's -8/headroom offsets cancel:
                        # tots and the sink term share the pivot).  Multi-split rows
                        # fold the sink exactly once, in the join, never here.
                        if live_splits == 1:
                            if has_sink == 1:
                                for n in cutlass.range_constexpr(NH):
                                    if n < gqa_ratio:
                                        s2 = mSNK.iterator[h_k_idx * gqa_ratio + n] * Float32(log2_e)
                                        mo = rM[n]
                                        mn = cute.arch.fmax(mo, s2)
                                        cf = exp2(mo - mn)
                                        if cutlass.const_expr(fp8):
                                            inv[n] = cf * cute.arch.rcp_approx(
                                                tots[n] * cf + exp2(s2 - mn)
                                            ) * v_scale
                                        else:
                                            inv[n] = cf * cute.arch.rcp_approx(
                                                tots[n] * cf + exp2(s2 - mn)
                                            )

                        # Workspace layout is [B, HQ, splits, D] partial-O then the
                        # per-split running max and colsum; the offsets are runtime
                        # products.  HQ is padded by one head tile so a GQA ratio below
                        # the tile can read the full tile back in the join without
                        # running off the end.
                        hq_pad = b_dim * HQ + pad_n
                        max_off = hq_pad * SPLITS * D
                        col_off = max_off + hq_pad * SPLITS
                        hq0 = b_idx * HQ + h_k_idx * gqa_ratio

                        # Global-join rows publish the per-split (m, l) pair ahead of
                        # the terminal drain; e4m3's clustered rows ship it through the
                        # mailbox instead.
                        pub_ml = cutlass.Boolean(True)
                        if cutlass.const_expr(fp8):
                            pub_ml = SPLITS > max_cluster_d
                        if live_splits != 1:
                            if pub_ml:
                                if row == 0:
                                    mbase = mWS.iterator + max_off + hq0 * SPLITS + split_idx
                                    lbase = mWS.iterator + col_off + hq0 * SPLITS + split_idx
                                    for n in cutlass.range_constexpr(NH):
                                        if n < gqa_ratio:
                                            (lbase + n * SPLITS).store(tots[n])
                                            (mbase + n * SPLITS).store(rM[n])

                        # BMM2 commits page s into ring slot s & (o_ring - 1): the
                        # terminal drain wait targets the last page's slot and parity.
                        while not prims.mbarrier_try_wait_parity(
                            obar.subview((iters - 1) & (o_ring - 1)).data_ptr(),
                            ((iters - 1) >> o_ring_log2) & 1,
                        ):
                            pass
                        rOs = []
                        for blk in cutlass.range_constexpr(o_blocks):
                            rOs.append(
                                prims.tcgen05_ld(
                                    "32x32b",
                                    prims.make_tmem_ptr(
                                        (trow << 16)
                                        | (tmem_base + o_col_offset + blk * o_blk_cols),
                                        Float32,
                                    ),
                                    num=NH,
                                )
                            )
                        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)

                        if live_splits == 1:
                            # Only one live split for this (b, h_k): the reduction
                            # collapses into the fused epilogue.  Every row whose KV
                            # fits one CTA window takes this path.  O TMEM rows past
                            # `d_lanes` (D < 128) are duplicates/garbage nothing emits.
                            for blk in cutlass.range_constexpr(o_blocks):
                                rO = rOs[blk]
                                obase = mO.iterator + hq0 * D + blk * o_blk_d + row
                                for n in cutlass.range_constexpr(NH):
                                    if n < gqa_ratio:
                                        if cutlass.const_expr(d_lanes == 128):
                                            (obase + n * D).store((rO[n] * inv[n]).to(Float16))
                                        else:
                                            if row < d_lanes:
                                                (obase + n * D).store((rO[n] * inv[n]).to(Float16))
                        else:
                            if cutlass.const_expr(fp8):
                                if clustered:
                                    # The partial is staged NORMALISED (O_s / l_s) and in
                                    # fp16 into the LOCAL mailbox slot -- O_s / l_s is a
                                    # convex combination of V rows, so fp16's 5e-4 relative
                                    # error sits two orders under the 1e-2 tolerance
                                    # whatever the split length.  Writers then push the
                                    # slot (and the (m, l) pair) into split 0's instance of
                                    # the SAME smem offset over the cluster interconnect;
                                    # split 0 just leaves its own slot in place.
                                    jb = split_idx * slot_o_elems
                                    for blk in cutlass.range_constexpr(o_blocks):
                                        rO = rOs[blk]
                                        for n in cutlass.range_constexpr(NH):
                                            if cutlass.const_expr(d_lanes == 128):
                                                sJO[jb + n * D + blk * o_blk_d + row] = (
                                                    rO[n] * inv[n]
                                                ).to(Float16)
                                            else:
                                                if row < d_lanes:
                                                    sJO[jb + n * D + blk * o_blk_d + row] = (
                                                        rO[n] * inv[n]
                                                    ).to(Float16)
                                    if wg_widx == 0:
                                        mlb = split_idx * slot_ml_elems
                                        for n in cutlass.range_constexpr(NH):
                                            if lane == n:
                                                sJML[mlb + n] = rM[n]
                                                sJML[mlb + mma_n + n] = tots[n]
                                    # Publish the staging stores to the async proxy (the
                                    # bulk engine reads them), then one thread ships both.
                                    cute.arch.fence_view_async_shared()
                                    prims.barrier_cta_sync(softmax_sync_barrier, thread_count=warpgroup_threads)
                                    if split_idx != 0:
                                        if row == 0:
                                            pb = prims.mapa(jbar.data_ptr(), 0)
                                            prims.cp_async_bulk_shared_cluster_shared_cta(
                                                prims.mapa(sJO.data_ptr() + jb, 0),
                                                sJO.data_ptr() + jb,
                                                pb,
                                                slot_o_bytes,
                                            )
                                            prims.cp_async_bulk_shared_cluster_shared_cta(
                                                prims.mapa(sJML.data_ptr() + split_idx * slot_ml_elems, 0),
                                                sJML.data_ptr() + split_idx * slot_ml_elems,
                                                pb,
                                                slot_ml_bytes,
                                            )
                                else:
                                    # Global-join rows: the partial goes to the fp16
                                    # workspace and every live split folds a slice.
                                    for blk in cutlass.range_constexpr(o_blocks):
                                        rO = rOs[blk]
                                        obase = (
                                            mWH.iterator
                                            + (hq0 * SPLITS + split_idx) * D + blk * o_blk_d + row
                                        )
                                        for n in cutlass.range_constexpr(NH):
                                            if n < gqa_ratio:
                                                if cutlass.const_expr(d_lanes == 128):
                                                    (obase + n * (SPLITS * D)).store(
                                                        (rO[n] * inv[n]).to(Float16)
                                                    )
                                                else:
                                                    if row < d_lanes:
                                                        (obase + n * (SPLITS * D)).store(
                                                            (rO[n] * inv[n]).to(Float16)
                                                        )
                                    prims.barrier_cta_sync(softmax_sync_barrier, thread_count=warpgroup_threads)
                                    # EVERY live split folds a SLICE of the (head, d)
                                    # space, so the whole group rendezvouses here on an
                                    # EPOCH FLAG LINE, one int32 word per split: a
                                    # writer publishes with `red.global.max.release.gpu`
                                    # (fire-and-forget) and the whole warp acquire-polls
                                    # the line until every live split shows this call's
                                    # epoch.  The token lives in DEVICE memory (a
                                    # host-side counter would be frozen into a CUDA
                                    # graph): every split reads the group's generation
                                    # before it publishes, and split 0 bumps it once all
                                    # flags are in, so stale flags read as "not yet".
                                    flagb = mSEM.iterator + coord_hb * sem_words
                                    epoch = prims.load_ext(
                                        flagb + gen_word,
                                        dtype=Int32,
                                        order=prims.MemOrder.ACQUIRE,
                                        scope=prims.MemScope.GPU,
                                    ) + Int32(1)
                                    if row == 0:
                                        prims.red(
                                            prims.ReductionOp.MAX,
                                            prims.ReductionType.S32,
                                            flagb + split_idx,
                                            epoch,
                                            mem_order=prims.MemOrder.RELEASE,
                                            mem_scope=prims.MemScope.GPU,
                                        )
                                    if wg_widx == 0:
                                        allin = cutlass.Boolean(False)
                                        while not allin:
                                            mine = cutlass.Boolean(True)
                                            for r in cutlass.range_constexpr(sreg):
                                                fi = Int32(r * warp_threads) + lane
                                                if fi < live_splits:
                                                    fv = prims.load_ext(
                                                        flagb + fi,
                                                        dtype=Int32,
                                                        order=prims.MemOrder.ACQUIRE,
                                                        scope=prims.MemScope.GPU,
                                                    )
                                                    if fv != epoch:
                                                        mine = cutlass.Boolean(False)
                                            allin = prims.vote_sync(
                                                0xFFFFFFFF, mine, prims.VoteSync.ALL
                                            )
                                        # Every split read the generation before it
                                        # published its flag, and all flags are in:
                                        # split 0 can advance the generation now.
                                        if split_idx == 0:
                                            if lane == 0:
                                                (flagb + gen_word).store(epoch)
                            else:
                                # The partial is published NORMALISED (O_s / l_s) and in
                                # fp16: bounded by max|V| however long the split is, so
                                # fp16's 5e-4 relative error sits two orders under the
                                # tolerance and the reducer gathers half the bytes.
                                for blk in cutlass.range_constexpr(o_blocks):
                                    rO = rOs[blk]
                                    obase = mWH.iterator + (hq0 * SPLITS + split_idx) * D + blk * o_blk_d + row
                                    for n in cutlass.range_constexpr(NH):
                                        if n < gqa_ratio:
                                            if cutlass.const_expr(d_lanes == 128):
                                                (obase + n * (SPLITS * D)).store(
                                                    (rO[n] * inv[n]).to(Float16)
                                                )
                                            else:
                                                if row < d_lanes:
                                                    (obase + n * (SPLITS * D)).store(
                                                        (rO[n] * inv[n]).to(Float16)
                                                    )
                                # split-K join: the last CTA of this (b, h_k) group
                                # combines every partial and writes the final fp16
                                # output.  The fold itself runs on ALL 256 threads
                                # below; this warpgroup only publishes the arrival count.
                                prims.barrier_cta_sync(softmax_sync_barrier, thread_count=warpgroup_threads)
                                if row == 0:
                                    old = prims.atomicrmw(
                                        prims.AtomicOp.ADD,
                                        mSEM.iterator + coord_hb,
                                        Int32(1),
                                        mem_order=prims.MemOrder.ACQ_REL,
                                        syncscope=prims.MemScope.GPU,
                                    )
                                    # The last arrival hands the slot back at zero, so
                                    # the counter never leaves [0, live_splits).
                                    sFlag[0] = old

            # ============ e4m3 RATIO-16: the 16-wide copy of the loop above =====
            # One n=16 chain: an S slot carries all 16 heads, one num=16 load
            # drains the page, P is the SAME 16 B / token store (all lanes
            # real now), and the per-head reductions ride warp_fold16 (17
            # shfl) instead of sixteen 5-step warp-max trees.  Narrow rows
            # never enter; wide rows never ran the loop above.
            if cutlass.const_expr(fp8):
              if wide:
                tok = row
                qsc16 = cutlass.Array(Float32, mma_n_wide)
                for nq in cutlass.range_constexpr(4):
                    qw = (sQS.data_ptr() + nq * 4).load(count=4, alignment=16)
                    for n in cutlass.range_constexpr(4):
                        qsc16[nq * 4 + n] = qw[n]
                qi16 = lane
                if qi16 >= mma_n_wide:
                    qi16 = Int32(0)
                qsl16 = sQS[qi16]
                rM16 = cutlass.Array(Float32, mma_n_wide)
                rL16 = cutlass.Array(Float32, mma_n_wide)
                for n in cutlass.range_constexpr(mma_n_wide):
                    rM16[n] = Float32(-1.0e30)
                    rL16[n] = Float32(0.0)
                rPq16 = cutlass.Array(e4m3, p_lanes, alignment=16)
                si16 = Int32(0)
                sph16 = Int32(0)
                pi16 = Int32(0)
                pph16 = Int32(1)
                for s in cutlass.range(iters):
                    mx_ph = s & 1
                    while not prims.mbarrier_try_wait_parity(
                        sbar.subview(si16).data_ptr(), sph16
                    ):
                        pass
                    sptr16 = prims.make_tmem_ptr(
                        (trow << 16) | (tmem_base + si16 * s_slot), Float32
                    )
                    ld16 = prims.tcgen05_ld("32x32b", sptr16, num=mma_n_wide)
                    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                    rS16 = cutlass.Array(Float32, mma_n_wide)
                    for n in cutlass.range_constexpr(mma_n_wide):
                        rS16[n] = ld16[n]
                    prims.mbarrier_arrive(sbar.subview(sstg + si16).data_ptr())
                    si16 += 1
                    if si16 == sstg:
                        si16 = Int32(0)
                        sph16 ^= 1

                    gtok = (p0 + s) * tokens_per_page + tok
                    msk = gtok >= seq_len
                    if gtok < lo_tok:
                        msk = cutlass.Boolean(True)
                    if msk:
                        for n in cutlass.range_constexpr(mma_n_wide):
                            rS16[n] = Float32(-1.0e30)

                    # per-head tile max: halving butterfly (head h lands on
                    # lanes 2h / 2h+1), then folded across the 4 warps.
                    mt16 = warp_fold16(rS16, lane, True)
                    if (lane & 1) == 0:
                        sMx[mx_ph * (mma_n_wide * 4) + ((lane >> 1) & 15) * 4 + wg_widx] = mt16
                    prims.barrier_cta_sync(softmax_sync_barrier, thread_count=warpgroup_threads)
                    mb16 = sMx.data_ptr() + mx_ph * (mma_n_wide * 4)
                    tmv16 = Float32(-1.0e30)
                    if lane < mma_n_wide:
                        q4w = (mb16 + lane * 4).load(count=4, alignment=16)
                        tmv16 = cute.arch.fmax(
                            cute.arch.fmax(q4w[0], q4w[1]), cute.arch.fmax(q4w[2], q4w[3])
                        ) * qsl16 - Float32(8.0)
                    corr16 = cutlass.Array(Float32, mma_n_wide)
                    mold16 = cutlass.Array(Float32, mma_n_wide)
                    chg16 = Int32(0)
                    for n in cutlass.range_constexpr(mma_n_wide):
                        tm16 = prims.shfl_sync(
                            thread_mask=0xFFFFFFFF,
                            val=tmv16,
                            offset=n,
                            mask_and_clamp=0x1F,
                            kind=prims.Shfl.IDX,
                            return_value_and_is_valid=False,
                        )
                        mold = rM16[n]
                        mnew = mold
                        if tm16 > mold:
                            mnew = tm16 + Float32(max_headroom)
                            chg16 = Int32(1)
                        rM16[n] = mnew
                        mold16[n] = mold
                    if chg16 != 0:
                        for n in cutlass.range_constexpr(mma_n_wide):
                            corr16[n] = exp2(mold16[n] - rM16[n])

                    rP16 = cutlass.Array(Float32, mma_n_wide)
                    for n in cutlass.range_constexpr(0, mma_n_wide, 2):
                        v2w = ffma2(
                            (qsc16[n], qsc16[n + 1]),
                            (rS16[n], rS16[n + 1]),
                            (-rM16[n], -rM16[n + 1]),
                        )
                        rP16[n] = exp2(v2w[0])
                        rP16[n + 1] = exp2(v2w[1])
                    if chg16 != 0:
                        for n in cutlass.range_constexpr(mma_n_wide):
                            rL16[n] = rL16[n] * corr16[n] + rP16[n]
                    else:
                        for n in cutlass.range_constexpr(mma_n_wide):
                            rL16[n] = rL16[n] + rP16[n]

                    # Rescale the previous page's O (rare, headroom-gated).
                    if s > 0:
                        if chg16 != 0:
                            sm1w = s - 1
                            while not prims.mbarrier_try_wait_parity(
                                obar.subview(sm1w & (o_ring - 1)).data_ptr(),
                                (sm1w >> o_ring_log2) & 1,
                            ):
                                pass
                            los16 = []
                            for blk in cutlass.range_constexpr(o_blocks):
                                optr16 = prims.make_tmem_ptr(
                                    (trow << 16)
                                    | (tmem_base + o_col_offset + blk * o_blk_cols),
                                    Float32,
                                )
                                los16.append(
                                    prims.tcgen05_ld("32x32b", optr16, num=mma_n_wide)
                                )
                            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                            cv16 = corr16.data_ptr().load(count=mma_n_wide, alignment=4)
                            for blk in cutlass.range_constexpr(o_blocks):
                                optr16 = prims.make_tmem_ptr(
                                    (trow << 16)
                                    | (tmem_base + o_col_offset + blk * o_blk_cols),
                                    Float32,
                                )
                                prims.tcgen05_st("32x32b", optr16, los16[blk] * cv16)
                            prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)

                    # publish P to smem for BMM2 (same 16 B/token as narrow)
                    while not prims.mbarrier_try_wait_parity(
                        pbar.subview(pstg + pi16).data_ptr(), pph16
                    ):
                        pass
                    for n in cutlass.range_constexpr(mma_n_wide):
                        rPq16[n] = rP16[n].to(e4m3)
                    sP.subview(pi16 * p_stage_elems + tok * p_lanes).data_ptr().store(
                        rPq16.data_ptr().load(count=p_lanes, alignment=16),
                        alignment=16,
                    )
                    cute.arch.fence_view_async_shared()
                    prims.mbarrier_arrive(pbar.subview(pi16).data_ptr())
                    pi16 += 1
                    if pi16 == pstg:
                        pi16 = Int32(0)
                        pph16 ^= 1

                # 16-wide epilogue
                if active:
                    lt16 = warp_fold16(rL16, lane, False)
                    if (lane & 1) == 0:
                        sLx[((lane >> 1) & 15) * 4 + wg_widx] = lt16
                    prims.barrier_cta_sync(softmax_sync_barrier, thread_count=warpgroup_threads)
                    inv16 = cutlass.Array(Float32, mma_n_wide)
                    tots16 = cutlass.Array(Float32, mma_n_wide)
                    for n in cutlass.range_constexpr(mma_n_wide):
                        q4l = (sLx.data_ptr() + n * 4).load(count=4, alignment=16)
                        tots16[n] = (q4l[0] + q4l[1]) + (q4l[2] + q4l[3])
                        inv16[n] = cute.arch.rcp_approx(tots16[n]) * v_scale

                    # Attention sink, single-split path (16-wide copy).
                    if live_splits == 1:
                        if has_sink == 1:
                            for n in cutlass.range_constexpr(mma_n_wide):
                                if n < gqa_ratio:
                                    s2w = mSNK.iterator[h_k_idx * gqa_ratio + n] * Float32(log2_e)
                                    mo16 = rM16[n]
                                    mn16 = cute.arch.fmax(mo16, s2w)
                                    cf16 = exp2(mo16 - mn16)
                                    inv16[n] = cf16 * cute.arch.rcp_approx(
                                        tots16[n] * cf16 + exp2(s2w - mn16)
                                    ) * v_scale

                    hq0w = b_idx * HQ + h_k_idx * gqa_ratio
                    hq_padw = b_dim * HQ + mma_n_wide
                    max_offw = hq_padw * SPLITS * D
                    col_offw = max_offw + hq_padw * SPLITS

                    # Wide multi-split rows ALWAYS join globally (never
                    # clustered): publish (m, l), spread over 16 lanes.
                    if live_splits != 1:
                        if row < mma_n_wide:
                            mvp = Float32(0.0)
                            tvp = Float32(0.0)
                            for n in cutlass.range_constexpr(mma_n_wide):
                                if n == row:
                                    mvp = rM16[n]
                                    tvp = tots16[n]
                            if row < gqa_ratio:
                                (mWS.iterator + max_offw + (hq0w + row) * SPLITS
                                 + split_idx).store(mvp)
                                (mWS.iterator + col_offw + (hq0w + row) * SPLITS
                                 + split_idx).store(tvp)

                    it1w = iters - 1
                    while not prims.mbarrier_try_wait_parity(
                        obar.subview(it1w & (o_ring - 1)).data_ptr(),
                        (it1w >> o_ring_log2) & 1,
                    ):
                        pass
                    rOs16 = []
                    for blk in cutlass.range_constexpr(o_blocks):
                        optrw = prims.make_tmem_ptr(
                            (trow << 16) | (tmem_base + o_col_offset + blk * o_blk_cols),
                            Float32,
                        )
                        rOs16.append(prims.tcgen05_ld("32x32b", optrw, num=mma_n_wide))
                    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)

                    if live_splits == 1:
                        for blk in cutlass.range_constexpr(o_blocks):
                            rOw = rOs16[blk]
                            obw = mO.iterator + hq0w * D + blk * mma_k + tok
                            for n in cutlass.range_constexpr(mma_n_wide):
                                if n < gqa_ratio:
                                    (obw + n * D).store((rOw[n] * inv16[n]).to(Float16))
                    else:
                        for blk in cutlass.range_constexpr(o_blocks):
                            rOw = rOs16[blk]
                            obw = (
                                mWH.iterator
                                + (hq0w * SPLITS + split_idx) * D + blk * mma_k + tok
                            )
                            for n in cutlass.range_constexpr(mma_n_wide):
                                if n < gqa_ratio:
                                    (obw + n * (SPLITS * D)).store(
                                        (rOw[n] * inv16[n]).to(Float16)
                                    )
                        prims.barrier_cta_sync(softmax_sync_barrier, thread_count=warpgroup_threads)
                        # epoch rendezvous: 16-wide copy of the narrow one.
                        flagw = mSEM.iterator + coord_hb * sem_words
                        epochw = prims.load_ext(
                            flagw + gen_word,
                            dtype=Int32,
                            order=prims.MemOrder.ACQUIRE,
                            scope=prims.MemScope.GPU,
                        ) + Int32(1)
                        if row == 0:
                            prims.red(
                                prims.ReductionOp.MAX,
                                prims.ReductionType.S32,
                                flagw + split_idx,
                                epochw,
                                mem_order=prims.MemOrder.RELEASE,
                                mem_scope=prims.MemScope.GPU,
                            )
                        if wg_widx == 0:
                            allinw = cutlass.Boolean(False)
                            while not allinw:
                                minew = cutlass.Boolean(True)
                                for r in cutlass.range_constexpr(sreg):
                                    fiw = Int32(r * warp_threads) + lane
                                    if fiw < live_splits:
                                        fvw = prims.load_ext(
                                            flagw + fiw,
                                            dtype=Int32,
                                            order=prims.MemOrder.ACQUIRE,
                                            scope=prims.MemScope.GPU,
                                        )
                                        if fvw != epochw:
                                            minew = cutlass.Boolean(False)
                                allinw = prims.vote_sync(
                                    0xFFFFFFFF, minew, prims.VoteSync.ALL
                                )
                            if split_idx == 0:
                                if lane == 0:
                                    (flagw + gen_word).store(epochw)

            # TMEM is handed back once every softmax warp's accumulator read
            # retired: a lone warp-4 dealloc can free the columns while warps
            # 5-7 are still loading O from them.
            prims.barrier_cta_sync(softmax_sync_barrier, thread_count=warpgroup_threads)
            if warp_idx == 4:
                prims.tcgen05_dealloc(
                    prims.make_tmem_ptr(tmem_holder.load(), cutlass.Int8),
                    Int32(tmem_cols),
                )

        # ======================================= split-K join (whole CTA)
        # `active` and `live_splits` are CTA-uniform, so this is a uniform
        # branch: dead splits and the single-split rows never enter it.  The
        # mechanism is per dtype (see the module docstring): e4m3 folds in
        # split 0's cluster mailbox or split-parallel off a device-memory
        # generation-word rendezvous; fp16 elects the last (or the first
        # `o_blocks`) arrivals on a GPU-scope counter and folds tiered batches.
        if cutlass.const_expr(fp8):
            # ======================================= split-K fold (folder CTA)
            # `active` and `live_splits` are CTA-uniform, so this is a uniform
            # branch: dead splits and the single-split rows never enter it.
            #
            # Split 0 is the folder.  Its warps 0-3 idle after the producer loops,
            # so they sit on the mailbox barrier while the softmax warpgroup is
            # still staging slot 0; the CTA barrier then orders slot 0's stores for
            # everyone and the whole fold runs out of LOCAL shared memory -- no
            # semaphore round trip, no far-L2 gather.  Warp w owns head w and lane
            # l owns d-quad l, exactly the old fold's dealing.
            if active:
              if live_splits != 1:
                if clustered:
                  if split_idx == 0:
                    while not prims.mbarrier_try_wait_parity(jbar.data_ptr(), 0):
                        pass
                    prims.barrier_cta_sync(join_barrier, thread_count=256)
                    nh = warp_idx
                    # Warps above the runtime GQA ratio own no real head.
                    if nh < gqa_ratio:
                     hq0 = b_idx * HQ + h_k_idx * gqa_ratio
                     mv = Float32(-1.0e30)
                     lv = Float32(0.0)
                     if lane < live_splits:
                         mv = sJML[lane * slot_ml_elems + nh]
                         lv = sJML[lane * slot_ml_elems + mma_n + nh]
                     acc = cutlass.Array(Float32, 4 * o_blocks)
                     for e in cutlass.range_constexpr(4 * o_blocks):
                         acc[e] = Float32(0.0)

                     # Partials are normalised, so a split's fold weight is
                     # `l_s * exp2(m_s - M)`; the denominator is that same sum.
                     # Lanes past the live split count carry -inf, hence exactly 0.
                     # Attention sink, mailbox path: folded HERE, exactly once,
                     # into the global max and the denominator (no V contribution).
                     gmn = warp_fmax(mv)
                     snk = Float32(0.0)
                     if has_sink == 1:
                         s2c = mSNK.iterator[h_k_idx * gqa_ratio + nh] * Float32(log2_e)
                         gmn = cute.arch.fmax(gmn, s2c)
                         snk = exp2(s2c - gmn)
                     fv = exp2(mv - gmn) * lv
                     dv = fv
                     for off in cutlass.range_constexpr(5):
                         dv = Float32(
                             dv
                             + prims.shfl_sync(
                                 thread_mask=0xFFFFFFFF,
                                 val=dv,
                                 offset=(16 >> off),
                                 mask_and_clamp=0x1F,
                                 kind=prims.Shfl.BFLY,
                                 return_value_and_is_valid=False,
                             )
                         )
                     dv = dv + snk

                     # D < 128: only the first d_quads lanes own a real 16 B
                     # granule of the head; the rest clamp their addresses and
                     # never store.
                     lq = lane
                     okq = cutlass.Boolean(True)
                     if cutlass.const_expr(D < mma_k):
                         okq = lane < d_quads
                         if lq >= d_quads:
                             lq = Int32(0)

                     # Dead slots hold stale shared memory (possibly NaN bit
                     # patterns), so the accumulate is predicated rather than
                     # weight-zeroed; the loads are shared-memory-fast either way.
                     for t in cutlass.range_constexpr(max_cluster_d):
                         f = prims.shfl_sync(
                             thread_mask=0xFFFFFFFF,
                             val=fv,
                             offset=t,
                             mask_and_clamp=0x1F,
                             kind=prims.Shfl.IDX,
                             return_value_and_is_valid=False,
                         )
                         if t < live_splits:
                             for blk in cutlass.range_constexpr(o_blocks):
                                 for e in cutlass.range_constexpr(4):
                                     acc[blk * 4 + e] = acc[blk * 4 + e] + f * sJO[
                                         t * slot_o_elems + nh * D + blk * mma_k
                                         + lq * 4 + e
                                     ].to(Float32)

                     rinv = cute.arch.rcp_approx(dv)
                     ob = mO.iterator + (hq0 + nh) * D + lq * 4
                     for blk in cutlass.range_constexpr(o_blocks):
                         for e in cutlass.range_constexpr(4):
                             av = acc[blk * 4 + e] * rinv
                             if okq:
                                 (ob + blk * mma_k + e).store(av.to(Float16))
                else:
                  # Global RENDEZVOUS join (SPLITS > 8 rows)
                  # Every live split folds a contiguous run of `D/4` d-quads of
                  # every head it owns, so one CTA's gather is `gqa * D` fp16 NO
                  # MATTER how wide the plan is.  The old form elected the last
                  # arrival as the sole folder and pulled `live_splits * gqa * D`
                  # fp16 through that one CTA's L1 -- a tail that grew with the
                  # split count and dominated every wide row (NCU: the folder's
                  # SM stayed active ~14 k cycles past the writers on the 32-split
                  # d512 row).  Warps above the GQA ratio own no head and just
                  # rendezvous.
                  prims.barrier_cta_sync(join_barrier, thread_count=256)
                  # Ratio-16: warp w folds head w, then head w + 8 (16 heads over
                  # the 8 warps); narrow rows predicate the hp == 1 pass off.
                  for hp in cutlass.range_constexpr(2):
                   nh = warp_idx + hp * mma_n
                   if nh < gqa_ratio:
                     hq_pad = b_dim * HQ + mma_n_wide
                     max_off = hq_pad * SPLITS * D
                     col_off = max_off + hq_pad * SPLITS
                     hq0 = b_idx * HQ + h_k_idx * gqa_ratio
                     jgb = mWH.iterator + (hq0 + nh) * (SPLITS * D)
                     mp = mWS.iterator + max_off + (hq0 + nh) * SPLITS
                     lp = mWS.iterator + col_off + (hq0 + nh) * SPLITS
                     # `sreg` registers of (m, l) per lane cover `32 * sreg`
                     # splits, so the plan is no longer pinned to "one split per
                     # lane of the fold warp": register r holds splits
                     # [32r, 32r + 32) and the gather picks its factor out of the
                     # register the 32-block belongs to.
                     mvr = []
                     lvr = []
                     for r in cutlass.range_constexpr(sreg):
                         mv_r = Float32(-1.0e30)
                         lv_r = Float32(0.0)
                         idx = Int32(r * warp_threads) + lane
                         if idx < live_splits:
                             mv_r = mp[idx]
                             lv_r = lp[idx]
                         mvr.append(mv_r)
                         lvr.append(lv_r)
                     ob = mO.iterator + (hq0 + nh) * D
                     # Attention sink, global-join path: folded exactly once per
                     # head, into the global max and denominator of every round.
                     s2g = Float32(0.0)
                     if has_sink == 1:
                         s2g = mSNK.iterator[h_k_idx * gqa_ratio + nh] * Float32(log2_e)

                     # SPLIT-PARALLEL FOLD.  This CTA owns quads [q0, q0 + qpc) of
                     # its head, and `qpc = ceil(d_quads / live_splits)` is TINY on
                     # every wide plan (4 at D = 512 / 32 splits, 2 at D = 256 / 32).
                     # Dealing one quad per LANE therefore left 28 of 32 lanes
                     # holding a clamped address while the warp still issued the
                     # whole `live_splits`-deep gather and its 4 FFMAs per slot --
                     # the fold measured (NCU, d512 kv4096) at 22 % of the kernel's
                     # entire instruction count for output it could have produced
                     # with 1/8 of it.
                     #
                     # The lane grid is now (quad, sublane): `L = 32 / 2^ceil(log2
                     # qpc)` sublanes cooperate on ONE quad, sublane s walking
                     # splits s, s + L, s + 2L, ...  A lane's gather is
                     # `ceil(live_splits / L)` slots instead of `live_splits`, and a
                     # log2(L)-step butterfly (contiguous lanes, so it never leaves
                     # the group) sums the sublanes back together.  Both the loads
                     # of one lane and the butterfly stay a SINGLE round trip.
                     qpc = (Int32(d_quads) + live_splits - 1) // live_splits
                     q0 = split_idx * qpc
                     lsh = Int32(5)
                     if qpc > 1:
                         lsh = Int32(4)
                     if qpc > 2:
                         lsh = Int32(3)
                     if qpc > 4:
                         lsh = Int32(2)
                     if qpc > 8:
                         lsh = Int32(1)
                     if qpc > 16:
                         lsh = Int32(0)
                     nsub = Int32(1) << lsh              # sublanes per quad (L)
                     kper = Int32(warp_threads) >> lsh   # quads per lane round
                     kq = lane >> lsh                    # this lane's quad slot
                     s0 = lane - (kq << lsh)             # this lane's sublane
                     nrq = (qpc + kper - 1) // kper
                     for i in cutlass.range(nrq):
                         qd = q0 + i * kper + kq
                         inr = (i * kper + kq) < qpc
                         if qd >= Int32(d_quads):
                             inr = cutlass.Boolean(False)
                         accs = cutlass.Array(Float32, 4)
                         for e in cutlass.range_constexpr(4):
                             accs[e] = Float32(0.0)
                         qs = qd
                         if not inr:
                             qs = Int32(0)
                         # `nsp_max` bounds `ceil(live_splits / L)`: L is at least
                         # `16 * live_splits / d_quads`, so the run per lane never
                         # exceeds `d_quads / 16` (2 / 4 / 8), and it is a
                         # COMPILE-TIME trip count, so the whole run issues as one
                         # batch of independent loads.
                         # The partial-O run is issued FIRST: its addresses need
                         # only the lane deal, never the (m, l) pair, but the
                         # `warp_fmax` that turns (m, l) into fold weights is a
                         # shuffle chain and ptxas will not hoist a load across it.
                         # Issuing the gather ahead of the weight math overlaps the
                         # two workspace round trips instead of serialising them.
                         jcs = []
                         vs = []
                         for t in cutlass.range_constexpr(nsp_max):
                             j = s0 + (Int32(t) << lsh)
                             inj = j < live_splits
                             jc = j
                             if not inj:
                                 jc = Int32(0)
                             jcs.append((inj, jc))
                             # nvvm.load.ext has no fp16 vector form; pull the four
                             # halves as one 64-bit b16x4.
                             vs.append(
                                 prims.load_ext(
                                     jgb + qs * 4 + jc * D,
                                     dtype=cutlass.Int16,
                                     count=4,
                                 )
                             )
                         # Partials are normalised, so a split's fold weight is
                         # `l_s * exp2(m_s - M)`; the denominator is that same sum.
                         mv = mvr[0]
                         for r in cutlass.range_constexpr(1, sreg):
                             mv = cute.arch.fmax(mv, mvr[r])
                         gmn = warp_fmax(mv)
                         if has_sink == 1:
                             gmn = cute.arch.fmax(gmn, s2g)
                         fvr = []
                         for r in cutlass.range_constexpr(sreg):
                             fvr.append(exp2(mvr[r] - gmn) * lvr[r])
                         dv = fvr[0]
                         for r in cutlass.range_constexpr(1, sreg):
                             dv = dv + fvr[r]
                         for off in cutlass.range_constexpr(5):
                             dv = Float32(
                                 dv
                                 + prims.shfl_sync(
                                     thread_mask=0xFFFFFFFF,
                                     val=dv,
                                     offset=(16 >> off),
                                     mask_and_clamp=0x1F,
                                     kind=prims.Shfl.BFLY,
                                     return_value_and_is_valid=False,
                                 )
                             )
                         if has_sink == 1:
                             dv = dv + exp2(s2g - gmn)
                         rinv = cute.arch.rcp_approx(dv)
                         for t in cutlass.range_constexpr(nsp_max):
                             inj, jc = jcs[t]
                             src = fvr[0]
                             for r in cutlass.range_constexpr(1, sreg):
                                 if jc >= Int32(r * warp_threads):
                                     src = fvr[r]
                             f = prims.shfl_sync(
                                 thread_mask=0xFFFFFFFF,
                                 val=src,
                                 offset=jc & Int32(warp_threads - 1),
                                 mask_and_clamp=0x1F,
                                 kind=prims.Shfl.IDX,
                                 return_value_and_is_valid=False,
                             )
                             if not inj:
                                 f = Float32(0.0)
                             v = vs[t]
                             for e in cutlass.range_constexpr(4):
                                 accs[e] = accs[e] + f * v[e].bitcast(Float16).to(
                                     Float32
                                 )
                         # Butterfly over the `nsub` sublanes of this quad.  The
                         # group is `nsub` CONTIGUOUS lanes, so an XOR shuffle by
                         # any offset below `nsub` stays inside it; the bound is
                         # warp-uniform, so every lane runs the same trip count.
                         for off in cutlass.range_constexpr(5):
                             if Int32(1 << off) < nsub:
                                 for e in cutlass.range_constexpr(4):
                                     accs[e] = Float32(
                                         accs[e]
                                         + prims.shfl_sync(
                                             thread_mask=0xFFFFFFFF,
                                             val=accs[e],
                                             offset=(1 << off),
                                             mask_and_clamp=0x1F,
                                             kind=prims.Shfl.BFLY,
                                             return_value_and_is_valid=False,
                                         )
                                     )
                         if inr:
                             if s0 == 0:
                                 for e in cutlass.range_constexpr(4):
                                     (ob + qd * 4 + e).store(
                                         (accs[e] * rinv).to(Float16)
                                     )

            # No CTA of a live multi-split CLUSTER may retire while the folder can
            # still be consuming mailbox bytes sourced from this CTA's shared
            # memory (an engine read is in flight until jbar fires), so every
            # cluster member -- writers, folder and dead splits alike -- leaves
            # through one cluster-wide rendezvous.  The folder arrives only after
            # its fold, so the grid's completion time is unchanged.
            if clustered and live_splits > 1:
                prims.barrier_cluster_arrive()
                prims.barrier_cluster_wait()
        else:
            # ======================================= split-K join (whole CTA)
            # `active` and `live_splits` are CTA-uniform, so this is a uniform
            # branch: dead splits and the single-split rows never enter it.
            #
            # The fold used to run on the softmax warpgroup alone, dealing two
            # head-windows to every thread and walking the split run in 8- then 4-
            # then 1-wide blocks.  Each block was a separate round trip -- the
            # loads of block b+1 could not issue until block b's FFMAs had retired
            # -- and the per-split (m, l) scale pair cost yet another one ahead of
            # them all.  Measured against the rest of the kernel that chain was
            # ~0.35 us per live split, i.e. the single largest term on every
            # multi-split row (a 14-split row paid ~5 us of pure pointer chasing).
            #
            # Now the whole CTA folds: warp w owns head w of the mma_n tile and
            # lane l owns d-quad l, so a thread carries ONE 16 B run per split, the
            # scale pair and the entire partial-O run are independent loads issued
            # as one batch, and the per-split factors reach the lanes by shfl.idx
            # instead of a smem staging array plus a second warpgroup barrier.
            if active:
              if live_splits != 1:
                prims.barrier_cta_sync(join_barrier, thread_count=256)
                myrank = sFlag[0]
                # Spread fold (D = 512): the arrival ticket doubles as a folder
                # rank -- the first min(o_blocks, live) arrivals each take one
                # 128-elem d-block of the output and spin until the whole group
                # has arrived.  Every CTA of the launch is resident (grid <= SM
                # count, one CTA per SM by smem), so the spin cannot deadlock.
                # At o_blocks <= 2 the fold stays on the LAST arrival: two blocks
                # do not repay the spin, the extra CTA barriers and the departure
                # atomic.
                njc = Int32(o_blocks)
                if live_splits < njc:
                    njc = live_splits
                # Spreading is now a RUNTIME choice, not a per-D constant.  It buys
                # o_blocks-way parallelism in the sweep but costs a spin poll, a
                # second CTA barrier and a departure atomic -- three L2 round trips
                # that a narrow group cannot repay.  At four live splits and D = 512
                # the last arrival's whole serial fold is 16 quad loads; the spread
                # machinery around it is more expensive than the 12 loads it saves.
                fold_me = myrank == live_splits - 1
                blk0 = Int32(0)
                bstep = Int32(1)
                if cutlass.const_expr(spread_fold):
                    fold_me = myrank < njc
                    blk0 = myrank
                    bstep = njc
                if fold_me:
                    if cutlass.const_expr(spread_fold):
                        if tidx == 0:
                            while (
                                prims.atomicrmw(
                                    prims.AtomicOp.ADD,
                                    mSEM.iterator + coord_hb,
                                    Int32(0),
                                    mem_order=prims.MemOrder.ACQUIRE,
                                    syncscope=prims.MemScope.GPU,
                                )
                                < live_splits
                            ):
                                pass
                        # The acquire above is done by one thread; the CTA barrier
                        # publishes its ordering to the other 255.
                        prims.barrier_cta_sync(join_barrier, thread_count=256)
                        # Hand both counters back HERE, before the sweep, not after
                        # it.  A folder only reaches this point once it has already
                        # PASSED the spin, so a reset can never strand one -- and
                        # the departure atomic used to be the very last instruction
                        # of the whole launch, an L2 round trip with nothing left to
                        # hide it.  Now it retires under the fold.
                        if tidx == 0:
                            old2 = prims.atomicrmw(
                                prims.AtomicOp.ADD,
                                mSEM.iterator + groups + coord_hb,
                                Int32(1),
                                mem_order=prims.MemOrder.ACQ_REL,
                                syncscope=prims.MemScope.GPU,
                            )
                            if old2 == njc - 1:
                                (mSEM.iterator + coord_hb).store(Int32(0))
                                (mSEM.iterator + groups + coord_hb).store(Int32(0))
                    else:
                        # The last arrival hands the slot back at zero, so the
                        # counter never leaves [0, live_splits).
                        if tidx == 0:
                            (mSEM.iterator + coord_hb).store(Int32(0))
                    nh = warp_idx
                    # Warps above the runtime GQA ratio own no real head.  They
                    # used to alias head 0 and replay the entire gather, on the
                    # argument that the duplicate loads were L1 hits; they are not.
                    # The smem carveout sits at its maximum, so the L1 data cache
                    # is a couple of KiB and every alias was a second set of L2
                    # round trips whose result the store predicate then dropped --
                    # half the join's traffic at gqa 4, three quarters at gqa 2.
                    if nh < gqa_ratio:
                     hq_pad = b_dim * HQ + mma_n
                     max_off = hq_pad * SPLITS * D
                     col_off = max_off + hq_pad * SPLITS
                     hq0 = b_idx * HQ + h_k_idx * gqa_ratio
                     jb = mWH.iterator + (hq0 + nh) * (SPLITS * D) + lane * 4
                     mp = mWS.iterator + max_off + (hq0 + nh) * SPLITS
                     lp = mWS.iterator + col_off + (hq0 + nh) * SPLITS
                     # Live-tiered fold.  Rows needing at most one 32-wide pass
                     # (every row below 33 live splits) pick a straight-line
                     # trace-time body by a runtime tier test: a 4-split D=512 row
                     # folds 8 slots instead of issuing 32 loads per d-block and
                     # clamping 28 away, and no runtime trip count sits on the
                     # fold's critical path.  Small tiers (T * o_blocks <= 32
                     # quads) stage the WHOLE burst -- every block interleaved --
                     # before the scale fold, so the in-order pipe issues one L2
                     # round trip and folds in its shadow; larger tiers stream
                     # load+FFMA with the per-split factor shuffled once and shared
                     # by every d-block.  33..64 splits take the two-round staged
                     # arm at the bottom.
                     mv = Float32(-1.0e30)
                     lv = Float32(0.0)
                     if lane < live_splits:
                         mv = mp[lane]
                         lv = lp[lane]
                     # Attention sink, multi-split path: folded HERE, exactly once, into
                     # the GLOBAL max across the splits and the denominator (no V row).
                     # Natural-log sink -> log2 domain.  The -1e30 sentinel matches `mv`,
                     # so a has_sink == 0 row never perturbs the fold.
                     s2 = Float32(-1.0e30)
                     snk = Float32(0.0)
                     if has_sink == 1:
                         s2 = mSNK.iterator[h_k_idx * gqa_ratio + nh] * Float32(log2_e)
                     ob = mO.iterator + (hq0 + nh) * D + lane * 4
                     if cutlass.const_expr(o_blocks <= 2):
                         if live_splits <= 32:
                             dvl = Float32(0.0)
                             acc = cutlass.Array(Float32, 4 * o_blocks)
                             for e in cutlass.range_constexpr(4 * o_blocks):
                                 acc[e] = Float32(0.0)
                             if live_splits <= 8:
                                 if cutlass.const_expr(8 * o_blocks <= 32 and o_blocks <= 2):
                                     vs = []
                                     for t in cutlass.range_constexpr(8):
                                         j = Int32(t)
                                         if j >= live_splits:
                                             j = Int32(0)
                                         for blk in cutlass.range_constexpr(o_blocks):
                                             vs.append(prims.load_ext(
                                                 jb + j * D + blk * v_slice_d,
                                                 dtype=cutlass.Int16, count=4,
                                             ))
                                     gmn = warp_fmax(mv)
                                     if has_sink == 1:
                                         gmn = cute.arch.fmax(gmn, s2)
                                         snk = exp2(s2 - gmn)
                                     fv = exp2(mv - gmn) * lv
                                     dvl = fv
                                     for t in cutlass.range_constexpr(8):
                                         f = prims.shfl_sync(
                                             thread_mask=0xFFFFFFFF,
                                             val=fv,
                                             offset=t,
                                             mask_and_clamp=0x1F,
                                             kind=prims.Shfl.IDX,
                                             return_value_and_is_valid=False,
                                         )
                                         for blk in cutlass.range_constexpr(o_blocks):
                                             v = vs[t * o_blocks + blk]
                                             for e in cutlass.range_constexpr(4):
                                                 acc[blk * 4 + e] = (
                                                     acc[blk * 4 + e]
                                                     + f * v[e].bitcast(Float16).to(Float32)
                                                 )
                                 else:
                                     gmn = warp_fmax(mv)
                                     if has_sink == 1:
                                         gmn = cute.arch.fmax(gmn, s2)
                                         snk = exp2(s2 - gmn)
                                     fv = exp2(mv - gmn) * lv
                                     dvl = fv
                                     for t in cutlass.range_constexpr(8):
                                         j = Int32(t)
                                         if j >= live_splits:
                                             j = Int32(0)
                                         f = prims.shfl_sync(
                                             thread_mask=0xFFFFFFFF,
                                             val=fv,
                                             offset=t,
                                             mask_and_clamp=0x1F,
                                             kind=prims.Shfl.IDX,
                                             return_value_and_is_valid=False,
                                         )
                                         for blk in cutlass.range_constexpr(o_blocks):
                                             v = prims.load_ext(
                                                 jb + j * D + blk * v_slice_d,
                                                 dtype=cutlass.Int16, count=4,
                                             )
                                             for e in cutlass.range_constexpr(4):
                                                 acc[blk * 4 + e] = (
                                                     acc[blk * 4 + e]
                                                     + f * v[e].bitcast(Float16).to(Float32)
                                                 )
                             else:
                              if live_splits <= 16:
                                  if cutlass.const_expr(16 * o_blocks <= 32 and o_blocks <= 2):
                                      vs = []
                                      for t in cutlass.range_constexpr(16):
                                          j = Int32(t)
                                          if j >= live_splits:
                                              j = Int32(0)
                                          for blk in cutlass.range_constexpr(o_blocks):
                                              vs.append(prims.load_ext(
                                                  jb + j * D + blk * v_slice_d,
                                                  dtype=cutlass.Int16, count=4,
                                              ))
                                      gmn = warp_fmax(mv)
                                      if has_sink == 1:
                                          gmn = cute.arch.fmax(gmn, s2)
                                          snk = exp2(s2 - gmn)
                                      fv = exp2(mv - gmn) * lv
                                      dvl = fv
                                      for t in cutlass.range_constexpr(16):
                                          f = prims.shfl_sync(
                                              thread_mask=0xFFFFFFFF,
                                              val=fv,
                                              offset=t,
                                              mask_and_clamp=0x1F,
                                              kind=prims.Shfl.IDX,
                                              return_value_and_is_valid=False,
                                          )
                                          for blk in cutlass.range_constexpr(o_blocks):
                                              v = vs[t * o_blocks + blk]
                                              for e in cutlass.range_constexpr(4):
                                                  acc[blk * 4 + e] = (
                                                      acc[blk * 4 + e]
                                                      + f * v[e].bitcast(Float16).to(Float32)
                                                  )
                                  else:
                                      gmn = warp_fmax(mv)
                                      if has_sink == 1:
                                          gmn = cute.arch.fmax(gmn, s2)
                                          snk = exp2(s2 - gmn)
                                      fv = exp2(mv - gmn) * lv
                                      dvl = fv
                                      for t in cutlass.range_constexpr(16):
                                          j = Int32(t)
                                          if j >= live_splits:
                                              j = Int32(0)
                                          f = prims.shfl_sync(
                                              thread_mask=0xFFFFFFFF,
                                              val=fv,
                                              offset=t,
                                              mask_and_clamp=0x1F,
                                              kind=prims.Shfl.IDX,
                                              return_value_and_is_valid=False,
                                          )
                                          for blk in cutlass.range_constexpr(o_blocks):
                                              v = prims.load_ext(
                                                  jb + j * D + blk * v_slice_d,
                                                  dtype=cutlass.Int16, count=4,
                                              )
                                              for e in cutlass.range_constexpr(4):
                                                  acc[blk * 4 + e] = (
                                                      acc[blk * 4 + e]
                                                      + f * v[e].bitcast(Float16).to(Float32)
                                                  )
                              else:
                                   if cutlass.const_expr(32 * o_blocks <= 32 and o_blocks <= 2):
                                       vs = []
                                       for t in cutlass.range_constexpr(32):
                                           j = Int32(t)
                                           if j >= live_splits:
                                               j = Int32(0)
                                           for blk in cutlass.range_constexpr(o_blocks):
                                               vs.append(prims.load_ext(
                                                   jb + j * D + blk * v_slice_d,
                                                   dtype=cutlass.Int16, count=4,
                                               ))
                                       gmn = warp_fmax(mv)
                                       if has_sink == 1:
                                           gmn = cute.arch.fmax(gmn, s2)
                                           snk = exp2(s2 - gmn)
                                       fv = exp2(mv - gmn) * lv
                                       dvl = fv
                                       for t in cutlass.range_constexpr(32):
                                           f = prims.shfl_sync(
                                               thread_mask=0xFFFFFFFF,
                                               val=fv,
                                               offset=t,
                                               mask_and_clamp=0x1F,
                                               kind=prims.Shfl.IDX,
                                               return_value_and_is_valid=False,
                                           )
                                           for blk in cutlass.range_constexpr(o_blocks):
                                               v = vs[t * o_blocks + blk]
                                               for e in cutlass.range_constexpr(4):
                                                   acc[blk * 4 + e] = (
                                                       acc[blk * 4 + e]
                                                       + f * v[e].bitcast(Float16).to(Float32)
                                                   )
                                   else:
                                       gmn = warp_fmax(mv)
                                       if has_sink == 1:
                                           gmn = cute.arch.fmax(gmn, s2)
                                           snk = exp2(s2 - gmn)
                                       fv = exp2(mv - gmn) * lv
                                       dvl = fv
                                       for t in cutlass.range_constexpr(32):
                                           j = Int32(t)
                                           if j >= live_splits:
                                               j = Int32(0)
                                           f = prims.shfl_sync(
                                               thread_mask=0xFFFFFFFF,
                                               val=fv,
                                               offset=t,
                                               mask_and_clamp=0x1F,
                                               kind=prims.Shfl.IDX,
                                               return_value_and_is_valid=False,
                                           )
                                           for blk in cutlass.range_constexpr(o_blocks):
                                               v = prims.load_ext(
                                                   jb + j * D + blk * v_slice_d,
                                                   dtype=cutlass.Int16, count=4,
                                               )
                                               for e in cutlass.range_constexpr(4):
                                                   acc[blk * 4 + e] = (
                                                       acc[blk * 4 + e]
                                                       + f * v[e].bitcast(Float16).to(Float32)
                                                   )
                             for off in cutlass.range_constexpr(5):
                                 dvl = Float32(
                                     dvl
                                     + prims.shfl_sync(
                                         thread_mask=0xFFFFFFFF,
                                         val=dvl,
                                         offset=(16 >> off),
                                         mask_and_clamp=0x1F,
                                         kind=prims.Shfl.BFLY,
                                         return_value_and_is_valid=False,
                                     )
                                 )
                             if has_sink == 1:
                                 dvl = dvl + snk
                             rinv = cute.arch.rcp_approx(dvl)
                             pk4 = cutlass.Array(cutlass.Int16, 4)
                             for blk in cutlass.range_constexpr(o_blocks):
                                 for e in cutlass.range_constexpr(4):
                                     pk4[e] = (acc[blk * 4 + e] * rinv).to(Float16).bitcast(cutlass.Int16)
                                 if cutlass.const_expr(d_lanes == 128):
                                     prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), ob + blk * v_slice_d)
                                 else:
                                     if lane < d_lanes // 4:
                                         prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), ob + blk * v_slice_d)
                         else:
                             if cutlass.const_expr(o_blocks == 1):
                                 # 33..64 live splits: two 32-slot batch rounds per d-block through a
                                 # staging buffer, packed-f32x2 folds, the whole scale fold hoisted above
                                 # the gather (the batch's loads are consumed as they land instead of
                                 # behind a warp reduction).  The host caps SPLITS at 64, so this arm is
                                 # the widest a launch can need.
                                 mv2 = Float32(-1.0e30)
                                 lv2 = Float32(0.0)
                                 li2 = warp_threads + lane
                                 if li2 >= live_splits:
                                     li2 = live_splits - 1
                                 mv2 = mp[li2]
                                 lv2 = lp[li2]
                                 if warp_threads + lane >= live_splits:
                                     mv2 = Float32(-1.0e30)
                                     lv2 = Float32(0.0)
                                 gmn = warp_fmax(cute.arch.fmax(mv, mv2))
                                 if has_sink == 1:
                                     gmn = cute.arch.fmax(gmn, s2)
                                     snk = exp2(s2 - gmn)
                                 fv = exp2(mv - gmn) * lv
                                 fv2 = exp2(mv2 - gmn) * lv2
                                 dv = fv + fv2
                                 for off in cutlass.range_constexpr(5):
                                     dv = Float32(
                                         dv
                                         + prims.shfl_sync(
                                             thread_mask=0xFFFFFFFF,
                                             val=dv,
                                             offset=(16 >> off),
                                             mask_and_clamp=0x1F,
                                             kind=prims.Shfl.BFLY,
                                             return_value_and_is_valid=False,
                                         )
                                     )
                                 if has_sink == 1:
                                     dv = dv + snk
                                 rinv = cute.arch.rcp_approx(dv)
                                 accb = cutlass.Array(Float32, 4)
                                 buf = cutlass.Array(Float32, 4 * warp_threads)
                                 jbb = jb
                                 obb = ob
                                 for e in cutlass.range_constexpr(4):
                                     accb[e] = Float32(0.0)
                                 for t in cutlass.range_constexpr(warp_threads):
                                     j = Int32(t)
                                     if j >= live_splits:
                                         j = Int32(0)
                                     v = prims.load_ext(jbb + j * D, dtype=cutlass.Int16, count=4)
                                     for e in cutlass.range_constexpr(4):
                                         buf[t * 4 + e] = v[e].bitcast(Float16).to(Float32)
                                 for t in cutlass.range_constexpr(warp_threads):
                                     f = prims.shfl_sync(
                                         thread_mask=0xFFFFFFFF,
                                         val=fv,
                                         offset=t,
                                         mask_and_clamp=0x1F,
                                         kind=prims.Shfl.IDX,
                                         return_value_and_is_valid=False,
                                     )
                                     for e2 in cutlass.range_constexpr(0, 4, 2):
                                         p2 = ffma2(
                                             (f, f),
                                             (buf[t * 4 + e2], buf[t * 4 + e2 + 1]),
                                             (accb[e2], accb[e2 + 1]),
                                         )
                                         accb[e2] = p2[0]
                                         accb[e2 + 1] = p2[1]
                                 for t in cutlass.range_constexpr(warp_threads):
                                     j = Int32(warp_threads + t)
                                     if j >= live_splits:
                                         j = Int32(0)
                                     v = prims.load_ext(jbb + j * D, dtype=cutlass.Int16, count=4)
                                     for e in cutlass.range_constexpr(4):
                                         buf[t * 4 + e] = v[e].bitcast(Float16).to(Float32)
                                 for t in cutlass.range_constexpr(warp_threads):
                                     f = prims.shfl_sync(
                                         thread_mask=0xFFFFFFFF,
                                         val=fv2,
                                         offset=t,
                                         mask_and_clamp=0x1F,
                                         kind=prims.Shfl.IDX,
                                         return_value_and_is_valid=False,
                                     )
                                     for e2 in cutlass.range_constexpr(0, 4, 2):
                                         p2 = ffma2(
                                             (f, f),
                                             (buf[t * 4 + e2], buf[t * 4 + e2 + 1]),
                                             (accb[e2], accb[e2 + 1]),
                                         )
                                         accb[e2] = p2[0]
                                         accb[e2 + 1] = p2[1]
                                 pk4 = cutlass.Array(cutlass.Int16, 4)
                                 for e in cutlass.range_constexpr(4):
                                     pk4[e] = (accb[e] * rinv).to(Float16).bitcast(cutlass.Int16)
                                 if cutlass.const_expr(d_lanes == 128):
                                     prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), obb)
                                 else:
                                     if lane < d_lanes // 4:
                                         prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), obb)
                             else:
                                 # o_blocks >= 2 at 33..64 splits: the per-block staged rounds
                                 # serialise a round trip per d-block (37.2 vs 35.9 us on the
                                 # 235-page row), so this arm keeps the streaming pass shape;
                                 # the second half's factors ride a second (m, l) register pair.
                                 mv2w = Float32(-1.0e30)
                                 lv2w = Float32(0.0)
                                 li2w = warp_threads + lane
                                 if li2w >= live_splits:
                                     li2w = live_splits - 1
                                 mv2w = mp[li2w]
                                 lv2w = lp[li2w]
                                 if warp_threads + lane >= live_splits:
                                     mv2w = Float32(-1.0e30)
                                     lv2w = Float32(0.0)
                                 gmnw = warp_fmax(cute.arch.fmax(mv, mv2w))
                                 if has_sink == 1:
                                     gmnw = cute.arch.fmax(gmnw, s2)
                                     snk = exp2(s2 - gmnw)
                                 fvw = exp2(mv - gmnw) * lv
                                 fv2w = exp2(mv2w - gmnw) * lv2w
                                 dvw = fvw + fv2w
                                 for off in cutlass.range_constexpr(5):
                                     dvw = Float32(
                                         dvw
                                         + prims.shfl_sync(
                                             thread_mask=0xFFFFFFFF,
                                             val=dvw,
                                             offset=(16 >> off),
                                             mask_and_clamp=0x1F,
                                             kind=prims.Shfl.BFLY,
                                             return_value_and_is_valid=False,
                                         )
                                     )
                                 if has_sink == 1:
                                     dvw = dvw + snk
                                 rinvw = cute.arch.rcp_approx(dvw)
                                 acc2 = cutlass.Array(Float32, 4 * o_blocks)
                                 for e in cutlass.range_constexpr(4 * o_blocks):
                                     acc2[e] = Float32(0.0)
                                 for t in cutlass.range_constexpr(32):
                                     j = Int32(0 + t)
                                     if j >= live_splits:
                                         j = Int32(0)
                                     f = prims.shfl_sync(
                                         thread_mask=0xFFFFFFFF,
                                         val=fvw,
                                         offset=t,
                                         mask_and_clamp=0x1F,
                                         kind=prims.Shfl.IDX,
                                         return_value_and_is_valid=False,
                                     )
                                     for blk in cutlass.range_constexpr(o_blocks):
                                         v = prims.load_ext(
                                             jb + j * D + blk * v_slice_d,
                                             dtype=cutlass.Int16, count=4,
                                         )
                                         for e in cutlass.range_constexpr(4):
                                             acc2[blk * 4 + e] = (
                                                 acc2[blk * 4 + e]
                                                 + f * v[e].bitcast(Float16).to(Float32)
                                             )
                                 for t in cutlass.range_constexpr(32):
                                     j = Int32(warp_threads + t)
                                     if j >= live_splits:
                                         j = Int32(0)
                                     f = prims.shfl_sync(
                                         thread_mask=0xFFFFFFFF,
                                         val=fv2w,
                                         offset=t,
                                         mask_and_clamp=0x1F,
                                         kind=prims.Shfl.IDX,
                                         return_value_and_is_valid=False,
                                     )
                                     for blk in cutlass.range_constexpr(o_blocks):
                                         v = prims.load_ext(
                                             jb + j * D + blk * v_slice_d,
                                             dtype=cutlass.Int16, count=4,
                                         )
                                         for e in cutlass.range_constexpr(4):
                                             acc2[blk * 4 + e] = (
                                                 acc2[blk * 4 + e]
                                                 + f * v[e].bitcast(Float16).to(Float32)
                                             )
                                 pk4 = cutlass.Array(cutlass.Int16, 4)
                                 for blk in cutlass.range_constexpr(o_blocks):
                                     for e in cutlass.range_constexpr(4):
                                         pk4[e] = (acc2[blk * 4 + e] * rinvw).to(Float16).bitcast(cutlass.Int16)
                                     prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), ob + blk * v_slice_d)


                     else:
                         # Splits 32..63: second (m, l) pair; -inf when unused = weight 0.
                         mv2 = Float32(-1.0e30)
                         lv2 = Float32(0.0)
                         if live_splits > warp_threads:
                             li2 = warp_threads + lane
                             if li2 >= live_splits:
                                 li2 = live_splits - 1
                             mv2 = mp[li2]
                             lv2 = lp[li2]
                             if warp_threads + lane >= live_splits:
                                 mv2 = Float32(-1.0e30)
                                 lv2 = Float32(0.0)
                         # The fold
                         # The partial-O sweep used to be a CONSTEXPR 32 slots (twice
                         # above 32 splits) whatever the live split count was: an
                         # 8-split row issued 32 loads, 32 shuffles and 128 FFMAs per
                         # d-block and threw 24/32 of the result away on a clamp.  At
                         # D = 512 (o_blocks = 4) that was ~1400 instructions per
                         # thread, on ONE CTA, at the very end of the launch.
                         #
                         # Sizing it as ceil(live/8) runtime groups fixed the waste but
                         # cost the wide rows their ILP: eight groups of eight expose
                         # the partial-O load latency eight times where one batch of
                         # thirty-two exposed it once.  So the WIDTH is picked by a
                         # single runtime tier test instead, and every sweep stays a
                         # constexpr batch of independent loads:
                         #   live <=  8 -> one batch of 8   (4x fewer slots)
                         #   live <= 16 -> one batch of 16  (2x fewer slots)
                         #   else       -> ceil(live/32) batches of 32, i.e. exactly
                         #                 the original sweep, ILP untouched.
                         # A batch never straddles a 32-slot block, so the (m, l)
                         # factor register is chosen once per batch and every shuffle
                         # keeps an immediate lane offset.
                         acc = cutlass.Array(Float32, 4)
                         buf = cutlass.Array(Float32, 4 * warp_threads)

                         # Partials are normalised, so a split's fold weight is
                         # `l_s * exp2(m_s - M)`; the denominator is that same sum.
                         # Hoisted above the gather: a handful of ALU ops on registers
                         # that are already live, so every batch's loads are consumed
                         # as soon as they land instead of behind a warp reduction.
                         gmn = warp_fmax(cute.arch.fmax(mv, mv2))
                         if has_sink == 1:
                             gmn = cute.arch.fmax(gmn, s2)
                             snk = exp2(s2 - gmn)
                         fv = exp2(mv - gmn) * lv
                         fv2 = exp2(mv2 - gmn) * lv2
                         dv = fv + fv2
                         for off in cutlass.range_constexpr(5):
                             dv = Float32(
                                 dv
                                 + prims.shfl_sync(
                                     thread_mask=0xFFFFFFFF,
                                     val=dv,
                                     offset=(16 >> off),
                                     mask_and_clamp=0x1F,
                                     kind=prims.Shfl.BFLY,
                                     return_value_and_is_valid=False,
                                 )
                             )
                         if has_sink == 1:
                             dv = dv + snk
                         rinv = cute.arch.rcp_approx(dv)
                         ob = mO.iterator + (hq0 + nh) * D + lane * 4

                         # The tier test is CTA-uniform and sits outside the d-block
                         # walk, so a sweep is always a straight-line batch of
                         # independent loads -- no runtime trip count anywhere on the
                         # fold's critical path.  DSL scopes cannot close over the
                         # accumulator arrays, and `range_constexpr` has to reach the
                         # preprocessor literally, so the bodies are written out.
                         #
                         # `o_blocks` is a per-D python constant, so the d-block walk
                         # is picked at TRACE time and only one of these two forms is
                         # ever built: at D = 64 the single block is straight-line
                         # (a runtime loop header around one iteration cost ~8 % on
                         # the 128-page row), at D >= 256 the blocks share one loop
                         # body rather than unrolling four copies of the widest sweep.
                         nblk = Int32(o_blocks)
                         # This folder owns d-blocks myrank, myrank+njc, ...  When the
                         # group has at least o_blocks live splits (every multi-block
                         # row here) that is exactly one block per folder.
                         nrep = Int32(1)
                         if cutlass.const_expr(spread_fold):
                             # One block per folder whenever the group has at least
                             # o_blocks live splits, which every multi-block row
                             # does.  The general ceil() form needs a runtime
                             # integer division -- ~30 instructions sitting on the
                             # fold's critical path -- so it is now reached only by
                             # the narrow groups that actually need it.
                             if njc < nblk:
                                 nrep = (nblk - blk0 + bstep - Int32(1)) // bstep
                         elif o_blocks > 1:
                             nrep = nblk
                         if o_blocks == 1:
                             if live_splits <= 8:
                                 jbb = jb
                                 obb = ob
                                 for e in cutlass.range_constexpr(4):
                                     acc[e] = Float32(0.0)
                                 for t in cutlass.range_constexpr(8):
                                     j = Int32(0) + t
                                     if j >= live_splits:
                                         j = Int32(0)
                                     v = prims.load_ext(jbb + j * D, dtype=cutlass.Int16, count=4)
                                     for e in cutlass.range_constexpr(4):
                                         buf[t * 4 + e] = v[e].bitcast(Float16).to(Float32)
                                 for t in cutlass.range_constexpr(8):
                                     f = prims.shfl_sync(
                                         thread_mask=0xFFFFFFFF,
                                         val=fv,
                                         offset=t,
                                         mask_and_clamp=0x1F,
                                         kind=prims.Shfl.IDX,
                                         return_value_and_is_valid=False,
                                     )
                                     for e2 in cutlass.range_constexpr(0, 4, 2):
                                         p2 = ffma2(
                                             (f, f),
                                             (buf[t * 4 + e2], buf[t * 4 + e2 + 1]),
                                             (acc[e2], acc[e2 + 1]),
                                         )
                                         acc[e2] = p2[0]
                                         acc[e2 + 1] = p2[1]
                                 pk4 = cutlass.Array(cutlass.Int16, 4)
                                 for e in cutlass.range_constexpr(4):
                                     pk4[e] = (acc[e] * rinv).to(Float16).bitcast(cutlass.Int16)
                                 if cutlass.const_expr(d_lanes == 128):
                                     prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), obb)
                                 else:
                                     if lane < d_lanes // 4:
                                         prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), obb)
                             else:
                                 if live_splits <= 16:
                                     jbb = jb
                                     obb = ob
                                     for e in cutlass.range_constexpr(4):
                                         acc[e] = Float32(0.0)
                                     for t in cutlass.range_constexpr(16):
                                         j = Int32(0) + t
                                         if j >= live_splits:
                                             j = Int32(0)
                                         v = prims.load_ext(jbb + j * D, dtype=cutlass.Int16, count=4)
                                         for e in cutlass.range_constexpr(4):
                                             buf[t * 4 + e] = v[e].bitcast(Float16).to(Float32)
                                     for t in cutlass.range_constexpr(16):
                                         f = prims.shfl_sync(
                                             thread_mask=0xFFFFFFFF,
                                             val=fv,
                                             offset=t,
                                             mask_and_clamp=0x1F,
                                             kind=prims.Shfl.IDX,
                                             return_value_and_is_valid=False,
                                         )
                                         for e2 in cutlass.range_constexpr(0, 4, 2):
                                             p2 = ffma2(
                                                 (f, f),
                                                 (buf[t * 4 + e2], buf[t * 4 + e2 + 1]),
                                                 (acc[e2], acc[e2 + 1]),
                                             )
                                             acc[e2] = p2[0]
                                             acc[e2 + 1] = p2[1]
                                     pk4 = cutlass.Array(cutlass.Int16, 4)
                                     for e in cutlass.range_constexpr(4):
                                         pk4[e] = (acc[e] * rinv).to(Float16).bitcast(cutlass.Int16)
                                     if cutlass.const_expr(d_lanes == 128):
                                         prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), obb)
                                     else:
                                         if lane < d_lanes // 4:
                                             prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), obb)
                                 else:
                                     if live_splits <= warp_threads:
                                         jbb = jb
                                         obb = ob
                                         for e in cutlass.range_constexpr(4):
                                             acc[e] = Float32(0.0)
                                         for t in cutlass.range_constexpr(warp_threads):
                                             j = Int32(0) + t
                                             if j >= live_splits:
                                                 j = Int32(0)
                                             v = prims.load_ext(jbb + j * D, dtype=cutlass.Int16, count=4)
                                             for e in cutlass.range_constexpr(4):
                                                 buf[t * 4 + e] = v[e].bitcast(Float16).to(Float32)
                                         for t in cutlass.range_constexpr(warp_threads):
                                             f = prims.shfl_sync(
                                                 thread_mask=0xFFFFFFFF,
                                                 val=fv,
                                                 offset=t,
                                                 mask_and_clamp=0x1F,
                                                 kind=prims.Shfl.IDX,
                                                 return_value_and_is_valid=False,
                                             )
                                             for e2 in cutlass.range_constexpr(0, 4, 2):
                                                 p2 = ffma2(
                                                     (f, f),
                                                     (buf[t * 4 + e2], buf[t * 4 + e2 + 1]),
                                                     (acc[e2], acc[e2 + 1]),
                                                 )
                                                 acc[e2] = p2[0]
                                                 acc[e2 + 1] = p2[1]
                                         pk4 = cutlass.Array(cutlass.Int16, 4)
                                         for e in cutlass.range_constexpr(4):
                                             pk4[e] = (acc[e] * rinv).to(Float16).bitcast(cutlass.Int16)
                                         if cutlass.const_expr(d_lanes == 128):
                                             prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), obb)
                                         else:
                                             if lane < d_lanes // 4:
                                                 prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), obb)
                                     else:
                                         # Splits 32..63 fold in a second batch off
                                         # the second (m, l) pair.
                                         jbb = jb
                                         obb = ob
                                         for e in cutlass.range_constexpr(4):
                                             acc[e] = Float32(0.0)
                                         for t in cutlass.range_constexpr(warp_threads):
                                             j = Int32(0) + t
                                             if j >= live_splits:
                                                 j = Int32(0)
                                             v = prims.load_ext(jbb + j * D, dtype=cutlass.Int16, count=4)
                                             for e in cutlass.range_constexpr(4):
                                                 buf[t * 4 + e] = v[e].bitcast(Float16).to(Float32)
                                         for t in cutlass.range_constexpr(warp_threads):
                                             f = prims.shfl_sync(
                                                 thread_mask=0xFFFFFFFF,
                                                 val=fv,
                                                 offset=t,
                                                 mask_and_clamp=0x1F,
                                                 kind=prims.Shfl.IDX,
                                                 return_value_and_is_valid=False,
                                             )
                                             for e2 in cutlass.range_constexpr(0, 4, 2):
                                                 p2 = ffma2(
                                                     (f, f),
                                                     (buf[t * 4 + e2], buf[t * 4 + e2 + 1]),
                                                     (acc[e2], acc[e2 + 1]),
                                                 )
                                                 acc[e2] = p2[0]
                                                 acc[e2 + 1] = p2[1]
                                         for t in cutlass.range_constexpr(warp_threads):
                                             j = Int32(warp_threads) + t
                                             if j >= live_splits:
                                                 j = Int32(0)
                                             v = prims.load_ext(jbb + j * D, dtype=cutlass.Int16, count=4)
                                             for e in cutlass.range_constexpr(4):
                                                 buf[t * 4 + e] = v[e].bitcast(Float16).to(Float32)
                                         for t in cutlass.range_constexpr(warp_threads):
                                             f = prims.shfl_sync(
                                                 thread_mask=0xFFFFFFFF,
                                                 val=fv2,
                                                 offset=t,
                                                 mask_and_clamp=0x1F,
                                                 kind=prims.Shfl.IDX,
                                                 return_value_and_is_valid=False,
                                             )
                                             for e2 in cutlass.range_constexpr(0, 4, 2):
                                                 p2 = ffma2(
                                                     (f, f),
                                                     (buf[t * 4 + e2], buf[t * 4 + e2 + 1]),
                                                     (acc[e2], acc[e2 + 1]),
                                                 )
                                                 acc[e2] = p2[0]
                                                 acc[e2 + 1] = p2[1]
                                         pk4 = cutlass.Array(cutlass.Int16, 4)
                                         for e in cutlass.range_constexpr(4):
                                             pk4[e] = (acc[e] * rinv).to(Float16).bitcast(cutlass.Int16)
                                         if cutlass.const_expr(d_lanes == 128):
                                             prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), obb)
                                         else:
                                             if lane < d_lanes // 4:
                                                 prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), obb)
                         else:
                             if live_splits <= 8:
                                 for i in cutlass.range(nrep):
                                     bb = blk0 + i * bstep
                                     jbb = jb + bb * v_slice_d
                                     obb = ob + bb * v_slice_d
                                     for e in cutlass.range_constexpr(4):
                                         acc[e] = Float32(0.0)
                                     for t in cutlass.range_constexpr(8):
                                         j = Int32(0) + t
                                         if j >= live_splits:
                                             j = Int32(0)
                                         v = prims.load_ext(jbb + j * D, dtype=cutlass.Int16, count=4)
                                         for e in cutlass.range_constexpr(4):
                                             buf[t * 4 + e] = v[e].bitcast(Float16).to(Float32)
                                     for t in cutlass.range_constexpr(8):
                                         f = prims.shfl_sync(
                                             thread_mask=0xFFFFFFFF,
                                             val=fv,
                                             offset=t,
                                             mask_and_clamp=0x1F,
                                             kind=prims.Shfl.IDX,
                                             return_value_and_is_valid=False,
                                         )
                                         for e2 in cutlass.range_constexpr(0, 4, 2):
                                             p2 = ffma2(
                                                 (f, f),
                                                 (buf[t * 4 + e2], buf[t * 4 + e2 + 1]),
                                                 (acc[e2], acc[e2 + 1]),
                                             )
                                             acc[e2] = p2[0]
                                             acc[e2 + 1] = p2[1]
                                     pk4 = cutlass.Array(cutlass.Int16, 4)
                                     for e in cutlass.range_constexpr(4):
                                         pk4[e] = (acc[e] * rinv).to(Float16).bitcast(cutlass.Int16)
                                     if cutlass.const_expr(d_lanes == 128):
                                         prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), obb)
                                     else:
                                         if lane < d_lanes // 4:
                                             prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), obb)
                             else:
                                 if live_splits <= 16:
                                     for i in cutlass.range(nrep):
                                         bb = blk0 + i * bstep
                                         jbb = jb + bb * v_slice_d
                                         obb = ob + bb * v_slice_d
                                         for e in cutlass.range_constexpr(4):
                                             acc[e] = Float32(0.0)
                                         for t in cutlass.range_constexpr(16):
                                             j = Int32(0) + t
                                             if j >= live_splits:
                                                 j = Int32(0)
                                             v = prims.load_ext(jbb + j * D, dtype=cutlass.Int16, count=4)
                                             for e in cutlass.range_constexpr(4):
                                                 buf[t * 4 + e] = v[e].bitcast(Float16).to(Float32)
                                         for t in cutlass.range_constexpr(16):
                                             f = prims.shfl_sync(
                                                 thread_mask=0xFFFFFFFF,
                                                 val=fv,
                                                 offset=t,
                                                 mask_and_clamp=0x1F,
                                                 kind=prims.Shfl.IDX,
                                                 return_value_and_is_valid=False,
                                             )
                                             for e2 in cutlass.range_constexpr(0, 4, 2):
                                                 p2 = ffma2(
                                                     (f, f),
                                                     (buf[t * 4 + e2], buf[t * 4 + e2 + 1]),
                                                     (acc[e2], acc[e2 + 1]),
                                                 )
                                                 acc[e2] = p2[0]
                                                 acc[e2 + 1] = p2[1]
                                         pk4 = cutlass.Array(cutlass.Int16, 4)
                                         for e in cutlass.range_constexpr(4):
                                             pk4[e] = (acc[e] * rinv).to(Float16).bitcast(cutlass.Int16)
                                         if cutlass.const_expr(d_lanes == 128):
                                             prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), obb)
                                         else:
                                             if lane < d_lanes // 4:
                                                 prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), obb)
                                 else:
                                     if live_splits <= warp_threads:
                                         for i in cutlass.range(nrep):
                                             bb = blk0 + i * bstep
                                             jbb = jb + bb * v_slice_d
                                             obb = ob + bb * v_slice_d
                                             for e in cutlass.range_constexpr(4):
                                                 acc[e] = Float32(0.0)
                                             for t in cutlass.range_constexpr(warp_threads):
                                                 j = Int32(0) + t
                                                 if j >= live_splits:
                                                     j = Int32(0)
                                                 v = prims.load_ext(jbb + j * D, dtype=cutlass.Int16, count=4)
                                                 for e in cutlass.range_constexpr(4):
                                                     buf[t * 4 + e] = v[e].bitcast(Float16).to(Float32)
                                             for t in cutlass.range_constexpr(warp_threads):
                                                 f = prims.shfl_sync(
                                                     thread_mask=0xFFFFFFFF,
                                                     val=fv,
                                                     offset=t,
                                                     mask_and_clamp=0x1F,
                                                     kind=prims.Shfl.IDX,
                                                     return_value_and_is_valid=False,
                                                 )
                                                 for e2 in cutlass.range_constexpr(0, 4, 2):
                                                     p2 = ffma2(
                                                         (f, f),
                                                         (buf[t * 4 + e2], buf[t * 4 + e2 + 1]),
                                                         (acc[e2], acc[e2 + 1]),
                                                     )
                                                     acc[e2] = p2[0]
                                                     acc[e2 + 1] = p2[1]
                                             pk4 = cutlass.Array(cutlass.Int16, 4)
                                             for e in cutlass.range_constexpr(4):
                                                 pk4[e] = (acc[e] * rinv).to(Float16).bitcast(cutlass.Int16)
                                             if cutlass.const_expr(d_lanes == 128):
                                                 prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), obb)
                                             else:
                                                 if lane < d_lanes // 4:
                                                     prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), obb)
                                     else:
                                         # Splits 32..63 fold in a second batch off
                                         # the second (m, l) pair.
                                         for i in cutlass.range(nrep):
                                             bb = blk0 + i * bstep
                                             jbb = jb + bb * v_slice_d
                                             obb = ob + bb * v_slice_d
                                             for e in cutlass.range_constexpr(4):
                                                 acc[e] = Float32(0.0)
                                             for t in cutlass.range_constexpr(warp_threads):
                                                 j = Int32(0) + t
                                                 if j >= live_splits:
                                                     j = Int32(0)
                                                 v = prims.load_ext(jbb + j * D, dtype=cutlass.Int16, count=4)
                                                 for e in cutlass.range_constexpr(4):
                                                     buf[t * 4 + e] = v[e].bitcast(Float16).to(Float32)
                                             for t in cutlass.range_constexpr(warp_threads):
                                                 f = prims.shfl_sync(
                                                     thread_mask=0xFFFFFFFF,
                                                     val=fv,
                                                     offset=t,
                                                     mask_and_clamp=0x1F,
                                                     kind=prims.Shfl.IDX,
                                                     return_value_and_is_valid=False,
                                                 )
                                                 for e2 in cutlass.range_constexpr(0, 4, 2):
                                                     p2 = ffma2(
                                                         (f, f),
                                                         (buf[t * 4 + e2], buf[t * 4 + e2 + 1]),
                                                         (acc[e2], acc[e2 + 1]),
                                                     )
                                                     acc[e2] = p2[0]
                                                     acc[e2 + 1] = p2[1]
                                             for t in cutlass.range_constexpr(warp_threads):
                                                 j = Int32(warp_threads) + t
                                                 if j >= live_splits:
                                                     j = Int32(0)
                                                 v = prims.load_ext(jbb + j * D, dtype=cutlass.Int16, count=4)
                                                 for e in cutlass.range_constexpr(4):
                                                     buf[t * 4 + e] = v[e].bitcast(Float16).to(Float32)
                                             for t in cutlass.range_constexpr(warp_threads):
                                                 f = prims.shfl_sync(
                                                     thread_mask=0xFFFFFFFF,
                                                     val=fv2,
                                                     offset=t,
                                                     mask_and_clamp=0x1F,
                                                     kind=prims.Shfl.IDX,
                                                     return_value_and_is_valid=False,
                                                 )
                                                 for e2 in cutlass.range_constexpr(0, 4, 2):
                                                     p2 = ffma2(
                                                         (f, f),
                                                         (buf[t * 4 + e2], buf[t * 4 + e2 + 1]),
                                                         (acc[e2], acc[e2 + 1]),
                                                     )
                                                     acc[e2] = p2[0]
                                                     acc[e2 + 1] = p2[1]
                                             pk4 = cutlass.Array(cutlass.Int16, 4)
                                             for e in cutlass.range_constexpr(4):
                                                 pk4[e] = (acc[e] * rinv).to(Float16).bitcast(cutlass.Int16)
                                             if cutlass.const_expr(d_lanes == 128):
                                                 prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), obb)
                                             else:
                                                 if lane < d_lanes // 4:
                                                     prims.store_ext(pk4.data_ptr().load(count=4, alignment=2), obb)


    @cute.jit
    def launch(
        pQ: cute.Pointer,
        pKV: cute.Pointer,
        pPT: cute.Pointer,
        pSL: cute.Pointer,
        pSNK: cute.Pointer,
        pO: cute.Pointer,
        pWS: cute.Pointer,
        pWH: cute.Pointer,
        pSEM: cute.Pointer,
        B: Int32,
        HKV: Int32,
        NP2: Int32,
        MAXP: Int32,
        gqa: Int32,
        splits: Int32,
        csplit: Int32,
        window: Int32,
        has_sink: Int32,
        hk_shift: Int32,
        qk_scale: Float32,
        v_scale: Float32,
        stream: cuda_drv.CUstream,
        pSKT: cute.Pointer,
        skt_len: Int32,
    ):
        # The pool stride (NP2, HKV) lives in the TMA descriptor, not in kernel
        # code, so a per-call rebuild covers every pool geometry; only the
        # *box* is fixed, and a box extent of 1 on the page and head modes lets
        # the page id and KV head ride in as runtime TMA coordinates.
        # `HKV` is the GRID's fast mode: at fp16 GQA ratio 16 it is twice the
        # pool's KV-head count (one CTA per 8-head group), so the pool geometry
        # -- and only the pool geometry -- uses the un-widened count.
        if cutlass.const_expr(fp8):
            hkv_r = HKV
        else:
            hkv_r = HKV >> hk_shift
        hq = HKV * gqa
        mQ = cute.make_tensor(pQ, cute.make_layout(B * hq * D))
        # Skip-softmax scale table: flattened (seqlen_bin, S) int32 pairs;
        # length 0 = dense.
        mSKT = cute.make_tensor(pSKT, cute.make_layout(skt_len))
        mPT = cute.make_tensor(pPT, cute.make_layout(B * 2 * MAXP))
        mSL = cute.make_tensor(pSL, cute.make_layout(B))
        mSNK = cute.make_tensor(pSNK, cute.make_layout(hq))
        mO = cute.make_tensor(pO, cute.make_layout(B * hq * D))
        mWS = cute.make_tensor(pWS, cute.make_layout(
            (B * hq + pad_n) * (splits * D + 2 * splits)))
        # fp16 alias of the SAME scratch.  The partial-O run occupies the
        # leading `hq_pad * splits * D` floats, so the half-precision view of
        # it ends exactly halfway through that run and never reaches the
        # running-max / colsum arrays that live behind it.
        mWH = cute.make_tensor(pWH, cute.make_layout(
            (B * hq + pad_n) * splits * D))
        if cutlass.const_expr(fp8):
            # One flag line (sem_words int32) per (b, kv-head) group.
            mSEM = cute.make_tensor(pSEM, cute.make_layout(B * HKV * sem_words))
            # TWO-mode view of the pool: `[NP2, 128, HKV, D]` has
            # stride(page) == 128 * stride(token) and stride(head) == D ==
            # extent(D) * stride(D), so both pairs of modes merge and the whole
            # pool is exactly the 2-D matrix `[NP2 * 128 tokens, HKV * D lanes]`.
            # The box is unchanged (128 token rows x 128 B) but each issue hands
            # over two coordinates instead of four.
            mKV = cute.make_tensor(pKV, cute.make_layout(
                (NP2 * tokens_per_page, HKV * D), stride=(HKV * D, 1)))
            tma_k = ccuda.create_tensor_map_tiled_from_view(
                mKV,
                box_dims=(tokens_per_k_stage, mma_k),
                stride_order=(1, 0),
                # A box row is the whole 128-elem token row: 128 e4m3 = 128 B =
                # one s128b period, so a KV page is one box, not two.
                swizzle=sw_tma,
                # 256 B promotion pulls the NEXT KV head's token row into L2,
                # which is what the adjacent-in-grid sibling CTA asks for.
                l2_promotion=ccuda.TensorMapL2Promotion.l2_256b,
            )
            tma_v = tma_k
        else:
            # Two counters per (b, head-group): arrivals, then folder departures.
            mSEM = cute.make_tensor(pSEM, cute.make_layout(2 * B * HKV))
            mKV = cute.make_tensor(pKV, cute.make_layout(
                (NP2, tokens_per_page, hkv_r, D),
                stride=(tokens_per_page * hkv_r * D, hkv_r * D, D, 1)))
            tma_k = ccuda.create_tensor_map_tiled_from_view(
                mKV,
                box_dims=(1, tokens_per_k_stage, 1, mma_k),
                stride_order=(3, 2, 1, 0),
                swizzle=sw_tma,
                # A box row is 64 halves = 128 B, but the two d-halves of a
                # token are adjacent in the pool, so a 256 B promotion turns the
                # second half of every token into a hit instead of a second DRAM
                # trip.  At D = 32 a row is 64 B and the promoted neighbour is
                # the NEXT KV head's slice, which the adjacent CTA of the fast
                # grid mode is fetching at the same time.
                l2_promotion=ccuda.TensorMapL2Promotion.l2_256b,
            )
            # V rides ONE box per stage where a stage carries two d-subtiles:
            # a fifth (extent-2, stride-64) d-slice mode folds both subtiles
            # into a single un-swizzled TMA box, so the smem stage becomes
            # token-major 256 B rows and the per-stage box count halves.
            # BMM2's descriptor walks that layout M-contiguously.
            tma_v = tma_k
            if cutlass.const_expr(v_boxes == 2):
                mKV5 = cute.make_tensor(pKV, cute.make_layout(
                    (NP2, tokens_per_page, hkv_r, D // mma_k, mma_k),
                    stride=(tokens_per_page * hkv_r * D, hkv_r * D, D, mma_k, 1)))
                tma_v = ccuda.create_tensor_map_tiled_from_view(
                    mKV5,
                    box_dims=(1, tokens_per_v_stage, 1, 2, mma_k),
                    # token BEFORE dslice in map order: the box fills smem
                    # subtile-major, byte-identical to the old two-box fill
                    # (row phase = token & 7 either way), so BMM2's descriptor
                    # is untouched.
                    stride_order=(4, 1, 2, 3, 0),
                    swizzle=sw_tma,
                    l2_promotion=ccuda.TensorMapL2Promotion.l2_256b,
                )
        # fp16: `qk_scale` is the plain 1/sqrt(D) and log2(e) folds in here.
        # e4m3: the host already folded k_scale * log2(e) / sqrt(D); the device
        # multiplies in the per-head Q dequant scalar.
        if cutlass.const_expr(fp8):
            # When the plan deals <= _mc(D) splits the splits of one (b, kv-head)
            # group launch as one CTA cluster (csplit == splits) so the split-K
            # join can ride the cluster interconnect instead of L2; wider plans
            # launch unclustered (csplit == 1) and keep the global join.
            decode(
                mQ, mPT, mSL, mSNK, mO, mWS, mWH, mSEM, mSKT, tma_k, tma_v,
                qk_scale, v_scale, MAXP, gqa, window, has_sink, hk_shift,
            ).launch(
                grid=[HKV, splits, B], block=[256, 1, 1],
                cluster=[1, csplit, 1], stream=stream,
            )
        else:
            decode(
                mQ, mPT, mSL, mSNK, mO, mWS, mWH, mSEM, mSKT, tma_k, tma_v,
                qk_scale * Float32(log2_e), v_scale, MAXP, gqa, window, has_sink, hk_shift,
            ).launch(grid=[HKV, splits, B], block=[256, 1, 1], stream=stream)

    return launch


# Tree-attention body
def _cfg(D, kv_dtype=KV_DTYPE_DEFAULT):
    """Per-(head dim, KV dtype) trace-time constants shared by the kernel and the host plan."""
    assert kv_dtype in ("fp16", "e4m3"), kv_dtype
    e4 = kv_dtype == "e4m3"
    if e4:
        assert D in (64, 128, 256, 512), f"e4m3 KV needs D in (64, 128, 256, 512), got {D}"
    else:
        assert D in (32, 64, 128, 256, 512), D
    wide = D >= 256 and not e4
    c = {"wide": wide, "D": D, "e4m3": e4}
    c["elem_bytes"] = 1 if e4 else 2
    # Elements of one d-slice row == one TMA swizzle period: 128 B (64 B at the
    # smallest head dim of each dtype).
    c["mma_k"] = (128 if D >= 128 else 64) if e4 else min(D, 64)
    c["k_slices"] = D // c["mma_k"]      # K d-slices per page
    c["k_granule"] = 32 if e4 else 16    # keys / d-elements per MMA K step
    # e4m3 D > 128 shards the head dim over `dsh_ways` sibling CTAs at trace
    # time so the 128-row tile's O pair still fits the 512-column TMEM; `Dc`
    # is the O width one CTA owns (the whole D everywhere else).
    c["dsh_ways"] = max(1, D // 128) if e4 else 1
    c["Dc"] = D // c["dsh_ways"]
    c["v_boxes"] = 1 if e4 else (2 if D >= 128 else 1)  # `mma_k`-wide TMA boxes per V stage
    c["v_slices"] = D // 128 if wide else 1   # V stages per page: 1 narrow, else D // 128
    if e4:
        c["rows_tile"] = 128
        c["cta_threads"] = 384
        # One e4m3 d-slice is half the bytes of the fp16 stage it replaces;
        # D = 512 stages 64 KiB of Q, so its rings are one shallower.
        c["kstg"] = 6 if D < 512 else 5
        c["v_stages"] = 4 if D < 512 else 3
        c["pstg"] = 0
        c["spec_chunks"] = 4
        c["narrow_depth"] = 3
        c["msg_wide"] = 8
        c["join_blk"] = 8
        c["jblocks"] = 2
        c["max_join_splits"] = 16
        c["l2_group_pages"] = 768
        c["can_dsplit"] = False
    elif wide:
        c["rows_tile"] = 64
        c["cta_threads"] = 256           # 4 producer/MMA warps + ONE softmax WG
        # K ring depth.  At D = 512 a page is eight 16 KiB d-slices and smem is
        # already at the 227 KiB ceiling, so the ring stays at four.  At D = 256
        # a page is exactly FOUR slices, i.e. the four-stage ring holds one page
        # and not a byte more: the producer can never issue into page p+1 while
        # BMM1 still walks page p; the fifth stage is 16 KiB of D = 256 headroom.
        c["kstg"] = 5 if D == 256 else 4
        c["v_stages"] = 2                # 32 KiB each: one d-pair slice of a page
        c["pstg"] = 4                    # SMEM P ring (8 KiB each)
        c["spec_chunks"] = 5             # cycle-zero speculative page-id window, in 32-page chunks
        c["narrow_depth"] = 2
        c["msg_wide"] = 4                # admission price for rows wider than one row block
        c["join_blk"] = 9                # partial slots gathered per fold block
        c["jblocks"] = 2
        c["max_join_splits"] = 18        # D = 512 cap; D = 256 stays at clu_ext2
        c["l2_group_pages"] = 768 * 128 // D
        c["can_dsplit"] = D == 512
    else:
        c["rows_tile"] = 128
        c["cta_threads"] = 384           # 4 producer/MMA warps + TWO softmax WGs
        # Pipeline depths are cut for the worst case against the 227 KiB smem
        # ceiling.  D = 128: 16 KiB K d-halves and a 32 KiB V page; D = 64 / 32:
        # a stage is one WHOLE page (16 / 8 KiB), so every stage the D = 128
        # build spent on a d-half is a page of walk depth here.
        c["kstg"] = 5 if D == 128 else (6 if D == 64 else 8)
        c["v_stages"] = 2 if D == 128 else (4 if D == 64 else 6)
        c["pstg"] = 0
        c["spec_chunks"] = 4
        c["narrow_depth"] = 3
        c["msg_wide"] = 8
        c["join_blk"] = 8
        c["jblocks"] = 2
        c["max_join_splits"] = 16
        c["l2_group_pages"] = 768
        c["can_dsplit"] = False
    return c


def _build_tree(D=D_DEFAULT, kv_dtype=KV_DTYPE_DEFAULT):
    """Spec-decode tree attention: the `@cute.jit` launcher for ONE (head dim, KV dtype).

    The query rows (S * gqa per (b, kv-head) group) are the MMA M axis and the
    keys the N axis.  Two row tilings live in the same kernel body behind
    trace-time branches:

      * "narrow" (fp16 D <= 128 and EVERY e4m3 D): 128-row tiles, one query row
        per TMEM lane and per softmax thread, TWO softmax warpgroups, P stored in
        place over S in TMEM and fed to BMM2 through the `[a-tmem]` form.
      * "wide" (fp16 D >= 256): 64-row tiles (tcgen05 M = 64, Layout F), ONE
        softmax warpgroup, P published through an SMEM ring, `D // 64` K
        d-slices streamed dk-outer.  D = 512 may run a runtime head-dim shard
        (`DSH`): a second CTA per (group, split) takes half the head dim.

    e4m3 KV: `kind::f8f6f4` MMAs, Q quantised to e4m3 per row in the staging
    warps, P quantised to e4m3 for BMM2, `k_scale` folded into the QK scale on
    the host and `v_scale` applied once on the normalised O.  D > 128 shards the
    head dim over `D // 128` sibling CTAs at trace time.

    D and kv_dtype are the only baked axes: they set the row tiling, the K/V
    d-slice counts, the MMA kind, the Q staging width, the O column budget and
    the join fold width.  Everything else -- B, S, HQ, HKV, the ratio, MAXP,
    window, contiguous, has_sink, both scales and the split plan -- stays a
    runtime scalar, so ONE compile serves every workload row of that pair.

    Launch contract (both dtypes): `DSH` is the runtime head-dim shard shift.
    fp16 D = 512 honours it (0 or 1).  e4m3 bakes its shard count from D and
    IGNORES `DSH`; the host passes `log2(D // 128)` so the grid / semaphore
    bookkeeping it implies matches what the kernel does.  fp16 passes
    `v_scale == 1` and never applies it.
    """
    # Tree body constants
    n_tok = 64         # keys per S tile = N of BMM1 / K of BMM2
    # Hysteresis (log2 domain) on the running softmax max: the max is only a
    # reference point, so holding it until a page max jumps this far past it is
    # EXACT math -- it just lets P grow to at most 2^max_hyst (4096 << fp16's
    # 65504) and turns the O-rescale into a once-per-walk rarity.
    max_hyst = 12.0
    # e4m3 P has a 448 ceiling and a 2^-9 subnormal floor, so the reference is
    # carried `max_drop_e4m3` BELOW the tile max (P <= 2^8 == 256) and a re-trigger
    # parks it `max_hyst_e4m3` above the tile that raised it.
    max_hyst_e4m3 = 4.0
    max_drop_e4m3 = 8.0
    # Hybrid exp2: the softmax's 64 MUFU.EX2 per thread per tile sit on the SFU
    # pipe (8 cycles/warp-inst) while the FMA pipe idles beside them.  The LAST
    # `poly_keys` keys of every tile evaluate exp2 as a degree-3 polynomial on the
    # FMA pipe instead: t = x + 1.5*2^23 rounds x to the nearest int in t's low
    # mantissa bits, p ~= 2^f for the remainder f in [-0.5, 0.5], and an integer
    # add of (t << 23) onto p's bit pattern applies the 2^round(x) scale.  Max
    # relative error 1.4e-4 over [-126, 12] -- under fp16 P's own 4.9e-4
    # quantisation.  The clamp at -126 keeps the exponent field in range; masked
    # keys (x ~ -1e37) come out as 2^-126, which fp16 flushes to the exact 0 the
    # MUFU path produced.  Hysteresis bounds x above at +12, so no upper clamp.
    exp_magic = 12582912.0
    exp_c0 = 0.9999513472
    exp_c1 = 0.6932530361
    exp_c2 = 0.2422568020
    exp_c3 = 0.0550293090
    poly_keys = 32

    # Mask sentinel.  A fully-masked S tile must NOT be able to move the running
    # max (P would then come out as exp2(0) == 1 for masked keys).  `neg_big` is
    # far enough below zero that `neg_big * scale` still clamps to `neg_floor`,
    # which is exactly the running max's initial value -- so an all-masked tile is
    # a no-op on the max and contributes exp2(-1.3e37) == 0 to the colsum.
    neg_big = -1.0e38
    neg_floor = -1.0e30
    # Signed 32-bit single-bit constants, so the per-key mask test is one LOP3
    # against an immediate instead of a shift plus an AND.
    _bit32 = [((1 << n) - (1 << 32) if n == 31 else (1 << n)) for n in range(32)]

    s_stages = 4       # TMEM S buffers (4*64 S columns; O takes the rest of the 512-col alloc)
    min_split_gain = 3            # pages of walk depth a split-K deal must save
    # A split-K deal buys walk depth and pays a join.  The join is a fixed ~2.4 us
    # -- release fence for partial-O visibility, GPU-scope ACQ_REL atomic, gmem
    # re-read of every partial -- while a page of walk costs ~0.6 us, so dealing a
    # sequence thinner than `narrow_depth` pages per CTA buys less than the join
    # it adds.  Deals wider than `narrow_splits` are therefore only cut for
    # sequences deep enough to pay for them; everything shorter is dealt
    # `narrow_splits` ways and walks deeper.
    narrow_splits = tree_narrow_splits
    join_batch = 4                # splits whose partials are in flight together
    max_splits = 8                # ceiling on the device-side split count
    min_join_ctas = 32            # ...relaxed when the group count alone is too narrow

    tmem_cols = 512               # allocation granularity is a power of two: S + O round to 512
    o_col_offset = s_stages * n_tok

    # barrier ids
    softmax_sync_barrier = 1                           # WG A internal sync (WG B uses +3 = 4)
    o_zero_barrier = 2                           # "O accumulator zeroed" handshake
    join_barrier = 3                           # CTA-wide release into the split-K fold
    wg_pair_barrier = 5                        # both softmax WGs (256 threads): merge + dealloc
    kv_init_barrier = 6                        # warp 0 -> warp 3: KV mbarriers live (wide only)

    # Thread-block-cluster extents for the cluster-barrier join (wide only): a cluster is a hardware
    # residency group (8 CTAs portable, 16 non-portable on SM100).
    clu_ext = 8
    clu_ext2 = tree_clu_ext2

    # One split-K arrival counter per 128 B line.  Every CTA of a group hammers the
    # same counter with a GPU-scope ACQ_REL atomic, and packing the groups' counters
    # into one line made every group's arrivals serialise against every other
    # group's at the same L2 sector -- on the narrow rows that is dozens of
    # round-trip-latency atomics on the join's critical path.
    sem_stride = tree_sem_stride

    # Walk depth (pages per CTA) at or above which the narrow BMM1 switches to the
    # page-fused N == 128 issue form.  Below it the pipeline never reaches steady
    # state and the extra half-BMM1 of fill latency costs more than the halved Q
    # re-reads save.
    fuse_depth = 6

    cfg = _cfg(D, kv_dtype)
    wide = cfg["wide"]
    kv_e4m3 = cfg["e4m3"]
    kv_t = e4m3 if kv_e4m3 else Float16
    elem_bytes = cfg["elem_bytes"]
    mma_kind = prims.Tcgen05MMAKind.F8F6F4 if kv_e4m3 else prims.Tcgen05MMAKind.F16
    mma_k = cfg["mma_k"]
    k_slices = cfg["k_slices"]
    k_granule = cfg["k_granule"]
    dsh_ways = cfg["dsh_ways"]
    dsh_log = dsh_ways.bit_length() - 1
    Dc = cfg["Dc"]
    v_boxes = cfg["v_boxes"]
    v_slices = cfg["v_slices"]
    rows_tile = cfg["rows_tile"]
    cta_threads = cfg["cta_threads"]
    kstg = cfg["kstg"]
    v_stages = cfg["v_stages"]
    pstg = cfg["pstg"]
    spec_chunks = cfg["spec_chunks"]
    spec_pages = spec_chunks * page_chunk
    narrow_depth = cfg["narrow_depth"]
    msg_wide = cfg["msg_wide"]
    join_blk = cfg["join_blk"]
    jblocks = cfg["jblocks"]
    l2_group_pages = cfg["l2_group_pages"]
    can_dsplit = cfg["can_dsplit"]

    # A row of the swizzled K/V/Q tile is `mma_k` elements == one swizzle
    # period: 128 B (the 128 B swizzle atom) or 64 B (fp16 D = 32, e4m3
    # D = 64), which is the 64 B TMA swizzle `Swizzle<2,4,3>` -- MMA layout
    # code 4, and an SBO of eight rows (512 B) instead of 1024 B.  The
    # hand-staged Q tile carries the matching XOR: `chunk ^ (row & 7)` at 128 B
    # rows, `chunk ^ ((row >> 1) & 3)` at 64 B.
    row_bytes = mma_k * elem_bytes
    q_chunks = row_bytes // 16           # 16 B chunks along d in one Q d-slice row
    sw_shift = 0 if row_bytes == 128 else 1
    sw_mask = 7 if row_bytes == 128 else 3
    smem_layout = cutlass.Int8(2) if row_bytes == 128 else cutlass.Int8(4)
    sbo = 8 * row_bytes                  # 1024 / 512
    dchunks = D // 8                     # 16 B pieces of one output row (== n_chunks_d)
    n_chunks_d = dchunks
    nchunk = Dc // 8                     # narrow: 16 B pieces of one CTA's partial row
    o_half_n = Dc // 2                   # narrow: O columns one softmax WG folds

    k_stage_elems = tokens_per_k_stage * mma_k
    v_subtile_elems = tokens_per_v_stage * mma_k
    v_stage_elems = v_boxes * v_subtile_elems
    q_stage_elems = rows_tile * mma_k
    p_stage_elems = rows_tile * n_tok               # wide only: 4096 elems = 8 KiB per P tile
    k_stage_bytes = k_stage_elems * elem_bytes
    v_stage_bytes = v_stage_elems * elem_bytes
    v_subtile_bytes = v_subtile_elems * elem_bytes
    tile_byte_step = n_tok * row_bytes       # byte offset of S tile 1 inside a K stage
    tiles_per_page = tokens_per_page // n_tok
    # MMA K-step geometry.  One K granule (`k_granule` elements) of a K-major
    # operand is 32 B == 2 descriptor start-address units for both dtypes; the
    # MN-major V operand steps `k_granule` tokens x `row_bytes`; the `[a-tmem]`
    # P operand steps 8 fp32 columns (32 B) per granule.
    kdesc_kstep = k_granule * elem_bytes // 16
    v_kblk_step = k_granule * row_bytes // 16
    n_mk1 = mma_k // k_granule           # BMM1 K steps per d-slice
    n_mk2 = n_tok // k_granule           # BMM2 K steps per S tile

    # Wide-only geometry: `D // 128` 64-column O blocks (d-halves packed at lane
    # alignments 0 / 16), the join fold staging and the head-dim shard.
    o_blocks = v_slices
    o_half = max(1, v_slices // 2)
    jslots = ((jblocks * join_blk) + 3) // 4 * 4 if wide else 2 * join_blk
    # Split 0's cycle-zero speculative page-0 KV issue needs the whole page to
    # fit fresh pipeline stages: true everywhere but D = 512 (8 K slices > 4 stages).
    spec_page0 = k_slices <= kstg and v_slices <= v_stages
    # Pipeline indices after that speculative issue: the ring lands at
    # `slices % stages` with the phase flipped once per full wrap.
    k_pre_idx = k_slices % kstg
    k_pre_ph = 1 ^ ((k_slices // kstg) & 1)
    v_pre_idx = v_slices % v_stages
    v_pre_ph = 1 ^ ((v_slices // v_stages) & 1)
    # The narrow Q staging pulls a whole row per thread: `k_slices * q_chunks` 16 B pieces.
    q_pieces = k_slices * q_chunks
    arrive0_count = 224 if wide else cta_threads
    magic_narrow = _magic(narrow_splits)
    magic_jblk = _magic(join_blk)

    @cute.kernel
    def decode(
        mQ: cute.Tensor,
        mPT: cute.Tensor,
        mSL: cute.Tensor,
        mSNK: cute.Tensor,
        mO: cute.Tensor,
        mWS: cute.Tensor,
        mWH: cute.Tensor,
        mSEM: cute.Tensor,
        mTM: cute.Tensor,
        tma_k: cutlass.GridConstant[ccuda.TensorMap],
        scale: Float32,
        v_scale: Float32,
        MAXP: Int32,
        gqa_ratio: Int32,
        window: Int32,
        CQ: Int32,
        has_sink: Int32,
        S: Int32,
        MW: Int32,
        RB: Int32,
        DSH: Int32,
        RBM: Int32, GQM: Int32, SPLM: Int32,
    ):
        h_k_idx, split_idx, z_idx = cute.arch.block_idx()
        # B / HKV / SPLITS ride in for free as grid dimensions instead of as
        # extra dispatch scalars (host arg marshalling is on the critical path
        # for the small rows).  HKV is the FAST grid mode: the KV-head CTAs of
        # one (b, split) walk the very same page ids, each taking a different
        # slice of the same token row, so dispatching them adjacently puts
        # their requests on the same DRAM rows at the same time.
        HKV, SPLITS, zdim = cute.arch.grid_dim()
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        wg_widx = warp_idx % 4                       # position inside warpgroup
        if cutlass.const_expr(wide):
            # Layout F row deal: alignment-0 puts row r on lane (r & 15) + 32 * (r >> 4); the
            # alignment-16 partner holds the same rows 16 lanes up.
            row = wg_widx * 16 + (lane & 15)         # query row this thread owns
            khalf = lane >> 4                        # which lane-alignment half
        else:
            row = wg_widx * warp_threads + lane      # TMEM row this thread owns

        # `S * gqa` query rows per (b, kv-head) group are dealt into row blocks
        # of `rows_tile`; the block rides on the SLOW grid mode so the KV-head
        # CTAs that share page ids stay adjacent.  The head-dim shard (D = 512)
        # is the INNERMOST bit of z, so the shards' duplicated K stream is one
        # L2 fill.  `RBM`/`GQM`/`SPLM` are host-built ceil(2**16 / d) magics
        # for the runtime divisors (RB, gqa, splits): `b_idx` addresses
        # `seq_lens` and the page table, so its division sat at the very head
        # of the cycle-zero critical path.
        ds = Int32(0)
        zt = z_idx
        ztdim = zdim
        if cutlass.const_expr(can_dsplit):
            ds = z_idx & DSH
            zt = z_idx >> DSH
            ztdim = zdim >> DSH
        elif cutlass.const_expr(dsh_ways > 1):
            # e4m3 D > 128: trace-time shard, still the innermost bit(s) of z
            # so the siblings' duplicated K stream is one L2 fill.
            ds = z_idx & Int32(dsh_ways - 1)
            zt = z_idx >> dsh_log
            ztdim = zdim >> dsh_log
        b_dim = (ztdim * RBM) >> 16
        b_idx = (zt * RBM) >> 16
        rb = zt - b_idx * RB
        HQ = HKV * gqa_ratio
        rows_total = S * gqa_ratio
        row_base = rb * rows_tile
        groups = b_dim * HKV
        if cutlass.const_expr(wide):
            coord_hb = ((((b_idx * HKV + h_k_idx) * RB + rb) << DSH) + ds) * sem_stride
            # Shard geometry, CTA-uniform: `Dc_r` columns from `vbase0`, `v_live` d-pair stages
            # into `ob_live` O blocks (`oh_live` per alignment).  `DSH == 0` gives the full width.
            Dc_r = Int32(D) >> DSH
            v_live = Int32(v_slices) >> DSH
            ob_live = Int32(o_blocks) >> DSH
            oh_live = v_live >> 1
            nc_live = Int32(n_chunks_d) >> DSH
            vbase0 = ds * Dc_r
        elif cutlass.const_expr(dsh_ways > 1):
            coord_hb = (
                (((b_idx * HKV + h_k_idx) * RB + rb) * Int32(dsh_ways) + ds) * sem_stride
            )
            vbase0 = ds * Int32(Dc)              # this shard's first head-dim column
        else:
            coord_hb = ((b_idx * HKV + h_k_idx) * RB + rb) * sem_stride

        # Smem
        tmem_holder = cutlass.Array(
            cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4
        )
        kbar = cutlass.Array(
            cutlass.Int64, kstg * 2, space=cutlass.AddressSpace.smem, alignment=8
        )
        vbar = cutlass.Array(
            cutlass.Int64, v_stages * 2, space=cutlass.AddressSpace.smem, alignment=8
        )
        sbar = cutlass.Array(
            cutlass.Int64, s_stages * 2, space=cutlass.AddressSpace.smem, alignment=8
        )
        # Wide: SMEM P ring full/free pairs (64 arrivals per stage: the low lanes publish
        # tile 0's stage, the high lanes tile 1's).  Narrow: P-ready, one per S stage (128
        # arrivals: the owning softmax WG's threads, once their P rows are stored to TMEM);
        # BMM2 hands the whole stage back through sbar's free side.
        pbar = cutlass.Array(
            cutlass.Int64, (pstg * 2) if wide else s_stages,
            space=cutlass.AddressSpace.smem, alignment=8
        )
        # [0]/[1]: WG A's rescale drain-commit / rescale-done pair; [3]/[4]:
        # the same pair for WG B (narrow only; each O accumulator has its own
        # rare max-jump events); [2]: the single terminal drain the epilogue O
        # reads wait on.
        obar = cutlass.Array(
            cutlass.Int64, 5, space=cutlass.AddressSpace.smem, alignment=8
        )
        if cutlass.const_expr(not wide):
            # Per-row (running max, colsum) exchange between the two softmax
            # WGs, written after each WG's walk and read by the other side so
            # the epilogue can merge the two O accumulators exactly.
            sRed = cutlass.Array(
                Float32, 2 * rows_tile * 2, space=cutlass.AddressSpace.smem, alignment=16
            )
        # Per-P-stage rescale flag, published by the softmax warpgroup
        # alongside P: the O-rescale handshake with BMM2 fires ONLY on the
        # rare max-jump page.
        sChg = cutlass.Array(
            cutlass.Int32, pstg if wide else s_stages,
            space=cutlass.AddressSpace.smem, alignment=16
        )
        # Per-warp rescale votes, published on a split arrive/wait mbarrier so
        # the whole warpgroup rendezvous hides under the exp block.  Two slots
        # per WG (ping-pong): a warp always passes its own wait on a slot before
        # re-arriving it, so phase reuse is ordered by program order alone.
        sV4 = cutlass.Array(
            cutlass.Int32, 4 * 4, space=cutlass.AddressSpace.smem, alignment=16
        )
        cbar = cutlass.Array(
            cutlass.Int64, 4, space=cutlass.AddressSpace.smem, alignment=8
        )
        sFlag = cutlass.Array(
            cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16
        )
        sK = cutlass.Array(
            kv_t, k_stage_elems * kstg, space=cutlass.AddressSpace.smem, alignment=1024
        )
        sV = cutlass.Array(
            kv_t, v_stage_elems * v_stages, space=cutlass.AddressSpace.smem, alignment=1024
        )
        sQ = cutlass.Array(
            kv_t, q_stage_elems * k_slices, space=cutlass.AddressSpace.smem, alignment=1024
        )
        if cutlass.const_expr(wide):
            sP = cutlass.Array(
                Float16, p_stage_elems * pstg, space=cutlass.AddressSpace.smem, alignment=1024
            )
        if cutlass.const_expr(kv_e4m3):
            # Per-ROW Q dequant scale (softmax scale * rowabsmax / 448), published
            # by the staging warps and read by the softmax thread that owns the row.
            sQS = cutlass.Array(
                Float32, rows_tile, space=cutlass.AddressSpace.smem, alignment=16
            )

        desc_k = tma_k.get_ptr()

        # `seq_lens` is issued HERE, at cycle zero, but first consumed only
        # after the whole seq_lens-independent prologue below: the scoreboard
        # stall lands on `live_pages`, so the mbarrier builds, the TMEM alloc,
        # the Q staging and the speculative page-table window all execute in
        # the shadow of this load instead of behind it.
        seq_len = mSL.iterator[b_idx]

        # Speculative page-id fetch
        # The page ids sit behind a TWO-deep dependent global chain
        # (`seq_lens` -> live_pages -> pages_per_cta -> p0 -> page id).  The
        # split walk deals every split a window inside [0, live_pages), so the
        # row's FIRST `spec_pages` page ids -- an address known at cycle zero --
        # cover every split's window whatever the plan turns out to be.  Warp 0
        # fetches that window immediately, alongside `seq_lens`; only the
        # wide-window rows still pay the dependent load.  Lanes past the live
        # window hold a page id that is never shuffled out, so the clamp is only
        # there to keep every probe inside this batch's own [2 * MAXP] row.
        kp_g = Int32(0)
        vp_g = Int32(0)
        vp_g3 = Int32(0)
        kw1 = Int32(0)
        kw2 = Int32(0)
        kw3 = Int32(0)
        kw4 = Int32(0)
        vw1 = Int32(0)
        vw2 = Int32(0)
        vw3 = Int32(0)
        vw4 = Int32(0)
        v3w1 = Int32(0)
        v3w2 = Int32(0)
        v3w3 = Int32(0)
        v3w4 = Int32(0)
        # Decode re-reads the SAME live KV set on every step, so the right L2
        # policy depends on whether that set can survive in the 126 MB L2
        # between calls.  Small rows keep their whole pool resident under
        # evict_last; the wide rows cannot retain anything, so protecting those
        # dead lines only makes incoming fills fight for victims.  Both policy
        # registers are built here (pure ALU); the WALK picks one per row once
        # the windowed live page count is known.
        l2pol = prims.inline_ptx(
            "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
            write_only_types=[cutlass.Int64],
        )
        l2pol_last = prims.inline_ptx(
            "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;",
            write_only_types=[cutlass.Int64],
        )
        if cutlass.const_expr(wide):
            # Half-fraction evict_last for the rows that re-read a line (row-block
            # siblings, head-dim shards) but whose live set cannot fit the L2 anyway.
            l2pol_half = prims.inline_ptx(
                "createpolicy.fractional.L2::evict_last.b64 $0, 0.5;",
                write_only_types=[cutlass.Int64],
            )
        if cutlass.const_expr(not wide):
            # TMEM allocation at CYCLE ZERO -- but only for WIDE trees (S * gqa
            # > 63): behind the plan the alloc sat serially after the seq_lens
            # round trip; hoisted, it overlaps that trip.  Narrow trees measured
            # +3..5 % under the hoist (dead splits touching the allocator), so
            # the row count picks the placement; it is a kernel argument, so the
            # branch is uniform.
            wide_alloc = rows_total > 63
            if warp_idx == 1:
                if wide_alloc:
                    prims.tcgen05_alloc(tmem_holder, Int32(tmem_cols))
                    prims.tcgen05_relinquish_alloc_permit()
        if warp_idx == 0:
            prims.prefetch_tensormap(desc_k)
            gi = lane
            gcap = MAXP - 1
            if gi > gcap:
                gi = gcap
            gpt = mPT.iterator + (b_idx * 2) * MAXP
            kp_g = gpt[gi]
            vp_g = gpt[MAXP + gi]
            # Chunks 1..spec_chunks-1 of the window, predicated on the capacity
            # so the cap-32 CI rows load exactly what they always did.
            if MAXP > page_chunk:
                gi1 = page_chunk + lane
                if gi1 > gcap:
                    gi1 = gcap
                kw1 = gpt[gi1]
                vw1 = gpt[MAXP + gi1]
            if MAXP > 2 * page_chunk:
                gi2 = 2 * page_chunk + lane
                if gi2 > gcap:
                    gi2 = gcap
                kw2 = gpt[gi2]
                vw2 = gpt[MAXP + gi2]
            if MAXP > 3 * page_chunk:
                gi3w = 3 * page_chunk + lane
                if gi3w > gcap:
                    gi3w = gcap
                kw3 = gpt[gi3w]
                vw3 = gpt[MAXP + gi3w]
            if cutlass.const_expr(spec_chunks > 4):
                if MAXP > 4 * page_chunk:
                    gi4w = 4 * page_chunk + lane
                    if gi4w > gcap:
                        gi4w = gcap
                    kw4 = gpt[gi4w]
                    vw4 = gpt[MAXP + gi4w]
            # cap > spec rows: every non-zero split's own page-table window is a
            # COLD dependent load on its critical path (plan -> pt -> first TMA).
            # Pull the whole [2, MAXP] row into L2 now, one 128 B line per lane.
            pfo = lane << 5
            if pfo < 2 * MAXP:
                prims.prefetch_l2(gpt + pfo)
            # The KV mbarriers and the CTA-wide arrive carry no dependency on
            # `seq_lens` either, so they are built here rather than after the
            # split plan.
            if prims.elect_sync():
                for i in cutlass.range_constexpr(kstg):
                    prims.mbarrier_init(kbar.subview(i).data_ptr(), 1)
                    prims.mbarrier_init(kbar.subview(kstg + i).data_ptr(), 1)
                for i in cutlass.range_constexpr(v_stages):
                    prims.mbarrier_init(vbar.subview(i).data_ptr(), 1)
                    prims.mbarrier_init(vbar.subview(v_stages + i).data_ptr(), 1)
                prims.fence_mbarrier_init()
            prims.bar_warp_sync(0xFFFFFFFF)
            prims.barrier_cta_arrive(0, arrive0_count)
            if cutlass.const_expr(wide):
                # ...plus a 64-thread release to the V producer alone.
                prims.barrier_cta_arrive(kv_init_barrier, 64)
            # Split 0's window starts at page 0 no matter what the plan says,
            # and live_pages >= 1 guarantees page 0 is always inside its window
            # -- so its first page's K and V boxes are issued HERE, gated only
            # on the speculative page-table window landing, never on seq_lens
            # or the plan ALU.  The walk below starts this CTA at page 1 with
            # the pipeline indices pre-advanced.
            # Sliding window moves the walk's first page to
            # `first_page = max(0, seq_len - window) >> 7`; as long as the
            # window start sits inside the register window the speculative
            # issue survives: shuffle from lane `first_page` instead of lane 0.
            # `window == 0` keeps `fp0 == 0` without ever consuming seq_len.
            fp0 = Int32(0)
            if window > 0:
                # Per-row window (XQA CONTIGUOUS_QUERY_SWA): the walk must start
                # at the MINIMUM left edge over the tile's rows, which belongs to
                # draft row 0 at absolute position `seq_len - S`.  With
                # `contiguous == 0` (XQA SPEC_DEC + SLIDING_WINDOW) every row
                # shares ONE left edge at `seq_len - window`.
                ft0 = seq_len - S - window + Int32(1)
                if CQ == 0:
                    ft0 = seq_len - window
                if ft0 < 0:
                    ft0 = Int32(0)
                fp0 = ft0 >> 7
            if cutlass.const_expr(spec_page0):
             if split_idx == 0:
              if fp0 < spec_pages:
                ksel = kp_g
                vsel = vp_g
                fpw = fp0 >> 5
                if fpw == 1:
                    ksel = kw1
                    vsel = vw1
                if fpw == 2:
                    ksel = kw2
                    vsel = vw2
                if fpw == 3:
                    ksel = kw3
                    vsel = vw3
                if cutlass.const_expr(spec_chunks > 4):
                    if fpw == 4:
                        ksel = kw4
                        vsel = vw4
                kp0 = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=ksel,
                    offset=fp0 & Int32(0x1F),
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                    return_value_and_is_valid=False,
                )
                vp0 = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=vsel,
                    offset=fp0 & Int32(0x1F),
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                    return_value_and_is_valid=False,
                )
                if prims.elect_sync():
                    # The WHOLE first page: k_slices K d-slice stages and v_slices V stages
                    # of v_boxes boxes (spec_page0 guarantees they fit the fresh rings).
                    for dk in cutlass.range_constexpr(k_slices):
                        prims.mbarrier_arrive_expect_tx(
                            kbar.subview(dk).data_ptr(), k_stage_bytes
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sK.subview(dk * k_stage_elems).data_ptr(),
                            desc_k,
                            (Int32(dk * mma_k), Int32(h_k_idx), Int32(0), kp0),
                            kbar.subview(dk).data_ptr(),
                            l2_cache_hint=l2pol,
                        )
                    for sk in cutlass.range_constexpr(v_slices):
                        prims.mbarrier_arrive_expect_tx(
                            vbar.subview(sk).data_ptr(), v_stage_bytes
                        )
                        for bx in cutlass.range_constexpr(v_boxes):
                            # An e4m3 shard streams only ITS `Dc` columns, from `vbase0`.
                            vd0 = Int32((sk * v_boxes + bx) * mma_k)
                            if cutlass.const_expr(dsh_ways > 1):
                                vd0 = vbase0 + vd0
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sV.subview(sk * v_stage_elems + bx * v_subtile_elems).data_ptr(),
                                desc_k,
                                (vd0, Int32(h_k_idx), Int32(0), vp0),
                                vbar.subview(sk).data_ptr(),
                                l2_cache_hint=l2pol,
                            )

        # Init warps
        if warp_idx == 2:
            if cutlass.const_expr(wide):
                # Tree-mask words ride a line the KV stream leaves cold: warm the
                # CTA's [S, MW] span at cycle zero (one line per lane covers it).
                tmoff = lane << 5
                if tmoff < S * MW:
                    prims.prefetch_l2(mTM.iterator + b_idx * S * MW + tmoff)
            if prims.elect_sync():
                # The sinks logits are consumed at the very END of the kernel as
                # a dependent gmem load on the terminal critical path; warm the
                # line at cycle zero.  `has_sink == 0` never touches the tensor.
                if has_sink == 1:
                    prims.prefetch_l2(mSNK.iterator + h_k_idx * gqa_ratio)
                for i in cutlass.range_constexpr(s_stages):
                    prims.mbarrier_init(sbar.subview(i).data_ptr(), 1)
                    if cutlass.const_expr(wide):
                        prims.mbarrier_init(sbar.subview(s_stages + i).data_ptr(), warpgroup_threads)
                    else:
                        # The stage-free side is a single tcgen05_commit from
                        # BMM2 (fires when the stage's P MMA group retires).
                        prims.mbarrier_init(sbar.subview(s_stages + i).data_ptr(), 1)
                        prims.mbarrier_init(pbar.subview(i).data_ptr(), warpgroup_threads)
                if cutlass.const_expr(wide):
                    for i in cutlass.range_constexpr(pstg):
                        prims.mbarrier_init(pbar.subview(i).data_ptr(), 64)
                        prims.mbarrier_init(pbar.subview(pstg + i).data_ptr(), 1)
                prims.mbarrier_init(obar.subview(0).data_ptr(), 1)
                prims.mbarrier_init(obar.subview(1).data_ptr(), warpgroup_threads)
                prims.mbarrier_init(obar.subview(2).data_ptr(), 1)
                prims.mbarrier_init(obar.subview(3).data_ptr(), 1)
                prims.mbarrier_init(obar.subview(4).data_ptr(), warpgroup_threads)
                # Vote mbarriers: one elected lane per softmax warp arrives.
                for i in cutlass.range_constexpr(4):
                    prims.mbarrier_init(cbar.subview(i).data_ptr(), 4)
                prims.fence_mbarrier_init()
        # V page ids (warp 3)
        if warp_idx == 3:
            # Warp 3 is the V producer: it holds its own copy of the cycle-zero
            # speculative V page-id window, issued HERE so the page-table load
            # runs in the shadow of the Q staging instead of behind the plan.
            gi3 = lane
            if gi3 > MAXP - 1:
                gi3 = MAXP - 1
            gpt3 = mPT.iterator + (b_idx * 2) * MAXP
            vp_g3 = gpt3[MAXP + gi3]
            if MAXP > page_chunk:
                g31 = page_chunk + lane
                if g31 > MAXP - 1:
                    g31 = MAXP - 1
                v3w1 = gpt3[MAXP + g31]
            if MAXP > 2 * page_chunk:
                g32 = 2 * page_chunk + lane
                if g32 > MAXP - 1:
                    g32 = MAXP - 1
                v3w2 = gpt3[MAXP + g32]
            if MAXP > 3 * page_chunk:
                g33 = 3 * page_chunk + lane
                if g33 > MAXP - 1:
                    g33 = MAXP - 1
                v3w3 = gpt3[MAXP + g33]
            if cutlass.const_expr(spec_chunks > 4):
                if MAXP > 4 * page_chunk:
                    g34 = 4 * page_chunk + lane
                    if g34 > MAXP - 1:
                        g34 = MAXP - 1
                    v3w4 = gpt3[MAXP + g34]

        if cutlass.const_expr(wide):
            # The 16-CTA cluster predicate is needed by the split plan below (a cluster deal must
            # not be collapsed by the join-block admission guard: there, the deal IS the cluster
            # extent), so it is derived here from launch quantities alone.
            clu16 = Int32(0)
            if SPLITS == Int32(clu_ext2):
                if (groups * RB) << DSH <= Int32(4):
                    clu16 = Int32(1)
        # Cost-proportional split plan
        # The page window is cut from the LIVE sequence length, not from the
        # engine capacity MAXP.  `seq_lens` is a device value, so the plan is
        # derived here rather than on the host; every CTA of the group computes
        # the same `live_splits`, which keeps the semaphore join exact.
        live_pages = (seq_len + Int32(tokens_per_page - 1)) >> 7
        if live_pages < 1:
            live_pages = Int32(1)
        # Sliding window: the walk starts at the page holding the first live
        # token, so the leading pages are never planned, loaded or scored.
        # `p0` below is an ABSOLUTE page-table index (first_page + dealt
        # offset).  `window == 0` keeps first_tok == first_page == 0: the plan,
        # the mask and the walk are bit-identical to the windowless kernel.
        # `first_q` is XQA's `firstQueryPosition`: the absolute cache position of
        # draft token 0.  Every key below it is pure context (always attended);
        # keys at or above it are the draft tokens the tree mask selects.
        first_q = seq_len - S
        if first_q < 0:
            first_q = Int32(0)
        first_tok = Int32(0)
        max_left = Int32(0)
        if window > 0:
            # `contiguous == 1`: the minimum per-row edge belongs to draft row 0.
            # `contiguous == 0`: one shared edge for every row.  `max_left` (the
            # MAXIMUM edge, the CTA-uniform mask gate) is `seq_len - window` in BOTH forms.
            ftw = first_q - window + Int32(1)
            if CQ == 0:
                ftw = seq_len - window
            if ftw > 0:
                first_tok = ftw
            mlw = seq_len - window
            if mlw > 0:
                max_left = mlw
        first_page = first_tok >> 7
        live_pages = live_pages - first_page
        # Sequences that already fit one window keep the fused epilogue: their
        # whole join -- the partial-O round trip through the workspace, the
        # GPU-scope semaphore and the fold -- costs more than the pages a
        # split could save.  Otherwise the window is cut as thin as the
        # launched split count allows and every page of slack is handed to an
        # idle SM.
        p0 = first_page
        iters = live_pages
        live_splits = Int32(1)
        if live_pages > min_pages_per_cta:
            # How many windows the sequence is dealt into.  A sequence that
            # would leave every window shallower than `narrow_depth` pages is
            # dealt `narrow_splits` ways instead and walks deeper.  The windows
            # past `deal` simply take no pages, so the launched grid is
            # unchanged and the semaphore still expects `live_splits` arrivals.
            deal = SPLITS
            dm = SPLM
            if live_pages <= narrow_splits * narrow_depth:
                if deal > narrow_splits:
                    deal = Int32(narrow_splits)
                    dm = Int32(magic_narrow)
            # Past `join_blk` live splits the fold gathers a SECOND partial
            # block per d-chunk, so a wider deal is only cut when it buys at
            # least two pages of walk depth over the join_blk-way one.
            if cutlass.const_expr(wide):
                if deal > join_blk and clu16 == 0:
                    d8 = (live_pages + Int32(join_blk - 1)) // join_blk
                    dd = ((live_pages + deal - Int32(1)) * dm) >> 16
                    if d8 - dd < 2:
                        deal = Int32(join_blk)
                        dm = Int32(magic_jblk)
            else:
                # Sequences deeper than 4096 pages (512 K tokens) exceed the
                # magic's exactness window and keep the true division.
                if deal > join_blk:
                    d8 = (live_pages + Int32(join_blk - 1)) >> 3
                    dq = (live_pages * dm) >> 16
                    if live_pages >= 4096:
                        dq = live_pages // deal
                    dd = dq
                    if live_pages - dq * deal != 0:
                        dd = dq + 1
                    if d8 - dd < 2:
                        deal = Int32(join_blk)
                        dm = Int32(magic_jblk)
            # Remainder-dealt windows, not a uniform ceil: `q + (s < r)` pages
            # for split s, so NO dealt split is empty and `live_splits` is
            # `min(deal, live_pages)`.
            q = (live_pages * dm) >> 16
            if cutlass.const_expr(not wide):
                if live_pages >= 4096:
                    q = live_pages // deal
            r = live_pages - q * deal
            # Split-K admission: the deal must SAVE at least `min_split_gain`
            # pages of walk depth over the unsplit walk, else the join costs
            # more than the pages it saves.  The price scales with the ROW
            # count: a wide tree (more than one row block) folds its join over
            # 240-480 rows AND already parallelises across RB CTAs.
            deep = q
            if r != 0:
                deep = q + 1
            msg = Int32(min_split_gain)
            if rows_total > rows_tile:
                msg = Int32(msg_wide)
            if live_pages - deep >= msg:
                iters = q
                p0 = first_page + split_idx * q + r
                if split_idx < r:
                    iters = q + 1
                    p0 = first_page + split_idx * (q + 1)
                if split_idx >= deal:
                    iters = Int32(0)
                live_splits = deal
                if live_pages < deal:
                    live_splits = live_pages
            elif split_idx != 0:
                iters = Int32(0)
        elif split_idx != 0:
            iters = Int32(0)
        # Splits whose window is entirely past the live length clamp to zero
        # iterations: their pipelines never turn, and they skip the epilogue so
        # the semaphore expects exactly `live_splits` arrivals.
        active = iters > 0
        # Every page id the walk needs is already in the cycle-zero register
        # window whenever the window [.., spec_pages) covers it.  Wide: a split
        # needs only ITS OWN window [p0, p0 + iters) inside the register window;
        # narrow: the row's LAST live page has to fit.
        if cutlass.const_expr(wide):
            spec_win = p0 + iters <= spec_pages
        else:
            spec_win = first_page + live_pages <= spec_pages
        # Adaptive KV L2 policy (see the createpolicy pair above).  The row's
        # live working set is groups * live_pages pages; `l2_group_pages` is
        # ~48 MB, comfortably inside the 126 MB L2.  `groups`, `live_pages` are
        # CTA-uniform, so the branch never diverges.
        if cutlass.const_expr(wide):
            # Cluster join: the group's split CTAs are ONE thread block cluster, so the partial-O
            # handshake is a cluster barrier instead of a GPU-scope atomic plus a polled counter.
            clux = Int32(0)
            if SPLITS == Int32(clu_ext):
                if (groups * RB) << DSH <= Int32(clu_ext):
                    clux = Int32(1)
            if clu16 != 0:
                clux = Int32(1)
        polw = l2pol
        over = groups * live_pages > l2_group_pages
        if not over:
            polw = l2pol_last
        # Spec-dec correction: with more than one ROW BLOCK per (b, kv-head)
        # group the sibling block's CTA walks the very same page ids at the
        # very same rate, so every KV line is consumed TWICE per call and only
        # microseconds apart -- never hint the L2 to drop it.
        if RB > 1:
            polw = l2pol_last
            if cutlass.const_expr(wide):
                if over:
                    polw = l2pol_half
        if cutlass.const_expr(wide):
            # Head-dim shards run the very same K walk in lockstep.
            if DSH > 0:
                polw = l2pol_last
                if over:
                    polw = l2pol_half

        # TMEM allocation gated on the split being live: a dead split's
        # alloc/dealloc pair still had to arbitrate for the SM's tensor memory,
        # and on the short rows the dead splits outnumber the live ones.  The
        # CTA-wide barrier below already waits for the Q staging warps, so the
        # alloc still lands long before tmem_base is consumed.  (Narrow wide
        # trees allocated at cycle zero above.)
        if warp_idx == 1:
            if cutlass.const_expr(wide):
                if active:
                    prims.tcgen05_alloc(tmem_holder, Int32(tmem_cols))
                    prims.tcgen05_relinquish_alloc_permit()
            else:
                if rows_total <= 63:
                    if active:
                        prims.tcgen05_alloc(tmem_holder, Int32(tmem_cols))
                        prims.tcgen05_relinquish_alloc_permit()

        # Q stage (softmax warpgroup)
        # Q is staged by the softmax threads that are idle until the first S
        # tile lands rather than by one warp: a hand-swizzled global->smem copy
        # is still cheaper than a third TMA tensormap (each
        # `cuTensorMapEncodeTiled` costs host time on every launch).  Layout
        # reproduced: the TMA swizzle over an 8-row atom, i.e. the physical
        # 16 B chunk index is `chunk ^ swz(row)`.  Row r of the tile is the
        # flattened (draft token, q-head-in-group) pair `row_base + r`, head
        # minor -- the XQA SPEC_DEC row space.
        # Q staging is UNGATED from `active`: every address here is derived
        # from kernel arguments alone, while `active` sits behind the seq_lens
        # load.  Dead splits pull the same Q lines as their live siblings,
        # which L2 dedups.
        q_tok = Int32(0)
        q_hd = Int32(0)
        m_w0 = Int32(0)
        m_w1 = Int32(0)
        r_sink = Float32(0.0)
        r_left = Int32(0)
        if warp_idx >= 4:
            grow = row_base + row
            live_row = grow < rows_total
            gclamp = grow
            if grow >= rows_total:
                gclamp = rows_total - 1
                if gclamp < 0:
                    gclamp = Int32(0)
            q_tok = (gclamp * GQM) >> 16
            q_hd = gclamp - q_tok * gqa_ratio
            if cutlass.const_expr(wide):
                # Staging deal: 128 threads over 64 rows, each thread pulls HALF of one row's
                # d-slices as D/16 contiguous 16 B pieces.  The staging row (qr, half qh) is a
                # DIFFERENT deal from the softmax row: thread t stages row t & 63, d-half t >> 6.
                tq = wg_widx * warp_threads + lane
                qr = tq & 63
                qh = tq >> 6
                sgrow = row_base + qr
                sclamp = sgrow
                if sgrow >= rows_total:
                    sclamp = rows_total - 1
                    if sclamp < 0:
                        sclamp = Int32(0)
                st_tok = (sclamp * GQM) >> 16
                st_hd = sclamp - st_tok * gqa_ratio
                qbase = ((b_idx * S + st_tok) * HQ + h_k_idx * gqa_ratio + st_hd) * D
                # Tree-mask words and sink logit of THIS row, once per CTA into registers.  Issued
                # AHEAD of the Q copy: the copy ends in an async-shared fence the load cannot be
                # hoisted over.
                mrow = mTM.iterator + (b_idx * S + q_tok) * MW
                m_w0 = mrow[0]
                if MW > 1:
                    m_w1 = mrow[1]
                if has_sink == 1:
                    r_sink = mSNK.iterator[h_k_idx * gqa_ratio + q_hd] * Float32(log2_e)
                # Rows at or past `S * gqa` stage the CLAMPED row the address math already produced
                # rather than a zero fill: S / P / O row r depend on Q row r alone, so a padded row
                # just recomputes a live row and the epilogue drops it.
                for it in cutlass.range_constexpr(D // 16):
                    dk = qh * (k_slices // 2) + (it >> 3)
                    qc = it & 7                         # 16 B chunk along d
                    sof = dk * q_stage_elems + qr * mma_k + ((qc ^ (qr & 7)) * 8)
                    gof = qbase + dk * mma_k + qc * 8
                    prims.store_ext(
                        prims.load_ext(mQ.iterator + gof, dtype=Float32, count=4),
                        sQ.subview(sof).data_ptr(),
                    )
                cute.arch.fence_view_async_shared()
            elif cutlass.const_expr(kv_e4m3):
                # Q reaches the MMA as e4m3, quantised per ROW with an absmax; the
                # dequant scalar `scale * absmax / 448` (k_scale and log2e/sqrt(D)
                # already folded into `scale`) goes to the softmax through `sQS`.
                # Both warpgroups stage, a PAIR of adjacent lanes per row (the
                # absmax merge is one shfl.bfly(1)), so a thread holds D/4 raw Q.
                tq = (warp_idx - 4) * warp_threads + lane
                qr = tq >> 1                             # staged row
                qh = tq & 1                              # d-half of that row
                sgrow = row_base + qr
                sclamp = sgrow
                if sgrow >= rows_total:
                    sclamp = rows_total - 1
                    if sclamp < 0:
                        sclamp = Int32(0)
                s_tok = (sclamp * GQM) >> 16
                s_hd = sclamp - s_tok * gqa_ratio
                # Padded rows alias onto a live row's addresses so the loads need no
                # predicate; `qmul == 0` is what zeroes them.
                qb = (
                    ((b_idx * S + s_tok) * HQ + h_k_idx * gqa_ratio + s_hd) * D
                    + qh * Int32(D // 2)
                )
                amx = Float32(0.0)
                if cutlass.const_expr(D <= 256):
                    # Every 16 B piece is issued BEFORE the absmax tree runs, so the
                    # whole staging costs ONE memory round trip, and the quantise
                    # pass re-reads the SAME registers instead of the L1.
                    qvs = []
                    for j in cutlass.range_constexpr(D // 16):
                        qvs.append(prims.load_ext(
                            mQ.iterator + qb + j * 8, dtype=cutlass.Int16, count=8
                        ))
                    for j in cutlass.range_constexpr(D // 16):
                        for e in cutlass.range_constexpr(8):
                            x = qvs[j][e].bitcast(Float16).to(Float32)
                            amx = cute.arch.fmax(amx, cute.arch.fmax(x, -x))
                else:
                    # D = 512 would hold 128 registers of raw Q; the absmax pass
                    # streams it in two 16-load batches and the quantise pass reloads
                    # out of L1 instead.
                    for g in cutlass.range_constexpr(D // 256):
                        gvs = []
                        for j in cutlass.range_constexpr(16):
                            gvs.append(prims.load_ext(
                                mQ.iterator + qb + (g * 16 + j) * 8,
                                dtype=cutlass.Int16, count=8,
                            ))
                        for j in cutlass.range_constexpr(16):
                            for e in cutlass.range_constexpr(8):
                                x = gvs[j][e].bitcast(Float16).to(Float32)
                                amx = cute.arch.fmax(amx, cute.arch.fmax(x, -x))
                amx = cute.arch.fmax(amx, prims.shfl_sync(
                    thread_mask=0xFFFFFFFF, val=amx, offset=1, mask_and_clamp=0x1F,
                    kind=prims.Shfl.BFLY, return_value_and_is_valid=False,
                ))
                # An all-zero row (GQA / draft padding) would divide by nothing.
                amx = cute.arch.fmax(amx, Float32(1.0e-30))
                qmul = cute.arch.rcp_approx(amx) * Float32(e4m3_max)
                if sgrow >= rows_total:
                    qmul = Float32(0.0)
                qq = cutlass.Array(e4m3, 16, alignment=16)
                for c in cutlass.range_constexpr(D // 32):
                    for h in cutlass.range_constexpr(2):
                        if cutlass.const_expr(D <= 256):
                            vv = qvs[c * 2 + h]
                        else:
                            vv = prims.load_ext(
                                mQ.iterator + qb + (c * 2 + h) * 8,
                                dtype=cutlass.Int16, count=8,
                            )
                        for e in cutlass.range_constexpr(8):
                            qq[h * 8 + e] = (
                                vv[e].bitcast(Float16).to(Float32) * qmul
                            ).to(e4m3)
                    # The 16 B chunk index over the whole D-wide row routes to its
                    # d-slice; inside it the swizzle is the physical chunk permute.
                    ch = qh * Int32(D // 32) + c
                    sl = ch >> (q_chunks.bit_length() - 1)
                    cc = ch & Int32(q_chunks - 1)
                    cc = cc ^ ((qr >> sw_shift) & Int32(sw_mask))
                    sQ.subview(
                        sl * q_stage_elems + qr * mma_k + cc * 16
                    ).data_ptr().store(
                        qq.data_ptr().load(count=16, alignment=16), alignment=16
                    )
                if qh == 0:
                    sQS[qr] = scale * amx * Float32(1.0 / e4m3_max)
                cute.arch.fence_view_async_shared()
                mrow = mTM.iterator + (b_idx * S + q_tok) * MW
                m_w0 = mrow[0]
                if MW > 1:
                    m_w1 = mrow[1]
                if has_sink == 1:
                    r_sink = mSNK.iterator[h_k_idx * gqa_ratio + q_hd] * Float32(log2_e)
            else:
                qbase = ((b_idx * S + q_tok) * HQ + h_k_idx * gqa_ratio + q_hd) * D
                # Only WG A (warps 4-7) stages Q: the two softmax WGs own the
                # same 128 rows, so a second staging pass would just
                # double-store the same bytes.  WG B still pulls its per-row
                # mask words and sink.  Rows at or past `S * gqa` are
                # zero-filled: a zero query row scores a finite 0 against every
                # key, and the epilogue simply never stores it.
                if warp_idx < 8:
                    qsw = (row >> sw_shift) & Int32(sw_mask)
                    for it in cutlass.range_constexpr(q_pieces):
                        dk = it // q_chunks
                        qc = it % q_chunks                  # 16 B chunk along d
                        sof = dk * q_stage_elems + row * mma_k + ((qc ^ qsw) * 8)
                        gof = qbase + dk * mma_k + qc * 8
                        if live_row:
                            prims.store_ext(
                                prims.load_ext(mQ.iterator + gof, dtype=Float32, count=4),
                                sQ.subview(sof).data_ptr(),
                            )
                        else:
                            prims.store_ext(
                                cutlass.vector.full((4,), 0, Float32),
                                sQ.subview(sof).data_ptr(),
                            )
                    cute.arch.fence_view_async_shared()
                # The tree-mask words of THIS row's draft token, and this row's
                # sink logit, are pulled once per CTA into registers.
                mrow = mTM.iterator + (b_idx * S + q_tok) * MW
                m_w0 = mrow[0]
                if MW > 1:
                    m_w1 = mrow[1]
                if has_sink == 1:
                    r_sink = mSNK.iterator[h_k_idx * gqa_ratio + q_hd] * Float32(log2_e)

        if cutlass.const_expr(wide):
            if warp_idx == 3:
                # V producer: released by warp 0's KV-mbarrier arrive alone, so its
                # first cp.async.bulk no longer waits on the Q staging warps.
                prims.barrier_cta_sync(kv_init_barrier, thread_count=64)
            elif warp_idx != 0:
                prims.barrier_cta_sync(0, thread_count=arrive0_count)
        else:
            if warp_idx != 0:
                prims.barrier_cta_sync(0, thread_count=arrive0_count)

        tmem_base = tmem_holder.load()

        # ======================================================== TMA warp
        kp_r = Int32(0)
        if warp_idx == 0:
            # One page id per lane; the walk refills the register window every
            # 32 pages, because a shfl.idx past lane 31 clamps instead of
            # wrapping.  `sh0` is where this split's window sits inside the
            # register window the lanes hold: 0 once the ids were pulled for
            # this split specifically, `p0` when the cycle-zero probe of the
            # row's first pages already covers it.
            sh0 = Int32(0)
            if spec_win:
                kp_r = kp_g
                sh0 = p0
            else:
                li = lane
                if li >= iters:
                    li = iters - 1
                if li < 0:
                    li = Int32(0)
                # A dead split under a sliding window can carry
                # p0 == first_page + live_pages == MAXP; its eager probe (the
                # value is never consumed) must still stay inside this batch's
                # own page-table row.
                p0c = p0
                if p0c >= MAXP:
                    p0c = Int32(0)
                pt0 = mPT.iterator + (b_idx * 2) * MAXP + p0c
                kp_r = pt0[li]
            # Split 0's page 0 was already issued speculatively above: its
            # pipeline indices start pre-advanced (the k_slices issued stages
            # land the ring at k_slices % kstg, phase flipped once per full
            # wrap) and chunk 0 of the walk starts at page 1.  Pre-advance only
            # when the speculative issue actually fired: a window start past
            # the cycle-zero register window skipped it.
            k_idx = Int32(0)
            k_ph = Int32(1)
            j00 = Int32(0)
            if cutlass.const_expr(spec_page0):
                if split_idx == 0:
                    if first_page < spec_pages:
                        k_idx = Int32(k_pre_idx)
                        k_ph = Int32(k_pre_ph)
                        j00 = Int32(1)
            pt = mPT.iterator + (b_idx * 2) * MAXP + p0
            # The refill is hoisted into an outer chunk loop rather than
            # predicated inside the page walk: a conditional multi-instruction
            # body inside a staged loop is predicated, not branched, so an
            # in-loop refill would issue its dependent global load on every page.
            n_chunks = (iters + Int32(page_chunk - 1)) >> 5
            for c in cutlass.range(n_chunks):
              cbase = c << 5
              if c != 0:
                li = cbase + lane
                if li >= iters:
                    li = iters - 1
                kp_r = pt[li]
              n_in_chunk = iters - cbase
              if n_in_chunk > page_chunk:
                  n_in_chunk = Int32(page_chunk)
              js = Int32(0)
              if c == 0:
                  js = j00
              # `sh0` is 0 whenever the ids were refetched for this split and
              # `p0` when the cycle-zero probe covers the whole sequence; a
              # refill only ever happens on the long rows, where `sh0` is 0.
              ooff = Int32(0)
              if c == 0:
                  ooff = sh0
              for j in cutlass.range(n_in_chunk - js):
                jj = ooff + js + j
                ksrc = kp_r
                if spec_win:
                    if c == 0:
                        kwsel = jj >> 5
                        if kwsel == 1:
                            ksrc = kw1
                        if kwsel == 2:
                            ksrc = kw2
                        if kwsel == 3:
                            ksrc = kw3
                        if cutlass.const_expr(spec_chunks > 4):
                            if kwsel == 4:
                                ksrc = kw4
                kp = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=ksrc,
                    offset=jj & Int32(0x1F),
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                    return_value_and_is_valid=False,
                )
                # A page is always 128 physically-valid tokens, so the whole box
                # is fetched even on the tail page: the rows past seq_len are
                # masked to -inf in the softmax (their P is exactly 0, so they
                # also contribute nothing to BMM2).
                for dk in cutlass.range_constexpr(k_slices):
                    while not prims.mbarrier_try_wait_parity(
                        kbar.subview(kstg + k_idx).data_ptr(), k_ph
                    ):
                        pass
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(
                            kbar.subview(k_idx).data_ptr(), k_stage_bytes
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sK.subview(k_idx * k_stage_elems).data_ptr(),
                            desc_k,
                            (Int32(dk * mma_k), Int32(h_k_idx), Int32(0), kp),
                            kbar.subview(k_idx).data_ptr(),
                            l2_cache_hint=polw,
                        )
                    k_idx += 1
                    if k_idx == kstg:
                        k_idx = Int32(0)
                        k_ph ^= 1
        # ================================================= V producer (warp 3)
        # V rides its own warp so the K stream never queues behind a V-stage
        # release: V credits only free at BMM2's pace, and an interleaved
        # producer would stall K page p+1 behind V page p, capping the KV bytes
        # in flight at the V-credit depth.
        elif warp_idx == 3:
            sh0v = Int32(0)
            vp_r3 = Int32(0)
            if spec_win:
                vp_r3 = vp_g3
                sh0v = p0
            else:
                li = lane
                if li >= iters:
                    li = iters - 1
                if li < 0:
                    li = Int32(0)
                p0c = p0
                if p0c >= MAXP:
                    p0c = Int32(0)
                vp_r3 = (mPT.iterator + (b_idx * 2) * MAXP + p0c)[MAXP + li]
            v_idx = Int32(0)
            v_ph = Int32(1)
            j00v = Int32(0)
            if cutlass.const_expr(spec_page0):
                if split_idx == 0:
                    if first_page < spec_pages:
                        v_idx = Int32(v_pre_idx)
                        v_ph = Int32(v_pre_ph)
                        j00v = Int32(1)
            ptv = mPT.iterator + (b_idx * 2) * MAXP + p0
            n_chunks = (iters + Int32(page_chunk - 1)) >> 5
            for c in cutlass.range(n_chunks):
              cbase = c << 5
              if c != 0:
                li = cbase + lane
                if li >= iters:
                    li = iters - 1
                vp_r3 = ptv[MAXP + li]
              n_in_chunk = iters - cbase
              if n_in_chunk > page_chunk:
                  n_in_chunk = Int32(page_chunk)
              js = Int32(0)
              if c == 0:
                  js = j00v
              ooffv = Int32(0)
              if c == 0:
                  ooffv = sh0v
              for j in cutlass.range(n_in_chunk - js):
                jj = ooffv + js + j
                vsrc = vp_r3
                if spec_win:
                    if c == 0:
                        vwsel = jj >> 5
                        if vwsel == 1:
                            vsrc = v3w1
                        if vwsel == 2:
                            vsrc = v3w2
                        if vwsel == 3:
                            vsrc = v3w3
                        if cutlass.const_expr(spec_chunks > 4):
                            if vwsel == 4:
                                vsrc = v3w4
                vp = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=vsrc,
                    offset=jj & Int32(0x1F),
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                    return_value_and_is_valid=False,
                )
                # Head-dim shard (D == 512 only): this CTA streams only ITS `v_live` d-pair stages,
                # starting at d = ds * Dc.  Every other D keeps the constexpr-unrolled walk.
                if cutlass.const_expr(can_dsplit):
                    for sk in cutlass.range(v_live):
                        vd0 = vbase0 + sk * Int32(2 * mma_k)
                        while not prims.mbarrier_try_wait_parity(
                            vbar.subview(v_stages + v_idx).data_ptr(), v_ph
                        ):
                            pass
                        if prims.elect_sync():
                            prims.mbarrier_arrive_expect_tx(
                                vbar.subview(v_idx).data_ptr(), v_stage_bytes
                            )
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sV.subview(v_idx * v_stage_elems).data_ptr(),
                                desc_k,
                                (vd0, Int32(h_k_idx), Int32(0), vp),
                                vbar.subview(v_idx).data_ptr(),
                                l2_cache_hint=polw,
                            )
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sV.subview(v_idx * v_stage_elems + v_subtile_elems).data_ptr(),
                                desc_k,
                                (vd0 + Int32(mma_k), Int32(h_k_idx), Int32(0), vp),
                                vbar.subview(v_idx).data_ptr(),
                                l2_cache_hint=polw,
                            )
                        v_idx += 1
                        if v_idx == v_stages:
                            v_idx = Int32(0)
                            v_ph ^= 1
                else:
                    for sk in cutlass.range_constexpr(v_slices):
                        while not prims.mbarrier_try_wait_parity(
                            vbar.subview(v_stages + v_idx).data_ptr(), v_ph
                        ):
                            pass
                        if prims.elect_sync():
                            prims.mbarrier_arrive_expect_tx(
                                vbar.subview(v_idx).data_ptr(), v_stage_bytes
                            )
                            for bx in cutlass.range_constexpr(v_boxes):
                                vdw = Int32((sk * v_boxes + bx) * mma_k)
                                if cutlass.const_expr(dsh_ways > 1):
                                    vdw = vbase0 + vdw
                                prims.cp_async_bulk_tensor_shared_cta_global(
                                    sV.subview(v_idx * v_stage_elems + bx * v_subtile_elems).data_ptr(),
                                    desc_k,
                                    (vdw, Int32(h_k_idx), Int32(0), vp),
                                    vbar.subview(v_idx).data_ptr(),
                                    l2_cache_hint=polw,
                                )
                        v_idx += 1
                        if v_idx == v_stages:
                            v_idx = Int32(0)
                            v_ph ^= 1
        # ======================================================== BMM1 warp
        elif warp_idx == 1:
          if cutlass.const_expr(wide):
            # S = Q @ K^T with M = the query rows and N = the keys: A is the
            # staged Q tile (K-major over d, exactly the swizzle sK carries) and
            # B is the K page (also K-major over d).  One KV page is `k_slices`
            # d-slice stages streamed DK-OUTER: every slice feeds BOTH S tiles
            # of the page (each tile accumulates over the slices), so a stage's
            # credit is returned after its two tile-MMAs and the ring never
            # needs to hold a whole page.
            idesc = prims.Tcgen05InstrDesc.build(
                c_dtype=Float32,
                a_dtype=Float16,
                b_dtype=Float16,
                n_dim=n_tok,
                m_dim=rows_tile,
            )
            s_idx = Int32(0)
            s_ph = Int32(1)
            k_idx = Int32(0)
            k_ph = Int32(0)
            for s in cutlass.range(iters):
                # The page's tile PAIR shares one 64-column TMEM block (tile 0 at lane alignment 0,
                # tile 1 at 16), so one free-wait covers both tiles.
                while not prims.mbarrier_try_wait_parity(
                    sbar.subview(s_stages + s_idx).data_ptr(), s_ph
                ):
                    pass
                for dk in cutlass.range_constexpr(k_slices):
                    while not prims.mbarrier_try_wait_parity(
                        kbar.subview(k_idx).data_ptr(), k_ph
                    ):
                        pass
                    dKs = cutlass.Int64(
                        prims.Tcgen05SmemDesc.build(
                            start_address=(sK.data_ptr() + k_idx * k_stage_elems).toint(Int32),
                            leading_byte_offset=16,
                            stride_byte_offset=1024,
                            base_offset=cutlass.Int8(0),
                            layout=cutlass.Int8(2),
                            leading_dim_mode=cutlass.Boolean(0),
                        )
                    )
                    dQ = cutlass.Int64(
                        prims.Tcgen05SmemDesc.build(
                            start_address=(
                                sQ.data_ptr() + dk * q_stage_elems
                            ).toint(Int32),
                            leading_byte_offset=16,
                            stride_byte_offset=1024,
                            base_offset=cutlass.Int8(0),
                            layout=cutlass.Int8(2),
                            leading_dim_mode=cutlass.Boolean(0),
                        )
                    )
                    # One election for the whole slice's MMA group (see BMM2).
                    if prims.elect_sync():
                        for tt in cutlass.range_constexpr(tiles_per_page):
                            acc = prims.make_tmem_ptr(
                                ((tt * 16) << 16) | (tmem_base + s_idx * n_tok),
                                Float32,
                            )
                            dKt = dKs + (tt * tile_byte_step) // 16
                            for mk in cutlass.range_constexpr(mma_k // 16):
                                prims.tcgen05_mma(
                                    prims.Tcgen05MMAKind.F16,
                                    prims.CTAGroup.CTA_1,
                                    acc,
                                    dQ + mk * 2,
                                    dKt + mk * 2,
                                    idesc,
                                    cutlass.Boolean(dk != 0 or mk != 0),
                                )
                        prims.tcgen05_commit(
                            kbar.subview(kstg + k_idx).data_ptr(),
                            group=prims.CTAGroup.CTA_1,
                        )
                    k_idx += 1
                    if k_idx == kstg:
                        k_idx = Int32(0)
                        k_ph ^= 1
                if prims.elect_sync():
                    prims.tcgen05_commit(
                        sbar.subview(s_idx).data_ptr(), group=prims.CTAGroup.CTA_1
                    )
                s_idx += 1
                if s_idx == s_stages:
                    s_idx = Int32(0)
                    s_ph ^= 1
          else:
            # S = Q @ K^T with M = the 128 query rows and N = the keys.  One KV
            # page feeds `tiles_per_page` S tiles of `n_tok` keys, and every
            # d-slice of the page (`k_slices` K stages) is needed by each of
            # them, so the stage credits are returned once, after the last tile
            # of the page.
            #
            # TWO issue forms, picked on the CTA-uniform walk depth:
            #  * PAGE-FUSED (deep walks).  The two S tiles of a page occupy
            #    CONTIGUOUS TMEM columns, so ONE N == 128 MMA group covers both:
            #    half the BMM1 instruction count and half the 32 KiB Q re-reads
            #    out of shared memory (L1/TEX was the highest-utilised unit).
            #  * PER-TILE (shallow walks).  The fused group can only signal both
            #    ready barriers after ALL its MMAs retire, so the first softmax
            #    warpgroup starts half a BMM1 later; a CTA that never reaches
            #    steady state pays that fill latency on every page it has.
            # The gate is `deep walk AND more than one row block`: a narrow tree
            # at the same depth is DRAM-paced and only feels the fill latency.
            idesc = prims.Tcgen05InstrDesc.build(
                c_dtype=Float32,
                a_dtype=kv_t,
                b_dtype=kv_t,
                n_dim=tokens_per_k_stage,
                m_dim=rows_tile,
            )
            idesc1 = prims.Tcgen05InstrDesc.build(
                c_dtype=Float32,
                a_dtype=kv_t,
                b_dtype=kv_t,
                n_dim=n_tok,
                m_dim=rows_tile,
            )
            # At ONE d-slice the Q descriptor is loop-invariant (one fixed smem
            # base), so it is built once outside the walk.
            if cutlass.const_expr(k_slices == 1):
                dQ0 = cutlass.Int64(
                    prims.Tcgen05SmemDesc.build(
                        start_address=(sQ.data_ptr()).toint(Int32),
                        leading_byte_offset=16,
                        stride_byte_offset=sbo,
                        base_offset=cutlass.Int8(0),
                        layout=smem_layout,
                        leading_dim_mode=cutlass.Boolean(0),
                    )
                )
            s_idx = Int32(0)
            s_ph = Int32(1)
            k_idx = Int32(0)
            k_ph = Int32(0)
            fuse_ok = Int32(0)
            if iters >= fuse_depth:
                if rows_total > rows_tile:
                    fuse_ok = Int32(1)
            if cutlass.const_expr(kv_e4m3 and k_slices > 1):
              # e4m3 D > 128: the page is `k_slices` d-slice stages streamed
              # DK-OUTER into the page-fused N == 128 accumulator, each slice
              # returning its credit as soon as its MMAs retire, so the K ring
              # never holds a whole page.  No tile finishes before the last
              # slice, so there is nothing for a per-tile form to release early.
              for s in cutlass.range(iters):
                for tt in cutlass.range_constexpr(tiles_per_page):
                    while not prims.mbarrier_try_wait_parity(
                        sbar.subview(s_stages + s_idx + tt).data_ptr(), s_ph
                    ):
                        pass
                acc = prims.make_tmem_ptr(tmem_base + s_idx * n_tok, Float32)
                for dk in cutlass.range_constexpr(k_slices):
                    while not prims.mbarrier_try_wait_parity(
                        kbar.subview(k_idx).data_ptr(), k_ph
                    ):
                        pass
                    dKs = cutlass.Int64(
                        prims.Tcgen05SmemDesc.build(
                            start_address=(
                                sK.data_ptr() + k_idx * k_stage_elems
                            ).toint(Int32),
                            leading_byte_offset=16,
                            stride_byte_offset=sbo,
                            base_offset=cutlass.Int8(0),
                            layout=smem_layout,
                            leading_dim_mode=cutlass.Boolean(0),
                        )
                    )
                    dQ = cutlass.Int64(
                        prims.Tcgen05SmemDesc.build(
                            start_address=(
                                sQ.data_ptr() + dk * q_stage_elems
                            ).toint(Int32),
                            leading_byte_offset=16,
                            stride_byte_offset=sbo,
                            base_offset=cutlass.Int8(0),
                            layout=smem_layout,
                            leading_dim_mode=cutlass.Boolean(0),
                        )
                    )
                    if prims.elect_sync():
                        for mk in cutlass.range_constexpr(n_mk1):
                            prims.tcgen05_mma(
                                mma_kind,
                                prims.CTAGroup.CTA_1,
                                acc,
                                dQ + mk * kdesc_kstep,
                                dKs + mk * kdesc_kstep,
                                idesc,
                                cutlass.Boolean(dk != 0 or mk != 0),
                            )
                        prims.tcgen05_commit(
                            kbar.subview(kstg + k_idx).data_ptr(),
                            group=prims.CTAGroup.CTA_1,
                        )
                    k_idx += 1
                    if k_idx == kstg:
                        k_idx = Int32(0)
                        k_ph ^= 1
                if prims.elect_sync():
                    for tt in cutlass.range_constexpr(tiles_per_page):
                        prims.tcgen05_commit(
                            sbar.subview(s_idx + tt).data_ptr(),
                            group=prims.CTAGroup.CTA_1,
                        )
                s_idx += tiles_per_page
                if s_idx == s_stages:
                    s_idx = Int32(0)
                    s_ph ^= 1
            else:
             if fuse_ok != 0:
              for s in cutlass.range(iters):
                if cutlass.const_expr(k_slices == 2):
                    k_idx1 = k_idx + 1
                    k_ph1 = k_ph
                    if k_idx1 == kstg:
                        k_idx1 = Int32(0)
                        k_ph1 = k_ph ^ 1
                    for dk in cutlass.range_constexpr(2):
                        kw = k_idx
                        kwp = k_ph
                        if cutlass.const_expr(dk == 1):
                            kw = k_idx1
                            kwp = k_ph1
                        while not prims.mbarrier_try_wait_parity(
                            kbar.subview(kw).data_ptr(), kwp
                        ):
                            pass
                else:
                    while not prims.mbarrier_try_wait_parity(
                        kbar.subview(k_idx).data_ptr(), k_ph
                    ):
                        pass
                dK0 = cutlass.Int64(
                    prims.Tcgen05SmemDesc.build(
                        start_address=(sK.data_ptr() + k_idx * k_stage_elems).toint(Int32),
                        leading_byte_offset=16,
                        stride_byte_offset=sbo,
                        base_offset=cutlass.Int8(0),
                        layout=smem_layout,
                        leading_dim_mode=cutlass.Boolean(0),
                    )
                )
                if cutlass.const_expr(k_slices == 2):
                    dK1 = cutlass.Int64(
                        prims.Tcgen05SmemDesc.build(
                            start_address=(sK.data_ptr() + k_idx1 * k_stage_elems).toint(Int32),
                            leading_byte_offset=16,
                            stride_byte_offset=1024,
                            base_offset=cutlass.Int8(0),
                            layout=cutlass.Int8(2),
                            leading_dim_mode=cutlass.Boolean(0),
                        )
                    )
                for tt in cutlass.range_constexpr(tiles_per_page):
                    while not prims.mbarrier_try_wait_parity(
                        sbar.subview(s_stages + s_idx + tt).data_ptr(), s_ph
                    ):
                        pass
                acc = prims.make_tmem_ptr(tmem_base + s_idx * n_tok, Float32)
                scale_d = False
                # One election for the whole page's MMA group (see BMM2).
                if prims.elect_sync():
                    for dk in cutlass.range_constexpr(k_slices):
                        if cutlass.const_expr(k_slices == 1):
                            dKs = dK0
                            dQ = dQ0
                        else:
                            dKs = dK0
                            if cutlass.const_expr(dk == 1):
                                dKs = dK1
                            dQ = cutlass.Int64(
                                prims.Tcgen05SmemDesc.build(
                                    start_address=(
                                        sQ.data_ptr() + dk * q_stage_elems
                                    ).toint(Int32),
                                    leading_byte_offset=16,
                                    stride_byte_offset=1024,
                                    base_offset=cutlass.Int8(0),
                                    layout=cutlass.Int8(2),
                                    leading_dim_mode=cutlass.Boolean(0),
                                )
                            )
                        for mk in cutlass.range_constexpr(n_mk1):
                            prims.tcgen05_mma(
                                mma_kind,
                                prims.CTAGroup.CTA_1,
                                acc,
                                dQ + mk * kdesc_kstep,
                                dKs + mk * kdesc_kstep,
                                idesc,
                                cutlass.Boolean(scale_d),
                            )
                            scale_d = True
                    for tt in cutlass.range_constexpr(tiles_per_page):
                        prims.tcgen05_commit(
                            sbar.subview(s_idx + tt).data_ptr(),
                            group=prims.CTAGroup.CTA_1,
                        )
                s_idx += tiles_per_page
                if s_idx == s_stages:
                    s_idx = Int32(0)
                    s_ph ^= 1
                if prims.elect_sync():
                    prims.tcgen05_commit(
                        kbar.subview(kstg + k_idx).data_ptr(),
                        group=prims.CTAGroup.CTA_1,
                    )
                    if cutlass.const_expr(k_slices == 2):
                        prims.tcgen05_commit(
                            kbar.subview(kstg + k_idx1).data_ptr(),
                            group=prims.CTAGroup.CTA_1,
                        )
                if cutlass.const_expr(k_slices == 2):
                    k_idx = k_idx1 + 1
                    k_ph = k_ph1
                else:
                    k_idx += 1
                if k_idx == kstg:
                    k_idx = Int32(0)
                    k_ph ^= 1
             else:
              for s in cutlass.range(iters):
                if cutlass.const_expr(k_slices == 2):
                    k_idx1 = k_idx + 1
                    k_ph1 = k_ph
                    if k_idx1 == kstg:
                        k_idx1 = Int32(0)
                        k_ph1 = k_ph ^ 1
                    for dk in cutlass.range_constexpr(2):
                        kw = k_idx
                        kwp = k_ph
                        if cutlass.const_expr(dk == 1):
                            kw = k_idx1
                            kwp = k_ph1
                        while not prims.mbarrier_try_wait_parity(
                            kbar.subview(kw).data_ptr(), kwp
                        ):
                            pass
                else:
                    while not prims.mbarrier_try_wait_parity(
                        kbar.subview(k_idx).data_ptr(), k_ph
                    ):
                        pass
                dK0 = cutlass.Int64(
                    prims.Tcgen05SmemDesc.build(
                        start_address=(sK.data_ptr() + k_idx * k_stage_elems).toint(Int32),
                        leading_byte_offset=16,
                        stride_byte_offset=sbo,
                        base_offset=cutlass.Int8(0),
                        layout=smem_layout,
                        leading_dim_mode=cutlass.Boolean(0),
                    )
                )
                if cutlass.const_expr(k_slices == 2):
                    dK1 = cutlass.Int64(
                        prims.Tcgen05SmemDesc.build(
                            start_address=(sK.data_ptr() + k_idx1 * k_stage_elems).toint(Int32),
                            leading_byte_offset=16,
                            stride_byte_offset=1024,
                            base_offset=cutlass.Int8(0),
                            layout=cutlass.Int8(2),
                            leading_dim_mode=cutlass.Boolean(0),
                        )
                    )
                for tt in cutlass.range_constexpr(tiles_per_page):
                    while not prims.mbarrier_try_wait_parity(
                        sbar.subview(s_stages + s_idx).data_ptr(), s_ph
                    ):
                        pass
                    acc = prims.make_tmem_ptr(tmem_base + s_idx * n_tok, Float32)
                    scale_d = False
                    # One election for the whole tile's MMA group (see BMM2).
                    if prims.elect_sync():
                        for dk in cutlass.range_constexpr(k_slices):
                            if cutlass.const_expr(k_slices == 1):
                                dKs = dK0 + (tt * tile_byte_step) // 16
                                dQ = dQ0
                            else:
                                dKs = dK0
                                if cutlass.const_expr(dk == 1):
                                    dKs = dK1
                                dKs = dKs + (tt * tile_byte_step) // 16
                                dQ = cutlass.Int64(
                                    prims.Tcgen05SmemDesc.build(
                                        start_address=(
                                            sQ.data_ptr() + dk * q_stage_elems
                                        ).toint(Int32),
                                        leading_byte_offset=16,
                                        stride_byte_offset=1024,
                                        base_offset=cutlass.Int8(0),
                                        layout=cutlass.Int8(2),
                                        leading_dim_mode=cutlass.Boolean(0),
                                    )
                                )
                            for mk in cutlass.range_constexpr(n_mk1):
                                prims.tcgen05_mma(
                                    mma_kind,
                                    prims.CTAGroup.CTA_1,
                                    acc,
                                    dQ + mk * kdesc_kstep,
                                    dKs + mk * kdesc_kstep,
                                    idesc1,
                                    cutlass.Boolean(scale_d),
                                )
                                scale_d = True
                        prims.tcgen05_commit(
                            sbar.subview(s_idx).data_ptr(), group=prims.CTAGroup.CTA_1
                        )
                    s_idx += 1
                    if s_idx == s_stages:
                        s_idx = Int32(0)
                        s_ph ^= 1
                if prims.elect_sync():
                    prims.tcgen05_commit(
                        kbar.subview(kstg + k_idx).data_ptr(),
                        group=prims.CTAGroup.CTA_1,
                    )
                    if cutlass.const_expr(k_slices == 2):
                        prims.tcgen05_commit(
                            kbar.subview(kstg + k_idx1).data_ptr(),
                            group=prims.CTAGroup.CTA_1,
                        )
                if cutlass.const_expr(k_slices == 2):
                    k_idx = k_idx1 + 1
                    k_ph = k_ph1
                else:
                    k_idx += 1
                if k_idx == kstg:
                    k_idx = Int32(0)
                    k_ph ^= 1

        # ======================================================== BMM2 warp
        elif warp_idx == 2:
          if cutlass.const_expr(wide):
            # O = P @ V, M = query rows, N = 128 d-columns per V d-pair stage: A is the published P
            # tile (K-major over keys, SMEM), B one V d-pair stage (MN-major over d).  V stages
            # stream VS-OUTER -- both P tiles are waited first, then every V stage multiplies both
            # into its own 64-column O block and returns credit.
            idesc = prims.Tcgen05InstrDesc.build(
                c_dtype=Float32,
                a_dtype=Float16,
                b_dtype=Float16,
                a_major=0,
                b_major=1,
                n_dim=2 * mma_k,
                m_dim=rows_tile,
            )
            # No O-zero handshake: the FIRST V-MMA of the CTA overwrites the
            # accumulator (enable-input-d is a runtime predicate).
            p_idx = Int32(0)
            p_ph = Int32(0)
            v_idx = Int32(0)
            v_ph = Int32(0)
            rph = Int32(0)
            for s in cutlass.range(iters):
                # Rescale handshake, fired ONLY on the rare max-jump page (the flag rides in with P,
                # set on tile 0 alone): commit an MMA drain so the softmax WG can rescale TMEM O,
                # then wait its 128-thread rescale-done arrive.  Past page 0 only.
                notfirst = s > 0
                while not prims.mbarrier_try_wait_parity(
                    pbar.subview(p_idx).data_ptr(), p_ph
                ):
                    pass
                if notfirst:
                    if sChg[p_idx] != 0:
                        if prims.elect_sync():
                            prims.tcgen05_commit(
                                obar.subview(0).data_ptr(),
                                group=prims.CTAGroup.CTA_1,
                            )
                        while not prims.mbarrier_try_wait_parity(
                            obar.subview(1).data_ptr(), rph
                        ):
                            pass
                        rph ^= 1
                pA0 = p_idx
                p_idx += 1
                if p_idx == pstg:
                    p_idx = Int32(0)
                    p_ph ^= 1
                while not prims.mbarrier_try_wait_parity(
                    pbar.subview(p_idx).data_ptr(), p_ph
                ):
                    pass
                pA1 = p_idx
                p_idx += 1
                if p_idx == pstg:
                    p_idx = Int32(0)
                    p_ph ^= 1
                # Head-dim shard (D == 512 only) walks a runtime stage count; every other D keeps
                # the constexpr-unrolled walk.
                if cutlass.const_expr(can_dsplit):
                  for vs in cutlass.range(v_live):
                      while not prims.mbarrier_try_wait_parity(
                          vbar.subview(v_idx).data_ptr(), v_ph
                      ):
                          pass
                      # One election for the whole stage's MMA group, not one per issue.
                      if prims.elect_sync():
                          # O d-halves pack like the S pair: blocks vs < oh_live sit at lane
                          # alignment 0, the rest at 16, sharing the same columns.
                          oal = Int32(0)
                          ocl = vs
                          if vs >= oh_live:
                              oal = Int32(16)
                              ocl = vs - oh_live
                          accT = prims.make_tmem_ptr(
                              (oal << 16)
                              | (tmem_base + o_col_offset + ocl * Int32(2 * mma_k)),
                              Float32,
                          )
                          for tt in cutlass.range_constexpr(tiles_per_page):
                              pAt = pA0
                              if cutlass.const_expr(tt == 1):
                                  pAt = pA1
                              dV = cutlass.Int64(
                                  prims.Tcgen05SmemDesc.build(
                                      start_address=(
                                          sV.data_ptr() + v_idx * v_stage_elems + tt * (n_tok * mma_k)
                                      ).toint(Int32),
                                      leading_byte_offset=v_subtile_elems * 2,
                                      stride_byte_offset=1024,
                                      base_offset=cutlass.Int8(0),
                                      layout=cutlass.Int8(2),
                                      leading_dim_mode=cutlass.Boolean(0),
                                  )
                              )
                              dP = cutlass.Int64(
                                  prims.Tcgen05SmemDesc.build(
                                      start_address=(
                                          sP.data_ptr() + pAt * p_stage_elems
                                      ).toint(Int32),
                                      leading_byte_offset=16,
                                      stride_byte_offset=1024,
                                      base_offset=cutlass.Int8(0),
                                      layout=cutlass.Int8(2),
                                      leading_dim_mode=cutlass.Boolean(0),
                                  )
                              )
                              for mk in cutlass.range_constexpr(n_tok // 16):
                                  prims.tcgen05_mma(
                                      prims.Tcgen05MMAKind.F16,
                                      prims.CTAGroup.CTA_1,
                                      accT,
                                      dP + mk * 2,
                                      dV + mk * v_kblk_step,
                                      idesc,
                                      notfirst if (tt == 0 and mk == 0) else cutlass.Boolean(True),
                                  )
                          prims.tcgen05_commit(
                              vbar.subview(v_stages + v_idx).data_ptr(),
                              group=prims.CTAGroup.CTA_1,
                          )
                      v_idx += 1
                      if v_idx == v_stages:
                          v_idx = Int32(0)
                          v_ph ^= 1
                else:
                  for vs in cutlass.range_constexpr(v_slices):
                      while not prims.mbarrier_try_wait_parity(
                          vbar.subview(v_idx).data_ptr(), v_ph
                      ):
                          pass
                      if prims.elect_sync():
                          # O d-halves pack like the S pair: blocks vs < o_half sit at lane
                          # alignment 0, the rest at 16, sharing the same columns.
                          accT = prims.make_tmem_ptr(
                              ((16 if vs >= o_half else 0) << 16)
                              | (tmem_base + o_col_offset + (vs % o_half) * 2 * mma_k),
                              Float32,
                          )
                          for tt in cutlass.range_constexpr(tiles_per_page):
                              pAt = pA0
                              if cutlass.const_expr(tt == 1):
                                  pAt = pA1
                              dV = cutlass.Int64(
                                  prims.Tcgen05SmemDesc.build(
                                      start_address=(
                                          sV.data_ptr() + v_idx * v_stage_elems + tt * (n_tok * mma_k)
                                      ).toint(Int32),
                                      leading_byte_offset=v_subtile_elems * 2,
                                      stride_byte_offset=1024,
                                      base_offset=cutlass.Int8(0),
                                      layout=cutlass.Int8(2),
                                      leading_dim_mode=cutlass.Boolean(0),
                                  )
                              )
                              dP = cutlass.Int64(
                                  prims.Tcgen05SmemDesc.build(
                                      start_address=(
                                          sP.data_ptr() + pAt * p_stage_elems
                                      ).toint(Int32),
                                      leading_byte_offset=16,
                                      stride_byte_offset=1024,
                                      base_offset=cutlass.Int8(0),
                                      layout=cutlass.Int8(2),
                                      leading_dim_mode=cutlass.Boolean(0),
                                  )
                              )
                              for mk in cutlass.range_constexpr(n_tok // 16):
                                  prims.tcgen05_mma(
                                      prims.Tcgen05MMAKind.F16,
                                      prims.CTAGroup.CTA_1,
                                      accT,
                                      dP + mk * 2,
                                      dV + mk * v_kblk_step,
                                      idesc,
                                      notfirst if (tt == 0 and mk == 0) else cutlass.Boolean(True),
                                  )
                          prims.tcgen05_commit(
                              vbar.subview(v_stages + v_idx).data_ptr(),
                              group=prims.CTAGroup.CTA_1,
                          )
                      v_idx += 1
                      if v_idx == v_stages:
                          v_idx = Int32(0)
                          v_ph ^= 1
                if prims.elect_sync():
                    prims.tcgen05_commit(
                        pbar.subview(pstg + pA0).data_ptr(),
                        group=prims.CTAGroup.CTA_1,
                    )
                    prims.tcgen05_commit(
                        pbar.subview(pstg + pA1).data_ptr(),
                        group=prims.CTAGroup.CTA_1,
                    )
          else:
            # O = P @ V, M = the query rows (same TMEM lanes S was produced on),
            # N = the head dim: A is the freshly published P tile (K-major over
            # the keys, packed fp16 in TMEM over the first 32 columns of its S
            # stage -- the `[a-tmem]` form) and B is the V page (MN-major over
            # d, spanning every d-subtile of the stage).  Keeping rows on the M
            # axis is what lets the epilogue rescale O with a thread-local factor.
            idesc = prims.Tcgen05InstrDesc.build(
                c_dtype=Float32,
                a_dtype=kv_t,
                b_dtype=kv_t,
                a_major=0,
                b_major=1,
                n_dim=Dc,
                m_dim=rows_tile,
            )
            accO = prims.make_tmem_ptr(tmem_base + o_col_offset, Float32)
            accO2 = prims.make_tmem_ptr(tmem_base + o_col_offset + Dc, Float32)
            # No O-zero handshake: the FIRST V-MMA of the CTA overwrites the
            # accumulator (enable-input-d is a runtime predicate).
            p_idx = Int32(0)
            p_ph = Int32(0)
            v_idx = Int32(0)
            v_ph = Int32(0)
            rph = Int32(0)
            rph2 = Int32(0)
            for s in cutlass.range(iters):
                while not prims.mbarrier_try_wait_parity(
                    vbar.subview(v_idx).data_ptr(), v_ph
                ):
                    pass
                for tt in cutlass.range_constexpr(tiles_per_page):
                    while not prims.mbarrier_try_wait_parity(
                        pbar.subview(p_idx).data_ptr(), p_ph
                    ):
                        pass
                    # Rescale handshake, fired ONLY on the rare max-jump tile
                    # (the flag rides in with P): commit an MMA drain so the
                    # owning softmax WG can safely rescale its TMEM O half,
                    # then wait for its 128-thread rescale-done arrive.  Tile
                    # parity picks the accumulator AND the obar pair: tile 0
                    # belongs to WG A / O_A, tile 1 to WG B / O_B.
                    notfirst = s > 0
                    if notfirst:
                        if sChg[p_idx] != 0:
                            if cutlass.const_expr(tt == 0):
                                if prims.elect_sync():
                                    prims.tcgen05_commit(
                                        obar.subview(0).data_ptr(),
                                        group=prims.CTAGroup.CTA_1,
                                    )
                                while not prims.mbarrier_try_wait_parity(
                                    obar.subview(1).data_ptr(), rph
                                ):
                                    pass
                                rph ^= 1
                            else:
                                if prims.elect_sync():
                                    prims.tcgen05_commit(
                                        obar.subview(3).data_ptr(),
                                        group=prims.CTAGroup.CTA_1,
                                    )
                                while not prims.mbarrier_try_wait_parity(
                                    obar.subview(4).data_ptr(), rph2
                                ):
                                    pass
                                rph2 ^= 1
                    dV = cutlass.Int64(
                        prims.Tcgen05SmemDesc.build(
                            start_address=(
                                sV.data_ptr() + v_idx * v_stage_elems + tt * (n_tok * mma_k)
                            ).toint(Int32),
                            leading_byte_offset=v_subtile_bytes,
                            stride_byte_offset=sbo,
                            base_offset=cutlass.Int8(0),
                            layout=smem_layout,
                            leading_dim_mode=cutlass.Boolean(0),
                        )
                    )
                    accT = accO
                    if cutlass.const_expr(tt == 1):
                        accT = accO2
                    # One election for the whole MMA group, not one per issue.
                    # A K-granule (16 fp16 or 32 e4m3 keys) is 32 B == 8 TMEM
                    # columns, so K-block mk starts at stage base + 8*mk.  The
                    # commit after the group hands the WHOLE stage back to BMM1.
                    if prims.elect_sync():
                        for mk in cutlass.range_constexpr(n_mk2):
                            prims.tcgen05_mma(
                                mma_kind,
                                prims.CTAGroup.CTA_1,
                                accT,
                                prims.make_tmem_ptr(
                                    tmem_base + p_idx * n_tok + mk * 8, Float32
                                ),
                                dV + mk * v_kblk_step,
                                idesc,
                                notfirst if mk == 0 else cutlass.Boolean(True),
                            )
                        prims.tcgen05_commit(
                            sbar.subview(s_stages + p_idx).data_ptr(),
                            group=prims.CTAGroup.CTA_1,
                        )
                    p_idx += 1
                    if p_idx == s_stages:
                        p_idx = Int32(0)
                        p_ph ^= 1
                if prims.elect_sync():
                    prims.tcgen05_commit(
                        vbar.subview(v_stages + v_idx).data_ptr(),
                        group=prims.CTAGroup.CTA_1,
                    )
                v_idx += 1
                if v_idx == v_stages:
                    v_idx = Int32(0)
                    v_ph ^= 1
          # One terminal drain commit for the whole walk: the epilogue's O
          # read waits a single phase-0 completion instead of a per-page
          # parity walk.  Dead splits never allocated TMEM, so they commit
          # nothing (the epilogue wait sits under `active` too).
          if active:
              if prims.elect_sync():
                  prims.tcgen05_commit(
                      obar.subview(2).data_ptr(), group=prims.CTAGroup.CTA_1
                  )

        # ============================================ softmax / correction warpgroup(s)
        elif warp_idx >= 4:
            trow = wg_widx * warp_threads
            grow = row_base + row
            live_row = grow < rows_total
            # Per-row sliding window (XQA CONTIGUOUS_QUERY_SWA): the row sitting
            # at absolute position `first_q + q_tok` attends
            # [p - window + 1, p], so the LEFT EDGE MOVES with the row.  The
            # page walk already started at the minimum edge over the tile;
            # every row clips its own leading keys here.
            r_left = Int32(0)
            if window > 0:
                # `contiguous == 0`: ONE shared left edge for every row.
                lv = seq_len - window
                if CQ == 1:
                    lv = first_q + q_tok - window + Int32(1)
                if lv > 0:
                    r_left = lv

            # Running max and colsum are plain per-thread scalars.  Narrow: a
            # thread owns ONE query row and all of its keys.  Wide: a thread
            # PAIR (lane, lane ^ 16) owns one row -- the low lane holds the
            # row's whole 64-key tile 0 of every page, the high lane tile 1;
            # the row max is ONE shfl.bfly(16), the colsum stays a per-thread
            # partial, and the running max sequence matches in both threads
            # of the pair (no corr exchange).
            rM = Float32(neg_floor)
            rL = Float32(0.0)
            # Score scale of THIS row: the warpgroup-uniform `scale`, or (e4m3)
            # the per-row Q dequant scalar the staging published, which already
            # carries `scale`.
            if cutlass.const_expr(kv_e4m3):
                qsc = sQS[row]
                # P is e4m3: staged here per tile and pushed into the S stage's
                # first 16 TMEM columns as one tcgen05.st.
                rPq = cutlass.Array(e4m3, n_tok, alignment=16)
            else:
                qsc = scale

            if cutlass.const_expr(wide):
                # Per-thread key base inside a page: keys [kh0, kh0 + 64).
                kh0 = khalf << 6
                s_idx = Int32(0)
                s_ph = Int32(0)
                p_idx = Int32(0)
                p_ph = Int32(1)
            else:
                # This WG's tile parity: WG A (warps 4-7) owns tile 0 of every
                # page, WG B (warps 8-11) tile 1.  All ring state advances by
                # TWO tiles per page: the S ring (period 4) wraps every second
                # page with a phase flip.  P rides inside the same stage, so
                # one index covers S-read and P-store.
                par = Int32(0)
                if warp_idx >= 8:
                    par = Int32(1)
                # Per-WG obar pair base (A: 0, B: 3).
                ob_evt = par * 3
                # This WG's O accumulator column base inside the 512-col alloc.
                o_col = o_col_offset + par * Dc
                s_idx = par + Int32(0)
                s_ph = Int32(0)
            rc = Int32(0)
            vping = Int32(0)
            vph = Int32(0)
            for s in cutlass.range(iters):
                tokp = (p0 + s) * tokens_per_page
                if cutlass.const_expr(True):
                    if cutlass.const_expr(wide):
                        tok0 = tokp + kh0
                    else:
                        tok0 = tokp + par * n_tok
                    while not prims.mbarrier_try_wait_parity(
                        sbar.subview(s_idx).data_ptr(), s_ph
                    ):
                        pass
                    sptr = prims.make_tmem_ptr(
                        (trow << 16) | (tmem_base + s_idx * n_tok), Float32
                    )
                    ld = prims.tcgen05_ld("32x32b", sptr, num=n_tok)
                    if cutlass.const_expr(not wide):
                        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                        rS = cutlass.Array(Float32, n_tok)
                        for n in cutlass.range_constexpr(n_tok):
                            rS[n] = ld[n]
                        # No stage-free arrive here: the stage stays owned
                        # through the P store below, and BMM2's post-MMA commit
                        # is what hands it back to BMM1.
                    # ---- masks (wide: BUILT UNDER THE TMEM LOAD -- the bitmap
                    # depends only on page/tile bases and tree words).  Both
                    # gates are CTA-uniform; the tree gate is XQA's
                    # `nbSeqItersWithoutMask`: every tile that ends at or below
                    # `first_q` is pure context and needs no mask work at all.
                    # A 64-bit ALLOW BITMAP: tree bits shifted into tile-key
                    # positions, OR'd with the pure-context prefix, AND'd with
                    # the window prefix, then one predicated select per key.
                    amsk = Int32(0)
                    alo = Int32(-1)
                    ahi = Int32(-1)
                    if window > 0:
                        if cutlass.const_expr(wide):
                            wgate = tokp < max_left
                        else:
                            wgate = tok0 < max_left
                        if wgate:
                            amsk = Int32(1)
                            wl = r_left - tok0
                            wt = wl
                            if wt > 32:
                                wt = Int32(32)
                            if wt < 0:
                                wt = Int32(0)
                            if wt < 32:
                                alo = ((Int32(1) << wt) - Int32(1)) ^ Int32(-1)
                            else:
                                alo = Int32(0)
                            wt2 = wl - Int32(32)
                            if wt2 > 32:
                                wt2 = Int32(32)
                            if wt2 < 0:
                                wt2 = Int32(0)
                            if wt2 < 32:
                                ahi = ((Int32(1) << wt2) - Int32(1)) ^ Int32(-1)
                            else:
                                ahi = Int32(0)
                    if cutlass.const_expr(wide):
                        tgate = tokp + tokens_per_page > first_q
                    else:
                        tgate = tok0 + n_tok > first_q
                    if tgate:
                        amsk = Int32(1)
                        # `e0` is the draft index of this thread's key 0, so key n
                        # carries draft index n + e0: the 32 keys of each half take
                        # mask bits [e0, e0 + 32) / [e0 + 32, e0 + 64) of this row,
                        # assembled from the two words straddling the span.  Words
                        # 0 / 1 sit in registers; a wider tree pulls words >= 2 from
                        # memory here, inside the gate (at most two tiles per CTA).
                        # Bits below draft index 0 are pure context (set by `pt`).
                        e0 = tok0 - first_q
                        mrow_off = (b_idx * S + q_tok) * MW
                        tw_w = e0 >> 5
                        tw_sh = e0 & Int32(31)
                        tw_lo = Int32(0)
                        tw_hi = Int32(0)
                        if tw_w == 0:
                            tw_lo = m_w0
                        elif tw_w == 1:
                            tw_lo = m_w1
                        elif tw_w > 1:
                            if tw_w < MW:
                                tw_lo = mTM.iterator[mrow_off + tw_w]
                        tw_w1 = tw_w + Int32(1)
                        if tw_w1 == 0:
                            tw_hi = m_w0
                        elif tw_w1 == 1:
                            tw_hi = m_w1
                        elif tw_w1 > 1:
                            if tw_w1 < MW:
                                tw_hi = mTM.iterator[mrow_off + tw_w1]
                        tw = tw_lo
                        if tw_sh > 0:
                            tw = (
                                (tw_lo >> tw_sh)
                                & (((Int32(1) << (Int32(31) - tw_sh)) << 1) - Int32(1))
                            ) | (tw_hi << (Int32(32) - tw_sh))
                        pt = Int32(0) - e0
                        if pt >= 32:
                            tw = Int32(-1)
                        elif pt > 0:
                            tw = tw | ((Int32(1) << pt) - Int32(1))
                        alo = alo & tw
                        f0 = e0 + Int32(32)
                        # Upper half: word tw_w1 is its low word, tw_w + 2 its high word.
                        tw_w2 = tw_w + Int32(2)
                        tw2_hi = Int32(0)
                        if tw_w2 == 0:
                            tw2_hi = m_w0
                        elif tw_w2 == 1:
                            tw2_hi = m_w1
                        elif tw_w2 > 1:
                            if tw_w2 < MW:
                                tw2_hi = mTM.iterator[mrow_off + tw_w2]
                        tw2 = tw_hi
                        if tw_sh > 0:
                            tw2 = (
                                (tw_hi >> tw_sh)
                                & (((Int32(1) << (Int32(31) - tw_sh)) << 1) - Int32(1))
                            ) | (tw2_hi << (Int32(32) - tw_sh))
                        pt2 = Int32(0) - f0
                        if pt2 >= 32:
                            tw2 = Int32(-1)
                        elif pt2 > 0:
                            tw2 = tw2 | ((Int32(1) << pt2) - Int32(1))
                        ahi = ahi & tw2
                    # Keys at or past `seq_len` are dropped whatever the tree bits say
                    # (XQA's `col < nbValidCols`): the mask words past S may be set, and
                    # the slots they select hold stale KV.
                    if cutlass.const_expr(wide):
                        sgate = tokp + tokens_per_page > seq_len
                    else:
                        sgate = tok0 + n_tok > seq_len
                    if sgate:
                        amsk = Int32(1)
                        vl = seq_len - tok0
                        if vl <= 0:
                            alo = Int32(0)
                        elif vl < 32:
                            alo = alo & ((Int32(1) << vl) - Int32(1))
                        vh = vl - Int32(32)
                        if vh <= 0:
                            ahi = Int32(0)
                        elif vh < 32:
                            ahi = ahi & ((Int32(1) << vh) - Int32(1))
                    if cutlass.const_expr(wide):
                        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                        rS = cutlass.Array(Float32, n_tok)
                        for n in cutlass.range_constexpr(n_tok):
                            rS[n] = ld[n]
                        prims.mbarrier_arrive(sbar.subview(s_stages + s_idx).data_ptr())
                        s_idx += 1
                        if s_idx >= s_stages:
                            s_idx -= s_stages
                            s_ph ^= 1
                    if amsk != 0:
                        for n in cutlass.range_constexpr(32):
                            if (alo & Int32(_bit32[n])) == 0:
                                rS[n] = Float32(neg_big)
                        for n in cutlass.range_constexpr(32):
                            if (ahi & Int32(_bit32[n])) == 0:
                                rS[32 + n] = Float32(neg_big)

                    # ---- thread-local tile max, folded eight ways so the
                    # reduction is a short tree instead of a 63-deep FMAX chain,
                    # and clamped at the running max's own floor so a fully
                    # masked tile can never move it.  Wide: the PAGE-wide row
                    # max is ONE butterfly shuffle across the 16-lane halves.
                    mx = cutlass.Array(Float32, 8)
                    for n in cutlass.range_constexpr(8):
                        mx[n] = rS[n]
                    for n in cutlass.range_constexpr(8, n_tok):
                        mx[n & 7] = cute.arch.fmax(mx[n & 7], rS[n])
                    for n in cutlass.range_constexpr(4):
                        mx[n] = cute.arch.fmax(mx[n], mx[n + 4])
                    tmx = cute.arch.fmax(
                        cute.arch.fmax(mx[0], mx[1]), cute.arch.fmax(mx[2], mx[3])
                    )
                    if cutlass.const_expr(wide):
                        tmx = cute.arch.fmax(
                            tmx,
                            prims.shfl_sync(
                                thread_mask=0xFFFFFFFF, val=tmx, offset=16,
                                mask_and_clamp=0x1F, kind=prims.Shfl.BFLY,
                            ),
                        )
                    # Lazy running max with hysteresis: the running max is only a
                    # REFERENCE point -- P = exp2(S*scale - m) is exact for any m,
                    # it just sets P's dynamic range.  fp16: hold m until a tile
                    # max exceeds it by `max_hyst`.  e4m3: carry the reference
                    # `max_drop_e4m3` BELOW the tile max (P <= 256 < 448, small
                    # keys 256x clear of the 2^-9 subnormal floor) and re-park it
                    # `max_hyst_e4m3` higher on a jump.  Either way the O-rescale
                    # event (and its BMM2 drain handshake below) vanishes from the
                    # steady state.
                    mold = rM
                    chg = Int32(0)
                    if cutlass.const_expr(kv_e4m3):
                        tm = cute.arch.fmax(tmx * qsc, Float32(neg_floor)) - Float32(max_drop_e4m3)
                        if tm > mold:
                            rM = tm + Float32(max_hyst_e4m3)
                            chg = Int32(1)
                    else:
                        tm = cute.arch.fmax(tmx * qsc, Float32(neg_floor))
                        if tm > mold + Float32(max_hyst):
                            rM = tm
                            chg = Int32(1)
                    corr = Float32(1.0)
                    if chg != 0:
                        corr = exp2(mold - rM)

                    # The rescale vote leaves the critical path: the ballot is
                    # decided by the tile max alone, so it is taken HERE, each
                    # warp's word published, and one elected lane per warp
                    # arrives the slot's mbarrier -- all before the exp block.
                    # The matching wait sits past the P math, so the whole
                    # warpgroup rendezvous hides under the ~64 SFU-bound exp2
                    # per thread instead of serialising after them.
                    wchg = prims.vote_sync(0xFFFFFFFF, chg != 0, "ballot")
                    if cutlass.const_expr(wide):
                        vsl = vping
                    else:
                        vsl = par * 2 + vping
                    if lane == 0:
                        sV4[vsl * 4 + wg_widx] = wchg
                        prims.mbarrier_arrive(cbar.subview(vsl).data_ptr())

                    # P = exp2(S*scale - m), written back over rS, and the
                    # running colsum -- both entirely inside this thread.
                    one2 = (Float32(1.0), Float32(1.0))
                    l0 = Float32(0.0)
                    l1 = Float32(0.0)
                    l2 = Float32(0.0)
                    l3 = Float32(0.0)
                    for n in cutlass.range_constexpr(0, n_tok - poly_keys, 4):
                        v2 = ffma2(
                            (qsc, qsc), (rS[n], rS[n + 1]), (-rM, -rM)
                        )
                        v3 = ffma2(
                            (qsc, qsc), (rS[n + 2], rS[n + 3]), (-rM, -rM)
                        )
                        rS[n] = exp2(v2[0])
                        rS[n + 1] = exp2(v2[1])
                        rS[n + 2] = exp2(v3[0])
                        rS[n + 3] = exp2(v3[1])
                        # Colsum folded with the SAME packed-f32x2 FFMA the
                        # rescale uses: two accumulator lanes per instruction.
                        a2 = ffma2(one2, (rS[n], rS[n + 1]), (l0, l1))
                        a3 = ffma2(one2, (rS[n + 2], rS[n + 3]), (l2, l3))
                        l0 = a2[0]
                        l1 = a2[1]
                        l2 = a3[0]
                        l3 = a3[1]
                    # Hybrid tail: the last `poly_keys` keys run the degree-3
                    # FFMA polynomial (see the module constants) so the SFU
                    # and FMA pipes split the exp work.
                    mg2 = (Float32(exp_magic), Float32(exp_magic))
                    ng2 = (Float32(-exp_magic), Float32(-exp_magic))
                    mo2 = (Float32(-1.0), Float32(-1.0))
                    e3_2 = (Float32(exp_c3), Float32(exp_c3))
                    e2_2 = (Float32(exp_c2), Float32(exp_c2))
                    e1_2 = (Float32(exp_c1), Float32(exp_c1))
                    e0_2 = (Float32(exp_c0), Float32(exp_c0))
                    for n in cutlass.range_constexpr(n_tok - poly_keys, n_tok, 4):
                        v2 = ffma2(
                            (qsc, qsc), (rS[n], rS[n + 1]), (-rM, -rM)
                        )
                        v3 = ffma2(
                            (qsc, qsc), (rS[n + 2], rS[n + 3]), (-rM, -rM)
                        )
                        x0 = cute.arch.fmax(v2[0], Float32(-126.0))
                        x1 = cute.arch.fmax(v2[1], Float32(-126.0))
                        x2 = cute.arch.fmax(v3[0], Float32(-126.0))
                        x3 = cute.arch.fmax(v3[1], Float32(-126.0))
                        t2 = ffma2(one2, (x0, x1), mg2)
                        t3 = ffma2(one2, (x2, x3), mg2)
                        u2 = ffma2(one2, t2, ng2)
                        u3 = ffma2(one2, t3, ng2)
                        f2 = ffma2(mo2, u2, (x0, x1))
                        f3 = ffma2(mo2, u3, (x2, x3))
                        p2 = ffma2(f2, e3_2, e2_2)
                        p3 = ffma2(f3, e3_2, e2_2)
                        p2 = ffma2(f2, p2, e1_2)
                        p3 = ffma2(f3, p3, e1_2)
                        p2 = ffma2(f2, p2, e0_2)
                        p3 = ffma2(f3, p3, e0_2)
                        r0 = (p2[0].bitcast(Int32) + (t2[0].bitcast(Int32) << 23)).bitcast(Float32)
                        r1 = (p2[1].bitcast(Int32) + (t2[1].bitcast(Int32) << 23)).bitcast(Float32)
                        r2 = (p3[0].bitcast(Int32) + (t3[0].bitcast(Int32) << 23)).bitcast(Float32)
                        r3 = (p3[1].bitcast(Int32) + (t3[1].bitcast(Int32) << 23)).bitcast(Float32)
                        rS[n] = r0
                        rS[n + 1] = r1
                        rS[n + 2] = r2
                        rS[n + 3] = r3
                        a2 = ffma2(one2, (r0, r1), (l0, l1))
                        a3 = ffma2(one2, (r2, r3), (l2, l3))
                        l0 = a2[0]
                        l1 = a2[1]
                        l2 = a3[0]
                        l3 = a3[1]
                    lsum = (l0 + l1) + (l2 + l3)
                    if chg != 0:
                        rL = rL * corr + lsum
                    else:
                        rL = rL + lsum

                    # The rescale handshake is per-WG, so the four warps of THIS
                    # group have to agree on whether this tile triggers one.
                    # The votes were published before the exp block; by now the
                    # slot's mbarrier is long complete, so this wait is free.
                    if cutlass.const_expr(wide):
                        # Per-thread P ring slot: the low lanes own stage p_idx (tile 0 of the page),
                        # the high lanes p_idx + 1, with the phase flipping when the +1 crosses the
                        # ring end.
                        pj = p_idx + khalf
                        pph = p_ph
                        if pj >= pstg:
                            pj -= pstg
                            pph ^= 1
                        while not prims.mbarrier_try_wait_parity(
                            pbar.subview(pstg + pj).data_ptr(), pph
                        ):
                            pass
                    while not prims.mbarrier_try_wait_parity(
                        cbar.subview(vsl).data_ptr(), vph
                    ):
                        pass
                    c4 = (sV4.data_ptr() + vsl * 4).load(count=4, alignment=16)
                    anyc = (c4[0] | c4[1]) | (c4[2] | c4[3])
                    vping ^= 1
                    if vping == 0:
                        vph ^= 1

                    if cutlass.const_expr(wide):
                        # Publish P (and the rescale flag) FIRST: the pbar arrive/wait pair orders
                        # the plain flag store, and BMM2 stops in front of this page's MMAs only when
                        # the flag is set.  A thread holds one whole 64-key row of ITS tile, so the
                        # page's two P tiles land in consecutive ring stages as eight 16 B STS each
                        # in the 128 B swizzle the MMA descriptors expect; 64 arrivals per stage.
                        if row == 0:
                            if khalf == 0:
                                sChg[p_idx] = anyc
                        pbase = pj * p_stage_elems + row * n_tok
                        for e in cutlass.range_constexpr(n_tok // 8):
                            sP.subview(pbase + ((Int32(e) ^ (row & 7)) * 8)).data_ptr().store(
                                (rS.data_ptr() + e * 8).load(count=8, alignment=4).to(Float16),
                                alignment=16,
                            )
                        cute.arch.fence_view_async_shared()
                        prims.mbarrier_arrive(pbar.subview(pj).data_ptr())
                        if anyc != 0:
                            # O is first written by tile 0 of page 0, so a rescale
                            # event on that very first tile has nothing to correct.
                            if s > 0:
                                while not prims.mbarrier_try_wait_parity(
                                    obar.subview(0).data_ptr(), rc
                                ):
                                    pass
                                rc ^= 1
                                # Runtime loop, not unrolled: the rescale fires once a walk
                                # (hysteresis).  Every thread rescales its own lane's O columns
                                # with its own corr (the pair shares the max sequence).
                                for cc in cutlass.range((Dc_r >> 1) >> 5):
                                    optr = prims.make_tmem_ptr(
                                        (trow << 16)
                                        | (tmem_base + o_col_offset + cc * 32),
                                        Float32,
                                    )
                                    lo = prims.tcgen05_ld("32x32b", optr, num=32)
                                    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                                    lc = cutlass.Array(Float32, 32)
                                    for n in cutlass.range_constexpr(32):
                                        lc[n] = lo[n] * corr
                                    prims.tcgen05_st(
                                        "32x32b",
                                        optr,
                                        lc.data_ptr().load(count=32, alignment=4),
                                    )
                                prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
                                prims.mbarrier_arrive(obar.subview(1).data_ptr())
                        p_idx += 2
                        if p_idx >= pstg:
                            p_idx -= pstg
                            p_ph ^= 1
                    else:
                        # Publish P (and the rescale flag): each thread packs its
                        # row's 64 P values to the KV dtype and stores them as one
                        # tcgen05.st over the first 32 (fp16) / 16 (e4m3) columns
                        # of THIS stage -- element k*c lands in byte-lane k of word
                        # c, exactly the packed K-major layout the `[a-tmem]` MMA
                        # form reads.  S is long since in registers, so the
                        # overwrite is safe.
                        if row == 0:
                            sChg[s_idx] = anyc
                        pptr = prims.make_tmem_ptr(
                            (trow << 16) | (tmem_base + s_idx * n_tok), kv_t
                        )
                        if cutlass.const_expr(kv_e4m3):
                            for n in cutlass.range_constexpr(n_tok):
                                rPq[n] = rS[n].to(e4m3)
                            prims.tcgen05_st(
                                "32x32b",
                                pptr,
                                rPq.data_ptr().load(count=n_tok, alignment=16),
                            )
                        else:
                            prims.tcgen05_st(
                                "32x32b",
                                pptr,
                                rS.data_ptr().load(count=n_tok, alignment=4).to(Float16),
                            )
                        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
                        cute.arch.fence_view_async_tmem_store()
                        prims.mbarrier_arrive(pbar.subview(s_idx).data_ptr())
                        if anyc != 0:
                            # This WG's O half is first written on page 0, so a
                            # rescale event on page 0 has nothing to correct.
                            if s > 0:
                                while not prims.mbarrier_try_wait_parity(
                                    obar.subview(ob_evt).data_ptr(), rc
                                ):
                                    pass
                                rc ^= 1
                                # Runtime loop, not unrolled: the rescale fires
                                # once a walk (hysteresis), so its body is pure
                                # ifetch weight on every row that never triggers it.
                                for cc in cutlass.range(Dc // 32):
                                    optr = prims.make_tmem_ptr(
                                        (trow << 16)
                                        | (tmem_base + o_col + cc * 32),
                                        Float32,
                                    )
                                    lo = prims.tcgen05_ld("32x32b", optr, num=32)
                                    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                                    lc = cutlass.Array(Float32, 32)
                                    for n in cutlass.range_constexpr(32):
                                        lc[n] = lo[n] * corr
                                    prims.tcgen05_st(
                                        "32x32b",
                                        optr,
                                        lc.data_ptr().load(count=32, alignment=4),
                                    )
                                prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
                                prims.mbarrier_arrive(obar.subview(ob_evt + 1).data_ptr())
                        s_idx += 2
                        if s_idx >= s_stages:
                            s_idx -= s_stages
                            s_ph ^= 1

            # Epilogue
            # It stays under `active`: a split whose window is past the live
            # length must not read O.
            if cutlass.const_expr(wide):
              # The thread pair shares one running-max sequence, so the row's denominator is a
              # plain ADD of the two colsum partials across the 16-lane halves -- one butterfly
              # shuffle, no smem, no barrier.
              if active:
                lo2 = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF, val=rL, offset=16,
                    mask_and_clamp=0x1F, kind=prims.Shfl.BFLY,
                )
                mm = rM
                tots = rL + lo2
                inv = cute.arch.rcp_approx(tots)

                # Attention sink, single-split path: the per-head logit joins
                # the merged row max and the denominator, contributing no V --
                # m' = max(m, s), l' = l*exp2(m-m'), O' = O*exp2(m-m'), all
                # folded into `inv` so O itself is untouched.  Sinks are
                # natural-log logits while mm/tots live in the log2 domain
                # (scale = log2e/sqrt(D)), hence the log2_e conversion done at
                # load time.  Multi-split rows fold the sink exactly once, in
                # the join below, never here.
                if live_splits == 1:
                    if has_sink == 1:
                        mn = cute.arch.fmax(mm, r_sink)
                        cf = exp2(mm - mn)
                        inv = cf * cute.arch.rcp_approx(
                            tots * cf + exp2(r_sink - mn)
                        )
                # Workspace layout is [B*S*HQ(+pad), splits, D] partial-O then
                # the per-split running max and colsum; the offsets are runtime
                # products.  The per-split (m, l) runs are padded to a multiple
                # of four so the join can pull them as aligned 16 B vectors.
                hq_pad = b_dim * S * HQ + rows_tile
                spw = (SPLITS + Int32(3)) & Int32(-4)
                max_off = hq_pad * SPLITS * D
                col_off = max_off + hq_pad * spw
                orow = (b_idx * S + q_tok) * HQ + h_k_idx * gqa_ratio + q_hd
                # Workspace rows are GROUP-CONTIGUOUS: the global row order leaves an HQ - gqa hole
                # after every gqa-run; scratch rows are permuted group-major instead.
                wrow = (h_k_idx * b_dim + b_idx) * rows_total \
                    + q_tok * gqa_ratio + q_hd

                if live_splits != 1:
                    if live_row:
                        if khalf == 0:
                            (mWS.iterator + max_off + wrow * spw + split_idx).store(mm)
                            (mWS.iterator + col_off + wrow * spw + split_idx).store(tots)

                # BMM2 fires a single terminal drain commit after its whole
                # walk, so this is one unambiguous phase-0 wait.
                while not prims.mbarrier_try_wait_parity(
                    obar.subview(2).data_ptr(), 0
                ):
                    pass
                if live_splits == 1:
                    # Only one live split for this (b, h_k, row block): the
                    # reduction kernel collapses into the fused epilogue.  Each
                    # O block of 64 TMEM columns holds this thread's 64 head-dim
                    # values of the block's 128 columns (Layout F: the 16-lane
                    # half picks d 0..63 vs 64..127 of the block).  A head-dim
                    # shard stores its `ob_live` blocks into its own `Dc_r`-wide
                    # slice at `vbase0`.
                    obase = mO.iterator + orow * D + vbase0
                    if cutlass.const_expr(can_dsplit):
                      for vs in cutlass.range(ob_live):
                        optrA = prims.make_tmem_ptr(
                            (trow << 16) | (tmem_base + o_col_offset + vs * Int32(mma_k)),
                            Float32,
                        )
                        rOa = prims.tcgen05_ld("32x32b", optrA, num=64)
                        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                        rOs = cutlass.Array(Float32, 64)
                        for n in cutlass.range_constexpr(64):
                            rOs[n] = rOa[n] * inv
                        if live_row:
                            for e in cutlass.range_constexpr(8):
                                # nvvm.store.ext has no fp16 vector form; push the eight halves out as one b16x8.
                                prims.store_ext(
                                    (rOs.data_ptr() + e * 8)
                                    .load(count=8, alignment=4)
                                    .to(Float16)
                                    .bitcast(cutlass.Int16),
                                    obase + khalf * (Dc_r >> 1) + vs * Int32(mma_k) + e * 8,
                                )
                    else:
                      # The `o_blocks` TMEM reads are issued back to back behind
                      # ONE drain: `tcgen05.wait::ld` retires every outstanding
                      # load, so a wait per block put a second TMEM round trip
                      # on a terminal path that has nothing left to hide it.
                      rOv = []
                      for vs in cutlass.range_constexpr(o_blocks):
                        optrA = prims.make_tmem_ptr(
                            (trow << 16) | (tmem_base + o_col_offset + vs * mma_k),
                            Float32,
                        )
                        rOv.append(prims.tcgen05_ld("32x32b", optrA, num=64))
                      prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                      for vs in cutlass.range_constexpr(o_blocks):
                        rOa = rOv[vs]
                        rOs = cutlass.Array(Float32, 64)
                        for n in cutlass.range_constexpr(64):
                            rOs[n] = rOa[n] * inv
                        if live_row:
                            for e in cutlass.range_constexpr(8):
                                prims.store_ext(
                                    (rOs.data_ptr() + e * 8)
                                    .load(count=8, alignment=4)
                                    .to(Float16)
                                    .bitcast(cutlass.Int16),
                                    obase + khalf * (D // 2) + vs * mma_k + e * 8,
                                )
                else:
                    # The partial is published NORMALISED (O_s / l_s) and in
                    # fp16.  O_s / l_s is a convex combination of V rows, so it
                    # is bounded by max|V| however long the split is; fp16's
                    # 5e-4 relative error sits two orders under the tolerance.
                    # The partial-O run is written d-CHUNK MAJOR:
                    # [split][d/8][row][8], so the same warp writes 32 adjacent
                    # 16 B pieces and the GPU-scope release drains faster.
                    cstride = hq_pad * 8
                    # A head-dim shard's chunks live at their GLOBAL chunk positions inside the
                    # split's [n_chunks_d] run, so the sibling shard's fold never touches them.
                    obase = (
                        mWH.iterator + split_idx * (n_chunks_d * cstride)
                        + (vbase0 >> 3) * cstride + wrow * 8
                    )
                    if cutlass.const_expr(can_dsplit):
                      for vs in cutlass.range(ob_live):
                        optrA = prims.make_tmem_ptr(
                            (trow << 16) | (tmem_base + o_col_offset + vs * Int32(mma_k)),
                            Float32,
                        )
                        rOa = prims.tcgen05_ld("32x32b", optrA, num=64)
                        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                        rOs = cutlass.Array(Float32, 64)
                        for n in cutlass.range_constexpr(64):
                            rOs[n] = rOa[n] * inv
                        if live_row:
                            for e in cutlass.range_constexpr(8):
                                prims.store_ext(
                                    (rOs.data_ptr() + e * 8)
                                    .load(count=8, alignment=4)
                                    .to(Float16)
                                    .bitcast(cutlass.Int16),
                                    obase + (khalf * (nc_live >> 1) + vs * 8 + e) * cstride,
                                )
                    else:
                      # One drain for both TMEM reads (see the single-split path).
                      rOv = []
                      for vs in cutlass.range_constexpr(o_blocks):
                        optrA = prims.make_tmem_ptr(
                            (trow << 16) | (tmem_base + o_col_offset + vs * mma_k),
                            Float32,
                        )
                        rOv.append(prims.tcgen05_ld("32x32b", optrA, num=64))
                      prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                      for vs in cutlass.range_constexpr(o_blocks):
                        rOa = rOv[vs]
                        rOs = cutlass.Array(Float32, 64)
                        for n in cutlass.range_constexpr(64):
                            rOs[n] = rOa[n] * inv
                        if live_row:
                            for e in cutlass.range_constexpr(8):
                                prims.store_ext(
                                    (rOs.data_ptr() + e * 8)
                                    .load(count=8, alignment=4)
                                    .to(Float16)
                                    .bitcast(cutlass.Int16),
                                    obase + (khalf * (n_chunks_d // 2) + vs * 8 + e) * cstride,
                                )

                    # split-K join: the fold itself runs on the whole CTA below; this barrier spans
                    # the softmax WG so every partial store is ordered before the release-carrying
                    # arrival.
                    prims.barrier_cta_sync(
                        wg_pair_barrier, thread_count=warpgroup_threads
                    )
                    if row == 0:
                        if khalf == 0:
                          if clux == 0:
                            old = prims.atomicrmw(
                                prims.AtomicOp.ADD,
                                mSEM.iterator + coord_hb,
                                Int32(1),
                                mem_order=prims.MemOrder.ACQ_REL,
                                syncscope=prims.MemScope.GPU,
                            )
                            # The last arrival hands the slot back at zero, so
                            # the counter never leaves [0, live_splits).
                            sFlag[0] = old

              # TMEM is handed back once every softmax warp's accumulator read
              # retired: a lone warp-4 dealloc can free the columns while any of
              # warps 5-7 is still loading O from them.
              prims.barrier_cta_sync(
                  wg_pair_barrier, thread_count=warpgroup_threads
              )
              if warp_idx == 4:
                  if active:
                      prims.tcgen05_dealloc(
                          prims.make_tmem_ptr(tmem_holder.load(), cutlass.Int8),
                          Int32(tmem_cols),
                      )
            else:
              # Each WG holds its own (running max, colsum) and its own TMEM O
              # half.  Exchange the per-row pairs through smem, merge exactly
              # (the same max/rescale algebra the split-K join uses), then each
              # WG folds BOTH O halves for ITS half of the head dim, so the
              # terminal TMEM drain is split across the two groups.
              if active:
                sRed[(par * rows_tile + row) * 2] = rM
                sRed[(par * rows_tile + row) * 2 + 1] = rL
                prims.barrier_cta_sync(wg_pair_barrier, thread_count=256)
                oix = ((Int32(1) - par) * rows_tile + row) * 2
                mo = sRed[oix]
                lo2 = sRed[oix + 1]
                mm = cute.arch.fmax(rM, mo)
                wS = exp2(rM - mm)
                wO = exp2(mo - mm)
                tots = rL * wS + lo2 * wO
                inv = cute.arch.rcp_approx(tots)

                # Attention sink, single-split path (see the wide epilogue).
                if live_splits == 1:
                    if has_sink == 1:
                        mn = cute.arch.fmax(mm, r_sink)
                        cf = exp2(mm - mn)
                        inv = cf * cute.arch.rcp_approx(
                            tots * cf + exp2(r_sink - mn)
                        )
                # e4m3: `v_scale` rides in HERE, on the NORMALISED accumulator,
                # and only here -- the single-split store and the multi-split
                # partial both go through these factors and the join folds
                # partials that already carry it, so no path double-scales V.
                if cutlass.const_expr(kv_e4m3):
                    inv = inv * v_scale
                # Per-accumulator fold factors: out = O_A * fac_a + O_B * fac_b.
                fac_s = wS * inv
                fac_o = wO * inv
                fac_a = fac_s
                fac_b = fac_o
                if par != 0:
                    fac_a = fac_o
                    fac_b = fac_s

                # Workspace rows are indexed by the CTA-LOCAL row inside this
                # (b, kv-head, row-block[, shard]) group, NOT by the output row:
                # orow interleaves (token, head) with head minor, so consecutive
                # fold threads only ever saw gqa*16 B contiguous runs.  The
                # workspace is pure scratch, so a dense `group-tile * 128 + row`
                # index makes every partial store AND every fold gather
                # coalesced; only the final `out` store still uses orow.
                ws_pad = groups * RB * rows_tile
                if cutlass.const_expr(dsh_ways > 1):
                    ws_pad = ws_pad * Int32(dsh_ways)
                spw = (SPLITS + Int32(3)) & Int32(-4)
                max_off = ws_pad * SPLITS * Dc
                col_off = max_off + ws_pad * spw
                orow = (b_idx * S + q_tok) * HQ + h_k_idx * gqa_ratio + q_hd
                wsrow = coord_hb * Int32(rows_tile // sem_stride) + row

                if live_splits != 1:
                    if live_row:
                        if par == 0:
                            (mWS.iterator + max_off + wsrow * spw + split_idx).store(mm)
                            (mWS.iterator + col_off + wsrow * spw + split_idx).store(tots)

                # BMM2 fires a single terminal drain commit after its whole
                # walk, so this is one unambiguous phase-0 wait.
                while not prims.mbarrier_try_wait_parity(
                    obar.subview(2).data_ptr(), 0
                ):
                    pass
                if live_splits == 1:
                    # Only one live split for this (b, h_k, row block): the
                    # reduction kernel collapses into the fused epilogue.  WG A
                    # folds the low Dc/2 columns, WG B the high ones -- each WG's
                    # slice of both accumulators pulled as ONE tcgen05.ld per
                    # accumulator, so the terminal path carries one blocking
                    # TMEM wait, not two.  An e4m3 shard stores into its own
                    # `Dc`-wide slice at `vbase0`.
                    obase = mO.iterator + orow * D
                    if cutlass.const_expr(dsh_ways > 1):
                        obase = obase + vbase0
                    optrA = prims.make_tmem_ptr(
                        (trow << 16) | (tmem_base + o_col_offset + par * o_half_n),
                        Float32,
                    )
                    optrB = prims.make_tmem_ptr(
                        (trow << 16)
                        | (tmem_base + o_col_offset + Dc + par * o_half_n),
                        Float32,
                    )
                    rOa = prims.tcgen05_ld("32x32b", optrA, num=o_half_n)
                    rOb = prims.tcgen05_ld("32x32b", optrB, num=o_half_n)
                    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                    rOs = cutlass.Array(Float32, o_half_n)
                    for n in cutlass.range_constexpr(o_half_n):
                        rOs[n] = rOa[n] * fac_a + rOb[n] * fac_b
                    if live_row:
                        for e in cutlass.range_constexpr(o_half_n // 8):
                            # nvvm.store.ext has no fp16 vector form; push
                            # the eight halves out as one b16x8.
                            prims.store_ext(
                                (rOs.data_ptr() + e * 8)
                                .load(count=8, alignment=4)
                                .to(Float16)
                                .bitcast(cutlass.Int16),
                                obase + par * o_half_n + e * 8,
                            )
                else:
                    # The partial is published NORMALISED (O_s / l_s) and in
                    # fp16, d-CHUNK MAJOR: [split][Dc/8][row][8] (see the wide
                    # epilogue for both arguments).
                    cstride = ws_pad * 8
                    obase = (mWH.iterator + split_idx * (nchunk * cstride)
                             + wsrow * 8)
                    optrA = prims.make_tmem_ptr(
                        (trow << 16) | (tmem_base + o_col_offset + par * o_half_n),
                        Float32,
                    )
                    optrB = prims.make_tmem_ptr(
                        (trow << 16)
                        | (tmem_base + o_col_offset + Dc + par * o_half_n),
                        Float32,
                    )
                    rOa = prims.tcgen05_ld("32x32b", optrA, num=o_half_n)
                    rOb = prims.tcgen05_ld("32x32b", optrB, num=o_half_n)
                    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                    rOs = cutlass.Array(Float32, o_half_n)
                    for n in cutlass.range_constexpr(o_half_n):
                        rOs[n] = rOa[n] * fac_a + rOb[n] * fac_b
                    if live_row:
                        for e in cutlass.range_constexpr(o_half_n // 8):
                            prims.store_ext(
                                (rOs.data_ptr() + e * 8)
                                .load(count=8, alignment=4)
                                .to(Float16)
                                .bitcast(cutlass.Int16),
                                obase + (par * (nchunk // 2) + e) * cstride,
                            )

                    # split-K join: the fold itself runs on the first 256
                    # threads below; this barrier spans BOTH softmax WGs so
                    # every partial store is ordered before the
                    # release-carrying arrival.
                    prims.barrier_cta_sync(
                        wg_pair_barrier, thread_count=256
                    )
                    if row == 0:
                        if par == 0:
                            old = prims.atomicrmw(
                                prims.AtomicOp.ADD,
                                mSEM.iterator + coord_hb,
                                Int32(1),
                                mem_order=prims.MemOrder.ACQ_REL,
                                syncscope=prims.MemScope.GPU,
                            )
                            # The last arrival hands the slot back at zero, so
                            # the counter never leaves [0, live_splits).
                            sFlag[0] = old

              # TMEM is handed back once every softmax warp's accumulator read
              # retired: a lone warp-4 dealloc can free the columns while any of
              # warps 5-11 is still loading O from them.
              prims.barrier_cta_sync(
                  wg_pair_barrier, thread_count=256
              )
              if warp_idx == 4:
                  # Wide trees alloc unconditionally (cycle-zero hoist), so
                  # every split hands its columns back; narrow trees alloc only
                  # when live, so only live splits dealloc.
                  de_ok = Int32(0)
                  if wide_alloc:
                      de_ok = Int32(1)
                  if active:
                      de_ok = Int32(1)
                  if de_ok != 0:
                      prims.tcgen05_dealloc(
                          prims.make_tmem_ptr(tmem_holder.load(), cutlass.Int8),
                          Int32(tmem_cols),
                      )

        # ======================================= split-K join (whole CTA)
        # `active` and `live_splits` are CTA-uniform, so this is a uniform
        # branch: dead splits and the single-split rows never enter it.
        #
        # The workspace is d-CHUNK major, so a warp folding 32 adjacent rows
        # of one chunk gathers four sectors instead of the thirty-two a
        # row-per-thread deal scattered across the split stride.
        if cutlass.const_expr(wide):
          # Cluster rendezvous FIRST and outside `active`: every CTA of the cluster must arrive,
          # and a dead split has no epilogue to reach.  `live_splits` is derived from seq_lens
          # alone, so the branch is cluster-uniform.  `barrier.cluster.arrive` carries the release
          # and `barrier.cluster.wait` the acquire.
          if live_splits != 1:
            if clux != 0:
                prims.barrier_cta_sync(join_barrier, thread_count=256)
                prims.barrier_cluster_arrive_aligned()
                prims.barrier_cluster_wait_aligned()
          if active:
           if live_splits != 1:
            rank = split_idx
            if clux == 0:
                prims.barrier_cta_sync(join_barrier, thread_count=256)
                # This CTA's arrival order inside its (b, kv-head, row-block)
                # group, published by the softmax warpgroup above.
                rank = sFlag[0]
            # Row window rounded up to a power of two at or above the live
            # row count, so spare threads take extra chunks instead of idling.
            nlive = rows_total - row_base
            if nlive > rows_tile:
                nlive = Int32(rows_tile)
            nr = Int32(rows_tile)
            lg = Int32(6)
            if nlive <= 32:
                nr = Int32(32)
                lg = Int32(5)
            # Distribute only when the single-CTA fold is deep.  The gate is the fold VOLUME (rows
            # x d-chunks), not the row count: a D=512 row folds 64 chunks.  A shard folds only its
            # own `nc_live` chunks.
            dist = Int32(0)
            if nlive * nc_live >= 512:
                dist = Int32(1)
            ranks = Int32(1)
            fold = Int32(0)
            if clux != 0:
                # Every live split CTA folds its own 1/live_splits slice; the cluster barrier
                # already proved the other partials landed, so there is no counter to read.
                fold = Int32(1)
                ranks = live_splits
            elif dist == 0:
                # Last-arrival fold: the counter is handed back at zero right
                # here, so it never leaves [0, live_splits) and no sibling is
                # left polling.
                if rank == live_splits - 1:
                    fold = Int32(1)
                    rank = Int32(0)
                    if tidx == 0:
                        (mSEM.iterator + coord_hb).store(Int32(0))
            else:
                # Distributed fold: every live split CTA takes its own
                # 1/live_splits slice.  RETIRE-FREE: the LAST-ARRIVING CTA hands
                # the counter back right here with one ACQ_REL subtract -- every
                # sibling's arrival is already in coherence order before it, so
                # a waiter observes either the full count or the zero that can
                # only follow it, and both mean "everyone arrived".
                fold = Int32(1)
                ranks = live_splits
                if rank == live_splits - 1:
                    if tidx == 0:
                        prims.atomicrmw(
                            prims.AtomicOp.ADD,
                            mSEM.iterator + coord_hb,
                            Int32(0) - live_splits,
                            mem_order=prims.MemOrder.ACQ_REL,
                            syncscope=prims.MemScope.GPU,
                        )
                else:
                    if tidx == 0:
                        # Probe FIRST: on the narrow rows the siblings usually
                        # arrived while this CTA drained its own partial.
                        arrived = prims.load_ext(
                            mSEM.iterator + coord_hb,
                            dtype=Int32,
                            order=prims.MemOrder.ACQUIRE,
                            scope=prims.MemScope.GPU,
                        )
                        keep = Int32(0)
                        if arrived != 0:
                            if arrived < live_splits:
                                keep = Int32(1)
                        while keep != 0:
                            prims.nanosleep(64)
                            arrived = prims.load_ext(
                                mSEM.iterator + coord_hb,
                                dtype=Int32,
                                order=prims.MemOrder.ACQUIRE,
                                scope=prims.MemScope.GPU,
                            )
                            keep = Int32(0)
                            if arrived != 0:
                                if arrived < live_splits:
                                    keep = Int32(1)
                    prims.barrier_cta_sync(join_barrier, thread_count=256)
            if fold != 0:
                # A folding thread owns ONE row and a run of that row's
                # d-chunks, so the (m, l) pair and the fold weights are
                # gathered once per thread whatever the deal.
                hq_pad = b_dim * S * HQ + rows_tile
                spw = (SPLITS + Int32(3)) & Int32(-4)
                max_off = hq_pad * SPLITS * D
                col_off = max_off + hq_pad * spw
                cstride = hq_pad * 8
                nb2 = Int32(0)
                if live_splits > join_blk:
                    nb2 = Int32(1)
                if cutlass.const_expr(jblocks > 2):
                    if live_splits > 2 * join_blk:
                        nb2 = Int32(2)
                # Thread groups per row, rounded DOWN to a power of two and
                # capped at the chunk count so `ngrp * cper` covers the n_chunks_d chunks exactly.
                ng0 = (ranks * Int32(256)) >> lg
                lgn = Int32(0)
                if ng0 >= 2:
                    lgn = Int32(1)
                if ng0 >= 4:
                    lgn = Int32(2)
                if ng0 >= 8:
                    lgn = Int32(3)
                if ng0 >= 16:
                    lgn = Int32(4)
                if ng0 >= 32:
                    lgn = Int32(5)
                if cutlass.const_expr(can_dsplit):
                    if nc_live >= 64:
                        if ng0 >= 64:
                            lgn = Int32(6)
                else:
                    if cutlass.const_expr(n_chunks_d >= 64):
                        if ng0 >= 64:
                            lgn = Int32(6)
                ngrp = Int32(1) << lgn
                cper = nc_live >> lgn      # ngrp is a power of two: shift
                gid = rank * Int32(256) + tidx
                jrow = gid & (nr - Int32(1))
                cgrp = gid >> lg
                jg = row_base + jrow
                if cgrp < ngrp:
                 if jg < rows_total:
                  jt = (jg * GQM) >> 16
                  jh = jg - jt * gqa_ratio
                  orow = (b_idx * S + jt) * HQ + h_k_idx * gqa_ratio + jh
                  # Group-contiguous workspace row: linear in `jg`, so a
                  # warp's rows gather ADJACENT 16 B pieces -- no holes.
                  wrow = (h_k_idx * b_dim + b_idx) * rows_total + jg
                  mp = mWS.iterator + max_off + wrow * spw
                  lp = mWS.iterator + col_off + wrow * spw
                  mvA = cutlass.Array(Float32, jslots)
                  lvA = cutlass.Array(Float32, jslots)
                  for blk in cutlass.range_constexpr(jslots // 4):
                      if blk * 4 < live_splits:
                          vm = prims.load_ext(mp + blk * 4, dtype=Float32, count=4)
                          vl = prims.load_ext(lp + blk * 4, dtype=Float32, count=4)
                          for e in cutlass.range_constexpr(4):
                              mvA[blk * 4 + e] = vm[e]
                              lvA[blk * 4 + e] = vl[e]
                      else:
                          for e in cutlass.range_constexpr(4):
                              mvA[blk * 4 + e] = Float32(neg_floor)
                              lvA[blk * 4 + e] = Float32(0.0)
                  # Attention sink, multi-split path: folded HERE, exactly
                  # once, into the GLOBAL max across the splits and the
                  # denominator (no V contribution).  Natural-log sink ->
                  # log2 domain.
                  gmn = Float32(neg_floor)
                  for tsp in cutlass.range_constexpr(jslots):
                      if tsp < live_splits:
                          gmn = cute.arch.fmax(gmn, mvA[tsp])
                  snk = Float32(0.0)
                  if has_sink == 1:
                      s2 = mSNK.iterator[h_k_idx * gqa_ratio + jh] * Float32(log2_e)
                      gmn = cute.arch.fmax(gmn, s2)
                      snk = exp2(s2 - gmn)
                  # Partials are normalised, so a split's fold weight is
                  # `l_s * exp2(m_s - M)`; the denominator is that same sum.
                  dv = snk
                  fw = cutlass.Array(Float32, jslots)
                  for tsp in cutlass.range_constexpr(jslots):
                      fw[tsp] = Float32(0.0)
                      if tsp < live_splits:
                          fw[tsp] = exp2(mvA[tsp] - gmn) * lvA[tsp]
                          dv = dv + fw[tsp]
                  rinv = cute.arch.rcp_approx(dv)
                  jb = mWH.iterator + wrow * 8
                  ob = mO.iterator + orow * D
                  # ONE d-chunk per fold pass: the join pays for the REQUEST
                  # count, not the byte count, so no gather is issued whose
                  # value no thread ever reads.  Out-of-range split slots clamp
                  # onto split 0 and carry a weight of exactly 0, so the gather
                  # stays branch-free; the gathered 16 B pieces stay RAW f16
                  # vectors until the fold pass converts them into the FFMAs.
                  acc = cutlass.Array(Float32, 8)
                  for i in cutlass.range(cper):
                   c0 = ds * nc_live + cgrp * cper + i
                   for e in cutlass.range_constexpr(8):
                       acc[e] = Float32(0.0)
                   # Split blocks of `join_blk` fold in a RUNTIME loop; the gathers of a block
                   # stay unrolled, so the memory-level parallelism is unchanged.
                   for blkk in cutlass.range(Int32(1) + nb2):
                       jbase = blkk * Int32(join_blk)
                       vecs = []
                       for t in cutlass.range_constexpr(join_blk):
                           j = jbase + Int32(t)
                           if j >= live_splits:
                               j = Int32(0)
                           jr = jb + (j * n_chunks_d + c0) * cstride
                           vecs.append(prims.load_ext(
                               jr, dtype=cutlass.Int16, count=8
                           ))
                       for t in cutlass.range_constexpr(join_blk):
                         # Dead slots of a partly-filled join block skip the fold on a CTA-uniform
                         # branch instead of running eight converts and FFMAs at weight zero.
                         if jbase + Int32(t) < live_splits:
                           wt = fw[t]
                           if blkk != 0:
                               wt = fw[join_blk + t]
                           if cutlass.const_expr(jblocks > 2):
                               if blkk == 2:
                                   wt = fw[2 * join_blk + t]
                           v = vecs[t]
                           for e in cutlass.range_constexpr(8):
                               acc[e] = acc[e] + wt * (
                                   v[e].bitcast(Float16).to(Float32)
                               )
                   for e in cutlass.range_constexpr(8):
                       acc[e] = acc[e] * rinv
                   prims.store_ext(
                       (acc.data_ptr() + 0).load(count=8, alignment=4)
                       .to(Float16).bitcast(cutlass.Int16),
                       ob + c0 * 8,
                   )
        else:
          if active:
           if live_splits != 1:
            prims.barrier_cta_sync(join_barrier, thread_count=384)
            # This CTA's arrival order inside its (b, kv-head, row-block)
            # group, published by the softmax warpgroup above.
            rank = sFlag[0]
            # Row window rounded up to a power of two at or above the live
            # row count, so spare threads take extra chunks instead of idling.
            nlive = rows_total - row_base
            if nlive > rows_tile:
                nlive = Int32(rows_tile)
            nr = Int32(rows_tile)
            lg = Int32(7)
            if nlive <= 64:
                nr = Int32(64)
                lg = Int32(6)
            if nlive <= 32:
                nr = Int32(32)
                lg = Int32(5)
            # Distribute only when the single-CTA fold is deep: a 32-row tile
            # folds in two chunks and the arrival poll costs more than it saves.
            dist = Int32(0)
            if nlive > 32:
                dist = Int32(1)
            ranks = Int32(1)
            fold = Int32(0)
            if dist == 0:
                # Last-arrival fold: the counter is handed back at zero right
                # here, so it never leaves [0, live_splits) and no sibling is
                # left polling.
                if rank == live_splits - 1:
                    fold = Int32(1)
                    rank = Int32(0)
                    if tidx == 0:
                        (mSEM.iterator + coord_hb).store(Int32(0))
            else:
                # Distributed fold: every live split CTA takes its own
                # 1/live_splits slice (see the wide join for the retire-free
                # counter argument).
                fold = Int32(1)
                ranks = live_splits
                if rank == live_splits - 1:
                    if tidx == 0:
                        prims.atomicrmw(
                            prims.AtomicOp.ADD,
                            mSEM.iterator + coord_hb,
                            Int32(0) - live_splits,
                            mem_order=prims.MemOrder.ACQ_REL,
                            syncscope=prims.MemScope.GPU,
                        )
                else:
                    if tidx == 0:
                        arrived = prims.load_ext(
                            mSEM.iterator + coord_hb,
                            dtype=Int32,
                            order=prims.MemOrder.ACQUIRE,
                            scope=prims.MemScope.GPU,
                        )
                        keep = Int32(0)
                        if arrived != 0:
                            if arrived < live_splits:
                                keep = Int32(1)
                        while keep != 0:
                            prims.nanosleep(64)
                            arrived = prims.load_ext(
                                mSEM.iterator + coord_hb,
                                dtype=Int32,
                                order=prims.MemOrder.ACQUIRE,
                                scope=prims.MemScope.GPU,
                            )
                            keep = Int32(0)
                            if arrived != 0:
                                if arrived < live_splits:
                                    keep = Int32(1)
                    prims.barrier_cta_sync(join_barrier, thread_count=384)
            if fold != 0:
                # A folding thread owns ONE row and a run of that row's
                # d-chunks, so the (m, l) pair and the fold weights are
                # gathered once per thread whatever the deal.
                ws_pad = groups * RB * rows_tile
                if cutlass.const_expr(dsh_ways > 1):
                    ws_pad = ws_pad * Int32(dsh_ways)
                spw = (SPLITS + Int32(3)) & Int32(-4)
                max_off = ws_pad * SPLITS * Dc
                col_off = max_off + ws_pad * spw
                cstride = ws_pad * 8
                nb2 = Int32(0)
                if live_splits > join_blk:
                    nb2 = Int32(1)
                # Thread groups per row, rounded DOWN to a power of two and
                # capped at `nchunk` so `ngrp * cper` covers the row's 16 B
                # pieces exactly: 16 at Dc = 128, 8 at Dc = 64, 4 at Dc = 32.
                ng0 = (ranks * Int32(256)) >> lg
                ngrp = Int32(1)
                cper = Int32(nchunk)                  # d-chunks per thread
                if ng0 >= 2:
                    ngrp = Int32(2)
                    cper = Int32(nchunk // 2)
                if ng0 >= 4:
                    ngrp = Int32(4)
                    cper = Int32(nchunk // 4)
                if cutlass.const_expr(nchunk >= 8):
                    if ng0 >= 8:
                        ngrp = Int32(8)
                        cper = Int32(nchunk // 8)
                if cutlass.const_expr(nchunk >= 16):
                    if ng0 >= 16:
                        ngrp = Int32(16)
                        cper = Int32(nchunk // 16)
                gid = rank * Int32(256) + tidx
                jrow = gid & (nr - Int32(1))
                cgrp = gid >> lg
                jg = row_base + jrow
                if cgrp < ngrp:
                 if jg < rows_total:
                  jt = (jg * GQM) >> 16
                  jh = jg - jt * gqa_ratio
                  orow = (b_idx * S + jt) * HQ + h_k_idx * gqa_ratio + jh
                  wsrow = coord_hb * Int32(rows_tile // sem_stride) + jrow
                  mp = mWS.iterator + max_off + wsrow * spw
                  lp = mWS.iterator + col_off + wsrow * spw
                  mvA = cutlass.Array(Float32, jslots)
                  lvA = cutlass.Array(Float32, jslots)
                  for blk in cutlass.range_constexpr(jslots // 4):
                      if blk * 4 < live_splits:
                          vm = prims.load_ext(mp + blk * 4, dtype=Float32, count=4)
                          vl = prims.load_ext(lp + blk * 4, dtype=Float32, count=4)
                          for e in cutlass.range_constexpr(4):
                              mvA[blk * 4 + e] = vm[e]
                              lvA[blk * 4 + e] = vl[e]
                      else:
                          for e in cutlass.range_constexpr(4):
                              mvA[blk * 4 + e] = Float32(neg_floor)
                              lvA[blk * 4 + e] = Float32(0.0)
                  # Attention sink, multi-split path: folded HERE, exactly
                  # once, into the GLOBAL max across the splits and the
                  # denominator (no V contribution).
                  gmn = Float32(neg_floor)
                  for tsp in cutlass.range_constexpr(jslots):
                      if tsp < live_splits:
                          gmn = cute.arch.fmax(gmn, mvA[tsp])
                  snk = Float32(0.0)
                  if has_sink == 1:
                      s2 = mSNK.iterator[h_k_idx * gqa_ratio + jh] * Float32(log2_e)
                      gmn = cute.arch.fmax(gmn, s2)
                      snk = exp2(s2 - gmn)
                  # Partials are normalised, so a split's fold weight is
                  # `l_s * exp2(m_s - M)`; the denominator is that same sum.
                  dv = snk
                  fw = cutlass.Array(Float32, jslots)
                  for tsp in cutlass.range_constexpr(jslots):
                      fw[tsp] = Float32(0.0)
                      if tsp < live_splits:
                          fw[tsp] = exp2(mvA[tsp] - gmn) * lvA[tsp]
                          dv = dv + fw[tsp]
                  rinv = cute.arch.rcp_approx(dv)
                  jb = mWH.iterator + wsrow * 8
                  ob = mO.iterator + orow * D
                  if cutlass.const_expr(dsh_ways > 1):
                      ob = ob + vbase0
                  # Chunks fold in PAIRS so sixteen 16 B gathers are in
                  # flight; a one-chunk deal predicates the second half away.
                  npr = cper >> 1
                  if npr < 1:
                      npr = Int32(1)
                  hcnt = cper
                  if hcnt > 2:
                      hcnt = Int32(2)
                  acc = cutlass.Array(Float32, 16)
                  for i in cutlass.range(npr):
                   c0 = cgrp * cper + 2 * i
                   # Out-of-range split slots clamp onto split 0 and carry a
                   # weight of exactly 0, so the gather stays branch-free.  The
                   # gathered 16 B pieces stay RAW f16 vectors (4 regs each)
                   # until the fold pass converts them into the FFMAs: an f32
                   # staging buffer pushed the 384-thread build into spills.
                   # Both 16 B chunks are pulled unconditionally -- a one-chunk
                   # deal's second load lands in the sibling chunk's run, still
                   # inside the workspace allocation, and its value is never
                   # consumed (`h < hcnt` gates the fold).
                   for e in cutlass.range_constexpr(16):
                       acc[e] = Float32(0.0)
                   # Split blocks of `join_blk` fold in a RUNTIME loop; the
                   # gathers of a block stay unrolled.
                   for blkk in cutlass.range(Int32(1) + nb2):
                       jbase = blkk * Int32(join_blk)
                       vecs = []
                       for t in cutlass.range_constexpr(join_blk):
                           j = jbase + Int32(t)
                           if j >= live_splits:
                               j = Int32(0)
                           jr = jb + (j * nchunk + c0) * cstride
                           for h in cutlass.range_constexpr(2):
                               vecs.append(prims.load_ext(
                                   jr + h * cstride, dtype=cutlass.Int16, count=8
                               ))
                       for t in cutlass.range_constexpr(join_blk):
                           wt = fw[t]
                           if blkk != 0:
                               wt = fw[join_blk + t]
                           for h in cutlass.range_constexpr(2):
                               if h < hcnt:
                                   v = vecs[t * 2 + h]
                                   for e in cutlass.range_constexpr(8):
                                       acc[h * 8 + e] = acc[h * 8 + e] + wt * (
                                           v[e].bitcast(Float16).to(Float32)
                                       )
                   for h in cutlass.range_constexpr(2):
                       if h < hcnt:
                           for e in cutlass.range_constexpr(8):
                               acc[h * 8 + e] = acc[h * 8 + e] * rinv
                           prims.store_ext(
                               (acc.data_ptr() + h * 8).load(count=8, alignment=4)
                               .to(Float16).bitcast(cutlass.Int16),
                               ob + (c0 + h) * 8,
                           )

    @cute.jit
    def launch(
        pQ: cute.Pointer,
        pKV: cute.Pointer,
        pPT: cute.Pointer,
        pSL: cute.Pointer,
        pSNK: cute.Pointer,
        pO: cute.Pointer,
        pWS: cute.Pointer,
        pWH: cute.Pointer,
        pSEM: cute.Pointer,
        pTM: cute.Pointer,
        B: Int32,
        HKV: Int32,
        NP2: Int32,
        MAXP: Int32,
        gqa: Int32,
        splits: Int32,
        W: Int32,
        CQ: Int32,
        SNK: Int32,
        S: Int32,
        MW: Int32,
        RB: Int32,
        DSH: Int32,
        RBM: Int32, GQM: Int32, SPLM: Int32,
        qk_scale: Float32,
        v_scale: Float32,
        stream: cuda_drv.CUstream,
    ):
        # `qk_scale` is 1/sqrt(D) with k_scale already folded in (e4m3);
        # `v_scale` is 1 for fp16.  The pool stride (NP2, HKV) lives in the TMA
        # descriptor, not in kernel code, so a per-call rebuild covers every
        # pool geometry; only the *box* is fixed, and a box extent of 1 on the
        # page and head modes lets the page id and KV head ride in as runtime
        # TMA coordinates.
        hq = HKV * gqa
        rows = B * S * hq
        mQ = cute.make_tensor(pQ, cute.make_layout(rows * D))
        mKV = cute.make_tensor(pKV, cute.make_layout(
            (NP2, tokens_per_page, HKV, D),
            stride=(tokens_per_page * HKV * D, HKV * D, D, 1)))
        mPT = cute.make_tensor(pPT, cute.make_layout(B * 2 * MAXP))
        mSL = cute.make_tensor(pSL, cute.make_layout(B))
        mSNK = cute.make_tensor(pSNK, cute.make_layout(hq))
        mO = cute.make_tensor(pO, cute.make_layout(rows * D))
        # Scratch: partial-O run, then the per-split running max and colsum.
        # Wide indexes it by global row (+ one tile of padding); narrow by the
        # dense group-tile row (one tile per e4m3 shard).  `pWH` is the fp16
        # alias of the SAME scratch: the half-precision view of the partial-O
        # run ends exactly halfway through it and never reaches the (m, l)
        # arrays behind it.
        if cutlass.const_expr(wide):
            wsn = rows + rows_tile
        else:
            wsn = B * HKV * RB * dsh_ways * rows_tile
        mWS = cute.make_tensor(pWS, cute.make_layout(
            wsn * (splits * Dc + 2 * ((splits + 3) // 4) * 4)))
        mWH = cute.make_tensor(pWH, cute.make_layout(wsn * splits * Dc))
        if cutlass.const_expr(wide):
            mSEM = cute.make_tensor(pSEM, cute.make_layout((B * HKV * RB * sem_stride) << DSH))
        else:
            mSEM = cute.make_tensor(pSEM, cute.make_layout(B * HKV * RB * dsh_ways * sem_stride))
        mTM = cute.make_tensor(pTM, cute.make_layout(B * S * MW))
        tma_k = ccuda.create_tensor_map_tiled_from_view(
            mKV,
            box_dims=(1, tokens_per_k_stage, 1, mma_k),
            stride_order=(3, 2, 1, 0),
            # The box row is one `row_bytes` swizzle period: 128 B (the 128 B
            # swizzle atom) or 64 B (the 64 B atom).  The MMA descriptors carry
            # the matching layout code and SBO, the hand-staged Q tile the XOR.
            swizzle=(ccuda.TensorMapSwizzle.s128b if row_bytes == 128
                     else ccuda.TensorMapSwizzle.s64b),
            # A 256 B promotion turns the bytes right after a box row -- the
            # token's next d-slice or the next kv head, read by the adjacent
            # CTA -- into a hit instead of a second DRAM trip.
            l2_promotion=ccuda.TensorMapL2Promotion.l2_256b,
        )
        if cutlass.const_expr(wide):
            if splits == clu_ext and ((B * HKV * RB) << DSH) <= clu_ext:
                decode(
                    mQ, mPT, mSL, mSNK, mO, mWS, mWH, mSEM, mTM, tma_k,
                    qk_scale * Float32(log2_e), v_scale, MAXP, gqa, W, CQ, SNK, S, MW, RB, DSH,
                    RBM, GQM, SPLM,
                ).launch(grid=[HKV, splits, (B * RB) << DSH], block=[cta_threads, 1, 1],
                         cluster=[1, clu_ext, 1], stream=stream)
            elif splits == clu_ext2 and ((B * HKV * RB) << DSH) <= 4:
                decode(
                    mQ, mPT, mSL, mSNK, mO, mWS, mWH, mSEM, mTM, tma_k,
                    qk_scale * Float32(log2_e), v_scale, MAXP, gqa, W, CQ, SNK, S, MW, RB, DSH,
                    RBM, GQM, SPLM,
                ).launch(grid=[HKV, splits, (B * RB) << DSH], block=[cta_threads, 1, 1],
                         cluster=[1, clu_ext2, 1], stream=stream)
            else:
                decode(
                    mQ, mPT, mSL, mSNK, mO, mWS, mWH, mSEM, mTM, tma_k,
                    qk_scale * Float32(log2_e), v_scale, MAXP, gqa, W, CQ, SNK, S, MW, RB, DSH,
                    RBM, GQM, SPLM,
                ).launch(grid=[HKV, splits, (B * RB) << DSH], block=[cta_threads, 1, 1],
                         stream=stream)
        else:
            # Narrow ignores DSH: fp16 D <= 128 has no shard (host passes 0);
            # e4m3 bakes `dsh_ways` from D (host passes log2(dsh_ways), which
            # is what this grid and the semaphore extent above imply).
            decode(
                mQ, mPT, mSL, mSNK, mO, mWS, mWH, mSEM, mTM, tma_k,
                qk_scale * Float32(log2_e), v_scale, MAXP, gqa, W, CQ, SNK, S, MW, RB, DSH,
                RBM, GQM, SPLM,
            ).launch(grid=[HKV, splits, B * RB * dsh_ways], block=[cta_threads, 1, 1],
                     stream=stream)

    return launch


# Host side
# ONE compiled program per (body, head dim, KV dtype); every other shape is a
# runtime value.  The only per-call state is scratch.  The single-token body
# keeps it per dtype (the fp16 join hands its counters back at zero while the
# e4m3 rendezvous keeps a persistent generation word, so the two must never
# alias) and the tree body keeps its own pair; `_ws` / `_sem` always name the
# buffers the LAST call used.
_launchers = {}            # (body, D, dtype key) -> (launcher, argmode)
_scratch = {}              # single: fp8 -> [ws, ws_ptr, ws_n, sem, sem_ptr, sem_n]
_tree_ws = None            # tree: split-K partials, pure workspace, fully rewritten
_tree_sem = None           # tree: split-K join counters, handed back at zero
_ws = None                 # the last call's split-K partials
_sem = None                # the last call's join counters / flag lines
_budget = 0

# `torch.cuda.current_stream()` builds a Python Stream wrapper on every call;
# the raw accessor hands back the same handle as an int.
try:
    from torch._C import _cuda_getCurrentRawStream as _raw_stream
except ImportError:                                       # pragma: no cover
    _raw_stream = None


def _cta_budget():
    """One fat CTA per SM. The split-K join reads partial-O written by sibling
    CTAs; a fixed 150 (B200's SM count) races on smaller devices (Thor: 20).
    Resolved from the device."""
    import torch as _t
    return _t.cuda.get_device_properties(_t.cuda.current_device()).multi_processor_count


_GM = cutlass.AddressSpace.gmem


def _p16(t):
    return make_ptr(Float16, t.data_ptr(), _GM, assumed_align=8)


def _p8(t):
    return make_ptr(e4m3, t.data_ptr(), _GM, assumed_align=16)


def _p32(t):
    return make_ptr(Int32, t.data_ptr(), _GM, assumed_align=4)


def _pkv(t, fp8):
    return _p8(t) if fp8 else _p16(t)


def _kv_dtype_of(kv_pool):
    if kv_pool.dtype == torch.float8_e4m3fn:
        return "e4m3"
    if kv_pool.dtype == torch.float16:
        return "fp16"
    raise TypeError(f"kv_pool dtype {kv_pool.dtype}: expected float16 or float8_e4m3fn")


# The definition contract fixes the softmax scale at 1/sqrt(D); the wrapper takes
# it as a runtime scalar, one value per head dim.
_ATTN_SCALES = {d: 1.0 / math.sqrt(float(d)) for d in (32, 64, 128, 256, 512)}


_dense_table = {}


def _skip_table_arg(skip_table, device):
    """(int32 device tensor, length) for the trailing skip-softmax table args; None = dense."""
    if skip_table is None:
        t = _dense_table.get(device)
        if t is None:
            t = _dense_table[device] = torch.zeros(2, dtype=torch.int32, device=device)
        return t, 0
    t = skip_table.to(device=device, dtype=torch.int32).contiguous()
    return t, t.numel()


def _run_single(q, kv_pool, page_table, seq_lens, sinks, k_scale, v_scale, window,
                has_sink, out, skip_table=None):
    """Single-token body.  kv_pool.dtype picks the engine: torch.float8_e4m3fn -> e4m3, else fp16.
    fp16 ignores `k_scale` / `v_scale` (its pool carries no dequant scale).
    `skip_table`: flattened int32 [bin_0, S_0, bin_1, S_1, ...] skip-softmax scale table
    (None = dense)."""
    global _ws, _sem, _budget
    fp8 = kv_pool.dtype == torch.float8_e4m3fn
    skt, skt_len = _skip_table_arg(skip_table, q.device)
    qs = q.shape
    B = qs[0]
    HQ = qs[1]
    D = qs[2]
    ks = kv_pool.shape
    HKV = ks[2]
    NP2 = ks[0]
    MAXP = page_table.shape[2]
    gqa = HQ // HKV
    window = int(window)
    hs = int(has_sink)

    if _budget == 0:
        _budget = _cta_budget()
    if fp8:
        # e4m3 plan.  Widening past 16 splits pays ONLY on the long engines
        # (MAXP <= 64 rows lose ~1 us to the extra CTAs), and the 64-wide plan
        # only where a page is expensive AND the engine is long (D = 512).
        hks = 0
        gqa_e = gqa
        hkv_g = HKV
        cap = single_narrow_join_splits
        if MAXP > single_wide_engine_pages and D == 512:
            cap = single_max_join_splits
        splits = _budget // (B * HKV)
        if splits > MAXP:
            splits = MAXP
        if splits > cap:
            splits = cap
        # A sliding window bounds the live span on the HOST: `window` keys
        # reach at most `window // 128 + 2` pages, so every split past that is
        # a CTA that runs a full prologue and then finds `iters == 0`.
        if window > 0:
            wcap = (window >> 7) + 2
            if splits > wcap:
                splits = wcap
        if splits < 1:
            splits = 1
        # Cluster the group's splits only when they fit the mailbox; the kernel
        # keys its join path on the same per-D predicate (`_mc`) AND the ratio
        # (a 16-wide slot would not fit, so ratio-16 rows always join globally).
        csplit = splits if (splits <= _mc(D) and gqa <= single_mma_n) else 1
        pad = single_mma_n_wide
        sem_need = B * HKV * single_sem_words
        # `qkScale *= kCacheScale`, exactly as XQA folds it; the per-head Q
        # dequant scalar is the only part left for the device.
        qk = float(k_scale) * log2_e / math.sqrt(D)
        vs = float(v_scale)
    else:
        # fp16 plan.  GQA ratio 16: the head tile stays 8 wide and the KV head
        # is served by TWO adjacent CTAs of the fast grid mode; `hk_shift` maps
        # a grid group back onto its pool KV head.
        hks = 1 if gqa > 8 else 0
        gqa_e = gqa >> hks
        hkv_g = HKV << hks
        trim_splits = D >= 512          # mirrors build()'s `trim_plan`
        splits = _budget // (B * hkv_g)
        if splits > MAXP:
            splits = MAXP
        if splits > single_max_join_splits:
            splits = single_max_join_splits
        if splits < 1:
            splits = 1
        # The launch width comes off MAXP, the ENGINE capacity, but a sliding
        # window bounds the live page span before `seq_lens` is ever read: the
        # walk covers at most (W>>7)+2 pages for ANY S, so cap the grid here.
        if window > 0:
            wcap = (window >> 7) + 2
            if splits > wcap:
                splits = wcap
        # The D = 512 plan halves its live split count on device whenever the
        # span fits one window per launched split and the group count is above
        # one, so a launch wider than ceil(MAXP/2) can never light more windows
        # -- the extra CTAs are dead by construction.
        elif trim_splits and B * hkv_g != 1:
            tcap = (MAXP + 1) >> 1
            if tcap < 8:
                tcap = 8
            if splits > tcap:
                splits = tcap
        # Join-width tiering: a narrower launch that keeps the SAME deepest
        # window (measured on MAXP) folds fewer partials -- 16 lands in the
        # staged single-round-trip join arm, 32 stays out of the two-round
        # 33..64 arm -- and sheds the dead trailing CTAs.
        if splits > 16:
            d0 = -(-MAXP // splits)
            if -(-MAXP // 16) == d0:
                splits = 16
            elif splits > 32 and -(-MAXP // 32) == d0:
                splits = 32
        csplit = 1
        pad = single_mma_n
        sem_need = 2 * B * hkv_g
        qk = 1.0 / math.sqrt(D)
        vs = 1.0

    # Scratch only: [B, HQ(+pad), splits, D] partial O, then the per-split
    # running max and colsum.  Grow-only and zero-filled, so the padded head
    # rows the join reads back are finite; nothing computed survives a call.
    sc = _scratch.get(fp8)
    if sc is None:
        sc = _scratch[fp8] = [None, 0, -1, None, 0, -1]
    need = (B * HQ + pad) * (splits * D + 2 * splits)
    if sc[2] < need:
        sc[0] = torch.zeros(need, dtype=torch.float32, device=q.device)
        sc[1] = sc[0].data_ptr()
        sc[2] = sc[0].numel()
    if sc[5] < sem_need:
        sc[3] = torch.zeros(sem_need, dtype=torch.int32, device=q.device)
        sc[4] = sc[3].data_ptr()
        sc[5] = sc[3].numel()
    _ws = sc[0]
    _sem = sc[3]
    ws_ptr = sc[1]
    sem_ptr = sc[4]

    if _raw_stream is not None:
        stream = _raw_stream(q.get_device())
    else:
        stream = torch.cuda.current_stream(q.device).cuda_stream

    key = ("single", D, fp8)
    ent = _launchers.get(key)
    if ent is None:
        # The COMPILE-time call still hands over fully typed operands: the
        # element type, address space and alignment of every pointer and the
        # dynamic-ness of every scalar are read off these objects when the IR
        # is built.  Only the LAUNCH path is stripped down.  ONE compile per
        # (head dim, dtype): the cache is keyed on those and nothing else.
        typed = (
            _p16(q), _pkv(kv_pool, fp8), _p32(page_table), _p32(seq_lens),
            make_ptr(Float32, sinks.data_ptr(), _GM, assumed_align=4), _p16(out),
            make_ptr(Float32, ws_ptr, _GM, assumed_align=16),
            make_ptr(Float16, ws_ptr, _GM, assumed_align=16), _p32(_sem),
            Int32(B), Int32(hkv_g), Int32(NP2), Int32(MAXP), Int32(gqa_e),
            Int32(splits), Int32(csplit), Int32(window), Int32(hs), Int32(hks),
            Float32(qk), Float32(vs), cuda_drv.CUstream(stream),
            _p32(skt), Int32(skt_len),
        )
        launcher = cute.compile(_build_single(D, "e4m3" if fp8 else "fp16"), *typed,
                                options="--enable-tvm-ffi")
        # Building the 9 `Pointer` wrappers and the `Int32` wrappers costs more
        # host time than the tiny CI rows spend on the GPU, and the compiled
        # tvm-ffi entry only ever consumes the raw address / raw integer behind
        # them.  Probe once, cheapest first, and keep whatever this build of
        # the DSL actually accepts; a rejected mode raises before the launch.
        raw = (
            q.data_ptr(), kv_pool.data_ptr(), page_table.data_ptr(),
            seq_lens.data_ptr(), sinks.data_ptr(), out.data_ptr(),
            ws_ptr, ws_ptr, sem_ptr,
        )
        scal = (B, hkv_g, NP2, MAXP, gqa_e, splits, csplit, window, hs, hks, qk, vs)
        tail_raw = (skt.data_ptr(), skt_len)
        argmode = 3
        for mode, probe in (
            (0, raw + scal + (stream,) + tail_raw),
            (1, raw + scal + (cuda_drv.CUstream(stream),) + tail_raw),
            (2, typed[:9] + scal + (cuda_drv.CUstream(stream),) + tail_raw),
            (3, typed),
        ):
            try:
                launcher(*probe)
            except Exception:
                continue
            argmode = mode
            break
        _launchers[key] = ent = (launcher, argmode)

    launcher, argmode = ent
    if argmode == 0:
        launcher(
            q.data_ptr(), kv_pool.data_ptr(), page_table.data_ptr(),
            seq_lens.data_ptr(), sinks.data_ptr(), out.data_ptr(), ws_ptr, ws_ptr,
            sem_ptr, B, hkv_g, NP2, MAXP, gqa_e, splits, csplit, window, hs, hks,
            qk, vs, stream, skt.data_ptr(), skt_len,
        )
    elif argmode == 1:
        launcher(
            q.data_ptr(), kv_pool.data_ptr(), page_table.data_ptr(),
            seq_lens.data_ptr(), sinks.data_ptr(), out.data_ptr(), ws_ptr, ws_ptr,
            sem_ptr, B, hkv_g, NP2, MAXP, gqa_e, splits, csplit, window, hs, hks,
            qk, vs, cuda_drv.CUstream(stream), skt.data_ptr(), skt_len,
        )
    elif argmode == 2:
        launcher(
            _p16(q), _pkv(kv_pool, fp8), _p32(page_table), _p32(seq_lens),
            make_ptr(Float32, sinks.data_ptr(), _GM, assumed_align=4), _p16(out),
            make_ptr(Float32, ws_ptr, _GM, assumed_align=16),
            make_ptr(Float16, ws_ptr, _GM, assumed_align=16),
            make_ptr(Int32, sem_ptr, _GM, assumed_align=4),
            B, hkv_g, NP2, MAXP, gqa_e, splits, csplit, window, hs, hks,
            qk, vs, cuda_drv.CUstream(stream), skt.data_ptr(), skt_len,
        )
    else:
        launcher(
            _p16(q), _pkv(kv_pool, fp8), _p32(page_table), _p32(seq_lens),
            make_ptr(Float32, sinks.data_ptr(), _GM, assumed_align=4), _p16(out),
            make_ptr(Float32, ws_ptr, _GM, assumed_align=16),
            make_ptr(Float16, ws_ptr, _GM, assumed_align=16),
            make_ptr(Int32, sem_ptr, _GM, assumed_align=4),
            Int32(B), Int32(hkv_g), Int32(NP2), Int32(MAXP), Int32(gqa_e),
            Int32(splits), Int32(csplit), Int32(window), Int32(hs), Int32(hks),
            Float32(qk), Float32(vs), cuda_drv.CUstream(stream),
            _p32(skt), Int32(skt_len),
        )


def _run_tree(q, kv_pool, page_table, seq_lens, tree_mask, sinks, *rest):
    """Tree body (SPEC_DEC): q / out are [B, S, HQ, D] with S runtime; the mask row
    is MW = ceil(S / 32) words (the first two live in registers, the rest are read
    on the masked tiles), so S is bounded by the caller, not the kernel.

    The KV dtype comes off `kv_pool.dtype`.  Three positional tails are
    accepted, in definition order:
      fp16 D = 128 (no `contiguous` input):  `..., window, has_sink, out`
      fp16 general:                           `..., window, contiguous, has_sink, out`
      e4m3:                `..., k_scale, v_scale, window, contiguous, has_sink, out`
    `window` / `contiguous` / `has_sink` arrive as host ints and ride into the
    kernel as Int32 arguments; `sinks` is a device fp32 [HQ] tensor passed by
    pointer.  e4m3: `k_scale` folds into the softmax scale on the host, exactly
    as XQA folds `kCacheScale` (`qkScale *= kCacheScale`); `v_scale` rides in as
    a runtime scalar and is applied ONCE, on the normalised accumulator.  The
    compile cache is keyed on (D, kv_dtype) and NOTHING else.
    """
    global _ws, _sem, _tree_ws, _tree_sem, _budget
    kv_dtype = _kv_dtype_of(kv_pool)
    k_scale = 1.0
    v_scale = 1.0
    if len(rest) == 6:
        k_scale, v_scale, window, contiguous, has_sink, out = rest
    elif len(rest) == 4:
        window, contiguous, has_sink, out = rest
    else:
        window, has_sink, out = rest
        contiguous = 1
    wv = int(window)
    cq = int(contiguous)
    hs = int(has_sink)
    vs = float(v_scale)
    qs = q.shape
    B = qs[0]
    S = qs[1]
    HQ = qs[2]
    D = qs[3]
    ks = kv_pool.shape
    HKV = ks[2]
    NP2 = ks[0]
    MAXP = page_table.shape[2]
    MW = tree_mask.shape[2]
    gqa = HQ // HKV
    e4 = kv_dtype == "e4m3"
    scale = _ATTN_SCALES.get(D)
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    if e4:
        # `qkScale *= kCacheScale`, exactly as XQA folds it.
        scale = float(k_scale) * scale
    wide = D >= 256 and not e4
    rows_tile = 64 if wide else 128
    # e4m3 head-dim shard (trace-time in the kernel): `dsh_ways` sibling CTAs per
    # (b, kv-head, row-block, split), each owning `Dc` output columns.
    dsh_ways = max(1, D // 128) if e4 else 1
    Dc = D // dsh_ways
    # Row blocks: the flattened (draft token, q-head-in-group) row space of one
    # (b, kv-head) group is S * gqa rows, dealt into tiles of `rows_tile`.
    RB = (S * gqa + rows_tile - 1) // rows_tile

    if _budget == 0:
        _budget = _cta_budget()
    # Split count, inlined (a Python call frame is measurable next to the 5 us
    # CI rows): as many splits as the CTA budget affords, never more than the
    # page-table capacity.  The row blocks (and the e4m3 shards) are already
    # parallelism, so the split count is cut against them too.  How many splits
    # actually do work is decided on device from seq_lens.
    splits = _budget // (B * HKV * RB * dsh_ways)
    if splits > MAXP:
        splits = MAXP
    if wide:
        mjs = 18 if D == 512 else tree_clu_ext2
    else:
        mjs = 16
    if splits > mjs:
        splits = mjs
    elif splits < 1:
        splits = 1
    # MAXP bounds live_pages, and under the join_blk admission guard a deal
    # past tree_narrow_splits must save two pages of depth over the 8-way one --
    # impossible for sequences at or under 3 * tree_narrow_splits pages.  Rows
    # whose capacity already rules that out never launch those splits.  Wide:
    # a sliding window bounds the live page count as hard as MAXP does (the
    # walk can never exceed ceil(window / 128) + 1 pages), and `tree_narrow_splits`
    # is also the cluster extent, so those rows take the cluster join for free.
    cap = MAXP
    if wide and wv > 0:
        wc = (wv >> 7) + 2
        if wc < cap:
            cap = wc
    if cap <= 4 * tree_narrow_splits and splits > tree_narrow_splits:
        splits = tree_narrow_splits
    # `DSH` is a shard SHIFT for both dtypes.  fp16 D == 512: a runtime choice,
    # two CTAs per (b, kv-head, row-block, split) when doubling the grid still
    # fits the machine.  e4m3: log2 of the baked shard count (the kernel
    # re-derives it from D and ignores the argument; passing it keeps the
    # contract's grid / semaphore arithmetic honest).
    dsh = 0
    if wide and D == 512 and 2 * B * HKV * RB * splits <= _budget:
        dsh = 1
    if e4:
        dsh = dsh_ways.bit_length() - 1
    ways = 1 << dsh
    rbm = _MAGIC[RB]
    gqm = _MAGIC[gqa] if gqa < 1025 else _magic(gqa)
    splm = _MAGIC[splits]

    # Scratch only: partial O then the per-split running max and colsum.
    # Grow-only and zero-filled, so the padded rows the join reads back are
    # finite; nothing computed survives a call.
    wsn = (B * S * HQ + rows_tile) if wide else (B * HKV * RB * ways * rows_tile)
    need = wsn * (splits * Dc + 2 * (((splits + 3) // 4) * 4))
    if _tree_ws is None or _tree_ws.numel() < need:
        _tree_ws = torch.zeros(need, dtype=torch.float32, device=q.device)
    nsem = B * HKV * RB * ways * tree_sem_stride
    if _tree_sem is None or _tree_sem.numel() < nsem:
        _tree_sem = torch.zeros(nsem, dtype=torch.int32, device=q.device)
    _ws = _tree_ws
    _sem = _tree_sem

    if _raw_stream is not None:
        stream = _raw_stream(q.device.index)
    else:
        stream = torch.cuda.current_stream(q.device).cuda_stream

    key = ("tree", D, kv_dtype)
    ent = _launchers.get(key)
    if ent is None:
        # The COMPILE-time call still hands over fully typed operands: the
        # element type, address space and alignment of every pointer and the
        # dynamic-ness of every scalar are read off these objects when the IR
        # is built.  Only the LAUNCH path is stripped down.
        typed = (
            _p16(q), _pkv(kv_pool, e4), _p32(page_table), _p32(seq_lens),
            make_ptr(Float32, sinks.data_ptr(), _GM, assumed_align=4), _p16(out),
            make_ptr(Float32, _tree_ws.data_ptr(), _GM, assumed_align=16),
            make_ptr(Float16, _tree_ws.data_ptr(), _GM, assumed_align=16), _p32(_tree_sem),
            _p32(tree_mask),
            Int32(B), Int32(HKV), Int32(NP2), Int32(MAXP), Int32(gqa),
            Int32(splits), Int32(wv), Int32(cq), Int32(hs), Int32(S), Int32(MW),
            Int32(RB), Int32(dsh), Int32(rbm), Int32(gqm), Int32(splm),
            Float32(scale), Float32(vs),
            cuda_drv.CUstream(stream),
        )
        launcher = cute.compile(_build_tree(D, kv_dtype), *typed, options="--enable-tvm-ffi")
        # Building the `Pointer` / `Int32` wrappers costs more host time than
        # the tiny CI rows spend on the GPU, and the compiled tvm-ffi entry
        # only ever consumes the raw address / raw integer behind them.  Probe
        # once, cheapest first, and keep whatever this build of the DSL
        # actually accepts; a rejected mode raises before the launch.
        raw = (
            q.data_ptr(), kv_pool.data_ptr(), page_table.data_ptr(),
            seq_lens.data_ptr(), sinks.data_ptr(), out.data_ptr(),
            _tree_ws.data_ptr(), _tree_ws.data_ptr(), _tree_sem.data_ptr(), tree_mask.data_ptr(),
        )
        scal = (B, HKV, NP2, MAXP, gqa, splits, wv, cq, hs, S, MW, RB, dsh,
                rbm, gqm, splm, scale, vs)
        argmode = 3
        for mode, probe in (
            (0, raw + scal + (stream,)),
            (1, raw + scal + (cuda_drv.CUstream(stream),)),
            (2, typed[:10] + scal + (cuda_drv.CUstream(stream),)),
            (3, typed),
        ):
            try:
                launcher(*probe)
            except Exception:
                continue
            argmode = mode
            break
        _launchers[key] = ent = (launcher, argmode)

    _launcher, _argmode = ent
    if _argmode == 0:
        _launcher(
            q.data_ptr(), kv_pool.data_ptr(), page_table.data_ptr(),
            seq_lens.data_ptr(), sinks.data_ptr(), out.data_ptr(),
            _tree_ws.data_ptr(), _tree_ws.data_ptr(), _tree_sem.data_ptr(), tree_mask.data_ptr(),
            B, HKV, NP2, MAXP, gqa, splits, wv, cq, hs, S, MW, RB, dsh, rbm, gqm, splm,
            scale, vs, stream,
        )
    elif _argmode == 1:
        _launcher(
            q.data_ptr(), kv_pool.data_ptr(), page_table.data_ptr(),
            seq_lens.data_ptr(), sinks.data_ptr(), out.data_ptr(),
            _tree_ws.data_ptr(), _tree_ws.data_ptr(), _tree_sem.data_ptr(), tree_mask.data_ptr(),
            B, HKV, NP2, MAXP, gqa, splits, wv, cq, hs, S, MW, RB, dsh, rbm, gqm, splm,
            scale, vs, cuda_drv.CUstream(stream),
        )
    elif _argmode == 2:
        _launcher(
            _p16(q), _pkv(kv_pool, e4), _p32(page_table), _p32(seq_lens),
            make_ptr(Float32, sinks.data_ptr(), _GM, assumed_align=4), _p16(out),
            make_ptr(Float32, _tree_ws.data_ptr(), _GM, assumed_align=16),
            make_ptr(Float16, _tree_ws.data_ptr(), _GM, assumed_align=16), _p32(_tree_sem),
            _p32(tree_mask),
            B, HKV, NP2, MAXP, gqa, splits, wv, cq, hs, S, MW, RB, dsh, rbm, gqm, splm,
            scale, vs, cuda_drv.CUstream(stream),
        )
    else:
        _launcher(
            _p16(q), _pkv(kv_pool, e4), _p32(page_table), _p32(seq_lens),
            make_ptr(Float32, sinks.data_ptr(), _GM, assumed_align=4), _p16(out),
            make_ptr(Float32, _tree_ws.data_ptr(), _GM, assumed_align=16),
            make_ptr(Float16, _tree_ws.data_ptr(), _GM, assumed_align=16), _p32(_tree_sem),
            _p32(tree_mask),
            Int32(B), Int32(HKV), Int32(NP2), Int32(MAXP), Int32(gqa),
            Int32(splits), Int32(wv), Int32(cq), Int32(hs), Int32(S), Int32(MW),
            Int32(RB), Int32(dsh), Int32(rbm), Int32(gqm), Int32(splm),
            Float32(scale), Float32(vs), cuda_drv.CUstream(stream),
        )


# Public API
def build(D=D_DEFAULT, kv_dtype="fp16", spec_dec=False):
    """The `@cute.jit` launcher for ONE (head dim, KV dtype, body).

    `spec_dec=False` -> the single-token body (24-argument launch contract),
    `spec_dec=True` -> the tree-attention body (29-argument launch contract);
    see the module docstring for both argument orders.
    """
    if kv_dtype not in KV_DTYPES:
        raise ValueError(f"kv_dtype must be one of {KV_DTYPES}, got {kv_dtype!r}")
    if D not in (32, 64, 128, 256, 512):
        raise ValueError(f"head dim must be one of (32, 64, 128, 256, 512), got {D}")
    if spec_dec:
        if kv_dtype == "e4m3" and D == 32:
            raise ValueError("e4m3 KV has no D = 32 tree kernel (use 64, 128, 256 or 512)")
        return _build_tree(D, kv_dtype)
    return _build_single(D, kv_dtype)


def run(*args, tree_mask=None, contiguous=0, skip_table=None):
    """Dispatch on the query rank / argument count; the KV dtype comes off `kv_pool.dtype`.

    Keyword form (canonical):
      run(q, kv_pool, page_table, seq_lens, sinks, k_scale, v_scale, window,
          has_sink, out, tree_mask=None, contiguous=0, skip_table=None)
      `tree_mask is None` -> single-token body; otherwise the tree body with
      `contiguous` (0 = one shared window edge for every draft row, 1 = per-row
      window, XQA CONTIGUOUS_QUERY_SWA).  `skip_table` (single-token body only):
      flattened int32 [bin_0, S_0, bin_1, S_1, ...] skip-softmax scale table,
      lambda = S / seq_len per row; None = dense.

    Positional forms (the campaign harness order, `out` last), single-token
    (q is [B, HQ, D]):
      10 args: q, kv_pool, page_table, seq_lens, sinks, k_scale, v_scale, window, has_sink, out
    tree (q is [B, S, HQ, D], tree_mask [B, S, MW] int32):
       9 args: q, kv_pool, page_table, seq_lens, tree_mask, sinks, window, has_sink, out
               (fp16 D = 128 definition; `contiguous` implied 1)
      10 args: q, kv_pool, page_table, seq_lens, tree_mask, sinks, window, contiguous, has_sink, out
      12 args: q, kv_pool, page_table, seq_lens, tree_mask, sinks, k_scale, v_scale, window,
               contiguous, has_sink, out
    fp16 ignores `k_scale` / `v_scale` (its pool carries no dequant scale).
    """
    if tree_mask is not None:
        if len(args) != 10:
            raise TypeError(f"run(..., tree_mask=) takes 10 positional arguments, got {len(args)}")
        q, kv_pool, page_table, seq_lens, sinks, k_scale, v_scale, window, has_sink, out = args
        return _run_tree(q, kv_pool, page_table, seq_lens, tree_mask, sinks, k_scale, v_scale,
                         window, contiguous, has_sink, out)
    n = len(args)
    if n == 10 and args[0].dim() == 3:
        return _run_single(*args, skip_table=skip_table)
    if n in (9, 10, 12) and args[0].dim() == 4:
        return _run_tree(*args)
    raise TypeError(f"run(): no body accepts {n} positional arguments with a rank-{args[0].dim()} q")


# # AOT export entry (build_cutedsl.py contract).  Every shape quantity is a
# runtime scalar, so one export per (head dim, KV dtype, body) serves all
# (B, S, HQ, HKV, MAXP).  `--target_sm` is accepted (build_cutedsl.py passes it)
# and read by `_target_has_f32_redux`; the DSL toolchain consumes it for the arch.
if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--file_name", required=True)
    ap.add_argument("--function_prefix", required=True)
    ap.add_argument("--export_only", action="store_true")
    ap.add_argument("--head_dim", type=int, default=D_DEFAULT, choices=(32, 64, 128, 256, 512))
    ap.add_argument("--kv_dtype", default=KV_DTYPE_DEFAULT, choices=KV_DTYPES)
    ap.add_argument("--spec_dec", action="store_true", help="export the tree-attention body")
    ap.add_argument("--target_sm", type=int, default=None)
    ns, _ = ap.parse_known_args()
    fp8 = ns.kv_dtype == "e4m3"
    if ns.spec_dec and fp8 and ns.head_dim == 32:
        raise SystemExit("[fmha_decode_ctm] e4m3 KV has no D = 32 tree kernel (use 64, 128, 256 or 512)")

    # Storage-free placeholder operands: only dtype / address space / alignment
    # and the dynamic-ness of each scalar are read at compile time, so the
    # export needs neither a GPU nor torch.
    from cutedsl_utils import aot_placeholders  # isort: skip
    p16 = aot_placeholders.make_ptr(Float16, 8)
    pkv = aot_placeholders.make_ptr(e4m3, 16) if fp8 else p16
    p32 = aot_placeholders.make_ptr(Int32, 4)
    ptrs = [
        p16, pkv, p32, p32,
        aot_placeholders.make_ptr(Float32, 4), p16,
        aot_placeholders.make_ptr(Float32, 16),
        aot_placeholders.make_ptr(Float16, 16), p32,
    ]
    stream = aot_placeholders.make_stream()
    if ns.spec_dec:
        # module + 29: 10 x void*, B HKV NP2 MAXP gqa splits W CQ SNK S MW RB DSH RBM GQM SPLM, qk v, stream
        args = ptrs + [
            p32,
            Int32(1), Int32(1), Int32(2), Int32(1), Int32(1), Int32(1),
            Int32(0), Int32(1), Int32(0), Int32(1), Int32(1), Int32(1), Int32(0),
            Int32(65536), Int32(65536), Int32(65536),  # magic recips
            Float32(1.0), Float32(1.0), stream,
        ]
    else:
        # module + 24: 9 x void*, B HKV NP2 MAXP gqa splits csplit window has_sink hk_shift, qk v, stream,
        # skip_softmax_scale_table, skip_softmax_scale_table_len
        args = ptrs + [
            Int32(1), Int32(1), Int32(2), Int32(1), Int32(1), Int32(1), Int32(1),
            Int32(0), Int32(0), Int32(0), Float32(1.0), Float32(1.0), stream,
            p32, Int32(0),
        ]
    compiled = cute.compile(build(ns.head_dim, ns.kv_dtype, ns.spec_dec), *args)  # plain C ABI (no tvm-ffi)
    os.makedirs(ns.output_dir, exist_ok=True)
    compiled.export_to_c(file_path=ns.output_dir, file_name=ns.file_name,
                         function_prefix=ns.function_prefix)
    print(f"[fmha_decode_ctm] exported {ns.file_name} ({'tree' if ns.spec_dec else 'single'}, "
          f"{ns.kv_dtype}, D={ns.head_dim}) -> {ns.output_dir}/{ns.file_name}.o")
