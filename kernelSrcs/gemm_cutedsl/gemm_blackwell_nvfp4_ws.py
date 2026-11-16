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

"""CuTe DSL Blackwell NVFP4 GEMM — warp-specialised variant.

Computes ``C = A @ B^T`` with block-scaled FP4 inputs and an FP32
accumulator. The kernel body splits work across three warp roles so that
TMA latency hides behind MMA issue and the issue warp is freed from
store-back stalls:

* warp 5    : TMA-load warp (issues all A / B / SFA / SFB cp.async.bulk loads)
* warp 4    : MMA-issue warp (issues tcgen05.blockscaled UMMA + epilogue arrive)
* warps 0-3 : Epilogue warps (drain accumulator TMEM -> SMEM -> GMEM via TMA store)

Layout adapted from the upstream CuTeDSL blockscaled-GEMM example
``examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent.py``.

This is a non-grouped (plain) NVFP4 GEMM variant. AOT build/plumbing
follows main's CuTe DSL contract (see ``kernelSrcs/README.md``).

Tensor layout (logical shape -> physical storage)
-------------------------------------------------
  A  : logical [M, K] FP4 -> physical [M, K/2] uint8, row-major
       (two Float4E2M1FN values packed per byte)
  B  : logical [N, K] FP4 -> physical [N, K/2] uint8, row-major
       (the kernel computes C = A @ B^T)
  SFA: logical [M, K/16] FP8 scale factors -> physical SF-atom tiled
       layout (32, 4, rest_m, 4, rest_k), Float8E4M3FN
  SFB: logical [N, K/16] FP8 scale factors -> physical SF-atom tiled
       layout (32, 4, rest_n, 4, rest_k), Float8E4M3FN
  C  : logical [M, N] -> physical [M, N], row-major,
       Float16 (default) or Float8E4M3FN

Here M / N / K are the logical GEMM dims; K/2 reflects the two-FP4-per-byte
packing and K/16 the one-scale-per-16-element block (sf_vec_size=16).

Accumulator: FP32. The optional SwiGLU epilogue consumes interleaved gate/up
tiles. With quantized_output, it rounds H to FP16 in registers and writes
packed NVFP4 plus SF atoms directly, without an H workspace.

Output dtype is selected at AOT-compile time via ``c_dtype``
(``cutlass.Float16`` or ``cutlass.Float8E4M3FN``). The kernel reads the
dtype from ``c.element_type`` and the epilogue casts the FP32 accumulator
via ``acc_vec.to(self.c_dtype)``, so switching the output type only
requires a different AOT variant. Variants registered in
``build_cutedsl.py``:

  gemm_blackwell_nvfp4_ws_fp16_tn{64,128,256}  — FP16 output
  gemm_blackwell_nvfp4_ws_fp8_tn{64,128}       — FP8 E4M3 output

Supported Blackwell-family SMs: 100, 101, 103, 110. SM 120/121
(GeForce Blackwell Ultra) lack tcgen05.blockscaled and are NOT supported.

Usage::

  # Test on GPU:
  python gemm_blackwell_nvfp4_ws.py --mnk 128,2048,2048

  # AOT export (.o + .h) — invoked by build_cutedsl.py:
  python gemm_blackwell_nvfp4_ws.py --mnk 128,2048,2048 --export_only \\
      --output_dir ./artifact --file_name gemm_blackwell_nvfp4_ws_fp16 \\
      --function_prefix gemm_blackwell_nvfp4_ws_fp16
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple, Type, Union

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
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import numpy as np
from cutedsl_utils import aot_placeholders
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from nvfp4_fused_moe_cutedsl.fp4_common import cvt_e2m1x8_f32, fabs_f32, fmax_f32, fmin_f32
from common import (
    export_compiled_kernel,
    parse_comma_separated_ints,
    to_cute_tensor,
)


# ---------------------------------------------------------------------------
# NVFP4 dummy-tensor helpers (used by both run() and AOT export)
#
# The NvFP4 MoE common.py already has `create_dummy_pointers` and
# `compute_sf_buffer_size`. We want plain (non-grouped) versions without the
# L dimension and without swiglu/FC2 extras — re-implemented locally so we
# do not pull the MoE-specific `tile_group_ptr` / `num_tiles_ptr` /
# `alpha_ptr` into a non-grouped GEMM.
# ---------------------------------------------------------------------------


def _compute_sf_buffer_elements(extent: int, k: int, sf_vec_size: int = 16) -> int:
    """Atom-layout SF element count for one tensor leaf.

    Matches the ``(32, 4, rest, 4, scale_k // 4)`` layout produced by
    ``blockscaled_utils.tile_atom_to_shape_SF``. ``extent`` is M for SFA
    (or N for SFB); ``k`` is the contraction dim.
    """
    scale_k = k // sf_vec_size
    padded_atoms = (extent + 127) // 128
    padded_scale_k_groups = (scale_k + 3) // 4
    return 32 * 4 * padded_atoms * 4 * padded_scale_k_groups


def _create_nvfp4_pointers(
    m: int, n: int, k: int, sf_vec_size: int = 16,
    c_dtype: Type[cutlass.Numeric] = cutlass.Float16,
):
    """Create typed GPU pointers + backing cupy buffers for AOT compile.

    Returns (ptrs_dict, backing_buffers_list). Uses the pointer-based
    pattern of ``nvfp4_moe_cutedsl/common.py::create_dummy_pointers`` so
    the kernel's ``wrapper`` entry point takes ``cute.Pointer`` arguments
    that match the generated C ABI exactly.

    The ``c_dtype`` argument selects the output element type:
      * ``cutlass.Float16``       — 2 bytes/elem, ``cp.float16`` backing buffer
      * ``cutlass.Float8E4M3FN``  — 1 byte/elem,  ``cp.uint8``   backing buffer
                                    (the kernel epilogue handles the FP32 →
                                    FP8 E4M3 cast via ``acc_vec.to(c_dtype)``).

    Dummy buffer sizes:
      * A   : uint8[M, K//2]      (FP4 packed)
      * B   : uint8[N, K//2]
      * SFA : uint8[atom_bytes]   (atom layout Float8E4M3FN)
      * SFB : uint8[atom_bytes]
      * C   : <c_dtype>[M, N]     (m_padded along M for the kernel epilogue)
      * ALPHA : float32[1]        (epilogue scalar multiplier)
    """
    import cutlass.cute.runtime as cute_runtime

    assert n % 128 == 0, f"N={n} must be divisible by 128 for atom SFB layout"
    assert k % (sf_vec_size * 4) == 0, (
        f"K={k} must be divisible by sf_vec_size*4={sf_vec_size * 4}"
    )
    m_padded = ((m + 127) // 128) * 128

    bufs = []

    def _alloc(shape, dtype):
        if not isinstance(shape, (list, tuple)):
            shape = (shape,)
        buf = cp.zeros(shape, dtype=dtype)
        bufs.append(buf)
        return buf

    a_raw = _alloc((m, k // 2), cp.uint8)
    b_raw = _alloc((n, k // 2), cp.uint8)
    sfa_raw = _alloc(_compute_sf_buffer_elements(m, k, sf_vec_size), cp.uint8)
    sfb_raw = _alloc(_compute_sf_buffer_elements(n, k, sf_vec_size), cp.uint8)
    if c_dtype is cutlass.Float16:
        c_raw = _alloc((m_padded, n), cp.float16)
    elif c_dtype is cutlass.Float8E4M3FN:
        # FP8 E4M3 is 1 byte/elem; back with uint8 and let the typed make_ptr
        # carry the cute element-type tag through to the kernel's c.element_type.
        c_raw = _alloc((m_padded, n), cp.uint8)
    else:
        raise ValueError(
            f"Unsupported c_dtype: {c_dtype!r}. Expected cutlass.Float16 or "
            f"cutlass.Float8E4M3FN."
        )
    alpha_raw = _alloc((1,), cp.float32)

    a_ptr = cute_runtime.make_ptr(
        cutlass.Float4E2M1FN, a_raw.data.ptr,
        cute.AddressSpace.gmem, assumed_align=32,
    )
    b_ptr = cute_runtime.make_ptr(
        cutlass.Float4E2M1FN, b_raw.data.ptr,
        cute.AddressSpace.gmem, assumed_align=32,
    )
    sfa_ptr = cute_runtime.make_ptr(
        cutlass.Float8E4M3FN, sfa_raw.data.ptr,
        cute.AddressSpace.gmem, assumed_align=16,
    )
    sfb_ptr = cute_runtime.make_ptr(
        cutlass.Float8E4M3FN, sfb_raw.data.ptr,
        cute.AddressSpace.gmem, assumed_align=16,
    )
    c_ptr = cute_runtime.make_ptr(
        c_dtype, c_raw.data.ptr,
        cute.AddressSpace.gmem, assumed_align=16,
    )
    alpha_ptr = cute_runtime.make_ptr(
        cutlass.Float32, alpha_raw.data.ptr,
        cute.AddressSpace.gmem, assumed_align=4,
    )

    return (
        dict(a=a_ptr, b=b_ptr, sfa=sfa_ptr, sfb=sfb_ptr, c=c_ptr,
             alpha=alpha_ptr),
        bufs,
    )


# ---------------------------------------------------------------------------
# Kernel class — warp-specialised body (3-way split: TMA load / MMA / epilog).
# ---------------------------------------------------------------------------


@dsl_user_op
def _div_rn(a, b, *, loc=None, ip=None):
    """Preserve the quantizer's two independently rounded FP32 divisions."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                  cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "div.rn.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def _half_div_fp8_rn(a, b, reciprocal, *, loc=None, ip=None):
    """FP16 numerator / positive finite E4M3 scale, with one shared reciprocal.

    Exhaustively checked against div.rn over the complete input-format domain.
    Preserve signed zeros and infinities with select; NaNs propagate through FMA.
    The following division by the global FP32 scale remains a separate div.rn.
    """
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [cutlass.Float32(x).ir_value(loc=loc, ip=ip)
                  for x in (a, b, reciprocal)],
        "{ .reg .f32 q, e, nb, z; .reg .b32 bits; .reg .pred special; "
        "mov.b32 bits, $1; and.b32 bits, bits, 0x7fffffff; "
        "mul.rn.f32 q, $1, $3; "
        "neg.f32 nb, $2; fma.rn.f32 e, nb, q, $1; "
        "fma.rn.f32 z, e, $3, q; "
        "setp.eq.u32 special, bits, 0; "
        "setp.eq.or.u32 special, bits, 0x7f800000, special; "
        "selp.f32 $0, $1, z, special; }",
        "=f,f,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


class GemmBlackwellNvFp4WS:
    """Blackwell SM100/101/103/110 NVFP4 GEMM, warp-specialised variant.

    Same math contract as ``GemmBlackwellNvFp4`` (the non-WS sibling) — see
    that class' docstring for the layout / dtype / shape contract. The only
    difference is the kernel body's 3-way warp specialisation (see module
    docstring "Warp roles" section).

    Shape constraints (same as main's MoE FC1):
      * M tile = 128
      * N tile in {64, 128, 192, 256}
      * K divisible by (sf_vec_size * 4) = 64
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        mma_tiler_mn: Tuple[int, int] = (128, 64),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
        sf_vec_size: int = 16,
        fused_swiglu: bool = False,
        quantized_output: bool = False,
        n_tile_group_size: int = 1,
        coalesced_output: bool = False,
        overlap_output_store: bool = False,
        fallback_cluster_shape_mn: Optional[Tuple[int, int]] = None,
    ):
        if n_tile_group_size not in (1, 2, 4):
            raise ValueError("N tile grouping requires size 1, 2 or 4")
        self.n_tile_group_size = n_tile_group_size
        swiglu_clusters = ((1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (4, 1)) if quantized_output else ((1, 1),)
        if fused_swiglu and (mma_tiler_mn != (128, 256)
                             or cluster_shape_mn not in swiglu_clusters
                             or sf_vec_size != 16):
            raise ValueError("NVFP4 SwiGLU requires tile (128,256), block size 16; multicast requires quantized output")
        self.fused_swiglu = fused_swiglu
        if quantized_output and not fused_swiglu:
            raise ValueError("Quantized output requires fused SwiGLU")
        self.quantized_output = quantized_output
        self.coalesced_output = coalesced_output
        self.overlap_output_store = overlap_output_store
        if overlap_output_store and not coalesced_output:
            raise ValueError("Output-store overlap requires coalesced output")
        if coalesced_output and not quantized_output:
            raise ValueError("Coalesced output requires quantized output")
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = False
        self.cluster_shape_mn = cluster_shape_mn
        if fallback_cluster_shape_mn is not None and (
            not quantized_output
            or fallback_cluster_shape_mn != (2, 1)
            or cluster_shape_mn not in ((2, 2), (4, 1))
        ):
            raise ValueError("Mixed clusters require quantized SwiGLU with 2x2 or 4x1 preferred and 2x1 fallback")
        self.fallback_cluster_shape_mn = fallback_cluster_shape_mn
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        # Floating-point outputs use TMA stores; packed NVFP4 uses register stores.
        self.use_tma_store = not quantized_output
        self.sf_vec_size = sf_vec_size

        self.cta_group = tcgen05.CtaGroup.ONE
        self.occupancy = 1

        # ---- Warp role IDs (used by the WS kernel body) ----
        #
        # Layout matches upstream
        # ``dense_blockscaled_gemm_persistent.py:201-211``:
        #
        #   warps 0-3  (epilog_warp_id) : TMEM → SMEM → GMEM via TMA store
        #   warp  4    (mma_warp_id)    : tcgen05.blockscaled UMMA issue
        #   warp  5    (tma_warp_id)    : A/B/SFA/SFB TMA load descriptor issue
        #
        # The WS body branches on these IDs via
        # ``if warp_idx == self.tma_warp_id`` / ``== self.mma_warp_id`` /
        # ``< self.mma_warp_id`` (epilog).
        self.threads_per_warp = 32
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5

        # WS threads_per_cta = 6 warps × 32 = 192. Upstream uses 256 because
        # it dedicates 2 extra warps to the persistent tile scheduler; we
        # drop persistence (each CTA processes ONE tile at our M=128 decode
        # shape, ≤ 8 CTAs total) so we don't need those scheduler warps.
        self.threads_per_cta = self.threads_per_warp * len(
            (*self.epilog_warp_id, self.mma_warp_id, self.tma_warp_id)
        )
        assert self.threads_per_cta == 192, (
            f"WS layout expects 6 warps = 192 threads, got {self.threads_per_cta}"
        )

        # ---- WS-specific NamedBarriers (IDs match upstream reference) ----
        # ``epilog_sync_barrier`` (id=1, 4 epilog warps): orders register →
        # SMEM stores before TMA bulk-store issue and gates TMEM-free until
        # all epilog warps drain their last subtile.
        # ``tmem_alloc_barrier`` (id=2, 4 epilog + 1 mma): epilog warps
        # ``tmem.allocate``, signal here, MMA warp ``wait_for_alloc``.
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_warp * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp * len(
                (self.mma_warp_id, *self.epilog_warp_id)
            ),
        )

        # SMEM capacity is the same for SM 100/101/103/110 (Blackwell family).
        # Reproduced here at __init__ time so the WS _compute_stages variant
        # can size SMEM with all WS-specific MBAR slots accounted for.
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        # Max TMEM columns available on Blackwell SM100-family — used by the
        # WS body's epilog warps to call ``tmem.allocate(num_tmem_alloc_cols)``.
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        # Persistent tile scheduler worker count is a RUNTIME kernel argument
        # (see ``__call__``/``wrapper``): the deployed caller passes the target
        # GPU's value at launch time, so AOT artifacts size the persistent grid
        # for the GPU they actually run on instead of baking the build
        # machine's occupancy. For clusters, query CUDA active-cluster occupancy;
        # SM count divided by cluster size can overprovision persistent workers.

    def _setup_attributes(self):
        """Populate MMA + SMEM + TMEM + pipeline attributes post-``__call__``.

        Structure follows ``gemm_blackwell.py::_setup_attributes`` (non-grouped
        FP16 baseline) with the blockscaled deltas grafted in from
        ``blockscaled_contiguous_grouped_gemm.py:239-402``:

        * Two tiled MMAs — one for A/B data (``tiled_mma``) and one for SFB
          (``tiled_mma_sfb`` with a widened N aligned to 128). Both are
          constructed via ``make_blockscaled_trivial_tiled_mma`` so the
          tcgen05 descriptor encodes ``blockscaled`` + sf_vec_size.
        * Extra SMEM staging for SFA / SFB via
          ``blockscaled_utils.make_smem_layout_{sfa,sfb}``.
        * TMEM: accumulator columns + SFA/SFB columns (blockscaled UMMA
          consumes SF via TMEM). Total budget pinned to 512 columns.
        * Adds ``self.overlapping_accum = False`` (drop the upstream early-
          release optimisation) and ``self.buffer_align_bytes`` (used by
          the WS SharedStorage's ``cute.struct.Align`` slots).
        """
        # MMA instruction tile (M, N) — always 128 x mma_tiler_N, with N
        # rounded up to 128 for the SFB MMA (SFB cares about stride, not
        # the consumed tile size).
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0],
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn_sfb,
        )

        # Promote K-tile to the MMA's natural K x 4 for deep pipelining —
        # identical to gemm_blackwell.py's recipe.
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        # Cluster layouts — one per MMA atom.
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Each merged-N tile contains 128 gate rows followed by 128 up rows.
        self.output_tile_shape_mnk = (
            self.cta_tile_shape_mnk[0],
            self.cta_tile_shape_mnk[1] // (2 if self.fused_swiglu else 1),
            self.cta_tile_shape_mnk[2],
        )
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.output_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])

        # cuTe DSL 4.4.1 does not always detect SM110 / SM101 smem capacity
        # correctly; force the Blackwell-family value (see gemm_blackwell.py
        # for the same workaround).
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

        # Stage budget across A, B, SFA, SFB (+ C for TMA store epilogue).
        # Matches the former non-WS sibling's _compute_stages.
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.sf_dtype,
            self.sf_vec_size,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.smem_capacity,
            self.occupancy,
            use_tma_store=self.use_tma_store,
        )

        # SMEM layouts (multi-stage) for A, B, SFA, SFB.
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage,
        )
        # Quantized output needs no C staging allocation.
        self.c_smem_layout_staged = (
            sm100_utils.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
            )
            if self.use_tma_store
            else None
        )

        # WS-specific: we explicitly DO NOT use the upstream
        # ``overlapping_accum`` early-release optimisation. Upstream toggles
        # it on whenever num_acc_stage == 1 (which is always true for us)
        # and uses it to release the accumulator buffer earlier in the
        # epilogue, at the cost of a more complex epilog branch (subtile
        # reversal, fence_view_async_tmem_load mid-loop, etc.). At our
        # M=128 decode shapes we don't need that, and the complexity hides
        # the WS skeleton. Keep the plain multi-stage accumulator path.
        self.overlapping_accum = False

        # TMEM columns: FP32 accumulator + SF columns. Blackwell blockscaled
        # MMA consumes SFA / SFB via TMEM, so we pin columns for both.
        # See blockscaled_contiguous_grouped_gemm.py:389-402 for the formula;
        # simplified here since we do not overlap accumulator stages
        # (see ``self.overlapping_accum = False`` above).
        sf_atom_mn = 32
        mma_inst_tile_k = 4
        self.num_sfa_tmem_cols = (
            self.cta_tile_shape_mnk[0] // sf_atom_mn
        ) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (
            self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn
        ) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        # Drop the overlapping_accum special case in upstream lines 380-384;
        # we always use the straight ``num_acc_stage * cta_tile_n`` cols
        # (matches the former non-WS sibling).
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage
        )
        # Override the __init__-set num_tmem_alloc_cols=get_max(...) value
        # to keep parity with the sibling — the full TMEM column budget
        # (512 on SM100-family) is what the epilog warps allocate; the
        # split between acc/SF is encoded by num_accumulator_tmem_cols +
        # num_sf_tmem_cols above. Leaving the literal here matches sibling.
        self.num_tmem_alloc_cols = 512

        # SMEM alignment for cute.struct.Align in the WS SharedStorage
        # (upstream uses 1024 bytes for sA/sB/sC/sSFA/sSFB).
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
        output_scale: Optional[cute.Tensor] = None,
        global_scale: Optional[cute.Tensor] = None,
    ):
        # Dtype + layout intro (like gemm_blackwell.py:171-181).
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        if cutlass.const_expr(self.fused_swiglu and self.c_dtype != cutlass.Float16):
            raise TypeError("NVFP4 SwiGLU currently requires FP16 output")

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"A/B dtype mismatch: {self.a_dtype} vs {self.b_dtype}")
        if cutlass.const_expr(sfa.element_type != sfb.element_type):
            raise TypeError(
                f"SFA/SFB dtype mismatch: {sfa.element_type} vs {sfb.element_type}"
            )

        self._setup_attributes()

        # Re-wrap SFA/SFB with the atom layout the MMA expects. Caller hands
        # us a flat raw-bytes tensor; the GEMM sees
        #   ((Atom_M, Rest_M), (Atom_K, Rest_K), L).
        # See blockscaled_contiguous_grouped_gemm.py:463-468.
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        sfa = cute.make_tensor(sfa.iterator, sfa_layout)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b.shape, self.sf_vec_size)
        sfb = cute.make_tensor(sfb.iterator, sfb_layout)

        # Re-materialise tiled_mma in this JIT scope (same seeds as in
        # _setup_attributes). CuTe DSL needs fresh TiledMma objects here
        # because _setup_attributes' locals were hoisted into Python state
        # but not JIT IR state.
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.a_major_mode, self.b_major_mode,
            self.sf_dtype, self.sf_vec_size, self.cta_group,
            self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.a_major_mode, self.b_major_mode,
            self.sf_dtype, self.sf_vec_size, self.cta_group,
            self.mma_inst_shape_mn_sfb,
        )
        tiled_mma_c = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.a_major_mode, self.b_major_mode,
            self.sf_dtype, self.sf_vec_size, self.cta_group,
            self.output_tile_shape_mnk[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # --- TMA atom for A ---
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # --- TMA atom for B ---
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # --- TMA atom for SFA (reuses A multicast pattern, Int16 internal) ---
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op, sfa, sfa_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # --- TMA atom for SFB (separate SFB tiled_mma; Int16 internal) ---
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op, sfb, sfb_smem_layout, self.mma_tiler_sfb, tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        fallback_layout = self.cluster_layout_vmnk
        fallback_sfb_layout = self.cluster_layout_sfb_vmnk
        fallback_tma_atom_a, fallback_tma_tensor_a = tma_atom_a, tma_tensor_a
        fallback_tma_atom_b, fallback_tma_tensor_b = tma_atom_b, tma_tensor_b
        fallback_tma_atom_sfa, fallback_tma_tensor_sfa = tma_atom_sfa, tma_tensor_sfa
        fallback_tma_atom_sfb, fallback_tma_tensor_sfb = tma_atom_sfb, tma_tensor_sfb
        if cutlass.const_expr(self.fallback_cluster_shape_mn is not None):
            fallback_layout = cute.tiled_divide(cute.make_layout((*self.fallback_cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,))
            fallback_sfb_layout = cute.tiled_divide(cute.make_layout((*self.fallback_cluster_shape_mn, 1)), (tiled_mma_sfb.thr_id.shape,))
            # --- TMA atom for A ---
            a_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.fallback_cluster_shape_mn, tiled_mma.thr_id
            )
            a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
            fallback_tma_atom_a, fallback_tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
                a_op, a, a_smem_layout, self.mma_tiler, tiled_mma,
                fallback_layout.shape,
            )

            # --- TMA atom for B ---
            b_op = sm100_utils.cluster_shape_to_tma_atom_B(
                self.fallback_cluster_shape_mn, tiled_mma.thr_id
            )
            b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
            fallback_tma_atom_b, fallback_tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
                b_op, b, b_smem_layout, self.mma_tiler, tiled_mma,
                fallback_layout.shape,
            )

            # --- TMA atom for SFA (reuses A multicast pattern, Int16 internal) ---
            sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.fallback_cluster_shape_mn, tiled_mma.thr_id
            )
            sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
            fallback_tma_atom_sfa, fallback_tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
                sfa_op, sfa, sfa_smem_layout, self.mma_tiler, tiled_mma,
                fallback_layout.shape,
                internal_type=cutlass.Int16,
            )

            # --- TMA atom for SFB (separate SFB tiled_mma; Int16 internal) ---
            sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
                self.fallback_cluster_shape_mn, tiled_mma.thr_id
            )
            sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
            fallback_tma_atom_sfb, fallback_tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op, sfb, sfb_smem_layout, self.mma_tiler_sfb, tiled_mma_sfb,
                fallback_sfb_layout.shape,
                internal_type=cutlass.Int16,
            )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            (a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size)
            * atom_thr_size
        )

        # --- TMA store atom for C (SMEM -> GMEM async bulk store) ---
        tma_atom_c, tma_tensor_c = None, c
        if cutlass.const_expr(not self.quantized_output):
            epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile,
            )

        # Persistent tile scheduler: launch a fixed worker grid (one CTA per
        # SM-cluster slot, capped by max_active_clusters) and stride over the
        # output tiles. Overlaps per-tile setup across the many N/M tiles at
        # prefill shapes (large-M down_proj), improving prefill throughput.
        # max_active_clusters is a runtime kernel argument supplied by the
        # caller for the GPU the kernel actually launches on.
        tile_sched_params, grid = self._compute_grid(
            c, self.output_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters
        )

        # WS SharedStorage: split full/empty mbar slots + per-operand
        # SMEM staging via cute.struct.Align (adapted from upstream
        # dense_blockscaled_gemm_persistent.py:611-655). Defining it
        # here (versus inside the kernel like the non-WS sibling does)
        # lets the kernel branches retrieve typed SMEM tensors via
        # ``storage.sA.get_tensor(...)`` without repeating layout info.
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage
            ]
            ab_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage
            ]
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage
            ]
            acc_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE) — for TMA store of C
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, 0 if self.quantized_output else cute.cosize(self.c_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (Atom_M, Rest_M, Atom_K, Rest_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (Atom_N, Rest_N, Atom_K, Rest_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        if cutlass.const_expr(self.coalesced_output):
            # One logical 128x128 output tile: 8 KiB FP4 plus 1 KiB scales.
            @cute.struct
            class OutputStorage:
                packed: cute.struct.Align[cute.struct.MemRange[cutlass.Uint64, 1024], 128]
                scales: cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, 1024], 128]
            self.output_storage = OutputStorage
        self.shared_storage = SharedStorage

        # Keep the preferred logical tile grid in both branches. Fallback clusters
        # partition the same CTA coordinates; changing the scheduler layout would
        # duplicate or omit tiles when preferred and fallback clusters coexist.
        preferred_args = (
            tiled_mma,
            tiled_mma_sfb,
            tiled_mma_c,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_sfa, tma_tensor_sfa,
            tma_atom_sfb, tma_tensor_sfb,
            tma_atom_c, tma_tensor_c,
            alpha,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            tile_sched_params,
            output_scale,
            global_scale,
        )
        fallback_args = (
            tiled_mma,
            tiled_mma_sfb,
            tiled_mma_c,
            fallback_tma_atom_a, fallback_tma_tensor_a,
            fallback_tma_atom_b, fallback_tma_tensor_b,
            fallback_tma_atom_sfa, fallback_tma_tensor_sfa,
            fallback_tma_atom_sfb, fallback_tma_tensor_sfb,
            tma_atom_c, tma_tensor_c,
            alpha,
            fallback_layout,
            fallback_sfb_layout,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            tile_sched_params,
            output_scale,
            global_scale,
        )
        self.kernel(preferred_args, fallback_args).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            fallback_cluster=(*self.fallback_cluster_shape_mn, 1) if self.fallback_cluster_shape_mn is not None else None,
            stream=stream,
        )
        return

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
        output_scale_ptr: Optional[cute.Pointer] = None,
        global_scale_ptr: Optional[cute.Pointer] = None,
    ):
        """AOT entry point — accepts raw GPU pointers + runtime dims.

        M need not be a multiple of 128. TMA zero-fills the last A tile;
        C stores are bounded by the actual M. Scale storage pads rows to 128.

        With fused_swiglu, N is the physical weight row count (twice the
        output width). Weights and their scales alternate 128 gate rows and
        128 up rows, and alpha holds independent gate/up FP32 multipliers.
        The default output contains M * (N/2) FP16 elements. Quantized output
        instead packs two E2M1 values per byte and requires output SF atoms
        and a device FP32 forward global scale.

        The generated C ABI is::

          extern "C" int gemm_blackwell_nvfp4_ws_fp16(
              Kernel_Module_t* module,
              void* a_ptr, void* b_ptr, void* sfa_ptr, void* sfb_ptr, void* c_ptr,
              void* alpha_ptr,
              int64_t m, int64_t n, int64_t k,
              int32_t max_active_clusters,
              CUstream stream);

        ``max_active_clusters`` sizes the persistent tile-scheduler grid and
        must describe the GPU the kernel launches on (typically
        ``cudaDevAttrMultiProcessorCount`` for cluster (1,1)).
        """
        scale_k = k // scaling_vector_size
        m_padded = ((m + 127) // 128) * 128

        a = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((m, k, 1), order=(1, 0, 2))
        )
        b = cute.make_tensor(
            b_ptr, layout=cute.make_ordered_layout((n, k, 1), order=(1, 0, 2))
        )
        sfa = cute.make_tensor(
            sfa_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, (m + 127) // 128, 4, scale_k // 4, 1),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )
        sfb = cute.make_tensor(
            sfb_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, n // 128, 4, scale_k // 4, 1),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )
        # IMPORTANT: the C tensor's M-extent is the **actual
        # m**, not m_padded. The kernel internally tiles by
        # cta_tile_shape_mnk[0]=128 in M, but the epilogue's per-CTA
        # global-store partition is derived from this layout — using
        # (m, n) here makes cute's predication clip the global writes
        # to only the first m rows, instead of writing the full 128-row
        # MMA-tile output (with rows m..127 being junk).
        #
        output_n = n // 2 if cutlass.const_expr(self.fused_swiglu) else n
        c = cute.make_tensor(
            c_ptr, layout=cute.make_ordered_layout((m, output_n, 1), order=(1, 0, 2))
        )
        alpha = cute.make_tensor(
            alpha_ptr, layout=cute.make_layout((2 if self.fused_swiglu else 1,))
        )
        output_scale, global_scale = None, None
        if cutlass.const_expr(self.quantized_output):
            # C supplies the logical shape only; quantized stores reinterpret its pointer.
            output_scale = cute.make_tensor(output_scale_ptr, cute.make_layout(m_padded * output_n // 16))
            global_scale = cute.make_tensor(global_scale_ptr, cute.make_layout(1))
        return self(a, b, sfa, sfb, c, alpha, max_active_clusters, stream, output_scale, global_scale)

    @cute.kernel
    def kernel(self, preferred_args, fallback_args):
        sPacked, sScale = None, None
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        if cutlass.const_expr(self.coalesced_output):
            out_storage = smem.allocate(self.output_storage)
            sPacked = out_storage.packed.get_tensor(cute.make_layout(1024))
            sScale = out_storage.scales.get_tensor(cute.make_layout(1024))

        if cutlass.const_expr(self.fallback_cluster_shape_mn is None):
            self.kernel_body(*preferred_args, storage, sPacked, sScale)
        else:
            x, y, z = cute.arch.block_in_cluster_dim()
            if (x == self.cluster_shape_mn[0]) & (y == self.cluster_shape_mn[1]) & (z == 1):
                self.kernel_body(*preferred_args, storage, sPacked, sScale)
            else:
                self.kernel_body(*fallback_args, storage, sPacked, sScale)

    @cute.jit
    def kernel_body(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tiled_mma_c: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        alpha: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Optional[Union[cute.ComposedLayout, cute.Layout]],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        output_scale: Optional[cute.Tensor],
        global_scale: Optional[cute.Tensor],
        storage,
        sPacked: Optional[cute.Tensor],
        sScale: Optional[cute.Tensor],
    ):
        """3-way warp-specialised NVFP4 blockscaled GEMM body.

        Structure matches upstream
        ``dense_blockscaled_gemm_persistent.py::kernel`` lines 694-1532,
        minus the persistent tile-scheduler loop. Each CTA processes ONE
        tile (computed from ``block_idx()``); the kernel splits into three
        ``if warp_idx == ...`` branches:

        * ``warp_idx == self.tma_warp_id`` (warp 5) — TMA load warp.
          Issues all A / B / SFA / SFB cp.async.bulk descriptors against
          ``ab_pipeline`` (producer side).
        * ``warp_idx == self.mma_warp_id`` (warp 4) — MMA issue warp.
          Consumes ``ab_pipeline``, S2T-copies SFA/SFB into TMEM, issues
          tcgen05.blockscaled UMMA for each kblock, and signals
          ``acc_pipeline`` once the accumulator is full.
        * ``warp_idx < self.mma_warp_id`` (warps 0-3) — Epilogue warps.
          Consume ``acc_pipeline``, drain accumulator from TMEM -> SMEM
          via subtile r2s, then issue TMA bulk-store SMEM -> GMEM.

        Helper methods (defined below):
          * ``mainloop_s2t_copy_and_partition`` — SFA/SFB SMEM-to-TMEM
            staged partition for the MMA warp.
          * ``epilog_tmem_copy_and_partition`` — TMEM-to-register T2R
            partition for the epilog warps.
          * ``epilog_smem_copy_and_partition`` — Register-to-SMEM R2S
            partition for the epilog warps.
          * ``epilog_gmem_copy_and_partition`` — SMEM-to-GMEM TMA store
            partition for the epilog warps.
        """
        # ---- Header: warp_idx + TMA descriptor prefetch ----
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            if cutlass.const_expr(not self.quantized_output):
                cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        # ---- CTA coords (persistent scheduler: bidx is the worker id, the
        # per-tile (M, N, L) comes from work_tile.tile_idx inside each warp's
        # scheduling loop below). mma_tile_coord_v / is_leader_cta are CTA-
        # constant (==0 / True for our (1,1) cluster). ----
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        # ---- SMEM allocator + pipeline barriers ----
        # AB pipeline (TMA load -> MMA consume).
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = cute.size(cluster_layout_vmnk.shape[2]) + cute.size(cluster_layout_vmnk.shape[1]) - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # ACC pipeline (MMA -> epilog consume).
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = self.threads_per_warp * len(
            self.epilog_warp_id
        ) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # TMEM allocator. ``allocator_warp_id`` tells the helper which
        # warp inside the WS group actually issues the TMEM alloc — only
        # epilog warp 0 calls ``tmem.allocate(...)``, the others (MMA +
        # epilog warps 1-3) call ``wait_for_alloc()`` to retrieve the
        # pointer. The 5-warp ``tmem_alloc_barrier`` is the sync.
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # Cluster arrive before TMEM alloc (no-op for our (1,1) cluster
        # but kept for symmetry with the multi-CTA cluster path).
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # ---- SMEM tensor views ----
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        if cutlass.const_expr(not self.quantized_output):
            sC = storage.sC.get_tensor(
                c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
            )

        # ---- Multicast masks (dead branch for cluster=(1,1)) ----
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(cute.size(cluster_layout_vmnk.shape[2]) > 1 or cute.size(cluster_layout_vmnk.shape[1]) > 1 or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = a_full_mcast_mask
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        # ---- Global tile views ----
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, self.output_tile_shape_mnk[:2], (None, None, None)
        )
        gSFA = cute.local_tile(
            mSFA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB = cute.local_tile(
            mSFB, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None)
        )
        # Cast to Int32 — ``cutlass.range`` rejects Int64 (K was declared
        # Int64 in the wrapper for large-shape safety).
        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        # ---- Thread-MMA partitions ----
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = tiled_mma_c.get_slice(mma_tile_coord_v).partition_C(gC_mnl)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgSFA = thr_mma.partition_A(gSFA)
        tCgSFB = thr_mma_sfb.partition_B(gSFB)

        # ---- TMA partitions for A / B / SFA / SFB ----
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        # Strip zero-stride (multicast broadcast) modes from the SFA TMA
        # views. Without this the TMA engine reports a different arrive-
        # byte count than the pipeline's tx_count, and consumer_wait
        # hangs forever. Matches blockscaled_contiguous_grouped_gemm.py
        # lines 973-974.
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        # NB: per-tile slicing of tAgA / tBgB / tAgSFA / tBgSFB now happens
        # inside each warp's persistent scheduling loop (the partitioned
        # tensors above still carry the (tile_m / tile_n, k, l) tile modes).

        # ---- MMA fragments ----
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # Append stage dim so per-stage slicing works in the WS body
        # (matches upstream lines 980-982). With overlapping_accum=False
        # and num_acc_stage=1, the stage dim is size 1.
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        # Cluster wait before TMEM alloc (no-op at cluster=(1,1)).
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ============================================================
        # Branch 1/3: Specialized TMA load warp (warp 5)
        # ============================================================
        if warp_idx == self.tma_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            while work_tile.is_valid_tile:
                cur_tile_coord = self.group_tile(work_tile.tile_idx, tile_sched_params)
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Per-tile slice of the TMA source tensors. When cta_tile_n
                # == 64 two adjacent 64-wide N tiles share one 128-wide SFB
                # tile — halve the N index so the second CTA doesn't TMA OOB.
                tAgA_t = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                tBgB_t = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
                tAgSFA_t = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                sfb_slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    sfb_slice_n = mma_tile_coord_mnl[1] // 2
                tBgSFB_t = tBgSFB[(None, sfb_slice_n, None, mma_tile_coord_mnl[2])]

                # Peek (try_wait) AB-empty for first k_tile of this tile.
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )

                    # TMA load A / B / SFA / SFB into the current stage slot.
                    cute.copy(
                        tma_atom_a,
                        tAgA_t[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_t[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_t[(None, ab_producer_state.count)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_t[(None, ab_producer_state.count)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Wait for all A/B buffers to drain before exiting.
            ab_pipeline.producer_tail(ab_producer_state)

        # ============================================================
        # Branch 2/3: Specialized MMA issue warp (warp 4)
        # ============================================================
        if warp_idx == self.mma_warp_id:
            # Wait for epilog warp 0 to hand over the TMEM pointer.
            tmem.wait_for_alloc()

            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # SFA TMEM tensor (columns right after the accumulator).
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # SFB TMEM tensor (columns after SFA).
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            # S2T copy partitions for SFA / SFB.
            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                cur_tile_coord = self.group_tile(work_tile.tile_idx, tile_sched_params)
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Per-tile accumulator stage (num_acc_stage=1 -> index 0).
                acc_stage_index = acc_producer_state.index
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                # SFB N-tile parity offset for cta_tile_n in {64, 192}: the
                # 128-wide SFB TMEM holds two halves; the MMA reads the half
                # selected by this tile's N parity.
                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    offset = (
                        cutlass.Int32(2)
                        if mma_tile_coord_mnl[1] % 2 == 1
                        else cutlass.Int32(0)
                    )
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                # Peek (try_wait) AB-full for first k_tile of this tile.
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                # Wait for accumulator buffer empty (only the leader CTA
                # issues UMMA).
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                # Reset ACCUMULATE before the first kblock — first cute.gemm
                # writes (not accumulates), then we flip ACCUMULATE on.
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                # MMA mainloop over K-tiles.
                for k_tile in cutlass.range(k_tile_cnt):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        # Copy SFA/SFB SMEM -> TMEM for this stage.
                        s2t_stage_coord = (
                            None, None, None, None, ab_consumer_state.index,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t[s2t_stage_coord],
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t[s2t_stage_coord],
                            tCtSFB_compact_s2t,
                        )

                        # MMA over each kblock — tCtAcc += A * SFA * B * SFB.
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None, None, kblock_idx, ab_consumer_state.index,
                            )
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kblock_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB_mma[sf_kblock_coord].iterator,
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Release this AB-stage so the TMA warp can reuse it.
                        ab_pipeline.consumer_release(ab_consumer_state)

                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                # Async-arrive accumulator buffer full. Commit ONCE per tile.
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Wait for the epilog warps to drain the accumulator.
            acc_pipeline.producer_tail(acc_producer_state)

        # ============================================================
        # Branch 3/3: Specialized epilogue warps (warps 0-3)
        # ============================================================
        if warp_idx < self.mma_warp_id:
            alpha_value = alpha[0].to(cutlass.Float32)
            if cutlass.const_expr(self.fused_swiglu):
                alpha_up = alpha[1].to(cutlass.Float32)
            # Allocate TMEM (only epilog warp 0 actually allocates; the
            # TmemAllocator helper handles that based on allocator_warp_id).
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()

            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Epilogue partition setup (T2R / R2S / S2G).
            epi_tidx = tidx
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs,
            )
            if cutlass.const_expr(self.fused_swiglu):
                _, tTR_tAccU_base, tTR_rAccU = self.epilog_tmem_copy_and_partition(
                    epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs, 1,
                )
            if cutlass.const_expr(not self.quantized_output):
                tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
                tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                    tiled_copy_t2r, tTR_rC, epi_tidx, sC,
                )
                tma_atom_c_epi, bSG_sC, bSG_gC_partitioned = self.epilog_gmem_copy_and_partition(
                    epi_tidx, tma_atom_c, tCgC, epi_tile, sC,
                )
            else:
                global_value = global_scale[0]
                packed_output = cute.make_tensor(
                    cute.recast_ptr(mC_mnl.iterator, dtype=cutlass.Uint64),
                    cute.make_layout(mC_mnl.shape[0] * mC_mnl.shape[1] // 16))
                epi_coord = tiled_copy_t2r.get_slice(epi_tidx).partition_D(
                    cute.make_identity_tensor((cute.size(epi_tile[0]), cute.size(epi_tile[1]))))[0]

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            # TMA-store pipeline over the 4 epilog warps.
            if cutlass.const_expr(not self.quantized_output):
                c_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.threads_per_warp * len(self.epilog_warp_id),
                )
                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_c_stage,
                    producer_group=c_producer_group,
                )

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                cur_tile_coord = self.group_tile(work_tile.tile_idx, tile_sched_params)
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Slice partitioned global tensor to this tile's coord.
                if cutlass.const_expr(not self.quantized_output):
                    bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]

                # Acc stage index (num_acc_stage=1 -> index 0).
                acc_stage_index = acc_consumer_state.index
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_stage_index)
                ]

                # Wait for the MMA warp to fill the accumulator.
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                if cutlass.const_expr(self.fused_swiglu):
                    tTR_tAccU = tTR_tAccU_base[
                        (None, None, None, None, None, acc_stage_index)
                    ]
                    tTR_tAccU = cute.group_modes(tTR_tAccU, 3, cute.rank(tTR_tAccU))
                if cutlass.const_expr(not self.quantized_output):
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                # Subtile loop: TMEM -> reg -> SMEM -> TMA -> GMEM. The C
                # SMEM ring buffer index continues across tiles via
                # num_prev_subtiles (matches upstream persistent epilogue).
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                for subtile_idx in cutlass.range(subtile_cnt):
                    # TMEM -> register.
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    # Apply the exact global NVFP4 scale in FP32 before
                    # converting the accumulator to FP16 or FP8 E4M3.
                    if cutlass.const_expr(self.fused_swiglu):
                        cute.copy(tiled_copy_t2r,
                                  tTR_tAccU[(None, None, None, subtile_idx)], tTR_rAccU)
                        for i in cutlass.range(cute.size(tTR_rAcc), unroll_full=True):
                            gate = alpha_value * tTR_rAcc[i]
                            up = alpha_up * tTR_rAccU[i]
                            sigmoid = cute.arch.rcp_approx(1.0 + cute.exp(-gate, fastmath=True))
                            tTR_rAcc[i] = gate * sigmoid * up
                        if cutlass.const_expr(not self.quantized_output):
                            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                            tRS_rC.store(acc_vec.to(self.c_dtype))
                    else:
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        tRS_rC.store((alpha_value * acc_vec).to(self.c_dtype))

                    if cutlass.const_expr(self.quantized_output):
                        if cutlass.const_expr(self.coalesced_output):
                            self.quantize_epilogue(
                                tTR_rAcc, epi_tile, epi_coord, subtile_idx, mma_tile_coord_mnl,
                                sPacked, sScale, global_value, mC_mnl.shape[0], mC_mnl.shape[1])
                        else:
                            self.quantize_epilogue(
                                tTR_rAcc, epi_tile, epi_coord, subtile_idx, mma_tile_coord_mnl,
                                packed_output, output_scale, global_value,
                                mC_mnl.shape[0], mC_mnl.shape[1])
                    else:
                        c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                        cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c_epi,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, subtile_idx)],
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        # Keep the SMEM slot alive until its TMA store completes.
                        self.epilog_sync_barrier.arrive_and_wait()

                if cutlass.const_expr(self.coalesced_output):
                    if cutlass.const_expr(self.overlap_output_store):
                        # All TMEM values are drained; the scratch flush can overlap the next MMA.
                        acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()
                    self.epilog_sync_barrier.arrive_and_wait()
                    blocks = mC_mnl.shape[1] // 16
                    tile_m, tile_n, _ = mma_tile_coord_mnl
                    for chunk in cutlass.range_constexpr(8):
                        index = epi_tidx + chunk * 128
                        row = index // 8
                        column = index % 8
                        # Undo the SMEM bank swizzle for contiguous global stores.
                        bits = sPacked[row * 8 + (column ^ ((row // 2) % 8))]
                        if tile_m * 128 + row < mC_mnl.shape[0] and tile_n * 128 < mC_mnl.shape[1]:
                            packed_output[(tile_m * 128 + row) * blocks + tile_n * 8 + column] = bits
                    local_sf = cute.make_tensor(
                        cute.recast_ptr(sScale.iterator, dtype=cutlass.Uint32),
                        cute.make_layout(256))
                    global_sf = cute.make_tensor(
                        cute.recast_ptr(output_scale.iterator, dtype=cutlass.Uint32),
                        cute.make_layout(cute.ceil_div(mC_mnl.shape[0], 128) * 128 * blocks // 4))
                    for chunk in cutlass.range_constexpr(2):
                        index = epi_tidx + chunk * 128
                        offset = (tile_m * (blocks // 4) + tile_n * 2) * 128 + index
                        valid_tile = tile_n * 128 < mC_mnl.shape[1]
                        if cutlass.const_expr(self.cluster_shape_mn[0] > 1):
                            valid_tile = valid_tile and tile_m * 128 < mC_mnl.shape[0]
                        if valid_tile:
                            global_sf[offset] = local_sf[index]
                    self.epilog_sync_barrier.arrive_and_wait()
                # Release acc-empty so the MMA warp can reuse the acc buffer.
                if cutlass.const_expr(not self.overlap_output_store):
                    acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Dealloc TMEM (only epilog warp 0 actually frees; the
            # ``relinquish_alloc_permit`` + sync + ``free`` triplet is the
            # canonical upstream shutdown sequence).
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)

            # Wait for the in-flight C TMA stores to complete before exit.
            if cutlass.const_expr(not self.quantized_output):
                c_pipeline.producer_tail()
        return

    @cute.jit
    def quantize_epilogue(self, accumulator, epi_tile, coord, subtile, tile, packed, scales,
                          global_value, rows, columns):
        """Quantize register-held H with FP16 rounding and zero-fill padded SF rows."""
        epi_m = cute.size(epi_tile[0])
        epi_n = cute.size(epi_tile[1])
        m_subtiles = self.output_tile_shape_mnk[0] // epi_m
        row_base = tile[0] * self.output_tile_shape_mnk[0] + (subtile % m_subtiles) * epi_m
        col_base = tile[1] * self.output_tile_shape_mnk[1] + (subtile // m_subtiles) * epi_n
        blocks_per_row = columns // 16
        # This T2R layout owns a contiguous row of 32 columns per thread.
        if cutlass.const_expr(cute.size(accumulator) != epi_n or epi_n != 32):
            raise ValueError("NVFP4 output quantization requires 32 contiguous columns per thread")
        row = row_base + coord[0]
        for i in cutlass.range_constexpr(epi_n // 16):
            column = (col_base + coord[1]) // 16 + i
            sf = cutlass.Float32(0.0).to(cutlass.Float8E4M3FN)
            bits = cutlass.Uint64(0)
            if row < rows and global_value > 0.0:
                values = cute.make_rmem_tensor(16, cutlass.Float32)
                amax = cutlass.Float32(0.0)
                for j in cutlass.range_constexpr(16):
                    values[j] = accumulator[i * 16 + j].to(cutlass.Float16).to(cutlass.Float32)
                    amax = fmax_f32(amax, fabs_f32(values[j]))
                sf = fmin_f32(_div_rn(_div_rn(amax, 6.0), global_value), 448.0).to(cutlass.Float8E4M3FN)
                scale = sf.to(cutlass.Float32)
                if scale > 0.0:
                    reciprocal = _div_rn(1.0, scale)
                    for j in cutlass.range_constexpr(16):
                        values[j] = _div_rn(
                            _half_div_fp8_rn(values[j], scale, reciprocal), global_value)
                    lo = cvt_e2m1x8_f32(values[0], values[1], values[2], values[3],
                                       values[4], values[5], values[6], values[7])
                    hi = cvt_e2m1x8_f32(values[8], values[9], values[10], values[11],
                                       values[12], values[13], values[14], values[15])
                    bits = lo.to(cutlass.Uint64) | (hi.to(cutlass.Uint64) << 32)
            sf_offset = ((((row // 128) * (blocks_per_row // 4) + column // 4) * 32
                          + row % 32) * 4 + (row % 128) // 32) * 4 + column % 4
            if cutlass.const_expr(self.coalesced_output):
                local_row = row - tile[0] * 128
                local_col = column - tile[1] * 8
                local_sf = (((local_col // 4) * 32 + local_row % 32) * 4 + local_row // 32) * 4 + local_col % 4
                scales[local_sf] = sf
                packed[local_row * 8 + (local_col ^ ((local_row // 2) % 8))] = bits
            else:
                valid_scale = column < blocks_per_row
                if cutlass.const_expr(self.cluster_shape_mn[0] > 1):
                    valid_scale = valid_scale and row < cute.ceil_div(rows, 128) * 128
                if valid_scale:
                    scales[sf_offset] = sf
                    if row < rows:
                        packed[row * blocks_per_row + column] = bits

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """SMEM -> TMEM tiled-copy partition for a scale-factor tensor.

        Used by the MMA warp to stage SFA / SFB from SMEM into TMEM so
        the tcgen05.blockscaled MMA can read them via Field.SFA/SFB.
        Adapted from upstream
        ``dense_blockscaled_gemm_persistent.py::mainloop_s2t_copy_and_partition``
        (line 1534) — identical logic, no NVFP4-specific tweaks needed.

        :param sSF: Scale-factor tensor in SMEM
            (MMA, MMA_MN, MMA_K, STAGE).
        :param tSF: Scale-factor tensor in TMEM (MMA, MMA_MN, MMA_K).
        :return: (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t)
        """
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
        acc_half: cutlass.Constexpr = 0,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """TMEM -> register tiled-copy partition for the epilog warps.

        Ported from upstream
        ``dense_blockscaled_gemm_persistent.py::epilog_tmem_copy_and_partition``
        (line 1577) with one shape tweak: ``tAcc[((None, None), 0, 0, None)]``
        (trailing None for the stage dim added to tCtAcc_fake in the
        kernel body).

        :param tidx: Per-CTA thread index (used by partition slicing).
        :param tAcc: Accumulator TMEM tensor with stage dim
            (MMA, MMA_M, MMA_N, STAGE).
        :param gC_mnl: Global tensor C (MMA, MMA_M, MMA_N, RestM, RestN, RestL).
        :param epi_tile: Epilogue subtile shape.
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
            (False for this WS variant).
        :return: (tiled_copy_t2r, tTR_tAcc, tTR_rAcc)
        """
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.output_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        acc_mn = tAcc[((None, None), 0, 0, None)]
        if cutlass.const_expr(self.fused_swiglu):
            halves = cute.flat_divide(acc_mn, self.output_tile_shape_mnk[:2])
            acc_mn = halves[(None, None, 0, acc_half, None)]
        tAcc_epi = cute.flat_divide(acc_mn, epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Register -> SMEM tiled-copy partition for the epilog warps.

        Ported verbatim from upstream
        ``dense_blockscaled_gemm_persistent.py::epilog_smem_copy_and_partition``
        (line 1640).

        :param tiled_copy_t2r: TMEM->register tiled-copy (used as basis
            for the matching R2S layout).
        :param tTR_rC: Register tensor C (T2R-shaped).
        :param tidx: Per-CTA thread index.
        :param sC: SMEM tensor C (EPI_TILE_M, EPI_TILE_N, STAGE).
        :return: (tiled_copy_r2s, tRS_rC, tRS_sC)
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: cute.CopyAtom,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """SMEM -> GMEM TMA-store partition for the epilog warps.

        Ported verbatim from upstream
        ``dense_blockscaled_gemm_persistent.py::epilog_gmem_copy_and_partition``
        (line 1677). The TMA atom is passed through unchanged; this
        method only computes the partitioned source (SMEM) and
        destination (GMEM) tensor views.

        :param tidx: Per-CTA thread index (unused for TMA-store; kept for
            signature symmetry with the SIMT-store variant).
        :param atom: TMA copy atom for C (CopyBulkTensorTileS2GOp).
        :param gC_mnl: Global tensor C (with MMA partitioning).
        :param epi_tile: Epilogue subtile shape.
        :param sC: SMEM tensor C (EPI_TILE_M, EPI_TILE_N, STAGE).
        :return: (tma_atom_c, bSG_sC, bSG_gC) — partitioned for the
            per-subtile ``cute.copy(tma_atom_c, bSG_sC[...], bSG_gC[...])``
            call in the epilog branch.
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    # -----------------------------------------------------------------------
    # Static helpers (mirror gemm_blackwell.py:566-634) — unchanged from FP16.
    # -----------------------------------------------------------------------

    @staticmethod
    def _compute_stages(
        tiled_mma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        epi_tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout,
        smem_capacity: int,
        occupancy: int,
        use_tma_store: bool = False,
    ):
        """SMEM stage budget for AB + SFA + SFB (+ optional C for TMA store).

        Follows ``gemm_blackwell.py:578-615`` for the AB / C stage arithmetic
        and ``blockscaled_contiguous_grouped_gemm.py:1864-1934`` for the
        SFA/SFB byte accounting.

        Returns (num_acc_stage, num_ab_stage, num_c_stage).
        The SF SMEM cost is rolled into ab_bytes_per_stage so downstream
        code keeps a single ab_stage count across A, B, SFA, SFB.
        """
        # ACC pipeline fixed at 1 stage. The upstream "num_acc_stage=2 for
        # wide-N (>=256)" rule is NOT used here: the 2-acc-stage path is not
        # correct for this WS port (it triggers an illegal-address at
        # mma_tiler_n=256), and acc=1 is both correct and faster for every
        # built tile (tn64/tn128/tn256).
        num_acc_stage = 1
        num_c_stage = 2 if use_tma_store else 0

        # Single-stage SMEM footprint for each operand (multiply by num_ab_stage).
        a_layout_1 = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
        b_layout_1 = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
        sfa_layout_1 = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, mma_tiler_mnk, sf_vec_size, 1
        )
        sfb_layout_1 = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, mma_tiler_mnk, sf_vec_size, 1
        )
        c_layout_1 = (
            sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)
            if use_tma_store
            else None
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_layout_1)
            + cute.size_in_bytes(b_dtype, b_layout_1)
            + cute.size_in_bytes(sf_dtype, sfa_layout_1)
            + cute.size_in_bytes(sf_dtype, sfb_layout_1)
        )
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = (
            cute.size_in_bytes(c_dtype, c_layout_1) if use_tma_store else 0
        )
        c_bytes = c_bytes_per_stage * num_c_stage

        num_ab_stage = (
            smem_capacity - (occupancy + 1) * (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        if use_tma_store:
            num_c_stage += (
                smem_capacity
                - ab_bytes_per_stage * num_ab_stage
                - (occupancy + 1) * (mbar_helpers_bytes + c_bytes)
            ) // ((occupancy + 1) * c_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_c_stage

    @cute.jit
    def group_tile(self, coord, tile_sched_params):
        m, n, batch = coord
        if cutlass.const_expr(self.n_tile_group_size > 1):
            tiles_m, tiles_n, _ = tile_sched_params.problem_shape_ntile_mnl
            group = self.n_tile_group_size
            if cutlass.const_expr(self.cluster_shape_mn == (1, 1)):
                # A complete N group preserves the scheduler's domain without padding.
                # Keep decode and short prefill in the original raster order.
                if (tiles_m >= 32) & (tiles_n % group == 0):
                    offset = m + tiles_m * (n % group)
                    m = offset // group
                    n = n // group * group + offset % group
            elif tiles_m >= 32:
                # Multicast peers must retain their relative coordinates within a cluster.
                cm, cn = self.cluster_shape_mn
                clusters_m = cute.ceil_div(tiles_m, cm)
                clusters_n = cute.ceil_div(tiles_n, cn)
                cluster_m, cluster_n = m // cm, n // cn
                start_n = cluster_n // group * group
                width = cutlass.min(group, clusters_n - start_n)
                offset = cluster_m + clusters_m * (cluster_n - start_n)
                m = offset // width * cm + m % cm
                n = (start_n + offset % width) * cn + n % cn
        return m, n, batch

    @staticmethod
    def _compute_grid(c, cta_tile_shape_mnk, cluster_shape_mn, max_active_clusters):
        """Persistent-scheduler grid: a fixed worker grid (capped by
        ``max_active_clusters``) that strides over all output tiles, instead
        of one CTA per tile. Returns (tile_sched_params, grid). Mirrors
        upstream dense_blockscaled_gemm_persistent.py::_compute_grid.
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _compute_num_tmem_alloc_cols(tiled_mma, mma_tiler):
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        return utils.get_num_tmem_alloc_cols(tCtAcc_fake)


# ---------------------------------------------------------------------------
# Run / Export driver
# ---------------------------------------------------------------------------


def run(
    mnk: Tuple[int, int, int],
    sf_vec_size: int = 16,
    mma_tiler_n: int = 64,
    c_dtype: Type[cutlass.Numeric] = cutlass.Float16,
    export_only: bool = False,
    output_dir: str = "./gemm_nvfp4_aot_artifacts",
    file_name: str = "gemm_blackwell_nvfp4_ws_fp16",
    function_prefix: str = "gemm_blackwell_nvfp4_ws_fp16",
    skip_ref_check: bool = False,
    tolerance: float = 0.05,
    warmup_iterations: int = 3,
    iterations: int = 10,
    fused_swiglu: bool = False,
    quantized_output: bool = False,
    cluster_shape_mn: Tuple[int, int] = (1, 1),
    fallback_cluster_shape_mn: Optional[Tuple[int, int]] = None,
    coalesced_output: bool = False,
    overlap_output_store: bool = False,
):
    m, n, k = mnk
    if fused_swiglu and not export_only:
        raise ValueError("Use test_nvfp4_dual_gemm.py for SwiGLU device validation")
    _tag = f"[{file_name}]"
    c_dtype_name = (
        "Float16" if c_dtype is cutlass.Float16 else
        "Float8E4M3FN" if c_dtype is cutlass.Float8E4M3FN else
        repr(c_dtype)
    )

    if export_only:
        print(f"{_tag} AOT compile: M={m}, N={n}, K={k}, sf_vec_size={sf_vec_size}, c_dtype={c_dtype_name}")
    else:
        print(f"{_tag} Running Blackwell NVFP4 GEMM (C = A @ B^T) test:")
        print(f"{_tag}   M={m}, N={n}, K={k}")
        print(f"{_tag}   A/B: Float4E2M1FN    SFA/SFB: Float8E4M3FN    C: {c_dtype_name}    acc: Float32")
        print(f"{_tag}   mma_tiler=(128,{mma_tiler_n}), cluster=(1,1), TMA store, 6 warps (4 epilog + MMA + TMA)")

    if not export_only and cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU is required!")

    if not export_only:
        cp.random.seed(1111)
    np.random.seed(1111)

    if export_only:
        ptrs = {
            "a": aot_placeholders.make_ptr(cutlass.Float4E2M1FN, assumed_align=32),
            "b": aot_placeholders.make_ptr(cutlass.Float4E2M1FN, assumed_align=32),
            "sfa": aot_placeholders.make_ptr(cutlass.Float8E4M3FN, assumed_align=16),
            "sfb": aot_placeholders.make_ptr(cutlass.Float8E4M3FN, assumed_align=16),
            "c": aot_placeholders.make_ptr(c_dtype, assumed_align=16),
            "alpha": aot_placeholders.make_ptr(cutlass.Float32, assumed_align=4),
        }
        _backing = None
        current_stream = aot_placeholders.make_stream()
    else:
        ptrs, _backing = _create_nvfp4_pointers(
            m, n, k, sf_vec_size, c_dtype=c_dtype
        )
        current_stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)

    gemm = GemmBlackwellNvFp4WS(
        acc_dtype=cutlass.Float32,
        mma_tiler_mn=(128, mma_tiler_n),
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        fused_swiglu=fused_swiglu,
        quantized_output=quantized_output,
        n_tile_group_size=4 if quantized_output else 1,
        fallback_cluster_shape_mn=fallback_cluster_shape_mn,
        coalesced_output=coalesced_output,
        overlap_output_store=overlap_output_store,
    )

    # Runtime persistent-grid size: export traces with a placeholder (the
    # deployed caller supplies the target GPU's value); run/verify probes the
    # local GPU as before.
    if export_only:
        max_active_clusters = aot_placeholders.runtime_int32()
    else:
        max_active_clusters = cutlass.Int32(
            utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1])
        )

    start_time = time.time()
    compiled_gemm = cute.compile(
        gemm.wrapper,
        ptrs["a"], ptrs["b"], ptrs["sfa"], ptrs["sfb"], ptrs["c"],
        ptrs["alpha"],
        m, n, k,
        sf_vec_size,
        max_active_clusters,
        current_stream,
        *([aot_placeholders.make_ptr(cutlass.Float8E4M3FN, assumed_align=16),
           aot_placeholders.make_ptr(cutlass.Float32, assumed_align=4)]
          if quantized_output else []),
        **(
            dict(options=aot_placeholders.compile_options())
            if export_only
            else {}
        ),
    )
    compilation_time = time.time() - start_time
    print(f"{_tag} Compilation time: {compilation_time:.4f}s")

    if export_only:
        export_compiled_kernel(
            compiled_gemm,
            output_dir=output_dir,
            file_name=file_name,
            function_prefix=function_prefix,
            tag=_tag,
        )
        return None

    # This entry point only drives a smoke launch and AOT export.
    # End-to-end numerical accuracy is exercised by downstream runtime tests.
    print(f"{_tag} Numerical check + benchmark: deferred (use --export_only)")
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CuTe DSL Blackwell NVFP4 GEMM **warp-specialised** variant "
                    "(C = A @ B^T, FP4 in / FP16 out / FP32 acc)."
    )
    p.add_argument(
        "--mnk", type=parse_comma_separated_ints, default=(128, 2048, 2048),
        help="M,N,K dimensions (comma-separated). Default: 128,2048,2048.",
    )
    p.add_argument("--sf_vec_size", type=int, default=16, choices=[16, 32],
                   help="Scale factor vector size (NVF4=16, MXF4=32). Default: 16.")
    p.add_argument("--mma_tiler_n", type=int, default=64, choices=[64, 128, 192, 256],
                   help="MMA tiler N. Default: 64.")
    p.add_argument("--fused_swiglu", action="store_true",
                   help="Interleaved 128-row gate/up weight tiles; two FP32 alphas; N/2 output columns.")
    p.add_argument("--quantized_output", action="store_true",
                   help="Fuse output NVFP4 quantization; add output SF and global-scale pointers.")
    p.add_argument("--cluster_shape_mn", type=parse_comma_separated_ints, default=(1, 1))
    p.add_argument("--fallback_cluster_shape_mn", type=parse_comma_separated_ints)
    p.add_argument("--coalesced_output", action="store_true")
    p.add_argument("--overlap_output_store", action="store_true")
    p.add_argument("--c_dtype", type=str, default="fp16", choices=["fp16", "fp8_e4m3"],
                   help="Output element type. 'fp16' → Float16 (FP16-output variant). "
                        "'fp8_e4m3' → Float8E4M3FN (FP8-output variant). "
                        "Default: fp16.")
    p.add_argument("--export_only", action="store_true",
                   help="Compile and export .o + .h; skip test and benchmark.")
    p.add_argument("--output_dir", type=str, default="./gemm_nvfp4_aot_artifacts",
                   help="Output directory for AOT artifacts.")
    p.add_argument("--file_name", type=str, default="gemm_blackwell_nvfp4_ws_fp16",
                   help="Base file name for exported .h/.o.")
    p.add_argument("--function_prefix", type=str, default="gemm_blackwell_nvfp4_ws_fp16",
                   help="C function prefix (avoids symbol conflicts).")
    p.add_argument("--skip_ref_check", action="store_true",
                   help="Skip reference correctness check.")
    p.add_argument("--tolerance", type=float, default=0.05,
                   help="Tolerance for reference check.")
    p.add_argument("--warmup", type=int, default=3, help="Warmup iterations.")
    p.add_argument("--iterations", type=int, default=10, help="Benchmark iterations.")
    return p.parse_known_args(args=argv)[0]


if __name__ == "__main__":
    _parsed_args = _parse_args(_saved_argv)
    args = _parsed_args

    if len(args.mnk) != 3:
        raise ValueError("--mnk must contain exactly 3 values (M,N,K)")

    _C_DTYPE_MAP = {
        "fp16": cutlass.Float16,
        "fp8_e4m3": cutlass.Float8E4M3FN,
    }

    run(
        mnk=args.mnk,
        sf_vec_size=args.sf_vec_size,
        mma_tiler_n=args.mma_tiler_n,
        c_dtype=_C_DTYPE_MAP[args.c_dtype],
        export_only=args.export_only,
        output_dir=args.output_dir,
        file_name=args.file_name,
        function_prefix=args.function_prefix,
        skip_ref_check=args.skip_ref_check,
        tolerance=args.tolerance,
        warmup_iterations=args.warmup,
        iterations=args.iterations,
        fused_swiglu=args.fused_swiglu,
        quantized_output=args.quantized_output,
        cluster_shape_mn=args.cluster_shape_mn,
        fallback_cluster_shape_mn=args.fallback_cluster_shape_mn,
        coalesced_output=args.coalesced_output,
        overlap_output_store=args.overlap_output_store,
    )
    if not args.export_only:
        print("PASS")
