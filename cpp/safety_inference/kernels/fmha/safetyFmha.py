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

# Origin: Adapted from xlite-dev/ffpa-attn (Apache-2.0):
#   https://github.com/xlite-dev/ffpa-attn
#   csrc/cuffpa/prefill.cuh + csrc/cuffpa/ffpa_attn_fwd.cuh
# Copyright (c) DefTruth & the xlite-dev community (Apache-2.0)
from types import SimpleNamespace
from typing import Callable, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, warp


@cute.jit
def skip_softmax_threshold_log2_from_table(skip_softmax_scale_table, seqlen_k):
    """Per-sequence log2(lambda), lambda = S / seqlen_k, with S the floor-bin
    lookup of seqlen_k in the flattened [bin_0, S_0, bin_1, S_1, ...] table
    (bins ascending). Below the first bin, or when S >= seqlen_k (lambda >= 1
    would pass every tile), returns -2^100 so no tile can win the vote: the
    sequence runs dense inside the skip kernel. Once per CTA, off the KV loop.
    Branch-free on purpose: the scf.if form costs 6% in the all-skip bound on
    GB10 (ptxas -O2 register allocation), the select form none."""
    num_rows = cute.size(skip_softmax_scale_table) // 2
    scale = cutlass.Int32(0)
    for row in cutlass.range(num_rows, unroll=1):
        if skip_softmax_scale_table[2 * row] <= seqlen_k:
            scale = skip_softmax_scale_table[2 * row + 1]
    valid = cutlass.Float32(cutlass.Int32((scale > 0) and (scale < seqlen_k)))
    threshold_log2 = cute.math.log2(
        cute.arch.fmax(cutlass.Float32(scale), cutlass.Float32(1.0)),
        fastmath=True) - cute.math.log2(cutlass.Float32(seqlen_k),
                                        fastmath=True)
    return threshold_log2 * valid - cutlass.Float32(2.0**100) * (1.0 - valid)


# Selected methods copied from kernelSrcs/fmha_v2_cutedsl/fmha.py
# at 52dfc7e00295d8c3d6875aa4acd1962c6003dbb7. Safety adds a device validity guard and
# launch bindings with independent Q/K lengths and explicit contiguous KV strides.
class FMHAV2Ampere:

    def __init__(
        self,
        head_dim: int,
        m_block_size: int = 128,
        n_block_size: int = 128,
        num_threads: int = 128,
        is_causal: bool = False,
        use_sliding_window: bool = False,
        packed_varlen: bool = False,
        skip_rescale: bool = False,
        hybrid_exp2: bool = False,
        skip_softmax: bool = False,
    ):
        """Initialize the FMHA-v2 Ampere-floor kernel.

        All contiguous dimensions must be at least 16 bytes aligned, which
        means ``head_dim`` should be a multiple of 8.

        GQA is a runtime axis: the kernel takes ``num_kv_heads`` as a launch
        argument (see ``__call__``), so a single compiled kernel serves MHA
        and any GQA group size without recompiling.

        :param head_dim: head dimension
        :param m_block_size: ``Br`` — query tile rows per CTA
        :param n_block_size: ``Bc`` — key/value tile rows per CTA
        :param num_threads: CTA thread count
        :param is_causal: enable causal mask
        :param skip_rescale: clamp per-row rescale factor to 1.0 when within
            ``2^-8`` of unity.  Trades one FFMA for a stabilized rescale, in
            line with the upstream xlite-dev/ffpa-attn tuning.
        :param hybrid_exp2: scaffold for FA4-style 75 % MUFU + 25 % polynomial
            exp2 in softmax.  Not validated in this variant; defer.
        :param skip_softmax: compile the skip-softmax (BLASST) path in: a
            warp skips softmax + P@V for a KV tile when every one of its rows
            has ``(tile_max - running_max) * scale < ln(lambda)``.  Adds the
            runtime ``skip_softmax_scale_table`` to the native-paged ABI.
        """
        self._head_dim = head_dim
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        # MMA consumes K in 16-element steps.  The D72 ViT specialization uses
        # D80 instead of wasting a full extra 32-element slice at D96.  Keep
        # the established 32-element padding for every other variant.
        self._head_dim_padded = 80 if head_dim == 72 else (head_dim +
                                                           31) // 32 * 32
        # D512 uses two CTAs per logical Q tile. Each CTA repeats QK/softmax
        # but owns only one 256-column V/O slice, keeping the FP32 output
        # accumulator small enough to stay in registers.
        self._output_block_size = 256 if head_dim == 512 else self._head_dim_padded
        self._num_output_blocks = 2 if head_dim == 512 else 1
        self._num_threads = num_threads
        self._is_causal = is_causal
        self._use_sliding_window = use_sliding_window
        self._packed_varlen = packed_varlen
        # D72's four short non-causal KV steps are faster without the
        # near-unity rescale clamp; the unclamped online-softmax path also
        # matches the long/ragged reference exactly.
        self._skip_rescale = skip_rescale and head_dim != 72
        self._hybrid_exp2 = hybrid_exp2
        self._skip_softmax = skip_softmax
        self._async_load_cache_mode = (cpasync.LoadCacheMode.ALWAYS if head_dim
                                       == 72 else cpasync.LoadCacheMode.GLOBAL)

        self.cta_sync_barrier = pipeline.NamedBarrier(barrier_id=1,
                                                      num_threads=num_threads)

    @cute.jit
    def __call__(
        self,
        validity: cute.Tensor,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mCuSeqLenQ: cute.Tensor,
        mCuSeqLenK: cute.Tensor,
        mBlockBegin: Optional[cute.Tensor],
        mBlockEnd: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        num_kv_heads: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Safety override of Standard's dense launcher.

        Adds the device validity guard. K/V use BSND views over the
        head-major contiguous cache; caller-supplied strides avoid repacking.
        Cumulative Q/K lengths preserve padded-chunk causal alignment.
        All tiling, scale conversion and launch geometry remain Standard.
        """
        if cutlass.const_expr(not (mQ.element_type == mK.element_type ==
                                   mV.element_type == mO.element_type)):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(not (mQ.element_type == cutlass.Float16
                                   or mQ.element_type == cutlass.BFloat16)):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if cutlass.const_expr((mBlockBegin is None) != (mBlockEnd is None)):
            raise TypeError(
                "mBlockBegin and mBlockEnd must both be provided or both be None"
            )
        if cutlass.const_expr(mBlockBegin is not None and not self._is_causal):
            raise TypeError(
                "The vision-block overlay is only defined for the causal variant"
            )
        self._dtype: Type[cutlass.Numeric] = mQ.element_type
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout: Q/K/V
        # ///////////////////////////////////////////////////////////////////////////////
        smem_k_block_size = (64 if self._head_dim_padded %
                             64 == 0 else 32 if self._head_dim_padded %
                             32 == 0 else 16)
        swizzle_bits = 3 if smem_k_block_size == 64 else 2 if smem_k_block_size == 32 else 1
        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size),
                             stride=(smem_k_block_size, 1)),
        )
        sQ_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self._m_block_size, self._head_dim_padded),
            (0, 1),
        )

        sKV_layout_atom = sQ_layout_atom
        sKV_layout = cute.tile_to_shape(
            sKV_layout_atom,
            (self._n_block_size, self._head_dim_padded),
            (0, 1),
        )
        sV_layout = sKV_layout
        sO_layout = sQ_layout
        if cutlass.const_expr(self._num_output_blocks == 2):
            sV_layout = cute.tile_to_shape(
                sKV_layout_atom,
                (self._n_block_size, self._output_block_size),
                (0, 1),
            )
            sO_layout = cute.tile_to_shape(
                sQ_layout_atom,
                (self._m_block_size, self._output_block_size),
                (0, 1),
            )

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[cute.struct.MemRange[self._dtype,
                                                       cute.cosize(sQ_layout)],
                                  1024]
            sK: cute.struct.Align[cute.struct.MemRange[
                self._dtype, cute.cosize(sKV_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[self._dtype,
                                                       cute.cosize(sV_layout)],
                                  1024]

        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self._dtype.width
        # atom_async_copy: async copy atom for QKV load
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=self._async_load_cache_mode),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # atom_universal_copy: universal copy atom for O store
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # tQKV_layout: thread layout for QKV load
        tQKV_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        tQKV_layout = cute.make_layout(
            (self._num_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
            stride=(tQKV_shape_dim_1, 1),
        )
        # tO_layout: thread layout for O store
        tO_layout = tQKV_layout

        # Value layouts for copies
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        vO_layout = vQKV_layout

        # gmem_tiled_copy_QKV: tiled copy for QKV load
        gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(atom_async_copy,
                                                      tQKV_layout, vQKV_layout)
        # gmem_tiled_copy_O: tiled copy for O store
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy,
                                                    tO_layout, vO_layout)

        # ///////////////////////////////////////////////////////////////////////////////
        # Tiled mma
        # ///////////////////////////////////////////////////////////////////////////////
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
        )

        # Keep Q heads that share a GQA K/V head adjacent in launch order so
        # their K/V working set can be reused from cache.  This also matches
        # the legacy FMHA-v2 grid ordering.
        # grid_dim: (m_block, num_head, batch_size)
        if cutlass.const_expr(self._num_output_blocks == 2):
            grid_dim = (
                cute.ceil_div(mQ.shape[1], self._m_block_size) * 2,
                cute.size(mQ.shape[2]),
                cute.size(mQ.shape[0]),
            )
        else:
            grid_dim = (
                cute.ceil_div(mQ.shape[1], self._m_block_size),
                cute.size(mQ.shape[2]),
                cute.size(mQ.shape[0]),
            )
        LOG2_E = 1.4426950408889634074
        softmax_scale_log2 = softmax_scale * LOG2_E
        self.kernel(
            validity,
            mQ,
            mK,
            mV,
            None,
            None,
            None,
            mO,
            mCuSeqLenQ,
            mCuSeqLenK,
            mBlockBegin,
            mBlockEnd,
            softmax_scale_log2,
            num_kv_heads,
            None,
            None,
            sQ_layout,
            sKV_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_QKV,
            gmem_tiled_copy_O,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self._num_threads, 1, 1],
            stream=stream,
        )

    @cute.jit
    def __call_vit__(
        self,
        validity: cute.Tensor,
        q_tensor: cute.Tensor,
        k_tensor: cute.Tensor,
        v_tensor: cute.Tensor,
        o_tensor: cute.Tensor,
        cu_seqlens: cute.Tensor,
        max_seqlen: cutlass.Int32,
        scale_softmax_log2: cutlass.Float32,
        scale_softmax: cutlass.Float32,
        scale_output: cutlass.Float32,
        sm_count: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Launch packed-varlen bidirectional ViT FMHA.

        Safety override: adds the leading device ``validity`` tensor and
        forwards it to the guarded kernel. Standard sequence boundaries,
        scale handling, and launch geometry are unchanged.

        Q/K/V/O use compact ``(total_S, H, D)`` storage.  The CTA grid is
        ``(ceil(max_seqlen / Br), B, Hq)`` and each batch pointer is shifted
        by ``cu_seqlens[b]`` inside the kernel, so sequences never attend
        across packed boundaries. Apart from ``validity``, the argument order
        mirrors the optimized Blackwell ViT ABI.
        """
        if cutlass.const_expr(not (
                q_tensor.element_type == k_tensor.element_type ==
                v_tensor.element_type == o_tensor.element_type)):
            raise TypeError(
                "Packed ViT Q, K, V, and O must have the same data type")
        if cutlass.const_expr(q_tensor.element_type != cutlass.Float16):
            raise TypeError(
                "FMHA-v2 packed ViT FMHA currently supports Float16 only")

        self._dtype = q_tensor.element_type
        smem_k_block_size = (64 if self._head_dim_padded %
                             64 == 0 else 32 if self._head_dim_padded %
                             32 == 0 else 16)
        swizzle_bits = 3 if smem_k_block_size == 64 else 2 if smem_k_block_size == 32 else 1
        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size),
                             stride=(smem_k_block_size, 1)),
        )
        sQ_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self._m_block_size, self._head_dim_padded),
            (0, 1),
        )
        sKV_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self._n_block_size, self._head_dim_padded),
            (0, 1),
        )
        sV_layout = sKV_layout
        sO_layout = sQ_layout
        if cutlass.const_expr(self._num_output_blocks == 2):
            sV_layout = cute.tile_to_shape(
                sQ_layout_atom,
                (self._n_block_size, self._output_block_size),
                (0, 1),
            )
            sO_layout = cute.tile_to_shape(
                sQ_layout_atom,
                (self._m_block_size, self._output_block_size),
                (0, 1),
            )

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[cute.struct.MemRange[self._dtype,
                                                       cute.cosize(sQ_layout)],
                                  1024]
            sK: cute.struct.Align[cute.struct.MemRange[
                self._dtype, cute.cosize(sKV_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[self._dtype,
                                                       cute.cosize(sV_layout)],
                                  1024]

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
        tQKV_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        tQKV_layout = cute.make_layout(
            (self._num_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
            stride=(tQKV_shape_dim_1, 1),
        )
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(atom_async_copy,
                                                      tQKV_layout, vQKV_layout)
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy,
                                                    tQKV_layout, vQKV_layout)
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
        )

        batch_size = cu_seqlens.shape[0] - 1
        if cutlass.const_expr(self._num_output_blocks == 2):
            grid_dim = (
                cute.ceil_div(max_seqlen, self._m_block_size) * 2,
                batch_size,
                q_tensor.shape[1],
            )
        else:
            grid_dim = (
                cute.ceil_div(max_seqlen, self._m_block_size),
                batch_size,
                q_tensor.shape[1],
            )
        # The optimized ABI carries both natural-log and log2 scales.  This kernel's
        # exp2 implementation consumes the pre-folded log2 value directly.
        _ = scale_softmax
        _ = scale_output
        _ = sm_count
        self.kernel(
            validity,
            q_tensor,
            k_tensor,
            v_tensor,
            None,
            None,
            None,
            o_tensor,
            cu_seqlens,
            cu_seqlens,
            None,
            None,
            scale_softmax_log2,
            k_tensor.shape[1],
            None,
            max_seqlen,
            sQ_layout,
            sKV_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_QKV,
            gmem_tiled_copy_O,
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
        validity: cute.Tensor,
        mQ: cute.Tensor,
        mK: Optional[cute.Tensor],
        mV: Optional[cute.Tensor],
        mKV: Optional[cute.Tensor],
        mPagedKV: Optional[cute.Tensor],
        mPageList: Optional[cute.Tensor],
        mO: cute.Tensor,
        mCuSeqLenQ: Optional[cute.Tensor],
        mCuSeqLenK: cute.Tensor,
        mBlockBegin: Optional[cute.Tensor],
        mBlockEnd: Optional[cute.Tensor],
        softmax_scale_log2: cutlass.Float32,
        num_kv_heads: cutlass.Int32,
        window_size_left: Optional[cutlass.Int32],
        max_seqlen: Optional[cutlass.Int32],
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_QKV: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        """FA2 kernel body: prologue cp.async Q/K, KV outer loop in reverse,
        ``compute_one_n_block`` for BMM1 → softmax_rescale → BMM2.  Epilogue
        rmem → smem (aliased over sQ) → gmem.

        Safety override: reads ``validity[0]`` before the Standard body and
        skips all metadata/Q/K/V accesses and output writes when it is zero.
        The caller must prepare the flag on the same stream and keep it
        unchanged during this launch so the guard is uniform across CTA
        threads. Metadata errors are reported by the validation kernel;
        the guarded FMHA leaves output untouched. MMA and softmax math are
        unchanged.
        """
        if validity[0] != 0:
            tidx, _, _ = cute.arch.thread_idx()
            m_block, grid_y, grid_z = cute.arch.block_idx()
            output_block = 0
            if cutlass.const_expr(self._num_output_blocks == 2):
                output_block = m_block % 2
                m_block = m_block // 2
            output_column_offset = output_block * self._output_block_size
            if cutlass.const_expr(self._packed_varlen):
                batch_size = grid_y
                num_head = grid_z
            else:
                num_head = grid_y
                batch_size = grid_z
            # GQA: group_size = H_q / H_kv (runtime); each K/V head is shared
            # across `kv_group_size` consecutive Q heads, so the K/V head index for
            # this CTA's Q head is `num_head // kv_group_size`.  Matches the
            # `q_head * H_kv / H_q` convention of the FP32 BSHD reference.
            if cutlass.const_expr(self._packed_varlen):
                num_q_heads = mQ.shape[1]
            else:
                num_q_heads = mQ.shape[2]
            kv_group_size = num_q_heads // num_kv_heads
            num_head_kv = num_head // kv_group_size

            # Per-batch logical lengths (varlen): mQ.shape[1]/mK.shape[1] stay the
            # physical padded extents (used for OOB predicates); the cu_seqlen
            # tensors carry the valid lengths.  offset_b is the bottom-right causal
            # offset (0 for right-padded own-KV prefill; the KV-cache prefix length
            # for chunked prefill).
            seqlen_q_b = mQ.shape[1]
            if cutlass.const_expr(mCuSeqLenQ is not None):
                seqlen_q_b = mCuSeqLenQ[batch_size +
                                        1] - mCuSeqLenQ[batch_size]
            seqlen_k_b = mCuSeqLenK[batch_size + 1] - mCuSeqLenK[batch_size]
            offset_b = seqlen_k_b - seqlen_q_b

            n_block_max = cute.ceil_div(seqlen_k_b, self._n_block_size)
            if self._is_causal:
                n_block_max = min(
                    cute.ceil_div(
                        (m_block + 1) * self._m_block_size + offset_b,
                        self._n_block_size,
                    ),
                    n_block_max,
                )
            # Skip whole-padding Q tiles: when the tile's FIRST row is already at
            # or past seqlen_q_b, every row of this m_block is a padding row and
            # there is nothing to compute.  The skip works by forcing
            # n_block_max = 0, which makes the KV loop below run zero iterations:
            # no K/V block is prefetched (the `n_block >= 0` guards), acc_O keeps
            # its zero fill and row_sum stays 0, so normalize_softmax's
            # `row_sum == 0` guard keeps scale = 1 and the epilogue stores exact
            # zeros for the whole tile.  Tiles that straddle seqlen_q_b are NOT
            # skipped: their valid rows compute normally, and their padding rows
            # are masked per-row in the masked steps (see softmax_rescale_O) —
            # those rows end up bounded (convex combinations of valid V rows),
            # not exact zeros, matching the CuTe DSL FMHA behaviour.
            n_block_max = (0 if m_block *
                           self._m_block_size >= seqlen_q_b else n_block_max)
            # Vision-block overlay: extend the per-CTA KV upper bound so tiles
            # covering [.., blockEnd[q]] beyond the causal diagonal are visited.
            # A block that crosses the upper Q-tile edge is the only interval that
            # can exceed the tile's causal bound, so its endpoint row supplies the
            # extension.  The extra tiles always take the masked path.  With no
            # block at that edge (blockEnd == -1), the extension is zero and the
            # traversal matches the plain causal kernel.
            n_block_max_causal = n_block_max
            extra_mask_steps = 0
            block_begin_tile_min = seqlen_k_b
            if cutlass.const_expr(mBlockEnd is not None):
                block_end_tile_max = cutlass.Int32(-1)
                row_lo = m_block * self._m_block_size
                row_hi = cutlass.min(row_lo + self._m_block_size, seqlen_q_b)
                row_hi = cutlass.max(row_hi, row_lo)
                if row_lo < row_hi:
                    # Vision intervals are contiguous and contain their query
                    # row.  Only a block crossing the lower tile edge can lower
                    # the sliding bound, and only one crossing the upper tile
                    # edge can extend the causal bound.  Intervals wholly inside
                    # the tile are already covered by those endpoint rows'
                    # analytic attention span.  Sampling the two endpoints avoids
                    # a redundant Br-element global scan in every CTA thread.
                    block_begin_first = mBlockBegin[batch_size, row_lo]
                    block_end_first = mBlockEnd[batch_size, row_lo]
                    if block_begin_first <= block_end_first:
                        block_begin_tile_min = cutlass.min(
                            block_begin_tile_min,
                            cutlass.max(block_begin_first, 0))
                    block_end_tile_max = mBlockEnd[batch_size, row_hi - 1]
                n_block_ext = cutlass.min(
                    cute.ceil_div(block_end_tile_max + 1, self._n_block_size),
                    cute.ceil_div(seqlen_k_b, self._n_block_size),
                )
                n_block_max = cutlass.max(n_block_max, n_block_ext)
                extra_mask_steps = n_block_max - n_block_max_causal

            # A sliding window never consumes complete KV tiles whose final key is
            # before the first allowed key of every row in this Q tile.  The
            # initial FMHA-v2 path still traversed those historical tiles and
            # merely score-masked them, which makes short-window Context FMHA do
            # nearly full-causal work.  Start at the tile containing the earliest
            # sliding-window key instead.  A vision-block interval is OR-ed with
            # the sliding/causal interval, so include its earliest valid begin as
            # well; the -1/-1 text-row sentinel does not lower the bound.
            n_block_min = cutlass.Int32(0)
            if cutlass.const_expr(self._use_sliding_window):
                first_query_row = m_block * self._m_block_size
                first_allowed_key = cutlass.max(
                    first_query_row + offset_b - window_size_left, 0)
                if cutlass.const_expr(mBlockBegin is not None):
                    first_allowed_key = cutlass.min(first_allowed_key,
                                                    block_begin_tile_min)
                n_block_min = cutlass.min(
                    n_block_max, first_allowed_key // self._n_block_size)
            n_block = n_block_max - 1

            # ///////////////////////////////////////////////////////////////////////////////
            # Get the appropriate tiles for this thread block.
            # Q is indexed by num_head (Q head); K/V by num_head_kv (shared head).
            # ///////////////////////////////////////////////////////////////////////////////
            q_extent = mQ.shape[1]
            q_copy_extent = mQ.shape[1]
            if cutlass.const_expr(self._packed_varlen):
                q_extent = max_seqlen
                q_copy_extent = seqlen_q_b
                q_seq_stride = num_q_heads * self._head_dim
                q_seq_stride = cute.assume(q_seq_stride, divby=8)
                q_head_offset = (mCuSeqLenQ[batch_size] * q_seq_stride +
                                 num_head * self._head_dim)
                q_head_ptr_raw = mQ.iterator + q_head_offset
                q_head_ptr = cute.make_ptr(
                    self._dtype,
                    q_head_ptr_raw.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                q_head_view = cute.make_tensor(
                    q_head_ptr,
                    cute.make_layout(
                        (max_seqlen, self._head_dim),
                        stride=(q_seq_stride, 1),
                    ),
                )
                gQ = cute.local_tile(
                    q_head_view,
                    (self._m_block_size, self._head_dim_padded),
                    (m_block, 0),
                )
            else:
                gQ = cute.local_tile(
                    mQ[batch_size, None, num_head, None],
                    (self._m_block_size, self._head_dim_padded),
                    (m_block, 0),
                )
            kv_capacity = 0
            kv_head_dim = 0
            if cutlass.const_expr(mPagedKV is not None):
                kv_capacity = mPageList.shape[2] * 128
                kv_head_dim = mPagedKV.shape[3]
            elif cutlass.const_expr(mKV is not None):
                kv_capacity = mKV.shape[3]
                kv_head_dim = mKV.shape[4]
                gK = cute.local_tile(
                    mKV[batch_size, 0, num_head_kv, None, None],
                    (self._n_block_size, self._head_dim_padded),
                    (None, 0),
                )
                gV = cute.local_tile(
                    mKV[batch_size, 1, num_head_kv, None, None],
                    (self._n_block_size, self._output_block_size),
                    (None, output_block),
                )
            elif cutlass.const_expr(self._packed_varlen):
                kv_capacity = max_seqlen
                kv_head_dim = mK.shape[2]
                kv_seq_stride = num_kv_heads * self._head_dim
                kv_seq_stride = cute.assume(kv_seq_stride, divby=8)
                kv_head_offset = (mCuSeqLenK[batch_size] * kv_seq_stride +
                                  num_head_kv * self._head_dim)
                k_head_ptr_raw = mK.iterator + kv_head_offset
                v_head_ptr_raw = mV.iterator + kv_head_offset
                k_head_ptr = cute.make_ptr(
                    self._dtype,
                    k_head_ptr_raw.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                v_head_ptr = cute.make_ptr(
                    self._dtype,
                    v_head_ptr_raw.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                k_head_view = cute.make_tensor(
                    k_head_ptr,
                    cute.make_layout(
                        (max_seqlen, self._head_dim),
                        stride=(kv_seq_stride, 1),
                    ),
                )
                v_head_view = cute.make_tensor(
                    v_head_ptr,
                    cute.make_layout(
                        (max_seqlen, self._head_dim),
                        stride=(kv_seq_stride, 1),
                    ),
                )
                gK = cute.local_tile(
                    k_head_view,
                    (self._n_block_size, self._head_dim_padded),
                    (None, 0),
                )
                gV = cute.local_tile(
                    v_head_view,
                    (self._n_block_size, self._output_block_size),
                    (None, output_block),
                )
            else:
                kv_capacity = mK.shape[1]
                kv_head_dim = mK.shape[3]
                gK = cute.local_tile(
                    mK[batch_size, None, num_head_kv, None],
                    (self._n_block_size, self._head_dim_padded),
                    (None, 0),
                )
                gV = cute.local_tile(
                    mV[batch_size, None, num_head_kv, None],
                    (self._n_block_size, self._output_block_size),
                    (None, output_block),
                )

            # ///////////////////////////////////////////////////////////////////////////////
            # Get shared memory buffer
            # ///////////////////////////////////////////////////////////////////////////////
            smem = cutlass.utils.SmemAllocator()

            storage = smem.allocate(SharedStorage)
            sQ = storage.sQ.get_tensor(sQ_layout)
            sK = storage.sK.get_tensor(sKV_layout)
            sV = storage.sV.get_tensor(sV_layout)

            # Transposed view of V: (head_dim, n_block_size) for the BMM2 mma.
            sVt = cute.composition(
                sV,
                cute.make_layout(
                    (self._output_block_size, self._n_block_size),
                    stride=(self._n_block_size, 1),
                ),
            )

            gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
            tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
            tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
            if cutlass.const_expr(mPagedKV is not None):
                tKgK = None
                tVgV = None
            else:
                tKgK = gmem_thr_copy_QKV.partition_S(gK)
                tVgV = gmem_thr_copy_QKV.partition_S(gV)
            tKsK = gmem_thr_copy_QKV.partition_D(sK)
            tVsV = gmem_thr_copy_QKV.partition_D(sV)

            # ///////////////////////////////////////////////////////////////////////////////
            # Tile MMA compute thread partitions and allocate accumulators
            # ///////////////////////////////////////////////////////////////////////////////
            thr_mma = tiled_mma.get_slice(tidx)
            tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
            tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
            tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
            acc_shape_O = thr_mma.partition_shape_C(
                (self._m_block_size, self._output_block_size))
            acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
            acc_O.fill(0.0)

            # ///////////////////////////////////////////////////////////////////////////////
            # Smem copy atom tiling
            # ///////////////////////////////////////////////////////////////////////////////
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
            smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q,
                                                       tiled_mma)
            smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K,
                                                       tiled_mma)
            smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V,
                                                       tiled_mma)

            smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
            smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
            smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)

            tSsQ = smem_thr_copy_Q.partition_S(sQ)
            tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
            tSsK = smem_thr_copy_K.partition_S(sK)
            tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
            tOsVt = smem_thr_copy_V.partition_S(sVt)
            tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

            # ///////////////////////////////////////////////////////////////////////////////
            # Predicate: mark indices that need to copy when problem_shape isn't a
            # multiple of tile_shape.
            # ///////////////////////////////////////////////////////////////////////////////
            mcQ = cute.make_identity_tensor((1, q_extent, 1, self._head_dim))
            # Use a synthetic rank-4 identity tensor for K/V coordinates so the
            # predicate logic is shared by the BSND and packed-varlen paths.
            mcKV = cute.make_identity_tensor((1, kv_capacity, 1, kv_head_dim))
            cQ = cute.local_tile(
                mcQ[0, None, 0, None],
                (self._m_block_size, self._head_dim_padded),
                (m_block, 0),
            )
            cKV = cute.local_tile(
                mcKV[0, None, 0, None],
                (self._n_block_size, self._head_dim_padded),
                (n_block, 0),
            )

            tQcQ = gmem_thr_copy_QKV.partition_S(cQ)
            tKVcKV = gmem_thr_copy_QKV.partition_S(cKV)
            tVcV = tKVcKV
            if cutlass.const_expr(self._num_output_blocks == 2):
                cV = cute.local_tile(
                    mcKV[0, None, 0, None],
                    (self._n_block_size, self._output_block_size),
                    (n_block, output_block),
                )
                tVcV = gmem_thr_copy_QKV.partition_S(cV)
            # Only the k-tile of the predicate is materialised; m/n predicates use
            # the first tile inline (-2-3 % perf gain vs. allocating the whole tile).
            tQpQ = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tQsQ.shape[0][1],
                        cute.size(tQsQ, mode=[1]),
                        cute.size(tQsQ, mode=[2]),
                    ),
                    stride=(cute.size(tQsQ, mode=[2]), 0, 1),
                ),
                cutlass.Boolean,
            )
            tKVpKV = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tKsK.shape[0][1],
                        cute.size(tKsK, mode=[1]),
                        cute.size(tKsK, mode=[2]),
                    ),
                    stride=(cute.size(tKsK, mode=[2]), 0, 1),
                ),
                cutlass.Boolean,
            )
            for rest_v in cutlass.range_constexpr(tQpQ.shape[0]):
                for rest_k in cutlass.range_constexpr(tQpQ.shape[2]):
                    tQpQ[rest_v, 0, rest_k] = cute.elem_less(
                        tQcQ[(0, rest_v), 0, rest_k][3], self._head_dim)
            for rest_v in cutlass.range_constexpr(tKVpKV.shape[0]):
                for rest_k in cutlass.range_constexpr(tKVpKV.shape[2]):
                    tKVpKV[rest_v, 0, rest_k] = cute.elem_less(
                        tKVcKV[(0, rest_v), 0, rest_k][3], kv_head_dim)
            tVpV = tKVpKV
            if cutlass.const_expr(self._num_output_blocks == 2):
                tVpV = cute.make_rmem_tensor(
                    cute.make_layout(
                        (
                            tVsV.shape[0][1],
                            cute.size(tVsV, mode=[1]),
                            cute.size(tVsV, mode=[2]),
                        ),
                        stride=(cute.size(tVsV, mode=[2]), 0, 1),
                    ),
                    cutlass.Boolean,
                )
                for rest_v in cutlass.range_constexpr(tVpV.shape[0]):
                    for rest_k in cutlass.range_constexpr(tVpV.shape[2]):
                        tVpV[rest_v, 0, rest_k] = cute.elem_less(
                            tVcV[(0, rest_v), 0, rest_k][3], kv_head_dim)
            # ///////////////////////////////////////////////////////////////////////////////
            # Prefetch Prologue — start async loads of the last mn-tile (mn residue handled here).
            # ///////////////////////////////////////////////////////////////////////////////
            for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
                if cute.elem_less(tQcQ[0, m, 0][1], q_copy_extent):
                    cute.copy(
                        gmem_tiled_copy_QKV,
                        tQgQ[None, m, None],
                        tQsQ[None, m, None],
                        pred=tQpQ[None, m, None],
                    )
                else:
                    tQsQ[None, m, None].fill(0)
            # n_block == -1 only for a degenerate empty batch (seqlen_k_b == 0);
            # skip the K prefetch entirely — the KV loop below runs zero steps and
            # the epilogue stores the zero-initialised acc_O.
            #
            # NaN hardening: rows of the boundary K/V tile
            # at logical positions >= seqlen_k_b are ZERO-FILLED, not just score
            # masked.  Score masking alone zeroes the softmax weight, but BMM2
            # still multiplies P(=0) x V, and IEEE 0 x NaN = NaN — padding rows can
            # legitimately hold NaN/Inf mid-network (fp16 overflow of garbage
            # embeddings in non-attention layers), which would poison the whole
            # boundary q-tile.  Zero-filled K/V make the masked columns inert
            # (0 x 0 = 0) regardless of what padding memory contains.  The
            # prologue loads the boundary tile (first processed block), so the
            # logical bound is applied here and in the first V load below; interior
            # blocks are entirely below seqlen_k_b and keep the fast full copies.
            if n_block >= 0:
                if cutlass.const_expr(mPagedKV is not None):
                    tPagedK = self._paged_partition(
                        mPagedKV,
                        mPageList,
                        batch_size,
                        num_head_kv,
                        0,
                        n_block,
                        gmem_thr_copy_QKV,
                    )
                    for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                        if cute.elem_less(tKVcKV[0, n, 0][1], seqlen_k_b):
                            cute.copy(
                                gmem_tiled_copy_QKV,
                                tPagedK[None, n, None],
                                tKsK[None, n, None],
                                pred=tKVpKV[None, n, None],
                            )
                        else:
                            tKsK[None, n, None].fill(0)
                else:
                    for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                        if cute.elem_less(tKVcKV[0, n, 0][1], seqlen_k_b):
                            cute.copy(
                                gmem_tiled_copy_QKV,
                                tKgK[None, n, None, n_block],
                                tKsK[None, n, None],
                                pred=tKVpKV[None, n, None],
                            )
                        else:
                            tKsK[None, n, None].fill(0)

            cute.arch.cp_async_commit_group()

            # ///////////////////////////////////////////////////////////////////////////////
            # Online-softmax intermediate state: row_max and row_sum (per-row scalars).
            # ///////////////////////////////////////////////////////////////////////////////
            row_max = cute.make_rmem_tensor(
                (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32)
            row_sum = cute.make_rmem_tensor(
                (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32)
            row_max.fill(-cutlass.Float32.inf)
            row_sum.fill(0.0)

            basic_params = SimpleNamespace(
                m_block=m_block,
                n_block=n_block,
                mQ=mQ,
                batch_size=batch_size,
                num_head=num_head,
                seqlen_q_b=seqlen_q_b,
                seqlen_k_b=seqlen_k_b,
                offset_b=offset_b,
                mBlockBegin=mBlockBegin,
                mBlockEnd=mBlockEnd,
                kv_capacity=kv_capacity,
                q_extent=q_extent,
                q_copy_extent=q_copy_extent,
                window_size_left=window_size_left,
                n_block_min=n_block_min,
            )
            mma_params = SimpleNamespace(
                thr_mma=thr_mma,
                tiled_mma=tiled_mma,
                tSrQ=tSrQ,
                tSrK=tSrK,
                tOrVt=tOrVt,
                acc_O=acc_O,
            )
            gmem_copy_params = SimpleNamespace(
                gmem_tiled_copy_QKV=gmem_tiled_copy_QKV,
                gmem_thr_copy_QKV=gmem_thr_copy_QKV,
                tKVcKV=tKVcKV,
                tKgK=tKgK,
                tKsK=tKsK,
                tVgV=tVgV,
                tVsV=tVsV,
                tKVpKV=tKVpKV,
                tVcV=tVcV,
                tVpV=tVpV,
                mPagedKV=mPagedKV,
                mPageList=mPageList,
                batch_size=batch_size,
                num_head_kv=num_head_kv,
                output_column_offset=output_column_offset,
            )
            smem_copy_params = SimpleNamespace(
                smem_tiled_copy_Q=smem_tiled_copy_Q,
                smem_tiled_copy_K=smem_tiled_copy_K,
                smem_tiled_copy_V=smem_tiled_copy_V,
                tSsQ=tSsQ,
                tSrQ_copy_view=tSrQ_copy_view,
                tSsK=tSsK,
                tSrK_copy_view=tSrK_copy_view,
                tOsVt=tOsVt,
                tOrVt_copy_view=tOrVt_copy_view,
            )
            softmax_params = SimpleNamespace(
                row_max=row_max,
                row_sum=row_sum,
                softmax_scale_log2=softmax_scale_log2,
            )

            # Two flavours of N-block iteration: masking (the block straddling the
            # per-batch seqlen_k_b boundary, and the causal tail) and unmasking.
            # Always at least one masking step.  Causal needs ceil(Br/Bc) diagonal
            # blocks +1: with a per-batch bottom-right offset the diagonal band
            # straddles one extra K block when offset_b % Bc != 0.  mask_steps is
            # a compile-time constant (this loop is unrolled) while offset_b is
            # runtime data, so the +1 is budgeted unconditionally — for aligned
            # offsets the extra masked pass simply finds nothing to mask (one
            # 16-key block per CTA takes the masked path instead of the fast
            # path; correctness over a cheap no-op).
            mask_steps = 1
            if cutlass.const_expr(self._is_causal):
                mask_steps = cute.ceil_div(self._m_block_size,
                                           self._n_block_size) + 1

            if cutlass.const_expr(self._use_sliding_window):
                n_block = n_block_max - 1
                basic_params.n_block = n_block
                if n_block >= 0:
                    self.compute_one_n_block(
                        basic_params,
                        mma_params,
                        gmem_copy_params,
                        smem_copy_params,
                        softmax_params,
                        is_first_n_block=True,
                        in_mask_steps=True,
                    )
                if cutlass.const_expr(mPagedKV is not None
                                      and mBlockBegin is None
                                      and mBlockEnd is None):
                    # Native-paged sliding attention has no vision-block overlay.
                    # Only tiles crossing either moving window edge need score
                    # predicates; keep the fully valid interior on the fast path.
                    first_q = m_block * self._m_block_size
                    last_q = (
                        cutlass.min(first_q + self._m_block_size, seqlen_q_b) -
                        1)
                    lower_last = cutlass.max(
                        last_q + offset_b - window_size_left, 0)
                    upper_first = cutlass.min(first_q + offset_b + 1,
                                              seqlen_k_b)
                    remaining_end = cutlass.max(n_block_min, n_block_max - 1)
                    interior_lo = cutlass.min(
                        remaining_end,
                        cutlass.max(
                            n_block_min,
                            cute.ceil_div(lower_last, self._n_block_size),
                        ),
                    )
                    interior_end = cutlass.min(
                        remaining_end,
                        cutlass.max(
                            interior_lo,
                            upper_first // self._n_block_size,
                        ),
                    )
                    for n_tile in range(0, remaining_end - interior_end, 1):
                        n_block = remaining_end - n_tile - 1
                        basic_params.n_block = n_block
                        self.compute_one_n_block(
                            basic_params,
                            mma_params,
                            gmem_copy_params,
                            smem_copy_params,
                            softmax_params,
                            is_first_n_block=False,
                            in_mask_steps=True,
                        )
                    for n_tile in range(0, interior_end - interior_lo, 1):
                        n_block = interior_end - n_tile - 1
                        basic_params.n_block = n_block
                        self.compute_one_n_block(
                            basic_params,
                            mma_params,
                            gmem_copy_params,
                            smem_copy_params,
                            softmax_params,
                            is_first_n_block=False,
                            in_mask_steps=False,
                        )
                    for n_tile in range(0, interior_lo - n_block_min, 1):
                        n_block = interior_lo - n_tile - 1
                        basic_params.n_block = n_block
                        self.compute_one_n_block(
                            basic_params,
                            mma_params,
                            gmem_copy_params,
                            smem_copy_params,
                            softmax_params,
                            is_first_n_block=False,
                            in_mask_steps=True,
                        )
                else:
                    for n_tile in range(1, n_block_max - n_block_min, 1):
                        n_block = n_block_max - n_tile - 1
                        basic_params.n_block = n_block
                        self.compute_one_n_block(
                            basic_params,
                            mma_params,
                            gmem_copy_params,
                            smem_copy_params,
                            softmax_params,
                            is_first_n_block=False,
                            in_mask_steps=True,
                        )
            else:
                for n_tile in cutlass.range_constexpr(mask_steps):
                    n_block = n_block_max - n_tile - 1
                    basic_params.n_block = n_block
                    if cutlass.const_expr(self._is_causal):
                        if n_block >= 0:
                            self.compute_one_n_block(
                                basic_params,
                                mma_params,
                                gmem_copy_params,
                                smem_copy_params,
                                softmax_params,
                                is_first_n_block=(n_tile == 0),
                                in_mask_steps=True,
                            )
                    else:
                        if n_block >= 0:
                            self.compute_one_n_block(
                                basic_params,
                                mma_params,
                                gmem_copy_params,
                                smem_copy_params,
                                softmax_params,
                                is_first_n_block=True,
                                in_mask_steps=True,
                            )

                # Vision-block extension tiles: KV tiles between the causal
                # diagonal and this Q tile's largest blockEnd.
                for n_tile in range(mask_steps, mask_steps + extra_mask_steps,
                                    1):
                    n_block = n_block_max - n_tile - 1
                    basic_params.n_block = n_block
                    if n_block >= 0:
                        self.compute_one_n_block(
                            basic_params,
                            mma_params,
                            gmem_copy_params,
                            smem_copy_params,
                            softmax_params,
                            is_first_n_block=False,
                            in_mask_steps=True,
                        )

                # Remaining K-tiles in reverse order — no k-residue handling.
                for n_tile in range(mask_steps + extra_mask_steps, n_block_max,
                                    1):
                    n_block = n_block_max - n_tile - 1
                    basic_params.n_block = n_block
                    self.compute_one_n_block(
                        basic_params,
                        mma_params,
                        gmem_copy_params,
                        smem_copy_params,
                        softmax_params,
                        is_first_n_block=False,
                        in_mask_steps=False,
                    )

            # ///////////////////////////////////////////////////////////////////////////////
            # Epilogue: normalize acc_O, rmem → smem (aliased over sQ) → gmem.
            # ///////////////////////////////////////////////////////////////////////////////
            self.normalize_softmax(acc_O, row_sum)
            rO = cute.make_fragment_like(acc_O, self._dtype)
            rO.store(acc_O.load().to(self._dtype))
            sO = cute.make_tensor(sQ.iterator, sO_layout)

            smem_copy_atom_O = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self._dtype)
            smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O,
                                                       tiled_mma)
            smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
            taccOrO = smem_thr_copy_O.retile(rO)
            taccOsO = smem_thr_copy_O.partition_D(sO)
            cute.copy(
                smem_copy_atom_O,
                taccOrO,
                taccOsO,
            )
            if cutlass.const_expr(self._packed_varlen):
                o_seq_stride = num_q_heads * self._head_dim
                o_seq_stride = cute.assume(o_seq_stride, divby=8)
                o_head_offset = (mCuSeqLenQ[batch_size] * o_seq_stride +
                                 num_head * self._head_dim)
                o_head_ptr_raw = mO.iterator + o_head_offset
                o_head_ptr = cute.make_ptr(
                    self._dtype,
                    o_head_ptr_raw.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                o_head_view = cute.make_tensor(
                    o_head_ptr,
                    cute.make_layout(
                        (max_seqlen, self._head_dim),
                        stride=(o_seq_stride, 1),
                    ),
                )
                gO = cute.local_tile(
                    o_head_view,
                    (self._m_block_size, self._output_block_size),
                    (m_block, output_block),
                )
            else:
                gO = cute.local_tile(
                    mO[batch_size, None, num_head, None],
                    (self._m_block_size, self._output_block_size),
                    (m_block, output_block),
                )

            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            tOgO = gmem_thr_copy_O.partition_D(gO)
            tOrO = cute.make_fragment_like(tOgO, self._dtype)
            self.cta_sync_barrier.arrive_and_wait()
            cute.copy(
                gmem_tiled_copy_O,
                tOsO,
                tOrO,
            )
            mcO = cute.make_identity_tensor((1, q_extent, 1, self._head_dim))
            cO = cute.local_tile(
                mcO[0, None, 0, None],
                (self._m_block_size, self._output_block_size),
                (m_block, output_block),
            )
            tOcO = gmem_thr_copy_O.partition_D(cO)
            tOpO = cute.make_rmem_tensor(
                cute.make_layout(
                    (tOgO.shape[0][1], tOgO.shape[1], tOgO.shape[2]),
                    stride=(tOgO.shape[2], 0, 1),
                ),
                cutlass.Boolean,
            )
            for rest_v in cutlass.range_constexpr(tOpO.shape[0]):
                for rest_n in cutlass.range_constexpr(cute.size(
                        tOpO.shape[2])):
                    tOpO[rest_v, 0, rest_n] = cute.elem_less(
                        tOcO[(0, rest_v), 0, rest_n][3], self._head_dim)
            for rest_m in cutlass.range_constexpr(cute.size(tOpO.shape[1])):
                if cute.elem_less(tOcO[0, rest_m, 0][1], q_copy_extent):
                    cute.copy(
                        gmem_tiled_copy_O,
                        tOrO[None, rest_m, None],
                        tOgO[None, rest_m, None],
                        pred=tOpO[None, rest_m, None],
                    )

    @cute.jit
    def _paged_tile(
        self,
        mPagedKV: cute.Tensor,
        mPageList: cute.Tensor,
        batch_size: cutlass.Int32,
        num_head_kv: cutlass.Int32,
        kv_plane: cutlass.Constexpr,
        n_block: cutlass.Int32,
        column_offset: cutlass.Int32,
        column_extent: cutlass.Constexpr,
    ):
        """Resolve one logical N tile to its physical paged NHD K/V view."""
        blocks_per_page = 128 // self._n_block_size
        logical_page = n_block // blocks_per_page
        physical_page = mPageList[batch_size, kv_plane, logical_page]
        return self._paged_tile_from_physical_page(
            mPagedKV,
            num_head_kv,
            physical_page,
            n_block,
            column_offset,
            column_extent,
        )

    @cute.jit
    def _paged_tile_from_physical_page(
        self,
        mPagedKV: cute.Tensor,
        num_head_kv: cutlass.Int32,
        physical_page: cutlass.Int32,
        n_block: cutlass.Int32,
        column_offset: cutlass.Int32,
        column_extent: cutlass.Constexpr,
    ):
        """Build one logical N tile after its page-table entry is prefetched."""
        blocks_per_page = 128 // self._n_block_size
        logical_page = n_block // blocks_per_page
        block_in_page = n_block - logical_page * blocks_per_page
        page_stride = mPagedKV.layout.stride[0]
        head_stride = mPagedKV.layout.stride[1]
        token_stride = mPagedKV.layout.stride[2]
        element_offset = (physical_page * page_stride +
                          num_head_kv * head_stride +
                          block_in_page * self._n_block_size * token_stride +
                          column_offset)
        raw_ptr = mPagedKV.iterator + element_offset
        aligned_ptr = cute.make_ptr(
            mPagedKV.element_type,
            raw_ptr.toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        gPagedKV = cute.make_tensor(
            aligned_ptr,
            cute.make_layout(
                (self._n_block_size, column_extent),
                stride=(token_stride, 1),
            ),
        )
        return cute.make_tensor(gPagedKV.iterator.align(16), gPagedKV.layout)

    @cute.jit
    def _paged_partition(
        self,
        mPagedKV: cute.Tensor,
        mPageList: cute.Tensor,
        batch_size: cutlass.Int32,
        num_head_kv: cutlass.Int32,
        kv_plane: cutlass.Constexpr,
        n_block: cutlass.Int32,
        gmem_thr_copy_QKV: cute.TiledCopy,
    ):
        """Return this thread's source partition for one FP16 paged KV tile."""
        gPagedKV = self._paged_tile(
            mPagedKV,
            mPageList,
            batch_size,
            num_head_kv,
            kv_plane,
            n_block,
            0,
            self._head_dim_padded,
        )
        tPagedKV = gmem_thr_copy_QKV.partition_S(gPagedKV)
        # Dynamic page/head/token offsets make partitioning conservatively
        # drop the ABI's 16-byte alignment. Each thread still begins on one
        # complete 128-bit copy vector, so restore that fact at the copy edge.
        return cute.make_tensor(tPagedKV.iterator.align(16), tPagedKV.layout)

    @cute.jit
    def _paged_partition_from_physical_page(
        self,
        mPagedKV: cute.Tensor,
        num_head_kv: cutlass.Int32,
        physical_page: cutlass.Int32,
        n_block: cutlass.Int32,
        gmem_thr_copy_QKV: cute.TiledCopy,
    ):
        """Partition one FP16 KV tile after its page-table entry is prefetched."""
        gPagedKV = self._paged_tile_from_physical_page(
            mPagedKV,
            num_head_kv,
            physical_page,
            n_block,
            0,
            self._head_dim_padded,
        )
        tPagedKV = gmem_thr_copy_QKV.partition_S(gPagedKV)
        return cute.make_tensor(tPagedKV.iterator.align(16), tPagedKV.layout)

    @cute.jit
    def _paged_partition_output_from_physical_page(
        self,
        mPagedKV: cute.Tensor,
        num_head_kv: cutlass.Int32,
        physical_page: cutlass.Int32,
        n_block: cutlass.Int32,
        output_column_offset: cutlass.Int32,
        gmem_thr_copy_QKV: cute.TiledCopy,
    ):
        """Partition one D512 V slice after its page-table entry is prefetched."""
        gPagedV = self._paged_tile_from_physical_page(
            mPagedKV,
            num_head_kv,
            physical_page,
            n_block,
            output_column_offset,
            self._output_block_size,
        )
        tPagedV = gmem_thr_copy_QKV.partition_S(gPagedV)
        return cute.make_tensor(tPagedV.iterator.align(16), tPagedV.layout)

    @cute.jit
    def compute_one_n_block(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        gmem_copy_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        is_first_n_block: cutlass.Constexpr,
        in_mask_steps: cutlass.Constexpr,
    ):
        """One outer N-block: BMM1 (Q @ K^T → S) → online softmax → BMM2 (P @ V → O)."""
        acc_shape_S = mma_params.thr_mma.partition_shape_C(
            (self._m_block_size, self._n_block_size))
        acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
        acc_S.fill(0.0)

        # Start both dependent page-table reads before waiting for the current
        # K tile. Their scalar results stay live in registers while cp.async
        # and BMM1 make forward progress.
        current_v_page = cutlass.Int32(0)
        next_k_page = cutlass.Int32(0)
        if cutlass.const_expr(gmem_copy_params.mPagedKV is not None):
            blocks_per_page = 128 // self._n_block_size
            current_v_logical_page = basic_params.n_block // blocks_per_page
            current_v_page = gmem_copy_params.mPageList[
                gmem_copy_params.batch_size, 1, current_v_logical_page]
            if basic_params.n_block > basic_params.n_block_min:
                next_k_logical_page = (basic_params.n_block -
                                       1) // blocks_per_page
                next_k_page = gmem_copy_params.mPageList[
                    gmem_copy_params.batch_size, 0, next_k_logical_page]

        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        # First tile: load V into smem with the n-residue predicate (otherwise
        # a single vectorised copy is enough — the `if` here is a constexpr).
        # First tile == the boundary K/V block: zero-fill V rows at logical
        # positions >= seqlen_k_b (NaN hardening — see the prologue K load).
        if cutlass.const_expr(gmem_copy_params.mPagedKV is not None):
            if cutlass.const_expr(self._num_output_blocks == 2):
                tPagedV = self._paged_partition_output_from_physical_page(
                    gmem_copy_params.mPagedKV,
                    gmem_copy_params.num_head_kv,
                    current_v_page,
                    basic_params.n_block,
                    gmem_copy_params.output_column_offset,
                    gmem_copy_params.gmem_thr_copy_QKV,
                )
            else:
                tPagedV = self._paged_partition_from_physical_page(
                    gmem_copy_params.mPagedKV,
                    gmem_copy_params.num_head_kv,
                    current_v_page,
                    basic_params.n_block,
                    gmem_copy_params.gmem_thr_copy_QKV,
                )
            if is_first_n_block:
                for n in cutlass.range_constexpr(
                        cute.size(gmem_copy_params.tVsV.shape[1])):
                    if cute.elem_less(
                            gmem_copy_params.tVcV[0, n, 0][1],
                            basic_params.seqlen_k_b,
                    ):
                        cute.copy(
                            gmem_copy_params.gmem_tiled_copy_QKV,
                            tPagedV[None, n, None],
                            gmem_copy_params.tVsV[None, n, None],
                            pred=gmem_copy_params.tVpV[None, n, None],
                        )
                    else:
                        gmem_copy_params.tVsV[None, n, None].fill(0.0)
            else:
                cute.copy(
                    gmem_copy_params.gmem_tiled_copy_QKV,
                    tPagedV,
                    gmem_copy_params.tVsV,
                    pred=gmem_copy_params.tVpV,
                )
        else:
            if is_first_n_block:
                for n in cutlass.range_constexpr(
                        cute.size(gmem_copy_params.tVsV.shape[1])):
                    if cute.elem_less(
                            gmem_copy_params.tVcV[0, n, 0][1],
                            basic_params.seqlen_k_b,
                    ):
                        cute.copy(
                            gmem_copy_params.gmem_tiled_copy_QKV,
                            gmem_copy_params.tVgV[None, n, None,
                                                  basic_params.n_block],
                            gmem_copy_params.tVsV[None, n, None],
                            pred=gmem_copy_params.tVpV[None, n, None],
                        )
                    else:
                        gmem_copy_params.tVsV[None, n, None].fill(0.0)
            else:
                cute.copy(
                    gmem_copy_params.gmem_tiled_copy_QKV,
                    gmem_copy_params.tVgV[None, None, None,
                                          basic_params.n_block],
                    gmem_copy_params.tVsV,
                    pred=gmem_copy_params.tVpV,
                )

        cute.arch.cp_async_commit_group()
        # ///////////////////////////////////////////////////////////////////////////////
        # S = Q @ K^T  (BMM1)
        # ///////////////////////////////////////////////////////////////////////////////
        cute.copy(
            smem_copy_params.smem_tiled_copy_Q,
            smem_copy_params.tSsQ[None, None, 0],
            smem_copy_params.tSrQ_copy_view[None, None, 0],
        )
        cute.copy(
            smem_copy_params.smem_tiled_copy_K,
            smem_copy_params.tSsK[None, None, 0],
            smem_copy_params.tSrK_copy_view[None, None, 0],
        )
        for k in cutlass.range_constexpr(
                cute.size(smem_copy_params.tSsQ.shape[2])):
            k_next = (k + 1) % cute.size(smem_copy_params.tSsQ.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_Q,
                smem_copy_params.tSsQ[None, None, k_next],
                smem_copy_params.tSrQ_copy_view[None, None, k_next],
            )
            cute.copy(
                smem_copy_params.smem_tiled_copy_K,
                smem_copy_params.tSsK[None, None, k_next],
                smem_copy_params.tSrK_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                acc_S,
                mma_params.tSrQ[None, None, k],
                mma_params.tSrK[None, None, k],
                acc_S,
            )

        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        if basic_params.n_block > basic_params.n_block_min:
            if cutlass.const_expr(gmem_copy_params.mPagedKV is not None):
                tPagedKNext = self._paged_partition_from_physical_page(
                    gmem_copy_params.mPagedKV,
                    gmem_copy_params.num_head_kv,
                    next_k_page,
                    basic_params.n_block - 1,
                    gmem_copy_params.gmem_thr_copy_QKV,
                )
                cute.copy(
                    gmem_copy_params.gmem_tiled_copy_QKV,
                    tPagedKNext,
                    gmem_copy_params.tKsK,
                    pred=gmem_copy_params.tKVpKV,
                )
            else:
                cute.copy(
                    gmem_copy_params.gmem_tiled_copy_QKV,
                    gmem_copy_params.tKgK[None, None, None,
                                          basic_params.n_block - 1],
                    gmem_copy_params.tKsK,
                    pred=gmem_copy_params.tKVpKV,
                )
            cute.arch.cp_async_commit_group()
        # ///////////////////////////////////////////////////////////////////////////////
        # Online softmax: rescale row_max/row_sum/acc_O, compute P = softmax(S).
        # ///////////////////////////////////////////////////////////////////////////////
        skip_tile = self.softmax_rescale_O(
            basic_params,
            mma_params,
            softmax_params,
            acc_S,
            is_first_n_block,
            in_mask_steps,
        )

        # A skipped tile leaves row_max/row_sum/acc_O untouched, so P@V is
        # elided outright (P would be exact zeros). No CTA barrier sits between
        # here and the next tile's cp.async wait, so the per-warp branch is safe.
        if cutlass.const_expr(self._skip_softmax):
            if not skip_tile:
                self.bmm2(acc_S, mma_params, smem_copy_params)
        else:
            self.bmm2(acc_S, mma_params, smem_copy_params)

    @cute.jit
    def bmm2(
        self,
        acc_S: cute.Tensor,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
    ):
        """O += P @ V (BMM2) for this warp's rows; ``acc_S`` already holds P."""
        rP = cute.make_fragment_like(acc_S, self._dtype)
        rP.store(acc_S.load().to(self._dtype))
        # mma.m16n8k16 layout: rearrange P from (4, MMA_M, MMA_N)
        # to ((4, 2), MMA_M, MMA_N / 2) to match the operand-A shape.
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
            smem_copy_params.tOsVt[None, None, 0],
            smem_copy_params.tOrVt_copy_view[None, None, 0],
        )
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            k_next = (k + 1) % cute.size(tOrS.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_V,
                smem_copy_params.tOsVt[None, None, k_next],
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
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        acc_S: cute.Tensor,
        is_first_n_block: cutlass.Constexpr,
        in_mask_steps: cutlass.Constexpr,
    ):
        """Apply online softmax to ``acc_S`` and rescale ``acc_O``.

        Distinguishes ``is_first_n_block`` (no prior row_max to fold in) from
        subsequent blocks, and masked vs. unmasked steps.  When
        ``skip_rescale`` is enabled and ``exp2(m_prev - m_cur) ≈ 1``, the
        rescale factor is snapped to exactly ``1.0`` to suppress precision
        wobble (see the upstream implementation).

        Returns the warp-uniform skip-softmax decision for this tile (Python
        ``False`` when the skip path is compiled out).
        """
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        acc_O_mn = self._make_acc_tensor_mn_view(mma_params.acc_O)
        row_max_prev = None
        if cutlass.const_expr(not is_first_n_block):
            row_max_prev = cute.make_fragment_like(softmax_params.row_max,
                                                   cutlass.Float32)
            cute.basic_copy(softmax_params.row_max, row_max_prev)
        # The first tile (causal diagonal) never skips: row_max is still -inf.
        enable_skip = self._skip_softmax and not is_first_n_block
        tScS_mn = None
        if cutlass.const_expr(in_mask_steps or enable_skip):
            mcS = cute.make_identity_tensor((
                1,
                basic_params.q_extent,
                1,
                basic_params.kv_capacity,
            ))
            cS = cute.local_tile(
                mcS[0, None, 0, None],
                (self._m_block_size, self._n_block_size),
                (basic_params.m_block, basic_params.n_block),
            )
            tScS = mma_params.thr_mma.partition_C(cS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)

        skip_tile = False
        if cutlass.const_expr(not enable_skip):
            for r in cutlass.range_constexpr(cute.size(
                    softmax_params.row_max)):
                if cutlass.const_expr(in_mask_steps):
                    self.mask_score_row(basic_params, acc_S_mn, tScS_mn, r)
                row_max_cur_row = self._threadquad_reduce_max(
                    acc_S_mn[r, None].load().reduce(cute.ReductionOp.MAX,
                                                    -cutlass.Float32.inf, 0))
                row_max_prev_row = None
                if cutlass.const_expr(not is_first_n_block):
                    row_max_prev_row = row_max_prev[r]
                    row_max_cur_row = cute.arch.fmax(row_max_prev_row,
                                                     row_max_cur_row)
                self.softmax_row(
                    softmax_params,
                    acc_S_mn,
                    acc_O_mn,
                    r,
                    row_max_cur_row,
                    row_max_prev_row,
                    is_first_n_block,
                )
        else:
            # Skip-softmax (BLASST): per-row masked tile max first, then ONE
            # warp-wide vote. Each warp owns its rows' O accumulator outright,
            # so a unanimous warp skips exp/rescale/P@V with no CTA exchange.
            # Per-thread predicate: (tile_max - running_max) * scale_log2 <
            # log2(lambda). Padding rows never contribute; they vote skip.
            tile_row_max = cute.make_fragment_like(softmax_params.row_max,
                                                   cutlass.Float32)
            thread_wants_skip = cutlass.Boolean(True)
            thread_all_oob = cutlass.Boolean(True)
            for r in cutlass.range_constexpr(cute.size(
                    softmax_params.row_max)):
                if cutlass.const_expr(in_mask_steps):
                    self.mask_score_row(basic_params, acc_S_mn, tScS_mn, r)
                tile_row_max[r] = self._threadquad_reduce_max(
                    acc_S_mn[r, None].load().reduce(cute.ReductionOp.MAX,
                                                    -cutlass.Float32.inf, 0))
                row_is_oob = tScS_mn[r, 0][1] >= basic_params.seqlen_q_b
                row_wants_skip = (
                    (tile_row_max[r] - row_max_prev[r]) *
                    softmax_params.softmax_scale_log2
                    < softmax_params.skip_softmax_threshold_log2) or row_is_oob
                thread_wants_skip = thread_wants_skip and row_wants_skip
                thread_all_oob = thread_all_oob and row_is_oob
            skip_tile = cute.arch.vote_all_sync(thread_wants_skip)
            if cutlass.const_expr(
                    softmax_params.skip_softmax_count is not None):
                # Self-test counters: exclude all-padding warps, which always
                # vote skip by construction and would inflate the ratio.
                warp_all_oob = cute.arch.vote_all_sync(thread_all_oob)
                if not warp_all_oob:
                    if cute.arch.lane_idx() == 0:
                        if skip_tile:
                            cute.arch.atomic_add(
                                softmax_params.skip_softmax_count.iterator.
                                llvm_ptr,
                                cutlass.Int32(1),
                            )
                        cute.arch.atomic_add(
                            softmax_params.total_softmax_count.iterator.
                            llvm_ptr,
                            cutlass.Int32(1),
                        )
            if not skip_tile:
                for r in cutlass.range_constexpr(
                        cute.size(softmax_params.row_max)):
                    row_max_prev_row = row_max_prev[r]
                    self.softmax_row(
                        softmax_params,
                        acc_S_mn,
                        acc_O_mn,
                        r,
                        cute.arch.fmax(row_max_prev_row, tile_row_max[r]),
                        row_max_prev_row,
                        is_first_n_block,
                    )
        return skip_tile

    @cute.jit
    def mask_score_row(
        self,
        basic_params: SimpleNamespace,
        acc_S_mn: cute.Tensor,
        tScS_mn: cute.Tensor,
        r: cutlass.Constexpr,
    ):
        """Per-batch variable-length masking of score row ``r``:
        residual K: mask index_k >= seqlen_k_b (padding keys);
        causal:     bottom-right aligned, limit = qi + offset_b + 1;
        residual Q: padding rows (qi >= seqlen_q_b) fully masked
        (masked steps only — in unmasked blocks padding rows still
        accumulate bounded valid-V combinations).
        """
        row_idx = tScS_mn[r, 0][1]
        if cutlass.const_expr(not self._is_causal):
            col_idx_limit = basic_params.seqlen_k_b
        else:
            col_idx_limit = cutlass.min(row_idx + 1 + basic_params.offset_b,
                                        basic_params.seqlen_k_b)
        col_idx_limit = (0 if row_idx +
                         1 > basic_params.seqlen_q_b else col_idx_limit)
        col_idx_begin = cutlass.Int32(0)
        if cutlass.const_expr(basic_params.window_size_left is not None):
            # AttentionPlugin passes W-1.  Therefore the first allowed
            # key for a query row q is q-(W-1), i.e. this inclusive
            # lower bound has no additional +1.
            col_idx_begin = cutlass.max(
                row_idx + basic_params.offset_b -
                basic_params.window_size_left,
                0,
            )
        if cutlass.const_expr(basic_params.mBlockBegin is not None):
            # Vision-block overlay: this row's extra allowed KV
            # interval.  The 0/-1 default keeps the interval empty
            # for padding rows (also guarding the (B, S_q) gather
            # against out-of-bounds row indices in residual tiles);
            # text rows store the -1/-1 sentinel which is empty for
            # the same reason (no key satisfies col <= -1).
            block_begin_r = cutlass.Int32(0)
            block_end_r = cutlass.Int32(-1)
            if row_idx < basic_params.seqlen_q_b:
                block_begin_r = basic_params.mBlockBegin[
                    basic_params.batch_size, row_idx]
                block_end_r = cutlass.min(
                    basic_params.mBlockEnd[basic_params.batch_size, row_idx],
                    basic_params.seqlen_k_b - 1,
                )
            for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                col_idx = tScS_mn[0, c][3]
                # Masked iff outside the analytic sliding-causal
                # interval AND outside the row's vision block.
                if cute.elem_less(col_idx, block_begin_r):
                    if cute.elem_less(col_idx + 1, col_idx_begin + 1):
                        acc_S_mn[r, c] = -cutlass.Float32.inf
                    if cute.elem_less(col_idx_limit, col_idx + 1):
                        acc_S_mn[r, c] = -cutlass.Float32.inf
                if cute.elem_less(block_end_r, col_idx):
                    if cute.elem_less(col_idx + 1, col_idx_begin + 1):
                        acc_S_mn[r, c] = -cutlass.Float32.inf
                    if cute.elem_less(col_idx_limit, col_idx + 1):
                        acc_S_mn[r, c] = -cutlass.Float32.inf
        else:
            for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                col_idx = tScS_mn[0, c][3]
                if cute.elem_less(col_idx + 1, col_idx_begin + 1):
                    acc_S_mn[r, c] = -cutlass.Float32.inf
                if cute.elem_less(col_idx_limit, col_idx + 1):
                    acc_S_mn[r, c] = -cutlass.Float32.inf

    @cute.jit
    def softmax_row(
        self,
        softmax_params: SimpleNamespace,
        acc_S_mn: cute.Tensor,
        acc_O_mn: cute.Tensor,
        r: cutlass.Constexpr,
        row_max_cur_row: cutlass.Float32,
        row_max_prev_row: Optional[cutlass.Float32],
        is_first_n_block: cutlass.Constexpr,
    ):
        """Row ``r``: P = exp2(S*scale - max*scale), fold row_sum, rescale acc_O."""
        acc_S_row = acc_S_mn[r, None].load()
        # Use a finite max for exponent arithmetic on fully-masked rows,
        # but preserve -inf in the running max until a valid score arrives.
        row_max_safe_row = (0.0 if row_max_cur_row == -cutlass.Float32.inf else
                            row_max_cur_row)

        acc_S_row_exp = cute.math.exp2(
            acc_S_row * softmax_params.softmax_scale_log2 -
            row_max_safe_row * softmax_params.softmax_scale_log2,
            fastmath=True,
        )
        acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD,
                                             cutlass.Float32.zero, 0)
        if cutlass.const_expr(not is_first_n_block):
            prev_minus_cur_exp = cute.math.exp2(
                row_max_prev_row * softmax_params.softmax_scale_log2 -
                row_max_safe_row * softmax_params.softmax_scale_log2,
                fastmath=True,
            )
            # skip_rescale: snap a near-unity rescale factor (within 2^-8)
            # to exactly 1.0 — trades one FFMA for a stabilized rescale.
            if cutlass.const_expr(self._skip_rescale):
                prev_minus_cur_exp = (
                    cutlass.Float32(1.0) if prev_minus_cur_exp
                    > cutlass.Float32(0.99609375) else prev_minus_cur_exp)
            acc_S_row_sum = (acc_S_row_sum +
                             softmax_params.row_sum[r] * prev_minus_cur_exp)
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
        """Final softmax normalisation: ``acc_O[r, :] /= row_sum[r]``."""
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[
                r] != row_sum[r]

            scale = (1.0 if acc_O_mn_row_is_zero_or_nan else
                     cute.arch.rcp_approx(row_sum[r]))

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

    def _threadquad_reduce(self, val: cutlass.Float32,
                           op: Callable) -> cutlass.Float32:
        """Reduce across the four threads holding the same column of an MMA fragment."""
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val,
                                        offset=2,
                                        mask=-1,
                                        mask_and_clamp=31),
        )
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val,
                                        offset=1,
                                        mask=-1,
                                        mask_and_clamp=31),
        )
        return val

    def _threadquad_reduce_max(self, val: cutlass.Float32) -> cutlass.Float32:
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val: cutlass.Float32) -> cutlass.Float32:
        return self._threadquad_reduce(val, lambda x, y: x + y)
