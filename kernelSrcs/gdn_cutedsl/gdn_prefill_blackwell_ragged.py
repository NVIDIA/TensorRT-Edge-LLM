# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Adapted from https://github.com/flashinfer-ai/flashinfer at revision
# 1989b509d345bd0c3c2ffa1f0ea39afec7cc4c33:
# - flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py
# - flashinfer/gdn_kernels/blackwell/gated_delta_net_tile_scheduler.py

"""
Chunked Gated Delta Net (GDN) prefill kernel for Blackwell SM100.

Algorithm overview (per chunk c, tokens [cC, (c+1)C)):
  Inputs : Q[BT,DK], K[BT,DK], V[BT,DV], a[BT], b[BT], A_log, dt_bias
  State  : S_prev[DK,DV]  (recurrent state, held in TMEM)

  Preprocessing (compute warp group 0):
    cumsumlog[t]     = sum_{l=0}^{t} log(gate_l)              cumulative log of gates
    cumprod[t]       = exp(cumsumlog[t])                       cumulative product of gates
    T_pairwise[i,j]  = cumprod[i] / cumprod[j]  (i>=j)       inter-token transfer weights
    (stored in registers; 128 regs/thread)

  GEMM 1 - kk   : W_kk[BT,BT]  = K  @ K^T       (lower-triangular intra scores)
  GEMM 2 - qk   : W_qk[BT,BT]  = Q  @ K^T       (output attention scores)
  GEMM 3 - k*state : KS[BT,DV] = K  @ S_prev    (key applied to state)
  GEMM 4 - q*state : QS[BT,DV] = Q  @ S_prev    (inter-chunk output, before T scaling)
  GEMM 5 - new v   : NV[BT,DV] = A_inv @ V       (corrected value vectors)
                      where A_inv = (I + M_kk)^{-1},  M_kk[i,j] = T[i,j]*beta[i]*W_kk[i,j]  (lower-tri, hierarchical blockwise inverse)
  GEMM 6 - qkv  : O_intra[BT,DV] = W_qkv @ NV   (intra-chunk output)
                   where W_qkv = T*beta*W_qk (scaled qk scores)
  GEMM 7 - kv update : dS[DK,DV] = K^T @ delta       (state update, BT contraction)
                        where delta[BT,DV] = V - KS    (delta rule residuals, after decay)

  Epilogue:
    O[BT,DV]  = O_intra + T_col * QS             (combine intra + inter)
    S_next    = cumprod[BT-1] * S_prev + dS        (update state in TMEM)

SMEM layout (225.5 KB total):
  Buffer                           Size (B)  Stages
  q                                32768     1
  k                                32768     2      <-- double-buffered (prefetch next chunk)
  v                                32768     1
  A_inverse / new_v                32768     1      <-- A_inv result, then overwritten with fp16 NV
  QK output / O store              32768     1      <-- W_qk scores, then O epilogue staging
  state / decay_v                  32768     1      <-- state GMEM<->TMEM staging, shared with decay_v ALU
  cumsumlog                          512     1      <-- BT x fp32 scalars
  cumprod                            512     1
  cumprod_scale                      512     1

TMEM layout (256 KB total):
  Buffer                  Size (B)  Stages
  state (S)               65536     1      <-- DKxDV fp32 = 128x128x4B
  q*state acc             65536     1      <-- BTxDV fp32 accumulator
  qk/kk/new_v/k*state/    65536     2      <-- shared accumulator for all other GEMMs
    kv/qkv acc

Warp assignments (12 warps = 384 threads):
  warps 0-3     : compute group 0 - T-pairwise, kk_epi, qk_epi, inverse
  warps 4-7     : compute group 1 - kv_decay_v, v-k*state, state*q_epi,
                                    new_v_epi, kv_update_epi, qkv_epilogue
  warp  8       : MMA warp       - issues all 7 GEMMs
  warp  9       : TMA load warp  - loads q, k (double-buf), v
  warp  10      : MMA warp       - issues state/output GEMMs
  warp  11      : epilogue warp  - prepares gate/beta and stores O
"""

import math
from typing import Optional, Tuple

import cuda.bindings.driver as cuda


import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass._mlir import ir
from cutlass.cutlass_dsl import (
    Integer,
    Int32,
    dsl_user_op,
    extract_mlir_values,
    min as dsl_min,
    new_from_mlir_values,
)
from cutlass.utils import TensorMapManager, TensorMapUpdateMode, WorkTileInfo
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05, OperandMajorMode
import cutlass.utils.blackwell_helpers as sm100_utils

# ---------------------------------------------------------------------------
# cutlass-dsl 4.4.2 compatibility: TmaInfo was removed; make_tiled_tma_atom_*
# now returns a plain (CopyAtom, Tensor) tuple instead of TmaInfo.
# ---------------------------------------------------------------------------
try:
    from cutlass.cute.nvgpu.cpasync import TmaInfo
except ImportError:
    from cutlass.base_dsl import (
        extract_mlir_values as _emv,
        new_from_mlir_values as _nfmv,
        extract_mlir_attributes as _ema,
        get_mlir_types as _gmt,
    )

    class TmaInfo:  # type: ignore[no-redef]
        """Compatibility shim replacing cpasync.TmaInfo for cutlass-dsl >= 4.4.2."""

        def __init__(self, atom, tma_tensor, smem_layout=None):
            self._atom = atom
            self._tma_tensor = tma_tensor

        @property
        def atom(self):
            return self._atom

        @property
        def tma_tensor(self):
            return self._tma_tensor

        def __extract_mlir_values__(self):
            return _emv(self._atom) + _emv(self._tma_tensor)

        def __extract_mlir_attributes__(self):
            return _ema(self._atom) + _ema(self._tma_tensor)

        def __new_from_mlir_values__(self, values):
            n = len(_gmt(self._atom))
            return TmaInfo(
                _nfmv(self._atom, values[:n]),
                _nfmv(self._tma_tensor, values[n:]),
            )

        def __iter__(self):
            yield self._atom
            yield self._tma_tensor

        def __getitem__(self, i):
            return (self._atom, self._tma_tensor)[i]

        def __len__(self):
            return 2


def _wrap_tma(ret):
    """Wrap make_tiled_tma_atom_* return value in TmaInfo if not already."""
    if isinstance(ret, TmaInfo):
        return ret
    # 4.4.2: returns (CopyAtom, Tensor) tuple
    return TmaInfo(ret[0], ret[1])


@cute.jit
def _get_safe_sequence_bounds(
    cu_seqlens: cute.Tensor,
    batch_idx: cutlass.Int32,
    total_tokens: cutlass.Int32,
):
    """Clamp one packed sequence interval to the physical token allocation."""
    raw_start = cutlass.Int32(cu_seqlens[batch_idx])
    raw_end = cutlass.Int32(cu_seqlens[batch_idx + 1])
    batch_start = max(0, min(raw_start, total_tokens))
    batch_end = max(batch_start, min(raw_end, total_tokens))
    return cutlass.Int32(batch_start), cutlass.Int32(batch_end)


class GDNTileSchedulerParams:
    def __init__(
        self,
        num_seqs: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        is_GQA: bool,
        total_tiles: Optional[cutlass.Int32] = None,
        num_seqs_fdd: cute.FastDivmodDivisor = None,
        num_heads_fdd: cute.FastDivmodDivisor = None,
        *,
        loc=None,
        ip=None,
    ):
        self.num_seqs = num_seqs
        self.num_q_heads = num_q_heads
        self.num_v_heads = num_v_heads
        self.is_GQA = is_GQA
        self.num_o_heads = num_q_heads if is_GQA else num_v_heads
        if total_tiles is None:
            self.total_tiles = num_seqs * self.num_o_heads
        else:
            self.total_tiles = total_tiles
        self.num_seqs_fdd = num_seqs_fdd
        self.num_heads_fdd = num_heads_fdd
        if num_seqs_fdd is None:
            self.num_seqs_fdd = cute.fast_divmod_create_divisor(
                num_seqs, loc=loc, ip=ip
            )
        if num_heads_fdd is None:
            self.num_heads_fdd = cute.fast_divmod_create_divisor(
                self.num_o_heads, loc=loc, ip=ip
            )
        self._loc = loc

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self.num_seqs,
            self.num_q_heads,
            self.num_v_heads,
            self.is_GQA,
            self.total_tiles,
            self.num_seqs_fdd,
            self.num_heads_fdd,
        ]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self.num_seqs,
                self.num_q_heads,
                self.num_v_heads,
                self.is_GQA,
                self.total_tiles,
                self.num_seqs_fdd,
                self.num_heads_fdd,
            ],
            self._values_pos,
            strict=True,
        ):
            obj_list.append(new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]

        return GDNTileSchedulerParams(*(tuple(obj_list)), loc=self._loc)

    @dsl_user_op
    def get_grid_shape(
        self, max_active_clusters: Int32, *, loc=None, ip=None
    ) -> Tuple[Integer, Integer, Integer]:
        return (dsl_min(self.total_tiles, max_active_clusters), 1, 1)


class GDNTileScheduler:
    """Assign one sequence/head tile at a time to each persistent CTA."""

    def __init__(
        self,
        params: GDNTileSchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        num_persistent_ctas: Int32,
        current_work_linear_idx: Int32,
    ):
        self.params = params
        self.block_idx = block_idx
        self.num_persistent_ctas = num_persistent_ctas
        self._current_work_linear_idx = current_work_linear_idx

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values, self._values_pos = [], []
        for obj in [
            self.params,
            self.block_idx,
            self.num_persistent_ctas,
            self._current_work_linear_idx,
        ]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "GDNTileScheduler":
        obj_list = []
        for obj, n_items in zip(
            [
                self.params,
                self.block_idx,
                self.num_persistent_ctas,
                self._current_work_linear_idx,
            ],
            self._values_pos,
            strict=True,
        ):
            obj_list.append(new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return GDNTileScheduler(*obj_list)

    @staticmethod
    @dsl_user_op
    def create(
        params: GDNTileSchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        *,
        loc=None,
        ip=None,
    ) -> "GDNTileScheduler":
        num_persistent_ctas = Int32(cute.size(grid_dim, loc=loc, ip=ip))
        bidx, _, _ = block_idx
        return GDNTileScheduler(
            params,
            block_idx,
            num_persistent_ctas,
            current_work_linear_idx=Int32(bidx),
        )

    @staticmethod
    def get_grid_shape(
        params: GDNTileSchedulerParams,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Integer, Integer, Integer]:
        return params.get_grid_shape(max_active_clusters, loc=loc, ip=ip)

    def _get_current_work_for_linear_idx(
        self, linear_idx: Int32, *, loc=None, ip=None
    ) -> WorkTileInfo:
        is_valid = linear_idx < cute.size(self.params.total_tiles, loc=loc, ip=ip)
        remain_work_idx, head_idx = divmod(linear_idx, self.params.num_heads_fdd)
        _, batch_idx = divmod(remain_work_idx, self.params.num_seqs_fdd)
        cur_tile_coord = tuple(Int32(x) for x in (batch_idx, head_idx, 1))
        return WorkTileInfo(cur_tile_coord, is_valid)

    @dsl_user_op
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self._get_current_work_for_linear_idx(
            self._current_work_linear_idx, loc=loc, ip=ip
        )

    @dsl_user_op
    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self.get_current_work(loc=loc, ip=ip)

    @dsl_user_op
    def advance_to_next_work(self, *, advance_count: int = 1, loc=None, ip=None):
        self._current_work_linear_idx += Int32(advance_count) * Int32(
            self.num_persistent_ctas
        )


# ---------------------------------------------------------------------------
# Combined configuration + execution class
# ---------------------------------------------------------------------------


class GatedDeltaNetChunkedKernel:
    """
    Configuration and execution class for the Chunked GDN kernel.

    Main responsibilities:
      - __init__    : warp IDs, barriers, tile shapes, SMEM/TMEM sizes
      - __call__    : @cute.jit host entry point (TMA setup, kernel launch)
      - kernel      : @cute.kernel device entry point (warp dispatch)
      - per-warp methods called from kernel's chunk loop

    The kernel is specialized for FP16 input/output, FP32 accumulation, and
    fixed MMA tile shapes.
    """

    # TMA descriptor size in bytes
    arch = "sm_100"
    bytes_per_tensormap = 128
    num_tensormaps = 4

    def __init__(
        self,
        is_GQA: bool,
        store_final_state: bool = True,
    ):
        self.io_dtype = cutlass.Float16
        self.mma_tiler_qk = (64, 64, 128)
        self.mma_tiler_qs = (128, 64, 128)
        self.mma_tiler_qkv = (128, 64, 64)
        self.mma_tiler_kv = (128, 128, 64)
        self.is_GQA = is_GQA
        self.store_final_state = store_final_state

        # ------------------------------------------------------------------
        # Warp assignments  (12 warps total)
        # ------------------------------------------------------------------
        # T-pairwise / kk_epi / qk_epi / inverse
        self.compute_group_0_warp_ids = [0, 1, 2, 3]
        # kv_decay_v / v-k*state / epi ops
        self.compute_group_1_warp_ids = [4, 5, 6, 7]
        self.mma_warp_id = 8
        self.tma_qkv_warp_id = 9
        # The second issuer owns the five state/output GEMMs.
        self.mma_cg1_warp_id = 10
        # store O
        self.epilogue_warp_id = 11
        # The lightly loaded O epilogue warp also prefetches gate/beta.
        self.load_gate_beta_warp_id = self.epilogue_warp_id

        # Give the MMA/TMA/gate/epilogue warps enough registers to keep their
        # pipeline handles resident. Fund the increase from CG1 first while
        # preserving the existing 64,512-register CTA budget.
        self.num_regs_compute_group_0 = 224
        self.num_regs_compute_group_1 = 256
        self.num_regs_other = 24

        self.threads_per_cta = 32 * (
            len(
                (
                    self.mma_warp_id,
                    self.tma_qkv_warp_id,
                    self.mma_cg1_warp_id,
                    self.epilogue_warp_id,
                )
            )
            + len(self.compute_group_0_warp_ids)
            + len(self.compute_group_1_warp_ids)
        )

        self.use_2cta_instrs = False
        self.cluster_shape_mnk = (1, 1, 1)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        self.occupancy = 1
        self.threads_per_warp = 32
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

        # ------------------------------------------------------------------
        # Named barriers - only TmemAllocator requires a NamedBarrier;
        # all other inter-warp synchronization uses mbarrier-based pipelines
        # created inside kernel().
        # ------------------------------------------------------------------
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_warp
            * len(
                (
                    self.mma_warp_id,
                    self.mma_cg1_warp_id,
                    *self.compute_group_0_warp_ids,
                    *self.compute_group_1_warp_ids,
                )
            ),
        )
        self.inverse_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp * len(self.compute_group_0_warp_ids),
        )
        self.init_state_store_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp * len(self.compute_group_1_warp_ids),
        )

    def _setup_attributes(self):
        # ------------------------------------------------------------------
        # SMEM sizes (bytes per stage) and stage counts
        # ------------------------------------------------------------------
        self.smem_q_stages = 2
        # Four K stages let TMA make K3 available while K0 remains live for
        # KV0, so the next pair's KK0/KK1 can be issued back-to-back.
        self.smem_k_stages = 4
        # Three V stages both break the double-KK lookahead dependency cycle
        # and let TMA stay ahead while CG1 consumes the current pair.
        self.smem_v_stages = 3
        # Mid-pair KK lookahead overlaps next-pair inverse preparation with the
        # current second chunk.  NV0 has released current Ainv0 by then, so the
        # live set is current Ainv1 plus next Ainv0/1: three stages total.
        self.smem_ainv_stages = 3
        self.smem_qk_stages = 2
        # Gate/beta work now shares the epilogue warp, so O uses two stages to
        # avoid back-pressuring CG1 while the warp publishes the next gate.
        self.smem_o_stages = 2
        # Five resident stages preserve four chunks of gate/beta lookahead.
        # Cumulative gate buffers are placed last in SMEM.
        self.smem_gate_stages = 5
        # Let the scalar producer enter the next pair after CG0 releases beta0
        # instead of waiting for both current-pair beta stages.
        self.smem_beta_stages = 5

        # ------------------------------------------------------------------
        # TMEM column offsets and buffer sizes (fp32, 32B per column)
        # ------------------------------------------------------------------
        self.tmem_kv_acc_stages = 1
        self.tmem_q_state_acc_stages = 1
        self.tmem_state_inp_stages = 1
        self.tmem_shared_inp_stages = 2
        # CG0 owns KK/QK and CG1 owns KS/NV.  Separate rings remove the
        # cross-group ownership handoff from the shared-acc critical path.
        self.tmem_cg0_shared_acc_stages = 2
        self.tmem_cg1_shared_acc_stages = 1

        self.tmem_state_offset = 0
        self.tmem_q_state_offset = (
            self.tmem_state_offset + self.tmem_kv_acc_stages * 128
        )
        self.tmem_state_inp_offset = (
            self.tmem_q_state_offset + self.tmem_q_state_acc_stages * 64
        )
        self.tmem_cg0_shared_acc_offset = (
            self.tmem_state_inp_offset + self.tmem_state_inp_stages * 64
        )
        self.tmem_cg1_shared_acc_offset = (
            self.tmem_cg0_shared_acc_offset + self.tmem_cg0_shared_acc_stages * 64
        )
        self.tmem_shared_inp_offset = (
            self.tmem_cg1_shared_acc_offset + self.tmem_cg1_shared_acc_stages * 64
        )

        self.buffer_align_bytes = 1024

    # -----------------------------------------------------------------------
    # Host entry point
    # -----------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        o: cute.Tensor,
        cu_seqlens: cute.Tensor,
        s_in: Optional[cute.Tensor],
        s_out: Optional[cute.Tensor],
        s_indices: Optional[cute.Tensor],
        scale: cutlass.Float32,
        tensormap_workspace: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        # chunk size
        self.b_t = 64
        h_q = q.shape[1]
        h_v = v.shape[1]
        batch_size = cu_seqlens.shape[0] - 1

        self._setup_attributes()

        if cutlass.const_expr(self.is_GQA):
            h_r = h_q // h_v
            h_qv = h_v
            q = cute.make_tensor(
                q.iterator,
                cute.make_layout(
                    (q.shape[0], q.shape[2], (h_r, h_v)),
                    stride=(q.stride[0], q.stride[2], (q.stride[1], h_r * q.stride[1])),
                ),
            )
            k = cute.make_tensor(
                k.iterator,
                cute.make_layout(
                    (k.shape[0], k.shape[2], (h_r, h_v)),
                    stride=(k.stride[0], k.stride[2], (0, k.stride[1])),
                ),
            )
            v = cute.make_tensor(
                v.iterator,
                cute.make_layout(
                    (v.shape[2], v.shape[0], (h_r, h_v)),
                    stride=(v.stride[2], v.stride[0], (0, v.stride[1])),
                ),
            )
        else:
            h_r = h_v // h_q
            h_qv = h_q
            q = cute.make_tensor(
                q.iterator,
                cute.make_layout(
                    (q.shape[0], q.shape[2], (h_r, h_q)),
                    stride=(q.stride[0], q.stride[2], (0, q.stride[1])),
                ),
            )
            k = cute.make_tensor(
                k.iterator,
                cute.make_layout(
                    (k.shape[0], k.shape[2], (h_r, h_q)),
                    stride=(k.stride[0], k.stride[2], (0, k.stride[1])),
                ),
            )
            v = cute.make_tensor(
                v.iterator,
                cute.make_layout(
                    (v.shape[2], v.shape[0], (h_r, h_q)),
                    stride=(v.stride[2], v.stride[0], (v.stride[1], h_r * v.stride[1])),
                ),
            )

        a = cute.make_tensor(
            a.iterator,
            cute.make_layout(
                (a.shape[0], (h_r, h_qv)),
                stride=(a.stride[0], (a.stride[1], h_r * a.stride[1])),
            ),
        )
        b = cute.make_tensor(
            b.iterator,
            cute.make_layout(
                (b.shape[0], (h_r, h_qv)),
                stride=(b.stride[0], (b.stride[1], h_r * b.stride[1])),
            ),
        )
        o = cute.make_tensor(
            o.iterator,
            cute.make_layout(
                (o.shape[2], o.shape[0], (h_r, h_qv)),
                stride=(o.stride[2], o.stride[0], (o.stride[1], h_r * o.stride[1])),
            ),
        )
        if cutlass.const_expr(s_in is not None):
            s_in = cute.make_tensor(
                s_in.iterator,
                cute.make_layout(
                    (s_in.shape[2], s_in.shape[3], (h_r, h_qv), s_in.shape[0]),
                    stride=(
                        s_in.stride[3],
                        s_in.stride[2],
                        (s_in.stride[1], h_r * s_in.stride[1]),
                        s_in.stride[0],
                    ),
                ),
            )
        if cutlass.const_expr(s_out is not None):
            s_out = cute.make_tensor(
                s_out.iterator,
                cute.make_layout(
                    (s_out.shape[2], s_out.shape[3], (h_r, h_qv), s_out.shape[0]),
                    stride=(
                        s_out.stride[3],
                        s_out.stride[2],
                        (s_out.stride[1], h_r * s_out.stride[1]),
                        s_out.stride[0],
                    ),
                ),
            )
        # ------------------------------------------------------------------
        # Build tiled MMAs  (one per logical GEMM group, differing in operand major modes)
        # ------------------------------------------------------------------
        def _mma_op(mma_tiler, a_major, b_major, OperandSourceA):
            # Derive MMA atom (M, N) from the first two dims of the tile shape;
            # K=16 is the hardware fp16 atom depth (fixed for SM100 tcgen05).
            return tcgen05.MmaF16BF16Op(
                self.io_dtype,
                cutlass.Float32,
                (mma_tiler[0], mma_tiler[1], 16),
                self.cta_group,
                OperandSourceA,
                a_major,
                b_major,
            )

        # GEMM 1 (kk: K@K^T) + GEMM 2 (qk: Q@K^T)          - KK-major: A=K, B=K
        tiled_mma_qk = cute.make_tiled_mma(
            _mma_op(
                self.mma_tiler_qk,
                OperandMajorMode.K,
                OperandMajorMode.K,
                tcgen05.OperandSource.SMEM,
            )
        )
        # GEMM 3 (k*state: K@S) + GEMM 4 (q*state: Q@S)     - KN-major: A=K, B=MN
        tiled_mma_qs = cute.make_tiled_mma(
            _mma_op(
                self.mma_tiler_qs,
                OperandMajorMode.K,
                OperandMajorMode.K,
                tcgen05.OperandSource.TMEM,
            )
        )
        # GEMM 5 (new_v: A_inv@V) + GEMM 6 (qkv: W_qkv@NV)  - KN-major: A=K, B=MN
        tiled_mma_qkv = cute.make_tiled_mma(
            _mma_op(
                self.mma_tiler_qkv,
                OperandMajorMode.K,
                OperandMajorMode.K,
                tcgen05.OperandSource.TMEM,
            )
        )
        # for v_smem_layout_staged
        tiled_mma_qkv_ss = cute.make_tiled_mma(
            _mma_op(
                self.mma_tiler_qkv,
                OperandMajorMode.MN,
                OperandMajorMode.K,
                tcgen05.OperandSource.SMEM,
            )
        )
        # GEMM 7 (kv_update: K^T@delta -> dS)                    - MN-major: A=MN, B=MN
        tiled_mma_kv = cute.make_tiled_mma(
            _mma_op(
                self.mma_tiler_kv,
                OperandMajorMode.K,
                OperandMajorMode.MN,
                tcgen05.OperandSource.TMEM,
            )
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

        # ------------------------------------------------------------------
        # SMEM layouts - computed before SharedStorage so cosize() is available
        # ------------------------------------------------------------------
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        # Q is A operand for tiled_mma_qk (GEMM 2: qk), K is B operand (GEMM 1+2: kk/qk)
        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.io_dtype, self.smem_q_stages
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.io_dtype, self.smem_k_stages
        )
        k_trans_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_kv, self.mma_tiler_kv, self.io_dtype, self.smem_k_stages
        )
        # V is A operand for tiled_mma_qkv (GEMM 5: new_v: V @ A_inv)
        v_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_qkv_ss, self.mma_tiler_qkv, self.io_dtype, self.smem_v_stages
        )
        # A_inv is B operand for tiled_mma_qkv (GEMM 5: new_v: V @ A_inv).
        ainv_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_qkv,
            self.mma_tiler_qkv,
            self.io_dtype,
            self.smem_ainv_stages,
        )
        # W_qkv is A operand for tiled_mma_qkv (GEMM 6: qkv: NV @ qk)
        qk_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_qkv, self.mma_tiler_qkv, self.io_dtype, self.smem_qk_stages
        )

        o_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            cutlass.utils.LayoutEnum.from_tensor(o),
            self.mma_tiler_qkv[:2],
            self.smem_o_stages,
        )
        # Gate scalar arrays (1D Float32, flat layout - no swizzle needed)
        cumsumlog_smem_layout_staged = cute.make_layout(
            (self.b_t, 1, self.smem_gate_stages)
        )
        beta_smem_layout_staged = cute.make_layout((self.b_t, 1, self.smem_beta_stages))

        # ------------------------------------------------------------------
        # Shared memory struct  (defined here to capture layout cosizes)
        # ------------------------------------------------------------------
        @cute.struct
        class SharedStorage:
            # Pipeline mbarriers - one entry per stage, 2 Int64 words per barrier
            # TMA load warp -> MMA warp (K is staged for next-chunk prefetch)
            load_k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_k_stages * 2]
            # TMA load warp -> MMA warp
            load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_q_stages * 2]
            # TMA load warp -> CG1 V-load signal
            load_v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_v_stages * 2]
            # Gate/beta load warp -> CG0/CG1
            load_gate_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_gate_stages * 2
            ]
            # Gate/beta load warp -> CG0
            load_beta_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_beta_stages * 2
            ]
            # MMA warp -> CG1 (Q*state acc ready in TMEM)
            q_state_acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_q_state_acc_stages * 2
            ]
            # MMA warp -> CG1 (GEMM 7 done)
            kv_acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_kv_acc_stages * 2
            ]
            # MMA warp -> CG0 (KK/QK ready)
            cg0_shared_acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_cg0_shared_acc_stages * 2
            ]
            # MMA warp -> CG1 (KS/NV ready)
            cg1_shared_acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_cg1_shared_acc_stages * 2
            ]
            # CG0 -> MMA warp (A_inv ready in SMEM)
            ainv_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_ainv_stages * 2
            ]
            # CG0 -> MMA warp (QK ready in SMEM)
            qk_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_qk_stages * 2
            ]
            state_inp_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_state_inp_stages * 2
            ]
            # CG1 -> MMA warp: fixed-slot TMEM inputs.  The empty halves are
            # unused because downstream accumulator-full signals prove reuse.
            vks_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            nv_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            decay_v_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            # CG1 -> epilogue warp
            o_store_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_o_stages * 2
            ]
            # TMEM allocation token
            tmem_holding_buf: cutlass.Int32
            # SMEM tensor buffers (aligned, in SMEM layout order)
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

            sV: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(v_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # A_inv result consumed by the next MMA; keep this in io_dtype.
            sAinv: cute.struct.Align[
                cute.struct.MemRange[
                    self.io_dtype, cute.cosize(ainv_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # W_qk scores
            sQk: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(qk_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(o_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # Cumulative gate scalars - placed last in SMEM
            cumsumlog: cute.struct.MemRange[
                cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged)
            ]
            cumprod: cute.struct.MemRange[
                cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged)
            ]
            beta: cute.struct.MemRange[
                cutlass.Float32, cute.cosize(beta_smem_layout_staged)
            ]

        self.shared_storage = SharedStorage

        # ------------------------------------------------------------------
        # Build TMA atoms
        # ------------------------------------------------------------------
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])

        tma_q = _wrap_tma(
            cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                q,
                q_smem_layout,
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
            )
        )
        tma_k = _wrap_tma(
            cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                k,
                k_smem_layout,
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
            )
        )
        tma_v = _wrap_tma(
            cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                v,
                v_smem_layout,
                self.mma_tiler_qkv,
                tiled_mma_qkv_ss,
                self.cluster_layout_vmnk.shape,
            )
        )

        o_smem_layout = cute.select(o_smem_layout_staged, mode=[0, 1])
        tma_o = _wrap_tma(
            cpasync.make_tiled_tma_atom(
                tma_store_op,
                o,
                o_smem_layout,
                self.mma_tiler_qkv[:2],
            )
        )

        self.tma_q_bytes = cute.size_in_bytes(self.io_dtype, q_smem_layout)
        self.tma_k_bytes = cute.size_in_bytes(self.io_dtype, k_smem_layout)
        self.tma_v_bytes = cute.size_in_bytes(self.io_dtype, v_smem_layout)
        self.tma_o_bytes = cute.size_in_bytes(self.io_dtype, o_smem_layout)

        # ------------------------------------------------------------------
        # Launch
        # ------------------------------------------------------------------
        scheduler_params = GDNTileSchedulerParams(
            num_seqs=batch_size,
            num_q_heads=h_q,
            num_v_heads=h_v,
            is_GQA=self.is_GQA,
        )
        grid_shape = GDNTileScheduler.get_grid_shape(
            scheduler_params, max_active_clusters
        )

        self.kernel(
            tiled_mma_qk,
            tiled_mma_qs,
            tiled_mma_qkv,
            tiled_mma_qkv_ss,
            tiled_mma_kv,
            tma_q,
            tma_k,
            tma_v,
            a,
            b,
            A_log,
            dt_bias,
            tma_o,
            cu_seqlens,
            s_in,
            s_out,
            s_indices,
            scale,
            q_smem_layout_staged,
            k_smem_layout_staged,
            k_trans_smem_layout_staged,
            v_smem_layout_staged,
            cumsumlog_smem_layout_staged,
            beta_smem_layout_staged,
            ainv_smem_layout_staged,
            qk_smem_layout_staged,
            o_smem_layout_staged,
            scheduler_params,
            q,
            k,
            v,
            o,
            tensormap_workspace,
        ).launch(
            grid=grid_shape,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
            min_blocks_per_mp=1,
        )

    # -----------------------------------------------------------------------
    # Device kernel
    # -----------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        # Tiled MMAs (one per logical GEMM group)
        # GEMM 1 (kk) + GEMM 2 (qk)
        tiled_mma_qk: cute.TiledMma,
        # GEMM 3 (k*state) + GEMM 4 (q*state)
        tiled_mma_qs: cute.TiledMma,
        # GEMM 5 (new_v) + GEMM 6 (qkv)
        tiled_mma_qkv: cute.TiledMma,
        # GEMM 5 (new_v: A_inv@V) first tile
        tiled_mma_qkv_ss: cute.TiledMma,
        # GEMM 7 (kv_update)
        tiled_mma_kv: cute.TiledMma,
        # TMA descriptors and cute tensors
        tma_q: TmaInfo,
        tma_k: TmaInfo,
        tma_v: TmaInfo,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mA_log: cute.Tensor,
        mDt_bias: cute.Tensor,
        tma_o: TmaInfo,
        # (B+1,)  int32  cumulative seq lengths
        cu_seqlens: cute.Tensor,
        # initial state (fp32) from GMEM
        mS_init: cute.Tensor,
        # final state output (fp32) to GMEM; None if not stored
        mS_out: Optional[cute.Tensor],
        mS_indices: Optional[cute.Tensor],
        scale: cutlass.Float32,
        # SMEM staged layouts (needed to view shared_storage tensor buffers)
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        k_trans_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        cumsumlog_smem_layout_staged: cute.Layout,
        beta_smem_layout_staged: cute.Layout,
        ainv_smem_layout_staged: cute.ComposedLayout,
        qk_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        # Scheduler
        scheduler_params: GDNTileSchedulerParams,
        # TMA descriptor workspace in GMEM (one q/k/v/o descriptor set per CTA)
        # Slots: Q=0, K=1, V=2, O=3
        mQ,
        mK,
        mV,
        # used for TMA descriptor update
        mO,
        # (num_ctas, num_tensormaps, bytes_per_tensormap) Int8 workspace
        tensormap_workspace: cute.Tensor,
    ):
        """
        Main GDN chunked kernel.

        Warp specialization is the outermost control flow: each warp role owns
        its own persistent tile-scheduler loop, iterating over (batch, head)
        tiles and then over chunks within each tile.
        """
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, bidz = cute.arch.block_idx()
        grid_dim = cute.arch.grid_dim()

        if cutlass.const_expr(self.store_final_state):
            assert mS_out is not None, (
                "mS_out must be provided if store_final_state is True"
            )
        else:
            assert mS_out is None, "mS_out must be None if store_final_state is False"

        # ------------------------------------------------------------------
        # TMA descriptor GMEM workspace - one q/k/v/o descriptor set per CTA.
        # Slots: Q=0, K=1, V=2, O=3.
        # ------------------------------------------------------------------
        cta_linear_idx = bidz * grid_dim[1] * grid_dim[0] + bidy * grid_dim[0] + bidx

        tensormap_manager = TensorMapManager(
            TensorMapUpdateMode.GMEM, self.bytes_per_tensormap
        )

        tensormap_workspace = self.initialize_workspace(tensormap_workspace, grid_dim)
        tensormap_q_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 0, None)].iterator
        )
        tensormap_k_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 1, None)].iterator
        )
        tensormap_v_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 2, None)].iterator
        )
        tensormap_o_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 3, None)].iterator
        )

        # ------------------------------------------------------------------
        # 1. Allocate SMEM / TMEM, prefetch TMA descriptors
        # ------------------------------------------------------------------
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        sK_trans = storage.sK.get_tensor(
            k_trans_smem_layout_staged.outer, swizzle=k_trans_smem_layout_staged.inner
        )
        sV = storage.sV.get_tensor(
            v_smem_layout_staged.outer, swizzle=v_smem_layout_staged.inner
        )
        # A_inverse MMA input.
        sAinv = storage.sAinv.get_tensor(
            ainv_smem_layout_staged.outer, swizzle=ainv_smem_layout_staged.inner
        )
        sAinvCal = sAinv
        # QK output / O store  (W_qk first, then O epilogue staging)
        sQk = storage.sQk.get_tensor(
            qk_smem_layout_staged.outer, swizzle=qk_smem_layout_staged.inner
        )
        sO = storage.sO.get_tensor(
            o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner
        )
        # Gate scalar arrays (1D Float32, flat - no swizzle)
        sCumsumlog = storage.cumsumlog.get_tensor(cumsumlog_smem_layout_staged)
        sCumprod = storage.cumprod.get_tensor(cumsumlog_smem_layout_staged)
        sBeta = storage.beta.get_tensor(beta_smem_layout_staged)

        if warp_idx == self.mma_warp_id:
            cpasync.prefetch_descriptor(tma_q.atom)
            cpasync.prefetch_descriptor(tma_k.atom)
            cpasync.prefetch_descriptor(tma_v.atom)
            cpasync.prefetch_descriptor(tma_o.atom)

        # TMEM allocator object - CG1 will issue the actual allocation
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            # CG1 owns allocation and is the last group to release TMEM state.
            allocator_warp_id=self.compute_group_1_warp_ids[0],
        )

        # ------------------------------------------------------------------
        # mbarrier-based pipelines
        # Each pipeline is created by all threads; barrier_storage points into SMEM.
        # defer_sync=True means pipeline_init_arrive() flushes all at once below.
        # ------------------------------------------------------------------
        def _cg(num_threads):
            return pipeline.CooperativeGroup(pipeline.Agent.Thread, num_threads)

        # 1 thread (TMA issuer)
        cg_tma = _cg(len([self.tma_qkv_warp_id]))
        # 1 warp (gate/beta scalar load warp)
        cg_gate = _cg(self.threads_per_warp * len([self.load_gate_beta_warp_id]))
        # One producer per result pipeline; K/Q are consumed by both issuers.
        cg_mma = _cg(len([self.mma_warp_id]))
        cg_mma_both = _cg(len([self.mma_warp_id, self.mma_cg1_warp_id]))
        # 128 threads (CG0)
        cg_cg0 = _cg(self.threads_per_warp * len(self.compute_group_0_warp_ids))
        # 128 threads (CG1)
        cg_cg1 = _cg(self.threads_per_warp * len(self.compute_group_1_warp_ids))
        # 4 threads (one per CG1 warp, used for V load signaling)
        cg_cg1_v = _cg(len(self.compute_group_1_warp_ids))
        # 256 threads (CG0 + CG1)
        cg_both = _cg(
            self.threads_per_warp * len(self.compute_group_0_warp_ids)
            + self.threads_per_warp * len(self.compute_group_1_warp_ids)
        )
        # 32 threads (epilogue warp)
        cg_epi = _cg(self.threads_per_warp * len([self.epilogue_warp_id]))

        # TMA load pipelines: K/Q feed MMA; V is signaled to CG1 for ALU work.
        load_k_producer, load_k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.smem_k_stages,
            producer_group=cg_tma,
            consumer_group=cg_mma_both,
            tx_count=self.tma_k_bytes,
            barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.smem_q_stages,
            producer_group=cg_tma,
            consumer_group=cg_mma_both,
            tx_count=self.tma_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        load_v_producer, load_v_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.smem_v_stages,
            producer_group=cg_tma,
            consumer_group=cg_cg1_v,
            tx_count=self.tma_v_bytes,
            barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        # Unified loader warp 9 -> CG0 / CG1: gate/beta (software-signaled)
        # Scalar-load paths do not use TMA barriers; producer calls commit() after writes.
        load_gate_producer, load_gate_consumer = pipeline.PipelineAsync.create(
            num_stages=self.smem_gate_stages,
            producer_group=cg_gate,
            consumer_group=cg_both,
            barrier_storage=storage.load_gate_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        load_beta_producer, load_beta_consumer = pipeline.PipelineAsync.create(
            num_stages=self.smem_beta_stages,
            producer_group=cg_gate,
            consumer_group=cg_cg0,
            barrier_storage=storage.load_beta_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # MMA warp -> CG1:  kv_acc
        kv_acc_producer, kv_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.tmem_kv_acc_stages,
            producer_group=cg_mma,
            consumer_group=cg_cg1,
            barrier_storage=storage.kv_acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # MMA warp -> CG1:  q_state_acc
        q_state_acc_producer, q_state_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.tmem_q_state_acc_stages,
            producer_group=cg_mma,
            consumer_group=cg_cg1,
            barrier_storage=storage.q_state_acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # MMA warp -> CG0: KK/QK accumulator ring.
        cg0_shared_acc_producer, cg0_shared_acc_consumer = (
            pipeline.PipelineUmmaAsync.create(
                num_stages=self.tmem_cg0_shared_acc_stages,
                producer_group=cg_mma,
                consumer_group=cg_cg0,
                barrier_storage=storage.cg0_shared_acc_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # MMA warp -> CG1: KS/NV accumulator ring.
        cg1_shared_acc_producer, cg1_shared_acc_consumer = (
            pipeline.PipelineUmmaAsync.create(
                num_stages=self.tmem_cg1_shared_acc_stages,
                producer_group=cg_mma,
                consumer_group=cg_cg1,
                barrier_storage=storage.cg1_shared_acc_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # CG0 -> MMA warp:  a_inv_done
        a_inv_ready_producer, a_inv_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.smem_ainv_stages,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.ainv_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # CG0 -> MMA warp:  qk_done
        qk_ready_producer, qk_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.smem_qk_stages,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.qk_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # CG1 -> MMA warp: state input.
        state_inp_ready_producer, state_inp_ready_consumer = (
            pipeline.PipelineAsyncUmma.create(
                num_stages=self.tmem_state_inp_stages,
                producer_group=cg_cg1,
                consumer_group=cg_mma,
                barrier_storage=storage.state_inp_ready_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # CG1 -> MMA warp: fixed-slot, ready-only input notifications.
        vks_ready_producer, vks_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg1,
            consumer_group=cg_mma,
            barrier_storage=storage.vks_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        nv_ready_producer, nv_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg1,
            consumer_group=cg_mma,
            barrier_storage=storage.nv_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        decay_v_ready_producer, decay_v_ready_consumer = (
            pipeline.PipelineAsyncUmma.create(
                num_stages=1,
                producer_group=cg_cg1,
                consumer_group=cg_mma,
                barrier_storage=storage.decay_v_ready_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # CG1 -> epilogue warp:  output_ready
        o_store_producer, o_store_consumer = pipeline.PipelineAsync.create(
            num_stages=self.smem_o_stages,
            producer_group=cg_cg1,
            consumer_group=cg_epi,
            barrier_storage=storage.o_store_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        pipeline_init_arrive(is_relaxed=True)

        pipeline_init_wait()

        # ------------------------------------------------------------------
        # 2. Warp specialization - each warp role owns its own scheduler loop
        # ------------------------------------------------------------------

        # ==============================================================
        # COMPUTE WARP GROUP 0 (warps 0-3)
        # ==============================================================
        if (
            warp_idx >= self.compute_group_0_warp_ids[0]
            and warp_idx <= self.compute_group_0_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.num_regs_compute_group_0)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)

            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()
            while work.is_valid_tile:
                batch_idx, head_idx, _ = work.tile_idx
                batch_start, batch_end = _get_safe_sequence_bounds(
                    cu_seqlens, batch_idx, mQ.shape[0]
                )
                seqlen_b = batch_end - batch_start
                num_pairs = 2
                num_pairs_b = cute.ceil_div(seqlen_b, self.b_t * num_pairs)

                sQk_pisl = self._transform_to_position_independent_layout(
                    sQk, qk_smem_layout_staged.inner
                )
                sAinv_pisl = self._transform_to_position_independent_layout(
                    sAinv, ainv_smem_layout_staged.inner
                )
                cg0_smem_args = (
                    sCumsumlog,
                    sBeta,
                    sAinv_pisl,
                    sAinvCal,
                    sQk_pisl,
                )
                # Runtime first/last predicates keep one pair body in SASS instead
                # of peeling a complete first-pair copy before the steady loop.
                for pair_idx in cutlass.range(num_pairs_b):
                    (
                        load_gate_consumer,
                        load_beta_consumer,
                        cg0_shared_acc_consumer,
                        a_inv_ready_producer,
                        qk_ready_producer,
                    ) = self.compute_group_0_pair(
                        tidx,
                        tmem_ptr,
                        scale,
                        (tiled_mma_qk,),
                        cg0_smem_args,
                        (
                            load_gate_consumer,
                            load_beta_consumer,
                            cg0_shared_acc_consumer,
                            a_inv_ready_producer,
                            qk_ready_producer,
                        ),
                        (pair_idx == 0, pair_idx < num_pairs_b - 1),
                    )

                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()
            a_inv_ready_producer.tail()
            qk_ready_producer.tail()

        # ==============================================================
        # COMPUTE WARP GROUP 1 (warps 4-7)
        # ==============================================================
        if (
            warp_idx >= self.compute_group_1_warp_ids[0]
            and warp_idx <= self.compute_group_1_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.num_regs_compute_group_1)
            # Total TMEM columns: state + q_state_acc + shared_acc + input staging.
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)

            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()
            while work.is_valid_tile:
                batch_idx, head_idx, _ = work.tile_idx
                batch_start, batch_end = _get_safe_sequence_bounds(
                    cu_seqlens, batch_idx, mQ.shape[0]
                )
                seqlen_b = batch_end - batch_start
                num_chunks_b = cute.ceil_div(seqlen_b, self.b_t)
                if num_chunks_b > 0:
                    kv_acc_producer = self._load_initial_state(
                        tidx,
                        mS_init,
                        mS_indices,
                        head_idx,
                        batch_idx,
                        tmem_ptr,
                        tiled_mma_kv,
                        kv_acc_producer,
                    )
                    sV_pisl = self._transform_to_position_independent_layout(
                        sV, v_smem_layout_staged.inner
                    )
                    sO_pisl = self._transform_to_position_independent_layout(
                        sO, o_smem_layout_staged.inner
                    )
                    num_pairs_b = cute.ceil_div(seqlen_b, self.b_t * 2)
                    num_chunks_padded = num_pairs_b * 2
                    for chunk_iter in cutlass.range(num_chunks_padded):
                        (
                            load_v_consumer,
                            load_gate_consumer,
                            cg1_shared_acc_consumer,
                            kv_acc_consumer,
                            q_state_acc_consumer,
                            kv_acc_producer,
                            state_inp_ready_producer,
                            vks_ready_producer,
                            nv_ready_producer,
                            decay_v_ready_producer,
                            o_store_producer,
                        ) = self.compute_group_1_chunk(
                            tidx,
                            tmem_ptr,
                            scale,
                            (tiled_mma_kv, tiled_mma_qs, tiled_mma_qkv),
                            (sV_pisl, sCumsumlog, sCumprod, sBeta, sO_pisl),
                            (
                                load_v_consumer,
                                load_gate_consumer,
                                cg1_shared_acc_consumer,
                                kv_acc_consumer,
                                q_state_acc_consumer,
                                kv_acc_producer,
                                state_inp_ready_producer,
                                vks_ready_producer,
                                nv_ready_producer,
                                decay_v_ready_producer,
                                o_store_producer,
                            ),
                            (
                                chunk_iter,
                                num_pairs_b,
                                head_idx,
                                seqlen_b,
                            ),
                        )
                    if cutlass.const_expr(self.store_final_state):
                        kv_acc_consumer = self._store_final_state(
                            tidx,
                            mS_out,
                            mS_indices,
                            head_idx,
                            batch_idx,
                            tmem_ptr,
                            tiled_mma_kv,
                            kv_acc_consumer,
                        )
                    else:
                        kv_acc_handle = kv_acc_consumer.wait_and_advance()
                        kv_acc_handle.release()
                elif cutlass.const_expr(self.store_final_state):
                    self._store_empty_final_state(
                        tidx,
                        mS_init,
                        mS_out,
                        mS_indices,
                        head_idx,
                        batch_idx,
                    )

                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

            o_store_producer.tail()
            state_inp_ready_producer.tail()

        # ==============================================================
        # CG0 MMA ISSUER (warp 8: KK/QK)
        # ==============================================================
        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()
            while work.is_valid_tile:
                batch_idx, _, _ = work.tile_idx
                batch_start, batch_end = _get_safe_sequence_bounds(
                    cu_seqlens, batch_idx, mQ.shape[0]
                )
                seqlen_b = batch_end - batch_start
                num_pairs_b = cute.ceil_div(seqlen_b, self.b_t * 2)
                for _pair_idx in cutlass.range(num_pairs_b):
                    (
                        cg0_shared_acc_producer,
                        load_k_consumer,
                        load_q_consumer,
                    ) = self.mma_cg0_pair(
                        tmem_ptr,
                        (tiled_mma_qk, tiled_mma_qkv),
                        (sQ, sK),
                        (
                            cg0_shared_acc_producer,
                            load_k_consumer,
                            load_q_consumer,
                        ),
                    )
                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

            cg0_shared_acc_producer.tail()

        # ==============================================================
        # CG1 MMA ISSUER (warp 10: KS/QS/NV/QKV/KV)
        # ==============================================================
        elif warp_idx == self.mma_cg1_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()
            while work.is_valid_tile:
                batch_idx, _, _ = work.tile_idx
                batch_start, batch_end = _get_safe_sequence_bounds(
                    cu_seqlens, batch_idx, mQ.shape[0]
                )
                seqlen_b = batch_end - batch_start
                num_chunks = cute.ceil_div(seqlen_b, self.b_t * 2) * 2

                for chunk_idx in cutlass.range(num_chunks):
                    is_first_chunk = chunk_idx == 0
                    (
                        cg1_shared_acc_producer,
                        q_state_acc_producer,
                        kv_acc_producer,
                        load_k_consumer,
                        load_q_consumer,
                        a_inv_ready_consumer,
                        qk_ready_consumer,
                        state_inp_ready_consumer,
                        vks_ready_consumer,
                        nv_ready_consumer,
                        decay_v_ready_consumer,
                    ) = self.mma_cg1_chunk(
                        tmem_ptr,
                        (
                            tiled_mma_qs,
                            tiled_mma_qkv,
                            tiled_mma_qkv_ss,
                            tiled_mma_kv,
                        ),
                        (sQ, sK, sK_trans, sV, sAinv, sQk),
                        (
                            cg1_shared_acc_producer,
                            q_state_acc_producer,
                            kv_acc_producer,
                            load_k_consumer,
                            load_q_consumer,
                            a_inv_ready_consumer,
                            qk_ready_consumer,
                            state_inp_ready_consumer,
                            vks_ready_consumer,
                            nv_ready_consumer,
                            decay_v_ready_consumer,
                        ),
                        is_first_chunk,
                    )

                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

            cg1_shared_acc_producer.tail()
            q_state_acc_producer.tail()
            kv_acc_producer.tail()

        # ==============================================================
        # TMA LOAD WARP (warp 9)
        # ==============================================================
        elif warp_idx == self.tma_qkv_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()

            # Init base descriptors once into GMEM (copies embedded atom descriptor)
            if work.is_valid_tile:
                tensormap_manager.init_tensormap_from_atom(
                    tma_q.atom, tensormap_q_ptr, self.tma_qkv_warp_id
                )
                tensormap_manager.init_tensormap_from_atom(
                    tma_k.atom, tensormap_k_ptr, self.tma_qkv_warp_id
                )
                tensormap_manager.init_tensormap_from_atom(
                    tma_v.atom, tensormap_v_ptr, self.tma_qkv_warp_id
                )
                tensormap_manager.fence_tensormap_initialization()

            while work.is_valid_tile:
                batch_idx, head_idx, _ = work.tile_idx
                batch_start, batch_end = _get_safe_sequence_bounds(
                    cu_seqlens, batch_idx, mQ.shape[0]
                )
                seqlen_b = batch_end - batch_start
                num_pairs = 2
                num_pairs_b = cute.ceil_div(seqlen_b, self.b_t * num_pairs)
                num_chunks_b = num_pairs_b * num_pairs

                # All warp roles skip empty workloads at the same granularity.
                if num_chunks_b > 0:
                    # Bounded descriptors zero-fill partial and padded chunks.
                    bounded_q = cute.make_tensor(
                        mQ.iterator,
                        cute.make_layout(
                            (batch_end, mQ.shape[1], mQ.shape[2]),
                            stride=(mQ.stride[0], mQ.stride[1], mQ.stride[2]),
                        ),
                    )
                    bounded_k = cute.make_tensor(
                        mK.iterator,
                        cute.make_layout(
                            (batch_end, mK.shape[1], mK.shape[2]),
                            stride=(mK.stride[0], mK.stride[1], mK.stride[2]),
                        ),
                    )
                    bounded_v = cute.make_tensor(
                        mV.iterator,
                        cute.make_layout(
                            (mV.shape[0], batch_end, mV.shape[2]),
                            stride=(mV.stride[0], mV.stride[1], mV.stride[2]),
                        ),
                    )
                    tensormap_manager.update_tensormap(
                        (bounded_q, bounded_k, bounded_v),
                        (tma_q.atom, tma_k.atom, tma_v.atom),
                        (tensormap_q_ptr, tensormap_k_ptr, tensormap_v_ptr),
                        self.tma_qkv_warp_id,
                        (None, None, None),
                    )

                    num_valid_chunks_b = cute.ceil_div(seqlen_b, self.b_t)
                    for chunk_idx in cutlass.range(num_valid_chunks_b - 1):
                        chunk_offset = batch_start + chunk_idx * self.b_t
                        load_q_producer, load_k_producer, load_v_producer = (
                            self.tma_qkv_warp(
                                (tiled_mma_qk, tiled_mma_qkv_ss, tiled_mma_kv),
                                (tma_q, tma_k, tma_v),
                                (sQ, sK, sV),
                                (load_q_producer, load_k_producer, load_v_producer),
                                (chunk_offset, chunk_idx, batch_idx, head_idx),
                                (
                                    tensormap_manager,
                                    tensormap_q_ptr,
                                    tensormap_k_ptr,
                                    tensormap_v_ptr,
                                ),
                            )
                        )
                    chunk_idx = num_valid_chunks_b - 1
                    chunk_offset = batch_start + chunk_idx * self.b_t
                    load_q_producer, load_k_producer, load_v_producer = (
                        self.tma_qkv_warp(
                            (tiled_mma_qk, tiled_mma_qkv_ss, tiled_mma_kv),
                            (tma_q, tma_k, tma_v),
                            (sQ, sK, sV),
                            (load_q_producer, load_k_producer, load_v_producer),
                            (chunk_offset, chunk_idx, batch_idx, head_idx),
                            (
                                tensormap_manager,
                                tensormap_q_ptr,
                                tensormap_k_ptr,
                                tensormap_v_ptr,
                            ),
                        )
                    )
                    for chunk_idx in cutlass.range(num_valid_chunks_b, num_chunks_b):
                        chunk_offset = batch_start + chunk_idx * self.b_t
                        load_q_producer, load_k_producer, load_v_producer = (
                            self.tma_qkv_warp(
                                (tiled_mma_qk, tiled_mma_qkv_ss, tiled_mma_kv),
                                (tma_q, tma_k, tma_v),
                                (sQ, sK, sV),
                                (load_q_producer, load_k_producer, load_v_producer),
                                (chunk_offset, chunk_idx, batch_idx, head_idx),
                                (
                                    tensormap_manager,
                                    tensormap_q_ptr,
                                    tensormap_k_ptr,
                                    tensormap_v_ptr,
                                ),
                            )
                        )
                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

            load_q_producer.tail()
            load_k_producer.tail()
            load_v_producer.tail()

        # ==============================================================
        # EPILOGUE WARP (warp 11)
        # ==============================================================
        if warp_idx == self.epilogue_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()

            # Init O descriptor once into GMEM
            if work.is_valid_tile:
                tensormap_manager.init_tensormap_from_atom(
                    tma_o.atom, tensormap_o_ptr, self.epilogue_warp_id
                )
                tensormap_manager.fence_tensormap_initialization()

            while work.is_valid_tile:
                batch_idx, head_idx, _ = work.tile_idx
                batch_start, batch_end = _get_safe_sequence_bounds(
                    cu_seqlens, batch_idx, mQ.shape[0]
                )
                seqlen_b = batch_end - batch_start
                num_pairs = 2
                num_pairs_b = cute.ceil_div(seqlen_b, self.b_t * num_pairs)
                num_chunks_b = num_pairs_b * num_pairs

                # Keep the fixed prefetch and zero-trip store loop under one
                # guard. The wider scope lets the compiler share the dynamic
                # chunk predicate without introducing divergent loop state.
                if num_chunks_b > 0:
                    num_valid_chunks_b = cute.ceil_div(seqlen_b, self.b_t)
                    bounded_o = cute.make_tensor(
                        mO.iterator,
                        cute.make_layout(
                            (mO.shape[0], batch_end, mO.shape[2]),
                            stride=(mO.stride[0], mO.stride[1], mO.stride[2]),
                        ),
                    )
                    tensormap_manager.update_tensormap(
                        (bounded_o,),
                        (tma_o.atom,),
                        (tensormap_o_ptr,),
                        self.epilogue_warp_id,
                        (None,),
                    )
                    tensormap_manager.fence_tensormap_update(tensormap_o_ptr)

                    for prefetch_idx in range(2):
                        prefetch_offset = batch_start + prefetch_idx * self.b_t
                        is_last_tile = prefetch_idx >= num_valid_chunks_b - 1
                        load_gate_producer, load_beta_producer = (
                            self.load_gate_beta_warp(
                                tidx,
                                (mA, mB, mA_log, mDt_bias),
                                (sCumsumlog, sCumprod, sBeta),
                                (load_gate_producer, load_beta_producer),
                                (
                                    prefetch_offset,
                                    head_idx,
                                    is_last_tile,
                                    batch_end,
                                ),
                            )
                        )

                    # Fill the remaining two lookahead stages together. Since the
                    # chunk count is even, chunks 2 and 3 are either both present
                    # or both absent.
                    if num_chunks_b > 2:
                        for prefetch_idx in range(2, 4):
                            prefetch_offset = batch_start + prefetch_idx * self.b_t
                            is_last_tile = prefetch_idx >= num_valid_chunks_b - 1
                            load_gate_producer, load_beta_producer = (
                                self.load_gate_beta_warp(
                                    tidx,
                                    (mA, mB, mA_log, mDt_bias),
                                    (sCumsumlog, sCumprod, sBeta),
                                    (load_gate_producer, load_beta_producer),
                                    (
                                        prefetch_offset,
                                        head_idx,
                                        is_last_tile,
                                        batch_end,
                                    ),
                                )
                            )

                    for chunk_idx in cutlass.range(num_chunks_b):
                        prefetch_idx = chunk_idx + 4
                        if prefetch_idx < num_chunks_b:
                            prefetch_offset = batch_start + prefetch_idx * self.b_t
                            is_last_tile = prefetch_idx >= num_valid_chunks_b - 1
                            load_gate_producer, load_beta_producer = (
                                self.load_gate_beta_warp(
                                    tidx,
                                    (mA, mB, mA_log, mDt_bias),
                                    (sCumsumlog, sCumprod, sBeta),
                                    (load_gate_producer, load_beta_producer),
                                    (
                                        prefetch_offset,
                                        head_idx,
                                        is_last_tile,
                                        batch_end,
                                    ),
                                )
                            )
                        chunk_offset = batch_start + chunk_idx * self.b_t
                        o_store_consumer = self.epilogue_warp(
                            (sO,),
                            (tma_o,),
                            (o_store_consumer,),
                            (head_idx, chunk_offset),
                            (tensormap_manager, tensormap_o_ptr),
                        )
                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

            load_gate_producer.tail()
            load_beta_producer.tail()

    # -----------------------------------------------------------------------
    # Per-warp methods  (called from kernel's chunk loop)
    # -----------------------------------------------------------------------
    @cute.jit
    def tma_qkv_warp(
        self,
        mma_args: tuple,
        tma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
        tensormap_args: tuple,
    ) -> tuple[
        pipeline.PipelineProducer, pipeline.PipelineProducer, pipeline.PipelineProducer
    ]:
        """Warp 9: load Q, K (double-buffered), V for the current chunk.

        TMA load pattern:
          1. domain_offset the TMA tensor to (chunk_offset, head_idx, 0) so that
             the logical tile (0, ...) maps to the current chunk.
          2. flat_divide to obtain the tiled global view.
          3. thr_mma.partition_{A,B} to get the TMA-compatible per-thread view.
          4. cpasync.tma_partition -> (tXsX, tXgX) SMEM/global pairs.
          5. acquire pipeline stage, issue cute.copy, signal mbarrier.

        Note on head coordinate: head_idx is the flat KV-head index in [0, h_qv).
        For the hierarchical head layout (h_r, h_qv) with h_r having stride 0
        (broadcast), the flat index maps correctly as long as head_idx < h_qv.
        """
        tiled_mma_qk, tiled_mma_qkv_ss, tiled_mma_kv = mma_args
        tma_q, tma_k, tma_v = tma_args
        sQ, sK, sV = smem_args
        load_q_producer, load_k_producer, load_v_producer = pipeline_args
        chunk_offset, chunk_idx, batch_idx, head_idx = work_args
        tensormap_manager, tensormap_q_ptr, tensormap_k_ptr, tensormap_v_ptr = (
            tensormap_args
        )

        # Single-CTA mode: no multicast, cta_v = 0.
        cta_layout = cute.make_layout(1)

        # Per-thread MMA slices (cta_v=0 for ONE-CTA mode).
        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_qkv_ss = tiled_mma_qkv_ss.get_slice(0)

        # Tile shapes from the MMA tiler (128, 128, 128):
        #   mode[0,2] = (BT, DK) - M,K tile for A (Q) and for B (K) of tiled_mma_qk
        #   mode[1,2] = (BT, DV) - tile shape for loading V (B operand in GEMMs 5/6)
        # (BT, DK)
        qk_tile = cute.select(self.mma_tiler_qk, mode=[0, 2])
        # (DV, BT)
        v_tile = cute.select(self.mma_tiler_qkv, mode=[0, 2])

        # ------------------------------------------------------------------
        # K  (B operand of GEMM-kk / GEMM-qk, double-buffered)
        # Tensor shape: (total_tokens, H_hier, DK)
        # TMA tile:     (BT, DK)
        # ------------------------------------------------------------------
        mK = cute.domain_offset(
            (chunk_offset, cutlass.Int32(0)), tma_k.tma_tensor[None, None, head_idx]
        )
        # (..., num_k_tiles, ...)
        gK = cute.flat_divide(mK, qk_tile)
        tCgK = thr_mma_qk.partition_B(gK)
        tKsK, tKgK = cpasync.tma_partition(
            tma_k.atom,
            0,
            cta_layout,
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tCgK, 0, 3),
        )

        # Load K for the current chunk into the next available pipeline stage.
        k_handle = load_k_producer.acquire_and_advance()
        if chunk_idx == 0:
            tensormap_manager.fence_tensormap_update(tensormap_k_ptr)

        cute.copy(
            tma_k.atom,
            tKgK[(None, 0, 0)],
            tKsK[(None, k_handle.index)],
            tma_bar_ptr=k_handle.barrier,
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                tensormap_k_ptr, cute.AddressSpace.generic
            ),
        )

        # ------------------------------------------------------------------
        # Q  (A operand of GEMM-qk, single-buffered)
        # ------------------------------------------------------------------
        mQ = cute.domain_offset(
            (chunk_offset, cutlass.Int32(0)), tma_q.tma_tensor[None, None, head_idx]
        )
        gQ = cute.flat_divide(mQ, qk_tile)
        tCgQ = thr_mma_qk.partition_A(gQ)
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_q.atom,
            0,
            cta_layout,
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tCgQ, 0, 3),
        )

        q_handle = load_q_producer.acquire_and_advance()
        if chunk_idx == 0:
            tensormap_manager.fence_tensormap_update(tensormap_q_ptr)
        cute.copy(
            tma_q.atom,
            tQgQ[(None, 0, 0)],
            tQsQ[(None, q_handle.index)],
            tma_bar_ptr=q_handle.barrier,
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                tensormap_q_ptr, cute.AddressSpace.generic
            ),
        )

        # ------------------------------------------------------------------
        # V  (B operand of GEMM-new_v / GEMM-qkv, single-buffered)
        # ------------------------------------------------------------------
        mV = cute.domain_offset(
            (cutlass.Int32(0), chunk_offset), tma_v.tma_tensor[None, None, head_idx]
        )
        gV = cute.flat_divide(mV, v_tile)
        tCgV = thr_mma_qkv_ss.partition_A(gV)
        tVsV, tVgV = cpasync.tma_partition(
            tma_v.atom,
            0,
            cta_layout,
            cute.group_modes(sV, 0, 3),
            cute.group_modes(tCgV, 0, 3),
        )

        v_handle = load_v_producer.acquire_and_advance()
        if chunk_idx == 0:
            tensormap_manager.fence_tensormap_update(tensormap_v_ptr)
        cute.copy(
            tma_v.atom,
            tVgV[(None, 0, 0)],
            tVsV[(None, v_handle.index)],
            tma_bar_ptr=v_handle.barrier,
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                tensormap_v_ptr, cute.AddressSpace.generic
            ),
        )

        return load_q_producer, load_k_producer, load_v_producer

    @cute.jit
    def load_gate_beta_warp(
        self,
        tidx: cutlass.Int32,
        gmem_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
    ) -> tuple[pipeline.PipelineProducer, pipeline.PipelineProducer]:
        """Epilogue warp 11: prepare log2(gate) and beta for one chunk.

        Raw a/b values are loaded to registers. Gate and beta are computed there,
        then stored to the staged SMEM buffers consumed by CG0 and CG1.

        The last tile uses predicated loads. Out-of-bounds elements receive neutral
        values (log2(gate)=0, beta=0).

        Thread tidx (lane 0..31) owns positions tidx and tidx+32.
        """
        a, b, A_log, dt_bias = gmem_args
        sCumsumlog, sCumprod, sBeta = smem_args
        load_gate_producer, load_beta_producer = pipeline_args
        chunk_offset, head_idx, is_last_tile, batch_end = work_args

        # lane index
        lidx = tidx % self.threads_per_warp

        gA = cute.domain_offset((chunk_offset,), a[None, head_idx])
        cA = cute.domain_offset(
            (chunk_offset,), cute.make_identity_tensor(a[None, head_idx].shape)
        )
        gB = cute.domain_offset((chunk_offset,), b[None, head_idx])
        gA = cute.flat_divide(gA, (self.b_t,))[None, 0]
        cA = cute.flat_divide(cA, (self.b_t,))[None, 0]
        gB = cute.flat_divide(gB, (self.b_t,))[None, 0]

        # Each lane owns two elements in the 64-token tile.
        thread_layout = cute.make_layout((self.threads_per_warp,), stride=(1,))
        value_layout = cute.make_layout((1,), stride=(1,))

        atom_raw_g2r = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), a.element_type, num_bits_per_copy=16
        )
        tiled_copy_raw_g2r = cute.make_tiled_copy_tv(
            atom_raw_g2r, thread_layout, value_layout
        )
        atom_f32 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=32
        )
        tiled_copy_f32 = cute.make_tiled_copy_tv(
            atom_f32, thread_layout, value_layout
        )

        thr_copy_raw_g2r = tiled_copy_raw_g2r.get_slice(lidx)
        tAgA = thr_copy_raw_g2r.partition_S(gA)
        tBgB = thr_copy_raw_g2r.partition_S(gB)
        thr_copy_f32 = tiled_copy_f32.get_slice(lidx)
        tFsCumsumlog = thr_copy_f32.partition_D(sCumsumlog)
        tFsCumprod = thr_copy_f32.partition_D(sCumprod)
        tFsBeta = thr_copy_f32.partition_D(sBeta)

        rA = cute.make_rmem_tensor_like(tAgA, a.element_type)
        rB = cute.make_rmem_tensor_like(tBgB, b.element_type)
        rGate = cute.make_rmem_tensor_like(tAgA, cutlass.Float32)
        rCumprod = cute.make_rmem_tensor_like(tAgA, cutlass.Float32)
        rBeta = cute.make_rmem_tensor_like(tBgB, cutlass.Float32)
        tArA = tiled_copy_raw_g2r.retile(rA)
        tBrB = tiled_copy_raw_g2r.retile(rB)
        tFrGate = tiled_copy_f32.retile(rGate)
        tFrCumprod = tiled_copy_f32.retile(rCumprod)
        tFrBeta = tiled_copy_f32.retile(rBeta)

        # Reuse the tail predicate for a and b.
        tAcA = thr_copy_raw_g2r.partition_S(cA)
        tApA = cute.make_rmem_tensor(
            ((tAcA.shape[0][1],), tAcA.shape[1]), cutlass.Boolean
        )
        if is_last_tile:
            for i in range(cute.size(tApA)):
                tApA[i] = cute.elem_less(tAcA[i][0], batch_end)

        if is_last_tile:
            tArA.fill(0.0)
            tBrB.fill(0.0)
            cute.copy(tiled_copy_raw_g2r, tAgA, tArA, pred=tApA)
            cute.copy(tiled_copy_raw_g2r, tBgB, tBrB, pred=tApA)
        else:
            cute.copy(tiled_copy_raw_g2r, tAgA, tArA)
            cute.copy(tiled_copy_raw_g2r, tBgB, tBrB)

        neg_exp_A_log_log2e = (
            -cute.math.exp(cutlass.Float32(A_log[head_idx]))
            * cutlass.Float32(1.4426950408889634)
        )
        dt_bias_f32 = cutlass.Float32(dt_bias[head_idx])
        for i in range(cute.size(tFrGate)):
            x = cutlass.Float32(tArA[i]) + dt_bias_f32
            softplus_x = x
            if x <= cutlass.Float32(20.0):
                softplus_x = cute.math.log(
                    cutlass.Float32(1.0) + cute.math.exp(x)
                )
            tFrGate[i] = neg_exp_A_log_log2e * softplus_x
            tFrBeta[i] = cutlass.Float32(1.0) / (
                cutlass.Float32(1.0) + cute.math.exp(-cutlass.Float32(tBrB[i]))
            )
            if is_last_tile and not tApA[i]:
                tFrGate[i] = cutlass.Float32(0.0)
                tFrBeta[i] = cutlass.Float32(0.0)

        for offset in [1, 2, 4, 8, 16]:
            for col in range(cute.size(tFrGate)):
                n = cute.arch.shuffle_sync_up(
                    tFrGate[col], offset, mask=0xFFFFFFFF, mask_and_clamp=0
                )
                if lidx >= offset:
                    tFrGate[col] = tFrGate[col] + n
        for col in range(1, cute.size(tFrGate)):
            last_v = cute.arch.shuffle_sync(
                tFrGate[col - 1],
                self.threads_per_warp - 1,
                mask=0xFFFFFFFF,
                mask_and_clamp=self.threads_per_warp - 1,
            )
            tFrGate[col] += last_v
        for col in range(cute.size(tFrGate)):
            tFrCumprod[col] = cute.math.exp2(tFrGate[col], fastmath=True)

        # Keep the gate stage available while the register-only prefix scan is
        # running.  The stage is needed only when its SMEM writes begin.
        gate_handle = load_gate_producer.acquire_and_advance()
        cute.copy(
            tiled_copy_f32,
            tFrGate,
            tFsCumsumlog[None, None, 0, gate_handle.index],
        )
        cute.copy(
            tiled_copy_f32,
            tFrCumprod,
            tFsCumprod[None, None, 0, gate_handle.index],
        )

        gate_handle.commit()

        beta_handle = load_beta_producer.acquire_and_advance()
        cute.copy(
            tiled_copy_f32,
            tFrBeta,
            tFsBeta[None, None, 0, beta_handle.index],
        )
        beta_handle.commit()

        return load_gate_producer, load_beta_producer

    @cute.jit
    def mma_cg0_pair(
        self,
        tmem_ptr: cutlass.Int64,
        mma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
    ) -> tuple[
        pipeline.PipelineProducer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
    ]:
        """Issue one pair's fixed KK0/KK1/QK0/QK1 sequence."""
        tiled_mma_qk, tiled_mma_qkv = mma_args
        sQ, sK = smem_args
        (
            cg0_acc_producer,
            load_k_consumer,
            load_q_consumer,
        ) = pipeline_args

        acc_shape = tiled_mma_qkv.partition_shape_C(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        tCtAcc_fake = tiled_mma_qkv.make_fragment_C(
            cute.append(acc_shape, self.tmem_cg0_shared_acc_stages)
        )
        tCtAcc = cute.make_tensor(
            tmem_ptr + self.tmem_cg0_shared_acc_offset, tCtAcc_fake.layout
        )
        tCrK_A = tiled_mma_qk.make_fragment_A(sK)
        tCrK_B = tiled_mma_qk.make_fragment_B(sK)
        tCrQ_A = tiled_mma_qk.make_fragment_A(sQ)
        num_kphases = cute.size(tCrK_A, mode=[2])

        kk0_handle = cg0_acc_producer.acquire_and_advance()
        k0_handle = load_k_consumer.wait_and_advance()
        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                tiled_mma_qk,
                tCtAcc[None, None, None, kk0_handle.index],
                tCrK_A[None, None, kphase_idx, k0_handle.index],
                tCrK_B[None, None, kphase_idx, k0_handle.index],
                tCtAcc[None, None, None, kk0_handle.index],
            )
        kk0_handle.commit()

        kk1_handle = cg0_acc_producer.acquire_and_advance()
        k1_handle = load_k_consumer.wait_and_advance()
        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                tiled_mma_qk,
                tCtAcc[None, None, None, kk1_handle.index],
                tCrK_A[None, None, kphase_idx, k1_handle.index],
                tCrK_B[None, None, kphase_idx, k1_handle.index],
                tCtAcc[None, None, None, kk1_handle.index],
            )
        kk1_handle.commit()

        q0_handle = load_q_consumer.wait_and_advance()
        qk0_handle = cg0_acc_producer.acquire_and_advance()
        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                tiled_mma_qk,
                tCtAcc[None, None, None, qk0_handle.index],
                tCrQ_A[None, None, kphase_idx, q0_handle.index],
                tCrK_B[None, None, kphase_idx, k0_handle.index],
                tCtAcc[None, None, None, qk0_handle.index],
            )
        qk0_handle.commit()

        q1_handle = load_q_consumer.wait_and_advance()
        qk1_handle = cg0_acc_producer.acquire_and_advance()
        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                tiled_mma_qk,
                tCtAcc[None, None, None, qk1_handle.index],
                tCrQ_A[None, None, kphase_idx, q1_handle.index],
                tCrK_B[None, None, kphase_idx, k1_handle.index],
                tCtAcc[None, None, None, qk1_handle.index],
            )
        qk1_handle.commit()

        q0_handle.release()
        q1_handle.release()
        k0_handle.release()
        k1_handle.release()

        return (
            cg0_acc_producer,
            load_k_consumer,
            load_q_consumer,
        )

    @cute.jit
    def mma_cg1_chunk(
        self,
        tmem_ptr: cutlass.Int64,
        mma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
        is_first_chunk: cutlass.Boolean,
    ) -> tuple:
        """Issue KS/QS/NV/QKV/KV for one chunk."""
        tiled_mma_qs, tiled_mma_qkv, _, tiled_mma_kv = mma_args
        sQ, sK, sK_trans, _, sAinv, sQk = smem_args
        (
            cg1_acc_producer,
            q_state_acc_producer,
            kv_acc_producer,
            load_k_consumer,
            load_q_consumer,
            a_inv_ready_consumer,
            qk_ready_consumer,
            state_inp_ready_consumer,
            vks_ready_consumer,
            nv_ready_consumer,
            decay_v_ready_consumer,
        ) = pipeline_args

        acc_shape = tiled_mma_qkv.partition_shape_C(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        tCtShared_fake = tiled_mma_qkv.make_fragment_C(
            cute.append(acc_shape, self.tmem_cg1_shared_acc_stages)
        )
        tCtShared = cute.make_tensor(
            tmem_ptr + self.tmem_cg1_shared_acc_offset, tCtShared_fake.layout
        )

        shared_inp_shape = tiled_mma_qkv.partition_shape_A(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[2])
        )
        tCtSharedInp_fake = tiled_mma_qkv.make_fragment_A(
            cute.append(shared_inp_shape, self.tmem_shared_inp_stages)
        )
        tCtSharedInp = cute.make_tensor(
            cute.recast_ptr(
                tmem_ptr + self.tmem_shared_inp_offset, dtype=self.io_dtype
            ),
            tCtSharedInp_fake.layout,
        )

        qs_acc_shape = tiled_mma_qs.partition_shape_C(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[1])
        )
        tCtQState_fake = tiled_mma_qs.make_fragment_C(
            cute.append(qs_acc_shape, self.tmem_q_state_acc_stages)
        )
        tCtQState = cute.make_tensor(
            tmem_ptr + self.tmem_q_state_offset, tCtQState_fake.layout
        )

        state_acc_shape = tiled_mma_kv.partition_shape_C(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        tCtState_fake = tiled_mma_kv.make_fragment_C(
            cute.append(state_acc_shape, self.tmem_kv_acc_stages)
        )
        tCtState = cute.make_tensor(
            tmem_ptr + self.tmem_state_offset, tCtState_fake.layout
        )

        state_inp_shape = tiled_mma_qs.partition_shape_A(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[2])
        )
        tCtStateInp_fake = tiled_mma_qs.make_fragment_A(
            cute.append(state_inp_shape, self.tmem_state_inp_stages)
        )
        tCtStateInp = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_state_inp_offset, dtype=self.io_dtype),
            tCtStateInp_fake.layout,
        )

        tCrK_B_qs = tiled_mma_qs.make_fragment_B(sK)
        tCrQ_B_qs = tiled_mma_qs.make_fragment_B(sQ)
        tCrAinv_B = tiled_mma_qkv.make_fragment_B(sAinv)
        tCrNv_B = tiled_mma_qkv.make_fragment_B(sQk)
        tCrKt_B = tiled_mma_kv.make_fragment_B(sK_trans)
        num_kphases_qs = cute.size(tCtStateInp, mode=[2])
        num_kphases_qkv = cute.size(tCrAinv_B, mode=[2])
        num_kphases_kv = cute.size(tCrKt_B, mode=[2])

        k_handle = load_k_consumer.wait_and_advance()
        q_handle = load_q_consumer.wait_and_advance()
        ks_handle = cg1_acc_producer.acquire_and_advance()
        state_handle = state_inp_ready_consumer.wait_and_advance()
        for kphase_idx in cutlass.range(num_kphases_qs, unroll_full=True):
            tiled_mma_qs.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                tiled_mma_qs,
                tCtShared[None, None, None, ks_handle.index],
                tCtStateInp[None, None, kphase_idx, state_handle.index],
                tCrK_B_qs[None, None, kphase_idx, k_handle.index],
                tCtShared[None, None, None, ks_handle.index],
            )
        ks_handle.commit()

        qs_handle = q_state_acc_producer.acquire_and_advance()
        for kphase_idx in cutlass.range(num_kphases_qs, unroll_full=True):
            tiled_mma_qs.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                tiled_mma_qs,
                tCtQState[None, None, None, qs_handle.index],
                tCtStateInp[None, None, kphase_idx, state_handle.index],
                tCrQ_B_qs[None, None, kphase_idx, q_handle.index],
                tCtQState[None, None, None, qs_handle.index],
            )
        qs_handle.commit()
        state_handle.release()

        q_handle.release()

        nv_handle = cg1_acc_producer.acquire_and_advance()
        vks_ready_consumer.wait_and_advance()
        ainv_handle = a_inv_ready_consumer.wait_and_advance()
        for kphase_idx in cutlass.range(num_kphases_qkv, unroll_full=True):
            tiled_mma_qkv.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                tiled_mma_qkv,
                tCtShared[None, None, None, nv_handle.index],
                tCtSharedInp[None, None, kphase_idx, 0],
                tCrAinv_B[None, None, kphase_idx, ainv_handle.index],
                tCtShared[None, None, None, nv_handle.index],
            )
        nv_handle.commit()
        ainv_handle.release()

        q_state_handle = q_state_acc_producer.acquire_and_advance()
        qk_handle = qk_ready_consumer.wait_and_advance()
        nv_ready_consumer.wait_and_advance()
        for kphase_idx in cutlass.range(num_kphases_qkv, unroll_full=True):
            tiled_mma_qkv.set(tcgen05.Field.ACCUMULATE, True)
            cute.gemm(
                tiled_mma_qkv,
                tCtQState[None, None, None, q_state_handle.index],
                tCtSharedInp[None, None, kphase_idx, 0],
                tCrNv_B[None, None, kphase_idx, qk_handle.index],
                tCtQState[None, None, None, q_state_handle.index],
            )
        qk_handle.release()
        q_state_handle.commit()

        if is_first_chunk:
            kv_acc_producer.advance()
        kv_handle = kv_acc_producer.acquire_and_advance()
        decay_v_ready_consumer.wait_and_advance()
        for kphase_idx in cutlass.range(num_kphases_kv, unroll_full=True):
            tiled_mma_kv.set(tcgen05.Field.ACCUMULATE, True)
            cute.gemm(
                tiled_mma_kv,
                tCtState[None, None, None, kv_handle.index],
                tCtSharedInp[None, None, kphase_idx, 1],
                tCrKt_B[None, None, kphase_idx, k_handle.index],
                tCtState[None, None, None, kv_handle.index],
            )
        kv_handle.commit()
        k_handle.release()

        return (
            cg1_acc_producer,
            q_state_acc_producer,
            kv_acc_producer,
            load_k_consumer,
            load_q_consumer,
            a_inv_ready_consumer,
            qk_ready_consumer,
            state_inp_ready_consumer,
            vks_ready_consumer,
            nv_ready_consumer,
            decay_v_ready_consumer,
        )

    @cute.jit
    def mma_issuer_warp(
        self,
        tmem_ptr: cutlass.Int64,
        scheduler_params: GDNTileSchedulerParams,
        block_coord: tuple,
        grid_dim: tuple,
        cu_seqlens: cute.Tensor,
        total_tokens: cutlass.Int32,
        mma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
    ):
        """Run the MMA issuer over its scheduler-owned work stream."""
        (
            cg0_shared_acc_producer,
            cg1_shared_acc_producer,
            q_state_acc_producer,
            kv_acc_producer,
            load_k_consumer,
            load_q_consumer,
            load_v_consumer,
            a_inv_ready_consumer,
            qk_ready_consumer,
            state_inp_ready_consumer,
            vks_ready_consumer,
            nv_ready_consumer,
            decay_v_ready_consumer,
        ) = pipeline_args
        load_k_releaser = load_k_consumer.clone()
        load_q_releaser = load_q_consumer.clone()
        scheduler = GDNTileScheduler.create(scheduler_params, block_coord, grid_dim)
        work = scheduler.initial_work_tile_info()

        while work.is_valid_tile:
            batch_idx, _, _ = work.tile_idx
            batch_start, batch_end = _get_safe_sequence_bounds(
                cu_seqlens, batch_idx, total_tokens
            )
            seqlen_b = batch_end - batch_start
            num_pairs_b = cute.ceil_div(seqlen_b, self.b_t * 2)
            num_chunks_padded = num_pairs_b * 2

            for chunk_iter in cutlass.range(num_chunks_padded):
                loop_is_first_chunk = chunk_iter == 0
                (
                    cg0_shared_acc_producer,
                    cg1_shared_acc_producer,
                    q_state_acc_producer,
                    kv_acc_producer,
                    load_k_consumer,
                    load_k_releaser,
                    load_q_consumer,
                    load_q_releaser,
                    load_v_consumer,
                    a_inv_ready_consumer,
                    qk_ready_consumer,
                    state_inp_ready_consumer,
                    vks_ready_consumer,
                    nv_ready_consumer,
                    decay_v_ready_consumer,
                ) = self.mma_warp_chunk(
                    tmem_ptr,
                    mma_args,
                    smem_args,
                    (
                        cg0_shared_acc_producer,
                        cg1_shared_acc_producer,
                        q_state_acc_producer,
                        kv_acc_producer,
                        load_k_consumer,
                        load_k_releaser,
                        load_q_consumer,
                        load_q_releaser,
                        load_v_consumer,
                        a_inv_ready_consumer,
                        qk_ready_consumer,
                        state_inp_ready_consumer,
                        vks_ready_consumer,
                        nv_ready_consumer,
                        decay_v_ready_consumer,
                    ),
                    (chunk_iter, num_pairs_b, loop_is_first_chunk),
                )

            scheduler.advance_to_next_work()
            work = scheduler.get_current_work()

        cg0_shared_acc_producer.tail()
        cg1_shared_acc_producer.tail()
        q_state_acc_producer.tail()
        kv_acc_producer.tail()

    @cute.jit
    def mma_warp_chunk(
        self,
        tmem_ptr: cutlass.Int64,
        mma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
    ) -> tuple[
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
    ]:
        """Warp 8: process one chunk from the caller-owned chunk stream.

        The next KK0/KK1 are issued after current NV0 so their MMA latency
        overlaps current QKV0/KV0 and chunk-1 work. CG0 and CG1 accumulators
        use disjoint two-stage rings, so the lookahead cannot alias KS/NV.
        """
        tiled_mma_qk, tiled_mma_qs, tiled_mma_qkv, _, tiled_mma_kv = mma_args
        sQ, sK, sK_trans, _, sAinv, sQk = smem_args
        (
            cg0_shared_acc_producer,
            cg1_shared_acc_producer,
            q_state_acc_producer,
            kv_acc_producer,
            load_k_consumer,
            load_k_releaser,
            load_q_consumer,
            load_q_releaser,
            load_v_consumer,
            a_inv_ready_consumer,
            qk_ready_consumer,
            state_inp_ready_consumer,
            vks_ready_consumer,
            nv_ready_consumer,
            decay_v_ready_consumer,
        ) = pipeline_args
        chunk_iter, num_pairs_b, is_first_chunk = work_args

        # ------------------------------------------------------------------
        # Build TMEM accumulator views  (identical to mma_warp)
        # ------------------------------------------------------------------
        acc_shape = tiled_mma_qkv.partition_shape_C(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        tCtCg0Shared_fake = tiled_mma_qkv.make_fragment_C(
            cute.append(acc_shape, self.tmem_cg0_shared_acc_stages)
        )
        tCtCg0Shared = cute.make_tensor(
            tmem_ptr + self.tmem_cg0_shared_acc_offset, tCtCg0Shared_fake.layout
        )
        tCtCg1Shared_fake = tiled_mma_qkv.make_fragment_C(
            cute.append(acc_shape, self.tmem_cg1_shared_acc_stages)
        )
        tCtCg1Shared = cute.make_tensor(
            tmem_ptr + self.tmem_cg1_shared_acc_offset, tCtCg1Shared_fake.layout
        )

        shared_inp_shape = tiled_mma_qkv.partition_shape_A(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[2])
        )
        tCtShared_inp_fake = tiled_mma_qkv.make_fragment_A(
            cute.append(shared_inp_shape, self.tmem_shared_inp_stages)
        )
        tCtShared_inp = cute.make_tensor(
            cute.recast_ptr(
                tmem_ptr + self.tmem_shared_inp_offset, dtype=self.io_dtype
            ),
            tCtShared_inp_fake.layout,
        )

        qs_acc_shape = tiled_mma_qs.partition_shape_C(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[1])
        )
        tCtQState_fake = tiled_mma_qs.make_fragment_C(
            cute.append(qs_acc_shape, self.tmem_q_state_acc_stages)
        )
        tCtQState = cute.make_tensor(
            tmem_ptr + self.tmem_q_state_offset, tCtQState_fake.layout
        )

        state_acc_shape = tiled_mma_kv.partition_shape_C(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        tCtState_fake = tiled_mma_kv.make_fragment_C(
            cute.append(state_acc_shape, self.tmem_kv_acc_stages)
        )
        tCtState = cute.make_tensor(
            tmem_ptr + self.tmem_state_offset, tCtState_fake.layout
        )

        state_inp_shape = tiled_mma_qs.partition_shape_A(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[2])
        )
        tCtState_inp_fake = tiled_mma_qs.make_fragment_A(
            cute.append(state_inp_shape, self.tmem_state_inp_stages)
        )
        tCtState_inp = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_state_inp_offset, dtype=self.io_dtype),
            tCtState_inp_fake.layout,
        )

        # ------------------------------------------------------------------
        # Pre-create operand fragments
        # ------------------------------------------------------------------
        tCrK_A = tiled_mma_qk.make_fragment_A(sK)
        tCrK_B = tiled_mma_qk.make_fragment_B(sK)
        tCrQ_A = tiled_mma_qk.make_fragment_A(sQ)
        tCrS_A = tCtState_inp
        tCrQ_B_qs = tiled_mma_qs.make_fragment_B(sQ)
        tCrK_B_qs = tiled_mma_qs.make_fragment_B(sK)
        tCrAinv_B = tiled_mma_qkv.make_fragment_B(sAinv)
        tCrQkv_A = tCtShared_inp
        tCrNv_B = tiled_mma_qkv.make_fragment_B(sQk)
        tCrDecayV_A = tCtShared_inp
        tCrKt_B = tiled_mma_kv.make_fragment_B(sK_trans)

        num_kphases = cute.size(tCrK_A, mode=[2])
        num_kphases_qs = cute.size(tCrS_A, mode=[2])
        num_kphases_qkv = cute.size(tCrAinv_B, mode=[2])
        num_kphases_kv = cute.size(tCrKt_B, mode=[2])

        # Pair membership is derived from the caller-provided chunk index.
        is_pair_first = (chunk_iter & 1) == 0
        has_next_pair = chunk_iter < num_pairs_b * 2 - 2

        # Both KK/QK accumulators must precede chunk 0 because CG0 builds
        # the pair inverse from KK0/KK1 before MMA can consume Ainv0.
        if is_pair_first:
            k0_cursor = load_k_releaser.clone()
            k0_ready_handle = k0_cursor.current_handle()
            k1_cursor = k0_cursor.clone()
            k1_cursor.advance()
            k1_ready_handle = k1_cursor.current_handle()

            if is_first_chunk:
                kk0_handle = cg0_shared_acc_producer.acquire_and_advance()
                k0_ready_handle = load_k_consumer.wait_and_advance()
                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                    tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                    cute.gemm(
                        tiled_mma_qk,
                        tCtCg0Shared[None, None, None, kk0_handle.index],
                        tCrK_A[None, None, kphase_idx, k0_ready_handle.index],
                        tCrK_B[None, None, kphase_idx, k0_ready_handle.index],
                        tCtCg0Shared[None, None, None, kk0_handle.index],
                    )
                kk0_handle.commit()

                kk1_handle = cg0_shared_acc_producer.acquire_and_advance()
                k1_ready_handle = load_k_consumer.wait_and_advance()
                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                    tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                    cute.gemm(
                        tiled_mma_qk,
                        tCtCg0Shared[None, None, None, kk1_handle.index],
                        tCrK_A[None, None, kphase_idx, k1_ready_handle.index],
                        tCrK_B[None, None, kphase_idx, k1_ready_handle.index],
                        tCtCg0Shared[None, None, None, kk1_handle.index],
                    )
                kk1_handle.commit()

            qk0_handle = cg0_shared_acc_producer.acquire_and_advance()
            q0_ready_handle = load_q_consumer.wait_and_advance()
            for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                cute.gemm(
                    tiled_mma_qk,
                    tCtCg0Shared[None, None, None, qk0_handle.index],
                    tCrQ_A[None, None, kphase_idx, q0_ready_handle.index],
                    tCrK_B[None, None, kphase_idx, k0_ready_handle.index],
                    tCtCg0Shared[None, None, None, qk0_handle.index],
                )
            qk0_handle.commit()

            qk1_handle = cg0_shared_acc_producer.acquire_and_advance()
            q1_ready_handle = load_q_consumer.wait_and_advance()
            for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                cute.gemm(
                    tiled_mma_qk,
                    tCtCg0Shared[None, None, None, qk1_handle.index],
                    tCrQ_A[None, None, kphase_idx, q1_ready_handle.index],
                    tCrK_B[None, None, kphase_idx, k1_ready_handle.index],
                    tCtCg0Shared[None, None, None, qk1_handle.index],
                )
            qk1_handle.commit()

        k_release_handle = load_k_releaser.current_handle()
        q_release_handle = load_q_releaser.current_handle()
        k_stage = k_release_handle.index
        q_stage = q_release_handle.index

        ks_handle = cg1_shared_acc_producer.acquire_and_advance()
        state_handle = state_inp_ready_consumer.wait_and_advance()
        for kphase_idx in cutlass.range(num_kphases_qs, unroll_full=True):
            tiled_mma_qs.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                tiled_mma_qs,
                tCtCg1Shared[None, None, None, ks_handle.index],
                tCrS_A[None, None, kphase_idx, state_handle.index],
                tCrK_B_qs[None, None, kphase_idx, k_stage],
                tCtCg1Shared[None, None, None, ks_handle.index],
            )
        ks_handle.commit()

        qs_handle = q_state_acc_producer.acquire_and_advance()
        for kphase_idx in cutlass.range(num_kphases_qs, unroll_full=True):
            tiled_mma_qs.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                tiled_mma_qs,
                tCtQState[None, None, None, qs_handle.index],
                tCrS_A[None, None, kphase_idx, state_handle.index],
                tCrQ_B_qs[None, None, kphase_idx, q_stage],
                tCtQState[None, None, None, qs_handle.index],
            )
        qs_handle.commit()
        state_handle.release()

        q_release_handle.release()
        load_q_releaser.advance()

        nv_handle = cg1_shared_acc_producer.acquire_and_advance()
        vks_ready_consumer.wait_and_advance()
        ainv_handle = a_inv_ready_consumer.wait_and_advance()
        for kphase_idx in cutlass.range(num_kphases_qkv, unroll_full=True):
            tiled_mma_qkv.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                tiled_mma_qkv,
                tCtCg1Shared[None, None, None, nv_handle.index],
                tCtShared_inp[None, None, kphase_idx, 0],
                tCrAinv_B[None, None, kphase_idx, ainv_handle.index],
                tCtCg1Shared[None, None, None, nv_handle.index],
            )
        nv_handle.commit()
        ainv_handle.release()

        # Preserve the mid-pair lookahead after NV0. The two successor KKs
        # are explicit operations, not a nested pair/chunk loop.
        if is_pair_first:
            if has_next_pair:
                pf_kk0_handle = cg0_shared_acc_producer.acquire_and_advance()
                pf_k0_handle = load_k_consumer.wait_and_advance()
                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                    tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                    cute.gemm(
                        tiled_mma_qk,
                        tCtCg0Shared[None, None, None, pf_kk0_handle.index],
                        tCrK_A[None, None, kphase_idx, pf_k0_handle.index],
                        tCrK_B[None, None, kphase_idx, pf_k0_handle.index],
                        tCtCg0Shared[None, None, None, pf_kk0_handle.index],
                    )
                pf_kk0_handle.commit()

                pf_kk1_handle = cg0_shared_acc_producer.acquire_and_advance()
                pf_k1_handle = load_k_consumer.wait_and_advance()
                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                    tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                    cute.gemm(
                        tiled_mma_qk,
                        tCtCg0Shared[None, None, None, pf_kk1_handle.index],
                        tCrK_A[None, None, kphase_idx, pf_k1_handle.index],
                        tCrK_B[None, None, kphase_idx, pf_k1_handle.index],
                        tCtCg0Shared[None, None, None, pf_kk1_handle.index],
                    )
                pf_kk1_handle.commit()

        q_state_handle = q_state_acc_producer.acquire_and_advance()
        qkv_qk_handle = qk_ready_consumer.wait_and_advance()
        nv_ready_consumer.wait_and_advance()
        for kphase_idx in cutlass.range(num_kphases_qkv, unroll_full=True):
            tiled_mma_qkv.set(tcgen05.Field.ACCUMULATE, True)
            cute.gemm(
                tiled_mma_qkv,
                tCtQState[None, None, None, q_state_handle.index],
                tCrQkv_A[None, None, kphase_idx, 0],
                tCrNv_B[None, None, kphase_idx, qkv_qk_handle.index],
                tCtQState[None, None, None, q_state_handle.index],
            )
        qkv_qk_handle.release()
        q_state_handle.commit()

        if is_first_chunk:
            kv_acc_producer.advance()
        kv_handle = kv_acc_producer.acquire_and_advance()
        decay_v_ready_consumer.wait_and_advance()
        for kphase_idx in cutlass.range(num_kphases_kv, unroll_full=True):
            tiled_mma_kv.set(tcgen05.Field.ACCUMULATE, True)
            cute.gemm(
                tiled_mma_kv,
                tCtState[None, None, None, kv_handle.index],
                tCrDecayV_A[None, None, kphase_idx, 1],
                tCrKt_B[None, None, kphase_idx, k_stage],
                tCtState[None, None, None, kv_handle.index],
            )
        kv_handle.commit()
        k_release_handle.release()
        load_k_releaser.advance()

        return (
            cg0_shared_acc_producer,
            cg1_shared_acc_producer,
            q_state_acc_producer,
            kv_acc_producer,
            load_k_consumer,
            load_k_releaser,
            load_q_consumer,
            load_q_releaser,
            load_v_consumer,
            a_inv_ready_consumer,
            qk_ready_consumer,
            state_inp_ready_consumer,
            vks_ready_consumer,
            nv_ready_consumer,
            decay_v_ready_consumer,
        )

    @cute.jit
    def compute_group_0_pair(
        self,
        tidx: cutlass.Int32,
        tmem_ptr: cutlass.Int64,
        scale: cutlass.Float32,
        mma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
    ) -> tuple[
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
    ]:
        """Warps 0-3: reuse one T-pairwise calculation across KK and QK.

        The first pair completes both inverses before QK. A steady pair consumes
        the prefetched KK0/KK1 from CG0's private ring, then reuses the same T
        registers for QK0/QK1 before publishing both inverses.
        """
        (tiled_mma_qk,) = mma_args
        sCumsumlog, sBeta, sAinv, sAinvCal, sQk = smem_args
        (
            load_gate_consumer,
            load_beta_consumer,
            cg0_shared_acc_consumer,
            a_inv_ready_producer,
            qk_ready_producer,
        ) = pipeline_args
        _, _ = work_args

        # ------------------------------------------------------------------
        # Preamble: identical to compute_group_0
        # ------------------------------------------------------------------
        num_threads_cg0 = self.threads_per_warp * len(self.compute_group_0_warp_ids)
        cg0_tidx = tidx % num_threads_cg0

        tAcc_shape = tiled_mma_qk.partition_shape_C(
            (self.mma_tiler_qk[0], self.mma_tiler_qk[1])
        )
        tAcc_wo_stages = tiled_mma_qk.make_fragment_C(tAcc_shape)
        tAcc = cute.make_tensor(
            tAcc_wo_stages.iterator,
            cute.flat_product(
                tAcc_wo_stages.layout,
                cute.make_layout((self.tmem_cg0_shared_acc_stages,), stride=(1,)),
            ),
        )
        tStS_staged = cute.make_tensor(
            tmem_ptr + self.tmem_cg0_shared_acc_offset, tAcc.layout
        )
        tStS_staged_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            tStS_staged
        )
        cS = cute.make_identity_tensor((self.mma_tiler_qk[0], self.mma_tiler_qk[1]))
        tStS_for_t2r = tStS_staged[(None, None), 0, 0, 0]
        atom_shared_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(16)), cutlass.Float32
        )
        tiled_shared_t2r = tcgen05.make_tmem_copy(atom_shared_t2r, tStS_for_t2r)
        thr_shared_t2r = tiled_shared_t2r.get_slice(cg0_tidx)

        tTR_tStS = thr_shared_t2r.partition_S(tStS_staged_mn_view)
        tTR_tScS = thr_shared_t2r.partition_D(cS)

        atom_ainv_r2s = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.io_dtype,
            num_bits_per_copy=128,
        )
        tiled_ainv_r2s = cute.make_tiled_copy_D(atom_ainv_r2s, tiled_shared_t2r)
        sAinvCal_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            sAinvCal
        )
        tiled_ainv_cal_r2s = tiled_ainv_r2s
        tCsAICal = tiled_ainv_cal_r2s.get_slice(cg0_tidx).partition_D(
            sAinvCal_mn_view
        )

        atom_qk_r2s = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.io_dtype,
            num_bits_per_copy=128,
        )
        tiled_qk_r2s = cute.make_tiled_copy_D(atom_qk_r2s, tiled_shared_t2r)
        sQk_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(sQk)
        tCsQK = tiled_qk_r2s.get_slice(cg0_tidx).partition_D(sQk_mn_view)
        # ------------------------------------------------------------------
        # Step 1a: T-pairwise for sub-chunk 0
        # ------------------------------------------------------------------
        gate0_handle = load_gate_consumer.wait_and_advance()
        sCumsumlog0 = sCumsumlog[None, 0, gate0_handle.index]
        row_coord = tTR_tScS[0][0]
        row_cumsumlog0 = sCumsumlog0[row_coord]
        tGrCumsumlog_0 = cute.make_rmem_tensor_like(tTR_tScS, cutlass.Float32)
        gate1_handle = load_gate_consumer.wait_and_advance()
        sCumsumlog1 = sCumsumlog[None, 0, gate1_handle.index]
        row_cumsumlog1 = sCumsumlog1[row_coord]
        tGrCumsumlog_1 = cute.make_rmem_tensor_like(tTR_tScS, cutlass.Float32)

        # The triangular predicate is identical for both chunks in the pair.
        # Compute the two named register fragments together so the predicate
        # and element traversal are emitted once without dynamic RMEM indexing.
        for k in cutlass.range_constexpr(cute.size(tTR_tScS)):
            coord = tTR_tScS[k]
            is_lower = row_coord >= coord[1]
            tGrCumsumlog_0[k] = (
                cute.math.exp2(
                    row_cumsumlog0 - sCumsumlog0[coord[1]],
                    fastmath=True,
                )
                if is_lower
                else 0.0
            )
            tGrCumsumlog_1[k] = (
                cute.math.exp2(
                    row_cumsumlog1 - sCumsumlog1[coord[1]],
                    fastmath=True,
                )
                if is_lower
                else 0.0
            )
        gate0_handle.release()
        gate1_handle.release()

        beta0_handle = load_beta_consumer.wait_and_advance()
        tGrBeta_0 = sBeta[tTR_tScS[0][0], 0, beta0_handle.index]
        beta1_handle = load_beta_consumer.wait_and_advance()
        tGrBeta_1 = sBeta[tTR_tScS[0][0], 0, beta1_handle.index]
        # ------------------------------------------------------------------
        # Step 2: kk_epi0 + kk_epi1
        # Consume both KK accumulators first so their inverse can overlap the
        # following QK0/QK1 MMA issues in the two shared-acc stages.
        # ------------------------------------------------------------------
        ainv0_handle = a_inv_ready_producer.acquire_and_advance()

        tKKrKK = cute.make_rmem_tensor_like(tTR_tScS, cutlass.Float32)
        tKKrKK_out_cal = cute.make_rmem_tensor_like(tKKrKK, self.io_dtype)
        tCrAICal = tiled_ainv_cal_r2s.retile(tKKrKK_out_cal)
        kk0_handle = cg0_shared_acc_consumer.wait_and_advance()
        for sub in cutlass.range(tKKrKK.shape[2]):
            cute.copy(
                tiled_shared_t2r,
                tTR_tStS[None, 0, sub, kk0_handle.index],
                tKKrKK[None, 0, sub],
            )
        cute.arch.fence_view_async_tmem_load()
        kk0_handle.release()
        for sub in cutlass.range(tKKrKK.shape[2]):
            for k in cutlass.range(
                cute.size(tGrCumsumlog_0.shape[0]),
                vectorize=True,
                unroll_full=True,
            ):
                tKKrKK[k, 0, sub] = (
                    tKKrKK[k, 0, sub] * tGrCumsumlog_0[k, 0, sub] * tGrBeta_0
                )
            tKKrKK_out_cal[None, 0, sub].store(
                tKKrKK[None, 0, sub].load().to(self.io_dtype)
            )
            cute.copy(
                tiled_ainv_cal_r2s,
                tCrAICal[None, 0, sub],
                tCsAICal[None, 0, sub, ainv0_handle.index],
            )

        ainv1_handle = a_inv_ready_producer.acquire_and_advance()
        kk1_handle = cg0_shared_acc_consumer.wait_and_advance()
        for sub in cutlass.range(tKKrKK.shape[2]):
            cute.copy(
                tiled_shared_t2r,
                tTR_tStS[None, 0, sub, kk1_handle.index],
                tKKrKK[None, 0, sub],
            )
        cute.arch.fence_view_async_tmem_load()
        kk1_handle.release()
        for sub in cutlass.range(tKKrKK.shape[2]):
            for k in cutlass.range(
                cute.size(tGrCumsumlog_1.shape[0]),
                vectorize=True,
                unroll_full=True,
            ):
                tKKrKK[k, 0, sub] = (
                    tKKrKK[k, 0, sub] * tGrCumsumlog_1[k, 0, sub] * tGrBeta_1
                )
            tKKrKK_out_cal[None, 0, sub].store(
                tKKrKK[None, 0, sub].load().to(self.io_dtype)
            )
            cute.copy(
                tiled_ainv_cal_r2s,
                tCrAICal[None, 0, sub],
                tCsAICal[None, 0, sub, ainv1_handle.index],
            )

        self._partial_pair_inverse(tidx, sAinvCal, ainv0_handle, ainv1_handle)
        self._finish_pair_inverse(
            tidx,
            (sBeta, sAinv, sAinvCal),
            (tiled_shared_t2r, tTR_tScS),
            (ainv0_handle, ainv1_handle, beta0_handle, beta1_handle),
        )

        qk0_ready_handle = qk_ready_producer.acquire_and_advance()
        qk0_handle = cg0_shared_acc_consumer.wait_and_advance()
        tQKrQK = cute.make_rmem_tensor_like(tTR_tScS, cutlass.Float32)
        tQKrQK_out = cute.make_rmem_tensor_like(tQKrQK, self.io_dtype)
        tCrQK = tiled_qk_r2s.retile(tQKrQK_out)
        for sub in cutlass.range(tQKrQK.shape[2]):
            cute.copy(
                tiled_shared_t2r,
                tTR_tStS[None, 0, sub, qk0_handle.index],
                tQKrQK[None, 0, sub],
            )
            for k in cutlass.range(
                cute.size(tGrCumsumlog_0.shape[0]), vectorize=True, unroll_full=True
            ):
                tQKrQK[k, 0, sub] = (
                    tQKrQK[k, 0, sub] * tGrCumsumlog_0[k, 0, sub] * scale
                )
            tQKrQK_out[None, 0, sub].store(
                tQKrQK[None, 0, sub].load().to(self.io_dtype)
            )
            cute.copy(
                tiled_qk_r2s,
                tCrQK[None, 0, sub],
                tCsQK[None, 0, sub, qk0_ready_handle.index],
            )
        cute.arch.fence_view_async_shared()
        cute.arch.fence_view_async_tmem_load()
        qk0_handle.release()
        qk0_ready_handle.commit()

        # ------------------------------------------------------------------
        # Step 3: qk_epi1
        # ------------------------------------------------------------------
        qk1_ready_handle = qk_ready_producer.acquire_and_advance()
        qk1_handle = cg0_shared_acc_consumer.wait_and_advance()

        for sub in cutlass.range(tQKrQK.shape[2]):
            cute.copy(
                tiled_shared_t2r,
                tTR_tStS[None, 0, sub, qk1_handle.index],
                tQKrQK[None, 0, sub],
            )
            for k in cutlass.range(
                cute.size(tGrCumsumlog_1.shape[0]), vectorize=True, unroll_full=True
            ):
                tQKrQK[k, 0, sub] = (
                    tQKrQK[k, 0, sub] * tGrCumsumlog_1[k, 0, sub] * scale
                )
            tQKrQK_out[None, 0, sub].store(
                tQKrQK[None, 0, sub].load().to(self.io_dtype)
            )
            cute.copy(
                tiled_qk_r2s,
                tCrQK[None, 0, sub],
                tCsQK[None, 0, sub, qk1_ready_handle.index],
            )
        cute.arch.fence_view_async_shared()
        cute.arch.fence_view_async_tmem_load()
        qk1_handle.release()
        qk1_ready_handle.commit()

        return (
            load_gate_consumer,
            load_beta_consumer,
            cg0_shared_acc_consumer,
            a_inv_ready_producer,
            qk_ready_producer,
        )

    @cute.jit
    def _partial_pair_inverse(
        self,
        tidx: cutlass.Int32,
        sAinvCal: cute.Tensor,
        ainv0_handle,
        ainv1_handle,
    ):
        """Invert 8x8 diagonal blocks and build the 16x16 corrections."""
        num_threads_cg0 = self.threads_per_warp * len(self.compute_group_0_warp_ids)
        cg0_tidx = tidx % num_threads_cg0
        warp_id = cg0_tidx // 32
        lane_id = cg0_tidx % 32
        inverse_group = warp_id // 2
        inverse_local_warp = warp_id % 2
        sAinvCal_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            sAinvCal
        )
        sA0Work = sAinvCal_mn_view[None, None, ainv0_handle.index]
        sA1Work = sAinvCal_mn_view[None, None, ainv1_handle.index]
        # Three stages can wrap between the two pair handles, so XOR is invalid.
        inverse_stage = ainv0_handle.index
        if inverse_group == 1:
            inverse_stage = ainv1_handle.index
        sAWork = sAinvCal_mn_view[None, None, inverse_stage]

        sM_8x8 = cute.flat_divide(sAWork, (8, 8))
        self.inverse_barrier.arrive_and_wait()
        idx_8x8 = (inverse_local_warp * self.threads_per_warp + lane_id) // 8
        self._invert_diagonal_NxN(sM_8x8[None, None, idx_8x8, idx_8x8], cg0_tidx, 8)
        self.inverse_barrier.arrive_and_wait()

        sM0_16x16 = cute.flat_divide(sA0Work, (16, 16))
        sM1_16x16 = cute.flat_divide(sA1Work, (16, 16))
        self._blockwise_diagonal_8x8_to_16x16(
            sM0_16x16[None, None, warp_id, warp_id], lane_id
        )
        self._blockwise_diagonal_8x8_to_16x16(
            sM1_16x16[None, None, warp_id, warp_id], lane_id
        )
        self.inverse_barrier.arrive_and_wait()

    @cute.jit
    def _finish_pair_inverse(
        self,
        tidx: cutlass.Int32,
        smem_args: tuple,
        copy_args: tuple,
        handle_args: tuple,
    ):
        """Invert and publish two prepared A-inverse stages in parallel."""
        sBeta, sAinv, sAinvCal = smem_args
        tiled_shared_t2r, tTR_tScS = copy_args
        ainv0_handle, ainv1_handle, beta0_handle, beta1_handle = handle_args

        num_threads_cg0 = self.threads_per_warp * len(self.compute_group_0_warp_ids)
        cg0_tidx = tidx % num_threads_cg0
        warp_id = cg0_tidx // 32
        lane_id = cg0_tidx % 32

        sAinv_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(sAinv)
        sAinvCal_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            sAinvCal
        )
        inverse_group = warp_id // 2
        inverse_local_warp = warp_id % 2
        inverse_stage = ainv0_handle.index
        if inverse_group == 1:
            inverse_stage = ainv1_handle.index
        sAWork = sAinvCal_mn_view[None, None, inverse_stage]

        # Resume after the partial inverse: 16x16 -> 32x32 -> 64x64.
        sM_32x32 = cute.flat_divide(sAWork, (32, 32))
        self._blockwise_diagonal_16x16_to_32x32(
            sM_32x32[None, None, inverse_local_warp, inverse_local_warp], lane_id
        )
        self.inverse_barrier.arrive_and_wait()
        sM_64x64 = cute.flat_divide(sAWork, (64, 64))
        self._blockwise_diagonal_32x32_to_64x64(
            sM_64x64[None, None, 0, 0], inverse_local_warp, lane_id
        )
        self.inverse_barrier.arrive_and_wait()

        # Apply beta column scaling and publish both stages to the MMA warp.
        atom_ainv_s2r = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.io_dtype, num_bits_per_copy=128
        )
        tiled_ainv_s2r = cute.make_tiled_copy_D(atom_ainv_s2r, tiled_shared_t2r)
        thr_ainv_s2r = tiled_ainv_s2r.get_slice(cg0_tidx)
        tiled_ainv_cal_s2r = tiled_ainv_s2r
        thr_ainv_cal_s2r = thr_ainv_s2r
        tKKrBeta = cute.make_rmem_tensor_like(tTR_tScS, cutlass.Float32)

        tCsAI0 = thr_ainv_s2r.partition_S(sAinv_mn_view)[
            None, None, None, ainv0_handle.index
        ]
        tCsAI0Cal = thr_ainv_cal_s2r.partition_S(sAinvCal_mn_view)[
            None, None, None, ainv0_handle.index
        ]
        tCrAI0 = cute.make_rmem_tensor_like(tCsAI0, self.io_dtype)
        tCrAI0Acc = cute.make_rmem_tensor_like(tCrAI0, cutlass.Float32)
        tCrAI0Cal = cute.make_rmem_tensor_like(tCsAI0Cal, self.io_dtype)
        for k in cutlass.range(cute.size(tTR_tScS)):
            coord = tTR_tScS[k]
            tKKrBeta[k] = sBeta[coord[1], 0, beta0_handle.index]
        cute.copy(tiled_ainv_cal_s2r, tCsAI0Cal, tCrAI0Cal)
        tCrAI0Acc.store(tCrAI0Cal.load().to(cutlass.Float32))
        for k in cutlass.range(cute.size(tCrAI0), vectorize=True, unroll_full=True):
            tCrAI0Acc[k] = tCrAI0Acc[k] * tKKrBeta[k]
        tCrAI0.store(tCrAI0Acc.load().to(self.io_dtype))
        cute.copy(tiled_ainv_s2r, tCrAI0, tCsAI0)
        cute.arch.fence_view_async_shared()
        ainv0_handle.commit()
        beta0_handle.release()

        tCsAI1 = thr_ainv_s2r.partition_S(sAinv_mn_view)[
            None, None, None, ainv1_handle.index
        ]
        tCsAI1Cal = thr_ainv_cal_s2r.partition_S(sAinvCal_mn_view)[
            None, None, None, ainv1_handle.index
        ]
        tCrAI1 = cute.make_rmem_tensor_like(tCsAI1, self.io_dtype)
        tCrAI1Acc = cute.make_rmem_tensor_like(tCrAI1, cutlass.Float32)
        tCrAI1Cal = cute.make_rmem_tensor_like(tCsAI1Cal, self.io_dtype)
        for k in cutlass.range(cute.size(tTR_tScS)):
            coord = tTR_tScS[k]
            tKKrBeta[k] = sBeta[coord[1], 0, beta1_handle.index]
        cute.copy(tiled_ainv_cal_s2r, tCsAI1Cal, tCrAI1Cal)
        tCrAI1Acc.store(tCrAI1Cal.load().to(cutlass.Float32))
        for k in cutlass.range(cute.size(tCrAI1), vectorize=True, unroll_full=True):
            tCrAI1Acc[k] = tCrAI1Acc[k] * tKKrBeta[k]
        tCrAI1.store(tCrAI1Acc.load().to(self.io_dtype))
        cute.copy(tiled_ainv_s2r, tCrAI1, tCsAI1)
        cute.arch.fence_view_async_shared()
        ainv1_handle.commit()
        beta1_handle.release()

    # ------------------------------------------------------------------
    # Hierarchical blockwise inverse helpers.
    # Compute X = (I + M)^{-1} for a 64x64 unit lower-triangular matrix in-place
    # on a row-major SMEM buffer.  4-stage algorithm:
    #   Stage 1: Gauss-Jordan inversion of 8 diagonal 8x8 blocks (warp shuffle)
    #   Stage 2: 8x8 -> 16x16 via warp MMA  (SM80_16x8x8)
    #   Stage 3: 16x16 -> 32x32 via warp MMA (SM80_16x8x16)
    #   Stage 4: 32x32 -> 64x64 via warp MMA, 2 warps per 64x64 tile
    # ------------------------------------------------------------------

    def _make_acc_tensor_into_a_view(self, acc: cute.Tensor) -> cute.Tensor:
        """Reinterpret accumulator tensor as an A-operand tensor for the next MMA.

        For SM80_16x8x8 (ratio=1) the layout is unchanged; for SM80_16x8x16 (ratio=2)
        the C-frag atom size differs from the A-frag atom size and requires a reshape.
        """
        acc_layout_divided = cute.logical_divide(acc.layout, (None, None, 2))
        acc_layout_a = cute.make_layout(
            (
                (acc_layout_divided.shape[0], acc_layout_divided.shape[2][0]),
                acc_layout_divided.shape[1],
                acc_layout_divided.shape[2][1],
            ),
            stride=(
                (acc_layout_divided.stride[0], acc_layout_divided.stride[2][0]),
                acc_layout_divided.stride[1],
                acc_layout_divided.stride[2][1],
            ),
        )
        return cute.make_tensor(acc.iterator, acc_layout_a)

    @cute.jit
    def _invert_diagonal_NxN(self, mat_NxN, tidx, N: int = 8):
        """Stage 1: Gauss-Jordan inversion of one diagonal NxN block in-place.

        Each thread owns one row (tidx_in_group = tidx % N).
        Uses warp shuffle to broadcast pivot values; no __syncthreads inside.
        N-1 pivot steps, compile-time unrolled.
        """
        tidx_in_group = tidx % N
        row_storage = cute.make_rmem_tensor((N,), mat_NxN.element_type)
        cute.autovec_copy(mat_NxN[tidx_in_group, None], row_storage)
        row = cute.make_rmem_tensor_like(row_storage, cutlass.Float32)
        row.store(row_storage.load().to(cutlass.Float32))

        for i in cutlass.range_constexpr(N):
            row[i] = 1.0 if tidx_in_group == i else row[i]
        for src_row in cutlass.range_constexpr(N - 1):
            row_scale = -row[src_row]
            for i in cutlass.range_constexpr(src_row):
                shfl_val = cute.arch.shuffle_sync(
                    row[i], src_row, mask=0xFFFFFFFF, mask_and_clamp=0b1100000011111
                )
                row[i] = (
                    row[i] + row_scale * shfl_val if tidx_in_group > src_row else row[i]
                )
            row[src_row] = row_scale if tidx_in_group > src_row else row[src_row]

        row_storage.store(row.load().to(mat_NxN.element_type))
        cute.autovec_copy(row_storage, mat_NxN[tidx_in_group, None])

    @cute.jit
    def _blockwise_diagonal_8x8_to_16x16(self, mat_16x16, lane_id):
        """Stage 2: off-diagonal correction for one 16x16 diagonal tile (8x8 -> 16x16).

        After Stage 1 each diagonal 8x8 is inverted.  Computes the bottom-left 8x8
        correction block: C <-- -D^{-1} C A^{-1}.
        MMA: SM80_16x8x8_F32F16F16F32_TN, single warp.  D^{-1} broadcast 8x8 -> 16x8.
        """
        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            mat_16x16.element_type, cutlass.Float32, (16, 8, 8)
        )
        tiled_mma = cute.make_tiled_mma(mma_atom, cute.make_layout((1, 1, 1)))
        thr_mma = tiled_mma.get_slice(lane_id)

        D_tiled_copy = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=1, transpose=False),
                mat_16x16.element_type,
            ),
            tiled_mma,
        )
        C_tiled_copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=1, transpose=True),
                mat_16x16.element_type,
            ),
            tiled_mma,
        )
        A_tiled_copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=1, transpose=True),
                mat_16x16.element_type,
            ),
            tiled_mma,
        )
        O_tiled_copy = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=1, transpose=False),
                mat_16x16.element_type,
            ),
            tiled_mma,
        )

        D_thr_copy = D_tiled_copy.get_slice(lane_id)
        C_thr_copy = C_tiled_copy.get_slice(lane_id)
        A_thr_copy = A_tiled_copy.get_slice(lane_id)
        O_thr_copy = O_tiled_copy.get_slice(lane_id)

        mat_8x8_2x2 = cute.flat_divide(mat_16x16, (8, 8))
        sDinv = mat_8x8_2x2[None, None, 1, 1]
        sC = mat_8x8_2x2[None, None, 1, 0]
        sC = cute.make_tensor(sC.iterator, cute.select(sC.layout, mode=[1, 0]))
        sAinv = mat_8x8_2x2[None, None, 0, 0]
        sAinv = cute.make_tensor(sAinv.iterator, cute.select(sAinv.layout, mode=[1, 0]))
        sO = mat_8x8_2x2[None, None, 1, 0]

        sDinv_m_bcast = cute.make_tensor(
            sDinv.iterator,
            cute.logical_product(sDinv.layout, (cute.make_layout((2,), stride=(0,)),)),
        )
        sO_m_bcast = cute.make_tensor(
            sO.iterator,
            cute.logical_product(sO.layout, (cute.make_layout((2,), stride=(0,)),)),
        )

        tOrDinv = tiled_mma.make_fragment_A(thr_mma.partition_A(sDinv_m_bcast))
        tOrC = tiled_mma.make_fragment_B(thr_mma.partition_B(sC))
        tOrAinv = tiled_mma.make_fragment_B(thr_mma.partition_B(sAinv))
        tDCrDC = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 8)))
        tOrO = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 8)))

        tOsDinv = D_thr_copy.partition_S(sDinv_m_bcast)
        tOrDinv_cv = D_thr_copy.retile(tOrDinv)
        tOsDinv = cute.logical_divide(tOsDinv, (tOsDinv.shape[0], None, None))
        tOrDinv_cv = cute.logical_divide(
            tOrDinv_cv, (tOrDinv_cv.shape[0], None, None)
        )
        tOsC = C_thr_copy.partition_S(sC)
        tOrC_cv = C_thr_copy.retile(tOrC)
        tOsAinv = A_thr_copy.partition_S(sAinv)
        tOrAinv_cv = A_thr_copy.retile(tOrAinv)
        tOsO = O_thr_copy.partition_D(sO_m_bcast)
        tOsO = cute.logical_divide(tOsO, (tOsO.shape[0], None, None))
        tOrO_cv = O_thr_copy.retile(tOrO)
        tOrO_cv = cute.logical_divide(tOrO_cv, (tOrO_cv.shape[0], None, None))
        cute.copy(
            D_tiled_copy,
            tOsDinv[(None, 0), None, None],
            tOrDinv_cv[(None, 0), None, None],
        )
        cute.copy(C_tiled_copy, tOsC, tOrC_cv)
        tDCrDC.fill(0.0)
        cute.gemm(tiled_mma, tDCrDC, tOrDinv, tOrC, tDCrDC)
        tDCrDC.store(cutlass.Float32(0.0) - tDCrDC.load())
        tDCrDC_a = self._make_acc_tensor_into_a_view(tDCrDC)
        tOrDC = cute.make_rmem_tensor_like(tDCrDC_a, mat_16x16.element_type)
        tOrDC.store(tDCrDC_a.load().to(mat_16x16.element_type))

        cute.copy(A_tiled_copy, tOsAinv, tOrAinv_cv)
        tOrO.fill(0.0)
        cute.gemm(tiled_mma, tOrO, tOrDC, tOrAinv, tOrO)

        tOrO_cv_cvt = cute.make_rmem_tensor_like(
            tOrO_cv[(None, 0), None, None], mat_16x16.element_type
        )
        tOrO_cv_cvt.store(
            tOrO_cv[(None, 0), None, None].load().to(mat_16x16.element_type)
        )
        cute.copy(O_tiled_copy, tOrO_cv_cvt, tOsO[(None, 0), None, None])

    @cute.jit
    def _blockwise_diagonal_16x16_to_32x32(self, mat_32x32, lane_id):
        """Stage 3: off-diagonal correction for one 32x32 diagonal tile (16x16 -> 32x32).

        After Stage 2 each diagonal 16x16 is inverted.  Computes C <-- -D^{-1} C A^{-1}.
        MMA: SM80_16x8x16_F32F16F16F32_TN, TileShape (16,16,16), single warp.
        make_acc_into_op ratio=2: A-frag atom size (8) / C-frag atom size (4).
        """
        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            mat_32x32.element_type, cutlass.Float32, (16, 8, 16)
        )
        tiled_mma = cute.make_tiled_mma(
            mma_atom, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 16, 16)
        )
        thr_mma = tiled_mma.get_slice(lane_id)

        D_tiled_copy = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=False),
                mat_32x32.element_type,
            ),
            tiled_mma,
        )
        C_tiled_copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True),
                mat_32x32.element_type,
            ),
            tiled_mma,
        )
        A_tiled_copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True),
                mat_32x32.element_type,
            ),
            tiled_mma,
        )
        O_tiled_copy = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=False),
                mat_32x32.element_type,
            ),
            tiled_mma,
        )

        D_thr_copy = D_tiled_copy.get_slice(lane_id)
        C_thr_copy = C_tiled_copy.get_slice(lane_id)
        A_thr_copy = A_tiled_copy.get_slice(lane_id)
        O_thr_copy = O_tiled_copy.get_slice(lane_id)

        mat_16x16_2x2 = cute.flat_divide(mat_32x32, (16, 16))
        sDinv = mat_16x16_2x2[None, None, 1, 1]
        sC = mat_16x16_2x2[None, None, 1, 0]
        sC = cute.make_tensor(sC.iterator, cute.select(sC.layout, mode=[1, 0]))
        sAinv = mat_16x16_2x2[None, None, 0, 0]
        sAinv = cute.make_tensor(sAinv.iterator, cute.select(sAinv.layout, mode=[1, 0]))
        sO = mat_16x16_2x2[None, None, 1, 0]

        tOrDinv = tiled_mma.make_fragment_A(thr_mma.partition_A(sDinv))
        tOrC = tiled_mma.make_fragment_B(thr_mma.partition_B(sC))
        tOrAinv = tiled_mma.make_fragment_B(thr_mma.partition_B(sAinv))
        tDCrDC = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 16)))
        tOrO = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 16)))

        tOsDinv = D_thr_copy.partition_S(sDinv)
        tOrDinv_cv = D_thr_copy.retile(tOrDinv)
        tOsC = C_thr_copy.partition_S(sC)
        tOrC_cv = C_thr_copy.retile(tOrC)
        tOsAinv = A_thr_copy.partition_S(sAinv)
        tOrAinv_cv = A_thr_copy.retile(tOrAinv)
        tOsO = O_thr_copy.partition_D(sO)
        tOrO_cv = O_thr_copy.retile(tOrO)

        cute.copy(D_tiled_copy, tOsDinv, tOrDinv_cv)
        cute.copy(C_tiled_copy, tOsC, tOrC_cv)
        tDCrDC.fill(0.0)
        cute.gemm(tiled_mma, tDCrDC, tOrDinv, tOrC, tDCrDC)

        tDCrDC.store(-tDCrDC.load())
        tDCrDC_a = self._make_acc_tensor_into_a_view(tDCrDC)
        tOrDC = cute.make_rmem_tensor_like(tDCrDC_a, mat_32x32.element_type)
        tOrDC.store(tDCrDC_a.load().to(mat_32x32.element_type))

        cute.copy(A_tiled_copy, tOsAinv, tOrAinv_cv)
        tOrO.fill(0.0)
        cute.gemm(tiled_mma, tOrO, tOrDC, tOrAinv, tOrO)

        tOrO_cv_cvt = cute.make_rmem_tensor_like(tOrO_cv, mat_32x32.element_type)
        tOrO_cv_cvt.store(tOrO_cv.load().to(mat_32x32.element_type))
        cute.copy(O_tiled_copy, tOrO_cv_cvt, tOsO)

    @cute.jit
    def _blockwise_diagonal_32x32_to_64x64(self, mat_64x64, local_warp_id, lane_id):
        """Stage 4: off-diagonal correction for one 64x64 diagonal tile (32x32 -> 64x64).

        Two warps collaborate, each owning one 16x32 slice of the bottom-left block.
        MMA: SM80_16x8x16 TileShape (16,32,32), permutation_mnk=(16,32,32).
        Ends with sync_threads() to protect the sO write from races.
        """
        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            mat_64x64.element_type, cutlass.Float32, (16, 8, 16)
        )
        tiled_mma = cute.make_tiled_mma(
            mma_atom, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 32, 32)
        )
        thr_mma = tiled_mma.get_slice(lane_id)

        D_tiled_copy = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=False),
                mat_64x64.element_type,
            ),
            tiled_mma,
        )
        C_tiled_copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True),
                mat_64x64.element_type,
            ),
            tiled_mma,
        )
        A_tiled_copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True),
                mat_64x64.element_type,
            ),
            tiled_mma,
        )
        O_tiled_copy = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=False),
                mat_64x64.element_type,
            ),
            tiled_mma,
        )

        D_thr_copy = D_tiled_copy.get_slice(lane_id)
        C_thr_copy = C_tiled_copy.get_slice(lane_id)
        A_thr_copy = A_tiled_copy.get_slice(lane_id)
        O_thr_copy = O_tiled_copy.get_slice(lane_id)

        mat_32x32_2x2 = cute.flat_divide(mat_64x64, (32, 32))
        sDinv_full = mat_32x32_2x2[None, None, 1, 1]
        sC_full = mat_32x32_2x2[None, None, 1, 0]
        sAinv_full = mat_32x32_2x2[None, None, 0, 0]

        sDinv = cute.flat_divide(sDinv_full, (16, 32))[None, None, local_warp_id, 0]
        sC = cute.make_tensor(
            sC_full.iterator, cute.select(sC_full.layout, mode=[1, 0])
        )
        sAinv = cute.make_tensor(
            sAinv_full.iterator, cute.select(sAinv_full.layout, mode=[1, 0])
        )
        sO = cute.flat_divide(sC_full, (16, 32))[None, None, local_warp_id, 0]

        tOrDinv = tiled_mma.make_fragment_A(thr_mma.partition_A(sDinv))
        tOrC = tiled_mma.make_fragment_B(thr_mma.partition_B(sC))
        tOrAinv = tiled_mma.make_fragment_B(thr_mma.partition_B(sAinv))
        tDCrDC = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 32)))
        tOrO = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 32)))

        tOsDinv = D_thr_copy.partition_S(sDinv)
        tOrDinv_cv = D_thr_copy.retile(tOrDinv)
        tOsC = C_thr_copy.partition_S(sC)
        tOrC_cv = C_thr_copy.retile(tOrC)
        tOsAinv = A_thr_copy.partition_S(sAinv)
        tOrAinv_cv = A_thr_copy.retile(tOrAinv)
        tOsO = O_thr_copy.partition_D(sO)
        tOrO_cv = O_thr_copy.retile(tOrO)

        cute.copy(D_tiled_copy, tOsDinv, tOrDinv_cv)
        cute.copy(C_tiled_copy, tOsC, tOrC_cv)
        tDCrDC.fill(0.0)
        cute.gemm(tiled_mma, tDCrDC, tOrDinv, tOrC, tDCrDC)

        tDCrDC.store(-tDCrDC.load())
        tDCrDC_a = self._make_acc_tensor_into_a_view(tDCrDC)
        tOrDC = cute.make_rmem_tensor_like(tDCrDC_a, mat_64x64.element_type)
        tOrDC.store(tDCrDC_a.load().to(mat_64x64.element_type))

        cute.copy(A_tiled_copy, tOsAinv, tOrAinv_cv)
        tOrO.fill(0.0)
        cute.gemm(tiled_mma, tOrO, tOrDC, tOrAinv, tOrO)

        tOrO_cv_cvt = cute.make_rmem_tensor_like(tOrO_cv, mat_64x64.element_type)
        tOrO_cv_cvt.store(tOrO_cv.load().to(mat_64x64.element_type))
        self.inverse_barrier.arrive_and_wait()
        cute.copy(O_tiled_copy, tOrO_cv_cvt, tOsO)

    @cute.jit
    def _load_initial_state(
        self,
        tidx,
        mS_init,
        mS_indices,
        head_idx,
        batch_idx,
        tmem_ptr,
        tiled_mma_kv,
        kv_acc_producer,
    ) -> pipeline.PipelineProducer:
        """Load S_init from GMEM into state TMEM (fp32).

        Two steps:
          1. GMEM fp32 -> registers
          2. registers -> state TMEM (fp32), signal kv_acc so MMA can start GEMM 7
        """
        num_threads_cg1 = self.threads_per_warp * len(self.compute_group_1_warp_ids)
        cg1_tidx = tidx % num_threads_cg1

        # Build state TMEM store copy (registers -> state TMEM)
        state_acc_shape = tiled_mma_kv.partition_shape_C(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        tCtState_fake = tiled_mma_kv.make_fragment_C(
            cute.append(state_acc_shape, self.tmem_kv_acc_stages)
        )
        tCtState = cute.make_tensor(
            tmem_ptr + self.tmem_state_offset, tCtState_fake.layout
        )
        tCtState_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            tCtState
        )
        state_r2t_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), cutlass.Float32
        )
        cState = cute.make_identity_tensor((self.mma_tiler_kv[0], self.mma_tiler_kv[1]))
        tCtState_for_r2t = tCtState[(None, None), 0, 0, 0]
        tiled_state_r2t = tcgen05.make_tmem_copy(state_r2t_atom, tCtState_for_r2t)
        thr_state_r2t = tiled_state_r2t.get_slice(cg1_tidx)
        tRT_tCtState = thr_state_r2t.partition_D(tCtState_mn_view)
        tRT_tCcState = thr_state_r2t.partition_S(cState)
        tRT_tCrState = cute.make_rmem_tensor_like(tRT_tCcState, cutlass.Float32)

        if cutlass.const_expr(mS_indices is not None):
            state_row = cutlass.Int32(mS_indices[batch_idx])
        else:
            state_row = batch_idx
        gS_init = cute.flat_divide(
            mS_init[None, None, head_idx, state_row],
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1]),
        )[None, None, 0, 0]
        tRT_tCgState = thr_state_r2t.partition_S(gS_init)
        kv_acc_handle = kv_acc_producer.acquire_and_advance()
        for sub in cutlass.range(tRT_tCrState.shape[2]):
            # 1. Load S_init FP32 GMEM -> FP32 registers
            cute.autovec_copy(
                tRT_tCgState[None, 0, sub],
                tRT_tCrState[None, 0, sub],
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )

            # 2. fp32 registers -> state TMEM; signal kv_acc (GEMM 7 accumulates dS on top)
            cute.copy(
                tiled_state_r2t,
                tRT_tCrState[None, 0, sub],
                tRT_tCtState[None, 0, sub, kv_acc_handle.index],
            )
        cute.arch.fence_view_async_tmem_store()

        # Manually sync before committing - CG1 is not the MMA warp so uses mbarrier_arrive.
        self.init_state_store_barrier.arrive_and_wait()
        if cg1_tidx == 0:
            cute.arch.mbarrier_arrive(kv_acc_handle.barrier)

        return kv_acc_producer

    @cute.jit
    def _store_empty_final_state(
        self,
        tidx,
        mS_init,
        mS_out,
        mS_indices,
        head_idx,
        batch_idx,
    ):
        """Copy initial state to final state for a sequence containing no tokens."""
        num_threads_cg1 = self.threads_per_warp * len(self.compute_group_1_warp_ids)
        cg1_tidx = tidx % num_threads_cg1
        state_elements = self.mma_tiler_kv[0] * self.mma_tiler_kv[1]
        if cutlass.const_expr(mS_indices is not None):
            state_row = cutlass.Int32(mS_indices[batch_idx])
        else:
            state_row = batch_idx
        for linear_idx in cutlass.range(
            cg1_tidx, state_elements, num_threads_cg1, unroll=1
        ):
            key_idx = linear_idx // self.mma_tiler_kv[0]
            value_idx = linear_idx - key_idx * self.mma_tiler_kv[0]
            mS_out[value_idx, key_idx, head_idx, state_row] = mS_init[
                value_idx, key_idx, head_idx, state_row
            ]

    @cute.jit
    def _store_final_state(
        self,
        tidx,
        # full output-state GMEM tensor (DK, DV, (h_r, h_qv), B) fp32
        mS_out,
        mS_indices,
        head_idx,
        batch_idx,
        tmem_ptr,
        tiled_mma_kv,
        # MMA -> CG1 consumer; waited+released inside this method
        kv_acc_consumer,
    ):
        """Store final recurrent state from TMEM (fp32) to GMEM mS_out.

        Waits for the last GEMM-7 (kv_acc) to complete, reads state TMEM -> registers,
        writes registers -> GMEM fp32, then releases the consumer handle.
        """
        num_threads_cg1 = self.threads_per_warp * len(self.compute_group_1_warp_ids)
        cg1_tidx = tidx % num_threads_cg1

        # Build state TMEM layout (mirrors compute_group_1 setup)
        state_acc_shape = tiled_mma_kv.partition_shape_C(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        tCtState_fake = tiled_mma_kv.make_fragment_C(
            cute.append(state_acc_shape, self.tmem_kv_acc_stages)
        )
        tCtState = cute.make_tensor(
            tmem_ptr + self.tmem_state_offset, tCtState_fake.layout
        )
        tCtState_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            tCtState
        )
        tCcState = cute.make_identity_tensor(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )

        # TMEM -> registers  (Ld32x32b)
        atom_state_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), cutlass.Float32
        )
        tCtState_for_t2r = tCtState[(None, None), 0, 0, 0]
        tiled_state_t2r = tcgen05.make_tmem_copy(atom_state_t2r, tCtState_for_t2r)
        thr_state_t2r = tiled_state_t2r.get_slice(cg1_tidx)
        tTR_tCtState = thr_state_t2r.partition_S(tCtState_mn_view)
        tTR_tCcState = thr_state_t2r.partition_D(tCcState)
        tTR_rState = cute.make_rmem_tensor_like(tTR_tCcState, cutlass.Float32)

        # Wait for last GEMM-7 to finish.
        kv_acc_handle = kv_acc_consumer.wait_and_advance()

        for sub in cutlass.range(tTR_rState.shape[2]):
            cute.copy(
                tiled_state_t2r,
                tTR_tCtState[None, 0, sub, kv_acc_handle.index],
                tTR_rState[None, 0, sub],
            )
            if cutlass.const_expr(self.store_final_state):
                if cutlass.const_expr(mS_indices is not None):
                    state_row = cutlass.Int32(mS_indices[batch_idx])
                else:
                    state_row = batch_idx
                gS_out = cute.flat_divide(
                    mS_out[None, None, head_idx, state_row],
                    (self.mma_tiler_kv[0], self.mma_tiler_kv[1]),
                )[None, None, 0, 0]
                tTR_tCgState = thr_state_t2r.partition_D(gS_out)
                cute.autovec_copy(
                    tTR_rState[None, 0, sub],
                    tTR_tCgState[None, 0, sub],
                    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                )
        kv_acc_handle.release()
        return kv_acc_consumer

    @cute.jit
    def compute_group_1_chunk(
        self,
        tidx: cutlass.Int32,
        tmem_ptr: cutlass.Int64,
        scale: cutlass.Float32,
        mma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
    ) -> tuple[
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
    ]:
        """Warps 4-7: process one chunk from the caller-owned chunk stream."""
        sV, sCumsumlog, sCumprod, sBeta, sO = smem_args
        (
            load_v_consumer,
            load_gate_consumer,
            cg1_shared_acc_consumer,
            kv_acc_consumer,
            q_state_acc_consumer,
            kv_acc_producer,
            state_inp_ready_producer,
            vks_ready_producer,
            nv_ready_producer,
            decay_v_ready_producer,
            o_store_producer,
        ) = pipeline_args
        tiled_mma_kv, tiled_mma_qs, tiled_mma_qkv = mma_args
        chunk_iter, num_pairs_b, head_idx, seqlen_b = work_args

        # ------------------------------------------------------------------
        # Preamble (identical to compute_group_1)
        # ------------------------------------------------------------------
        num_threads_cg1 = self.threads_per_warp * len(self.compute_group_1_warp_ids)
        cg1_tidx = tidx % num_threads_cg1

        state_acc_shape = tiled_mma_kv.partition_shape_C(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        tCtState_fake = tiled_mma_kv.make_fragment_C(
            cute.append(state_acc_shape, self.tmem_kv_acc_stages)
        )
        tCtState = cute.make_tensor(
            tmem_ptr + self.tmem_state_offset, tCtState_fake.layout
        )
        tCtState_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            tCtState
        )
        tCcState = cute.make_identity_tensor(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        atom_state_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), cutlass.Float32
        )
        atom_state_r2t = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), cutlass.Float32
        )
        tCtState_for_t2r = tCtState[(None, None), 0, 0, 0]
        tiled_state_t2r = tcgen05.make_tmem_copy(atom_state_t2r, tCtState_for_t2r)
        tiled_state_r2t = tcgen05.make_tmem_copy(atom_state_r2t, tCtState_for_t2r)
        thr_state_t2r = tiled_state_t2r.get_slice(cg1_tidx)
        thr_state_r2t = tiled_state_r2t.get_slice(cg1_tidx)
        tTR_tCtState = thr_state_t2r.partition_S(tCtState_mn_view)
        tTR_tCcState = thr_state_t2r.partition_D(tCcState)
        tRT_tCtState = thr_state_r2t.partition_D(tCtState_mn_view)
        tTR_rState = cute.make_rmem_tensor_like(tTR_tCcState, cutlass.Float32)

        state_inp_shape = tiled_mma_qs.partition_shape_A(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[2])
        )
        tCtState_inp_fake = tiled_mma_qs.make_fragment_A(
            cute.append(state_inp_shape, self.tmem_state_inp_stages)
        )
        tCtState_inp = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_state_inp_offset, dtype=self.io_dtype),
            tCtState_inp_fake.layout,
        )
        tCtState_inp_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            tCtState_inp
        )
        tCcState_inp = cute.make_identity_tensor(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[2])
        )
        atom_state_inp_r2t = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), self.io_dtype
        )
        tCtState_inp_for_r2t = tCtState_inp_mn_view[None, None, 0]
        tiled_state_inp_r2t = tcgen05.make_tmem_copy(
            atom_state_inp_r2t, tCtState_inp_for_r2t
        )
        thr_state_inp_r2t = tiled_state_inp_r2t.get_slice(cg1_tidx)
        tRT_tCcState_inp = thr_state_inp_r2t.partition_S(tCcState_inp)
        tRT_tCtState_inp = thr_state_inp_r2t.partition_D(tCtState_inp_mn_view)
        tRT_rState_inp = cute.make_rmem_tensor_like(tRT_tCcState_inp, self.io_dtype)

        qkv_acc_shape = tiled_mma_qkv.partition_shape_C(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        tCtShared_fake = tiled_mma_qkv.make_fragment_C(qkv_acc_shape)
        tCtShared = cute.make_tensor(
            tmem_ptr + self.tmem_cg1_shared_acc_offset,
            cute.flat_product(
                tCtShared_fake.layout,
                cute.make_layout((self.tmem_cg1_shared_acc_stages,)),
            ),
        )
        tCtShared_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            tCtShared
        )
        tCcShared = cute.make_identity_tensor(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        atom_shared_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), cutlass.Float32
        )
        tCtShared_for_t2r = tCtShared[(None, None), 0, 0, 0]
        tiled_shared_t2r = tcgen05.make_tmem_copy(atom_shared_t2r, tCtShared_for_t2r)
        thr_shared_t2r = tiled_shared_t2r.get_slice(cg1_tidx)
        tTR_tCtShared = thr_shared_t2r.partition_S(tCtShared_mn_view)
        tTR_tCcShared = thr_shared_t2r.partition_D(tCcShared)

        # Dedicated full-tile KS t2r (separate atom so it can be tuned
        # independently of atom_shared_t2r used by NV/decay_v reads).
        atom_ks_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), cutlass.Float32
        )
        tiled_ks_t2r = tcgen05.make_tmem_copy(atom_ks_t2r, tCtShared_for_t2r)
        thr_ks_t2r = tiled_ks_t2r.get_slice(cg1_tidx)
        tTR_tCtKS = thr_ks_t2r.partition_S(tCtShared_mn_view)
        tTR_tCcKS = thr_ks_t2r.partition_D(tCcShared)

        qkv_inp_shape = tiled_mma_qkv.partition_shape_A(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[2])
        )
        tCtShared_inp_fake = tiled_mma_qkv.make_fragment_A(
            cute.append(qkv_inp_shape, self.tmem_shared_inp_stages)
        )
        tCtShared_inp = cute.make_tensor(
            cute.recast_ptr(
                tmem_ptr + self.tmem_shared_inp_offset, dtype=self.io_dtype
            ),
            tCtShared_inp_fake.layout,
        )
        tCtShared_inp_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            tCtShared_inp
        )
        tCcShared_inp = cute.make_identity_tensor(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[2])
        )
        atom_shared_inp_r2t = cute.make_copy_atom(
            tcgen05.copy.St16x128bOp(tcgen05.copy.Repetition(8)), self.io_dtype
        )
        tCtShared_inp_for_r2t = tCtShared_inp_mn_view[None, None, 0]
        tiled_shared_inp_r2t = tcgen05.make_tmem_copy(
            atom_shared_inp_r2t, tCtShared_inp_for_r2t
        )
        thr_shared_inp_r2t = tiled_shared_inp_r2t.get_slice(cg1_tidx)
        tRT_tCtShared_inp = thr_shared_inp_r2t.partition_D(tCtShared_inp_mn_view)

        # Dedicated full-tile VKS r2t (separate atom so it can be tuned
        # independently of atom_shared_inp_r2t used by NV/decay_v writes).
        atom_vks_r2t = cute.make_copy_atom(
            tcgen05.copy.St16x128bOp(tcgen05.copy.Repetition(8)), self.io_dtype
        )
        tiled_vks_r2t = tcgen05.make_tmem_copy(atom_vks_r2t, tCtShared_inp_for_r2t)
        thr_vks_r2t = tiled_vks_r2t.get_slice(cg1_tidx)
        tRT_tCtVKS_inp = thr_vks_r2t.partition_D(tCtShared_inp_mn_view)

        qs_acc_shape = tiled_mma_qs.partition_shape_C(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[1])
        )
        tCtQState_fake = tiled_mma_qs.make_fragment_C(
            cute.append(qs_acc_shape, self.tmem_q_state_acc_stages)
        )
        tCtQState = cute.make_tensor(
            tmem_ptr + self.tmem_q_state_offset, tCtQState_fake.layout
        )
        tCtQState_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            tCtQState
        )
        tCcQState = cute.make_identity_tensor(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[1])
        )
        atom_qs_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), cutlass.Float32
        )
        atom_qs_r2t = cute.make_copy_atom(
            tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(8)), cutlass.Float32
        )
        tCtQState_for_t2r = tCtQState[(None, None), 0, 0, 0]
        tiled_qs_t2r = tcgen05.make_tmem_copy(atom_qs_t2r, tCtQState_for_t2r)
        tCtQState_for_r2t = tCtQState[(None, None), 0, 0, 0]
        tiled_qs_r2t = tcgen05.make_tmem_copy(atom_qs_r2t, tCtQState_for_r2t)
        thr_qs_t2r = tiled_qs_t2r.get_slice(cg1_tidx)
        thr_qs_r2t = tiled_qs_r2t.get_slice(cg1_tidx)
        tTR_tCtQS = thr_qs_t2r.partition_S(tCtQState_mn_view)
        tTR_tCcQS = thr_qs_t2r.partition_D(tCcQState)
        tRT_tCtQS = thr_qs_r2t.partition_D(tCtQState_mn_view)
        tTR_rQS = cute.make_rmem_tensor_like(tTR_tCcQS, cutlass.Float32)

        tRT_tCcV = thr_shared_inp_r2t.partition_S(tCcShared_inp)
        atom_v_s2r = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True),
            self.io_dtype,
        )
        tiled_v_s2r = cute.make_tiled_copy_S(
            atom_v_s2r,
            tiled_shared_inp_r2t,
        )
        thr_v_s2r = tiled_v_s2r.get_slice(cg1_tidx)

        atom_o_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), cutlass.Float32
        )
        tiled_o_t2r = tcgen05.make_tmem_copy(atom_o_t2r, tCtQState_for_t2r)
        thr_o_t2r = tiled_o_t2r.get_slice(cg1_tidx)
        tTR_tOtO = thr_o_t2r.partition_S(tCtQState_mn_view)
        tTR_tOcO = thr_o_t2r.partition_D(tCcQState)
        atom_o_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=True),
            self.io_dtype,
        )
        tiled_o_r2s = cute.make_tiled_copy_D(atom_o_r2s, tiled_o_t2r)
        thr_o_r2s = tiled_o_r2s.get_slice(cg1_tidx)
        tCsO = thr_o_r2s.partition_D(sO)

        sub_tile_size = 32
        sV_vt_view = utils.gemm.sm100.transform_partitioned_tensor_layout(sV)
        tCsV = thr_v_s2r.partition_S(sV_vt_view)

        rCumprod = cute.make_rmem_tensor((1, cute.size(tTR_tCcShared)), cutlass.Float32)
        tGrCumprod = thr_shared_t2r.partition_D(rCumprod)
        tGrDecayScale = cute.make_rmem_tensor_like(tTR_tCcShared, cutlass.Float32)
        # Pair membership is derived from the caller-provided chunk index.
        is_pair_first = (chunk_iter & 1) == 0

        if is_pair_first:
            kv_acc_producer.advance()
            kv_acc_producer.advance()

        # Only the total decay is needed to finish the state update.  Defer
        # the per-row gate work until the previous state has been published
        # and decayed, keeping its register fragment in one contiguous region.
        gate_handle = load_gate_consumer.wait_and_advance()
        # valid_len = max(0, min(self.b_t, seqlen_b - chunk_iter * self.b_t))
        # gamma_end = 1.0 if valid_len == 0 else sCumprod[valid_len - 1, 0, gate_handle.index]
        # OOB alpha is padded with 1 before the prefix scan, so the last physical slot equals gamma_end.
        cumprod_total = sCumprod[self.b_t - 1, 0, gate_handle.index]

        kv_prev_handle = kv_acc_consumer.wait_and_advance()
        # No empty-stage wait is needed before reusing state_inp.  For the
        # first publication the stage is initialized empty.  Thereafter,
        # CG1 reaches this point only after waiting for the preceding NewV;
        # the MMA warp releases state_inp before publishing that NewV.
        state_inp_handle = state_inp_ready_producer.current_handle()
        state_inp_ready_producer.advance()
        cute.copy(
            tiled_state_t2r,
            tTR_tCtState[None, 0, None, kv_prev_handle.index],
            tTR_rState[None, 0, None],
        )
        tRT_rState_inp[None, 0, None].store(
            tTR_rState[None, 0, None].load().to(self.io_dtype)
        )
        cute.copy(
            tiled_state_inp_r2t,
            tRT_rState_inp[None, 0, None],
            tRT_tCtState_inp[None, 0, None, state_inp_handle.index],
        )
        cute.arch.fence_view_async_tmem_store()
        state_inp_handle.commit()
        for k in cutlass.range(cute.size(tTR_rState), vectorize=True):
            tTR_rState[k] = tTR_rState[k] * cumprod_total
        cute.copy(
            tiled_state_r2t,
            tTR_rState[None, 0, None],
            tRT_tCtState[None, 0, None, kv_prev_handle.index],
        )
        cute.arch.fence_view_async_tmem_store()
        kv_prev_handle.release()
        for k in cutlass.range_constexpr(cute.size(tTR_tCcShared)):
            coord = tTR_tCcShared[k]
            tGrCumprod[k] = sCumprod[coord[1], 0, gate_handle.index]
        last_cumsumlog = sCumsumlog[self.b_t - 1, 0, gate_handle.index]
        for k in cutlass.range_constexpr(0, cute.size(tTR_tCcShared), 2):
            coord0 = tTR_tCcShared[k]
            coord1 = tTR_tCcShared[k + 1]
            decay_diff = cute.arch.add_packed_f32x2(
                (last_cumsumlog, last_cumsumlog),
                (
                    -sCumsumlog[coord0[1], 0, gate_handle.index],
                    -sCumsumlog[coord1[1], 0, gate_handle.index],
                ),
                ftz=False,
                rnd="rn",
            )
            tGrDecayScale[k] = cute.math.exp2(decay_diff[0], fastmath=True)
            tGrDecayScale[k + 1] = cute.math.exp2(decay_diff[1], fastmath=True)
        gate_handle.release()

        vks_handle = vks_ready_producer.current_handle()
        vks_ready_producer.advance()
        v_handle = load_v_consumer.wait_and_advance()
        tRT_rV = cute.make_rmem_tensor_like(tRT_tCcV, self.io_dtype)
        tCrV = tiled_v_s2r.retile(tRT_rV)
        # Always publish V through fixed TMEM slot 0. The SMEM V ring cursor
        # survives persistent work boundaries, so a new work item cannot
        # assume that its first V tile resides in SMEM stage 0.
        cute.copy(
            tiled_v_s2r,
            tCsV[None, None, None, v_handle.index],
            tCrV,
        )

        ks_handle = cg1_shared_acc_consumer.wait_and_advance()
        tTR_rKS = cute.make_rmem_tensor_like(tTR_tCcKS, cutlass.Float32)
        cute.copy(
            tiled_ks_t2r,
            tTR_tCtKS[None, None, None, ks_handle.index],
            tTR_rKS,
        )
        for k in cutlass.range(cute.size(tTR_rKS), vectorize=True):
            tTR_rKS[k] = tTR_rKS[k] * tGrCumprod[k]
        ks_handle.release()
        for k in cutlass.range(cute.size(tTR_rKS), vectorize=True):
            tRT_rV[k] = tRT_rV[k] - tTR_rKS[k].to(self.io_dtype)
        cute.copy(
            tiled_vks_r2t,
            tRT_rV,
            tRT_tCtVKS_inp[None, None, None, 0],
        )
        cute.arch.fence_view_async_tmem_store()
        vks_handle.commit()

        qs_handle = q_state_acc_consumer.wait_and_advance()
        cute.copy(
            tiled_qs_t2r,
            tTR_tCtQS[None, None, 0, qs_handle.index],
            tTR_rQS[None, None, 0],
        )
        for k in cutlass.range(cute.size(tTR_rQS), vectorize=True):
            tTR_rQS[k] = tTR_rQS[k] * tGrCumprod[k] * scale
        cute.copy(
            tiled_qs_r2t,
            tTR_rQS[None, None, 0],
            tRT_tCtQS[None, None, 0, qs_handle.index],
        )
        cute.arch.fence_view_async_tmem_store()
        qs_handle.release()

        nv_handle = cg1_shared_acc_consumer.wait_and_advance()
        v_handle.release()
        tTR_rNv = cute.make_rmem_tensor_like(tTR_tCcShared, cutlass.Float32)
        tTR_rNv_inp = cute.make_rmem_tensor_like(tTR_rNv, self.io_dtype)
        for sub in cutlass.range(tTR_rNv.shape[1]):
            cute.copy(
                tiled_shared_t2r,
                tTR_tCtShared[None, sub, 0, nv_handle.index],
                tTR_rNv[None, sub, 0],
            )
            tTR_rNv_inp[None, sub, 0].store(
                tTR_rNv[None, sub, 0].load().to(self.io_dtype)
            )
        nv_handle.release()

        tTR_rDv = tTR_rNv
        for sub in cutlass.range(tTR_rDv.shape[1]):
            for k in cutlass.range(sub_tile_size, vectorize=True):
                tTR_rDv[k, sub, 0] = tTR_rDv[k, sub, 0] * tGrDecayScale[k, sub, 0]

        nv_ready_handle = nv_ready_producer.current_handle()
        nv_ready_producer.advance()
        decay_v_ready_handle = decay_v_ready_producer.current_handle()
        decay_v_ready_producer.advance()
        # Both chunks publish NV for GEMM6/QKV before decayV for GEMM7/KV.
        for sub in cutlass.range(tTR_rDv.shape[1]):
            cute.copy(
                tiled_shared_inp_r2t,
                tTR_rNv_inp[None, sub, 0],
                tRT_tCtShared_inp[None, sub, 0, 0],
            )
            tTR_rNv_inp[None, sub, 0].store(
                tTR_rDv[None, sub, 0].load().to(self.io_dtype)
            )
            cute.copy(
                tiled_shared_inp_r2t,
                tTR_rNv_inp[None, sub, 0],
                tRT_tCtShared_inp[None, sub, 0, 1],
            )
        cute.arch.fence_view_async_tmem_store()
        nv_ready_handle.commit()
        decay_v_ready_handle.commit()

        # Drain this chunk's output at the end of the same chunk.  This keeps
        # O ownership uniform and removes the first/last pending-output cases.
        o_handle = o_store_producer.acquire_and_advance()
        o_qs_handle = q_state_acc_consumer.wait_and_advance()
        tTR_tOrO = cute.make_rmem_tensor_like(tTR_tOcO, cutlass.Float32)
        tTR_rO_out = cute.make_rmem_tensor_like(tTR_tOrO, self.io_dtype)
        tRS_tOrO = tiled_o_r2s.retile(tTR_rO_out)
        cute.copy(
            tiled_o_t2r,
            tTR_tOtO[None, None, None, o_qs_handle.index],
            tTR_tOrO,
        )
        tTR_rO_out.store(tTR_tOrO.load().to(self.io_dtype))
        cute.copy(
            tiled_o_r2s,
            tRS_tOrO,
            tCsO[None, None, None, o_handle.index],
        )
        cute.arch.fence_view_async_shared()
        o_qs_handle.release()
        o_handle.commit()

        return (
            load_v_consumer,
            load_gate_consumer,
            cg1_shared_acc_consumer,
            kv_acc_consumer,
            q_state_acc_consumer,
            kv_acc_producer,
            state_inp_ready_producer,
            vks_ready_producer,
            nv_ready_producer,
            decay_v_ready_producer,
            o_store_producer,
        )

    @cute.jit
    def epilogue_warp(
        self,
        smem_args,
        tma_args,
        pipeline_args,
        work_args,
        tensormap_args,
    ) -> pipeline.PipelineConsumer:
        """Warp 11: TMA bulk-store O from SMEM staging buffer to global memory.

        Steps:
          1. Wait for CG1 to signal O is ready in sO (via o_store_consumer).
          2. Domain-offset the TMA tensor to (chunk_offset, head_idx), flat-divide
             into tiles, tma_partition -> (tOsO, tOgO).
          3. Issue TMA S2G bulk copy using the per-work-tile updated descriptor.
          4. Commit the async group and wait for the store to land in GMEM.
          5. Release the pipeline slot back to CG1.
        """
        (sO,) = smem_args
        (tma_o,) = tma_args
        (o_store_consumer,) = pipeline_args
        head_idx, chunk_offset = work_args
        tensormap_manager, tensormap_o_ptr = tensormap_args

        o_handle = o_store_consumer.wait_and_advance()

        cta_layout = cute.make_layout(1)
        # (BT, DV)
        o_tile = cute.select(self.mma_tiler_qkv, mode=[0, 1])

        # Position global O tile at current chunk / head
        mO = cute.domain_offset(
            (cutlass.Int32(0), chunk_offset),
            tma_o.tma_tensor[None, None, head_idx],
        )
        # (BT, DV, num_o_tiles, ...)
        gO = cute.flat_divide(mO, o_tile)

        # TMA partition: tOsO = SMEM source, tOgO = GMEM destination
        tOsO, tOgO = cpasync.tma_partition(
            tma_o.atom,
            0,
            cta_layout,
            cute.group_modes(sO, 0, 2),
            cute.group_modes(gO, 0, 2),
        )

        # TMA bulk store SMEM -> GMEM using the descriptor updated per work tile
        cute.copy(
            tma_o.atom,
            tOsO[(None, o_handle.index)],
            tOgO[(None, 0, 0)],
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                tensormap_o_ptr, cute.AddressSpace.generic
            ),
        )

        # Wait for the store to complete before releasing the SMEM slot
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0)

        o_handle.release()

        return o_store_consumer

    @cute.jit
    def _transform_to_position_independent_layout(
        self, tensor: cute.Tensor, swizzle_inner: cute.Swizzle
    ) -> cute.Tensor:
        wo_swizzle_iter = cute.recast_ptr(tensor.iterator, swizzle_=None)
        pisl_swizzle_base = int(math.log2(self.io_dtype.width)) - 1
        pisl_swizzle = cute.make_swizzle(
            swizzle_inner.num_bits, pisl_swizzle_base, swizzle_inner.num_shift
        )
        tensor_pisl = cute.make_composed_layout(pisl_swizzle, 0, tensor.layout)
        return cute.make_tensor(wo_swizzle_iter, tensor_pisl)

    @staticmethod
    def get_workspace_size(num_sm: int):
        return (
            GatedDeltaNetChunkedKernel.bytes_per_tensormap
            * GatedDeltaNetChunkedKernel.num_tensormaps
            * num_sm
        )

    @cute.jit
    def initialize_workspace(
        self, workspace: cute.Tensor, grid_dim: Tuple[int, int, int]
    ):
        workspace = cute.make_tensor(
            workspace.iterator,
            cute.make_layout(
                (
                    grid_dim[0] * grid_dim[1] * grid_dim[2],
                    GatedDeltaNetChunkedKernel.num_tensormaps,
                    GatedDeltaNetChunkedKernel.bytes_per_tensormap,
                ),
                stride=(
                    GatedDeltaNetChunkedKernel.num_tensormaps
                    * GatedDeltaNetChunkedKernel.bytes_per_tensormap,
                    GatedDeltaNetChunkedKernel.bytes_per_tensormap,
                    1,
                ),
            ),
        )
        return workspace


# ---------------------------------------------------------------------------
# Test / validation entry point
# ---------------------------------------------------------------------------
