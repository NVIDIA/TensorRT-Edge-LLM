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
"""
QsaAttentionPlugin (Qwen Sparse Attention, prefill + decode) unit tests.

Two layers of coverage:

1. ``QsaTorchReference`` / ``QsaDecodeTorchReference`` self-tests (pure
   torch, CPU, no TensorRT / CUDA): the dense-equivalence invariant (for
   S <= 2051 the sparse selection is exactly the full causal prefix, so
   sparse output == causal SDPA), the structural invariants of the produced
   index lists, and the decode-step == prefill-row equivalence of the
   stateful decode reference.

2. Plugin tests driving the TRT ``QsaAttentionPlugin`` through the
   ``PluginRunner`` harness on CUDA: dense-equivalence prefill, sparse
   prefill vs the full FP32 QSA torch reference, paged-KV write-through,
   determinism, and the decode mode — prefill + N single-token steps (flat
   and ragged batches), persisted pool-tail indexer state, NaN tail
   poisoning, S > 1 rejection, and a CUDA-graph capture smoke test.

Algorithm contract under test (frozen with the C++ plugin):

* Main path: split packed qkv [B,S,(Hq+2*Hkv)*D] -> per-head Gemma qk-norm
  (``normalize(x) * gamma`` with gamma PRE-FOLDED as (1+w)) -> partial neox
  RoPE (rotary dim 64 of head size 256, theta=1e7) -> paged KV write
  (roped K, raw V).
* Indexer (NH=4, D_idx=128, ratio=4, block_topk=512 / budget=2048 tokens, width=2051): q Gemma-norm
  with RAW w (kernel computes 1+w internally) + rope@t; compressed keys per
  complete block g: fp32 mean of 4 raw keys (fixed order) -> cast fp16 ->
  Gemma-norm -> rope@4g; scores = sum_h relu(q_h . kbar_g) / sqrt(128) in
  FP32, visible g < (t+1)//4; top-512 blocks -> expand blk*4+[0..3] ->
  ALWAYS append tail [4*n_vis .. t] ((t+1)%4 tokens) -> -1 pad to 2051.
* Sparse GQA: attend only listed tokens; FP32 softmax at 1/sqrt(256) applied
  on FP32 scores after the QK dot; -1 masked; all-invalid row -> zeros; rows
  past context_lengths -> exact zeros.
* Decode (kvcache_start_index runtime shape [B] — values are the past
  lengths, never dereferenced — REQUIRES S == 1; context_lengths = TOTAL
  lengths including the new token): rope appends K/V at pool row ctx-1; the
  indexer either persists the new token's raw index-K (indexQk columns
  [512:640)) BIT-UNMODIFIED into its K-row tail (head 0, pool columns
  [256:384) of the widened row) while its block is still incomplete
  (ctx % 4 != 0), or, when ctx % 4 == 0, compresses the completed block from
  the three persisted raw keys plus the new one and writes kbar into the
  V-row tail of token 4g (the completing token's raw key is never stored);
  then top-512 select + expand to outIdx [B, 1, 2051] and split-K sparse
  attention over the paged pool. Prefill persists every complete block's
  kbar and only the trailing incomplete block's raw keys, so decode-era
  indexer state is bit-compatible with prefill (see qsaIndexerKernels.h).

Top-k tie-break caveat: block scores are compared in FP32, and the plugin's
warp-reduction accumulation order differs from torch's, so blocks whose
scores sit exactly at the top-512 boundary may be swapped between plugin and
reference for S > 2051. All tests use continuous random inputs (ties are
measure-zero) and the sparse comparisons use fp16 tolerances plus a cosine
gate instead of exact index-list equality.

Run:
    python3 -m pytest tests/python-unittests/test_qsa_attention_plugin.py -v
Reference-only (no GPU needed):
    python3 -m pytest tests/python-unittests/test_qsa_attention_plugin.py -k reference -v
"""

from __future__ import annotations

import math

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              PluginRunner, _device_sm, _get_logger,
                              assert_close, cosine_sim, find_plugin_library,
                              load_edgellm_plugins, pf_float32, pf_int32,
                              poison_padding)

# The reference self-tests only need torch (CPU is fine); the plugin tests
# additionally need TensorRT + CUDA (DEPENDENCIES_AVAILABLE from the harness).
try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - exercised only without torch
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(e)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt

DEV = "cuda"

requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason=f"torch not available: {TORCH_IMPORT_ERROR}")
requires_gpu = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE,
    reason=f"TensorRT/torch CUDA not available: {IMPORT_ERROR}")

# --------------------------------------------------------------------------- #
# Frozen QSA model constants (Qwen3.8-Flash-Next qwen4_exp)
# --------------------------------------------------------------------------- #
NUM_Q_HEADS = 24
NUM_KV_HEADS = 2
HEAD_SIZE = 256
INDEXER_N_HEADS = 4
INDEXER_HEAD_DIM = 128
INDEXER_COMPRESS_RATIO = 4
# Top blocks selected per query row (kQSA_BLOCK_TOPK).
INDEXER_BLOCK_TOPK = 512
# The plugin's indexer_budget attribute counts EXPANDED TOKENS
# (kQSA_INDEX_BUDGET = block_topk * ratio), not blocks.
INDEXER_BUDGET = INDEXER_BLOCK_TOPK * INDEXER_COMPRESS_RATIO  # 2048
# Widest possible index row: the token budget + a (ratio-1)-token tail.
INDEX_WIDTH = INDEXER_BUDGET + (INDEXER_COMPRESS_RATIO - 1)  # 2051
ROTARY_DIM = 64  # partial rope: first 64 dims of each head
ROPE_THETA = 1.0e7
RMS_NORM_EPS = 1e-6
# Tokens per page of the paged-KV pool. Must match kTOKENS_PER_PAGE.
PAGE_SIZE = 128
# Widened pool row: [roped K or raw V (HEAD_SIZE) | indexer-state tail]. The
# plugin DERIVES it as head_size + indexer_head_dim (no separate attribute);
# the rope kernel writes only the leading HEAD_SIZE columns of each row (see
# applyRopeWriteKV.h).
POOL_HEAD_DIM = HEAD_SIZE + INDEXER_HEAD_DIM  # 384

QKV_PACKED = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_SIZE  # 7168
INDEX_QK_PACKED = (INDEXER_N_HEADS + 1) * INDEXER_HEAD_DIM  # 640

# QSA CuTe-DSL sparse kernel variants are registered for these SMs only.
QSA_SMS = frozenset({100, 101, 110})


# --------------------------------------------------------------------------- #
# RoPE table + reference math helpers
# --------------------------------------------------------------------------- #
def make_qsa_rope_table(max_pos: int, device="cpu") -> "torch.Tensor":
    """Build the shared QSA cos/sin table [max_pos, 64] (FP32).

    Layout matches the plugin binding (and test_attention_plugin's
    ``_make_rope``): cos in [0:32], sin in [32:64];
    ``inv_freq[i] = theta**(-i/half)`` with theta=1e7, half=32.
    """
    half = ROTARY_DIM // 2
    pos = torch.arange(max_pos, dtype=torch.float32, device=device)[:, None]
    inv_freq = 1.0 / (ROPE_THETA**(torch.arange(
        0, half, dtype=torch.float32, device=device) / half))[None, :]
    ang = pos * inv_freq  # [max_pos, half]
    table = torch.zeros((max_pos, ROTARY_DIM),
                        dtype=torch.float32,
                        device=device)
    table[:, :half] = torch.cos(ang)
    table[:, half:] = torch.sin(ang)
    return table


def apply_partial_rope(x32: "torch.Tensor",
                       table: "torch.Tensor",
                       positions: "torch.Tensor",
                       interleaved: bool = False) -> "torch.Tensor":
    """Partial RoPE on the first ROTARY_DIM dims of the last axis (FP32).

    ``x32``: [N, ..., Dh] with the position axis first; ``positions``: [N]
    int64; ``table``: [max_pos, 64] from :func:`make_qsa_rope_table`.

    ``interleaved=False`` (default) is the neox rotate-half convention:
    pairs are (dim i, dim i+32) within the first 64 dims. ``interleaved=True``
    pairs (dim 2i, dim 2i+1) — kept as a one-line flip in case the HF
    qwen4_exp cross-check pins the other layout.
    """
    half = ROTARY_DIM // 2
    cos = table[positions, :half].float()
    sin = table[positions, half:].float()
    bshape = [x32.shape[0]] + [1] * (x32.dim() - 2) + [half]
    cos = cos.reshape(bshape)
    sin = sin.reshape(bshape)
    rot = x32[..., :ROTARY_DIM]
    if interleaved:
        x1, x2 = rot[..., 0::2], rot[..., 1::2]
    else:
        x1, x2 = rot[..., :half], rot[..., half:]
    r1 = x1 * cos - x2 * sin
    r2 = x1 * sin + x2 * cos
    if interleaved:
        roped = torch.stack((r1, r2), dim=-1).reshape(rot.shape)
    else:
        roped = torch.cat((r1, r2), dim=-1)
    return torch.cat((roped, x32[..., ROTARY_DIM:]), dim=-1)


def gemma_rms_norm(x32: "torch.Tensor", gamma32: "torch.Tensor",
                   eps: float) -> "torch.Tensor":
    """FP32 RMSNorm ``x * rsqrt(mean(x^2) + eps) * gamma`` (llama convention).

    The Gemma ``(1+w)`` semantics live in the caller: main-path gammas arrive
    pre-folded, indexer gammas arrive raw and the caller adds the 1.
    """
    return x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * gamma32


# --------------------------------------------------------------------------- #
# Pure-torch FP32 QSA reference
# --------------------------------------------------------------------------- #
class QsaTorchReference:
    """FP32 torch reference for the full QSA plugin math (prefill).

    Follows the frozen cast chain: indexer q/kbar are stored through an FP16
    cast (like the CUDA kernels) before the FP32 score dot; compressed keys
    take the fixed-order fp32 mean -> fp16 cast -> Gemma-norm -> rope@4g
    chain. The main path keeps FP32 throughout (the plugin's fp16 rounding is
    covered by the comparison tolerances).
    """

    def __init__(self,
                 num_q_heads: int = NUM_Q_HEADS,
                 num_kv_heads: int = NUM_KV_HEADS,
                 head_size: int = HEAD_SIZE,
                 indexer_n_heads: int = INDEXER_N_HEADS,
                 indexer_head_dim: int = INDEXER_HEAD_DIM,
                 indexer_block_topk: int = INDEXER_BLOCK_TOPK,
                 indexer_compress_ratio: int = INDEXER_COMPRESS_RATIO,
                 attention_scale: float = 0.0,
                 rms_norm_eps: float = RMS_NORM_EPS,
                 interleaved: bool = False):
        assert num_q_heads % num_kv_heads == 0
        self.hq = num_q_heads
        self.hkv = num_kv_heads
        self.d = head_size
        self.nh = indexer_n_heads
        self.di = indexer_head_dim
        self.budget = indexer_block_topk  # blocks per row
        self.ratio = indexer_compress_ratio
        self.width = indexer_block_topk * indexer_compress_ratio + (
            indexer_compress_ratio - 1)
        # attention_scale == 0.0 selects the plugin default 1/sqrt(head_size).
        self.scale = (attention_scale if attention_scale != 0.0 else 1.0 /
                      math.sqrt(head_size))
        self.eps = rms_norm_eps
        self.interleaved = interleaved

    # -- main path ---------------------------------------------------------- #
    def main_qkv(self, qkv, q_gamma_folded, k_gamma_folded, table):
        """Split packed qkv, qk-norm (pre-folded gammas), partial rope.

        qkv [B,S,(Hq+2*Hkv)*D] (any float dtype); gammas [D] hold (1+w).
        Returns FP32 (q_roped, k_roped, v) each [B,S,H,D].
        """
        b, s, _ = qkv.shape
        qh = self.hq * self.d
        kvh = self.hkv * self.d
        q = qkv[..., :qh].reshape(b, s, self.hq, self.d).float()
        k = qkv[..., qh:qh + kvh].reshape(b, s, self.hkv, self.d).float()
        v = qkv[..., qh + kvh:].reshape(b, s, self.hkv, self.d).float()
        q = gemma_rms_norm(q, q_gamma_folded.float(), self.eps)
        k = gemma_rms_norm(k, k_gamma_folded.float(), self.eps)
        positions = torch.arange(s, device=qkv.device, dtype=torch.long)
        q = torch.stack([
            apply_partial_rope(q[bi], table, positions, self.interleaved)
            for bi in range(b)
        ])
        k = torch.stack([
            apply_partial_rope(k[bi], table, positions, self.interleaved)
            for bi in range(b)
        ])
        return q, k, v

    # -- indexer ------------------------------------------------------------ #
    def compress_kbar(self, k_raw, ik_gamma_raw, table):
        """Compressed keys of every complete block of one sequence.

        ``k_raw`` [L, D_idx] FP32 raw index-K history. Returns FP32
        [L // ratio, D_idx] (values exact fp16): FIXED-ORDER fp32 mean
        (((k0+k1)+k2)+k3) * 0.25 -> cast fp16 -> Gemma-norm (1 + w_raw) ->
        rope@4g -> fp16 storage cast. Bit-compat-critical: the decode-path
        recompress from the persisted raw-K pool tails must land on the
        same bits as the prefill compress.
        """
        dev = k_raw.device
        num_blocks = k_raw.shape[0] // self.ratio
        if num_blocks == 0:
            return torch.zeros((0, self.di), dtype=torch.float32, device=dev)
        kg = k_raw[:num_blocks * self.ratio].reshape(num_blocks, self.ratio,
                                                     self.di)
        mean = kg[:, 0]
        for j in range(1, self.ratio):
            mean = mean + kg[:, j]
        mean = (mean * (1.0 / self.ratio)).to(torch.float16).float()
        kbar = gemma_rms_norm(mean, 1.0 + ik_gamma_raw.float(), self.eps)
        block_pos = torch.arange(num_blocks, device=dev,
                                 dtype=torch.long) * self.ratio
        kbar = apply_partial_rope(kbar, table, block_pos, self.interleaved)
        return kbar.to(torch.float16).float()

    def build_indices(self, index_qk, iq_gamma_raw, ik_gamma_raw, table,
                      context_lengths):
        """Full indexer pipeline -> int32 index lists [B, S, width] (-1 pad).

        ``iq_gamma_raw`` / ``ik_gamma_raw`` [D_idx] hold RAW w; the effective
        Gemma gamma is (1 + w) computed here in FP32 (as the CUDA kernel does).
        Rows past a request's context length are all -1.
        """
        b, s, _ = index_qk.shape
        dev = index_qk.device
        out = torch.full((b, s, self.width), -1, dtype=torch.int32, device=dev)
        iq_gamma = 1.0 + iq_gamma_raw.float()
        ar_ratio = torch.arange(self.ratio, device=dev, dtype=torch.int64)
        for bi in range(b):
            length = int(context_lengths[bi])
            if length <= 0:
                continue
            q_raw = index_qk[bi, :length, :self.nh * self.di].reshape(
                length, self.nh, self.di).float()
            k_raw = index_qk[bi, :length,
                             self.nh * self.di:].reshape(length,
                                                         self.di).float()
            positions = torch.arange(length, device=dev, dtype=torch.long)
            # q: gemma-norm (1 + w_raw) -> rope@t -> fp16 storage cast.
            q_idx = gemma_rms_norm(q_raw, iq_gamma, self.eps)
            q_idx = apply_partial_rope(q_idx, table, positions,
                                       self.interleaved)
            q_idx = q_idx.to(torch.float16).float()
            num_blocks = length // self.ratio  # complete blocks only
            n_vis = (positions + 1) // self.ratio  # [length]
            top_idx = None
            if num_blocks > 0:
                kbar = self.compress_kbar(k_raw, ik_gamma_raw, table)
                # scores[t, g] = sum_h relu(q[t,h] . kbar[g]) / sqrt(D_idx):
                # relu per head, THEN sum over heads, THEN scale — all FP32.
                dots = torch.einsum("thd,gd->thg", q_idx, kbar)
                scores = torch.relu(dots).sum(dim=1) / math.sqrt(self.di)
                visible = (torch.arange(num_blocks, device=dev)[None, :]
                           < n_vis[:, None])
                scores = scores.masked_fill(~visible, float("-inf"))
                k_top = min(self.budget, num_blocks)
                # topk sorts descending; -inf (masked) blocks rank after every
                # finite score, so slicing to n_sel keeps only visible blocks.
                top_idx = torch.topk(scores, k_top, dim=-1).indices
            for t in range(length):
                nv = int(n_vis[t])
                n_sel = min(self.budget, nv)
                parts = []
                if n_sel > 0:
                    blocks = top_idx[t, :n_sel].to(torch.int64)
                    parts.append(
                        (blocks[:, None] * self.ratio + ar_ratio).reshape(-1))
                tail_start = nv * self.ratio
                if tail_start <= t:  # (t+1) % ratio tokens, ALWAYS appended
                    parts.append(
                        torch.arange(tail_start,
                                     t + 1,
                                     device=dev,
                                     dtype=torch.int64))
                row = torch.cat(parts) if parts else None
                if row is not None and row.numel() > 0:
                    out[bi, t, :row.numel()] = row.to(torch.int32)
        return out

    # -- sparse attention ---------------------------------------------------- #
    def sparse_attention(self,
                         q,
                         k,
                         v,
                         indices,
                         context_lengths,
                         row_chunk: int = 64):
        """Sparse GQA over the listed tokens only (no causal mask in here).

        q [B,S,Hq,D], k/v [B,S,Hkv,D] FP32; indices [B,S,width] int32 (-1 =
        masked). FP32 softmax; scale applied on the FP32 scores after the QK
        dot. All-invalid rows and rows past context_lengths -> exact zeros.
        """
        b, s, hq, d = q.shape
        group = self.hq // self.hkv
        out = torch.zeros((b, s, hq, d), dtype=torch.float32, device=q.device)
        for bi in range(b):
            length = int(context_lengths[bi])
            for r0 in range(0, length, row_chunk):
                r1 = min(r0 + row_chunk, length)
                idx = indices[bi, r0:r1].to(torch.int64)  # [R, W]
                valid = idx >= 0
                idx_cl = idx.clamp(min=0)
                kg = k[bi][idx_cl]  # [R, W, Hkv, D]
                vg = v[bi][idx_cl]
                qc = q[bi, r0:r1].reshape(r1 - r0, self.hkv, group, d)
                scores = torch.einsum("rkgd,rwkd->rkgw", qc, kg) * self.scale
                scores = scores.masked_fill(~valid[:, None, None, :],
                                            float("-inf"))
                probs = torch.softmax(scores, dim=-1)
                # All-invalid rows softmax to NaN -> force exact zero output.
                probs = torch.nan_to_num(probs, nan=0.0)
                out[bi, r0:r1] = torch.einsum("rkgw,rwkd->rkgd", probs,
                                              vg).reshape(r1 - r0, hq, d)
        return out

    def dense_attention(self, q, k, v, context_lengths):
        """Causal dense SDPA on the same post-norm/rope q/k/v (FP32).

        The dense-equivalence variant: for context lengths <= width the
        sparse selection is exactly the causal prefix, so this must match
        :meth:`sparse_attention`. Padding rows -> exact zeros.
        """
        b, s, hq, d = q.shape
        group = self.hq // self.hkv
        out = torch.zeros((b, s, hq, d), dtype=torch.float32, device=q.device)
        for bi in range(b):
            length = int(context_lengths[bi])
            if length <= 0:
                continue
            qb = q[bi, :length].permute(1, 0, 2)  # [Hq, L, D]
            kb = k[bi, :length].permute(1, 0, 2).repeat_interleave(group,
                                                                   dim=0)
            vb = v[bi, :length].permute(1, 0, 2).repeat_interleave(group,
                                                                   dim=0)
            scores = torch.matmul(qb, kb.transpose(-1, -2)) * self.scale
            causal = torch.ones((length, length),
                                dtype=torch.bool,
                                device=q.device).tril()
            scores = scores.masked_fill(~causal[None], float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[bi, :length] = torch.matmul(probs, vb).permute(1, 0, 2)
        return out

    # -- full forward -------------------------------------------------------- #
    def forward(self, qkv, index_qk, q_gamma_folded, k_gamma_folded,
                iq_gamma_raw, ik_gamma_raw, table, context_lengths):
        """Full QSA forward. Returns (out, indices, k_roped, v) — k/v FP32
        [B,S,Hkv,D] as the plugin stores them (roped K, raw V)."""
        q, k, v = self.main_qkv(qkv, q_gamma_folded, k_gamma_folded, table)
        indices = self.build_indices(index_qk, iq_gamma_raw, ik_gamma_raw,
                                     table, context_lengths)
        out = self.sparse_attention(q, k, v, indices, context_lengths)
        return out, indices, k, v

    def dense_forward(self, qkv, q_gamma_folded, k_gamma_folded, table,
                      context_lengths):
        """Causal-dense variant of :meth:`forward` (no indexer)."""
        q, k, v = self.main_qkv(qkv, q_gamma_folded, k_gamma_folded, table)
        return self.dense_attention(q, k, v, context_lengths)


class QsaDecodeTorchReference:
    """Stateful decode-step reference over a full token history (pure torch).

    Sequence b's token t is row t of the given history tensors: prefill
    consumed rows [0, L0_b) and decode step s consumes row L0_b + s. For any
    TOTAL context length ctx (including the token being decoded) the class
    reproduces the plugin decode contract with exactly the prefill
    reference's numerics restricted to the last row — the plugin persists
    raw index-K tails bit-unmodified and (re)compresses kbar through the
    same fp16 chain, so decode-era indexer state is bit-compatible with
    prefill and the decode row must equal prefill row ctx - 1 (pinned by
    ``test_reference_decode_step_matches_prefill_rows``).

    Rows past a sequence's own history (ragged batches) are never read:
    every access is bounded by that sequence's ctx.
    """

    def __init__(self, ref: QsaTorchReference, qkv_hist, index_qk_hist,
                 q_gamma_folded, k_gamma_folded, iq_gamma_raw, ik_gamma_raw,
                 table):
        self.ref = ref
        batch, length, _ = index_qk_hist.shape
        dev = index_qk_hist.device
        # Indexer state: q rows Gemma-normed (1 + w_raw) -> rope@t -> fp16
        # cast; kbar per complete block through the shared compress chain.
        q_raw = index_qk_hist[..., :ref.nh * ref.di].reshape(
            batch, length, ref.nh, ref.di).float()
        k_raw = index_qk_hist[..., ref.nh * ref.di:].float()
        positions = torch.arange(length, device=dev, dtype=torch.long)
        q_idx = gemma_rms_norm(q_raw, 1.0 + iq_gamma_raw.float(), ref.eps)
        q_idx = torch.stack([
            apply_partial_rope(q_idx[bi], table, positions, ref.interleaved)
            for bi in range(batch)
        ])
        self.q_idx = q_idx.to(torch.float16).float()  # [B, L, NH, D_idx]
        self.kbar = torch.stack([
            ref.compress_kbar(k_raw[bi], ik_gamma_raw, table)
            for bi in range(batch)
        ])  # [B, L // ratio, D_idx]
        # Main-path q/k/v feed only step_output — computed lazily so
        # index-only checks (the long-context self-test) stay cheap.
        self._main_args = (qkv_hist, q_gamma_folded, k_gamma_folded, table)
        self._main = None

    def _main_qkv(self):
        if self._main is None:
            self._main = self.ref.main_qkv(*self._main_args)
        return self._main

    def block_scores(self, bi: int, ctx: int) -> "torch.Tensor":
        """FP32 block logits of the decode row: [ctx // ratio] (may be
        empty). Only complete blocks are visible: n_vis = ctx // ratio."""
        n_vis = ctx // self.ref.ratio
        dots = torch.einsum("hd,gd->hg", self.q_idx[bi, ctx - 1],
                            self.kbar[bi, :n_vis])
        return torch.relu(dots).sum(dim=0) / math.sqrt(self.ref.di)

    def step_indices(self, bi: int, ctx: int) -> "torch.Tensor":
        """Decode-row token indices (int64; all valid — no -1 padding):
        min(512, ctx // 4) top-score blocks expanded to 4 tokens each in
        descending-logit order, then the ctx % 4 tail tokens."""
        ref = self.ref
        dev = self.q_idx.device
        n_vis = ctx // ref.ratio
        parts = []
        if n_vis > 0:
            n_sel = min(ref.budget, n_vis)
            blocks = torch.topk(self.block_scores(bi, ctx),
                                n_sel).indices.to(torch.int64)
            ar = torch.arange(ref.ratio, device=dev, dtype=torch.int64)
            parts.append((blocks[:, None] * ref.ratio + ar).reshape(-1))
        if n_vis * ref.ratio < ctx:  # ctx % ratio tokens, ALWAYS appended
            parts.append(
                torch.arange(n_vis * ref.ratio,
                             ctx,
                             device=dev,
                             dtype=torch.int64))
        return torch.cat(parts)

    def step_output(self, bi: int, ctx: int):
        """(FP32 attention row [Hq, D] at position ctx - 1, token indices).

        Sparse GQA over the listed tokens only; FP32 softmax with the scale
        applied on the FP32 scores after the QK dot (as the plugin does).
        """
        ref = self.ref
        idx = self.step_indices(bi, ctx)
        q, k, v = self._main_qkv()
        group = ref.hq // ref.hkv
        qc = q[bi, ctx - 1].reshape(ref.hkv, group, ref.d)
        kg = k[bi][idx]  # [N, Hkv, D]
        vg = v[bi][idx]
        scores = torch.einsum("kgd,nkd->kgn", qc, kg) * ref.scale
        probs = torch.softmax(scores, dim=-1)  # every listed token is valid
        out = torch.einsum("kgn,nkd->kgd", probs, vg).reshape(ref.hq, ref.d)
        return out, idx


# --------------------------------------------------------------------------- #
# Shared input builders
# --------------------------------------------------------------------------- #
def _make_gammas(gen, device="cpu"):
    """Random Gemma weights. Returns (q_folded, k_folded, iq_raw, ik_raw) FP16.

    Main-path gammas are PRE-FOLDED (1 + w) — exactly what the plugin (and
    reference) multiply by. Indexer gammas are RAW w — the kernel (and
    reference) compute (1 + w) internally.
    """
    w_q = 0.25 * torch.randn(HEAD_SIZE, generator=gen, dtype=torch.float32)
    w_k = 0.25 * torch.randn(HEAD_SIZE, generator=gen, dtype=torch.float32)
    w_iq = 0.25 * torch.randn(
        INDEXER_HEAD_DIM, generator=gen, dtype=torch.float32)
    w_ik = 0.25 * torch.randn(
        INDEXER_HEAD_DIM, generator=gen, dtype=torch.float32)
    return ((1.0 + w_q).to(torch.float16).to(device), (1.0 + w_k).to(
        torch.float16).to(device), w_iq.to(torch.float16).to(device),
            w_ik.to(torch.float16).to(device))


def _make_activations(batch, seq, gen, device="cpu"):
    """Random packed qkv / index_qk (FP16), generated on CPU for determinism."""
    qkv = torch.randn((batch, seq, QKV_PACKED),
                      generator=gen,
                      dtype=torch.float32).to(device).to(torch.float16)
    index_qk = torch.randn((batch, seq, INDEX_QK_PACKED),
                           generator=gen,
                           dtype=torch.float32).to(device).to(torch.float16)
    return qkv, index_qk


# =========================================================================== #
# A/B. Reference self-tests — pure torch on CPU, no plugin / CUDA required
# =========================================================================== #
@requires_torch
@pytest.mark.parametrize("seq_lens", [[1], [4], [33], [129, 87]],
                         ids=lambda sl: "L" + "_".join(map(str, sl)))
def test_reference_dense_equivalence(seq_lens):
    """Invariant: for L <= 2051, sparse reference == causal SDPA reference.

    n_vis = (t+1)//4 <= 512 for every t < 2051, so top-512 keeps ALL visible
    blocks and expansion + tail reproduce exactly the causal prefix {0..t}.
    """
    torch.manual_seed(20260829)
    gen = torch.Generator().manual_seed(20260829)
    batch = len(seq_lens)
    seq = max(seq_lens)
    ref = QsaTorchReference()
    q_g, k_g, iq_g, ik_g = _make_gammas(gen)
    qkv, index_qk = _make_activations(batch, seq, gen)
    ctx = torch.tensor(seq_lens, dtype=torch.int32)
    table = make_qsa_rope_table(max(seq, 64))

    sparse_out, indices, _, _ = ref.forward(qkv, index_qk, q_g, k_g, iq_g,
                                            ik_g, table, ctx)
    dense_out = ref.dense_forward(qkv, q_g, k_g, table, ctx)

    # The token sets are identical; only the softmax/matmul accumulation
    # order differs (the sparse list is unsorted), hence the tiny fp32 slack.
    torch.testing.assert_close(sparse_out, dense_out, rtol=1e-5, atol=1e-5)

    # Short-seq index rows must be exactly the causal prefix (as a set).
    for bi, length in enumerate(seq_lens):
        for t in (0, length // 2, length - 1):
            row = indices[bi, t]
            valid = row[row >= 0].tolist()
            assert sorted(valid) == list(range(t + 1)), \
                f"row (b={bi}, t={t}) is not the causal prefix"


@requires_torch
def test_reference_index_list_invariants():
    """Structural invariants of the index lists, incl. the sparse regime.

    Uses S=2178 (S % 4 == 2) so rows cover n_vis <= 512 (dense-equivalent)
    and n_vis > 512 (true top-k), plus a short ragged second request.
    """
    gen = torch.Generator().manual_seed(4242)
    batch_lengths = [2178, 517]
    seq = max(batch_lengths)
    ref = QsaTorchReference()
    _, _, iq_g, ik_g = _make_gammas(gen)
    _, index_qk = _make_activations(len(batch_lengths), seq, gen)
    ctx = torch.tensor(batch_lengths, dtype=torch.int32)
    table = make_qsa_rope_table(seq)

    indices = ref.build_indices(index_qk, iq_g, ik_g, table, ctx)

    assert indices.shape == (len(batch_lengths), seq, INDEX_WIDTH)
    assert INDEX_WIDTH == 2051

    ratio, budget = INDEXER_COMPRESS_RATIO, INDEXER_BLOCK_TOPK
    for bi, length in enumerate(batch_lengths):
        # Padding rows are all -1.
        assert (indices[bi, length:] == -1).all(), "padding rows must be -1"
        sample_rows = sorted(
            set([0, 1, 2, 3, 4, 7, length // 3, length // 2, length - 1] +
                list(range(2044, min(length, 2060)))))
        for t in sample_rows:
            row = indices[bi, t]
            valid = row[row >= 0]
            n_vis = (t + 1) // ratio
            n_sel = min(budget, n_vis)
            expected_count = n_sel * ratio + (t + 1) - n_vis * ratio
            # Valid entries are packed to the front; the rest is -1 padding.
            assert (row[expected_count:] == -1).all(), \
                f"(b={bi}, t={t}) expected -1 padding after {expected_count}"
            assert valid.numel() == expected_count, \
                f"(b={bi}, t={t}) count {valid.numel()} != {expected_count}"
            # Distinct, in-range (< t+1 keeps causality inside the list).
            assert valid.unique().numel() == valid.numel(), \
                f"(b={bi}, t={t}) duplicate token indices"
            assert int(valid.min()) >= 0 and int(valid.max()) <= t, \
                f"(b={bi}, t={t}) token index out of causal range"
            # Tail [4*n_vis .. t] is ALWAYS present ((t+1)%4 tokens).
            tail = set(range(n_vis * ratio, t + 1))
            assert tail.issubset(set(valid.tolist())), \
                f"(b={bi}, t={t}) tail tokens missing"
            # Selected blocks are expanded as complete runs of 4.
            block_tokens = [int(x) for x in valid if x < n_vis * ratio]
            blocks = set(tok // ratio for tok in block_tokens)
            assert len(block_tokens) == ratio * len(blocks), \
                f"(b={bi}, t={t}) partial block expansion"
            assert len(blocks) == n_sel
            # Short-seq rows are exactly the causal prefix.
            if n_vis <= budget:
                assert sorted(valid.tolist()) == list(range(t + 1)), \
                    f"(b={bi}, t={t}) short row is not the causal prefix"


@requires_torch
def test_reference_sparse_regime_differs_from_dense():
    """Sanity: past 2051 tokens the sparse output actually drops tokens
    (guards against a reference that silently stays dense)."""
    gen = torch.Generator().manual_seed(77)
    length = 2100  # n_vis = 525 > 512 for the last rows
    ref = QsaTorchReference()
    _, _, iq_g, ik_g = _make_gammas(gen)
    _, index_qk = _make_activations(1, length, gen)
    ctx = torch.tensor([length], dtype=torch.int32)
    table = make_qsa_rope_table(length)
    indices = ref.build_indices(index_qk, iq_g, ik_g, table, ctx)
    t = length - 1
    valid = indices[0, t][indices[0, t] >= 0]
    n_vis = (t + 1) // INDEXER_COMPRESS_RATIO
    assert n_vis > INDEXER_BLOCK_TOPK
    assert valid.numel() == INDEXER_BLOCK_TOPK * INDEXER_COMPRESS_RATIO + (
        t + 1) % INDEXER_COMPRESS_RATIO
    assert valid.numel() < t + 1  # tokens were dropped


# =========================================================================== #
# C. Plugin tests (TensorRT engine on CUDA)
# =========================================================================== #
def _require_qsa_plugin():
    """Skip (not fail) when the QSA plugin cannot run here.

    Unlike the mature plugins (which fail loudly on a missing build), QSA is
    SM 100/101/110-only and is compiled only when ENABLE_CUTE_DSL includes
    "qsa", so a missing library or unregistered creator is an expected state
    and skips cleanly.
    """
    if _device_sm() not in QSA_SMS:
        pytest.skip(f"QSA kernels not registered for SM{_device_sm()} "
                    f"(supported: {sorted(QSA_SMS)})")
    if find_plugin_library() is None:
        pytest.skip("libNvInfer_edgellm_plugin.so not built")
    try:
        load_edgellm_plugins(_get_logger(False))
    except RuntimeError as e:
        pytest.skip(f"could not load Edge-LLM plugins: {e}")
    creator = trt.get_plugin_registry().get_creator("QsaAttentionPlugin", "1",
                                                    "")
    if creator is None:
        pytest.skip("QsaAttentionPlugin not registered in this plugin build")


class QsaPluginRunner:
    """Builds + runs a QsaAttentionPlugin engine for prefill/decode testing.

    The 11-input plugin contract (all required):
    qkv, index_qk, past_key_value (widened paged pool [2, numPages,
    PAGE_SIZE, Hkv, POOL_HEAD_DIM] — K/V in the first HEAD_SIZE of each row,
    indexer-state tails after), context_lengths, rope_rotary_cos_sin
    [1, maxPos, 64], kvcache_start_index (runtime shape [0] = prefill),
    kv_page_table [B, 2, mpps], and the four FP16 gamma constants (engine
    weights).
    """

    def __init__(self,
                 *,
                 batch_size: int,
                 seq_len: int,
                 max_position_embeddings: int,
                 q_gamma: "torch.Tensor",
                 k_gamma: "torch.Tensor",
                 iq_gamma: "torch.Tensor",
                 ik_gamma: "torch.Tensor",
                 attention_scale: float = 0.0,
                 rms_norm_eps: float = RMS_NORM_EPS,
                 flip_page_table: bool = False):
        self.batch = batch_size
        self.seq = seq_len
        self.mpe = max_position_embeddings
        self.flip_page_table = flip_page_table
        self.cap = -(-seq_len // PAGE_SIZE) * PAGE_SIZE
        self.mpps = self.cap // PAGE_SIZE
        self.num_pages = batch_size * self.mpps
        self._pool = None
        self._page_table = None

        pool_shape = (2, self.num_pages, PAGE_SIZE, NUM_KV_HEADS,
                      POOL_HEAD_DIM)
        rope_shape = (1, self.mpe, ROTARY_DIM)
        input_specs = [
            ("qkv", trt.float16, (-1, -1, QKV_PACKED)),
            ("index_qk", trt.float16, (-1, -1, INDEX_QK_PACKED)),
            ("past_key_value", trt.float16, (2, -1, PAGE_SIZE, NUM_KV_HEADS,
                                             POOL_HEAD_DIM)),
            ("context_lengths", trt.int32, (-1, )),
            ("rope_rotary_cos_sin", trt.float32, rope_shape),
            ("kvcache_start_index", trt.int32, (-1, )),
            ("kv_page_table", trt.int32, (-1, 2, self.mpps)),
        ]
        profiles = {
            "qkv": ((1, 1, QKV_PACKED), (batch_size, seq_len, QKV_PACKED),
                    (batch_size, seq_len, QKV_PACKED)),
            "index_qk":
            ((1, 1, INDEX_QK_PACKED), (batch_size, seq_len, INDEX_QK_PACKED),
             (batch_size, seq_len, INDEX_QK_PACKED)),
            # The pool never resizes: numPages fixed per engine.
            "past_key_value": (pool_shape, pool_shape, pool_shape),
            "context_lengths": ((1, ), (batch_size, ), (batch_size, )),
            "rope_rotary_cos_sin": (rope_shape, rope_shape, rope_shape),
            # [0] is the prefill sentinel; max allows binding [B], which
            # selects decode mode (and, with S > 1, the runtime rejection).
            "kvcache_start_index": ((0, ), (0, ), (batch_size, )),
            "kv_page_table": ((1, 2, self.mpps), (batch_size, 2, self.mpps),
                              (batch_size, 2, self.mpps)),
        }
        constant_specs = [
            ("q_norm_gamma", trt.float16, (HEAD_SIZE, ), q_gamma),
            ("k_norm_gamma", trt.float16, (HEAD_SIZE, ), k_gamma),
            ("indexer_q_norm_gamma", trt.float16, (INDEXER_HEAD_DIM, ),
             iq_gamma),
            ("indexer_k_norm_gamma", trt.float16, (INDEXER_HEAD_DIM, ),
             ik_gamma),
        ]
        # Frozen plugin input order (must match the C++ plugin contract).
        plugin_input_order = [
            "qkv", "index_qk", "past_key_value", "context_lengths",
            "rope_rotary_cos_sin", "kvcache_start_index", "kv_page_table",
            "q_norm_gamma", "k_norm_gamma", "indexer_q_norm_gamma",
            "indexer_k_norm_gamma"
        ]
        fields = [
            pf_int32("num_q_heads", NUM_Q_HEADS),
            pf_int32("num_kv_heads", NUM_KV_HEADS),
            pf_int32("head_size", HEAD_SIZE),
            pf_int32("indexer_n_heads", INDEXER_N_HEADS),
            pf_int32("indexer_head_dim", INDEXER_HEAD_DIM),
            pf_int32("indexer_budget", INDEXER_BUDGET),
            pf_int32("indexer_compress_ratio", INDEXER_COMPRESS_RATIO),
            pf_float32("attention_scale", attention_scale),
            pf_float32("rms_norm_eps", rms_norm_eps),
        ]
        self.runner = PluginRunner()
        self.runner.build(
            input_specs=input_specs,
            output_names=["attention_output", "present_key_value"],
            plugin_name="QsaAttentionPlugin",
            plugin_version="1",
            plugin_fields=fields,
            profiles=profiles,
            constant_specs=constant_specs,
            plugin_input_order=plugin_input_order,
            # Indexer workspace (sort buffers + [B,S,2051] outIdx) exceeds 1GB
            # only far past these test shapes; 2GB gives headroom.
            workspace_bytes=2 << 30,
        )

    def _get_pool(self):
        if self._pool is None:
            self._pool = torch.zeros(
                (2, self.num_pages, PAGE_SIZE, NUM_KV_HEADS, POOL_HEAD_DIM),
                dtype=torch.float16,
                device=DEV)
        return self._pool

    def _make_page_table(self, batch):
        if self._page_table is None or self._page_table.shape[0] != batch:
            k_ids = torch.arange(batch * self.mpps,
                                 dtype=torch.int32,
                                 device=DEV)
            if self.flip_page_table:
                k_ids = k_ids.flip(0)
            k_ids = k_ids.reshape(batch, self.mpps)
            self._page_table = torch.stack((k_ids, k_ids + self.num_pages),
                                           dim=1).contiguous()
        return self._page_table

    def run(self, qkv, index_qk, context_lengths, rope_table, zero_pool=True):
        """Execute one prefill (kvcache_start_index bound with the [0]
        sentinel). Returns the [B, S, Hq, D] FP16 output.

        The pool is zeroed before each run — so KV comparisons never see
        stale pages — unless ``zero_pool`` is False (state-poisoning tests
        prepare the pool themselves).
        """
        batch, seq, _ = qkv.shape
        pool = self._get_pool()
        if zero_pool:
            pool.zero_()
        page_table = self._make_page_table(batch)
        kv_start = torch.zeros((self.batch, ), dtype=torch.int32, device=DEV)
        attn_out = torch.empty((batch, seq, NUM_Q_HEADS, HEAD_SIZE),
                               dtype=torch.float16,
                               device=DEV)
        tensors = {
            "qkv": qkv,
            "index_qk": index_qk,
            "past_key_value": pool,
            "context_lengths": context_lengths,
            "rope_rotary_cos_sin": rope_table,
            "kvcache_start_index": kv_start,
            "kv_page_table": page_table,
            "attention_output": attn_out,
            "present_key_value": pool,  # aliased in-place to the pool binding
        }
        self.runner.execute(tensors, {"kvcache_start_index": (0, )})
        return attn_out

    def gather_kv(self, batch, length):
        """Gather the paged pool through the page table into token-major K/V.

        Returns (k, v) FP16 [batch, length, Hkv, HEAD_SIZE] — roped K and raw
        V as the plugin stores them. Only the HEAD_SIZE prefix of each
        widened pool row is K/V; the [HEAD_SIZE:POOL_HEAD_DIM) tails hold
        indexer state and are excluded here.
        """
        pool = self._get_pool()
        page_table = self._make_page_table(batch)
        k = torch.zeros((batch, self.cap, NUM_KV_HEADS, HEAD_SIZE),
                        dtype=torch.float16,
                        device=DEV)
        v = torch.zeros_like(k)
        for bi in range(batch):
            for lp in range(self.mpps):
                k_page = int(page_table[bi, 0, lp])
                v_page = int(page_table[bi, 1, lp]) - self.num_pages
                begin = lp * PAGE_SIZE
                k[bi, begin:begin + PAGE_SIZE] = pool[0,
                                                      k_page][..., :HEAD_SIZE]
                v[bi, begin:begin + PAGE_SIZE] = pool[1,
                                                      v_page][..., :HEAD_SIZE]
        return k[:, :length], v[:, :length]

    def run_decode(self,
                   qkv,
                   index_qk,
                   context_lengths,
                   rope_table,
                   kv_start=None,
                   attention_output=None,
                   synchronize=True):
        """Execute one decode step. Returns the [B, 1, Hq, D] FP16 output.

        The pool is NOT zeroed: decode appends to the state left by the
        preceding prefill / decode enqueues. ``context_lengths`` holds the
        TOTAL lengths including the token being decoded; ``kv_start``
        (shape [B] selects decode mode) defaults to context_lengths - 1 —
        the past lengths, whose values the plugin never dereferences.
        ``synchronize=False`` supports CUDA-graph capture with static
        bindings (pass ``kv_start`` and ``attention_output`` explicitly so
        nothing is allocated during capture).
        """
        batch = qkv.shape[0]
        pool = self._get_pool()
        page_table = self._make_page_table(batch)
        if kv_start is None:
            kv_start = context_lengths - 1
        attn_out = attention_output
        if attn_out is None:
            attn_out = torch.empty(
                (batch, qkv.shape[1], NUM_Q_HEADS, HEAD_SIZE),
                dtype=torch.float16,
                device=DEV)
        tensors = {
            "qkv": qkv,
            "index_qk": index_qk,
            "past_key_value": pool,
            "context_lengths": context_lengths,
            "rope_rotary_cos_sin": rope_table,
            "kvcache_start_index": kv_start,
            "kv_page_table": page_table,
            "attention_output": attn_out,
            "present_key_value": pool,  # aliased in-place to the pool binding
        }
        self.runner.execute(tensors, synchronize=synchronize)
        return attn_out

    def gather_tails(self, batch):
        """Gather the widened rows' indexer-state tails through the page
        table.

        Returns (k_tail, v_tail) FP16 [batch, cap, Hkv, INDEXER_HEAD_DIM]:
        columns [HEAD_SIZE:POOL_HEAD_DIM) of every pool row, token-major.
        Per the QsaIndexerPoolState contract only head 0 holds state — the
        V-plane tail of token 4g the block's kbar, the K-plane tail of a
        token its raw index-K only while that token's block is incomplete —
        but both heads are returned so tests can also assert the untouched
        regions.
        """
        pool = self._get_pool()
        page_table = self._make_page_table(batch)
        k_tail = torch.empty((batch, self.cap, NUM_KV_HEADS, INDEXER_HEAD_DIM),
                             dtype=torch.float16,
                             device=DEV)
        v_tail = torch.empty_like(k_tail)
        for bi in range(batch):
            for lp in range(self.mpps):
                k_page = int(page_table[bi, 0, lp])
                v_page = int(page_table[bi, 1, lp]) - self.num_pages
                begin = lp * PAGE_SIZE
                k_tail[bi, begin:begin + PAGE_SIZE] = pool[0,
                                                           k_page][...,
                                                                   HEAD_SIZE:]
                v_tail[bi, begin:begin + PAGE_SIZE] = pool[1,
                                                           v_page][...,
                                                                   HEAD_SIZE:]
        return k_tail, v_tail


def _plugin_case(batch,
                 seq,
                 seed,
                 *,
                 attention_scale=0.0,
                 flip_page_table=False):
    """Build runner + reference + matched random inputs for one config."""
    gen = torch.Generator().manual_seed(seed)
    q_g, k_g, iq_g, ik_g = _make_gammas(gen, device=DEV)
    qkv, index_qk = _make_activations(batch, seq, gen, device=DEV)
    mpe = max(seq, ROTARY_DIM)
    table = make_qsa_rope_table(mpe, device=DEV)
    rope_binding = table[None]  # [1, mpe, 64]
    runner = QsaPluginRunner(batch_size=batch,
                             seq_len=seq,
                             max_position_embeddings=mpe,
                             q_gamma=q_g,
                             k_gamma=k_g,
                             iq_gamma=iq_g,
                             ik_gamma=ik_g,
                             attention_scale=attention_scale,
                             flip_page_table=flip_page_table)
    ref = QsaTorchReference(attention_scale=attention_scale)
    return runner, ref, qkv, index_qk, (q_g, k_g, iq_g, ik_g), table, \
        rope_binding


@requires_gpu
@pytest.mark.parametrize("seq_len", [33, 384])
def test_plugin_dense_equivalence_prefill(seq_len):
    """For S <= 2051 the plugin must equal the causal-dense SDPA reference
    (with qk-norm + partial rope): the index list is exactly {0..t}."""
    _require_qsa_plugin()
    batch = 1
    runner, ref, qkv, index_qk, gammas, table, rope = _plugin_case(batch,
                                                                   seq_len,
                                                                   seed=1000 +
                                                                   seq_len)
    q_g, k_g, _, _ = gammas
    ctx = torch.full((batch, ), seq_len, dtype=torch.int32, device=DEV)

    out = runner.run(qkv, index_qk, ctx, rope)

    ref_out = ref.dense_forward(qkv, q_g, k_g, table, ctx)
    assert_close(f"qsa_dense_equiv[s{seq_len}]", ref_out, out)


@requires_gpu
@pytest.mark.parametrize("context_lengths", [[4096], [4096, 2700]],
                         ids=["bs1_s4096", "bs2_ragged"])
def test_plugin_sparse_prefill(context_lengths):
    """S=4096 sparse prefill vs the full FP32 QSA torch reference.

    fp16 tolerances + a cosine gate (not exact index equality): FP32 score
    accumulation order differs between the warp kernel and torch, so blocks
    at the top-512 boundary may swap (see the module docstring caveat).
    """
    _require_qsa_plugin()
    batch = len(context_lengths)
    seq = max(context_lengths)
    runner, ref, qkv, index_qk, gammas, table, rope = _plugin_case(batch,
                                                                   seq,
                                                                   seed=2026)
    q_g, k_g, iq_g, ik_g = gammas
    ctx = torch.tensor(context_lengths, dtype=torch.int32, device=DEV)

    out = runner.run(qkv, index_qk, ctx, rope)

    ref_out, _, _, _ = ref.forward(qkv, index_qk, q_g, k_g, iq_g, ik_g, table,
                                   ctx)
    for bi, length in enumerate(context_lengths):
        case = f"qsa_sparse[b{bi} L{length}]"
        assert_close(case,
                     ref_out[bi, :length],
                     out[bi, :length],
                     cos_threshold=0.999)
        assert cosine_sim(ref_out[bi, :length], out[bi, :length]) > 0.999
        # Rows past the context length must be exact zeros.
        if length < seq:
            assert (out[bi, length:] == 0).all(), \
                f"{case}: padding rows must be exactly zero"


@requires_gpu
def test_plugin_kv_write_through():
    """Paged-KV write-through: pool gathered via a flipped page table equals
    the roped-K / raw-V reference; poisoned padding must not leak anywhere."""
    _require_qsa_plugin()
    context_lengths = [384, 200]
    batch, seq = len(context_lengths), max(context_lengths)
    runner, ref, qkv, index_qk, gammas, table, rope = _plugin_case(
        batch, seq, seed=333, flip_page_table=True)
    q_g, k_g, iq_g, ik_g = gammas
    ctx = torch.tensor(context_lengths, dtype=torch.int32, device=DEV)
    # Any kernel that reads a row's padding region now corrupts the result.
    poison_padding([qkv, index_qk], context_lengths)

    out = runner.run(qkv, index_qk, ctx, rope)

    ref_out, _, ref_k, ref_v = ref.forward(qkv, index_qk, q_g, k_g, iq_g, ik_g,
                                           table, ctx)
    plugin_k, plugin_v = runner.gather_kv(batch, seq)
    for bi, length in enumerate(context_lengths):
        case = f"b{bi} L{length}"
        assert_close(f"qsa_kv_k[{case}]", ref_k[bi, :length],
                     plugin_k[bi, :length])
        assert_close(f"qsa_kv_v[{case}]", ref_v[bi, :length],
                     plugin_v[bi, :length])
        assert_close(f"qsa_attn_poisoned[{case}]", ref_out[bi, :length],
                     out[bi, :length])
        if length < seq:
            assert (out[bi, length:] == 0).all(), \
                f"qsa_attn_poisoned[{case}]: padding rows must be zero"


@requires_gpu
def test_plugin_determinism():
    """Two identical sparse prefills produce bitwise-identical outputs and KV
    (cub sort + gather kernels must be run-to-run deterministic)."""
    _require_qsa_plugin()
    batch, seq = 1, 4096
    runner, _, qkv, index_qk, _, _, rope = _plugin_case(batch, seq, seed=777)
    ctx = torch.full((batch, ), seq, dtype=torch.int32, device=DEV)

    out1 = runner.run(qkv.clone(), index_qk.clone(), ctx, rope).clone()
    k1, v1 = runner.gather_kv(batch, seq)
    k1, v1 = k1.clone(), v1.clone()
    out2 = runner.run(qkv.clone(), index_qk.clone(), ctx, rope)
    k2, v2 = runner.gather_kv(batch, seq)

    assert torch.equal(out1.view(torch.int16), out2.view(torch.int16)), \
        "attention output is not deterministic"
    assert torch.equal(k1.view(torch.int16), k2.view(torch.int16)), \
        "K cache write is not deterministic"
    assert torch.equal(v1.view(torch.int16), v2.view(torch.int16)), \
        "V cache write is not deterministic"


# =========================================================================== #
# D. Decode reference self-tests — pure torch on CPU, no plugin / CUDA
# =========================================================================== #
def _assert_index_sets_match(case,
                             ref_row,
                             dec_idx,
                             block_scores,
                             tie_tol=1e-4):
    """Index SET equality between a prefill reference row and a decode row.

    Disagreements are tolerated only as whole-block swaps whose FP32 scores
    tie (within ``tie_tol``) at the top-512 selection boundary — the two
    references reduce the score dot in different orders, so exact-cutoff
    ties may resolve differently (measure-zero for continuous random
    inputs, but kept as the documented tolerance). Anything else fails.
    """
    dec_list = dec_idx.tolist()
    dec_set = set(dec_list)
    assert len(dec_set) == len(dec_list), f"{case}: duplicate decode indices"
    ref_set = set(ref_row[ref_row >= 0].tolist())
    if ref_set == dec_set:
        return
    ratio, budget = INDEXER_COMPRESS_RATIO, INDEXER_BLOCK_TOPK
    only_ref = ref_set - dec_set
    only_dec = dec_set - ref_set
    blocks_ref = {t // ratio for t in only_ref}
    blocks_dec = {t // ratio for t in only_dec}
    assert (len(only_ref) == ratio * len(blocks_ref)
            and len(only_dec) == ratio * len(blocks_dec)
            and len(blocks_ref) == len(blocks_dec)), \
        f"{case}: index sets differ beyond whole-block swaps"
    assert block_scores is not None and block_scores.numel() > budget, \
        f"{case}: index sets differ below the top-{budget} boundary"
    cutoff = float(torch.topk(block_scores, budget).values[-1])
    for g in blocks_ref | blocks_dec:
        assert abs(float(block_scores[g]) - cutoff) <= tie_tol, \
            f"{case}: swapped block {g} does not tie at the top-{budget} " \
            f"cutoff"


@requires_torch
def test_reference_decode_step_matches_prefill_rows():
    """Decode-reference invariant: the step at ctx == prefill row ctx - 1.

    Short regime (n_vis <= 512): exact index-set equality plus fp32-allclose
    attention for every decode step of a prefill(44) + 17-step sequence.
    Long regime (n_vis > 512): index sets compared with the top-512 tie
    tolerance (the decode row's attention math is already pinned by the
    short regime; the long rows only add the budget cutoff).
    """
    gen = torch.Generator().manual_seed(20260901)
    ref = QsaTorchReference()

    # -- short regime: 44 prefill tokens + 17 decode rows, attention gated --
    length, l0 = 61, 44
    q_g, k_g, iq_g, ik_g = _make_gammas(gen)
    qkv, index_qk = _make_activations(1, length, gen)
    table = make_qsa_rope_table(max(length, ROTARY_DIM))
    prefill_out, prefill_idx, _, _ = ref.forward(
        qkv, index_qk, q_g, k_g, iq_g, ik_g, table,
        torch.tensor([length], dtype=torch.int32))
    dec = QsaDecodeTorchReference(ref, qkv, index_qk, q_g, k_g, iq_g, ik_g,
                                  table)
    for ctx in range(l0 + 1, length + 1):
        out, idx = dec.step_output(0, ctx)
        n_vis = ctx // INDEXER_COMPRESS_RATIO
        assert idx.numel() == (
            min(INDEXER_BLOCK_TOPK, n_vis) * INDEXER_COMPRESS_RATIO + ctx -
            n_vis * INDEXER_COMPRESS_RATIO)
        _assert_index_sets_match(f"short ctx={ctx}", prefill_idx[0, ctx - 1],
                                 idx, dec.block_scores(0, ctx))
        torch.testing.assert_close(out,
                                   prefill_out[0, ctx - 1],
                                   rtol=1e-4,
                                   atol=1e-4)

    # -- long regime: n_vis > 512 exercises the budget cutoff (2100//4=525) --
    length = 2104
    qkv, index_qk = _make_activations(1, length, gen)
    table = make_qsa_rope_table(length)
    prefill_idx = ref.build_indices(index_qk, iq_g, ik_g, table,
                                    torch.tensor([length], dtype=torch.int32))
    dec = QsaDecodeTorchReference(ref, qkv, index_qk, q_g, k_g, iq_g, ik_g,
                                  table)
    for ctx in (2100, 2101, 2103, 2104):
        idx = dec.step_indices(0, ctx)
        assert ctx // INDEXER_COMPRESS_RATIO > INDEXER_BLOCK_TOPK
        assert idx.numel() == (INDEXER_BLOCK_TOPK * INDEXER_COMPRESS_RATIO +
                               ctx % INDEXER_COMPRESS_RATIO)
        _assert_index_sets_match(f"long ctx={ctx}", prefill_idx[0, ctx - 1],
                                 idx, dec.block_scores(0, ctx))


# =========================================================================== #
# E. Plugin decode tests (TensorRT engine on CUDA)
# =========================================================================== #
def _decode_case(l0s, steps, seed, *, build_seq, flip_page_table=False):
    """Runner + stateful decode reference + pregenerated full history.

    Row t of the [B, build_seq] random history is sequence b's token t:
    prefill consumes rows [0, l0s[b]) and decode step s consumes row
    l0s[b] + s — a shorter sequence's later rows double as (masked) prefill
    padding first and as its real decode tokens afterwards.
    """
    assert max(l0s) + steps <= build_seq
    runner, ref, qkv, index_qk, gammas, table, rope = _plugin_case(
        len(l0s), build_seq, seed=seed, flip_page_table=flip_page_table)
    q_g, k_g, iq_g, ik_g = gammas
    dec = QsaDecodeTorchReference(ref, qkv, index_qk, q_g, k_g, iq_g, ik_g,
                                  table)
    return runner, ref, dec, qkv, index_qk, gammas, table, rope


def _run_prefill(runner, qkv, index_qk, l0s, rope, zero_pool=True):
    """Prefill rows [0, l0s[b]) of the history; returns the plugin output."""
    s0 = max(l0s)
    ctx0 = torch.tensor(l0s, dtype=torch.int32, device=DEV)
    return runner.run(qkv[:, :s0].contiguous(),
                      index_qk[:, :s0].contiguous(),
                      ctx0,
                      rope,
                      zero_pool=zero_pool)


def _decode_step_inputs(qkv, index_qk, l0s, step):
    """S=1 bindings for decode step ``step``: sequence b decodes token row
    l0s[b] + step; context_lengths are the TOTAL lengths including it."""
    rows = [l0 + step for l0 in l0s]
    qkv_s = torch.stack([qkv[b, r] for b, r in enumerate(rows)])[:, None]
    iq_s = torch.stack([index_qk[b, r] for b, r in enumerate(rows)])[:, None]
    ctx = torch.tensor([r + 1 for r in rows],
                       dtype=torch.int32,
                       device=qkv.device)
    return qkv_s.contiguous(), iq_s.contiguous(), ctx


@requires_gpu
def test_plugin_decode_ragged_batch():
    """B=2 ragged decode: different prefill lengths, both sequences advance
    one token per step, and sequence 0 crosses the 128-token page boundary
    mid-decode (ctx 120 -> 136 with maxPagesPerSeq = 2)."""
    _require_qsa_plugin()
    l0s, steps = [120, 87], 16
    runner, ref, dec, qkv, index_qk, gammas, table, rope = _decode_case(
        l0s, steps, seed=8802, build_seq=256)
    _run_prefill(runner, qkv, index_qk, l0s, rope)
    for s in range(steps):
        qkv_s, iq_s, ctx_t = _decode_step_inputs(qkv, index_qk, l0s, s)
        out = runner.run_decode(qkv_s, iq_s, ctx_t, rope)
        for bi, l0 in enumerate(l0s):
            ctx = l0 + s + 1
            ref_out, _ = dec.step_output(bi, ctx)
            assert_close(f"qsa_decode_ragged[b{bi} ctx{ctx}]", ref_out, out[bi,
                                                                            0])


def _raw_key_rows(ctx, l0):
    """Boolean mask over rows [0, ctx): the tokens whose raw index-K the
    kernels persist — every token that arrived into an incomplete block
    (never the block-completing token, t % 4 == 3) and, from the prefill,
    only its trailing incomplete block (t >= l0 - l0 % 4)."""
    rows = torch.arange(ctx, device=DEV)
    return rows, (
        (rows >= l0 - l0 % INDEXER_COMPRESS_RATIO)
        & (rows % INDEXER_COMPRESS_RATIO != INDEXER_COMPRESS_RATIO - 1))


@requires_gpu
def test_plugin_decode_pool_tail_state():
    """Persisted indexer state in the widened pool tails.

    After prefill (L0=120) + 16 decode steps (final ctx=136 — block-aligned
    so every V-tail row within ctx is asserted — crossing the page boundary,
    with a flipped page table to exercise the paged addressing): the V-plane
    tail of every token 4g holds the block's kbar (fp16 chain) within 2e-3;
    the K-plane tail of every decode-era token that arrived into an
    incomplete block (t >= 120, t % 4 != 3) holds its raw index-K columns
    [512:640) BIT-EXACTLY; every other tail row is untouched (still zero) —
    in particular the K-tails of the blocks prefill completed (L0 % 4 == 0,
    so prefill stores no raw keys) and of every block-completing token.
    """
    _require_qsa_plugin()
    l0s, steps = [120], 16
    runner, ref, dec, qkv, index_qk, gammas, table, rope = _decode_case(
        l0s, steps, seed=8803, build_seq=256, flip_page_table=True)
    _run_prefill(runner, qkv, index_qk, l0s, rope)
    out = None
    for s in range(steps):
        qkv_s, iq_s, ctx_t = _decode_step_inputs(qkv, index_qk, l0s, s)
        out = runner.run_decode(qkv_s, iq_s, ctx_t, rope)
    ctx = l0s[0] + steps  # 136
    ref_out, _ = dec.step_output(0, ctx)
    assert_close(f"qsa_tail_state[ctx{ctx}]", ref_out, out[0, 0])

    k_tail, v_tail = runner.gather_tails(1)
    # K tails: bit-exact raw index-K for exactly the tokens whose block was
    # incomplete when they arrived (decode era only: L0 is block-aligned).
    rows, raw = _raw_key_rows(ctx, l0s[0])
    expect_k = index_qk[0, rows[raw], INDEXER_N_HEADS * INDEXER_HEAD_DIM:]
    assert torch.equal(
        k_tail[0, rows[raw], 0].contiguous().view(torch.int16),
        expect_k.contiguous().view(torch.int16)), \
        "K-plane tails must hold the raw index-K bit-exactly"
    assert (k_tail[0, rows[~raw], 0] == 0).all(), \
        "K tails of completed blocks / block-completing tokens must stay untouched"
    # V tails at rows 4g: the kbar of every complete block.
    kbar_err = (v_tail[0, 0:ctx:INDEXER_COMPRESS_RATIO, 0].float() -
                dec.kbar[0, :ctx // INDEXER_COMPRESS_RATIO]).abs().max()
    assert float(kbar_err) <= 2e-3, \
        f"V-plane kbar tails off by {float(kbar_err):.5f} (limit 2e-3)"
    # Untouched regions: rows at/after ctx (pool was zeroed pre-prefill),
    # V-tail rows off the 4g grid, and head 1 everywhere.
    assert (k_tail[0, ctx:] == 0).all(), "K tails beyond ctx were touched"
    assert (v_tail[0, ctx:] == 0).all(), "V tails beyond ctx were touched"
    off_grid = rows[rows % INDEXER_COMPRESS_RATIO != 0]
    assert (v_tail[0, off_grid, 0] == 0).all(), \
        "V tails off the 4g grid were touched"
    assert (k_tail[0, :, 1] == 0).all() and (v_tail[0, :, 1] == 0).all(), \
        "indexer state must live in head 0 tails only"


@requires_gpu
def test_plugin_decode_nan_tail_poisoning():
    """NaN-poisoned tails never leak: the attention path must not read the
    pool tails, and the indexer must overwrite every tail it later reads.

    All tails start as NaN. Prefill + decode outputs must stay finite and
    match the reference (assert_close rejects non-finite actuals), while
    never-written tail rows must keep their NaN — pinning that both reads
    and writes stay exactly on the contracted state locations. L0 = 43 is
    NOT block-aligned, so the prefill scatter's positive path (raw keys of
    the trailing block 40..42, consumed by the ctx = 44 compress) is covered
    at the plugin level too; the final ctx = 60 is block-aligned.
    """
    _require_qsa_plugin()
    l0, steps = 43, 17
    runner, ref, dec, qkv, index_qk, gammas, table, rope = _decode_case(
        [l0], steps, seed=8804, build_seq=64)
    pool = runner._get_pool()
    pool.zero_()
    pool[..., HEAD_SIZE:] = float("nan")
    out0 = _run_prefill(runner, qkv, index_qk, [l0], rope, zero_pool=False)
    q_g, k_g, _, _ = gammas
    ctx0 = torch.tensor([l0], dtype=torch.int32, device=DEV)
    assert_close("qsa_nan_tails_prefill",
                 ref.dense_forward(qkv[:, :l0], q_g, k_g, table, ctx0), out0)
    for s in range(steps):
        qkv_s, iq_s, ctx_t = _decode_step_inputs(qkv, index_qk, [l0], s)
        out = runner.run_decode(qkv_s, iq_s, ctx_t, rope)
        ctx = l0 + s + 1
        ref_out, _ = dec.step_output(0, ctx)
        assert_close(f"qsa_nan_tails[ctx{ctx}]", ref_out, out[0, 0])
    # Written tails lost their poison, never-written tails kept it (ctx = 60
    # is block-aligned: every 4g row within ctx was completed).
    ctx = l0 + steps
    k_tail, v_tail = runner.gather_tails(1)
    rows, raw = _raw_key_rows(ctx, l0)
    assert not torch.isnan(k_tail[0, rows[raw], 0]).any(), \
        "prefill/decode must overwrite the K tail of every incomplete-block token"
    assert torch.isnan(k_tail[0, rows[~raw], 0]).all(), \
        "K tails of completed blocks / block-completing tokens must keep the poison"
    on_grid = rows[rows % INDEXER_COMPRESS_RATIO == 0]
    assert not torch.isnan(v_tail[0, on_grid, 0]).any(), \
        "every complete block's kbar tail must be overwritten"
    assert torch.isnan(v_tail[0, rows[rows % INDEXER_COMPRESS_RATIO != 0],
                              0]).all(), \
        "V tails off the 4g grid must keep the poison (never written)"
    assert torch.isnan(k_tail[0, ctx:]).all() \
        and torch.isnan(v_tail[0, ctx:]).all(), \
        "tails beyond ctx must keep the poison (never written)"
    assert torch.isnan(k_tail[0, :, 1]).all() \
        and torch.isnan(v_tail[0, :, 1]).all(), \
        "head 1 tails must keep the poison (state lives in head 0)"


@requires_gpu
def test_plugin_decode_s_gt1_rejected():
    """Decode mode (kvcache_start_index shape [B]) requires S == 1: S = 4
    must fail the enqueue cleanly (no crash), and the execution context must
    stay usable — the surrounding valid decode steps still match."""
    _require_qsa_plugin()
    l0 = 44
    runner, ref, dec, qkv, index_qk, gammas, table, rope = _decode_case(
        [l0], 8, seed=8805, build_seq=64)
    _run_prefill(runner, qkv, index_qk, [l0], rope)
    qkv_s, iq_s, ctx_t = _decode_step_inputs(qkv, index_qk, [l0], 0)
    out = runner.run_decode(qkv_s, iq_s, ctx_t, rope)
    ref_out, _ = dec.step_output(0, l0 + 1)
    assert_close("qsa_s_gt1[pre]", ref_out, out[0, 0])
    # S=4 with a decode-shaped start index: rejected before any launch.
    with pytest.raises(RuntimeError, match="execute_async_v3 returned False"):
        runner.run_decode(qkv[:, l0 + 1:l0 + 5].contiguous(),
                          index_qk[:, l0 + 1:l0 + 5].contiguous(), ctx_t + 1,
                          rope)
    # The failure consumed nothing: the next valid step still matches.
    qkv_s, iq_s, ctx_t = _decode_step_inputs(qkv, index_qk, [l0], 1)
    out = runner.run_decode(qkv_s, iq_s, ctx_t, rope)
    ref_out, _ = dec.step_output(0, l0 + 2)
    assert_close("qsa_s_gt1[post]", ref_out, out[0, 0])


@requires_gpu
def test_plugin_decode_graph_capture_smoke():
    """One decode step captured into a CUDA graph replays bit-identically.

    The decode path is contractually graph-safe (no cub, no allocation,
    in-enqueue split-counter zeroing) and idempotent for a fixed ctx (it
    rewrites the same pool rows with the same values), so warmup + capture
    + N replays of the same step are equivalent. The output buffer is
    zeroed before each replay so a silently empty graph cannot pass.
    """
    _require_qsa_plugin()
    l0 = 44
    runner, ref, dec, qkv, index_qk, gammas, table, rope = _decode_case(
        [l0], 4, seed=8806, build_seq=64)
    _run_prefill(runner, qkv, index_qk, [l0], rope)
    # One eager step so capture starts from a realistic mid-decode state.
    qkv_s, iq_s, ctx_t = _decode_step_inputs(qkv, index_qk, [l0], 0)
    runner.run_decode(qkv_s, iq_s, ctx_t, rope)
    # Static bindings for the captured step (ctx = l0 + 2): everything is
    # allocated here so nothing allocates during capture.
    ctx = l0 + 2
    qkv_s, iq_s, ctx_t = _decode_step_inputs(qkv, index_qk, [l0], 1)
    kv_start = ctx_t - 1
    static_out = torch.empty((1, 1, NUM_Q_HEADS, HEAD_SIZE),
                             dtype=torch.float16,
                             device=DEV)

    def enqueue():
        runner.run_decode(qkv_s,
                          iq_s,
                          ctx_t,
                          rope,
                          kv_start=kv_start,
                          attention_output=static_out,
                          synchronize=False)

    # TensorRT wants one enqueue before capture (lazy resource init); the
    # side stream keeps the warmup out of the ambient capture stream, per
    # torch's CUDA-graph usage pattern.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        enqueue()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        enqueue()

    replays = []
    for _ in range(2):
        static_out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        replays.append(static_out.clone())
    assert (replays[0] != 0).any(), "graph replay produced no output"
    assert torch.equal(replays[0].view(torch.int16),
                       replays[1].view(torch.int16)), \
        "graph replays are not bitwise identical"
    ref_out, _ = dec.step_output(0, ctx)
    assert_close("qsa_graph_decode", ref_out, replays[1][0, 0])
