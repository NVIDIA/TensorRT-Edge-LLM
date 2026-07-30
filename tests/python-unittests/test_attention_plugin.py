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
AttentionPlugin unit tests vs a PyTorch reference.

Drives the custom AttentionPlugin entirely through torch CUDA tensors and
validates it against a PyTorch reference across a sweep of configs:
prefill / decode, grouped-query attention, head-size variants, FP8 KV cache,
chunked prefill, ragged context lengths, batch-order permutation invariance,
tree (speculative) attention, shared-KV (donor-cache) layers, and the Q
pre-scaling convention for non-standard softmax scales. Sliding-window
attention is covered in test_sliding_window_attention_plugin.py.

The plugin's KV-cache ABI is paged: a pool binding [2, numPages, PAGE_SIZE,
Hkv, D] plus an int32 page table [batch, 2, maxPagesPerSeq]. The tests (and
the torch reference) keep working in the logical per-slot layout
[batch, 2, Hkv, cap, D]; AttentionPluginRunner converts between the two under
an identity page table.

Run:
    python3 -m pytest tests/python-unittests/test_attention_plugin.py -v
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              RAGGED_CASES, PluginRunner, assert_close,
                              pf_float32, pf_int32, poison_padding)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

pytestmark = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE,
    reason=f"TensorRT/torch CUDA not available: {IMPORT_ERROR}")

DEV = "cuda"

# Tokens per page of the paged-KV pool. Must match kTOKENS_PER_PAGE
# (cpp/common/pagedKvTypes.h).
PAGE_SIZE = 128


@dataclass
class AttentionParams:
    """Parameters for attention testing (torch reference)."""
    batch_size: int = 1
    seq_len: int = 1
    num_q_heads: int = 8
    num_kv_heads: int = 8
    head_size: int = 128
    kv_cache_capacity: int = 64
    max_batch_size: int = 8
    max_seq_len: int = 8
    max_position_embeddings: int = 64
    qk_scale: Optional[float] = None
    is_prefill: bool = False
    enable_fp8_kv_cache: bool = False
    sliding_window_size: int = -1  # -1 disables
    qkv_scales: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    def __post_init__(self):
        assert self.num_q_heads % self.num_kv_heads == 0, \
            "num_q_heads must be a multiple of num_kv_heads (GQA)"
        self.qkv_hidden_size = (self.num_q_heads +
                                2 * self.num_kv_heads) * self.head_size
        if self.qk_scale is None:
            self.qk_scale = 1.0 / (self.head_size**0.5)

    @property
    def q_hidden(self) -> int:
        return self.num_q_heads * self.head_size

    @property
    def kv_hidden(self) -> int:
        return self.num_kv_heads * self.head_size


def fp8_round_trip(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Model FP8 (e4m3) storage with a dequant ``scale`` (stored * scale = orig).

    Returns the value recovered after quantize → fp8 cast → dequantize, i.e. the
    magnitude the plugin's FP8 KV cache would actually hold.
    """
    if not hasattr(torch, "float8_e4m3fn"):
        return x
    q = (x.float() / scale).to(torch.float8_e4m3fn)
    return q.float() * scale


def apply_rotary_embedding(x: torch.Tensor, cos_cache: torch.Tensor,
                           sin_cache: torch.Tensor,
                           position_ids: torch.Tensor) -> torch.Tensor:
    """NeoX rotate-half RoPE.

    x:            [batch, num_heads, seq_len, head_size]
    cos/sin_cache:[max_pos_emb, head_size // 2]
    position_ids: [batch, seq_len]
    """
    batch, num_heads, seq_len, head_size = x.shape
    half = head_size // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = cos_cache[position_ids]  # [b, s, half]
    sin = sin_cache[position_ids]
    cos = cos[:, None, :, :]  # [b, 1, s, half]
    sin = sin[:, None, :, :]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return torch.cat([rot1, rot2], dim=-1)


def sliding_window_mask(seq_q: int, seq_k: int, window: int,
                        device) -> torch.Tensor:
    """Causal mask with a sliding window. 1 = attend, 0 = masked.

    ``window`` is the number of keys attended in total: the query at absolute
    position p attends to keys [p - window + 1, p]. This is the plugin's single
    window semantic across prefill (CuTe DSL FMHA and FMHA_v2) and decode (XQA,
    matching the ``sliceKVWindow`` reference in the XQA decoding gtest), and
    the HF ``sliding_window`` convention.

    window <= 0 disables the window (causal only).
    """
    offset = seq_k - seq_q
    qi = torch.arange(seq_q, device=device)[:, None] + offset
    kj = torch.arange(seq_k, device=device)[None, :]
    mask = kj <= qi
    if window and window > 0:
        mask &= kj >= (qi - (window - 1))
    return mask.to(torch.int32)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor,
                                 v: torch.Tensor, scale: float,
                                 attn_mask: Optional[torch.Tensor],
                                 num_q_heads: int,
                                 num_kv_heads: int) -> torch.Tensor:
    """SDPA with grouped-query expansion. q:[b,Hq,sq,d] k/v:[b,Hkv,sk,d]."""
    if num_kv_heads != num_q_heads:
        rep = num_q_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask[None, None] == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def compute_attention(
    qkv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    cache_indices: torch.Tensor,
    params: AttentionParams,
    attn_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full attention with RoPE + KV cache. Returns (out, k_cache, v_cache).

    Shapes: qkv [b, s, qkv_hidden]; k_cache/v_cache [b, Hkv, cap, d]
            (post-RoPE keys, raw values — as the plugin stores them).
    Caches are returned updated (not in-place).
    """
    b, s = params.batch_size, params.seq_len
    Hq, Hkv, d = params.num_q_heads, params.num_kv_heads, params.head_size

    q = qkv[:, :, :params.q_hidden]
    k = qkv[:, :, params.q_hidden:params.q_hidden + params.kv_hidden]
    v = qkv[:, :, params.q_hidden + params.kv_hidden:]

    q = q.reshape(b, s, Hq, d).transpose(1, 2)
    k = k.reshape(b, s, Hkv, d).transpose(1, 2)
    v = v.reshape(b, s, Hkv, d).transpose(1, 2)

    q = apply_rotary_embedding(q, cos_cache, sin_cache, position_ids)
    k = apply_rotary_embedding(k, cos_cache, sin_cache, position_ids)

    if params.enable_fp8_kv_cache:
        qs, ks, vs = params.qkv_scales
        q = fp8_round_trip(q, qs)
        k = fp8_round_trip(k, ks)
        v = fp8_round_trip(v, vs)

    k_cache = k_cache.clone()
    v_cache = v_cache.clone()
    for bi in range(b):
        idx = int(cache_indices[bi])
        k_cache[bi, :, idx:idx + s, :] = k[bi]
        v_cache[bi, :, idx:idx + s, :] = v[bi]

    present = int(cache_indices[0]) + s
    k_present = k_cache[:, :, :present, :]
    v_present = v_cache[:, :, :present, :]

    out = scaled_dot_product_attention(q, k_present, v_present,
                                       params.qk_scale, attn_mask, Hq, Hkv)
    out = out.transpose(1, 2).reshape(b, s, Hq * d)
    return out, k_cache, v_cache


# --------------------------------------------------------------------------- #
# Tree-attention helpers (ported from test_attention_utils.py)
# --------------------------------------------------------------------------- #
def get_tree_attention_mask(seq_len: int):
    """Fixed tree mask [seq_len, seq_len] + accepted indices (tokens 0 and 2)."""
    base = torch.tensor(
        [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 1, 1]],
        dtype=torch.int32)
    if seq_len <= 4:
        mask = base[:seq_len, :seq_len].clone()
    else:
        mask = torch.zeros((seq_len, seq_len), dtype=torch.int32)
        mask[:4, :4] = base
        for i in range(4, seq_len):
            mask[i, 0] = mask[i, i] = 1
    accepted = torch.tensor([0], dtype=torch.int32)
    if seq_len > 2:
        accepted = torch.cat([accepted, torch.tensor([2], dtype=torch.int32)])
    return mask, accepted


def pack_tree_mask(mask: torch.Tensor, seq_len: int,
                   batch_size: int) -> torch.Tensor:
    """Bit-pack tree mask to [B, S, ceil(S/32)] int32 for the XQA kernel.

    Column ``c`` of a row goes to word ``c // 32``, bit ``c % 32`` (so widths
    above 32 span multiple words). Word values are wrapped to signed int32 so
    bit 31 is representable.
    """
    num_packed = (seq_len + 31) // 32
    packed = torch.zeros((seq_len, num_packed), dtype=torch.int32)
    for row in range(seq_len):
        words = [0] * num_packed
        for col in range(seq_len):
            if int(mask[row, col]) == 1:
                words[col // 32] |= (1 << (col % 32))
        for w, val in enumerate(words):
            packed[row, w] = val - (1 << 32) if val >= (1 << 31) else val
    return packed[None].expand(batch_size, *packed.shape).contiguous()


def commit_kv_cache(k_cache: torch.Tensor, v_cache: torch.Tensor,
                    accepted_indices: torch.Tensor, current_pos: int,
                    seq_len: int):
    """Keep only accepted tokens in the cache (tree-attention commit step)."""
    acc = accepted_indices.to(torch.long)
    dst = torch.arange(current_pos, current_pos + len(acc))
    k = k_cache.clone()
    v = v_cache.clone()
    k[:, :, current_pos:current_pos + seq_len, :] = 0
    v[:, :, current_pos:current_pos + seq_len, :] = 0
    k[:, :, dst, :] = k_cache[:, :, current_pos + acc, :]
    v[:, :, dst, :] = v_cache[:, :, current_pos + acc, :]
    return k, v


def _fp8_supported() -> bool:
    return DEPENDENCIES_AVAILABLE and hasattr(torch, "float8_e4m3fn") \
        and hasattr(trt, "fp8")


def _device_sm() -> int:
    if not (DEPENDENCIES_AVAILABLE and torch.cuda.is_available()):
        return 0
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


# FP8 KV-cache support differs by phase: decode needs the FP8 XQA cubins
# (FP8-capable devices, sm89+), prefill additionally needs the CuTe DSL FMHA
# FP8 path (sm100/101/110 only -- FMHA_v2 and FFPA are FP16-only). Gate on
# the kernel support surface instead of trying the build: on unsupported SMs
# an FP8 engine build/enqueue can crash the process instead of erroring.
def _fp8_decode_supported() -> bool:
    return _fp8_supported() and _device_sm() >= 89


def _fp8_prefill_supported() -> bool:
    return _fp8_supported() and _device_sm() in (100, 101, 110)


class AttentionPluginRunner:
    """Builds + runs the AttentionPlugin for a given AttentionParams config.

    The plugin's kv_cache binding is a paged POOL [2, numPages, PAGE_SIZE,
    Hkv, D] plus an int32 page table [batch, 2, maxPagesPerSeq], but the tests
    keep working in the LOGICAL per-slot layout [batch, 2, Hkv, cap, D]:
    ``run`` scatters the logical cache into the pool under an identity page
    table (slot b owns pages [b*mpps, (b+1)*mpps)), executes, and gathers the
    pool back into the logical tensor in place.

    ``allow_empty_kv`` lowers the K/V profile minimum to sequence length 0 so
    the same engine also accepts shared-KV calls (K/V with S=0, Gemma4
    KV-sharing layers); the plugin deduces shared-KV per enqueue from the
    runtime K/V dims.
    """

    def __init__(self,
                 p: AttentionParams,
                 enable_tree_attention=False,
                 allow_empty_kv=False,
                 attention_scale: Optional[float] = None):
        self.p = p
        self.tree = enable_tree_attention
        self.allow_empty_kv = allow_empty_kv
        self.attention_scale = attention_scale
        self.kv_dtype = trt.fp8 if p.enable_fp8_kv_cache else trt.float16
        # Paged-pool geometry: capacity padded up to whole pages, one fixed
        # page range per batch slot (identity page table).
        self.cap_padded = -(-p.kv_cache_capacity // PAGE_SIZE) * PAGE_SIZE
        self.mpps = self.cap_padded // PAGE_SIZE  # maxPagesPerSeq
        self.num_pages = p.max_batch_size * self.mpps
        self._pool = None
        self._page_table = None
        self.runner = PluginRunner()
        self._build()

    def _build(self):
        p = self.p
        qh, kvh = p.q_hidden, p.kv_hidden
        D, Hkv = p.head_size, p.num_kv_heads
        mb, ms, mpe = p.max_batch_size, p.max_seq_len, p.max_position_embeddings

        input_specs = [
            ("q", trt.float16, (-1, -1, qh)),
            ("k", trt.float16, (-1, -1, kvh)),
            ("v", trt.float16, (-1, -1, kvh)),
            ("kv_cache", self.kv_dtype, (2, -1, PAGE_SIZE, Hkv, D)),
            ("context_lengths", trt.int32, (-1, )),
            ("rope_cos_sin", trt.float32, (1, mpe, D)),
            ("kv_cache_indices", trt.int32, (-1, )),
            ("kv_page_table", trt.int32, (-1, 2, self.mpps)),
        ]
        kv_min_seq = 0 if self.allow_empty_kv else 1
        # The pool never resizes: numPages is fixed per engine (min=opt=max).
        pool_shape = (2, self.num_pages, PAGE_SIZE, Hkv, D)
        profiles = {
            "q": ((1, 1, qh), (p.batch_size, p.seq_len, qh), (mb, ms, qh)),
            "k": ((1, kv_min_seq, kvh), (p.batch_size, p.seq_len, kvh),
                  (mb, ms, kvh)),
            "v": ((1, kv_min_seq, kvh), (p.batch_size, p.seq_len, kvh),
                  (mb, ms, kvh)),
            "kv_cache": (pool_shape, pool_shape, pool_shape),
            "context_lengths": ((1, ), (p.batch_size, ), (mb, )),
            "rope_cos_sin": ((1, mpe, D), (1, mpe, D), (1, mpe, D)),
            "kv_cache_indices": ((1, ), (p.batch_size, ), (mb, )),
            "kv_page_table": ((1, 2, self.mpps), (p.batch_size, 2, self.mpps),
                              (mb, 2, self.mpps)),
        }
        if self.tree:
            input_specs += [
                ("tree_mask", trt.int32, (-1, -1, -1)),
                ("position_ids", trt.int32, (-1, -1)),
            ]
            profiles["tree_mask"] = ((1, 1, 1), (p.batch_size, p.seq_len,
                                                 p.seq_len), (mb, ms, ms))
            profiles["position_ids"] = ((1, 1), (p.batch_size, p.seq_len),
                                        (mb, ms))

        fields = [
            pf_int32("num_q_heads", p.num_q_heads),
            pf_int32("num_kv_heads", p.num_kv_heads),
            pf_int32("head_size", p.head_size),
            pf_int32("enable_tree_attention", int(self.tree)),
            pf_int32("enable_fp8_kv_cache", int(p.enable_fp8_kv_cache)),
            pf_int32("sliding_window_size", p.sliding_window_size),
        ]
        if p.enable_fp8_kv_cache:
            fields.append(pf_float32("qkv_scales", p.qkv_scales))
        if self.attention_scale is not None:
            fields.append(pf_float32("attention_scale", self.attention_scale))

        self.runner.build(
            input_specs=input_specs,
            output_names=["attention_output", "kv_cache_output"],
            plugin_name="AttentionPlugin",
            plugin_version="1",
            plugin_fields=fields,
            profiles=profiles,
        )

    def _pool_views(self, kv_dtype):
        """The (lazily allocated) pool plus its K/V halves viewed logically.

        Under the identity page table the K half [numPages, PAGE_SIZE, Hkv, D]
        is exactly slot-major/token-major, so viewing it as [max_batch,
        cap_padded, Hkv, D] gives slot b's token t at [b, t] (NHD); same for
        the V half. Tokens [cap:cap_padded] are padding and stay zero.
        """
        p = self.p
        if self._pool is None:
            self._pool = torch.zeros(
                (2, self.num_pages, PAGE_SIZE, p.num_kv_heads, p.head_size),
                dtype=kv_dtype,
                device=DEV)
        view_shape = (p.max_batch_size, self.cap_padded, p.num_kv_heads,
                      p.head_size)
        return (self._pool, self._pool[0].view(view_shape),
                self._pool[1].view(view_shape))

    def _identity_page_table(self, batch):
        """int32 [batch, 2, mpps]: K row b -> pages [b*mpps, (b+1)*mpps),
        V row = K row + numPages (the V half of the pool)."""
        if self._page_table is None or self._page_table.shape[0] != batch:
            k_ids = torch.arange(batch * self.mpps,
                                 dtype=torch.int32,
                                 device=DEV).reshape(batch, self.mpps)
            self._page_table = torch.stack((k_ids, k_ids + self.num_pages),
                                           dim=1)
        return self._page_table

    def run(self,
            q,
            k,
            v,
            kv_cache,
            context_lengths,
            rope_cos_sin,
            cache_indices,
            tree_mask=None,
            position_ids=None,
            input_shapes=None):
        """Execute; returns (attn_output fp16, kv_cache after update).

        ``kv_cache`` is the LOGICAL cache [batch, 2, Hkv, cap, D]: it is
        scattered into the paged pool before the enqueue and the pool is
        gathered back into it (in place) afterwards, so callers never see the
        pool layout.

        ``input_shapes`` optionally overrides runtime input shapes (see
        PluginRunner.execute); used to bind K/V with sequence length 0 for
        shared-KV calls. kv_page_table always binds its full runtime shape
        (taken from the tensor itself).
        """
        p = self.p
        batch, cap = kv_cache.shape[0], p.kv_cache_capacity
        pool, pool_k, pool_v = self._pool_views(kv_cache.dtype)
        # Scatter logical [batch, 2, Hkv, cap, D] (HND) into the pool's
        # slot-major NHD view (identity page table).
        pool_k[:batch, :cap] = kv_cache[:, 0].permute(0, 2, 1, 3)
        pool_v[:batch, :cap] = kv_cache[:, 1].permute(0, 2, 1, 3)
        attn_out = torch.empty((q.shape[0], q.shape[1], p.q_hidden),
                               dtype=torch.float16,
                               device=DEV)
        tensors = {
            "q": q,
            "k": k,
            "v": v,
            "kv_cache": pool,
            "context_lengths": context_lengths,
            "rope_cos_sin": rope_cos_sin,
            "kv_cache_indices": cache_indices,
            "kv_page_table": self._identity_page_table(batch),
            "attention_output": attn_out,
            "kv_cache_output": pool,  # aliased in-place to the pool binding
        }
        if self.tree:
            tensors["tree_mask"] = tree_mask
            tensors["position_ids"] = position_ids
        self.runner.execute(tensors, input_shapes)
        # Gather the (possibly updated) pool back into the logical cache.
        kv_cache[:, 0] = pool_k[:batch, :cap].permute(0, 2, 1, 3)
        kv_cache[:, 1] = pool_v[:batch, :cap].permute(0, 2, 1, 3)
        return attn_out, kv_cache


# --------------------------------------------------------------------------- #
# Shared fixtures / builders
# --------------------------------------------------------------------------- #
def _make_rope(p: AttentionParams, gen):
    """Returns (cos_cache[mpe,half], sin_cache[mpe,half], combined[1,mpe,D]).

    Uses real RoPE (cos^2+sin^2=1, norm-preserving) rather than random values so
    the rotated q/k stay bounded at long sequence lengths -- random cos/sin grow
    the dot products unbounded and overflow fp16 in the plugin past ~1k tokens.
    """
    half = p.head_size // 2
    pos = torch.arange(p.max_position_embeddings, dtype=torch.float32)[:, None]
    inv_freq = 1.0 / (10000.0**(torch.arange(0, half, dtype=torch.float32) /
                                half))[None, :]
    ang = pos * inv_freq  # [mpe, half]
    cos = torch.cos(ang)
    sin = torch.sin(ang)
    combined = torch.zeros((1, p.max_position_embeddings, p.head_size),
                           dtype=torch.float32)
    combined[0, :, :half] = cos
    combined[0, :, half:] = sin
    return cos.to(DEV), sin.to(DEV), combined.to(DEV)


def _empty_caches(p: AttentionParams):
    """Reference caches (fp32) + plugin cache (fp16/fp8)."""
    ref_k = torch.zeros(
        (p.batch_size, p.num_kv_heads, p.kv_cache_capacity, p.head_size),
        dtype=torch.float32,
        device=DEV)
    ref_v = torch.zeros_like(ref_k)
    kv_dtype = torch.float8_e4m3fn if p.enable_fp8_kv_cache else torch.float16
    plugin_kv = torch.zeros(
        (p.batch_size, 2, p.num_kv_heads, p.kv_cache_capacity, p.head_size),
        dtype=kv_dtype,
        device=DEV)
    return ref_k, ref_v, plugin_kv


def _split_qkv(qkv, p):
    q = qkv[:, :, :p.q_hidden].to(torch.float16)
    k = qkv[:, :, p.q_hidden:p.q_hidden + p.kv_hidden].to(torch.float16)
    v = qkv[:, :, p.q_hidden + p.kv_hidden:].to(torch.float16)
    return q, k, v


def _plugin_kv_to_ref(plugin_kv, p):
    """Dequantize plugin KV cache -> (k_fp32, v_fp32) matching the reference."""
    ks, vs = (p.qkv_scales[1],
              p.qkv_scales[2]) if p.enable_fp8_kv_cache else (1.0, 1.0)
    k = plugin_kv[:, 0].float() * ks
    v = plugin_kv[:, 1].float() * vs
    return k, v


def _run_rounds(p: AttentionParams,
                num_rounds: int,
                atol: float,
                rtol: float,
                seed: int = 42,
                cos_threshold: float = 0.99999,
                attention_scale: Optional[float] = None):
    """Generic multi-round decode/prefill driver comparing plugin vs reference."""
    gen = torch.Generator().manual_seed(seed)
    runner = AttentionPluginRunner(p, attention_scale=attention_scale)
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)

    pos = 0
    for r in range(num_rounds):
        qkv = torch.randn((p.batch_size, p.seq_len, p.qkv_hidden_size),
                          generator=gen,
                          dtype=torch.float32).to(DEV)
        q, k, v = _split_qkv(qkv, p)
        position_ids = torch.arange(pos,
                                    pos + p.seq_len,
                                    dtype=torch.int32,
                                    device=DEV)[None].repeat(p.batch_size, 1)
        cache_idx = torch.full((p.batch_size, ),
                               pos,
                               dtype=torch.int32,
                               device=DEV)
        if p.is_prefill:
            ctx_len = torch.full((p.batch_size, ),
                                 p.seq_len,
                                 dtype=torch.int32,
                                 device=DEV)
            mask = sliding_window_mask(p.seq_len, pos + p.seq_len,
                                       p.sliding_window_size, DEV)
        else:
            ctx_len = torch.full((p.batch_size, ),
                                 pos + p.seq_len,
                                 dtype=torch.int32,
                                 device=DEV)
            mask = sliding_window_mask(p.seq_len, pos + p.seq_len,
                                       p.sliding_window_size, DEV) \
                if p.sliding_window_size > 0 else None

        ref_out, ref_k, ref_v = compute_attention(qkv.float(), ref_k, ref_v,
                                                  cos, sin, position_ids,
                                                  cache_idx, p, mask)

        attn_out, plugin_kv = runner.run(q, k, v, plugin_kv, ctx_len, combined,
                                         cache_idx)
        pk, pv = _plugin_kv_to_ref(plugin_kv, p)

        assert_close(f"attn[r{r}]",
                     ref_out,
                     attn_out,
                     atol=atol,
                     rtol=rtol,
                     cos_threshold=cos_threshold)
        # Only compare the populated cache region.
        end = pos + p.seq_len
        assert_close(f"k_cache[r{r}]",
                     ref_k[:, :, :end],
                     pk[:, :, :end],
                     atol=atol,
                     rtol=rtol,
                     cos_threshold=cos_threshold)
        assert_close(f"v_cache[r{r}]",
                     ref_v[:, :, :end],
                     pv[:, :, :end],
                     atol=atol,
                     rtol=rtol,
                     cos_threshold=cos_threshold)
        pos += p.seq_len


BASE = dict(num_q_heads=8,
            num_kv_heads=8,
            head_size=128,
            kv_cache_capacity=64,
            max_batch_size=8,
            max_seq_len=8,
            max_position_embeddings=64)

# --------------------------------------------------------------------------- #
# (head_size, num_q_heads, num_kv_heads) sweep, shared by the GQA prefill and
# decode tests. The plugin requires BOTH a prefill path (FMHA, or FFPA for
# head 512) and the decode XQA path to support a config. Supported space:
#   head 64/128 -> GQA ratio 1..8 (head 128 also supports ratio 16, Nemotron-H);
#   head 256    -> GQA ratio 2/4/6/8 only (XQA constraint; Qwen3.5 family);
#   head 512    -> GQA ratio 4/8 only (FFPA prefill + XQA-512 decode,
#                  Gemma4 E4B/E2B global attention layers).
# head 32 is excluded from this sweep: the prefill FMHA has no head-32 kernel,
# so the plugin runs in the degraded XQA-only mode (decode works, covered by
# test_decode_head32 below; prefill has no kernel and enqueue fails).
# --------------------------------------------------------------------------- #
ATTN_CONFIGS = [
    (64, 8, 8),
    (64, 8, 2),
    (64, 8, 1),
    (64, 16, 4),
    (64, 24, 3),
    (64, 32, 4),
    (128, 8, 8),
    (128, 8, 4),
    (128, 8, 2),
    (128, 8, 1),
    (128, 16, 4),
    (128, 32, 8),
    (128, 24, 3),
    (128, 32, 2),
    (256, 32, 8),
    (256, 24, 4),
    (256, 32, 4),
    (256, 16, 8),
    (512, 8, 2),
    (512, 8, 1),
]

# Prefill side of the sweep: head 512 is excluded (FFPA dense-causal prefill
# has its own dedicated tests below).
PREFILL_CONFIGS = [c for c in ATTN_CONFIGS if c[0] != 512]


# --------------------------------------------------------------------------- #
# Core: prefill / decode across batch sizes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 8], ids=lambda b: f"bs{b}")
def test_prefill(batch_size):
    p = AttentionParams(batch_size=batch_size,
                        seq_len=8,
                        is_prefill=True,
                        **BASE)
    _run_rounds(p, num_rounds=3, atol=1e-2, rtol=1e-2)


# --------------------------------------------------------------------------- #
# Grouped-query attention (prefill; the decode side is the test_gqa_decode
# sweep further down). Head 64/128 run CuTe DSL FMHA where available and
# FMHA_v2 elsewhere; head 256 runs FMHA_v2 on every SKU.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "head_size,num_q_heads,num_kv_heads",
    PREFILL_CONFIGS,
    ids=[f"head{h}_q{q}_kv{kv}" for h, q, kv in PREFILL_CONFIGS])
def test_gqa_prefill(head_size, num_q_heads, num_kv_heads):
    cfg = dict(BASE)
    cfg["head_size"] = head_size
    cfg["num_q_heads"] = num_q_heads
    cfg["num_kv_heads"] = num_kv_heads
    p = AttentionParams(batch_size=2, seq_len=8, is_prefill=True, **cfg)
    _run_rounds(p, num_rounds=2, atol=1e-2, rtol=1e-2)


# --------------------------------------------------------------------------- #
# head 512 prefill (FFPA path). FMHA has no head-512 kernels, so prefill runs
# the FFPA d512 causal kernel (Ampere instruction floor, all SMs). kv1/kv2 are
# the Gemma4 E2B / E4B global-attention-layer configs. Round 2 is a chunked
# continuation: FFPA reads the cache prefix back and per-batch cu_kv_seqlens
# drive the bottom-right causal offset.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_kv_heads", [1, 2], ids=lambda k: f"kv{k}")
def test_prefill_head512(num_kv_heads):
    cfg = dict(BASE)
    cfg["head_size"] = 512
    cfg["num_kv_heads"] = num_kv_heads
    p = AttentionParams(batch_size=2, seq_len=8, is_prefill=True, **cfg)
    _run_rounds(p, num_rounds=2, atol=1e-2, rtol=1e-2)


# --------------------------------------------------------------------------- #
# FP8 KV cache (+ non-unit qkv scales)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _fp8_decode_supported(),
                    reason="FP8 XQA decode not supported on this device")
@pytest.mark.parametrize("scales", [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]],
                         ids=["scale1.0", "scale0.5"])
def test_fp8_kv_cache_decode(scales):
    p = AttentionParams(batch_size=2,
                        seq_len=1,
                        enable_fp8_kv_cache=True,
                        qkv_scales=scales,
                        **BASE)
    # FP8 e4m3 KV-cache storage (~2-3 mantissa bits) lands at cos_sim ~0.99987,
    # below the 0.9999 bar used for FP16 paths -- this is the precision floor of
    # FP8 storage, not an error. Use a relaxed (but still tight) threshold.
    _run_rounds(p, num_rounds=4, atol=2e-1, rtol=2e-1, cos_threshold=0.999)


# --------------------------------------------------------------------------- #
# FP8 prefill (CuTe DSL FMHA): the RoPE kernel quantizes Q to FP8 and writes
# FP8 K/V to the interleaved cache, then the CuTe DSL FMHA consumes FP8 Q +
# FP8 cache directly. Only available where the CuTe DSL FMHA runs (sm100/101/
# 110, head 64/128); FMHA_v2 and FFPA prefill are FP16-only, so other SMs skip.
# Round 2 continues the prefill at an advancing cache offset, so the FMHA also
# reads back the FP8 KV written in round 1.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _fp8_prefill_supported(),
                    reason="FP8 CuTe DSL FMHA prefill not supported here")
@pytest.mark.parametrize("scales", [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]],
                         ids=["scale1.0", "scale0.5"])
def test_fp8_kv_cache_prefill(scales):
    p = AttentionParams(batch_size=2,
                        seq_len=8,
                        is_prefill=True,
                        enable_fp8_kv_cache=True,
                        qkv_scales=scales,
                        **BASE)
    # Same relaxed threshold as test_fp8_kv_cache_decode: the FP8 e4m3
    # storage precision floor, not an error.
    _run_rounds(p, num_rounds=2, atol=2e-1, rtol=2e-1, cos_threshold=0.999)


# --------------------------------------------------------------------------- #
# Configurable softmax scale through the real prefill and decode kernels.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("is_prefill", [True, False],
                         ids=["prefill", "decode"])
def test_configurable_softmax_scale(is_prefill):
    desired_scale = 0.37
    p = AttentionParams(batch_size=2,
                        seq_len=8 if is_prefill else 1,
                        is_prefill=is_prefill,
                        qk_scale=desired_scale,
                        **BASE)
    _run_rounds(p,
                num_rounds=2 if is_prefill else 4,
                atol=1e-2,
                rtol=1e-2,
                attention_scale=desired_scale)


# --------------------------------------------------------------------------- #
# Successive prefill chunks: prefill a short chunk, then prefill again at an
# advancing cache offset (each chunk attends within itself).
# --------------------------------------------------------------------------- #
def test_prefill_successive_chunks():
    p = AttentionParams(batch_size=2, seq_len=4, is_prefill=True, **BASE)
    _run_rounds(p, num_rounds=2, atol=1e-2, rtol=1e-2)


# --------------------------------------------------------------------------- #
# Ragged context lengths across the batch (decode with differing histories)
# --------------------------------------------------------------------------- #
def test_ragged_context_lengths_decode():
    p = AttentionParams(batch_size=4, seq_len=1, **BASE)
    gen = torch.Generator().manual_seed(7)
    runner = AttentionPluginRunner(p)
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)

    # Pre-fill differing amounts of history per batch row.
    histories = torch.tensor([0, 3, 7, 15], dtype=torch.int32)
    # Warm the caches by replaying decode steps up to each row's history.
    max_h = int(histories.max())
    for step in range(max_h):
        qkv = torch.randn((p.batch_size, 1, p.qkv_hidden_size),
                          generator=gen,
                          dtype=torch.float32).to(DEV)
        q, k, v = _split_qkv(qkv, p)
        pos_ids = torch.full((p.batch_size, 1),
                             step,
                             dtype=torch.int32,
                             device=DEV)
        cache_idx = torch.full((p.batch_size, ),
                               step,
                               dtype=torch.int32,
                               device=DEV)
        active = (histories > step).to(DEV)
        ctx_len = torch.full((p.batch_size, ),
                             step + 1,
                             dtype=torch.int32,
                             device=DEV)
        rk, rv = ref_k.clone(), ref_v.clone()
        ro, ref_k, ref_v = compute_attention(qkv.float(), ref_k, ref_v, cos,
                                             sin, pos_ids, cache_idx, p, None)
        attn_out, plugin_kv = runner.run(q, k, v, plugin_kv, ctx_len, combined,
                                         cache_idx)
        # Roll back inactive rows so each row only accumulates its own history.
        for bi in range(p.batch_size):
            if not bool(active[bi]):
                ref_k[bi], ref_v[bi] = rk[bi], rv[bi]
    # Compare the populated cache regions per row.
    pk, pv = _plugin_kv_to_ref(plugin_kv, p)
    for bi in range(p.batch_size):
        h = int(histories[bi])
        if h == 0:
            continue
        assert_close(f"k_cache[b{bi}]", ref_k[bi, :, :h], pk[bi, :, :h], 1e-2,
                     1e-2)
        assert_close(f"v_cache[b{bi}]", ref_v[bi, :, :h], pv[bi, :, :h], 1e-2,
                     1e-2)


# --------------------------------------------------------------------------- #
# Batch-order permutation invariance: shuffling the batch must permute the
# per-row decode outputs identically.
# --------------------------------------------------------------------------- #
def test_batch_permutation_invariance_decode():
    """Permuting the batch must permute outputs identically (no cross-row leak)."""
    p = AttentionParams(batch_size=4, seq_len=1, **BASE)
    gen = torch.Generator().manual_seed(123)
    cos, sin, combined = _make_rope(p, gen)

    qkv = torch.randn((p.batch_size, 1, p.qkv_hidden_size),
                      generator=gen,
                      dtype=torch.float32).to(DEV)
    q, k, v = _split_qkv(qkv, p)
    cache_idx = torch.zeros((p.batch_size, ), dtype=torch.int32, device=DEV)
    ctx_len = torch.ones((p.batch_size, ), dtype=torch.int32, device=DEV)

    runner = AttentionPluginRunner(p)
    _, _, kv0 = _empty_caches(p)
    out0, _ = runner.run(q, k, v, kv0, ctx_len, combined, cache_idx)

    perm = torch.tensor([2, 0, 3, 1], device=DEV)
    _, _, kv1 = _empty_caches(p)
    out1, _ = runner.run(q[perm], k[perm], v[perm], kv1, ctx_len, combined,
                         cache_idx)
    assert_close("batch-perm", out0[perm], out1, 1e-3, 1e-3)


# --------------------------------------------------------------------------- #
# Tree (speculative) attention
# --------------------------------------------------------------------------- #
def test_tree_attention():
    p = AttentionParams(batch_size=4, seq_len=4, **BASE)
    num_rounds = 5
    gen = torch.Generator().manual_seed(42)
    runner = AttentionPluginRunner(p, enable_tree_attention=True)
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)

    tree_mask, accepted = get_tree_attention_mask(p.seq_len)
    packed = pack_tree_mask(tree_mask, p.seq_len, p.batch_size).to(DEV)
    tree_mask = tree_mask.to(DEV)

    pos = 0
    base_depth = torch.tensor([0, 1, 1, 2], dtype=torch.int32)
    for r in range(num_rounds):
        qkv = torch.randn((p.batch_size, p.seq_len, p.qkv_hidden_size),
                          generator=gen,
                          dtype=torch.float32).to(DEV)
        q, k, v = _split_qkv(qkv, p)
        depth = base_depth[:p.seq_len]
        pos_ids = (pos + depth)[None].repeat(p.batch_size,
                                             1).to(torch.int32).to(DEV)
        cache_idx = torch.full((p.batch_size, ),
                               pos,
                               dtype=torch.int32,
                               device=DEV)
        ctx_len = torch.full((p.batch_size, ),
                             pos + p.seq_len,
                             dtype=torch.int32,
                             device=DEV)

        full_mask = torch.ones((p.seq_len, pos + p.seq_len),
                               dtype=torch.int32,
                               device=DEV)
        full_mask[:, pos:] = tree_mask
        ref_out, ref_k_out, ref_v_out = compute_attention(
            qkv.float(), ref_k, ref_v, cos, sin, pos_ids, cache_idx, p,
            full_mask)

        attn_out, plugin_kv = runner.run(q, k, v, plugin_kv, ctx_len, combined,
                                         cache_idx, packed, pos_ids)
        pk, pv = _plugin_kv_to_ref(plugin_kv, p)

        assert_close(f"tree-attn[r{r}]", ref_out, attn_out, 1e-2, 1e-2)
        end = pos + p.seq_len
        assert_close(f"tree-k[r{r}]", ref_k_out[:, :, :end], pk[:, :, :end],
                     1e-2, 1e-2)
        assert_close(f"tree-v[r{r}]", ref_v_out[:, :, :end], pv[:, :, :end],
                     1e-2, 1e-2)

        # Commit accepted tokens for the next round.
        ref_k, ref_v = commit_kv_cache(ref_k_out, ref_v_out, accepted, pos,
                                       p.seq_len)
        pk_c, pv_c = commit_kv_cache(pk, pv, accepted, pos, p.seq_len)
        # Re-quantize committed cache back into the plugin buffer for next round.
        kv_dtype = plugin_kv.dtype
        if p.enable_fp8_kv_cache:
            ks, vs = p.qkv_scales[1], p.qkv_scales[2]
            plugin_kv[:, 0] = (pk_c / ks).to(kv_dtype)
            plugin_kv[:, 1] = (pv_c / vs).to(kv_dtype)
        else:
            plugin_kv[:, 0] = pk_c.to(kv_dtype)
            plugin_kv[:, 1] = pv_c.to(kv_dtype)
        pos += int(len(accepted))


def _tree_from_parents(parent):
    """Build (mask[W,W] int32, depth[W] int32) from a parent list (parent[0]=-1).
    mask[i, j] = 1 iff j is on i's ancestor path (incl. itself); depth[i] is the
    node's tree depth (its RoPE position offset)."""
    width = len(parent)
    mask = torch.zeros((width, width), dtype=torch.int32)
    depth = torch.zeros(width, dtype=torch.int32)
    for i in range(width):
        if parent[i] != -1:
            depth[i] = depth[parent[i]] + 1
        j = i
        while j != -1:
            mask[i, j] = 1
            j = parent[j]
    return mask, depth


def _random_tree(width: int, gen):
    """Random speculative-decoding tree over ``width`` nodes: node i>0 picks a
    random parent in [0, i)."""
    parent = [-1] + [
        int(torch.randint(0, i, (1, ), generator=gen))
        for i in range(1, width)
    ]
    return _tree_from_parents(parent)


def _tree_for_kind(width: int, kind: str, gen):
    """Distinct tree topologies of a given width, to vary the mask shape:
    two random trees, a degenerate chain (max depth) and a star (max width)."""
    if kind == "chain":  # 0->1->2->...: single path, depth = width-1
        return _tree_from_parents([-1] + list(range(width - 1)))
    if kind == "star":  # every node is a direct child of the root, depth 1
        return _tree_from_parents([-1] + [0] * (width - 1))
    return _random_tree(width, gen)


# --------------------------------------------------------------------------- #
# Comprehensive tree attention: sweep tree width AND mask shape (two random
# trees plus degenerate chain/star topologies per width), one verification round.
# --------------------------------------------------------------------------- #
_TREE_KINDS = {"rand-a": 0, "rand-b": 1, "chain": 2, "star": 3}


@pytest.mark.parametrize("width", [8, 16, 32, 48, 64],
                         ids=lambda w: f"width{w}")
@pytest.mark.parametrize("kind", list(_TREE_KINDS))
def test_tree_attention_topology(width, kind):
    cfg = dict(BASE)
    cfg["kv_cache_capacity"] = 64
    cfg["max_seq_len"] = 64
    cfg["max_position_embeddings"] = 64
    p = AttentionParams(batch_size=2, seq_len=width, **cfg)
    gen = torch.Generator().manual_seed(300 + width * 4 + _TREE_KINDS[kind])
    runner = AttentionPluginRunner(p, enable_tree_attention=True)
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)

    tree_mask, depth = _tree_for_kind(width, kind, gen)
    packed = pack_tree_mask(tree_mask, width, p.batch_size).to(DEV)
    tree_mask = tree_mask.to(DEV)

    qkv = torch.randn((p.batch_size, width, p.qkv_hidden_size),
                      generator=gen,
                      dtype=torch.float32).to(DEV)
    q, k, v = _split_qkv(qkv, p)
    pos_ids = depth[None].repeat(p.batch_size, 1).to(DEV)
    cache_idx = torch.zeros(p.batch_size, dtype=torch.int32, device=DEV)
    ctx_len = torch.full((p.batch_size, ),
                         width,
                         dtype=torch.int32,
                         device=DEV)

    ref_out, _, _ = compute_attention(qkv.float(), ref_k, ref_v, cos, sin,
                                      pos_ids, cache_idx, p, tree_mask)
    attn_out, _ = runner.run(q, k, v, plugin_kv, ctx_len, combined, cache_idx,
                             packed, pos_ids)
    assert_close(f"tree-topology[{kind},W{width}]", ref_out, attn_out)


# --------------------------------------------------------------------------- #
# Decode across the full required batch-size set (1/2/3/4/8)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 8], ids=lambda b: f"bs{b}")
def test_decode(batch_size):
    p = AttentionParams(batch_size=batch_size, seq_len=1, **BASE)
    _run_rounds(p, num_rounds=4, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "head_size,num_q_heads,num_kv_heads",
    ATTN_CONFIGS,
    ids=[f"head{h}_q{q}_kv{kv}" for h, q, kv in ATTN_CONFIGS])
def test_gqa_decode(head_size, num_q_heads, num_kv_heads):
    cfg = dict(BASE)
    cfg["num_q_heads"] = num_q_heads
    cfg["num_kv_heads"] = num_kv_heads
    cfg["head_size"] = head_size
    p = AttentionParams(batch_size=2, seq_len=1, **cfg)
    _run_rounds(p, num_rounds=4, atol=1e-2, rtol=1e-2)


# --------------------------------------------------------------------------- #
# head 32 decode: the degraded "XQA-only" mode (no prefill kernel exists for
# head 32). XQA supports head 32 with GQA ratios 1-8. Note the plugin
# constructor advertises "naive attention for prefill" for this mode, but
# enqueue() has no such path -- a head-32 prefill call fails at enqueue time
# (verified on Thor), so only decode is tested.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_kv_heads", [8, 1], ids=lambda k: f"kv{k}")
def test_decode_head32(num_kv_heads):
    cfg = dict(BASE)
    cfg["head_size"] = 32
    cfg["num_kv_heads"] = num_kv_heads
    p = AttentionParams(batch_size=2, seq_len=1, **cfg)
    _run_rounds(p, num_rounds=4, atol=1e-2, rtol=1e-2)


# --------------------------------------------------------------------------- #
# Prefill -> decode handoff: a long prefill (ISL in 10..2048) fills the KV
# cache, then N decode steps continue from it. The plugin KV cache is shared
# across both phases; compared against a continuous reference.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2], ids=lambda s: f"seed{s}")
def test_prefill_decode_handoff(seed):
    # Prefill ISL randomly chosen in [10, 2048] (seeded for reproducibility).
    prefill_len = random.Random(9000 + seed).randint(10, 2048)
    bs, n_decode = 2, 4
    cap = prefill_len + n_decode + 8
    cfg = dict(BASE)
    cfg["kv_cache_capacity"] = cap
    cfg["max_seq_len"] = prefill_len
    cfg["max_position_embeddings"] = cap
    p = AttentionParams(batch_size=bs,
                        seq_len=prefill_len,
                        is_prefill=True,
                        **cfg)
    p_dec = AttentionParams(batch_size=bs, seq_len=1, **cfg)
    gen = torch.Generator().manual_seed(42424 + prefill_len)
    runner = AttentionPluginRunner(p)  # one engine handles prefill + decode
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)

    # --- prefill ---
    qkv = torch.randn((bs, prefill_len, p.qkv_hidden_size),
                      generator=gen,
                      dtype=torch.float32).to(DEV)
    q, k, v = _split_qkv(qkv, p)
    pos_ids = torch.arange(prefill_len, dtype=torch.int32,
                           device=DEV)[None].repeat(bs, 1)
    cache_idx = torch.zeros(bs, dtype=torch.int32, device=DEV)
    ctx_len = torch.full((bs, ), prefill_len, dtype=torch.int32, device=DEV)
    mask = sliding_window_mask(prefill_len, prefill_len, -1, DEV)
    ref_out, ref_k, ref_v = compute_attention(qkv.float(), ref_k, ref_v, cos,
                                              sin, pos_ids, cache_idx, p, mask)
    attn_out, plugin_kv = runner.run(q, k, v, plugin_kv, ctx_len, combined,
                                     cache_idx)
    assert_close("handoff-prefill", ref_out, attn_out)

    # --- decode steps, continuing from the prefilled KV cache ---
    pos = prefill_len
    for i in range(n_decode):
        qkv_d = torch.randn((bs, 1, p.qkv_hidden_size),
                            generator=gen,
                            dtype=torch.float32).to(DEV)
        q, k, v = _split_qkv(qkv_d, p_dec)
        pos_ids = torch.full((bs, 1), pos, dtype=torch.int32, device=DEV)
        cache_idx = torch.full((bs, ), pos, dtype=torch.int32, device=DEV)
        ctx_len = torch.full((bs, ), pos + 1, dtype=torch.int32, device=DEV)
        ref_out, ref_k, ref_v = compute_attention(qkv_d.float(), ref_k, ref_v,
                                                  cos, sin, pos_ids, cache_idx,
                                                  p_dec, None)
        attn_out, plugin_kv = runner.run(q, k, v, plugin_kv, ctx_len, combined,
                                         cache_idx)
        assert_close(f"handoff-decode[t={pos}]", ref_out, attn_out)
        pos += 1


# --------------------------------------------------------------------------- #
# Ragged prefill: one prefill call where each batch row has a different valid
# length (padded to the max), per-row context_lengths, padding poisoned. The
# reference computes each row's causal attention over its own valid length.
# --------------------------------------------------------------------------- #
def _ragged_prefill_ref(qkv, cos, sin, seqlens, p):
    """Per-row causal attention over each row's valid length. Returns a list of
    [L_b, q_hidden] outputs."""
    outs = []
    for b, L in enumerate(seqlens):
        q = qkv[b:b + 1, :L, :p.q_hidden].reshape(1, L, p.num_q_heads,
                                                  p.head_size).transpose(1, 2)
        k = qkv[b:b + 1, :L, p.q_hidden:p.q_hidden + p.kv_hidden].reshape(
            1, L, p.num_kv_heads, p.head_size).transpose(1, 2)
        v = qkv[b:b + 1, :L, p.q_hidden + p.kv_hidden:].reshape(
            1, L, p.num_kv_heads, p.head_size).transpose(1, 2)
        pos = torch.arange(L, dtype=torch.int32, device=DEV)[None]
        q = apply_rotary_embedding(q, cos, sin, pos)
        k = apply_rotary_embedding(k, cos, sin, pos)
        mask = sliding_window_mask(L, L, -1, DEV)
        o = scaled_dot_product_attention(q, k, v, p.qk_scale, mask,
                                         p.num_q_heads, p.num_kv_heads)
        outs.append(o.transpose(1, 2).reshape(L, p.num_q_heads * p.head_size))
    return outs


def _ragged_attn_params(seqlens):
    maxlen = max(seqlens)
    cfg = dict(BASE)
    cfg["num_kv_heads"] = 8
    cfg["head_size"] = 64
    cfg["kv_cache_capacity"] = maxlen
    cfg["max_seq_len"] = maxlen
    cfg["max_position_embeddings"] = maxlen
    cfg["max_batch_size"] = 8
    return AttentionParams(batch_size=len(seqlens),
                           seq_len=maxlen,
                           is_prefill=True,
                           **cfg)


def _ragged_qkv(seqlens, p, gen):
    maxlen = max(seqlens)
    qkv = torch.randn((len(seqlens), maxlen, p.qkv_hidden_size),
                      generator=gen,
                      dtype=torch.float32).to(DEV)
    poison_padding([qkv], seqlens)
    return qkv


# Required even/uneven batch cases (bs 1/2/3/4/8, seq up to 2048) as one ragged
# prefill call, per-row context_lengths, padding poisoned.
@pytest.mark.parametrize("label,seqlens", RAGGED_CASES)
def test_ragged_prefill(label, seqlens):
    p = _ragged_attn_params(seqlens)
    gen = torch.Generator().manual_seed(7777 + max(seqlens) + len(seqlens))
    runner = AttentionPluginRunner(p)
    cos, sin, combined = _make_rope(p, gen)
    _, _, plugin_kv = _empty_caches(p)
    qkv = _ragged_qkv(seqlens, p, gen)
    q, k, v = _split_qkv(qkv, p)
    ctx_len = torch.tensor(seqlens, dtype=torch.int32, device=DEV)
    cache_idx = torch.zeros(len(seqlens), dtype=torch.int32, device=DEV)
    attn_out, _ = runner.run(q, k, v, plugin_kv, ctx_len, combined, cache_idx)
    ref_rows = _ragged_prefill_ref(qkv.float(), cos, sin, seqlens, p)
    for b, L in enumerate(seqlens):
        assert_close(f"ragged[{label}].b{b}", ref_rows[b], attn_out[b, :L])


# Batch invariance on RAGGED input (plugin-vs-plugin): permuting the rows (and
# their context lengths) must permute the per-row outputs identically.
#
# NOTE: the AttentionPlugin applies RoPE in place to its Q/K input buffers
# (launchApplyRopeWriteKV writes the rotated Q/K back into the input tensors).
# That is benign in the real graph, where Q/K come fresh from the QKV projection
# each iteration, but a test that reuses the same Q/K buffer across two enqueues
# would rotate it twice. So each enqueue must get its own copy of Q/K/V.
def test_ragged_prefill_batch_invariance():
    seqlens = [10, 2048, 128]
    p = _ragged_attn_params(seqlens)
    gen = torch.Generator().manual_seed(8888)
    runner = AttentionPluginRunner(p)
    cos, sin, combined = _make_rope(p, gen)
    qkv = _ragged_qkv(seqlens, p, gen)
    q, k, v = _split_qkv(qkv, p)
    cache_idx = torch.zeros(len(seqlens), dtype=torch.int32, device=DEV)

    _, _, kv0 = _empty_caches(p)
    ctx0 = torch.tensor(seqlens, dtype=torch.int32, device=DEV)
    # clone so the in-place RoPE does not corrupt the buffers reused below
    out0, _ = runner.run(q.clone(), k.clone(), v.clone(), kv0, ctx0, combined,
                         cache_idx)

    perm = [2, 0, 1]
    sl_p = [seqlens[i] for i in perm]
    _, _, kv1 = _empty_caches(p)
    ctx1 = torch.tensor(sl_p, dtype=torch.int32, device=DEV)
    out1, _ = runner.run(q[perm].contiguous(), k[perm].contiguous(),
                         v[perm].contiguous(), kv1, ctx1, combined, cache_idx)
    for new_i, orig in enumerate(perm):
        L = seqlens[orig]
        assert_close(f"ragged-batch-inv[{new_i}]", out0[orig, :L],
                     out1[new_i, :L])


# --------------------------------------------------------------------------- #
# Shared KV (Gemma4 KV-sharing layers): the plugin detects shared-KV mode from
# K/V inputs with sequence length 0. The KV-cache input is then a DONOR layer's
# cache (already-RoPE'd K + raw V); the plugin applies RoPE to Q ONLY and
# attends against the donor cache without writing to it. Each test first runs
# an own-KV pass through the same engine to populate the donor cache (the K/V
# profile min is lowered to S=0 so one engine serves both modes).
# --------------------------------------------------------------------------- #
def _empty_kv(p: AttentionParams):
    """K/V binding for shared-KV calls: (dummy tensor, input-shape overrides).

    The runtime K/V shape must be [B, 0, Hkv*D], but a 0-element torch tensor
    reports ``data_ptr() == 0`` and TensorRT rejects a null binding address.
    So bind a 1-token dummy buffer and override the runtime shape to S=0 (the
    plugin never reads K/V in shared-KV mode)."""
    dummy = torch.zeros((p.batch_size, 1, p.kv_hidden),
                        dtype=torch.float16,
                        device=DEV)
    shapes = {
        "k": (p.batch_size, 0, p.kv_hidden),
        "v": (p.batch_size, 0, p.kv_hidden),
    }
    return dummy, shapes


def _assert_cache_untouched(name: str, before: "torch.Tensor",
                            after: "torch.Tensor"):
    """Bit-exact check that a shared-KV call did not write the donor cache."""
    assert torch.equal(before.view(torch.int16), after.view(torch.int16)), \
        f"{name}: shared-KV call must not modify the donor KV cache"


# Shared-KV prefill. head 128 runs the CuTe DSL FMHA path where available
# (SM100+) and FMHA_v2 elsewhere; head 256 forces the FMHA_v2 deinterleave
# path on every SKU (CuTe DSL FMHA only supports head 64/128); head 512 runs
# FFPA against the deinterleaved donor cache (the Gemma4 E2B/E4B
# KV-sharing-layer configs).
@pytest.mark.parametrize("head_size,num_q_heads,num_kv_heads", [(128, 8, 4),
                                                                (256, 16, 8),
                                                                (512, 8, 1),
                                                                (512, 8, 2)],
                         ids=[
                             "head128_q8_kv4", "head256_q16_kv8",
                             "head512_q8_kv1", "head512_q8_kv2"
                         ])
def test_shared_kv_prefill(head_size, num_q_heads, num_kv_heads):
    cfg = dict(BASE)
    cfg["head_size"] = head_size
    cfg["num_q_heads"] = num_q_heads
    cfg["num_kv_heads"] = num_kv_heads
    p = AttentionParams(batch_size=2, seq_len=8, is_prefill=True, **cfg)
    gen = torch.Generator().manual_seed(2400 + head_size)
    runner = AttentionPluginRunner(p, allow_empty_kv=True)
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)
    b, s = p.batch_size, p.seq_len

    pos_ids = torch.arange(s, dtype=torch.int32, device=DEV)[None].repeat(b, 1)
    cache_idx = torch.zeros(b, dtype=torch.int32, device=DEV)
    ctx_len = torch.full((b, ), s, dtype=torch.int32, device=DEV)
    mask = sliding_window_mask(s, s, -1, DEV)

    # Donor pass (own KV) populates the cache: RoPE'd K + raw V.
    qkv = torch.randn((b, s, p.qkv_hidden_size),
                      generator=gen,
                      dtype=torch.float32).to(DEV)
    q, k, v = _split_qkv(qkv, p)
    _, ref_k, ref_v = compute_attention(qkv.float(), ref_k, ref_v, cos, sin,
                                        pos_ids, cache_idx, p, mask)
    runner.run(q, k, v, plugin_kv, ctx_len, combined, cache_idx)

    # Shared-KV pass: fresh Q for the same positions, K/V with S=0, the donor
    # cache as KV-cache input.
    q2 = torch.randn((b, s, p.q_hidden), generator=gen,
                     dtype=torch.float32).to(DEV)
    kv_dummy, kv_shapes = _empty_kv(p)
    donor_before = plugin_kv.clone()
    attn_out, plugin_kv = runner.run(q2.to(torch.float16),
                                     kv_dummy,
                                     kv_dummy,
                                     plugin_kv,
                                     ctx_len,
                                     combined,
                                     cache_idx,
                                     input_shapes=kv_shapes)

    # Reference: RoPE Q at positions 0..S-1, causal attention against the
    # donor cache contents.
    q2r = apply_rotary_embedding(
        q2.reshape(b, s, p.num_q_heads, p.head_size).transpose(1, 2), cos, sin,
        pos_ids)
    ref_out = scaled_dot_product_attention(q2r, ref_k[:, :, :s],
                                           ref_v[:, :, :s], p.qk_scale, mask,
                                           p.num_q_heads, p.num_kv_heads)
    ref_out = ref_out.transpose(1, 2).reshape(b, s, p.q_hidden)
    assert_close("shared-kv-prefill", ref_out, attn_out)
    _assert_cache_untouched("shared-kv-prefill", donor_before, plugin_kv)


# Shared-KV decode (XQA with RoPE-Q-only): each step first runs an own-KV
# donor decode that writes the new K/V slot, then a shared-KV decode whose Q
# (roped at position ctx-1) attends the donor cache including that slot.
def test_shared_kv_decode():
    cfg = dict(BASE)
    cfg["num_kv_heads"] = 4
    p = AttentionParams(batch_size=2, seq_len=8, is_prefill=True, **cfg)
    p_dec = AttentionParams(batch_size=2, seq_len=1, **cfg)
    gen = torch.Generator().manual_seed(2500)
    # One engine (dynamic S, K/V min S=0) serves donor prefill, donor decode
    # and shared-KV decode.
    runner = AttentionPluginRunner(p, allow_empty_kv=True)
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)
    b, s = p.batch_size, p.seq_len

    # Donor prefill populates the cache.
    qkv = torch.randn((b, s, p.qkv_hidden_size),
                      generator=gen,
                      dtype=torch.float32).to(DEV)
    q, k, v = _split_qkv(qkv, p)
    pos_ids = torch.arange(s, dtype=torch.int32, device=DEV)[None].repeat(b, 1)
    cache_idx = torch.zeros(b, dtype=torch.int32, device=DEV)
    ctx_len = torch.full((b, ), s, dtype=torch.int32, device=DEV)
    mask = sliding_window_mask(s, s, -1, DEV)
    _, ref_k, ref_v = compute_attention(qkv.float(), ref_k, ref_v, cos, sin,
                                        pos_ids, cache_idx, p, mask)
    runner.run(q, k, v, plugin_kv, ctx_len, combined, cache_idx)

    pos = s
    for step in range(3):
        pos_ids = torch.full((b, 1), pos, dtype=torch.int32, device=DEV)
        cache_idx = torch.full((b, ), pos, dtype=torch.int32, device=DEV)
        ctx_len = torch.full((b, ), pos + 1, dtype=torch.int32, device=DEV)

        # Donor decode: own-KV step that writes the K/V slot at ``pos``.
        qkv_d = torch.randn((b, 1, p.qkv_hidden_size),
                            generator=gen,
                            dtype=torch.float32).to(DEV)
        q, k, v = _split_qkv(qkv_d, p_dec)
        _, ref_k, ref_v = compute_attention(qkv_d.float(), ref_k, ref_v, cos,
                                            sin, pos_ids, cache_idx, p_dec,
                                            None)
        runner.run(q, k, v, plugin_kv, ctx_len, combined, cache_idx)

        # Shared-KV decode: fresh Q, K/V with S=0, no cache write.
        q_s = torch.randn((b, 1, p.q_hidden),
                          generator=gen,
                          dtype=torch.float32).to(DEV)
        kv_dummy, kv_shapes = _empty_kv(p_dec)
        donor_before = plugin_kv.clone()
        attn_out, plugin_kv = runner.run(q_s.to(torch.float16),
                                         kv_dummy,
                                         kv_dummy,
                                         plugin_kv,
                                         ctx_len,
                                         combined,
                                         cache_idx,
                                         input_shapes=kv_shapes)

        # Reference: RoPE Q at position ctx-1, attend all ctx donor entries.
        qr = apply_rotary_embedding(
            q_s.reshape(b, 1, p.num_q_heads, p.head_size).transpose(1, 2), cos,
            sin, pos_ids)
        ref_out = scaled_dot_product_attention(qr, ref_k[:, :, :pos + 1],
                                               ref_v[:, :, :pos + 1],
                                               p.qk_scale, None, p.num_q_heads,
                                               p.num_kv_heads)
        ref_out = ref_out.transpose(1, 2).reshape(b, 1, p.q_hidden)
        assert_close(f"shared-kv-decode[t={pos}]", ref_out, attn_out)
        _assert_cache_untouched(f"shared-kv-decode[t={pos}]", donor_before,
                                plugin_kv)
        pos += 1


# Shared-KV chunked prefill: the second chunk's Q must also attend the donor
# cache prefix, driving the chunked shared-KV kernel variants (CuTe DSL FMHA
# via padded cu_kv_seqlens, FMHA_v2 with s_kv=capacity, FFPA with the
# bottom-right causal offset -- head 512 is the Gemma4 E4B config).
@pytest.mark.parametrize(
    "head_size,num_q_heads,num_kv_heads", [(128, 8, 4), (256, 16, 8),
                                           (512, 8, 2)],
    ids=["head128_q8_kv4", "head256_q16_kv8", "head512_q8_kv2"])
def test_shared_kv_chunked_prefill(head_size, num_q_heads, num_kv_heads):
    cfg = dict(BASE)
    cfg["head_size"] = head_size
    cfg["num_q_heads"] = num_q_heads
    cfg["num_kv_heads"] = num_kv_heads
    p = AttentionParams(batch_size=2, seq_len=8, is_prefill=True, **cfg)
    gen = torch.Generator().manual_seed(2600 + head_size)
    runner = AttentionPluginRunner(p, allow_empty_kv=True)
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)
    b, s = p.batch_size, p.seq_len

    # Two donor chunks (own KV) build a 2S-token cache.
    ctx_len = torch.full((b, ), s, dtype=torch.int32, device=DEV)
    for pos in (0, s):
        pos_ids = torch.arange(pos, pos + s, dtype=torch.int32,
                               device=DEV)[None].repeat(b, 1)
        cache_idx = torch.full((b, ), pos, dtype=torch.int32, device=DEV)
        mask = sliding_window_mask(s, pos + s, -1, DEV)
        qkv = torch.randn((b, s, p.qkv_hidden_size),
                          generator=gen,
                          dtype=torch.float32).to(DEV)
        q, k, v = _split_qkv(qkv, p)
        _, ref_k, ref_v = compute_attention(qkv.float(), ref_k, ref_v, cos,
                                            sin, pos_ids, cache_idx, p, mask)
        runner.run(q, k, v, plugin_kv, ctx_len, combined, cache_idx)

    # Shared-KV second chunk: fresh Q at positions S..2S-1 attends the full
    # donor cache (prefix + current chunk). pos_ids/cache_idx/mask still hold
    # the chunk-2 values from the loop.
    q2 = torch.randn((b, s, p.q_hidden), generator=gen,
                     dtype=torch.float32).to(DEV)
    kv_dummy, kv_shapes = _empty_kv(p)
    donor_before = plugin_kv.clone()
    attn_out, plugin_kv = runner.run(q2.to(torch.float16),
                                     kv_dummy,
                                     kv_dummy,
                                     plugin_kv,
                                     ctx_len,
                                     combined,
                                     cache_idx,
                                     input_shapes=kv_shapes)

    q2r = apply_rotary_embedding(
        q2.reshape(b, s, p.num_q_heads, p.head_size).transpose(1, 2), cos, sin,
        pos_ids)
    ref_out = scaled_dot_product_attention(q2r, ref_k[:, :, :2 * s],
                                           ref_v[:, :, :2 * s], p.qk_scale,
                                           mask, p.num_q_heads, p.num_kv_heads)
    ref_out = ref_out.transpose(1, 2).reshape(b, s, p.q_hidden)
    assert_close("shared-kv-chunked", ref_out, attn_out)
    _assert_cache_untouched("shared-kv-chunked", donor_before, plugin_kv)


# Shared-KV tree (speculative) decode: the donor layer's own-KV tree pass
# writes the candidate K/V, then the shared layer's Q -- roped with the tree
# position IDs (the RoPE-Q-only tree dispatch) -- attends the donor cache
# under the same tree mask. Round 2 runs with a committed cache prefix.
def test_shared_kv_tree_decode():
    cfg = dict(BASE)
    cfg["num_kv_heads"] = 4
    p = AttentionParams(batch_size=2, seq_len=4, **cfg)
    gen = torch.Generator().manual_seed(2700)
    runner = AttentionPluginRunner(p,
                                   enable_tree_attention=True,
                                   allow_empty_kv=True)
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)
    b, s = p.batch_size, p.seq_len

    tree_mask, accepted = get_tree_attention_mask(s)
    packed = pack_tree_mask(tree_mask, s, b).to(DEV)
    tree_mask = tree_mask.to(DEV)
    base_depth = torch.tensor([0, 1, 1, 2], dtype=torch.int32)

    pos = 0
    for r in range(2):
        pos_ids = (pos + base_depth)[None].repeat(b, 1).to(torch.int32).to(DEV)
        cache_idx = torch.full((b, ), pos, dtype=torch.int32, device=DEV)
        ctx_len = torch.full((b, ), pos + s, dtype=torch.int32, device=DEV)
        full_mask = torch.ones((s, pos + s), dtype=torch.int32, device=DEV)
        full_mask[:, pos:] = tree_mask

        # Donor tree pass (own KV) writes the candidate K/V slots.
        qkv = torch.randn((b, s, p.qkv_hidden_size),
                          generator=gen,
                          dtype=torch.float32).to(DEV)
        q, k, v = _split_qkv(qkv, p)
        _, ref_k_out, ref_v_out = compute_attention(qkv.float(), ref_k, ref_v,
                                                    cos, sin, pos_ids,
                                                    cache_idx, p, full_mask)
        runner.run(q, k, v, plugin_kv, ctx_len, combined, cache_idx, packed,
                   pos_ids)

        # Shared-KV tree pass: fresh Q, K/V with S=0, donor cache read-only.
        q2 = torch.randn((b, s, p.q_hidden),
                         generator=gen,
                         dtype=torch.float32).to(DEV)
        kv_dummy, kv_shapes = _empty_kv(p)
        donor_before = plugin_kv.clone()
        attn_out, plugin_kv = runner.run(q2.to(torch.float16),
                                         kv_dummy,
                                         kv_dummy,
                                         plugin_kv,
                                         ctx_len,
                                         combined,
                                         cache_idx,
                                         packed,
                                         pos_ids,
                                         input_shapes=kv_shapes)

        end = pos + s
        q2r = apply_rotary_embedding(
            q2.reshape(b, s, p.num_q_heads, p.head_size).transpose(1, 2), cos,
            sin, pos_ids)
        ref_out = scaled_dot_product_attention(q2r, ref_k_out[:, :, :end],
                                               ref_v_out[:, :, :end],
                                               p.qk_scale, full_mask,
                                               p.num_q_heads, p.num_kv_heads)
        ref_out = ref_out.transpose(1, 2).reshape(b, s, p.q_hidden)
        assert_close(f"shared-kv-tree[r{r}]", ref_out, attn_out)
        _assert_cache_untouched(f"shared-kv-tree[r{r}]", donor_before,
                                plugin_kv)

        # Commit accepted tokens in both caches for the next round.
        ref_k, ref_v = commit_kv_cache(ref_k_out, ref_v_out, accepted, pos, s)
        pk, pv = _plugin_kv_to_ref(plugin_kv, p)
        pk_c, pv_c = commit_kv_cache(pk, pv, accepted, pos, s)
        plugin_kv[:, 0] = pk_c.to(plugin_kv.dtype)
        plugin_kv[:, 1] = pv_c.to(plugin_kv.dtype)
        pos += int(len(accepted))
