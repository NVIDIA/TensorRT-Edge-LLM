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
tree (speculative) attention, vision-block attention, shared-KV (donor-cache)
layers, fused qk_norm (per-head RMSNorm on Q/K before RoPE), and the Q
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

import mmap
import multiprocessing
import os
import random
import traceback
from dataclasses import dataclass, field, replace

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              RAGGED_CASES, PluginRunner,
                              PluginUnsupportedError, _device_sm, assert_close,
                              find_plugin_library, pf_float32, pf_int32,
                              poison_padding)

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

# D512 native-paged CuTe DSL SM-support contract, hand-maintained (not read
# from build_cutedsl.py) so a dropped kernel on a supported SM fails, not skips
# green.
NATIVE_SMS = frozenset({80, 86, 87, 89, 90, 100, 101, 110, 120, 121})
CUTEDSL_D512_FP8_SMS = frozenset({100, 101, 110})
# FP8 KV-cache support is constrained by the bundled XQA decode kernels.
FP8_KV_CACHE_SMS = frozenset({89, 90, 100, 101, 110, 120, 121})
_CUTEDSL_MODULE_FAILURE_HOOK = "TRT_EDGELLM_TEST_FAIL_CUTEDSL_MODULE"
_CUTEDSL_LAZY_ATTENTION_MODULE_BY_SM = {
    80: "fmha_v2_d128_paged",
    86: "fmha_v2_d128_paged",
    87: "fmha_v2_d128_paged",
    89: "fmha_v2_d128_paged",
    90: "fmha_v2_d128_paged",
    100: "fmha_d128_paged",
    101: "fmha_d128_paged",
    110: "fmha_d128_paged",
    120: "fmha_v2_d128_paged",
    121: "fmha_v2_d128_paged",
}


def _cutedsl_module_failure_hook_built() -> bool:
    """Return whether the hook and this SM's target module exist in the plugin."""
    plugin_path = find_plugin_library()
    module_name = _CUTEDSL_LAZY_ATTENTION_MODULE_BY_SM.get(_device_sm())
    if plugin_path is None or module_name is None:
        return False
    try:
        with open(plugin_path, "rb") as plugin_file:
            if os.fstat(plugin_file.fileno()).st_size == 0:
                return False
            with mmap.mmap(plugin_file.fileno(), 0,
                           access=mmap.ACCESS_READ) as plugin_image:
                return (plugin_image.find(
                    _CUTEDSL_MODULE_FAILURE_HOOK.encode()) >= 0
                        and plugin_image.find(module_name.encode()) >= 0)
    except OSError:
        return False


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
    rms_norm_eps: float = 1e-6  # qk_norm epsilon (used when gammas are set)

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

    @property
    def tag(self) -> str:
        """Compact config label for assert_close failure messages."""
        fp8 = " fp8kv" if self.enable_fp8_kv_cache else ""
        return (f"h{self.head_size} q{self.num_q_heads}/kv{self.num_kv_heads} "
                f"bs{self.batch_size} s{self.seq_len} "
                f"sw{self.sliding_window_size}{fp8}")


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


def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-head RMSNorm in FP32 (qk_norm): x * rsqrt(mean(x^2) + eps) * gamma.

    x: [..., head_size]; gamma: [head_size]. Mirrors the plugin's fused
    qk_norm, which normalizes Q and K per head before RoPE (V is untouched).
    """
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) +
                            eps) * gamma.float()


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
        mask = attn_mask[None,
                         None] if attn_mask.ndim == 2 else attn_mask[:, None]
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def attention_bidirectional_mask(
        vision_block_ids: torch.Tensor,
        context_lengths: torch.Tensor,
        sliding_window_size: int = -1) -> torch.Tensor:
    """Causal/sliding mask OR each contiguous non-negative vision-ID run."""
    batch_size, seq_len = vision_block_ids.shape
    base = sliding_window_mask(seq_len, seq_len, sliding_window_size, DEV)
    mask = base[None].expand(batch_size, -1, -1).clone().to(torch.bool)
    block_ids_cpu = vision_block_ids.cpu()
    context_lengths_cpu = context_lengths.cpu()
    for batch_idx in range(batch_size):
        context_len = min(int(context_lengths_cpu[batch_idx]), seq_len)
        mask[batch_idx, context_len:, :] = False
        mask[batch_idx, :, context_len:] = False
        begin = 0
        while begin < context_len:
            block_id = int(block_ids_cpu[batch_idx, begin])
            end = begin + 1
            while end < context_len and int(block_ids_cpu[batch_idx,
                                                          end]) == block_id:
                end += 1
            if block_id >= 0:
                mask[batch_idx, begin:end, begin:end] = True
            begin = end
    return mask.to(torch.int32)


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
    q_norm_gamma: Optional[torch.Tensor] = None,
    k_norm_gamma: Optional[torch.Tensor] = None,
    rms_norm_eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full attention with RoPE + KV cache. Returns (out, k_cache, v_cache).

    Shapes: qkv [b, s, qkv_hidden]; k_cache/v_cache [b, Hkv, cap, d]
            (post-RoPE keys, raw values — as the plugin stores them).
    Caches are returned updated (not in-place).

    ``q_norm_gamma`` / ``k_norm_gamma`` ([head_size]) enable qk_norm: per-head
    FP32 RMSNorm applied to Q / K before RoPE (V is never normalized).
    """
    b, s = params.batch_size, params.seq_len
    Hq, Hkv, d = params.num_q_heads, params.num_kv_heads, params.head_size

    q = qkv[:, :, :params.q_hidden]
    k = qkv[:, :, params.q_hidden:params.q_hidden + params.kv_hidden]
    v = qkv[:, :, params.q_hidden + params.kv_hidden:]

    q = q.reshape(b, s, Hq, d).transpose(1, 2)
    k = k.reshape(b, s, Hkv, d).transpose(1, 2)
    v = v.reshape(b, s, Hkv, d).transpose(1, 2)

    if q_norm_gamma is not None:
        q = rms_norm(q, q_norm_gamma, rms_norm_eps)
    if k_norm_gamma is not None:
        k = rms_norm(k, k_norm_gamma, rms_norm_eps)

    q = apply_rotary_embedding(q, cos_cache, sin_cache, position_ids)
    k = apply_rotary_embedding(k, cos_cache, sin_cache, position_ids)

    if params.enable_fp8_kv_cache:
        qs, ks, vs = params.qkv_scales
        # Optimized common FMHA consumes FP8 Q on SM100/101/110. FMHA-v2
        # prefill and XQA decode consume FP16 Q, while K/V always round-trip
        # through the FP8 cache.
        if params.is_prefill and _device_sm() in (100, 101, 110):
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


# Cache-reading prefill works on every supported FP8 KV-cache SKU: native FP8
# on Blackwell, else the split-KV gather dequantizes FP8->FP16 for the
# FMHA-v2 consumers.
def _fp8_available() -> bool:
    return _fp8_supported() and _device_sm() in FP8_KV_CACHE_SMS


class AttentionPluginRunner:
    """Builds + runs the AttentionPlugin for a given AttentionParams config.

    The plugin takes a single packed QKV input [B, S, C].  The
    ``enable_kv_shared`` plugin field selects the layout: C = (Hq + 2*Hkv)*D
    for normal (own-KV) layers, or C = Hq*D (Q only) for shared-KV (Gemma4
    KV-sharing) layers, where K/V are read from a donor layer's cache without
    being written.

    The plugin's kv_cache binding is a paged POOL [2, numPages, PAGE_SIZE,
    Hkv, D] plus an int32 page table [batch, 2, maxPagesPerSeq], but the tests
    keep working in the LOGICAL per-slot layout [batch, 2, Hkv, cap, D]:
    ``run`` scatters the logical cache into the pool under an identity page
    table (slot b owns pages [b*mpps, (b+1)*mpps)), executes, and gathers the
    pool back into the logical tensor in place.

    ``q_norm_gamma`` / ``k_norm_gamma`` ([head_size] float tensors) enable
    fused qk_norm: per-head FP32 RMSNorm on Q and K before RoPE. They are
    wired as OPTIONAL engine-weight constant inputs (enable_qk_norm=1); when
    omitted the plugin gets no gamma inputs at all (enable_qk_norm=0).
    """

    def __init__(self,
                 p: AttentionParams,
                 enable_tree_attention=False,
                 q_norm_gamma=None,
                 k_norm_gamma=None,
                 attention_scale: Optional[float] = None,
                 enable_kv_shared: int = 0,
                 enable_context_mask_selector: bool = False,
                 enable_vision_block_attention: bool = False,
                 expect_unsupported: bool = False,
                 shuffle_pages: bool = False):
        self.p = p
        self.tree = enable_tree_attention
        self.vision = enable_vision_block_attention
        self.q_norm_gamma = q_norm_gamma
        self.k_norm_gamma = k_norm_gamma
        self.attention_scale = attention_scale
        self.kv_shared = enable_kv_shared
        self.context_mask_selector = enable_context_mask_selector
        self.expect_unsupported = expect_unsupported
        # shuffle_pages: give each slot non-contiguous physical pages so the
        # page table stops being identity -- proves the kernel follows it.
        self.shuffle_pages = shuffle_pages
        self._page_perm = None
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
        qh = p.q_hidden
        D, Hkv = p.head_size, p.num_kv_heads
        mb, ms, mpe = p.max_batch_size, p.max_seq_len, p.max_position_embeddings

        input_specs = [
            ("qkv", trt.float16, (-1, -1, -1)),
            ("kv_cache", self.kv_dtype, (2, -1, PAGE_SIZE, Hkv, D)),
            ("context_lengths", trt.int32, (-1, )),
            ("rope_cos_sin", trt.float32, (1, mpe, D)),
            ("kv_cache_indices", trt.int32, (-1, )),
            ("kv_page_table", trt.int32, (-1, 2, self.mpps)),
        ]
        # The pool never resizes: numPages is fixed per engine (min=opt=max).
        pool_shape = (2, self.num_pages, PAGE_SIZE, Hkv, D)
        # Channel width is fixed by the mode: Q-only (q_hidden) for shared-KV
        # engines, the full packed width otherwise.
        qkv_c = qh if self.kv_shared else p.qkv_hidden_size
        profiles = {
            "qkv":
            ((1, 1, qkv_c), (p.batch_size, p.seq_len, qkv_c), (mb, ms, qkv_c)),
            "kv_cache": (pool_shape, pool_shape, pool_shape),
            "context_lengths": ((1, ), (p.batch_size, ), (mb, )),
            "rope_cos_sin": ((1, mpe, D), (1, mpe, D), (1, mpe, D)),
            # min 0: an empty kv_cache_indices binding selects normal prefill.
            "kv_cache_indices": ((0, ), (p.batch_size, ), (mb, )),
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
        if self.vision:
            # vision_block_ids [B, S]: -1 for text/pad, non-negative per image
            # run. Occupies the optional attention-mask input slot.
            input_specs.append(("vision_block_ids", trt.int32, (-1, -1)))
            profiles["vision_block_ids"] = ((1, 1), (p.batch_size, p.seq_len),
                                            (mb, ms))

        # qk_norm gammas are OPTIONAL engine-weight constant inputs, wired only
        # when enable_qk_norm=1.
        qk_norm = self.q_norm_gamma is not None or self.k_norm_gamma is not None
        constant_specs = []
        plugin_input_order = [
            "qkv", "kv_cache", "context_lengths", "rope_cos_sin",
            "kv_cache_indices", "kv_page_table"
        ]
        if qk_norm:
            constant_specs = [
                ("q_norm_gamma", trt.float16, (D, ), self.q_norm_gamma),
                ("k_norm_gamma", trt.float16, (D, ), self.k_norm_gamma),
            ]
            plugin_input_order += ["q_norm_gamma", "k_norm_gamma"]
        if self.context_mask_selector:
            input_specs.append(("context_mask_selector", trt.int32, (-1, )))
            profiles["context_mask_selector"] = ((0, ), (p.batch_size, ),
                                                 (mb, ))
            plugin_input_order.append("context_mask_selector")
        if self.tree:
            plugin_input_order += ["tree_mask", "position_ids"]
        if self.vision:
            plugin_input_order.append("vision_block_ids")

        fields = [
            pf_int32("num_q_heads", p.num_q_heads),
            pf_int32("num_kv_heads", p.num_kv_heads),
            pf_int32("head_size", p.head_size),
            pf_int32("enable_tree_attention", int(self.tree)),
            pf_int32("enable_vision_block_attention", int(self.vision)),
            pf_int32("enable_qk_norm", int(qk_norm)),
            pf_int32("enable_kv_shared", int(self.kv_shared)),
            pf_int32("enable_context_mask_selector",
                     int(self.context_mask_selector)),
            pf_int32("enable_fp8_kv_cache", int(p.enable_fp8_kv_cache)),
            pf_int32("sliding_window_size", p.sliding_window_size),
        ]
        if p.enable_fp8_kv_cache:
            fields.append(pf_float32("qkv_scales", p.qkv_scales))
        if self.attention_scale is not None:
            fields.append(pf_float32("attention_scale", self.attention_scale))
        if qk_norm:
            fields.append(pf_float32("rms_norm_eps", p.rms_norm_eps))

        self.runner.build(
            input_specs=input_specs,
            output_names=["attention_output", "kv_cache_output"],
            plugin_name="AttentionPlugin",
            plugin_version="1",
            plugin_fields=fields,
            profiles=profiles,
            constant_specs=constant_specs,
            plugin_input_order=plugin_input_order,
            expect_unsupported=self.expect_unsupported,
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

    def _k_page_ids(self, batch):
        """[batch, mpps] physical K page ids. Identity = b*mpps+pi; shuffled =
        a fixed random permutation of all num_pages pages (non-contiguous)."""
        if self.shuffle_pages:
            if self._page_perm is None:
                perm = torch.randperm(
                    self.num_pages, generator=torch.Generator().manual_seed(7))
                self._page_perm = perm.to(torch.int32).to(DEV).reshape(
                    self.p.max_batch_size, self.mpps)
            return self._page_perm[:batch]
        return torch.arange(batch * self.mpps, dtype=torch.int32,
                            device=DEV).reshape(batch, self.mpps)

    def _identity_page_table(self, batch):
        """int32 [batch, 2, mpps]: K page ids then V page ids (K + numPages,
        the V half of the pool). K ids are contiguous (identity) or a fixed
        permutation (shuffle_pages)."""
        if self._page_table is None or self._page_table.shape[0] != batch:
            k_ids = self._k_page_ids(batch)
            self._page_table = torch.stack((k_ids, k_ids + self.num_pages),
                                           dim=1)
        return self._page_table

    def run(self,
            qkv,
            kv_cache,
            context_lengths,
            rope_cos_sin,
            cache_indices,
            tree_mask=None,
            position_ids=None,
            vision_block_ids=None,
            input_shapes=None,
            context_mask_selector=None,
            attention_output=None):
        """Execute; returns (attn_output fp16, kv_cache after update).

        ``qkv`` is the packed [B, S, (Hq+2*Hkv)*D] input, or a Q-only
        [B, S, Hq*D] tensor when the runner was built with
        ``enable_kv_shared=1``.

        ``kv_cache`` is the LOGICAL cache [batch, 2, Hkv, cap, D]: it is
        scattered into the paged pool before the enqueue and the pool is
        gathered back into it (in place) afterwards, so callers never see the
        pool layout.

        ``input_shapes`` optionally overrides runtime input shapes (see
        PluginRunner.execute). kv_page_table always binds its full runtime
        shape (taken from the tensor itself).

        ``attention_output`` optionally supplies the output buffer so failure
        tests can verify that enqueue returned before any output write.
        """
        p = self.p
        batch, cap = kv_cache.shape[0], p.kv_cache_capacity
        pool, pool_k, pool_v = self._pool_views(kv_cache.dtype)
        if self.shuffle_pages:
            self._scatter_paged(pool, kv_cache, batch)
        else:
            # Scatter logical [batch, 2, Hkv, cap, D] (HND) into the pool's
            # slot-major NHD view (identity page table).
            pool_k[:batch, :cap] = kv_cache[:, 0].permute(0, 2, 1, 3)
            pool_v[:batch, :cap] = kv_cache[:, 1].permute(0, 2, 1, 3)
        attn_out = attention_output
        if attn_out is None:
            attn_out = torch.empty((qkv.shape[0], qkv.shape[1], p.q_hidden),
                                   dtype=torch.float16,
                                   device=DEV)
        tensors = {
            "qkv": qkv,
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
        if self.context_mask_selector:
            tensors["context_mask_selector"] = context_mask_selector
        if self.vision:
            tensors["vision_block_ids"] = vision_block_ids
        self.runner.execute(tensors, input_shapes)
        # Gather the (possibly updated) pool back into the logical cache.
        if self.shuffle_pages:
            self._gather_paged(pool, kv_cache, batch)
        else:
            kv_cache[:, 0] = pool_k[:batch, :cap].permute(0, 2, 1, 3)
            kv_cache[:, 1] = pool_v[:batch, :cap].permute(0, 2, 1, 3)
        return attn_out, kv_cache

    def _paged_indices(self, batch):
        """Flat physical page ids into the [2*numPages, ...] pool for the K
        then V halves of each slot's pages, in logical page order."""
        k_ids = self._k_page_ids(batch).long().reshape(-1)  # [batch*mpps]
        return k_ids, k_ids + self.num_pages

    def _scatter_paged(self, pool, kv_cache, batch):
        """Place each logical page at its (possibly shuffled) physical page."""
        p = self.p
        cap, capp = p.kv_cache_capacity, self.cap_padded
        Hkv, D = p.num_kv_heads, p.head_size
        flat = pool.view(2 * self.num_pages, PAGE_SIZE, Hkv, D)
        k_idx, v_idx = self._paged_indices(batch)
        for half, idx in ((0, k_idx), (1, v_idx)):
            logical = torch.zeros((batch, capp, Hkv, D),
                                  dtype=pool.dtype,
                                  device=DEV)
            logical[:, :cap] = kv_cache[:, half].permute(0, 2, 1, 3)
            flat[idx] = logical.reshape(batch * self.mpps, PAGE_SIZE, Hkv, D)

    def _gather_paged(self, pool, kv_cache, batch):
        p = self.p
        cap, capp = p.kv_cache_capacity, self.cap_padded
        Hkv, D = p.num_kv_heads, p.head_size
        flat = pool.view(2 * self.num_pages, PAGE_SIZE, Hkv, D)
        k_idx, v_idx = self._paged_indices(batch)
        for half, idx in ((0, k_idx), (1, v_idx)):
            pages = flat[idx].reshape(batch, capp, Hkv, D)
            kv_cache[:, half] = pages[:, :cap].permute(0, 2, 1, 3)


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
                q_prescale: float = 1.0,
                q_norm_gamma=None,
                k_norm_gamma=None,
                attention_scale: Optional[float] = None,
                shuffle_pages: bool = False):
    """Generic multi-round decode/prefill driver comparing plugin vs reference.

    ``q_prescale`` multiplies the plugin-side Q only (the export-time Q
    pre-scaling convention for a non-default softmax scale); the reference
    keeps the unscaled Q and applies ``p.qk_scale`` directly.

    ``q_norm_gamma`` / ``k_norm_gamma`` enable fused qk_norm in the plugin and
    the matching RMSNorm in the reference.

    ``attention_scale`` passes a non-default absolute QK^T multiplier to the
    plugin via the ``attention_scale`` field (the reference applies
    ``p.qk_scale`` directly).
    """
    gen = torch.Generator().manual_seed(seed)
    runner = AttentionPluginRunner(p,
                                   q_norm_gamma=q_norm_gamma,
                                   k_norm_gamma=k_norm_gamma,
                                   attention_scale=attention_scale,
                                   shuffle_pages=shuffle_pages)
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)

    pos = 0
    for r in range(num_rounds):
        qkv = torch.randn((p.batch_size, p.seq_len, p.qkv_hidden_size),
                          generator=gen,
                          dtype=torch.float32).to(DEV)
        qkv_plugin = qkv.to(torch.float16)  # fresh copy per enqueue
        if q_prescale != 1.0:
            qkv_plugin[:, :, :p.q_hidden] = (qkv[:, :, :p.q_hidden] *
                                             q_prescale).to(torch.float16)
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

        ref_out, ref_k, ref_v = compute_attention(qkv.float(),
                                                  ref_k,
                                                  ref_v,
                                                  cos,
                                                  sin,
                                                  position_ids,
                                                  cache_idx,
                                                  p,
                                                  mask,
                                                  q_norm_gamma=q_norm_gamma,
                                                  k_norm_gamma=k_norm_gamma,
                                                  rms_norm_eps=p.rms_norm_eps)

        # First prefill
        input_shapes = {"kv_cache_indices": (0, )} \
            if p.is_prefill and r == 0 else None
        attn_out, plugin_kv = runner.run(qkv_plugin,
                                         plugin_kv,
                                         ctx_len,
                                         combined,
                                         cache_idx,
                                         input_shapes=input_shapes)
        pk, pv = _plugin_kv_to_ref(plugin_kv, p)

        phase = (("chunked-prefill" if r else "prefill")
                 if p.is_prefill else "decode")
        case = f"{p.tag} {phase} r{r}"
        assert_close(f"attn[{case}]",
                     ref_out,
                     attn_out,
                     atol=atol,
                     rtol=rtol,
                     cos_threshold=cos_threshold)
        # Only compare the populated cache region.
        end = pos + p.seq_len
        assert_close(f"k_cache[{case}]",
                     ref_k[:, :, :end],
                     pk[:, :, :end],
                     atol=atol,
                     rtol=rtol,
                     cos_threshold=cos_threshold)
        assert_close(f"v_cache[{case}]",
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


def _exercise_cutedsl_lazy_module_failure_hook():
    """Exercise one exact attention module through a real plugin enqueue."""
    sm_version = _device_sm()
    module_name = _CUTEDSL_LAZY_ATTENTION_MODULE_BY_SM[sm_version]
    os.environ[_CUTEDSL_MODULE_FAILURE_HOOK] = module_name
    try:
        p = AttentionParams(batch_size=1, seq_len=8, is_prefill=True, **BASE)
        generator = torch.Generator().manual_seed(314159)
        runner = AttentionPluginRunner(p)
        cos, sin, combined = _make_rope(p, generator)
        ref_k, ref_v, plugin_kv = _empty_caches(p)
        qkv = torch.randn((p.batch_size, p.seq_len, p.qkv_hidden_size),
                          generator=generator,
                          dtype=torch.float32).to(DEV).to(torch.float16)
        context_lengths = torch.full((p.batch_size, ),
                                     p.seq_len,
                                     dtype=torch.int32,
                                     device=DEV)
        cache_indices = torch.zeros(p.batch_size,
                                    dtype=torch.int32,
                                    device=DEV)
        prefill_shapes = {"kv_cache_indices": (0, )}

        pool, pool_k, pool_v = runner._pool_views(plugin_kv.dtype)
        pool_k[:p.batch_size, :p.kv_cache_capacity] = plugin_kv[:, 0].permute(
            0, 2, 1, 3)
        pool_v[:p.batch_size, :p.kv_cache_capacity] = plugin_kv[:, 1].permute(
            0, 2, 1, 3)
        failed_output = torch.full((p.batch_size, p.seq_len, p.q_hidden),
                                   -123.0,
                                   dtype=torch.float16,
                                   device=DEV)
        torch.cuda.synchronize()
        qkv_before = qkv.clone()
        pool_before = pool.clone()
        output_before = failed_output.clone()

        with pytest.raises(RuntimeError,
                           match="execute_async_v3 returned False"):
            runner.run(qkv,
                       plugin_kv,
                       context_lengths,
                       combined,
                       cache_indices,
                       input_shapes=prefill_shapes,
                       attention_output=failed_output)

        assert torch.equal(qkv_before.view(torch.int16),
                           qkv.view(torch.int16)), \
            "lazy-load failure must precede AttentionPlugin QKV preprocessing"
        assert torch.equal(pool_before.view(torch.int16),
                           pool.view(torch.int16)), \
            "lazy-load failure must precede AttentionPlugin KV-cache writes"
        assert torch.equal(output_before.view(torch.int16),
                           failed_output.view(torch.int16)), \
            "lazy-load failure must precede AttentionPlugin output writes"

        del os.environ[_CUTEDSL_MODULE_FAILURE_HOOK]
        failed_context = runner.runner.context
        runner.runner.context = runner.runner.engine.create_execution_context()
        assert runner.runner.context is not None
        del failed_context

        position_ids = torch.arange(p.seq_len, dtype=torch.int32,
                                    device=DEV)[None]
        causal_mask = sliding_window_mask(p.seq_len, p.seq_len,
                                          p.sliding_window_size, DEV)
        ref_out, ref_k, ref_v = compute_attention(qkv.float(), ref_k, ref_v,
                                                  cos, sin, position_ids,
                                                  cache_indices, p,
                                                  causal_mask)
        retry_output = torch.full_like(failed_output, -123.0)
        actual_out, actual_kv = runner.run(qkv.clone(),
                                           plugin_kv,
                                           context_lengths,
                                           combined,
                                           cache_indices,
                                           input_shapes=prefill_shapes,
                                           attention_output=retry_output)
        actual_k, actual_v = _plugin_kv_to_ref(actual_kv, p)
        assert_close("lazy-module-retry-attention", ref_out, actual_out)
        assert_close("lazy-module-retry-k-cache", ref_k, actual_k)
        assert_close("lazy-module-retry-v-cache", ref_v, actual_v)
    finally:
        os.environ.pop(_CUTEDSL_MODULE_FAILURE_HOOK, None)


def _cutedsl_lazy_module_failure_hook_child(connection):
    try:
        _exercise_cutedsl_lazy_module_failure_hook()
    except BaseException:
        connection.send(traceback.format_exc())
        raise
    else:
        connection.send("")
    finally:
        connection.close()


@pytest.mark.skipif(
    _device_sm() not in _CUTEDSL_LAZY_ATTENTION_MODULE_BY_SM,
    reason="representative CuTe DSL attention module is unsupported on this SM"
)
def test_cutedsl_lazy_module_failure_propagates_before_attention_mutation(
        request):
    """A failed first load is non-mutating and retryable on a fresh context."""
    if not _cutedsl_module_failure_hook_built():
        message = ("plugin lacks the CuTe DSL test hook or target attention "
                   "module")
        if request.config.getoption("--priority") == "l0_python_ut":
            pytest.fail(message)
        pytest.skip(message)

    spawn_context = multiprocessing.get_context("spawn")
    receiver, sender = spawn_context.Pipe(duplex=False)
    process = spawn_context.Process(
        target=_cutedsl_lazy_module_failure_hook_child, args=(sender, ))
    process.start()
    sender.close()
    process.join(timeout=300)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("spawned CuTe DSL lazy-module plugin test timed out")

    child_error = receiver.recv() if receiver.poll() else ""
    receiver.close()
    assert process.exitcode == 0, \
        child_error or f"spawned plugin test exited with code {process.exitcode}"


# --------------------------------------------------------------------------- #
# (head_size, num_q_heads, num_kv_heads) sweep, shared by the GQA prefill and
# decode tests. The plugin requires BOTH a prefill FMHA path and the decode XQA
# path to support a config. Supported space:
#   head 64/128 -> GQA ratio 1..8 (head 128 also supports ratio 16, Nemotron-H);
#   head 256    -> GQA ratio 2/4/6/8 only (XQA constraint; Qwen3.5 family);
#   head 512    -> any GQA ratio through D512 CuTe DSL paged prefill, with
#                  XQA-512 decode (Gemma4 E4B/E2B global attention layers).
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
    (512, 16, 1),
    (512, 4, 2),
]

# Prefill side of the sweep: head 512 has its own routing test below.
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
# head 512 prefill routing. On SM100/101/110, normal and chunked prefill use
# optimized common FMHA's native-paged ABI. On the remaining CuTe DSL targets,
# both modes use FMHA-v2's native-paged ABI.
# kv1/kv2 are the Gemma4 E2B / E4B global-attention-layer configs.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(_device_sm() not in NATIVE_SMS,
                    reason="D512 CuTe DSL FMHA is unavailable on this SM")
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(8, 1), (8, 2), (16, 1)],
                         ids=["q8_kv1", "q8_kv2", "q16_kv1"])
def test_prefill_head512(num_q_heads, num_kv_heads):
    cfg = dict(BASE)
    cfg["head_size"] = 512
    cfg["num_q_heads"] = num_q_heads
    cfg["num_kv_heads"] = num_kv_heads
    p = AttentionParams(batch_size=2, seq_len=8, is_prefill=True, **cfg)
    _run_rounds(p, num_rounds=2, atol=1e-2, rtol=1e-2)


# Numerical success covers the optimized path on SM100/101/110 and the FMHA-v2
# D512 implementation elsewhere; Nsight route validation separately
# distinguishes native-paged from gather.
@pytest.mark.skipif(_device_sm() not in NATIVE_SMS,
                    reason="D512 CuTe DSL FMHA is unavailable on this SM")
@pytest.mark.parametrize("sliding_window", [-1, 4], ids=["causal", "sliding"])
def test_prefill_head512_cutedsl_gqa_2(sliding_window):
    cfg = dict(BASE)
    cfg["head_size"] = 512
    cfg["num_q_heads"] = 4
    cfg["num_kv_heads"] = 2
    cfg["sliding_window_size"] = sliding_window
    p = AttentionParams(batch_size=2, seq_len=8, is_prefill=True, **cfg)
    _run_rounds(p, num_rounds=2, atol=1e-2, rtol=1e-2)


def _run_vision_prefill_case(*,
                             label: str,
                             head_size: int,
                             num_q_heads: int,
                             num_kv_heads: int,
                             seq_len: int,
                             sliding_window_size: int,
                             vision_ranges,
                             attention_scale: float = 1.0):
    # Require plugin construction explicitly because PluginRunner otherwise
    # treats an unavailable plugin as a platform skip.
    torch.empty(0, device=DEV)
    PluginRunner()
    creator = trt.get_plugin_registry().get_creator("AttentionPlugin", "1", "")
    assert creator is not None
    plugin = creator.create_plugin(
        "AttentionPlugin",
        trt.PluginFieldCollection([
            pf_int32("num_q_heads", num_q_heads),
            pf_int32("num_kv_heads", num_kv_heads),
            pf_int32("head_size", head_size),
            pf_int32("enable_vision_block_attention", 1),
            pf_int32("sliding_window_size", sliding_window_size),
        ]), trt.TensorRTPhase.BUILD)
    assert plugin is not None

    p = AttentionParams(batch_size=1,
                        seq_len=seq_len,
                        num_q_heads=num_q_heads,
                        num_kv_heads=num_kv_heads,
                        head_size=head_size,
                        kv_cache_capacity=seq_len,
                        max_batch_size=1,
                        max_seq_len=seq_len,
                        max_position_embeddings=seq_len,
                        qk_scale=attention_scale,
                        is_prefill=True,
                        sliding_window_size=sliding_window_size)
    gen = torch.Generator().manual_seed(625)
    runner = AttentionPluginRunner(p,
                                   attention_scale=attention_scale,
                                   enable_vision_block_attention=True)
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)
    qkv = (0.1 * torch.randn(
        (1, seq_len, p.qkv_hidden_size), generator=gen,
        dtype=torch.float32)).to(DEV)
    position_ids = torch.arange(seq_len, dtype=torch.int32, device=DEV)[None]
    cache_idx = torch.zeros(1, dtype=torch.int32, device=DEV)
    context_lengths = torch.full((1, ), seq_len, dtype=torch.int32, device=DEV)
    vision_block_ids = torch.full((1, seq_len),
                                  -1,
                                  dtype=torch.int32,
                                  device=DEV)
    for block_id, (begin, end) in enumerate(vision_ranges):
        vision_block_ids[:, begin:end] = block_id
    mask = attention_bidirectional_mask(vision_block_ids, context_lengths,
                                        sliding_window_size)
    ref_out, _, _ = compute_attention(qkv, ref_k, ref_v, cos, sin,
                                      position_ids, cache_idx, p, mask)
    attn_out, _ = runner.run(qkv.to(torch.float16),
                             plugin_kv,
                             context_lengths,
                             combined,
                             cache_idx,
                             vision_block_ids=vision_block_ids,
                             input_shapes={"kv_cache_indices": (0, )})
    assert_close(label, ref_out, attn_out, atol=2e-2, rtol=2e-2)


def _run_head512_vision_prefill_case(case: str):
    gemma4_shape = case == "gemma4_blocks"
    seq_len = 128 if gemma4_shape else 129
    num_q_heads = 16 if gemma4_shape else 4
    num_kv_heads = 1 if gemma4_shape else 2
    sliding_window_size = 32 if case == "blocks_sliding" else -1
    vision_ranges = ()
    if case != "all_text":
        vision_ranges = ((8, 40), (80, 112)) if gemma4_shape else ((8, 32),
                                                                   (120, 129))

    _run_vision_prefill_case(label=f"vision-d512-{case}",
                             head_size=512,
                             num_q_heads=num_q_heads,
                             num_kv_heads=num_kv_heads,
                             seq_len=seq_len,
                             sliding_window_size=sliding_window_size,
                             vision_ranges=vision_ranges)


@pytest.mark.skipif(
    _device_sm() not in (80, 86, 87, 89, 90, 100, 101, 110, 120, 121),
    reason="D256 FMHA-v2 bidirectional attention requires a supported CUDA SM")
@pytest.mark.parametrize("case", ["all_text", "page_crossing"])
def test_prefill_head256_vision_fmha_v2_bidirectional(case):
    """Exercise the Gemma4 D256 FMHA-v2 bidirectional route."""
    vision_ranges = () if case == "all_text" else ((120, 129), )
    sliding_window_size = 1024 if case == "all_text" else 32
    _run_vision_prefill_case(label=f"vision-d256-{case}",
                             head_size=256,
                             num_q_heads=16,
                             num_kv_heads=8,
                             seq_len=129,
                             sliding_window_size=sliding_window_size,
                             vision_ranges=vision_ranges)


@pytest.mark.skipif(_device_sm() not in NATIVE_SMS,
                    reason="D512 CuTe DSL FMHA is unavailable on this SM")
@pytest.mark.parametrize(
    "case", ["all_text", "blocks", "blocks_sliding", "gemma4_blocks"])
def test_prefill_head512_vision_cutedsl(case):
    """Prove all D512 vision modes route through a CuTe DSL FMHA kernel.

    SM100/101/110 take the paged bidirectional kernel; the remaining supported
    SMs take the FMHA-v2 D512 vision-block kernel.
    """
    _run_head512_vision_prefill_case(case)


@pytest.mark.skipif(_device_sm() not in NATIVE_SMS,
                    reason="D512 CuTe DSL FMHA is unavailable on this SM")
def test_prefill_head512_vision_scaled():
    """A non-default attention scale must reach the D512 vision kernel."""
    _run_vision_prefill_case(label="vision-d512-scaled",
                             head_size=512,
                             num_q_heads=16,
                             num_kv_heads=1,
                             seq_len=128,
                             sliding_window_size=-1,
                             vision_ranges=((8, 40), (80, 112)),
                             attention_scale=0.37)


def _run_vision_prefill_then_decode(*,
                                    head_size,
                                    num_q_heads,
                                    num_kv_heads,
                                    window,
                                    seq_len=24,
                                    decode_rounds=3):
    """Gemma4 Unified generation path: a vision-block prefill (image tokens
    bidirectional within their block) then vanilla XQA decode (vision_block_ids
    all -1) over a shuffled page table."""
    b = 2
    cfg = dict(BASE)
    cfg["head_size"] = head_size
    cfg["num_q_heads"] = num_q_heads
    cfg["num_kv_heads"] = num_kv_heads
    cfg["kv_cache_capacity"] = 256
    cfg["max_seq_len"] = 256
    cfg["max_position_embeddings"] = 256
    p = AttentionParams(batch_size=b,
                        seq_len=seq_len,
                        is_prefill=True,
                        sliding_window_size=window,
                        **cfg)
    gen = torch.Generator().manual_seed(808)
    runner = AttentionPluginRunner(p,
                                   enable_vision_block_attention=True,
                                   shuffle_pages=True)
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)
    vbid_row = torch.full((seq_len, ), -1, dtype=torch.int32)
    vbid_row[4:seq_len - 4] = 0
    vbid = vbid_row[None].repeat(b, 1).to(DEV)
    ctx = torch.full((b, ), seq_len, dtype=torch.int32, device=DEV)
    mask = attention_bidirectional_mask(vbid, ctx, window)
    qkv = torch.randn((b, seq_len, p.qkv_hidden_size),
                      generator=gen,
                      dtype=torch.float32).to(DEV)
    pos_ids = torch.arange(seq_len, dtype=torch.int32,
                           device=DEV)[None].repeat(b, 1)
    zero_idx = torch.zeros(b, dtype=torch.int32, device=DEV)
    ref_out, ref_k, ref_v = compute_attention(qkv.float(), ref_k, ref_v, cos,
                                              sin, pos_ids, zero_idx, p, mask)
    idx_dummy, idx_shapes = _empty_cache_indices(p)
    attn_out, plugin_kv = runner.run(qkv.to(torch.float16),
                                     plugin_kv,
                                     ctx,
                                     combined,
                                     idx_dummy,
                                     input_shapes=idx_shapes,
                                     vision_block_ids=vbid)
    assert_close("vision-prefill", ref_out, attn_out, 1e-2, 1e-2)

    # Decode generated text tokens (window >> history so no clipping).
    p_dec = replace(p, seq_len=1, is_prefill=False)
    decode_vbid = torch.full((b, 1), -1, dtype=torch.int32, device=DEV)
    pos = seq_len
    for r in range(decode_rounds):
        qkv_d = torch.randn((b, 1, p.qkv_hidden_size),
                            generator=gen,
                            dtype=torch.float32).to(DEV)
        pos_ids = torch.full((b, 1), pos, dtype=torch.int32, device=DEV)
        cache_idx = torch.full((b, ), pos, dtype=torch.int32, device=DEV)
        ctx = torch.full((b, ), pos + 1, dtype=torch.int32, device=DEV)
        ref_out, ref_k, ref_v = compute_attention(qkv_d.float(), ref_k, ref_v,
                                                  cos, sin, pos_ids, cache_idx,
                                                  p_dec, None)
        attn_out, plugin_kv = runner.run(qkv_d.to(torch.float16),
                                         plugin_kv,
                                         ctx,
                                         combined,
                                         cache_idx,
                                         vision_block_ids=decode_vbid)
        pk, pv = _plugin_kv_to_ref(plugin_kv, p)
        end = pos + 1
        assert_close(f"vision-decode[r{r}]", ref_out, attn_out, 1e-2, 1e-2)
        assert_close(f"vision-decode-k[r{r}]", ref_k[:, :, :end],
                     pk[:, :, :end], 1e-2, 1e-2)
        assert_close(f"vision-decode-v[r{r}]", ref_v[:, :, :end],
                     pv[:, :, :end], 1e-2, 1e-2)
        pos += 1


# Both Gemma4 Unified layer types: vision-block prefill -> vanilla XQA decode.
# Only the D512 global case needs the CuTe DSL FMHA; the D256 sliding case runs
# FMHA-v2 bidirectional and must exercise Orin too, so gate per-param.
@pytest.mark.parametrize("head,num_q,num_kv,window", [
    pytest.param(512,
                 16,
                 1,
                 -1,
                 id="global_d512_16q1kv",
                 marks=pytest.mark.skipif(
                     _device_sm() not in NATIVE_SMS,
                     reason="D512 CuTe DSL FMHA is unavailable on this SM")),
    pytest.param(256, 16, 8, 1024, id="sliding_d256_16q8kv"),
])
def test_vision_block_prefill_then_decode(head, num_q, num_kv, window):
    _run_vision_prefill_then_decode(head_size=head,
                                    num_q_heads=num_q,
                                    num_kv_heads=num_kv,
                                    window=window)


@pytest.mark.parametrize("case", ["vision_fp8", "tree_vision"])
def test_head512_vision_rejections(case):
    """Vision D512 rejects unsupported precision and mask combinations."""
    PluginRunner()
    creator = trt.get_plugin_registry().get_creator("AttentionPlugin", "1", "")
    assert creator is not None
    enable_fp8 = case == "vision_fp8"
    fields = [
        pf_int32("num_q_heads", 8),
        pf_int32("num_kv_heads", 2),
        pf_int32("head_size", 512),
        pf_int32("enable_tree_attention", int(case == "tree_vision")),
        pf_int32("enable_vision_block_attention", 1),
        pf_int32("enable_fp8_kv_cache", int(enable_fp8)),
        pf_int32("sliding_window_size", -1),
    ]
    if enable_fp8:
        fields.append(pf_float32("qkv_scales", [0.5, 0.25, 0.125]))
    plugin = creator.create_plugin("AttentionPlugin",
                                   trt.PluginFieldCollection(fields),
                                   trt.TensorRTPhase.BUILD)
    assert plugin is None


@pytest.mark.skipif(_device_sm() == 0 or _device_sm() >= 100,
                    reason="requires a non-Blackwell CUDA device")
def test_head512_sliding_vision_accepted_off_blackwell():
    """The FMHA-v2 D512 vision-block kernel serves sliding layers off Blackwell.

    The window is a runtime argument of that kernel, so the same variant covers
    both the sliding and the full-causal Gemma4 global layers.
    """
    PluginRunner()
    creator = trt.get_plugin_registry().get_creator("AttentionPlugin", "1", "")
    assert creator is not None
    plugin = creator.create_plugin(
        "AttentionPlugin",
        trt.PluginFieldCollection([
            pf_int32("num_q_heads", 8),
            pf_int32("num_kv_heads", 2),
            pf_int32("head_size", 512),
            pf_int32("enable_vision_block_attention", 1),
            pf_int32("sliding_window_size", 1024),
        ]), trt.TensorRTPhase.BUILD)
    assert plugin is not None


@pytest.mark.skipif(_device_sm() not in NATIVE_SMS,
                    reason="D512 CuTe DSL FMHA is unavailable on this SM")
@pytest.mark.parametrize("route", ["chunked", "shared"])
def test_head512_vision_prefill_route_rejections(route):
    """Vision blocks are supported only for normal prefill with owned KV."""
    p = AttentionParams(batch_size=1,
                        seq_len=8,
                        num_q_heads=4,
                        num_kv_heads=2,
                        head_size=512,
                        kv_cache_capacity=16,
                        max_batch_size=1,
                        max_seq_len=8,
                        max_position_embeddings=16,
                        is_prefill=True)
    shared = route == "shared"
    runner = AttentionPluginRunner(p,
                                   enable_kv_shared=int(shared),
                                   enable_vision_block_attention=True)
    gen = torch.Generator().manual_seed(626)
    _, _, combined = _make_rope(p, gen)
    _, _, plugin_kv = _empty_caches(p)
    width = p.q_hidden if shared else p.qkv_hidden_size
    qkv = torch.randn((1, p.seq_len, width),
                      generator=gen,
                      dtype=torch.float32).to(DEV)
    context_lengths = torch.full((1, ),
                                 p.seq_len,
                                 dtype=torch.int32,
                                 device=DEV)
    cache_idx = torch.zeros(1, dtype=torch.int32, device=DEV)
    input_shapes = {"kv_cache_indices": (0, )}
    if route == "chunked":
        cache_idx.fill_(1)
        context_lengths.fill_(p.seq_len + 1)
        input_shapes = None
    vision_block_ids = torch.full((1, p.seq_len),
                                  -1,
                                  dtype=torch.int32,
                                  device=DEV)
    with pytest.raises(RuntimeError, match="execute_async_v3 returned False"):
        runner.run(qkv.to(torch.float16),
                   plugin_kv,
                   context_lengths,
                   combined,
                   cache_idx,
                   vision_block_ids=vision_block_ids,
                   input_shapes=input_shapes)


@pytest.mark.skipif(not _fp8_available(),
                    reason="FP8 KV cache not supported on this device")
def test_head512_shared_fp8_prefill_rejected():
    cfg = dict(BASE)
    cfg.update(head_size=512,
               num_q_heads=4,
               num_kv_heads=2,
               enable_fp8_kv_cache=True,
               qkv_scales=[0.5, 0.25, 0.125])
    p = AttentionParams(batch_size=2, seq_len=8, is_prefill=True, **cfg)
    runner = AttentionPluginRunner(p, enable_kv_shared=1)
    gen = torch.Generator().manual_seed(5152)
    _, _, combined = _make_rope(p, gen)
    _, _, plugin_kv = _empty_caches(p)
    q = torch.randn((p.batch_size, p.seq_len, p.q_hidden),
                    generator=gen,
                    dtype=torch.float32).to(DEV)
    ctx_len = torch.full((p.batch_size, ),
                         p.seq_len,
                         dtype=torch.int32,
                         device=DEV)
    cache_idx = torch.zeros(p.batch_size, dtype=torch.int32, device=DEV)
    with pytest.raises(RuntimeError, match="execute_async_v3 returned False"):
        runner.run(q.to(torch.float16),
                   plugin_kv,
                   ctx_len,
                   combined,
                   cache_idx,
                   input_shapes={"kv_cache_indices": (0, )})


# --------------------------------------------------------------------------- #
# FP8 KV cache (+ non-unit qkv scales)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(_device_sm() == 0 or _device_sm() in FP8_KV_CACHE_SMS,
                    reason="requires a CUDA SM without FP8 KV-cache support")
def test_fp8_kv_cache_rejected_on_unsupported_sm():
    PluginRunner()
    creator = trt.get_plugin_registry().get_creator("AttentionPlugin", "1", "")
    assert creator is not None
    plugin = creator.create_plugin(
        "AttentionPlugin",
        trt.PluginFieldCollection([
            pf_int32("num_q_heads", 8),
            pf_int32("num_kv_heads", 2),
            pf_int32("head_size", 128),
            pf_int32("enable_fp8_kv_cache", 1),
            pf_float32("qkv_scales", [0.5, 0.25, 0.125]),
        ]), trt.TensorRTPhase.BUILD)
    assert plugin is None


@pytest.mark.skipif(not _fp8_available(),
                    reason="FP8 XQA decode not supported on this device")
@pytest.mark.parametrize("scales", [[1.0, 1.0, 1.0], [0.5, 0.25, 0.125]],
                         ids=["unit-scales", "distinct-scales"])
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
# FP8 prefill across both supported contracts. Optimized SM100/101/110 kernels
# consume paged FP8 directly. FMHA-v2 on SM89/120/121 preserves its legacy
# dense FP16 fallback: round 1 exercises fresh dense scratch, and round 2
# advances the cache offset to exercise paged gather/dequantization.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _fp8_available(),
                    reason="FP8 CuTe DSL FMHA prefill not supported here")
@pytest.mark.parametrize("head_size", [128, 256], ids=["head128", "head256"])
@pytest.mark.parametrize("scales", [[1.0, 1.0, 1.0], [0.5, 0.25, 0.125]],
                         ids=["unit-scales", "distinct-scales"])
def test_fp8_kv_cache_prefill(scales, head_size):
    cfg = dict(BASE)
    cfg["head_size"] = head_size
    if head_size == 256:
        # Qwen3.5 shape: 16 Q / 8 KV heads (XQA head-256 needs GQA ratio >= 2).
        cfg["num_q_heads"], cfg["num_kv_heads"] = 16, 8
    p = AttentionParams(batch_size=2,
                        seq_len=8,
                        is_prefill=True,
                        enable_fp8_kv_cache=True,
                        qkv_scales=scales,
                        **cfg)
    # FP8 e4m3 storage precision floor, not an error.
    _run_rounds(p, num_rounds=2, atol=2e-1, rtol=2e-1, cos_threshold=0.999)


# FP8 D512 exercises the CuTe DSL D512 FP8 FMHA; round 2 reads back round 1's
# FP8 KV.
@pytest.mark.skipif(_device_sm() not in CUTEDSL_D512_FP8_SMS,
                    reason="D512 FP8 FMHA only on SM100/101/110")
@pytest.mark.parametrize("scales", [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]],
                         ids=["scale1.0", "scale0.5"])
def test_fp8_kv_cache_prefill_head512(scales):
    cfg = dict(BASE)
    cfg["head_size"] = 512
    cfg["num_q_heads"] = 4
    cfg["num_kv_heads"] = 2
    p = AttentionParams(batch_size=2,
                        seq_len=8,
                        is_prefill=True,
                        enable_fp8_kv_cache=True,
                        qkv_scales=scales,
                        **cfg)
    # FP8 e4m3 storage precision floor, not an error (see test_fp8_kv_cache_decode).
    _run_rounds(p, num_rounds=2, atol=2e-1, rtol=2e-1, cos_threshold=0.999)


# --------------------------------------------------------------------------- #
# Normal prefill (kNORMAL_PREFILL): an EMPTY kv_cache_indices binding (shape
# [0]) dispatches the fresh-prefill path -- attention reads the input K/V
# directly (s_kv = S) while RoPE still writes the cache at offset 0.
# --------------------------------------------------------------------------- #
def _empty_cache_indices(p: AttentionParams):
    """kv_cache_indices binding for normal prefill: (dummy, shape overrides).

    A 0-element tensor has data_ptr()==0 which TRT rejects, so bind a 1-element
    dummy and override the runtime shape to [0] to dispatch kNORMAL_PREFILL.
    """
    dummy = torch.zeros((1, ), dtype=torch.int32, device=DEV)
    shapes = {"kv_cache_indices": (0, )}
    return dummy, shapes


def _run_normal_prefill_then_decode(p: AttentionParams,
                                    decode_rounds: int,
                                    seed: int = 42,
                                    shuffle_pages: bool = False,
                                    **tol):
    """Normal-prefill round (empty indices) + regular decode continuation.

    The decode rounds prove the KV cache written by the normal prefill is
    consumable by the non-empty-indices path. ``tol`` is forwarded to
    assert_close (empty = default thresholds)."""
    gen = torch.Generator().manual_seed(seed)
    runner = AttentionPluginRunner(p, shuffle_pages=shuffle_pages)
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)
    b, s = p.batch_size, p.seq_len

    # Normal prefill; reference = first chunked round at cache offset 0.
    qkv = torch.randn((b, s, p.qkv_hidden_size),
                      generator=gen,
                      dtype=torch.float32).to(DEV)
    pos_ids = torch.arange(s, dtype=torch.int32, device=DEV)[None].repeat(b, 1)
    ctx_len = torch.full((b, ), s, dtype=torch.int32, device=DEV)
    mask = sliding_window_mask(s, s, -1, DEV)
    zero_idx = torch.zeros(b, dtype=torch.int32, device=DEV)
    ref_out, ref_k, ref_v = compute_attention(qkv.float(), ref_k, ref_v, cos,
                                              sin, pos_ids, zero_idx, p, mask)
    idx_dummy, idx_shapes = _empty_cache_indices(p)
    attn_out, plugin_kv = runner.run(qkv.to(torch.float16),
                                     plugin_kv,
                                     ctx_len,
                                     combined,
                                     idx_dummy,
                                     input_shapes=idx_shapes)
    pk, pv = _plugin_kv_to_ref(plugin_kv, p)
    assert_close("normal-prefill", ref_out, attn_out, **tol)
    assert_close("normal-prefill-k_cache", ref_k[:, :, :s], pk[:, :, :s],
                 **tol)
    assert_close("normal-prefill-v_cache", ref_v[:, :, :s], pv[:, :, :s],
                 **tol)

    # Regular decode continuation (non-empty indices) off that cache.
    p_dec = replace(p, seq_len=1, is_prefill=False)
    pos = s
    for r in range(decode_rounds):
        qkv_d = torch.randn((b, 1, p.qkv_hidden_size),
                            generator=gen,
                            dtype=torch.float32).to(DEV)
        pos_ids = torch.full((b, 1), pos, dtype=torch.int32, device=DEV)
        cache_idx = torch.full((b, ), pos, dtype=torch.int32, device=DEV)
        ctx_len = torch.full((b, ), pos + 1, dtype=torch.int32, device=DEV)
        ref_out, ref_k, ref_v = compute_attention(qkv_d.float(), ref_k, ref_v,
                                                  cos, sin, pos_ids, cache_idx,
                                                  p_dec, None)
        attn_out, plugin_kv = runner.run(qkv_d.to(torch.float16), plugin_kv,
                                         ctx_len, combined, cache_idx)
        pk, pv = _plugin_kv_to_ref(plugin_kv, p)
        end = pos + 1
        assert_close(f"decode[r{r}]", ref_out, attn_out, **tol)
        assert_close(f"decode-k_cache[r{r}]", ref_k[:, :, :end],
                     pk[:, :, :end], **tol)
        assert_close(f"decode-v_cache[r{r}]", ref_v[:, :, :end],
                     pv[:, :, :end], **tol)
        pos += 1


def test_normal_prefill():
    p = AttentionParams(batch_size=2, seq_len=8, is_prefill=True, **BASE)
    _run_normal_prefill_then_decode(p, decode_rounds=1)


# FP8 KV cache + normal prefill: the fresh-prefill kernel never READS the FP8
# cache -- it attends the FP16 input K/V (the cache is only RoPE-written, read
# back by XQA decode). Works on every FP8-capable SKU (sm89+).
@pytest.mark.skipif(not _fp8_available(),
                    reason="FP8 KV cache not supported on this device")
def test_fp8_kv_cache_normal_prefill():
    p = AttentionParams(batch_size=2,
                        seq_len=8,
                        is_prefill=True,
                        enable_fp8_kv_cache=True,
                        qkv_scales=[1.0, 1.0, 1.0],
                        **BASE)
    # FP8 e4m3 storage precision floor, not an error.
    _run_normal_prefill_then_decode(p,
                                    decode_rounds=2,
                                    atol=2e-1,
                                    rtol=2e-1,
                                    cos_threshold=0.999)


# --------------------------------------------------------------------------- #
# Graceful failure: a config the kernels cannot serve must fail the call
# (RuntimeError on a failed enqueue) with the process intact. Runs on the SKUs
# the support gates above skip.
# --------------------------------------------------------------------------- #
def _run_once(p: AttentionParams, runner, shared_kv=False):
    """One well-formed enqueue against a pre-built runner (no accuracy check)."""
    gen = torch.Generator().manual_seed(7)
    cos, sin, combined = _make_rope(p, gen)
    _, _, plugin_kv = _empty_caches(p)
    b, s = p.batch_size, p.seq_len
    ctx_len = torch.full((b, ), s, dtype=torch.int32, device=DEV)
    cache_idx = torch.zeros(b, dtype=torch.int32, device=DEV)
    width = p.q_hidden if shared_kv else p.qkv_hidden_size
    qkv = torch.randn((b, s, width), generator=gen,
                      dtype=torch.float32).to(DEV).to(torch.float16)
    return runner.run(qkv, plugin_kv, ctx_len, combined, cache_idx)


def _assert_graceful_failure(p, runner_kwargs=None, run_kwargs=None):
    """The config must fail cleanly (no process crash): either the engine
    build rejects it (PluginUnsupportedError) or it builds and the enqueue
    fails (execute_async_v3 returns False). Both are acceptable."""
    try:
        runner = AttentionPluginRunner(p,
                                       expect_unsupported=True,
                                       **(runner_kwargs or {}))
    except PluginUnsupportedError:
        return  # build-time rejection is a clean failure
    with pytest.raises(RuntimeError, match="execute_async_v3 returned False"):
        _run_once(p, runner, **(run_kwargs or {}))


@pytest.mark.skipif(not _fp8_supported() or _device_sm() >= 89,
                    reason="needs a device without FP8 XQA cubins (sm<89)")
def test_fp8_kv_cache_decode_fails_without_fp8_xqa():
    # Below sm89 there are no FP8 XQA cubins: the fp8 engine either builds
    # decode-less and the decode call fails, or the build is rejected
    # outright (sm80 has no fp8 at all) -- both are graceful, not a crash.
    p = AttentionParams(batch_size=1,
                        seq_len=1,
                        enable_fp8_kv_cache=True,
                        qkv_scales=[1.0, 1.0, 1.0],
                        **BASE)
    _assert_graceful_failure(p)


def test_decode_head256_no_gqa_fails_gracefully():
    # XQA has no head-256 ratio-1 kernel on any SM, so the engine builds
    # prefill-only and the decode dispatch throws inside the plugin; the
    # noexcept enqueue boundary must turn that into a failed call.
    cfg = dict(BASE)
    cfg["head_size"] = 256
    p = AttentionParams(batch_size=1, seq_len=1, **cfg)
    _assert_graceful_failure(p)


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


# Custom attention scale on a D512 CuTe DSL prefill path (ratio 4, Gemma4 E4B).
@pytest.mark.skipif(_device_sm() not in NATIVE_SMS,
                    reason="D512 CuTe DSL FMHA is unavailable on this SM")
def test_configurable_softmax_scale_head512():
    desired_scale = 0.37
    cfg = dict(BASE)
    cfg["head_size"] = 512
    cfg["num_q_heads"] = 8
    cfg["num_kv_heads"] = 2
    p = AttentionParams(batch_size=2,
                        seq_len=8,
                        is_prefill=True,
                        qk_scale=desired_scale,
                        **cfg)
    _run_rounds(p,
                num_rounds=2,
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
# Paged KV page table: a random non-contiguous permutation (vs the identity
# table every other test uses) proves the kernel follows the table -- the
# KV-cache readback, gathered through it, fails otherwise.
# --------------------------------------------------------------------------- #
def test_paged_kv_shuffled_page_table():
    cfg = dict(BASE)
    cfg["kv_cache_capacity"] = 256  # 2 pages/slot (PAGE_SIZE=128)
    cfg["max_seq_len"] = 256
    cfg["max_position_embeddings"] = 256
    p = AttentionParams(batch_size=2, seq_len=200, is_prefill=True, **cfg)
    _run_rounds(p, num_rounds=1, atol=1e-2, rtol=1e-2, shuffle_pages=True)


# Same shuffled table across a two-page prefill AND the decode: XQA decode reads
# through its own pageList path, so a table-ignoring decode reads wrong pages.
def test_paged_kv_shuffled_page_table_decode():
    cfg = dict(BASE)
    cfg["kv_cache_capacity"] = 256
    cfg["max_seq_len"] = 256
    cfg["max_position_embeddings"] = 256
    p = AttentionParams(batch_size=2, seq_len=200, is_prefill=True, **cfg)
    _run_normal_prefill_then_decode(p,
                                    decode_rounds=3,
                                    shuffle_pages=True,
                                    atol=1e-2,
                                    rtol=1e-2)


# --------------------------------------------------------------------------- #
# Ragged context lengths across the batch (decode with differing histories)
# --------------------------------------------------------------------------- #
def test_ragged_context_lengths_decode():
    # Ragged decode: rows carry DIFFERENT histories, decode in ONE step with
    # per-row cache_idx/context_lengths. If XQA ignored context_lengths, row 0
    # (ctx=1) would read parked garbage past its history and diverge.
    p = AttentionParams(batch_size=4, seq_len=1, **BASE)
    Hq, Hkv, d = p.num_q_heads, p.num_kv_heads, p.head_size
    gen = torch.Generator().manual_seed(7)
    runner = AttentionPluginRunner(p)
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)
    histories = [0, 3, 7, 15]
    park = max(histories) + 1  # scratch slot for full rows during warmup
    b = p.batch_size

    def roped_kv(qkv, pos_ids):
        k = qkv[:, :, p.q_hidden:p.q_hidden + p.kv_hidden]
        v = qkv[:, :, p.q_hidden + p.kv_hidden:]
        k = apply_rotary_embedding(
            k.reshape(b, 1, Hkv, d).transpose(1, 2), cos, sin, pos_ids)
        return k, v.reshape(b, 1, Hkv, d).transpose(1, 2)

    # Warm row bi to exactly `histories[bi]` tokens; full rows park writes at
    # `park` (never attended) so their valid prefix stays intact.
    for s in range(max(histories)):
        qkv = torch.randn((b, 1, p.qkv_hidden_size),
                          generator=gen,
                          dtype=torch.float32).to(DEV)
        cache_idx = torch.tensor([s if s < h else park for h in histories],
                                 dtype=torch.int32,
                                 device=DEV)
        pos_ids = cache_idx[:, None].to(torch.int32)
        _, plugin_kv = runner.run(qkv.to(torch.float16), plugin_kv,
                                  cache_idx + 1, combined, cache_idx)
        k, v = roped_kv(qkv.float(), pos_ids)
        for bi in range(b):
            idx = int(cache_idx[bi])
            ref_k[bi, :, idx], ref_v[bi, :, idx] = k[bi, :, 0], v[bi, :, 0]

    # One ragged decode step.
    qkv = torch.randn((b, 1, p.qkv_hidden_size),
                      generator=gen,
                      dtype=torch.float32).to(DEV)
    cache_idx = torch.tensor(histories, dtype=torch.int32, device=DEV)
    pos_ids = cache_idx[:, None].to(torch.int32)
    q = apply_rotary_embedding(
        qkv[:, :, :p.q_hidden].reshape(b, 1, Hq, d).transpose(1, 2), cos, sin,
        pos_ids)
    k, v = roped_kv(qkv.float(), pos_ids)
    attn_out, plugin_kv = runner.run(qkv.to(torch.float16), plugin_kv,
                                     cache_idx + 1, combined, cache_idx)
    ref_out = torch.empty((b, 1, Hq * d), dtype=torch.float32, device=DEV)
    for bi in range(b):
        idx = histories[bi]
        ref_k[bi, :, idx], ref_v[bi, :, idx] = k[bi, :, 0], v[bi, :, 0]
        L = idx + 1
        o = scaled_dot_product_attention(q[bi:bi + 1], ref_k[bi:bi + 1, :, :L],
                                         ref_v[bi:bi + 1, :, :L], p.qk_scale,
                                         None, Hq, Hkv)
        ref_out[bi] = o.transpose(1, 2).reshape(1, Hq * d)
    assert_close("ragged-decode-out", ref_out, attn_out, 1e-2, 1e-2)
    pk, pv = _plugin_kv_to_ref(plugin_kv, p)
    for bi in range(b):
        L = histories[bi] + 1
        assert_close(f"ragged-decode-k[b{bi}]", ref_k[bi, :, :L],
                     pk[bi, :, :L], 1e-2, 1e-2)
        assert_close(f"ragged-decode-v[b{bi}]", ref_v[bi, :, :L],
                     pv[bi, :, :L], 1e-2, 1e-2)


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
    qkv16 = qkv.to(torch.float16)
    cache_idx = torch.zeros((p.batch_size, ), dtype=torch.int32, device=DEV)
    ctx_len = torch.ones((p.batch_size, ), dtype=torch.int32, device=DEV)

    runner = AttentionPluginRunner(p)
    _, _, kv0 = _empty_caches(p)
    out0, _ = runner.run(qkv16.clone(), kv0, ctx_len, combined, cache_idx)

    perm = torch.tensor([2, 0, 3, 1], device=DEV)
    _, _, kv1 = _empty_caches(p)
    out1, _ = runner.run(qkv16[perm].contiguous(), kv1, ctx_len, combined,
                         cache_idx)
    assert_close("batch-perm", out0[perm], out1, 1e-3, 1e-3)


# --------------------------------------------------------------------------- #
# Tree (speculative) attention
# --------------------------------------------------------------------------- #
def _tree_attention_rounds(q_norm_gamma=None, k_norm_gamma=None):
    p = AttentionParams(batch_size=4, seq_len=4, **BASE)
    num_rounds = 5
    gen = torch.Generator().manual_seed(42)
    runner = AttentionPluginRunner(p,
                                   enable_tree_attention=True,
                                   q_norm_gamma=q_norm_gamma,
                                   k_norm_gamma=k_norm_gamma)
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
            qkv.float(),
            ref_k,
            ref_v,
            cos,
            sin,
            pos_ids,
            cache_idx,
            p,
            full_mask,
            q_norm_gamma=q_norm_gamma,
            k_norm_gamma=k_norm_gamma,
            rms_norm_eps=p.rms_norm_eps)

        attn_out, plugin_kv = runner.run(qkv.to(torch.float16), plugin_kv,
                                         ctx_len, combined, cache_idx, packed,
                                         pos_ids)
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


def test_tree_attention():
    _tree_attention_rounds()


def test_tree_attention_qknorm():
    # EAGLE3 cross path: fused qk_norm combined with tree attention.
    gen = torch.Generator().manual_seed(1012)
    qg, kg = _make_qk_norm_gammas(BASE["head_size"], gen)
    _tree_attention_rounds(q_norm_gamma=qg, k_norm_gamma=kg)


@pytest.mark.skipif(_device_sm() not in NATIVE_SMS,
                    reason="D512 CuTe DSL FMHA is unavailable on this SM")
@pytest.mark.parametrize("sliding_window", [-1, 4], ids=["causal", "sliding"])
def test_tree_attention_head512_prefill_then_decode(sliding_window):
    cfg = dict(BASE)
    cfg.update(head_size=512,
               num_q_heads=4,
               num_kv_heads=2,
               sliding_window_size=sliding_window)
    p = AttentionParams(batch_size=2, seq_len=8, is_prefill=True, **cfg)
    runner = AttentionPluginRunner(p, enable_tree_attention=True)
    gen = torch.Generator().manual_seed(5120 + max(sliding_window, 0))
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)
    placeholder_mask = torch.ones((p.batch_size, 1, 1),
                                  dtype=torch.int32,
                                  device=DEV)
    placeholder_pos = torch.zeros((p.batch_size, 1),
                                  dtype=torch.int32,
                                  device=DEV)

    # Tree-enabled engines use native-paged FMHA for normal and chunked
    # prefill. A length-one position-id input is the runtime mode convention.
    for round_idx, pos in enumerate((0, p.seq_len)):
        qkv = torch.randn((p.batch_size, p.seq_len, p.qkv_hidden_size),
                          generator=gen,
                          dtype=torch.float32).to(DEV)
        pos_ids = torch.arange(pos,
                               pos + p.seq_len,
                               dtype=torch.int32,
                               device=DEV)[None].repeat(p.batch_size, 1)
        cache_idx = torch.full((p.batch_size, ),
                               pos,
                               dtype=torch.int32,
                               device=DEV)
        ctx_len = torch.full((p.batch_size, ),
                             pos + p.seq_len,
                             dtype=torch.int32,
                             device=DEV)
        mask = sliding_window_mask(p.seq_len, pos + p.seq_len, sliding_window,
                                   DEV)
        ref_out, ref_k, ref_v = compute_attention(qkv.float(), ref_k, ref_v,
                                                  cos, sin, pos_ids, cache_idx,
                                                  p, mask)
        input_shapes = {"kv_cache_indices": (0, )} if round_idx == 0 else None
        attn_out, plugin_kv = runner.run(qkv.to(torch.float16),
                                         plugin_kv,
                                         ctx_len,
                                         combined,
                                         cache_idx,
                                         placeholder_mask,
                                         placeholder_pos,
                                         input_shapes=input_shapes)
        assert_close(f"tree-d512-prefill[r{round_idx}]",
                     ref_out,
                     attn_out,
                     atol=8e-2,
                     rtol=2e-2,
                     cos_threshold=0.9999)

    # A full-width position-id input selects tree decode, which must use XQA.
    width = 4
    p_tree = AttentionParams(batch_size=p.batch_size,
                             seq_len=width,
                             is_prefill=False,
                             **cfg)
    tree_mask, _ = get_tree_attention_mask(width)
    packed = pack_tree_mask(tree_mask, width, p.batch_size).to(DEV)
    tree_mask = tree_mask.to(DEV)
    pos = 2 * p.seq_len
    depth = torch.tensor([0, 1, 1, 2], dtype=torch.int32, device=DEV)
    pos_ids = (pos + depth)[None].repeat(p.batch_size, 1)
    cache_idx = torch.full((p.batch_size, ),
                           pos,
                           dtype=torch.int32,
                           device=DEV)
    ctx_len = torch.full((p.batch_size, ),
                         pos + width,
                         dtype=torch.int32,
                         device=DEV)
    full_mask = torch.ones((width, pos + width), dtype=torch.int32, device=DEV)
    full_mask[:, pos:] = tree_mask
    if sliding_window > 0:
        full_mask[:, :max(0, pos + width - sliding_window)] = 0
    qkv = torch.randn((p.batch_size, width, p.qkv_hidden_size),
                      generator=gen,
                      dtype=torch.float32).to(DEV)
    ref_out, _, _ = compute_attention(qkv.float(), ref_k, ref_v, cos, sin,
                                      pos_ids, cache_idx, p_tree, full_mask)
    attn_out, _ = runner.run(qkv.to(torch.float16), plugin_kv, ctx_len,
                             combined, cache_idx, packed, pos_ids)
    assert_close("tree-d512-decode",
                 ref_out,
                 attn_out,
                 atol=8e-2,
                 rtol=2e-2,
                 cos_threshold=0.9999)


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
    pos_ids = depth[None].repeat(p.batch_size, 1).to(DEV)
    cache_idx = torch.zeros(p.batch_size, dtype=torch.int32, device=DEV)
    ctx_len = torch.full((p.batch_size, ),
                         width,
                         dtype=torch.int32,
                         device=DEV)

    ref_out, _, _ = compute_attention(qkv.float(), ref_k, ref_v, cos, sin,
                                      pos_ids, cache_idx, p, tree_mask)
    attn_out, _ = runner.run(qkv.to(torch.float16), plugin_kv, ctx_len,
                             combined, cache_idx, packed, pos_ids)
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
# head 32 decode: degraded XQA-only mode (no head-32 prefill FMHA kernel; XQA
# supports ratios 1-8). The constructor advertises "naive attention for
# prefill" but enqueue() has no such path, so head-32 prefill fails at enqueue.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_kv_heads", [8, 1], ids=lambda k: f"kv{k}")
def test_decode_head32(num_kv_heads):
    cfg = dict(BASE)
    cfg["head_size"] = 32
    cfg["num_kv_heads"] = num_kv_heads
    p = AttentionParams(batch_size=2, seq_len=1, **cfg)
    _run_rounds(p, num_rounds=4, atol=1e-2, rtol=1e-2)


# --------------------------------------------------------------------------- #
# Fused qk_norm (per-head FP32 RMSNorm on Q/K before RoPE, e.g. Qwen3):
# enable_qk_norm=1 wires the [head_size] gamma constants as optional plugin inputs.
# --------------------------------------------------------------------------- #
def _make_qk_norm_gammas(head_size: int, gen):
    """Random per-head Q/K gammas in [0.5, 1.5]."""
    qg = torch.empty(head_size).uniform_(0.5, 1.5, generator=gen).to(DEV)
    kg = torch.empty(head_size).uniform_(0.5, 1.5, generator=gen).to(DEV)
    return qg, kg


def test_prefill_qknorm():
    p = AttentionParams(batch_size=2, seq_len=8, is_prefill=True, **BASE)
    gen = torch.Generator().manual_seed(1010)
    qg, kg = _make_qk_norm_gammas(p.head_size, gen)
    _run_rounds(p,
                num_rounds=3,
                atol=1e-2,
                rtol=1e-2,
                q_norm_gamma=qg,
                k_norm_gamma=kg)


def test_decode_qknorm():
    p = AttentionParams(batch_size=2, seq_len=1, **BASE)
    gen = torch.Generator().manual_seed(1011)
    qg, kg = _make_qk_norm_gammas(p.head_size, gen)
    _run_rounds(p,
                num_rounds=4,
                atol=1e-2,
                rtol=1e-2,
                q_norm_gamma=qg,
                k_norm_gamma=kg)


def test_decode_qknorm_odd_tokens():
    """bs=1 decode: 1 token/step — an ODD total token count.

    At head_size=128 the packed RoPE kernel packs two token rows per warp,
    so an odd token count leaves a tail row that must stay alive through the
    fused-norm warp collectives without storing anything.
    """
    p = AttentionParams(batch_size=1, seq_len=1, **BASE)
    gen = torch.Generator().manual_seed(1012)
    qg, kg = _make_qk_norm_gammas(p.head_size, gen)
    _run_rounds(p,
                num_rounds=5,
                atol=1e-2,
                rtol=1e-2,
                q_norm_gamma=qg,
                k_norm_gamma=kg)


def test_prefill_qknorm_odd_tokens():
    """bs=1 prefill with an odd sequence length (7 tokens) — same warp-tail
    coverage as test_decode_qknorm_odd_tokens but through the prefill path."""
    p = AttentionParams(batch_size=1, seq_len=7, is_prefill=True, **BASE)
    gen = torch.Generator().manual_seed(1013)
    qg, kg = _make_qk_norm_gammas(p.head_size, gen)
    _run_rounds(p,
                num_rounds=2,
                atol=1e-2,
                rtol=1e-2,
                q_norm_gamma=qg,
                k_norm_gamma=kg)


@pytest.mark.skipif(not _fp8_available(),
                    reason="FP8 XQA decode not supported on this device")
def test_decode_qknorm_fp8kv():
    """Fused qk_norm combined with the FP8 KV cache path (norm + RoPE + FP8
    quantized K/V cache write in one kernel). Relaxed threshold matches
    test_fp8_kv_cache_decode: the FP8 e4m3 storage precision floor."""
    p = AttentionParams(batch_size=2,
                        seq_len=1,
                        enable_fp8_kv_cache=True,
                        qkv_scales=[1.0, 1.0, 1.0],
                        **BASE)
    gen = torch.Generator().manual_seed(1014)
    qg, kg = _make_qk_norm_gammas(p.head_size, gen)
    _run_rounds(p,
                num_rounds=4,
                atol=2e-1,
                rtol=2e-1,
                cos_threshold=0.999,
                q_norm_gamma=qg,
                k_norm_gamma=kg)


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
    pos_ids = torch.arange(prefill_len, dtype=torch.int32,
                           device=DEV)[None].repeat(bs, 1)
    cache_idx = torch.zeros(bs, dtype=torch.int32, device=DEV)
    ctx_len = torch.full((bs, ), prefill_len, dtype=torch.int32, device=DEV)
    mask = sliding_window_mask(prefill_len, prefill_len, -1, DEV)
    ref_out, ref_k, ref_v = compute_attention(qkv.float(), ref_k, ref_v, cos,
                                              sin, pos_ids, cache_idx, p, mask)
    attn_out, plugin_kv = runner.run(qkv.to(torch.float16), plugin_kv, ctx_len,
                                     combined, cache_idx)
    assert_close("handoff-prefill", ref_out, attn_out)

    # --- decode steps, continuing from the prefilled KV cache ---
    pos = prefill_len
    for i in range(n_decode):
        qkv_d = torch.randn((bs, 1, p.qkv_hidden_size),
                            generator=gen,
                            dtype=torch.float32).to(DEV)
        pos_ids = torch.full((bs, 1), pos, dtype=torch.int32, device=DEV)
        cache_idx = torch.full((bs, ), pos, dtype=torch.int32, device=DEV)
        ctx_len = torch.full((bs, ), pos + 1, dtype=torch.int32, device=DEV)
        ref_out, ref_k, ref_v = compute_attention(qkv_d.float(), ref_k, ref_v,
                                                  cos, sin, pos_ids, cache_idx,
                                                  p_dec, None)
        attn_out, plugin_kv = runner.run(qkv_d.to(torch.float16), plugin_kv,
                                         ctx_len, combined, cache_idx)
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


# D512 normal-prefill ragged coverage. On SM100/101/110 this uses optimized
# native-paged common FMHA; elsewhere it exercises native-paged FMHA-v2. The
# physical S=257 tensor crosses both 128-token CTA boundaries, while poisoned
# padding and valid-row checks catch reads outside each request's logical prefix.
@pytest.mark.skipif(_device_sm() not in NATIVE_SMS,
                    reason="D512 CuTe DSL FMHA is unavailable on this SM")
def test_ragged_prefill_head512_cutedsl():
    seqlens = [1, 128, 257]
    cfg = dict(BASE)
    cfg["num_q_heads"] = 4
    cfg["num_kv_heads"] = 2
    cfg["head_size"] = 512
    cfg["kv_cache_capacity"] = 257
    cfg["max_batch_size"] = len(seqlens)
    cfg["max_seq_len"] = 257
    cfg["max_position_embeddings"] = 257
    p = AttentionParams(batch_size=len(seqlens),
                        seq_len=257,
                        is_prefill=True,
                        **cfg)
    gen = torch.Generator().manual_seed(512257)
    runner = AttentionPluginRunner(p)
    cos, sin, combined = _make_rope(p, gen)
    _, _, plugin_kv = _empty_caches(p)
    qkv = _ragged_qkv(seqlens, p, gen)
    ctx_len = torch.tensor(seqlens, dtype=torch.int32, device=DEV)
    cache_idx = torch.zeros(len(seqlens), dtype=torch.int32, device=DEV)
    attn_out, _ = runner.run(qkv.to(torch.float16),
                             plugin_kv,
                             ctx_len,
                             combined,
                             cache_idx,
                             input_shapes={"kv_cache_indices": (0, )})
    ref_rows = _ragged_prefill_ref(qkv.float(), cos, sin, seqlens, p)
    # D512's two-CTA FP16 accumulation order has a wider elementwise envelope
    # at S=257; retain a tight cosine gate to catch structural masking errors.
    for b, length in enumerate(seqlens):
        assert_close(f"d512-ragged.b{b}",
                     ref_rows[b],
                     attn_out[b, :length],
                     atol=8e-2,
                     rtol=2e-2,
                     cos_threshold=0.9999)


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
    ctx_len = torch.tensor(seqlens, dtype=torch.int32, device=DEV)
    cache_idx = torch.zeros(len(seqlens), dtype=torch.int32, device=DEV)
    attn_out, _ = runner.run(qkv.to(torch.float16),
                             plugin_kv,
                             ctx_len,
                             combined,
                             cache_idx,
                             input_shapes={"kv_cache_indices": (0, )})
    ref_rows = _ragged_prefill_ref(qkv.float(), cos, sin, seqlens, p)
    for b, L in enumerate(seqlens):
        assert_close(f"ragged[{label}].b{b}", ref_rows[b], attn_out[b, :L])


# DiffusionGemma denoise uses a non-empty context-mask selector to switch from
# bottom-right causal attention to dense PADDING attention. Cache starts
# [20, 8] plus valid Q lengths [4, 3] produce the exact logical KV lengths
# [24, 11] while the physical Q tensor remains batch-strided at S=4.
@pytest.mark.skipif(_device_sm()
                    not in (80, 86, 87, 89, 90, 100, 101, 110, 120, 121),
                    reason="D256 CuTe DSL PADDING FMHA is unsupported")
def test_diffusion_gemma_padding_prefill():
    q_lens = [4, 3]
    cache_starts = [20, 8]
    kv_lens = [24, 11]
    cfg = dict(BASE)
    cfg["num_q_heads"] = 16
    cfg["num_kv_heads"] = 8
    cfg["head_size"] = 256
    cfg["kv_cache_capacity"] = 32
    cfg["max_batch_size"] = len(q_lens)
    cfg["max_seq_len"] = max(q_lens)
    cfg["max_position_embeddings"] = cfg["kv_cache_capacity"]
    p = AttentionParams(batch_size=len(q_lens),
                        seq_len=max(q_lens),
                        is_prefill=True,
                        **cfg)
    gen = torch.Generator().manual_seed(2562411)
    runner = AttentionPluginRunner(p, enable_context_mask_selector=True)
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)

    # Seed an already-RoPE'd cache prefix, as the runtime does before a denoise
    # chunk. Copy through FP16 so the reference starts from the exact values
    # consumed by the plugin.
    for b, prefix_len in enumerate(cache_starts):
        prefix_k = torch.randn((p.num_kv_heads, prefix_len, p.head_size),
                               generator=gen,
                               dtype=torch.float32).to(DEV)
        prefix_v = torch.randn((p.num_kv_heads, prefix_len, p.head_size),
                               generator=gen,
                               dtype=torch.float32).to(DEV)
        plugin_kv[b, 0, :, :prefix_len] = prefix_k.to(torch.float16)
        plugin_kv[b, 1, :, :prefix_len] = prefix_v.to(torch.float16)
        ref_k[b, :, :prefix_len] = plugin_kv[b, 0, :, :prefix_len].float()
        ref_v[b, :, :prefix_len] = plugin_kv[b, 1, :, :prefix_len].float()

    qkv = torch.randn((p.batch_size, p.seq_len, p.qkv_hidden_size),
                      generator=gen,
                      dtype=torch.float32).to(DEV)
    poison_padding(qkv, q_lens)
    qkv_plugin = qkv.to(torch.float16)
    qkv_ref = qkv_plugin.float()
    ref_rows = []
    for b, (q_len, cache_start,
            kv_len) in enumerate(zip(q_lens, cache_starts, kv_lens)):
        q = qkv_ref[b:b + 1, :q_len, :p.q_hidden].reshape(
            1, q_len, p.num_q_heads, p.head_size).transpose(1, 2)
        k = qkv_ref[b:b + 1, :q_len,
                    p.q_hidden:p.q_hidden + p.kv_hidden].reshape(
                        1, q_len, p.num_kv_heads, p.head_size).transpose(1, 2)
        v = qkv_ref[b:b + 1, :q_len, p.q_hidden + p.kv_hidden:].reshape(
            1, q_len, p.num_kv_heads, p.head_size).transpose(1, 2)
        position_ids = torch.arange(cache_start,
                                    cache_start + q_len,
                                    dtype=torch.int32,
                                    device=DEV)[None]
        q = apply_rotary_embedding(q, cos, sin, position_ids)
        k = apply_rotary_embedding(k, cos, sin, position_ids)
        ref_k[b, :, cache_start:kv_len] = k[0]
        ref_v[b, :, cache_start:kv_len] = v[0]
        ref_out = scaled_dot_product_attention(q, ref_k[b:b + 1, :, :kv_len],
                                               ref_v[b:b + 1, :, :kv_len],
                                               p.qk_scale, None, p.num_q_heads,
                                               p.num_kv_heads)
        ref_rows.append(ref_out.transpose(1, 2).reshape(q_len, p.q_hidden))

    ctx_len = torch.tensor(q_lens, dtype=torch.int32, device=DEV)
    cache_idx = torch.tensor(cache_starts, dtype=torch.int32, device=DEV)
    selector = torch.zeros(p.batch_size, dtype=torch.int32, device=DEV)
    attn_out, plugin_kv = runner.run(qkv_plugin,
                                     plugin_kv,
                                     ctx_len,
                                     combined,
                                     cache_idx,
                                     context_mask_selector=selector)
    plugin_k, plugin_v = _plugin_kv_to_ref(plugin_kv, p)

    for b, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        assert_close(f"diffusion-padding-attn.b{b}", ref_rows[b],
                     attn_out[b, :q_len])
        assert_close(f"diffusion-padding-k-cache.b{b}", ref_k[b, :, :kv_len],
                     plugin_k[b, :, :kv_len])
        assert_close(f"diffusion-padding-v-cache.b{b}", ref_v[b, :, :kv_len],
                     plugin_v[b, :, :kv_len])


# Batch invariance on RAGGED input (plugin-vs-plugin): permuting the rows (and
# their context lengths) must permute the per-row outputs identically.
#
# The AttentionPlugin applies RoPE in place to its Q/K input buffers (benign in
# the real graph where Q/K are fresh each iter, but reusing a Q/K buffer across
# two enqueues rotates it twice). Each enqueue gets its own copy of Q/K/V.
def test_ragged_prefill_batch_invariance():
    seqlens = [10, 2048, 128]
    p = _ragged_attn_params(seqlens)
    gen = torch.Generator().manual_seed(8888)
    runner = AttentionPluginRunner(p)
    cos, sin, combined = _make_rope(p, gen)
    qkv = _ragged_qkv(seqlens, p, gen)
    qkv16 = qkv.to(torch.float16)
    cache_idx = torch.zeros(len(seqlens), dtype=torch.int32, device=DEV)

    _, _, kv0 = _empty_caches(p)
    ctx0 = torch.tensor(seqlens, dtype=torch.int32, device=DEV)
    # clone so the in-place RoPE does not corrupt the buffer reused below
    prefill_shapes = {"kv_cache_indices": (0, )}
    out0, _ = runner.run(qkv16.clone(),
                         kv0,
                         ctx0,
                         combined,
                         cache_idx,
                         input_shapes=prefill_shapes)

    perm = [2, 0, 1]
    sl_p = [seqlens[i] for i in perm]
    _, _, kv1 = _empty_caches(p)
    ctx1 = torch.tensor(sl_p, dtype=torch.int32, device=DEV)
    out1, _ = runner.run(qkv16[perm].contiguous(),
                         kv1,
                         ctx1,
                         combined,
                         cache_idx,
                         input_shapes=prefill_shapes)
    for new_i, orig in enumerate(perm):
        L = seqlens[orig]
        assert_close(f"ragged-batch-inv[{new_i}]", out0[orig, :L],
                     out1[new_i, :L])


# --------------------------------------------------------------------------- #
# Shared KV (Gemma4 KV-sharing layers): a shared engine (enable_kv_shared=1) takes a
# Q-only packed input and reads a donor engine's cache without writing it. Each test
# populates the cache with a donor (enable_kv_shared=0) pass first.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _fp8_available(),
                    reason="FP8 KV cache not supported on this device")
def test_fp8_shared_kv_prefill_rejected():
    """Shared-KV prefill rejects FP8 because the shared layer lacks donor K/V
    scales -- verified on every FP8-capable SKU (incl. SM120, which has FP8
    decode but no shared-KV prefill route)."""
    p = AttentionParams(batch_size=2,
                        seq_len=8,
                        is_prefill=True,
                        enable_fp8_kv_cache=True,
                        qkv_scales=[0.5, 0.25, 0.125],
                        **BASE)
    shared_runner = AttentionPluginRunner(p, enable_kv_shared=1)
    _, _, combined = _make_rope(p, torch.Generator().manual_seed(2300))
    _, _, plugin_kv = _empty_caches(p)
    q = torch.zeros((p.batch_size, p.seq_len, p.q_hidden),
                    dtype=torch.float16,
                    device=DEV)
    context_lengths = torch.full((p.batch_size, ),
                                 p.seq_len,
                                 dtype=torch.int32,
                                 device=DEV)
    cache_indices = torch.zeros(p.batch_size, dtype=torch.int32, device=DEV)

    with pytest.raises(RuntimeError, match="execute_async_v3 returned False"):
        shared_runner.run(q, plugin_kv, context_lengths, combined,
                          cache_indices)


def _assert_cache_untouched(name: str, before: "torch.Tensor",
                            after: "torch.Tensor"):
    """Bit-exact check that a shared-KV call did not write the donor cache."""
    assert torch.equal(before.view(torch.int16), after.view(torch.int16)), \
        f"{name}: shared-KV call must not modify the donor KV cache"


# Shared-KV prefill. head 128 runs the CuTe DSL FMHA path where available
# (SM100+) and FMHA-v2 elsewhere; head 256 uses native-paged FMHA-v2 where
# supported. FP16 head 512 runs native-paged common FMHA on SM100/101/110 and
# native-paged FMHA-v2 on the remaining CuTe DSL targets. Nsight route
# validation separately excludes the gather fallback.
@pytest.mark.parametrize("head_size,num_q_heads,num_kv_heads,sliding_window", [
    pytest.param(128, 8, 4, -1, id="head128_q8_kv4"),
    pytest.param(256, 16, 8, -1, id="head256_q16_kv8"),
    pytest.param(512,
                 4,
                 2,
                 -1,
                 id="head512_q4_kv2",
                 marks=pytest.mark.skipif(
                     _device_sm() not in NATIVE_SMS,
                     reason="D512 CuTe DSL FMHA is unavailable on this SM")),
    pytest.param(512,
                 4,
                 2,
                 4,
                 id="head512_q4_kv2_sliding",
                 marks=pytest.mark.skipif(
                     _device_sm() not in NATIVE_SMS,
                     reason="D512 CuTe DSL FMHA is unavailable on this SM")),
    pytest.param(512,
                 8,
                 1,
                 -1,
                 id="head512_q8_kv1",
                 marks=pytest.mark.skipif(
                     _device_sm() not in NATIVE_SMS,
                     reason="D512 CuTe DSL FMHA is unavailable on this SM")),
    pytest.param(512,
                 8,
                 2,
                 -1,
                 id="head512_q8_kv2",
                 marks=pytest.mark.skipif(
                     _device_sm() not in NATIVE_SMS,
                     reason="D512 CuTe DSL FMHA is unavailable on this SM")),
])
def test_shared_kv_prefill(head_size, num_q_heads, num_kv_heads,
                           sliding_window):
    cfg = dict(BASE)
    cfg["head_size"] = head_size
    cfg["num_q_heads"] = num_q_heads
    cfg["num_kv_heads"] = num_kv_heads
    cfg["sliding_window_size"] = sliding_window
    p = AttentionParams(batch_size=2, seq_len=8, is_prefill=True, **cfg)
    gen = torch.Generator().manual_seed(2400 + head_size)
    runner = AttentionPluginRunner(p)
    shared_runner = AttentionPluginRunner(p, enable_kv_shared=1)
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)
    b, s = p.batch_size, p.seq_len

    pos_ids = torch.arange(s, dtype=torch.int32, device=DEV)[None].repeat(b, 1)
    cache_idx = torch.zeros(b, dtype=torch.int32, device=DEV)
    ctx_len = torch.full((b, ), s, dtype=torch.int32, device=DEV)
    mask = sliding_window_mask(s, s, sliding_window, DEV)

    # Donor pass (own KV) populates the cache: RoPE'd K + raw V.
    qkv = torch.randn((b, s, p.qkv_hidden_size),
                      generator=gen,
                      dtype=torch.float32).to(DEV)
    _, ref_k, ref_v = compute_attention(qkv.float(), ref_k, ref_v, cos, sin,
                                        pos_ids, cache_idx, p, mask)
    prefill_shapes = {"kv_cache_indices": (0, )}
    runner.run(qkv.to(torch.float16),
               plugin_kv,
               ctx_len,
               combined,
               cache_idx,
               input_shapes=prefill_shapes)

    # Shared-KV pass: a Q-only QKV (C = Hq*D) for the same positions, the
    # donor cache as KV-cache input.
    q2 = torch.randn((b, s, p.q_hidden), generator=gen,
                     dtype=torch.float32).to(DEV)
    donor_before = plugin_kv.clone()
    attn_out, plugin_kv = shared_runner.run(q2.to(torch.float16),
                                            plugin_kv,
                                            ctx_len,
                                            combined,
                                            cache_idx,
                                            input_shapes=prefill_shapes)

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


# Shared-KV prefill with a non-default QK^T scale: the donor cache is real
# (written by an own-KV prefill) and the shared layer attends it with the scale.
def test_shared_kv_prefill_scaled():
    scale = 0.37
    p = AttentionParams(batch_size=2,
                        seq_len=8,
                        is_prefill=True,
                        qk_scale=scale,
                        **BASE)
    gen = torch.Generator().manual_seed(2401)
    runner = AttentionPluginRunner(p)
    shared_runner = AttentionPluginRunner(p,
                                          enable_kv_shared=1,
                                          attention_scale=scale)
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)
    b, s = p.batch_size, p.seq_len
    pos_ids = torch.arange(s, dtype=torch.int32, device=DEV)[None].repeat(b, 1)
    cache_idx = torch.zeros(b, dtype=torch.int32, device=DEV)
    ctx_len = torch.full((b, ), s, dtype=torch.int32, device=DEV)
    mask = sliding_window_mask(s, s, -1, DEV)
    prefill_shapes = {"kv_cache_indices": (0, )}

    qkv = torch.randn((b, s, p.qkv_hidden_size),
                      generator=gen,
                      dtype=torch.float32).to(DEV)
    _, ref_k, ref_v = compute_attention(qkv.float(), ref_k, ref_v, cos, sin,
                                        pos_ids, cache_idx, p, mask)
    runner.run(qkv.to(torch.float16),
               plugin_kv,
               ctx_len,
               combined,
               cache_idx,
               input_shapes=prefill_shapes)
    q2 = torch.randn((b, s, p.q_hidden), generator=gen,
                     dtype=torch.float32).to(DEV)
    donor_before = plugin_kv.clone()
    attn_out, plugin_kv = shared_runner.run(q2.to(torch.float16),
                                            plugin_kv,
                                            ctx_len,
                                            combined,
                                            cache_idx,
                                            input_shapes=prefill_shapes)
    q2r = apply_rotary_embedding(
        q2.reshape(b, s, p.num_q_heads, p.head_size).transpose(1, 2), cos, sin,
        pos_ids)
    ref_out = scaled_dot_product_attention(q2r, ref_k[:, :, :s],
                                           ref_v[:, :, :s], scale, mask,
                                           p.num_q_heads, p.num_kv_heads)
    ref_out = ref_out.transpose(1, 2).reshape(b, s, p.q_hidden)
    assert_close("shared-kv-prefill-scaled", ref_out, attn_out)
    _assert_cache_untouched("shared-kv-prefill-scaled", donor_before,
                            plugin_kv)


# Shared-KV decode (XQA, RoPE-Q-only): an own-KV donor decode writes the new
# K/V slot, then a shared-KV decode's Q (roped at ctx-1) attends the donor cache
# including it. The custom scale exercises the shared decode's softmaxScale.
@pytest.mark.parametrize("attention_scale", [None, 0.37],
                         ids=["default_scale", "scale0.37"])
def test_shared_kv_decode(attention_scale):
    cfg = dict(BASE)
    cfg["num_kv_heads"] = 4
    scale_kw = {} if attention_scale is None else {"qk_scale": attention_scale}
    p = AttentionParams(batch_size=2,
                        seq_len=8,
                        is_prefill=True,
                        **scale_kw,
                        **cfg)
    p_dec = AttentionParams(batch_size=2, seq_len=1, **scale_kw, **cfg)
    gen = torch.Generator().manual_seed(2500)
    # Donor engine (own KV) serves prefill + decode writes; a separate
    # enable_kv_shared=1 engine serves the Q-only shared-KV decode reads.
    runner = AttentionPluginRunner(p)
    shared_runner = AttentionPluginRunner(p,
                                          enable_kv_shared=1,
                                          attention_scale=attention_scale)
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)
    b, s = p.batch_size, p.seq_len

    # Donor prefill populates the cache.
    qkv = torch.randn((b, s, p.qkv_hidden_size),
                      generator=gen,
                      dtype=torch.float32).to(DEV)
    pos_ids = torch.arange(s, dtype=torch.int32, device=DEV)[None].repeat(b, 1)
    cache_idx = torch.zeros(b, dtype=torch.int32, device=DEV)
    ctx_len = torch.full((b, ), s, dtype=torch.int32, device=DEV)
    mask = sliding_window_mask(s, s, -1, DEV)
    _, ref_k, ref_v = compute_attention(qkv.float(), ref_k, ref_v, cos, sin,
                                        pos_ids, cache_idx, p, mask)
    runner.run(qkv.to(torch.float16),
               plugin_kv,
               ctx_len,
               combined,
               cache_idx,
               input_shapes={"kv_cache_indices": (0, )})

    pos = s
    for step in range(3):
        pos_ids = torch.full((b, 1), pos, dtype=torch.int32, device=DEV)
        cache_idx = torch.full((b, ), pos, dtype=torch.int32, device=DEV)
        ctx_len = torch.full((b, ), pos + 1, dtype=torch.int32, device=DEV)

        # Donor decode: own-KV step that writes the K/V slot at ``pos``.
        qkv_d = torch.randn((b, 1, p.qkv_hidden_size),
                            generator=gen,
                            dtype=torch.float32).to(DEV)
        _, ref_k, ref_v = compute_attention(qkv_d.float(), ref_k, ref_v, cos,
                                            sin, pos_ids, cache_idx, p_dec,
                                            None)
        runner.run(qkv_d.to(torch.float16), plugin_kv, ctx_len, combined,
                   cache_idx)

        # Shared-KV decode: Q-only QKV (C = Hq*D), no cache write.
        q_s = torch.randn((b, 1, p.q_hidden),
                          generator=gen,
                          dtype=torch.float32).to(DEV)
        donor_before = plugin_kv.clone()
        attn_out, plugin_kv = shared_runner.run(q_s.to(torch.float16),
                                                plugin_kv, ctx_len, combined,
                                                cache_idx)

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
# cache prefix, driving the chunked shared-KV kernel variants. D512 uses
# native-paged common FMHA on SM100/101/110 and native-paged FMHA-v2 elsewhere.
# Nsight route validation separately proves that shared chunked prefill does
# not use the gather fallback.
@pytest.mark.parametrize("head_size,num_q_heads,num_kv_heads,sliding_window", [
    pytest.param(128, 8, 4, -1, id="head128_q8_kv4"),
    pytest.param(256, 16, 8, -1, id="head256_q16_kv8"),
    pytest.param(512,
                 4,
                 2,
                 -1,
                 id="head512_q4_kv2",
                 marks=pytest.mark.skipif(
                     _device_sm() not in NATIVE_SMS,
                     reason="D512 CuTe DSL FMHA is unavailable on this SM")),
    pytest.param(512,
                 4,
                 2,
                 4,
                 id="head512_q4_kv2_sliding",
                 marks=pytest.mark.skipif(
                     _device_sm() not in NATIVE_SMS,
                     reason="D512 CuTe DSL FMHA is unavailable on this SM")),
    pytest.param(512,
                 8,
                 2,
                 -1,
                 id="head512_q8_kv2",
                 marks=pytest.mark.skipif(
                     _device_sm() not in NATIVE_SMS,
                     reason="D512 CuTe DSL FMHA is unavailable on this SM")),
])
def test_shared_kv_chunked_prefill(head_size, num_q_heads, num_kv_heads,
                                   sliding_window):
    cfg = dict(BASE)
    cfg["head_size"] = head_size
    cfg["num_q_heads"] = num_q_heads
    cfg["num_kv_heads"] = num_kv_heads
    cfg["sliding_window_size"] = sliding_window
    p = AttentionParams(batch_size=2, seq_len=8, is_prefill=True, **cfg)
    gen = torch.Generator().manual_seed(2600 + head_size)
    runner = AttentionPluginRunner(p)
    shared_runner = AttentionPluginRunner(p, enable_kv_shared=1)
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)
    b, s = p.batch_size, p.seq_len

    # Two donor chunks (own KV) build a 2S-token cache.
    ctx_len = torch.full((b, ), s, dtype=torch.int32, device=DEV)
    for pos in (0, s):
        pos_ids = torch.arange(pos, pos + s, dtype=torch.int32,
                               device=DEV)[None].repeat(b, 1)
        cache_idx = torch.full((b, ), pos, dtype=torch.int32, device=DEV)
        mask = sliding_window_mask(s, pos + s, sliding_window, DEV)
        qkv = torch.randn((b, s, p.qkv_hidden_size),
                          generator=gen,
                          dtype=torch.float32).to(DEV)
        _, ref_k, ref_v = compute_attention(qkv.float(), ref_k, ref_v, cos,
                                            sin, pos_ids, cache_idx, p, mask)
        runner.run(qkv.to(torch.float16),
                   plugin_kv,
                   ctx_len,
                   combined,
                   cache_idx,
                   input_shapes={"kv_cache_indices":
                                 (0, )} if pos == 0 else None)

    # Shared-KV second chunk: Q-only QKV at positions S..2S-1 attends the full
    # donor cache (pos_ids/cache_idx/mask keep the chunk-2 values from the loop).
    q2 = torch.randn((b, s, p.q_hidden), generator=gen,
                     dtype=torch.float32).to(DEV)
    donor_before = plugin_kv.clone()
    attn_out, plugin_kv = shared_runner.run(q2.to(torch.float16), plugin_kv,
                                            ctx_len, combined, cache_idx)

    q2r = apply_rotary_embedding(
        q2.reshape(b, s, p.num_q_heads, p.head_size).transpose(1, 2), cos, sin,
        pos_ids)
    ref_out = scaled_dot_product_attention(q2r, ref_k[:, :, :2 * s],
                                           ref_v[:, :, :2 * s], p.qk_scale,
                                           mask, p.num_q_heads, p.num_kv_heads)
    ref_out = ref_out.transpose(1, 2).reshape(b, s, p.q_hidden)
    assert_close("shared-kv-chunked", ref_out, attn_out)
    _assert_cache_untouched("shared-kv-chunked", donor_before, plugin_kv)


# Shared-KV tree (speculative) decode: the donor's own-KV tree pass writes the
# candidate K/V, then the shared layer's Q (roped with tree position IDs)
# attends the donor cache under the same tree mask. Round 2 commits a prefix.
def test_shared_kv_tree_decode():
    cfg = dict(BASE)
    cfg["num_kv_heads"] = 4
    p = AttentionParams(batch_size=2, seq_len=4, **cfg)
    gen = torch.Generator().manual_seed(2700)
    runner = AttentionPluginRunner(p, enable_tree_attention=True)
    shared_runner = AttentionPluginRunner(p,
                                          enable_tree_attention=True,
                                          enable_kv_shared=1)
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
        _, ref_k_out, ref_v_out = compute_attention(qkv.float(), ref_k, ref_v,
                                                    cos, sin, pos_ids,
                                                    cache_idx, p, full_mask)
        runner.run(qkv.to(torch.float16), plugin_kv, ctx_len, combined,
                   cache_idx, packed, pos_ids)

        # Shared-KV tree pass: Q-only QKV (C = Hq*D), donor cache read-only.
        q2 = torch.randn((b, s, p.q_hidden),
                         generator=gen,
                         dtype=torch.float32).to(DEV)
        donor_before = plugin_kv.clone()
        attn_out, plugin_kv = shared_runner.run(q2.to(torch.float16),
                                                plugin_kv, ctx_len, combined,
                                                cache_idx, packed, pos_ids)

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
