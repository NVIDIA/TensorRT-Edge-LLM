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
Sliding-window tests for full-cache and bounded-cache AttentionPlugin modes.

Split into its own file since sliding window is a distinct feature (and a
scale-up target). Reuses the AttentionPlugin harness/reference from
test_attention_plugin. Validated against the PyTorch reference at cos > 0.99999.

The plugin uses one window semantic everywhere (prefill via CuTe DSL FMHA or
FMHA_v2, decode via XQA): ``sliding_window_size`` is the number of keys
attended in total, query included -- the HF ``sliding_window`` convention. See
sliding_window_mask in test_attention_plugin.
"""

from __future__ import annotations

import pytest
import test_attention_plugin as attention_test
from test_attention_plugin import (BASE, DEPENDENCIES_AVAILABLE, DEV,
                                   IMPORT_ERROR, AttentionParams,
                                   AttentionPluginRunner, _device_sm,
                                   _empty_caches, _make_rope,
                                   _ragged_prefill_ref, _ragged_qkv,
                                   _run_rounds)
from test_plugin_base import assert_close

if DEPENDENCIES_AVAILABLE:
    import torch

pytestmark = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE,
    reason=f"TensorRT/torch CUDA not available: {IMPORT_ERROR}")


@pytest.mark.parametrize("window", [4, 16], ids=lambda w: f"window{w}")
def test_sliding_window_prefill(window):
    p = AttentionParams(batch_size=2,
                        seq_len=8,
                        is_prefill=True,
                        sliding_window_size=window,
                        **BASE)
    _run_rounds(p, num_rounds=2, atol=1e-2, rtol=1e-2)


# Gemma4 sliding-attention layers use head 256 with a window. Head 256 has no
# CuTe DSL sliding kernel, so this exercises the FMHA_v2 sliding path on every
# SKU.
@pytest.mark.parametrize("num_kv_heads", [2], ids=lambda k: f"kv{k}")
def test_sliding_window_prefill_head256(num_kv_heads):
    cfg = dict(BASE)
    cfg["head_size"] = 256
    cfg["num_kv_heads"] = num_kv_heads
    p = AttentionParams(batch_size=2,
                        seq_len=8,
                        is_prefill=True,
                        sliding_window_size=4,
                        **cfg)
    _run_rounds(p, num_rounds=2, atol=1e-2, rtol=1e-2)


# Head 512 sliding-window prefill routes to the optimized D512 CuTe DSL FMHA
# sliding variant (fmha_d512_sw_paged) on SM100/101/110.
@pytest.mark.skipif(_device_sm() not in (100, 101, 110),
                    reason="D512 CuTe DSL sliding FMHA requires SM100/101/110")
def test_sliding_window_prefill_head512():
    cfg = dict(BASE)
    cfg["head_size"] = 512
    cfg["num_q_heads"] = 4
    cfg["num_kv_heads"] = 2
    p = AttentionParams(batch_size=2,
                        seq_len=8,
                        is_prefill=True,
                        sliding_window_size=4,
                        **cfg)
    _run_rounds(p, num_rounds=2, atol=1e-2, rtol=1e-2)


def test_sliding_window_decode():
    # capacity > window so the window actually clips history during decode
    cfg = dict(BASE)
    cfg["kv_cache_capacity"] = 32
    cfg["max_seq_len"] = 24
    cfg["max_position_embeddings"] = 64
    p = AttentionParams(batch_size=2, seq_len=1, sliding_window_size=8, **cfg)
    _run_rounds(p, num_rounds=12, atol=1e-2, rtol=1e-2)


def test_bounded_swa_ragged_prefill_scrambled_page_table():
    seqlens = [8, 5]
    cfg = dict(BASE)
    cfg["kv_cache_capacity"] = 32
    p = AttentionParams(batch_size=len(seqlens),
                        seq_len=max(seqlens),
                        is_prefill=True,
                        sliding_window_size=4,
                        **cfg)
    gen = torch.Generator().manual_seed(9087)
    runner = AttentionPluginRunner(p,
                                   bounded_swa_cache=True,
                                   scramble_page_table=True)
    cos, sin, combined = _make_rope(p, gen)
    _, _, plugin_kv = _empty_caches(p)
    qkv = _ragged_qkv(seqlens, p, gen)
    context_lengths = torch.tensor(seqlens, dtype=torch.int32, device=DEV)
    cache_indices = torch.zeros(len(seqlens), dtype=torch.int32, device=DEV)

    attention_output, _ = runner.run(qkv.to(torch.float16),
                                     plugin_kv,
                                     context_lengths,
                                     combined,
                                     cache_indices,
                                     input_shapes={"kv_cache_indices": (0, )})
    reference_rows = _ragged_prefill_ref(qkv.float(), cos, sin, seqlens, p,
                                         p.sliding_window_size)
    for batch_idx, seq_len in enumerate(seqlens):
        assert_close(f"bounded-swa-ragged.b{batch_idx}",
                     reference_rows[batch_idx],
                     attention_output[batch_idx, :seq_len])

    k_page_ids = runner._make_page_table(len(seqlens))[:, 0].flatten()
    identity = torch.arange(len(seqlens) * runner.mpps,
                            dtype=torch.int32,
                            device=DEV)
    assert not torch.equal(k_page_ids, identity)


def _set_sparse_pages(runner, logical_pages):
    page_table = runner._make_page_table(1)
    page_table.fill_(-1)
    assert len(logical_pages) <= runner.num_pages
    for physical_page, logical_page in enumerate(reversed(logical_pages)):
        page_table[0, 0, logical_page] = physical_page
        page_table[0, 1, logical_page] = runner.num_pages + physical_page


def test_bounded_swa_long_normal_prefill_with_sparse_page_table():
    seq_len = 320
    window = 64
    cfg = dict(BASE)
    cfg.update(kv_cache_capacity=384,
               max_batch_size=1,
               max_seq_len=seq_len,
               max_position_embeddings=384)
    p = AttentionParams(batch_size=1,
                        seq_len=seq_len,
                        is_prefill=True,
                        sliding_window_size=window,
                        **cfg)
    gen = torch.Generator().manual_seed(9183)
    runner = AttentionPluginRunner(p,
                                   bounded_swa_cache=True,
                                   num_physical_pages=2)
    _set_sparse_pages(runner, [1, 2])
    cos, sin, combined = _make_rope(p, gen)
    _, _, plugin_kv = _empty_caches(p)
    qkv = torch.randn((1, seq_len, p.qkv_hidden_size),
                      generator=gen,
                      dtype=torch.float32).to(DEV)
    context_lengths = torch.tensor([seq_len], dtype=torch.int32, device=DEV)
    cache_indices = torch.zeros(1, dtype=torch.int32, device=DEV)

    actual, _ = runner.run(qkv.to(torch.float16),
                           plugin_kv,
                           context_lengths,
                           combined,
                           cache_indices,
                           input_shapes={"kv_cache_indices": (0, )})
    reference = _ragged_prefill_ref(qkv, cos, sin, [seq_len], p, window)[0]
    assert_close("bounded-swa-long-normal-sparse", reference, actual[0])


def test_bounded_swa_shared_kv_long_chunk_uses_current_donor_kv():
    seq_len = 256
    window = 64
    cfg = dict(BASE)
    cfg.update(kv_cache_capacity=512,
               max_batch_size=1,
               max_seq_len=seq_len,
               max_position_embeddings=512)
    p = AttentionParams(batch_size=1,
                        seq_len=seq_len,
                        is_prefill=True,
                        sliding_window_size=window,
                        **cfg)
    gen = torch.Generator().manual_seed(9271)
    writer = AttentionPluginRunner(p,
                                   bounded_swa_cache=True,
                                   num_physical_pages=2)
    consumer = AttentionPluginRunner(p,
                                     enable_kv_shared=1,
                                     bounded_swa_cache=True,
                                     shared_current_kv=True,
                                     num_physical_pages=2)
    cos, sin, combined = _make_rope(p, gen)
    ref_k, ref_v, plugin_kv = _empty_caches(p)
    context_lengths = torch.tensor([seq_len], dtype=torch.int32, device=DEV)

    _set_sparse_pages(writer, [1])
    first_qkv = torch.randn((1, seq_len, p.qkv_hidden_size),
                            generator=gen,
                            dtype=torch.float32).to(DEV)
    first_positions = torch.arange(seq_len, dtype=torch.int32,
                                   device=DEV)[None]
    first_indices = torch.zeros(1, dtype=torch.int32, device=DEV)
    first_mask = attention_test.sliding_window_mask(seq_len, seq_len, window,
                                                    DEV)
    first_ref, ref_k, ref_v = attention_test.compute_attention(
        first_qkv, ref_k, ref_v, cos, sin, first_positions, first_indices, p,
        first_mask)
    first_actual, plugin_kv = writer.run(
        first_qkv.to(torch.float16),
        plugin_kv,
        context_lengths,
        combined,
        first_indices,
        input_shapes={"kv_cache_indices": (0, )})
    assert_close("bounded-swa-writer-first", first_ref, first_actual)

    _set_sparse_pages(writer, [1, 3])
    second_qkv = torch.randn((1, seq_len, p.qkv_hidden_size),
                             generator=gen,
                             dtype=torch.float32).to(DEV)
    second_positions = torch.arange(seq_len,
                                    2 * seq_len,
                                    dtype=torch.int32,
                                    device=DEV)[None]
    second_indices = torch.tensor([seq_len], dtype=torch.int32, device=DEV)
    second_mask = attention_test.sliding_window_mask(seq_len, 2 * seq_len,
                                                     window, DEV)
    second_ref, ref_k, ref_v = attention_test.compute_attention(
        second_qkv, ref_k, ref_v, cos, sin, second_positions, second_indices,
        p, second_mask)
    second_actual, plugin_kv = writer.run(second_qkv.to(torch.float16),
                                          plugin_kv, context_lengths, combined,
                                          second_indices)
    assert_close("bounded-swa-writer-long-chunk", second_ref, second_actual)

    _set_sparse_pages(consumer, [1, 3])
    shared_q = torch.randn((1, seq_len, p.q_hidden),
                           generator=gen,
                           dtype=torch.float32).to(DEV)
    current_kv = second_qkv[..., p.q_hidden:]
    shared_packed = torch.cat((shared_q, current_kv), dim=-1)
    donor_before = plugin_kv.clone()
    shared_actual, plugin_kv = consumer.run(shared_packed.to(torch.float16),
                                            plugin_kv, context_lengths,
                                            combined, second_indices)
    shared_q_roped = attention_test.apply_rotary_embedding(
        shared_q.reshape(1, seq_len, p.num_q_heads,
                         p.head_size).transpose(1, 2), cos, sin,
        second_positions)
    shared_ref = attention_test.scaled_dot_product_attention(
        shared_q_roped, ref_k, ref_v, p.qk_scale, second_mask, p.num_q_heads,
        p.num_kv_heads)
    shared_ref = shared_ref.transpose(1, 2).reshape(1, seq_len, p.q_hidden)
    assert_close("bounded-swa-shared-long-chunk", shared_ref, shared_actual)
    assert torch.equal(donor_before.view(torch.int16),
                       plugin_kv.view(torch.int16))


def test_bounded_swa_decode_scrambled_page_table():
    cfg = dict(BASE)
    cfg["kv_cache_capacity"] = 32
    cfg["max_seq_len"] = 24
    cfg["max_position_embeddings"] = 64
    p = AttentionParams(batch_size=2, seq_len=1, sliding_window_size=8, **cfg)
    _run_rounds(p,
                num_rounds=20,
                atol=1e-2,
                rtol=1e-2,
                bounded_swa_cache=True,
                scramble_page_table=True)


def test_swa_capable_full_cache_decode_scrambled_page_table():
    """The same plugin capability also accepts the runtime full-cache mode."""
    cfg = dict(BASE)
    cfg["kv_cache_capacity"] = 32
    cfg["max_seq_len"] = 24
    cfg["max_position_embeddings"] = 64
    p = AttentionParams(batch_size=2, seq_len=1, sliding_window_size=8, **cfg)
    _run_rounds(p,
                num_rounds=12,
                atol=1e-2,
                rtol=1e-2,
                swa_cache_mode="full",
                scramble_page_table=True)
