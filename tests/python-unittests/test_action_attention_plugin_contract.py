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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import pytest
from test_attention_plugin import (DEV, apply_rotary_embedding,
                                   scaled_dot_product_attention)
from test_plugin_base import _device_sm, torch

ATOL = 1e-2
RTOL = 1e-2
STEPS = 10
SEQ_LEN = 64
Q_HEADS = 16
KV_HEADS = 8
HEAD_SIZE = 128
Q_HIDDEN = Q_HEADS * HEAD_SIZE
KV_HIDDEN = KV_HEADS * HEAD_SIZE
QKV_HIDDEN = (Q_HEADS + 2 * KV_HEADS) * HEAD_SIZE
ATTENTION_SCALE = 0.0883883461


@dataclass(frozen=True)
class ActionAttentionCase:
    name: str
    starts: tuple[int, ...]
    capacity: int
    scrambled: bool


CASES = (
    ActionAttentionCase("b1_c320_aligned384_identity", (320, ), 512, False),
    ActionAttentionCase("b1_prefix384_aligned_identity", (384, ), 512, False),
    ActionAttentionCase("b2_c300_380_scrambled", (300, 380), 512, True),
    ActionAttentionCase("b2_partial127_129_identity", (127, 129), 256, False),
)


def _make_rope(max_pos: int):
    half = HEAD_SIZE // 2
    positions = torch.arange(max_pos, dtype=torch.float32, device=DEV)[:, None]
    inv_freq = 1.0 / (10000.0**(
        torch.arange(half, dtype=torch.float32, device=DEV)[None, :] / half))
    angles = positions * inv_freq
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin, torch.cat((cos, sin), dim=-1)[None].contiguous()


def _finite_qkv(batch: int, step: int, generator) -> torch.Tensor:
    qkv = torch.randn((batch, SEQ_LEN, QKV_HIDDEN),
                      generator=generator,
                      dtype=torch.float32,
                      device=DEV)
    qkv = qkv * (0.16 + 0.004 * step) + torch.linspace(
        -0.02, 0.02, QKV_HIDDEN, device=DEV)
    qkv[:, :, Q_HIDDEN + KV_HIDDEN:] += torch.linspace(-0.75,
                                                       0.75,
                                                       SEQ_LEN,
                                                       device=DEV)[None, :,
                                                                   None]
    return qkv.to(torch.float16)


def _seed_prefix(plugin_kv: torch.Tensor, starts: tuple[int, ...],
                 generator) -> None:
    for batch_idx, start in enumerate(starts):
        k = torch.randn((KV_HEADS, start, HEAD_SIZE),
                        generator=generator,
                        dtype=torch.float32,
                        device=DEV)
        v = torch.randn((KV_HEADS, start, HEAD_SIZE),
                        generator=generator,
                        dtype=torch.float32,
                        device=DEV)
        plugin_kv[batch_idx, 0, :, :start] = (k * 0.025).to(torch.float16)
        plugin_kv[batch_idx, 1, :, :start] = (v * 0.025).to(torch.float16)


def _dense_reference(qkv,
                     before,
                     starts,
                     cos,
                     sin,
                     *,
                     causal=False,
                     skip_rope=False,
                     start_delta=0,
                     kv_scales=None):
    after = before.clone()
    rows = []
    for batch_idx, start in enumerate(starts):
        effective_start = start + start_delta
        q = qkv[batch_idx:batch_idx + 1, :, :Q_HIDDEN].reshape(
            1, SEQ_LEN, Q_HEADS, HEAD_SIZE).transpose(1, 2)
        k = qkv[batch_idx:batch_idx + 1, :,
                Q_HIDDEN:Q_HIDDEN + KV_HIDDEN].reshape(1, SEQ_LEN, KV_HEADS,
                                                       HEAD_SIZE).transpose(
                                                           1, 2)
        v = qkv[batch_idx:batch_idx + 1, :,
                Q_HIDDEN + KV_HIDDEN:].reshape(1, SEQ_LEN, KV_HEADS,
                                               HEAD_SIZE).transpose(1, 2)
        if not skip_rope:
            pos = torch.arange(effective_start,
                               effective_start + SEQ_LEN,
                               dtype=torch.int32,
                               device=DEV)[None]
            q = apply_rotary_embedding(q, cos, sin, pos)
            k = apply_rotary_embedding(k, cos, sin, pos)
        if kv_scales is not None:
            k = ((k.float() / kv_scales[0]).to(torch.float8_e4m3fn).float() *
                 kv_scales[0]).half()
            v = ((v.float() / kv_scales[1]).to(torch.float8_e4m3fn).float() *
                 kv_scales[1]).half()
        after[batch_idx, 0, :, start:start + SEQ_LEN] = k[0].to(torch.float16)
        after[batch_idx, 1, :, start:start + SEQ_LEN] = v[0]
        mask = None
        if causal:
            q_abs = torch.arange(start, start + SEQ_LEN, device=DEV)[:, None]
            k_abs = torch.arange(start + SEQ_LEN, device=DEV)[None, :]
            mask = (k_abs <= q_abs).to(torch.int32)
        out = scaled_dot_product_attention(
            q.float(), after[batch_idx:batch_idx + 1,
                             0, :, :start + SEQ_LEN].float(),
            after[batch_idx:batch_idx + 1, 1, :, :start + SEQ_LEN].float(),
            ATTENTION_SCALE, mask, Q_HEADS, KV_HEADS)
        rows.append(out.transpose(1, 2).reshape(SEQ_LEN, Q_HIDDEN))
    return torch.stack(rows, dim=0).to(torch.float16), after


def _metric(expected, actual):
    diff = (expected.float() - actual.float()).abs()
    violations = int((diff
                      > ATOL + RTOL * expected.float().abs()).sum().item())
    return violations, float(diff.max().item()) if diff.numel() else 0.0


def _assert_close(expected, actual) -> None:
    assert torch.isfinite(expected).all()
    assert torch.isfinite(actual).all()
    violations, max_abs = _metric(expected, actual)
    assert violations == 0, f"violations={violations}, max_abs={max_abs}"


def _assert_mutation_fails(expected, actual) -> None:
    violations, _ = _metric(expected, actual)
    assert violations > 0


@pytest.mark.skipif(
    _device_sm() not in (80, 86, 87, 89, 90, 100, 101, 103, 110, 120, 121),
    reason="D128 CuTe DSL action attention contract requires a supported SM",
)
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_action_attention_direct_qkv_contract(case: ActionAttentionCase):
    """Repeated FP16 denoising preserves prefix bytes and matches non-causal attention."""
    from test_action_two_pool_attention import _run_two_pool_case
    _run_two_pool_case(case.starts, False, case.scrambled, case.capacity)


@pytest.mark.skipif(
    _device_sm() not in (80, 86, 87, 89, 90, 100, 101, 103, 110, 120, 121),
    reason="FP8 paged action attention requires a supported SM",
)
@pytest.mark.parametrize("starts,capacity", [((4095, ), 4224)])
def test_action_attention_fp8_repeated_writes(starts, capacity):
    """FP8 prefixes are dequantized while repeated action KV writes remain FP16."""
    from test_action_two_pool_attention import _run_two_pool_case
    _run_two_pool_case(starts, True, True, capacity)
