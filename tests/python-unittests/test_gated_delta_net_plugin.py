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
gated_delta_net (GDN) plugin tests vs a PyTorch reference (gdn_ref).

GDN is the linear-attention mixer used by Qwen3-Next (not Nemotron). k_dim ==
v_dim == 128 are the only supported dims. Decode (seq=1) carries the recurrent
state; prefill (seq>1, Blackwell chunked path on SM>=100) starts fresh. Ragged
prefill verifies the per-row final state and poisons the padding region.

Run:
    python3 -m pytest tests/python-unittests/test_gated_delta_net_plugin.py -v
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              RAGGED_CASES, PluginRunner, _device_sm,
                              assert_close, pf_int32, poison_padding)

# CuTe DSL SM-support contract for GDN (mirrors kernelSrcs/build_cutedsl.py,
# kept independent on purpose: a supported SM must RUN so a dropped kernel
# fails, not skips green).
CUTEDSL_GDN_SMS = frozenset({80, 86, 87, 89, 90, 100, 101, 110, 120, 121})

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

# GDN is a CuTe DSL kernel, gated below on its promised SM support. On a
# supported SM the build must provide the kernel -- a missing one fails the
# test, it does not skip.
pytestmark = [
    pytest.mark.skipif(
        not DEPENDENCIES_AVAILABLE,
        reason=f"TensorRT/torch CUDA not available: {IMPORT_ERROR}"),
    pytest.mark.skipif(_device_sm() not in CUTEDSL_GDN_SMS,
                       reason="GDN (CuTe DSL) unsupported on this SM"),
]

DEV = "cuda"


def gdn_ref(
        q: torch.Tensor,  # [n, s, h, kd]
        k: torch.Tensor,  # [n, s, h, kd]
        v: torch.Tensor,  # [n, s, hv, vd]
        a: torch.Tensor,  # [n, s, hv]   (input gate)
        b: torch.Tensor,  # [n, s, hv]   (output gate)
        A_log: torch.Tensor,  # [hv]  fp32
        dt_bias: torch.Tensor,  # [hv]
        h0: torch.Tensor,  # [n, hv, kd, vd]  fp32 (initial recurrent state)
        context_lengths: Optional[torch.Tensor] = None,  # [n]
):
    """Gated-delta-net reference (matches gdn_cutedsl kernels). h == hv assumed.

    Per head, per time step (constexpr from the kernel: scale = kd**-0.5,
    use_qk_l2norm, softplus_beta=1, threshold=20):
        qn, kn = l2norm(q), l2norm(k);  qs = qn * scale
        g    = exp(-exp(A_log) * softplus(a + dt_bias))    # decay
        beta = sigmoid(b)
        S'   = g * S
        u    = (v - kn @ S') * beta
        S    = S' + outer(kn, u)
        o    = qs @ S                                       # uses updated S
    Returns (o [n,s,hv,vd], final_state [n,hv,kd,vd]). Ragged: only the first
    context_lengths[ni] steps are processed per row.
    """
    n, s, hv, vd = v.shape
    kd = q.shape[-1]
    scale = kd**-0.5
    qf, kf, vf = q.float(), k.float(), v.float()
    af, bf = a.float(), b.float()
    a_log_f = A_log.float()
    bias = dt_bias.float()

    if context_lengths is None:
        context_lengths = torch.full((n, ), s, dtype=torch.int64)
    else:
        context_lengths = context_lengths.to(torch.int64)

    def l2norm(x):  # over last dim
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    o = torch.zeros((n, s, hv, vd), dtype=torch.float32, device=q.device)
    state = h0.float().clone()
    for ni in range(n):
        L = int(context_lengths[ni])
        S = state[ni].clone()  # [hv, kd, vd]
        for t in range(L):
            qn = l2norm(qf[ni, t])  # [hv, kd]  scale applied once (at readout)
            kn = l2norm(kf[ni, t])  # [hv, kd]
            g = torch.exp(
                -torch.exp(a_log_f) *
                torch.nn.functional.softplus(af[ni, t] + bias))  # [hv]
            beta = torch.sigmoid(bf[ni, t])  # [hv]
            Sp = g[:, None, None] * S  # [hv, kd, vd]
            pred = torch.einsum("hk,hkv->hv", kn, Sp)  # [hv, vd]
            u = (vf[ni, t] - pred) * beta[:, None]  # [hv, vd]
            S = Sp + kn[:, :, None] * u[:, None, :]  # [hv, kd, vd]
            o[ni, t] = torch.einsum("hk,hkv->hv", qn * scale, S)
        state[ni] = S
    return o, state


@dataclass
class GDNConfig:
    heads: int = 4  # h == hv
    k_dim: int = 128  # only 128 supported
    v_dim: int = 128
    max_batch: int = 4
    max_seq: int = 64


class GDNRunner:
    """Builds + runs gated_delta_net (one engine handles decode + prefill)."""

    def __init__(self, cfg: GDNConfig):
        self.cfg = cfg
        self.runner = PluginRunner()
        self._build()

    def _build(self):
        c = self.cfg
        h, kd, vd, mb, ms = c.heads, c.k_dim, c.v_dim, c.max_batch, c.max_seq
        F16, F32, I32 = trt.float16, trt.float32, trt.int32
        input_specs = [
            ("q", F16, (-1, h, kd)),
            ("k", F16, (-1, h, kd)),
            ("v", F16, (-1, h, vd)),
            ("a", F16, (-1, h)),
            ("b", F16, (-1, h)),
            ("A_log", F32, (h, )),
            ("dt_bias", F16, (h, )),
            ("h0", F32, (-1, h, kd, vd)),
            ("context_lengths", I32, (-1, )),
            ("query_start_offsets", I32, (-1, )),
            ("state_indices", I32, (-1, )),
            ("execution_phase_marker", I32, (-1, )),
            ("context_sequence_count_carrier", I32, (-1, )),
        ]
        profiles = {
            "q": ((1, h, kd), (8, h, kd), (mb * ms, h, kd)),
            "k": ((1, h, kd), (8, h, kd), (mb * ms, h, kd)),
            "v": ((1, h, vd), (8, h, vd), (mb * ms, h, vd)),
            "a": ((1, h), (8, h), (mb * ms, h)),
            "b": ((1, h), (8, h), (mb * ms, h)),
            "A_log": ((h, ), (h, ), (h, )),
            "dt_bias": ((h, ), (h, ), (h, )),
            "h0": ((1, h, kd, vd), (1, h, kd, vd), (mb, h, kd, vd)),
            "context_lengths": ((1, ), (1, ), (mb, )),
            "query_start_offsets": ((2, ), (mb + 1, ), (mb + 1, )),
            "state_indices": ((1, ), (mb, ), (mb, )),
            "execution_phase_marker": ((1, ), (1, ), (8, )),
            "context_sequence_count_carrier": ((0, ), (1, ), (mb, )),
        }
        self.runner.build(
            input_specs=input_specs,
            output_names=["o", "h0_out"],
            plugin_name="gated_delta_net",
            plugin_version="1",
            plugin_fields=[
                pf_int32("k_dim", kd),
                pf_int32("v_dim", vd),
                pf_int32("use_mtp", 0),
            ],
            profiles=profiles,
        )

    def run(self, q, k, v, a, b, A_log, dt_bias, h0, ctx):
        batch, seq = q.shape[:2]
        tokens = batch * seq
        o = torch.empty((tokens, self.cfg.heads, self.cfg.v_dim),
                        dtype=v.dtype,
                        device=DEV)
        phase = 3 if seq == 1 else 1
        carrier_extent = batch if phase == 1 else 0
        bindings = {
            "q":
            q.reshape(tokens, self.cfg.heads, self.cfg.k_dim),
            "k":
            k.reshape(tokens, self.cfg.heads, self.cfg.k_dim),
            "v":
            v.reshape(tokens, self.cfg.heads, self.cfg.v_dim),
            "a":
            a.reshape(tokens, self.cfg.heads),
            "b":
            b.reshape(tokens, self.cfg.heads),
            "A_log":
            A_log,
            "dt_bias":
            dt_bias,
            "h0":
            h0,
            "context_lengths":
            ctx,
            "query_start_offsets":
            torch.arange(0, tokens + 1, seq, dtype=torch.int32, device=DEV),
            "state_indices":
            torch.arange(batch, dtype=torch.int32, device=DEV),
            "execution_phase_marker":
            torch.empty(phase, dtype=torch.int32, device=DEV),
            "context_sequence_count_carrier":
            torch.empty(max(1, carrier_extent), dtype=torch.int32, device=DEV),
            "o":
            o,
            "h0_out":
            h0,
        }
        input_shapes = ({
            "context_sequence_count_carrier": (0, )
        } if carrier_extent == 0 else None)
        self.runner.execute(bindings, input_shapes=input_shapes)
        return o.reshape(batch, seq, self.cfg.heads, self.cfg.v_dim), h0


def _rand(cfg, n, s, gen):
    h, kd, vd = cfg.heads, cfg.k_dim, cfg.v_dim

    def rn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(DEV)

    q = rn(n, s, h, kd).to(torch.float16)
    k = rn(n, s, h, kd).to(torch.float16)
    v = rn(n, s, h, vd).to(torch.float16)
    a = rn(n, s, h).to(torch.float16)
    b = rn(n, s, h).to(torch.float16)
    A_log = rn(h)  # fp32
    dt_bias = (rn(h) * 0.1).to(torch.float16)
    return q, k, v, a, b, A_log, dt_bias


def _check(cfg, o, h0_out, ref_o, ref_state, ctx, atol, rtol):
    assert_close("h0_out", ref_state, h0_out, atol, rtol)
    for ni in range(o.shape[0]):
        L = int(ctx[ni])
        assert_close(f"o[n{ni}]", ref_o[ni, :L], o[ni, :L], atol, rtol)


# --------------------------------------------------------------------------- #
# Decode (seq=1): recurrent state carry from a non-zero initial state
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [1, 2, 4], ids=lambda b: f"bs{b}")
def test_decode(batch):
    cfg = GDNConfig()
    gen = torch.Generator().manual_seed(700 + batch)
    r = GDNRunner(cfg)
    q, k, v, a, b, A_log, dt_bias = _rand(cfg, batch, 1, gen)
    h0 = (torch.randn(batch,
                      cfg.heads,
                      cfg.k_dim,
                      cfg.v_dim,
                      generator=gen,
                      dtype=torch.float32) * 0.1).to(DEV)
    ctx = torch.ones(batch, dtype=torch.int32, device=DEV)
    o, h0_out = r.run(q, k, v, a, b, A_log, dt_bias, h0.clone(), ctx)
    ref_o, ref_state = gdn_ref(q, k, v, a, b, A_log, dt_bias, h0, ctx)
    _check(cfg, o, h0_out, ref_o, ref_state, ctx, 5e-2, 5e-2)


# --------------------------------------------------------------------------- #
# Prefill (seq>1): fresh state
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [1, 2], ids=lambda b: f"bs{b}")
@pytest.mark.parametrize("seq", [16, 64], ids=lambda s: f"seq{s}")
def test_prefill(batch, seq):
    cfg = GDNConfig()
    gen = torch.Generator().manual_seed(800 + seq + batch)
    r = GDNRunner(cfg)
    q, k, v, a, b, A_log, dt_bias = _rand(cfg, batch, seq, gen)
    h0 = torch.zeros(batch,
                     cfg.heads,
                     cfg.k_dim,
                     cfg.v_dim,
                     dtype=torch.float32,
                     device=DEV)
    ctx = torch.full((batch, ), seq, dtype=torch.int32, device=DEV)
    o, h0_out = r.run(q, k, v, a, b, A_log, dt_bias, h0.clone(), ctx)
    ref_o, ref_state = gdn_ref(q, k, v, a, b, A_log, dt_bias, h0, ctx)
    _check(cfg, o, h0_out, ref_o, ref_state, ctx, 8e-2, 8e-2)


# --------------------------------------------------------------------------- #
# RAGGED prefill: variable per-row lengths + poisoned padding
# --------------------------------------------------------------------------- #
def test_ragged_prefill():
    cfg = GDNConfig()
    seq = 64
    batch = 3
    gen = torch.Generator().manual_seed(909)
    r = GDNRunner(cfg)
    q, k, v, a, b, A_log, dt_bias = _rand(cfg, batch, seq, gen)
    ctx = torch.tensor([64, 33, 7], dtype=torch.int32, device=DEV)
    poison_padding([q, k, v, a, b], ctx)
    h0 = torch.zeros(batch,
                     cfg.heads,
                     cfg.k_dim,
                     cfg.v_dim,
                     dtype=torch.float32,
                     device=DEV)
    o, h0_out = r.run(q, k, v, a, b, A_log, dt_bias, h0.clone(), ctx)
    ref_o, ref_state = gdn_ref(q, k, v, a, b, A_log, dt_bias, h0, ctx)
    _check(cfg, o, h0_out, ref_o, ref_state, ctx, 8e-2, 8e-2)


def _ragged_cfg():
    return GDNConfig(heads=4, k_dim=128, v_dim=128, max_batch=8, max_seq=2048)


# --------------------------------------------------------------------------- #
# Required even/uneven batch cases (bs 1/2/3/4/8, seq up to 2048).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,seqlens", RAGGED_CASES)
def test_ragged_prefill_batch_sizes(label, seqlens):
    cfg = _ragged_cfg()
    bs, maxlen = len(seqlens), max(seqlens)
    gen = torch.Generator().manual_seed(1200 + maxlen + bs)
    q, k, v, a, b, A_log, dt_bias = _rand(cfg, bs, maxlen, gen)
    ctx = torch.tensor(seqlens, dtype=torch.int32, device=DEV)
    poison_padding([q, k, v, a, b], ctx)
    h0 = torch.zeros(bs,
                     cfg.heads,
                     cfg.k_dim,
                     cfg.v_dim,
                     dtype=torch.float32,
                     device=DEV)
    r = GDNRunner(cfg)
    o, h0_out = r.run(q, k, v, a, b, A_log, dt_bias, h0.clone(), ctx)
    ref_o, ref_state = gdn_ref(q, k, v, a, b, A_log, dt_bias, h0, ctx)
    _check(cfg, o, h0_out, ref_o, ref_state, ctx, 8e-2, 8e-2)


# --------------------------------------------------------------------------- #
# Batch invariance (plugin-vs-plugin): permuting batch rows permutes outputs.
# --------------------------------------------------------------------------- #
def test_batch_invariance():
    cfg = _ragged_cfg()
    seqlens = [10, 2048, 128]
    bs, maxlen = len(seqlens), max(seqlens)
    gen = torch.Generator().manual_seed(2468)
    q, k, v, a, b, A_log, dt_bias = _rand(cfg, bs, maxlen, gen)
    ctx = torch.tensor(seqlens, dtype=torch.int32, device=DEV)
    poison_padding([q, k, v, a, b], ctx)
    h0 = torch.zeros(bs,
                     cfg.heads,
                     cfg.k_dim,
                     cfg.v_dim,
                     dtype=torch.float32,
                     device=DEV)
    r = GDNRunner(cfg)
    o0, st0 = r.run(q, k, v, a, b, A_log, dt_bias, h0.clone(), ctx)
    p = torch.tensor([2, 0, 1], device=DEV)
    o1, st1 = r.run(q[p].contiguous(), k[p].contiguous(), v[p].contiguous(),
                    a[p].contiguous(), b[p].contiguous(), A_log, dt_bias,
                    h0.clone(), ctx[p].contiguous())
    pc = p.cpu()
    for new_i in range(bs):
        orig = int(pc[new_i])
        L = seqlens[orig]
        assert_close(f"gdn-batch-inv.o[{new_i}]", o0[orig, :L], o1[new_i, :L])
    assert_close("gdn-batch-inv.state", st0[pc], st1)


# --------------------------------------------------------------------------- #
# prefill -> decode state handoff: prefill L tokens, feed the recurrent state
# through N decode steps, compare to one continuous reference scan over L+N.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2], ids=lambda s: f"seed{s}")
def test_prefill_decode_handoff(seed):
    # Prefill ISL randomly chosen in [10, 2048] (seeded for reproducibility).
    prefill_len = random.Random(6000 + seed).randint(10, 2048)
    n_decode, batch = 4, 2
    cfg = GDNConfig(heads=4,
                    k_dim=128,
                    v_dim=128,
                    max_batch=2,
                    max_seq=prefill_len + n_decode + 8)
    total = prefill_len + n_decode
    gen = torch.Generator().manual_seed(3690 + prefill_len)
    q, k, v, a, b, A_log, dt_bias = _rand(cfg, batch, total, gen)
    h0 = torch.zeros(batch,
                     cfg.heads,
                     cfg.k_dim,
                     cfg.v_dim,
                     dtype=torch.float32,
                     device=DEV)
    ref_o, ref_state = gdn_ref(q, k, v, a, b, A_log, dt_bias, h0)

    r = GDNRunner(cfg)
    ctx_pre = torch.full((batch, ), prefill_len, dtype=torch.int32, device=DEV)
    sl = slice(0, prefill_len)
    _, state = r.run(q[:, sl].contiguous(), k[:, sl].contiguous(),
                     v[:, sl].contiguous(), a[:, sl].contiguous(),
                     b[:,
                       sl].contiguous(), A_log, dt_bias, h0.clone(), ctx_pre)
    ctx_dec = torch.ones(batch, dtype=torch.int32, device=DEV)
    for i in range(n_decode):
        t = prefill_len + i
        st = slice(t, t + 1)
        o, state = r.run(q[:, st].contiguous(), k[:, st].contiguous(),
                         v[:, st].contiguous(), a[:, st].contiguous(),
                         b[:, st].contiguous(), A_log, dt_bias, state, ctx_dec)
        assert_close(f"gdn-handoff.o[t={t}]", ref_o[:, t], o[:, 0])
    assert_close("gdn-handoff.state", ref_state, state)
