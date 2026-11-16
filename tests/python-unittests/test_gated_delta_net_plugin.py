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
                              RAGGED_CASES, PluginRunner,
                              PluginUnsupportedError, _device_sm, assert_close,
                              assert_compact_workspace,
                              assert_execute_rejected, pf_int32,
                              poison_padding)

# CuTe DSL SM-support contract for GDN (mirrors kernelSrcs/build_cutedsl.py,
# kept independent on purpose: a supported SM must RUN so a dropped kernel
# fails, not skips green).
CUTEDSL_GDN_SMS = frozenset({80, 86, 87, 89, 90, 100, 101, 103, 110, 120, 121})

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
    heads: int = 4
    k_dim: int = 128  # only 128 supported
    v_dim: int = 128
    max_batch: int = 4
    max_seq: int = 64
    value_heads: int = 0  # 0 means value_heads == heads

    @property
    def hv(self) -> int:
        return self.value_heads or self.heads


def _gdn_io_specs(c: GDNConfig, verify=False):
    h, kd, vd, mb, ms = c.heads, c.k_dim, c.v_dim, c.max_batch, c.max_seq
    hv = c.hv
    F16, F32, I32 = trt.float16, trt.float32, trt.int32
    input_specs = [
        ("q", F16, (-1, h, kd)),
        ("k", F16, (-1, h, kd)),
        ("v", F16, (-1, hv, vd)),
        ("a", F16, (-1, hv)),
        ("b", F16, (-1, hv)),
        ("A_log", F32, (hv, )),
        ("dt_bias", F16, (hv, )),
        ("h0", F32, (-1, hv, kd, vd)),
        ("context_lengths", I32, (-1, )),
        ("query_start_offsets", I32, (-1, )),
        ("state_indices", I32, (-1, )),
        ("execution_phase_marker", I32, (-1, )),
        ("context_sequence_count_carrier", I32, (-1, )),
    ]
    profiles = {
        "q": ((1, h, kd), (8, h, kd), (mb * ms, h, kd)),
        "k": ((1, h, kd), (8, h, kd), (mb * ms, h, kd)),
        "v": ((1, hv, vd), (8, hv, vd), (mb * ms, hv, vd)),
        "a": ((1, hv), (8, hv), (mb * ms, hv)),
        "b": ((1, hv), (8, hv), (mb * ms, hv)),
        "A_log": ((hv, ), (hv, ), (hv, )),
        "dt_bias": ((hv, ), (hv, ), (hv, )),
        "h0": ((1, hv, kd, vd), (1, hv, kd, vd), (mb, hv, kd, vd)),
        "context_lengths": ((1, ), (1, ), (mb, )),
        "query_start_offsets": ((2, ), (mb + 1, ), (mb + 1, )),
        "state_indices": ((1, ), (mb, ), (mb, )),
        "execution_phase_marker": ((1, ), (1, ), (8, )),
        "context_sequence_count_carrier": ((0, ), (1, ), (mb, )),
    }
    if verify:
        for name in ("tree_parent_ids", "tree_depths"):
            input_specs.append((name, I32, (-1, )))
            profiles[name] = ((1, ), (8, ), (mb * ms, ))
    return input_specs, profiles


def _gdn_plugin_fields(c: GDNConfig, verify=False, diffusion=False):
    return [
        pf_int32("k_dim", c.k_dim),
        pf_int32("v_dim", c.v_dim),
        pf_int32("use_spec_verify_state", int(verify)),
    ] + ([pf_int32("use_diffusion_state", 1)] if diffusion else [])


class GDNRunner:
    """Builds + runs gated_delta_net (one engine handles decode + prefill)."""

    def __init__(self, cfg: GDNConfig, verify=False, diffusion=False):
        self.cfg = cfg
        self.verify = verify
        self.diffusion = diffusion
        self.runner = PluginRunner()
        self._build()

    def _build(self):
        input_specs, profiles = _gdn_io_specs(self.cfg, self.verify)
        self.runner.build(
            input_specs=input_specs,
            output_names=["o", "h0_out"] +
            (["intermediate"] if self.verify else []),
            plugin_name="gated_delta_net",
            plugin_version="1",
            plugin_fields=_gdn_plugin_fields(self.cfg, self.verify,
                                             self.diffusion),
            profiles=profiles,
        )

    def run(self,
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            h0,
            ctx,
            phase=None,
            parents=None,
            depths=None,
            cuda_graph=False):
        batch, seq = q.shape[:2]
        tokens = batch * seq
        o = torch.empty((tokens, self.cfg.hv, self.cfg.v_dim),
                        dtype=v.dtype,
                        device=DEV)
        phase = (3 if seq == 1 else 1) if phase is None else phase
        carrier_extent = batch if phase == 1 else 0
        bindings = {
            "q":
            q.reshape(tokens, self.cfg.heads, self.cfg.k_dim),
            "k":
            k.reshape(tokens, self.cfg.heads, self.cfg.k_dim),
            "v":
            v.reshape(tokens, self.cfg.hv, self.cfg.v_dim),
            "a":
            a.reshape(tokens, self.cfg.hv),
            "b":
            b.reshape(tokens, self.cfg.hv),
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
        input_shapes = input_shapes or {}
        if self.verify:
            if parents is None:
                parents = torch.arange(-1,
                                       seq - 1,
                                       dtype=torch.int32,
                                       device=DEV).repeat(batch)
            bindings.update({
                "tree_parent_ids":
                parents,
                "tree_depths":
                depths if depths is not None else torch.arange(
                    seq, dtype=torch.int32, device=DEV).repeat(batch),
                "intermediate":
                torch.empty(
                    (tokens, self.cfg.heads, self.cfg.k_dim, self.cfg.v_dim),
                    dtype=torch.float32,
                    device=DEV),
            })
        if cuda_graph:
            initial = h0.clone()
            self.runner.execute(bindings, input_shapes=input_shapes)
            h0.copy_(initial)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.runner.execute(bindings,
                                    input_shapes=input_shapes,
                                    synchronize=False)
            graph.replay()
            torch.cuda.current_stream().synchronize()
        else:
            self.runner.execute(bindings, input_shapes=input_shapes)
        return o.reshape(batch, seq, self.cfg.hv, self.cfg.v_dim), h0

    def run_tokens(self, tokens, A_log, dt_bias, pool, query_lengths,
                   state_indices, phase, num_prefill):
        """Execute one step over an unpadded token run against a resident
        state pool (updated in place); returns o [T, hv, vd].

        ``tokens`` is (q, k, v, a, b) token-major; sequence ``i`` owns
        ``query_lengths[i]`` consecutive rows and pool row
        ``state_indices[i]``.
        """
        q, k, v, a, b = tokens
        q_len = torch.tensor(query_lengths, dtype=torch.int32, device=DEV)
        offsets = torch.zeros(len(query_lengths) + 1,
                              dtype=torch.int32,
                              device=DEV)
        offsets[1:] = torch.cumsum(q_len, 0)
        o = torch.empty_like(v)
        bindings = {
            "q":
            q,
            "k":
            k,
            "v":
            v,
            "a":
            a,
            "b":
            b,
            "A_log":
            A_log,
            "dt_bias":
            dt_bias,
            "h0":
            pool,
            "context_lengths":
            q_len,
            "query_start_offsets":
            offsets,
            "state_indices":
            torch.tensor(state_indices, dtype=torch.int32, device=DEV),
            "execution_phase_marker":
            torch.empty(phase, dtype=torch.int32, device=DEV),
            "context_sequence_count_carrier":
            torch.empty(max(1, num_prefill), dtype=torch.int32, device=DEV),
            "o":
            o,
            "h0_out":
            pool,
        }
        if self.verify:
            # Inert chain ancestry; a verify layer's phase 8 is rejected.
            total = q.shape[0]
            bindings.update({
                "tree_parent_ids":
                torch.arange(-1, total - 1, dtype=torch.int32, device=DEV),
                "tree_depths":
                torch.arange(total, dtype=torch.int32, device=DEV),
                "intermediate":
                torch.empty(
                    (total, self.cfg.hv, self.cfg.k_dim, self.cfg.v_dim),
                    dtype=torch.float32,
                    device=DEV),
            })
        input_shapes = ({
            "context_sequence_count_carrier": (0, )
        } if num_prefill == 0 else None)
        # The packed-varlen prefill requires an explicit (non-default) stream.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.runner.execute(bindings, input_shapes=input_shapes)
        return o


def _spec_state_output_shape(*, context_profile: bool):
    cfg = GDNConfig(heads=2, max_batch=4, max_seq=4)
    h, kd, vd = cfg.heads, cfg.k_dim, cfg.v_dim
    token_profile = ((1, h, kd), (8, h, kd), (16, h, kd))
    value_profile = ((1, h, vd), (8, h, vd), (16, h, vd))
    gate_profile = ((1, h), (8, h), (16, h))
    batch_profile = ((1, ), (2, ), (4, ))
    phase = 1 if context_profile else 5
    carrier_profile = (batch_profile if context_profile else
                       ((0, ), (0, ), (0, )))
    profiles = {
        "q": token_profile,
        "k": token_profile,
        "v": value_profile,
        "a": gate_profile,
        "b": gate_profile,
        "A_log": ((h, ), (h, ), (h, )),
        "dt_bias": ((h, ), (h, ), (h, )),
        "h0": ((1, h, kd, vd), (2, h, kd, vd), (4, h, kd, vd)),
        "context_lengths": batch_profile,
        "query_start_offsets": ((2, ), (3, ), (5, )),
        "state_indices": batch_profile,
        "execution_phase_marker": ((phase, ), (phase, ), (phase, )),
        "context_sequence_count_carrier": carrier_profile,
        "tree_parent_ids": ((1, ), (8, ), (16, )),
        "tree_depths": ((1, ), (8, ), (16, )),
    }
    runner = PluginRunner().build(
        input_specs=[
            ("q", trt.float16, (-1, h, kd)),
            ("k", trt.float16, (-1, h, kd)),
            ("v", trt.float16, (-1, h, vd)),
            ("a", trt.float16, (-1, h)),
            ("b", trt.float16, (-1, h)),
            ("A_log", trt.float32, (h, )),
            ("dt_bias", trt.float16, (h, )),
            ("h0", trt.float32, (-1, h, kd, vd)),
            ("context_lengths", trt.int32, (-1, )),
            ("query_start_offsets", trt.int32, (-1, )),
            ("state_indices", trt.int32, (-1, )),
            ("execution_phase_marker", trt.int32, (-1, )),
            ("context_sequence_count_carrier", trt.int32, (-1, )),
            ("tree_parent_ids", trt.int32, (-1, )),
            ("tree_depths", trt.int32, (-1, )),
        ],
        output_names=["o", "h0_out", "intermediate_states"],
        plugin_name="gated_delta_net",
        plugin_version="1",
        plugin_fields=[
            pf_int32("k_dim", kd),
            pf_int32("v_dim", vd),
            pf_int32("use_spec_verify_state", 1),
        ],
        profiles=profiles,
    )
    for name, (_, opt_shape, _) in profiles.items():
        assert runner.context.set_input_shape(name, opt_shape)
    return tuple(runner.context.get_tensor_shape("intermediate_states"))


def test_spec_state_output_is_empty_for_context_profile():
    assert _spec_state_output_shape(context_profile=True) == (0, 2, 128, 128)


def test_spec_state_output_keeps_token_rows_for_verify_profile():
    assert _spec_state_output_shape(context_profile=False) == (8, 2, 128, 128)


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


@pytest.mark.parametrize("batch", (1, 2))
@pytest.mark.parametrize("nodes", (1, 8, 16))
def test_one_verify_artifact_executes_inactive_chain_and_tree(batch, nodes):
    # Given a single GDN verify artifact, when changing ancestry with graphs on/off,
    # then ancestry controls output and verification preserves the resident state.
    cfg = GDNConfig(max_batch=batch, max_seq=16)
    runner = GDNRunner(cfg, verify=True)
    inputs = _rand(cfg, batch, nodes, torch.Generator().manual_seed(173))
    initial = torch.full((batch, cfg.heads, cfg.k_dim, cfg.v_dim),
                         0.25,
                         device=DEV)
    lengths = torch.full((batch, ), nodes, dtype=torch.int32, device=DEV)
    expected, expected_state = gdn_ref(*inputs, initial, lengths)
    parent_ids = [-1] + [(node - 1) // 2 for node in range(1, nodes)]
    parents = torch.tensor(parent_ids, dtype=torch.int32,
                           device=DEV).repeat(batch)
    tree_outputs = []
    depth_values = []
    for node in range(nodes):
        path = []
        while node >= 0:
            path.append(node)
            node = parent_ids[node]
        path.reverse()
        depth_values.append(len(path) - 1)
        path_inputs = [value[:, path]
                       for value in inputs[:5]] + list(inputs[5:])
        path_output, _ = gdn_ref(*path_inputs, initial)
        tree_outputs.append(path_output[:, -1])
    tree_expected = torch.stack(tree_outputs, dim=1)
    depths = torch.tensor(depth_values, dtype=torch.int32,
                          device=DEV).repeat(batch)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cases = ((False, False), (True, False), (True, True))
        for cuda_graph in (False, True):
            for verify, branching in cases:
                actual, state = runner.run(
                    *inputs,
                    initial.clone(),
                    lengths,
                    phase=5 if verify else 1,
                    parents=parents if branching else None,
                    depths=depths if branching else None,
                    cuda_graph=cuda_graph)
                assert_close("GDN ancestry output",
                             tree_expected if branching else expected, actual,
                             8e-2, 8e-2)
                assert_close("GDN resident state",
                             initial if verify else expected_state, state,
                             8e-2, 8e-2)


# --------------------------------------------------------------------------- #
# Compact phase 8: varlen prefill rows first, then single-step decode rows,
# against a shared resident state pool with non-contiguous slots.
# --------------------------------------------------------------------------- #
COMPACT_GDN_SMS = frozenset({80, 86, 87, 89, 90, 100, 101, 103, 110, 120, 121})
_GDN_POOL_ROWS = 6
# (label, prefill [(slot, query_length, continued)], decode slots)
_GDN_COMPACT_CASES = [
    ("prefill_and_decode", [(4, 37, True), (1, 130, False)], [3, 0]),
    ("prefill_only", [(2, 64, True), (5, 9, False)], []),
    ("single_row_prefill", [(5, 1, True)], [1, 2, 4]),
    ("decode_only", [], [3, 0, 5]),
    # Profile 0 upper bound of the test engine: Np = B = 6, T_p = T_max.
    ("profile_upper_bound_prefill_only", [(0, 256, True), (1, 256, False),
                                          (2, 256, True), (3, 256, False),
                                          (4, 256, True), (5, 256, True)], []),
]


def _gdn_compact_inputs(cfg, query_lengths, gen):
    h, hv, kd, vd = cfg.heads, cfg.hv, cfg.k_dim, cfg.v_dim
    total = sum(query_lengths)

    def rn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(DEV)

    tokens = (rn(total, h, kd).to(torch.float16), rn(total, h,
                                                     kd).to(torch.float16),
              rn(total, hv, vd).to(torch.float16), rn(total,
                                                      hv).to(torch.float16),
              rn(total, hv).to(torch.float16))
    A_log = rn(hv)
    dt_bias = (rn(hv) * 0.1).to(torch.float16)
    return tokens, A_log, dt_bias


def _gdn_slice(tokens, begin, end):
    return tuple(t[begin:end].contiguous() for t in tokens)


@pytest.mark.skipif(
    _device_sm() not in COMPACT_GDN_SMS,
    reason="packed-varlen GDN prefill is unavailable on this SM")
@pytest.mark.parametrize("value_heads", [4, 8], ids=lambda v: f"hv{v}")
@pytest.mark.parametrize("label,prefill,decode", _GDN_COMPACT_CASES)
def test_compact_phase_matches_separate_prefill_and_decode(
        label, prefill, decode, value_heads):
    """Phase 8 equals separate execution on one state pool: each prefill
    sequence (fresh or continued) matches its own prefill step within
    tolerance (the packed-varlen kernel accumulates in a different order from
    the padded one), the decode rows and their states match a phase 3 step bit
    for bit, and pool rows outside the step stay unchanged, with value heads
    equal to or twice the Q/K heads, up to the profile bound Np = B,
    T_p = T_max."""
    cfg = GDNConfig(heads=4,
                    max_batch=_GDN_POOL_ROWS,
                    max_seq=256,
                    value_heads=value_heads)
    gen = torch.Generator().manual_seed(5150 + len(prefill) + len(decode))
    query_lengths = [q for _, q, _ in prefill] + [1] * len(decode)
    slots = [slot for slot, _, _ in prefill] + decode
    tokens, A_log, dt_bias = _gdn_compact_inputs(cfg, query_lengths, gen)
    pool = (torch.randn(
        _GDN_POOL_ROWS, cfg.hv, cfg.k_dim, cfg.v_dim, generator=gen) *
            0.1).to(DEV)
    for slot, _, continued in prefill:
        if not continued:
            pool[slot].zero_()
    r = GDNRunner(cfg)

    compact_pool = pool.clone()
    compact_o = r.run_tokens(tokens, A_log, dt_bias, compact_pool,
                             query_lengths, slots, 8, len(prefill))

    offset = 0
    for slot, q_len, continued in prefill:
        ref_pool = pool.clone()
        ref_o = r.run_tokens(_gdn_slice(tokens, offset, offset + q_len), A_log,
                             dt_bias, ref_pool, [q_len], [slot],
                             2 if continued else 1, 1)
        assert_close(f"gdn-compact[{label}].o[slot{slot}]", ref_o,
                     compact_o[offset:offset + q_len], 8e-2, 8e-2)
        assert_close(f"gdn-compact[{label}].state[slot{slot}]", ref_pool[slot],
                     compact_pool[slot], 8e-2, 8e-2)
        offset += q_len
    if decode:
        ref_pool = pool.clone()
        ref_o = r.run_tokens(_gdn_slice(tokens, offset,
                                        offset + len(decode)), A_log, dt_bias,
                             ref_pool, [1] * len(decode), decode, 3, 0)
        assert torch.equal(ref_o, compact_o[offset:]), \
            f"gdn-compact[{label}]: decode rows differ from phase 3"
        for slot in decode:
            assert torch.equal(ref_pool[slot], compact_pool[slot]), \
                f"gdn-compact[{label}]: decode state of slot {slot} differs"
    for slot in set(range(_GDN_POOL_ROWS)) - set(slots):
        assert torch.equal(pool[slot], compact_pool[slot]), \
            f"gdn-compact[{label}]: untouched slot {slot} changed"


@pytest.mark.parametrize("runner_kwargs,state", [({
    "verify": True
}, "spec-verify"), ({
    "diffusion": True
}, "diffusion")],
                         ids=["spec_verify", "diffusion"])
def test_compact_phase_rejected_for_unsupported_layer(runner_kwargs, state,
                                                      capfd):
    """On every GPU, a spec-verify or diffusion layer rejects phase 8 with the
    unsupported-layer error, which names the layer and its shape, without
    touching the resident state pool."""
    cfg = GDNConfig(heads=4, max_batch=8, max_seq=64)
    gen = torch.Generator().manual_seed(77)
    tokens, A_log, dt_bias = _gdn_compact_inputs(cfg, [5, 1], gen)
    pool = (torch.randn(
        _GDN_POOL_ROWS, cfg.hv, cfg.k_dim, cfg.v_dim, generator=gen) *
            0.1).to(DEV)
    before = pool.clone()
    r = GDNRunner(cfg, **runner_kwargs)
    message = (
        f"gated_delta_net 'gated_delta_net' (K head dim 128, V head "
        f"dim 128, {state} state): compact execution is not supported for this "
        "layer on this GPU.")
    assert_execute_rejected(
        capfd, message, lambda: r.run_tokens(tokens, A_log, dt_bias, pool,
                                             [5, 1], [2, 4], 8, 1))
    assert torch.equal(pool, before), \
        "phase 8 rejection must not write the state pool"


@pytest.mark.parametrize("runner_kwargs", [{}, {
    "verify": True
}, {
    "diffusion": True
}],
                         ids=["plain", "spec_verify", "diffusion"])
def test_build_reports_whether_the_layer_reserved_the_compact_workspace(
        capfd, runner_kwargs):
    """The engine build reports the layer as reserving the compact
    workspace exactly where it accepts compact execution: a plain layer on
    the packed-varlen GDN SMs, a spec-verify or diffusion layer nowhere; a
    layer that does not reserve it is named in a WARNING."""
    r = GDNRunner(GDNConfig(heads=4, max_batch=8, max_seq=64), **runner_kwargs)
    assert_compact_workspace(
        r.runner, not runner_kwargs and _device_sm() in COMPACT_GDN_SMS, capfd)


def test_value_heads_not_a_multiple_of_qk_heads_fail_to_build():
    """Every GDN kernel maps value head i to Q/K head i / (Hv / H), so a layer
    whose value heads are not a positive multiple of its Q/K heads is rejected
    at build time instead of computing wrong outputs."""
    for heads, value_heads in ((4, 6), (4, 2)):
        cfg = GDNConfig(heads=heads, value_heads=value_heads)
        input_specs, profiles = _gdn_io_specs(cfg)
        with pytest.raises(PluginUnsupportedError):
            PluginRunner().build(input_specs=input_specs,
                                 output_names=["o", "h0_out"],
                                 plugin_name="gated_delta_net",
                                 plugin_version="1",
                                 plugin_fields=_gdn_plugin_fields(cfg),
                                 profiles=profiles,
                                 expect_unsupported=True)
