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
Mamba selective-state-update plugin (``update_ssm_state``) tests vs a PyTorch
reference (``selective_scan_ref``).

Centerpiece is ragged prefill (variable per-row sequence lengths in one padded
call, as Nemotron-H hits during batched MMLU prefill): the test verifies the
recurrent final state per row (the primary signal for the SSD varlen state bug)
and poisons the padding region so any kernel that reads past ``context_lengths``
is caught.

Both prefill code paths are exercised:
  * single-step loop      (2 <= seq_len < 128)
  * CuTeDSL SSD chunk-scan (seq_len >= 128; generic dim in {64,128}, plus Blackwell D80/N128)

Nemotron-realistic dims: head_dim=64, ssm_state=128, n_groups=8. bs kept <= 4.
Run:
    python3 -m pytest tests/python-unittests/test_mamba_plugin.py -v
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              RAGGED_CASES, PluginRunner, assert_close,
                              pf_int32, poison_padding)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

pytestmark = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE,
    reason=f"TensorRT/torch CUDA not available: {IMPORT_ERROR}")

DEV = "cuda"
PLUGIN_NAME = "update_ssm_state"
PLUGIN_VERSION = "1"


def selective_scan_ref(
        x: torch.Tensor,  # [b, s, h, dim]  (or [b, h, dim] for decode)
        A: torch.Tensor,  # [h]            fp32
        B: torch.Tensor,  # [b, s, g, n]   (or [b, g, n])
        C: torch.Tensor,  # [b, s, g, n]   (or [b, g, n])
        dt: torch.Tensor,  # [b, s, h]      (or [b, h])
        dt_bias: Optional[torch.Tensor],  # [h]
        D: Optional[torch.Tensor],  # [h]
        state0: torch.Tensor,  # [b, h, dim, n]
        ngroups: int,
        dt_softplus: bool,
        context_lengths: Optional[torch.Tensor] = None,  # [b] int
):
    """Sequential fp32 reference. Returns (y, final_state).

    y has the same (b,[s,]h,dim) shape as x; padding rows of y are left zero.
    """
    decode = (x.dim() == 3)
    if decode:
        x = x[:, None]
        B = B[:, None]
        C = C[:, None]
        dt = dt[:, None]
    b, s, h, dim = x.shape
    hpg = h // ngroups

    x = x.float()
    A = A.float()
    B = B.float()
    C = C.float()
    dt = dt.float()
    bias = dt_bias.float() if dt_bias is not None else None
    Df = D.float() if D is not None else None

    if context_lengths is None:
        context_lengths = torch.full((b, ), s, dtype=torch.int64)
    else:
        context_lengths = context_lengths.to(torch.int64)

    y = torch.zeros((b, s, h, dim), dtype=torch.float32, device=x.device)
    final_state = state0.float().clone()

    for bi in range(b):
        L = int(context_lengths[bi])
        st = state0[bi].float().clone()  # [h, dim, n]
        for t in range(L):
            dt_t = dt[bi, t]  # [h]
            if bias is not None:
                dt_t = dt_t + bias
            if dt_softplus:
                dt_t = torch.nn.functional.softplus(dt_t)
            dA = torch.exp(A * dt_t)  # [h]
            x_t = x[bi, t]  # [h, dim]
            B_h = B[bi, t].repeat_interleave(hpg, dim=0)  # [h, n]
            C_h = C[bi, t].repeat_interleave(hpg, dim=0)  # [h, n]
            # state update: [h,dim,n] = [h,dim,n]*dA + (dt*x)[h,dim,1]*B[h,1,n]
            st = st * dA[:, None, None] + \
                (dt_t[:, None] * x_t)[:, :, None] * B_h[:, None, :]
            y_t = (st * C_h[:, None, :]).sum(-1)  # [h, dim]
            if Df is not None:
                y_t = y_t + Df[:, None] * x_t
            y[bi, t] = y_t
        final_state[bi] = st

    if decode:
        y = y[:, 0]
    return y, final_state


SSD_CHUNK = 128  # CuTeDSL SSD chunk size; seq_len >= 128 selects that path


def _supports_d80_blackwell():
    if not DEPENDENCIES_AVAILABLE or not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor in (100, 101, 110)


REQUIRES_D80_BLACKWELL = pytest.mark.skipif(
    not _supports_d80_blackwell(),
    reason=
    "D80 SSD requires compiled Blackwell artifacts on SM100, SM101, or SM110")


@dataclass
class MambaConfig:
    nheads: int = 8
    head_dim: int = 64  # "dim"   (64/128 generic; 80 with N=128 on supported Blackwell GPUs)
    dstate: int = 128  # "dstate" (must be 64 or 128 for the SSD path)
    ngroups: int = 2
    dt_softplus: bool = True
    max_batch: int = 4
    max_seq: int = 256


class MambaRunner:
    """Builds + runs update_ssm_state for prefill (4D x) or decode (3D x)."""

    def __init__(self, cfg: MambaConfig, prefill: bool):
        self.cfg = cfg
        self.prefill = prefill
        self.runner = PluginRunner()
        self._build()

    def _build(self):
        c = self.cfg
        h, dim, n, g = c.nheads, c.head_dim, c.dstate, c.ngroups
        mb, ms = c.max_batch, c.max_seq
        F16, F32, I32 = trt.float16, trt.float32, trt.int32
        if self.prefill:
            x = ("x", F16, (-1, -1, h, dim))
            B = ("B", F16, (-1, -1, g, n))
            C = ("C", F16, (-1, -1, g, n))
            dt = ("dt", F16, (-1, -1, h))
            prof = {
                "x": ((1, 1, h, dim), (1, 16, h, dim), (mb, ms, h, dim)),
                "B": ((1, 1, g, n), (1, 16, g, n), (mb, ms, g, n)),
                "C": ((1, 1, g, n), (1, 16, g, n), (mb, ms, g, n)),
                "dt": ((1, 1, h), (1, 16, h), (mb, ms, h)),
            }
        else:
            x = ("x", F16, (-1, h, dim))
            B = ("B", F16, (-1, g, n))
            C = ("C", F16, (-1, g, n))
            dt = ("dt", F16, (-1, h))
            prof = {
                "x": ((1, h, dim), (1, h, dim), (mb, h, dim)),
                "B": ((1, g, n), (1, g, n), (mb, g, n)),
                "C": ((1, g, n), (1, g, n), (mb, g, n)),
                "dt": ((1, h), (1, h), (mb, h)),
            }
        input_specs = [
            x,
            ("A", F32, (h, )),
            B,
            C,
            ("D", F16, (h, )),
            dt,
            ("dt_bias", F16, (h, )),
            ("state", F16, (-1, h, dim, n)),
            ("context_lengths", I32, (-1, )),
            ("state_start_index", I32, (-1, )),
        ]
        prof.update({
            "A": ((h, ), (h, ), (h, )),
            "D": ((h, ), (h, ), (h, )),
            "dt_bias": ((h, ), (h, ), (h, )),
            "state": ((1, h, dim, n), (1, h, dim, n), (mb, h, dim, n)),
            "context_lengths": ((1, ), (1, ), (mb, )),
            "state_start_index": ((0, ), (1, ), (mb, )),
        })
        self.runner.build(
            input_specs=input_specs,
            output_names=["output", "state_out"],
            plugin_name=PLUGIN_NAME,
            plugin_version=PLUGIN_VERSION,
            plugin_fields=[
                pf_int32("dim", dim),
                pf_int32("dstate", n),
                pf_int32("nheads", h),
                pf_int32("ngroups", g),
                pf_int32("dt_softplus", int(c.dt_softplus)),
            ],
            profiles=prof,
        )

    def run(self,
            x,
            A,
            B,
            C,
            D,
            dt,
            dt_bias,
            state,
            context_lengths,
            state_start_index=None,
            synchronize=True):
        out = torch.empty_like(x)
        state_out = torch.empty_like(state)
        input_shapes = None
        if state_start_index is None or state_start_index.numel() == 0:
            # TensorRT requires a non-null address even when the runtime input shape is the cold-prefill [0] sentinel.
            state_start_index = torch.empty(1, dtype=torch.int32, device=DEV)
            input_shapes = {"state_start_index": (0, )}
        bindings = {
            "x": x,
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "dt": dt,
            "dt_bias": dt_bias,
            "state": state,
            "context_lengths": context_lengths,
            "state_start_index": state_start_index,
            "output": out,
            "state_out": state_out,
        }
        self.runner.execute(bindings,
                            input_shapes=input_shapes,
                            synchronize=synchronize)
        return out, state_out


def _rand_inputs(cfg: MambaConfig, b, s, gen, seq_dim=True):
    """Stable random Mamba inputs (A<0 so the recurrence decays)."""
    h, dim, n, g = cfg.nheads, cfg.head_dim, cfg.dstate, cfg.ngroups

    def rn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(DEV)

    if seq_dim:
        x = rn(b, s, h, dim).to(torch.float16)
        B = (rn(b, s, g, n) * 0.5).to(torch.float16)
        C = (rn(b, s, g, n) * 0.5).to(torch.float16)
        dt = (rn(b, s, h) * 0.1).to(torch.float16)
    else:
        x = rn(b, h, dim).to(torch.float16)
        B = (rn(b, g, n) * 0.5).to(torch.float16)
        C = (rn(b, g, n) * 0.5).to(torch.float16)
        dt = (rn(b, h) * 0.1).to(torch.float16)
    A = -torch.exp(rn(h))  # [h] fp32, negative
    D = rn(h).to(torch.float16)
    dt_bias = (rn(h) * 0.1).to(torch.float16)
    return x, A, B, C, D, dt, dt_bias


def _check(cfg, out, state_out, ref_y, ref_state, context_lengths, atol, rtol):
    """Compare final state (primary) and per-row valid outputs."""
    assert_close("state_out", ref_state, state_out, atol, rtol)
    if out.dim() == 4:  # prefill: compare only valid tokens per row
        for bi in range(out.shape[0]):
            L = int(context_lengths[bi])
            assert_close(f"y[b{bi}]", ref_y[bi, :L], out[bi, :L], atol, rtol)
    else:
        assert_close("y", ref_y, out, atol, rtol)


# --------------------------------------------------------------------------- #
# Decode (seq_len=1): state carry-over from a non-zero initial state
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [1, 2, 4], ids=lambda b: f"bs{b}")
@pytest.mark.parametrize("dt_softplus", [True, False],
                         ids=["softplus", "nosoftplus"])
def test_decode(batch, dt_softplus):
    cfg = MambaConfig(dt_softplus=dt_softplus)
    gen = torch.Generator().manual_seed(100 + batch + int(dt_softplus))
    r = MambaRunner(cfg, prefill=False)
    x, A, B, C, D, dt, dt_bias = _rand_inputs(cfg,
                                              batch,
                                              1,
                                              gen,
                                              seq_dim=False)
    state0 = (torch.randn(batch,
                          cfg.nheads,
                          cfg.head_dim,
                          cfg.dstate,
                          generator=gen,
                          dtype=torch.float32) * 0.3).to(torch.float16).to(DEV)
    ctx = torch.ones(batch, dtype=torch.int32, device=DEV)
    out, state_out = r.run(x, A, B, C, D, dt, dt_bias, state0.clone(), ctx)
    ref_y, ref_state = selective_scan_ref(x, A, B, C, dt, dt_bias, D, state0,
                                          cfg.ngroups, dt_softplus)
    _check(cfg, out, state_out, ref_y, ref_state, ctx, 2e-2, 2e-2)


# --------------------------------------------------------------------------- #
# Single-step-loop prefill (2 <= seq_len < 128), fresh (zero) state
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [1, 2], ids=lambda b: f"bs{b}")
def test_prefill_singlestep(batch):
    cfg = MambaConfig()
    seq = 16
    gen = torch.Generator().manual_seed(200 + batch)
    r = MambaRunner(cfg, prefill=True)
    x, A, B, C, D, dt, dt_bias = _rand_inputs(cfg, batch, seq, gen)
    state0 = torch.zeros(batch,
                         cfg.nheads,
                         cfg.head_dim,
                         cfg.dstate,
                         dtype=torch.float16,
                         device=DEV)
    ctx = torch.full((batch, ), seq, dtype=torch.int32, device=DEV)
    out, state_out = r.run(x, A, B, C, D, dt, dt_bias, state0.clone(), ctx)
    ref_y, ref_state = selective_scan_ref(x, A, B, C, dt, dt_bias, D, state0,
                                          cfg.ngroups, True, ctx)
    _check(cfg, out, state_out, ref_y, ref_state, ctx, 2e-2, 2e-2)


# --------------------------------------------------------------------------- #
# RAGGED single-step prefill: variable per-row lengths + poisoned padding
# --------------------------------------------------------------------------- #
def test_ragged_prefill_singlestep():
    cfg = MambaConfig()
    seq = 24
    batch = 3
    gen = torch.Generator().manual_seed(303)
    r = MambaRunner(cfg, prefill=True)
    x, A, B, C, D, dt, dt_bias = _rand_inputs(cfg, batch, seq, gen)
    ctx = torch.tensor([24, 9, 1], dtype=torch.int32, device=DEV)
    poison_padding([x, B, C, dt], ctx)
    state0 = torch.zeros(batch,
                         cfg.nheads,
                         cfg.head_dim,
                         cfg.dstate,
                         dtype=torch.float16,
                         device=DEV)
    out, state_out = r.run(x, A, B, C, D, dt, dt_bias, state0.clone(), ctx)
    ref_y, ref_state = selective_scan_ref(x, A, B, C, dt, dt_bias, D, state0,
                                          cfg.ngroups, True, ctx)
    _check(cfg, out, state_out, ref_y, ref_state, ctx, 2e-2, 2e-2)


# --------------------------------------------------------------------------- #
# CuTeDSL SSD chunk-scan prefill (seq_len >= 128), fresh state
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seq", [128, 200, 256], ids=lambda s: f"seq{s}")
@pytest.mark.parametrize("batch", [1, 2], ids=lambda b: f"bs{b}")
def test_prefill_ssd(seq, batch):
    cfg = MambaConfig()
    gen = torch.Generator().manual_seed(400 + seq + batch)
    r = MambaRunner(cfg, prefill=True)
    x, A, B, C, D, dt, dt_bias = _rand_inputs(cfg, batch, seq, gen)
    state0 = torch.zeros(batch,
                         cfg.nheads,
                         cfg.head_dim,
                         cfg.dstate,
                         dtype=torch.float16,
                         device=DEV)
    ctx = torch.full((batch, ), seq, dtype=torch.int32, device=DEV)
    out, state_out = r.run(x, A, B, C, D, dt, dt_bias, state0.clone(), ctx)
    ref_y, ref_state = selective_scan_ref(x, A, B, C, dt, dt_bias, D, state0,
                                          cfg.ngroups, True, ctx)
    # SSD path accumulates over a long sequence in fp16 I/O -> looser tol.
    _check(cfg, out, state_out, ref_y, ref_state, ctx, 5e-2, 5e-2)


@pytest.mark.parametrize("restored", [False, True], ids=["cold", "restored"])
def test_prefill_ssd_state_start_index_contract(restored):
    cfg = MambaConfig(max_batch=1, max_seq=128)
    seq = 128
    batch = 1
    gen = torch.Generator().manual_seed(900 + int(restored))
    r = MambaRunner(cfg, prefill=True)
    x, A, B, C, D, dt, dt_bias = _rand_inputs(cfg, batch, seq, gen)
    if restored:
        state0 = (torch.randn(batch,
                              cfg.nheads,
                              cfg.head_dim,
                              cfg.dstate,
                              generator=gen,
                              device=DEV) * 0.1).to(torch.float16)
        state_start_index = torch.zeros(batch, dtype=torch.int32, device=DEV)
    else:
        state0 = torch.zeros(batch,
                             cfg.nheads,
                             cfg.head_dim,
                             cfg.dstate,
                             dtype=torch.float16,
                             device=DEV)
        state_start_index = torch.empty(0, dtype=torch.int32, device=DEV)
    ctx = torch.full((batch, ), seq, dtype=torch.int32, device=DEV)
    out, state_out = r.run(x, A, B, C, D, dt, dt_bias, state0.clone(), ctx,
                           state_start_index)
    ref_y, ref_state = selective_scan_ref(x, A, B, C, dt, dt_bias, D, state0,
                                          cfg.ngroups, True, ctx)
    _check(cfg, out, state_out, ref_y, ref_state, ctx, 5e-2, 5e-2)


# --------------------------------------------------------------------------- #
# RAGGED SSD prefill: partial final chunk (len % 128 != 0) + poisoned padding.
# This is the Nemotron MMLU prefill shape (a partial final chunk stresses the
# TMA bounds handling).
# --------------------------------------------------------------------------- #
def test_ragged_prefill_ssd():
    cfg = MambaConfig()
    seq = 256
    batch = 2
    gen = torch.Generator().manual_seed(505)
    r = MambaRunner(cfg, prefill=True)
    x, A, B, C, D, dt, dt_bias = _rand_inputs(cfg, batch, seq, gen)
    # row0 full 2 chunks; row1 partial final chunk (130 = 128 + 2) + padding.
    ctx = torch.tensor([256, 130], dtype=torch.int32, device=DEV)
    poison_padding([x, B, C, dt], ctx)
    state0 = torch.zeros(batch,
                         cfg.nheads,
                         cfg.head_dim,
                         cfg.dstate,
                         dtype=torch.float16,
                         device=DEV)
    out, state_out = r.run(x, A, B, C, D, dt, dt_bias, state0.clone(), ctx)
    ref_y, ref_state = selective_scan_ref(x, A, B, C, dt, dt_bias, D, state0,
                                          cfg.ngroups, True, ctx)
    _check(cfg, out, state_out, ref_y, ref_state, ctx, 5e-2, 5e-2)


def _ssd_prefill_check(cfg, seq, batch, seed, atol=5e-2, rtol=5e-2):
    """Fresh-state SSD chunk-scan prefill, plugin vs reference (used by the
    dim/dstate and nheads/ngroups sweeps below)."""
    gen = torch.Generator().manual_seed(seed)
    r = MambaRunner(cfg, prefill=True)
    x, A, B, C, D, dt, dt_bias = _rand_inputs(cfg, batch, seq, gen)
    state0 = torch.zeros(batch,
                         cfg.nheads,
                         cfg.head_dim,
                         cfg.dstate,
                         dtype=torch.float16,
                         device=DEV)
    ctx = torch.full((batch, ), seq, dtype=torch.int32, device=DEV)
    out, state_out = r.run(x, A, B, C, D, dt, dt_bias, state0.clone(), ctx)
    ref_y, ref_state = selective_scan_ref(x, A, B, C, dt, dt_bias, D, state0,
                                          cfg.ngroups, True, ctx)
    _check(cfg, out, state_out, ref_y, ref_state, ctx, atol, rtol)


# --------------------------------------------------------------------------- #
# dim x dstate sweep: the SSD kernel canImplement() supports dim in {64,128}
# and dstate in {64,128} -- cover all four combinations (seq=256 -> 2 chunks).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [64, 128], ids=lambda d: f"dim{d}")
@pytest.mark.parametrize("dstate", [64, 128], ids=lambda d: f"dstate{d}")
def test_ssd_dim_sweep(dim, dstate):
    cfg = MambaConfig(nheads=8,
                      head_dim=dim,
                      dstate=dstate,
                      ngroups=2,
                      max_batch=2,
                      max_seq=256)
    _ssd_prefill_check(cfg, seq=256, batch=2, seed=1700 + dim + dstate)


# --------------------------------------------------------------------------- #
# nheads x ngroups sweep: ngroups must divide nheads (head->group mapping).
# Covers single group, per-head groups, and intermediate sharing.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nheads,ngroups", [(8, 1), (8, 4), (8, 8), (16, 2)],
                         ids=[
                             "nheads8_ngroups1", "nheads8_ngroups4",
                             "nheads8_ngroups8", "nheads16_ngroups2"
                         ])
def test_ssd_group_sweep(nheads, ngroups):
    cfg = MambaConfig(nheads=nheads,
                      head_dim=64,
                      dstate=128,
                      ngroups=ngroups,
                      max_batch=2,
                      max_seq=256)
    _ssd_prefill_check(cfg, seq=256, batch=2, seed=1800 + nheads + ngroups)


def _d80_cfg(max_seq):
    return MambaConfig(nheads=8,
                       head_dim=80,
                       dstate=128,
                       ngroups=1,
                       max_batch=2,
                       max_seq=max_seq)


def _nemotron_d80_cfg(max_seq):
    return MambaConfig(nheads=96,
                       head_dim=80,
                       dstate=128,
                       ngroups=8,
                       max_batch=1,
                       max_seq=max_seq)


@REQUIRES_D80_BLACKWELL
def test_d80_prefill_routing_boundary():
    cfg = _d80_cfg(max_seq=129)
    runner = MambaRunner(cfg, prefill=True)
    for seq in (127, 128, 129):
        gen = torch.Generator().manual_seed(8000 + seq)
        x, A, B, C, D, dt, dt_bias = _rand_inputs(cfg, 1, seq, gen)
        state0 = torch.zeros(1,
                             cfg.nheads,
                             cfg.head_dim,
                             cfg.dstate,
                             dtype=torch.float16,
                             device=DEV)
        ctx = torch.full((1, ), seq, dtype=torch.int32, device=DEV)
        out, state_out = runner.run(x, A, B, C, D, dt, dt_bias, state0.clone(),
                                    ctx)
        ref_y, ref_state = selective_scan_ref(x, A, B, C, dt, dt_bias, D,
                                              state0, cfg.ngroups, True, ctx)
        _check(cfg, out, state_out, ref_y, ref_state, ctx, 6e-2, 6e-2)


@REQUIRES_D80_BLACKWELL
def test_nemotron_d80_prefill_decode_handoff():
    prefill_len = 128
    cfg = _nemotron_d80_cfg(max_seq=prefill_len)
    gen = torch.Generator().manual_seed(8100)
    x, A, B, C, D, dt, dt_bias = _rand_inputs(cfg, 1, prefill_len + 1, gen)
    state0 = torch.zeros(1,
                         cfg.nheads,
                         cfg.head_dim,
                         cfg.dstate,
                         dtype=torch.float16,
                         device=DEV)
    ref_y, ref_state = selective_scan_ref(x, A, B, C, dt, dt_bias, D, state0,
                                          cfg.ngroups, True)

    prefill = MambaRunner(cfg, prefill=True)
    decode = MambaRunner(cfg, prefill=False)
    prefill_ctx = torch.full((1, ), prefill_len, dtype=torch.int32, device=DEV)
    prefill_out, state = prefill.run(x[:, :prefill_len].contiguous(), A,
                                     B[:, :prefill_len].contiguous(),
                                     C[:, :prefill_len].contiguous(), D,
                                     dt[:, :prefill_len].contiguous(), dt_bias,
                                     state0.clone(), prefill_ctx)
    assert_close("d80-prefill-output", ref_y[:, :prefill_len], prefill_out,
                 6e-2, 6e-2)

    decode_ctx = torch.ones(1, dtype=torch.int32, device=DEV)
    decode_out, state = decode.run(x[:, prefill_len].contiguous(), A,
                                   B[:, prefill_len].contiguous(),
                                   C[:, prefill_len].contiguous(), D,
                                   dt[:, prefill_len].contiguous(), dt_bias,
                                   state, decode_ctx)
    assert_close("d80-decode-output", ref_y[:, prefill_len], decode_out, 6e-2,
                 6e-2)
    assert_close("d80-handoff-state", ref_state, state, 6e-2, 6e-2)


@REQUIRES_D80_BLACKWELL
def test_d80_prefill_cuda_graph():
    seq = 128
    cfg = _d80_cfg(max_seq=seq)
    gen = torch.Generator().manual_seed(8200)
    x, A, B, C, D, dt, dt_bias = _rand_inputs(cfg, 1, seq, gen)
    state0 = torch.zeros(1,
                         cfg.nheads,
                         cfg.head_dim,
                         cfg.dstate,
                         dtype=torch.float16,
                         device=DEV)
    ctx = torch.full((1, ), seq, dtype=torch.int32, device=DEV)
    runner = MambaRunner(cfg, prefill=True)

    runner.run(x, A, B, C, D, dt, dt_bias, state0.clone(), ctx)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_state = runner.run(x,
                                            A,
                                            B,
                                            C,
                                            D,
                                            dt,
                                            dt_bias,
                                            state0.clone(),
                                            ctx,
                                            synchronize=False)
    graph.replay()
    torch.cuda.synchronize()

    ref_y, ref_state = selective_scan_ref(x, A, B, C, dt, dt_bias, D, state0,
                                          cfg.ngroups, True, ctx)
    _check(cfg, graph_out, graph_state, ref_y, ref_state, ctx, 6e-2, 6e-2)


# Exact NemotronH Nano mamba config (from the model config.json): 64 heads,
# head_dim 64, ssm_state 128, n_groups 8. n_groups=8 / 64 heads differ from the
# small default config above.
def _nemotron_cfg(max_seq):
    return MambaConfig(nheads=64,
                       head_dim=64,
                       dstate=128,
                       ngroups=8,
                       max_batch=2,
                       max_seq=max_seq)


# --------------------------------------------------------------------------- #
# (a) Exact-Nemotron config + long multi-chunk SSD prefill (seq=512 -> 4 chunks)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [1, 2], ids=lambda b: f"bs{b}")
def test_nemotron_prefill_multichunk(batch):
    cfg = _nemotron_cfg(max_seq=512)
    seq = 512  # 4 SSD chunks
    gen = torch.Generator().manual_seed(600 + batch)
    r = MambaRunner(cfg, prefill=True)
    x, A, B, C, D, dt, dt_bias = _rand_inputs(cfg, batch, seq, gen)
    state0 = torch.zeros(batch,
                         cfg.nheads,
                         cfg.head_dim,
                         cfg.dstate,
                         dtype=torch.float16,
                         device=DEV)
    ctx = torch.full((batch, ), seq, dtype=torch.int32, device=DEV)
    out, state_out = r.run(x, A, B, C, D, dt, dt_bias, state0.clone(), ctx)
    ref_y, ref_state = selective_scan_ref(x, A, B, C, dt, dt_bias, D, state0,
                                          cfg.ngroups, True, ctx)
    _check(cfg, out, state_out, ref_y, ref_state, ctx, 6e-2, 6e-2)


def test_nemotron_ragged_prefill_multichunk():
    cfg = _nemotron_cfg(max_seq=512)
    seq = 512
    batch = 2
    gen = torch.Generator().manual_seed(611)
    r = MambaRunner(cfg, prefill=True)
    x, A, B, C, D, dt, dt_bias = _rand_inputs(cfg, batch, seq, gen)
    # row0 full 4 chunks; row1 partial final chunk (387 = 3*128 + 3) + padding.
    ctx = torch.tensor([512, 387], dtype=torch.int32, device=DEV)
    poison_padding([x, B, C, dt], ctx)
    state0 = torch.zeros(batch,
                         cfg.nheads,
                         cfg.head_dim,
                         cfg.dstate,
                         dtype=torch.float16,
                         device=DEV)
    out, state_out = r.run(x, A, B, C, D, dt, dt_bias, state0.clone(), ctx)
    ref_y, ref_state = selective_scan_ref(x, A, B, C, dt, dt_bias, D, state0,
                                          cfg.ngroups, True, ctx)
    _check(cfg, out, state_out, ref_y, ref_state, ctx, 6e-2, 6e-2)


# --------------------------------------------------------------------------- #
# (b) prefill -> decode state handoff (real autoregressive contract): prefill L
# tokens from a fresh state, then feed the captured ssm_state into N decode
# steps. Compare against one continuous reference scan over L+N tokens. Catches
# prefill-capture / decode-consume state-layout mismatches that single-phase
# tests miss.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2], ids=lambda s: f"seed{s}")
def test_prefill_decode_handoff(seed):
    # Prefill ISL randomly chosen in [10, 2048] (seeded for reproducibility).
    prefill_len = random.Random(5000 + seed).randint(10, 2048)
    n_decode = 4
    batch = 2
    cfg = _nemotron_cfg(max_seq=prefill_len + n_decode + 8)
    total = prefill_len + n_decode
    gen = torch.Generator().manual_seed(640 + prefill_len)
    x, A, B, C, D, dt, dt_bias = _rand_inputs(cfg, batch, total, gen)

    # Continuous reference scan over the whole L+N sequence (fresh state).
    state0 = torch.zeros(batch,
                         cfg.nheads,
                         cfg.head_dim,
                         cfg.dstate,
                         dtype=torch.float16,
                         device=DEV)
    ref_y, ref_state = selective_scan_ref(x, A, B, C, dt, dt_bias, D, state0,
                                          cfg.ngroups, True)

    # Plugin: prefill the first L tokens, then decode the rest one at a time.
    pre = MambaRunner(cfg, prefill=True)
    dec = MambaRunner(cfg, prefill=False)
    ctx_pre = torch.full((batch, ), prefill_len, dtype=torch.int32, device=DEV)
    # .contiguous(): seq-dim slices are non-contiguous views; the plugin reads
    # via data_ptr() assuming a contiguous layout.
    _, state = pre.run(x[:, :prefill_len].contiguous(), A,
                       B[:, :prefill_len].contiguous(),
                       C[:, :prefill_len].contiguous(),
                       D, dt[:, :prefill_len].contiguous(), dt_bias,
                       state0.clone(), ctx_pre)
    ctx_dec = torch.ones(batch, dtype=torch.int32, device=DEV)
    for i in range(n_decode):
        t = prefill_len + i
        o, state = dec.run(x[:, t].contiguous(), A, B[:, t].contiguous(),
                           C[:, t].contiguous(), D, dt[:, t].contiguous(),
                           dt_bias, state, ctx_dec)
        assert_close(f"decode-out[t={t}]", ref_y[:, t], o, 6e-2, 6e-2)
    # Final recurrent state after the full L+N sequence must match.
    assert_close("handoff-final-state", ref_state, state, 6e-2, 6e-2)


# --------------------------------------------------------------------------- #
# Required even/uneven batch cases (bs 1/2/3/4/8, seq up to 2048) -- SSD path.
# Each row's context_length is independent; padding is poisoned.
# --------------------------------------------------------------------------- #
def _ragged_cfg():
    return MambaConfig(nheads=8,
                       head_dim=64,
                       dstate=128,
                       ngroups=2,
                       max_batch=8,
                       max_seq=2048)


@pytest.mark.parametrize("label,seqlens", RAGGED_CASES)
def test_ragged_prefill_batch_sizes(label, seqlens):
    cfg = _ragged_cfg()
    bs, maxlen = len(seqlens), max(seqlens)
    gen = torch.Generator().manual_seed(1000 + maxlen + bs)
    x, A, B, C, D, dt, dt_bias = _rand_inputs(cfg, bs, maxlen, gen)
    ctx = torch.tensor(seqlens, dtype=torch.int32, device=DEV)
    poison_padding([x, B, C, dt], ctx)
    state0 = torch.zeros(bs,
                         cfg.nheads,
                         cfg.head_dim,
                         cfg.dstate,
                         dtype=torch.float16,
                         device=DEV)
    r = MambaRunner(cfg, prefill=True)
    out, state_out = r.run(x, A, B, C, D, dt, dt_bias, state0.clone(), ctx)
    ref_y, ref_state = selective_scan_ref(x, A, B, C, dt, dt_bias, D, state0,
                                          cfg.ngroups, True, ctx)
    _check(cfg, out, state_out, ref_y, ref_state, ctx, 6e-2, 6e-2)


# --------------------------------------------------------------------------- #
# Batch invariance (plugin-vs-plugin): permuting the batch rows must permute
# the outputs and final states identically (uneven case).
# --------------------------------------------------------------------------- #
def test_batch_invariance():
    cfg = _ragged_cfg()
    seqlens = [10, 2048, 128]
    bs, maxlen = len(seqlens), max(seqlens)
    gen = torch.Generator().manual_seed(1234)
    x, A, B, C, D, dt, dt_bias = _rand_inputs(cfg, bs, maxlen, gen)
    ctx = torch.tensor(seqlens, dtype=torch.int32, device=DEV)
    poison_padding([x, B, C, dt], ctx)
    state0 = torch.zeros(bs,
                         cfg.nheads,
                         cfg.head_dim,
                         cfg.dstate,
                         dtype=torch.float16,
                         device=DEV)
    r = MambaRunner(cfg, prefill=True)
    out0, st0 = r.run(x, A, B, C, D, dt, dt_bias, state0.clone(), ctx)

    perm = torch.tensor([2, 0, 1], device=DEV)
    out1, st1 = r.run(x[perm].contiguous(), A, B[perm].contiguous(),
                      C[perm].contiguous(), D, dt[perm].contiguous(), dt_bias,
                      state0.clone(), ctx[perm].contiguous())
    # Compare per-row valid regions of the permuted run vs the original.
    pcpu = perm.cpu()
    for new_i in range(bs):
        orig = int(pcpu[new_i])
        L = seqlens[orig]
        assert_close(f"batch-inv-y[{new_i}]", out0[orig, :L], out1[new_i, :L])
    assert_close("batch-inv-state", st0[pcpu], st1)
