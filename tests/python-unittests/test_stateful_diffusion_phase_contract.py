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

import pytest
from test_gated_delta_net_plugin import CUTEDSL_GDN_SMS, GDNConfig
from test_gated_delta_net_plugin import _rand as gdn_rand
from test_gated_delta_net_plugin import gdn_ref
from test_mamba_plugin import MambaConfig
from test_mamba_plugin import _rand_inputs as mamba_rand
from test_mamba_plugin import selective_scan_ref
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              PluginRunner, _device_sm, assert_close, pf_int32,
                              poison_padding)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

pytestmark = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE,
    reason=f"TensorRT/torch CUDA not available: {IMPORT_ERROR}")

DEV = "cuda"


class TokenMajorMambaRunner:

    def __init__(self, cfg, max_state_rows=4):
        self.cfg = cfg
        self.max_state_rows = max_state_rows
        self.runner = PluginRunner()
        self._build()

    def _build(self):
        c = self.cfg
        h, dim, n, g = c.nheads, c.head_dim, c.dstate, c.ngroups
        max_tokens = c.max_batch * c.max_seq
        f16, f32, i32 = trt.float16, trt.float32, trt.int32
        self.runner.build(
            input_specs=[
                ("x", f16, (-1, h, dim)),
                ("A", f32, (h, )),
                ("B", f16, (-1, g, n)),
                ("C", f16, (-1, g, n)),
                ("D", f16, (h, )),
                ("dt", f16, (-1, h)),
                ("dt_bias", f16, (h, )),
                ("state", f16, (-1, h, dim, n)),
                ("query_lengths", i32, (-1, )),
                ("query_start_offsets", i32, (-1, )),
                ("state_indices", i32, (-1, )),
                ("execution_phase_marker", i32, (-1, )),
                ("context_sequence_count_carrier", i32, (-1, )),
            ],
            output_names=["output", "state_out"],
            plugin_name="update_ssm_state",
            plugin_version="1",
            plugin_fields=[
                pf_int32("dim", dim),
                pf_int32("dstate", n),
                pf_int32("nheads", h),
                pf_int32("ngroups", g),
                pf_int32("dt_softplus", int(c.dt_softplus)),
            ],
            profiles={
                "x": ((1, h, dim), (16, h, dim), (max_tokens, h, dim)),
                "A": ((h, ), (h, ), (h, )),
                "B": ((1, g, n), (16, g, n), (max_tokens, g, n)),
                "C": ((1, g, n), (16, g, n), (max_tokens, g, n)),
                "D": ((h, ), (h, ), (h, )),
                "dt": ((1, h), (16, h), (max_tokens, h)),
                "dt_bias": ((h, ), (h, ), (h, )),
                "state": ((1, h, dim, n), (c.max_batch, h, dim, n),
                          (self.max_state_rows, h, dim, n)),
                "query_lengths": ((1, ), (c.max_batch, ), (c.max_batch, )),
                "query_start_offsets":
                ((2, ), (c.max_batch + 1, ), (c.max_batch + 1, )),
                "state_indices": ((1, ), (c.max_batch, ), (c.max_batch, )),
                "execution_phase_marker": ((1, ), (6, ), (8, )),
                "context_sequence_count_carrier":
                ((0, ), (0, ), (c.max_batch, )),
            })

    def run(self, tensors, state, lengths, state_indices, phase):
        x, A, B, C, D, dt, dt_bias = tensors
        batch, seq = x.shape[:2]
        offsets = torch.arange(batch + 1, dtype=torch.int32, device=DEV) * seq
        output = torch.empty_like(x).reshape(batch * seq, *x.shape[2:])
        context_sequences = batch if phase in (1, 2) else 0
        bindings = {
            "x":
            x.reshape(batch * seq, *x.shape[2:]),
            "A":
            A,
            "B":
            B.reshape(batch * seq, *B.shape[2:]),
            "C":
            C.reshape(batch * seq, *C.shape[2:]),
            "D":
            D,
            "dt":
            dt.reshape(batch * seq, *dt.shape[2:]),
            "dt_bias":
            dt_bias,
            "state":
            state,
            "query_lengths":
            lengths,
            "query_start_offsets":
            offsets,
            "state_indices":
            state_indices,
            "execution_phase_marker":
            torch.empty(phase, dtype=torch.int32, device=DEV),
            "context_sequence_count_carrier":
            torch.empty(max(1, context_sequences),
                        dtype=torch.int32,
                        device=DEV),
            "output":
            output,
            "state_out":
            state,
        }
        input_shapes = ({
            "context_sequence_count_carrier": (0, )
        } if context_sequences == 0 else None)
        self.runner.execute(bindings, input_shapes=input_shapes)
        return output.reshape_as(x)


@pytest.mark.parametrize(("phase", "seq", "commits"), [(2, 5, True),
                                                       (6, 128, False),
                                                       (7, 5, True)])
def test_mamba_diffusion_phase_transaction(phase, seq, commits):
    cfg = MambaConfig(nheads=2,
                      head_dim=64,
                      dstate=64,
                      ngroups=1,
                      max_batch=2,
                      max_seq=128)
    gen = torch.Generator().manual_seed(4100 + phase)
    runner = TokenMajorMambaRunner(cfg)
    tensors = list(mamba_rand(cfg, 2, seq, gen))
    lengths = torch.tensor([seq, seq - 2], dtype=torch.int32, device=DEV)
    for tensor in (tensors[0], tensors[2], tensors[3], tensors[5]):
        poison_padding(tensor, lengths)
    state = (torch.randn(4,
                         cfg.nheads,
                         cfg.head_dim,
                         cfg.dstate,
                         generator=gen,
                         dtype=torch.float32) * 0.05).to(torch.float16).to(DEV)
    before = state.clone()
    indices = torch.tensor([3, 1], dtype=torch.int32, device=DEV)
    selected = before.index_select(0, indices.to(torch.int64))
    output = runner.run(tensors, state, lengths, indices, phase)
    x, A, B, C, D, dt, dt_bias = tensors
    ref_output, ref_state = selective_scan_ref(x, A, B, C, dt, dt_bias, D,
                                               selected, cfg.ngroups,
                                               cfg.dt_softplus, lengths)
    for batch_idx, length in enumerate(lengths.tolist()):
        assert_close(f"mamba phase{phase} output[{batch_idx}]",
                     ref_output[batch_idx, :length],
                     output[batch_idx, :length], 3e-2, 3e-2)
    if commits:
        assert_close(f"mamba phase{phase} state[3]", ref_state[0], state[3],
                     3e-2, 3e-2)
        assert_close(f"mamba phase{phase} state[1]", ref_state[1], state[1],
                     3e-2, 3e-2)
        assert torch.equal(state[0], before[0])
        assert torch.equal(state[2], before[2])
    else:
        assert torch.equal(state, before)


class TokenMajorGDNRunner:

    def __init__(self, cfg, max_state_rows=4, use_diffusion_state=True):
        self.cfg = cfg
        self.max_state_rows = max_state_rows
        self.use_diffusion_state = use_diffusion_state
        self.runner = PluginRunner()
        self._build()

    def _build(self):
        c = self.cfg
        h, kd, vd = c.heads, c.k_dim, c.v_dim
        max_tokens = c.max_batch * c.max_seq
        opt_tokens = min(16, max_tokens)
        f16, f32, i32 = trt.float16, trt.float32, trt.int32
        token_hk = ((1, h, kd), (opt_tokens, h, kd), (max_tokens, h, kd))
        token_hv = ((1, h, vd), (opt_tokens, h, vd), (max_tokens, h, vd))
        token_h = ((1, h), (opt_tokens, h), (max_tokens, h))
        self.runner.build(
            input_specs=[
                ("q", f16, (-1, h, kd)),
                ("k", f16, (-1, h, kd)),
                ("v", f16, (-1, h, vd)),
                ("a", f16, (-1, h)),
                ("b", f16, (-1, h)),
                ("A_log", f32, (h, )),
                ("dt_bias", f16, (h, )),
                ("h0", f32, (-1, h, kd, vd)),
                ("query_lengths", i32, (-1, )),
                ("query_start_offsets", i32, (-1, )),
                ("state_indices", i32, (-1, )),
                ("execution_phase_marker", i32, (-1, )),
                ("context_sequence_count_carrier", i32, (-1, )),
            ],
            output_names=["o", "h0_out"],
            plugin_name="gated_delta_net",
            plugin_version="1",
            plugin_fields=[
                pf_int32("k_dim", kd),
                pf_int32("v_dim", vd),
                pf_int32("use_mtp", 0),
                pf_int32("use_diffusion_state", self.use_diffusion_state)
            ],
            profiles={
                "q":
                token_hk,
                "k":
                token_hk,
                "v":
                token_hv,
                "a":
                token_h,
                "b":
                token_h,
                "A_log": ((h, ), (h, ), (h, )),
                "dt_bias": ((h, ), (h, ), (h, )),
                "h0": ((1, h, kd, vd), (c.max_batch, h, kd, vd),
                       (self.max_state_rows, h, kd, vd)),
                "query_lengths": ((1, ), (c.max_batch, ), (c.max_batch, )),
                "query_start_offsets":
                ((2, ), (c.max_batch + 1, ), (c.max_batch + 1, )),
                "state_indices": ((1, ), (c.max_batch, ), (c.max_batch, )),
                "execution_phase_marker": ((1, ), (6, ), (8, )),
                "context_sequence_count_carrier":
                ((0, ), (0, ), (c.max_batch, )),
            })

    def run(self, tensors, state, lengths, state_indices, phase):
        q, k, v, a, b, A_log, dt_bias = tensors
        batch, seq = q.shape[:2]
        offsets = torch.arange(batch + 1, dtype=torch.int32, device=DEV) * seq
        output = torch.empty_like(v).reshape(batch * seq, *v.shape[2:])
        bindings = {
            "q":
            q.reshape(batch * seq, *q.shape[2:]),
            "k":
            k.reshape(batch * seq, *k.shape[2:]),
            "v":
            v.reshape(batch * seq, *v.shape[2:]),
            "a":
            a.reshape(batch * seq, *a.shape[2:]),
            "b":
            b.reshape(batch * seq, *b.shape[2:]),
            "A_log":
            A_log,
            "dt_bias":
            dt_bias,
            "h0":
            state,
            "query_lengths":
            lengths,
            "query_start_offsets":
            offsets,
            "state_indices":
            state_indices,
            "execution_phase_marker":
            torch.empty(phase, dtype=torch.int32, device=DEV),
            "context_sequence_count_carrier":
            torch.empty(max(1, batch if phase in (1, 2) else 0),
                        dtype=torch.int32,
                        device=DEV),
            "o":
            output,
            "h0_out":
            state,
        }
        input_shapes = ({
            "context_sequence_count_carrier": (0, )
        } if phase not in (1, 2) else None)
        self.runner.execute(bindings, input_shapes=input_shapes)
        return output.reshape_as(v)


@pytest.mark.skipif(_device_sm() not in CUTEDSL_GDN_SMS,
                    reason="GDN unsupported on this SM")
@pytest.mark.parametrize(("phase", "commits"), [(2, True), (6, False),
                                                (7, True)])
def test_gdn_diffusion_phase_transaction(phase, commits):
    cfg = GDNConfig(heads=2, max_batch=2, max_seq=5)
    gen = torch.Generator().manual_seed(5100 + phase)
    runner = TokenMajorGDNRunner(cfg)
    tensors = list(gdn_rand(cfg, 2, 5, gen))
    lengths = torch.tensor([5, 3], dtype=torch.int32, device=DEV)
    for tensor in tensors[:5]:
        poison_padding(tensor, lengths)
    state = (torch.randn(
        4, cfg.heads, cfg.k_dim, cfg.v_dim, generator=gen, dtype=torch.float32)
             * 0.02).to(DEV)
    before = state.clone()
    indices = torch.tensor([3, 1], dtype=torch.int32, device=DEV)
    selected = before.index_select(0, indices.to(torch.int64))
    output = runner.run(tensors, state, lengths, indices, phase)
    ref_output, ref_state = gdn_ref(*tensors, selected, lengths)
    for batch_idx, length in enumerate(lengths.tolist()):
        assert_close(f"gdn phase{phase} output[{batch_idx}]",
                     ref_output[batch_idx, :length],
                     output[batch_idx, :length], 8e-2, 8e-2)
    if commits:
        assert_close(f"gdn phase{phase} state[3]", ref_state[0], state[3],
                     8e-2, 8e-2)
        assert_close(f"gdn phase{phase} state[1]", ref_state[1], state[1],
                     8e-2, 8e-2)
        assert torch.equal(state[0], before[0])
        assert torch.equal(state[2], before[2])
    else:
        assert torch.equal(state, before)


@pytest.mark.skipif(_device_sm() not in CUTEDSL_GDN_SMS,
                    reason="GDN unsupported on this SM")
@pytest.mark.parametrize("phase", (6, 7))
def test_gdn_rejects_diffusion_phase_without_engine_capability(phase):
    cfg = GDNConfig(heads=2, max_batch=1, max_seq=2)
    runner = TokenMajorGDNRunner(cfg, use_diffusion_state=False)
    tensors = list(
        gdn_rand(cfg, 1, 2,
                 torch.Generator().manual_seed(6100 + phase)))
    state = torch.zeros(2,
                        cfg.heads,
                        cfg.k_dim,
                        cfg.v_dim,
                        dtype=torch.float32,
                        device=DEV)
    lengths = torch.tensor([2], dtype=torch.int32, device=DEV)
    indices = torch.tensor([1], dtype=torch.int32, device=DEV)

    with pytest.raises(RuntimeError):
        runner.run(tensors, state, lengths, indices, phase)
