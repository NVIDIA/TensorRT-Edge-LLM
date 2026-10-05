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
causal_conv1d plugin tests vs a PyTorch reference (causal_conv1d_ref).

Depthwise causal conv1d as used by the Nemotron-H mamba mixer (conv_kernel=4).
Centerpiece is ragged prefill (variable per-row lengths in one padded call,
as Nemotron hits during batched MMLU prefill): verifies per-row valid outputs
and the captured conv-state, and poisons the padding so any read past
query_lengths is caught. Decode is tested with a non-zero conv-state to
exercise state carry-over.

Run:
    python3 -m pytest tests/python-unittests/test_causal_conv1d_plugin.py -v
"""

from __future__ import annotations

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


def causal_conv1d_ref(
        x: torch.Tensor,  # [b, s, c]
        weight: torch.Tensor,  # [c, width]  (depthwise; width == conv kernel)
        bias: Optional[torch.Tensor],  # [c]
        conv_state0: torch.Tensor,  # [b, c, width]
        activation: bool,
        context_lengths: Optional[torch.Tensor] = None,  # [b]
):
    """Depthwise causal conv1d reference. Returns (y, final_conv_state).

    Matches the plugin's conventions (verified empirically): the conv-state
    buffer is ``width`` (== kernel) columns holding the most recent ``width``
    inputs (oldest..newest). Output token t uses the last ``width`` inputs
    ending at t; the relevant history for the first new token is the buffer's
    last ``width-1`` columns ``conv_state0[:, :, 1:]`` (zeros on a fresh state,
    giving causal left zero-pad). The captured final state is the last
    ``width`` inputs. For ragged rows only the first ``context_lengths[bi]``
    tokens are valid.
    """
    b, s, c = x.shape
    width = weight.shape[-1]
    xf = x.float().transpose(1, 2)  # [b, c, s]
    wf = weight.float()  # [c, width]
    state0 = conv_state0.float()  # [b, c, width]

    if context_lengths is None:
        context_lengths = torch.full((b, ), s, dtype=torch.int64)
    else:
        context_lengths = context_lengths.to(torch.int64)

    y = torch.zeros((b, c, s), dtype=torch.float32, device=x.device)
    final_state = state0.clone()
    for bi in range(b):
        L = int(context_lengths[bi])
        # Usable history = buffer's last (width-1) columns, then valid inputs.
        seq = torch.cat([state0[bi, :, 1:width], xf[bi, :, :L]], dim=-1)
        for t in range(L):
            window = seq[:, t:t + width]  # [c, width]
            out = (window * wf).sum(-1)
            if bias is not None:
                out = out + bias.float()
            y[bi, :, t] = out
        # Captured state = last `width` inputs (left zero-pad if seq shorter).
        if seq.shape[-1] >= width:
            final_state[bi] = seq[:, -width:]
        else:
            final_state[bi] = torch.nn.functional.pad(
                seq, (width - seq.shape[-1], 0))
    if activation:
        y = torch.nn.functional.silu(y)
    return y.transpose(1, 2), final_state  # y: [b, s, c]


@dataclass
class ConvConfig:
    dim: int = 256  # conv channels (depthwise)
    width: int = 4  # conv kernel (Nemotron conv_kernel=4)
    max_batch: int = 4
    max_seq: int = 64
    max_state_rows: int = 8


class ConvRunner:
    """Build and run the token-major ragged causal-conv ABI."""

    def __init__(self, cfg: ConvConfig, *, use_mtp=False, use_ddtree=False):
        self.cfg = cfg
        self.use_mtp = use_mtp
        self.use_ddtree = use_ddtree
        self.runner = PluginRunner()
        self._build()

    def _build(self):
        c = self.cfg
        dim, w, mb, ms = c.dim, c.width, c.max_batch, c.max_seq
        F16, I32 = trt.float16, trt.int32
        input_specs = [
            ("x", F16, (-1, dim)),
            ("weight", F16, (dim, 1, w)),
            ("bias", F16, (dim, )),
            ("conv_state", F16, (-1, dim, w)),
            ("query_lengths", I32, (-1, )),
            ("query_start_offsets", I32, (-1, )),
            ("state_indices", I32, (-1, )),
            ("execution_phase_marker", I32, (-1, )),
            ("context_sequence_count_carrier", I32, (-1, )),
        ]
        if self.use_ddtree:
            input_specs += [("tree_parent_ids", I32, (-1, )),
                            ("tree_depths", I32, (-1, ))]
        profiles = {
            "x": ((1, dim), (min(16, mb * ms), dim), (mb * ms, dim)),
            "weight": ((dim, 1, w), (dim, 1, w), (dim, 1, w)),
            "bias": ((dim, ), (dim, ), (dim, )),
            "conv_state":
            ((1, dim, w), (mb, dim, w), (c.max_state_rows, dim, w)),
            "query_lengths": ((1, ), (mb, ), (mb, )),
            "query_start_offsets": ((2, ), (mb + 1, ), (mb + 1, )),
            "state_indices": ((1, ), (mb, ), (mb, )),
            "execution_phase_marker": ((1, ), (1, ), (8, )),
            "context_sequence_count_carrier": ((0, ), (mb, ), (mb, )),
        }
        if self.use_ddtree:
            profiles["tree_parent_ids"] = ((1, ), (min(16, mb * ms), ),
                                           (mb * ms, ))
            profiles["tree_depths"] = profiles["tree_parent_ids"]
        outputs = ["output", "conv_state_out"]
        if self.use_mtp or self.use_ddtree:
            outputs.append("intermediate_states")
        self.runner.build(
            input_specs=input_specs,
            output_names=outputs,
            plugin_name="causal_conv1d",
            plugin_version="1",
            plugin_fields=[
                pf_int32("stride", 1),
                pf_int32("padding", w - 1),  # causal left pad
                pf_int32("dilation", 1),
                pf_int32("groups", dim),
                pf_int32("use_mtp", int(self.use_mtp)),
                pf_int32("use_ddtree", int(self.use_ddtree)),
            ],
            profiles=profiles,
        )

    def run(self,
            x,
            weight,
            bias,
            state_pool,
            query_lengths,
            *,
            phase,
            state_indices=None,
            tree_parent_ids=None,
            tree_depths=None):
        batch, seq, dim = x.shape
        if state_indices is None:
            state_indices = torch.arange(batch, dtype=torch.int32, device=DEV)
        offsets = torch.arange(batch + 1, dtype=torch.int32, device=DEV) * seq
        flat_x = x.reshape(batch * seq, dim).contiguous()
        output = torch.empty_like(flat_x)
        bindings = {
            "x":
            flat_x,
            "weight":
            weight,
            "bias":
            bias,
            "conv_state":
            state_pool,
            "query_lengths":
            query_lengths,
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
            "output":
            output,
            # Successful enqueue proves the required pointer alias.
            "conv_state_out":
            state_pool,
        }
        intermediate = None
        if self.use_mtp or self.use_ddtree:
            intermediate = torch.empty(batch * seq,
                                       dim,
                                       self.cfg.width,
                                       dtype=torch.float16,
                                       device=DEV)
            bindings["intermediate_states"] = intermediate
        if self.use_ddtree:
            bindings["tree_parent_ids"] = tree_parent_ids.reshape(-1)
            bindings["tree_depths"] = tree_depths.reshape(-1)
        input_shapes = None
        if phase not in (1, 2):
            input_shapes = {"context_sequence_count_carrier": (0, )}
        self.runner.execute(bindings, input_shapes=input_shapes)
        return (output.reshape(batch, seq,
                               dim), state_pool, None if intermediate is None
                else intermediate.reshape(batch, seq, dim, self.cfg.width))


def _rand(cfg, b, s, gen):
    """Create entry-padded data; the runner flattens it to token-major ABI."""
    dim, w = cfg.dim, cfg.width

    def rn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(DEV)

    x = rn(b, s, dim).to(torch.float16)
    weight = (rn(dim, 1, w) * 0.3).to(torch.float16)
    bias = (rn(dim) * 0.1).to(torch.float16)
    return x, weight, bias


def _decode_steps_ref(x, weight, bias, state0):
    state = state0.float().clone()
    outputs = []
    checkpoints = []
    for token in x.float().unbind(dim=1):
        state = torch.cat([state[:, :, 1:], token.unsqueeze(-1)], dim=-1)
        outputs.append((state * weight[:, 0].float()).sum(-1) + bias.float())
        checkpoints.append(state.clone())
    return torch.stack(outputs, dim=1), torch.stack(checkpoints, dim=1)


def _tree_ref(x, weight, bias, state0, parents, depths):
    batch, seq, dim = x.shape
    width = state0.shape[-1]
    output = torch.zeros(batch, seq, dim, dtype=torch.float32, device=DEV)
    states = torch.empty(batch,
                         seq,
                         dim,
                         width,
                         dtype=torch.float32,
                         device=DEV)
    for bi in range(batch):
        for node in range(seq):
            parent = int(parents[bi, node])
            depth = int(depths[bi, node])
            valid = ((node == 0 and parent < 0 and depth == 0)
                     or (node > 0 and 0 <= parent < node and depth > 0))
            if not valid:
                states[bi, node] = state0[bi].float()
                continue
            path = []
            current = node
            for _ in range(min(depth + 1, width)):
                if not 0 <= current < seq:
                    break
                path.append(current)
                if current == 0:
                    break
                current = int(parents[bi, current])
            state = torch.zeros_like(state0[bi], dtype=torch.float32)
            if len(path) < width:
                state[:, :width - len(path)] = state0[bi, :, len(path):]
            for path_offset, path_node in enumerate(path):
                state[:, width - 1 - path_offset] = x[bi, path_node].float()
            states[bi, node] = state
            output[bi, node] = ((state * weight[:, 0].float()).sum(-1) +
                                bias.float())
    return output, states


# --------------------------------------------------------------------------- #
# Prefill: causal conv output + captured conv-state (fresh zero state)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [1, 2, 4], ids=lambda b: f"bs{b}")
def test_prefill(batch):
    cfg = ConvConfig()
    seq = 16
    gen = torch.Generator().manual_seed(10 + batch)
    r = ConvRunner(cfg)
    x, weight, bias = _rand(cfg, batch, seq, gen)
    state0 = torch.zeros(batch,
                         cfg.dim,
                         cfg.width,
                         dtype=torch.float16,
                         device=DEV)
    ctx = torch.full((batch, ), seq, dtype=torch.int32, device=DEV)
    out, state_out, _ = r.run(x, weight, bias, state0.clone(), ctx, phase=1)
    w2 = weight[:, 0, :]  # [dim, width]
    ref_y, ref_state = causal_conv1d_ref(x, w2, bias, state0, False, ctx)
    for bi in range(batch):
        assert_close(f"y[b{bi}]", ref_y[bi], out[bi], 2e-2, 2e-2)
    assert_close("conv_state_out", ref_state, state_out, 2e-2, 2e-2)


# --------------------------------------------------------------------------- #
# RAGGED prefill: variable per-row lengths + poisoned padding
# --------------------------------------------------------------------------- #
def test_ragged_prefill():
    cfg = ConvConfig()
    seq = 32
    batch = 3
    gen = torch.Generator().manual_seed(202)
    r = ConvRunner(cfg)
    x, weight, bias = _rand(cfg, batch, seq, gen)
    ctx = torch.tensor([32, 17, 5], dtype=torch.int32, device=DEV)
    poison_padding(x, ctx)
    state0 = torch.zeros(batch,
                         cfg.dim,
                         cfg.width,
                         dtype=torch.float16,
                         device=DEV)
    out, state_out, _ = r.run(x, weight, bias, state0.clone(), ctx, phase=1)
    w2 = weight[:, 0, :]
    ref_y, ref_state = causal_conv1d_ref(x, w2, bias, state0, False, ctx)
    for bi in range(batch):
        L = int(ctx[bi])
        assert_close(f"y[b{bi}]", ref_y[bi, :L], out[bi, :L], 2e-2, 2e-2)
        assert torch.count_nonzero(out[bi, L:]) == 0
    assert_close("conv_state_out", ref_state, state_out, 2e-2, 2e-2)


@pytest.mark.parametrize(("phase", "commits_state"), [(2, True), (6, False),
                                                      (7, True)])
def test_context_phase_state_commit_contract(phase, commits_state):
    cfg = ConvConfig(dim=16, width=4, max_batch=2, max_seq=5, max_state_rows=4)
    gen = torch.Generator().manual_seed(2600 + phase)
    runner = ConvRunner(cfg)
    x, weight, bias = _rand(cfg, 2, 5, gen)
    state_pool = (torch.randn(
        4, cfg.dim, cfg.width, generator=gen, dtype=torch.float32) * 0.2).to(
            dtype=torch.float16, device=DEV)
    before = state_pool.clone()
    state_indices = torch.tensor([3, 1], dtype=torch.int32, device=DEV)
    lengths = torch.tensor([5, 3], dtype=torch.int32, device=DEV)
    poison_padding(x, lengths)

    output, _, _ = runner.run(x,
                              weight,
                              bias,
                              state_pool,
                              lengths,
                              phase=phase,
                              state_indices=state_indices)

    selected = before.index_select(0, state_indices.to(torch.int64))
    expected_y, expected_state = causal_conv1d_ref(x, weight[:, 0], bias,
                                                   selected, False, lengths)
    for batch_idx, length in enumerate(lengths.tolist()):
        assert_close(f"phase{phase}.output[{batch_idx}]",
                     expected_y[batch_idx, :length],
                     output[batch_idx, :length], 2e-2, 2e-2)
        assert torch.count_nonzero(output[batch_idx, length:]) == 0
    if commits_state:
        assert_close(f"phase{phase}.resident[3]", expected_state[0],
                     state_pool[3], 2e-2, 2e-2)
        assert_close(f"phase{phase}.resident[1]", expected_state[1],
                     state_pool[1], 2e-2, 2e-2)
        assert torch.equal(state_pool[0], before[0])
        assert torch.equal(state_pool[2], before[2])
    else:
        assert torch.equal(state_pool, before)


# --------------------------------------------------------------------------- #
# Decode (seq=1): state carry-over from a non-zero conv-state.
# The conv-state buffer holds the last `width` inputs; decode rolls left and
# appends the new input.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [1, 2], ids=lambda b: f"bs{b}")
def test_decode(batch):
    cfg = ConvConfig()
    gen = torch.Generator().manual_seed(30 + batch)
    r = ConvRunner(cfg)
    x, weight, bias = _rand(cfg, batch, 1, gen)  # x is [b, 1, dim]
    state0 = (torch.randn(
        batch, cfg.dim, cfg.width, generator=gen, dtype=torch.float32) *
              0.3).to(torch.float16).to(DEV)
    ctx = torch.ones(batch, dtype=torch.int32, device=DEV)
    out, state_out, _ = r.run(x, weight, bias, state0.clone(), ctx, phase=3)
    w2 = weight[:, 0, :]
    ref_y, ref_state = causal_conv1d_ref(x, w2, bias, state0, False, ctx)
    assert_close("y", ref_y, out, 2e-2, 2e-2)
    assert_close("conv_state_out", ref_state, state_out, 2e-2, 2e-2)


@pytest.mark.parametrize(("phase", "seq"), [(1, 5), (3, 1), (4, 3)])
def test_nonidentity_resident_slots_and_aliasing(phase, seq):
    cfg = ConvConfig(dim=16, width=4, max_batch=2, max_seq=5, max_state_rows=4)
    gen = torch.Generator().manual_seed(811 + phase)
    runner = ConvRunner(cfg)
    x, weight, bias = _rand(cfg, 2, seq, gen)
    state_pool = (torch.randn(
        4, cfg.dim, cfg.width, generator=gen, dtype=torch.float32) * 0.2).to(
            dtype=torch.float16, device=DEV)
    before = state_pool.clone()
    state_indices = torch.tensor([2, 0], dtype=torch.int32, device=DEV)
    lengths = torch.full((2, ), seq, dtype=torch.int32, device=DEV)

    output, state_out, _ = runner.run(x,
                                      weight,
                                      bias,
                                      state_pool,
                                      lengths,
                                      phase=phase,
                                      state_indices=state_indices)

    selected = before.index_select(0, state_indices.to(torch.int64))
    expected_y, expected_state = causal_conv1d_ref(x, weight[:, 0], bias,
                                                   selected, False, lengths)
    assert state_out is state_pool
    assert state_out.data_ptr() == state_pool.data_ptr()
    assert_close("indexed output", expected_y, output, 2e-2, 2e-2)
    assert_close("resident row 2", expected_state[0], state_pool[2], 2e-2,
                 2e-2)
    assert_close("resident row 0", expected_state[1], state_pool[0], 2e-2,
                 2e-2)
    assert torch.equal(state_pool[1], before[1])
    assert torch.equal(state_pool[3], before[3])


def test_mtp_verify_uses_slots_without_mutating_resident_state():
    cfg = ConvConfig(dim=16, width=4, max_batch=2, max_seq=4, max_state_rows=4)
    gen = torch.Generator().manual_seed(902)
    runner = ConvRunner(cfg, use_mtp=True)
    x, weight, bias = _rand(cfg, 2, 4, gen)
    state_pool = (torch.randn(
        4, cfg.dim, cfg.width, generator=gen, dtype=torch.float32) * 0.2).to(
            dtype=torch.float16, device=DEV)
    before = state_pool.clone()
    state_indices = torch.tensor([3, 1], dtype=torch.int32, device=DEV)
    lengths = torch.full((2, ), 4, dtype=torch.int32, device=DEV)

    output, state_out, intermediate = runner.run(x,
                                                 weight,
                                                 bias,
                                                 state_pool,
                                                 lengths,
                                                 phase=5,
                                                 state_indices=state_indices)

    selected = before.index_select(0, state_indices.to(torch.int64))
    expected_y, expected_intermediate = _decode_steps_ref(
        x, weight, bias, selected)
    assert state_out is state_pool
    assert torch.equal(state_pool, before)
    assert_close("mtp output", expected_y, output, 2e-2, 2e-2)
    assert_close("mtp checkpoints", expected_intermediate, intermediate, 2e-2,
                 2e-2)


def test_ddtree_verify_uses_nonidentity_resident_slots():
    cfg = ConvConfig(dim=16, width=4, max_batch=2, max_seq=6, max_state_rows=4)
    gen = torch.Generator().manual_seed(947)
    runner = ConvRunner(cfg, use_ddtree=True)
    x, weight, bias = _rand(cfg, 2, 6, gen)
    state_pool = (torch.randn(
        4, cfg.dim, cfg.width, generator=gen, dtype=torch.float32) * 0.2).to(
            dtype=torch.float16, device=DEV)
    before = state_pool.clone()
    state_indices = torch.tensor([2, 0], dtype=torch.int32, device=DEV)
    parents = torch.tensor([[-1, 0, 0, 1, 1, 2], [-1, 0, 0, -1, -1, -1]],
                           dtype=torch.int32,
                           device=DEV)
    depths = torch.tensor([[0, 1, 1, 2, 2, 2], [0, 1, 1, -1, -1, -1]],
                          dtype=torch.int32,
                          device=DEV)
    lengths = torch.tensor([6, 3], dtype=torch.int32, device=DEV)
    poison_padding(x, lengths)

    output, state_out, intermediate = runner.run(x,
                                                 weight,
                                                 bias,
                                                 state_pool,
                                                 lengths,
                                                 phase=5,
                                                 state_indices=state_indices,
                                                 tree_parent_ids=parents,
                                                 tree_depths=depths)

    selected = before.index_select(0, state_indices.to(torch.int64))
    expected_y, expected_intermediate = _tree_ref(x, weight, bias, selected,
                                                  parents, depths)
    assert state_out is state_pool
    assert torch.equal(state_pool, before)
    for batch_idx, length in enumerate(lengths.tolist()):
        assert_close(f"ddtree output[{batch_idx}]",
                     expected_y[batch_idx, :length],
                     output[batch_idx, :length], 2e-2, 2e-2)
        assert_close(f"ddtree checkpoints[{batch_idx}]",
                     expected_intermediate[batch_idx, :length],
                     intermediate[batch_idx, :length], 2e-2, 2e-2)
        assert torch.count_nonzero(output[batch_idx, length:]) == 0


# --------------------------------------------------------------------------- #
# Required even/uneven batch cases (bs 1/2/3/4/8, seq up to 2048).
# --------------------------------------------------------------------------- #
def _ragged_cfg():
    return ConvConfig(dim=256, width=4, max_batch=8, max_seq=2048)


@pytest.mark.parametrize("label,seqlens", RAGGED_CASES)
def test_ragged_prefill_batch_sizes(label, seqlens):
    cfg = _ragged_cfg()
    bs, maxlen = len(seqlens), max(seqlens)
    gen = torch.Generator().manual_seed(1100 + maxlen + bs)
    x, weight, bias = _rand(cfg, bs, maxlen, gen)
    ctx = torch.tensor(seqlens, dtype=torch.int32, device=DEV)
    poison_padding(x, ctx)
    state0 = torch.zeros(bs,
                         cfg.dim,
                         cfg.width,
                         dtype=torch.float16,
                         device=DEV)
    r = ConvRunner(cfg)
    out, state_out, _ = r.run(x, weight, bias, state0.clone(), ctx, phase=1)
    ref_y, ref_state = causal_conv1d_ref(x, weight[:, 0, :], bias, state0,
                                         False, ctx)
    for bi in range(bs):
        L = seqlens[bi]
        assert_close(f"conv[{label}].y[{bi}]", ref_y[bi, :L], out[bi, :L])
        assert torch.count_nonzero(out[bi, L:]) == 0
    assert_close(f"conv[{label}].state", ref_state, state_out)


# --------------------------------------------------------------------------- #
# Batch invariance (plugin-vs-plugin): permuting batch rows permutes outputs.
# --------------------------------------------------------------------------- #
def test_batch_invariance():
    cfg = _ragged_cfg()
    seqlens = [10, 2048, 128]
    bs, maxlen = len(seqlens), max(seqlens)
    gen = torch.Generator().manual_seed(1357)
    x, weight, bias = _rand(cfg, bs, maxlen, gen)
    ctx = torch.tensor(seqlens, dtype=torch.int32, device=DEV)
    poison_padding(x, ctx)
    state0 = torch.zeros(bs,
                         cfg.dim,
                         cfg.width,
                         dtype=torch.float16,
                         device=DEV)
    r = ConvRunner(cfg)
    out0, st0, _ = r.run(x, weight, bias, state0.clone(), ctx, phase=1)
    perm = torch.tensor([2, 0, 1], device=DEV)
    out1, st1, _ = r.run(x[perm].contiguous(),
                         weight,
                         bias,
                         state0.clone(),
                         ctx[perm].contiguous(),
                         phase=1)
    pcpu = perm.cpu()
    for new_i in range(bs):
        orig = int(pcpu[new_i])
        L = seqlens[orig]
        assert_close(f"conv-batch-inv.y[{new_i}]", out0[orig, :L],
                     out1[new_i, :L])
    assert_close("conv-batch-inv.state", st0[pcpu], st1)
