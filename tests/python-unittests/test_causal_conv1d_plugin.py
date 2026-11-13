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
from typing import Optional

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              RAGGED_CASES, PluginRunner,
                              PluginUnsupportedError, _device_sm, assert_close,
                              assert_compact_workspace,
                              assert_execute_rejected, pf_int32,
                              poison_padding)

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
    dilation: int = 1
    # Columns of history the conv state holds. Mamba/GDN use `width`; the dilated
    # HuggingFace/vLLM convention is the tight `(width - 1) * dilation`.
    state_len: Optional[int] = None

    @property
    def window(self) -> int:
        return self.width if self.state_len is None else self.state_len


class ConvRunner:
    """Build and run the token-major ragged causal-conv ABI."""

    def __init__(self,
                 cfg: ConvConfig,
                 *,
                 use_mtp=False,
                 use_ddtree=False,
                 expect_unsupported=False):
        self.cfg = cfg
        self.use_mtp = use_mtp
        self.use_ddtree = use_ddtree
        self.expect_unsupported = expect_unsupported
        self.runner = PluginRunner()
        self._build()

    def _build(self):
        c = self.cfg
        dim, w, mb, ms = c.dim, c.width, c.max_batch, c.max_seq
        sl = c.window
        F16, I32 = trt.float16, trt.int32
        input_specs = [
            ("x", F16, (-1, dim)),
            ("weight", F16, (dim, 1, w)),
            ("bias", F16, (dim, )),
            ("conv_state", F16, (-1, dim, sl)),
            ("query_lengths", I32, (-1, )),
            ("query_start_offsets", I32, (-1, )),
            ("state_indices", I32, (-1, )),
            ("execution_phase_marker", I32, (-1, )),
            ("context_sequence_count_carrier", I32, (-1, )),
        ]
        if self.use_mtp or self.use_ddtree:
            input_specs += [("tree_parent_ids", I32, (-1, )),
                            ("tree_depths", I32, (-1, )),
                            ("spec_verify_layout", I32, (-1, ))]
        profiles = {
            "x": ((1, dim), (min(16, mb * ms), dim), (mb * ms, dim)),
            "weight": ((dim, 1, w), (dim, 1, w), (dim, 1, w)),
            "bias": ((dim, ), (dim, ), (dim, )),
            "conv_state":
            ((1, dim, sl), (mb, dim, sl), (c.max_state_rows, dim, sl)),
            "query_lengths": ((1, ), (mb, ), (mb, )),
            "query_start_offsets": ((2, ), (mb + 1, ), (mb + 1, )),
            "state_indices": ((1, ), (mb, ), (mb, )),
            "execution_phase_marker": ((1, ), (1, ), (8, )),
            "context_sequence_count_carrier": ((0, ), (mb, ), (mb, )),
        }
        if self.use_mtp or self.use_ddtree:
            profiles["tree_parent_ids"] = ((1, ), (min(16, mb * ms), ),
                                           (mb * ms, ))
            profiles["tree_depths"] = profiles["tree_parent_ids"]
            profiles["spec_verify_layout"] = ((0, ), (1, ), (2, ))
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
                pf_int32("padding", (w - 1) * c.dilation),  # causal left pad
                pf_int32("dilation", c.dilation),
                pf_int32("groups", dim),
                pf_int32("use_spec_verify_state",
                         int(self.use_mtp or self.use_ddtree)),
            ],
            profiles=profiles,
            expect_unsupported=self.expect_unsupported,
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
            tree_depths=None,
            layout=None,
            cuda_graph=False):
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
                                       self.cfg.window,
                                       dtype=torch.float16,
                                       device=DEV)
            bindings["intermediate_states"] = intermediate
        if self.use_mtp or self.use_ddtree:
            if tree_parent_ids is None:
                tree_parent_ids = (
                    torch.arange(seq, dtype=torch.int32, device=DEV) -
                    1).repeat(batch)
                tree_depths = torch.arange(seq, dtype=torch.int32,
                                           device=DEV).repeat(batch)
            extent = (
                2 if self.use_ddtree else 1) if layout is None else layout
            extent = extent if phase == 5 else 0
            bindings["spec_verify_layout"] = torch.empty(max(1, extent),
                                                         dtype=torch.int32,
                                                         device=DEV)
            bindings["tree_parent_ids"] = tree_parent_ids.reshape(-1)
            bindings["tree_depths"] = tree_depths.reshape(-1)
        input_shapes = None
        if phase not in (1, 2):
            input_shapes = {"context_sequence_count_carrier": (0, )}
        if self.use_mtp or self.use_ddtree:
            input_shapes = dict(input_shapes or {},
                                spec_verify_layout=(extent, ))
        if cuda_graph:
            original_state = state_pool.clone()
            self.runner.execute(bindings, input_shapes=input_shapes)
            state_pool.copy_(original_state)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.runner.execute(bindings,
                                    input_shapes=input_shapes,
                                    synchronize=False)
            graph.replay()
            torch.cuda.current_stream().synchronize()
        else:
            self.runner.execute(bindings, input_shapes=input_shapes)
        return (output.reshape(batch, seq,
                               dim), state_pool, None if intermediate is None
                else intermediate.reshape(batch, seq, dim, self.cfg.window))

    def run_tokens(self, x, weight, bias, state_pool, query_lengths,
                   state_indices, phase, num_prefill):
        """Execute one step over an unpadded token run [T, dim] against a
        resident state pool (updated in place); returns the output [T, dim].
        """
        q_len = torch.tensor(query_lengths, dtype=torch.int32, device=DEV)
        offsets = torch.zeros(len(query_lengths) + 1,
                              dtype=torch.int32,
                              device=DEV)
        offsets[1:] = torch.cumsum(q_len, 0)
        output = torch.empty_like(x)
        bindings = {
            "x":
            x,
            "weight":
            weight,
            "bias":
            bias,
            "conv_state":
            state_pool,
            "query_lengths":
            q_len,
            "query_start_offsets":
            offsets,
            "state_indices":
            torch.tensor(state_indices, dtype=torch.int32, device=DEV),
            "execution_phase_marker":
            torch.empty(phase, dtype=torch.int32, device=DEV),
            "context_sequence_count_carrier":
            torch.empty(max(1, num_prefill), dtype=torch.int32, device=DEV),
            "output":
            output,
            "conv_state_out":
            state_pool,
        }
        input_shapes = ({
            "context_sequence_count_carrier": (0, )
        } if num_prefill == 0 else None)
        if self.use_mtp or self.use_ddtree:
            # Inactive verify descriptor; a verify layer's phase 8 is rejected.
            total = x.shape[0]
            bindings.update({
                "tree_parent_ids":
                torch.arange(-1, total - 1, dtype=torch.int32, device=DEV),
                "tree_depths":
                torch.arange(total, dtype=torch.int32, device=DEV),
                "spec_verify_layout":
                torch.empty(1, dtype=torch.int32, device=DEV),
                "intermediate_states":
                torch.empty(total,
                            x.shape[1],
                            self.cfg.window,
                            dtype=torch.float16,
                            device=DEV),
            })
            input_shapes = dict(input_shapes or {}, spec_verify_layout=(0, ))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.runner.execute(bindings, input_shapes=input_shapes)
        return output


def _spec_state_output_shape(*, context_profile: bool):
    dim, width = 16, 4
    batch_profile = ((1, ), (2, ), (4, ))
    phase = 1 if context_profile else 5
    carrier_profile = (batch_profile if context_profile else
                       ((0, ), (0, ), (0, )))
    profiles = {
        "x": ((1, dim), (8, dim), (16, dim)),
        "weight": ((dim, 1, width), (dim, 1, width), (dim, 1, width)),
        "bias": ((dim, ), (dim, ), (dim, )),
        "conv_state": ((1, dim, width), (2, dim, width), (4, dim, width)),
        "query_lengths": batch_profile,
        "query_start_offsets": ((2, ), (3, ), (5, )),
        "state_indices": batch_profile,
        "execution_phase_marker": ((phase, ), (phase, ), (phase, )),
        "context_sequence_count_carrier": carrier_profile,
        "tree_parent_ids": ((1, ), (8, ), (16, )),
        "tree_depths": ((1, ), (8, ), (16, )),
        "spec_verify_layout": ((0, ), (1, ), (2, )),
    }
    runner = PluginRunner().build(
        input_specs=[
            ("x", trt.float16, (-1, dim)),
            ("weight", trt.float16, (dim, 1, width)),
            ("bias", trt.float16, (dim, )),
            ("conv_state", trt.float16, (-1, dim, width)),
            ("query_lengths", trt.int32, (-1, )),
            ("query_start_offsets", trt.int32, (-1, )),
            ("state_indices", trt.int32, (-1, )),
            ("execution_phase_marker", trt.int32, (-1, )),
            ("context_sequence_count_carrier", trt.int32, (-1, )),
            ("tree_parent_ids", trt.int32, (-1, )),
            ("tree_depths", trt.int32, (-1, )),
            ("spec_verify_layout", trt.int32, (-1, )),
        ],
        output_names=["output", "conv_state_out", "intermediate_states"],
        plugin_name="causal_conv1d",
        plugin_version="1",
        plugin_fields=[
            pf_int32("stride", 1),
            pf_int32("padding", width - 1),
            pf_int32("dilation", 1),
            pf_int32("groups", dim),
            pf_int32("use_spec_verify_state", 1),
        ],
        profiles=profiles,
    )
    for name, (_, opt_shape, _) in profiles.items():
        assert runner.context.set_input_shape(name, opt_shape)
    return tuple(runner.context.get_tensor_shape("intermediate_states"))


def test_spec_state_output_is_empty_for_context_profile():
    assert _spec_state_output_shape(context_profile=True) == (0, 16, 4)


def test_spec_state_output_keeps_token_rows_for_verify_profile():
    assert _spec_state_output_shape(context_profile=False) == (8, 16, 4)


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


# --------------------------------------------------------------------------- #
# Dilated short convolution (Qwen3.8-Flash-Next n-gram embedding: k=4, dilation=3)
# --------------------------------------------------------------------------- #
def dilated_conv1d_ref(
        x: torch.Tensor,  # [b, s, c]
        weight: torch.Tensor,  # [c, 1, width]
        bias: torch.Tensor,  # [c]
        conv_state0: torch.Tensor,  # [b, c, state_len]
        dilation: int,
        context_lengths: Optional[torch.Tensor] = None,  # [b]
):
    """Golden from ``F.conv1d(dilation=...)``. Returns (y, final_conv_state).

    The conv state supplies the left context, so ``cat([state, x])`` is already
    padded and a plain valid convolution over its last ``(width-1)*dilation + L``
    columns yields the ``L`` outputs of the row.
    """
    b, s, c = x.shape
    width = weight.shape[-1]
    state_len = conv_state0.shape[-1]
    span = (width - 1) * dilation
    xf = x.float().transpose(1, 2)  # [b, c, s]
    wf = weight.float()  # [c, 1, width]
    state0 = conv_state0.float()

    if context_lengths is None:
        context_lengths = torch.full((b, ), s, dtype=torch.int64)
    else:
        context_lengths = context_lengths.to(torch.int64)

    y = torch.zeros((b, c, s), dtype=torch.float32, device=x.device)
    final_state = state0.clone()
    for bi in range(b):
        length = int(context_lengths[bi])
        seq = torch.cat([state0[bi], xf[bi, :, :length]], dim=-1)
        if length > 0:
            window = seq[:, state_len - span:].unsqueeze(0)
            y[bi, :, :length] = torch.nn.functional.conv1d(window,
                                                           wf,
                                                           bias.float(),
                                                           dilation=dilation,
                                                           groups=c)[0]
        final_state[bi] = seq[:, -state_len:]
    return y.transpose(1, 2), final_state


def _dilated_cfg(**kwargs):
    base = dict(dim=128,
                width=4,
                dilation=3,
                state_len=9,
                max_batch=4,
                max_seq=32,
                max_state_rows=8)
    base.update(kwargs)
    return ConvConfig(**base)


@pytest.mark.parametrize("state_len", [9, 10], ids=lambda s: f"state{s}")
def test_dilated_prefill(state_len):
    """state_len 10 keeps one column the dilated taps must ignore."""
    cfg = _dilated_cfg(state_len=state_len)
    batch, seq = 3, 24
    gen = torch.Generator().manual_seed(4100 + state_len)
    runner = ConvRunner(cfg)
    x, weight, bias = _rand(cfg, batch, seq, gen)
    ctx = torch.tensor([24, 11, 2], dtype=torch.int32, device=DEV)
    poison_padding(x, ctx)
    state0 = (torch.randn(
        batch, cfg.dim, state_len, generator=gen, dtype=torch.float32) *
              0.3).to(torch.float16).to(DEV)

    out, state_out, _ = runner.run(x,
                                   weight,
                                   bias,
                                   state0.clone(),
                                   ctx,
                                   phase=1)

    ref_y, ref_state = dilated_conv1d_ref(x, weight, bias, state0,
                                          cfg.dilation, ctx)
    for bi in range(batch):
        length = int(ctx[bi])
        assert_close(f"dilated y[b{bi}]", ref_y[bi, :length], out[bi, :length],
                     2e-2, 2e-2)
        assert torch.count_nonzero(out[bi, length:]) == 0
    assert_close("dilated conv_state_out", ref_state, state_out, 2e-2, 2e-2)


def test_dilated_decode():
    cfg = _dilated_cfg()
    batch = 2
    gen = torch.Generator().manual_seed(4200)
    runner = ConvRunner(cfg)
    x, weight, bias = _rand(cfg, batch, 1, gen)
    state0 = (torch.randn(
        batch, cfg.dim, cfg.state_len, generator=gen, dtype=torch.float32) *
              0.3).to(torch.float16).to(DEV)
    ctx = torch.ones(batch, dtype=torch.int32, device=DEV)

    out, state_out, _ = runner.run(x,
                                   weight,
                                   bias,
                                   state0.clone(),
                                   ctx,
                                   phase=3)

    ref_y, ref_state = dilated_conv1d_ref(x, weight, bias, state0,
                                          cfg.dilation, ctx)
    assert_close("dilated decode y", ref_y, out, 2e-2, 2e-2)
    assert_close("dilated decode conv_state_out", ref_state, state_out, 2e-2,
                 2e-2)


def test_dilated_mtp_verify():
    cfg = _dilated_cfg(max_seq=4)
    batch, seq = 2, 4
    gen = torch.Generator().manual_seed(4300)
    runner = ConvRunner(cfg, use_mtp=True)
    x, weight, bias = _rand(cfg, batch, seq, gen)
    state_pool = (torch.randn(
        4, cfg.dim, cfg.state_len, generator=gen, dtype=torch.float32) *
                  0.2).to(dtype=torch.float16, device=DEV)
    before = state_pool.clone()
    state_indices = torch.tensor([3, 1], dtype=torch.int32, device=DEV)
    lengths = torch.full((batch, ), seq, dtype=torch.int32, device=DEV)

    output, state_out, intermediate = runner.run(x,
                                                 weight,
                                                 bias,
                                                 state_pool,
                                                 lengths,
                                                 phase=5,
                                                 state_indices=state_indices)

    # Each draft token is one more step of the same dilated recurrence.
    selected = before.index_select(0, state_indices.to(torch.int64))
    state = selected.float()
    expected_y = []
    expected_states = []
    for token in x.float().unbind(dim=1):
        step_y, _ = dilated_conv1d_ref(
            token.unsqueeze(1).half(), weight, bias, state.half(),
            cfg.dilation)
        expected_y.append(step_y[:, 0])
        state = torch.cat([state[:, :, 1:], token.unsqueeze(-1)], dim=-1)
        expected_states.append(state.clone())

    assert state_out is state_pool
    assert torch.equal(state_pool, before)
    assert_close("dilated mtp output", torch.stack(expected_y, dim=1), output,
                 2e-2, 2e-2)
    assert_close("dilated mtp checkpoints",
                 torch.stack(expected_states, dim=1), intermediate, 2e-2, 2e-2)


# --------------------------------------------------------------------------- #
# Shapes outside the supported range must be rejected at build time.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("width", "dilation", "state_len"), [
    (1, 1, 1),
    (9, 1, 9),
    (4, 5, 15),
    (4, 3, 8),
    (2, 1, 17),
    (5, 4, 16),
])
def test_rejects_unsupported_shapes(width, dilation, state_len):
    cfg = ConvConfig(dim=16,
                     width=width,
                     dilation=dilation,
                     state_len=state_len,
                     max_batch=2,
                     max_seq=4,
                     max_state_rows=4)
    with pytest.raises(PluginUnsupportedError):
        ConvRunner(cfg, expect_unsupported=True)


def test_one_verify_artifact_executes_inactive_chain_and_tree():
    # Given one verify-capable convolution artifact, when switching descriptor
    # extents, then inactive prefill and both layouts retain causal outputs.
    cfg = ConvConfig(dim=64, max_batch=1, max_seq=4)
    runner = ConvRunner(cfg, use_mtp=True)
    gen = torch.Generator().manual_seed(73)
    x, weight, bias = _rand(cfg, 1, 4, gen)
    lengths = torch.tensor([4], dtype=torch.int32, device=DEV)
    initial = torch.zeros(1,
                          cfg.dim,
                          cfg.window,
                          dtype=torch.float16,
                          device=DEV)
    expected, expected_state = causal_conv1d_ref(x, weight.squeeze(1), bias,
                                                 initial, False)
    parents = torch.tensor([[-1, 0, 0, 2]], dtype=torch.int32, device=DEV)
    depths = torch.tensor([[0, 1, 1, 2]], dtype=torch.int32, device=DEV)
    tree_output, tree_states = _tree_ref(x, weight, bias, initial, parents,
                                         depths)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for cuda_graph in (False, True):
            for phase, layout in ((1, 0), (5, 1), (5, 2)):
                actual, state, snapshots = runner.run(
                    x,
                    weight,
                    bias,
                    initial.clone(),
                    lengths,
                    phase=phase,
                    layout=layout,
                    tree_parent_ids=parents if layout == 2 else None,
                    tree_depths=depths if layout == 2 else None,
                    cuda_graph=cuda_graph)
                assert_close("layout output",
                             tree_output if layout == 2 else expected, actual)
                retained = state if phase == 1 else snapshots[:, -1]
                assert_close(
                    "retained state",
                    tree_states[:, -1] if layout == 2 else expected_state,
                    retained)


@pytest.mark.parametrize("name,outputs",
                         (("causal_conv1d", 3), ("gated_delta_net", 3),
                          ("update_ssm_state", 6)))
def test_recurrent_creator_declares_one_verify_capability(name, outputs):
    # Given a version-1 creator, when creating a verify instance, then its
    # fixed output arity needs neither topology nor replay-version fields.
    PluginRunner()
    creator = trt.get_plugin_registry().get_creator(name, "1", "")
    fields = [pf_int32("use_spec_verify_state", 1)]
    plugin = creator.create_plugin(name, trt.PluginFieldCollection(fields),
                                   trt.TensorRTPhase.BUILD)
    build = plugin.get_capability_interface(trt.PluginCapabilityType.BUILD)
    assert (build.num_outputs, {field.name
                                for field in creator.field_names}
            & {"use_ddtree", "replay_format_version", "use_spec_verify_state"
               }) == (outputs, {"use_spec_verify_state"})


@pytest.mark.parametrize(
    "name", ("causal_conv1d", "gated_delta_net", "update_ssm_state"))
def test_recurrent_creator_rejects_missing_verify_capability(name):
    # Given fields from an old artifact, when the version-1 creator loads them,
    # then missing fixed-ABI capability is rejected instead of guessed.
    PluginRunner()
    creator = trt.get_plugin_registry().get_creator(name, "1", "")
    plugin = creator.create_plugin(name, trt.PluginFieldCollection([]),
                                   trt.TensorRTPhase.BUILD)
    assert plugin is None


# --------------------------------------------------------------------------- #
# Compact phase 8: packed varlen prefill rows first, then single-step decode
# rows, against one resident state pool with non-contiguous slots.
# --------------------------------------------------------------------------- #
COMPACT_CONV_SMS = frozenset(
    {80, 86, 87, 89, 90, 100, 101, 103, 110, 120, 121})
_CONV_POOL_ROWS = 6
# (label, prefill [(slot, query_length, continued)], decode slots)
_CONV_COMPACT_CASES = [
    ("prefill_and_decode", [(4, 37, True), (1, 130, False)], [3, 0]),
    ("prefill_only", [(2, 64, True), (5, 2, False)], []),
    ("single_row_prefill", [(5, 1, True)], [1, 2, 4]),
    ("decode_only", [], [3, 0, 5]),
    # Profile 0 upper bound of the test engine: Np = B = 6, T_p = T_max.
    ("profile_upper_bound_prefill_only", [(0, 256, True), (1, 256, False),
                                          (2, 256, True), (3, 256, False),
                                          (4, 256, True), (5, 256, True)], []),
]


def _conv_compact_setup(cfg, query_lengths, gen):
    x = (torch.randn(sum(query_lengths), cfg.dim,
                     generator=gen)).to(torch.float16).to(DEV)
    weight = (torch.randn(cfg.dim, 1, cfg.width, generator=gen) * 0.3).to(
        torch.float16).to(DEV)
    bias = (torch.randn(cfg.dim, generator=gen) * 0.1).to(
        torch.float16).to(DEV)
    pool = (torch.randn(_CONV_POOL_ROWS, cfg.dim, cfg.window, generator=gen) *
            0.3).to(torch.float16).to(DEV)
    return x, weight, bias, pool


@pytest.mark.skipif(_device_sm() not in COMPACT_CONV_SMS,
                    reason="packed varlen causal conv1d is unsupported here")
@pytest.mark.parametrize("state_len", [None, 3], ids=["state_w", "state_w1"])
@pytest.mark.parametrize("label,prefill,decode", _CONV_COMPACT_CASES)
def test_compact_phase_matches_separate_prefill_and_decode(
        label, prefill, decode, state_len):
    """Phase 8 equals separate execution on one state pool: each prefill
    sequence (fresh or continued) matches its own prefill step in output and
    its whole conv state bit for bit, the decode rows and their states match a
    phase 3 step bit for bit, and pool rows outside the step stay unchanged,
    up to the profile bound Np = B, T_p = T_max, with a conv state of width or
    width - 1 columns."""
    cfg = ConvConfig(dim=256,
                     width=4,
                     max_batch=_CONV_POOL_ROWS,
                     max_seq=256,
                     max_state_rows=8,
                     state_len=state_len)
    gen = torch.Generator().manual_seed(6160 + len(prefill) + len(decode))
    query_lengths = [q for _, q, _ in prefill] + [1] * len(decode)
    slots = [slot for slot, _, _ in prefill] + decode
    x, weight, bias, pool = _conv_compact_setup(cfg, query_lengths, gen)
    for slot, _, continued in prefill:
        if not continued:
            pool[slot].zero_()
    runner = ConvRunner(cfg)

    compact_pool = pool.clone()
    compact_out = runner.run_tokens(x, weight, bias, compact_pool,
                                    query_lengths, slots, 8, len(prefill))

    offset = 0
    for slot, q_len, continued in prefill:
        ref_pool = pool.clone()
        ref_out = runner.run_tokens(x[offset:offset + q_len].contiguous(),
                                    weight, bias, ref_pool, [q_len], [slot],
                                    2 if continued else 1, 1)
        assert_close(f"conv-compact[{label}].y[slot{slot}]", ref_out,
                     compact_out[offset:offset + q_len], 2e-2, 2e-2)
        assert torch.equal(ref_pool[slot], compact_pool[slot]), \
            f"conv-compact[{label}]: state of slot {slot} differs from padded"
        offset += q_len
    if decode:
        ref_pool = pool.clone()
        ref_out = runner.run_tokens(
            x[offset:offset + len(decode)].contiguous(), weight, bias,
            ref_pool, [1] * len(decode), decode, 3, 0)
        assert torch.equal(ref_out, compact_out[offset:]), \
            f"conv-compact[{label}]: decode rows differ from phase 3"
        for slot in decode:
            assert torch.equal(ref_pool[slot], compact_pool[slot]), \
                f"conv-compact[{label}]: decode state of slot {slot} differs"
    for slot in set(range(_CONV_POOL_ROWS)) - set(slots):
        assert torch.equal(pool[slot], compact_pool[slot]), \
            f"conv-compact[{label}]: untouched slot {slot} changed"


@pytest.mark.skipif(_device_sm() not in COMPACT_CONV_SMS,
                    reason="packed varlen causal conv1d is unsupported here")
def test_compact_prefill_skips_out_of_range_state_indices():
    """Prefill rows whose state index lies outside the pool are skipped like
    padding slots: no state outside the pool is read or written, and the
    valid rows of the step are unaffected."""
    cfg = ConvConfig(dim=256,
                     width=4,
                     max_batch=8,
                     max_seq=64,
                     max_state_rows=8)
    gen = torch.Generator().manual_seed(63)
    query_lengths = [9, 5, 4]
    x, weight, bias, pool = _conv_compact_setup(cfg, query_lengths, gen)
    runner = ConvRunner(cfg)
    # Guard rows on both sides of the pool catch out-of-range accesses.
    guarded = torch.randn(_CONV_POOL_ROWS + 2,
                          cfg.dim,
                          cfg.width,
                          generator=gen).to(torch.float16).to(DEV)
    guarded[1:-1] = pool
    before = guarded.clone()

    compact_out = runner.run_tokens(x, weight, bias, guarded[1:-1],
                                    query_lengths, [-1, 2, _CONV_POOL_ROWS], 8,
                                    3)

    ref_pool = pool.clone()
    ref_out = runner.run_tokens(x[9:14].contiguous(), weight, bias, ref_pool,
                                [5], [2], 2, 1)
    assert_close("conv-compact[out_of_range].y[slot2]", ref_out,
                 compact_out[9:14], 2e-2, 2e-2)
    assert torch.equal(guarded[0], before[0]) and \
        torch.equal(guarded[-1], before[-1]), \
        "conv-compact[out_of_range]: state outside the pool was written"
    for slot in set(range(_CONV_POOL_ROWS)) - {2}:
        assert torch.equal(guarded[1 + slot], before[1 + slot]), \
            f"conv-compact[out_of_range]: pool row {slot} changed"


@pytest.mark.parametrize("cfg_overrides,runner_kwargs,shape", [
    pytest.param({}, {"use_mtp": True},
                 "256 channels, width 4, dilation 1, spec-verify state",
                 id="spec_verify"),
    pytest.param({
        "dim": 128,
        "dilation": 3,
        "state_len": 9
    }, {},
                 "128 channels, width 4, dilation 3",
                 id="dilation_3"),
])
def test_compact_phase_rejected_for_unsupported_layer(cfg_overrides,
                                                      runner_kwargs, shape,
                                                      capfd):
    """On every GPU, a spec-verify layer or a dilated convolution rejects
    phase 8 with the unsupported-layer error, which names the layer and its
    shape, without touching the resident state pool."""
    fields = dict(dim=256, width=4, max_batch=8, max_seq=64, max_state_rows=8)
    fields.update(cfg_overrides)
    cfg = ConvConfig(**fields)
    gen = torch.Generator().manual_seed(61)
    x, weight, bias, pool = _conv_compact_setup(cfg, [5, 1], gen)
    before = pool.clone()
    runner = ConvRunner(cfg, **runner_kwargs)
    assert_execute_rejected(
        capfd,
        f"causal_conv1d 'causal_conv1d' ({shape}): compact execution is not "
        "supported for this layer on this GPU.",
        lambda: runner.run_tokens(x, weight, bias, pool, [5, 1], [2, 4], 8, 1))
    assert torch.equal(pool, before), \
        "phase 8 rejection must not write the state pool"


@pytest.mark.parametrize("cfg_overrides,runner_kwargs,supported", [
    pytest.param({}, {}, True, id="depthwise"),
    pytest.param({}, {"use_mtp": True}, False, id="spec_verify"),
    pytest.param({
        "dim": 128,
        "dilation": 3,
        "state_len": 9
    }, {},
                 False,
                 id="dilation_3"),
])
def test_build_reports_whether_the_layer_reserved_the_compact_workspace(
        capfd, cfg_overrides, runner_kwargs, supported):
    """The engine build reports the layer as reserving the compact
    workspace exactly where it accepts compact execution: a depthwise layer
    on the packed-varlen conv SMs, a spec-verify or dilated layer nowhere; a
    layer that does not reserve it is named in a WARNING."""
    fields = dict(dim=256, width=4, max_batch=8, max_seq=64, max_state_rows=8)
    fields.update(cfg_overrides)
    runner = ConvRunner(ConvConfig(**fields), **runner_kwargs)
    assert_compact_workspace(runner.runner, supported
                             and _device_sm() in COMPACT_CONV_SMS, capfd)
