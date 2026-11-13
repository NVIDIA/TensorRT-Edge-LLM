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
"""Packed SSD contracts against an independent token-by-token FP32 recurrence.

Run on SM120/121 with CuPy and nvidia-cutlass-dsl[cu13]:
  pytest -c /dev/null --confcutdir=tests/python-unittests \
      tests/python-unittests/test_ssd_sm12x.py -v -s
"""

import pathlib
import sys

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
cute = pytest.importorskip("cutlass.cute")
from cuda.bindings import driver as cuda
from cutlass.cute import runtime

sys.path.insert(
    0,
    str(
        pathlib.Path(__file__).resolve().parents[2] / "kernelSrcs" /
        "ssd_cutedsl"))
import ssd_prefill_sm12x


def make_inputs(lengths,
                dim,
                dstate,
                seed=1280,
                softplus=True,
                heads=4,
                groups=2):
    rng = np.random.default_rng(seed)
    total = sum(lengths)
    slots = len(lengths) + 3
    half = lambda shape, scale: (rng.standard_normal(shape) * scale).astype(
        np.float16)
    return {
        "x":
        half((total, heads, dim), 0.5),
        "dt":
        half((total, heads), 0.5) if softplus else rng.uniform(
            0.01, 0.2, (total, heads)).astype(np.float16),
        "a":
        -rng.uniform(0.2, 1.0, heads).astype(np.float32),
        "b":
        half((total, groups, dstate), 0.5),
        "c":
        half((total, groups, dstate), 0.5),
        "d":
        half((heads, ), 0.5),
        "bias":
        np.full(heads, -2 if softplus else 0, dtype=np.float16),
        "cu":
        np.asarray([0] + list(np.cumsum(lengths)), dtype=np.int32),
        "indices":
        rng.permutation(slots)[:len(lengths)].astype(np.int32),
        "state":
        half((slots, heads, dim, dstate), 0.125),
    }


def reference(data, softplus=True):
    """Apply S_t = exp(A*dt)*S_(t-1) + dt*x*B^T, y_t = S_t*C + D*x."""
    x, b, c = (data[name].astype(np.float32) for name in ("x", "b", "c"))
    states = data["state"].astype(np.float32).copy()
    out = np.empty_like(x)
    delta = data["dt"].astype(np.float32) + data["bias"].astype(np.float32)
    if softplus:
        delta = np.logaddexp(np.float32(0), delta)
    group = np.arange(x.shape[1]) // (x.shape[1] // b.shape[1])
    for seq, slot in enumerate(data["indices"]):
        for token in range(data["cu"][seq], data["cu"][seq + 1]):
            dt = delta[token]
            states[slot] *= np.exp(data["a"] * dt)[:, None, None]
            states[slot] += dt[:, None, None] * x[token, :, :,
                                                  None] * b[token, group,
                                                            None, :]
            out[token] = np.einsum("hdn,hn->hd", states[slot],
                                   c[token,
                                     group]) + data["d"][:, None] * x[token]
    return out, states


class Invocation:

    def __init__(self,
                 data,
                 chunk_size=64,
                 dim_tile=64,
                 softplus=True,
                 dynamic=False,
                 compiled=None):
        self.stream = cp.cuda.Stream(non_blocking=True)
        with self.stream:
            self.gpu = {
                name: cp.asarray(array)
                for name, array in data.items()
            }
            self.output = cp.full(data["x"].shape, np.nan, dtype=cp.float16)
            self.delta = cp.empty(data["dt"].shape, dtype=cp.float32)
            self.prefix = cp.empty_like(self.delta)
        self.stream.synchronize()
        arrays = [
            self.gpu[key] for key in ("x", "dt", "a", "b", "c", "d", "bias",
                                      "cu", "indices", "state")
        ]
        arrays += [self.output, self.delta, self.prefix]
        self.args = [
            runtime.from_dlpack(array, assumed_align=16) for array in arrays
        ]
        if dynamic:
            for index in (0, 1, 3, 4, 7, 8, 9, 10, 11, 12):
                self.args[index] = self.args[index].mark_compact_shape_dynamic(
                    mode=0, stride_order=tuple(range(arrays[index].ndim)))
        self.args.append(cuda.CUstream(self.stream.ptr))
        kernel = ssd_prefill_sm12x.PackedSSD(data["x"].shape[2],
                                             data["b"].shape[2], chunk_size,
                                             dim_tile, softplus)
        self.compiled = compiled if compiled is not None else cute.compile(
            kernel, *self.args)

    def run(self):
        self.compiled(*self.args)
        self.stream.synchronize()
        return cp.asnumpy(self.output), cp.asnumpy(self.gpu["state"])


def check_result(data, actual, expected):
    output, state = actual
    expected_output, expected_state = expected
    np.testing.assert_allclose(output, expected_output, atol=0.004, rtol=0.005)
    np.testing.assert_allclose(state[data["indices"]],
                               expected_state[data["indices"]],
                               atol=0.002,
                               rtol=0.005)
    preserved = np.ones(state.shape[0], dtype=bool)
    for seq, slot in enumerate(data["indices"]):
        if data["cu"][seq] != data["cu"][seq + 1]:
            preserved[slot] = False
    np.testing.assert_array_equal(state[preserved].view(np.uint16),
                                  data["state"][preserved].view(np.uint16))
    print(
        f"max_abs output={np.max(np.abs(output.astype(np.float32) - expected_output), initial=0):.6g} "
        f"state={np.max(np.abs(state.astype(np.float32) - expected_state), initial=0):.6g}"
    )


@pytest.mark.parametrize("dim,dstate,chunk_size,dim_tile,softplus", [
    (64, 64, 64, 64, True),
    (64, 128, 64, 64, True),
    (128, 64, 64, 64, True),
    (128, 128, 64, 64, True),
    (128, 128, 128, 32, True),
    (64, 128, 64, 32, False),
])
def test_packed_recurrence(dim, dstate, chunk_size, dim_tile, softplus):
    """Ragged tails, empty sequences, grouped heads and indirect nonzero states match recurrence."""
    data = make_inputs([0, 1, 63, 64, 65, 127, 128, 129, 257],
                       dim,
                       dstate,
                       softplus=softplus)
    invocation = Invocation(data, chunk_size, dim_tile, softplus)
    check_result(data, invocation.run(), reference(data, softplus))


def test_continuation():
    """Split prefill resumes from updated state slots across an unaligned chunk boundary."""
    full = make_inputs([259, 134], 64, 128)
    split = [71, 3]
    pieces = []
    state = full["state"].copy()
    for second in (False, True):
        positions = []
        lengths = []
        for seq, first_len in enumerate(split):
            lo, hi = full["cu"][seq:seq + 2]
            begin, end = (lo + first_len, hi) if second else (lo,
                                                              lo + first_len)
            positions.extend(range(begin, end))
            lengths.append(end - begin)
        part = {key: value.copy() for key, value in full.items()}
        for key in ("x", "dt", "b", "c"):
            part[key] = full[key][positions].copy()
        part["cu"] = np.asarray([0] + list(np.cumsum(lengths)), dtype=np.int32)
        part["state"] = state.copy()
        result = Invocation(part).run()
        check_result(part, result, reference(part))
        output, state = result
        pieces.append((positions, output))
    combined = np.empty_like(full["x"])
    for positions, output in pieces:
        combined[positions] = output
    check_result(full, (combined, state), reference(full))


def test_graph_replay_metadata():
    """Captured launches read fresh sequence boundaries, slot mapping and state on each replay."""
    data = make_inputs([1, 127, 65], 64, 64)
    invocation = Invocation(data)
    invocation.run()
    invocation.stream.begin_capture()
    invocation.compiled(*invocation.args)
    graph = invocation.stream.end_capture()
    for lengths, indices in (([1, 127, 65], [2, 4, 0]), ([64, 1,
                                                          128], [4, 0, 2])):
        data["cu"] = np.asarray([0] + list(np.cumsum(lengths)), dtype=np.int32)
        data["indices"] = np.asarray(indices, dtype=np.int32)
        with invocation.stream:
            for key in ("cu", "indices", "state"):
                invocation.gpu[key].set(data[key])
            invocation.output.fill(np.nan)
            graph.launch(invocation.stream)
        invocation.stream.synchronize()
        actual = cp.asnumpy(invocation.output), cp.asnumpy(
            invocation.gpu["state"])
        check_result(data, actual, reference(data))


def test_dynamic_batch_and_tokens():
    """One compiled entry accepts different token, batch and state-pool extents."""
    first = make_inputs([3, 65], 64, 128)
    invocation = Invocation(first, dynamic=True)
    check_result(first, invocation.run(), reference(first))
    second = make_inputs([0, 129, 1, 7], 64, 128)
    reused = Invocation(second, dynamic=True, compiled=invocation.compiled)
    check_result(second, reused.run(), reference(second))


def test_all_empty():
    """A batch with zero packed tokens preserves every state slot exactly."""
    data = make_inputs([0, 0, 0], 64, 64)
    check_result(data, Invocation(data).run(), reference(data))
