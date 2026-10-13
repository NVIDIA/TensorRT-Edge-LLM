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
"""TensorRT engine lifecycle and CUDA graph tests for DFlash2 plugins."""

from __future__ import annotations

import numpy as np
import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              PluginRunner, assert_close, pf_int32)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch


@pytest.fixture(autouse=True)
def require_plugin_test_dependencies(request):
    """Do not let the GPU-only L0 suite pass through dependency skips."""
    if DEPENDENCIES_AVAILABLE:
        return
    reason = f"TensorRT/torch CUDA not available: {IMPORT_ERROR}"
    if request.config.getoption("--priority") == "l0_python_ut":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


def _round_trip_engine(runner: PluginRunner) -> None:
    serialized = runner.engine.serialize()
    assert serialized is not None
    runtime = trt.Runtime(runner.logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    runner.engine = engine
    runner.context = context
    # TensorRT objects retain logger/runtime process state; keep the explicit
    # Python owner alive for the rest of the test as well.
    runner._dflash2_test_runtime = runtime


def _bind_tensors(runner: PluginRunner, tensors) -> None:
    for index in range(runner.engine.num_io_tensors):
        name = runner.engine.get_tensor_name(index)
        tensor = tensors[name]
        if runner.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            assert runner.context.set_input_shape(name, tuple(tensor.shape))
        assert runner.context.set_tensor_address(name, tensor.data_ptr())


def _capture_engine(runner: PluginRunner, tensors) -> "torch.cuda.CUDAGraph":
    """Capture one enqueue with fixed tensor addresses and return its graph."""
    _bind_tensors(runner, tensors)
    stream = torch.cuda.current_stream()
    # Warm up the exact binding set outside capture.
    assert runner.context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert runner.context.execute_async_v3(
            torch.cuda.current_stream().cuda_stream)
    graph.replay()
    stream.synchronize()
    return graph


def _dynamic_profile(shape, max_batch=2):
    return ((1, *shape[1:]), (1, *shape[1:]), (max_batch, *shape[1:]))


def _build_grouped_conv(fuse_residual: bool) -> PluginRunner:
    runtime_block, profile_block, kernel, group, hidden = 8, 16, 2, 16, 32
    input_specs = [
        ("hidden", trt.float16, (-1, runtime_block, hidden)),
        ("delta", trt.float16, (-1, runtime_block, kernel, hidden // group)),
    ]
    if fuse_residual:
        input_specs.append(
            ("residual", trt.float32, (-1, runtime_block, hidden)))
    profiles = {
        name: _dynamic_profile(shape)
        for name, _, shape in input_specs
    }
    input_order = ["hidden", "delta", "base_kernel"]
    if fuse_residual:
        input_order.append("residual")
    return PluginRunner().build(input_specs=input_specs,
                                constant_specs=[
                                    ("base_kernel", trt.float16, (kernel,
                                                                  hidden),
                                     np.ones((kernel, hidden),
                                             dtype=np.float16)),
                                ],
                                plugin_input_order=input_order,
                                output_names=["output"],
                                plugin_name="DFlash2GroupedDynamicConvPlugin",
                                plugin_version="1",
                                plugin_fields=[
                                    pf_int32("block_size", profile_block),
                                    pf_int32("kernel_size", kernel),
                                    pf_int32("group_size", group),
                                    pf_int32("fuse_residual",
                                             int(fuse_residual)),
                                ],
                                profiles=profiles)


def _build_grouped_conv_with_conservative_block_bound() -> PluginRunner:
    profile_block, kernel, group, hidden = 16, 2, 16, 32
    input_specs = [
        ("hidden", trt.float16, (-1, -1, hidden)),
        ("delta", trt.float16, (-1, -1, kernel, hidden // group)),
    ]
    profiles = {
        "hidden": ((1, 1, hidden), (2, profile_block, hidden),
                   (2, 2 * profile_block, hidden)),
        "delta": ((1, 1, kernel, hidden // group), (2, profile_block, kernel,
                                                    hidden // group),
                  (2, 2 * profile_block, kernel, hidden // group)),
    }
    return PluginRunner().build(
        input_specs=input_specs,
        constant_specs=[
            ("base_kernel", trt.float16, (kernel, hidden),
             np.ones((kernel, hidden), dtype=np.float16)),
        ],
        plugin_input_order=["hidden", "delta", "base_kernel"],
        output_names=["output"],
        plugin_name="DFlash2GroupedDynamicConvPlugin",
        plugin_version="1",
        plugin_fields=[
            pf_int32("block_size", profile_block),
            pf_int32("kernel_size", kernel),
            pf_int32("group_size", group),
            pf_int32("fuse_residual", 0),
        ],
        profiles=profiles)


def _grouped_conv_reference(hidden, residual=None):
    result = hidden.float().clone()
    result[:, 1:] += hidden[:, :-1].float()
    if residual is not None:
        result += residual
    return result


def test_grouped_dynamic_conv_accepts_conservative_profile_block_bound():
    batch, block, kernel, group, hidden_size = 2, 16, 2, 16, 32
    runner = _build_grouped_conv_with_conservative_block_bound()
    tensors = {
        "hidden":
        torch.ones((batch, block, hidden_size),
                   device="cuda",
                   dtype=torch.float16),
        "delta":
        torch.zeros((batch, block, kernel, hidden_size // group),
                    device="cuda",
                    dtype=torch.float16),
        "output":
        torch.empty((batch, block, hidden_size),
                    device="cuda",
                    dtype=torch.float16),
    }

    runner.execute(tensors)

    assert_close("grouped-conv conservative profile bound",
                 _grouped_conv_reference(tensors["hidden"]), tensors["output"])

    oversized = {
        "hidden":
        torch.ones((batch, 2 * block, hidden_size),
                   device="cuda",
                   dtype=torch.float16),
        "delta":
        torch.zeros((batch, 2 * block, kernel, hidden_size // group),
                    device="cuda",
                    dtype=torch.float16),
        "output":
        torch.empty((batch, 2 * block, hidden_size),
                    device="cuda",
                    dtype=torch.float16),
    }
    with pytest.raises(RuntimeError, match="execute_async_v3 returned False"):
        runner.execute(oversized)


@pytest.mark.parametrize("fuse_residual", [False, True], ids=["pre", "post"])
def test_grouped_dynamic_conv_engine_roundtrip_graph_replay_and_block_boundaries(
        fuse_residual):
    batch, block, kernel, group, hidden_size = 2, 8, 2, 16, 32
    runner = _build_grouped_conv(fuse_residual)
    _round_trip_engine(runner)

    hidden = torch.ones((batch, block, hidden_size),
                        device="cuda",
                        dtype=torch.float16)
    hidden[1].fill_(100.0)
    residual = torch.full(
        (batch, block, hidden_size), 10.0, device="cuda",
        dtype=torch.float32) if fuse_residual else None
    tensors = {
        "hidden":
        hidden,
        "delta":
        torch.zeros((batch, block, kernel, hidden_size // group),
                    device="cuda",
                    dtype=torch.float16),
        "output":
        torch.empty((batch, block, hidden_size),
                    device="cuda",
                    dtype=torch.float32 if fuse_residual else torch.float16),
    }
    if residual is not None:
        tensors["residual"] = residual

    runner.execute(tensors)
    expected = _grouped_conv_reference(hidden, residual)
    assert_close("grouped-conv eager after engine roundtrip", expected,
                 tensors["output"])
    # The first token of row 1 starts a new logical block. A cross-block read
    # would add row 0's final token here.
    assert_close("grouped-conv block boundary", expected[1, 0],
                 tensors["output"][1, 0])

    graph = _capture_engine(runner, tensors)
    first_output = tensors["output"].clone()
    hidden.add_(2.0)
    if residual is not None:
        residual.add_(7.0)
    tensors["output"].zero_()
    graph.replay()
    torch.cuda.current_stream().synchronize()

    replay_expected = _grouped_conv_reference(hidden, residual)
    assert not torch.equal(tensors["output"], first_output)
    assert_close("grouped-conv graph replay", replay_expected,
                 tensors["output"])
    assert_close("grouped-conv replay block boundary", replay_expected[1, 0],
                 tensors["output"][1, 0])
