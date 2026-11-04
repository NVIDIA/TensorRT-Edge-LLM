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
"""Dynamic-prefill accuracy and lifecycle coverage for AlpamayoFusedGateUpPlugin."""

import pytest
import test_plugin_base as plugin_base

if plugin_base.DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

_PLUGIN_NAME = "AlpamayoFusedGateUpPlugin"
_HIDDEN = 2048
_INTERMEDIATE = 6144


@pytest.fixture(autouse=True)
def require_plugin_test_dependencies(request):
    """Require the GPU dependencies when this module is selected by L0."""
    if plugin_base.DEPENDENCIES_AVAILABLE:
        return
    reason = f"TensorRT/torch CUDA not available: {plugin_base.IMPORT_ERROR}"
    if request.config.getoption("--priority") == "l0_python_ut":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


@pytest.fixture
def sm110():
    """Execute positive kernel coverage only on the supported architecture."""
    if plugin_base._device_sm() != 110:
        pytest.skip("AlpamayoFusedGateUpPlugin requires SM110")


def _build_runner(expect_unsupported=False,
                  hidden=_HIDDEN,
                  weight_rows=2 * _INTERMEDIATE,
                  scale_shape=(1, ),
                  activation_type=None,
                  min_tokens=2048,
                  max_tokens=4096):
    activation_type = trt.fp8 if activation_type is None else activation_type
    runner = plugin_base.PluginRunner()
    runner.build(input_specs=[("x", activation_type, (-1, hidden)),
                              ("w", trt.fp8, (weight_rows, _HIDDEN)),
                              ("gate_decode", trt.float32, scale_shape),
                              ("up_encode", trt.float32, (1, ))],
                 output_names=["y"],
                 plugin_name=_PLUGIN_NAME,
                 plugin_version="1",
                 plugin_fields=[],
                 profiles={
                     "x": ((min_tokens, hidden), (3006, hidden), (max_tokens,
                                                                  hidden))
                 },
                 expect_unsupported=expect_unsupported)
    return runner


def _fixture():
    generator = torch.Generator().manual_seed(1743)
    x = (torch.randn((4096, _HIDDEN), generator=generator) * 0.25).to(
        device="cuda", dtype=torch.float8_e4m3fn)
    gate = (torch.randn((_INTERMEDIATE, _HIDDEN), generator=generator) *
            0.125).to(device="cuda", dtype=torch.float8_e4m3fn)
    up = (torch.randn((_INTERMEDIATE, _HIDDEN), generator=generator) *
          0.0625).to(device="cuda", dtype=torch.float8_e4m3fn)
    # The reference retains the original matrices; only the plugin sees packed rows.
    packed = torch.stack(
        (gate.reshape(24, 2, 128, _HIDDEN), up.reshape(24, 2, 128, _HIDDEN)),
        dim=2).reshape(2 * _INTERMEDIATE, _HIDDEN).contiguous()
    return x, gate, up, packed


def _reference(x, gate, up, gate_decode, up_encode):
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        gate_values = (x.float() @ gate.float().t()) * gate_decode
        up_values = (x.float() @ up.float().t()) * up_encode
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    values = torch.nn.functional.silu(gate_values) * up_values
    return values.clamp(-448, 448).to(torch.float8_e4m3fn).float()


def _assert_output(actual, expected):
    actual = actual.float()
    assert bool(torch.isfinite(actual).all())
    # Allow one E4M3 step for the fast tanh SiLU at quantization boundaries.
    torch.testing.assert_close(actual, expected, rtol=0.125, atol=2**-9)


def test_dynamic_prefill_accuracy_and_serialization(sm110):
    """A serialized engine handles changing M, packed gate/up scales, and partial tiles."""
    runner = _build_runner()
    serialized = runner.engine.serialize()
    assert serialized is not None
    runtime = trt.Runtime(runner.logger)
    runner.engine = runtime.deserialize_cuda_engine(serialized)
    assert runner.engine is not None
    runner.context = runner.engine.create_execution_context()
    assert runner.context is not None
    x, gate, up, packed = _fixture()
    gate_decode = torch.tensor([0.5], device="cuda")
    up_encode = torch.tensor([1.5], device="cuda")
    for m in (2048, 2049, 3006, 4096, 2048):
        actual = torch.empty((m, _INTERMEDIATE),
                             dtype=torch.float8_e4m3fn,
                             device="cuda")
        actual.view(torch.uint8).fill_(0x7F)
        runner.execute({
            "x": x[:m],
            "w": packed,
            "gate_decode": gate_decode,
            "up_encode": up_encode,
            "y": actual
        })
        expected = _reference(x[:m], gate, up, 0.5, 1.5)
        _assert_output(actual, expected)


def test_prefill_cuda_graph_replay(sm110):
    """Graph capture and replay read current inputs and scales on the supplied stream."""
    runner = _build_runner()
    x, gate, up, packed = _fixture()
    x = x[:2049].clone()
    actual = torch.empty((2049, _INTERMEDIATE),
                         dtype=torch.float8_e4m3fn,
                         device="cuda")
    gate_decode = torch.tensor([0.5], device="cuda")
    up_encode = torch.tensor([1.5], device="cuda")
    tensors = {
        "x": x,
        "w": packed,
        "gate_decode": gate_decode,
        "up_encode": up_encode,
        "y": actual
    }
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner.execute(tensors)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner.execute(tensors, synchronize=False)
        x.copy_((-x.float()).to(torch.float8_e4m3fn))
        gate_decode.fill_(0.25)
        up_encode.fill_(0.75)
        actual.view(torch.uint8).fill_(0x7F)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    expected = _reference(x, gate, up, 0.25, 0.75)
    _assert_output(actual, expected)


@pytest.mark.parametrize("invalid", [
    "hidden", "weights", "scale", "dtype", "empty_profile", "overflow_profile"
])
def test_build_rejects_invalid_contract(sm110, invalid):
    """The builder rejects unsupported dimensions and input types on SM110."""
    kwargs = {
        "hidden": {
            "hidden": _HIDDEN + 1
        },
        "weights": {
            "weight_rows": 2 * _INTERMEDIATE - 1
        },
        "scale": {
            "scale_shape": (2, )
        },
        "dtype": {
            "activation_type": trt.float16
        },
        "empty_profile": {
            "min_tokens": 0
        },
        "overflow_profile": {
            "max_tokens": 2**31 - 1
        },
    }[invalid]
    with pytest.raises(plugin_base.PluginUnsupportedError):
        _build_runner(expect_unsupported=True, **kwargs)


def test_build_on_unsupported_device_is_rejected():
    """A non-SM110 device rejects the plugin before launching device code."""
    if plugin_base._device_sm() == 110:
        pytest.skip("Negative device gate applies only away from SM110")
    with pytest.raises(plugin_base.PluginUnsupportedError):
        _build_runner(expect_unsupported=True)
