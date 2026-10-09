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
"""GPU regression coverage for speculative decoding log probabilities."""

import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from experimental.builder.ops import backend


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape,axis", [
    ((1, 1, 32), 2),
    ((2, 4, 32), 2),
    ((2, 32, 4), 1),
    ((1, 4, 151936), 2),
    ((1, 4, 248320), 2),
])
def test_log_softmax_build_and_execute(dtype, shape, axis):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    net = backend.Net(builder, network)
    trt_dtype = trt.float32 if dtype == torch.float32 else trt.float16
    x = net.add_input("x", trt_dtype, (-1, ) + shape[1:])
    y = net.log_softmax(x, axis)
    y.name = "y"
    network.mark_output(y)
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape("x", (1, ) + shape[1:], shape, (4, ) + shape[1:])
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    generator = torch.Generator(device="cuda").manual_seed(223)
    for batch in (1, shape[0], 4):
        runtime_shape = (batch, ) + shape[1:]
        for scale in (0.0, 1.0, 100.0):
            with torch.cuda.stream(stream):
                values = torch.randn(runtime_shape,
                                     generator=generator,
                                     device="cuda",
                                     dtype=dtype) * scale
                output = torch.empty_like(values)
                assert context.set_input_shape("x", runtime_shape)
                assert context.set_tensor_address("x", values.data_ptr())
                assert context.set_tensor_address("y", output.data_ptr())
                assert context.execute_async_v3(stream.cuda_stream)
                expected = torch.log_softmax(values.float(),
                                             dim=axis).to(dtype)
            stream.synchronize()
            assert torch.isfinite(output).all()
            tolerance = 2e-3 if dtype == torch.float16 else 2e-5
            torch.testing.assert_close(output,
                                       expected,
                                       rtol=tolerance,
                                       atol=tolerance)
