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
"""Build and numerical regressions for native NVFP4 linear projections."""

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from experimental.builder.ops import backend


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("shape,shapes,width", [
    ((-1, 1, 2048), [(1, 1, 2048)] * 3, 6144),
    ((-1, -1, 128), [(1, 1, 128), (2, 3, 128), (2, 7, 128)], 128),
    ((-1, 2, 1, 128), [(1, 2, 1, 128), (2, 2, 1, 128), (4, 2, 1, 128)], 128),
    ((-1, 128), [(1, 128), (7, 128), (8, 128)], 128),
])
def test_nvfp4_linear_build_and_execute(shape, shapes, width, with_bias):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(
    )[0] < 10:
        pytest.skip("NVFP4 requires Blackwell or newer")
    rng = np.random.default_rng(216)
    inner = shape[-1]
    packed = rng.integers(0, 256, (width, inner // 2), dtype=np.uint8)
    scales = rng.choice(np.array([0x30, 0x38, 0x40], np.uint8),
                        (width, inner // 16))
    bias = (rng.integers(-4, 5, width) * .25).astype(np.float16)
    raw = dict(packed=packed,
               weight_scale=scales,
               weight_scale_2=.03125,
               input_scale=.0625)
    if with_bias:
        raw["bias"] = bias
    levels = np.array(
        [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
        np.float32)
    dense = np.empty((width, inner), np.float32)
    dense[:, 0::2] = levels[packed & 15]
    dense[:, 1::2] = levels[packed >> 4]
    weight_scales = np.where(scales == 0x30, .5,
                             np.where(scales == 0x38, 1., 2.))
    dense = (dense.reshape(width, inner // 16, 16) *
             weight_scales[:, :, None] * .03125).reshape(width, inner)
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    net = backend.Net(builder, network)
    x = net.add_input("x", trt.float16, shape)
    y = net.cast(net.nvfp4_linear(x, raw, len(shape)), trt.float32)
    y.name = "y"
    network.mark_output(y)
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape("x", *shapes)
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for runtime_shape in dict.fromkeys(shapes):
            # Each block has an exact FP4 scale and a maximum of 6 * scale,
            # so activation QDQ is exact and the oracle isolates linear layout.
            values = rng.choice(levels,
                                runtime_shape).reshape(-1, inner // 16, 16)
            values[:, :, 0] = 6
            values *= rng.choice(np.array([.5, 1., 2.], np.float32),
                                 values.shape[:-1])[..., None]
            values = values.reshape(runtime_shape).astype(np.float16)
            with torch.cuda.stream(stream):
                inputs = torch.tensor(values, device="cuda")
                weights = torch.tensor(dense,
                                       device="cuda",
                                       dtype=torch.float32)
                output = torch.empty((*runtime_shape[:-1], width),
                                     device="cuda",
                                     dtype=torch.float32)
                product = inputs.float() @ weights.T
                expected = product.half().float()
                if with_bias:
                    bias_tensor = torch.tensor(bias, device="cuda")
                    expected = (product + bias_tensor.float()).half().float()
                    separate = (product.half() + bias_tensor).float()
                else:
                    separate = expected
                assert context.set_input_shape("x", runtime_shape)
                assert context.set_tensor_address("x", inputs.data_ptr())
                assert context.set_tensor_address("y", output.data_ptr())
                assert context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            assert torch.isfinite(output).all()
            # TensorRT may fuse bias into GEMM or round before the bias add.
            error = torch.minimum((output - expected).abs(),
                                  (output - separate).abs())
            assert torch.count_nonzero(error).item() == 0
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
