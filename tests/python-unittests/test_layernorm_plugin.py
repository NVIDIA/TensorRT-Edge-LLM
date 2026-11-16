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
"""Accuracy tests for ``LayerNormPlugin``.

The plugin keeps the mean, centered-square variance, normalization, and affine
math in FP32, then converts once to the homogeneous FP16/BF16 storage dtype.
The dynamic profiles below cover zero and odd row counts; H=4097 exercises the
masked odd-hidden-size AOT variant and every case uses a nonzero beta.
"""

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              PluginRunner, assert_close, pf_float32)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

_PLUGIN_NAME = "LayerNormPlugin"
_PLUGIN_VERSION = "1"
_EPSILON = 1e-5


@pytest.fixture(autouse=True)
def require_plugin_test_dependencies(request):
    """Prevent the GPU-only L0 suite from passing through dependency skips."""
    if not DEPENDENCIES_AVAILABLE:
        reason = f"TensorRT/torch CUDA not available: {IMPORT_ERROR}"
        if request.config.getoption("--priority") == "l0_python_ut":
            pytest.fail(reason, pytrace=False)
        pytest.skip(reason)

    # LayerNorm is an optional exact-SM CuTe DSL group. Some jobs intentionally
    # omit it (including SM86 jobs that reuse an SM80 artifact), so keep the
    # module registered while cleanly skipping that declared capability gap.
    PluginRunner()
    creator = trt.get_plugin_registry().get_creator(_PLUGIN_NAME,
                                                    _PLUGIN_VERSION, "")
    if creator is None:
        pytest.skip(
            "LayerNormPlugin creator is unavailable: the optional exact-SM "
            "CuTe DSL layernorm artifact group is not enabled in this build")


def _dtype(dtype_name):
    return {
        "fp16": (trt.float16, torch.float16),
        "bf16": (trt.bfloat16, torch.bfloat16),
    }[dtype_name]


def _make_affine(hidden_size, torch_dtype, device="cuda"):
    gamma = torch.linspace(0.75,
                           1.25,
                           hidden_size,
                           dtype=torch.float32,
                           device=device).to(torch_dtype)
    beta = torch.linspace(-0.2,
                          0.3,
                          hidden_size,
                          dtype=torch.float32,
                          device=device).to(torch_dtype)
    assert bool(torch.any(beta != 0))
    return gamma.contiguous(), beta.contiguous()


def _reference(x, gamma, beta):
    if x.numel() == 0:
        return torch.empty_like(x)
    x32 = x.float()
    mean = x32.mean(dim=-1, keepdim=True)
    variance = (x32 - mean).square().mean(dim=-1, keepdim=True)
    return (((x32 - mean) * torch.rsqrt(variance + _EPSILON) * gamma.float()) +
            beta.float()).to(x.dtype)


# Written into the output allocation before every launch. The row past the
# logical extent must still hold it afterwards: that is what detects a kernel
# writing outside the rows it was given, which a multi-row tiling can do
# whenever the row count is not a multiple of its rows-per-CTA.
_SENTINEL = -12345.0

# (mean, stddev) of the input. Zero-mean input cannot see a padding lane that
# wrongly contributes mean**2 to the variance: dropping the mask entirely at
# H=5120 (3072 padding lanes) moves the output by 0.0039 at mean 0 -- under the
# 0.004 tolerance below -- but by 1.6 at mean 1 and 4.4 at mean 10. Real
# LayerNorm inputs, which follow a residual add, are not zero-mean.
_DISTRIBUTIONS = ((0.0, 1.0), (10.0, 1.0), (3.0, 0.1))

_ROW_COUNTS = (0, 1, 2, 3, 5, 7, 9)


@pytest.mark.parametrize("dtype_name", ("fp16", "bf16"))
@pytest.mark.parametrize("hidden_size", (4096, 4097, 5120, 7168, 8192))
def test_layernorm_plugin_parity_zero_and_odd_rows(dtype_name, hidden_size):
    trt_dtype, torch_dtype = _dtype(dtype_name)
    runner = PluginRunner()
    runner.build(
        input_specs=[("x", trt_dtype, (-1, hidden_size)),
                     ("gamma", trt_dtype, (hidden_size, )),
                     ("beta", trt_dtype, (hidden_size, ))],
        output_names=["y"],
        plugin_name=_PLUGIN_NAME,
        plugin_version=_PLUGIN_VERSION,
        plugin_fields=[pf_float32("epsilon", _EPSILON)],
        profiles={"x": ((0, hidden_size), (3, hidden_size), (9, hidden_size))},
        plugin_input_order=["x", "gamma", "beta"],
    )

    gamma, beta = _make_affine(hidden_size, torch_dtype)
    for rows in _ROW_COUNTS:
        for mean, stddev in _DISTRIBUTIONS:
            generator = torch.Generator(
                device="cuda").manual_seed(778000 + hidden_size + rows +
                                           int(mean * 10))

            # One row past the logical extent, so every launch is bounded by a
            # guard row. It also gives rows=0 a nonempty allocation: TensorRT
            # requires a non-null address for every binding, and an empty torch
            # view reports data_ptr()==0.
            allocation_rows = rows + 1
            noise = torch.randn((allocation_rows, hidden_size),
                                generator=generator,
                                dtype=torch.float32,
                                device="cuda")
            # Vary the mean per row so no single scalar offset cancels it.
            offsets = mean * torch.linspace(
                0.5, 1.5, allocation_rows, device="cuda").unsqueeze(1)
            x_storage = (noise * stddev + offsets).to(torch_dtype)
            sentinel_row = torch.full((hidden_size, ),
                                      _SENTINEL,
                                      dtype=torch_dtype,
                                      device="cuda")
            y_storage = sentinel_row.expand(allocation_rows,
                                            hidden_size).clone()

            expected = _reference(x_storage[:rows], gamma, beta)
            runner.execute(
                {
                    "x": x_storage,
                    "gamma": gamma,
                    "beta": beta,
                    "y": y_storage,
                },
                input_shapes={"x": (rows, hidden_size)},
            )

            case = (f"{dtype_name}[rows={rows},H={hidden_size},"
                    f"mean={mean},stddev={stddev}]")
            assert bool(torch.equal(
                y_storage[rows],
                sentinel_row)), (f"{case}: the kernel wrote past row {rows}")
            assert_close(case,
                         expected,
                         y_storage[:rows],
                         atol=0.004 if dtype_name == "fp16" else 0.025,
                         rtol=0.004 if dtype_name == "fp16" else 0.02)
