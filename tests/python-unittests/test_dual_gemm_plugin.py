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
"""TensorRT lifecycle and accuracy tests for the SwiGLU dual GEMM plugin (Blackwell datacenter / Thor).

``DualGemmPlugin`` computes H = silu(X @ Wg^T) * (X @ Wu^T) from one packed gate/up weight
(dual_gemm_blackwell layout: per 128-wide output tile and 64-wide K tile, the Wg tile then the Wu tile).
Each test builds one dynamic-profile engine and executes the token counts that hit all three AOT variants
(decode M <= 64, merged-N prefill 64 < M <= 2048, grouped-raster large M), comparing against an fp32 torch
reference of the unpacked weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              PluginRunner, assert_close, pf_int32)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

_PLUGIN_NAME = "DualGemmPlugin"
_PLUGIN_VERSION = "1"
_ACTIVATION_SILU = 0
_TILE_N = 128
_TILE_K = 64


@pytest.fixture(autouse=True)
def require_plugin_test_dependencies(request):
    """Prevent the GPU-only L0 suite from passing through dependency skips."""
    if DEPENDENCIES_AVAILABLE:
        return
    reason = f"TensorRT/torch CUDA not available: {IMPORT_ERROR}"
    if request.config.getoption("--priority") == "l0_python_ut":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


def _require_blackwell():
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor not in (100, 101, 103, 110):
        pytest.skip(
            "DualGemmPlugin ships Blackwell datacenter / Thor kernels only")


@dataclass(frozen=True)
class DualGemmCase:
    name: str
    inter_size: int
    hidden_size: int
    max_m: int
    token_counts: tuple


_CASES = [
    # Small shapes: every variant, cheap.
    DualGemmCase("n512_k256", 512, 256, 4096,
                 (1, 8, 64, 65, 1024, 2048, 2049, 4096)),
    # Qwen3-8B MLP shape at the M ranges of the three variants (decode / merged-N prefill / large).
    DualGemmCase("qwen3_8b", 12288, 4096, 4096, (1, 64, 1024, 4096)),
]


def pack_gate_up_weights(wg: "torch.Tensor",
                         wu: "torch.Tensor") -> "torch.Tensor":
    """Mirror of kernelSrcs/gemm_cutedsl/dual_gemm_blackwell.py::pack_gate_up_weights (N, K) -> (2N, K)."""
    n, k = wg.shape
    assert n % _TILE_N == 0 and k % _TILE_K == 0
    g = wg.reshape(n // _TILE_N, _TILE_N, k // _TILE_K,
                   _TILE_K).permute(0, 2, 1, 3)
    u = wu.reshape(n // _TILE_N, _TILE_N, k // _TILE_K,
                   _TILE_K).permute(0, 2, 1, 3)
    return torch.stack([g, u], dim=2).reshape(2 * n, k).contiguous()


def _make_weights(case: DualGemmCase, generator: "torch.Generator"):
    scale = 1.0 / (case.hidden_size**0.5)
    wg = ((torch.rand((case.inter_size, case.hidden_size),
                      generator=generator,
                      dtype=torch.float32) * 2 - 1) * scale).to(torch.float16)
    wu = ((torch.rand((case.inter_size, case.hidden_size),
                      generator=generator,
                      dtype=torch.float32) * 2 - 1) * scale).to(torch.float16)
    return wg, wu


def _build_runner(case: DualGemmCase, packed: "torch.Tensor",
                  rank: int) -> PluginRunner:
    runner = PluginRunner()
    if rank == 2:
        shape = (-1, case.hidden_size)
        profile = ((1, case.hidden_size), (min(64,
                                               case.max_m), case.hidden_size),
                   (case.max_m, case.hidden_size))
    else:
        shape = (1, -1, case.hidden_size)
        profile = ((1, 1, case.hidden_size), (1, min(64, case.max_m),
                                              case.hidden_size),
                   (1, case.max_m, case.hidden_size))
    runner.build(
        input_specs=[("x", trt.float16, shape)],
        output_names=["h"],
        plugin_name=_PLUGIN_NAME,
        plugin_version=_PLUGIN_VERSION,
        plugin_fields=[
            pf_int32("inter_size", case.inter_size),
            pf_int32("hidden_size", case.hidden_size),
            pf_int32("activation", _ACTIVATION_SILU),
        ],
        profiles={"x": profile},
        constant_specs=[("w_packed", trt.float16, tuple(packed.shape), packed)
                        ],
        plugin_input_order=["x", "w_packed"],
    )
    return runner


def _reference(x: "torch.Tensor", wg: "torch.Tensor",
               wu: "torch.Tensor") -> "torch.Tensor":
    x32 = x.to(torch.float32)
    gate = x32 @ wg.to(torch.float32).t().to(x.device)
    up = x32 @ wu.to(torch.float32).t().to(x.device)
    return torch.nn.functional.silu(gate) * up


def _run_case(case: DualGemmCase, rank: int):
    _require_blackwell()
    generator = torch.Generator().manual_seed(6704525 + case.inter_size)
    wg, wu = _make_weights(case, generator)
    runner = _build_runner(case, pack_gate_up_weights(wg, wu), rank)
    for num_rows in case.token_counts:
        if num_rows > case.max_m:
            continue
        leading = (num_rows, ) if rank == 2 else (1, num_rows)
        x = ((torch.rand(
            (*leading, case.hidden_size),
            generator=generator,
            dtype=torch.float32) * 2 - 1).to(torch.float16).to("cuda"))
        h = torch.empty((*leading, case.inter_size),
                        dtype=torch.float16,
                        device="cuda")
        runner.execute({"x": x, "h": h})
        expected = _reference(x.reshape(-1, case.hidden_size), wg,
                              wu).reshape(*leading, case.inter_size)
        assert bool(torch.isfinite(h.to(torch.float32)).all())
        assert_close(f"{case.name}[rank={rank}, M={num_rows}]",
                     expected,
                     h.to(torch.float32),
                     atol=2e-2,
                     rtol=2e-2)


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("rank", [2, 3])
def test_dual_gemm_all_variants(case, rank):
    _run_case(case, rank)


def test_dual_gemm_engine_roundtrip():
    """Serialize / deserialize the engine and execute a decode and a prefill token count again."""
    _require_blackwell()
    case = _CASES[0]
    generator = torch.Generator().manual_seed(4711)
    wg, wu = _make_weights(case, generator)
    runner = _build_runner(case, pack_gate_up_weights(wg, wu), rank=2)
    serialized = runner.engine.serialize()
    assert serialized is not None
    runtime = trt.Runtime(runner.logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    assert engine is not None
    runner.engine = engine
    runner.context = engine.create_execution_context()
    assert runner.context is not None
    for num_rows in (8, 1024):
        x = ((torch.rand(
            (num_rows, case.hidden_size),
            generator=generator,
            dtype=torch.float32) * 2 - 1).to(torch.float16).to("cuda"))
        h = torch.empty((num_rows, case.inter_size),
                        dtype=torch.float16,
                        device="cuda")
        runner.execute({"x": x, "h": h})
        assert_close(f"roundtrip[M={num_rows}]",
                     _reference(x, wg, wu),
                     h.to(torch.float32),
                     atol=2e-2,
                     rtol=2e-2)


@pytest.mark.parametrize("inter_size,hidden_size", [(500, 256), (512, 250)])
def test_build_rejects_misaligned_dimensions(inter_size, hidden_size):
    """N must be a multiple of 128 and K a multiple of 64 (packed weight tiles)."""
    registry = trt.get_plugin_registry()
    PluginRunner()  # loads the plugin library
    creator = registry.get_creator(_PLUGIN_NAME, _PLUGIN_VERSION, "")
    assert creator is not None
    fields = trt.PluginFieldCollection([
        pf_int32("inter_size", inter_size),
        pf_int32("hidden_size", hidden_size),
        pf_int32("activation", _ACTIVATION_SILU),
    ])
    assert creator.create_plugin("dual_gemm_bad", fields,
                                 trt.TensorRTPhase.BUILD) is None
