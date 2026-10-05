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
"""TensorRT lifecycle and accuracy tests for the dense NVFP4 (W4A16) GEMM plugin.

The dense ``Nvfp4A16GemmPlugin`` is a thin wrapper over the MoE Marlin
BF16xE2M1 kernel driven with a single expert and top_k = 1. Each positive test
builds one dynamic-profile engine, round-trips its serialization, and executes
both a decode row count (M == 1, Marlin block 8) and a prefill row count
(M > 1, Marlin block 32).

Weights use the same tile-constant Marlin layout as the MoE plugin test: codes
and E4M3 block scales are constant inside each 16x64 Marlin tile, so the Marlin
permutation is a no-op and the fixture stays independent of the checkpoint
repacking code. The reference dequantizes the original E2M1 codes with their
E4M3 block scales and per-tensor global scale, then does a dense fp32 matmul.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              PluginRunner, assert_close, pf_int32)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

_PLUGIN_NAME = "Nvfp4A16GemmPlugin"
_PLUGIN_VERSION = "1"
_FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@pytest.fixture(autouse=True)
def require_plugin_test_dependencies(request):
    """Prevent the GPU-only L0 suite from passing through dependency skips."""
    if DEPENDENCIES_AVAILABLE:
        return
    reason = f"TensorRT/torch CUDA not available: {IMPORT_ERROR}"
    if request.config.getoption("--priority") == "l0_python_ut":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


@dataclass(frozen=True)
class GemmCase:
    name: str
    gemm_n: int
    gemm_k: int
    max_m: int = 128  # plugin attribute (0 == auto: size from the profile)

    @property
    def profile_max(self) -> int:
        # Optimization-profile max token count; independent of the (possibly
        # auto == 0) plugin attribute.
        return self.max_m if self.max_m > 0 else 128


# gemm_n must be a multiple of 128 (Marlin thread_n) and of 64 (tile-constant
# packing); gemm_k must be a multiple of 64 (Marlin thread_k) and 16 (group).
_CASES = [
    GemmCase(name="n128_k128", gemm_n=128, gemm_k=128),
    GemmCase(name="n256_k64", gemm_n=256, gemm_k=64),
    GemmCase(name="n256_k256", gemm_n=256, gemm_k=256),
    GemmCase(name="k_proj_like_n256_k2688", gemm_n=256, gemm_k=2688),
    GemmCase(name="o_proj_like_n2688_k4096",
             gemm_n=2688,
             gemm_k=4096,
             max_m=32),
]


def _make_tile_constant_weight(size_n: int, size_k: int, global_scale: float,
                               generator: "torch.Generator"):
    """Build one synthetic dense projection already encoded in Marlin layout.

    Returns (packed_qweights [1, K/16, 8*N] int8, packed_block_scales
    [1, K/16, N] int8, packed_global_scale [1] bf16, original_codes [1, N, K]
    uint8, original_block_scales [1, N, K/16] f8e4m3). The leading dim of 1 is
    the degenerate expert dimension the MoE kernel shape checks expect.
    """
    assert size_n % 64 == 0 and size_k % 16 == 0
    num_n_tiles = size_n // 64
    num_k_tiles = size_k // 16
    tile_codes = torch.randint(0,
                               16, (1, num_n_tiles, num_k_tiles),
                               generator=generator,
                               dtype=torch.uint8)
    levels = torch.tensor([2.0**-8, 0.25, 0.5], dtype=torch.float32)
    indices = torch.randint(0,
                            levels.numel(),
                            tile_codes.shape,
                            generator=generator,
                            dtype=torch.int64)
    tile_scales = levels[indices].to(torch.float8_e4m3fn)

    original_codes = tile_codes.repeat_interleave(64, dim=1).repeat_interleave(
        16, dim=2).contiguous()
    original_scales = tile_scales.repeat_interleave(64, dim=1).contiguous()

    packed_bytes = tile_codes | (tile_codes << 4)
    packed_qweights = packed_bytes.permute(
        0, 2,
        1).contiguous().repeat_interleave(8 * 64,
                                          dim=2).contiguous().view(torch.int8)
    packed_block_scales = tile_scales.permute(
        0, 2,
        1).contiguous().repeat_interleave(64,
                                          dim=2).contiguous().view(torch.int8)
    packed_global_scale = torch.tensor([global_scale],
                                       dtype=torch.float16) * float(2**7)
    return (packed_qweights, packed_block_scales, packed_global_scale,
            original_codes, original_scales)


def _dequantize_original(codes: "torch.Tensor", block_scales: "torch.Tensor",
                         global_scale: float) -> "torch.Tensor":
    """Dequantize original E2M1 codes to a dense [N, K] fp32 weight."""
    levels = torch.tensor(_FP4_LEVELS, dtype=torch.float32)
    magnitudes = levels[(codes & 0x7).to(torch.int64)]
    values = torch.where((codes & 0x8) != 0, -magnitudes, magnitudes)
    expanded_scales = block_scales.to(torch.float32).repeat_interleave(16,
                                                                       dim=2)
    dense = values * expanded_scales * global_scale
    return dense[0]  # drop the degenerate expert dim -> [N, K]


def _plugin_fields(case: GemmCase):
    return [
        pf_int32("gemm_n", case.gemm_n),
        pf_int32("gemm_k", case.gemm_k),
        pf_int32("max_m", case.max_m),
    ]


def _io_specs(case: GemmCase, rank: int = 3):
    return [
        ("activation", trt.float16, (-1, case.gemm_k) if rank == 2 else
         (-1, -1, case.gemm_k)),
        ("qweights", trt.int8, (1, case.gemm_k // 16, 8 * case.gemm_n)),
        ("block_scales", trt.int8, (1, case.gemm_k // 16, case.gemm_n)),
        ("global_scale", trt.float16, (1, )),
    ]


def _profiles(case: GemmCase, input_specs):
    profiles = {}
    for name, _, shape in input_specs:
        if name == "activation":
            profiles[name] = (((1, case.gemm_k), (1, case.gemm_k),
                               (case.profile_max,
                                case.gemm_k)) if len(shape) == 2 else
                              ((1, 1, case.gemm_k), (1, 1, case.gemm_k),
                               (1, case.profile_max, case.gemm_k)))
        else:
            profiles[name] = (shape, shape, shape)
    return profiles


def _build_runner(case: GemmCase, rank: int = 3) -> PluginRunner:
    runner = PluginRunner()
    input_specs = _io_specs(case, rank)
    runner.build(input_specs=input_specs,
                 output_names=["output"],
                 plugin_name=_PLUGIN_NAME,
                 plugin_version=_PLUGIN_VERSION,
                 plugin_fields=_plugin_fields(case),
                 profiles=_profiles(case, input_specs))
    return runner


def _round_trip_engine(runner: PluginRunner) -> None:
    """Serialize and deserialize the built engine a second time."""
    serialized = runner.engine.serialize()
    assert serialized is not None
    runtime = trt.Runtime(runner.logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    runner.engine = engine
    runner.context = context
    runner._nvfp4_test_runtime = runtime


def _execute_case(case: GemmCase, rank: int = 3) -> None:
    generator = torch.Generator().manual_seed(4711 + case.gemm_n + case.gemm_k)
    global_scale = 2.0**-3
    (qweights, block_scales, global_scale_t, codes,
     scales) = _make_tile_constant_weight(case.gemm_n, case.gemm_k,
                                          global_scale, generator)
    dense_w = _dequantize_original(codes, scales, global_scale).to("cuda")

    runner = _build_runner(case, rank)
    _round_trip_engine(runner)
    static_inputs = {
        "qweights": qweights.to("cuda").contiguous(),
        "block_scales": block_scales.to("cuda").contiguous(),
        "global_scale": global_scale_t.to("cuda").contiguous(),
    }

    row_counts = [
        m for m in (1, 33, case.profile_max) if m <= case.profile_max
    ]
    for num_rows in [*sorted(set(row_counts)), 1]:
        leading_shape = (num_rows, ) if rank == 2 else (1, num_rows)
        hidden = (torch.randn(
            (*leading_shape, case.gemm_k),
            generator=generator,
            dtype=torch.float32) * 0.25).to(torch.float16).to("cuda")
        # Plugin computes out[m, n] = sum_k act[m, k] * W_dequant[n, k].
        expected = hidden.reshape(num_rows, case.gemm_k).to(
            torch.float32) @ dense_w.t()
        expected = expected.reshape(*leading_shape, case.gemm_n)
        actual = torch.empty((*leading_shape, case.gemm_n),
                             dtype=torch.float16,
                             device="cuda")
        tensors = {"activation": hidden, "output": actual, **static_inputs}
        runner.execute(tensors)

        assert bool(torch.isfinite(actual.to(torch.float32)).all())
        assert_close(f"{case.name}[M={num_rows}]",
                     expected,
                     actual,
                     atol=0.05,
                     rtol=0.02,
                     cos_threshold=0.99)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_dense_nvfp4_a16_gemm_decode_and_prefill(case, rank):
    _execute_case(case, rank)


@pytest.mark.parametrize("rank", [2, 3])
def test_dense_nvfp4_a16_gemm_auto_max_m(rank):
    # max_m == 0 asks the plugin to size its workspace from the optimization
    # profile instead of an explicit token cap.
    _execute_case(GemmCase(name="auto", gemm_n=256, gemm_k=256, max_m=0), rank)


@pytest.mark.parametrize(
    "gemm_n,gemm_k",
    [
        (130, 128),  # N not a multiple of 128
        (128, 100),  # K not a multiple of 64
        (128, 0),  # non-positive K
    ])
def test_build_rejects_misaligned_dimensions(gemm_n, gemm_k):
    # validateAttributes() throws for these, so the creator returns a null
    # plugin instead of a valid one.
    PluginRunner()  # ensures the plugin library is loaded and registered
    registry = trt.get_plugin_registry()
    creator = registry.get_creator(_PLUGIN_NAME, _PLUGIN_VERSION, "")
    assert creator is not None
    fields = [
        pf_int32("gemm_n", gemm_n),
        pf_int32("gemm_k", gemm_k),
        pf_int32("max_m", 128),
    ]
    fc = trt.PluginFieldCollection(fields)
    plugin = creator.create_plugin(_PLUGIN_NAME, fc, trt.TensorRTPhase.BUILD)
    assert plugin is None
