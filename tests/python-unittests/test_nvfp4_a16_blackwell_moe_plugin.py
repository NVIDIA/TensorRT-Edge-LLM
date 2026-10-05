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
"""``Nvfp4A16BlackwellMoePlugin`` (Thor SM110 W4A16 routed MoE).

The positive tests build one dynamic-profile engine, round-trip its
serialization and execute one token, the smallest grouped-GEMM token set
(T=9), and a longer token set (T=64). Weights are real ModelOpt-style NVFP4
tensors repacked with ``repack_nvfp4_a16_blackwell_moe_experts`` (random
codes, no tile-constant trick) and the reference dequantizes the original
codes with ``decode_modelopt_nvfp4``. They run on SM110 only; the attribute
and dtype rejection tests run on any GPU with the plugin library.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from typing import Dict

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              PluginRunner, PluginUnsupportedError,
                              assert_close, pf_float32, pf_int32)

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

    from tensorrt_edgellm.checkpoint.repacking import (
        decode_modelopt_nvfp4, repack_nvfp4_a16_blackwell_moe_experts)


@pytest.fixture(autouse=True)
def require_plugin_test_dependencies(request):
    if DEPENDENCIES_AVAILABLE:
        return
    reason = f"TensorRT/torch CUDA not available: {IMPORT_ERROR}"
    if request.config.getoption("--priority") == "l0_python_ut":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


_PLUGIN_NAME = "Nvfp4A16BlackwellMoePlugin"
_PLUGIN_VERSION = "1"
_MAX_SEQUENCE_LENGTH = 64
_TILE_N = 128
_TILE_K = 64


@dataclass(frozen=True)
class MoeCase:
    name: str
    num_experts: int = 128
    top_k: int = 6
    hidden_size: int = 256
    moe_inter_size: int = 192  # pads to 256 inside the layout
    activation_type: int = 4
    routing_mode: int = 1
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: int = 1
    routed_scaling_factor: float = 2.5
    max_routed_rows: int = 0
    layout: int = 1
    backend: int = 0

    @property
    def inter_size_padded(self) -> int:
        return (self.moe_inter_size + _TILE_N - 1) // _TILE_N * _TILE_N


_SMALL_CASE = MoeCase(name="small_relu2_sigmoid")


def _is_thor() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(
    ) == (11, 0)


def _random_expert(n: int, k: int, generator):
    packed = torch.randint(0,
                           256, (n, k // 2),
                           generator=generator,
                           dtype=torch.int64).to(torch.uint8)
    # E4M3 codes 0x28..0x3F decode to [0.25, 2): a realistic block-scale spread.
    scales = torch.randint(0x28,
                           0x40, (n, k // 16),
                           generator=generator,
                           dtype=torch.int64).to(torch.int8)
    ws2 = (torch.rand(
        (1, ), generator=generator) * 0.008 + 0.004).to(torch.float32)
    return packed, scales, ws2


@dataclass
class MoeFixture:
    case: MoeCase
    packed_inputs: Dict[str, "torch.Tensor"]
    dense_fc1: "torch.Tensor"  # [E, I, H] fp32, alpha included
    dense_fc2: "torch.Tensor"  # [E, H, I] fp32, alpha included


def _make_fixture(case: MoeCase, seed: int = 1234) -> MoeFixture:
    generator = torch.Generator().manual_seed(seed)
    fc1 = [
        _random_expert(case.moe_inter_size, case.hidden_size, generator)
        for _ in range(case.num_experts)
    ]
    fc2 = [
        _random_expert(case.hidden_size, case.moe_inter_size, generator)
        for _ in range(case.num_experts)
    ]
    fc1_q, fc1_s, fc1_g, fc2_q, fc2_s, fc2_g = repack_nvfp4_a16_blackwell_moe_experts(
        [t[0] for t in fc1], [t[1] for t in fc1], [t[2] for t in fc1],
        [t[0] for t in fc2], [t[1] for t in fc2], [t[2] for t in fc2])
    dense_fc1 = torch.stack([
        torch.from_numpy(decode_modelopt_nvfp4(*fc1[e]))
        for e in range(case.num_experts)
    ])
    dense_fc2 = torch.stack([
        torch.from_numpy(decode_modelopt_nvfp4(*fc2[e]))
        for e in range(case.num_experts)
    ])
    packed = {
        "fc1_qweights": fc1_q,
        "fc1_block_scales": fc1_s,
        "fc1_global_scales": fc1_g,
        "fc2_qweights": fc2_q,
        "fc2_block_scales": fc2_s,
        "fc2_global_scales": fc2_g,
    }
    return MoeFixture(case, packed, dense_fc1, dense_fc2)


def _routing_reference(case: MoeCase, router_logits, expert_score_bias):
    scores = torch.sigmoid(router_logits)
    biased = scores + expert_score_bias
    experts_per_group = case.num_experts // case.n_group
    grouped = biased.view(-1, case.n_group, experts_per_group)
    group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
    selected_groups = group_scores.topk(case.topk_group, dim=-1).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    expert_mask = group_mask[:, :, None].expand_as(grouped).reshape_as(biased)
    selected = biased.masked_fill(~expert_mask, float("-inf"))
    indices = selected.topk(case.top_k, dim=-1).indices
    weights = scores.gather(1, indices)
    if case.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * case.routed_scaling_factor, indices


def _moe_reference(fixture: MoeFixture, hidden_states, router_logits,
                   expert_score_bias):
    case = fixture.case
    hidden_2d = hidden_states.reshape(-1, case.hidden_size).to(torch.float32)
    weights, indices = _routing_reference(case,
                                          router_logits.to(torch.float32),
                                          expert_score_bias.to(torch.float32))
    output = torch.zeros((hidden_2d.shape[0], case.hidden_size),
                         dtype=torch.float32,
                         device=hidden_states.device)
    # One expert's dense fp32 weights on the device at a time: the full
    # stacks (4.8 GiB for the Nemotron shape) next to the packed plugin
    # inputs leave no headroom on the 6 GiB Thor CI device.
    for expert_id in range(case.num_experts):
        token_slot = (indices == expert_id).nonzero(as_tuple=False)
        if token_slot.numel() == 0:
            continue
        tokens = token_slot[:, 0]
        slots = token_slot[:, 1]
        fc1_w = fixture.dense_fc1[expert_id].to(hidden_states.device)
        fc2_w = fixture.dense_fc2[expert_id].to(hidden_states.device)
        fc1 = hidden_2d[tokens] @ fc1_w.T
        act = torch.relu(fc1).square().to(torch.float16).to(torch.float32)
        down = act @ fc2_w.T
        output.index_add_(0, tokens, down * weights[tokens, slots, None])
    return output.to(torch.float16).reshape_as(hidden_states)


def _plugin_fields(case: MoeCase):
    return [
        pf_int32("num_experts", case.num_experts),
        pf_int32("top_k", case.top_k),
        pf_int32("hidden_size", case.hidden_size),
        pf_int32("moe_inter_size", case.moe_inter_size),
        pf_int32("activation_type", case.activation_type),
        pf_int32("n_group", case.n_group),
        pf_int32("topk_group", case.topk_group),
        pf_int32("norm_topk_prob", case.norm_topk_prob),
        pf_float32("routed_scaling_factor", case.routed_scaling_factor),
        pf_int32("routing_mode", case.routing_mode),
        pf_int32("max_routed_rows", case.max_routed_rows),
        pf_int32("layout", case.layout),
        pf_int32("backend", case.backend),
    ]


def _io_specs(case: MoeCase,
              *,
              hidden_dtype=None,
              global_dtype=None,
              legacy_rank3=False):
    hidden_dtype = trt.float16 if hidden_dtype is None else hidden_dtype
    global_dtype = trt.float32 if global_dtype is None else global_dtype
    e, h, i_pad, i = (case.num_experts, case.hidden_size,
                      case.inter_size_padded, case.moe_inter_size)
    return [
        ("router_logits", trt.float32, (-1, e)),
        ("hidden_states", hidden_dtype, (-1, -1, h) if legacy_rank3 else
         (-1, h)),
        ("fc1_qweights", trt.int8, (e, i_pad // _TILE_N, h // _TILE_K, _TILE_N,
                                    32)),
        ("fc1_block_scales", trt.int8, (e, i_pad // _TILE_N, h // _TILE_K,
                                        _TILE_N, 4)),
        ("fc1_global_scales", global_dtype, (e, )),
        ("fc2_qweights", trt.int8, (e, h // _TILE_N, i // _TILE_K, _TILE_N,
                                    32)),
        ("fc2_block_scales", trt.int8, (e, h // _TILE_N, i // _TILE_K, _TILE_N,
                                        4)),
        ("fc2_global_scales", global_dtype, (e, )),
        ("expert_score_bias", trt.float32, (e, )),
    ]


def _profiles(case: MoeCase, input_specs):
    profiles = {}
    legacy_rank3 = next(shape for name, _, shape in input_specs
                        if name == "hidden_states")[0:2] == (-1, -1)
    for name, _, shape in input_specs:
        if name == "router_logits":
            max_tokens = 16 * 9 if legacy_rank3 else _MAX_SEQUENCE_LENGTH
            opt_tokens = 9 if legacy_rank3 else 1
            profiles[name] = ((1, case.num_experts), (opt_tokens,
                                                      case.num_experts),
                              (max_tokens, case.num_experts))
        elif name == "hidden_states":
            if len(shape) == 3:
                profiles[name] = ((1, 1, case.hidden_size),
                                  (1, 9, case.hidden_size), (16, 9,
                                                             case.hidden_size))
            else:
                profiles[name] = ((1, case.hidden_size), (1, case.hidden_size),
                                  (_MAX_SEQUENCE_LENGTH, case.hidden_size))
        else:
            profiles[name] = (shape, shape, shape)
    return profiles


def _build_runner(case: MoeCase,
                  expect_unsupported: bool = False,
                  **io_kwargs) -> PluginRunner:
    runner = PluginRunner()
    input_specs = _io_specs(case, **io_kwargs)
    runner.build(input_specs=input_specs,
                 output_names=["output"],
                 plugin_name=_PLUGIN_NAME,
                 plugin_version=_PLUGIN_VERSION,
                 plugin_fields=_plugin_fields(case),
                 profiles=_profiles(case, input_specs),
                 expect_unsupported=expect_unsupported)
    return runner


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
    runner._nvfp4_test_runtime = runtime


def _execute_case(case: MoeCase) -> None:
    fixture = _make_fixture(case)
    runner = _build_runner(case)
    _round_trip_engine(runner)
    static_inputs = {
        name: tensor.to("cuda").contiguous()
        for name, tensor in fixture.packed_inputs.items()
    }
    # T=1 uses decode kernels; larger token sets use grouped tcgen05 GEMM.
    for sequence_length in (1, 9, _MAX_SEQUENCE_LENGTH):
        generator = torch.Generator().manual_seed(40000 + sequence_length)
        hidden_states = torch.randn(
            (sequence_length, case.hidden_size),
            generator=generator,
            dtype=torch.float32).to(torch.float16).to("cuda")
        router_logits = torch.randn((sequence_length, case.num_experts),
                                    generator=generator,
                                    dtype=torch.float32).to("cuda")
        expert_score_bias = (torch.randn(
            (case.num_experts, ), generator=generator, dtype=torch.float32) *
                             0.05).to("cuda")
        expected = _moe_reference(fixture, hidden_states, router_logits,
                                  expert_score_bias)
        actual = torch.empty_like(hidden_states)
        tensors = {
            "router_logits": router_logits,
            "hidden_states": hidden_states,
            "expert_score_bias": expert_score_bias,
            "output": actual,
            **static_inputs,
        }
        runner.execute(tensors)
        assert bool(torch.isfinite(actual.to(torch.float32)).all())
        assert_close(
            f"{case.name}[backend={case.backend}][T={sequence_length}]",
            expected,
            actual,
            atol=0.05,
            rtol=0.02,
            cos_threshold=0.999)


def _execute_legacy_rank3_case(case: MoeCase) -> None:
    fixture = _make_fixture(case)
    runner = _build_runner(case, legacy_rank3=True)
    _round_trip_engine(runner)
    static_inputs = {
        name: tensor.to("cuda").contiguous()
        for name, tensor in fixture.packed_inputs.items()
    }
    for batch_size, sequence_length in ((1, 9), (2, 4), (16, 1)):
        num_tokens = batch_size * sequence_length
        generator = torch.Generator().manual_seed(50000 + num_tokens)
        hidden_states = torch.randn(
            (batch_size, sequence_length, case.hidden_size),
            generator=generator,
            dtype=torch.float32).to(torch.float16).to("cuda")
        router_logits = torch.randn((num_tokens, case.num_experts),
                                    generator=generator,
                                    dtype=torch.float32).to("cuda")
        expert_score_bias = (torch.randn(
            (case.num_experts, ), generator=generator, dtype=torch.float32) *
                             0.05).to("cuda")
        expected = _moe_reference(fixture, hidden_states, router_logits,
                                  expert_score_bias)
        actual = torch.empty_like(hidden_states)
        runner.execute({
            "router_logits": router_logits,
            "hidden_states": hidden_states,
            "expert_score_bias": expert_score_bias,
            "output": actual,
            **static_inputs,
        })
        assert_close(f"{case.name}[legacy B={batch_size},S={sequence_length}]",
                     expected,
                     actual,
                     atol=0.05,
                     rtol=0.02,
                     cos_threshold=0.999)


@pytest.mark.parametrize("backend", [0, 1, 2],
                         ids=["auto", "decode", "prefill"])
def test_small_relu2_sigmoid_decode_and_prefill_dynamic_engine(backend):
    if not _is_thor():
        pytest.skip("Nvfp4A16BlackwellMoePlugin requires SM110 (Thor)")
    _execute_case(replace(_SMALL_CASE, backend=backend))


def test_grouped_routing_decode_and_prefill_dynamic_engine():
    if not _is_thor():
        pytest.skip("Nvfp4A16BlackwellMoePlugin requires SM110 (Thor)")
    # n_group > 1 takes the shared moeSigmoidGroupTopk routing in both the
    # decode path (S=1) and the grouped-GEMM path (S=9 / S=64).
    _execute_case(
        replace(_SMALL_CASE,
                name="small_relu2_grouped",
                n_group=2,
                topk_group=1))


def test_rank3_hidden_states_are_rejected():
    if not _is_thor():
        pytest.skip("Nvfp4A16BlackwellMoePlugin requires SM110 (Thor)")
    _expect_build_rejected(_SMALL_CASE, legacy_rank3=True)


def test_nemotron_shape_decode_and_prefill_dynamic_engine():
    if not _is_thor():
        pytest.skip("Nvfp4A16BlackwellMoePlugin requires SM110 (Thor)")
    # Real Nemotron 3.5 Lightning routed-expert shape; 730 MB of weights.
    _execute_case(
        replace(_SMALL_CASE,
                name="nemotron_relu2",
                hidden_size=2688,
                moe_inter_size=1856))


def test_build_on_non_thor_gpu_is_rejected():
    if _is_thor():
        pytest.skip("negative SM gate only applies away from SM110")
    # configurePlugin refuses every other SM; the graceful build-time rejection
    # surfaces as PluginUnsupportedError through expect_unsupported.
    with pytest.raises(PluginUnsupportedError):
        _build_runner(_SMALL_CASE, expect_unsupported=True)


def _expect_build_rejected(case: MoeCase, **io_kwargs):
    """The creator (attribute validation) or the builder (dtype / SM gates)
    must refuse the contract cleanly on every device."""
    with pytest.raises(PluginUnsupportedError):
        _build_runner(case, expect_unsupported=True, **io_kwargs)


def test_build_rejects_marlin_layout_and_unknown_backend():
    _expect_build_rejected(replace(_SMALL_CASE, layout=0))
    _expect_build_rejected(replace(_SMALL_CASE, backend=3))


def test_build_rejects_unsupported_activation_or_routing():
    _expect_build_rejected(replace(_SMALL_CASE, activation_type=2))
    _expect_build_rejected(replace(_SMALL_CASE, routing_mode=0))


def test_build_rejects_misaligned_dimensions():
    _expect_build_rejected(replace(_SMALL_CASE, moe_inter_size=200))
    _expect_build_rejected(replace(_SMALL_CASE, hidden_size=200))


def test_build_rejects_bf16_hidden_or_fp16_global_scales():
    _expect_build_rejected(_SMALL_CASE, hidden_dtype=trt.bfloat16)
    _expect_build_rejected(_SMALL_CASE, global_dtype=trt.float16)


def test_create_plugin_accepts_nemotron_contract():
    PluginRunner()
    creator = trt.get_plugin_registry().get_creator(_PLUGIN_NAME,
                                                    _PLUGIN_VERSION, "")
    assert creator is not None
    names = [f.name for f in creator.field_names]
    # 13 ONNX attributes plus the runtime-only NVRTC bundle of the layer's
    # CUDA-core kernels (compiled at engine build, never set on the ONNX node).
    assert names == [
        "num_experts", "top_k", "hidden_size", "moe_inter_size",
        "activation_type", "n_group", "topk_group", "norm_topk_prob",
        "routed_scaling_factor", "routing_mode", "max_routed_rows", "layout",
        "backend", "moe_jit_bundle"
    ]
    fields = _plugin_fields(
        replace(_SMALL_CASE, hidden_size=2688, moe_inter_size=1856))
    plugin = creator.create_plugin(_PLUGIN_NAME,
                                   trt.PluginFieldCollection(fields),
                                   trt.TensorRTPhase.BUILD)
    assert plugin is not None
