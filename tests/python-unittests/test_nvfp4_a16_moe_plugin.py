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
"""TensorRT lifecycle and accuracy tests for Marlin MoE plugins.

Each positive test builds one dynamic-profile engine, round-trips its
serialization, and executes both decode (B=1, S=1) and prefill (B=1, S=128).
The reference dequantizes the original E2M1 codes with their original E4M3
block scales and per-expert global scales; it never decodes packed Marlin data.

A focused ``Int4MoePlugin`` case also locks the legacy FP16 SwiGLU rounding
behavior after that activation moved onto the shared MoE activation kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              PluginRunner, assert_close, pf_float32, pf_int32)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch


@pytest.fixture(autouse=True)
def require_plugin_test_dependencies(request):
    """Prevent the GPU-only L0 suite from passing through dependency skips."""
    if DEPENDENCIES_AVAILABLE:
        return

    reason = f"TensorRT/torch CUDA not available: {IMPORT_ERROR}"
    if request.config.getoption("--priority") == "l0_python_ut":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


_PLUGIN_NAME = "Nvfp4A16MoePlugin"
_PLUGIN_VERSION = "1"
_INT4_PLUGIN_NAME = "Int4MoePlugin"
_NUM_EXPERTS = 128
_HIDDEN_SIZE = 128
_MAX_SEQUENCE_LENGTH = 128
_FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@dataclass(frozen=True)
class MoeCase:
    name: str
    top_k: int
    moe_inter_size: int
    activation_type: int
    routing_mode: int
    n_group: int
    topk_group: int
    norm_topk_prob: int
    routed_scaling_factor: float
    hidden_size: int = _HIDDEN_SIZE
    num_experts: int = _NUM_EXPERTS
    max_routed_rows: Optional[int] = None

    @property
    def fc1_out_dim(self) -> int:
        return (2 * self.moe_inter_size
                if self.activation_type == 2 else self.moe_inter_size)

    @property
    def routed_row_capacity(self) -> int:
        if self.max_routed_rows is not None:
            return self.max_routed_rows
        # The dynamic profile max is prefill, which uses a 32-row Marlin tile.
        return (_MAX_SEQUENCE_LENGTH * self.top_k + self.num_experts *
                (32 - 1))


_QWEN_CASE = MoeCase(
    name="qwen_swiglu_softmax",
    top_k=8,
    moe_inter_size=64,
    activation_type=2,
    routing_mode=0,
    n_group=1,
    topk_group=1,
    norm_topk_prob=1,
    # Softmax routing does not consume this attribute.
    routed_scaling_factor=0.0)

_NEMOTRON_CASE = MoeCase(name="nemotron_relu2_sigmoid_group",
                         top_k=6,
                         moe_inter_size=128,
                         activation_type=4,
                         routing_mode=1,
                         n_group=8,
                         topk_group=4,
                         norm_topk_prob=1,
                         routed_scaling_factor=1.0)


@dataclass
class MoeFixture:
    case: MoeCase
    activation_dtype: "torch.dtype"
    packed_inputs: Dict[str, "torch.Tensor"]
    original_codes: Dict[str, "torch.Tensor"]
    original_block_scales: Dict[str, "torch.Tensor"]
    original_global_scales: Dict[str, "torch.Tensor"]


def _make_tile_constant_projection(num_experts: int, size_n: int, size_k: int,
                                   global_scales: "torch.Tensor",
                                   activation_dtype: "torch.dtype",
                                   generator: "torch.Generator"):
    """Build a synthetic projection already encoded in Marlin layout.

    Codes and scales vary by expert and 16x64 Marlin tile, but are constant
    inside each tile. The Marlin permutations therefore leave each tile
    unchanged, which keeps this standalone plugin test independent of model
    checkpoint repacking code.
    """
    assert size_n % 64 == 0 and size_k % 16 == 0
    num_n_tiles = size_n // 64
    num_k_tiles = size_k // 16
    tile_codes = torch.randint(0,
                               16, (num_experts, num_n_tiles, num_k_tiles),
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
    global_scale_exponent = 119 if activation_dtype == torch.bfloat16 else 7
    packed_global_scales = (
        global_scales * float(2**global_scale_exponent)).to(activation_dtype)
    return (packed_qweights, packed_block_scales, packed_global_scales,
            original_codes, original_scales)


def _make_qwen_fixture(activation_dtype: "torch.dtype") -> MoeFixture:
    case = _QWEN_CASE
    generator = torch.Generator().manual_seed(14122)
    e, h, i = case.num_experts, case.hidden_size, case.moe_inter_size
    expert_ids = torch.arange(e)
    small_global = torch.full((e, ), 2.0**-3, dtype=torch.float32)
    large_global = torch.full((e, ), 2.0**-2, dtype=torch.float32)
    # The plugin accepts one FC1 global scale, so use the same exact power of
    # two for gate and up in these prepacked standalone fixtures.
    fc1_global = torch.where(expert_ids % 2 == 0, small_global, large_global)
    down_global = torch.full((e, ), 2.0**-3, dtype=torch.float32)
    gate = _make_tile_constant_projection(e, i, h, fc1_global,
                                          activation_dtype, generator)
    up = _make_tile_constant_projection(e, i, h, fc1_global, activation_dtype,
                                        generator)
    down = _make_tile_constant_projection(e, h, i, down_global,
                                          activation_dtype, generator)

    return MoeFixture(
        case=case,
        activation_dtype=activation_dtype,
        packed_inputs={
            "fc1_qweights": torch.cat([gate[0], up[0]], dim=2),
            "fc1_block_scales": torch.cat([gate[1], up[1]], dim=2),
            "fc1_global_scales": gate[2],
            "fc2_qweights": down[0],
            "fc2_block_scales": down[1],
            "fc2_global_scales": down[2],
        },
        original_codes={
            "gate": gate[3],
            "up": up[3],
            "fc2": down[3]
        },
        original_block_scales={
            "gate": gate[4],
            "up": up[4],
            "fc2": down[4]
        },
        original_global_scales={
            "gate": fc1_global,
            "up": fc1_global,
            "fc2": down_global
        },
    )


def _make_nemotron_fixture(activation_dtype: "torch.dtype") -> MoeFixture:
    case = _NEMOTRON_CASE
    generator = torch.Generator().manual_seed(19652)
    e, h, i = case.num_experts, case.hidden_size, case.moe_inter_size
    fc1_global = torch.full((e, ), 2.0**-3, dtype=torch.float32)
    fc2_global = torch.full((e, ), 2.0**-3, dtype=torch.float32)
    fc1 = _make_tile_constant_projection(e, i, h, fc1_global, activation_dtype,
                                         generator)
    fc2 = _make_tile_constant_projection(e, h, i, fc2_global, activation_dtype,
                                         generator)

    return MoeFixture(
        case=case,
        activation_dtype=activation_dtype,
        packed_inputs={
            "fc1_qweights": fc1[0],
            "fc1_block_scales": fc1[1],
            "fc1_global_scales": fc1[2],
            "fc2_qweights": fc2[0],
            "fc2_block_scales": fc2[1],
            "fc2_global_scales": fc2[2],
        },
        original_codes={
            "fc1": fc1[3],
            "fc2": fc2[3]
        },
        original_block_scales={
            "fc1": fc1[4],
            "fc2": fc2[4]
        },
        original_global_scales={
            "fc1": fc1_global,
            "fc2": fc2_global
        },
    )


def _dequantize_original(codes: "torch.Tensor", block_scales: "torch.Tensor",
                         global_scales: "torch.Tensor") -> "torch.Tensor":
    """Dequantize original E2M1 codes without consulting Marlin tensors."""
    levels = torch.tensor(_FP4_LEVELS, dtype=torch.float32)
    magnitudes = levels[(codes & 0x7).to(torch.int64)]
    values = torch.where((codes & 0x8) != 0, -magnitudes, magnitudes)
    expanded_scales = block_scales.to(torch.float32).repeat_interleave(16,
                                                                       dim=2)
    return values * expanded_scales * global_scales[:, None, None]


def _reference_dense_weights(fixture: MoeFixture):
    return {
        name:
        _dequantize_original(codes, fixture.original_block_scales[name],
                             fixture.original_global_scales[name]).to("cuda")
        for name, codes in fixture.original_codes.items()
    }


def _routing_reference(case: MoeCase, router_logits: "torch.Tensor",
                       expert_score_bias: "torch.Tensor"):
    if case.routing_mode == 0:
        scores = torch.softmax(router_logits + expert_score_bias, dim=-1)
        weights, indices = torch.topk(scores, case.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights, indices

    scores = torch.sigmoid(router_logits)
    biased = scores + expert_score_bias
    experts_per_group = case.num_experts // case.n_group
    grouped = biased.view(-1, case.n_group, experts_per_group)
    group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
    selected_groups = group_scores.topk(case.topk_group, dim=-1).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    expert_mask = group_mask[:, :, None].expand_as(grouped).reshape_as(biased)
    selected_scores = biased.masked_fill(~expert_mask, float("-inf"))
    indices = selected_scores.topk(case.top_k, dim=-1).indices
    weights = scores.gather(1, indices)
    if case.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * case.routed_scaling_factor, indices


def _moe_reference(fixture: MoeFixture, hidden_states: "torch.Tensor",
                   router_logits: "torch.Tensor",
                   expert_score_bias: "torch.Tensor",
                   dense_weights: Dict[str, "torch.Tensor"]):
    """Reference the FP16/BF16 FC1 -> activation -> FC2 -> sum pipeline."""
    case = fixture.case
    activation_dtype = fixture.activation_dtype
    hidden_2d = hidden_states.reshape(-1, case.hidden_size)
    route_weights, route_indices = _routing_reference(case, router_logits,
                                                      expert_score_bias)
    slot_outputs = torch.zeros(
        (hidden_2d.shape[0], case.top_k, case.hidden_size),
        dtype=activation_dtype,
        device=hidden_states.device)

    for expert_id in range(case.num_experts):
        token_slot = (route_indices == expert_id).nonzero(as_tuple=False)
        if token_slot.numel() == 0:
            continue
        token_ids = token_slot[:, 0]
        slot_ids = token_slot[:, 1]
        expert_input = hidden_2d[token_ids].to(torch.float32)
        if case.activation_type == 2:
            gate = (expert_input @ dense_weights["gate"][expert_id].T
                    ).to(activation_dtype).to(torch.float32)
            up = (expert_input @ dense_weights["up"][expert_id].T
                  ).to(activation_dtype).to(torch.float32)
            activated_gate = torch.nn.functional.silu(gate)
            if activation_dtype == torch.float16:
                # The shared FP16 activation preserves legacy rounding: SiLU
                # rounds to FP16 before the multiplication by up.
                activated_gate = activated_gate.to(activation_dtype).to(
                    torch.float32)
            activated = (activated_gate * up).to(activation_dtype).to(
                torch.float32)
        else:
            fc1 = (expert_input @ dense_weights["fc1"][expert_id].T
                   ).to(activation_dtype).to(torch.float32)
            activated = torch.relu(fc1).square().to(activation_dtype).to(
                torch.float32)

        down = (activated @ dense_weights["fc2"][expert_id].T
                ).to(activation_dtype).to(torch.float32)
        weighted = (
            down *
            route_weights[token_ids, slot_ids, None]).to(activation_dtype)
        slot_outputs[token_ids, slot_ids] = weighted

    output = slot_outputs.to(torch.float32).sum(dim=1).to(activation_dtype)
    return output.reshape_as(hidden_states)


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
        pf_int32("max_routed_rows", case.routed_row_capacity),
    ]


def _io_specs(case: MoeCase,
              activation_dtype=None,
              *,
              hidden_dtype=None,
              fc1_global_dtype=None,
              fc2_global_dtype=None):
    activation_dtype = (trt.bfloat16
                        if activation_dtype is None else activation_dtype)
    hidden_dtype = (activation_dtype if hidden_dtype is None else hidden_dtype)
    fc1_global_dtype = (activation_dtype
                        if fc1_global_dtype is None else fc1_global_dtype)
    fc2_global_dtype = (activation_dtype
                        if fc2_global_dtype is None else fc2_global_dtype)
    return [
        ("router_logits", trt.float32, (-1, case.num_experts)),
        ("hidden_states", hidden_dtype, (-1, -1, case.hidden_size)),
        ("fc1_qweights", trt.int8, (case.num_experts, case.hidden_size // 16,
                                    8 * case.fc1_out_dim)),
        ("fc1_block_scales", trt.int8,
         (case.num_experts, case.hidden_size // 16, case.fc1_out_dim)),
        ("fc1_global_scales", fc1_global_dtype, (case.num_experts, )),
        ("fc2_qweights", trt.int8,
         (case.num_experts, case.moe_inter_size // 16, 8 * case.hidden_size)),
        ("fc2_block_scales", trt.int8,
         (case.num_experts, case.moe_inter_size // 16, case.hidden_size)),
        ("fc2_global_scales", fc2_global_dtype, (case.num_experts, )),
        ("expert_score_bias", trt.float32, (case.num_experts, )),
    ]


def _profiles(case: MoeCase, input_specs):
    profiles = {}
    for name, _, shape in input_specs:
        if name == "router_logits":
            profiles[name] = ((1, case.num_experts), (1, case.num_experts),
                              (_MAX_SEQUENCE_LENGTH, case.num_experts))
        elif name == "hidden_states":
            profiles[name] = ((1, 1, case.hidden_size), (1, 1,
                                                         case.hidden_size),
                              (1, _MAX_SEQUENCE_LENGTH, case.hidden_size))
        else:
            profiles[name] = (shape, shape, shape)
    return profiles


def _build_runner(case: MoeCase, activation_dtype) -> PluginRunner:
    runner = PluginRunner()
    input_specs = _io_specs(case, activation_dtype)
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
    # Keep the runtime alive for TensorRT versions where it owns resources used
    # by the deserialized engine.
    runner._nvfp4_test_runtime = runtime


def _execute_decode_and_prefill(fixture: MoeFixture) -> None:
    case = fixture.case
    trt_dtype = (trt.bfloat16 if fixture.activation_dtype == torch.bfloat16
                 else trt.float16)
    runner = _build_runner(case, trt_dtype)
    _round_trip_engine(runner)
    static_inputs = {
        name: tensor.to("cuda").contiguous()
        for name, tensor in fixture.packed_inputs.items()
    }
    dense_weights = _reference_dense_weights(fixture)

    for sequence_length in (1, _MAX_SEQUENCE_LENGTH):
        generator = torch.Generator().manual_seed(30000 + sequence_length +
                                                  case.routing_mode)
        hidden_states = (torch.randn((1, sequence_length, case.hidden_size),
                                     generator=generator,
                                     dtype=torch.float32) * 0.25).to(
                                         fixture.activation_dtype).to("cuda")
        router_logits = torch.randn((sequence_length, case.num_experts),
                                    generator=generator,
                                    dtype=torch.float32).to("cuda")
        expert_score_bias = (torch.randn(
            (case.num_experts, ), generator=generator, dtype=torch.float32) *
                             0.01).to("cuda")
        expected = _moe_reference(fixture, hidden_states, router_logits,
                                  expert_score_bias, dense_weights)
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
            f"{case.name}[{fixture.activation_dtype}][S={sequence_length}]",
            expected,
            actual,
            atol=0.05,
            rtol=0.02,
            cos_threshold=0.99)


@pytest.mark.parametrize("activation_dtype_name", ["bfloat16", "float16"],
                         ids=["bf16", "fp16"])
def test_qwen_softmax_swiglu_decode_and_prefill_dynamic_engine(
        activation_dtype_name):
    _execute_decode_and_prefill(
        _make_qwen_fixture(getattr(torch, activation_dtype_name)))


@pytest.mark.parametrize("activation_dtype_name", ["bfloat16", "float16"],
                         ids=["bf16", "fp16"])
def test_nemotron_sigmoid_group_relu2_decode_and_prefill_dynamic_engine(
        activation_dtype_name):
    _execute_decode_and_prefill(
        _make_nemotron_fixture(getattr(torch, activation_dtype_name)))


def test_int4_fp16_swiglu_preserves_legacy_rounding():
    """Exercise the shared FP16 activation through the complete INT4 plugin."""
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("The four-stage Marlin kernel requires SM80 or newer")

    num_experts = 1
    top_k = 1
    hidden_size = 128
    moe_inter_size = 64
    quantization_group_size = 64
    input_specs = [
        ("router_logits", trt.float32, (1, num_experts)),
        ("hidden_states", trt.float16, (1, 1, hidden_size)),
        ("fc_gate_up_qweights", trt.int8, (num_experts, hidden_size // 16,
                                           16 * moe_inter_size)),
        ("fc_gate_up_scales", trt.float16,
         (num_experts, hidden_size // quantization_group_size,
          2 * moe_inter_size)),
        ("fc_down_qweights", trt.int8, (num_experts, moe_inter_size // 16,
                                        8 * hidden_size)),
        ("fc_down_scales", trt.float16,
         (num_experts, moe_inter_size // quantization_group_size,
          hidden_size)),
    ]
    profiles = {name: (shape, shape, shape) for name, _, shape in input_specs}
    runner = PluginRunner()
    runner.build(input_specs=input_specs,
                 output_names=["output"],
                 plugin_name=_INT4_PLUGIN_NAME,
                 plugin_version=_PLUGIN_VERSION,
                 plugin_fields=[
                     pf_int32("num_experts", num_experts),
                     pf_int32("top_k", top_k),
                     pf_int32("hidden_size", hidden_size),
                     pf_int32("moe_inter_size", moe_inter_size),
                     pf_int32("activation_type", 0),
                     pf_int32("quantization_group_size",
                              quantization_group_size),
                 ],
                 profiles=profiles)

    hidden_states = torch.zeros((1, 1, hidden_size),
                                dtype=torch.float16,
                                device="cuda")
    hidden_states[..., 0] = 1.0
    # Every 0x99 byte holds two unsigned-INT4 code-9 values. Since all
    # nibbles are identical, the Marlin weight permutation leaves them
    # unchanged and each dequantized weight is +1 times its channel scale.
    packed_positive_one = -103
    fc_gate_up_qweights = torch.full(input_specs[2][2],
                                     packed_positive_one,
                                     dtype=torch.int8,
                                     device="cuda")
    fc_down_qweights = torch.full(input_specs[4][2],
                                  packed_positive_one,
                                  dtype=torch.int8,
                                  device="cuda")
    fc_gate_up_scales = torch.zeros(input_specs[3][2],
                                    dtype=torch.float16,
                                    device="cuda")
    gate_value = torch.tensor([0x4898],
                              dtype=torch.int16).view(torch.float16).item()
    up_value = torch.tensor([0x5F66],
                            dtype=torch.int16).view(torch.float16).item()
    fc_gate_up_scales[0, 0, 0] = gate_value
    fc_gate_up_scales[0, 0, moe_inter_size] = up_value
    fc_down_scales = torch.ones(input_specs[5][2],
                                dtype=torch.float16,
                                device="cuda")
    actual = torch.empty_like(hidden_states)
    runner.execute({
        "router_logits":
        torch.zeros((1, num_experts), dtype=torch.float32, device="cuda"),
        "hidden_states":
        hidden_states,
        "fc_gate_up_qweights":
        fc_gate_up_qweights,
        "fc_gate_up_scales":
        fc_gate_up_scales,
        "fc_down_qweights":
        fc_down_qweights,
        "fc_down_scales":
        fc_down_scales,
        "output":
        actual,
    })

    # Legacy behavior rounds SiLU(gate) to FP16 before multiplying by up.
    # That produces 0x6c40; a single rounding after the multiply is 0x6c3f.
    legacy_result_bits = 0x6C40
    final_round_only_bits = 0x6C3F
    actual_bits = actual.cpu().view(torch.int16)
    assert actual_bits.unique().tolist() == [legacy_result_bits]
    assert not bool(torch.any(actual_bits == final_round_only_bits))


def _build_serialized_network_without_skip(case: MoeCase,
                                           activation_dtype=None,
                                           **dtype_overrides):
    """Low-level builder used by negative tests; never converts failure to skip."""
    harness = PluginRunner()
    builder = trt.Builder(harness.logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    input_specs = _io_specs(case, activation_dtype, **dtype_overrides)
    inputs = [
        network.add_input(name, dtype, shape)
        for name, dtype, shape in input_specs
    ]

    creator = trt.get_plugin_registry().get_creator(_PLUGIN_NAME,
                                                    _PLUGIN_VERSION, "")
    if creator is None:
        raise RuntimeError(f"{_PLUGIN_NAME} v{_PLUGIN_VERSION} not registered")
    fields = _plugin_fields(case)
    plugin = creator.create_plugin(_PLUGIN_NAME,
                                   trt.PluginFieldCollection(fields),
                                   trt.TensorRTPhase.BUILD)
    if plugin is None:
        return None
    layer = network.add_plugin_v3(inputs, [], plugin)
    if layer is None:
        return None
    network.mark_output(layer.get_output(0))

    profile = builder.create_optimization_profile()
    for name, shapes in _profiles(case, input_specs).items():
        profile.set_shape(name, *shapes)
    config.add_optimization_profile(profile)
    try:
        return builder.build_serialized_network(network, config)
    except RuntimeError:
        # Some TensorRT Python versions raise on an invalid strongly-typed
        # network while others log the rejection and return None.
        return None


def test_build_rejects_insufficient_max_routed_rows():
    required = _QWEN_CASE.routed_row_capacity
    case = replace(_QWEN_CASE, max_routed_rows=required - 1)
    assert _build_serialized_network_without_skip(case) is None


@pytest.mark.parametrize("case", [
    replace(_QWEN_CASE, activation_type=3),
    replace(_QWEN_CASE, routing_mode=2),
],
                         ids=["activation", "routing"])
def test_build_rejects_unsupported_activation_or_routing(case):
    assert _build_serialized_network_without_skip(case) is None


def test_build_rejects_wrong_hidden_state_dtype():
    assert _build_serialized_network_without_skip(
        _QWEN_CASE, hidden_dtype=trt.float32) is None


@pytest.mark.parametrize("global_scale_name", ["fc1", "fc2"])
def test_build_rejects_mixed_global_scale_dtype(global_scale_name):
    dtype_overrides = {f"{global_scale_name}_global_dtype": trt.bfloat16}
    assert _build_serialized_network_without_skip(_QWEN_CASE, trt.float16,
                                                  **dtype_overrides) is None


def test_build_rejects_nonpositive_sigmoid_routed_scaling_factor():
    case = replace(_NEMOTRON_CASE, routed_scaling_factor=0.0)
    assert _build_serialized_network_without_skip(case) is None


@pytest.mark.parametrize("case", [
    replace(_QWEN_CASE, hidden_size=64),
    replace(_QWEN_CASE, moe_inter_size=96),
],
                         ids=["hidden", "swiglu_intermediate"])
def test_build_rejects_misaligned_dimensions(case):
    assert _build_serialized_network_without_skip(case) is None


def test_build_rejects_unsupported_num_experts():
    assert _build_serialized_network_without_skip(
        replace(_QWEN_CASE, num_experts=64)) is None


def test_create_plugin_accepts_256_experts():
    """Qwen3.6-35B-A3B uses 256 routed experts; createPlugin must not reject it."""
    PluginRunner()
    creator = trt.get_plugin_registry().get_creator(_PLUGIN_NAME,
                                                    _PLUGIN_VERSION, "")
    assert creator is not None
    fields = _plugin_fields(replace(_QWEN_CASE, num_experts=256))
    plugin = creator.create_plugin(_PLUGIN_NAME,
                                   trt.PluginFieldCollection(fields),
                                   trt.TensorRTPhase.BUILD)
    assert plugin is not None


def test_create_plugin_accepts_512_experts():
    """Nemotron-3-Super-120B uses 512 routed experts; createPlugin must not reject it."""
    PluginRunner()
    creator = trt.get_plugin_registry().get_creator(_PLUGIN_NAME,
                                                    _PLUGIN_VERSION, "")
    assert creator is not None
    fields = _plugin_fields(replace(_QWEN_CASE, num_experts=512))
    plugin = creator.create_plugin(_PLUGIN_NAME,
                                   trt.PluginFieldCollection(fields),
                                   trt.TensorRTPhase.BUILD)
    assert plugin is not None
