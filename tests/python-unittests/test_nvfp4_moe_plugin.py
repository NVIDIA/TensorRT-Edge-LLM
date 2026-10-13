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
"""Token-major integration for ``Nvfp4MoePlugin``."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              PluginRunner, PluginUnsupportedError, pf_float32,
                              pf_int32)

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

    from tensorrt_edgellm.checkpoint.repacking import (
        _interleave_gated_moe_fc1, _nvfp4_moe_plugin_tensors,
        _quantize_nvfp4_moe_weight)


@pytest.fixture(autouse=True)
def require_plugin_test_dependencies(request):
    if DEPENDENCIES_AVAILABLE:
        return
    reason = f"TensorRT/torch CUDA not available: {IMPORT_ERROR}"
    if request.config.getoption("--priority") == "l0_python_ut":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


_NUM_EXPERTS = 128
_HIDDEN_SIZE = 128
_INTER_SIZE = 128
_MAX_TOKENS = 16 * 9


def _pack_test_weight(dense_weight):
    qweight, scale_bytes = _quantize_nvfp4_moe_weight(dense_weight,
                                                      group_size=16,
                                                      global_scale=1.0)
    return _nvfp4_moe_plugin_tensors(qweight, scale_bytes)


def _fields(activation_type, routing_mode, top_k):
    return [
        pf_int32("num_experts", _NUM_EXPERTS),
        pf_int32("top_k", top_k),
        pf_int32("hidden_size", _HIDDEN_SIZE),
        pf_int32("moe_inter_size", _INTER_SIZE),
        pf_int32("activation_type", activation_type),
        pf_int32("n_group", 1),
        pf_int32("topk_group", 1),
        pf_int32("norm_topk_prob", 1),
        pf_float32("routed_scaling_factor", 1.0),
        pf_int32("routing_mode", routing_mode),
        pf_int32("backend", 0),
        pf_int32("max_routed_rows", 0),
        pf_int32("io_dtype", 1),
    ]


def _packed_inputs(gated):
    generator = np.random.default_rng(6752959)
    if gated:
        gate = generator.normal(0, 0.02,
                                (_INTER_SIZE, _HIDDEN_SIZE)).astype(np.float32)
        up = generator.normal(0, 0.02,
                              (_INTER_SIZE, _HIDDEN_SIZE)).astype(np.float32)
        fc1_dense = _interleave_gated_moe_fc1(gate, up, _HIDDEN_SIZE,
                                              _INTER_SIZE)
    else:
        fc1_dense = generator.normal(
            0, 0.02, (_INTER_SIZE, _HIDDEN_SIZE)).astype(np.float32)
    fc2_dense = generator.normal(0, 0.02, (_HIDDEN_SIZE, _INTER_SIZE)).astype(
        np.float32)
    fc1_qw, fc1_sf = _pack_test_weight(fc1_dense)
    fc2_qw, fc2_sf = _pack_test_weight(fc2_dense)

    def repeat(value):
        return value.unsqueeze(0).repeat(_NUM_EXPERTS, *([1] * value.ndim))

    ones = torch.ones((_NUM_EXPERTS, ), dtype=torch.float32)
    return {
        "fc1_qweights": repeat(fc1_qw),
        "fc1_blocks_scale": repeat(fc1_sf),
        "fc1_alpha": ones,
        "fc2_qweights": repeat(fc2_qw),
        "fc2_blocks_scale": repeat(fc2_sf),
        "fc2_alpha": ones,
        "input_global_scale": ones,
        "down_input_scale": ones,
        "e_score_correction_bias": torch.zeros_like(ones),
    }


def _specs(hidden_shape, gated):
    fc1_size = 2 * _INTER_SIZE if gated else _INTER_SIZE
    return [
        ("router_logits", trt.float32, (-1, _NUM_EXPERTS)),
        ("hidden_states", trt.float16, hidden_shape),
        ("fc1_qweights", trt.int8, (_NUM_EXPERTS, fc1_size,
                                    _HIDDEN_SIZE // 2)),
        ("fc1_blocks_scale", trt.int8, (_NUM_EXPERTS, fc1_size // 128, 2, 32,
                                        4, 4)),
        ("fc1_alpha", trt.float32, (_NUM_EXPERTS, )),
        ("fc2_qweights", trt.int8, (_NUM_EXPERTS, _HIDDEN_SIZE,
                                    _INTER_SIZE // 2)),
        ("fc2_blocks_scale", trt.int8, (_NUM_EXPERTS, 1, 2, 32, 4, 4)),
        ("fc2_alpha", trt.float32, (_NUM_EXPERTS, )),
        ("input_global_scale", trt.float32, (_NUM_EXPERTS, )),
        ("down_input_scale", trt.float32, (_NUM_EXPERTS, )),
        ("e_score_correction_bias", trt.float32, (_NUM_EXPERTS, )),
    ]


def _build(legacy_rank3,
           activation_type,
           routing_mode,
           top_k,
           gated,
           *,
           plugin_name="Nvfp4MoePlugin",
           expect_unsupported=False):
    hidden_shape = (-1, -1, _HIDDEN_SIZE) if legacy_rank3 else (-1,
                                                                _HIDDEN_SIZE)
    specs = _specs(hidden_shape, gated)
    profiles = {}
    for name, _, shape in specs:
        if name == "router_logits":
            profiles[name] = ((1, _NUM_EXPERTS), (8, _NUM_EXPERTS),
                              (_MAX_TOKENS, _NUM_EXPERTS))
        elif name == "hidden_states":
            profiles[name] = (((1, 1, _HIDDEN_SIZE), (1, 8, _HIDDEN_SIZE),
                               (16, 9, _HIDDEN_SIZE)) if legacy_rank3 else
                              ((1, _HIDDEN_SIZE), (8, _HIDDEN_SIZE),
                               (_MAX_TOKENS, _HIDDEN_SIZE)))
        else:
            profiles[name] = (shape, shape, shape)
    return PluginRunner().build(input_specs=specs,
                                output_names=["output"],
                                plugin_name=plugin_name,
                                plugin_version="1",
                                plugin_fields=_fields(activation_type,
                                                      routing_mode, top_k),
                                profiles=profiles,
                                expect_unsupported=expect_unsupported)


def _round_trip(runner):
    serialized = runner.engine.serialize()
    assert serialized is not None
    runtime = trt.Runtime(runner.logger)
    runner.engine = runtime.deserialize_cuda_engine(serialized)
    assert runner.engine is not None
    runner.context = runner.engine.create_execution_context()
    assert runner.context is not None
    runner._nvfp4_test_runtime = runtime


@pytest.mark.parametrize(
    "activation_type,routing_mode,top_k,gated",
    [(4, 1, 1, False), (2, 0, 8, True)],
    ids=["nemotron-relu2-sigmoid", "qwen-swiglu-softmax"],
)
def test_rank2_is_stable_after_engine_round_trip(activation_type, routing_mode,
                                                 top_k, gated):
    if torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("Nvfp4MoePlugin test requires SM110 (Thor)")
    original = _build(False, activation_type, routing_mode, top_k, gated)
    restored = _build(False, activation_type, routing_mode, top_k, gated)
    _round_trip(restored)

    static = {
        name: value.to("cuda").contiguous()
        for name, value in _packed_inputs(gated).items()
    }
    for batch_size, sequence_length in ((1, 1), (1, 8), (2, 4), (1, 9), (16,
                                                                         1)):
        num_tokens = batch_size * sequence_length
        generator = torch.Generator().manual_seed(60000 + num_tokens)
        hidden2 = torch.randn((num_tokens, _HIDDEN_SIZE),
                              generator=generator,
                              dtype=torch.float16).to("cuda")
        router = torch.randn((num_tokens, _NUM_EXPERTS),
                             generator=generator,
                             dtype=torch.float32).to("cuda")
        expected = torch.empty_like(hidden2)
        actual = torch.empty_like(hidden2)
        original.execute({
            "router_logits": router,
            "hidden_states": hidden2,
            "output": expected,
            **static,
        })
        restored.execute({
            "router_logits": router,
            "hidden_states": hidden2,
            "output": actual,
            **static,
        })
        max_abs_diff = float(
            (expected.float() - actual.float()).abs().max().item())
        assert max_abs_diff <= 1e-4, (
            f"serialized rank-2 output differs for B={batch_size}, "
            f"S={sequence_length}: max_abs_diff={max_abs_diff}")


@pytest.mark.parametrize(
    "activation_type,routing_mode,top_k,gated",
    [(4, 1, 1, False), (2, 0, 8, True)],
    ids=["nemotron-relu2-sigmoid", "qwen-swiglu-softmax"],
)
@pytest.mark.parametrize("plugin_name",
                         ["Nvfp4MoePlugin", "NvFP4MoEPluginGeforce"])
def test_rank3_hidden_states_are_rejected(activation_type, routing_mode, top_k,
                                          gated, plugin_name):
    if torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("Nvfp4MoePlugin test requires SM110 (Thor)")
    with pytest.raises(PluginUnsupportedError):
        _build(True,
               activation_type,
               routing_mode,
               top_k,
               gated,
               plugin_name=plugin_name,
               expect_unsupported=True)
