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
"""Rank propagation contracts for token-major experimental decoders."""

from types import SimpleNamespace

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from experimental.builder.models.dflash.modeling_dflash_draft import \
    DFlashTargetProjection
from experimental.builder.models.dspark.modeling_dspark_draft import \
    DSparkTargetProjection
from experimental.builder.models.phi4mm.modeling_phi4mm_text import \
    Phi4MultimodalMLP
from experimental.builder.ops import functional as F
from experimental.builder.ops.backend import Net
from experimental.builder.ops.functional import core as functional_core
from experimental.builder.ops.functional import moe as functional_moe
from experimental.builder.ops.linear import Linear
from experimental.builder.ops.module import BuildContext, BuildOptions
from experimental.builder.ops.normalization import RMSNorm


class _Tensor:

    def __init__(self, ndim):
        self.ndim = ndim
        self.dtype = trt.float16


class _MoeTensor(_Tensor):

    def __init__(self, ndim, operations):
        super().__init__(ndim)
        self.operations = operations

    def unsqueeze(self, axis, rank):
        self.operations.append(("unsqueeze", axis, rank))
        return _MoeTensor(rank + 1, self.operations)

    def reshape(self, shape):
        self.operations.append(("reshape", shape))
        return _MoeTensor(len(shape), self.operations)


@pytest.mark.parametrize("sm12x", [False, True], ids=["thor", "geforce"])
def test_nvfp4_moe_preserves_token_major_plugin_contract(monkeypatch, sm12x):
    operations = []
    router = _MoeTensor(2, operations)
    hidden = _MoeTensor(2, operations)
    plugin_output = _MoeTensor(2, operations)
    captured = {}
    monkeypatch.setattr(functional_moe, "parameter",
                        lambda *args, **kwargs: object())

    def fake_operation(name, inputs, **attributes):
        captured.update(name=name, inputs=inputs, attributes=attributes)
        return plugin_output

    monkeypatch.setattr(functional_moe, "operation", fake_operation)
    weights = {
        name: np.ones(1)
        for name in ("fc1_qweights", "fc1_blocks_scale", "fc1_alpha",
                     "fc2_qweights", "fc2_blocks_scale", "fc2_alpha",
                     "input_global_scale", "down_input_scale",
                     "e_score_correction_bias")
    }

    result = functional_moe.nvfp4_moe(
        router,
        hidden,
        weights,
        2,
        1,
        16,
        128,
        functional_moe.MoeActivation.RELU2,
        1,
        1,
        0,
        1.0,
        functional_moe.MoeRouting.SIGMOID_GROUP_TOPK,
        sm12x,
        weight_prefix="experts")

    expected_operation = "nvfp4_moe_sm12x" if sm12x else "nvfp4_moe"
    assert captured["name"] == expected_operation
    assert captured["inputs"][0] is router
    assert captured["inputs"][1] is hidden
    assert captured["inputs"][1].ndim == 2
    assert result.ndim == 2
    assert operations == []


@pytest.mark.parametrize("quant_type", ["fp16", "int4", "nvfp4_a16"])
def test_other_moe_variants_preserve_token_major_plugin_contract(
        monkeypatch, quant_type):
    operations = []
    router = _MoeTensor(2, operations)
    hidden = _MoeTensor(2, operations)
    captured = {}
    monkeypatch.setattr(functional_moe, "parameter",
                        lambda *args, **kwargs: object())

    def fake_operation(name, inputs, **attributes):
        captured.update(name=name, inputs=inputs, attributes=attributes)
        return _MoeTensor(2, operations)

    monkeypatch.setattr(functional_moe, "operation", fake_operation)
    if quant_type == "fp16":
        weights = {name: np.ones(1) for name in ("fc1_weights", "fc2_weights")}
        result = functional_moe.fp16_moe(
            router,
            hidden,
            weights,
            2,
            1,
            16,
            128,
            weight_prefix="experts",
            weight_bindings={name: None
                             for name in weights})
    elif quant_type == "int4":
        weights = {
            name: np.ones(1)
            for name in ("fc_gate_up_qweights", "fc_gate_up_scales",
                         "fc_down_qweights", "fc_down_scales")
        }
        result = functional_moe.int4_moe(
            router,
            hidden,
            weights,
            2,
            1,
            16,
            128,
            64,
            weight_prefix="experts",
            weight_bindings={name: None
                             for name in weights})
    else:
        weights = {
            name: np.ones(1)
            for name in ("fc1_qweights", "fc1_block_scales",
                         "fc1_global_scales", "fc2_qweights",
                         "fc2_block_scales", "fc2_global_scales",
                         "e_score_correction_bias")
        }
        result = functional_moe.nvfp4_a16_moe(
            router,
            hidden,
            weights,
            2,
            1,
            16,
            128,
            functional_moe.MoeActivation.RELU2,
            1,
            1,
            1,
            1.0,
            functional_moe.MoeRouting.SIGMOID_GROUP_TOPK,
            weight_prefix="experts")

    assert captured["inputs"][0] is router
    assert captured["inputs"][1] is hidden
    assert result.ndim == 2
    assert operations == []


@pytest.mark.parametrize("moe_variant", ["fp16", "int4", "nvfp4_a16", "nvfp4"])
def test_experimental_moe_rejects_rank3_hidden_states(monkeypatch,
                                                      moe_variant):
    router = _MoeTensor(2, [])
    hidden = _MoeTensor(3, [])
    monkeypatch.setattr(functional_moe, "parameter",
                        lambda *args, **kwargs: object())
    monkeypatch.setattr(
        functional_moe, "operation",
        lambda *args, **kwargs: pytest.fail("operation called"))

    with pytest.raises(ValueError,
                       match=r"hidden_states must have shape \[T, H\]"):
        if moe_variant == "fp16":
            weights = {
                name: np.ones(1)
                for name in ("fc1_weights", "fc2_weights")
            }
            functional_moe.fp16_moe(
                router,
                hidden,
                weights,
                2,
                1,
                16,
                128,
                weight_prefix="experts",
                weight_bindings={name: None
                                 for name in weights})
        elif moe_variant == "int4":
            names = ("fc_gate_up_qweights", "fc_gate_up_scales",
                     "fc_down_qweights", "fc_down_scales")
            weights = {name: np.ones(1) for name in names}
            functional_moe.int4_moe(
                router,
                hidden,
                weights,
                2,
                1,
                16,
                128,
                64,
                weight_prefix="experts",
                weight_bindings={name: None
                                 for name in weights})
        elif moe_variant == "nvfp4_a16":
            names = ("fc1_qweights", "fc1_block_scales", "fc1_global_scales",
                     "fc2_qweights", "fc2_block_scales", "fc2_global_scales",
                     "e_score_correction_bias")
            functional_moe.nvfp4_a16_moe(
                router,
                hidden, {name: np.ones(1)
                         for name in names},
                2,
                1,
                16,
                128,
                functional_moe.MoeActivation.RELU2,
                1,
                1,
                1,
                1.0,
                functional_moe.MoeRouting.SIGMOID_GROUP_TOPK,
                weight_prefix="experts")
        else:
            names = ("fc1_qweights", "fc1_blocks_scale", "fc1_alpha",
                     "fc2_qweights", "fc2_blocks_scale", "fc2_alpha",
                     "input_global_scale", "down_input_scale",
                     "e_score_correction_bias")
            functional_moe.nvfp4_moe(
                router,
                hidden, {name: np.ones(1)
                         for name in names},
                2,
                1,
                16,
                128,
                functional_moe.MoeActivation.RELU2,
                1,
                1,
                1,
                1.0,
                functional_moe.MoeRouting.SIGMOID_GROUP_TOPK,
                False,
                weight_prefix="experts")


class _SliceTensor(_Tensor):

    def __init__(self, ndim, ranks):
        super().__init__(ndim)
        self.ranks = ranks

    def slice_last_dim(self, _offset, _size, rank):
        self.ranks.append(rank)
        return self

    def activation(self, _name):
        return self

    def __mul__(self, _other):
        return self


def _context():
    weights = SimpleNamespace(
        module_quant_type=lambda *args, **kwargs: "fp16",
        linear_descriptor=lambda *args, **kwargs: SimpleNamespace(
            in_features=16, out_features=16),
        shard_linear=lambda descriptor, *args: descriptor,
        fp16_parameter=lambda *args: SimpleNamespace(shape=(16, )),
        f16=lambda *args: None,
        linear_adapter=lambda *args: None,
    )
    cfg = SimpleNamespace(tp_size=1,
                          tp_rank=0,
                          rms_norm_eps=1e-6,
                          tie_word_embeddings=False)
    return BuildContext(net=SimpleNamespace(),
                        cfg=cfg,
                        weights=weights,
                        options=BuildOptions(),
                        bundle=SimpleNamespace(),
                        args=SimpleNamespace())


def test_linear_uses_the_actual_activation_rank(monkeypatch):
    ranks = []
    monkeypatch.setattr(
        F,
        "linear_from_weights",
        lambda hidden, descriptor, rank, name="": ranks.append(rank) or hidden)

    Linear(_context(), "model.layers.0.self_attn.q_proj")(_Tensor(2))

    assert ranks == [2]


def test_linear_preserves_explicit_encoder_rank(monkeypatch):
    ranks = []
    monkeypatch.setattr(
        F,
        "linear_from_weights",
        lambda hidden, descriptor, rank, name="": ranks.append(rank) or hidden)

    Linear(_context(), "visual.proj", rank=3,
           tensor_parallel=False)(_Tensor(2))

    assert ranks == [3]


def test_direct_weight_projection_infers_token_major_rank(monkeypatch):
    ranks = []
    backend = SimpleNamespace(linear_from_weights=lambda hidden, weights, rank,
                              name="": ranks.append(rank) or hidden)
    monkeypatch.setattr(functional_core, "current_net", lambda: backend)
    monkeypatch.setattr(functional_core, "tensor", lambda value: value)

    F.linear_from_weights(_Tensor(2), object(), name="fused_qkv")

    assert ranks == [2]


def test_direct_weight_projection_preserves_explicit_rank(monkeypatch):
    ranks = []
    backend = SimpleNamespace(linear_from_weights=lambda hidden, weights, rank,
                              name="": ranks.append(rank) or hidden)
    monkeypatch.setattr(functional_core, "current_net", lambda: backend)
    monkeypatch.setattr(functional_core, "tensor", lambda value: value)

    F.linear_from_weights(_Tensor(2), object(), rank=3, name="vision_proj")

    assert ranks == [3]


@pytest.mark.parametrize("sm110", [False, True], ids=["generic", "sm110"])
def test_nvfp4_a16_backend_accepts_token_major_rank(monkeypatch, sm110):
    from experimental.builder.weight_packing import nvfp4

    packed = (np.zeros((1, ), dtype=np.int8), np.zeros(
        (1, ), dtype=np.int8), np.zeros((1, ), dtype=np.float16), 16, 16)
    monkeypatch.setattr(nvfp4, "pack_nvfp4_a16_linear", lambda *_args: packed)
    monkeypatch.setattr(nvfp4, "pack_nvfp4_a16_blackwell_linear",
                        lambda *_args: packed)
    output = object()
    layer = SimpleNamespace(get_output=lambda _index: output)
    backend = SimpleNamespace(
        const=lambda value, _name: value,
        _unwrap=lambda value: value,
        operation=lambda *_args: layer,
        slice_last_dim=lambda value, *_args: value,
        _add_bias=lambda value, *_args: value,
    )
    weights = SimpleNamespace(weight=np.zeros((1, ), dtype=np.int8),
                              weight_scale=np.zeros((1, ), dtype=np.int8),
                              weight_scale_2=np.zeros((1, ), dtype=np.float16),
                              in_features=16,
                              bias=None)

    assert Net.nvfp4_a16_linear(backend,
                                object(),
                                weights,
                                rank=2,
                                sm110=sm110) is output


def test_rmsnorm_uses_the_actual_activation_rank(monkeypatch):
    ranks = []
    monkeypatch.setattr(
        F, "rms_norm", lambda hidden, weight, eps, rank, **kwargs: ranks.
        append(rank) or hidden)

    RMSNorm(_context(), "model.layers.0.input_layernorm")(_Tensor(2))

    assert ranks == [2]


def test_rmsnorm_preserves_explicit_encoder_rank(monkeypatch):
    ranks = []
    monkeypatch.setattr(
        F, "rms_norm", lambda hidden, weight, eps, rank, **kwargs: ranks.
        append(rank) or hidden)

    RMSNorm(_context(), "visual.norm", rank=3)(_Tensor(2))

    assert ranks == [3]


@pytest.mark.parametrize("projection",
                         (DFlashTargetProjection, DSparkTargetProjection))
def test_target_projection_uses_token_major_rank(monkeypatch, projection):
    ranks = []
    monkeypatch.setattr(
        F,
        "linear_f32_from_weights",
        lambda hidden, descriptor, name, rank=3: ranks.append(rank) or hidden)

    projection(_context(), "fc")(_Tensor(2))

    assert ranks == [2]


def test_phi4_multimodal_mlp_slices_token_major_activations():
    ranks = []
    mlp = object.__new__(Phi4MultimodalMLP)
    mlp.ctx = SimpleNamespace(
        cfg=SimpleNamespace(intermediate_size=16, hidden_act="silu"))
    mlp.gate_up_proj = lambda _hidden: _SliceTensor(2, ranks)
    mlp.down_proj = lambda hidden: hidden

    mlp.forward(_Tensor(2))

    assert ranks == [2, 2]
