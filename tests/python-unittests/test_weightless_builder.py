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
"""Behavioral CPU tests for the internal-TRT weightless builder contract."""

from types import SimpleNamespace

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from tensorrt_edgellm.builder.core import (builder, quantization,
                                           weight_policy, weights)
from tensorrt_edgellm.builder.core.artifacts.external_weights import \
    _share_tied_embedding
from tensorrt_edgellm.builder.ops import backend


class FakeNetwork:
    """Record constant descriptors, input tensors and canonical refit names."""

    def __init__(self):
        self.constants = []
        self.inputs = []
        self.names = []

    def add_constant(self, shape, descriptor):
        tensor = SimpleNamespace(shape=shape, dtype=descriptor[0])
        layer = SimpleNamespace(weights=descriptor,
                                name="",
                                get_output=lambda index: tensor)
        self.constants.append(layer)
        return layer

    def set_weights_name(self, descriptor, name):
        self.names.append((descriptor, name))
        return True

    def add_input(self, name, dtype, shape):
        tensor = SimpleNamespace(name=name, dtype=dtype, shape=shape)
        self.inputs.append(tensor)
        return tensor


@pytest.fixture
def network(monkeypatch):
    # Public TRT bindings need not accept the internal null-pointer constructor.
    monkeypatch.setattr(
        backend, "trt",
        SimpleNamespace(float16=trt.float16,
                        float32=trt.float32,
                        DataType=trt.DataType,
                        MatrixOperation=trt.MatrixOperation,
                        Weights=lambda *args: args))
    return FakeNetwork()


@pytest.fixture
def llm_config():
    return SimpleNamespace(model_type="qwen3",
                           num_experts=0,
                           quant=quantization.QuantConfig())


def test_proxy_is_named_null_constant_with_recipe(network):
    """FP16 metadata produces a named null constant and a refit recipe."""
    # Given FP16 metadata, when lowered weightlessly, then no weight input or payload is created.
    net = backend.Net(None, network, weightless_engine=True)
    name = "model.layers.0.self_attn.q_proj.weight"
    shape = (8, 4)
    recipe = {"checkpoint_keys": [name], "source_layout": "fp16"}
    value = weights.ParameterSpec(shape, np.float16)
    tensor = net.parameter(name, value, "fp16", recipe=recipe)
    assert tensor.shape == shape
    assert network.inputs == []
    assert network.constants[0].weights == (trt.float16, 0, 32)
    assert network.constants[0].name == name
    assert network.names == [(network.constants[0].weights, name)]
    assert net.take_weight_bindings() == [{
        "engine_name": name,
        "checkpoint_keys": [name],
        "source_layout": "fp16",
        "dtype": "F16",
        "shape": list(shape),
        "destination": "refit",
    }]
    assert recipe == {"checkpoint_keys": [name], "source_layout": "fp16"}


def test_externalized_parameter_still_uses_engine_input(network):
    """Non-weightless builds retain requested FP16 inputs without refit."""
    net = backend.Net(None,
                      network,
                      policy=weight_policy.WeightPolicy.from_request(
                          ("fp16", )))
    tensor = net.parameter("w", weights.ParameterSpec((4, ), np.float16),
                           "fp16", {"checkpoint_keys": ["w"]})
    assert tensor is network.inputs[0]
    assert network.constants == []
    assert "destination" not in net.take_weight_bindings()[0]


@pytest.mark.parametrize("kind", ("q8_0", "iq2_xxs"))
def test_weightless_plugin_input_weights_stay_engine_inputs(network, kind):
    """Packed plugin weights stay runtime-owned instead of becoming refit copies."""
    # Given a weightless build, when a plugin-input weight is lowered, then it is an engine input.
    net = backend.Net(None, network, weightless_engine=True)
    tensor = net.parameter("w", weights.ParameterSpec((8, 34), np.int8), kind,
                           {"checkpoint_keys": ["w"]})
    assert tensor is network.inputs[0]
    assert network.constants == []
    assert "destination" not in net.take_weight_bindings()[0]


@pytest.fixture
def shared_embedding():
    return {
        "engine_name": "__embedding__",
        "role": "embedding",
        "checkpoint_keys": ["embedding.weight"],
        "source_layout": "fp16",
        "dtype": "F16",
        "shape": [8, 4],
        "embedding_scale": 1.0,
    }


def test_refit_preserves_compatible_embedding_input(network, shared_embedding):
    """Matching tied weights share persistent storage without refitting the head."""
    net = backend.Net(None,
                      network,
                      weightless_engine=True,
                      shared_embedding=shared_embedding)
    recipe = {"checkpoint_keys": ["embedding.weight"], "source_layout": "fp16"}
    value = weights.ParameterSpec((8, 4), np.float16)
    tensor = net.parameter("decoder.output.weight",
                           value,
                           "fp16",
                           recipe=recipe)
    assert network.inputs == [tensor]
    assert network.constants == []
    bindings = net.take_weight_bindings()
    assert "destination" not in bindings[0]
    _share_tied_embedding([*bindings, shared_embedding], True)
    assert shared_embedding["storage_alias_of"] == "decoder.output.weight"


def test_refit_keeps_incompatible_embedding_weight_constant(
        network, shared_embedding):
    """Different checkpoint materialization cannot bypass refit or alias storage."""
    shared_embedding["checkpoint_keys"] = ["different.weight"]
    net = backend.Net(None,
                      network,
                      weightless_engine=True,
                      shared_embedding=shared_embedding)
    net.parameter("lm_head.weight", weights.ParameterSpec((8, 4), np.float16),
                  "fp16", {
                      "checkpoint_keys": ["embedding.weight"],
                      "source_layout": "fp16"
                  })
    assert network.inputs == []
    assert network.constants[0].weights == (trt.float16, 0, 32)
    assert net.take_weight_bindings()[0]["destination"] == "refit"


def test_weightless_scope_rejects_explicit_external_weights():
    """Refit owns weight routing and rejects an explicit external selection."""
    # Given explicit weight routing, when validating refit mode, then the conflicting selection is rejected.
    args = builder.BuildArgs("checkpoint",
                             "engine",
                             weightless_engine=True,
                             externalize_weights=("fp16", ))
    with pytest.raises(ValueError, match="weightless-engine"):
        args.validate()


@pytest.mark.parametrize("enabled", [False, True])
def test_weightless_flags_only_modify_opt_in_builds(llm_config, enabled):
    """Refit flags depend on explicit opt-in."""
    # Given a direct build, when configuring, then refit flags depend on opt-in.
    args = builder.BuildArgs("checkpoint", "engine", weightless_engine=enabled)
    args.validate()
    flags = []
    config = SimpleNamespace(set_flag=flags.append,
                             builder_optimization_level=5)
    builder._configure_weightless_engine(args, llm_config, config)
    assert flags == ([
        trt.BuilderFlag.STRIP_PLAN, trt.BuilderFlag.REFIT_INDIVIDUAL
    ] if enabled else [])
    assert config.builder_optimization_level == 5


def test_weightless_requires_component_quantization_metadata():
    """Auxiliary refit configuration requires component quantization metadata."""
    # Given missing component metadata, when configuring refit, then no builder flags are set.
    args = builder.BuildArgs("checkpoint", "engine", weightless_engine=True)
    flags = []
    with pytest.raises(ValueError, match="weightless-engine"):
        builder._configure_weightless_engine(
            args, None, SimpleNamespace(set_flag=flags.append))
    assert flags == []


def test_refit_preserves_parameter_storage_type(network):
    """Packed plugin parameters retain their actual storage dtype and count."""
    net = backend.Net(None, network, weightless_engine=True)
    value = weights.ParameterSpec((4, 8), np.int32)
    net.parameter("w", value, "int4_ffn", {"checkpoint_keys": ["w"]})
    assert network.constants[0].weights == (backend._NP_TO_TRT[value.dtype], 0,
                                            32)
    assert net.take_weight_bindings(
    )[0]["dtype"] == backend._SAFETENSORS_DTYPE[value.dtype]


def test_typed_refit_separates_logical_shape_from_storage(network):
    """Quantized constants use logical counts and byte-addressable recipes."""
    shape = (4, 16)
    net = backend.Net(None, network, weightless_engine=True)
    net._refit_parameter("w",
                         weights.ParameterSpec((4, 8), np.uint8),
                         "quantized", {"checkpoint_keys": ["w"]},
                         trt_dtype=trt.DataType.FP4,
                         shape=shape,
                         storage_dtype="U8")
    assert network.constants[0].weights == (trt.DataType.FP4, 0,
                                            int(np.prod(shape)))
    binding = net.take_weight_bindings()[0]
    assert binding["shape"] == [4, 8]
    assert binding["dtype"] == "U8"
    assert binding["destination"] == "refit"


@pytest.mark.parametrize("missing_flag", ["STRIP_PLAN", "REFIT_INDIVIDUAL"])
def test_missing_trt_flags_fail_with_dependency_diagnostic(
        monkeypatch, llm_config, missing_flag):
    """Missing TensorRT flags produce an actionable dependency error."""
    # Given a TRT without refit flags, when configuring, then startup fails with an actionable dependency error.
    flags = {
        name: getattr(trt.BuilderFlag, name)
        for name in ("STRIP_PLAN", "REFIT_INDIVIDUAL") if name != missing_flag
    }
    monkeypatch.setattr(builder, "trt",
                        SimpleNamespace(BuilderFlag=SimpleNamespace(**flags)))
    args = builder.BuildArgs("checkpoint", "engine", weightless_engine=True)
    with pytest.raises(RuntimeError, match="internal TensorRT"):
        builder._configure_weightless_engine(args, llm_config,
                                             SimpleNamespace())


def test_dense_linear_routes_checkpoint_metadata_to_refit(
        network, monkeypatch):
    """Dense linear weights use refit constants rather than engine inputs."""
    # Given a dense projection, when lowering the actual linear path, then its matrix is a refit constant.
    net = backend.Net(None,
                      network,
                      policy=weight_policy.WeightPolicy.from_request(),
                      weightless_engine=True)
    name = "model.layers.0.mlp.gate_proj"
    descriptor = weights.LinearWeights(
        quant_type="fp16",
        weight=weights.ParameterSpec((8, 4), np.float16),
        weight_recipe={"checkpoint_keys": [name + ".weight"]})
    operands = []
    monkeypatch.setattr(net, "matmul",
                        lambda *args: operands.append(args) or args[1])
    result = net.linear_from_weights(object(), descriptor, rank=2, name=name)
    assert operands[0][1] is result
    assert network.inputs == []
    assert network.names[0][1] == name + ".weight"
    assert net.take_weight_bindings()[0]["destination"] == "refit"


def test_fp32_linear_routes_checkpoint_metadata_to_refit(network, monkeypatch):
    net = backend.Net(None, network, weightless_engine=True)
    name = "model.layers.0.mlp.gate_proj"
    descriptor = weights.LinearWeights(
        quant_type="fp16",
        weight=weights.ParameterSpec((8, 4), np.float16),
        weight_recipe={"checkpoint_keys": [name + ".weight"]})
    operands = []
    monkeypatch.setattr(net, "cast", lambda value, dtype: value)
    monkeypatch.setattr(net, "matmul",
                        lambda *args: operands.append(args) or args[1])
    result = net.linear_f32_from_weights(object(), descriptor, name, rank=2)
    assert operands[0][1] is result
    assert network.inputs == []
    assert network.names[0][1] == name + ".weight"
    assert net.take_weight_bindings()[0]["destination"] == "refit"


def test_fp32_linear_widens_an_external_weight_once_at_load(
        network, monkeypatch):
    """An FP32-accumulation projection with widen_at_load binds its external
    FP16 checkpoint weight as an FP32 engine input filled by cast_to_fp32."""
    # Given an externalized FP16 projection, when lowered with widen_at_load, then the engine takes an FP32 input
    # whose binding widens the checkpoint tensor, and only the activation is cast inside the engine.
    net = backend.Net(None,
                      network,
                      policy=weight_policy.WeightPolicy(
                          kinds=(weight_policy.EXTERNAL_WEIGHT_FP16, )))
    descriptor = weights.LinearWeights(quant_type="fp16",
                                       weight=weights.ParameterSpec(
                                           (8, 4), np.float16),
                                       weight_recipe={
                                           "checkpoint_keys": ["fc.weight"],
                                           "source_layout": "fp16"
                                       })
    casts = []
    monkeypatch.setattr(net, "cast",
                        lambda value, dtype: casts.append(value) or value)
    monkeypatch.setattr(net, "matmul", lambda *args: args[1])
    activation = object()

    weight = net.linear_f32_from_weights(activation,
                                         descriptor,
                                         "fc",
                                         rank=2,
                                         widen_at_load=True)

    weight_input, = network.inputs
    assert weight is weight_input
    assert (weight_input.name, weight_input.dtype,
            tuple(weight_input.shape)) == ("fc.weight", trt.float32, (8, 4))
    binding, = net.take_weight_bindings()
    assert (binding["assemble"], binding["dtype"],
            binding["checkpoint_keys"]) == ("cast_to_fp32", "F32",
                                            ["fc.weight"])
    assert casts == [activation]


@pytest.mark.parametrize("accumulate_f32", [False, True])
def test_weightless_linear_keeps_materialized_weight_with_recipe(
        network, monkeypatch, accumulate_f32):
    net = backend.Net(None, network, weightless_engine=True)
    name = "lm_head"
    weight = np.ones((8, 4), dtype=np.float16)
    descriptor = weights.LinearWeights(
        quant_type="fp16",
        weight=weight,
        weight_recipe={"checkpoint_keys": [name + ".weight"]})
    if accumulate_f32:
        constants = []
        monkeypatch.setattr(net, "cast", lambda value, dtype: value)
        monkeypatch.setattr(
            net, "const",
            lambda value, layer_name: constants.append(value) or value)
        monkeypatch.setattr(net, "matmul", lambda *args: args[1])
        result = net.linear_f32_from_weights(object(),
                                             descriptor,
                                             name,
                                             rank=2)
        assert result is constants[0]
    else:
        monkeypatch.setattr(net, "linear", lambda x, value, bias, rank: value)
        result = net.linear_from_weights(object(),
                                         descriptor,
                                         rank=2,
                                         name=name)
        assert result is weight
    assert network.inputs == []
    assert network.names == []
    assert net.take_weight_bindings() == []


def test_weightless_fp16_descriptor_uses_checkpoint_metadata(monkeypatch):
    loader = weights.Weights.__new__(weights.Weights)
    loader._weightless_engine = True
    descriptor = weights.LinearWeights(
        quant_type="fp16",
        weight=weights.ParameterSpec((8, 4), np.float16),
        weight_recipe={"checkpoint_keys": ["model.embed_tokens.weight"]})
    monkeypatch.setattr(loader, "linear_metadata",
                        lambda prefix, quant_type: descriptor)
    monkeypatch.setattr(
        loader, "linear", lambda prefix, quant_type: pytest.fail(
            "weightless FP16 weight was materialized"))
    assert loader.linear_descriptor("lm_head", "fp16") is descriptor


def test_cli_defaults_off_and_passes_explicit_opt_in():
    """CLI builds use weightless mode only when explicitly requested."""
    # Given CLI arguments, when parsed into BuildArgs, then weightless mode is disabled unless requested.
    from tensorrt_edgellm.builder import cli
    parser = cli._parser()
    for enabled in (False, True):
        command = ["--model-dir", "checkpoint", "--engine-dir", "engine"]
        if enabled:
            command.append("--weightless-engine")
        args = parser.parse_args(command)
        assert cli._build_args(args, "llm").weightless_engine is enabled
