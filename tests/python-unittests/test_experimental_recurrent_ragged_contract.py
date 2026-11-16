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
"""Contract tests for experimental token-major recurrent operations."""

import dataclasses
from types import SimpleNamespace

import pytest

pytest.importorskip("tensorrt")

from tensorrt_edgellm.builder.core import config
from tensorrt_edgellm.builder.models.nemotron_h.modeling_nemotron_h import \
    NemotronHForCausalLM
from tensorrt_edgellm.builder.models.nemotron_omni.modeling_nemotron_omni_text import \
    NemotronOmniCausalLM
from tensorrt_edgellm.builder.models.qwen3_5.modeling_qwen3_5_text import \
    Qwen3_5ForCausalLM
from tensorrt_edgellm.builder.ops import (BuildContext, BuildOptions,
                                          NetworkModule)
from tensorrt_edgellm.builder.ops import functional as F
from tensorrt_edgellm.builder.ops.ragged import RaggedDecoderInputs
from tensorrt_edgellm.builder.ops.scope import build_scope


class _Value:

    def __init__(self, name, shape=(-1, )):
        self.name = name
        self.shape = shape
        self.dtype = None

    def __add__(self, other):
        return _Value(f"{self.name}+{other.name}", self.shape)


class _Layer:

    def __init__(self, count):
        self.outputs = tuple(
            _Value(f"output_{index}") for index in range(count))

    def get_output(self, index):
        return self.outputs[index]


class _RecordingNet:

    def __init__(self):
        self.calls = []

    def operation(self, name, attributes, inputs):
        output_count = {
            "causal_conv1d": 3,
            "gated_delta_net": 3,
            "update_ssm_state": 6
        }[name]
        self.calls.append((name, attributes, inputs))
        return _Layer(output_count)


def _context(net):
    return BuildContext(net=net,
                        cfg=SimpleNamespace(),
                        weights=SimpleNamespace(),
                        options=BuildOptions(),
                        bundle=SimpleNamespace(),
                        args=SimpleNamespace())


@pytest.mark.parametrize("channel_scales", [False, True])
def test_fp8_conv_projection_has_dense_channel_output(channel_scales):
    # Given FP8 projection weights, when lowering the conv partition, then
    # TensorRT receives a dense token-major conv projection and matching scales.
    import numpy as np
    import tensorrt as trt

    from tensorrt_edgellm.builder.core.weights import LinearWeights
    from tensorrt_edgellm.builder.models.nemotron_h.modeling_nemotron_h import \
        NemotronHMamba2Mixer
    from tensorrt_edgellm.builder.ops.backend import Net

    logger = trt.Logger(trt.Logger.ERROR)
    with trt.Builder(logger) as builder:
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        net = Net(builder, network)
        assert hasattr(net, "fp8_conv")
        projection_calls = []
        fp8_linear = net.fp8_linear
        fp8_conv = net.fp8_conv

        def record_fp8_linear(*args, **kwargs):
            projection_calls.append("linear")
            return fp8_linear(*args, **kwargs)

        def record_fp8_conv(*args, **kwargs):
            projection_calls.append("conv")
            return fp8_conv(*args, **kwargs)

        net.fp8_linear = record_fp8_linear
        net.fp8_conv = record_fp8_conv
        hidden = network.add_input("hidden", trt.float16, (-1, 16))
        weights = LinearWeights("fp8",
                                np.ones((50, 16), dtype=np.uint8),
                                input_scale=np.array(0.03125,
                                                     dtype=np.float16),
                                weight_scale=np.array(0.0625,
                                                      dtype=np.float16))
        if channel_scales:
            weights = dataclasses.replace(weights,
                                          weight_scale=np.full(
                                              50, 0.0625, dtype=np.float16))
        ctx = _context(net)
        ctx.cfg = SimpleNamespace(tp_size=1,
                                  tp_rank=0,
                                  mamba_cfg=SimpleNamespace(
                                      intermediate_size=16,
                                      conv_dim=32,
                                      num_heads=2))
        ctx.weights = SimpleNamespace(
            module_quant_type=lambda *args, **kwargs: "fp8",
            linear_descriptor=lambda *args, **kwargs: weights,
            shard_linear=lambda descriptor, *args: descriptor,
            linear_adapter=lambda *args: None)
        with build_scope(ctx):
            outputs = NemotronHMamba2Mixer(ctx, "mixer")._project_inputs(
                F.tensor(hidden))
        layers = [network.get_layer(i) for i in range(network.num_layers)]
        conv_channels = [
            layer.get_output(0).shape[1] for layer in layers
            if layer.type == trt.LayerType.CONVOLUTION
        ]
        conv = next(layer for layer in layers
                    if layer.type == trt.LayerType.CONVOLUTION)
        weight_dq = next(layer for layer in layers
                         if layer.get_output(0).name == conv.get_input(1).name)
        output_shapes = tuple(tuple(output.shape) for output in outputs)
        scale_shape = tuple(weight_dq.get_input(1).shape)
        expected_shapes = ((-1, 16), (-1, 32), (-1, 2))
        expected_scale_shape = (32, ) if channel_scales else (1, )
        assert (output_shapes, conv_channels, scale_shape,
                projection_calls) == (expected_shapes, [32],
                                      expected_scale_shape,
                                      ["linear", "conv", "linear"])


def test_weightless_fp8_projection_keeps_single_refit_parameter(monkeypatch):
    # Given a metadata-only FP8 matrix, when partitioning Mamba inputs, then
    # the checkpoint parameter remains whole and the projection output is sliced.
    import numpy as np

    from tensorrt_edgellm.builder.core.weights import (LinearWeights,
                                                       ParameterSpec)
    from tensorrt_edgellm.builder.models.nemotron_h.modeling_nemotron_h import \
        NemotronHMamba2Mixer

    descriptor = LinearWeights("fp8",
                               ParameterSpec((50, 16), np.uint8),
                               weight_scale=np.ones(1, np.float16))

    class Projection:

        def __init__(self):
            self.calls = []

        def quant_type(self):
            return "fp8"

        def has_adapter(self):
            return False

        def weight_descriptor(self):
            return descriptor

        def __call__(self, value):
            self.calls.append(value)
            return "projected"

    ctx = _context(_RecordingNet())
    ctx.cfg = SimpleNamespace(mamba_cfg=SimpleNamespace(
        intermediate_size=16, conv_dim=32, num_heads=2))
    mixer = NemotronHMamba2Mixer(ctx, "mixer")
    mixer.in_proj = Projection()
    slices = []
    monkeypatch.setattr(
        F, "slice_last_dim", lambda value, offset, size, rank: slices.append(
            (value, offset, size, rank)) or size)

    assert mixer._project_inputs("hidden") == (16, 32, 2)
    assert mixer.in_proj.calls == ["hidden"]
    assert slices == [("projected", 0, 16, 2), ("projected", 16, 32, 2),
                      ("projected", 48, 2, 2)]


def _ragged():
    return RaggedDecoderInputs(
        positions=_Value("positions"),
        query_start_offsets=_Value("query_start_offsets"),
        query_lengths=_Value("query_lengths"),
        past_lengths=_Value("past_lengths"),
        attention_sequence_lengths=_Value("attention_sequence_lengths"),
        state_indices=_Value("state_indices", (2, )),
        logits_indices=_Value("logits_indices"),
        execution_phase_marker=_Value("execution_phase_marker"),
        context_sequence_count_carrier=_Value(
            "context_sequence_count_carrier"),
        kv_page_table=_Value("kv_page_table"),
        spec_verify_layout=_Value("spec_verify_layout"),
    )


def _names(values):
    return [value.name for value in values]


@pytest.mark.parametrize("call,prefix_count,kwargs,output_count", (
    (F.causal_conv1d, 4, {
        "groups": 64,
        "padding": 3
    }, 3),
    (F.gated_delta_net, 8, {
        "key_head_dim": 128,
        "value_head_dim": 128
    }, 3),
    (F.update_ssm_state, 8, {
        "dim": 128,
        "dstate": 64,
        "nheads": 4,
        "ngroups": 1
    }, 6),
))
def test_verify_recurrent_operation_has_fixed_layout_contract(
        call, prefix_count, kwargs, output_count):
    # Given a verify-capable recurrent operation, when lowering it, then the
    # outputs and ancestry stay fixed; only conv/Mamba consume a layout marker.
    net = _RecordingNet()
    metadata = _ragged()
    ancestry = ["tree_parent_ids", "tree_depths"]
    if call is F.gated_delta_net:
        metadata = dataclasses.replace(metadata, spec_verify_layout=None)
    else:
        ancestry.append("spec_verify_layout")
    with build_scope(_context(net)):
        outputs = call(*[_Value(f"input_{i}") for i in range(prefix_count)],
                       metadata,
                       tree_parent_ids=_Value("tree_parent_ids"),
                       tree_depths=_Value("tree_depths"),
                       use_intermediate=True,
                       **kwargs)
    _, attributes, inputs = net.calls[0]
    assert (_names(inputs)[prefix_count + 5:],
            attributes.get("use_spec_verify_state"), "use_ddtree"
            in attributes, len(outputs)) == (ancestry, 1, False, output_count)


def test_causal_conv_uses_resident_slot_and_phase_order():
    net = _RecordingNet()
    with build_scope(_context(net)):
        F.causal_conv1d(_Value("x"), _Value("weight"), _Value("bias"),
                        _Value("resident_conv_pool"), _ragged(), 64, 3)

    name, attributes, inputs = net.calls[0]
    assert name == "causal_conv1d"
    assert attributes["use_spec_verify_state"] == 0
    assert _names(inputs) == [
        "x", "weight", "bias", "resident_conv_pool", "query_lengths",
        "query_start_offsets", "state_indices", "execution_phase_marker",
        "context_sequence_count_carrier"
    ]


def test_causal_conv_ddtree_appends_parent_and_depth_after_common_metadata():
    net = _RecordingNet()
    with build_scope(_context(net)):
        F.causal_conv1d(_Value("x"),
                        _Value("weight"),
                        _Value("bias"),
                        _Value("resident_conv_pool"),
                        _ragged(),
                        64,
                        3,
                        tree_parent_ids=_Value("tree_parent_ids"),
                        tree_depths=_Value("tree_depths"),
                        use_intermediate=True)

    _, attributes, inputs = net.calls[0]
    assert attributes["use_spec_verify_state"] == 1
    assert _names(inputs)[4:] == [
        "query_lengths", "query_start_offsets", "state_indices",
        "execution_phase_marker", "context_sequence_count_carrier",
        "tree_parent_ids", "tree_depths", "spec_verify_layout"
    ]


def test_gdn_ddtree_uses_explicit_tree_metadata_after_common_state_metadata():
    # Given GDN verification, when lowering ancestry, then both parent/depth
    # inputs follow resident metadata without a redundant topology marker.
    net = _RecordingNet()
    with build_scope(_context(net)):
        result = F.gated_delta_net(*[
            _Value(name) for name in ("q", "k", "v", "a", "b", "a_log",
                                      "dt_bias", "resident_state_pool")
        ],
                                   _ragged(),
                                   128,
                                   128,
                                   tree_parent_ids=_Value("tree_parent_ids"),
                                   tree_depths=_Value("tree_depths"),
                                   use_intermediate=True)

    _, attributes, inputs = net.calls[0]
    assert attributes["use_spec_verify_state"] == 1
    assert attributes["use_diffusion_state"] == 0
    assert _names(inputs)[8:] == [
        "query_lengths", "query_start_offsets", "state_indices",
        "execution_phase_marker", "context_sequence_count_carrier",
        "tree_parent_ids", "tree_depths"
    ]
    assert result[2] is not None


def test_mamba_uses_same_nonidentity_resident_slot_binding():
    net = _RecordingNet()
    values = [
        _Value(name) for name in ("x", "a", "b", "c", "d", "dt", "dt_bias",
                                  "resident_state_pool")
    ]
    with build_scope(_context(net)):
        result = F.update_ssm_state(*values,
                                    _ragged(),
                                    dim=128,
                                    dstate=64,
                                    nheads=4,
                                    ngroups=1,
                                    tree_parent_ids=_Value("tree_parent_ids"),
                                    tree_depths=_Value("tree_depths"),
                                    use_intermediate=True)

    name, attributes, inputs = net.calls[0]
    assert name == "update_ssm_state"
    assert attributes["use_spec_verify_state"] == 1
    assert attributes["dim"] == 128
    assert _names(inputs)[8:] == [
        "query_lengths", "query_start_offsets", "state_indices",
        "execution_phase_marker", "context_sequence_count_carrier",
        "tree_parent_ids", "tree_depths", "spec_verify_layout"
    ]
    assert len(result) == 6


@pytest.mark.parametrize(
    "call,kwargs",
    ((F.causal_conv1d, {
        "groups": 64,
        "padding": 3
    }), (F.gated_delta_net, {
        "key_head_dim": 128,
        "value_head_dim": 128
    })),
)
def test_tree_metadata_must_be_a_pair_and_requires_intermediate(call, kwargs):
    prefix_count = 4 if call is F.causal_conv1d else 8
    args = [_Value(f"input_{index}") for index in range(prefix_count)]
    with build_scope(_context(_RecordingNet())):
        with pytest.raises(ValueError, match="supplied together"):
            call(*args,
                 _ragged(),
                 tree_parent_ids=_Value("tree_parent_ids"),
                 **kwargs)
        with pytest.raises(ValueError, match="intermediate"):
            call(*args,
                 _ragged(),
                 tree_parent_ids=_Value("tree_parent_ids"),
                 tree_depths=_Value("tree_depths"),
                 **kwargs)


class _FakeHybridLayer:

    def __init__(self, states):
        self.states = states
        self.ragged = None

    def __call__(self, hidden_states, ragged, **kwargs):
        self.ragged = ragged
        return hidden_states, self.states


class _RecordingMambaMixer:

    def __init__(self):
        self.tree_parent_ids = None
        self.tree_depths = None

    def __call__(self, hidden_states, conv_state, recurrent_state, ragged,
                 tree_parent_ids, tree_depths, collect_intermediate):
        self.tree_parent_ids = tree_parent_ids
        self.tree_depths = tree_depths
        return (hidden_states, _Value("conv_out"), _Value("state_out"), None,
                None, None, None, None)


def _top_level_io():
    io = _ragged().as_dict()
    io.update({
        "inputs_embeds": _Value("inputs_embeds"),
        "past_key_values": [],
        "rope": _Value("rope"),
        "conv_states": [_Value("conv_state")],
        "recurrent_states": [_Value("recurrent_state")],
        "attention_mask": None,
        "attention_pos_id": None,
        "tree_parent_ids": None,
        "tree_depths": None,
        "valid_tree_counts": None,
    })
    return io


def test_qwen35_top_level_threads_nonidentity_ragged_state_indices(
        monkeypatch):
    cfg = SimpleNamespace(layer_types=[config.LAYER_GDN], engine_role="llm")
    model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
    NetworkModule.__init__(model, _context(_RecordingNet()))
    model.ctx.cfg = cfg
    layer = _FakeHybridLayer(
        (_Value("conv_out"), _Value("state_out"), None, None))
    model.layers = [layer]
    model.norm = lambda value: value
    model.lm_head = lambda value: _Value("logits")
    monkeypatch.setattr(F, "gather_token_rows", lambda value, indices: value)
    monkeypatch.setattr(F, "cast", lambda value, dtype: value)

    outputs = model.forward(**_top_level_io())

    assert layer.ragged.state_indices.name == "state_indices"
    assert outputs["present_conv_state_0"].name == "conv_out"
    assert outputs["present_recurrent_state_0"].name == "state_out"


@pytest.mark.parametrize("model_class",
                         (NemotronHForCausalLM, NemotronOmniCausalLM))
def test_nemotron_top_level_preserves_all_mamba_replay_bindings(
        monkeypatch, model_class):
    cfg = SimpleNamespace(layer_types=[config.LAYER_MAMBA], engine_role="base")
    model = model_class.__new__(model_class)
    NetworkModule.__init__(model, _context(_RecordingNet()))
    model.ctx.cfg = cfg
    states = tuple(
        _Value(name)
        for name in ("conv_out", "state_out", "inter_conv", "replay_da",
                     "replay_u", "replay_b", "replay_dt"))
    layer = _FakeHybridLayer(states)
    model.layers = [layer]
    model.norm = lambda value: value
    model.lm_head = lambda value, out_dtype=None: _Value("logits")
    monkeypatch.setattr(F, "gather_token_rows", lambda value, indices: value)
    monkeypatch.setattr(F, "cast", lambda value, dtype: value)
    monkeypatch.setattr(F, "hidden_state_feedback",
                        lambda *args, **kwargs: _Value("hidden_feedback"))

    outputs = model.forward(**_top_level_io())

    assert layer.ragged.state_indices.name == "state_indices"
    assert outputs["replay_da_state_0"].name == "replay_da"
    assert outputs["replay_u_state_0"].name == "replay_u"
    assert outputs["replay_b_state_0"].name == "replay_b"
    assert outputs["replay_dt_state_0"].name == "replay_dt"


@pytest.mark.parametrize(
    "module_name,class_name",
    (("tensorrt_edgellm.builder.models.nemotron_h.modeling_nemotron_h",
      "NemotronHBlock"),
     ("tensorrt_edgellm.builder.models.nemotron_omni.modeling_nemotron_omni_text",
      "NemotronOmniBlock")),
)
def test_nemotron_mamba_block_routes_ddtree_metadata_to_causal_conv_boundary(
        module_name, class_name):
    import importlib

    block_class = getattr(importlib.import_module(module_name), class_name)
    block = block_class.__new__(block_class)
    block.ctx = _context(_RecordingNet())
    block.ctx.cfg = SimpleNamespace()
    block.layer_type = config.LAYER_MAMBA
    block.input_norm = lambda value: value
    block.mixer = _RecordingMambaMixer()
    parent = _Value("tree_parent_ids")
    depth = _Value("tree_depths")

    block.forward(_Value("hidden"),
                  _ragged(),
                  conv_state=_Value("conv_state"),
                  recurrent_state=_Value("recurrent_state"),
                  tree_parent_ids=parent,
                  tree_depths=depth,
                  collect_intermediate=True)

    assert block.mixer.tree_parent_ids is parent
    assert block.mixer.tree_depths is depth
