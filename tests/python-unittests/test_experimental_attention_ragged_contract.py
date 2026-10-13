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
"""Operation-level contract tests for experimental decoder attention."""

from types import SimpleNamespace

import pytest

pytest.importorskip("tensorrt")

from experimental.builder.ops import BuildContext, BuildOptions
from experimental.builder.ops import functional as F
from experimental.builder.ops.ragged import RaggedDecoderInputs
from experimental.builder.ops.scope import build_scope
from experimental.builder.ops.transformer import pack_qkv


class _Value:

    def __init__(self, name, shape=(-1, )):
        self.name = name
        self.shape = shape
        self.dtype = None


class _Layer:

    def __init__(self, outputs):
        self.outputs = outputs

    def get_output(self, index):
        return self.outputs[index]


class _RecordingNet:

    def __init__(self):
        self.calls = []
        self.concat_axes = []
        self.reshape_calls = []
        self.inputs = []

    def add_input(self, name, dtype, shape):
        value = _Value(name, tuple(shape))
        value.dtype = dtype
        self.inputs.append(value)
        return value

    def operation(self, name, attributes, inputs):
        self.calls.append((name, attributes, inputs))
        return _Layer((_Value("attention_output", (-1, 4, 8)),
                       _Value("present_key_value", (2, -1, 128, 2, 8))))

    def concat(self, inputs, dim):
        self.concat_axes.append(dim)
        return _Value("packed_qkv", (-1, 64))

    def reshape(self, value, shape):
        self.reshape_calls.append((value, shape))
        return _Value("reshaped", tuple(shape))


def _context(net):
    return BuildContext(net=net,
                        cfg=SimpleNamespace(),
                        weights=SimpleNamespace(),
                        options=BuildOptions(),
                        bundle=SimpleNamespace(),
                        args=SimpleNamespace())


def _ragged():
    return RaggedDecoderInputs(
        positions=_Value("positions"),
        query_start_offsets=_Value("query_start_offsets"),
        query_lengths=_Value("query_lengths"),
        past_lengths=_Value("past_lengths"),
        attention_sequence_lengths=_Value("attention_sequence_lengths"),
        state_indices=_Value("state_indices"),
        logits_indices=_Value("logits_indices"),
        execution_phase_marker=_Value("execution_phase_marker"),
        context_sequence_count_carrier=_Value(
            "context_sequence_count_carrier"),
        kv_page_table=_Value("kv_page_table"),
    )


def _names(values):
    return [value.name for value in values]


@pytest.mark.parametrize("spec_decode_type",
                         ("dflash", "jetspec", "dspark", "eagle3"))
def test_speculative_hidden_feedback_concatenates_token_major_features(
        spec_decode_type):
    net = _RecordingNet()
    hidden_states = [_Value(f"hidden_{index}", (-1, 16)) for index in range(6)]
    config = SimpleNamespace(spec_decode_type=spec_decode_type,
                             dflash_target_layer_ids=[0, 1],
                             dspark_target_layer_ids=[0, 1],
                             eagle3_target_layer_ids=[1, 2, 3])

    with build_scope(_context(net)):
        F.hidden_state_feedback(hidden_states[-1], hidden_states, config)

    assert net.concat_axes == [1]


@pytest.mark.parametrize("invalid_layer_ids", ([0, 6], [-1, 0]))
def test_speculative_hidden_feedback_rejects_partially_invalid_layers(
        invalid_layer_ids):
    net = _RecordingNet()
    hidden_states = [_Value(f"hidden_{index}", (-1, 16)) for index in range(6)]
    config = SimpleNamespace(spec_decode_type="dflash",
                             dflash_target_layer_ids=invalid_layer_ids)

    with build_scope(_context(net)), pytest.raises(
            ValueError, match="DFlash target layer IDs are out of range"):
        F.hidden_state_feedback(hidden_states[-1], hidden_states, config)


def test_attention_uses_exact_unified_optional_and_ragged_input_order():
    net = _RecordingNet()
    ragged = _ragged()
    optional = {
        name: _Value(name)
        for name in (
            "q_norm_gamma",
            "k_norm_gamma",
            "context_mask_selector",
            "packed_attention_mask",
            "attention_position_ids",
        )
    }

    with build_scope(_context(net)):
        output, _ = F.attention(
            _Value("qkv", (-1, 64)),
            _Value("past_key_value"),
            _Value("rope_rotary_cos_sin", (-1, 8)),
            ragged,
            num_q_heads=4,
            num_kv_heads=2,
            head_size=8,
            q_norm_gamma=optional["q_norm_gamma"],
            k_norm_gamma=optional["k_norm_gamma"],
            context_mask_selector=optional["context_mask_selector"],
            attention_mask=optional["packed_attention_mask"],
            attention_pos_id=optional["attention_position_ids"],
            skip_softmax_scale_factor=1.5,
        )

    name, attributes, inputs = net.calls[0]
    assert name == "attention"
    assert attributes["enable_tree_attention"] == 1
    assert "enable_tree_metadata" not in attributes
    assert attributes["skip_softmax_scale_factor"] == 1.5
    assert _names(inputs) == [
        "qkv",
        "past_key_value",
        "query_lengths",
        "rope_rotary_cos_sin",
        "past_lengths",
        "kv_page_table",
        "q_norm_gamma",
        "k_norm_gamma",
        "context_mask_selector",
        "packed_attention_mask",
        "attention_position_ids",
        "skip_softmax_scale",
        "query_start_offsets",
        "attention_sequence_lengths",
        "execution_phase_marker",
        "context_sequence_count_carrier",
    ]
    assert output.shape == (-1, 4, 8)
    assert net.reshape_calls == []


def test_attention_places_vision_block_input_before_ragged_metadata():
    net = _RecordingNet()
    with build_scope(_context(net)):
        F.attention(
            _Value("qkv"),
            _Value("past_key_value"),
            _Value("rope_rotary_cos_sin"),
            _ragged(),
            num_q_heads=4,
            num_kv_heads=2,
            head_size=8,
            vision_block_ids=_Value("vision_block_ids"),
        )

    _, attributes, inputs = net.calls[0]
    assert attributes["enable_vision_block_attention"] == 1
    assert _names(inputs)[6:9] == [
        "vision_block_ids",
        "query_start_offsets",
        "attention_sequence_lengths",
    ]


@pytest.mark.parametrize(
    "kwargs,match",
    (
        ({
            "q_norm_gamma": _Value("q_norm_gamma")
        }, "supplied together"),
        ({
            "attention_mask": _Value("mask")
        }, "supplied together"),
        ({
            "attention_mask": _Value("mask"),
            "attention_pos_id": _Value("position"),
            "vision_block_ids": _Value("vision"),
        }, "mutually exclusive"),
    ),
)
def test_attention_rejects_inconsistent_optional_groups(kwargs, match):
    with build_scope(_context(_RecordingNet())):
        with pytest.raises(ValueError, match=match):
            F.attention(_Value("qkv"),
                        _Value("past"),
                        _Value("rope"),
                        _ragged(),
                        num_q_heads=4,
                        num_kv_heads=2,
                        head_size=8,
                        **kwargs)


def test_pack_qkv_concatenates_on_token_major_channel_axis():
    net = _RecordingNet()
    value_projection = SimpleNamespace(
        ctx=SimpleNamespace(
            backend="edgellm",
            options=SimpleNamespace(int4_gemm_plugin_version=1)),
        quant_type=lambda: "fp16",
    )
    with build_scope(_context(net)):
        packed = pack_qkv(_Value("q", (-1, 32)), _Value("k", (-1, 16)),
                          _Value("v", (-1, 16)), value_projection)

    assert packed.shape == (-1, 64)
    assert net.concat_axes == [1]


def test_dflash_cache_update_uses_token_owner_and_paged_pool_contract():
    net = _RecordingNet()
    net.operation = lambda name, attributes, inputs: _Layer(
        (_Value("present_key_value"), ))
    values = {
        name: _Value(name)
        for name in (
            "key_delta",
            "value_delta",
            "past_key_value",
            "delta_rope_cos_sin",
            "delta_positions",
            "delta_token_to_sequence",
            "kv_page_table",
        )
    }
    calls = []

    def record(name, attributes, inputs):
        calls.append((name, attributes, inputs))
        return _Layer((_Value("present_key_value"), ))

    net.operation = record
    with build_scope(_context(net)):
        F.update_dflash_target_cache(*values.values())

    assert calls[0][0] == "dflash_target_cache_update"
    assert _names(calls[0][2]) == list(values)
