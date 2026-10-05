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

import math

import onnx
import pytest
import torch

from tensorrt_edgellm.config import ModelConfig
from tensorrt_edgellm.models.gemma4.modeling_gemma4_assistant import \
    Gemma4AssistantForCausalLM
from tensorrt_edgellm.models.gemma4.modeling_gemma4_text import \
    Gemma4ForCausalLM
from tensorrt_edgellm.onnx.export import _export_model


def _config() -> ModelConfig:
    return ModelConfig(
        model_type="gemma4_unified_text",
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        head_dim=8,
        rms_norm_eps=1e-6,
        vocab_size=32,
        rope_theta=10_000.0,
        max_position_embeddings=4096,
        default_attention_scale=1.0 / math.sqrt(8.0),
        attention_layer_types=["sliding_attention", "full_attention"],
        layer_types=["attention", "attention"],
        sliding_window_size=16,
        use_vision_bidirectional_attention=True,
        hidden_size_per_layer_input=4,
        vocab_size_per_layer_input=32,
        sliding_rope_config={
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 1.0,
        },
        full_rope_config={
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 1.0,
        },
    )


def test_gemma4_ple_vision_export_uses_token_major_contract():
    spec = Gemma4ForCausalLM(_config()).onnx_export_spec()
    args = dict(zip(spec.input_names, spec.args))
    shapes = dict(zip(spec.input_names, spec.dynamic_shapes))

    assert args["inputs_embeds"].ndim == 2
    assert args["ple_token_embeds_0"].ndim == 2
    assert args["rope_rotary_cos_sin_sliding"].ndim == 2
    assert args["rope_rotary_cos_sin_full"].ndim == 2
    assert args["vision_block_ids"].ndim == 1
    assert args["vision_block_ids"].dtype == torch.int32
    assert "context_lengths" not in args
    assert "kvcache_start_index" not in args
    assert "last_token_ids" not in args
    for name in ("positions", "query_start_offsets", "query_lengths",
                 "past_lengths", "logits_indices"):
        assert name in args
    assert shapes["context_sequence_count_carrier"][0] != shapes[
        "query_lengths"][0]


def test_gemma4_tiny_onnx_connects_ragged_vision_metadata(tmp_path):
    output = tmp_path / "gemma4-ragged.onnx"
    _export_model(Gemma4ForCausalLM(_config()), str(output), optimize=False)
    graph = onnx.load(str(output), load_external_data=False).graph
    attention_nodes = [
        node for node in graph.node if node.op_type == "AttentionPlugin"
    ]

    assert len(attention_nodes) == 2
    for node in attention_nodes:
        assert set(node.input) >= {
            "query_start_offsets", "query_lengths", "past_lengths",
            "attention_sequence_lengths", "execution_phase_marker",
            "context_sequence_count_carrier", "vision_block_ids"
        }


def test_gemma4_tree_base_logits_selection_has_independent_dynamic_axis(
        tmp_path):
    config = _config()
    config.gemma4_mtp_base = True
    output = tmp_path / "gemma4-tree-base-ragged.onnx"

    _export_model(Gemma4ForCausalLM(config), str(output), optimize=False)

    graph = onnx.load(str(output), load_external_data=False).graph
    inputs = {tensor.name: tensor for tensor in graph.input}
    assert inputs["inputs_embeds"].type.tensor_type.shape.dim[
        0].dim_param == "physical_tokens"
    assert inputs["logits_indices"].type.tensor_type.shape.dim[
        0].dim_param == "logits_rows"
    assert inputs["packed_attention_mask"].type.tensor_type.shape.dim[
        1].dim_param == "packed_mask_width"


def test_gemma4_assistant_uses_token_major_shared_kv_contract():
    config = _config()
    config.backbone_hidden_size = config.hidden_size
    spec = Gemma4AssistantForCausalLM(config).onnx_export_spec()
    args = dict(zip(spec.input_names, spec.args))

    assert args["inputs_embeds"].ndim == 2
    assert args["hidden_states_input"].ndim == 2
    assert args["rope_rotary_cos_sin_sliding"].ndim == 2
    assert args["rope_rotary_cos_sin_full"].ndim == 2
    for name in ("positions", "query_start_offsets", "query_lengths",
                 "past_lengths", "attention_sequence_lengths", "state_indices",
                 "execution_phase_marker", "context_sequence_count_carrier"):
        assert name in args

    logits, hidden_states = spec.wrapped(*spec.args)
    assert logits.shape == (1, config.vocab_size)
    assert hidden_states.shape == (1, config.backbone_hidden_size)


def test_gemma4_mtp_ragged_base_emits_final_norm_hidden(monkeypatch):
    config = _config()
    config.gemma4_mtp_base = True
    model = Gemma4ForCausalLM(config)
    pre_norm_hidden = torch.full((1, config.hidden_size),
                                 2.0,
                                 dtype=torch.float16)
    final_norm_hidden = torch.full((1, config.hidden_size),
                                   3.0,
                                   dtype=torch.float16)
    model.model.last_pre_norm_hidden_states = pre_norm_hidden
    model.model.target_hidden_concat = None

    def fake_forward_ragged(*args, **kwargs):
        return final_norm_hidden, ()

    monkeypatch.setattr(model.model, "forward_ragged", fake_forward_ragged)
    token = torch.zeros((1, config.hidden_size), dtype=torch.float16)
    sequence = torch.zeros(1, dtype=torch.int32)
    emitted_hidden = model.forward_ragged(
        token,
        (),
        None,
        sequence,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
        sequence,
        torch.ones(1, dtype=torch.int32),
        sequence,
        torch.zeros(4, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.zeros((1, 2, 1), dtype=torch.int32),
        torch.zeros(1, dtype=torch.int64),
    )[1]

    torch.testing.assert_close(emitted_hidden, final_norm_hidden)


@pytest.mark.parametrize("num_layers", [8, 32, 36])
@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("target_layers", [(), (5, 1, 3)])
def test_gemma4_eagle_feedback_layers(monkeypatch, num_layers, ragged,
                                      target_layers):
    config = _config()
    config.num_hidden_layers = num_layers
    config.attention_layer_types = ["full_attention"] * num_layers
    config.layer_types = ["attention"] * num_layers
    config.hidden_size_per_layer_input = 0
    config.eagle_base = True
    config.eagle3_target_layer_ids = list(target_layers)
    model = Gemma4ForCausalLM(config)
    model.model.norm = torch.nn.Identity()

    def layer_forward(hidden_states, past_key_value, *args, **kwargs):
        return hidden_states + 1, past_key_value, None

    # Stub decoder arithmetic, leaving hidden-state collection and selection intact.
    for layer in model.model.layers:
        monkeypatch.setattr(layer, "forward", layer_forward)
        monkeypatch.setattr(layer, "forward_ragged", layer_forward)

    shape = (2, config.hidden_size) if ragged else (1, 2, config.hidden_size)
    hidden_states = tuple(
        torch.full(shape, float(index), dtype=torch.float16)
        for index in range(num_layers + 1))
    past_key_values = tuple(torch.empty(0) for _ in range(num_layers))
    rope = torch.empty(0)
    sequence = torch.zeros(1, dtype=torch.int32)
    lengths = torch.tensor([2], dtype=torch.int32)
    kv_page_table = torch.zeros((1, 2, 1), dtype=torch.int32)
    if ragged:
        emitted_hidden = model.forward_ragged(
            hidden_states[0],
            past_key_values,
            rope,
            torch.arange(2, dtype=torch.int32),
            torch.tensor([0, 2], dtype=torch.int32),
            lengths,
            sequence,
            lengths,
            sequence,
            torch.zeros(4, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            kv_page_table,
            torch.tensor([1], dtype=torch.int64),
        )[1]
    else:
        emitted_hidden = model(hidden_states[0], past_key_values, rope,
                               lengths, sequence, kv_page_table,
                               torch.tensor([[1]], dtype=torch.int64))[1]

    if target_layers:
        # Explicit Gemma4 layer IDs name block outputs in provider order.
        selected = [hidden_states[index + 1] for index in target_layers]
    else:
        selected = [
            hidden_states[2], hidden_states[num_layers // 2], hidden_states[-4]
        ]
    torch.testing.assert_close(emitted_hidden,
                               torch.cat(selected, dim=-1),
                               rtol=0,
                               atol=0)


def test_gemma4_eagle_default_hidden_output_survives_onnx_export(tmp_path):
    config = _config()
    config.eagle_base = True
    config.eagle3_target_layer_ids = []
    output = tmp_path / "gemma4-eagle-default-hidden.onnx"

    _export_model(Gemma4ForCausalLM(config), str(output), optimize=False)

    graph = onnx.load(str(output), load_external_data=False).graph
    assert [tensor.name for tensor in graph.output] == [
        "logits", "hidden_states", "present_key_values_0",
        "present_key_values_1"
    ]
    hidden_output = graph.output[1]
    assert hidden_output.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert hidden_output.type.tensor_type.shape.dim[0].dim_param == \
        "physical_tokens"
    assert hidden_output.type.tensor_type.shape.dim[1].dim_value == \
        3 * config.hidden_size
