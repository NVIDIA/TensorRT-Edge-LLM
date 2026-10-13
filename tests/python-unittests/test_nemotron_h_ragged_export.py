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

from tensorrt_edgellm.config import (LAYER_ATTN, LAYER_MAMBA, LAYER_MOE,
                                     QUANT_NVFP4, MambaConfig, ModelConfig,
                                     QuantConfig)
from tensorrt_edgellm.models.nemotron_h.modeling_nemotron_h import \
    NemotronHCausalLM
from tensorrt_edgellm.models.nemotron_h.modeling_nemotron_h_mtp import \
    NemotronHMtpDraftModel
from tensorrt_edgellm.onnx.export import _export_model


def _config() -> ModelConfig:
    return ModelConfig(model_type="nemotron_h",
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
                       layer_types=[LAYER_MAMBA, LAYER_ATTN],
                       mamba_cfg=MambaConfig(num_heads=2,
                                             head_dim=8,
                                             ssm_state_size=8,
                                             conv_dim=32,
                                             conv_kernel=4,
                                             n_groups=1))


def _moe_config() -> ModelConfig:
    config = _config()
    config.num_hidden_layers = 3
    config.layer_types = [LAYER_MAMBA, LAYER_MOE, LAYER_ATTN]
    config.quant = QuantConfig(quant_type=QUANT_NVFP4, group_size=16)
    config.n_routed_experts = 2
    config.num_experts_per_tok = 1
    config.moe_intermediate_size = 16
    config.moe_shared_expert_intermediate_size = 16
    return config


def test_nemotron_h_export_uses_token_major_resident_state_contract():
    spec = NemotronHCausalLM(_config()).onnx_export_spec()
    args = dict(zip(spec.input_names, spec.args))
    shapes = dict(zip(spec.input_names, spec.dynamic_shapes))

    assert args["inputs_embeds"].ndim == 2
    assert args["conv_state_0"].shape[0] > args["query_lengths"].shape[0]
    assert args["recurrent_state_0"].shape[0] > args["query_lengths"].shape[0]
    assert "context_lengths" not in args
    assert "state_indices" in args
    assert args["execution_phase_marker"].shape == (2, )
    assert args["context_sequence_count_carrier"].shape == args[
        "query_lengths"].shape
    assert shapes["context_sequence_count_carrier"][0] != shapes[
        "query_lengths"][0]
    assert "execution_phase_marker" in args
    assert "context_sequence_count_carrier" in args

    outputs = spec.wrapped(*spec.args)
    assert outputs[0].shape == (2, 32)
    assert outputs[1].shape == args["past_key_values_0"].shape
    assert outputs[2].shape == args["conv_state_0"].shape
    assert outputs[3].shape == args["recurrent_state_0"].shape


def test_nemotron_h_moe_exports_token_major_hidden_states(tmp_path):
    output = tmp_path / "nemotron-h-moe-ragged.onnx"

    _export_model(NemotronHCausalLM(_moe_config()),
                  str(output),
                  optimize=False)

    graph = onnx.load(str(output), load_external_data=False).graph
    moe = next(node for node in graph.node if node.op_type == "Nvfp4MoePlugin")
    assert moe.input[1]
    producers = {value: node for node in graph.node for value in node.output}
    value_info = {
        value.name: value
        for value in list(graph.input) + list(graph.value_info)
    }
    assert len(value_info[moe.input[0]].type.tensor_type.shape.dim) == 2
    assert len(value_info[moe.input[1]].type.tensor_type.shape.dim) == 2
    assert producers[moe.input[1]].op_type != "Unsqueeze"
    assert not any(node.op_type == "Reshape" and node.input[0] == moe.output[0]
                   for node in graph.node)
    nvfp4_activations = {
        value
        for node in graph.node if node.op_type == "DequantizeLinear"
        for value in node.output if value.startswith("nvfp4_act_qdq")
    }
    assert nvfp4_activations
    nvfp4_gemms = [
        node for node in graph.node
        if node.op_type == "Gemm" and node.input[0] in nvfp4_activations
    ]
    assert nvfp4_gemms
    assert all([
        len(value_info[name].type.tensor_type.shape.dim)
        for name in (*node.input[:2], node.output[0])
    ] == [2, 2, 2] for node in nvfp4_gemms)
    assert all(producers[node.input[1]].op_type != "Unsqueeze"
               for node in nvfp4_gemms)
    dynamic_quantize = [
        node for node in graph.node if node.op_type == "TRT_FP4DynamicQuantize"
    ]
    assert dynamic_quantize
    assert all(
        len(value_info[node.input[0]].type.tensor_type.shape.dim) == 2
        for node in dynamic_quantize)


def test_nemotron_h_export_routes_ragged_metadata_to_plugins(tmp_path):
    output = tmp_path / "nemotron-h-ragged.onnx"
    _export_model(NemotronHCausalLM(_config()), str(output), optimize=False)
    graph = onnx.load(str(output), load_external_data=False).graph
    nodes = {
        node.op_type: node
        for node in graph.node if node.op_type in
        {"AttentionPlugin", "causal_conv1d", "update_ssm_state"}
    }

    assert set(nodes) == {
        "AttentionPlugin", "causal_conv1d", "update_ssm_state"
    }
    assert set(nodes["AttentionPlugin"].input) >= {
        "query_start_offsets", "query_lengths", "past_lengths",
        "attention_sequence_lengths", "execution_phase_marker",
        "context_sequence_count_carrier"
    }
    for op_type in ("causal_conv1d", "update_ssm_state"):
        assert set(nodes[op_type].input) >= {
            "query_start_offsets", "query_lengths", "state_indices",
            "execution_phase_marker", "context_sequence_count_carrier"
        }


def test_nemotron_h_spec_base_logits_selection_has_independent_dynamic_axis(
        tmp_path):
    config = _config()
    config.mtp_base = True
    output = tmp_path / "nemotron-h-spec-base-ragged.onnx"

    _export_model(NemotronHCausalLM(config), str(output), optimize=False)

    graph = onnx.load(str(output), load_external_data=False).graph
    inputs = {tensor.name: tensor for tensor in graph.input}
    assert inputs["inputs_embeds"].type.tensor_type.shape.dim[
        0].dim_param == "physical_tokens"
    assert inputs["logits_indices"].type.tensor_type.shape.dim[
        0].dim_param == "logits_rows"


def test_nemotron_h_mtp_draft_exports_token_major_attention(tmp_path):
    config = _config()
    config.num_hidden_layers = 1
    config.layer_types = [LAYER_ATTN]
    output = tmp_path / "nemotron-h-mtp-ragged.onnx"
    model = NemotronHMtpDraftModel(config)
    spec = model.onnx_export_spec()
    args = dict(zip(spec.input_names, spec.args))
    shapes = dict(zip(spec.input_names, spec.dynamic_shapes))

    assert args["inputs_embeds"].ndim == 2
    assert "context_lengths" not in args
    assert "state_indices" in args
    assert args["execution_phase_marker"].shape == (2, )
    assert args["context_sequence_count_carrier"].shape == args[
        "query_lengths"].shape
    assert shapes["context_sequence_count_carrier"][0] != shapes[
        "query_lengths"][0]
    assert shapes["logits_indices"][0] != shapes["query_lengths"][0]
    outputs = spec.wrapped(*spec.args)
    assert outputs[0].shape == (2, 32)
    assert outputs[1].shape == (2, 16)

    proposal_args = list(spec.args)
    proposal_args[spec.input_names.index("logits_indices")] = torch.tensor(
        [0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.int64)
    proposal_outputs = spec.wrapped(*proposal_args)
    assert proposal_outputs[0].shape == (8, 32)
    assert proposal_outputs[1].shape == (8, 16)

    _export_model(model, str(output), optimize=False)
    graph = onnx.load(str(output), load_external_data=False).graph
    inputs = {tensor.name: tensor for tensor in graph.input}
    assert inputs["logits_indices"].type.tensor_type.shape.dim[
        0].dim_param == "logits_rows"
    attention = next(node for node in graph.node
                     if node.op_type == "AttentionPlugin")
    assert set(attention.input) >= {
        "query_start_offsets", "query_lengths", "past_lengths",
        "attention_sequence_lengths", "execution_phase_marker",
        "context_sequence_count_carrier"
    }


@pytest.mark.parametrize(
    "mode", ["vanilla", "dspark", "dspark_tree", "dflash", "mtp", "mtp_tree"])
def test_nemotron_spec_state_metadata_matches_decoder_mode(tmp_path, mode):
    config = _config()
    if mode != "vanilla":
        setattr(config, mode.split("_")[0] + "_base", True)
    if mode.startswith("dspark"):
        config.dspark_target_layer_ids = [0]
        config.dspark_tree_base = mode == "dspark_tree"
    if mode == "dflash":
        config.dflash_target_layer_ids = [0]
    if mode.startswith("mtp"):
        config.mtp_tree_base = mode == "mtp_tree"
    model = NemotronHCausalLM(config)
    spec = model.onnx_export_spec()
    assert len(spec.input_names) == len(spec.args) == len(spec.dynamic_shapes)
    state_names = {"tree_parent_ids", "tree_depths", "valid_tree_counts"}
    has_tree_state = mode in {"dspark_tree", "dflash", "mtp_tree"}
    assert state_names.issubset(spec.input_names) == has_tree_state
    if not has_tree_state:
        assert state_names.isdisjoint(spec.input_names)
    path = tmp_path / f"nemotron-{mode}.onnx"
    _export_model(model, str(path), optimize=False)
    graph = onnx.load(str(path), load_external_data=False).graph
    graph_inputs = {value.name for value in graph.input}
    assert state_names.issubset(graph_inputs) == has_tree_state
    if not has_tree_state:
        assert state_names.isdisjoint(graph_inputs)
    if mode != "vanilla":
        assert {"attention_position_ids",
                "packed_attention_mask"}.issubset(graph_inputs)
        assert any("replay" in value.name for value in graph.output)
    state_nodes = [
        node for node in graph.node
        if node.op_type in {"causal_conv1d", "update_ssm_state"}
    ]
    assert len(state_nodes) == 2
    for node in state_nodes:
        attributes = {
            attribute.name: attribute.i
            for attribute in node.attribute
        }
        assert bool(attributes.get("use_ddtree", 0)) == has_tree_state
