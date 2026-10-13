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
import re

import onnx
import torch

from tensorrt_edgellm.config import (LAYER_ATTN, LAYER_GDN, QUANT_INT4_AWQ,
                                     QUANT_NVFP4, GdnConfig, ModelConfig,
                                     QuantConfig)
from tensorrt_edgellm.models.default.modeling_default import (
    CausalLM, _concat_hidden_in_provider_order, fuse_qkv_projections)
from tensorrt_edgellm.models.linear import AWQLinear
from tensorrt_edgellm.models.ops import (attention_plugin, gated_delta_net,
                                         set_int4_gemm_plugin_version)
from tensorrt_edgellm.models.qwen3_5.modeling_qwen3_5_text import (
    Qwen3_5CausalLM, Qwen3_5DecoderLayer)
from tensorrt_edgellm.models.qwen3_omni_next.modeling_qwen3_omni_next_code_predictor import \
    Qwen3OmniNextCodePredictorCausalLM
from tensorrt_edgellm.models.qwen3_omni_next.modeling_qwen3_omni_next_moe_text import \
    Qwen3OmniNextMoeLanguageModel
from tensorrt_edgellm.models.qwen3_omni_next.modeling_qwen3_omni_next_talker import \
    Qwen3OmniNextTalkerCausalLM
from tensorrt_edgellm.models.qwen3_tts.modeling_code_predictor import \
    CodePredictorCausalLM
from tensorrt_edgellm.models.qwen3_tts.modeling_qwen3_tts_talker import \
    TalkerCausalLM
from tensorrt_edgellm.onnx.dynamo_translations import \
    build_custom_translation_table
from tensorrt_edgellm.onnx.export import (_export_model,
                                          _permissive_inline_opset)
from tensorrt_edgellm.onnx.onnx_custom_schemas import _attention_plugin_schema


def _config() -> ModelConfig:
    return ModelConfig(model_type="qwen3_5_text",
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
                       layer_types=[LAYER_GDN, LAYER_ATTN],
                       gdn_cfg=GdnConfig(num_key_heads=2,
                                         num_value_heads=2,
                                         key_head_dim=8,
                                         value_head_dim=8,
                                         conv_kernel=4),
                       attn_output_gate=True)


def _attention_config() -> ModelConfig:
    config = _config()
    config.model_type = "qwen3"
    config.num_hidden_layers = 1
    config.layer_types = [LAYER_ATTN]
    config.gdn_cfg = None
    config.attn_output_gate = False
    config.num_deepstack_features = 1
    return config


def _omni_next_moe_config() -> ModelConfig:
    config = _config()
    config.model_type = "qwen3_omni_next_moe"
    config.num_experts = 2
    config.num_experts_per_tok = 1
    config.moe_intermediate_size = 64
    config.accept_hidden_layer = 1
    config.mtp_base = True
    return config


def test_qwen35_vanilla_export_uses_token_major_ragged_abi():
    spec = Qwen3_5CausalLM(_config()).onnx_export_spec()
    args = dict(zip(spec.input_names, spec.args))
    shapes = dict(zip(spec.input_names, spec.dynamic_shapes))

    expected_metadata = {
        "positions": torch.int32,
        "query_start_offsets": torch.int32,
        "query_lengths": torch.int32,
        "past_lengths": torch.int32,
        "state_indices": torch.int32,
        "context_sequence_count_carrier": torch.int32,
        "logits_indices": torch.int64,
    }
    assert args["inputs_embeds"].ndim == 2
    assert args["rope_rotary_cos_sin"].ndim == 2
    assert "context_lengths" not in args
    assert "kvcache_start_index" not in args
    assert "last_token_ids" not in args
    assert "token_to_sequence" not in args

    for name, dtype in expected_metadata.items():
        assert args[name].ndim == 1
        assert args[name].dtype == dtype
        assert 0 in shapes[name]

    context_dim = shapes["context_sequence_count_carrier"][0]
    assert context_dim != shapes["query_lengths"][0]
    assert context_dim.min == 0

    assert args["query_start_offsets"].shape[
        0] == args["query_lengths"].shape[0] + 1
    assert args["conv_state_0"].shape[0] > args["query_lengths"].shape[0]
    assert args["recurrent_state_0"].shape[0] > args["query_lengths"].shape[0]


class _GdnDiffusionCapabilityModule(torch.nn.Module):

    def forward(self, q, k, v, a, b, a_log, dt_bias, state, query_lengths,
                query_start_offsets, state_indices, execution_phase_marker,
                context_sequence_count_carrier):
        return gated_delta_net(q, k, v, a, b, a_log, dt_bias, state,
                               query_lengths, 8, 8, query_start_offsets,
                               state_indices, execution_phase_marker,
                               context_sequence_count_carrier, True)


def test_official_gdn_op_exports_diffusion_state_capability(tmp_path):
    q = torch.zeros(2, 2, 8, dtype=torch.float16)
    gate = torch.zeros(2, 2, dtype=torch.float16)
    a_log = torch.zeros(2, dtype=torch.float32)
    state = torch.zeros(2, 2, 8, 8, dtype=torch.float32)
    query_lengths = torch.ones(2, dtype=torch.int32)
    query_start_offsets = torch.tensor([0, 1, 2], dtype=torch.int32)
    state_indices = torch.tensor([0, 1], dtype=torch.int32)
    phase = torch.ones(1, dtype=torch.int32)
    context_count = torch.zeros(0, dtype=torch.int32)
    args = (q, q, q, gate, gate, a_log, gate[0], state, query_lengths,
            query_start_offsets, state_indices, phase, context_count)

    with _permissive_inline_opset():
        program = torch.onnx.export(
            _GdnDiffusionCapabilityModule().eval(),
            args,
            dynamo=True,
            opset_version=24,
            custom_translation_table=build_custom_translation_table(),
            optimize=False)

    output = tmp_path / "gdn-diffusion-capability.onnx"
    program.save(str(output))
    onnx.checker.check_model(str(output))
    node = next(node for node in onnx.load(str(output)).graph.node
                if node.op_type == "gated_delta_net")
    attributes = {attribute.name: attribute.i for attribute in node.attribute}
    assert attributes["use_diffusion_state"] == 1
    assert len(node.input) == 13
    assert len(node.output) == 2


def test_qwen35_ragged_layer_flattens_input_before_gdn_projections():

    class RankThreeGdn(torch.nn.Module):

        def forward_ragged(self, hidden_states, *_args, **_kwargs):
            assert hidden_states.shape == (6, 16)
            return hidden_states, None, None, None, None

    layer = Qwen3_5DecoderLayer(_config(), _config().gdn_cfg, 0, LAYER_GDN)
    layer.input_layernorm = torch.nn.Identity()
    layer.post_attention_layernorm = torch.nn.Identity()
    layer.linear_attn = RankThreeGdn()
    layer.mlp = torch.nn.Identity()

    output = layer.forward_ragged(torch.randn(2, 3, 16, dtype=torch.float16),
                                  conv_state=None,
                                  recurrent_state=None,
                                  query_start_offsets=None,
                                  query_lengths=None,
                                  state_indices=None,
                                  execution_phase_marker=None,
                                  context_sequence_count_carrier=None)[0]

    assert output.shape == (6, 16)


def test_attention_plugin_signature_freezes_stage1_contract():
    schema = str(attention_plugin._schema)
    parameters = re.findall(
        r"(?:Tensor|SymInt|bool|float)(?:\[\])?\?? ([a-zA-Z0-9_]+)",
        schema.split(" -> ", maxsplit=1)[0])

    assert parameters[:6] == [
        "qkv", "past_key_value", "query_lengths", "rope_rotary_cos_sin",
        "past_lengths", "kv_page_table"
    ]
    assert parameters[-4:] == [
        "query_start_offsets", "attention_sequence_lengths",
        "execution_phase_marker", "context_sequence_count_carrier"
    ]
    for redundant_input in ("positions", "token_to_sequence",
                            "sequence_lengths", "state_indices"):
        assert redundant_input not in parameters
    assert "context_mask_selector" in parameters

    schema_types = {
        parameter.name: parameter.type_str
        for parameter in _attention_plugin_schema.inputs
    }
    assert schema_types["query_lengths"] == "tensor(int32)"
    assert schema_types["skip_softmax_scale"] == "tensor(int8)"
    assert schema_types["swa_kv_cache_mode"] == "tensor(int8)"
    assert schema_types["context_sequence_count_carrier"] == "tensor(int32)"


def test_qwen35_speculative_base_uses_unified_token_major_contract():
    config = _config()
    config.mtp_base = True
    spec = Qwen3_5CausalLM(config).onnx_export_spec()
    args = dict(zip(spec.input_names, spec.args))

    assert args["inputs_embeds"].ndim == 2
    assert "context_lengths" not in args
    assert "last_token_ids" not in args
    assert "positions" in args
    assert "state_indices" in args
    assert "attention_sequence_lengths" in args
    assert "execution_phase_marker" in args


def test_qwen35_mtp_base_tiny_onnx_supports_linear_verify(tmp_path):
    config = _config()
    config.mtp_base = True
    output = tmp_path / "qwen35-mtp-base.onnx"

    _export_model(Qwen3_5CausalLM(config), str(output), optimize=False)

    graph = onnx.load(str(output), load_external_data=False).graph
    inputs = {tensor.name: tensor for tensor in graph.input}
    assert inputs["logits_indices"].type.tensor_type.shape.dim[
        0].dim_param == "logits_rows"
    nodes = {
        node.op_type: node
        for node in graph.node
        if node.op_type in {"causal_conv1d", "gated_delta_net"}
    }
    assert set(nodes) == {"causal_conv1d", "gated_delta_net"}
    for node in nodes.values():
        attributes = {
            attribute.name: attribute.i
            for attribute in node.attribute
        }
        assert attributes.get("use_ddtree", 0) == 0


def test_qwen35_awq_v1_decoder_linears_are_token_major(tmp_path):
    config = _config()
    config.hidden_size = 128
    config.intermediate_size = 128
    config.vocab_size = 128
    config.head_dim = 64
    config.num_attention_heads = 2
    config.num_key_value_heads = 1
    config.gdn_cfg = GdnConfig(num_key_heads=2,
                               num_value_heads=2,
                               key_head_dim=64,
                               value_head_dim=64,
                               conv_kernel=4)
    config.quant = QuantConfig(quant_type=QUANT_INT4_AWQ,
                               group_size=128,
                               excluded=["lm_head"])
    output = tmp_path / "qwen35-awq-v1.onnx"
    set_int4_gemm_plugin_version(1)
    try:
        model = Qwen3_5CausalLM(config)
        for module in model.modules():
            if isinstance(module, AWQLinear):
                module.qweight = torch.zeros(module.out_features // 2,
                                             module.in_features,
                                             dtype=torch.int8)
        _export_model(model, str(output), optimize=True)
    finally:
        set_int4_gemm_plugin_version(2)

    graph = onnx.shape_inference.infer_shapes(
        onnx.load(str(output), load_external_data=False)).graph
    values = {
        value.name: value
        for value in (*graph.input, *graph.value_info, *graph.output)
    }
    linear_nodes = [
        node for node in graph.node
        if node.op_type in {"Gemm", "Int4GroupwiseGemmPlugin"}
    ]
    assert linear_nodes
    assert {
        node.name: len(values[node.input[0]].type.tensor_type.shape.dim)
        for node in linear_nodes
    } == {
        node.name: 2
        for node in linear_nodes
    }
    assert {
        node.name: len(values[node.output[0]].type.tensor_type.shape.dim)
        for node in linear_nodes
    } == {
        node.name: 2
        for node in linear_nodes
    }
    qkv_concat = next(node for node in graph.node
                      if node.op_type == "QkvConcatPlugin")
    assert len(values[qkv_concat.output[0]].type.tensor_type.shape.dim) == 2


def test_qwen35_dflash_linear_base_keeps_tree_attention_without_metadata(
        tmp_path):
    config = _config()
    config.dflash_base = True
    output = tmp_path / "qwen35-dflash-base.onnx"

    _export_model(Qwen3_5CausalLM(config), str(output), optimize=False)

    graph = onnx.load(str(output), load_external_data=False).graph
    inputs = {tensor.name for tensor in graph.input}
    assert {"attention_position_ids", "packed_attention_mask"} <= inputs
    assert not {"tree_parent_ids", "tree_depths", "valid_tree_counts"} & inputs
    attention = next(node for node in graph.node
                     if node.op_type == "AttentionPlugin")
    attributes = {
        attribute.name: attribute.i
        for attribute in attention.attribute
    }
    assert attributes["enable_tree_attention"] == 1
    assert "enable_tree_metadata" not in attributes
    assert set(attention.input) >= {
        "attention_position_ids", "packed_attention_mask"
    }
    assert all(attention.input)


def test_qwen35_jetspec_base_emits_target_hidden():
    config = _config()
    config.jetspec_base = True
    config.jetspec_target_layer_ids = [0, 1]

    spec = Qwen3_5CausalLM(config).onnx_export_spec()
    outputs = spec.wrapped(*spec.args)

    assert spec.output_names[1] == "hidden_states"
    assert outputs[1].shape == (spec.args[0].shape[0], 2 * config.hidden_size)


def test_target_hidden_concat_preserves_provider_order():
    layer_zero = torch.zeros((2, 3), dtype=torch.float16)
    layer_one = torch.ones((2, 3), dtype=torch.float16)

    actual = _concat_hidden_in_provider_order({
        0: layer_zero,
        1: layer_one
    }, [1, 0])

    torch.testing.assert_close(actual,
                               torch.cat((layer_one, layer_zero), dim=-1))


def test_qwen35_mtp_tree_base_exports_tree_state_metadata():
    config = _config()
    config.mtp_base = True
    config.mtp_tree_base = True

    spec = Qwen3_5CausalLM(config).onnx_export_spec()
    args = dict(zip(spec.input_names, spec.args))

    assert args["tree_parent_ids"].shape == args["positions"].shape
    assert args["tree_depths"].shape == args["positions"].shape
    assert args["valid_tree_counts"].shape == args["query_lengths"].shape
    spec.wrapped(*spec.args)


def test_qwen35_mtp_tree_tiny_onnx_keeps_tree_metadata_out_of_attention(
        tmp_path):
    config = _config()
    config.mtp_base = True
    config.mtp_tree_base = True
    output = tmp_path / "qwen35-mtp-tree-base.onnx"

    _export_model(Qwen3_5CausalLM(config), str(output), optimize=False)

    graph = onnx.load(str(output), load_external_data=False).graph
    attention = next(node for node in graph.node
                     if node.op_type == "AttentionPlugin")
    attributes = {attribute.name for attribute in attention.attribute}
    assert "enable_tree_metadata" not in attributes
    assert not {"tree_parent_ids", "tree_depths", "valid_tree_counts"} & set(
        attention.input)
    assert all(attention.input)


def test_qwen35_exported_graph_connects_ragged_metadata_to_plugins():
    spec = Qwen3_5CausalLM(_config()).onnx_export_spec()

    exported = torch.export.export(spec.wrapped,
                                   spec.args,
                                   dynamic_shapes=spec.dynamic_shapes,
                                   strict=False)
    graph = str(exported.graph)

    assert "trt.attention_plugin" in graph
    assert "trt_edgellm.causal_conv1d" in graph
    assert "trt_edgellm.gated_delta_net" in graph
    for name in ("positions", "query_start_offsets", "query_lengths",
                 "past_lengths", "state_indices"):
        assert name in graph


def test_qwen35_tiny_onnx_preserves_plugin_metadata_inputs(tmp_path):
    output = tmp_path / "qwen35-ragged.onnx"
    _export_model(Qwen3_5CausalLM(_config()), str(output), optimize=False)
    graph = onnx.load(str(output), load_external_data=False).graph
    nodes = {
        node.op_type: node
        for node in graph.node if node.op_type in
        {"AttentionPlugin", "causal_conv1d", "gated_delta_net"}
    }

    assert set(nodes) == {
        "AttentionPlugin", "causal_conv1d", "gated_delta_net"
    }
    assert set(nodes["AttentionPlugin"].input) >= {
        "query_start_offsets", "query_lengths", "past_lengths",
        "attention_sequence_lengths", "execution_phase_marker",
        "context_sequence_count_carrier"
    }
    for op_type in ("causal_conv1d", "gated_delta_net"):
        assert set(nodes[op_type].input) >= {
            "query_start_offsets", "query_lengths", "state_indices",
            "execution_phase_marker", "context_sequence_count_carrier"
        }


def test_default_attention_export_uses_common_token_major_contract():
    spec = CausalLM(_attention_config()).onnx_export_spec()
    args = dict(zip(spec.input_names, spec.args))
    shapes = dict(zip(spec.input_names, spec.dynamic_shapes))

    assert args["inputs_embeds"].ndim == 2
    assert args["rope_rotary_cos_sin"].ndim == 2
    assert args["deepstack_embeds_0"].ndim == 2
    assert "state_indices" in args
    assert "positions" in args
    assert "token_to_sequence" not in args
    assert "query_start_offsets" in args
    assert "query_lengths" in args
    assert "past_lengths" in args
    assert "sequence_lengths" not in args
    assert "attention_sequence_lengths" in args
    assert "execution_phase_marker" in args
    assert "context_sequence_count_carrier" in args
    assert shapes["context_sequence_count_carrier"][0] != shapes[
        "query_lengths"][0]
    assert "logits_indices" in args

    outputs = spec.wrapped(*spec.args)
    assert outputs[0].shape == (2, 32)


def test_default_ragged_decoder_preserves_matrix_rank_through_mlp():
    model = CausalLM(_attention_config())
    projection_input_shapes = []
    hooks = [
        projection.register_forward_pre_hook(
            lambda _module, args: projection_input_shapes.append(
                tuple(args[0].shape))) for projection in (
                    model.model.layers[0].mlp.gate_proj,
                    model.model.layers[0].mlp.up_proj,
                    model.model.layers[0].mlp.down_proj,
                )
    ]
    try:
        spec = model.onnx_export_spec()
        outputs = spec.wrapped(*spec.args)
    finally:
        for hook in hooks:
            hook.remove()

    physical_tokens = spec.args[0].shape[0]
    assert projection_input_shapes
    assert all(shape[:2] == (physical_tokens, 1)
               for shape in projection_input_shapes), projection_input_shapes
    assert outputs[0].shape == (2, 32)


def test_tp_nvfp4_mlp_uses_token_matrix_and_restores_decoder_rank():
    config = _attention_config()
    config.num_key_value_heads = 2
    config.quant = QuantConfig(quant_type=QUANT_NVFP4, group_size=16)
    model = CausalLM(config.for_rank(rank=0, world=2))
    projection_input_shapes = []
    projections = (
        model.model.layers[0].mlp.gate_proj,
        model.model.layers[0].mlp.up_proj,
        model.model.layers[0].mlp.down_proj,
    )
    hooks = [
        projection.register_forward_pre_hook(
            lambda _module, args: projection_input_shapes.append(
                tuple(args[0].shape))) for projection in projections
    ]
    hidden_states = torch.zeros(4, 1, config.hidden_size, dtype=torch.float16)
    try:
        output = model.model.layers[0].mlp(hidden_states)
    finally:
        for hook in hooks:
            hook.remove()

    assert projection_input_shapes == [(4, config.hidden_size),
                                       (4, config.hidden_size), (4, 16)]
    assert output.shape == hidden_states.shape


def test_default_attention_tiny_onnx_uses_ragged_attention(tmp_path):
    output = tmp_path / "attention-ragged.onnx"
    model = CausalLM(_attention_config())
    assert fuse_qkv_projections(model) == 1
    _export_model(model, str(output), optimize=True)
    graph = onnx.shape_inference.infer_shapes(
        onnx.load(str(output), load_external_data=False)).graph
    attention = next(node for node in graph.node
                     if node.op_type == "AttentionPlugin")

    assert set(attention.input) >= {
        "query_start_offsets", "query_lengths", "past_lengths",
        "attention_sequence_lengths", "execution_phase_marker",
        "context_sequence_count_carrier"
    }
    assert all(node.op_type not in {"causal_conv1d", "gated_delta_net"}
               for node in graph.node)
    values = {
        value.name: value
        for value in (*graph.input, *graph.value_info, *graph.output)
    }
    decoder_linears = [node for node in graph.node if node.op_type == "MatMul"]
    assert decoder_linears
    assert all(
        len(values[node.input[0]].type.tensor_type.shape.dim) == 3
        for node in decoder_linears)
    assert all(
        values[node.input[0]].type.tensor_type.shape.dim[1].dim_value == 1
        for node in decoder_linears)
    assert len(values[attention.input[0]].type.tensor_type.shape.dim) == 2
    lm_head = next(node for node in graph.node if node.op_type == "Gemm")
    assert len(values[lm_head.input[0]].type.tensor_type.shape.dim) == 2


def test_default_attention_logits_selection_uses_axis_gather(tmp_path):
    output = tmp_path / "attention-token-row-gather.onnx"
    _export_model(CausalLM(_attention_config()), str(output), optimize=False)
    graph = onnx.load(str(output), load_external_data=False).graph
    op_types = [node.op_type for node in graph.node]

    assert "Gather" in op_types
    assert "GatherND" not in op_types


def test_tree_base_logits_selection_has_independent_dynamic_axis(tmp_path):
    config = _attention_config()
    config.eagle_base = True
    config.num_hidden_layers = 8
    config.layer_types = [LAYER_ATTN] * config.num_hidden_layers
    output = tmp_path / "tree-base-ragged.onnx"

    _export_model(CausalLM(config), str(output), optimize=False)

    graph = onnx.load(str(output), load_external_data=False).graph
    inputs = {tensor.name: tensor for tensor in graph.input}
    token_axis = inputs["inputs_embeds"].type.tensor_type.shape.dim[0]
    logits_axis = inputs["logits_indices"].type.tensor_type.shape.dim[0]
    assert token_axis.dim_param == "physical_tokens"
    assert logits_axis.dim_param == "logits_rows"


def test_omni_next_talker_emits_hidden_states_on_token_major_contract():
    spec = Qwen3OmniNextTalkerCausalLM(_config()).onnx_export_spec()
    args = dict(zip(spec.input_names, spec.args))
    outputs = spec.wrapped(*spec.args)

    assert "context_lengths" not in args
    assert spec.output_names[:2] == ["logits", "hidden_states"]
    assert outputs[1].shape[0] == args["inputs_embeds"].shape[0]


def test_omni_next_moe_mtp_base_runs_token_major_wrapper(tmp_path):
    model = Qwen3OmniNextMoeLanguageModel(_omni_next_moe_config())
    for layer in model.model.layers:
        layer.mlp._prepare_moe_weights()
    spec = model.onnx_export_spec()
    args = dict(zip(spec.input_names, spec.args))
    outputs = spec.wrapped(*spec.args)

    assert "context_lengths" not in args
    assert spec.output_names[:3] == [
        "logits", "hidden_states", "accept_hidden_states"
    ]
    assert outputs[1].shape[0] == args["inputs_embeds"].shape[0]
    assert outputs[2].shape[0] == args["inputs_embeds"].shape[0]

    output = tmp_path / "omni-next-moe-mtp-base.onnx"
    _export_model(model, str(output), optimize=False)
    graph = onnx.load(str(output), load_external_data=False).graph
    inputs = {tensor.name: tensor for tensor in graph.input}
    assert len(inputs["inputs_embeds"].type.tensor_type.shape.dim) == 2
    moe_nodes = [
        node for node in graph.node if node.op_type == "Fp16MoePlugin"
    ]
    assert len(moe_nodes) == 2
    value_info = {
        value.name: value
        for value in list(graph.input) + list(graph.value_info) +
        list(graph.output)
    }
    assert all(
        len(value_info[node.input[1]].type.tensor_type.shape.dim) == 2
        for node in moe_nodes)
    assert all(
        len(value_info[node.output[0]].type.tensor_type.shape.dim) == 2
        for node in moe_nodes)


def test_omni_next_code_predictor_dynamic_head_uses_token_major_indices():
    config = _config()
    config.num_code_groups = 3
    spec = Qwen3OmniNextCodePredictorCausalLM(config).onnx_export_spec()
    args = dict(zip(spec.input_names, spec.args))
    outputs = spec.wrapped(*spec.args)

    assert "context_lengths" not in args
    assert "logits_indices" in args
    assert spec.output_names[:2] == ["logits", "hidden_states"]
    assert outputs[0].shape == (2, 32)
    assert outputs[1].shape[0] == args["inputs_embeds"].shape[0]


def test_qwen3_tts_talker_uses_common_token_major_contract():
    spec = TalkerCausalLM(_attention_config()).onnx_export_spec()
    args = dict(zip(spec.input_names, spec.args))
    outputs = spec.wrapped(*spec.args)

    assert "context_lengths" not in args
    assert spec.output_names[:2] == ["logits", "hidden_states"]
    assert outputs[1].shape[0] == args["inputs_embeds"].shape[0]


def test_qwen3_tts_code_predictor_uses_common_token_major_contract():
    config = _attention_config()
    config.num_code_groups = 3
    spec = CodePredictorCausalLM(config).onnx_export_spec()
    args = dict(zip(spec.input_names, spec.args))
    outputs = spec.wrapped(*spec.args)

    assert "context_lengths" not in args
    assert "logits_indices" in args
    assert outputs[0].shape == (2, 32)
    assert outputs[1].shape[0] == args["inputs_embeds"].shape[0]
