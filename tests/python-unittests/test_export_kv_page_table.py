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
"""ONNX-graph assertions for the ``kv_page_table`` engine binding.

Exports a tiny default-arch (attention-only) model and checks that the
exported graph carries the paged-KV plugin contract: a required
``kv_page_table`` graph input of shape ``[batch, 2, max_pages_per_seq]``,
AttentionPlugin nodes with the six fixed inputs plus four ragged metadata
inputs, and that the ``past_key_values_i`` KV-cache
binding is declared as the AttentionPlugin's paged-pool contract
``[2, num_pages, KV_PAGE_SIZE, num_kv_heads, head_dim]``.
"""

import ast
import os
import pathlib
from types import SimpleNamespace

import onnx
import pytest
import torch

from tensorrt_edgellm.checkpoint.checkpoint_utils import \
    build_runtime_llm_config_dict
from tensorrt_edgellm.config import (LAYER_GDN, QUANT_NVFP4, GdnConfig,
                                     ModelConfig, QuantConfig)
from tensorrt_edgellm.models import ops
from tensorrt_edgellm.models.default.modeling_default import (
    CausalLM, fuse_qkv_projections)
from tensorrt_edgellm.models.ops import (KV_PAGE_SIZE,
                                         dflash_target_kv_cache_update,
                                         use_generic_nvfp4_gemm_allreduce)
from tensorrt_edgellm.onnx.export import (_export_model,
                                          setup_fp8_qkv_scales_for_export)
from tensorrt_edgellm.onnx.onnx_custom_schemas import \
    register_tensorrt_edgellm_onnx_custom_schemas

_REQUIRED_ATTENTION_INPUTS = (
    "query_key_value",
    "past_key_value",
    "query_lengths",
    "rope_rotary_cos_sin",
    "past_lengths",
    "kv_page_table",
    "query_start_offsets",
    "attention_sequence_lengths",
    "execution_phase_marker",
    "context_sequence_count_carrier",
)

# ``attention_plugin``'s positional signature ends
# (..., kvcache_start_index, kv_page_table, ...) — with the packed-QKV
# contract that is positional slot 6; a call site must supply at least this
# many positional args, or pass ``kv_page_table`` as a keyword.
_MIN_ATTENTION_PLUGIN_POSITIONAL_ARGS = 6

_MODELS_DIR = (pathlib.Path(__file__).resolve().parents[2] /
               "tensorrt_edgellm" / "models")


def _tiny_default_config() -> ModelConfig:
    return ModelConfig(
        model_type="llama",
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        head_dim=4,
        rms_norm_eps=1e-6,
        vocab_size=32,
        rope_theta=10000.0,
        max_position_embeddings=128,
        default_attention_scale=4**-0.5,  # 1/sqrt(head_dim)
    )


def _export_tiny_model(tmp_path) -> str:
    config = _tiny_default_config()
    model = CausalLM(config)
    model.eval()
    output_path = os.path.join(str(tmp_path), "model.onnx")
    _export_model(model, output_path)
    return output_path


def _export_tiny_tp_nvfp4_model(tmp_path) -> str:
    config = _tiny_default_config()
    config.hidden_size = 64
    config.num_hidden_layers = 1
    config.intermediate_size = 128
    config.head_dim = 16
    config.default_attention_scale = 16**-0.5
    config.quant = QuantConfig(quant_type=QUANT_NVFP4, group_size=16)
    model = CausalLM(config.for_rank(rank=0, world=2))
    model.eval()
    output_path = os.path.join(str(tmp_path), "model.onnx")
    _export_model(model, output_path)
    return output_path


@pytest.mark.parametrize(
    "target,expected_op,unexpected_op",
    [
        ("sm110", "FusedNvfp4GemmAllReducePlugin", "AllReducePlugin"),
        ("sm103", "FusedNvfp4GemmAllReducePlugin", "AllReducePlugin"),
        ("sm121", "AllReducePlugin", "FusedNvfp4GemmAllReducePlugin"),
    ],
)
def test_nvfp4_tp_gemm_allreduce_target_export(monkeypatch, tmp_path, target,
                                               expected_op, unexpected_op):
    monkeypatch.setenv("EDGELLM_NVFP4_GEMM_ALLREDUCE_TARGET", target)
    assert use_generic_nvfp4_gemm_allreduce() is (target == "sm121")

    output_path = _export_tiny_tp_nvfp4_model(tmp_path)
    model = onnx.load(output_path, load_external_data=False)
    op_types = [node.op_type for node in model.graph.node]

    # One attention output projection and one MLP down projection per layer.
    assert op_types.count(expected_op) == 2
    assert unexpected_op not in op_types
    if target == "sm121":
        assert "MatMul" in op_types


def test_nvfp4_fp8_kv_qkv_fusion_preserves_export_scales(tmp_path):
    config = _tiny_default_config()
    config.quant = QuantConfig(quant_type=QUANT_NVFP4,
                               group_size=16,
                               kv_cache_quant="fp8")
    model = CausalLM(config)

    expected_scales = []
    expected_weights = []
    for layer_index, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        scales = [0.5 + layer_index, 0.25 + layer_index, 0.125 + layer_index]
        for proj, scale_name, scale in zip(
            (attn.q_proj, attn.k_proj, attn.v_proj),
            ("q_scale", "k_scale", "v_scale"), scales):
            getattr(proj, scale_name).fill_(scale)
        expected_scales.append(scales)
        expected_weights.append(
            torch.cat([
                attn.q_proj.weight,
                attn.k_proj.weight,
                attn.v_proj.weight,
            ],
                      dim=0).clone())

    assert fuse_qkv_projections(model) == config.num_hidden_layers
    setup_fp8_qkv_scales_for_export(model)

    for layer, scales, weight in zip(model.model.layers, expected_scales,
                                     expected_weights):
        attn = layer.self_attn
        assert not hasattr(attn, "q_proj")
        assert not hasattr(attn, "k_proj")
        assert not hasattr(attn, "v_proj")
        assert attn._qkv_scales_float == scales
        assert torch.equal(attn.qkv_proj_fused.weight, weight)
        assert not any(
            hasattr(attn.qkv_proj_fused, name)
            for name in ("q_scale", "k_scale", "v_scale"))

    output_path = os.path.join(str(tmp_path), "fused_nvfp4_fp8kv.onnx")
    _export_model(model, output_path)
    onnx_model = onnx.load(output_path, load_external_data=False)
    attention_nodes = [
        node for node in onnx_model.graph.node
        if node.op_type == "AttentionPlugin"
    ]
    assert len(attention_nodes) == config.num_hidden_layers
    for node, scales in zip(attention_nodes, expected_scales):
        attributes = {
            attr.name: onnx.helper.get_attribute_value(attr)
            for attr in node.attribute
        }
        assert attributes["enable_fp8_kv_cache"] == 1
        assert attributes["qkv_scales"] == scales


def test_tp_runtime_config_preserves_global_gdn_metadata():
    global_config = _tiny_default_config()
    global_config.model_type = "qwen3_5_text"
    global_config.layer_types = [LAYER_GDN, "full_attention"]
    global_config.gdn_cfg = GdnConfig(
        num_key_heads=4,
        num_value_heads=4,
        key_head_dim=4,
        value_head_dim=8,
        conv_kernel=4,
    )
    rank_config = global_config.for_rank(rank=1, world=2)
    model = SimpleNamespace(
        config=rank_config,
        RECURRENT_STATE_DTYPE=torch.float32,
        CONV_STATE_DTYPE=torch.float16,
    )
    global_llm_config = {
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
    }

    runtime_config = build_runtime_llm_config_dict(
        model, global_llm_config=global_llm_config)

    assert runtime_config["num_attention_heads"] == 4
    assert runtime_config["num_key_value_heads"] == 2
    assert runtime_config["intermediate_size"] == 32
    assert runtime_config["recurrent_state_num_heads"] == 4
    assert runtime_config["conv_dim"] == 64
    assert runtime_config["kv_layer_configs"] == [
        None,
        {
            "num_kv_heads": 2,
            "head_dim": 4,
        },
    ]
    assert runtime_config["rank_configs"] == [{
        "rank": 1,
        "config_overrides": {
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 16,
            "recurrent_state_num_heads": 2,
            "conv_dim": 32,
            "kv_layer_configs": [
                None,
                {
                    "num_kv_heads": 1,
                    "head_dim": 4,
                },
            ],
        },
    }]


def test_kv_page_table_graph_input_shape(tmp_path):
    output_path = _export_tiny_model(tmp_path)
    model = onnx.load(output_path, load_external_data=False)

    page_table_inputs = [
        graph_input for graph_input in model.graph.input
        if graph_input.name == "kv_page_table"
    ]
    assert len(page_table_inputs) == 1, (
        "Expected exactly one 'kv_page_table' graph input, found "
        f"{len(page_table_inputs)}")

    dims = page_table_inputs[0].type.tensor_type.shape.dim
    assert len(dims) == 3, f"Expected kv_page_table to be rank 3, got {dims}"
    # dim 0 (batch) and dim 2 (max_pages_per_seq) are dynamic; dim 1 is the
    # fixed K/V split and must be the constant 2.
    assert dims[0].dim_param, "kv_page_table dim 0 (batch) must be dynamic"
    assert dims[1].dim_value == 2, (
        "kv_page_table dim 1 must be the fixed K/V split (2), got "
        f"{dims[1]}")
    assert dims[2].dim_param, (
        "kv_page_table dim 2 (max_pages_per_seq) must be dynamic")


def test_attention_plugin_nodes_have_required_inputs(tmp_path):
    output_path = _export_tiny_model(tmp_path)
    model = onnx.load(output_path, load_external_data=False)

    attention_nodes = [
        node for node in model.graph.node if node.op_type == "AttentionPlugin"
    ]
    assert attention_nodes, "Expected at least one AttentionPlugin node"
    for node in attention_nodes:
        assert len(node.input) == len(_REQUIRED_ATTENTION_INPUTS), (
            f"AttentionPlugin node {node.name!r} has {len(node.input)} "
            f"inputs, expected {len(_REQUIRED_ATTENTION_INPUTS)}")
        assert node.input[5] != "", (
            "AttentionPlugin input index 5 (kv_page_table) must not be empty")
        assert tuple(node.input[2:]) == _REQUIRED_ATTENTION_INPUTS[2:]


def test_kv_cache_graph_input_is_pool_shaped(tmp_path):
    """Assert every past_key_values_i input uses the paged-pool contract."""
    output_path = _export_tiny_model(tmp_path)
    model = onnx.load(output_path, load_external_data=False)

    kv_cache_inputs = [
        graph_input for graph_input in model.graph.input
        if graph_input.name.startswith("past_key_values_")
    ]
    assert kv_cache_inputs, "Expected at least one past_key_values_i graph input"

    for graph_input in kv_cache_inputs:
        dims = graph_input.type.tensor_type.shape.dim
        assert len(dims) == 5, (
            f"{graph_input.name} must be rank 5, got {len(dims)}: {dims}")
        assert dims[0].dim_value == 2, (
            f"{graph_input.name} dim 0 must be the fixed K/V split (2), got "
            f"{dims[0]}")
        assert dims[1].dim_param, (
            f"{graph_input.name} dim 1 (num_pages) must be dynamic, got "
            f"{dims[1]}")
        assert dims[2].dim_value == KV_PAGE_SIZE, (
            f"{graph_input.name} dim 2 must be the fixed page size "
            f"({KV_PAGE_SIZE}), got {dims[2]}")
        assert dims[3].dim_value > 0, (
            f"{graph_input.name} dim 3 (num_kv_heads) must be a fixed "
            f"positive value, got {dims[3]}")
        assert dims[4].dim_value > 0, (
            f"{graph_input.name} dim 4 (head_dim) must be a fixed positive "
            f"value, got {dims[4]}")


def _iter_attention_plugin_calls():
    for path in sorted(_MODELS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "attention_plugin"):
                yield path, node


def test_attention_plugin_call_sites_pass_kv_page_table():
    """Static regression guard: every ``attention_plugin(`` call must pass ``kv_page_table``.

    A future modeling-file reuser of the shared Attention/Transformer stack
    could add its own ``attention_plugin(`` call site (or copy an old one)
    without the paged-KV argument; catch that at the source level instead of
    relying on model instantiation, since some reusers' checkpoints are
    large/unavailable in CI.
    """
    call_sites = list(_iter_attention_plugin_calls())
    assert call_sites, ("Expected to find attention_plugin( call sites under "
                        f"{_MODELS_DIR}")

    failures = []
    for path, node in call_sites:
        if any(kw.arg == "kv_page_table" for kw in node.keywords):
            continue
        # Positional form: the 6th positional argument slot IS kv_page_table
        # (packed contract). Merely counting args would let a stale call with
        # enough positional args slip through,
        # so require the expression in that slot to visibly be a page table.
        if len(node.args) < _MIN_ATTENTION_PLUGIN_POSITIONAL_ARGS:
            failures.append(
                f"{path}:{node.lineno}: attention_plugin( call has "
                f"{len(node.args)} positional args and no kv_page_table "
                "keyword -- missing the paged-KV ABI argument")
            continue
        slot = node.args[_MIN_ATTENTION_PLUGIN_POSITIONAL_ARGS - 1]
        slot_src = ast.unparse(slot)
        if "page_table" not in slot_src:
            failures.append(
                f"{path}:{node.lineno}: attention_plugin( positional arg 6 "
                f"is {slot_src!r}, expected the kv_page_table tensor -- the "
                "call site predates the paged-KV ABI or binds arguments in "
                "the wrong order")
    assert not failures, "\n".join(failures)


def test_dflash_target_kv_update_requires_page_table_argument():
    parameters = [
        argument.name for argument in
        dflash_target_kv_cache_update._opoverload._schema.arguments
    ]
    assert parameters == [
        "k_delta",
        "v_delta",
        "past_key_value",
        "token_aligned_rope_cos_sin",
        "delta_positions",
        "delta_token_to_sequence",
        "kv_page_table",
    ]


def test_dflash_target_kv_update_schema_keeps_name_and_has_seven_inputs():
    register_tensorrt_edgellm_onnx_custom_schemas()
    schemas = [
        schema for schema in onnx.defs.get_all_schemas_with_history()
        if schema.name == "DFlashTargetKVCacheUpdate"
        and schema.domain == "trt_edgellm"
    ]
    assert len(schemas) == 1
    schema = schemas[0]
    assert [parameter.name for parameter in schema.inputs] == [
        "k_delta",
        "v_delta",
        "past_key_value",
        "token_aligned_rope_cos_sin",
        "delta_positions",
        "delta_token_to_sequence",
        "kv_page_table",
    ]


def test_dflash_target_kv_update_call_sites_pass_page_table():
    failures = []
    for path in sorted(_MODELS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "dflash_target_kv_cache_update"):
                continue
            if len(node.args) < 7 or "page_table" not in ast.unparse(
                    node.args[6]):
                failures.append(
                    f"{path}:{node.lineno}: dflash_target_kv_cache_update "
                    "argument 7 must be kv_page_table")
    assert not failures, "\\n".join(failures)


def test_attention_plugin_direct_call_sites_pass_required_static_flags():
    """Static guard for required bool attrs in direct ``attention_plugin`` calls.

    Calls routed through a local ``**kwargs`` dict are checked by their owning
    model tests. Direct call sites must pass the required bool attributes
    explicitly so ``torch.export`` cannot drop default-valued arguments.
    """
    call_sites = list(_iter_attention_plugin_calls())
    failures = []
    required_flags = {
        "enable_context_mask_selector",
        "enable_vision_block_attention",
    }
    for path, node in call_sites:
        keyword_names = {kw.arg for kw in node.keywords}
        if None in keyword_names:
            continue
        missing = sorted(required_flags - keyword_names)
        if missing:
            failures.append(
                f"{path}:{node.lineno}: attention_plugin( direct call is "
                f"missing required static flag(s): {', '.join(missing)}")
    assert not failures, "\n".join(failures)


def test_attention_plugin_rejects_dense_rank_three_input():
    qkv = torch.zeros(1, 2, 8, dtype=torch.float16)
    kv_cache = torch.zeros(2, 1, 128, 1, 2, dtype=torch.float16)
    sequence_lengths = torch.ones(1, dtype=torch.int32)
    rope = torch.zeros(2, 2, dtype=torch.float32)
    past_lengths = torch.zeros(1, dtype=torch.int32)
    page_table = torch.zeros(1, 2, 1, dtype=torch.int32)

    with pytest.raises(ValueError, match="token-major rank-2"):
        ops.attention_plugin(qkv, kv_cache, sequence_lengths, rope,
                             past_lengths, page_table, 1, 1, 2, 0, False,
                             False, 1.0, False, False, 0.0)
