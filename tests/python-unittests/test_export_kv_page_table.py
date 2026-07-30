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
AttentionPlugin nodes with 8 inputs (no stray tree-attention optionals for a
non-tree-attention model), and — a previously-shipped regression — that the
``past_key_values_i`` KV-cache binding is declared as the AttentionPlugin's
paged-pool contract ``[2, num_pages, KV_PAGE_SIZE, num_kv_heads, head_dim]``,
not the legacy per-slot ``[batch, 2, num_kv_heads, capacity, head_dim]``
shape (a legacy-shaped binding makes the paged CuTe DSL prefill kernel
misread ``numPages`` from ``dims.d[1]`` and corrupts prefill).
"""

import ast
import os
import pathlib

import onnx

from tensorrt_edgellm.config import ModelConfig
from tensorrt_edgellm.models.default.modeling_default import CausalLM
from tensorrt_edgellm.models.ops import KV_PAGE_SIZE
from tensorrt_edgellm.onnx.export import _export_model

_NUM_REQUIRED_ATTENTION_INPUTS = 8

# ``attention_plugin``'s positional signature ends
# (..., kvcache_start_index, kv_page_table, ...); a call site must supply at
# least this many positional args, or pass ``kv_page_table`` as a keyword.
_MIN_ATTENTION_PLUGIN_POSITIONAL_ARGS = 8

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
        assert len(node.input) == _NUM_REQUIRED_ATTENTION_INPUTS, (
            f"AttentionPlugin node {node.name!r} has {len(node.input)} "
            f"inputs, expected {_NUM_REQUIRED_ATTENTION_INPUTS}")
        assert node.input[7] != "", (
            "AttentionPlugin input index 7 (kv_page_table) must not be empty")


def test_kv_cache_graph_input_is_pool_shaped(tmp_path):
    """Regression test: past_key_values_i must be the paged-pool contract.

    A binding declared with the legacy per-slot shape (rank 5 but
    [batch, 2, num_kv_heads, capacity, head_dim]) passes a naive rank check
    but makes the CuTe DSL prefill kernel misread numPages from dim 1.
    Assert the actual per-dim contract: dim 0 == 2
    (fixed K/V split), dim 1 dynamic (num_pages), dim 2 == KV_PAGE_SIZE
    (fixed), dims 3/4 fixed (num_kv_heads, head_dim).
    """
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
            f"{dims[1]} — a static/legacy-ordered dim here means the "
            "binding is not pool-shaped")
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
        has_keyword = any(kw.arg == "kv_page_table" for kw in node.keywords)
        if not has_keyword and len(
                node.args) < _MIN_ATTENTION_PLUGIN_POSITIONAL_ARGS:
            failures.append(
                f"{path}:{node.lineno}: attention_plugin( call has "
                f"{len(node.args)} positional args and no kv_page_table "
                "keyword -- missing the paged-KV ABI argument")
    assert not failures, "\n".join(failures)
