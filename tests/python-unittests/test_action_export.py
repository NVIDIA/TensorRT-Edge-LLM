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
"""ONNX graph assertions for Alpamayo action export."""

import onnx
import pytest
import torch

from tensorrt_edgellm.config import ActionConfig
from tensorrt_edgellm.onnx.export_encoder import export_action_onnx


def test_action_export_compacts_attention_optional_inputs(tmp_path):
    # Given an Alpamayo action graph that enables only context-mask attention,
    # When the graph is exported for TensorRT, Then disabled optional
    # AttentionPlugin slots are removed while ragged metadata remains wired.
    output_path = tmp_path / "model.onnx"
    cfg = ActionConfig(
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        hidden_size=8,
        intermediate_size=16,
        n_diffusion_tokens=2,
        in_proj_hidden_size=8,
        in_proj_num_enc_layers=1,
        in_proj_num_fourier_feats=4,
    )

    export_action_onnx(
        output_path=str(output_path),
        weights={},
        config=cfg,
        max_kv_cache_capacity=128,
        dtype=torch.float16,
    )

    graph = onnx.load(str(output_path), load_external_data=False).graph
    attention = next(node for node in graph.node
                     if node.op_type == "AttentionPlugin")

    assert all(attention.input)
    assert list(attention.input[-6:]) == [
        "query_start_offsets",
        "attention_sequence_lengths",
        "execution_phase_marker",
        "context_sequence_count_carrier",
        "past_key_values_0",
        "kv_page_table",
    ]
    assert "context_mask_selector" in attention.input
    assert "attention_pos_id" not in attention.input
    assert "attention_mask" not in attention.input


def test_action_config_uses_llm_fp8_scales():
    """Action cache codec must use every LLM layer's calibration, not expert weights."""
    import pytest

    from tensorrt_edgellm.onnx.cli import _build_action_config

    root = {"expert_cfg": {"head_dim": 128, "num_attention_heads": 1}}
    weights = {}
    for layer, (k_scale, v_scale) in enumerate(
        ((0.03125, 0.015625), (0.0625, 0.125))):
        weights[
            f"expert.layers.{layer}.self_attn.k_proj.weight"] = torch.zeros(
                128, 128)
        prefix = f"vlm.model.language_model.layers.{layer}.self_attn"
        weights[f"{prefix}.k_proj.k_scale"] = torch.tensor(k_scale)
        weights[f"{prefix}.v_proj.v_scale"] = torch.tensor(v_scale)
    weights["expert.layers.0.self_attn.k_proj.k_scale"] = torch.tensor(99.0)
    cfg = _build_action_config(root, weights)
    assert cfg.kv_cache_scales == [[0.03125, 0.015625], [0.0625, 0.125]]
    del weights["vlm.model.language_model.layers.1.self_attn.v_proj.v_scale"]
    with pytest.raises(ValueError, match="every action layer"):
        _build_action_config(root, weights)


@pytest.mark.parametrize("target_sm", [None, 80, 86, 87, 120])
def test_action_export_fp8_pool_codec(tmp_path, monkeypatch, target_sm):
    """FP8 pools and per-layer scales must survive export while action Q/O stay FP16."""
    import json

    from tensorrt_edgellm.onnx.export_encoder import write_action_config

    monkeypatch.setattr(
        "tensorrt_edgellm.models.ops.nvfp4_a16_export_target_sm",
        lambda: target_sm)
    prefix_type = (onnx.TensorProto.INT8 if target_sm in (80, 86, 87) else
                   onnx.TensorProto.FLOAT8E4M3FN)
    scales = [[0.03125, 0.015625], [0.0625, 0.125]]
    cfg = ActionConfig(
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        hidden_size=128,
        intermediate_size=128,
        n_diffusion_tokens=64,
        in_proj_hidden_size=8,
        in_proj_num_enc_layers=1,
        in_proj_num_fourier_feats=4,
        kv_cache_scales=scales,
    )
    output = tmp_path / "model.onnx"
    export_action_onnx(str(output), {}, cfg, 256, torch.float16)
    graph = onnx.load(str(output), load_external_data=False).graph
    nodes = [node for node in graph.node if node.op_type == "AttentionPlugin"]
    assert len(nodes) == 2
    inputs = {
        value.name: value.type.tensor_type.elem_type
        for value in graph.input
    }
    outputs = {
        value.name: value.type.tensor_type.elem_type
        for value in graph.output
    }
    for node, pair in zip(nodes, scales):
        attrs = {
            attr.name: onnx.helper.get_attribute_value(attr)
            for attr in node.attribute
        }
        assert attrs["enable_fp8_kv_cache"] == 0
        assert attrs["external_kv_prefix"] == 1
        assert attrs["fp8_kv_prefix"] == 1
        assert list(attrs["qkv_scales"]) == [1.0, *pair]
        assert inputs[node.input[1]] == onnx.TensorProto.FLOAT16
        assert inputs[node.input[-2]] == prefix_type
        assert outputs[node.output[1]] == onnx.TensorProto.FLOAT16
    write_action_config(cfg, 256, str(tmp_path))
    assert json.loads(
        (tmp_path / "config.json").read_text())["kv_cache_scales"] == scales


@pytest.mark.parametrize("tp_size", [1, 2])
def test_action_export_selects_kv_contract_for_llm_tp(tmp_path, tp_size):
    """TP=1 borrows FP8 pages; TP>1 keeps dense FP16 action inputs and native attention."""
    import json

    from tensorrt_edgellm.onnx.cli import _export_action

    root = {
        "expert_cfg": {
            "head_dim": 128,
            "num_attention_heads": 2,
            "hidden_size": 256,
            "intermediate_size": 256,
        },
        "action_space_cfg": {
            "n_waypoints": 64
        },
        "action_in_proj_cfg": {
            "hidden_size": 8,
            "num_enc_layers": 1,
            "num_fourier_feats": 4,
        },
    }
    weights = {
        "expert.layers.0.self_attn.k_proj.weight":
        torch.zeros(128, 256),
        "vlm.model.language_model.layers.0.self_attn.k_proj.k_scale":
        torch.tensor(0.5),
        "vlm.model.language_model.layers.0.self_attn.v_proj.v_scale":
        torch.tensor(0.25),
    }
    _export_action("",
                   str(tmp_path),
                   weights,
                   root,
                   256,
                   torch.float16,
                   tp_size=tp_size)
    graph = onnx.load(str(tmp_path / "model.onnx"),
                      load_external_data=False).graph
    kv_inputs = {
        value.name: (value.type.tensor_type.elem_type,
                     len(value.type.tensor_type.shape.dim))
        for value in graph.input
        if value.name.startswith(("past_key_values_", "action_key_values_",
                                  "k_cache_", "v_cache_"))
    }
    config = json.loads((tmp_path / "config.json").read_text())
    actual = (kv_inputs, [(node.op_type, node.domain) for node in graph.node
                          if node.op_type in ("Attention", "AttentionPlugin")],
              config.get("kv_cache_scales"))
    if tp_size == 1:
        expected = ({
            "past_key_values_0": (onnx.TensorProto.FLOAT8E4M3FN, 5),
            "action_key_values_0": (onnx.TensorProto.FLOAT16, 5),
        }, [("AttentionPlugin", "trt_edgellm")], [[0.5, 0.25]])
    else:
        expected = ({
            "k_cache_0": (onnx.TensorProto.FLOAT16, 4),
            "v_cache_0": (onnx.TensorProto.FLOAT16, 4),
        }, [("Attention", "trt")], None)
    assert actual == expected
