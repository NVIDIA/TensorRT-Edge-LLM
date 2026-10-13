# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tensorrt_edgellm.checkpoint.repacking import \
    repack_nvfp4_a16_marlin_linear
from tensorrt_edgellm.config import QUANT_NVFP4_A16, ModelConfig, QuantConfig
from tensorrt_edgellm.models import ops
from tensorrt_edgellm.models.dflash.modeling_dflash_draft import \
    DFlashDraftModel
from tensorrt_edgellm.models.dspark.modeling_dspark_draft import \
    DSparkDraftModel
from tensorrt_edgellm.onnx.export import _export_model


def _config(kind: str) -> ModelConfig:
    kwargs = ({
        "dflash_target_layer_ids": [0],
        "dflash_block_size": 4,
    } if kind == "dflash" else {
        "dspark_target_layer_ids": [0],
        "dspark_block_size": 4,
    })
    return ModelConfig(model_type="qwen3",
                       hidden_size=16,
                       num_hidden_layers=1,
                       num_attention_heads=2,
                       num_key_value_heads=1,
                       intermediate_size=32,
                       head_dim=8,
                       rms_norm_eps=1e-6,
                       vocab_size=32,
                       rope_theta=10_000.0,
                       max_position_embeddings=4096,
                       default_attention_scale=1.0 / math.sqrt(8.0),
                       **kwargs)


@pytest.mark.parametrize(("kind", "model_cls", "expected_outputs"), [
    ("dflash", DFlashDraftModel, 2),
    ("dspark", DSparkDraftModel, 3),
])
def test_block_draft_export_uses_token_major_proposal_and_delta_portals(
        kind, model_cls, expected_outputs):
    config = _config(kind)
    spec = model_cls(config).onnx_export_spec()
    args = dict(zip(spec.input_names, spec.args))

    assert args["inputs_embeds"].ndim == 2
    assert args["dflash_target_hidden_concat"].ndim == 2
    assert args["rope_rotary_cos_sin"].ndim == 2
    assert args["dflash_delta_rope_cos_sin"].ndim == 2
    assert args["dflash_delta_positions"].ndim == 1
    assert args["dflash_delta_token_to_sequence"].ndim == 1
    assert args["attention_position_ids"].ndim == 1
    assert args["packed_attention_mask"].ndim == 2
    assert args["execution_phase_marker"].shape == (4, )
    assert args["context_sequence_count_carrier"].shape == (0, )
    assert args["context_sequence_count_carrier"].dtype == torch.int32

    outputs = spec.wrapped(*spec.args)
    assert len(outputs) == expected_outputs
    assert outputs[0].shape == (8, config.vocab_size)
    assert outputs[0].dtype == torch.float32


@pytest.mark.parametrize(("kind", "model_cls"), [
    ("dflash", DFlashDraftModel),
    ("dspark", DSparkDraftModel),
])
def test_block_draft_attention_omits_absent_ddtree_metadata(
        tmp_path, kind, model_cls):
    output = tmp_path / f"{kind}-draft.onnx"
    _export_model(model_cls(_config(kind)), str(output), optimize=False)

    graph = onnx.load(str(output), load_external_data=False).graph
    attention = next(node for node in graph.node
                     if node.op_type == "AttentionPlugin")
    attributes = {
        attribute.name: attribute.i
        for attribute in attention.attribute
    }

    assert attributes["enable_tree_attention"] == 1
    assert "enable_tree_metadata" not in attributes
    assert list(attention.input)[-4:] == [
        "query_start_offsets",
        "attention_sequence_lengths",
        "execution_phase_marker",
        "context_sequence_count_carrier",
    ]
    assert len(attention.input) == 12
    assert all(attention.input)


def test_dspark_ragged_attention_preserves_sink_and_contiguous_swa(tmp_path):
    config = _config("dspark")
    config.attention_sink_bias = True
    config.sliding_window_size = 128
    config.raw_layer_types = ["sliding_attention"]
    output = tmp_path / "dspark-sink-swa.onnx"

    _export_model(DSparkDraftModel(config), str(output), optimize=False)

    graph = onnx.load(str(output), load_external_data=False).graph
    attention = next(node for node in graph.node
                     if node.op_type == "AttentionPlugin")
    attributes = {
        attribute.name: attribute.i
        for attribute in attention.attribute
    }
    assert attributes["enable_attention_sink"] == 1
    assert attributes["enable_contiguous_query_swa"] == 1
    assert list(attention.input)[-5].startswith("attention_sinks_fp32")


def test_dspark_ragged_native_fc_exports_repacked_weights(
        tmp_path, monkeypatch):
    monkeypatch.setattr(ops, "_NVFP4_A16_EXPORT_TARGET_SM", 120)
    config = _config("dspark")
    config.hidden_size = 128
    config.head_dim = 64
    config.intermediate_size = 256
    config.vocab_size = 256
    config.default_attention_scale = 1.0 / math.sqrt(64.0)
    config.dspark_target_layer_ids = [0, 1]
    config.dspark_fc_native_precision = True
    config.quant = QuantConfig(group_size=16,
                               layer_overrides={"fc": QUANT_NVFP4_A16})
    model = DSparkDraftModel(config)
    packed = torch.full((128, 128), 0x22, dtype=torch.uint8)
    scales = torch.full((128, 16), 0x28, dtype=torch.int8)
    qweight, block_scales, global_scale, _, n_padded = repack_nvfp4_a16_marlin_linear(
        packed, scales, torch.tensor([0.125], dtype=torch.float32))
    model.fc.register_buffer("qweight", qweight)
    model.fc.register_buffer("block_scales", block_scales)
    model.fc.register_buffer("global_scale", global_scale)
    model.fc.n_padded = n_padded
    output = tmp_path / "dspark-packed-fc.onnx"
    _export_model(model, str(output), optimize=False)
    graph = onnx.load(str(output), load_external_data=False).graph
    gemms = [
        node for node in graph.node if node.op_type == "Nvfp4A16GemmPlugin"
    ]
    assert len(gemms) == 1
    values = {
        value.name: value
        for value in [*graph.input, *graph.value_info, *graph.output]
    }
    assert len(values[gemms[0].input[0]].type.tensor_type.shape.dim) == 2
    assert len(values[gemms[0].output[0]].type.tensor_type.shape.dim) == 2


def test_dspark_ragged_default_fc_preserves_fp32_accumulation():
    model = DSparkDraftModel(_config("dspark"))
    with torch.no_grad():
        model.fc.weight.fill_(100.0)
    spec = model.onnx_export_spec()
    args = list(spec.args)
    index = spec.input_names.index("dflash_target_hidden_concat")
    args[index] = torch.full_like(args[index], 1000.0)
    observed = []
    hook = model.hidden_norm.register_forward_pre_hook(
        lambda _module, values: observed.append(values[0].detach().clone()))
    try:
        spec.wrapped(*args)
    finally:
        hook.remove()
    expected = torch.nn.functional.linear(args[index].float(),
                                          model.fc.weight.float())
    assert len(observed) == 1
    assert observed[0].dtype == torch.float32
    assert torch.isfinite(observed[0]).all()
    assert expected.min() > torch.finfo(torch.float16).max
    torch.testing.assert_close(observed[0], expected)
