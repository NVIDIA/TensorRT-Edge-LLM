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
"""Gemma4 pool inputs preserve the existing PLE numerical path."""

import json
import math
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from tensorrt_edgellm.config import Mapping, ModelConfig
from tensorrt_edgellm.models import ops
from tensorrt_edgellm.models.gemma4.modeling_gemma4_text import \
    Gemma4ForCausalLM
from tensorrt_edgellm.onnx import dynamo_translations
from tensorrt_edgellm.onnx.export import _permissive_inline_opset


def _model():
    config = ModelConfig(model_type="gemma4_text",
                         hidden_size=32,
                         num_hidden_layers=2,
                         num_attention_heads=4,
                         num_key_value_heads=2,
                         head_dim=8,
                         intermediate_size=64,
                         rms_norm_eps=1e-6,
                         vocab_size=128,
                         rope_theta=10000.0,
                         max_position_embeddings=128,
                         default_attention_scale=8**-0.5,
                         hidden_size_per_layer_input=8,
                         vocab_size_per_layer_input=128,
                         hidden_activation="gelu_pytorch_tanh",
                         attention_layer_types=["full_attention"] * 2)
    torch.manual_seed(7)
    model = Gemma4ForCausalLM(config).half().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.1)
    return model


@pytest.mark.parametrize("capacity", [8, 64, 129])
@pytest.mark.parametrize("query_lengths", [[3, 1], [1, 1, 1, 1]])
def test_pool_and_expanded_gemma_hidden_states_match(capacity, query_lengths):
    model = _model()
    pooled = model.onnx_export_spec()
    tokens = pooled.args[0].shape[0]
    pool = torch.randn(capacity, 16, dtype=torch.float16)
    pool[0].zero_()
    row_map = (torch.arange(tokens) % 3).reshape(tokens, 1).int()
    row_map[row_map == 2] = capacity - 1
    values = pool[row_map.long()].reshape(tokens, 2, 8)
    inputs = torch.randn_like(pooled.args[0])
    pooled_args = (inputs, pool, row_map, *pooled.args[3:])
    lengths = torch.tensor(query_lengths, dtype=torch.int32)
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).int()))
    metadata = {
        "positions":
        torch.cat([torch.arange(n, dtype=torch.int32) for n in query_lengths]),
        "query_lengths":
        lengths,
        "query_start_offsets":
        offsets,
        "past_lengths":
        torch.zeros_like(lengths),
        "attention_sequence_lengths":
        lengths,
        "state_indices":
        torch.arange(len(query_lengths) - 1, -1, -1, dtype=torch.int32),
        "context_sequence_count_carrier":
        torch.zeros_like(lengths),
        "kv_page_table":
        torch.zeros(len(query_lengths), 2, 2, dtype=torch.int32),
        "logits_indices":
        offsets[1:].long() - 1,
    }
    pooled_args = tuple(
        metadata.get(name, arg)
        for name, arg in zip(pooled.input_names, pooled_args))
    reference = dict(zip(pooled.input_names, pooled_args))
    reference.pop("ple_cache_pool")
    reference.pop("ple_row_map")
    reference["past_key_values"] = tuple(
        reference.pop(f"past_key_values_{i}") for i in range(2))
    reference["ple_token_embeds"] = tuple(values[:, i] for i in range(2))
    with torch.no_grad():
        model.forward_ragged(**reference)
        expected = model.model.last_pre_norm_hidden_states.clone()
        pooled.wrapped(*pooled_args)
        actual = model.model.last_pre_norm_hidden_states.clone()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        changed_pool = pool + 1
        pooled.wrapped(inputs, changed_pool, row_map, *pooled_args[3:])
        assert not torch.equal(actual, model.model.last_pre_norm_hidden_states)


@pytest.mark.parametrize("ple_scale", [None, 0.125])
def test_gemma_pool_exports_dynamic_capacity_and_internal_layer_slices(
        tmp_path, ple_scale):
    """PLE graph input precision and dequantization match its pool contract."""
    model = _model()
    model._ple_pool_scale = ple_scale
    spec = model.onnx_export_spec()
    with _permissive_inline_opset():
        program = torch.onnx.export(
            spec.wrapped,
            spec.args,
            dynamo=True,
            opset_version=23,
            input_names=spec.input_names,
            output_names=spec.output_names,
            dynamic_shapes=spec.dynamic_shapes,
            custom_translation_table=dynamo_translations.
            build_custom_translation_table())
    path = tmp_path / "gemma_pool.onnx"
    program.save(path, external_data=False)
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    inputs = {value.name: value for value in graph.graph.input}
    assert "ple_cache_pool" in inputs and "ple_row_map" in inputs
    assert inputs["ple_cache_pool"].type.tensor_type.elem_type == (
        onnx.TensorProto.FLOAT16
        if ple_scale is None else onnx.TensorProto.FLOAT8E4M3FN)
    if ple_scale is not None:
        dequant = next(node for node in graph.graph.node
                       if node.op_type == "DequantizeLinear")
        initializers = {value.name: value for value in graph.graph.initializer}
        assert float(onnx.numpy_helper.to_array(
            initializers[dequant.input[1]])) == ple_scale
    assert not any(name.startswith("ple_token_embeds_") for name in inputs)
    pool_shape = inputs["ple_cache_pool"].type.tensor_type.shape.dim
    map_shape = inputs["ple_row_map"].type.tensor_type.shape.dim
    assert pool_shape[0].dim_param
    assert len(map_shape) == 2 and map_shape[1].dim_value == 1
    assert map_shape[0].dim_param == inputs[
        "inputs_embeds"].type.tensor_type.shape.dim[0].dim_param
    assert map_shape[0].dim_param != pool_shape[0].dim_param
    assert not {"context_lengths", "kvcache_start_index", "last_token_ids"
                } & inputs.keys()
    for node in graph.graph.node:
        if node.op_type == "AttentionPlugin":
            assert {
                "query_start_offsets", "query_lengths", "past_lengths",
                "execution_phase_marker", "context_sequence_count_carrier"
            } <= set(node.input)
    assert [n.op_type for n in graph.graph.node].count("PleGatherPlugin") == 1


@pytest.mark.parametrize("field",
                         ["eagle_base", "gemma4_mtp_base", "dflash_base"])
def test_pool_preserves_speculative_model_interfaces(field):
    model = _model()
    setattr(model.config, field, True)
    spec = model.onnx_export_spec()
    assert "ple_cache_pool" in spec.input_names
    assert "ple_row_map" in spec.input_names
    assert "hidden_states" in spec.output_names
    assert "packed_attention_mask" in spec.input_names
    torch.export.export(spec.wrapped,
                        spec.args,
                        dynamic_shapes=spec.dynamic_shapes,
                        strict=False)


def _checkpoint_source(
        path,
        dtype=torch.bfloat16,
        key="model.language_model.embed_tokens_per_layer.weight"):
    from safetensors.torch import save_file

    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type":
                                                  "gemma4_text"}))
    (path / "chat_template.jinja").write_text("{{ messages[0].content }}")
    values = (torch.arange(128 * 16).reshape(128, 16).float() / 4096).to(dtype)
    save_file({key: values}, str(path / "model.safetensors"))
    return values


@pytest.mark.parametrize("quantize_at_export", [False, True])
def test_fp8_ple_export_metadata_matches_graph(tmp_path, quantize_at_export):
    """Checkpoint and export-time PLE FP8 both feed scaled FP16 layer inputs."""
    from safetensors.torch import load_file, save_file

    from tensorrt_edgellm.checkpoint.checkpoint_utils import \
        write_runtime_artifacts
    from tensorrt_edgellm.checkpoint.embedding_quantization import \
        prepare_ple_export

    model = _model()
    source = tmp_path / "source"
    values = _checkpoint_source(source)
    key = "model.language_model.embed_tokens_per_layer.weight"
    if not quantize_at_export:
        save_file(
            {
                key: values.to(torch.float8_e4m3fn),
                key + "_scale": torch.tensor(0.125)
            }, str(source / "model.safetensors"))
    before = (source / "model.safetensors").read_bytes()
    output = tmp_path / "output"
    prepare_ple_export(model, str(source), str(output), quantize_at_export)
    write_runtime_artifacts(model, str(source), str(output))
    config = json.loads((output / "config.json").read_text())
    pool = config["ple_pool"]
    assert pool["dtype"] == "F8_E4M3"
    assert pool["source"]["scale"] == 1.0
    assert model._ple_pool_scale == pool["scale"]
    assert (source / "model.safetensors").read_bytes() == before
    table = load_file(str(
        Path(config["checkpoint_dir"]) / "model.safetensors"))
    spec = model.onnx_export_spec()
    assert spec.args[1].dtype == torch.float8_e4m3fn
    row_map = torch.tensor([[0], [3], [3], [127]], dtype=torch.int32)
    expected = (table[key].float()[row_map.long()] *
                torch.tensor(pool["scale"], dtype=torch.float32)).half()
    actual = ops.ple_pool_gather(
        table[key], row_map, torch.tensor(pool["scale"],
                                          dtype=torch.float32)).half()
    torch.testing.assert_close(actual, expected.flatten(-2), rtol=0, atol=0)


@pytest.mark.parametrize("dtype,tag", [(torch.float16, "F16"),
                                       (torch.bfloat16, "BF16"),
                                       (torch.float32, "F32")])
def test_pool_metadata_references_raw_checkpoint_without_copy(
        tmp_path, dtype, tag):
    from tensorrt_edgellm.checkpoint.checkpoint_utils import \
        write_runtime_artifacts

    model = _model()
    source = tmp_path / "source"
    values = _checkpoint_source(source, dtype)
    output = tmp_path / "output"
    before = (source / "model.safetensors").read_bytes()
    write_runtime_artifacts(model, str(source), str(output))
    config = json.loads((output / "config.json").read_text())
    pool = config["ple_pool"]
    assert pool["dtype"] == "F16" and pool["scale"] == 1.0
    assert pool["row_count"] == 128 and pool["row_width"] == 16
    assert pool["source"]["dtype"] == tag
    assert pool["source"][
        "tensor_key"] == "model.language_model.embed_tokens_per_layer.weight"
    assert pool["source"]["bytes"] == values.numel() * values.element_size()
    assert pool["source"]["scale"] == float(torch.tensor(math.sqrt(8)).half())
    raw = values.view(torch.uint8).numpy().tobytes()
    for sample in pool["source"]["samples"]:
        data = bytes.fromhex(sample["data"])
        assert data == raw[sample["offset"]:sample["offset"] + len(data)]
    assert config["checkpoint_dir"] == str(source.resolve())
    assert not (output / "ple").exists()
    assert not (output / "ple_embedding.safetensors").exists()
    assert not hasattr(model.model, "embed_tokens_per_layer")
    assert (source / "model.safetensors").read_bytes() == before


def test_missing_ple_checkpoint_does_not_publish_config(tmp_path):
    from tensorrt_edgellm.checkpoint.checkpoint_utils import \
        write_runtime_artifacts

    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    with pytest.raises((ValueError, FileNotFoundError)):
        write_runtime_artifacts(_model(), str(source), str(output))
    assert not (output / "config.json").exists()
    assert not (output / "embedding.safetensors").exists()


def test_non_ple_model_has_no_pool_inputs():
    config = _model().config
    config.hidden_size_per_layer_input = 0
    spec = Gemma4ForCausalLM(config).half().onnx_export_spec()
    assert not any(name.startswith("ple_") for name in spec.input_names)


def test_nonshared_rank_preserves_pool_metadata(tmp_path):
    from tensorrt_edgellm.checkpoint.checkpoint_utils import \
        write_runtime_artifacts

    model = _model()
    source = tmp_path / "source"
    _checkpoint_source(source)
    output = tmp_path / "output"
    model.config.mapping = Mapping(world_size=2, tp_size=2)
    write_runtime_artifacts(model,
                            str(source),
                            str(output),
                            config_filename="config_world2.json")
    descriptor = json.loads(
        (output / "config_world2.json").read_text())["ple_pool"]
    model.config.mapping = Mapping(world_size=2, rank=1, tp_size=2, tp_rank=1)
    write_runtime_artifacts(model,
                            str(source),
                            str(output),
                            config_filename="config_world2.json",
                            write_shared_artifacts=False)
    config = json.loads((output / "config_world2.json").read_text())
    assert config["ple_pool"] == descriptor
    assert [entry["rank"] for entry in config["rank_configs"]] == [0, 1]
    assert not (output / "ple").exists()


def test_reexport_updates_graph_without_copying_ple(tmp_path):
    from tensorrt_edgellm.onnx.export import export_onnx

    model = _model()
    source = tmp_path / "source"
    _checkpoint_source(source)
    output = tmp_path / "output"
    before = (source / "model.safetensors").read_bytes()
    export_onnx(model, str(output / "model.onnx"), model_dir=str(source))
    metadata = json.loads((output / "config.json").read_text())["ple_pool"]
    original_weights = {
        tensor.name: onnx.numpy_helper.to_array(tensor)
        for tensor in onnx.load(output / "model.onnx").graph.initializer
    }
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.25)
    export_onnx(model, str(output / "model.onnx"), model_dir=str(source))
    updated_weights = {
        tensor.name: onnx.numpy_helper.to_array(tensor)
        for tensor in onnx.load(output / "model.onnx").graph.initializer
    }
    assert any(name in updated_weights
               and not np.array_equal(value, updated_weights[name])
               for name, value in original_weights.items())
    assert json.loads(
        (output / "config.json").read_text())["ple_pool"] == metadata
    assert (source / "model.safetensors").read_bytes() == before
    assert not (output / "ple").exists()
    assert not (output / "ple_embedding.safetensors").exists()


def test_checkpoint_loader_does_not_materialize_runtime_ple(
        tmp_path, monkeypatch):
    from safetensors.torch import save_file

    from tensorrt_edgellm.checkpoint import loader
    from tensorrt_edgellm.model import AutoModel

    model = _model()
    config = {
        "model_type": "gemma4_text",
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "intermediate_size": 64,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "rope_theta": 10000.0,
        "max_position_embeddings": 128,
        "hidden_size_per_layer_input": 8,
        "vocab_size_per_layer_input": 128,
        "layer_types": ["full_attention"] * 2,
        "hidden_activation": "gelu_pytorch_tanh"
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    weights = model.state_dict()
    weights["model.embed_tokens_per_layer.weight"] = torch.ones(
        128, 16, dtype=torch.bfloat16)
    save_file(weights, str(tmp_path / "model.safetensors"))
    safe_open = loader.safe_open

    class HeaderOnlyPle:

        def __init__(self, *args, **kwargs):
            self.source = safe_open(*args, **kwargs)

        def __enter__(self):
            self.source.__enter__()
            return self

        def __exit__(self, *args):
            return self.source.__exit__(*args)

        def keys(self):
            return self.source.keys()

        def get_tensor(self, key):
            if key.endswith("embed_tokens_per_layer.weight"):
                raise AssertionError(
                    "PLE payload must remain in the checkpoint")
            return self.source.get_tensor(key)

    monkeypatch.setattr(loader, "safe_open", HeaderOnlyPle)
    loaded = AutoModel.from_pretrained(str(tmp_path), device="cpu")
    assert not hasattr(loaded.model, "embed_tokens_per_layer")
    torch.testing.assert_close(loaded.model.embed_tokens.weight,
                               model.model.embed_tokens.weight)
