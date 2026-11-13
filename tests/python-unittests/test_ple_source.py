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
"""Raw checkpoint PLE metadata and bounded source inspection."""

import json
import struct
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from tensorrt_edgellm.checkpoint.checkpoint_utils import \
    _gemma_ple_checkpoint_metadata
from tensorrt_edgellm.checkpoint.ple_source import (
    gemma_ple_identity, make_gemma_ple_source_metadata)


def _config(diffusion=False):
    return SimpleNamespace(is_diffusion_gemma=diffusion,
                           vocab_size_per_layer_input=4,
                           num_hidden_layers=2,
                           hidden_size_per_layer_input=3)


def _write_ple_shard(path, checkpoint_format):
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.arange(24).reshape(4, 6).bfloat16()
    values = {"model.embed_tokens_per_layer.weight": tensor}
    if checkpoint_format == "safetensors":
        save_file(values, str(path))
    else:
        torch.save(values, path)
    return tensor.view(torch.uint8).numpy().tobytes()


def _write_ple_index(checkpoint, checkpoint_format, shard_name):
    checkpoint.mkdir(parents=True, exist_ok=True)
    index_name = ("model.safetensors.index.json" if checkpoint_format
                  == "safetensors" else "pytorch_model.bin.index.json")
    (checkpoint / index_name).write_text(
        json.dumps({
            "weight_map": {
                "model.embed_tokens_per_layer.weight": shard_name
            }
        }))


@pytest.mark.parametrize("contents,message", [
    (b"short", "Truncated safetensors header"),
    (struct.pack("<Q", 200) + b"{}", "Truncated safetensors header"),
    (struct.pack("<Q", 16 * 1024 * 1024 + 1), "bounded header limit"),
    (struct.pack("<Q", 15) + b'{"x":1,"x":2}  ',
     "Duplicate safetensors header key"),
],
                         ids=[
                             "short-prefix", "truncated-header",
                             "oversized-header", "duplicate-key"
                         ])
def test_source_rejects_malformed_safetensors_headers(tmp_path, contents,
                                                      message):
    (tmp_path / "weights.safetensors").write_bytes(contents)
    _write_ple_index(tmp_path, "safetensors", "weights.safetensors")
    with pytest.raises(ValueError, match=message):
        _gemma_ple_checkpoint_metadata(str(tmp_path), _config())


@pytest.mark.parametrize("override,message", [
    ({
        "dtype": "F32"
    }, "Invalid PLE checkpoint"),
    ({
        "shape": [4, 5]
    }, "Invalid PLE checkpoint"),
    ({
        "shape": [24]
    }, "Invalid PLE checkpoint"),
    ({
        "shape": [0, 6]
    }, "Invalid PLE checkpoint"),
    ({
        "shape": [True, 6]
    }, "Invalid safetensors tensor shape"),
    ({
        "shape": [-1, 6]
    }, "Invalid safetensors tensor shape"),
    ({
        "data_offsets": [-1, 47]
    }, "tensor range exceeds"),
    ({
        "data_offsets": [48, 0]
    }, "tensor range exceeds"),
    ({
        "data_offsets": [0, 49]
    }, "tensor range exceeds"),
    ({
        "data_offsets": [0, 47]
    }, "Invalid PLE checkpoint"),
    ({
        "data_offsets": [0, "48"]
    }, "Invalid safetensors data_offsets"),
])
def test_source_rejects_invalid_safetensors_records(tmp_path, override,
                                                    message):
    record = {"dtype": "BF16", "shape": [4, 6], "data_offsets": [0, 48]}
    record.update(override)
    header = json.dumps({
        "model.embed_tokens_per_layer.weight": record
    }).encode()
    (tmp_path / "weights.safetensors"
     ).write_bytes(struct.pack("<Q", len(header)) + header + bytes(48))
    _write_ple_index(tmp_path, "safetensors", "weights.safetensors")
    with pytest.raises(ValueError, match=message):
        _gemma_ple_checkpoint_metadata(str(tmp_path), _config())


@pytest.mark.parametrize("checkpoint_format", ["safetensors", "pytorch"])
@pytest.mark.parametrize("path_kind", ["absolute", "parent", "nested_parent"])
def test_indexed_ple_source_rejects_unsafe_shard_paths(tmp_path,
                                                       checkpoint_format,
                                                       path_kind):
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "nested").mkdir(parents=True)
    filename = ("weights.safetensors"
                if checkpoint_format == "safetensors" else "weights.bin")
    if path_kind == "absolute":
        shard_name = str(tmp_path / "outside" / filename)
    elif path_kind == "parent":
        shard_name = "../outside/" + filename
    else:
        shard_name = "nested/../" + filename
    _write_ple_shard(checkpoint / shard_name, checkpoint_format)
    _write_ple_index(checkpoint, checkpoint_format, shard_name)

    with pytest.raises(ValueError, match="Invalid PLE checkpoint shard path"):
        _gemma_ple_checkpoint_metadata(str(checkpoint), _config())


@pytest.mark.parametrize("checkpoint_format", ["safetensors", "pytorch"])
@pytest.mark.parametrize("shard_name", ["", None, 17, ["weights.bin"]],
                         ids=["empty", "null", "integer", "list"])
def test_indexed_ple_source_rejects_invalid_shard_names(
        tmp_path, checkpoint_format, shard_name):
    _write_ple_index(tmp_path, checkpoint_format, shard_name)
    with pytest.raises(ValueError, match="Invalid PLE checkpoint shard path"):
        _gemma_ple_checkpoint_metadata(str(tmp_path), _config())


@pytest.mark.parametrize("checkpoint_format", ["safetensors", "pytorch"])
def test_indexed_ple_source_accepts_nested_relative_shards(
        tmp_path, checkpoint_format):
    shard_name = ("nested/weights.safetensors" if checkpoint_format
                  == "safetensors" else "nested/weights.bin")
    raw = _write_ple_shard(tmp_path / shard_name, checkpoint_format)
    _write_ple_index(tmp_path, checkpoint_format, shard_name)
    source = _gemma_ple_checkpoint_metadata(str(tmp_path), _config())["source"]
    assert source["dtype"] == "BF16" and source["bytes"] == len(raw)
    for sample in source["samples"]:
        data = bytes.fromhex(sample["data"])
        assert raw[sample["offset"]:sample["offset"] + len(data)] == data
    if checkpoint_format == "pytorch":
        assert source["location"]["file"] == shard_name


@pytest.mark.parametrize("checkpoint_format", ["safetensors", "pytorch"])
def test_indexed_ple_source_accepts_snapshot_symlinks_to_external_blobs(
        tmp_path, checkpoint_format):
    checkpoint = tmp_path / "snapshots" / "revision"
    checkpoint.mkdir(parents=True)
    raw = _write_ple_shard(tmp_path / "blobs" / "hash", checkpoint_format)
    shard_name = ("weights.safetensors"
                  if checkpoint_format == "safetensors" else "weights.bin")
    (checkpoint / shard_name).symlink_to("../../blobs/hash")
    _write_ple_index(checkpoint, checkpoint_format, shard_name)
    source = _gemma_ple_checkpoint_metadata(str(checkpoint),
                                            _config())["source"]
    assert source["dtype"] == "BF16" and source["bytes"] == len(raw)
    for sample in source["samples"]:
        data = bytes.fromhex(sample["data"])
        assert raw[sample["offset"]:sample["offset"] + len(data)] == data
    if checkpoint_format == "pytorch":
        assert source["location"]["file"] == shard_name


@pytest.mark.parametrize("key,diffusion", [
    ("model.embed_tokens_per_layer.weight", False),
    ("model.language_model.embed_tokens_per_layer.weight", False),
    ("language_model.model.embed_tokens_per_layer.weight", False),
    ("embed_tokens_per_layer.weight", False),
    ("model.decoder.embed_tokens_per_layer.weight", True),
])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
def test_source_metadata_preserves_provider_key_and_dtype(
        tmp_path, key, diffusion, dtype):
    value = (torch.arange(24).reshape(4, 6).float() / 37).to(dtype)
    values = {key: value}
    if diffusion:
        values[
            "model.encoder.language_model.embed_tokens_per_layer.weight"] = value + 1
    save_file(values, str(tmp_path / "weights.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {
                name: "weights.safetensors"
                for name in values
            }}))
    metadata = _gemma_ple_checkpoint_metadata(str(tmp_path),
                                              _config(diffusion))
    assert metadata["dtype"] == "F16" and metadata["scale"] == 1.0
    source = metadata["source"]
    assert source["tensor_key"] == key
    assert source["dtype"] == {
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.float32: "F32"
    }[dtype]
    assert source["scale"] == struct.unpack("<e", struct.pack("<e", 3**0.5))[0]
    assert "location" not in source
    raw = value.view(torch.uint8).numpy().tobytes()
    assert source["bytes"] == len(raw)
    assert sum(
        len(bytes.fromhex(sample["data"]))
        for sample in source["samples"]) <= 48
    for sample in source["samples"]:
        data = bytes.fromhex(sample["data"])
        assert raw[sample["offset"]:sample["offset"] + len(data)] == data


def test_pytorch_source_records_exact_original_archive_range(tmp_path):
    key = "model.language_model.embed_tokens_per_layer.weight"
    values = torch.arange(24).reshape(4, 6).bfloat16()
    path = tmp_path / "pytorch_model.bin"
    torch.save({key: values}, path)
    metadata = _gemma_ple_checkpoint_metadata(str(tmp_path), _config())
    source = metadata["source"]
    location = source["location"]
    assert location["file"] == path.name
    assert location["bytes"] == values.numel() * values.element_size()
    with path.open("rb") as reader:
        reader.seek(location["offset"])
        assert reader.read(location["bytes"]) == values.view(
            torch.uint8).numpy().tobytes()
    assert source["tensor_key"] == key and source["dtype"] == "BF16"


def test_ambiguous_ple_source_is_rejected(tmp_path):
    save_file(
        {
            "model.embed_tokens_per_layer.weight": torch.ones(4, 6),
            "other.embed_tokens_per_layer.weight": torch.ones(4, 6)
        }, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="Expected one"):
        gemma_ple_identity(tmp_path)


def test_fp8_gemma_ple_metadata_preserves_checkpoint_scale(tmp_path):
    """Both exporters must carry the FP8 PLE scale into the runtime pool."""
    from tensorrt_edgellm.builder.core.artifacts.embeddings import \
        ple_pool_metadata
    from tensorrt_edgellm.builder.core.weights import Weights

    key = "model.embed_tokens_per_layer.weight"
    save_file(
        {
            key: torch.ones(4, 6).to(torch.float8_e4m3fn),
            key + "_scale": torch.tensor(0.25, dtype=torch.float32),
        }, str(tmp_path / "model.safetensors"))
    embed_scale = struct.unpack("<f", struct.pack("<f", 3**0.5))[0]
    expected = struct.unpack("<f", struct.pack("<f", 0.25 * embed_scale))[0]
    metadata = _gemma_ple_checkpoint_metadata(str(tmp_path), _config())
    weights = Weights(str(tmp_path))
    try:
        assert ple_pool_metadata(weights, _config()) == metadata
    finally:
        weights.close()
    assert metadata["dtype"] == "F8_E4M3"
    assert metadata["scale"] == expected
    assert metadata["source"]["dtype"] == "F8_E4M3"
    assert metadata["source"]["scale"] == 1.0


@pytest.mark.parametrize("field,value", [("dtype", "F8_E4M3"),
                                         ("shape", [4, 7]), ("bytes", 47),
                                         ("samples", [])])
def test_invalid_ple_metadata_is_rejected(field, value):
    identity = {
        "dtype": "BF16",
        "shape": [4, 6],
        "bytes": 48,
        "samples": [{
            "offset": 0,
            "data": "00" * 16
        }]
    }
    identity[field] = value
    with pytest.raises(ValueError, match="Invalid PLE"):
        make_gemma_ple_source_metadata("weight", identity, 4, 6, 3)


@pytest.mark.parametrize("key", [
    "model.embed_tokens_per_layer.weight",
    "model.language_model.embed_tokens_per_layer.weight",
    "language_model.model.embed_tokens_per_layer.weight",
    "embed_tokens_per_layer.weight",
])
def test_direct_and_onnx_producers_emit_identical_checkpoint_contract(
        tmp_path, key):
    from tensorrt_edgellm.builder.core.artifacts.embeddings import \
        ple_pool_metadata
    from tensorrt_edgellm.builder.core.weights import Weights
    from tensorrt_edgellm.builder.models.gemma4 import weights as conversion

    save_file({key: torch.arange(24).reshape(4, 6).bfloat16()},
              str(tmp_path / "model.safetensors"))
    weights = Weights(str(tmp_path), conversion=conversion)
    try:
        direct = ple_pool_metadata(weights, _config())
    finally:
        weights.close()
    assert direct == _gemma_ple_checkpoint_metadata(str(tmp_path), _config())
    assert not (tmp_path / "ple").exists()
    assert not (tmp_path / "ple_embedding.safetensors").exists()


@pytest.mark.parametrize("checkpoint_override", ["", "/relocated/checkpoint"])
def test_direct_weight_config_preserves_ple_checkpoint_fallback(
        tmp_path, checkpoint_override):
    from tensorrt_edgellm.builder.core.artifacts.external_weights import \
        patch_external_weight_config

    path = tmp_path / "config.json"
    pool = {"source": {"tensor_key": "model.embed_tokens_per_layer.weight"}}
    path.write_text(
        json.dumps({
            "ple_pool": pool,
            "checkpoint_dir": "/original/checkpoint"
        }))
    patch_external_weight_config(str(path), [{
        "engine_name": "weight"
    }], {"tensors": []}, checkpoint_override)
    config = json.loads(path.read_text())
    assert config["checkpoint_dir"] == (checkpoint_override
                                        or "/original/checkpoint")
    assert config["ple_pool"] == pool


def test_export_time_ngram_fp8_writes_released_layout(tmp_path):
    """``--fp8-ple`` turns a BF16 n-gram table into ``split_ngram_parts`` FP8 shards under one
    F32 scale, copies the hash constants beside them and points the model at that directory;
    an FP8 table passes through untouched."""
    from tensorrt_edgellm.checkpoint.embedding_quantization import \
        prepare_ple_export
    from tensorrt_edgellm.checkpoint.ple_source import make_ngram_pool_metadata

    prefix = "model.language_model.layers.1.ple.ple_embedding"
    torch.manual_seed(0)
    table = torch.randn(12, 4).bfloat16()
    constants = {
        f"{prefix}.layer_multipliers": torch.tensor([3, 5, 7]),
        f"{prefix}.ngram_heads_vocab_sizes": torch.tensor([5, 7]),
        f"{prefix}.ngram_heads_offsets": torch.tensor([0, 5]),
    }
    source = tmp_path / "source"
    source.mkdir()
    save_file(
        {
            f"{prefix}.ngram_embedding.shard_0.weight": table[:7],
            f"{prefix}.ngram_embedding.shard_1.weight": table[7:],
            **constants
        }, str(source / "model.safetensors"))

    def model_with_scale_buffer():
        model = torch.nn.Module()
        model.ple_embedding = torch.nn.Module()
        model.ple_embedding.ngram_embedding = torch.nn.Module()
        model.ple_embedding.ngram_embedding.register_buffer(
            "weight_scale", torch.zeros(1, dtype=torch.float16))
        model.config = SimpleNamespace(ple_enabled=False,
                                       ngram_layers=[1],
                                       split_ngram_parts=3)
        return model

    model = model_with_scale_buffer()
    output = tmp_path / "output"
    prepare_ple_export(model, str(source), str(output), fp8_ple=True)

    assert model._ple_checkpoint_dir == str(output / "ple")
    written = load_file(str(output / "ple" / "model.safetensors"))
    shards = [
        written[f"{prefix}.ngram_embedding.shard_{index}.weight"]
        for index in range(3)
    ]
    scale = written[f"{prefix}.ngram_embedding.weight_scale"]
    assert len(written) == len(shards) + 1 + len(constants)
    assert [tuple(shard.shape) for shard in shards] == [(4, 4)] * 3
    assert all(shard.dtype == torch.float8_e4m3fn for shard in shards)
    assert scale.dtype == torch.float32 and scale.shape == (1, )
    assert scale.item() == (table.abs().amax().float() / 448).item()
    assert all(
        torch.equal(written[key], value) for key, value in constants.items())
    torch.testing.assert_close(torch.cat(shards).float() * scale,
                               table.float(),
                               rtol=2**-3,
                               atol=2**-9 * scale.item())
    assert model.ple_embedding.ngram_embedding.weight_scale.item(
    ) == scale.half().item()
    pool = make_ngram_pool_metadata(str(output / "ple"), embed_dim=8, heads=2)
    assert (pool["row_count"], pool["scale"]) == (12, scale.item())

    again = model_with_scale_buffer()
    prepare_ple_export(again,
                       str(output / "ple"),
                       str(tmp_path / "again"),
                       fp8_ple=True)
    assert not hasattr(again, "_ple_checkpoint_dir")
    assert not (tmp_path / "again").exists()
