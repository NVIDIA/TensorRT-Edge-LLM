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

import json
from types import SimpleNamespace

import numpy as np
import pytest

safetensors_numpy = pytest.importorskip("safetensors.numpy")

from experimental.builder.core import quantization, weight_policy
from experimental.builder.core.artifacts.external_weights import (
    _share_tied_embedding, checkpoint_identity, patch_external_weight_config)
from experimental.builder.core.safetensors_np import (SafetensorsStore,
                                                      _ShardFile)
from experimental.builder.core.weights import (LinearWeights, ParameterSpec,
                                               Weights)


def _checkpoint(path, values):
    path.mkdir()
    safetensors_numpy.save_file(
        {"model.embed_tokens.weight": np.asarray(values, dtype=np.float16)},
        str(path / "model.safetensors"),
    )


def _binding(engine_name, key="model.embed_tokens.weight", **extra):
    return {
        "engine_name": engine_name,
        "checkpoint_keys": [key],
        "source_layout": "fp16",
        "dtype": "F16",
        "shape": [8, 4],
        **extra,
    }


def test_checkpoint_identity_rejects_same_shape_different_content(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _checkpoint(first, np.arange(32).reshape(8, 4))
    _checkpoint(second, np.arange(32, 64).reshape(8, 4))
    bindings = [_binding("lm_head.weight")]

    first_identity = checkpoint_identity(bindings, str(first))
    second_identity = checkpoint_identity(bindings, str(second))
    first_tensor = first_identity["sources"]["component"]["tensors"][
        "model.embed_tokens.weight"]
    second_tensor = second_identity["sources"]["component"]["tensors"][
        "model.embed_tokens.weight"]

    assert first_tensor["dtype"] == second_tensor["dtype"]
    assert first_tensor["shape"] == second_tensor["shape"]
    assert first_tensor["bytes"] == second_tensor["bytes"]
    assert first_tensor["samples"] != second_tensor["samples"]
    assert set(first_identity["sources"]["component"]["files"]) == {
        "model.safetensors"
    }


def test_checkpoint_identity_reads_bounded_payload(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    _checkpoint(checkpoint, np.zeros((1 << 18, 4), dtype=np.float16))
    sampled_bytes = 0
    original = _ShardFile.raw_slice

    def counted(self, name, offset, length):
        nonlocal sampled_bytes
        sampled_bytes += length
        return original(self, name, offset, length)

    monkeypatch.setattr(_ShardFile, "raw_slice", counted)
    with SafetensorsStore(str(checkpoint)) as store:
        identity = store.tensor_identity("model.embed_tokens.weight")

    assert len(identity["samples"]) == 3
    assert sampled_bytes == 48


def test_checkpoint_identity_records_shard_index(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    key = "model.embed_tokens.weight"
    shard_name = "model-00001-of-00001.safetensors"
    safetensors_numpy.save_file(
        {key: np.arange(32, dtype=np.float16).reshape(8, 4)},
        str(checkpoint / shard_name),
    )
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {
            key: shard_name
        }}))

    identity = checkpoint_identity([_binding("lm_head.weight")],
                                   str(checkpoint))

    assert set(identity["sources"]["component"]["files"]) == {
        "model.safetensors.index.json",
        shard_name,
    }


def test_checkpoint_identity_sampling_cost_is_model_bounded(
        tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    tensors = {
        f"model.layers.{index}.weight": np.full((32, 32),
                                                index,
                                                dtype=np.float16)
        for index in range(64)
    }
    safetensors_numpy.save_file(tensors, str(checkpoint / "model.safetensors"))
    bindings = [
        _binding(f"layer_{index}.weight", key=key)
        for index, key in enumerate(tensors)
    ]
    sampled_bytes = 0
    original = _ShardFile.raw_slice

    def counted(self, name, offset, length):
        nonlocal sampled_bytes
        sampled_bytes += length
        return original(self, name, offset, length)

    monkeypatch.setattr(_ShardFile, "raw_slice", counted)
    identity = checkpoint_identity(bindings, str(checkpoint))
    recorded = identity["sources"]["component"]["tensors"]

    assert len(recorded) == len(tensors)
    assert sum(bool(tensor["samples"]) for tensor in recorded.values()) == 16
    assert sampled_bytes == 16 * 3 * 16


@pytest.mark.parametrize(("tied", "aliases"), ((True, True), (False, False)))
def test_embedding_alias_requires_model_tying(tied, aliases):
    bindings = [
        _binding("lm_head.weight"),
        _binding("__embedding__", role="embedding", embedding_scale=1.0),
    ]

    _share_tied_embedding(bindings, tied)

    assert (bindings[1].get("storage_alias_of") == "lm_head.weight") is aliases


@pytest.mark.parametrize(
    ("candidate", "embedding"),
    [
        (
            _binding("lm_head.weight"),
            _binding("__embedding__", role="embedding", embedding_scale=2.0),
        ),
        (
            _binding("lm_head.weight", embedding_scale=2.0),
            _binding("__embedding__", role="embedding", embedding_scale=1.0),
        ),
        (
            _binding("lm_head.weight"),
            _binding("__embedding__",
                     key="other.embed_tokens.weight",
                     role="embedding",
                     embedding_scale=1.0),
        ),
    ],
)
def test_embedding_with_different_transform_does_not_alias(
        candidate, embedding):
    bindings = [candidate, embedding]

    _share_tied_embedding(bindings, True)

    assert "storage_alias_of" not in bindings[1]


@pytest.mark.parametrize(
    ("tied", "expected"),
    ((True, "model.embed_tokens.weight"), (False, "lm_head.weight")),
)
def test_lm_head_checkpoint_resolution_follows_model_tying(
        tmp_path, tied, expected):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    safetensors_numpy.save_file(
        {
            "model.embed_tokens.weight": np.zeros((8, 4), dtype=np.float16),
            "lm_head.weight": np.ones((8, 4), dtype=np.float16),
        },
        str(checkpoint / "model.safetensors"),
    )

    def resolve_candidates(name, **_):
        return (("model.embed_tokens.weight", )
                if name == "lm_head.weight" else ())

    weights = Weights(
        str(checkpoint),
        conversion=SimpleNamespace(resolve_candidates=resolve_candidates),
        tie_word_embeddings=tied)
    try:
        assert weights.checkpoint_key("lm_head.weight") == expected
    finally:
        weights.close()


def test_runtime_config_records_checkpoint_identity(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    identity = {
        "version": 1,
        "sources": {
            "component": {
                "build_source": "/checkpoint",
                "tensors": {},
            },
        },
    }

    patch_external_weight_config(str(config_path), [_binding("weight")],
                                 identity)

    config = json.loads(config_path.read_text())
    assert config["checkpoint_identity"] == identity


def test_runtime_config_requires_checkpoint_identity(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")

    with pytest.raises(ValueError, match="requires a checkpoint identity"):
        patch_external_weight_config(str(config_path), [_binding("weight")],
                                     {})


def test_unknown_quantization_algorithm_is_not_fp16():
    with pytest.raises(ValueError, match="unsupported.*algorithm"):
        quantization.algorithm_to_type("UNSUPPORTED_QUANT")


@pytest.mark.parametrize("algorithm", ("W4A8_MXFP4_FP8", "W4A8_MXFP4_MXFP8"))
def test_mxfp4_checkpoint_is_not_misclassified_as_fp8(algorithm):
    with pytest.raises(ValueError, match="MXFP4 weight layouts"):
        quantization.algorithm_to_type(algorithm)


def test_nvfp4_weight_layout_takes_precedence_over_activation_precision():
    assert (quantization.algorithm_to_type("W4A8_NVFP4_FP8") ==
            quantization.QUANT_NVFP4)


def test_unknown_compressed_tensor_format_is_not_fp16(tmp_path):
    embedded = {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
        },
    }

    with pytest.raises(ValueError, match="unsupported compressed-tensors"):
        quantization.parse_quantization(str(tmp_path), embedded, {})


def test_declared_nvfp4_layer_without_payload_is_not_fp16():
    weights = object.__new__(Weights)
    weights.is_nvfp4 = lambda prefix: False

    with pytest.raises(ValueError, match="selects NVFP4"):
        weights.linear_descriptor("model.layers.0.mlp.up_proj",
                                  quantization.QUANT_NVFP4)


@pytest.mark.parametrize(
    "mode, source_shape, expected_shape, shard_axis",
    (
        ("column", (12288, 4096), (6144, 4096), 0),
        ("row", (4096, 12288), (4096, 6144), 1),
    ),
)
def test_tp_external_fp16_shard_recipe_is_rank_neutral(mode, source_shape,
                                                       expected_shape,
                                                       shard_axis):
    descriptor = LinearWeights(
        quantization.QUANT_FP16,
        ParameterSpec(source_shape, np.float16),
        weight_recipe={"checkpoint_keys": ["weight"]},
    )

    rank0 = Weights.shard_linear(descriptor, mode, 2, 0)
    rank1 = Weights.shard_linear(descriptor, mode, 2, 1)

    assert rank0.weight.shape == expected_shape
    assert rank0.weight_recipe == rank1.weight_recipe
    assert rank0.weight_recipe["extra"]["tp_shard"] == {
        "axis": shard_axis,
        "size": 2,
    }


def test_tp_nvfp4_row_parallel_shards_scale_on_input_axis():
    descriptor = LinearWeights(
        quantization.QUANT_NVFP4,
        np.zeros((4096, 2048), dtype=np.uint8),
        weight_scale=np.zeros((4096, 256), dtype=np.uint8),
        group_size=16,
    )

    sharded = Weights.shard_linear(descriptor, "row", 2, 0)

    assert sharded.weight.shape == (4096, 1024)
    assert sharded.weight_scale.shape == (4096, 128)


def test_tp_external_nvfp4_shard_recipe_is_rank_neutral():
    descriptor = LinearWeights(
        quantization.QUANT_NVFP4,
        ParameterSpec((4096, 2048), np.uint8),
        weight_scale=ParameterSpec((4096, 256), np.uint8),
        group_size=16,
        weight_recipe={"checkpoint_keys": ["weight"]},
        scale_recipe={"checkpoint_keys": ["weight_scale"]},
        logical_out_features=4096,
        logical_in_features=4096,
    )

    rank0 = Weights.shard_linear(descriptor, "row", 2, 0)
    rank1 = Weights.shard_linear(descriptor, "row", 2, 1)

    assert rank0.weight.shape == (4096, 1024)
    assert rank0.weight_scale.shape == (4096, 128)
    assert rank0.weight_recipe == rank1.weight_recipe
    assert rank0.scale_recipe == rank1.scale_recipe
    assert rank0.weight_recipe["extra"]["tp_shard"] == {
        "axis": 1,
        "size": 2,
    }
    assert rank0.scale_recipe["extra"]["tp_shard"] == {
        "axis": 1,
        "size": 2,
    }


def test_tp_nvfp4_resource_rejects_signed_packed_storage():
    prefix = "model.layers.0.self_attn.o_proj"
    dtypes = {
        prefix + ".weight": "I8",
        prefix + ".weight_scale": "F8_E4M3",
        prefix + ".weight_scale_2": "F32",
    }

    class DtypeOnlyStore:

        @staticmethod
        def has(name):
            return name in dtypes

        @staticmethod
        def dtype(name):
            return dtypes[name]

    weights = object.__new__(Weights)
    weights.store = DtypeOnlyStore()
    weights.conversion = None
    weights.vocab_map = None
    weights.component = "llm"
    weights.spec_type = "none"
    weights.spec_role = "none"
    weights.tie_word_embeddings = False

    assert weights.linear_nvfp4_tp_metadata(prefix) is None


def test_tp_default_bakes_small_fp16_parameters():
    from experimental.builder.core.builder import BuildArgs

    default = BuildArgs("/checkpoint", "/engine", tp_size=2).weight_policy
    assert default.wants(weight_policy.EXTERNAL_WEIGHT_EMBEDDING)
    assert default.wants(weight_policy.EXTERNAL_WEIGHT_LM_HEAD)
    assert default.wants(weight_policy.EXTERNAL_WEIGHT_NVFP4_TP)
    assert not default.wants(weight_policy.EXTERNAL_WEIGHT_FP16)

    fully_external = BuildArgs(
        "/checkpoint",
        "/engine",
        tp_size=2,
        externalize_weights=(weight_policy.EXTERNAL_WEIGHT_ALL, ),
    ).weight_policy
    assert fully_external.wants(weight_policy.EXTERNAL_WEIGHT_EMBEDDING)
    assert fully_external.wants(weight_policy.EXTERNAL_WEIGHT_LM_HEAD)
    assert fully_external.wants(weight_policy.EXTERNAL_WEIGHT_FP16)
