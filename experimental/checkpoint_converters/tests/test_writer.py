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
"""Behavioral contracts for HF checkpoint publication."""

import json

import pytest
import torch

from experimental.checkpoint_converters import writer as writer_module
from experimental.checkpoint_converters.models.gemma4 import \
    schema as gemma_schema
from experimental.checkpoint_converters.models.gemma4.flax_adapter import \
    Gemma4FlaxAdapter
from experimental.checkpoint_converters.models.gemma4.writer import \
    Gemma4CheckpointWritePolicy
from experimental.checkpoint_converters.schema import ConversionError
from experimental.checkpoint_converters.transforms import (MappingPlan,
                                                           MappingRule,
                                                           Transform,
                                                           TransformKind)
from experimental.checkpoint_converters.writer import HFCheckpointWriter

from .tokenizer_fixtures import create_conversion_paths


def _writer(output, max_checkpoint_file_bytes):
    return HFCheckpointWriter(output, max_checkpoint_file_bytes,
                              Gemma4CheckpointWritePolicy())


@pytest.fixture
def conversion_paths(tmp_path):
    """Create compatible local paths for writer validation."""
    return create_conversion_paths(tmp_path)


def _small_rule(source, target, shape=(2, )):
    """Create one identity rule for writer-only behavior tests."""
    return MappingRule(
        source_path=(source, ),
        source_shape=shape,
        target_names=(target, ),
        target_shapes=(shape, ),
        transform=Transform(TransformKind.IDENTITY, output_shapes=(shape, )),
    )


def test_checkpoint_partitioning_is_deterministic_bounded_and_keeps_rules_whole(
        tmp_path):
    """Checkpoint files use canonical order while split-source targets stay together."""
    split = MappingRule(
        source_path=("split", ),
        source_shape=(2, 2),
        target_names=("b", "c"),
        target_shapes=((2, ), (2, )),
        transform=Transform(TransformKind.SPLIT_TRANSPOSE_RESHAPE,
                            output_shapes=((2, ), (2, )),
                            split_axis=0),
    )
    ple = _small_rule("ple",
                      "model.language_model.embed_tokens_per_layer.weight",
                      (10, ))
    plan = MappingPlan((ple, split, _small_rule("a", "a")), "synthetic")
    writer = _writer(tmp_path / "out", 8)

    first = writer.partition_tensors_into_checkpoint_files(plan)
    second = writer.partition_tensors_into_checkpoint_files(plan)

    assert first == second
    assert tuple(item.name for item in first) == (
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    )
    assert tuple(item.target_names for item in first) == (
        ("a", ),
        ("b", "c"),
        ("model.language_model.embed_tokens_per_layer.weight", ),
    )
    assert first[-1].target_bytes == 20


def test_checkpoint_partitioning_rejects_invalid_bounds_and_oversize_rules(
        tmp_path):
    """Only the indivisible PLE embedding may exceed the explicit file bound."""
    plan = MappingPlan((_small_rule("large", "large", (5, )), ), "synthetic")
    writer = _writer(tmp_path / "out", 8)

    with pytest.raises(ConversionError, match="exceeds checkpoint file bound"):
        writer.partition_tensors_into_checkpoint_files(plan)
    with pytest.raises(ConversionError, match="must be positive"):
        _writer(tmp_path / "invalid",
                0).partition_tensors_into_checkpoint_files(plan)


def test_writer_publishes_complete_hf_metadata_and_allowlisted_tokenizer(
        conversion_paths, monkeypatch):
    """A successful write atomically publishes one internally consistent directory."""
    safetensors_torch = pytest.importorskip("safetensors.torch")
    _, tokenizer, output = conversion_paths
    rules = (_small_rule("z", "model.language_model.z.weight"),
             _small_rule("a", "model.language_model.a.weight"))
    plan = MappingPlan(rules, "synthetic")
    source_tree = {"z": object(), "a": object()}

    def materialize(source, rule):
        value = 2 if source is source_tree["z"] else 1
        return {
            rule.target_names[0]:
            torch.full(rule.target_shapes[0], value, dtype=torch.bfloat16)
        }

    monkeypatch.setattr(writer_module, "convert_tensor", materialize)
    report = _writer(output, 8).write(tokenizer, source_tree, plan,
                                      {"model_type": "gemma4"})
    index = json.loads(
        (output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (output / "conversion.json").read_text(encoding="utf-8"))

    assert report.output_path == output
    assert report.tensor_count == 2
    assert report.checkpoint_files == ("model-00001-of-00001.safetensors", )
    assert report.total_size == 8
    assert index["metadata"] == {"total_size": 8}
    assert index["weight_map"] == {
        "model.language_model.a.weight": report.checkpoint_files[0],
        "model.language_model.z.weight": report.checkpoint_files[0],
    }
    tensors = safetensors_torch.load_file(
        str(output / report.checkpoint_files[0]))
    assert tensors["model.language_model.a.weight"].tolist() == [1.0, 1.0]
    assert tensors["model.language_model.z.weight"].tolist() == [2.0, 2.0]
    assert manifest["schema_version"] == 2
    assert manifest["source"]["revision"] == gemma_schema.GEMMA_REVISION
    assert manifest["source_dtype"] == gemma_schema.SOURCE_DTYPE == "float32"
    assert manifest["target_dtype"] == gemma_schema.TARGET_DTYPE == "bfloat16"
    assert manifest["rounding_policy"] == gemma_schema.ROUNDING_POLICY == (
        "round_to_nearest_even")
    assert manifest["hf_reference"]["revision"] == gemma_schema.HF_REVISIONS[
        gemma_schema.SUPPORTED_VARIANT]
    assert manifest["source_tensor_count"] == 2
    assert manifest["target_tensor_count"] == 2
    assert set(path.name for path in output.iterdir()) == {
        "chat_template.jinja",
        "config.json",
        "conversion.json",
        "generation_config.json",
        "model-00001-of-00001.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    for name in gemma_schema.TOKENIZER_FILES:
        assert (output / name).read_bytes() == (tokenizer / name).read_bytes()


def test_writer_rejects_extra_tokenizer_assets_without_copying(tmp_path):
    """Tokenizer publication copies exactly the reviewed family assets."""
    tokenizer = tmp_path / "tokenizer"
    staging = tmp_path / "staging"
    tokenizer.mkdir()
    staging.mkdir()
    for name in gemma_schema.TOKENIZER_FILES:
        (tokenizer / name).write_text("{}", encoding="utf-8")
    (tokenizer / "unreviewed.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ConversionError, match="unexpected"):
        _writer(tmp_path / "out",
                8).copy_required_tokenizer_files(tokenizer, staging)

    assert not tuple(staging.iterdir())


def test_writer_cleans_staging_and_leaves_destination_unpublished_on_failure(
        conversion_paths, monkeypatch):
    """A metadata failure removes transient files and never exposes partial output."""
    _, tokenizer, output = conversion_paths
    plan = MappingPlan((_small_rule("leaf", "target"), ), "synthetic")
    writer = _writer(output, 8)
    monkeypatch.setattr(
        writer_module, "convert_tensor",
        lambda source, rule: {"target": torch.zeros(2, dtype=torch.bfloat16)})

    def fail(*args, **kwargs):
        raise OSError("metadata failed")

    monkeypatch.setattr(HFCheckpointWriter, "write_checkpoint_metadata", fail)
    with pytest.raises(OSError, match="metadata failed"):
        writer.write(tokenizer, {"leaf": object()}, plan, {})

    assert not output.exists()
    assert not tuple(output.parent.glob(f".{output.name}.*.tmp"))


def test_atomic_publish_replaces_existing_empty_destination(tmp_path):
    """One atomic rename replaces a validated existing empty destination."""
    output = tmp_path / "output"
    output.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "complete.txt").write_text("complete", encoding="utf-8")

    _writer(output, 8).atomic_publish(staging)

    assert not staging.exists()
    assert (output / "complete.txt").read_text(encoding="utf-8") == "complete"


def test_atomic_publish_preserves_empty_destination_when_replace_fails(
        tmp_path, monkeypatch):
    """A failed atomic rename leaves both the empty destination and staging intact."""
    output = tmp_path / "output"
    output.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    payload = staging / "complete.txt"
    payload.write_text("complete", encoding="utf-8")

    def fail_replace(*args):
        raise OSError("rename failed")

    monkeypatch.setattr(writer_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="rename failed"):
        _writer(output, 8).atomic_publish(staging)

    assert output.is_dir()
    assert not tuple(output.iterdir())
    assert payload.read_text(encoding="utf-8") == "complete"


def test_writer_preserves_existing_empty_destination_when_publish_fails(
        conversion_paths, monkeypatch):
    """A failed final rename preserves the empty destination and removes staging."""
    _, tokenizer, output = conversion_paths
    output.mkdir()
    plan = MappingPlan((_small_rule("leaf", "target"), ), "synthetic")
    writer = _writer(output, 8)

    def write_checkpoint(staging, checkpoint_file, source_tree):
        path = staging / checkpoint_file.name
        path.write_bytes(b"complete")
        return writer_module.CheckpointFileReport(checkpoint_file.name,
                                                  "digest", len(b"complete"),
                                                  checkpoint_file.target_bytes)

    monkeypatch.setattr(HFCheckpointWriter, "write_checkpoint_file",
                        lambda self, *args: write_checkpoint(*args))
    monkeypatch.setattr(HFCheckpointWriter, "write_checkpoint_metadata",
                        lambda *args: None)

    def fail_replace(*args):
        raise OSError("rename failed")

    monkeypatch.setattr(writer_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="rename failed"):
        writer.write(tokenizer, {"leaf": object()}, plan, {})

    assert output.is_dir()
    assert not tuple(output.iterdir())
    assert not tuple(output.parent.glob(f".{output.name}.*.tmp"))


def test_generated_checkpoint_metadata_loads_through_edge_model_config(
        tmp_path):
    """Current Edge ModelConfig accepts the converter's generated config and index."""
    from tensorrt_edgellm.config import ModelConfig

    config = Gemma4FlaxAdapter().build_config()
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {
                "total_size": 2
            },
            "weight_map": {
                "model.language_model.layers.0.self_attn.q_norm.weight":
                "model-00001-of-00001.safetensors"
            },
        }),
        encoding="utf-8",
    )

    loaded = ModelConfig.from_pretrained(str(tmp_path), lambda head_dim: 1.0)

    assert loaded.model_type == "gemma4_text"
    assert loaded.num_hidden_layers == 35
    assert loaded.head_dim == 256
    assert loaded.global_head_dim == 512
    assert loaded.hidden_size_per_layer_input == 256
    assert loaded.num_kv_shared_layers == 20
    assert loaded.has_qk_norm is True
