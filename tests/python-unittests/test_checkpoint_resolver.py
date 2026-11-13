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
"""Contracts for resolving local and Hugging Face checkpoints."""

import json
import sys
from types import SimpleNamespace

import pytest
from conftest import EnvironmentConfig
from defs import checkpoint_resolver
from defs.config import ModelType, TaskType
from defs.config import TestConfig as _TestConfig
from defs.config import _hf_download_settings
from defs.model_test_config import ModelTest


def _checkpoint(directory):
    """Create the minimum files recognized as a local checkpoint."""
    directory.mkdir(parents=True)
    (directory / "config.json").touch()
    (directory / "model.safetensors").touch()
    return directory


def test_nested_indexed_checkpoint_requires_every_weight_file(tmp_path):
    """Recognize a complete nested Diffusers checkpoint, but not a partial one."""
    checkpoint = tmp_path / "cosmos-policy"
    transformer = checkpoint / "transformer"
    transformer.mkdir(parents=True)
    (checkpoint / "config.json").touch()
    (checkpoint / "model_index.json").touch()
    weight = transformer / "model.safetensors"
    weight.touch()
    index = {
        "weight_map": {
            "transformer.layer.weight": "transformer/model.safetensors"
        }
    }
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps(index),
                                                             encoding="utf-8")

    assert checkpoint_resolver.is_checkpoint_dir(checkpoint)

    weight.unlink()
    assert not checkpoint_resolver.is_checkpoint_dir(checkpoint)


def _public_environment(tmp_path):
    """Create a test environment without a managed model root."""
    return EnvironmentConfig(
        llm_sdk_dir=str(tmp_path / "source"),
        llm_models_dir=None,
        edgellm_data_dir=None,
        onnx_dir=str(tmp_path / "onnx"),
        engine_dir=str(tmp_path / "engines"),
        build_dir=str(tmp_path / "build"),
        test_log_dir=str(tmp_path / "logs"),
        trt_package_dir=str(tmp_path / "tensorrt"),
    )


def _public_model_test(task_type):
    """Create a model case whose checkpoint is a Hugging Face ID."""
    parameters = {
        "model_name": "Qwen3-0.6B",
        "checkpoint_dir": "Qwen/Qwen3-0.6B",
        "llm_precision": "fp16",
    }
    if task_type == TaskType.CHECKPOINT_BUILD:
        parameters.update({
            "max_kv_cache_capacity": 1024,
            "max_input_len": 1024,
            "max_batch_size": 1,
            "test_case": "llm_basic",
        })
    return ModelTest(
        node="unused",
        platform="l0_e2e_a30",
        case_id="Qwen3-0.6B/public",
        model_type="llm",
        model_parameters=parameters,
        requirements={},
    )


@pytest.fixture
def empty_inventory(monkeypatch):
    """Remove managed checkpoint aliases for public-resolution tests."""
    monkeypatch.setattr(checkpoint_resolver, "CHECKPOINTS_BY_ID", {})
    monkeypatch.setattr(checkpoint_resolver, "DIRECT_CHECKPOINT_ID_BY_ALIAS",
                        {})
    monkeypatch.setattr(checkpoint_resolver,
                        "QUANTIZED_CHECKPOINT_ID_BY_ALIAS", {})


@pytest.fixture
def inventory(monkeypatch):
    """Build resolver registries from test-owned checkpoint declarations."""

    def configure(checkpoints, private_ids=()):
        monkeypatch.setattr(checkpoint_resolver, "CHECKPOINTS_BY_ID",
                            checkpoints)
        monkeypatch.setattr(checkpoint_resolver, "PRIVATE_CHECKPOINT_IDS",
                            frozenset(private_ids))
        monkeypatch.setattr(
            checkpoint_resolver, "DIRECT_CHECKPOINT_ID_BY_ALIAS",
            checkpoint_resolver._build_alias_index(quantized=False))
        monkeypatch.setattr(
            checkpoint_resolver, "QUANTIZED_CHECKPOINT_ID_BY_ALIAS",
            checkpoint_resolver._build_alias_index(quantized=True))

    return configure


def test_public_variant_can_use_private_quantized_mirror(tmp_path, inventory):
    """A public quantized variant can resolve from a private local mirror."""
    inventory({
        "example/model": {
            "quantized": {
                "name": "model",
                "quantizations": ("NVFP4", )
            },
        },
        "example/model-NVFP4": {},
    })
    checkpoint = _checkpoint(tmp_path / "private" / "model-NVFP4")

    resolved = checkpoint_resolver.resolve_checkpoint(
        "model-NVFP4", torch_roots=(), quantized_root=str(tmp_path))

    assert resolved == str(checkpoint)


def test_quantized_variant_can_use_declared_managed_path(tmp_path, inventory):
    """A quantized variant resolves from its declared managed location."""
    inventory({
        "example/model": {
            "quantized": {
                "name": "model",
                "quantizations": ("NVFP4", ),
                "paths": {
                    "NVFP4": ("models/quantized-model", )
                },
            },
        },
    })
    checkpoint = _checkpoint(tmp_path / "models" / "quantized-model")

    resolved = checkpoint_resolver.resolve_checkpoint(
        "model-NVFP4",
        torch_roots=(),
        quantized_root=str(tmp_path / "quantized"),
        managed_roots=(str(tmp_path), ),
    )

    assert resolved == str(checkpoint)


def test_private_checkpoint_never_falls_back_to_hugging_face(
        tmp_path, monkeypatch, inventory):
    """A missing private checkpoint is refused without requesting a download."""
    inventory({"private/model": {}}, private_ids=("private/model", ))

    def fail_download(*args, **kwargs):
        pytest.fail("private checkpoint attempted a Hugging Face download")

    monkeypatch.setattr(checkpoint_resolver, "_download_from_hf",
                        fail_download)
    with pytest.raises(ValueError, match="Private CI checkpoint"):
        checkpoint_resolver.resolve_checkpoint(
            "private/model",
            torch_roots=(),
            allow_download=True,
            download_root=str(tmp_path / "downloads"),
        )


def test_absent_inventory_leaves_the_registry_empty(tmp_path, monkeypatch):
    """An absent optional inventory leaves checkpoint registries empty."""
    monkeypatch.setattr(checkpoint_resolver, "_INVENTORY_PATH",
                        tmp_path / "checkpoint_inventory.py")

    assert checkpoint_resolver._load_inventory() == ({}, frozenset())


def test_unreadable_inventory_is_still_an_error(tmp_path, monkeypatch):
    """An existing but unreadable inventory fails during registry loading."""
    inventory = tmp_path / "checkpoint_inventory.py"
    inventory.write_text("raise ImportError('broken')\n", encoding="utf-8")
    monkeypatch.setattr(checkpoint_resolver, "_INVENTORY_PATH", inventory)

    with pytest.raises(ImportError):
        checkpoint_resolver._load_inventory()


def test_hugging_face_id_resolves_without_an_inventory(tmp_path,
                                                       empty_inventory):
    """A full Hugging Face ID resolves from a supplied local search root."""
    _checkpoint(tmp_path / "Qwen" / "Qwen3-0.6B")

    resolved = checkpoint_resolver.resolve_checkpoint(
        "Qwen/Qwen3-0.6B", torch_roots=(str(tmp_path), ))

    assert resolved == str(tmp_path / "Qwen" / "Qwen3-0.6B")


def test_hugging_face_id_uses_the_standard_cache_unless_a_root_is_specified(
        tmp_path, monkeypatch, empty_inventory):
    """A full model ID uses the standard cache unless given a download root."""
    monkeypatch.delenv("HF_CHECKPOINT_DOWNLOAD_DIR", raising=False)
    monkeypatch.setenv("EDGELLM_DATA_DIR", str(tmp_path / "edge-llm-data"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    calls = []

    cached = _checkpoint(tmp_path / "hf-home" / "hub" / "snapshots" / "abc")

    def snapshot_download(*, repo_id, local_files_only=False):
        """Record the cache policy used by checkpoint resolution."""
        calls.append((repo_id, local_files_only))
        return str(cached)

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        SimpleNamespace(snapshot_download=snapshot_download))
    allow_download, download_root = _hf_download_settings()

    resolved = checkpoint_resolver.resolve_checkpoint(
        "Qwen/Qwen3-0.6B",
        torch_roots=(),
        allow_download=allow_download,
        download_root=download_root,
    )

    assert allow_download is True
    assert download_root is None
    assert resolved == str(cached)
    assert calls == [("Qwen/Qwen3-0.6B", False)]


def test_downloads_disabled_still_uses_the_standard_hf_cache(
        tmp_path, monkeypatch, empty_inventory):
    """Disabling downloads still permits a standard-cache snapshot."""
    monkeypatch.delenv("HF_CHECKPOINT_DOWNLOAD_DIR", raising=False)
    monkeypatch.delenv("EDGELLM_DATA_DIR", raising=False)
    monkeypatch.setenv("EDGE_LLM_ALLOW_HF_DOWNLOAD", "0")
    cached = _checkpoint(tmp_path / "hf-home" / "hub" / "snapshots" / "abc")
    calls = []

    def snapshot_download(*, repo_id, local_files_only=False):
        """Record that cache-only resolution was requested."""
        calls.append((repo_id, local_files_only))
        return str(cached)

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        SimpleNamespace(snapshot_download=snapshot_download))
    allow_download, download_root = _hf_download_settings()

    resolved = checkpoint_resolver.resolve_checkpoint(
        "Qwen/Qwen3-0.6B",
        torch_roots=(),
        allow_download=allow_download,
        download_root=download_root,
    )

    assert allow_download is False
    assert download_root is None
    assert resolved == str(cached)
    assert calls == [("Qwen/Qwen3-0.6B", True)]


@pytest.mark.parametrize("task_type",
                         [TaskType.EXPORT, TaskType.CHECKPOINT_BUILD])
def test_explicit_model_id_does_not_require_a_model_root(tmp_path, task_type):
    """An explicit model ID satisfies export and direct-build validation."""
    config = _TestConfig.from_param_string(
        _public_model_test(task_type),
        ModelType.LLM,
        task_type,
        _public_environment(tmp_path),
    )

    assert config.checkpoint_dir == "Qwen/Qwen3-0.6B"
    assert config.llm_models_dir is None


def test_legacy_case_without_model_id_still_requires_a_model_root(tmp_path):
    """A legacy encoded case still requires LLM_MODELS_DIR."""
    with pytest.raises(ValueError, match="LLM_MODELS_DIR"):
        _TestConfig.from_param_string(
            "Qwen3-0.6B-fp16",
            ModelType.LLM,
            TaskType.CHECKPOINT_BUILD,
            _public_environment(tmp_path),
        )


def test_explicit_download_root_must_remain_absolute(monkeypatch):
    """A configured checkpoint download root must be absolute."""
    monkeypatch.setenv("HF_CHECKPOINT_DOWNLOAD_DIR", "relative/checkpoints")

    allow_download, download_root = _hf_download_settings()

    assert allow_download is True
    with pytest.raises(ValueError, match="must be absolute"):
        checkpoint_resolver.resolve_checkpoint(
            "Qwen/Qwen3-0.6B",
            torch_roots=(),
            allow_download=allow_download,
            download_root=download_root,
        )


def test_unregistered_name_without_an_inventory_is_refused(
        tmp_path, empty_inventory):
    """A bare unregistered name is neither inferred nor downloaded."""

    with pytest.raises(ValueError, match="not registered"):
        checkpoint_resolver.resolve_checkpoint("Qwen3-0.6B",
                                               torch_roots=(str(tmp_path), ))
