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
"""Unit tests for ``_resolve_shard`` path-traversal guard in checkpoint/loader.py."""

import os
import sys

import pytest

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from tensorrt_edgellm.checkpoint.loader import _resolve_shard
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"tensorrt_edgellm not importable: {exc}",
                allow_module_level=True)


def test_resolve_shard_basename(tmp_path):
    shard = tmp_path / "model.safetensors"
    shard.touch()

    result = _resolve_shard(str(tmp_path), shard.name)

    assert result == str(shard)


def test_resolve_shard_subdir_allowed(tmp_path):
    shard = tmp_path / "subfolder" / "model.safetensors"
    shard.parent.mkdir()
    shard.touch()

    result = _resolve_shard(str(tmp_path), "subfolder/model.safetensors")

    assert result == str(shard)


def test_resolve_shard_traversal_rejected(tmp_path):
    with pytest.raises(ValueError, match="escapes model_dir"):
        _resolve_shard(str(tmp_path), "../../../etc/passwd")


def test_resolve_shard_single_dotdot_rejected(tmp_path):
    with pytest.raises(ValueError, match="escapes model_dir"):
        _resolve_shard(str(tmp_path), "../sibling.bin")


def test_resolve_shard_absolute_path_rejected(tmp_path):
    with pytest.raises(ValueError, match="escapes model_dir"):
        _resolve_shard(str(tmp_path), "/etc/passwd")


def test_resolve_shard_returns_str(tmp_path):
    result = _resolve_shard(str(tmp_path), "weights.bin")
    assert isinstance(result, str)


@pytest.mark.parametrize(
    "shard_name",
    ["model-00001-of-00002.safetensors", "pytorch_model-00001-of-00002.bin"])
def test_resolve_shard_hugging_face_cache_symlink(tmp_path, shard_name):
    repository = tmp_path / "models--org--model"
    snapshot = repository / "snapshots" / "revision"
    blobs = repository / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()

    blob = blobs / "abcdef123456"
    blob.touch()
    shard = snapshot / shard_name
    shard.symlink_to(os.path.join("..", "..", "blobs", blob.name))

    result = _resolve_shard(str(snapshot), shard_name)

    assert result == str(shard)


def test_resolve_shard_external_symlink_rejected(tmp_path):
    snapshot = tmp_path / "models--org--model" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    external = tmp_path / "external.safetensors"
    external.touch()
    shard = snapshot / "model.safetensors"
    shard.symlink_to(external)

    with pytest.raises(ValueError, match="escapes model_dir"):
        _resolve_shard(str(snapshot), shard.name)
