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
"""Unit tests for checkpoint path and tensor sharding helpers."""

import os
import sys

import pytest

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import torch

    from tensorrt_edgellm.checkpoint.loader import (
        _load_segmented_weight_shard, _resolve_shard)
    from tensorrt_edgellm.config import Mapping
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"tensorrt_edgellm not importable: {exc}",
                allow_module_level=True)


def test_resolve_shard_basename(tmp_path):
    result = _resolve_shard(str(tmp_path), "model.safetensors")
    assert result == str(tmp_path / "model.safetensors")


def test_resolve_shard_subdir_allowed(tmp_path):
    result = _resolve_shard(str(tmp_path), "subfolder/model.safetensors")
    assert result == str(tmp_path / "subfolder" / "model.safetensors")


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
    "rank,expected",
    [
        (0, [0, 1, 4, 5, 8]),
        (1, [2, 3, 6, 7, 9]),
    ],
)
def test_segmented_weight_shard_preserves_each_logical_segment(rank, expected):
    tensor = torch.arange(10)
    mapping = Mapping(world_size=2, rank=rank, tp_size=2, tp_rank=rank)

    shard = _load_segmented_weight_shard(tensor, 0, (2, 2, 1), mapping)

    assert shard.tolist() == expected


def test_segmented_weight_shard_rejects_incompatible_shape():
    mapping = Mapping(world_size=2, rank=0, tp_size=2, tp_rank=0)
    with pytest.raises(ValueError, match="does not match"):
        _load_segmented_weight_shard(torch.arange(9), 0, (2, 2, 1), mapping)
