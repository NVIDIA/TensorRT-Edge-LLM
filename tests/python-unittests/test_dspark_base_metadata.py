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

import json
from types import SimpleNamespace

import pytest

from tensorrt_edgellm.model import _inherit_dspark_base_metadata


def _base_config():
    return SimpleNamespace(dspark_target_layer_ids=[99],
                           dspark_block_size=7,
                           dspark_mask_token_id=151669,
                           dspark_enable_confidence_head=True,
                           dspark_confidence_head_with_markov=True,
                           dspark_markov_head_type="stale",
                           dspark_markov_rank=256,
                           dspark_sample_from_anchor=True)


def test_dspark_base_inherits_authoritative_nested_draft_contract(tmp_path):
    draft_config = {
        "dspark_config": {
            "target_layer_ids": [1, 5, 19, 29, 41, 51],
            "block_size": 8,
            "mask_token_id": 990,
            "enable_confidence_head": False,
            "confidence_head_with_markov": False,
            "markov_head_type": "vanilla",
            "markov_rank": 512,
            "sample_from_anchor": False,
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(draft_config))
    config = _base_config()

    _inherit_dspark_base_metadata(config, str(tmp_path))

    assert config.dspark_target_layer_ids == [1, 5, 19, 29, 41, 51]
    assert config.dspark_block_size == 8
    assert config.dspark_mask_token_id == 990
    assert config.dspark_enable_confidence_head is False
    assert config.dspark_confidence_head_with_markov is False
    assert config.dspark_markov_head_type == "vanilla"
    assert config.dspark_markov_rank == 512
    assert config.dspark_sample_from_anchor is False


def test_dspark_base_inherits_legacy_top_level_anchor_flag(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({
            "target_layer_ids": [2, 6],
            "block_size": 4,
            "mask_token_id": 7,
            "sample_from_anchor": False,
        }))
    config = _base_config()

    _inherit_dspark_base_metadata(config, str(tmp_path))

    assert config.dspark_target_layer_ids == [2, 6]
    assert config.dspark_block_size == 4
    assert config.dspark_mask_token_id == 7
    assert config.dspark_sample_from_anchor is False
    assert config.dspark_markov_rank == 256


def test_dspark_base_inherits_dflash_fallback_and_gemma4_mask(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_type": "gemma4_text",
            "dflash_config": {
                "target_layer_ids": [3, 9],
                "block_size": 6,
            },
        }))
    config = _base_config()

    _inherit_dspark_base_metadata(config, str(tmp_path))

    assert config.dspark_target_layer_ids == [3, 9]
    assert config.dspark_block_size == 6
    assert config.dspark_mask_token_id == 4


def test_dspark_base_metadata_is_unchanged_without_explicit_draft_dir():
    config = _base_config()
    expected = vars(config).copy()

    _inherit_dspark_base_metadata(config, None)

    assert vars(config) == expected


def test_dspark_base_rejects_explicit_dir_without_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="DSpark draft config"):
        _inherit_dspark_base_metadata(_base_config(), str(tmp_path))
