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

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from lilicorr_checkpoint_config import LILICORR_CONFIG as _LILICORR_CONFIG

from tensorrt_edgellm.checkpoint.checkpoint_utils import \
    build_runtime_llm_config_dict
from tensorrt_edgellm.config import make_dflash2_draft_config
from tensorrt_edgellm.dflash import DFlashVersion, resolve_dflash_contract

_LILICORR_RUNTIME_DFLASH_CONFIG = {
    "version": 2,
    "target_layer_ids": [1, 18, 35, 51, 68, 85],
    "block_size": 8,
    "mask_token_id": 990,
    "is_causal": False,
    "conv_kernel_size": 2,
    "conv_group_size": 16,
    "supports_probabilistic_sampling": True,
    "projector_type": "lilicorr",
    "lilicorr_candidate_topk": 8,
}


def _lilicorr_config(**dflash_overrides):
    config = copy.deepcopy(_LILICORR_CONFIG)
    config["dflash_config"].update(dflash_overrides)
    return config


def _lilicorr_runtime_model_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_lilicorr_config()))
    config = make_dflash2_draft_config(str(tmp_path),
                                       lambda head_dim: head_dim**-0.5)
    return replace(config,
                   dflash_projector_type="lilicorr",
                   lilicorr_candidate_topk=8)


def test_lilicorr_config_resolves_as_v2():
    """The real LiLiCorr checkpoint config resolves to a V2 LiLiCorr contract."""
    contract = resolve_dflash_contract(_lilicorr_config(), {})

    assert contract.version == DFlashVersion.V2
    assert contract.is_lilicorr
    assert contract.block_size == 8
    assert contract.mask_token_id == 990
    assert contract.target_layer_ids == (1, 18, 35, 51, 68, 85)
    assert contract.is_causal is False
    assert contract.conv_kernel_size == 2
    assert contract.conv_group_size == 16
    assert contract.selector_rank == 0
    assert contract.selector_top_k == 0
    assert contract.lilicorr_candidate_topk == 8
    assert contract.lilicorr_hidden_size == 1024
    assert contract.lilicorr_num_layers == 2
    assert contract.lilicorr_num_heads == 8
    assert contract.lilicorr_mlp_ratio == 2.0
    assert contract.lilicorr_factor_dim == 1024
    assert contract.lilicorr_vector_eps == 1e-4
    assert contract.lilicorr_logit_scale == 8.0
    assert contract.supports_probabilistic_sampling


def test_lilicorr_architecture_implies_projector():
    """LiLiCorrDraftModel without projector_type still resolves as LiLiCorr."""
    config = _lilicorr_config()
    del config["dflash_config"]["projector_type"]

    contract = resolve_dflash_contract(config, {})

    assert contract.version == DFlashVersion.V2
    assert contract.is_lilicorr


def test_lilicorr_projector_on_dflash_v1_architecture_is_v2():
    """projector_type='lilicorr' selects V2 regardless of the architecture."""
    config = _lilicorr_config()
    config["architectures"] = ["DFlashDraftModel"]

    contract = resolve_dflash_contract(config, {})

    assert contract.version == DFlashVersion.V2
    assert contract.is_lilicorr


def test_causal_key_is_an_is_causal_alias():
    """dflash_config.causal=true is honored and rejected for a V2 draft."""
    config = _lilicorr_config(causal=True)

    with pytest.raises(ValueError, match="is_causal=false"):
        resolve_dflash_contract(config, {})


@pytest.mark.parametrize("topk", [0, 6, 32])
def test_lilicorr_rejects_non_power_of_two_topk(topk):
    """candidate_topk must be a power of two no larger than 16."""
    with pytest.raises(ValueError, match="power of two"):
        resolve_dflash_contract(_lilicorr_config(lilicorr_candidate_topk=topk),
                                {})


def test_lilicorr_rejects_disabled_flag():
    """A checkpoint with lilicorr_enabled=false is not a usable LiLiCorr draft."""
    with pytest.raises(ValueError, match="lilicorr_enabled"):
        resolve_dflash_contract(_lilicorr_config(lilicorr_enabled=False), {})


def test_lilicorr_requires_enabled_flag():
    """lilicorr_enabled is a required checkpoint field, not defaulted to true."""
    config = _lilicorr_config()
    del config["dflash_config"]["lilicorr_enabled"]

    with pytest.raises(ValueError, match="lilicorr_enabled"):
        resolve_dflash_contract(config, {})


@pytest.mark.parametrize("name", [
    "lilicorr_candidate_topk", "lilicorr_hidden_size", "lilicorr_num_layers",
    "lilicorr_num_heads", "lilicorr_mlp_ratio", "lilicorr_factor_dim",
    "lilicorr_vector_eps", "lilicorr_logit_scale"
])
def test_lilicorr_requires_head_hyperparameters(name):
    """Every LiLiCorr head hyperparameter must come from the checkpoint."""
    config = _lilicorr_config()
    del config["dflash_config"][name]

    with pytest.raises(ValueError, match=name):
        resolve_dflash_contract(config, {})


@pytest.mark.parametrize(("name", "bad_value"), [
    ("lilicorr_hidden_size", 0),
    ("lilicorr_num_layers", 0),
    ("lilicorr_num_heads", 0),
    ("lilicorr_mlp_ratio", 0.0),
    ("lilicorr_factor_dim", -4),
    ("lilicorr_vector_eps", 0.0),
    ("lilicorr_logit_scale", -8.0),
])
def test_lilicorr_rejects_non_positive_head_hyperparameters(name, bad_value):
    """Head sizes, eps and logit scale must be positive."""
    with pytest.raises(ValueError, match=name):
        resolve_dflash_contract(_lilicorr_config(**{name: bad_value}), {})


def test_lilicorr_rejects_empty_target_layers():
    """LiLiCorr needs at least one target layer."""
    with pytest.raises(ValueError, match="target-layer"):
        resolve_dflash_contract(_lilicorr_config(target_layer_ids=[]), {})


def test_lilicorr_rejects_duplicate_target_layers():
    """LiLiCorr target-layer IDs stay unique."""
    with pytest.raises(ValueError, match="unique"):
        resolve_dflash_contract(
            _lilicorr_config(target_layer_ids=[1, 18, 18, 51, 68, 85]), {})


@pytest.mark.parametrize(("field", "bad_value"), [
    ("conv_kernel_size", 3),
    ("conv_group_size", 8),
    ("block_size", 17),
])
def test_lilicorr_keeps_dflash2_geometry_checks(field, bad_value):
    """LiLiCorr keeps the DFlash2 conv 2/16 and block [2, 16] contract."""
    with pytest.raises(ValueError, match="DFlash V2"):
        resolve_dflash_contract(_lilicorr_config(**{field: bad_value}), {})


def test_unknown_projector_still_rejected():
    """Unknown projector types stay rejected beside domino and lilicorr."""
    with pytest.raises(ValueError, match="projector_type 'foo'"):
        resolve_dflash_contract(_lilicorr_config(projector_type="foo"), {})


def test_lilicorr_runtime_config_omits_selector_fields(tmp_path):
    """The LiLiCorr draft runtime dflash_config is exactly the dict the C++ runtime parses."""
    config = _lilicorr_runtime_model_config(tmp_path)

    runtime = build_runtime_llm_config_dict(SimpleNamespace(config=config))

    assert runtime["spec_decode_type"] == "dflash"
    assert runtime["engine_role"] == "draft"
    assert runtime["dflash_config"] == _LILICORR_RUNTIME_DFLASH_CONFIG


def test_lilicorr_base_runtime_config_matches_draft(tmp_path):
    """The LiLiCorr base runtime dflash_config is exactly the dict the C++ runtime parses."""
    config = replace(_lilicorr_runtime_model_config(tmp_path),
                     is_dflash_draft_flag=False,
                     dflash_base=True)

    runtime = build_runtime_llm_config_dict(SimpleNamespace(config=config))

    assert runtime["spec_decode_type"] == "dflash"
    assert runtime["engine_role"] == "base"
    assert runtime["dflash_config"] == _LILICORR_RUNTIME_DFLASH_CONFIG
