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

import pytest

pytest.importorskip("tensorrt")

from experimental.builder.models.cosmos3.configuration import (
    Cosmos3PolicyGeometry, prepare_root)
from experimental.builder.models.cosmos3.runtime_config import \
    component_runtime_config


def test_policy_droid_metadata_owns_direct_builder_geometry(tmp_path):
    (tmp_path / "checkpoint.json").write_text(
        json.dumps({
            "policy": {
                "action_chunk_size": 32,
                "conditioning_fps": 15.0,
                "domain_name": "droid_lerobot",
            }
        }))

    root = prepare_root(str(tmp_path), {})
    geometry = Cosmos3PolicyGeometry.from_bundle(
        SimpleNamespace(root=root),
        SimpleNamespace(max_input_len=512),
    )

    assert geometry.action_chunk_size == 32
    assert geometry.action_token_count == 33
    assert geometry.conditioning_frames == 1
    assert geometry.content_height == 540
    assert geometry.content_width == 640
    assert geometry.latent_h == 33
    assert geometry.latent_w == 40
    assert geometry.num_frames == 33
    assert geometry.fps == 15.0
    assert geometry.raw_action_dim == 8
    assert geometry.state_rows == 1
    assert geometry.use_state is True


def test_generic_cosmos3_geometry_keeps_legacy_defaults():
    geometry = Cosmos3PolicyGeometry.from_bundle(
        SimpleNamespace(root={}),
        SimpleNamespace(max_input_len=512),
    )

    assert geometry.action_chunk_size == 16
    assert geometry.action_token_count == 16
    assert geometry.conditioning_frames == 17
    assert geometry.num_frames == 17
    assert geometry.raw_action_dim == 10
    assert geometry.use_state is False


@pytest.mark.parametrize(("height", "width"), ((545, 736), (544, 737)))
def test_cosmos3_geometry_rejects_unaligned_vae_canvas(height, width):
    geometry = Cosmos3PolicyGeometry(height=height, width=width)

    with pytest.raises(ValueError, match="VAE downsample factor"):
        geometry.validate()


def test_policy_component_contracts_separate_prefix_and_generation_geometry():
    root = {
        "_direct_policy_config": {
            "action_chunk_size": 32,
            "conditioning_fps": 15.0,
            "domain_name": "droid_lerobot",
        },
        "_direct_transformer_config": {
            "hidden_size": 2560,
            "intermediate_size": 6912,
            "head_dim": 128,
            "num_attention_heads": 20,
            "num_key_value_heads": 4,
            "num_hidden_layers": 2,
            "hidden_act": "silu",
            "rope_theta": 1_000_000.0,
        },
        "_direct_vae_config": {
            "z_dim": 48,
            "patch_size": 2,
            "latents_mean": [0.0] * 48,
            "latents_std": [1.0] * 48,
        },
    }
    bundle = SimpleNamespace(root=root)
    args = SimpleNamespace(max_batch_size=1, max_input_len=512)

    und = component_runtime_config(bundle, "und_prefill", args)
    vae = component_runtime_config(bundle, "vae_encoder", args)
    gen = component_runtime_config(bundle, "gen", args)

    assert "attention_position_ids" in und["optimization_profile"]
    assert "attention_pos_id" not in und["optimization_profile"]
    assert vae["optimization_profile"]["pixel_values"]["opt"] == [
        1, 3, 1, 544, 736
    ]
    assert gen["optimization_profile"]["video_latent"]["opt"] == [
        1, 48, 9, 33, 40
    ]
    assert gen["num_video_tokens"] == 9 * 17 * 20
