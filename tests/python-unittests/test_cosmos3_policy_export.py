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
"""Contracts for Cosmos3 policy export artifacts."""

import json
import os
import stat
from pathlib import Path

import torch

from tensorrt_edgellm.models.cosmos3 import export
from tensorrt_edgellm.models.cosmos3.modeling_gen import (
    Cosmos3Gen, Cosmos3GenConfig, gen_config_from_transformer, make_gen_config)
from tensorrt_edgellm.models.cosmos3.modeling_und_prefill import (
    Cosmos3UndPrefill, make_und_prefill_config)
from tensorrt_edgellm.models.cosmos3.modeling_vae_encoder import \
    make_vae_encoder_config


def test_embedding_sidecars_are_readable_by_device_jobs(tmp_path, monkeypatch):
    """Make policy sidecars readable when export and device jobs use different UIDs."""
    from tensorrt_edgellm import _safetensors_io

    def save_owner_only(_tensors, path, metadata=None):
        path = Path(path)
        path.write_bytes(b"sidecar")
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    monkeypatch.setattr(_safetensors_io, "_save_file", save_owner_only)
    weights = {
        "embed_tokens.weight": torch.ones(1),
        "lm_head.weight": torch.ones(1),
    }

    export._save_embedding_artifacts(tmp_path, weights, torch.float16)

    for filename in ("embed_tokens.safetensors", "lm_head.safetensors"):
        mode = os.stat(tmp_path / filename).st_mode
        assert mode & stat.S_IRGRP
        assert mode & stat.S_IROTH


def test_policy_export_uses_droid_checkpoint_geometry(tmp_path, monkeypatch):
    """Use released DROID metadata for every exported policy component."""
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.json").write_text(
        json.dumps({
            "policy": {
                "action_chunk_size": 32,
                "conditioning_fps": 15.0,
                "domain_name": "droid_lerobot",
            }
        }))
    captured = {}

    monkeypatch.setattr(export, "_write_tokenizer_artifacts",
                        lambda *_args: None)
    monkeypatch.setattr(
        export, "export_gen", lambda _checkpoint, _out_dir, _dtype, geometry,
        **_kwargs: captured.setdefault("gen", geometry))
    monkeypatch.setattr(
        export, "export_und_prefill",
        lambda _checkpoint, _out_dir, _dtype, max_und_len: captured.setdefault(
            "max_und_len", max_und_len))
    monkeypatch.setattr(
        export, "export_vae_encoder",
        lambda _checkpoint, _out_dir, _dtype, geometry: captured.setdefault(
            "vae", geometry))

    export.export_cosmos3_components(str(checkpoint), str(tmp_path / "output"))

    geometry = captured["gen"]
    assert captured["vae"] is geometry
    assert captured["max_und_len"] == 512
    assert geometry.action_chunk_size == 32
    assert geometry.action_token_count == 33
    assert geometry.conditioning_frames == 1
    assert geometry.num_frames == 33
    assert geometry.content_height == 540
    assert geometry.content_width == 640
    assert geometry.fps == 15.0
    assert geometry.raw_action_dim == 8
    assert geometry.use_state is True


def test_droid_component_configs_preserve_state_and_content_geometry():
    """Encode DROID state tokens and cropped image geometry in runtime configs."""
    transformer_config = {
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "intermediate_size": 256,
        "rms_norm_eps": 1e-6,
        "hidden_act": "relu2",
        "rope_theta": 1_000_000.0,
        "rope_scaling": {
            "mrope_section": [16, 24, 24],
        },
    }
    gen = gen_config_from_transformer(transformer_config,
                                      action_chunk_size=32,
                                      num_frames=33,
                                      raw_action_dim=8,
                                      state_rows=1,
                                      latent_height=33,
                                      latent_width=40)
    config = make_gen_config(gen, transformer_config, max_und_len=512, fps=15)
    vae = make_vae_encoder_config(544, 736, 1, 540, 640)

    assert config["optimization_profile"]["video_latent"]["opt"] == [
        1, 48, 9, 33, 40
    ]
    assert config["optimization_profile"]["action_latent"]["opt"] == [
        1, 33, 64
    ]
    assert config["action_chunk_size"] == 32
    assert config["action_token_count"] == 33
    assert config["raw_action_dim"] == 8
    assert config["state_rows"] == 1
    assert config["history_length"] == 1
    assert config["use_state"] is True
    assert config["action_start_frame_offset"] == 0
    assert config["fps"] == 15
    assert vae["optimization_profile"]["pixel_values"]["opt"] == [
        1, 3, 1, 544, 736
    ]
    assert vae["builder_config"]["content_height"] == 540
    assert vae["builder_config"]["content_width"] == 640


def test_onnx_components_name_attention_positions_like_runtime():
    """Name the GEN and UND-prefill position input as the runtime binds it."""
    binding = "attention_position_ids"
    gen = Cosmos3GenConfig(hidden_size=8,
                           num_hidden_layers=0,
                           num_attention_heads=1,
                           num_key_value_heads=1,
                           head_dim=8,
                           intermediate_size=16)
    transformer = {
        "rope_theta": 1.0,
        "rope_scaling": {
            "mrope_section": [1, 1, 2]
        },
    }
    und = {
        **transformer,
        "hidden_size": 8,
        "num_hidden_layers": 0,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "rms_norm_eps": 1e-6,
    }

    _, gen_inputs, _, _ = Cosmos3Gen(gen).get_onnx_export_args(16, "cpu")
    _, und_inputs, _, _ = Cosmos3UndPrefill(und).get_onnx_export_args()
    gen_config = make_gen_config(gen, transformer, max_und_len=16)
    und_config = make_und_prefill_config(und, max_und_len=16)

    assert binding in gen_inputs
    assert binding in und_inputs
    assert binding in gen_config["optimization_profile"]
    assert binding in und_config["optimization_profile"]
    assert binding in und_config["tensor_contract"]["inputs"]


def test_und_prefill_masks_future_tokens_explicitly(monkeypatch):
    """UND attention takes an explicit causal mask, not TensorRT's is_causal."""
    from tensorrt_edgellm.models import ops
    calls = []

    def attention(query, key, value, attn_mask, is_causal, scale):
        calls.append((attn_mask, is_causal))
        return query

    monkeypatch.setattr(ops, "attention_onnx", attention)
    monkeypatch.setattr(ops, "rope_onnx", lambda x, cos, sin, pos: x)
    und = Cosmos3UndPrefill({
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "intermediate_size": 16,
        "hidden_act": "relu2",
        "rms_norm_eps": 1e-6,
    })
    positions = torch.arange(3, dtype=torch.int32)[None]

    und(torch.zeros(1, 3, 8), torch.zeros(1, 3, 8), positions)

    (mask, is_causal), = calls
    assert not is_causal
    expected = torch.zeros(3, 3).masked_fill(
        torch.ones(3, 3, dtype=torch.bool).triu(1), float("-inf"))
    torch.testing.assert_close(mask[0, 0].float(), expected)


def test_gen_patchify_preserves_odd_droid_latent_grid():
    """Pad and crop the DROID 33x40 latent grid without losing values."""
    config = Cosmos3GenConfig(hidden_size=8,
                              num_hidden_layers=0,
                              num_attention_heads=1,
                              num_key_value_heads=1,
                              head_dim=8,
                              intermediate_size=16,
                              latent_channel=2,
                              max_action_dim=4,
                              latent_t=9,
                              latent_h=33,
                              latent_w=40)
    model = Cosmos3Gen(config)
    latent = torch.arange(2 * 2 * 9 * 33 * 40,
                          dtype=torch.float32).reshape(2, 2, 9, 33, 40)

    tokens = model._patchify(latent)
    restored = model._unpatchify(tokens, 9)

    assert tokens.shape == (2, 9 * 17 * 20, 8)
    torch.testing.assert_close(restored, latent)

    class RoundTrip(torch.nn.Module):

        def __init__(self, gen):
            super().__init__()
            self.gen = gen

        def forward(self, video_latent):
            return self.gen._unpatchify(self.gen._patchify(video_latent),
                                        video_latent.shape[2])

    batch = torch.export.Dim("batch", min=1, max=256)
    latent_t = torch.export.Dim("latent_t", min=1, max=256)
    program = torch.export.export(RoundTrip(model), (latent, ),
                                  dynamic_shapes=({
                                      0: batch,
                                      2: latent_t
                                  }, ),
                                  strict=False)
    torch.testing.assert_close(program.module()(latent), latent)


def test_policy_geometry_uses_checkpoint_domain_id(tmp_path):
    """Select the checkpoint's action domain unless the caller overrides it."""
    from types import SimpleNamespace

    from tensorrt_edgellm.models.cosmos3.policy_geometry import \
        Cosmos3PolicyGeometry

    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(
        json.dumps({"domain_id": 3}))
    bundle = SimpleNamespace(
        root={"_direct_transformer_config": {
            "domain_id": 3
        }})

    assert Cosmos3PolicyGeometry.from_checkpoint(tmp_path).domain_id == 3
    assert Cosmos3PolicyGeometry.from_checkpoint(tmp_path).domain_name is None
    assert Cosmos3PolicyGeometry.from_bundle(bundle).domain_id == 3
    assert Cosmos3PolicyGeometry.from_checkpoint(tmp_path, {
        "domain_id": 5
    }).domain_id == 5
