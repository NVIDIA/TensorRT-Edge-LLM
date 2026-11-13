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
"""Focused contracts for the pi0.5 direct builder."""

import json
from types import SimpleNamespace

import pytest

from tensorrt_edgellm.builder import cli
from tensorrt_edgellm.builder.core import contracts
from tensorrt_edgellm.builder.core.bundle import BundleConfig
from tensorrt_edgellm.builder.models import registry
from tensorrt_edgellm.builder.models.pi05 import (artifacts, configuration,
                                                  runtime_config, weights)
from tensorrt_edgellm.models.pi05 import component_contracts, policy_assets

POLICY_CASES = {
    "pi05_libero": {
        "state_dim": 8,
        "raw_action_dim": 7,
        "horizon": 10,
        "cameras":
        (("observation/image", True), ("observation/wrist_image", True)),
        "opt_views": 2,
        "minimum_prefix": 712,
        "cache_capacity": 1024,
        "pages": 8,
        "mask_words": 1,
    },
    "pi05_droid": {
        "state_dim":
        8,
        "raw_action_dim":
        8,
        "horizon":
        15,
        "cameras": (("observation/exterior_image_1_left", True),
                    ("observation/wrist_image_left", True)),
        "opt_views":
        2,
        "minimum_prefix":
        712,
        "cache_capacity":
        1024,
        "pages":
        8,
        "mask_words":
        1,
    },
    "pi05_aloha": {
        "state_dim":
        14,
        "raw_action_dim":
        14,
        "horizon":
        50,
        "cameras": (("cam_high", True), ("cam_left_wrist", False),
                    ("cam_right_wrist", False)),
        "opt_views":
        3,
        "minimum_prefix":
        968,
        "cache_capacity":
        1280,
        "pages":
        10,
        "mask_words":
        2,
    },
}
POLICY_NAMES = [
    pytest.param(name, id=name.replace("pi05_", "", 1))
    for name in POLICY_CASES
]


def _checkpoint(tmp_path,
                policy_name="pi05_libero",
                *,
                quantization_config=None):
    case = POLICY_CASES[policy_name]
    model_dir = tmp_path / policy_name
    model_dir.mkdir()
    input_features = {
        "observation.state": {
            "type": "STATE",
            "shape": [case["state_dim"]],
        },
    }
    input_features.update({
        name: {
            "type": "VISUAL",
            "shape": [3, 224, 224],
        }
        for name, _ in case["cameras"]
    })
    root = {
        "type": "pi05",
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "max_action_dim": 32,
        "n_action_steps": case["horizon"],
        "input_features": input_features,
        "output_features": {
            "action": {
                "type": "ACTION",
                "shape": [case["raw_action_dim"]],
            }
        },
        "image_resolution": [224, 224],
    }
    if quantization_config is not None:
        root["quantization_config"] = quantization_config
    (model_dir / "config.json").write_text(json.dumps(root), encoding="utf-8")
    return model_dir


def _args(component=contracts.Component.ACTION,
          policy_config="pi05_libero",
          max_batch_size=2,
          max_prefix_len=0):
    return SimpleNamespace(
        resolved_component=component,
        resolved_spec_role=contracts.SpecRole.NONE,
        spec_type="none",
        spec_role="none",
        dflash_version="v1",
        eagle4_draft=False,
        dense_quant="auto",
        externalize_weights=(),
        fp8_embedding=False,
        pi05_policy_config=policy_config,
        pi05_max_prefix_len=max_prefix_len,
        pi05_denoise_steps=0,
        max_batch_size=max_batch_size,
    )


def _file_snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }


def test_pi05_registry_and_runtime_bundle_layout(tmp_path):
    """The pi0.5 registry exposes exactly the four runtime components."""
    bundle = BundleConfig.from_pretrained(str(_checkpoint(tmp_path)))

    assert bundle.components == registry.components_for("pi05") == frozenset({
        contracts.Component.VISUAL,
        contracts.Component.PREFIX,
        contracts.Component.ACTION,
        contracts.Component.COND,
    })
    assert contracts.component_spec(
        contracts.Component.PREFIX).engine_filename == "prefix.engine"
    assert contracts.component_spec(
        contracts.Component.COND).output_dir("/engines") == "/engines/cond"


def test_pi05_rejects_unavailable_component(tmp_path):
    """Component selection rejects engines outside the pi0.5 policy bundle."""
    args = SimpleNamespace(model_dir=str(_checkpoint(tmp_path)),
                           components="llm")

    with pytest.raises(ValueError, match="has no requested component.*llm"):
        cli._resolve_build_selection(args)


@pytest.mark.parametrize("policy_name", POLICY_NAMES)
def test_pi05_geometry_and_runtime_profiles_match_policy(
        tmp_path, policy_name):
    """Every supported policy produces its required action and cache profile."""
    case = POLICY_CASES[policy_name]
    bundle = BundleConfig.from_pretrained(
        str(_checkpoint(tmp_path, policy_name)))
    args = _args(policy_config=policy_name)
    geometry = configuration.Pi05Geometry.from_bundle(bundle, args)

    assert geometry.max_prefix_len == 976
    assert geometry.action_horizon == case["horizon"]
    assert geometry.opt_views == case["opt_views"]
    assert geometry.max_views == policy_assets.MAX_CAMERA_SLOTS
    assert geometry.cache_capacity == case["cache_capacity"]
    assert geometry.pages_per_seq == case["pages"]
    assert component_contracts.packed_mask_words(
        geometry.action_horizon) == case["mask_words"]

    action = runtime_config.component_runtime_config(
        bundle, contracts.Component.ACTION, args)
    assert "onnx_filename" not in action
    assert action["optimization_profile"]["execution_phase_marker"]["max"] == [
        6
    ]
    assert action["optimization_profile"]["context_sequence_count_carrier"][
        "max"] == [0]
    assert action["optimization_profile"]["attention_mask"]["max"] == [
        2 * case["horizon"], case["mask_words"]
    ]
    assert action["optimization_profile"]["kv_cache_layer00"]["max"] == [
        2, 2 * case["pages"], 128, 1, 256
    ]
    assert action["batch_stride"]["kv_cache_layer00"] == case["pages"]


def test_visual_profile_does_not_scale_camera_slots_with_batch(tmp_path):
    """The visual profile counts camera slots independently of request batch."""
    bundle = BundleConfig.from_pretrained(str(_checkpoint(tmp_path)))
    args = _args(contracts.Component.VISUAL, max_batch_size=2)
    config = runtime_config.component_runtime_config(
        bundle, contracts.Component.VISUAL, args)

    assert config["optimization_profile"]["pixel_values"]["max"] == [
        policy_assets.MAX_CAMERA_SLOTS, 3, 224, 224
    ]
    assert config["builder_config"]["max_views"] == 3


@pytest.mark.parametrize("policy_name", ("pi05_libero", "pi05_aloha"))
def test_prefix_capacity_covers_declared_camera_slots(tmp_path, policy_name):
    """Prefix capacity covers the two- and three-camera policy contracts."""
    case = POLICY_CASES[policy_name]
    bundle = BundleConfig.from_pretrained(
        str(_checkpoint(tmp_path, policy_name)))

    with pytest.raises(ValueError, match=f">= {case['minimum_prefix']}"):
        configuration.Pi05Geometry.from_bundle(
            bundle,
            _args(policy_config=policy_name,
                  max_prefix_len=case["minimum_prefix"] - 1))

    geometry = configuration.Pi05Geometry.from_bundle(
        bundle,
        _args(policy_config=policy_name,
              max_prefix_len=case["minimum_prefix"]))
    assert geometry.max_prefix_len == case["minimum_prefix"]


@pytest.mark.parametrize(
    "override,match",
    [
        ({
            "dense": "nvfp4-qdq"
        }, "FP16 dense"),
        ({
            "externalize_weights": ("fp16", )
        }, "external weights"),
        ({
            "fp8_embedding": True
        }, "FP16 embeddings"),
    ],
)
def test_fp16_only_cli_options_are_rejected(override, match):
    """CLI validation rejects every unsupported pi0.5 precision mode."""
    args = _args()
    args.dense = "auto"
    for name, value in override.items():
        setattr(args, name, value)

    with pytest.raises(ValueError, match=match):
        configuration.validate_build(args, (contracts.Component.ACTION, ))


def test_quantized_checkpoint_is_rejected_before_network_creation(tmp_path):
    """Checkpoint validation rejects quantized weights before graph creation."""
    checkpoint = _checkpoint(tmp_path,
                             quantization_config={
                                 "quant_algo": "NVFP4",
                                 "group_size": 16,
                             })
    bundle = BundleConfig.from_pretrained(str(checkpoint))

    with pytest.raises(ValueError, match="unquantized checkpoints"):
        configuration.validate_checkpoint_precision(bundle, _args())


@pytest.mark.parametrize(
    "policy_name,match",
    [
        pytest.param("pi05_libero", "different pi0.5 build", id="identity"),
        pytest.param(
            "pi05_aloha", "do not describe the same policy", id="policy"),
    ],
)
def test_cli_preflight_rejects_mismatch_without_mutating_files(
        monkeypatch, tmp_path, policy_name, match):
    """CLI preflight rejects bundle mismatches without changing existing files."""
    checkpoint = _checkpoint(tmp_path)
    bundle = BundleConfig.from_pretrained(str(checkpoint))
    engine_dir = tmp_path / "engines"
    action_dir = engine_dir / "action"
    action_dir.mkdir(parents=True)
    (action_dir / "action.engine").write_bytes(b"existing-engine")
    (action_dir / "config.json").write_text(json.dumps(
        {"export_id": "old-export"}),
                                            encoding="utf-8")
    (engine_dir / "policy.json").write_text(json.dumps({
        "checkpoint_fingerprint":
        "old-fingerprint",
        "export_id":
        "old-export",
    }),
                                            encoding="utf-8")
    before = _file_snapshot(engine_dir)

    monkeypatch.setattr(artifacts.policy_assets, "checkpoint_fingerprint",
                        lambda _: "new-fingerprint")
    build_args = _args(contracts.Component.ACTION, policy_name)
    monkeypatch.setattr(cli, "_build_args", lambda *_: build_args)
    monkeypatch.setattr(registry, "artifact_writer_for",
                        lambda *_, **__: artifacts)
    plan = (("action", build_args, bundle, contracts.Component.ACTION), )

    with pytest.raises(ValueError, match=match):
        cli._preflight_plan(SimpleNamespace(engine_dir=str(engine_dir)), plan)

    assert _file_snapshot(engine_dir) == before


def test_preflight_rejects_policy_semantics_without_mutating_files(
        monkeypatch, tmp_path):
    """Preflight preserves files when matching builds disagree on policy semantics."""
    checkpoint = _checkpoint(tmp_path)
    bundle = BundleConfig.from_pretrained(str(checkpoint))
    args = _args()
    geometry = configuration.Pi05Geometry.from_bundle(bundle, args)
    monkeypatch.setattr(artifacts.policy_assets, "checkpoint_fingerprint",
                        lambda _: "checkpoint-fingerprint")
    fingerprint, export_id = artifacts._export_identity(bundle, geometry)
    engine_dir = tmp_path / "engines"
    action_dir = engine_dir / "action"
    action_dir.mkdir(parents=True)
    (action_dir / "action.engine").write_bytes(b"existing-engine")
    (action_dir / "config.json").write_text(json.dumps(
        {"export_id": export_id}),
                                            encoding="utf-8")
    contract = policy_assets.build_policy_contract(
        str(checkpoint), {"vocab_size": 257152}, {
            "contract_version": component_contracts.CONTRACT_VERSION,
            "checkpoint_fingerprint": fingerprint,
            "export_id": export_id,
            "components": ["action"],
        }, "pi05_libero")
    contract["adapter"] = "different-adapter"
    (engine_dir / "policy.json").write_text(json.dumps(contract),
                                            encoding="utf-8")
    before = _file_snapshot(engine_dir)

    with pytest.raises(ValueError, match="different pi0.5 policy semantics"):
        artifacts.preflight_bundle(bundle, args, str(engine_dir))

    assert _file_snapshot(engine_dir) == before


def test_missing_norm_stats_warns_without_staging_partial_assets(
        tmp_path, caplog):
    """Missing normalization statistics never leave a partial asset tree."""
    checkpoint = _checkpoint(tmp_path)
    source = checkpoint / "assets"
    source.mkdir()
    (source / "metadata.json").write_text("{}", encoding="utf-8")
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()

    artifacts._stage_assets(SimpleNamespace(model_dir=str(checkpoint)),
                            str(engine_dir))

    assert "has no norm_stats.json" in caplog.text
    assert not (engine_dir / "assets").exists()


def test_asset_staging_rejects_symlink_escape_without_writes(tmp_path):
    """Asset staging rejects links that escape the checkpoint tree."""
    checkpoint = _checkpoint(tmp_path)
    source = checkpoint / "assets"
    source.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"secret": true}', encoding="utf-8")
    (source / "norm_stats.json").symlink_to(outside)
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()

    with pytest.raises(ValueError, match="asset symlink escapes"):
        artifacts._stage_assets(SimpleNamespace(model_dir=str(checkpoint)),
                                str(engine_dir))

    assert not (engine_dir / "assets").exists()


def test_pi05_weight_aliases_cover_native_and_lerobot_wrapped_keys():
    """Weight aliases cover native and LeRobot-wrapped pi0.5 checkpoints."""
    visual = weights.resolve_candidates(
        "vision_tower.vision_model.embeddings.patch_embedding.weight",
        component="visual",
        spec_type="none",
        spec_role="none",
        quant_type="fp16")
    action = weights.resolve_candidates(
        "layers.0.input_layernorm.dense.weight",
        component="action",
        spec_type="none",
        spec_role="none",
        quant_type="fp16")
    embedding = weights.resolve_candidates("embed_tokens.weight",
                                           component="prefix",
                                           spec_type="none",
                                           spec_role="none",
                                           quant_type="fp16")

    assert visual[1] == "model." + visual[0]
    assert action[1] == "model." + action[0]
    assert "paligemma_with_expert.paligemma.lm_head.weight" in embedding


def test_embedding_sidecar_is_staged_at_runtime_bundle_root(
        monkeypatch, tmp_path):
    """Prefix artifacts bind the root embedding sidecar into its manifest."""
    table = SimpleNamespace(shape=(257152, 2048))
    saved = {}

    class _Checkpoint:

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def f16(self, name):
            assert name == "embed_tokens.weight"
            return table

        def close(self):
            pass

    monkeypatch.setattr(artifacts, "Weights", _Checkpoint)
    monkeypatch.setattr(
        artifacts, "save_safetensors",
        lambda path, tensors: saved.update(path=path, tensors=tensors))
    monkeypatch.setattr(artifacts.policy_assets, "sha256_file",
                        lambda path: "digest")
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    (prefix / "config.json").write_text("{}", encoding="utf-8")

    artifacts._write_embedding(SimpleNamespace(model_dir="/checkpoint"),
                               str(tmp_path))

    assert saved["path"] == str(tmp_path / "embed_tokens.safetensors")
    assert saved["tensors"] == {"embed_tokens.weight": table}
    config = json.loads((prefix / "config.json").read_text(encoding="utf-8"))
    assert config["artifacts"]["embed_tokens.safetensors"]["shape"] == [
        257152, 2048
    ]
