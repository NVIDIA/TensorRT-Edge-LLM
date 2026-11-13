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
"""Focused contracts for the GR00T N1.7 direct builder."""

import json
import struct
from types import SimpleNamespace

import pytest

from tensorrt_edgellm.builder import cli
from tensorrt_edgellm.builder.core import contracts
from tensorrt_edgellm.builder.core.bundle import BundleConfig
from tensorrt_edgellm.builder.models import registry
from tensorrt_edgellm.builder.models.gr00t_n1d7 import (artifacts,
                                                        configuration,
                                                        runtime_config,
                                                        weights)
from tensorrt_edgellm.models.gr00t_n1d7 import configuration as facts
from tensorrt_edgellm.models.gr00t_n1d7 import identity, policy_contract

LM = facts.LM_PREFIX
VISUAL = "backbone.model.model.visual."
HORIZON = 16
STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
STATE_DIMS = (1, 1, 1, 1, 1, 1, 2)


def _header_shapes():
    """Tensor shapes validate_backbone() and the geometry read from the shard header."""
    shapes = {
        VISUAL + "patch_embed.proj.weight": (1024, 3, 2, 16, 16),
        VISUAL + "pos_embed.weight": (2304, 1024),
        VISUAL + "blocks.0.attn.qkv.weight": (3072, 1024),
        VISUAL + "blocks.0.mlp.linear_fc1.weight": (4096, 1024),
        VISUAL + "merger.linear_fc2.weight": (2048, 4096),
        LM + "0.self_attn.k_proj.weight": (1024, 2048),
        LM + "0.self_attn.q_norm.weight": (128, ),
        LM + "0.mlp.gate_proj.weight": (6144, 2048),
        "backbone.model.model.language_model.embed_tokens.weight":
        (151936, 2048),
        facts.DIT_INNER_DIM_KEY: (1536, 1536),
    }
    for block in range(facts.VISION_CONFIG["depth"]):
        shapes[VISUAL + f"blocks.{block}.norm1.weight"] = (1024, )
    for merger in range(len(facts.VISION_CONFIG["deepstack_visual_indexes"])):
        shapes[VISUAL +
               f"deepstack_merger_list.{merger}.linear_fc1.weight"] = (4096,
                                                                       4096)
    for layer in range(16):
        shapes[LM + f"{layer}.self_attn.q_proj.weight"] = (2048, 2048)
    return shapes


def _write_header_only_shard(path, shapes):
    """A safetensors header with the real shapes and no data: shape checks and the
    fingerprint read headers only, and the tests never load a tensor."""
    header, offset = {}, 0
    for name, shape in sorted(shapes.items()):
        size = 2
        for dim in shape:
            size *= dim
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)


def _statistics():
    stats = {}
    for key, dim in zip(STATE_KEYS, STATE_DIMS):
        stats[key] = {
            "min": [-1.0] * dim,
            "max": [1.0] * dim,
            "q01": [-0.5] * dim,
            "q99": [0.5] * dim,
        }
    action = {key: dict(stats[key]) for key in STATE_KEYS}
    action["gripper"] = {
        "min": [-1.0],
        "max": [1.0],
        "q01": [-1.0],
        "q99": [1.0]
    }
    return {"libero_sim": {"state": stats, "action": action}}


def _checkpoint(tmp_path, *, name="libero_10", tokenizer=True):
    model_dir = tmp_path / name
    model_dir.mkdir()
    root = {
        "model_type": facts.MODEL_TYPE,
        "architectures": [facts.MODEL_TYPE],
        "action_horizon": 40,
        "max_action_dim": 132,
        "max_state_dim": 132,
        "num_inference_timesteps": 4,
        "num_timestep_buckets": 1000,
        "backbone_embedding_dim": 2048,
        "select_layer": 16,
        "diffusion_model_cfg": {
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "num_layers": 32,
            "output_dim": 1024,
        },
        "vl_self_attention_cfg": {
            "num_layers": 4,
            "num_attention_heads": 32,
            "attention_head_dim": 64,
        },
        "use_alternate_vl_dit": True,
        "use_percentiles": True,
        "use_albumentations": True,
        "letter_box_transform": False,
        "shortest_image_edge": 256,
        "image_target_size": [256, 256],
        "crop_fraction": 0.95,
        "formalize_language": True,
    }
    (model_dir / "config.json").write_text(json.dumps(root), encoding="utf-8")
    (model_dir / "embodiment_id.json").write_text(json.dumps({
        "libero_sim": 2,
        "unknown": 22
    }),
                                                  encoding="utf-8")
    absolute = {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT"}
    modality = {
        "video": {
            "delta_indices": [0],
            "modality_keys": ["image", "wrist_image"]
        },
        "state": {
            "delta_indices": [0],
            "modality_keys": list(STATE_KEYS)
        },
        "action": {
            "delta_indices": list(range(HORIZON)),
            "modality_keys": list(STATE_KEYS),
            "action_configs": [dict(absolute) for _ in STATE_KEYS],
        },
    }
    (model_dir / "processor_config.json").write_text(json.dumps(
        {"processor_kwargs": {
            "modality_configs": {
                "libero_sim": modality
            }
        }}),
                                                     encoding="utf-8")
    (model_dir / "statistics.json").write_text(json.dumps(_statistics()),
                                               encoding="utf-8")
    shard = "model-00001-of-00001.safetensors"
    shapes = _header_shapes()
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {
            name: shard
            for name in shapes
        }}),
                                                            encoding="utf-8")
    _write_header_only_shard(model_dir / shard, shapes)
    if tokenizer:
        source = model_dir / "tokenizer"
        source.mkdir()
        (source / "tokenizer.json").write_text("{}", encoding="utf-8")
        (source / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (source / "chat_template.json").write_text(json.dumps(
            {"chat_template": "{{ messages[0].content }}"}),
                                                   encoding="utf-8")
    return model_dir


def _args(model_dir, engine_dir, **overrides):
    values = {
        "model_dir": str(model_dir),
        "engine_dir": str(engine_dir),
        "components": "all",
        "spec_type": "none",
        "spec_role": "none",
        "tp_size": 1,
        "max_input_len": 1024,
        "max_batch_size": 1,
        "gr00t_embodiment_id": None,
        "gr00t_max_image_len": None,
        "min_image_tokens": 4,
        "max_image_tokens": 1024,
        "max_image_tokens_per_image": 512,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _file_snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }


def _stamped_bundle(tmp_path, bundle, args, *, export_id=None, frontend=None):
    """An engine directory holding every component of ``bundle``'s build, optionally
    carrying another identity."""
    engine_dir = tmp_path / "engines"
    stamp = dict(runtime_config.bundle_identity(bundle, args))
    if export_id is not None:
        stamp["export_id"] = export_id
    if frontend is not None:
        stamp["frontend"] = frontend
    for component in identity.COMPONENTS:
        directory = engine_dir / component
        directory.mkdir(parents=True)
        (directory / f"{component}.engine").write_bytes(b"existing-engine")
        (directory / "config.json").write_text(json.dumps(stamp),
                                               encoding="utf-8")
    (engine_dir / "action" / "policy.json").write_text(json.dumps(
        identity.stamp_contract({}, stamp)),
                                                       encoding="utf-8")
    return engine_dir


def test_gr00t_registry_and_runtime_bundle_layout(tmp_path):
    """The GR00T registry exposes exactly the three policy components at their bundle paths."""
    bundle = BundleConfig.from_pretrained(str(_checkpoint(tmp_path)))

    assert bundle.components == registry.components_for(
        facts.MODEL_TYPE) == frozenset({
            contracts.Component.VISUAL,
            contracts.Component.VL_BACKBONE,
            contracts.Component.ACTION,
        })
    spec = contracts.component_spec(contracts.Component.VL_BACKBONE)
    assert spec.engine_filename == "vl_backbone.engine"
    assert spec.output_dir("/engines") == "/engines/vl_backbone"


def test_gr00t_rejects_unavailable_component(tmp_path):
    """Component selection rejects engines outside the GR00T policy bundle."""
    args = _args(_checkpoint(tmp_path), tmp_path / "engines", components="llm")

    with pytest.raises(ValueError, match="has no requested component.*llm"):
        cli._resolve_build_selection(args)


def test_geometry_profiles_and_policy_contract_agree(tmp_path):
    """Engine profiles and policy.json are cut from one geometry: the directory-resolved
    embodiment, the checkpoint's horizons and the requested context length."""
    checkpoint = _checkpoint(tmp_path)
    bundle = BundleConfig.from_pretrained(str(checkpoint))
    args = _args(checkpoint, tmp_path / "engines", max_input_len=512)
    geometry = runtime_config.geometry_from_bundle(bundle, args)

    assert geometry.embodiment_id == 2
    assert (geometry.action_horizon, geometry.action_dim,
            geometry.state_dim) == (40, 132, 132)
    assert geometry.max_vl_len == 512

    action = runtime_config.component_runtime_config(
        bundle, contracts.Component.ACTION, args)
    profile = action["optimization_profile"]
    assert profile["noise_trajectory"]["max"] == [1, 40, 132]
    assert profile["state"]["max"] == [1, 1, 132]
    assert profile["text_attention_mask"]["max"] == [1, 1, 1, 512]
    assert action["frontend"] == "direct"

    contract = policy_contract.build_policy_contract(
        str(checkpoint), bundle.root,
        bundle.component_dict(contracts.Component.VISUAL), geometry)
    assert (contract["embodiment_tag"],
            contract["embodiment_id"]) == ("libero_sim", 2)
    assert contract["action"]["horizon"] == HORIZON
    assert contract["state"]["total_dim"] == sum(STATE_DIMS)
    assert contract["video_keys"] == ["image", "wrist_image"]
    assert contract["vision"]["tokens_per_view"] == 64


def test_embodiment_falls_back_to_unknown_when_no_tag_matches(tmp_path):
    """A directory name matching no embodiment tag resolves to the unknown slot."""
    assert facts.resolve_embodiment(str(_checkpoint(tmp_path))) == 2
    assert facts.resolve_embodiment(str(_checkpoint(
        tmp_path, name="ckpt"))) == facts.UNKNOWN_EMBODIMENT_ID


@pytest.mark.parametrize(
    "override,match",
    [
        ({
            "spec_type": "eagle3"
        }, "speculative decoding"),
        ({
            "tp_size": 2
        }, "single-rank"),
    ],
)
def test_unsupported_build_options_are_rejected(tmp_path, override, match):
    """Build validation rejects speculative and tensor-parallel GR00T builds."""
    args = _args(_checkpoint(tmp_path), tmp_path / "engines", **override)

    with pytest.raises(ValueError, match=match):
        configuration.validate_build(args, (contracts.Component.ACTION, ))


@pytest.mark.parametrize(
    "stamp,match",
    [
        pytest.param({"export_id": "0" * 32}, "export_id=", id="build-id"),
        pytest.param({"frontend": "onnx"}, "frontend=", id="route"),
    ],
)
def test_preflight_rejects_another_build_without_writing(
        tmp_path, stamp, match):
    """A component from another build id or route is refused before any file changes."""
    checkpoint = _checkpoint(tmp_path)
    bundle = BundleConfig.from_pretrained(str(checkpoint))
    args = _args(checkpoint, tmp_path / "engines", components="action")
    engine_dir = _stamped_bundle(tmp_path, bundle, args, **stamp)
    before = _file_snapshot(engine_dir)

    with pytest.raises(ValueError, match=f"different GR00T build.*{match}"):
        configuration.validate_build(args, (contracts.Component.ACTION, ))

    assert _file_snapshot(engine_dir) == before


def test_preflight_accepts_partial_rebuild_from_the_same_build(tmp_path):
    """Rebuilding one component from the same checkpoint, options and route is allowed."""
    checkpoint = _checkpoint(tmp_path)
    bundle = BundleConfig.from_pretrained(str(checkpoint))
    args = _args(checkpoint, tmp_path / "engines", components="action")
    engine_dir = _stamped_bundle(tmp_path, bundle, args)
    before = _file_snapshot(engine_dir)

    configuration.validate_build(args, (contracts.Component.ACTION, ))

    assert _file_snapshot(engine_dir) == before


def test_missing_tokenizer_is_refused_before_any_engine_is_written(tmp_path):
    """A checkpoint the runtime could not load is refused by the CLI preflight."""
    checkpoint = _checkpoint(tmp_path, tokenizer=False)
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()

    with pytest.raises(ValueError, match="missing tokenizer/: "):
        cli._resolve_build_selection(_args(checkpoint, engine_dir))

    assert not any(engine_dir.iterdir())


def test_tokenizer_without_a_chat_template_is_refused(tmp_path):
    """The runtime renders the checkpoint's own chat template, so a tokenizer directory
    without one is refused as well."""
    checkpoint = _checkpoint(tmp_path)
    (checkpoint / "tokenizer" / "chat_template.json").unlink()

    with pytest.raises(ValueError, match="no chat template"):
        facts.require_tokenizer(str(checkpoint))


def test_gr00t_weight_aliases_cover_both_checkpoint_roots():
    """Backbone names map under the language model root, head names under action_head."""
    backbone = weights.resolve_candidates(
        "model.layers.0.self_attn.q_proj.weight",
        component="vl-backbone",
        spec_type="none",
        spec_role="none",
        quant_type="fp16")
    head = weights.resolve_candidates(
        "model.transformer_blocks.0.attn1.to_q.weight",
        component="action",
        spec_type="none",
        spec_role="none",
        quant_type="fp16")

    assert backbone[0] == LM + "0.self_attn.q_proj.weight"
    assert head == (
        "action_head.model.transformer_blocks.0.attn1.to_q.weight", )


def test_backbone_sidecars_are_staged_at_runtime_bundle_paths(
        monkeypatch, tmp_path):
    """The embedding table lands at the bundle root and the tokenizer is copied whole with
    a rendered chat_template.jinja beside it."""
    checkpoint = _checkpoint(tmp_path)
    bundle = BundleConfig.from_pretrained(str(checkpoint))
    table = SimpleNamespace(shape=(151936, 2048))
    saved = {}

    class _Checkpoint:

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def f16(self, name):
            assert name == "model.embed_tokens.weight"
            return table

        def close(self):
            pass

    monkeypatch.setattr(artifacts, "Weights", _Checkpoint)
    monkeypatch.setattr(
        artifacts, "save_safetensors",
        lambda path, tensors: saved.update(path=path, tensors=tensors))
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()

    artifacts._write_backbone_sidecars(bundle, str(engine_dir))

    assert saved["path"] == str(engine_dir / "embed_tokens.safetensors")
    assert saved["tensors"] == {"embed_tokens.weight": table}
    assert (engine_dir / "tokenizer" / "tokenizer.json").is_file()
    assert (engine_dir / "tokenizer" /
            "chat_template.jinja").read_text() == "{{ messages[0].content }}"
