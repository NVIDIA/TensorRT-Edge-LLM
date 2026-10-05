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

import numpy as np
import pytest

pytest.importorskip("tensorrt")

from experimental.builder.core import contracts
from experimental.builder.core.artifacts.runtime_config import \
    build_runtime_config
from experimental.builder.core.artifacts.tensors import save_safetensors
from experimental.builder.core.builder import BuildArgs, _setup_llm_profiles
from experimental.builder.core.config import DeviceConfig
from experimental.builder.core.safetensors_np import SafetensorsStore
from experimental.builder.models import registry
from experimental.builder.models.dflash2 import artifacts as dflash2_artifacts
from experimental.builder.models.dflash2 import modeling_dflash2_draft
from experimental.builder.ops.functional import speculative
from tensorrt_edgellm.dflash import DFlashVersion


def _qwen38_config(*, draft: bool = False) -> DeviceConfig:
    return DeviceConfig(
        model_type="qwen3",
        root_model_type="qwen3",
        hidden_size=5120,
        num_hidden_layers=5 if draft else 64,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=17408,
        vocab_size=248320,
        rms_norm_eps=1e-6,
        rope_theta=10_000_000.0,
        max_position_embeddings=262144,
        dflash_version=DFlashVersion.V2 if draft else DFlashVersion.V1,
        dflash_target_layer_ids=[5, 19, 33, 47, 61] if draft else [],
        dflash_block_size=8 if draft else 16,
        dflash_mask_token_id=248070,
        dflash2_target_layer_ids=[5, 19, 33, 47, 61] if draft else [],
        dflash2_block_size=8 if draft else 0,
        dflash2_mask_token_id=248070 if draft else -1,
        dflash2_is_causal=not draft,
        dflash2_conv_kernel_size=2 if draft else 0,
        dflash2_conv_group_size=16 if draft else 0,
        dflash2_selector_rank=256 if draft else 0,
        dflash2_selector_top_k=16 if draft else 0)


def _write_dflash2_config(model_dir):
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["DFlash2DraftModel"],
        "model_type":
        "qwen3",
        "hidden_size":
        5120,
        "num_hidden_layers":
        5,
        "num_attention_heads":
        40,
        "num_key_value_heads":
        8,
        "head_dim":
        128,
        "intermediate_size":
        17408,
        "vocab_size":
        248320,
        "rms_norm_eps":
        1e-6,
        "rope_theta":
        10_000_000.0,
        "max_position_embeddings":
        262144,
        "is_causal":
        False,
        "dflash_config": {
            "target_layer_ids": [5, 19, 33, 47, 61],
            "block_size": 8,
            "mask_token_id": 248070,
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "selector_rank": 256,
            "selector_top_k": 16
        }
    }),
                                           encoding="utf-8")
    return model_dir


@pytest.mark.parametrize("local_name", ["lm_head.weight", "lm_head.qweight"])
def test_dflash2_uses_local_fp16_or_quantized_target_head(
        monkeypatch, local_name):
    sentinel = object()

    class Weights:

        def has(self, name):
            return name == local_name

        def causal_lm_head_prefix(self):
            return "lm_head"

    ctx = SimpleNamespace(weights=Weights())
    monkeypatch.setattr(modeling_dflash2_draft, "Linear",
                        lambda context, prefix: sentinel)

    assert modeling_dflash2_draft._resolve_target_lm_head(ctx,
                                                          None) is sentinel


def test_dflash2_prefers_injected_paired_target_head():
    sentinel = object()
    ctx = SimpleNamespace(weights=None)

    assert modeling_dflash2_draft._resolve_target_lm_head(ctx,
                                                          sentinel) is sentinel


def test_dflash2_drops_anchor_with_runtime_block_dimension(monkeypatch):
    hidden = object()
    captured = {}

    def dynamic_slice(value, start, size):
        captured.update(value=value, start=start, size=size)
        return "prediction_hidden"

    monkeypatch.setattr(
        modeling_dflash2_draft, "F",
        SimpleNamespace(
            shape_of=lambda value: np.asarray([8, 16, 5120], dtype=np.int32),
            constant=lambda value, name: value,
            dynamic_slice=dynamic_slice,
        ))

    result = modeling_dflash2_draft._drop_anchor_token(hidden, 5120)

    assert result == "prediction_hidden"
    assert captured["value"] is hidden
    assert tuple(int(np.asarray(value).item())
                 for value in captured["start"]) == (0, 1, 0)
    assert tuple(int(np.asarray(value).item())
                 for value in captured["size"]) == (8, 15, 5120)


def test_dflash2_checkpoint_contract_is_parsed_exactly(tmp_path):
    cfg = DeviceConfig.from_pretrained(
        str(_write_dflash2_config(tmp_path / "draft")))

    assert cfg.dflash_version == DFlashVersion.V2
    assert cfg.dflash2_is_causal is False
    assert cfg.dflash2_target_layer_ids == [5, 19, 33, 47, 61]
    assert cfg.dflash2_block_size == 8
    assert cfg.dflash2_mask_token_id == 248070
    assert cfg.dflash2_conv_kernel_size == 2
    assert cfg.dflash2_conv_group_size == 16
    assert cfg.dflash2_selector_rank == 256
    assert cfg.dflash2_selector_top_k == 16


def test_dflash2_tree_base_is_rejected():
    args = BuildArgs(
        model_dir="target",
        engine_dir="engine",
        spec_role="base",
        spec_type="dflash",
        dflash_version=DFlashVersion.V2,
        draft_model_dir="draft",
        tree_base=True,
    )

    with pytest.raises(ValueError, match="tree-base.*not supported.*V2"):
        args.validate()


def test_dflash2_base_requires_paired_draft():
    args = BuildArgs(
        model_dir="target",
        engine_dir="engine",
        spec_role="base",
        spec_type="dflash",
        dflash_version=DFlashVersion.V2,
    )

    with pytest.raises(ValueError, match="dflash base requires"):
        args.validate()


def test_dflash2_cannot_use_dflash_reduced_vocab_option():
    args = BuildArgs(
        model_dir="draft",
        engine_dir="engine",
        spec_role="draft",
        spec_type="dflash",
        dflash_version=DFlashVersion.V2,
        target_model_dir="target",
        draft_reduced_vocab_dir="reduced",
    )

    with pytest.raises(ValueError, match="does not support reduced"):
        args.validate()


def test_dflash2_draft_configuration_validates_published_contract():
    target = _qwen38_config()
    draft = _qwen38_config(draft=True)

    configured = registry.configure_for_build(
        draft,
        contracts.SpecRole.DRAFT,
        "dflash",
        paired_target=target,
        build_args=SimpleNamespace(dflash_version=DFlashVersion.V2))

    assert configured.spec_decode_type == "dflash"
    assert configured.engine_role == "draft"
    assert configured.dflash2_is_causal is False


def test_dflash2_base_configuration_reads_paired_draft_contract(monkeypatch):
    target = _qwen38_config()
    draft = _qwen38_config(draft=True)
    monkeypatch.setattr(DeviceConfig, "from_pretrained",
                        lambda model_dir: draft)

    configured = registry.configure_for_build(
        target,
        contracts.SpecRole.BASE,
        "dflash",
        paired_draft_dir="draft",
        build_args=SimpleNamespace(dflash_version=DFlashVersion.V2),
    )

    assert configured.dflash_version == DFlashVersion.V2
    assert configured.dflash_tree_base
    assert configured.dflash2_target_layer_ids == [5, 19, 33, 47, 61]
    assert configured.dflash2_block_size == 8
    assert configured.dflash2_selector_top_k == 16


def test_dflash2_rejects_missing_architecture_field():
    draft = _qwen38_config(draft=True)
    draft.dflash_version = DFlashVersion.V1
    target = _qwen38_config()

    with pytest.raises(ValueError, match="DFlash2DraftModel"):
        registry.configure_for_build(
            draft,
            contracts.SpecRole.DRAFT,
            "dflash",
            paired_target=target,
            build_args=SimpleNamespace(dflash_version=DFlashVersion.V2))


@pytest.mark.parametrize("draft_layers,target_layer_ids", [
    (4, [5, 19, 33, 47, 61]),
    (5, [5, 19, 33, 47]),
])
def test_dflash2_rejects_non_production_layer_contract(draft_layers,
                                                       target_layer_ids):
    draft = _qwen38_config(draft=True)
    draft.num_hidden_layers = draft_layers
    draft.dflash2_target_layer_ids = target_layer_ids

    with pytest.raises(ValueError, match="exactly five"):
        registry.configure_for_build(
            draft,
            contracts.SpecRole.DRAFT,
            "dflash",
            paired_target=_qwen38_config(),
            build_args=SimpleNamespace(dflash_version=DFlashVersion.V2))


def test_dflash2_runtime_metadata_is_versioned_and_sparse(monkeypatch):
    target = _qwen38_config()
    draft = _qwen38_config(draft=True)
    monkeypatch.setattr(DeviceConfig, "from_pretrained",
                        lambda model_dir: draft)
    registry.configure_for_build(
        target,
        contracts.SpecRole.BASE,
        "dflash",
        paired_draft_dir="draft",
        build_args=SimpleNamespace(dflash_version=DFlashVersion.V2),
    )
    args = BuildArgs(
        model_dir="target",
        engine_dir="engine",
        spec_role="base",
        spec_type="dflash",
        dflash_version=DFlashVersion.V2,
        draft_model_dir="draft",
        max_verify_tree_size=8,
        max_draft_tree_size=8,
    )

    metadata = build_runtime_config(target, args)

    assert metadata["spec_decode_type"] == "dflash"
    assert metadata["dflash_config"] == {
        "version": 2,
        "target_layer_ids": [5, 19, 33, 47, 61],
        "block_size": 8,
        "mask_token_id": 248070,
        "is_causal": False,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
        "selector_file": "dflash2_selector.safetensors",
        "supports_probabilistic_sampling": True,
    }


def test_dflash2_draft_resolves_its_selector_sidecar_writer():
    writer = registry.artifact_writer_for("qwen3", "dflash",
                                          contracts.SpecRole.DRAFT,
                                          DFlashVersion.V2)

    assert writer.__name__.endswith(".dflash2.artifacts")


def test_dflash2_artifact_writer_preserves_selector_codebooks(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    predecessor = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float16)
    successor = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float16)
    save_safetensors(
        str(model_dir / "model.safetensors"), {
            "candidate_selector.predecessor_codebook": predecessor,
            "candidate_selector.successor_codebook": successor,
        })
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    args = SimpleNamespace(model_dir=str(model_dir),
                           resolved_component=contracts.Component.LLM)
    config = SimpleNamespace(vocab_size=3, dflash2_selector_rank=2)

    dflash2_artifacts._write_selector_sidecar(config, args, str(engine_dir))

    with SafetensorsStore(str(engine_dir)) as store:
        np.testing.assert_array_equal(store.get_f16("predecessor_codebook"),
                                      predecessor)
        np.testing.assert_array_equal(store.get_f16("successor_codebook"),
                                      successor)


def test_dflash2_build_profile_rejects_sizes_above_supported_maximum():
    args = BuildArgs(
        model_dir="target",
        engine_dir="engine",
        spec_role="base",
        spec_type="dflash",
        dflash_version=DFlashVersion.V2,
        draft_model_dir="draft",
        max_verify_tree_size=60,
        max_draft_tree_size=60,
    )

    with pytest.raises(ValueError, match=r"verify.*draft.*\[2, 16\]"):
        args.validate()


def test_dflash2_build_profile_accepts_runtime_block_16():
    args = BuildArgs(
        model_dir="target",
        engine_dir="engine",
        spec_role="base",
        spec_type="dflash",
        dflash_version=DFlashVersion.V2,
        draft_model_dir="draft",
        max_verify_tree_size=16,
        max_draft_tree_size=16,
    )

    args.validate()


def test_dflash_target_cache_update_threads_managed_page_table(monkeypatch):
    tensors = [object() for _ in range(7)]
    observed = {}

    def record_operation(name, inputs, **attributes):
        observed.update(name=name, inputs=inputs, attributes=attributes)
        return "layer"

    monkeypatch.setattr(speculative, "operation", record_operation)

    result = speculative.update_dflash_target_cache(*tensors)

    assert result == "layer"
    assert observed == {
        "name": "dflash_target_cache_update",
        "inputs": tensors,
        "attributes": {},
    }


def test_dflash2_draft_profiles_do_not_bind_runtime_selector_inputs():

    class Profile:

        def __init__(self):
            self.shapes = {}

        def set_shape(self, name, minimum, optimum, maximum):
            self.shapes[name] = (minimum, optimum, maximum)

    class Builder:

        def __init__(self):
            self.profiles = []

        def create_optimization_profile(self):
            profile = Profile()
            self.profiles.append(profile)
            return profile

    input_shapes = {
        "spec_sampling_temperature": (-1, ),
        "spec_proposal_greedy_mask": (-1, ),
        "spec_proposal_uniforms": (-1, 7),
        "spec_anchor_token_ids": (-1, ),
        "inputs_embeds": (-1, 5120),
        "positions": (-1, ),
        "query_start_offsets": (-1, ),
        "query_lengths": (-1, ),
        "past_lengths": (-1, ),
        "attention_sequence_lengths": (-1, ),
        "state_indices": (-1, ),
        "execution_phase_marker": (-1, ),
        "context_sequence_count_carrier": (-1, ),
        "kv_page_table": (-1, 2, -1),
        "attention_position_ids": (-1, ),
        "packed_attention_mask": (-1, -1),
    }
    tensors = [
        SimpleNamespace(name=name, shape=shape)
        for name, shape in input_shapes.items()
    ]
    network = SimpleNamespace(
        num_inputs=len(tensors),
        get_input=lambda index: tensors[index],
    )
    builder = Builder()
    config = SimpleNamespace(profiles=[])
    config.add_optimization_profile = lambda profile: config.profiles.append(
        profile)
    cfg = SimpleNamespace(
        hidden_size=5120,
        num_key_value_heads=8,
        head_dim=128,
        rotary_dim=128,
        is_hybrid=False,
        num_attn_layers=0,
        num_hidden_layers=0,
        mamba_cfg=None,
        gdn_cfg=None,
        num_deepstack_features=0,
        vocab_size=248320,
        shares_target_kv=False,
        hidden_size_per_layer_input=5120,
        dflash2_block_size=8,
    )
    args = BuildArgs(model_dir="draft",
                     engine_dir="engine",
                     spec_role="draft",
                     spec_type="dflash",
                     dflash_version=DFlashVersion.V2,
                     target_model_dir="target",
                     max_batch_size=8,
                     max_input_len=2048,
                     max_kv_cache_capacity=4096,
                     max_verify_tree_size=16,
                     max_draft_tree_size=16)

    _setup_llm_profiles(builder, config, network, cfg, args)

    for profile in builder.profiles:
        assert "spec_sampling_temperature" not in profile.shapes
        assert "spec_proposal_greedy_mask" not in profile.shapes
        assert "spec_proposal_uniforms" not in profile.shapes
        assert "spec_anchor_token_ids" not in profile.shapes
        assert "kvcache_start_index" not in profile.shapes
        assert "context_lengths" not in profile.shapes
        assert "dflash_delta_lengths" not in profile.shapes
        assert profile.shapes["inputs_embeds"] == ((2, 5120), (64, 5120),
                                                   (128, 5120))
        assert profile.shapes["attention_position_ids"] == ((2, ), (64, ),
                                                            (128, ))
        assert profile.shapes["packed_attention_mask"] == ((2, 1), (64, 1),
                                                           (128, 1))
