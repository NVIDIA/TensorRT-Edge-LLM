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

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt")

from experimental.builder.core import contracts, quantization
from experimental.builder.core.artifacts import embeddings as direct_embeddings
from experimental.builder.core.artifacts.runtime_config import \
    build_runtime_config
from experimental.builder.core.artifacts.tensors import save_safetensors
from experimental.builder.core.builder import BuildArgs
from experimental.builder.core.config import DeviceConfig
from experimental.builder.core.safetensors_np import SafetensorsStore
from experimental.builder.models.dflash import configuration as dflash_config
from experimental.builder.models.dspark import artifacts as dspark_artifacts
from experimental.builder.models.dspark import configuration as dspark_config
from experimental.builder.models.nemotron_h import weights as nemotron_weights
from experimental.builder.ops.functional import recurrent
from experimental.builder.ops.ragged import RaggedDecoderInputs
from experimental.builder.weight_packing import nvfp4 as direct_nvfp4

direct_attention = importlib.import_module(
    "experimental.builder.ops.functional.attention")


def _target_config():
    return DeviceConfig(model_type="nemotron_h",
                        root_model_type="nemotron_h",
                        hidden_size=2688,
                        num_hidden_layers=52,
                        num_attention_heads=32,
                        num_key_value_heads=2,
                        head_dim=128,
                        intermediate_size=2688,
                        vocab_size=131072,
                        rms_norm_eps=1e-5,
                        rope_theta=10000.0,
                        max_position_embeddings=131072)


@pytest.mark.parametrize("spec_type", ["mtp", "dflash", "dspark"])
def test_tree_base_validation_accepts_supported_decoders(spec_type):
    args = BuildArgs(model_dir="target",
                     engine_dir="engine",
                     spec_role="base",
                     spec_type=spec_type,
                     draft_model_dir="draft" if spec_type != "mtp" else "",
                     tree_base=True)

    args.validate()


@pytest.mark.parametrize("tree_base", [False, True])
def test_dflash_v1_base_is_tree_capable_for_chain_replay(
        monkeypatch, tree_base):
    draft = {
        "hidden_size": 2688,
        "vocab_size": 131072,
        "dflash_config": {
            "target_layer_ids": [1, 5, 19, 29, 41, 51],
            "block_size": 7,
            "mask_token_id": 990,
        },
    }
    bundle = SimpleNamespace(component_dict=lambda component: draft, )
    monkeypatch.setattr(dflash_config.BundleConfig, "from_pretrained",
                        lambda path: bundle)
    config = _target_config()

    dflash_config.configure_base(
        config,
        paired_draft_dir="draft",
        build_args=SimpleNamespace(tree_base=tree_base),
    )

    assert config.dflash_tree_base is True


@pytest.mark.parametrize("tree_base", [False, True])
def test_dspark_base_preserves_requested_tree_mode(monkeypatch, tree_base):
    draft = {
        "hidden_size": 2688,
        "vocab_size": 131072,
        "target_layer_ids": [1, 5, 19, 29, 41, 51],
        "block_size": 6,
        "mask_token_id": 990,
        "markov_head_type": "vanilla",
        "dflash_query_causal": True,
        "sliding_window": 1024,
        "sample_from_anchor": False,
    }
    bundle = SimpleNamespace(component_dict=lambda component: draft, )
    monkeypatch.setattr(dspark_config.BundleConfig, "from_pretrained",
                        lambda path: bundle)
    config = _target_config()

    dspark_config.configure_base(
        config,
        paired_draft_dir="draft",
        build_args=SimpleNamespace(tree_base=tree_base,
                                   max_verify_tree_size=7,
                                   max_draft_tree_size=7,
                                   reduced_vocab_dir="",
                                   draft_reduced_vocab_dir=""),
    )

    assert config.dspark_tree_base is tree_base
    assert config.dspark_causal_proposal is True
    assert config.dspark_contiguous_query_swa is True
    assert config.dspark_sample_from_anchor is False


def test_dspark_runtime_config_preserves_official_proposal_contract():
    config = _target_config()
    config.spec_decode_type = "dspark"
    config.dspark_target_layer_ids = [1, 5, 19, 29, 41, 51]
    config.dspark_block_size = 8
    config.dspark_mask_token_id = 990
    config.dspark_markov_head_type = "vanilla"
    config.dspark_markov_rank = 512
    config.dspark_causal_proposal = True
    config.dspark_contiguous_query_swa = True
    config.dspark_sample_from_anchor = False
    args = SimpleNamespace(
        resolved_spec_role=contracts.SpecRole.BASE,
        spec_type="dspark",
        tp_size=1,
        max_input_len=256,
        max_batch_size=2,
        max_lora_rank=0,
        max_kv_cache_capacity=1024,
        max_verify_tree_size=3,
        max_draft_tree_size=2,
    )

    runtime = build_runtime_config(config, args)

    assert runtime["dspark_config"]["causal_head"] is True
    assert runtime["dspark_config"]["contiguous_query_swa"] is True
    assert runtime["dspark_config"]["sample_from_anchor"] is False


def test_dspark_draft_runtime_config_excludes_non_anchor_slot():
    config = _target_config()
    config.spec_decode_type = "dspark"
    config.dspark_sample_from_anchor = False
    args = SimpleNamespace(
        resolved_spec_role=contracts.SpecRole.DRAFT,
        spec_type="dspark",
        tp_size=1,
        max_input_len=256,
        max_batch_size=2,
        max_lora_rank=0,
        max_kv_cache_capacity=1024,
        max_verify_tree_size=9,
        max_draft_tree_size=9,
    )

    runtime = build_runtime_config(config, args)

    assert runtime["builder_config"]["max_draft_tree_size"] == 8


def test_dspark_direct_sidecar_dequantizes_nvfp4_markov_w2(tmp_path):
    weight_key = "markov_head.markov_w2.weight"
    scale_key, scale_2_key = dspark_artifacts._NVFP4_HEAD_SCALES["markov_w2"]
    packed = np.zeros((2, 8), dtype=np.uint8)
    packed[:, 0] = 0x21
    scale = np.full((2, 1), 0x38, dtype=np.uint8)
    scale_2 = np.array([2.0], dtype=np.float32)
    checkpoint = tmp_path / "model.safetensors"
    save_safetensors(str(checkpoint), {
        weight_key: packed,
        scale_key: scale,
        scale_2_key: scale_2,
    },
                     dtype_overrides={scale_key: "F8_E4M3"})

    with SafetensorsStore(str(tmp_path)) as store:
        dense, metadata = dspark_artifacts._load_head_tensor(
            store, "markov_w2", weight_key, 16)

    assert dense.shape == (2, 16)
    assert dense.dtype == np.float16
    assert metadata["nvfp4_dequantized"] is True
    assert metadata["packed_shape"] == [2, 8]
    assert metadata["group_size"] == 16


def test_nemotron_w4a16_is_model_specific():
    algorithm = "W4A16_NVFP4"
    assert quantization.algorithm_to_type(
        algorithm) == quantization.QUANT_NVFP4
    assert (quantization._algorithm_to_type(
        algorithm, nemotron_weights) == quantization.QUANT_NVFP4_A16)


def test_direct_a16_marlin_linear_matches_reference():
    torch = pytest.importorskip("torch")
    from tensorrt_edgellm.checkpoint.repacking import \
        repack_nvfp4_a16_marlin_linear

    rng = np.random.default_rng(17)
    packed = rng.integers(0, 256, size=(65, 64), dtype=np.uint8)
    scales = rng.integers(0, 256, size=(65, 8), dtype=np.uint8)
    global_scale = np.float32(0.03125)
    direct = direct_nvfp4.pack_nvfp4_a16_linear(packed, scales, global_scale)
    reference = repack_nvfp4_a16_marlin_linear(
        torch.from_numpy(packed), torch.from_numpy(scales.view(np.int8)),
        torch.tensor([global_scale], dtype=torch.float32))
    for actual, expected in zip(direct[:3], reference[:3]):
        np.testing.assert_array_equal(actual, expected.cpu().numpy())
    assert direct[3:] == reference[3:]


def test_direct_a16_blackwell_linear_matches_reference():
    torch = pytest.importorskip("torch")
    from tensorrt_edgellm.checkpoint.repacking import \
        repack_nvfp4_a16_blackwell_linear

    rng = np.random.default_rng(23)
    packed = rng.integers(0, 256, size=(65, 64), dtype=np.uint8)
    scales = rng.integers(0, 256, size=(65, 8), dtype=np.uint8)
    global_scale = np.float32(0.015625)
    direct = direct_nvfp4.pack_nvfp4_a16_blackwell_linear(
        packed, scales, global_scale)
    reference = repack_nvfp4_a16_blackwell_linear(
        torch.from_numpy(packed), torch.from_numpy(scales.view(np.int8)),
        torch.tensor([global_scale], dtype=torch.float32))
    for actual, expected in zip(direct[:3], reference[:3]):
        np.testing.assert_array_equal(actual, expected.cpu().numpy())
    assert direct[3:] == reference[3:]


def test_direct_a16_nongated_moe_matches_reference():
    torch = pytest.importorskip("torch")
    from tensorrt_edgellm.checkpoint.repacking import \
        repack_nvfp4_a16_marlin_moe_experts

    rng = np.random.default_rng(31)
    experts = []
    for _ in range(2):
        experts.append({
            "up_packed":
            rng.integers(0, 256, (144, 64), dtype=np.uint8),
            "up_sf":
            rng.integers(0, 256, (144, 8), dtype=np.uint8),
            "up_alpha":
            np.float32(0.03125),
            "down_packed":
            rng.integers(0, 256, (128, 72), dtype=np.uint8),
            "down_sf":
            rng.integers(0, 256, (128, 9), dtype=np.uint8),
            "down_alpha":
            np.float32(0.015625),
        })
    direct = direct_nvfp4.pack_nvfp4_a16_nongated_experts(
        experts.__getitem__, 2, 128, 144)
    reference = repack_nvfp4_a16_marlin_moe_experts(
        [torch.from_numpy(expert["up_packed"]) for expert in experts], [
            torch.from_numpy(expert["up_sf"].view(np.int8))
            for expert in experts
        ], [torch.tensor([expert["up_alpha"]]) for expert in experts],
        [torch.from_numpy(expert["down_packed"]) for expert in experts], [
            torch.from_numpy(expert["down_sf"].view(np.int8))
            for expert in experts
        ], [torch.tensor([expert["down_alpha"]]) for expert in experts], 256)
    names = ("fc1_qweights", "fc1_block_scales", "fc1_global_scales",
             "fc2_qweights", "fc2_block_scales", "fc2_global_scales")
    for name, expected in zip(names, reference):
        np.testing.assert_array_equal(direct[name], expected.cpu().numpy())
    assert direct["padded_intermediate"] == 256


def test_dspark_direct_sidecar_rejects_unscaled_packed_markov_w2(tmp_path):
    weight_key = "markov_head.markov_w2.weight"
    checkpoint = tmp_path / "model.safetensors"
    save_safetensors(str(checkpoint), {
        weight_key: np.zeros((2, 8), dtype=np.uint8),
    })

    with SafetensorsStore(str(tmp_path)) as store:
        with pytest.raises(KeyError, match="scale tensors are missing"):
            dspark_artifacts._load_head_tensor(store, "markov_w2", weight_key,
                                               16)


def test_direct_attention_emits_sink_and_contiguous_swa_contract(monkeypatch):
    captured = {}

    def fake_operation(name, inputs, output_count, **attributes):
        captured.update(name=name,
                        inputs=inputs,
                        output_count=output_count,
                        attributes=attributes)
        return "attention", "present"

    monkeypatch.setattr(direct_attention, "operation", fake_operation)
    ragged = RaggedDecoderInputs(
        positions="positions",
        query_start_offsets="query_offsets",
        query_lengths="lengths",
        past_lengths="cache_start",
        attention_sequence_lengths="attention_lengths",
        state_indices="state_indices",
        logits_indices="logits_indices",
        execution_phase_marker="phase",
        context_sequence_count_carrier="context_count",
        kv_page_table="page_table",
    )
    result = direct_attention.attention(
        "qkv",
        "past",
        "rope",
        ragged,
        32,
        2,
        128,
        sliding_window_size=1024,
        attention_mask="mask",
        attention_pos_id="positions",
        attention_sinks="sinks",
        enable_contiguous_query_swa=True,
    )

    assert result == ("attention", "present")
    assert captured["inputs"] == [
        "qkv", "past", "lengths", "rope", "cache_start", "page_table", "mask",
        "positions", "sinks", "query_offsets", "attention_lengths", "phase",
        "context_count"
    ]
    assert captured["attributes"]["enable_tree_attention"] == 1
    assert captured["attributes"]["enable_attention_sink"] == 1
    assert captured["attributes"]["enable_contiguous_query_swa"] == 1
    assert captured["attributes"]["sliding_window_size"] == 1024


def test_update_ssm_state_emits_tree_replay(monkeypatch):
    captured = {}

    def fake_operation(name, inputs, output_count, **attributes):
        captured.update(name=name,
                        inputs=inputs,
                        output_count=output_count,
                        attributes=attributes)
        return tuple(range(output_count))

    monkeypatch.setattr(recurrent, "operation", fake_operation)
    tensors = list(range(15))
    ragged = RaggedDecoderInputs(
        positions=None,
        query_start_offsets=tensors[9],
        query_lengths=tensors[8],
        past_lengths=None,
        attention_sequence_lengths=None,
        state_indices=tensors[10],
        logits_indices=None,
        execution_phase_marker=tensors[11],
        context_sequence_count_carrier=tensors[12],
        kv_page_table=None,
    )

    result = recurrent.update_ssm_state(*tensors[:8],
                                        ragged,
                                        2688,
                                        128,
                                        42,
                                        8,
                                        tree_parent_ids=tensors[13],
                                        tree_depths=tensors[14],
                                        use_intermediate=True)

    assert result == tuple(range(6))
    assert captured["name"] == "update_ssm_state"
    assert captured["inputs"] == tensors
    assert captured["output_count"] == 6
    assert captured["attributes"]["use_spec_verify_state"] == 1
    assert captured["attributes"]["use_ddtree"] == 1


def test_cached_draft_base_embedding_is_not_externalized():
    policy = SimpleNamespace(externalizes_embedding=True)
    for spec_type in ("dflash", "dspark"):
        args = SimpleNamespace(
            weight_policy=policy,
            resolved_spec_role=contracts.SpecRole.BASE,
            spec_type=spec_type,
        )
        assert not direct_embeddings.externalizes_embedding(
            args, nemotron_weights)

    mtp = SimpleNamespace(
        weight_policy=policy,
        resolved_spec_role=contracts.SpecRole.BASE,
        spec_type="mtp",
    )
    assert direct_embeddings.externalizes_embedding(mtp, nemotron_weights)


def test_cached_draft_mask_row_is_folded_into_base_embedding(tmp_path):
    output_dir = tmp_path / "engine"
    draft_dir = tmp_path / "draft"
    output_dir.mkdir()
    draft_dir.mkdir()
    base = np.arange(20, dtype=np.float16).reshape(5, 4)
    draft = base + np.float16(7.0)
    save_safetensors(str(output_dir / "embedding.safetensors"),
                     {"embedding": base.copy()})
    save_safetensors(str(draft_dir / "model.safetensors"),
                     {"embed_tokens.weight": draft})

    direct_embeddings.write_cached_draft_embedding(str(output_dir),
                                                   str(draft_dir), 2, 0.5)

    patched = direct_embeddings.load_safetensors_tensor(
        str(output_dir / "embedding.safetensors"), "embedding")
    np.testing.assert_array_equal(patched[:2], base[:2])
    np.testing.assert_array_equal(patched[2], draft[2] * np.float16(0.5))
    np.testing.assert_array_equal(patched[3:], base[3:])


def test_safetensors_store_reads_single_bf16_row(tmp_path):
    values = np.array([[1.0, -2.0, 3.5], [4.0, 5.25, -6.5]], dtype=np.float32)
    bf16 = (values.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    save_safetensors(str(tmp_path / "model.safetensors"),
                     {"weight": bf16.view(np.float16)}, {"weight": "BF16"})

    with SafetensorsStore(str(tmp_path)) as store:
        row = store.get_f16_row("weight", 1)

    np.testing.assert_array_equal(row, values[1].astype(np.float16))


def test_safetensors_store_reads_sharded_f32_row(tmp_path):
    values = np.array([[1.25, -2.5], [3.75, 4.5]], dtype=np.float32)
    shard = "model-00001-of-00001.safetensors"
    save_safetensors(str(tmp_path / shard), {"weight": values})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {
            "weight": shard
        }}))

    with SafetensorsStore(str(tmp_path)) as store:
        row = store.get_f16_row("weight", 1)
        with pytest.raises(ValueError, match="outside shape"):
            store.get_f16_row("weight", 2)

    np.testing.assert_array_equal(row, values[1].astype(np.float16))


def test_pytorch_store_rejects_integer_embedding_row(tmp_path):
    torch = pytest.importorskip("torch")
    torch.save({"weight": torch.ones((2, 3), dtype=torch.int32)},
               tmp_path / "pytorch_model.bin")

    with SafetensorsStore(str(tmp_path)) as store:
        with pytest.raises(TypeError, match="cannot decode dtype .I32."):
            store.get_f16_row("weight", 0)


def test_cached_draft_without_embedding_keeps_base_sidecar(tmp_path):
    output_dir = tmp_path / "engine"
    draft_dir = tmp_path / "draft"
    output_dir.mkdir()
    draft_dir.mkdir()
    base = np.arange(12, dtype=np.uint8).reshape(3, 4)
    scales = np.ones((3, 1), dtype=np.float32)
    save_safetensors(str(output_dir / "embedding.safetensors"), {
        "embedding": base.copy(),
        "embedding_scale": scales
    }, {"embedding": "F8_E4M3"})
    save_safetensors(str(draft_dir / "model.safetensors"),
                     {"other.weight": np.ones((1, 1), dtype=np.float16)})

    direct_embeddings.write_cached_draft_embedding(str(output_dir),
                                                   str(draft_dir), 1, 1.0)

    actual = direct_embeddings.load_safetensors_tensor(
        str(output_dir / "embedding.safetensors"), "embedding")
    np.testing.assert_array_equal(actual, base)


def test_dspark_profile_resolution_uses_legacy_mask_slot_contract():
    draft = {
        "dflash_config": {
            "block_size": 6,
            "sample_from_anchor": False,
        },
    }

    verify_size, draft_size = dspark_config.resolve_build_profile(
        draft, None, None, False)

    assert draft_size == 7
    assert verify_size == 7


def test_dspark_profile_resolution_defaults_legacy_checkpoint_to_anchor_slot():
    draft = {"dflash_config": {"block_size": 8}}

    verify_size, draft_size = dspark_config.resolve_build_profile(
        draft, None, None, False)

    assert draft_size == 8
    assert verify_size == 9


def test_dspark_profile_resolution_reserves_explicit_non_anchor_slot():
    draft = {
        "dflash_config": {
            "block_size": 8,
            "sample_from_anchor": False,
        },
    }

    verify_size, draft_size = dspark_config.resolve_build_profile(
        draft, None, None, False)

    assert draft_size == 9
    assert verify_size == 9


def test_dspark_tree_profile_uses_non_anchor_slot_as_verify_root():
    draft = {
        "dflash_config": {
            "block_size": 8,
            "sample_from_anchor": False,
        },
    }

    verify_size, draft_size = dspark_config.resolve_build_profile(
        draft, None, None, True)

    assert draft_size == 9
    assert verify_size == 9


def test_dspark_config_precedence_is_nested_then_top_level_then_legacy():
    values = dspark_config.resolve_dspark_config({
        "block_size": 8,
        "sample_from_anchor": False,
        "dflash_config": {
            "block_size": 6,
            "sample_from_anchor": True,
            "markov_rank": 128,
        },
        "dspark_config": {
            "block_size": 9,
        },
    })

    assert values["block_size"] == 9
    assert values["sample_from_anchor"] is False
    assert values["markov_rank"] == 128


def test_dspark_modern_attention_metadata_overrides_top_level():
    values = dspark_config.resolve_dspark_config(
        {
            "attention_sink_bias": False,
            "layer_types": ["sliding_attention"],
            "sliding_window": 1024,
            "dspark_config": {
                "attention_sink_bias": True,
                "use_swa": False,
                "swa_window_size": 2048,
            },
        }, 1024)

    assert values["attention_sink_bias"] is True
    assert values["sliding_window_size"] == -1
    assert values["contiguous_query_swa"] is False


def test_dspark_resolver_respects_normalized_disabled_sliding_window():
    draft = {
        "sliding_window": None,
        "use_sliding_window": False,
        "layer_types": ["full_attention"],
    }
    values = dspark_config.resolve_dspark_config(draft)

    assert values["sliding_window_size"] == -1
    assert values["contiguous_query_swa"] is False


def test_dspark_device_config_reads_legacy_dflash_contract(monkeypatch):
    draft = {
        "model_type": "qwen3",
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "intermediate_size": 128,
        "vocab_size": 1000,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 1024,
        "dflash_config": {
            "target_layer_ids": [0],
            "block_size": 6,
            "mask_token_id": 990,
            "causal": True,
            "swa_window_size": 1024,
            "sample_from_anchor": False,
        },
    }
    bundle = SimpleNamespace(
        root_model_type="qwen3",
        root=draft,
        component_dict=lambda component: draft,
    )
    monkeypatch.setattr(
        "experimental.builder.core.config.BundleConfig.from_pretrained",
        lambda path: bundle)

    config = DeviceConfig.from_pretrained("draft")

    assert config.dspark_target_layer_ids == [0]
    assert config.dspark_block_size == 6
    assert config.dspark_mask_token_id == 990
    assert config.dspark_causal_proposal is True
    assert config.dspark_contiguous_query_swa is True
    assert config.sliding_window_size == 1024
    assert config.dspark_sample_from_anchor is False


def test_dspark_tree_draft_skips_base_chain_topology_validation():
    config = _target_config()
    config.dspark_block_size = 8
    config.dspark_mask_token_id = 990
    config.dspark_markov_head_type = "vanilla"
    config.dspark_sample_from_anchor = False
    args = SimpleNamespace(
        resolved_spec_role=contracts.SpecRole.DRAFT,
        tree_base=False,
        max_verify_tree_size=3,
        max_draft_tree_size=2,
        reduced_vocab_dir="",
        draft_reduced_vocab_dir="",
    )

    dspark_config._validate_runtime_contract(config, args,
                                             contracts.SpecRole.DRAFT)
