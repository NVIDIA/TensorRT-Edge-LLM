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

from tensorrt_edgellm.builder.core import contracts, quantization
from tensorrt_edgellm.builder.core.artifacts import \
    embeddings as direct_embeddings
from tensorrt_edgellm.builder.core.artifacts.runtime_config import \
    build_runtime_config
from tensorrt_edgellm.builder.core.artifacts.tensors import save_safetensors
from tensorrt_edgellm.builder.core.builder import BuildArgs
from tensorrt_edgellm.builder.core.config import DeviceConfig
from tensorrt_edgellm.builder.core.safetensors_np import SafetensorsStore
from tensorrt_edgellm.builder.models.dflash import \
    configuration as dflash_config
from tensorrt_edgellm.builder.models.dspark import \
    artifacts as dspark_artifacts
from tensorrt_edgellm.builder.models.dspark import \
    configuration as dspark_config
from tensorrt_edgellm.builder.models.eagle3 import \
    configuration as eagle3_config
from tensorrt_edgellm.builder.models.nemotron_h import \
    weights as nemotron_weights
from tensorrt_edgellm.builder.ops.functional import recurrent
from tensorrt_edgellm.builder.ops.ragged import RaggedDecoderInputs
from tensorrt_edgellm.builder.weight_packing import nvfp4 as direct_nvfp4

direct_attention = importlib.import_module(
    "tensorrt_edgellm.builder.ops.functional.attention")


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


@pytest.mark.parametrize(
    ("module", "config_name"),
    ((dflash_config, "dflash_config"), (dspark_config, "dspark_config")),
)
def test_provider_aux_layers_are_normalized_for_direct_builder(
        module, config_name):
    root = {
        "aux_hidden_state_layer_ids": [2, 10, 18, 26, 34],
        "block_size": 8,
        "transformer_layer_config": {},
    }

    config = module.component_config(root, contracts.Component.LLM)

    assert config[config_name]["target_layer_ids"] == [1, 9, 17, 25, 33]


def test_eagle3_provider_default_layers_match_speculators_contract():
    assert eagle3_config._target_layers({}, 36) == [2, 18, 33]


@pytest.mark.parametrize("spec_type", ["mtp", "dflash", "dspark"])
def test_tree_base_validation_accepts_supported_decoders(spec_type):
    args = BuildArgs(model_dir="target",
                     engine_dir="engine",
                     spec_role="base",
                     spec_type=spec_type,
                     draft_model_dir="draft" if spec_type != "mtp" else "")

    args.validate()


@pytest.mark.parametrize("verify_tokens",
                         [60, contracts.MAX_SPEC_VERIFY_TREE_TOKENS])
def test_base_accepts_verify_budget_up_to_tree_limit(verify_tokens):
    """Given a spec base build whose verify budget is at or below the shared tree
    limit, When the arguments are validated, Then validation passes, so the
    documented 128-token JetSpec recipe still builds."""
    args = BuildArgs(model_dir="target",
                     engine_dir="engine",
                     spec_role="base",
                     spec_type="mtp",
                     max_spec_verify_tokens=verify_tokens)

    args.validate()


def test_base_rejects_verify_budget_past_tree_limit():
    """Given a spec base build whose verify budget exceeds the shared tree limit
    by one token, When the arguments are validated, Then validation fails naming
    the limit, so no engine is built that no tree verifier or decode kernel can
    serve."""
    args = BuildArgs(
        model_dir="target",
        engine_dir="engine",
        spec_role="base",
        spec_type="mtp",
        max_spec_verify_tokens=contracts.MAX_SPEC_VERIFY_TREE_TOKENS + 1)

    with pytest.raises(ValueError, match="tree verification limit"):
        args.validate()


def test_direct_spec_plan_passes_target_validation_mode_by_keyword(
        monkeypatch):
    """Protect the direct builder's keyword-only validator boundary."""
    from tensorrt_edgellm.builder import cli

    args = SimpleNamespace(spec_type="jetspec")
    bundle = SimpleNamespace(
        component_dict=lambda component: {"model_type": "qwen3"})

    def validate(config, *, mode):
        assert config == {"model_type": "qwen3"}
        assert mode == "jetspec"
        raise RuntimeError("validation boundary reached")

    monkeypatch.setattr(cli, "validate_external_speculative_target", validate)

    # Given an external speculative direct-build request, when its plan is
    # validated, then the selected mode crosses the keyword-only API boundary.
    with pytest.raises(RuntimeError, match="validation boundary reached"):
        cli._speculative_build_plan(args, bundle, (contracts.Component.LLM, ))


def test_dflash_v1_base_is_tree_capable_for_chain_replay(monkeypatch):
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
        build_args=SimpleNamespace(spec_type="dflash"),
    )

    assert config.dflash_base is True
    assert not hasattr(config, "dflash_tree_base")


def test_dspark_base_preserves_requested_tree_mode(monkeypatch):
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
        build_args=SimpleNamespace(max_spec_verify_tokens=7,
                                   max_spec_draft_tokens=7,
                                   reduced_vocab_dir="",
                                   draft_reduced_vocab_dir=""),
    )

    assert config.dspark_base is True
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
        max_spec_verify_tokens=3,
        max_spec_draft_tokens=2,
    )

    runtime = build_runtime_config(config, args)

    assert runtime["dspark_config"]["causal_head"] is True
    assert runtime["dspark_config"]["contiguous_query_swa"] is True
    assert runtime["dspark_config"]["sample_from_anchor"] is False


def test_dspark_draft_runtime_config_serializes_physical_query_capacity():
    # Given a false-anchor draft whose capacity includes its leading mask row,
    # when runtime metadata is emitted, then the physical capacity is preserved
    # exactly instead of being converted back to a proposal-only count.
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
        max_spec_verify_tokens=9,
        max_spec_draft_tokens=9,
    )

    runtime = build_runtime_config(config, args)

    assert runtime["builder_config"]["max_spec_draft_tokens"] == 9


def test_dspark_direct_sidecar_preserves_nvfp4_markov_w2(tmp_path):
    """Keep checkpoint W2 nibbles and scales intact for native W4A16 execution."""
    weight_key = "markov_head.markov_w2.weight"
    scale_key, scale_2_key = dspark_artifacts._NVFP4_HEAD_SCALES["markov_w2"]
    packed = np.zeros((2, 8), dtype=np.uint8)
    packed[:, 0] = 0x21
    scale = np.full((2, 1), 0x38, dtype=np.uint8)
    scale_2 = np.array([2.0], dtype=np.float32)
    checkpoint = tmp_path / "model.safetensors"
    save_safetensors(
        str(checkpoint), {
            "markov_head.markov_w1.weight": np.ones((2, 16), dtype=np.float16),
            weight_key: packed,
            scale_key: scale,
            scale_2_key: scale_2,
        },
        dtype_overrides={scale_key: "F8_E4M3"})

    config = SimpleNamespace(
        quant=SimpleNamespace(module_group_size=lambda _: 16),
        dspark_enable_confidence_head=False,
        dspark_markov_head_type="vanilla",
        dspark_markov_rank=16,
        dspark_confidence_head_with_markov=False,
    )
    args = SimpleNamespace(model_dir=str(tmp_path),
                           resolved_component=contracts.Component.LLM)
    output = tmp_path / "engine"
    output.mkdir()
    dspark_artifacts._write_head_sidecars(config, args, str(output))
    with SafetensorsStore(str(output)) as store:
        np.testing.assert_array_equal(store.get_packed_fp4("packed_weight"),
                                      packed)
        np.testing.assert_array_equal(store.get_fp8_bytes("block_scales"),
                                      scale)
        np.testing.assert_array_equal(store.get_f32("global_scale"), scale_2)
        assert not store.has("markov_w2")
    info = json.loads((output / "dspark_heads_info.json").read_text())
    assert info["w2_format"] == "nvfp4"


def test_w4a16_nvfp4_is_weight_only_for_every_model():
    algorithm = "W4A16_NVFP4"
    assert quantization.algorithm_to_type(
        algorithm) == quantization.QUANT_NVFP4_A16
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


def test_direct_a16_gated_moe_matches_reference():
    torch = pytest.importorskip("torch")
    from tensorrt_edgellm.checkpoint.repacking import \
        repack_nvfp4_a16_marlin_gated_moe_experts

    rng = np.random.default_rng(37)
    experts = []
    for _ in range(2):
        expert = {}
        for projection in ("gate", "up"):
            expert[projection] = {
                "packed": rng.integers(0, 256, (144, 64), dtype=np.uint8),
                "sf": rng.integers(0, 256, (144, 8), dtype=np.uint8),
                "alpha": np.float32(0.03125),
            }
        expert["down"] = {
            "packed": rng.integers(0, 256, (128, 72), dtype=np.uint8),
            "sf": rng.integers(0, 256, (128, 9), dtype=np.uint8),
            "alpha": np.float32(0.015625),
        }
        experts.append(expert)

    direct = direct_nvfp4.pack_nvfp4_a16_gated_experts(experts.__getitem__, 2,
                                                       128, 144)
    reference = repack_nvfp4_a16_marlin_gated_moe_experts(
        [torch.from_numpy(expert["gate"]["packed"]) for expert in experts], [
            torch.from_numpy(expert["gate"]["sf"].view(np.int8))
            for expert in experts
        ], [torch.tensor([expert["gate"]["alpha"]]) for expert in experts],
        [torch.from_numpy(expert["up"]["packed"]) for expert in experts], [
            torch.from_numpy(expert["up"]["sf"].view(np.int8))
            for expert in experts
        ], [torch.tensor([expert["up"]["alpha"]]) for expert in experts],
        [torch.from_numpy(expert["down"]["packed"]) for expert in experts], [
            torch.from_numpy(expert["down"]["sf"].view(np.int8))
            for expert in experts
        ], [torch.tensor([expert["down"]["alpha"]])
            for expert in experts], 256)
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
            dspark_artifacts._load_head_tensor(store, "markov_w2", weight_key)


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
    tensors = list(range(16))
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
        spec_verify_layout=tensors[15],
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
    assert "use_ddtree" not in captured["attributes"]


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
        draft, None, None)

    assert draft_size == 7
    assert verify_size == 7


def test_dspark_omitted_target_capacity_is_independent_of_short_draft():
    # Given a block-seven DSpark checkpoint and an explicit four-row draft,
    # when target capacity is omitted, then the unified target still covers
    # block plus root so the same artifact can later select chain or tree.
    draft = {"dspark_config": {"block_size": 7}}

    verify_size, draft_size = dspark_config.resolve_build_profile(
        draft, 4, None)

    assert draft_size == 4
    assert verify_size == 8


def test_dspark_profile_resolution_defaults_legacy_checkpoint_to_anchor_slot():
    draft = {"dflash_config": {"block_size": 8}}

    verify_size, draft_size = dspark_config.resolve_build_profile(
        draft, None, None)

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
        draft, None, None)

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
        draft, None, None)

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
        "tensorrt_edgellm.builder.core.config.BundleConfig.from_pretrained",
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
        max_spec_verify_tokens=3,
        max_spec_draft_tokens=2,
        reduced_vocab_dir="",
        draft_reduced_vocab_dir="",
    )

    dspark_config._validate_runtime_contract(config, args,
                                             contracts.SpecRole.DRAFT)


def test_dspark_non_anchor_draft_rejects_mask_only_capacity():
    config = _target_config()
    config.dspark_block_size = 8
    config.dspark_mask_token_id = 990
    config.dspark_markov_head_type = "vanilla"
    config.dspark_sample_from_anchor = False
    args = SimpleNamespace(
        resolved_spec_role=contracts.SpecRole.DRAFT,
        max_spec_verify_tokens=2,
        max_spec_draft_tokens=1,
        reduced_vocab_dir="",
        draft_reduced_vocab_dir="",
    )

    with pytest.raises(ValueError, match="proposal token"):
        dspark_config._validate_runtime_contract(config, args,
                                                 contracts.SpecRole.DRAFT)


@pytest.mark.parametrize("frontend,flag", [
    ("official", "--mtp-tree-base"),
    ("official", "--dflash-tree-base"),
    ("official", "--jetspec-tree-base"),
    ("official", "--dspark-tree-base"),
    ("direct", "--tree-base"),
])
def test_target_cli_rejects_topology_artifact_options(monkeypatch, capsys,
                                                      frontend, flag):
    # Given one target artifact for both layouts, when a topology export option is supplied,
    # then argument parsing rejects it before reading a checkpoint.
    if frontend == "official":
        import sys

        from tensorrt_edgellm.onnx import cli as export
        monkeypatch.setattr(sys, "argv",
                            ["export", "checkpoint", "output", flag])
        monkeypatch.setattr(
            export, "_resolve_model_dir", lambda _: pytest.fail(
                "obsolete option reached checkpoint resolution"))
        parse = export.main
    else:
        from tensorrt_edgellm.builder.cli import _parser
        parse = lambda: _parser().parse_args(
            ["--model-dir", "checkpoint", "--engine-dir", "output", flag])
    with pytest.raises(SystemExit):
        parse()
    assert f"unrecognized arguments: {flag}" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["dflash", "jetspec", "dspark"])
def test_ci_target_artifact_paths_ignore_runtime_topology(mode):
    # Given identical checkpoint/precision/capacity, when runtime topology changes,
    # then CI resolves the same regenerated target ONNX and engine artifacts.
    from defs.config import TaskType, TestConfig
    config = SimpleNamespace(is_mtp=False,
                             is_eagle=False,
                             is_dflash=mode == "dflash",
                             is_jetspec=mode == "jetspec",
                             is_domino=False,
                             is_dspark=mode == "dspark",
                             is_dflash_tree=False,
                             is_jetspec_tree=False,
                             is_dspark_tree=False,
                             engine_dir="engines",
                             task_type=TaskType.BUILD,
                             draft_model_id="paired",
                             draft_llm_precision="fp16",
                             get_onnx_base_dir=lambda: "onnx",
                             _get_llm_onnx_model_id=lambda: "fp16-fp16",
                             get_engine_base_dir=lambda: "engines",
                             get_engine_id=lambda: "capacity-profile")
    chain = (TestConfig.get_llm_onnx_dir(config),
             TestConfig.get_llm_engine_dir(config))
    setattr(config, f"is_{mode}_tree", True)
    tree = (TestConfig.get_llm_onnx_dir(config),
            TestConfig.get_llm_engine_dir(config))
    assert chain == tree
