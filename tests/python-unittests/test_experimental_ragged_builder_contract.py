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
"""Packaged-builder contracts for the unified token-major decoder ABI."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

pytest.importorskip("tensorrt")

from experimental.builder.core import contracts
from experimental.builder.core.artifacts.runtime_config import \
    build_runtime_config
from experimental.builder.core.builder import (_setup_diffusion_profiles,
                                               _setup_llm_profiles)
from experimental.builder.core.config import (LAYER_ATTN, LAYER_GDN,
                                              LAYER_MAMBA, DeviceConfig,
                                              GdnConfig, MambaConfig)
from experimental.builder.core.ragged import (checked_kv_pool_pages,
                                              decoder_profile_ranges)
from experimental.builder.models.alpamayo.modeling_alpamayo_text import \
    AlpamayoForCausalLM
from experimental.builder.models.dflash.modeling_dflash_draft import \
    DFlashDraftModel
from experimental.builder.models.diffusion_gemma.modeling_diffusion_gemma import \
    DiffusionGemmaForBlockDiffusion
from experimental.builder.models.diffusion_gemma.runtime_config import \
    _diffusion_config
from experimental.builder.models.dspark.configuration import \
    configure_draft as configure_dspark_draft
from experimental.builder.models.dspark.modeling_dspark_draft import \
    DSparkDraftModel
from experimental.builder.models.eagle3.modeling_eagle3_draft import \
    Eagle3DraftModel
from experimental.builder.models.gemma4.modeling_gemma4_assistant import \
    Gemma4AssistantForCausalLM
from experimental.builder.models.gemma4.modeling_gemma4_text import \
    Gemma4ForCausalLM
from experimental.builder.models.llama.modeling_llama import LlamaForCausalLM
from experimental.builder.models.muse_glimmer.modeling_muse_glimmer_text import \
    MuseGlimmerForCausalLM
from experimental.builder.models.nemotron_h.modeling_nemotron_h import \
    NemotronHForCausalLM
from experimental.builder.models.nemotron_omni.modeling_nemotron_omni_text import \
    NemotronOmniCausalLM
from experimental.builder.models.qwen3_5.modeling_qwen3_5_mtp import \
    Qwen35MtpDraftModel
from experimental.builder.models.qwen3_5.modeling_qwen3_5_text import \
    Qwen3_5ForCausalLM
from experimental.builder.models.qwen3_5_moe.modeling_qwen3_5_moe import \
    Qwen3_5MoeForCausalLM
from experimental.builder.models.qwen3_moe.modeling_qwen3_moe import \
    Qwen3MoeForCausalLM
from experimental.builder.models.qwen3_omni.modeling_qwen3_omni_code_predictor import \
    Qwen3OmniCodePredictor
from experimental.builder.models.qwen3_omni.modeling_qwen3_omni_moe_talker import \
    Qwen3OmniMoeTalker
from experimental.builder.models.qwen3_omni.modeling_qwen3_omni_moe_text import \
    Qwen3OmniMoeThinker
from experimental.builder.models.qwen3_omni.modeling_qwen3_omni_talker import \
    Qwen3OmniTalker
from experimental.builder.models.qwen3_omni_next.modeling_qwen3_omni_next_talker import \
    Qwen3OmniNextTalker
from experimental.builder.models.qwen3_omni_next.modeling_qwen3_omni_next_text import \
    Qwen3OmniNextThinker
from experimental.builder.models.qwen3_tts.modeling_qwen3_tts_code_predictor import \
    Qwen3TTSCodePredictor
from experimental.builder.models.qwen3_tts.modeling_qwen3_tts_talker import \
    Qwen3TTSTalker
from experimental.builder.ops import BuildContext, BuildOptions, NetworkModule
from experimental.builder.ops.ragged import add_ragged_decoder_inputs
from experimental.builder.ops.scope import build_scope

_COMMON_DECODER_METADATA = {
    "positions",
    "query_start_offsets",
    "query_lengths",
    "past_lengths",
    "attention_sequence_lengths",
    "state_indices",
    "execution_phase_marker",
    "context_sequence_count_carrier",
    "kv_page_table",
}
_COMMON_DECODER_BINDINGS = _COMMON_DECODER_METADATA | {"logits_indices"}
_LEGACY_DECODER_BINDINGS = {
    "kvcache_start_index",
    "spec_verify_phase_marker",
}


@dataclass
class _RecordedInput:
    name: str
    dtype: object
    shape: tuple


class _InputContractNet:
    """Record real NetworkModule input declarations without initializing CUDA."""

    def __init__(self):
        self.inputs = {}

    def add_input(self, name, dtype, shape):
        tensor = _RecordedInput(name, dtype, tuple(shape))
        self.inputs[name] = tensor
        return tensor

    @staticmethod
    def operation_attributes(name):
        if name in ("causal_conv1d", "gated_delta_net"):
            return frozenset(("use_ddtree", ))
        return frozenset()


class _ProfileNetwork:

    def __init__(self, inputs):
        self._inputs = inputs

    @property
    def num_inputs(self):
        return len(self._inputs)

    def get_input(self, index):
        return self._inputs[index]


class _RecordedProfile:

    def __init__(self):
        self.shapes = {}

    def set_shape(self, name, minimum, optimum, maximum):
        self.shapes[name] = (minimum, optimum, maximum)


class _ProfileBuilder:

    @staticmethod
    def create_optimization_profile():
        return _RecordedProfile()


class _ProfileConfig:

    def __init__(self):
        self.profiles = []

    def add_optimization_profile(self, profile):
        self.profiles.append(profile)


def _dense_config(**overrides):
    values = {
        "model_type": "llama",
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "intermediate_size": 32,
        "vocab_size": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 128,
        "layer_types": [LAYER_ATTN],
    }
    values.update(overrides)
    return DeviceConfig(**values)


def _hybrid_config(**overrides):
    values = {
        "model_type":
        "qwen3_5_text",
        "layer_types": [LAYER_ATTN, LAYER_GDN],
        "num_hidden_layers":
        2,
        "gdn_cfg":
        GdnConfig(num_key_heads=2,
                  num_value_heads=2,
                  key_head_dim=4,
                  value_head_dim=4,
                  conv_kernel=4),
    }
    values.update(overrides)
    return _dense_config(**values)


def _mamba_config(**overrides):
    values = {
        "model_type":
        "nemotron_h",
        "layer_types": [LAYER_ATTN, LAYER_MAMBA],
        "num_hidden_layers":
        2,
        "mamba_cfg":
        MambaConfig(num_heads=2,
                    head_dim=4,
                    ssm_state_size=8,
                    conv_dim=16,
                    conv_kernel=4,
                    n_groups=1),
    }
    values.update(overrides)
    return _dense_config(**values)


def _build_args(**overrides):
    values = {
        "model_dir": "",
        "resolved_component": contracts.Component.LLM,
        "resolved_spec_role": contracts.SpecRole.NONE,
        "spec_type": "none",
        "max_batch_size": 3,
        "max_input_len": 17,
        "max_kv_cache_capacity": 64,
        "max_verify_tree_size": 9,
        "max_draft_tree_size": 7,
        "max_lora_rank": 0,
        "tp_size": 1,
        "reduced_vocab_dir": "",
        "draft_reduced_vocab_dir": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _network_inputs(model_class, cfg):
    net = _InputContractNet()
    context = BuildContext(net=net,
                           cfg=cfg,
                           weights=SimpleNamespace(),
                           options=BuildOptions(),
                           bundle=SimpleNamespace(),
                           args=SimpleNamespace())
    model = model_class.__new__(model_class)
    NetworkModule.__init__(model, context)
    model.qwen35 = False
    if model_class is Qwen3_5ForCausalLM:
        model.qwen35 = True
    with build_scope(context):
        model.input_tensors()
    return net.inputs


def _assert_decoder_contract(inputs,
                             required_portals=(),
                             hidden_inputs=(),
                             include_logits_indices=True):
    names = set(inputs)
    expected = (_COMMON_DECODER_BINDINGS
                if include_logits_indices else _COMMON_DECODER_METADATA)
    missing = sorted((expected | set(required_portals)) - names)
    unexpected = (["logits_indices"] if not include_logits_indices
                  and "logits_indices" in names else [])
    legacy = sorted(_LEGACY_DECODER_BINDINGS & names)
    wrong_ranks = {
        name: len(inputs[name].shape)
        for name in ("inputs_embeds", "rope_rotary_cos_sin", *hidden_inputs)
        if name in inputs and len(inputs[name].shape) != 2
    }
    declared = []
    for value in inputs.values():
        declared.extend(value if isinstance(value, list) else (value, ))
    declared_names = {value.name for value in declared if value is not None}
    legacy.extend(sorted(_LEGACY_DECODER_BINDINGS & declared_names))
    token_prefixes = (
        "inputs_embeds",
        "rope_rotary_cos_sin",
        "hidden_states_",
        "dflash_target_hidden_concat",
        "deepstack_embeds_",
        "ple_token_embeds_",
        "prev_self_conditioning_embeds",
    )
    wrong_ranks.update({
        value.name: len(value.shape)
        for value in declared if value is not None
        and value.name.startswith(token_prefixes) and len(value.shape) != 2
    })
    if "attention_mask" in inputs and inputs["attention_mask"] is not None:
        assert inputs["attention_mask"].name == "packed_attention_mask"
    if "attention_pos_id" in inputs and inputs["attention_pos_id"] is not None:
        assert inputs["attention_pos_id"].name == "attention_position_ids"
    assert not (missing or unexpected or legacy or wrong_ranks), {
        "missing": missing,
        "unexpected": unexpected,
        "legacy": legacy,
        "wrong_token_aligned_ranks": wrong_ranks,
    }


def test_dense_runtime_config_serializes_only_ragged_backend_contract():
    cfg = _dense_config()
    args = _build_args()

    builder = build_runtime_config(cfg, args)["builder_config"]

    assert builder == {
        "tp_size": 1,
        "max_input_len": 17,
        "spec_draft": False,
        "spec_base": False,
        "max_batch_size": 3,
        "max_lora_rank": 0,
        "max_kv_cache_capacity": 64,
        "max_kv_pool_pages": 3,
        "ragged_backend": "entry_padded_compatibility",
    }


@pytest.mark.parametrize("role",
                         (contracts.SpecRole.BASE, contracts.SpecRole.DRAFT))
def test_spec_runtime_capacity_covers_default_generation_profile(role):
    args = _build_args(resolved_spec_role=role,
                       spec_type="mtp",
                       max_batch_size=2,
                       max_input_len=32,
                       max_verify_tree_size=60,
                       max_draft_tree_size=60)

    prefill, generation = decoder_profile_ranges(args)

    assert prefill.physical_tokens[2] == 64
    assert generation.physical_tokens[2] == 120


def test_diffusion_runtime_capacity_covers_default_canvas_profile():
    cfg = _dense_config(raw_root={"canvas_length": 256})
    args = _build_args(resolved_component=contracts.Component.DLLM,
                       max_batch_size=2,
                       max_input_len=32)
    network = _ProfileNetwork(
        [_RecordedInput("inputs_embeds", None, (-1, 16))])
    profile_config = _ProfileConfig()

    builder = build_runtime_config(cfg, args)["builder_config"]
    _setup_diffusion_profiles(_ProfileBuilder(), profile_config, network, cfg,
                              args)

    assert builder["ragged_backend"] == "entry_padded_compatibility"
    assert profile_config.profiles[0].shapes["inputs_embeds"][2] == (64, 16)
    assert profile_config.profiles[1].shapes["inputs_embeds"][2] == (512, 16)


def test_generation_config_canvas_is_shared_by_all_diffusion_artifacts(
        tmp_path):
    (tmp_path / "generation_config.json").write_text('{"canvas_length": 512}')
    cfg = _dense_config(raw_root={})
    args = _build_args(model_dir=str(tmp_path),
                       resolved_component=contracts.Component.DLLM,
                       max_batch_size=2,
                       max_input_len=32)
    network = _ProfileNetwork(
        [_RecordedInput("inputs_embeds", None, (-1, 16))])
    profile_config = _ProfileConfig()

    builder = build_runtime_config(cfg, args)["builder_config"]
    _setup_diffusion_profiles(_ProfileBuilder(), profile_config, network, cfg,
                              args)
    diffusion = _diffusion_config({}, {"canvas_length": 512})

    assert builder["ragged_backend"] == "entry_padded_compatibility"
    assert profile_config.profiles[1].shapes["inputs_embeds"][2] == (1024, 16)
    assert diffusion["canvas_length"] == 512


def test_root_canvas_precedes_generation_config_for_all_diffusion_artifacts(
        tmp_path):
    (tmp_path / "generation_config.json").write_text('{"canvas_length": 512}')
    cfg = _dense_config(raw_root={"canvas_length": 128})
    args = _build_args(model_dir=str(tmp_path),
                       resolved_component=contracts.Component.DLLM,
                       max_batch_size=2,
                       max_input_len=32)
    network = _ProfileNetwork(
        [_RecordedInput("inputs_embeds", None, (-1, 16))])
    profile_config = _ProfileConfig()

    builder = build_runtime_config(cfg, args)["builder_config"]
    _setup_diffusion_profiles(_ProfileBuilder(), profile_config, network, cfg,
                              args)
    diffusion = _diffusion_config({"canvas_length": 128},
                                  {"canvas_length": 512})

    assert builder["ragged_backend"] == "entry_padded_compatibility"
    assert profile_config.profiles[1].shapes["inputs_embeds"][2] == (256, 16)
    assert diffusion["canvas_length"] == 128


def test_vanilla_profiles_track_max_input_length():
    prefill, generation = decoder_profile_ranges(
        _build_args(max_batch_size=2, max_input_len=32))

    assert prefill.physical_tokens[2] == 64
    assert generation.physical_tokens[2] == 2


def test_explicit_larger_input_dominates_spec_prefill_capacity():
    args = _build_args(resolved_spec_role=contracts.SpecRole.BASE,
                       spec_type="mtp",
                       max_batch_size=2,
                       max_input_len=96,
                       max_verify_tree_size=60)

    prefill, generation = decoder_profile_ranges(args)

    assert prefill.physical_tokens[2] == 192
    assert generation.physical_tokens[2] == 120


def test_spec_generation_capacity_overflow_is_rejected_actionably():
    args = _build_args(resolved_spec_role=contracts.SpecRole.BASE,
                       spec_type="mtp",
                       max_batch_size=(1 << 31) // 60 + 1,
                       max_input_len=32,
                       max_verify_tree_size=60)

    with pytest.raises(ValueError,
                       match="ragged physical-token capacity exceeds int32"):
        decoder_profile_ranges(args)


def test_diffusion_canvas_capacity_is_validated_actionably():
    args = _build_args(resolved_component=contracts.Component.DLLM,
                       max_input_len=32)
    cfg = _dense_config(raw_root={"canvas_length": 0})
    network = _ProfileNetwork(
        [_RecordedInput("inputs_embeds", None, (-1, 16))])

    with pytest.raises(ValueError,
                       match="canvas_length must fit a positive int32"):
        _setup_diffusion_profiles(_ProfileBuilder(), _ProfileConfig(), network,
                                  cfg, args)


def test_generation_config_canvas_is_validated_actionably(tmp_path):
    (tmp_path / "generation_config.json").write_text('{"canvas_length": 0}')
    args = _build_args(model_dir=str(tmp_path),
                       resolved_component=contracts.Component.DLLM,
                       max_input_len=32)
    cfg = _dense_config(raw_root={})
    network = _ProfileNetwork(
        [_RecordedInput("inputs_embeds", None, (-1, 16))])

    with pytest.raises(ValueError,
                       match="canvas_length must fit a positive int32"):
        _setup_diffusion_profiles(_ProfileBuilder(), _ProfileConfig(), network,
                                  cfg, args)


@pytest.mark.parametrize(("root", "generation"), (
    ({
        "canvas_length": 0
    }, {
        "canvas_length": 512
    }),
    ({}, {
        "canvas_length": 0
    }),
    ({}, {
        "canvas_length": 1 << 31
    }),
))
def test_diffusion_runtime_artifact_validates_shared_canvas(root, generation):
    with pytest.raises(ValueError,
                       match="canvas_length must fit a positive int32"):
        _diffusion_config(root, generation)


def test_generation_config_canvas_capacity_overflow_is_rejected(tmp_path):
    (tmp_path / "generation_config.json").write_text('{"canvas_length": 512}')
    args = _build_args(model_dir=str(tmp_path),
                       resolved_component=contracts.Component.DLLM,
                       max_batch_size=(1 << 31) // 512 + 1,
                       max_input_len=32)
    cfg = _dense_config(raw_root={})
    network = _ProfileNetwork(
        [_RecordedInput("inputs_embeds", None, (-1, 16))])

    with pytest.raises(ValueError,
                       match="ragged physical-token capacity exceeds int32"):
        _setup_diffusion_profiles(_ProfileBuilder(), _ProfileConfig(), network,
                                  cfg, args)


def test_hybrid_multimodal_runtime_config_does_not_duplicate_capabilities():
    cfg = _hybrid_config(
        num_deepstack_features=2,
        raw_root={
            "image_token_id": 7,
            "use_vision_bidirectional_attention": True,
        },
    )

    builder = build_runtime_config(cfg, _build_args())["builder_config"]

    assert builder["ragged_backend"] == "entry_padded_compatibility"
    assert not any(key.endswith("_supported") for key in builder)


def test_hybrid_runtime_config_declares_recurrent_commit_representation():
    gdn = build_runtime_config(_hybrid_config(), _build_args())
    mamba = build_runtime_config(_mamba_config(), _build_args())

    assert gdn["recurrent_spec_verify_mode"] == "snapshot"
    assert mamba["recurrent_spec_verify_mode"] == "replay"
    assert mamba["recurrent_state_num_groups"] == 1


def test_runtime_config_does_not_add_capabilities_after_model_augmentation():
    config = build_runtime_config(_dense_config(), _build_args())

    config["image_token_id"] = 42

    assert not any(
        key.endswith("_supported") for key in config["builder_config"])


def test_runtime_config_rejects_physical_token_capacity_overflow():
    args = _build_args(max_batch_size=1 << 29, max_input_len=5)

    with pytest.raises(ValueError,
                       match="physical-token capacity exceeds int32"):
        decoder_profile_ranges(args)


def test_kv_pool_profiles_reject_capacity_that_cannot_align_in_int32():
    with pytest.raises(ValueError, match="int32 page limit"):
        checked_kv_pool_pages(1, (1 << 31) - 1)


def test_decoder_profiles_use_token_and_sequence_address_spaces():
    prefill, decode = decoder_profile_ranges(_build_args())

    assert prefill.token(16) == ((1, 16), (24, 16), (51, 16))
    assert prefill.sequence() == ((1, ), (3, ), (3, ))
    assert prefill.offsets() == ((2, ), (4, ), (4, ))
    assert decode.token(16) == ((1, 16), (3, 16), (3, 16))


def test_draft_profiles_bound_entry_padded_proposal_rows():
    args = _build_args(resolved_spec_role=contracts.SpecRole.DRAFT,
                       spec_type="mtp")

    _, proposal = decoder_profile_ranges(args)

    assert proposal.token() == ((1, ), (21, ), (21, ))
    assert proposal.logits_rows == (1, 21, 21)


def test_spec_prefill_profiles_keep_sequence_and_logits_extents_distinct():
    args = _build_args(resolved_spec_role=contracts.SpecRole.DRAFT,
                       spec_type="mtp")

    prefill, _ = decoder_profile_ranges(args)

    assert prefill.sequence() == ((1, ), (3, ), (3, ))
    assert prefill.logits_rows == (1, 24, 51)


def test_common_decoder_input_bundle_declares_exact_mandatory_contract():
    net = _InputContractNet()

    inputs = add_ragged_decoder_inputs(net.add_input).as_dict()

    assert set(inputs) == _COMMON_DECODER_BINDINGS
    assert inputs["kv_page_table"].shape == (-1, 2, -1)
    assert all(
        len(tensor.shape) == 1 for name, tensor in inputs.items()
        if name != "kv_page_table")


def test_common_decoder_input_bundle_can_omit_role_specific_logits_selector():
    net = _InputContractNet()

    inputs = add_ragged_decoder_inputs(net.add_input,
                                       include_logits_indices=False).as_dict()

    assert set(inputs) == _COMMON_DECODER_METADATA
    assert "logits_indices" not in net.inputs


def test_block_draft_profiles_separate_proposal_and_delta_address_spaces():
    names_and_shapes = {
        "inputs_embeds": (-1, 16),
        "rope_rotary_cos_sin": (-1, 4),
        "positions": (-1, ),
        "query_start_offsets": (-1, ),
        "query_lengths": (-1, ),
        "past_lengths": (-1, ),
        "attention_sequence_lengths": (-1, ),
        "state_indices": (-1, ),
        "execution_phase_marker": (-1, ),
        "context_sequence_count_carrier": (-1, ),
        "kv_page_table": (-1, 2, -1),
        "skip_softmax_scale": (-1, ),
        "packed_attention_mask": (-1, -1),
        "dflash_target_hidden_concat": (-1, 80),
        "dflash_delta_rope_cos_sin": (-1, 4),
        "dflash_delta_positions": (-1, ),
        "dflash_delta_token_to_sequence": (-1, ),
        "dflash_delta_lengths": (-1, ),
    }
    network = _ProfileNetwork([
        _RecordedInput(name, None, shape)
        for name, shape in names_and_shapes.items()
    ])
    config = _ProfileConfig()
    args = _build_args(resolved_spec_role=contracts.SpecRole.DRAFT,
                       spec_type="dflash")

    _setup_llm_profiles(_ProfileBuilder(), config, network, _dense_config(),
                        args)

    context, generation = config.profiles
    assert context.shapes["inputs_embeds"] == ((1, 16), (21, 16), (21, 16))
    assert generation.shapes["inputs_embeds"] == ((1, 16), (21, 16), (21, 16))
    assert context.shapes["dflash_target_hidden_concat"] == ((1, 80), (24, 80),
                                                             (51, 80))
    assert generation.shapes["dflash_target_hidden_concat"] == ((1, 80),
                                                                (24, 80), (24,
                                                                           80))
    assert context.shapes["dflash_delta_rope_cos_sin"] == ((1, 4), (24, 4),
                                                           (51, 4))
    assert generation.shapes["dflash_delta_rope_cos_sin"] == ((1, 4), (24, 4),
                                                              (24, 4))
    assert context.shapes["execution_phase_marker"][1] == (4, )
    assert context.shapes["context_sequence_count_carrier"] == ((0, ), (0, ),
                                                                (0, ))
    assert generation.shapes["context_sequence_count_carrier"] == ((0, ),
                                                                   (0, ),
                                                                   (0, ))
    assert context.shapes["skip_softmax_scale"] == ((0, ), (0, ), (64, ))
    assert generation.shapes["skip_softmax_scale"] == ((0, ), (0, ), (64, ))


def test_dspark_sample_without_anchor_uses_resolved_physical_capacity():
    network = _ProfileNetwork([
        _RecordedInput("inputs_embeds", None, (-1, 16)),
        _RecordedInput("positions", None, (-1, )),
        _RecordedInput("rope_rotary_cos_sin", None, (-1, 4)),
        _RecordedInput("packed_attention_mask", None, (-1, -1)),
    ])
    config = _ProfileConfig()
    cfg = _dense_config(dspark_sample_from_anchor=False)
    args = _build_args(resolved_spec_role=contracts.SpecRole.DRAFT,
                       spec_type="dspark")

    _setup_llm_profiles(_ProfileBuilder(), config, network, cfg, args)

    for profile in config.profiles:
        assert profile.shapes["inputs_embeds"] == ((1, 16), (21, 16), (21, 16))
        assert profile.shapes["positions"] == ((1, ), (21, ), (21, ))
        assert profile.shapes["rope_rotary_cos_sin"] == ((1, 4), (21, 4), (21,
                                                                           4))
        assert profile.shapes["packed_attention_mask"] == ((1, 1), (21, 1),
                                                           (21, 1))


def test_dspark_runtime_config_preserves_execution_contract():
    cfg = _dense_config(dspark_target_layer_ids=[0],
                        dspark_causal_proposal=True,
                        dspark_contiguous_query_swa=False,
                        dspark_sample_from_anchor=False)
    args = _build_args(resolved_spec_role=contracts.SpecRole.DRAFT,
                       spec_type="dspark")

    dspark = build_runtime_config(cfg, args)["dspark_config"]

    assert dspark["causal_head"] is True
    assert dspark["contiguous_query_swa"] is False
    assert dspark["sample_from_anchor"] is False


def test_experimental_dspark_accepts_shared_kv_and_value_norm():
    args = _build_args(resolved_spec_role=contracts.SpecRole.DRAFT,
                       spec_type="dspark")
    target = _dense_config()
    cfg = _dense_config(dspark_target_layer_ids=[0],
                        dspark_markov_rank=1,
                        attention_k_eq_v=True,
                        has_value_norm=True)

    configure_dspark_draft(cfg, paired_target=target, build_args=args)

    assert cfg.attention_k_eq_v is True
    assert cfg.has_value_norm is True


def test_ordinary_draft_profiles_use_context_prefill_then_proposal():
    network = _ProfileNetwork([
        _RecordedInput("context_sequence_count_carrier", None, (-1, )),
        _RecordedInput("execution_phase_marker", None, (-1, )),
    ])
    config = _ProfileConfig()
    args = _build_args(resolved_spec_role=contracts.SpecRole.DRAFT,
                       spec_type="mtp")

    _setup_llm_profiles(_ProfileBuilder(), config, network, _dense_config(),
                        args)

    assert config.profiles[0].shapes["context_sequence_count_carrier"] == ((
        1, ), (3, ), (3, ))
    assert config.profiles[1].shapes["context_sequence_count_carrier"] == ((
        0, ), (0, ), (0, ))
    assert config.profiles[0].shapes["execution_phase_marker"] == ((1, ),
                                                                   (1, ),
                                                                   (8, ))
    assert config.profiles[1].shapes["execution_phase_marker"] == ((1, ),
                                                                   (4, ),
                                                                   (8, ))


def test_diffusion_context_mask_selector_profiles_nonempty_optimum_batch():
    network = _ProfileNetwork(
        [_RecordedInput("context_mask_selector", None, (-1, ))])
    config = _ProfileConfig()

    _setup_diffusion_profiles(_ProfileBuilder(), config, network,
                              _dense_config(raw_root={"canvas_length": 11}),
                              _build_args())

    expected = ((0, ), (3, ), (3, ))
    assert config.profiles[0].shapes["context_mask_selector"] == expected
    assert config.profiles[1].shapes["context_mask_selector"] == expected


def test_diffusion_profiles_restore_phase_and_selection_bindings():
    network = _ProfileNetwork([
        _RecordedInput("phase_is_encoder", None, (-1, )),
        _RecordedInput("select_token_indices", None, (-1, )),
        _RecordedInput("execution_phase_marker", None, (-1, )),
        _RecordedInput("context_sequence_count_carrier", None, (-1, )),
    ])
    config = _ProfileConfig()

    _setup_diffusion_profiles(_ProfileBuilder(), config, network,
                              _dense_config(raw_root={"canvas_length": 11}),
                              _build_args())

    prefill, diffusion = config.profiles
    assert prefill.shapes["phase_is_encoder"] == ((1, ), (1, ), (1, ))
    assert diffusion.shapes["phase_is_encoder"] == ((1, ), (1, ), (1, ))
    assert prefill.shapes["select_token_indices"] == ((1, ), (24, ), (51, ))
    assert diffusion.shapes["select_token_indices"] == ((1, ), (33, ), (33, ))
    assert prefill.shapes["execution_phase_marker"] == ((1, ), (7, ), (8, ))
    assert diffusion.shapes["execution_phase_marker"] == ((1, ), (6, ), (8, ))
    assert prefill.shapes["context_sequence_count_carrier"] == ((0, ), (0, ),
                                                                (0, ))
    assert diffusion.shapes["context_sequence_count_carrier"] == ((0, ), (0, ),
                                                                  (0, ))


def test_packed_tree_mask_keeps_unit_minimum_width_above_one_word():
    network = _ProfileNetwork(
        [_RecordedInput("packed_attention_mask", None, (-1, -1))])
    config = _ProfileConfig()
    args = _build_args(resolved_spec_role=contracts.SpecRole.DRAFT,
                       spec_type="dflash",
                       max_draft_tree_size=60)

    _setup_llm_profiles(_ProfileBuilder(), config, network, _dense_config(),
                        args)

    expected = ((1, 1), (180, 2), (180, 2))
    assert config.profiles[0].shapes["packed_attention_mask"] == expected
    assert config.profiles[1].shapes["packed_attention_mask"] == expected


@pytest.mark.parametrize(
    ("model_class", "cfg"),
    (
        (LlamaForCausalLM, _dense_config()),
        (Qwen3_5ForCausalLM, _hybrid_config()),
    ),
    ids=("dense-text", "qwen3.5-hybrid"),
)
def test_decoder_network_inputs_use_common_token_major_contract(
        model_class, cfg):
    inputs = _network_inputs(model_class, cfg)

    _assert_decoder_contract(inputs)


def test_llama_tree_bindings_match_runtime_names_and_token_major_ranks():
    inputs = _network_inputs(LlamaForCausalLM,
                             _dense_config(engine_role="base"))

    assert inputs["attention_position_ids"].shape == (-1, )
    assert inputs["packed_attention_mask"].shape == (-1, -1)
    assert "attention_pos_id" not in inputs
    assert "attention_mask" not in inputs


def test_muse_glimmer_base_uses_dual_rope_and_ragged_tree_bindings():
    cfg = _dense_config(
        root_model_type="muse_glimmer",
        model_type="muse_glimmer_text",
        engine_role="base",
        spec_decode_type="dflash",
        dflash_tree_base=True,
        sliding_rope_config={"rope_type": "default"},
        full_rope_config={"rope_type": "nope"},
    )

    inputs = _network_inputs(MuseGlimmerForCausalLM, cfg)

    _assert_decoder_contract(
        inputs,
        required_portals=("rope_rotary_cos_sin_sliding",
                          "rope_rotary_cos_sin_full", "attention_position_ids",
                          "packed_attention_mask", "tree_parent_ids",
                          "tree_depths", "valid_tree_counts"))
    assert inputs["rope_rotary_cos_sin_sliding"].shape == (-1, 4)
    assert inputs["rope_rotary_cos_sin_full"].shape == (-1, 4)


@pytest.mark.parametrize(
    "model_class",
    (Qwen3_5MoeForCausalLM, Qwen3OmniNextThinker, Qwen3OmniNextTalker,
     NemotronHForCausalLM, NemotronOmniCausalLM),
)
def test_hybrid_network_inputs_share_common_token_major_contract(model_class):
    inputs = _network_inputs(model_class, _hybrid_config())

    _assert_decoder_contract(inputs)


def test_mtp_draft_network_uses_common_contract_and_hidden_portals():
    cfg = _hybrid_config(engine_role="draft", spec_decode_type="mtp")

    inputs = _network_inputs(Qwen35MtpDraftModel, cfg)

    _assert_decoder_contract(
        inputs,
        required_portals=("hidden_states_input", "hidden_states_from_draft",
                          "packed_attention_mask", "attention_position_ids"),
        hidden_inputs=("hidden_states_input", "hidden_states_from_draft"))


def test_qwen35_target_verify_network_uses_common_contract_and_tree_portals():
    cfg = _hybrid_config(engine_role="base",
                         spec_decode_type="mtp",
                         mtp_tree_base=True)

    inputs = _network_inputs(Qwen3_5ForCausalLM, cfg)

    _assert_decoder_contract(
        inputs,
        required_portals=("packed_attention_mask", "attention_position_ids",
                          "tree_parent_ids", "tree_depths",
                          "valid_tree_counts"))


@pytest.mark.parametrize(
    ("model_class", "cfg", "hidden_inputs"),
    (
        (AlpamayoForCausalLM, _dense_config(), ("deepstack_embeds", )),
        (Qwen3MoeForCausalLM, _dense_config(), ()),
        (Qwen3OmniMoeThinker, _dense_config(), ("deepstack_embeds", )),
        (Qwen3OmniMoeTalker, _dense_config(), ()),
        (Qwen3OmniTalker, _dense_config(), ()),
        (Qwen3OmniCodePredictor, _dense_config(), ()),
        (Qwen3TTSTalker, _dense_config(), ()),
        (Qwen3TTSCodePredictor, _dense_config(), ()),
        (Gemma4ForCausalLM, _dense_config(), ("ple", )),
        (Eagle3DraftModel,
         _dense_config(eagle3_target_layer_ids=[0], target_hidden_size=16),
         ("hidden_states_input", "hidden_states_from_draft")),
    ),
)
def test_specialized_decoder_networks_declare_ragged_token_major_inputs(
        model_class, cfg, hidden_inputs):
    inputs = _network_inputs(model_class, cfg)

    _assert_decoder_contract(inputs, hidden_inputs=hidden_inputs)


@pytest.mark.parametrize(
    ("model_class", "cfg", "hidden_inputs", "required_portals"),
    (
        (Gemma4AssistantForCausalLM, _dense_config(backbone_hidden_size=16),
         ("hidden_states_input", ), ()),
        (DFlashDraftModel, _dense_config(dflash_target_layer_ids=[0]),
         ("dflash_target_hidden_concat", ), ()),
        (DSparkDraftModel, _dense_config(dspark_target_layer_ids=[0]),
         ("dflash_target_hidden_concat", ), ()),
        (DiffusionGemmaForBlockDiffusion, _dense_config(),
         ("prev_self_conditioning_embeds", ),
         ("phase_is_encoder", "select_token_indices")),
    ),
    ids=("gemma4-assistant", "dflash", "dspark", "diffusion"),
)
def test_specialized_full_row_decoders_omit_logits_indices(
        model_class, cfg, hidden_inputs, required_portals):
    inputs = _network_inputs(model_class, cfg)

    _assert_decoder_contract(inputs,
                             required_portals=required_portals,
                             hidden_inputs=hidden_inputs,
                             include_logits_indices=False)
