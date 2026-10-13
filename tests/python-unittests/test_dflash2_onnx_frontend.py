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
from dataclasses import replace
from types import SimpleNamespace

import onnx
import pytest
import torch
from safetensors.torch import load_file, save_file

import tensorrt_edgellm.model as model_module
import tensorrt_edgellm.scripts.export as export_script
from tensorrt_edgellm.checkpoint.checkpoint_utils import \
    build_runtime_llm_config_dict
from tensorrt_edgellm.checkpoint.loader import load_weights
from tensorrt_edgellm.config import (QUANT_FP8, QUANT_NVFP4, QuantConfig,
                                     make_dflash2_draft_config)
from tensorrt_edgellm.dflash import DFlashVersion, resolve_dflash_contract
from tensorrt_edgellm.model import _inherit_dflash_lm_head_quant
from tensorrt_edgellm.models.dflash2 import DFlash2DraftModel
from tensorrt_edgellm.models.ops import dflash2_grouped_dynamic_conv
from tensorrt_edgellm.models.qwen3_5.modeling_qwen3_5_text import \
    _is_spec_tree_base_export
from tensorrt_edgellm.onnx.export import _export_model


class _TinyCheckpointModel(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = torch.nn.Parameter(torch.ones(4, 4))


class _TinyCheckpointModelWithDefault(_TinyCheckpointModel):

    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("default_bias", torch.zeros(4))

    def materialize_checkpoint_defaults(self, device):
        if self.default_bias.device.type == "meta":
            self.default_bias = torch.zeros(4, device=device)


def _write_dflash2_config(tmp_path, **dflash_overrides):
    dflash_config = {
        "target_layer_ids": [5, 19, 33, 47, 61],
        "block_size": 8,
        "mask_token_id": 248070,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
    }
    dflash_config.update(dflash_overrides)
    config = {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "hidden_size": 5120,
        "num_hidden_layers": 5,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 17408,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
        "rope_theta": 10000000,
        "rms_norm_eps": 1e-6,
        "is_causal": False,
        "dflash_config": dflash_config,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def _dflash2_contract_fields():
    return {
        "target_layer_ids": [5, 19, 33, 47, 61],
        "block_size": 16,
        "mask_token_id": 248070,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
    }


@pytest.mark.parametrize("nested", [False, True])
def test_dflash_version_is_resolved_from_root_or_nested_architecture(nested):
    root = {"is_causal": False}
    llm = {"is_causal": False}
    owner = llm if nested else root
    owner["architectures"] = ["DFlash2DraftModel"]
    owner["dflash_config"] = _dflash2_contract_fields()

    contract = resolve_dflash_contract(root, llm)

    assert contract.version == DFlashVersion.V2
    assert contract.block_size == 16
    assert contract.supports_probabilistic_sampling


def test_legacy_dflash_architecture_keeps_version_one():
    contract = resolve_dflash_contract(
        {
            "architectures": ["DFlashDraftModel"],
            "dflash_config": _dflash2_contract_fields(),
        }, {})

    assert contract.version == DFlashVersion.V1
    assert not contract.supports_probabilistic_sampling


@pytest.mark.parametrize("missing",
                         ["block_size", "mask_token_id", "is_causal"])
def test_dflash2_contract_requires_common_fields(missing):
    root = {
        "architectures": ["DFlash2DraftModel"],
        "dflash_config": _dflash2_contract_fields(),
        "is_causal": False,
    }
    if missing == "is_causal":
        del root[missing]
    else:
        del root["dflash_config"][missing]

    with pytest.raises(ValueError, match=missing):
        resolve_dflash_contract(root, {})


def test_low_memory_model_initialization_materializes_only_checkpoint_tensors(
        tmp_path):
    expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    save_file({"weight": expected}, str(tmp_path / "model.safetensors"))

    model = model_module._instantiate_model(_TinyCheckpointModel,
                                            SimpleNamespace(),
                                            device="cpu",
                                            low_cpu_mem_usage=True)

    assert model.weight.device.type == "meta"
    load_weights(model, str(tmp_path))
    assert model.weight.device.type == "cpu"
    torch.testing.assert_close(model.weight, expected)


def test_low_memory_model_initialization_restores_declared_checkpoint_defaults(
        tmp_path):
    expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    save_file({"weight": expected}, str(tmp_path / "model.safetensors"))
    model = model_module._instantiate_model(_TinyCheckpointModelWithDefault,
                                            SimpleNamespace(),
                                            device="cpu",
                                            low_cpu_mem_usage=True)
    load_weights(model, str(tmp_path))

    model_module._materialize_checkpoint_defaults(model, "cpu")

    torch.testing.assert_close(model.default_bias, torch.zeros(4))


def test_dflash2_config_preserves_the_production_checkpoint_contract(tmp_path):
    config = make_dflash2_draft_config(str(_write_dflash2_config(tmp_path)),
                                       lambda head_dim: head_dim**-0.5)

    assert config.is_dflash_draft
    assert config.dflash2_target_layer_ids == [5, 19, 33, 47, 61]
    assert config.dflash2_block_size == 8
    assert config.dflash2_mask_token_id == 248070
    assert config.dflash2_conv_kernel_size == 2
    assert config.dflash2_conv_group_size == 16
    assert config.dflash2_selector_rank == 256
    assert config.dflash2_selector_top_k == 16
    assert config.dflash2_is_causal is False
    assert config.dflash_version == DFlashVersion.V2


def test_dflash2_config_accepts_muse_block_size_16(tmp_path):
    config = make_dflash2_draft_config(
        str(_write_dflash2_config(tmp_path, block_size=16)),
        lambda head_dim: head_dim**-0.5)

    assert config.dflash2_block_size == 16


@pytest.mark.parametrize(("field", "bad_value"), [
    ("block_size", 17),
    ("conv_kernel_size", 3),
    ("conv_group_size", 8),
    ("selector_rank", 128),
    ("selector_top_k", 8),
])
def test_dflash2_config_rejects_incompatible_runtime_contract(
        tmp_path, field, bad_value):
    model_dir = _write_dflash2_config(tmp_path, **{field: bad_value})

    with pytest.raises(ValueError, match="DFlash V2"):
        make_dflash2_draft_config(str(model_dir),
                                  lambda head_dim: head_dim**-0.5)


def test_dflash2_runtime_config_is_self_describing(tmp_path):
    config = make_dflash2_draft_config(str(_write_dflash2_config(tmp_path)),
                                       lambda head_dim: head_dim**-0.5)
    runtime = build_runtime_llm_config_dict(SimpleNamespace(config=config))

    assert runtime["spec_decode_type"] == "dflash"
    assert runtime["engine_role"] == "draft"
    assert runtime["base_model_hidden_size"] == 25600
    assert runtime["dflash_config"] == {
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


def test_dflash2_base_does_not_emit_legacy_dflash_metadata(tmp_path):
    config = make_dflash2_draft_config(str(_write_dflash2_config(tmp_path)),
                                       lambda head_dim: head_dim**-0.5)
    config = replace(config, is_dflash_draft_flag=False, dflash_base=True)

    runtime = build_runtime_llm_config_dict(SimpleNamespace(config=config))

    assert runtime["spec_decode_type"] == "dflash"
    assert runtime["engine_role"] == "base"
    assert runtime["dflash_config"]["version"] == 2


def test_dflash_base_exports_tree_metadata_for_linear_and_branching_modes():
    config = SimpleNamespace(dflash_base=False,
                             dflash_tree_base=False,
                             jetspec_tree_base=False,
                             mtp_tree_base=False,
                             dflash_version=DFlashVersion.V2)
    config.dflash_base = True

    assert _is_spec_tree_base_export(config)


def test_shared_lm_head_inherits_base_group_size(tmp_path):
    draft = make_dflash2_draft_config(str(_write_dflash2_config(tmp_path)),
                                      lambda head_dim: head_dim**-0.5)
    draft = replace(draft,
                    quant=QuantConfig(quant_type=QUANT_NVFP4, group_size=128))
    base = replace(draft,
                   quant=QuantConfig(quant_type=QUANT_NVFP4, group_size=16))

    inherited = _inherit_dflash_lm_head_quant(draft, base)

    assert inherited.quant.group_size == 128
    assert inherited.quant.layer_overrides["lm_head"] == QUANT_NVFP4
    assert inherited.quant.layer_group_sizes["lm_head"] == 16


def test_shared_lm_head_uses_module_group_size_for_mixed_precision(tmp_path):
    draft = make_dflash2_draft_config(str(_write_dflash2_config(tmp_path)),
                                      lambda head_dim: head_dim**-0.5)
    base = replace(draft,
                   quant=QuantConfig(quant_type=QUANT_FP8,
                                     group_size=1,
                                     layer_overrides={"lm_head": QUANT_NVFP4},
                                     layer_group_sizes={"lm_head": 16},
                                     is_mixed_precision=True))

    inherited = _inherit_dflash_lm_head_quant(draft, base)

    assert inherited.quant.group_size == 1
    assert inherited.quant.layer_overrides["lm_head"] == QUANT_NVFP4
    assert inherited.quant.layer_group_sizes["lm_head"] == 16


def test_dflash2_grouped_conv_fake_contract_matches_plugin_shapes():
    hidden = torch.zeros(2, 8, 64, dtype=torch.float16)
    delta = torch.zeros(2, 8, 2, 4, dtype=torch.float16)
    base = torch.zeros(2, 64, dtype=torch.float16)
    residual = torch.zeros(2, 8, 64, dtype=torch.float32)

    pre = dflash2_grouped_dynamic_conv(hidden,
                                       delta,
                                       base,
                                       None,
                                       block_size=8,
                                       kernel_size=2,
                                       group_size=16,
                                       fuse_residual=0)
    post = dflash2_grouped_dynamic_conv(hidden,
                                        delta,
                                        base,
                                        residual,
                                        block_size=8,
                                        kernel_size=2,
                                        group_size=16,
                                        fuse_residual=1)

    assert pre.shape == hidden.shape
    assert pre.dtype == torch.float16
    assert post.shape == residual.shape
    assert post.dtype == torch.float32


def test_dflash2_model_exports_selector_inputs_for_runtime(tmp_path):
    config = make_dflash2_draft_config(str(_write_dflash2_config(tmp_path)),
                                       lambda head_dim: head_dim**-0.5)
    config = replace(config,
                     hidden_size=64,
                     num_attention_heads=4,
                     num_key_value_heads=2,
                     head_dim=16,
                     intermediate_size=128,
                     vocab_size=128)
    model = DFlash2DraftModel(config)

    parameters = dict(model.named_parameters())
    assert "layers.0.attention_conv.base_kernel" in parameters
    assert "layers.0.attention_conv.kernel_projection.weight" in parameters
    assert "candidate_selector.hidden_projection.weight" in parameters
    assert "candidate_selector.predecessor_codebook" in parameters
    assert "candidate_selector.successor_codebook" in parameters

    spec = model.onnx_export_spec()
    assert "spec_sampling_temperature" not in spec.input_names
    assert "spec_proposal_greedy_mask" not in spec.input_names
    assert "spec_proposal_uniforms" not in spec.input_names
    assert "spec_anchor_token_ids" not in spec.input_names
    assert spec.output_names[:3] == [
        "spec_proposal_support_ids", "spec_proposal_unary_values",
        "spec_proposal_projected_hidden"
    ]
    assert "context_lengths" not in spec.input_names
    assert "kvcache_start_index" not in spec.input_names
    assert "dflash_delta_lengths" not in spec.input_names
    assert {
        "query_start_offsets", "query_lengths", "past_lengths",
        "attention_sequence_lengths", "state_indices",
        "execution_phase_marker", "context_sequence_count_carrier",
        "dflash_delta_positions", "dflash_delta_token_to_sequence"
    } <= set(spec.input_names)
    token_dim = spec.dynamic_shapes[0][0]
    assert token_dim.min == 2
    assert token_dim.max == 8_388_608


def test_dflash2_official_export_writes_runtime_selector_sidecar(tmp_path):
    selector = SimpleNamespace(
        predecessor_codebook=torch.nn.Parameter(torch.tensor(
            [[1, 2], [3, 4], [5, 6]], dtype=torch.float16),
                                                requires_grad=False),
        successor_codebook=torch.nn.Parameter(torch.tensor(
            [[7, 8], [9, 10], [11, 12]], dtype=torch.float16),
                                              requires_grad=False),
    )
    model = SimpleNamespace(candidate_selector=selector,
                            config=SimpleNamespace(vocab_size=3,
                                                   dflash2_selector_rank=2))

    export_script._export_dflash2_selector_sidecar(model, str(tmp_path))

    tensors = load_file(str(tmp_path / "dflash2_selector.safetensors"))
    torch.testing.assert_close(tensors["predecessor_codebook"],
                               selector.predecessor_codebook)
    torch.testing.assert_close(tensors["successor_codebook"],
                               selector.successor_codebook)


def test_dflash2_onnx_leaves_candidate_selection_to_runtime(tmp_path):
    config = make_dflash2_draft_config(str(_write_dflash2_config(tmp_path)),
                                       lambda head_dim: head_dim**-0.5)
    config = replace(config,
                     hidden_size=64,
                     num_attention_heads=4,
                     num_key_value_heads=2,
                     head_dim=16,
                     intermediate_size=128,
                     vocab_size=128)
    model = DFlash2DraftModel(config)
    output_path = tmp_path / "dflash2.onnx"

    _export_model(model, str(output_path), optimize=False)

    onnx.checker.check_model(str(output_path))
    graph = onnx.load(str(output_path), load_external_data=False)
    inputs = {value.name: value for value in graph.graph.input}
    input_shape = inputs["inputs_embeds"].type.tensor_type.shape.dim
    assert len(input_shape) == 2
    assert input_shape[0].dim_param
    assert "context_lengths" not in inputs
    assert "kvcache_start_index" not in inputs
    assert "dflash_delta_lengths" not in inputs
    assert "query_start_offsets" in inputs
    assert "dflash_delta_token_to_sequence" in inputs
    assert "spec_proposal_uniforms" not in inputs
    assert all(node.op_type != "DFlash2CandidateSelectorPlugin"
               for node in graph.graph.node)
    outputs = {value.name for value in graph.graph.output}
    assert {
        "spec_proposal_support_ids", "spec_proposal_unary_values",
        "spec_proposal_projected_hidden"
    } <= outputs
