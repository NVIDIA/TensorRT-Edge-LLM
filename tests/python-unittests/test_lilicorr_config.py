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

import onnx
import pytest
import torch
from lilicorr_checkpoint_config import LILICORR_CONFIG as _LILICORR_CONFIG
from onnx import numpy_helper

from tensorrt_edgellm.checkpoint.checkpoint_utils import (
    build_runtime_llm_config_dict, validate_external_speculative_target)
from tensorrt_edgellm.config import (QUANT_NVFP4_A16, QUANT_Q8_0, ModelConfig,
                                     make_dflash2_draft_config,
                                     make_lilicorr_draft_config,
                                     module_quant_type)
from tensorrt_edgellm.model import _inherit_dflash_lm_head_quant
from tensorrt_edgellm.models.dflash2 import DFlash2DraftModel
from tensorrt_edgellm.onnx.export import _export_model

# Abridged hf_quant_config.json shipped beside the same checkpoint.
_LILICORR_HF_QUANT_CONFIG = {
    "producer": {
        "name": "modelopt",
        "version": "0.0.0"
    },
    "quantization": {
        "quant_algo":
        "W4A16_NVFP4",
        "kv_cache_quant_algo":
        None,
        "group_size":
        16,
        "exclude_modules": [
            "layers.0.self_attn*", "lilicorr.in_head", "lilicorr.out_head",
            "hidden_norm", "norm", "lm_head", "*lm_head", "*qkv_proj"
        ],
    },
}

_DFLASH2_CONFIG = {
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
    "dflash_config": {
        "target_layer_ids": [5, 19, 33, 47, 61],
        "block_size": 8,
        "mask_token_id": 248070,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
    },
}


def _attention_scale(head_dim):
    return head_dim**-0.5


def _write_checkpoint(directory, config, hf_quant_config=None):
    (directory / "config.json").write_text(json.dumps(config))
    if hf_quant_config is not None:
        (directory / "hf_quant_config.json").write_text(
            json.dumps(hf_quant_config))
    return str(directory)


def _lilicorr_draft_config(tmp_path, with_hf_quant_config=False):
    checkpoint = copy.deepcopy(_LILICORR_CONFIG)
    hf_quant_config = None
    if with_hf_quant_config:
        del checkpoint["quantization_config"]
        hf_quant_config = _LILICORR_HF_QUANT_CONFIG
    return make_lilicorr_draft_config(
        _write_checkpoint(tmp_path, checkpoint, hf_quant_config),
        _attention_scale)


def _tiny_dflash2_config(tmp_path, **overrides) -> ModelConfig:
    config = make_dflash2_draft_config(
        _write_checkpoint(tmp_path, _DFLASH2_CONFIG), _attention_scale)
    return replace(config,
                   hidden_size=64,
                   num_attention_heads=4,
                   num_key_value_heads=2,
                   head_dim=16,
                   intermediate_size=128,
                   vocab_size=128,
                   **overrides)


def _sliding_sink_config(tmp_path) -> ModelConfig:
    layer_types = ["sliding_attention"] * 5
    return _tiny_dflash2_config(tmp_path,
                                attention_sink_bias=True,
                                sliding_window_size=1024,
                                attention_layer_types=layer_types,
                                raw_layer_types=layer_types)


def _constant_values(graph, name):
    for initializer in graph.initializer:
        if initializer.name == name:
            return numpy_helper.to_array(initializer).flatten().tolist()
    producer = next(node for node in graph.node if name in node.output)
    assert producer.op_type == "Constant"
    value = producer.attribute[0]
    if value.floats:
        return list(value.floats)
    return numpy_helper.to_array(value.t).flatten().tolist()


@pytest.mark.parametrize("with_hf_quant_config", [False, True])
def test_lilicorr_w4a16_maps_to_nvfp4_a16(tmp_path, with_hf_quant_config):
    """W4A16_NVFP4 from either quant source resolves to weight-only NVFP4."""
    config = _lilicorr_draft_config(tmp_path, with_hf_quant_config)

    assert config.quant.quant_type == QUANT_NVFP4_A16
    assert config.quant.group_size == 16


def test_lilicorr_fc_not_excluded(tmp_path):
    """LiLiCorr loads fc from its packed NVFP4 tensors instead of FP16."""
    config = _lilicorr_draft_config(tmp_path)

    assert "fc" not in config.quant.excluded


@pytest.mark.parametrize("with_hf_quant_config", [False, True])
def test_lilicorr_lm_head_inherits_quantized_base_head(tmp_path,
                                                       with_hf_quant_config):
    """A LiLiCorr draft excluding `*lm_head` by glob still takes the base's quantized lm_head layout."""
    draft = _lilicorr_draft_config(tmp_path, with_hf_quant_config)
    base = replace(draft,
                   quant=replace(draft.quant,
                                 excluded=[],
                                 layer_overrides={"lm_head": QUANT_Q8_0},
                                 layer_group_sizes={"lm_head": 32}))

    config = _inherit_dflash_lm_head_quant(draft, base)

    assert module_quant_type("lm_head", config) == QUANT_Q8_0


@pytest.mark.parametrize("mode, accepted", [("lilicorr", True),
                                            ("dflash", False),
                                            ("dspark", False),
                                            ("dspark_onnx", False),
                                            ("eagle3", False)])
def test_bolt_target_accepts_only_lilicorr_drafts(mode, accepted):
    """A Nemotron-H Puzzle (Bolt) target accepts only external LiLiCorr drafts, not other DFlash drafts or modes."""
    config = {"model_type": "nemotron_h_puzzle"}

    if accepted:
        validate_external_speculative_target(config, mode=mode)
    else:
        with pytest.raises(ValueError, match="Nemotron external"):
            validate_external_speculative_target(config, mode=mode)


@pytest.mark.parametrize("draft_config, expected_mode",
                         [(_LILICORR_CONFIG, "lilicorr"),
                          (_DFLASH2_CONFIG, "dflash")])
def test_export_validates_target_for_the_draft_projector(
        tmp_path, capsys, monkeypatch, draft_config, expected_mode):
    """The exporter validates the target for a LiLiCorr draft as mode 'lilicorr' and for other DFlash drafts as 'dflash'."""
    from tensorrt_edgellm.onnx import cli as export_script

    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(
        json.dumps({"model_type": "nemotron_h_puzzle"}))
    draft = tmp_path / "draft"
    draft.mkdir()
    _write_checkpoint(draft, draft_config)

    def validate(config, *, mode):
        raise ValueError(f"validated mode={mode}")

    monkeypatch.setattr(export_script.checkpoint_utils,
                        "validate_external_speculative_target", validate)

    with pytest.raises(SystemExit):
        export_script.main([
            str(target),
            str(tmp_path / "output"), "--dflash-base", "--dflash-draft-dir",
            str(draft)
        ])
    assert f"validated mode={expected_mode}" in capsys.readouterr().err


def test_dflash2_fc_stays_excluded(tmp_path):
    """Non-LiLiCorr DFlash2 drafts keep fc as a dense FP16 projector."""
    config = make_dflash2_draft_config(
        _write_checkpoint(tmp_path, _DFLASH2_CONFIG), _attention_scale)

    assert "fc" in config.quant.excluded
    assert config.dflash_projector_type == ""
    assert config.attention_sink_bias is False


def test_lilicorr_head_dims_parsed(tmp_path):
    """Every LiLiCorr head hyperparameter reaches ModelConfig unchanged."""
    config = _lilicorr_draft_config(tmp_path)

    assert config.dflash_projector_type == "lilicorr"
    assert config.lilicorr_candidate_topk == 8
    assert config.lilicorr_hidden_size == 1024
    assert config.lilicorr_num_layers == 2
    assert config.lilicorr_num_heads == 8
    assert config.lilicorr_mlp_ratio == 2.0
    assert config.lilicorr_factor_dim == 1024
    assert config.lilicorr_vector_eps == 1e-4
    assert config.lilicorr_logit_scale == 8.0


def test_lilicorr_rejects_head_as_wide_as_draft(tmp_path):
    """A head as wide as the draft needs an identity token_proj, which is unsupported."""
    checkpoint = copy.deepcopy(_LILICORR_CONFIG)
    checkpoint["dflash_config"]["lilicorr_hidden_size"] = 4096

    with pytest.raises(ValueError, match="identity token_proj"):
        make_lilicorr_draft_config(_write_checkpoint(tmp_path, checkpoint),
                                   _attention_scale)


def test_lilicorr_rejects_draft_local_vocabulary(tmp_path):
    """A draft with its own reduced vocabulary is rejected: LiLiCorr candidate ids are target-vocabulary ids."""
    checkpoint = copy.deepcopy(_LILICORR_CONFIG)
    checkpoint["draft_vocab_size"] = checkpoint["vocab_size"] // 2

    with pytest.raises(ValueError, match="full target vocabulary"):
        make_lilicorr_draft_config(_write_checkpoint(tmp_path, checkpoint),
                                   _attention_scale)


@pytest.mark.parametrize("topk", [4, 8])
def test_lilicorr_runtime_config_reports_candidate_topk(tmp_path, topk):
    """The LiLiCorr draft runtime config carries the checkpoint top-k."""
    checkpoint = copy.deepcopy(_LILICORR_CONFIG)
    checkpoint["dflash_config"]["lilicorr_candidate_topk"] = topk
    config = make_lilicorr_draft_config(
        _write_checkpoint(tmp_path, checkpoint), _attention_scale)

    runtime = build_runtime_llm_config_dict(SimpleNamespace(config=config))

    assert runtime["dflash_config"]["projector_type"] == "lilicorr"
    assert runtime["dflash_config"]["lilicorr_candidate_topk"] == topk


def test_lilicorr_sinks_and_window_enabled(tmp_path):
    """LiLiCorr drafts enable attention sinks and the 1024-token window."""
    config = _lilicorr_draft_config(tmp_path)

    assert config.attention_sink_bias is True
    assert config.sliding_window_size == 1024


def test_lilicorr_reads_top_level_attention_sink_flag(tmp_path):
    """attention_sink_bias falls back to the top-level config key."""
    checkpoint = copy.deepcopy(_LILICORR_CONFIG)
    del checkpoint["dflash_config"]["attention_sink_bias"]

    config = make_lilicorr_draft_config(
        _write_checkpoint(tmp_path, checkpoint), _attention_scale)

    assert config.attention_sink_bias is True


def test_lilicorr_factory_rejects_dflash2_checkpoint(tmp_path):
    """The LiLiCorr factory only accepts projector_type='lilicorr'."""
    with pytest.raises(ValueError, match="lilicorr"):
        make_lilicorr_draft_config(
            _write_checkpoint(tmp_path, _DFLASH2_CONFIG), _attention_scale)


def test_dflash_attention_sink_kwargs_without_contiguous_swa(tmp_path):
    """Sink kwargs carry the loaded FP32 sinks and keep the DFlash SWA path."""
    model = DFlash2DraftModel(_sliding_sink_config(tmp_path))
    attention = model.layers[0].self_attn
    sinks = torch.arange(attention.num_heads, dtype=torch.float32)

    assert attention.attention_sink_bias.dtype == torch.float32
    assert attention.attention_sink_bias.shape == (attention.num_heads, )
    attention.attention_sink_bias.copy_(sinks)
    attention._capture_attention_sink_list()

    assert attention.sliding_window_size == 1024
    assert attention._ragged_attention_kwargs() == {
        "attention_sinks": sinks.tolist(),
        "enable_attention_sink": 1,
    }


def test_dflash_attention_exports_sinks(tmp_path):
    """Exported DFlash2 AttentionPlugins carry each layer's sinks."""
    model = DFlash2DraftModel(_sliding_sink_config(tmp_path))
    expected = []
    for layer_idx, layer in enumerate(model.layers):
        attention = layer.self_attn
        sinks = torch.arange(attention.num_heads,
                             dtype=torch.float32) + layer_idx * 0.5
        attention.attention_sink_bias.copy_(sinks)
        expected.append(sinks.tolist())
    output = tmp_path / "dflash2-sinks.onnx"

    _export_model(model, str(output), optimize=False)

    graph = onnx.load(str(output), load_external_data=False).graph
    attentions = [
        node for node in graph.node if node.op_type == "AttentionPlugin"
    ]
    assert len(attentions) == len(model.layers)
    exported = []
    for node in attentions:
        attributes = {
            attribute.name: attribute.i
            for attribute in node.attribute
        }
        assert attributes["enable_attention_sink"] == 1
        assert attributes.get("enable_contiguous_query_swa", 0) == 0
        assert attributes["sliding_window_size"] == 1024
        sink_inputs = [
            name for name in node.input
            if name.startswith("attention_sinks_fp32")
        ]
        assert len(sink_inputs) == 1
        exported.append(_constant_values(graph, sink_inputs[0]))
    assert exported == expected


def test_dflash2_without_sinks_unchanged(tmp_path):
    """DFlash2 drafts without attention_sink_bias pass no sink kwargs."""
    model = DFlash2DraftModel(_tiny_dflash2_config(tmp_path))
    attention = model.layers[0].self_attn
    attention._capture_attention_sink_list()

    assert not hasattr(attention, "attention_sink_bias")
    assert attention._ragged_attention_kwargs() == {}
