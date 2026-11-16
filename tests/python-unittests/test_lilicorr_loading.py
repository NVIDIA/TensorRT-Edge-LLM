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
"""LiLiCorr draft loading: packed NVFP4 linears and strict coverage.

The synthetic checkpoints reuse the real LiLiCorr tensor names (179 tensors)
with scaled-down dimensions.
"""

import copy
import dataclasses
import json
import logging
import os
import re

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_edgellm.config import ModelConfig
from tensorrt_edgellm.dflash import DFlashVersion
from tensorrt_edgellm.model import (AutoModel, _assert_lilicorr_weights_loaded,
                                    _resolve_model_variant, load_model_config)
from tensorrt_edgellm.models.linear import NVFP4A16Linear

_H = 64
_V = 128
_LAYERS = 5
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 16
_INTERMEDIATE = 128
_TARGET_LAYER_IDS = [1, 18, 35, 51, 68, 85]
_BLOCK = 8
_TOPK = 8
_D = 32
_LILICORR_LAYERS = 2
_LILICORR_HEADS = 4
_LILICORR_MLP = 64
_FACTOR = 24
_CONV_PROJ_OUT = 2 * 2 * (_H // 16)
_GROUP = 16

_DRAFT_CONFIG = {
    "architectures": ["LiLiCorrDraftModel"],
    "model_type": "qwen3",
    "block_size": _BLOCK,
    "dflash_config": {
        "mask_token_id": 990,
        "target_layer_ids": _TARGET_LAYER_IDS,
        "use_swa": True,
        "swa_window_size": 1024,
        "causal": False,
        "attention_sink_bias": True,
        "projector_type": "lilicorr",
        "lilicorr_enabled": True,
        "lilicorr_candidate_topk": _TOPK,
        "lilicorr_hidden_size": _D,
        "lilicorr_num_layers": _LILICORR_LAYERS,
        "lilicorr_num_heads": _LILICORR_HEADS,
        "lilicorr_mlp_ratio": 2.0,
        "lilicorr_factor_dim": _FACTOR,
        "lilicorr_vector_eps": 1e-4,
        "lilicorr_logit_scale": 8.0,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
    },
    "hidden_size": _H,
    "num_hidden_layers": _LAYERS,
    "num_attention_heads": _HEADS,
    "num_key_value_heads": _KV_HEADS,
    "head_dim": _HEAD_DIM,
    "intermediate_size": _INTERMEDIATE,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-06,
    "vocab_size": _V,
    "max_position_embeddings": 262144,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "layer_types": ["sliding_attention"] * _LAYERS,
    "sliding_window": 1024,
    "attention_sink_bias": True,
    "use_sliding_window": True,
}

# Abridged hf_quant_config.json of the real draft.
_DRAFT_HF_QUANT_CONFIG = {
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
        _GROUP,
        "exclude_modules": [
            "*attention_conv*", "*mlp_conv*", "*self_attn*", "*norm",
            "lilicorr.anchor_out_head", "lilicorr.factor_input_proj",
            "lilicorr.feature_mlp*", "lilicorr.in_head", "lilicorr.out_head",
            "lm_head", "*lm_head", "*qkv_proj"
        ],
    },
}

_BASE_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": _H,
    "num_hidden_layers": 86,
    "num_attention_heads": _HEADS,
    "num_key_value_heads": _KV_HEADS,
    "head_dim": _HEAD_DIM,
    "intermediate_size": _INTERMEDIATE,
    "rms_norm_eps": 1e-6,
    "vocab_size": _V,
    "max_position_embeddings": 4096,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}

_BASE_EMBEDDING_KEY = "backbone.embeddings.weight"
_HEAD_NVFP4_LINEARS = [
    "lilicorr.token_proj", "lilicorr.pass_hidden_proj",
    "lilicorr.context_proj", "lilicorr.layers.0.attn.out_proj",
    "lilicorr.layers.1.mlp.up_proj", "lilicorr.layers.1.mlp.down_proj"
]


def _nvfp4_modules():
    """NVFP4 module names and ``(out, in)`` shapes of the real draft."""
    modules = {"fc": (_H, len(_TARGET_LAYER_IDS) * _H)}
    for i in range(_LAYERS):
        modules[f"layers.{i}.mlp.gate_proj"] = (_INTERMEDIATE, _H)
        modules[f"layers.{i}.mlp.up_proj"] = (_INTERMEDIATE, _H)
        modules[f"layers.{i}.mlp.down_proj"] = (_H, _INTERMEDIATE)
    for name in ("token_proj", "pass_hidden_proj", "context_proj"):
        modules[f"lilicorr.{name}"] = (_D, _H)
    for j in range(_LILICORR_LAYERS):
        modules[f"lilicorr.layers.{j}.attn.out_proj"] = (_D, _D)
        modules[f"lilicorr.layers.{j}.mlp.up_proj"] = (_LILICORR_MLP, _D)
        modules[f"lilicorr.layers.{j}.mlp.down_proj"] = (_D, _LILICORR_MLP)
    return modules


def _dense_shapes():
    """BF16 tensor names and shapes of the real draft."""
    shapes = {"hidden_norm.weight": (_H, ), "norm.weight": (_H, )}
    for i in range(_LAYERS):
        prefix = f"layers.{i}"
        for conv in ("attention_conv", "mlp_conv"):
            shapes[f"{prefix}.{conv}.base_kernel"] = (2, 2, _H)
            shapes[f"{prefix}.{conv}.kernel_projection.weight"] = (
                _CONV_PROJ_OUT, _H)
        shapes[f"{prefix}.input_layernorm.weight"] = (_H, )
        shapes[f"{prefix}.post_attention_layernorm.weight"] = (_H, )
        shapes[f"{prefix}.self_attn.attention_sink_bias"] = (_HEADS, )
        shapes[f"{prefix}.self_attn.q_norm.weight"] = (_HEAD_DIM, )
        shapes[f"{prefix}.self_attn.k_norm.weight"] = (_HEAD_DIM, )
        shapes[f"{prefix}.self_attn.q_proj.weight"] = (_HEADS * _HEAD_DIM, _H)
        shapes[f"{prefix}.self_attn.k_proj.weight"] = (_KV_HEADS * _HEAD_DIM,
                                                       _H)
        shapes[f"{prefix}.self_attn.v_proj.weight"] = (_KV_HEADS * _HEAD_DIM,
                                                       _H)
        shapes[f"{prefix}.self_attn.o_proj.weight"] = (_H, _HEADS * _HEAD_DIM)
    head = "lilicorr"
    for name in ("token_proj", "pass_hidden_proj", "context_proj"):
        shapes[f"{head}.{name}.bias"] = (_D, )
    for j in range(_LILICORR_LAYERS):
        layer = f"{head}.layers.{j}"
        shapes[f"{layer}.attn.in_proj_weight"] = (3 * _D, _D)
        shapes[f"{layer}.attn.in_proj_bias"] = (3 * _D, )
        shapes[f"{layer}.attn.out_proj.bias"] = (_D, )
        shapes[f"{layer}.attn_norm.weight"] = (_D, )
        shapes[f"{layer}.mlp.up_proj.bias"] = (_LILICORR_MLP, )
        shapes[f"{layer}.mlp.down_proj.bias"] = (_D, )
        shapes[f"{layer}.mlp_norm.weight"] = (_D, )
    for name in ("anchor_out_head", "in_head", "out_head"):
        shapes[f"{head}.{name}.weight"] = (_FACTOR, _D)
        shapes[f"{head}.{name}.bias"] = (_FACTOR, )
    shapes.update({
        f"{head}.anchor_norm.weight": (_D, ),
        f"{head}.factor_input_proj.weight": (_D, 3 * _D),
        f"{head}.factor_input_proj.bias": (_D, ),
        f"{head}.feature_mlp.up_proj.weight": (_D, 5),
        f"{head}.feature_mlp.up_proj.bias": (_D, ),
        f"{head}.feature_mlp.down_proj.weight": (_D, _D),
        f"{head}.feature_mlp.down_proj.bias": (_D, ),
        f"{head}.feature_norm.weight": (5, ),
        f"{head}.feature_norm.bias": (5, ),
        f"{head}.output_norm.weight": (_D, ),
        f"{head}.rank_embedding": (1, 1, 1, _TOPK, _D),
        f"{head}.relative_slot_bias": (_LILICORR_HEADS, 2 * _BLOCK - 1),
        f"{head}.same_slot_bias": (_LILICORR_HEADS, ),
        f"{head}.slot_embedding": (1, 1, _BLOCK - 1, 1, _D),
    })
    return shapes


def _draft_tensors(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, (out_features, in_features) in _nvfp4_modules().items():
        tensors[f"{name}.weight"] = torch.randint(
            0,
            256, (out_features, in_features // 2),
            generator=g,
            dtype=torch.uint8)
        block_scales = torch.tensor([0.25, 0.5, 1.0])[torch.randint(
            0, 3, (out_features, in_features // _GROUP), generator=g)]
        tensors[f"{name}.weight_scale"] = block_scales.to(torch.float8_e4m3fn)
        tensors[f"{name}.weight_scale_2"] = torch.tensor(0.125,
                                                         dtype=torch.float32)
    for name, shape in _dense_shapes().items():
        tensors[name] = (torch.randn(shape, generator=g) * 0.1).to(
            torch.bfloat16)
    return tensors


def _write_draft(directory, drop=(), extra=None):
    directory.mkdir(parents=True, exist_ok=True)
    tensors = _draft_tensors()
    for key in drop:
        del tensors[key]
    tensors.update(extra or {})
    (directory / "config.json").write_text(json.dumps(_DRAFT_CONFIG))
    (directory / "hf_quant_config.json").write_text(
        json.dumps(_DRAFT_HF_QUANT_CONFIG))
    save_file(tensors, str(directory / "model.safetensors"))
    return tensors


def _write_base(directory):
    directory.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(7)
    tensors = {
        _BASE_EMBEDDING_KEY: torch.randn(_V, _H,
                                         generator=g).to(torch.bfloat16),
        "lm_head.weight": torch.randn(_V, _H, generator=g).to(torch.bfloat16),
    }
    (directory / "config.json").write_text(json.dumps(_BASE_CONFIG))
    save_file(tensors, str(directory / "model.safetensors"))
    return tensors


def _load_draft(base_dir, draft_dir, **kwargs):
    return AutoModel.from_pretrained(str(base_dir),
                                     dflash_draft=True,
                                     dflash_draft_dir=str(draft_dir),
                                     **kwargs)


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("lilicorr_loading")
    base_tensors = _write_base(root / "base")
    draft_tensors = _write_draft(root / "draft")
    return root / "base", root / "draft", base_tensors, draft_tensors


@pytest.fixture(scope="module")
def loaded(checkpoints):
    base_dir, draft_dir, base_tensors, draft_tensors = checkpoints
    return _load_draft(base_dir, draft_dir), base_tensors, draft_tensors


def test_lilicorr_variant_resolved(checkpoints):
    """A V2 draft with projector_type 'lilicorr' routes to the lilicorr_draft variant."""
    config = load_model_config(str(checkpoints[0]))

    variant = _resolve_model_variant(config,
                                     eagle_base=False,
                                     mtp_base=False,
                                     mtp_draft=False,
                                     dflash_draft=True,
                                     dflash_version=DFlashVersion.V2,
                                     dflash_projector="lilicorr")

    assert variant == "lilicorr_draft"


def test_fc_stays_packed_nvfp4(loaded):
    """fc keeps the checkpoint's NVFP4 weights and runs as the packed W4A16 GEMM, not a dense FP16 copy."""
    model, _, _ = loaded

    assert type(model.fc) is NVFP4A16Linear
    assert model.fc_native_precision
    assert (model.fc.in_features, model.fc.out_features) == (6 * _H, _H)


@pytest.mark.parametrize("name", _HEAD_NVFP4_LINEARS)
def test_head_nvfp4_linears_stay_packed(loaded, name):
    """NVFP4-stored head linears stay packed NVFP4A16Linear and keep their checkpoint bias."""
    model, _, draft_tensors = loaded
    module = model.get_submodule(name)

    assert type(module) is NVFP4A16Linear
    assert torch.equal(module.bias.to(torch.bfloat16),
                       draft_tensors[f"{name}.bias"])


def _assert_packed_linears_hold_checkpoint_bytes(model, tensors, names,
                                                 blackwell):
    from tensorrt_edgellm.checkpoint.repacking import (
        repack_nvfp4_a16_blackwell_linear, repack_nvfp4_a16_marlin_linear)
    repack = (repack_nvfp4_a16_blackwell_linear
              if blackwell else repack_nvfp4_a16_marlin_linear)
    for name in names:
        module = model.get_submodule(name)
        qweight, block_scales, global_scale, _, n_padded = repack(
            tensors[f"{name}.weight"],
            tensors[f"{name}.weight_scale"],
            tensors[f"{name}.weight_scale_2"],
            pad_n_to=128)
        assert module._use_blackwell_gemm == blackwell, name
        assert torch.equal(module.qweight, qweight), name
        assert torch.equal(module.block_scales, block_scales), name
        assert torch.equal(module.global_scale, global_scale), name
        assert module.n_padded == n_padded, name


def test_packed_linears_hold_checkpoint_bytes(loaded):
    """After load, fc and every NVFP4 head linear hold exactly the Marlin buffers repacked from their checkpoint bytes."""
    model, _, draft_tensors = loaded

    _assert_packed_linears_hold_checkpoint_bytes(model,
                                                 draft_tensors,
                                                 ["fc"] + _HEAD_NVFP4_LINEARS,
                                                 blackwell=False)


def test_no_derived_token_table(loaded):
    """The draft holds no folded token table; candidate embeddings come from the runtime table."""
    model, _, _ = loaded

    names = [name for name, _ in model.named_buffers()]
    names += [name for name, _ in model.named_parameters()]
    assert not any("token_table" in name for name in names)


def _write_scaled_base(directory, scale):
    _write_base(directory)
    config = dict(_BASE_CONFIG, embedding_scale=scale)
    (directory / "config.json").write_text(json.dumps(config))


def test_draft_vocab_must_equal_base_vocab(checkpoints, tmp_path):
    """A LiLiCorr draft whose vocabulary differs from the base's is rejected: its ids index the base embedding table."""
    _write_base(tmp_path / "base")
    config = dict(_BASE_CONFIG, vocab_size=2 * _V)
    (tmp_path / "base" / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match=f"draft {_V}, base {2 * _V}"):
        _load_draft(tmp_path / "base", checkpoints[1])


def test_prescaled_base_embedding_rejected(checkpoints, tmp_path):
    """A base whose runtime embedding table is pre-scaled is rejected for both the LiLiCorr draft and base exports."""
    _write_scaled_base(tmp_path / "base", 2.0)

    with pytest.raises(ValueError, match="pre-scales it by 2.0"):
        _load_draft(tmp_path / "base", checkpoints[1])
    with pytest.raises(ValueError, match="pre-scales it by 2.0"):
        AutoModel.from_pretrained(str(tmp_path / "base"),
                                  dflash_base=True,
                                  dflash_draft_dir=str(checkpoints[1]))


def test_fp8_embedding_rejected_with_lilicorr_draft(checkpoints, tmp_path,
                                                    capsys):
    """Export fails up front when --fp8-embedding meets a LiLiCorr draft, whose engine needs the FP16 table."""
    from tensorrt_edgellm.onnx import cli as export_script
    base_dir, draft_dir, _, _ = checkpoints

    with pytest.raises(SystemExit):
        export_script.main([
            str(base_dir),
            str(tmp_path / "out"), "--dflash-base", "--dflash-draft-dir",
            str(draft_dir), "--fp8-embedding"
        ])

    assert "--fp8-embedding cannot be combined with a LiLiCorr draft" in (
        capsys.readouterr().err)
    assert not (tmp_path / "out").exists()


def test_lm_head_inherited_from_base(loaded):
    """The draft has no lm_head; it shares the base checkpoint's untied lm_head."""
    model, base_tensors, _ = loaded

    assert torch.equal(model.lm_head.weight,
                       base_tensors["lm_head.weight"].to(torch.float16))


def test_backbone_mlp_stays_nvfp4_a16(loaded):
    """Backbone MLPs stay weight-only NVFP4 (W4A16) linears."""
    model, _, _ = loaded

    assert isinstance(model.layers[0].mlp.gate_proj, NVFP4A16Linear)


def test_loaded_draft_passes_coverage_and_materializes_head(loaded):
    """A complete checkpoint leaves no sentinel and builds the lattice bias."""
    model, _, _ = loaded

    _assert_lilicorr_weights_loaded(model)
    assert torch.isfinite(model.lilicorr.lattice_attention_bias).all()


def test_assert_reports_sentinel_and_packed_tensors(loaded):
    """The load assertion names NaN-sentinel tensors and still-packed FP16 linears."""
    model = copy.deepcopy(loaded[0])
    with torch.no_grad():
        model.layers[1].self_attn.attention_sink_bias[0] = float("nan")
    model.lilicorr.out_head.weight = torch.nn.Parameter(torch.zeros(
        _FACTOR, _D // 2, dtype=torch.uint8),
                                                        requires_grad=False)

    with pytest.raises(ValueError) as error:
        _assert_lilicorr_weights_loaded(model)

    assert "layers.1.self_attn.attention_sink_bias" in str(error.value)
    assert "lilicorr.out_head.weight" in str(error.value)


@pytest.mark.parametrize("missing, reported", [
    ("lilicorr.in_head.weight", "lilicorr.in_head.weight"),
    ("lilicorr.rank_embedding", "lilicorr.rank_embedding"),
    ("layers.0.mlp_conv.base_kernel", "layers.0.mlp_conv.base_kernel"),
    ("hidden_norm.weight", "hidden_norm.weight"),
    ("fc.weight_scale_2", "fc.weight_scale_2"),
    ("lilicorr.token_proj.bias", "lilicorr.token_proj.bias"),
    ("layers.0.mlp.gate_proj.weight_scale_2",
     "layers.0.mlp.gate_proj.weight_scale_2"),
])
def test_missing_head_tensor_raises(checkpoints, tmp_path, missing, reported):
    """A checkpoint missing a floating tensor fails loudly and names it."""
    _write_draft(tmp_path / "draft", drop=[missing])

    with pytest.raises(ValueError, match=re.escape(reported)):
        _load_draft(checkpoints[0], tmp_path / "draft")


def test_missing_sink_raises(checkpoints, tmp_path):
    """A missing attention_sink_bias fails loudly instead of loading zeros."""
    missing = "layers.0.self_attn.attention_sink_bias"
    _write_draft(tmp_path / "draft", drop=[missing])

    with pytest.raises(ValueError, match=re.escape(missing)):
        _load_draft(checkpoints[0], tmp_path / "draft")


@pytest.mark.parametrize("missing", [
    "layers.0.mlp.gate_proj.weight",
    "lilicorr.context_proj.weight_scale",
    "lilicorr.token_proj.weight",
])
def test_missing_packed_tensor_raises(checkpoints, tmp_path, missing):
    """Missing non-floating NVFP4 tensors (packed weight, block scale) fail loudly."""
    _write_draft(tmp_path / "draft", drop=[missing])

    with pytest.raises(ValueError, match=re.escape(missing)):
        _load_draft(checkpoints[0], tmp_path / "draft")


@pytest.mark.parametrize("extra", [
    "lilicorr.layers.0.attn.unexpected_scale",
    "layers.1.mlp_conv.unexpected",
    "fc.input_scale",
])
def test_unconsumed_checkpoint_tensor_raises(checkpoints, tmp_path, extra):
    """Every draft checkpoint tensor must be consumed; a stray key fails loudly."""
    _write_draft(tmp_path / "draft",
                 extra={extra: torch.ones((), dtype=torch.float32)})

    with pytest.raises(ValueError, match=re.escape(extra)):
        _load_draft(checkpoints[0], tmp_path / "draft")


@pytest.mark.parametrize("draft_vocab_size, with_d2t", [(_V // 2, False),
                                                        (None, True)])
def test_draft_local_vocabulary_rejected(checkpoints, tmp_path,
                                         draft_vocab_size, with_d2t):
    """A draft with its own vocabulary (draft_vocab_size or a d2t map) is rejected: candidates must be target ids."""
    extra = {
        "d2t": torch.zeros(_V // 2, dtype=torch.int64)
    } if with_d2t else {}
    _write_draft(tmp_path / "draft", extra=extra)
    if draft_vocab_size is not None:
        config = dict(_DRAFT_CONFIG, draft_vocab_size=draft_vocab_size)
        (tmp_path / "draft" / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="full target vocabulary"):
        _load_draft(checkpoints[0], tmp_path / "draft")


def test_reduced_vocab_rejected(checkpoints, tmp_path):
    """Reduced-vocab export is rejected: the candidate ids index the full vocabulary and its embedding table."""
    with pytest.raises(ValueError, match="reduced"):
        _load_draft(checkpoints[0],
                    checkpoints[1],
                    reduced_vocab_dir=str(tmp_path / "reduced_vocab"))


def test_base_config_carries_lilicorr_contract(checkpoints):
    """The base export copies the LiLiCorr contract and no selector fields."""
    base_dir, draft_dir, _, _ = checkpoints
    defaults = {
        field.name: field.default
        for field in dataclasses.fields(ModelConfig)
    }

    model = AutoModel.from_pretrained(str(base_dir),
                                      dflash_base=True,
                                      dflash_draft_dir=str(draft_dir))
    config = model.config

    assert config.dflash_target_layer_ids == _TARGET_LAYER_IDS
    assert config.dflash2_target_layer_ids == _TARGET_LAYER_IDS
    assert config.dflash_version == DFlashVersion.V2
    assert config.dflash_projector_type == "lilicorr"
    assert config.dflash2_is_causal is False
    assert config.dflash2_conv_kernel_size == 2
    assert config.dflash2_conv_group_size == 16
    assert config.lilicorr_candidate_topk == _TOPK
    assert config.lilicorr_hidden_size == _D
    assert config.lilicorr_num_layers == _LILICORR_LAYERS
    assert config.lilicorr_num_heads == _LILICORR_HEADS
    assert config.lilicorr_factor_dim == _FACTOR
    assert config.lilicorr_logit_scale == 8.0
    assert config.lilicorr_vector_eps == 1e-4
    assert config.dflash2_selector_rank == defaults["dflash2_selector_rank"]
    assert config.dflash2_selector_top_k == defaults["dflash2_selector_top_k"]


@pytest.mark.skipif(not (os.environ.get("LILICORR_DRAFT_DIR")
                         and os.environ.get("BOLT_BASE_DIR")),
                    reason="needs LILICORR_DRAFT_DIR and BOLT_BASE_DIR")
def test_real_checkpoint_coverage(caplog):
    """The real LiLiCorr draft consumes all 179 tensors and leaves no sentinel."""
    caplog.set_level(logging.INFO, logger="tensorrt_edgellm.model")

    model = AutoModel.from_pretrained(
        os.environ["BOLT_BASE_DIR"],
        dflash_draft=True,
        dflash_draft_dir=os.environ["LILICORR_DRAFT_DIR"])

    _assert_lilicorr_weights_loaded(model)
    match = re.search(r"consumed (\d+) of (\d+) draft checkpoint tensors",
                      caplog.text)
    assert match is not None
    assert match.group(1) == match.group(2) == "179"
    assert type(model.fc) is NVFP4A16Linear
    assert type(model.lilicorr.token_proj) is NVFP4A16Linear


_REAL_HEAD_NVFP4_LINEARS = [
    "fc", "lilicorr.token_proj", "lilicorr.pass_hidden_proj",
    "lilicorr.context_proj"
] + [
    f"lilicorr.layers.{i}.{name}" for i in range(2)
    for name in ("attn.out_proj", "mlp.up_proj", "mlp.down_proj")
]


@pytest.mark.skipif(not (os.environ.get("LILICORR_DRAFT_DIR")
                         and os.environ.get("BOLT_BASE_DIR")),
                    reason="needs LILICORR_DRAFT_DIR and BOLT_BASE_DIR")
@pytest.mark.parametrize("target_sm", [None, 110])
def test_real_checkpoint_packed_linears_hold_checkpoint_bytes(target_sm):
    """H2 on the real draft: fc and the head NVFP4 linears hold the Marlin (default) or SM110 Blackwell buffers of their checkpoint bytes."""
    from safetensors.torch import load_file

    from tensorrt_edgellm.models.ops import set_nvfp4_a16_export_target_sm
    draft_dir = os.environ["LILICORR_DRAFT_DIR"]
    set_nvfp4_a16_export_target_sm(target_sm)
    try:
        model = AutoModel.from_pretrained(os.environ["BOLT_BASE_DIR"],
                                          dflash_draft=True,
                                          dflash_draft_dir=draft_dir)
    finally:
        set_nvfp4_a16_export_target_sm(None)
    tensors = load_file(os.path.join(draft_dir, "model.safetensors"))

    _assert_packed_linears_hold_checkpoint_bytes(model,
                                                 tensors,
                                                 _REAL_HEAD_NVFP4_LINEARS,
                                                 blackwell=target_sm == 110)
