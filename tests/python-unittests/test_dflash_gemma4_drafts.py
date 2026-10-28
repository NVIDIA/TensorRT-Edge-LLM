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

import json
import types

import torch

from tensorrt_edgellm import config as model_config
from tensorrt_edgellm import dflash
from tensorrt_edgellm.checkpoint import checkpoint_utils
from tensorrt_edgellm.quantization import quantization_configs
from tensorrt_edgellm.quantization.models import dflash_draft, layers
from tensorrt_edgellm.scripts import quantize as quantize_script

_HIDDEN = 8
_GLOBAL_HEAD_DIM = 8
_NUM_HEADS = 2
_NUM_TARGET_LAYERS = 5


def _gemma4_config(num_layers=2):
    return types.SimpleNamespace(
        architectures=["Gemma4DSparkModel"],
        model_type="gemma4_text",
        hidden_size=_HIDDEN,
        intermediate_size=16,
        num_attention_heads=_NUM_HEADS,
        num_key_value_heads=2,
        head_dim=4,
        global_head_dim=_GLOBAL_HEAD_DIM,
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
        layer_types=["full_attention"] * num_layers,
        num_hidden_layers=num_layers,
        vocab_size=32,
        rms_norm_eps=1e-6,
        rope_parameters={
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
            }
        },
        dflash_config={
            "target_layer_ids": list(range(_NUM_TARGET_LAYERS)),
            "block_size": 4,
            "mask_token_id": 4,
        },
    )


def _checkpoint_shapes(num_layers=2):
    """Tensor names/shapes of deepseek-ai/dflash_gemma4_12b_block7, scaled down."""
    q_dim = _NUM_HEADS * _GLOBAL_HEAD_DIM
    shapes = {
        "fc.weight": (_HIDDEN, _NUM_TARGET_LAYERS * _HIDDEN),
        "hidden_norm.weight": (_HIDDEN, ),
        "norm.weight": (_HIDDEN, ),
        "lm_head.weight": (32, _HIDDEN),
    }
    for i in range(num_layers):
        p = f"layers.{i}."
        shapes.update({
            p + "input_layernorm.weight": (_HIDDEN, ),
            p + "post_attention_layernorm.weight": (_HIDDEN, ),
            p + "pre_feedforward_layernorm.weight": (_HIDDEN, ),
            p + "post_feedforward_layernorm.weight": (_HIDDEN, ),
            p + "layer_scalar": (1, ),
            p + "self_attn.q_proj.weight": (q_dim, _HIDDEN),
            p + "self_attn.k_proj.weight": (_GLOBAL_HEAD_DIM, _HIDDEN),
            p + "self_attn.o_proj.weight": (_HIDDEN, q_dim),
            p + "self_attn.q_norm.weight": (_GLOBAL_HEAD_DIM, ),
            p + "self_attn.k_norm.weight": (_GLOBAL_HEAD_DIM, ),
            p + "mlp.gate_proj.weight": (16, _HIDDEN),
            p + "mlp.up_proj.weight": (16, _HIDDEN),
            p + "mlp.down_proj.weight": (_HIDDEN, 16),
        })
    return shapes


def test_deepseek_gemma4_draft_routes_to_dflash_quantization(tmp_path):
    """A Gemma4DSparkModel draft without dflash_config uses the DFlash flow, not Eagle3."""
    (tmp_path / "config.json").write_text(
        json.dumps({
            "architectures": ["Gemma4DSparkModel"],
            "model_type": "gemma4_text",
            "block_size": 7,
            "target_layer_ids": [5, 17, 29, 41, 46],
        }))

    assert quantize_script._is_dflash_family_draft(str(tmp_path))


def test_deepseek_gemma4_draft_config_loads_for_export(tmp_path):
    """Exporter config loading tolerates the draft's mixed scalar/per-layer rope_parameters."""
    (tmp_path / "config.json").write_text(
        json.dumps({
            "architectures": ["Gemma4DSparkModel"],
            "model_type": "gemma4_text",
            "hidden_size": 3840,
            "num_attention_heads": 16,
            "num_hidden_layers": 5,
            "layer_types": ["full_attention"] * 5,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional",
                },
                "rope_theta": None,
                "rope_type": "default",
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default",
                },
            },
            "block_size": 7,
            "target_layer_ids": [5, 17, 29, 41, 46],
        }))

    _, llm = checkpoint_utils.load_checkpoint_config_dicts(str(tmp_path))

    assert llm["target_layer_ids"] == [5, 17, 29, 41, 46]
    assert llm["block_size"] == 7


def test_dflash_shift_label_follows_checkpoint_convention():
    """DeepSpec drafts predict token i+1 at block row i; z-lab drafts and explicit opt-outs do not."""
    deepspec = {
        "architectures": ["Gemma4DSparkModel"],
        "block_size": 7,
        "target_layer_ids": [5, 17, 29, 41, 46],
    }
    zlab = {
        "architectures": ["DFlashDraftModel"],
        "dflash_config": {
            "mask_token_id": 4,
            "target_layer_ids": [1, 10, 19, 27, 36, 45]
        },
    }
    opt_out = dict(deepspec, dflash_config={"shift_label": False})

    assert dflash.resolve_dflash_contract(deepspec, deepspec).shift_label
    assert not dflash.resolve_dflash_contract(zlab, zlab).shift_label
    assert not dflash.resolve_dflash_contract(opt_out, opt_out).shift_label


def _write_draft_config(path, **overrides):
    cfg = {
        "architectures": ["Gemma4DSparkModel"],
        "model_type": "gemma4_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "num_hidden_layers": 1,
        "layer_types": ["full_attention"],
        "vocab_size": 128,
        "rms_norm_eps": 1e-6,
        "block_size": 7,
        "target_layer_ids": [1, 2],
        "mask_token_id": 4,
    }
    cfg.update(overrides)
    (path / "config.json").write_text(json.dumps(cfg))


def test_dflash_draft_export_config_carries_shift_label(tmp_path):
    """The exporter's DFlash draft config enables the row shift for DeepSpec drafts."""
    _write_draft_config(tmp_path)

    config = model_config.make_dflash_draft_config(str(tmp_path),
                                                   lambda d: d**-0.5)

    assert config.dflash_shift_label


def test_gemma4_calibration_model_matches_checkpoint_tensors():
    """Calibration modules load the Gemma4 draft checkpoint strictly (fc = Nl*H, K=V, no v_proj)."""
    model = dflash_draft.DFlashCalibDraftModel(_gemma4_config())
    checkpoint = {
        name: torch.randn(shape)
        for name, shape in _checkpoint_shapes().items()
    }

    model.load_state_dict(checkpoint, strict=True)

    assert model.is_gemma4


def test_gemma4_k_eq_v_uses_raw_key_with_weightless_value_norm():
    """K=V attention takes V from the pre-k_norm key projection, then applies v_norm."""
    attention = dflash_draft.DFlashCalibDraftModel(
        _gemma4_config()).layers[0].self_attn
    k_weight = torch.arange(1.0, _GLOBAL_HEAD_DIM + 1.0)
    with torch.no_grad():
        attention.k_norm.weight.copy_(k_weight)
    hidden = torch.randn(1, 2, _HIDDEN)

    key, value = attention.project_kv(hidden)

    raw = attention.k_proj(hidden).view(1, 2, 1, _GLOBAL_HEAD_DIM)
    expected_value = raw * torch.rsqrt(
        raw.pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(value, expected_value)
    torch.testing.assert_close(key, k_weight * expected_value)


def test_proportional_rope_rotates_only_leading_frequencies():
    """Gemma4 proportional RoPE leaves frequencies past partial_rotary_factor unrotated."""
    rope = layers.RotaryEmbedding(16,
                                  base=10000.0,
                                  proportional_rotary_factor=0.25)
    cos, sin = rope(torch.zeros(1, 3, 16), torch.arange(3).unsqueeze(0))

    # 0.25 * 16 // 2 = 2 rotated frequencies per half.
    assert torch.equal(rope.inv_freq[2:], torch.zeros(6))
    assert torch.isclose(rope.inv_freq[1],
                         torch.tensor(1.0 / 10000.0**(2 / 16)))
    assert torch.equal(cos[..., 2:8], torch.ones_like(cos[..., 2:8]))
    assert torch.equal(sin[..., 10:], torch.zeros_like(sin[..., 10:]))


def test_gemma4_mlp_clamps_intermediate_before_down_proj():
    """down_proj calibration sees the exporter's +/-2048 GeGLU clamp."""
    mlp = dflash_draft.Gemma4DFlashCalibMLP(_HIDDEN, 16)
    with torch.no_grad():
        mlp.gate_proj.weight.fill_(100.0)
        mlp.up_proj.weight.fill_(100.0)
    seen = []
    mlp.down_proj.register_forward_pre_hook(
        lambda _module, inputs: seen.append(inputs[0]))

    mlp(torch.ones(1, 1, _HIDDEN))

    assert seen[0].abs().max() == 2048.0


def test_gemma4_4bit_quantization_keeps_every_down_proj_fp16():
    """int4/nvfp4 Gemma4 drafts disable both down_proj quantizers on every layer."""
    model = dflash_draft.DFlashCalibDraftModel(_gemma4_config())
    cfg = quantization_configs.build_quant_config("nvfp4")

    dflash_draft._disable_gemma4_down_proj_quantization(cfg, model)

    section = cfg["quant_cfg"]
    entries = (section if isinstance(section, list) else [{
        "quantizer_name": key,
        **value
    } for key, value in section.items() if isinstance(value, dict)])
    disabled = {
        e["quantizer_name"]
        for e in entries if e.get("enable") is False
    }
    for i in range(2):
        assert f"*layers.{i}.mlp.down_proj.weight_quantizer" in disabled
        assert f"*layers.{i}.mlp.down_proj.input_quantizer" in disabled
    assert "fp8" not in dflash_draft._GEMMA4_FP16_DOWN_PROJ_QUANTS
