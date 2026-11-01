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
"""Checkpoint-provided NVFP4 W4A16 lm_head for Qwen3.6-style models."""

import os
import sys
from types import SimpleNamespace

import pytest

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tensorrt_edgellm import config


@pytest.fixture(autouse=True)
def _restore_activation_precision():
    config.set_default_quantize_activations(True)
    yield
    config.set_default_quantize_activations(True)


@pytest.mark.parametrize(("model_type", "expected_quant_type"),
                         [("qwen3_moe", config.QUANT_NVFP4),
                          ("nemotron_h", config.QUANT_NVFP4_A16)],
                         ids=["qwen-generic-nvfp4", "nemotron-h-nvfp4-a16"])
def test_mixed_precision_w4a16_dispatch_is_model_specific(
        model_type, expected_quant_type):
    """Qwen uses its established repacker; Nemotron-H uses Marlin A16."""
    dominant, group_size, overrides, group_sizes = config._parse_mixed_precision(
        {
            "layers.0.mlp.experts": {
                "quant_algo": "W4A16_NVFP4",
                "group_size": 16
            },
            "lm_head": {
                "quant_algo": "W4A16_NVFP4",
                "group_size": 16
            },
        }, model_type)
    assert dominant == expected_quant_type
    assert group_size == 16
    assert overrides["lm_head"] == expected_quant_type
    assert overrides["layers.0.mlp.experts"] == expected_quant_type
    assert group_sizes["lm_head"] == 16
    assert group_sizes["layers.0.mlp.experts"] == 16


def test_mixed_precision_preserves_group_sizes_for_fused_overrides():
    dominant, group_size, overrides, group_sizes = config._parse_mixed_precision(
        {
            "layers.0.self_attn.o_proj": {
                "quant_algo": "FP8",
                "group_size": 1
            },
            "layers.1.self_attn.o_proj": {
                "quant_algo": "FP8",
                "group_size": 1
            },
            "layers.0.mlp.gate_up_proj": {
                "quant_algo": "NVFP4",
                "group_size": 16
            },
            "lm_head": {
                "quant_algo": "NVFP4",
                "group_size": 16
            },
        })

    assert dominant == config.QUANT_FP8
    assert group_size == 1
    assert overrides["layers.0.mlp.gate_proj"] == config.QUANT_NVFP4
    assert overrides["layers.0.mlp.up_proj"] == config.QUANT_NVFP4
    assert group_sizes["layers.0.mlp.gate_proj"] == 16
    assert group_sizes["layers.0.mlp.up_proj"] == 16
    assert group_sizes["lm_head"] == 16


def test_no_activation_quantization_selects_nvfp4_a16():
    config.set_default_quantize_activations(False)

    quant = config._parse_quant(
        "/unused", {
            "model_type": "qwen3_moe",
            "quantization_config": {
                "quant_algo": "NVFP4",
                "group_size": 16,
            },
        })

    assert quant.quant_type == config.QUANT_NVFP4_A16
    assert quant.quantize_activations is False
    assert quant.uses_nvfp4_weights is True


def test_no_activation_quantization_only_changes_nvfp4_layers():
    config.set_default_quantize_activations(False)

    dominant, _, overrides, _ = config._parse_mixed_precision(
        {
            "layers.0.mlp.experts": {
                "quant_algo": "NVFP4",
                "group_size": 16,
            },
            "layers.0.self_attn.q_proj": {
                "quant_algo": "FP8",
            },
        }, "qwen3_5_moe_text")

    assert dominant == config.QUANT_NVFP4_A16
    assert overrides["layers.0.mlp.experts"] == config.QUANT_NVFP4_A16
    assert overrides["layers.0.self_attn.q_proj"] == config.QUANT_FP8


def test_parse_quant_keeps_excluded_fp16_lm_head():
    """Nemotron-style excluded FP16/BF16 heads stay excluded; no self-pack."""
    quant = config._parse_quant(
        "/unused", {
            "quantization_config": {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "ignore": ["lm_head", "visual"],
            }
        })
    assert "lm_head" in quant.excluded
    assert quant.layer_overrides.get("lm_head") is None


def test_parse_quant_does_not_invent_lm_head_override():
    quant = config._parse_quant("/unused", {"hidden_size": 2048})
    assert quant.layer_overrides.get("lm_head") is None


def _fake_nvfp4_a16(n, k, group_size=16):
    torch = pytest.importorskip("torch")
    packed = torch.zeros(n, k // 2, dtype=torch.uint8)
    scale = torch.ones(n, k // group_size, dtype=torch.float8_e4m3fn)
    scale2 = torch.ones(1, dtype=torch.float32)
    return packed, scale, scale2


def test_repack_nvfp4_a16_gated_moe_shapes():
    pytest.importorskip("torch")
    from tensorrt_edgellm.checkpoint.repacking import \
        repack_nvfp4_a16_marlin_gated_moe_experts

    hidden, inter, experts = 128, 128, 2
    gate = [_fake_nvfp4_a16(inter, hidden) for _ in range(experts)]
    up = [_fake_nvfp4_a16(inter, hidden) for _ in range(experts)]
    down = [_fake_nvfp4_a16(hidden, inter) for _ in range(experts)]
    fc1_q, fc1_s, fc1_g, fc2_q, fc2_s, fc2_g = (
        repack_nvfp4_a16_marlin_gated_moe_experts(
            [t[0] for t in gate], [t[1] for t in gate], [t[2] for t in gate],
            [t[0] for t in up], [t[1] for t in up], [t[2] for t in up],
            [t[0] for t in down], [t[1] for t in down], [t[2] for t in down],
            inter))
    assert tuple(fc1_q.shape) == (experts, hidden // 16, 8 * 2 * inter)
    assert tuple(fc1_s.shape) == (experts, hidden // 16, 2 * inter)
    assert tuple(fc1_g.shape) == (2, experts)
    assert tuple(fc2_q.shape) == (experts, inter // 16, 8 * hidden)
    assert tuple(fc2_s.shape) == (experts, inter // 16, hidden)
    assert tuple(fc2_g.shape) == (experts, )


def test_qwen_w4a4_moe_repack_keeps_calibrated_activation_scales(monkeypatch):
    torch = pytest.importorskip("torch")
    from tensorrt_edgellm.checkpoint import repacking
    from tensorrt_edgellm.models.qwen3_moe import modeling_qwen3_moe

    experts = []
    for fc1_scale, fc2_scale in ((2.0, 3.0), (5.0, 7.0)):
        experts.append(
            SimpleNamespace(
                gate_proj=SimpleNamespace(input_scale=torch.tensor(fc1_scale)),
                up_proj=SimpleNamespace(input_scale=torch.tensor(fc1_scale)),
                down_proj=SimpleNamespace(input_scale=torch.tensor(fc2_scale)),
            ))

    packed = tuple(torch.zeros(2, 1) for _ in range(6))
    monkeypatch.setattr(repacking, "repack_nvfp4_gated_moe_experts",
                        lambda *args, **kwargs: packed)
    monkeypatch.setattr(modeling_qwen3_moe, "use_geforce_nvfp4_moe",
                        lambda: False)

    holder = SimpleNamespace(
        gate=SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(2, 4))),
        hidden_size=4,
        num_experts=2,
        moe_intermediate_size=4,
        group_size=16,
        experts=experts,
    )
    holder.register_buffer = lambda name, value: setattr(holder, name, value)

    modeling_qwen3_moe.Qwen3SparseMoeBlock._prepare_nvfp4_moe_weights_impl(
        holder)

    assert torch.equal(holder.input_global_scale, torch.tensor([2.0, 5.0]))
    assert torch.equal(holder.down_input_scale, torch.tensor([3.0, 7.0]))


@pytest.mark.parametrize(
    ("module_name", "gated"),
    (("gemma4.modeling_gemma4_text", True),
     ("nemotron_h.modeling_nemotron_h", False)),
)
def test_w4a4_moe_helpers_keep_calibrated_activation_scales(
        module_name, gated):
    torch = pytest.importorskip("torch")
    models = pytest.importorskip(f"tensorrt_edgellm.models.{module_name}")

    experts = []
    for fc1_scale, fc2_scale in ((2.0, 3.0), (5.0, 7.0)):
        expert = SimpleNamespace(
            up_proj=SimpleNamespace(input_scale=torch.tensor(fc1_scale)),
            down_proj=SimpleNamespace(input_scale=torch.tensor(fc2_scale)),
        )
        if gated:
            expert.gate_proj = SimpleNamespace(
                input_scale=torch.tensor(fc1_scale))
        experts.append(expert)

    fc1, fc2 = models._stack_nvfp4_moe_input_scales(experts)

    assert torch.equal(fc1, torch.tensor([2.0, 5.0]))
    assert torch.equal(fc2, torch.tensor([3.0, 7.0]))


def test_gemma4_expert_quant_lookup_uses_checkpoint_names(monkeypatch):
    torch = pytest.importorskip("torch")
    from tensorrt_edgellm.models.gemma4 import modeling_gemma4_text

    names = []

    def make_linear(*args, module_name, **kwargs):
        del args, kwargs
        names.append(module_name)
        return torch.nn.Identity()

    monkeypatch.setattr(modeling_gemma4_text, "make_linear", make_linear)
    cfg = SimpleNamespace(hidden_size=128,
                          moe_intermediate_size=128,
                          num_experts=2)

    modeling_gemma4_text.Gemma4NvFP4MoEExperts(cfg,
                                               "layers.0.moe_block.experts")

    assert names == [
        f"layers.0.moe_block.experts.{expert}.{projection}_proj"
        for expert in range(2) for projection in ("gate", "up", "down")
    ]


def test_muse_glimmer_keeps_packed_embedding_for_w4a16():
    from tensorrt_edgellm.models.muse_glimmer.modeling_muse_glimmer_text import (
        MuseGlimmerRuntimeEmbedding, MuseGlimmerTransformer)

    quant = SimpleNamespace(
        excluded=[],
        quant_type=config.QUANT_NVFP4_A16,
        layer_overrides={},
        layer_group_sizes={},
        is_mixed_precision=False,
        group_size=16,
    )
    model_config = SimpleNamespace(
        quant=quant,
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=0,
        rms_norm_eps=1e-6,
    )

    model = MuseGlimmerTransformer(model_config)

    assert isinstance(model.embed_tokens, MuseGlimmerRuntimeEmbedding)
