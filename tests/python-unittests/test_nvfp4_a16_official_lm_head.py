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
"""Checkpoint-provided NVFP4 W4A16 lm_head (issue 703 / Qwen3.6-style)."""

import os
import sys

import pytest

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tensorrt_edgellm import config


@pytest.mark.parametrize(("model_type", "expected_quant_type"),
                         [("qwen3_moe", config.QUANT_NVFP4),
                          ("nemotron_h", config.QUANT_NVFP4_A16)],
                         ids=["qwen-generic-nvfp4", "nemotron-h-nvfp4-a16"])
def test_mixed_precision_w4a16_dispatch_is_model_specific(
        model_type, expected_quant_type):
    """Qwen uses its established repacker; Nemotron-H uses Marlin A16."""
    dominant, group_size, overrides = config._parse_mixed_precision(
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
    # Plugin takes one FC1 global; keep gate/up identical.
    up = [(u[0], u[1], g[2]) for u, g in zip(up, gate)]
    down = [_fake_nvfp4_a16(hidden, inter) for _ in range(experts)]
    fc1_q, fc1_s, fc1_g, fc2_q, fc2_s, fc2_g = (
        repack_nvfp4_a16_marlin_gated_moe_experts(
            [t[0] for t in gate], [t[1] for t in gate], [t[2] for t in gate],
            [t[0] for t in up], [t[1] for t in up], [t[2] for t in up],
            [t[0] for t in down], [t[1] for t in down], [t[2] for t in down],
            inter))
    assert tuple(fc1_q.shape) == (experts, hidden // 16, 8 * 2 * inter)
    assert tuple(fc1_s.shape) == (experts, hidden // 16, 2 * inter)
    assert tuple(fc1_g.shape) == (experts, )
    assert tuple(fc2_q.shape) == (experts, inter // 16, 8 * hidden)
    assert tuple(fc2_s.shape) == (experts, inter // 16, hidden)
    assert tuple(fc2_g.shape) == (experts, )
