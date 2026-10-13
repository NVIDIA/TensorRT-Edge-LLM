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
import math

import pytest

from experimental.builder.core.config import DeviceConfig

_A, _B = 100.0, 4.0


def _write_config(model_dir, root_extra=None, text_extra=None):
    model_dir.mkdir()
    text_config = {
        "model_type": "qwen3_5_text",
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "intermediate_size": 3584,
        "vocab_size": 248320,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000_000.0,
        "max_position_embeddings": 262144,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_value_head_dim": 128,
        **(text_extra or {}),
    }
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type":
        "qwen3_5",
        "text_config":
        text_config,
        **(root_extra or {}),
    }),
                                           encoding="utf-8")
    return str(model_dir)


def test_skip_softmax_defaults_to_disabled(tmp_path):
    cfg = DeviceConfig.from_pretrained(_write_config(tmp_path / "dense"))

    assert cfg.skip_softmax_scale_factor == 0.0


def test_skip_softmax_scale_factor_is_read_from_text_config(tmp_path):
    cfg = DeviceConfig.from_pretrained(
        _write_config(tmp_path / "scale",
                      text_extra={"skip_softmax_scale_factor": 2048.0}))

    assert cfg.skip_softmax_scale_factor == 2048.0


def test_skip_softmax_target_sparsity_uses_calibration_formula(tmp_path):
    cfg = DeviceConfig.from_pretrained(
        _write_config(tmp_path / "calib",
                      root_extra={
                          "skip_softmax_calibration": {
                              "a": _A,
                              "b": _B
                          },
                          "skip_softmax_target_sparsity": 0.75,
                      }))

    assert cfg.skip_softmax_scale_factor == pytest.approx(_A *
                                                          math.exp(_B * 0.75))


def test_skip_softmax_explicit_scale_factor_wins_over_target_sparsity(
        tmp_path):
    cfg = DeviceConfig.from_pretrained(
        _write_config(tmp_path / "both",
                      root_extra={
                          "skip_softmax_calibration": {
                              "a": _A,
                              "b": _B
                          },
                          "skip_softmax_target_sparsity": 0.75,
                      },
                      text_extra={"skip_softmax_scale_factor": 512.0}))

    assert cfg.skip_softmax_scale_factor == 512.0


def test_skip_softmax_target_sparsity_requires_calibration(tmp_path):
    with pytest.raises(ValueError, match="skip_softmax_calibration"):
        DeviceConfig.from_pretrained(
            _write_config(tmp_path / "nocalib",
                          root_extra={"skip_softmax_target_sparsity": 0.75}))
