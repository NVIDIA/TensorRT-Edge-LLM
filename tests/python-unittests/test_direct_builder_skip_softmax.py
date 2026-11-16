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

pytest.importorskip("tensorrt")

from tensorrt_edgellm.builder.core.artifacts.runtime_config import \
    build_runtime_config
from tensorrt_edgellm.builder.core.builder import BuildArgs
from tensorrt_edgellm.builder.core.config import DeviceConfig

_A, _B = 100.0, 4.0
_TABLE = [[8192, 6380], [12288, 7676], [16384, 9398]]


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

    assert cfg.skip_softmax_prefill_scale_table == []
    assert cfg.skip_softmax_decode_scale_table == []


@pytest.mark.parametrize("factor", [2048.0, 0.0])
def test_checkpoint_scale_factor_key_is_ignored(tmp_path, factor):
    """A checkpoint skip_softmax_scale_factor key, positive or zero, is not
    read: alone the engine is dense, next to a prefill fit + target the
    formula table deploys."""
    alone = DeviceConfig.from_pretrained(
        _write_config(tmp_path / "alone",
                      text_extra={"skip_softmax_scale_factor": factor}))
    assert alone.skip_softmax_prefill_scale_table == []
    assert "skip_softmax_prefill_scale_table" not in build_runtime_config(
        alone, _args())
    fit = DeviceConfig.from_pretrained(
        _write_config(tmp_path / "fit",
                      root_extra={
                          "skip_softmax_prefill_calibration": {
                              "a": _A,
                              "b": _B
                          },
                          "skip_softmax_prefill_target_sparsity": 0.75,
                      },
                      text_extra={"skip_softmax_scale_factor": factor}))
    assert fit.skip_softmax_prefill_scale_table == [[
        0, round(_A * math.exp(_B * 0.75))
    ]]


def test_skip_softmax_target_sparsity_uses_calibration_formula(tmp_path):
    cfg = DeviceConfig.from_pretrained(
        _write_config(tmp_path / "calib",
                      root_extra={
                          "skip_softmax_prefill_calibration": {
                              "a": _A,
                              "b": _B
                          },
                          "skip_softmax_prefill_target_sparsity": 0.75,
                      }))

    assert build_runtime_config(
        cfg, _args())["skip_softmax_prefill_scale_table"] == [[
            0, round(_A * math.exp(_B * 0.75))
        ]]


def test_skip_softmax_target_sparsity_requires_calibration(tmp_path):
    with pytest.raises(ValueError, match="skip_softmax_prefill_calibration"):
        DeviceConfig.from_pretrained(
            _write_config(
                tmp_path / "nocalib",
                root_extra={"skip_softmax_prefill_target_sparsity": 0.75}))


def _args():
    return BuildArgs(model_dir="ckpt", engine_dir="engine")


def test_scale_table_enables_layers_and_is_deployed_sorted(tmp_path):
    cfg = DeviceConfig.from_pretrained(
        _write_config(tmp_path / "table",
                      root_extra={
                          "skip_softmax_prefill_scale_table":
                          list(reversed(_TABLE))
                      }))

    assert cfg.skip_softmax_prefill_scale_table == _TABLE
    assert build_runtime_config(
        cfg, _args())["skip_softmax_prefill_scale_table"] == _TABLE


def test_scale_table_wins_over_target_sparsity(tmp_path):
    cfg = DeviceConfig.from_pretrained(
        _write_config(tmp_path / "both",
                      root_extra={
                          "skip_softmax_prefill_scale_table": _TABLE,
                          "skip_softmax_prefill_calibration": {
                              "a": _A,
                              "b": _B
                          },
                          "skip_softmax_prefill_target_sparsity": 0.75,
                      }))

    assert build_runtime_config(
        cfg, _args())["skip_softmax_prefill_scale_table"] == _TABLE


@pytest.mark.parametrize("factor", [2048.0, 0.0])
def test_checkpoint_scale_table_deploys_whatever_the_scale_factor_key_says(
        tmp_path, factor):
    """A checkpoint prefill table deploys next to a positive or zero
    skip_softmax_scale_factor key."""
    cfg = DeviceConfig.from_pretrained(
        _write_config(tmp_path / "table",
                      root_extra={"skip_softmax_prefill_scale_table": _TABLE},
                      text_extra={"skip_softmax_scale_factor": factor}))

    assert cfg.skip_softmax_prefill_scale_table == _TABLE
    assert build_runtime_config(
        cfg, _args())["skip_softmax_prefill_scale_table"] == _TABLE


def test_dense_runtime_config_carries_no_table(tmp_path):
    cfg = DeviceConfig.from_pretrained(_write_config(tmp_path / "dense"))

    assert "skip_softmax_prefill_scale_table" not in build_runtime_config(
        cfg, _args())


_BAD_ROWS = [[], [[8192]], [[8192, -1]], [[8192, 1.5]],
             [[b, 1] for b in range(129)], [[8192, 1], [8192, 2]]]


@pytest.mark.parametrize("rows", _BAD_ROWS)
def test_malformed_scale_table_is_rejected(tmp_path, rows):
    """Malformed prefill tables (shape, sign, integrality, capacity, repeated
    bins) fail at config load, named after the offending key."""
    with pytest.raises(ValueError, match="skip_softmax_prefill_scale_table"):
        DeviceConfig.from_pretrained(
            _write_config(
                tmp_path / "bad",
                root_extra={"skip_softmax_prefill_scale_table": rows}))


@pytest.mark.parametrize("rows", _BAD_ROWS)
def test_malformed_decode_scale_table_is_rejected(tmp_path, rows):
    """The decode table is validated by the same rules as the prefill one."""
    with pytest.raises(ValueError, match="skip_softmax_decode_scale_table"):
        DeviceConfig.from_pretrained(
            _write_config(tmp_path / "bad",
                          root_extra={"skip_softmax_decode_scale_table":
                                      rows}))


def test_unphased_prefill_table_key_is_not_read(tmp_path):
    """The pre-prefix checkpoint key skip_softmax_scale_table is not read:
    alone the engine is dense, next to skip_softmax_prefill_scale_table only
    the phased table deploys and the engine config carries only that name."""
    alone = DeviceConfig.from_pretrained(
        _write_config(tmp_path / "alone",
                      root_extra={"skip_softmax_scale_table": _TABLE}))
    assert alone.skip_softmax_prefill_scale_table == []
    assert alone.skip_softmax_prefill_scale_table == []
    assert "skip_softmax_prefill_scale_table" not in build_runtime_config(
        alone, _args())

    both = DeviceConfig.from_pretrained(
        _write_config(tmp_path / "both",
                      root_extra={
                          "skip_softmax_scale_table": [[0, 1]],
                          "skip_softmax_prefill_scale_table": _TABLE,
                      }))
    assert both.skip_softmax_prefill_scale_table == _TABLE
    runtime = build_runtime_config(both, _args())
    assert runtime["skip_softmax_prefill_scale_table"] == _TABLE
    assert "skip_softmax_scale_table" not in runtime
