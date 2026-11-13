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
"""calibrate_skip_softmax.py records its fit in the checkpoint config.json;
both export front-ends resolve what to deploy from those keys and the export
flags through ``tensorrt_edgellm.skip_softmax.resolve``. The tool module is
loaded from its path: its top level is argparse/json only, torch and modelopt
are imported inside main()."""
import argparse
import importlib.util
import json
import math
import os
import re

import pytest

from tensorrt_edgellm.builder.core.config import DeviceConfig
from tensorrt_edgellm.skip_softmax import (SKIP_SOFTMAX_SCALE_TABLE_MAX_LEN,
                                           SkipSoftmaxConfigError,
                                           SkipSoftmaxDeployment, phase_prefix,
                                           resolve, tier_key, validate_table)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
_TOOL_PATH = os.path.join(_ROOT, "kernelSrcs", "fmha_cutedsl_blackwell",
                          "calibrate_skip_softmax.py")

_A, _B, _TARGET = 100.0, 4.0, 0.75
_FORMULA_S = int(round(_A * math.exp(_B * _TARGET)))
_PREFILL = [[8192, 6380], [12288, 7676]]
_DECODE = [[8192, 250], [16384, 400]]


def _load_tool():
    spec = importlib.util.spec_from_file_location("calibrate_skip_softmax",
                                                  _TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_checkpoint_config(model_dir, root_extra=None, text_extra=None):
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type":
        "qwen3_5",
        "text_config": {
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
        },
        **(root_extra or {}),
    }),
                                           encoding="utf-8")
    return model_dir


def _builder_deploys(model_dir):
    """(prefill table, decode table) the ONNX-less builder emits."""
    from tensorrt_edgellm.builder.core.artifacts.runtime_config import \
        build_runtime_config
    from tensorrt_edgellm.builder.core.builder import BuildArgs
    cfg = DeviceConfig.from_pretrained(str(model_dir))
    runtime = build_runtime_config(
        cfg, BuildArgs(model_dir=str(model_dir), engine_dir="engine"))
    return (runtime.get("skip_softmax_prefill_scale_table"),
            runtime.get("skip_softmax_decode_scale_table"))


def _export_deploys(model_dir, **flags):
    """Same pair through the exporter's resolver with CLI-equivalent flags."""
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    dep = resolve(raw, raw["text_config"], **flags)
    return dep.prefill_table, dep.decode_table


def test_record_calibration_merges_root_keys_and_keeps_the_rest(tmp_path):
    tool = _load_tool()
    model_dir = _write_checkpoint_config(tmp_path / "ckpt")

    cfg_path = tool.record_calibration(model_dir, _A, _B, _TARGET)

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["skip_softmax_prefill_calibration"] == {
        "a": _A,
        "b": _B,
        "formula": "a * exp(b * target_sparsity)",
    }
    assert cfg["skip_softmax_prefill_target_sparsity"] == _TARGET
    assert cfg["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert cfg["text_config"]["head_dim"] == 256
    assert "skip_softmax_prefill_calibration" not in cfg["text_config"]


def test_recorded_config_drives_the_onnxless_builder(tmp_path):
    """A recorded fit + target deploys the flat table [[0, round(a*e^(bR))]]
    with no flags; the attribute is that table's S."""
    tool = _load_tool()
    model_dir = _write_checkpoint_config(tmp_path / "ckpt")
    tool.record_calibration(model_dir, _A, _B, _TARGET)

    assert _builder_deploys(model_dir) == ([[0, _FORMULA_S]], None)


def test_calibrate_writes_config_by_default_with_opt_out():
    tool = _load_tool()
    source = open(_TOOL_PATH, encoding="utf-8").read()
    assert re.search(
        r'"--no-write-config",\s*dest="write_config",\s*'
        r'action="store_false"', source)
    assert "if args.write_config:" in source
    assert hasattr(tool, "record_calibration")


# ---- per-length table -------------------------------------------------------

_THRESHOLDS = [1e-4, 1e-3, 1e-2, 1e-1]


def _sample(length, sparsities):
    return {"sample_length": length, "sparsity": sparsities}


def test_build_scale_table_interpolates_per_bin_and_guards_gaps():
    tool = _load_tool()
    bins = [4096, 8192, 12288]
    tap = {
        "thresholds":
        _THRESHOLDS,
        "stats": [
            # 4k: target 0.75 out of range -> below the first calibratable bin,
            # omitted (the runtime treats 4k requests as below the table).
            _sample(4100, [0.1, 0.2, 0.3, 0.4]),
            # 8k: two samples averaging to [0.2, 0.5, 0.8, 0.9]; 0.75 sits
            # between thresholds 1e-3 and 1e-2 -> lambda = 1e-3 + (0.25/0.3)*9e-3
            _sample(8200, [0.2, 0.4, 0.8, 0.9]),
            _sample(8100, [0.2, 0.6, 0.8, 0.9]),
            # a degenerate chunk (never sparse) is ignored
            _sample(8192, [0.0, 0.0, 0.01, 0.02]),
            # 12k: target out of range after a calibratable bin -> dense guard
            _sample(12300, [0.1, 0.2, 0.3, 0.4]),
        ],
    }

    table, per_bin = tool.build_scale_table(tap, bins, 0.75)

    lam = 1e-3 + (0.75 - 0.5) / (0.8 - 0.5) * (1e-2 - 1e-3)
    assert table == [[8192, int(round(lam * 8192))], [12288, 1]]
    assert per_bin[4096][3] is None
    assert per_bin[8192][0] == 2
    assert per_bin[12288][3] is None


def test_build_scale_table_uses_nearest_bin_for_off_grid_lengths():
    tool = _load_tool()
    tap = {
        "thresholds": _THRESHOLDS,
        "stats": [_sample(7000, [0.5, 0.75, 0.9, 0.95])],
    }

    table, per_bin = tool.build_scale_table(tap, [4096, 8192], 0.75)

    assert per_bin[4096][0] == 0
    assert table == [[8192, int(round(1e-3 * 8192))]]


def test_record_calibration_writes_table_and_clears_a_stale_one(tmp_path):
    tool = _load_tool()
    model_dir = _write_checkpoint_config(tmp_path / "ckpt")
    table = [[8192, 6380], [12288, 7676]]

    cfg_path = tool.record_calibration(model_dir, _A, _B, _TARGET, table)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["skip_softmax_prefill_scale_table"] == table
    assert DeviceConfig.from_pretrained(
        str(model_dir)).skip_softmax_prefill_scale_table == table

    tool.record_calibration(model_dir, _A, _B, _TARGET)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert "skip_softmax_prefill_scale_table" not in cfg
    assert cfg["skip_softmax_prefill_target_sparsity"] == _TARGET


# ---- tier ladder (no --target-sparsity) ---------------------------------------


def test_default_ladder_records_every_tier_and_deploys_75(tmp_path):
    tool = _load_tool()
    model_dir = _write_checkpoint_config(tmp_path / "ckpt")
    tables = {
        tool.tier_key(t): [[8192, int(1000 * t)]]
        for t in tool.DEFAULT_TIERS
    }

    cfg_path = tool.record_calibration(model_dir, _A, _B, tool.DEPLOY_TIER,
                                       tables[tool.tier_key(tool.DEPLOY_TIER)],
                                       tables)

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert tool.DEFAULT_TIERS == (0.5, 0.6, 0.75,
                                  0.8) and tool.DEPLOY_TIER == 0.75
    assert set(cfg["skip_softmax_prefill_scale_tables"]) == {
        "50%", "60%", "75%", "80%"
    }
    assert cfg["skip_softmax_prefill_scale_table"] == [[8192, 750]]
    assert cfg["skip_softmax_prefill_target_sparsity"] == 0.75


def test_builder_deploys_the_target_tiers_table(tmp_path):
    tool = _load_tool()
    model_dir = _write_checkpoint_config(tmp_path / "ckpt")
    tables = {"50%": [[8192, 500]], "75%": [[8192, 750]], "80%": [[8192, 800]]}
    tool.record_calibration(model_dir, _A, _B, 0.75, tables["75%"], tables)
    assert DeviceConfig.from_pretrained(
        str(model_dir)).skip_softmax_prefill_scale_table == [[8192, 750]]

    # re-target by editing the checkpoint's target: the builder follows the tier key
    cfg_path = model_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["skip_softmax_prefill_target_sparsity"] = 0.8
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    assert DeviceConfig.from_pretrained(
        str(model_dir)).skip_softmax_prefill_scale_table == [[8192, 800]]


def test_price_lengths_drop_what_the_tables_do_not_cover():
    tool = _load_tool()
    assert tool.price_lengths(list(tool.DEFAULT_PRICE_LENGTHS),
                              40960) == [8192, 16384, 32768, 40960]
    assert tool.price_lengths([40960, 8192, 8192, 16384],
                              16384) == [8192, 16384]
    assert tool.price_lengths([8192], 4096) == []


def test_parse_bench_ms_reads_prefill_stage_not_request_time():
    # Given a report with different request and prefill times,
    # when calibration reads it, then only the prefill-stage mean is priced.
    tool = _load_tool()
    report = {
        "schema_version": 2,
        "mode": "fixed_round",
        "statistics": {
            "request_wall_ms": {
                "mean": 99.0
            },
            "invocation_gpu_ms": {
                "llm_prefill": {
                    "count": 3,
                    "mean": 11.875
                }
            }
        }
    }
    assert tool.parse_bench_ms(report) == 11.875


@pytest.mark.parametrize("report", [
    {},
    {
        "schema_version": 1,
        "mode": "fixed_round",
        "statistics": {
            "invocation_gpu_ms": {
                "llm_prefill": {
                    "count": 1,
                    "mean": 1.0
                }
            }
        }
    },
    {
        "schema_version": 2,
        "mode": "generate",
        "statistics": {
            "invocation_gpu_ms": {
                "llm_prefill": {
                    "count": 1,
                    "mean": 1.0
                }
            }
        }
    },
    {
        "schema_version": 2,
        "mode": "fixed_round",
        "statistics": {}
    },
    {
        "schema_version": 2,
        "mode": "fixed_round",
        "statistics": {
            "invocation_gpu_ms": {
                "llm_prefill": {
                    "count": 0,
                    "mean": 1.0
                }
            }
        }
    },
    {
        "schema_version": 2,
        "mode": "fixed_round",
        "statistics": {
            "invocation_gpu_ms": {
                "llm_prefill": {
                    "count": 1,
                    "mean": float("nan")
                }
            }
        }
    },
])
def test_parse_bench_ms_rejects_missing_or_incompatible_samples(report):
    # Given missing or incompatible timing data, when pricing a report,
    # then calibration rejects it rather than using an unrelated timing field.
    with pytest.raises(ValueError):
        _load_tool().parse_bench_ms(report)


def test_config_tiers_follow_the_recorded_tables(tmp_path):
    tool = _load_tool()
    model_dir = _write_checkpoint_config(tmp_path / "ckpt")
    tables = {"50%": [[8192, 500]], "75%": [[8192, 750]], "80%": [[8192, 800]]}
    tool.record_calibration(model_dir, _A, _B, 0.75, tables["75%"], tables)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    assert tool.config_tiers(cfg) == [0.5, 0.75, 0.8]

    tool.record_calibration(model_dir, _A, _B, 0.6)  # a/b fit only
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    assert tool.config_tiers(cfg) == [0.6]
    with pytest.raises(ValueError, match="calibrate"):
        tool.config_tiers({})


def test_price_list_report_and_record(tmp_path):
    tool = _load_tool()
    model_dir = _write_checkpoint_config(tmp_path / "ckpt")
    tool.record_calibration(model_dir, _A, _B, 0.75, [[8192, 750]],
                            {"75%": [[8192, 750]]})
    price_list = {
        "lengths": [8192, 32768],
        "dense": {
            "8192": {
                "ttft_ms": 100.0,
                "ruler": 0.9,
                "longbench": 0.5
            },
            "32768": {
                "ttft_ms": 400.0,
                "longbench": 0.45
            },
        },
        "tiers": {
            "75%": {
                "8192": {
                    "ttft_ms": 90.0,
                    "speedup": 100.0 / 90.0,
                    "ruler": 0.89,
                    "ruler_drop": 0.01,
                    "longbench": 0.5,
                    "longbench_drop": 0.0
                },
                "32768": {
                    "ttft_ms": 320.0,
                    "speedup": 1.25,
                    "longbench": 0.44,
                    "longbench_drop": 0.01
                },
            }
        },
    }

    report = "\n".join(tool.format_price_report(price_list))
    assert "75%" in report and "1.11x" in report and "1.25x" in report
    assert "R-0.010" in report and "LB-0.010" in report

    cfg_path = tool.record_price_list(model_dir, price_list)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["skip_softmax_price_list"] == price_list
    assert cfg["skip_softmax_prefill_scale_table"] == [[8192,
                                                        750]]  # untouched


# ---- decode phase -----------------------------------------------------------


def test_record_calibration_decode_phase_writes_decode_keys_only(tmp_path):
    # Given a checkpoint with a prefill calibration, When the decode phase is
    # recorded, Then the skip_softmax_decode_* keys carry the decode fit and
    # tables and the prefill keys are left untouched.
    tool = _load_tool()
    model_dir = _write_checkpoint_config(tmp_path / "ckpt")
    tool.record_calibration(model_dir, _A, _B, 0.75, [[8192, 750]],
                            {"75%": [[8192, 750]]})
    tool.record_calibration(model_dir,
                            7.0,
                            3.0,
                            0.5, [[8192, 250]], {"50%": [[8192, 250]]},
                            phase="decode")

    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg["skip_softmax_decode_calibration"]["a"] == 7.0
    assert cfg["skip_softmax_decode_target_sparsity"] == 0.5
    assert cfg["skip_softmax_decode_scale_table"] == [[8192, 250]]
    assert cfg["skip_softmax_decode_scale_tables"] == {"50%": [[8192, 250]]}
    assert cfg["skip_softmax_prefill_scale_table"] == [[8192, 750]]
    assert cfg["skip_softmax_prefill_target_sparsity"] == 0.75
    with pytest.raises(ValueError, match="phase"):
        tool.record_calibration(model_dir, _A, _B, 0.75, phase="verify")


def test_builder_deploys_the_decode_table_and_enables_the_layers(tmp_path):
    # Given a checkpoint calibrated for decode only, When the ONNX-less builder
    # reads it, Then the decode table deploys, the attribute enables the
    # skip-softmax layers from that table and the runtime config carries the
    # decode table but no prefill table.
    from tensorrt_edgellm.builder.core.artifacts.runtime_config import \
        build_runtime_config
    from tensorrt_edgellm.builder.core.builder import BuildArgs
    tool = _load_tool()
    model_dir = _write_checkpoint_config(tmp_path / "ckpt")
    tool.record_calibration(model_dir,
                            7.0,
                            3.0,
                            0.5, [[8192, 250]], {
                                "50%": [[8192, 250]],
                                "75%": [[8192, 400]]
                            },
                            phase="decode")

    cfg = DeviceConfig.from_pretrained(str(model_dir))
    assert cfg.skip_softmax_decode_scale_table == [[8192, 250]]
    assert cfg.skip_softmax_prefill_scale_table == []
    runtime = build_runtime_config(
        cfg, BuildArgs(model_dir=str(model_dir), engine_dir="engine"))
    assert runtime["skip_softmax_decode_scale_table"] == [[8192, 250]]
    assert "skip_softmax_prefill_scale_table" not in runtime

    # re-target by the decode tier key
    cfg_path = model_dir / "config.json"
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    raw["skip_softmax_decode_target_sparsity"] = 0.75
    cfg_path.write_text(json.dumps(raw), encoding="utf-8")
    assert DeviceConfig.from_pretrained(
        str(model_dir)).skip_softmax_decode_scale_table == [[8192, 400]]


def test_arm_engine_removes_exactly_the_other_phases_table(tmp_path):
    # Given one skip engine carrying both tables, When an arm is selected,
    # Then `both` runs the engine itself and every other arm runs a linked copy
    # whose config.json lacks exactly the other phase's table (dense: both),
    # with the engine files reachable; an unknown arm is rejected.
    tool = _load_tool()
    eng = tmp_path / "eng"
    eng.mkdir()
    (eng / "llm.engine").write_bytes(b"x")
    (eng / "config.json").write_text(
        json.dumps({
            "skip_softmax_prefill_scale_table": [[0, 6000]],
            "skip_softmax_decode_scale_table": [[0, 60]],
            "k": 1
        }))
    assert tool._arm_engine(eng, "both") == eng
    keys = lambda arm: set(
        json.loads((tool._arm_engine(eng, arm) / "config.json").read_text()))
    assert keys("prefill") == {"skip_softmax_prefill_scale_table", "k"}
    assert keys("decode") == {"skip_softmax_decode_scale_table", "k"}
    assert keys("dense") == {"k"}
    assert (tool._arm_engine(eng, "dense") / "llm.engine").read_bytes() == b"x"
    dense = tool._arm_engine(eng, "dense")
    assert dense == tool._arm_engine(eng, "dense")
    with pytest.raises(ValueError):
        tool._arm_engine(eng, "verify")
    # the linked copies are temporary: the exit hook removes every one of them
    tool._cleanup_arm_dirs()
    assert not dense.exists() and tool._ARM_DIRS == {}


def test_price_list_report_shows_tpot_and_single_arm_rows():
    # Given a priced tier with TPOT and ablation arms, When the report is
    # rendered, Then each length cell shows TTFT x / TPOT x plus the `both`
    # accuracy delta, the single arms get their own accuracy rows and the
    # dense TPOT line is printed.
    tool = _load_tool()
    price_list = {
        "lengths": [16384],
        "dense": {
            "16384": {
                "ttft_ms": 200.0,
                "tpot_ms": 20.0,
                "ruler": 0.9
            }
        },
        "tiers": {
            "75%": {
                "16384": {
                    "ttft_ms": 160.0,
                    "speedup": 1.25,
                    "tpot_ms": 16.0,
                    "tpot_speedup": 1.25,
                    "ruler": 0.89,
                    "ruler_drop": 0.01,
                    "arms": {
                        "prefill": {
                            "ruler": 0.895,
                            "ruler_drop": 0.005
                        },
                        "decode": {
                            "ruler": 0.90,
                            "ruler_drop": 0.0
                        }
                    }
                }
            }
        },
        "arms": ["both", "prefill", "decode"],
    }
    report = "\n".join(tool.format_price_report(price_list))
    assert "1.25x/1.25x" in report and "R-0.010" in report
    assert "R-0.005" in report and "R+0.000" in report
    assert "tpot" in report and "20.00 ms" in report


# ---- shared resolver: both front-ends deploy the same tables ----------------


def test_shared_vocabulary_matches_the_runtime_contract():
    """tier_key / phase_prefix are the checkpoint key vocabulary and the table
    capacity is the one the builder profile and the runtime enforce."""
    assert tier_key(0.75) == "75%" and tier_key(0.8) == "80%"
    assert phase_prefix("prefill") == "skip_softmax_prefill_"
    assert phase_prefix("decode") == "skip_softmax_decode_"
    with pytest.raises(ValueError, match="phase"):
        phase_prefix("verify")
    assert SKIP_SOFTMAX_SCALE_TABLE_MAX_LEN == 256


@pytest.mark.parametrize("rows", [
    [],
    [[8192]],
    [[8192, -1]],
    [[8192, 1.5]],
    [[True, 1]],
    "[[0, 1]]",
    [[b, 1] for b in range(129)],
    [[8192, 1], [8192, 2]],
])
def test_validate_table_rejects_what_the_runtime_parser_rejects(rows):
    """A table is non-empty integral non-negative [bin, S] rows, at most 128
    of them, with unique bins; the error names the offending key."""
    with pytest.raises(SkipSoftmaxConfigError, match="my_table"):
        validate_table(rows, "my_table")


def test_validate_table_sorts_rows_by_bin():
    """Rows are deployed bin-ascending whatever order the file lists them."""
    assert validate_table([[12288, 7676.0], [8192, 6380]]) == _PREFILL


def test_deployment_flags_follow_the_deployed_tables():
    """prefill_enabled / decode_enabled iff that phase's table is deployed;
    enabled iff either is."""
    dense = SkipSoftmaxDeployment()
    assert not (dense.prefill_enabled or dense.decode_enabled or dense.enabled)
    prefill_only = SkipSoftmaxDeployment(prefill_table=_PREFILL)
    assert prefill_only.prefill_enabled and not prefill_only.decode_enabled
    decode_only = SkipSoftmaxDeployment(decode_table=_DECODE)
    assert decode_only.decode_enabled and not decode_only.prefill_enabled
    assert decode_only.enabled


def test_both_paths_formula_prefill_plus_decode_table(tmp_path):
    """Formula prefill (fit + target, no prefill table) and a decode table:
    prefill deploys the fit's flat S and decode its table on both paths."""
    model_dir = _write_checkpoint_config(
        tmp_path / "ckpt",
        root_extra={
            "skip_softmax_prefill_calibration": {
                "a": _A,
                "b": _B
            },
            "skip_softmax_prefill_target_sparsity": _TARGET,
            "skip_softmax_decode_scale_table": _DECODE,
        })
    expected = ([[0, _FORMULA_S]], _DECODE)
    assert _builder_deploys(model_dir) == expected
    assert _export_deploys(model_dir) == expected


@pytest.mark.parametrize("factor", [2048.0, 0.0])
def test_both_paths_ignore_the_checkpoint_scale_factor_key(tmp_path, factor):
    """A checkpoint skip_softmax_scale_factor key, positive or zero, neither
    deploys nor disables anything on either path: the phase tables / tiers
    decide."""
    tables = _write_checkpoint_config(
        tmp_path / "tables",
        root_extra={
            "skip_softmax_prefill_scale_table": _PREFILL,
            "skip_softmax_decode_scale_table": _DECODE,
        },
        text_extra={"skip_softmax_scale_factor": factor})
    expected = (_PREFILL, _DECODE)
    assert _builder_deploys(tables) == expected
    assert _export_deploys(tables) == expected
    alone = _write_checkpoint_config(
        tmp_path / "alone", text_extra={"skip_softmax_scale_factor": factor})
    assert _builder_deploys(alone) == (None, None)
    assert _export_deploys(alone) == (None, None)


def test_both_paths_decode_formula_from_the_checkpoint_fit(tmp_path):
    """Decode mirrors prefill: a checkpoint decode fit + target with no decode
    table deploys the flat [[0, round(a*e^(bR))]] on both paths."""
    s_decode = int(round(7.0 * math.exp(3.0 * 0.5)))
    model_dir = _write_checkpoint_config(
        tmp_path / "ckpt",
        root_extra={
            "skip_softmax_decode_calibration": {
                "a": 7.0,
                "b": 3.0
            },
            "skip_softmax_decode_target_sparsity": 0.5,
        })
    expected = (None, [[0, s_decode]])
    assert _builder_deploys(model_dir) == expected
    assert _export_deploys(model_dir) == expected
    with pytest.raises(SkipSoftmaxConfigError,
                       match="skip_softmax_decode_calibration"):
        _export_deploys(
            _write_checkpoint_config(
                tmp_path / "nofit",
                root_extra={"skip_softmax_decode_target_sparsity": 0.5}))


def test_both_paths_decode_only_keeps_prefill_dense(tmp_path):
    """Decode-only checkpoint: no prefill table is synthesised, the attribute
    comes from the decode table, and the engine config carries only it."""
    model_dir = _write_checkpoint_config(
        tmp_path / "ckpt",
        root_extra={"skip_softmax_decode_scale_table": _DECODE})
    assert _builder_deploys(model_dir) == (None, _DECODE)
    assert _export_deploys(model_dir) == (None, _DECODE)
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    dep = resolve(raw, raw["text_config"])
    assert dep.decode_enabled and dep.prefill_calibration is None
    assert dep.prefill_target_sparsity is None


def test_export_flags_select_phases_independently(tmp_path):
    """Prefill tiers 75/80 plus decode tiers: --decode-target-sparsity alone
    keeps the checkpoint's prefill tier, --prefill-target-sparsity alone keeps
    the checkpoint's decode tier, and each flag switches only its own phase."""
    tables = {"75%": [[8192, 750]], "80%": [[8192, 800]]}
    decode_tables = {"50%": [[8192, 250]], "80%": [[8192, 280]]}
    model_dir = _write_checkpoint_config(
        tmp_path / "ckpt",
        root_extra={
            "skip_softmax_prefill_scale_tables": tables,
            "skip_softmax_prefill_scale_table": tables["75%"],
            "skip_softmax_prefill_target_sparsity": 0.75,
            "skip_softmax_decode_scale_tables": decode_tables,
            "skip_softmax_decode_scale_table": decode_tables["50%"],
            "skip_softmax_decode_target_sparsity": 0.5,
        })
    assert _builder_deploys(model_dir) == ([[8192, 750]], [[8192, 250]])
    assert _export_deploys(model_dir) == ([[8192, 750]], [[8192, 250]])
    assert _export_deploys(model_dir,
                           decode_target_sparsity=0.8) == ([[8192, 750]],
                                                           [[8192, 280]])
    assert _export_deploys(model_dir,
                           prefill_target_sparsity=0.8) == ([[8192, 800]],
                                                            [[8192, 250]])
    assert _export_deploys(model_dir,
                           prefill_target_sparsity=0.8,
                           decode_target_sparsity=0.8) == ([[8192, 800]],
                                                           [[8192, 280]])


def test_export_tier_follows_the_checkpoint_target_over_the_deploy_default(
        tmp_path):
    """Without flags the checkpoint's target tier beats the deploy-default
    table (re-target by editing the target, like the builder test above)."""
    tables = {"75%": [[8192, 750]], "80%": [[8192, 800]]}
    model_dir = _write_checkpoint_config(
        tmp_path / "ckpt",
        root_extra={
            "skip_softmax_prefill_scale_tables": tables,
            "skip_softmax_prefill_scale_table": tables["75%"],
            "skip_softmax_prefill_target_sparsity": 0.8,
        })
    assert _export_deploys(model_dir) == ([[8192, 800]], None)
    assert _builder_deploys(model_dir) == ([[8192, 800]], None)


def test_export_prefill_flag_precedence(tmp_path):
    """Prefill: table file > tier of --prefill-target-sparsity > checkpoint
    keys > fit; a target with no tier table and no fit errors."""
    model_dir = _write_checkpoint_config(
        tmp_path / "ckpt",
        root_extra={
            "skip_softmax_prefill_scale_tables": {
                "80%": [[8192, 800]]
            },
            "skip_softmax_prefill_scale_table": _PREFILL,
            "skip_softmax_prefill_calibration": {
                "a": _A,
                "b": _B
            },
        })
    file_table = [[4096, 10], [8192, 20]]
    assert _export_deploys(model_dir,
                           prefill_table=file_table,
                           prefill_target_sparsity=0.8) == (file_table, None)
    assert _export_deploys(model_dir,
                           prefill_target_sparsity=0.8) == ([[8192,
                                                              800]], None)
    assert _export_deploys(model_dir,
                           prefill_target_sparsity=0.75) == ([[0, _FORMULA_S]],
                                                             None)
    assert _export_deploys(model_dir) == (_PREFILL, None)
    bare = _write_checkpoint_config(tmp_path / "bare")
    with pytest.raises(SkipSoftmaxConfigError,
                       match="skip_softmax_prefill_calibration"):
        _export_deploys(bare, prefill_target_sparsity=0.75)


def test_export_decode_tier_without_table_uses_the_fit_or_errors(tmp_path):
    """--decode-target-sparsity for a tier with no table derives S from
    skip_softmax_decode_calibration, and errors without one; a decode table
    file outranks both."""
    fit = _write_checkpoint_config(
        tmp_path / "fit",
        root_extra={"skip_softmax_decode_calibration": {
            "a": 7.0,
            "b": 3.0
        }})
    s = int(round(7.0 * math.exp(3.0 * 0.5)))
    assert _export_deploys(fit, decode_target_sparsity=0.5) == (None, [[0, s]])
    assert _export_deploys(fit,
                           decode_target_sparsity=0.5,
                           decode_table=_DECODE) == (None, _DECODE)
    bare = _write_checkpoint_config(tmp_path / "bare")
    with pytest.raises(SkipSoftmaxConfigError, match="no '50%' table"):
        _export_deploys(bare, decode_target_sparsity=0.5)


def _calibrated_both_phases(tmp_path):
    return _write_checkpoint_config(tmp_path / "ckpt",
                                    root_extra={
                                        "skip_softmax_prefill_scale_tables": {
                                            "80%": [[8192, 800]]
                                        },
                                        "skip_softmax_prefill_scale_table":
                                        _PREFILL,
                                        "skip_softmax_prefill_calibration": {
                                            "a": _A,
                                            "b": _B
                                        },
                                        "skip_softmax_prefill_target_sparsity":
                                        _TARGET,
                                        "skip_softmax_decode_scale_table":
                                        _DECODE,
                                    })


def test_export_prefill_none_makes_prefill_dense_and_keeps_decode(tmp_path):
    """--skip-softmax-prefill-scale-table none: prefill dense over every
    checkpoint prefill key and over --prefill-target-sparsity, no prefill
    metadata recorded; decode still comes from the checkpoint."""
    model_dir = _calibrated_both_phases(tmp_path)
    assert _export_deploys(model_dir, prefill_table=[]) == (None, _DECODE)
    assert _export_deploys(model_dir,
                           prefill_table=[],
                           prefill_target_sparsity=0.8) == (None, _DECODE)
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    dep = resolve(raw, raw["text_config"], prefill_table=[])
    assert (dep.prefill_calibration, dep.prefill_target_sparsity) == (None,
                                                                      None)


def test_export_decode_none_makes_decode_dense_and_keeps_prefill(tmp_path):
    """--skip-softmax-decode-scale-table none: decode dense over the
    checkpoint decode table and over --decode-target-sparsity; prefill still
    comes from the checkpoint."""
    model_dir = _calibrated_both_phases(tmp_path)
    assert _export_deploys(model_dir, decode_table=[]) == (_PREFILL, None)
    assert _export_deploys(model_dir,
                           decode_table=[],
                           decode_target_sparsity=0.5) == (_PREFILL, None)


def test_export_both_none_is_dense_and_records_nothing(tmp_path):
    """Both table flags none on a fully calibrated checkpoint: attribute 0
    and the engine config gets no skip_softmax key at all."""
    from tensorrt_edgellm.onnx.cli import _record_skip_softmax
    model_dir = _calibrated_both_phases(tmp_path)
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    dep = resolve(raw, raw["text_config"], prefill_table=[], decode_table=[])
    assert dep == SkipSoftmaxDeployment()
    assert not dep.enabled
    out = tmp_path / "llm"
    out.mkdir()
    (out / "config.json").write_text(json.dumps({"k": 1}))
    _record_skip_softmax(str(out), dep)
    assert json.loads((out / "config.json").read_text()) == {"k": 1}


@pytest.mark.parametrize("spelling", ["none", "NONE", "0", "FILE"])
def test_cli_table_flag_none_zero_or_empty_file_mean_dense(tmp_path, spelling):
    """The CLI maps `none` (any case), `0` and a JSON file holding [] to the
    same explicit-dense value, and a table file to its rows."""
    from tensorrt_edgellm.onnx.cli import _skip_softmax_table_flag
    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    value = str(empty) if spelling == "FILE" else spelling
    assert _skip_softmax_table_flag(value) == []
    assert _skip_softmax_table_flag(None) is None
    table = tmp_path / "table.json"
    table.write_text(json.dumps(_DECODE))
    assert _skip_softmax_table_flag(str(table)) == _DECODE
    model_dir = _calibrated_both_phases(tmp_path)
    args = argparse.Namespace(prefill_target_sparsity=None,
                              skip_softmax_prefill_scale_table=value,
                              decode_target_sparsity=None,
                              skip_softmax_decode_scale_table=str(table))
    from tensorrt_edgellm.onnx.cli import _resolve_skip_softmax
    dep = _resolve_skip_softmax(
        args, json.loads((model_dir / "config.json").read_text()))
    assert (dep.prefill_table, dep.decode_table) == (None, _DECODE)


@pytest.mark.parametrize("flag",
                         ["prefill_target_sparsity", "decode_target_sparsity"])
@pytest.mark.parametrize("value", [0.0, 1.0, 1.5, 75])
def test_export_target_sparsity_must_lie_in_the_open_unit_interval(
        tmp_path, flag, value):
    """Targets outside (0, 1) are rejected instead of becoming a "7500%" tier
    that silently falls back to the formula."""
    model_dir = _write_checkpoint_config(
        tmp_path / "ckpt",
        root_extra={
            "skip_softmax_prefill_calibration": {
                "a": _A,
                "b": _B
            },
            "skip_softmax_decode_calibration": {
                "a": _A,
                "b": _B
            },
        })
    with pytest.raises(SkipSoftmaxConfigError, match=r"\(0, 1\)"):
        _export_deploys(model_dir, **{flag: value})


@pytest.mark.parametrize("fit", [{"a": 1.0}, {"a": "x", "b": 1.0}, [1.0, 2.0]])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_export_checkpoint_fit_is_validated(tmp_path, fit, phase):
    """A checkpoint skip_softmax_<phase>_calibration must be an object with
    numeric a and b; the error names the key."""
    model_dir = _write_checkpoint_config(
        tmp_path / "ckpt",
        root_extra={f"skip_softmax_{phase}_calibration": fit})
    with pytest.raises(SkipSoftmaxConfigError,
                       match=f"skip_softmax_{phase}_calibration.*numeric"):
        _export_deploys(model_dir)


def test_export_metadata_only_describes_a_deployed_prefill_table(tmp_path):
    """calibration / target_sparsity are recorded only with a prefill table:
    an explicit-dense prefill carries none, a table file carries the fit but
    no target, a formula export carries both."""
    model_dir = _write_checkpoint_config(
        tmp_path / "ckpt",
        root_extra={
            "skip_softmax_prefill_calibration": {
                "a": _A,
                "b": _B
            },
            "skip_softmax_prefill_target_sparsity": _TARGET,
        })
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    formula = resolve(raw, raw["text_config"])
    assert (formula.prefill_calibration, formula.prefill_target_sparsity) == ({
        "a":
        _A,
        "b":
        _B
    }, _TARGET)
    table = resolve(raw, raw["text_config"], prefill_table=_PREFILL)
    assert (table.prefill_calibration, table.prefill_target_sparsity) == ({
        "a":
        _A,
        "b":
        _B
    }, None)
    dense = resolve(raw, raw["text_config"], prefill_table=[])
    assert (dense.prefill_calibration, dense.prefill_target_sparsity) == (None,
                                                                          None)
    # a fit that selected nothing is not recorded either: a decode-only
    # deployment next to a prefill fit
    decode_only = resolve(raw,
                          raw["text_config"],
                          prefill_table=[],
                          decode_table=_DECODE)
    assert decode_only.decode_table == _DECODE
    assert (decode_only.prefill_calibration,
            decode_only.prefill_target_sparsity) == (None, None)


def test_export_reads_checkpoint_keys_from_text_config_too(tmp_path):
    """The exporter reads the nested (thinker) text config before the root,
    like the builder, and a decode table flag leaves that prefill key alone."""
    from tensorrt_edgellm.onnx.cli import _resolve_skip_softmax
    table_path = tmp_path / "decode.json"
    table_path.write_text(json.dumps(_DECODE))
    args = argparse.Namespace(prefill_target_sparsity=None,
                              skip_softmax_prefill_scale_table=None,
                              decode_target_sparsity=None,
                              skip_softmax_decode_scale_table=str(table_path))
    config = {
        "thinker_config": {
            "text_config": {
                "skip_softmax_prefill_scale_table": _PREFILL
            }
        }
    }
    dep = _resolve_skip_softmax(args, config)
    assert dep.prefill_table == _PREFILL and dep.decode_table == _DECODE
    args.skip_softmax_prefill_scale_table = "none"
    dep = _resolve_skip_softmax(args, config)
    assert dep.prefill_table is None and dep.decode_table == _DECODE


@pytest.mark.parametrize("factor", [2048.0, 0.0])
def test_model_config_does_not_read_the_checkpoint_scale_factor(
        tmp_path, factor):
    """A checkpoint skip_softmax_scale_factor key enables nothing on the
    ONNX-side ModelConfig: the exporter sets the two enable flags from the
    resolved deployment."""
    from tensorrt_edgellm.config import ModelConfig
    model_dir = _write_checkpoint_config(
        tmp_path / "ckpt", text_extra={"skip_softmax_scale_factor": factor})
    cfg = ModelConfig.from_pretrained(str(model_dir),
                                      lambda head_dim: head_dim**-0.5)
    assert not cfg.skip_softmax_prefill_enabled
    assert not cfg.skip_softmax_decode_enabled


# ---- unphased (pre-prefix) prefill spellings are not read -----------------

_UNPHASED_PREFILL = {
    "skip_softmax_scale_tables": {
        "75%": [[8192, 750]],
        "80%": [[8192, 800]]
    },
    "skip_softmax_scale_table": [[8192, 750]],
    "skip_softmax_calibration": {
        "a": _A,
        "b": _B
    },
    "skip_softmax_target_sparsity": 0.8,
}


def test_unphased_prefill_keys_are_not_read(tmp_path):
    """Checkpoint keys without a phase (skip_softmax_{scale_tables,
    scale_table,calibration,target_sparsity}) deploy nothing on either path;
    only the skip_softmax_prefill_* spellings are read, and next to them the
    unphased keys change nothing."""
    unphased = _write_checkpoint_config(tmp_path / "unphased",
                                        root_extra=_UNPHASED_PREFILL)
    assert _builder_deploys(unphased) == (None, None)
    assert _export_deploys(unphased) == (None, None)
    raw = json.loads((unphased / "config.json").read_text(encoding="utf-8"))
    dep = resolve(raw, raw["text_config"])
    assert (dep.prefill_calibration, dep.prefill_target_sparsity) == (None,
                                                                      None)
    mixed = _write_checkpoint_config(
        tmp_path / "mixed",
        root_extra={
            **_UNPHASED_PREFILL, "skip_softmax_prefill_scale_table": _PREFILL
        })
    assert _builder_deploys(mixed) == (_PREFILL, None)
    assert _export_deploys(mixed) == (_PREFILL, None)


def test_unphased_target_without_a_phased_fit_is_not_a_target(tmp_path):
    """--prefill-target-sparsity next to only an unphased fit / tier table
    errors (no phased calibration), instead of silently using them."""
    unphased = _write_checkpoint_config(tmp_path / "unphased",
                                        root_extra=_UNPHASED_PREFILL)
    with pytest.raises(SkipSoftmaxConfigError,
                       match="skip_softmax_prefill_calibration"):
        _export_deploys(unphased, prefill_target_sparsity=0.8)


def test_tool_ignores_unphased_prefill_keys_but_still_drops_them(tmp_path):
    """The tool's readers (tiers, price arms) see only skip_softmax_prefill_*;
    a prefill calibration still removes the unphased keys it does not read,
    and an engine config.json's unphased table is left alone by every arm."""
    tool = _load_tool()
    with pytest.raises(ValueError, match="calibrate"):
        tool.config_tiers({
            "skip_softmax_scale_tables": {
                "50%": [[8192, 500]]
            },
            "skip_softmax_target_sparsity": 0.6,
        })
    assert tool.config_tiers({
        "skip_softmax_prefill_scale_tables": {
            "50%": [[8192, 500]]
        },
        "skip_softmax_target_sparsity": 0.6,
    }) == [0.5]
    model_dir = _write_checkpoint_config(tmp_path / "ckpt",
                                         root_extra=_UNPHASED_PREFILL)
    tool.record_calibration(model_dir, 2.0, 3.0, 0.5, [[8192, 5]],
                            {"50%": [[8192, 5]]})
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    assert not any(k in cfg for k in _UNPHASED_PREFILL)
    assert cfg["skip_softmax_prefill_scale_tables"] == {"50%": [[8192, 5]]}
    assert cfg["skip_softmax_prefill_target_sparsity"] == 0.5
    eng = tmp_path / "eng"
    eng.mkdir()
    (eng / "config.json").write_text(
        json.dumps({
            "skip_softmax_scale_table": [[0, 6000]],
            "skip_softmax_prefill_scale_table": [[0, 6000]],
            "skip_softmax_decode_scale_table": [[0, 60]],
        }))
    keys = lambda arm: set(
        json.loads((tool._arm_engine(eng, arm) / "config.json").read_text()))
    assert keys("decode") == {
        "skip_softmax_scale_table", "skip_softmax_decode_scale_table"
    }
    assert keys("dense") == {"skip_softmax_scale_table"}
    tool._cleanup_arm_dirs()


def test_exporter_records_canonical_keys_in_every_engine_config(tmp_path):
    """The exporter writes the deployed tables and the prefill fit / target
    under skip_softmax_prefill_* / skip_softmax_decode_* into config.json
    and every config_tp*.json, nothing under the legacy names, and leaves a
    dense export's configs untouched."""
    from tensorrt_edgellm.onnx.cli import _record_skip_softmax
    out = tmp_path / "llm"
    out.mkdir()
    for name in ("config.json", "config_tp2.json", "other.json"):
        (out / name).write_text(json.dumps({"k": 1}))
    _record_skip_softmax(str(out), SkipSoftmaxDeployment())
    assert all(
        json.loads((out / n).read_text()) == {"k": 1}
        for n in ("config.json", "config_tp2.json", "other.json"))
    _record_skip_softmax(
        str(out),
        SkipSoftmaxDeployment(prefill_table=_PREFILL,
                              decode_table=_DECODE,
                              prefill_calibration={
                                  "a": _A,
                                  "b": _B
                              },
                              prefill_target_sparsity=_TARGET))
    for name in ("config.json", "config_tp2.json"):
        cfg = json.loads((out / name).read_text())
        assert cfg == {
            "k": 1,
            "skip_softmax_prefill_scale_table": _PREFILL,
            "skip_softmax_decode_scale_table": _DECODE,
            "skip_softmax_prefill_calibration": {
                "a": _A,
                "b": _B,
                "formula": "a * exp(b * target_sparsity)"
            },
            "skip_softmax_prefill_target_sparsity": _TARGET,
        }
    assert json.loads((out / "other.json").read_text()) == {"k": 1}


# ---- shape advisor: mass share priced against sparsity ------------------------


def test_tile_mass_share_prices_exactly_the_tiles_the_skip_rule_drops():
    # Given a 4x4 score matrix in 2x2 tiles whose second column tile sits
    # 20 nats below the first in every row,
    # When the dropped mass share is read at three lambdas,
    # Then it is the exact softmax mass of the tiles the streaming rule drops:
    # nothing at ln(lambda) below the gap, the second tile's mass above it,
    # everything at lambda >= 1 (gap <= 0 can never beat ln(lambda) >= 0).
    torch = pytest.importorskip("torch")
    tool = _load_tool()
    scores = torch.tensor([[0.0, 0.0, -20.0, -20.0]] * 4).view(1, 1, 4, 4)

    shares = tool.tile_mass_share(scores, [-25.0, -10.0, 0.0], br=2, bc=2)

    expected_second_tile = 2 * math.exp(-20.0) / (2 + 2 * math.exp(-20.0))
    assert shares[0] == 0.0
    assert shares[1] == pytest.approx(expected_second_tile, rel=1e-3)
    assert shares[2] == pytest.approx(1.0)

    # Given the second tile RAISES the running max by 5 instead (streaming
    # rule), When read at ln(lambda) = -3, Then neither tile is dropped: the
    # first tile's gap is measured against the running max at its own
    # position (0), not the final row max (which would drop it at -5 < -3).
    rising = torch.tensor([[0.0, 0.0, 5.0, 5.0]] * 4).view(1, 1, 4, 4)
    assert tool.tile_mass_share(rising, [-3.0], br=2, bc=2) == [0.0]

    # Given rows with different softmax denominators (rows 0-1 spread, rows
    # 2-3 peaked), When the second column tile is dropped everywhere,
    # Then the share is the slide's mass(b) = sum exp(x - x_max) ratio — the
    # spread rows weigh more — not the row-normalized mean (0.1345 here).
    mixed = torch.tensor([[0.0, 0.0, -1.0, -1.0]] * 2 +
                         [[0.0, -30.0, -30.0, -30.0]] * 2).view(1, 1, 4, 4)
    e1, e30 = math.exp(-1.0), math.exp(-30.0)
    numerator_share = (2 * 2 * e1 + 2 * 2 * e30) / (2 * (2 + 2 * e1) + 2 *
                                                    (1 + 3 * e30))
    assert tool.tile_mass_share(mixed, [-0.5], br=2,
                                bc=2)[0] == pytest.approx(numerator_share,
                                                          rel=1e-4)


def test_shape_verdict_is_one_line_on_mass_over_sparsity():
    # Given the advisor's two numbers at the deployed lambda,
    # When the verdict is taken,
    # Then it is ON below mass / sparsity = 0.3 and OFF at or above it, the
    # line is movable, and the four slide models fall on the right sides
    # (Qwen3-8B 0.8% / Llama 3% of a 50% ask -> ON; hybrids 48-50% -> OFF).
    tool = _load_tool()
    assert tool.FLAT_MASS_RATIO == 0.3
    assert tool.shape_verdict(0.5, 0.008)[0] == "on"
    assert tool.shape_verdict(0.5, 0.03)[0] == "on"
    assert tool.shape_verdict(0.5, 0.48)[0] == "off"
    assert tool.shape_verdict(0.5, 0.50)[0] == "off"
    assert tool.shape_verdict(0.5, 0.15)[0] == "off"  # exactly on the line
    assert tool.shape_verdict(0.5, 0.149)[0] == "on"
    assert tool.shape_verdict(0.5, 0.15, ratio=0.35)[0] == "on"
    assert tool.shape_verdict(0.0, 0.0)[0] == "on"  # nothing skipped


def _point(layer, length, sparsity, mass, thresholds=(0.1, 0.5, 0.9)):
    return {
        "layer": layer,
        "sample_length": length,
        "thresholds": list(thresholds),
        "sparsity": list(sparsity),
        "mass_share": list(mass)
    }


def test_advise_shape_reads_the_deployed_lambda_and_pools_per_request():
    # Given two layers tapped over three prefill calls each — request 0 at
    # L=4096 and request 1 at L=4200 (the table row [4096, 3072] deploys
    # lambda = 3072 / L for both), request 2 at L=2048 (below the first bin:
    # the kernel runs dense there) — with an A-shaped layer (mass <<
    # sparsity) and a flat layer (mass == sparsity),
    # When the deployed table is priced,
    # Then the dense request is dropped, the two others are pooled per request
    # (layer mean) with curves interpolated in ln(lambda) between the 0.5 and
    # 0.9 trials, the mass TAIL is the larger request (not the mean of the
    # two), the per-bin row exists for 4096, and only the flat layer is flagged.
    tool = _load_tool()
    a_shape = ([0.5, 0.75, 0.9], [0.005, 0.01, 0.02])
    flat = ([0.5, 0.75, 0.9], [0.5, 0.75, 0.9])
    scatter = []
    for length in (4096, 4200, 2048):
        scatter.append(_point("A", length, *a_shape))
        scatter.append(_point("F", length, *flat))

    advice = tool.advise_shape(scatter, [[4096, 3072]], bins=[4096, 8192])

    def at(length):
        t = math.log(3072 / length / 0.5) / math.log(0.9 / 0.5)
        sparsity = 0.75 + t * (0.9 - 0.75)
        a_mass = 0.01 + t * (0.02 - 0.01)
        return sparsity, (a_mass + sparsity) / 2  # pooled over the two layers

    (s0, m0), (s1, m1) = at(4096), at(4200)
    assert m0 > m1 + 1e-3  # the longer request runs the smaller lambda
    assert advice["requests"] == 2
    assert advice["sparsity"] == pytest.approx((s0 + s1) / 2, abs=1e-4)
    assert advice["mass_share_tail"] == pytest.approx(m0, abs=1e-4)
    assert advice["mass_share_mean"] == pytest.approx((m0 + m1) / 2, abs=1e-4)
    assert advice["verdict"] == "off"  # half the layers flat -> ratio ~0.5
    assert list(advice["per_bin"]) == ["4096"]
    assert advice["flat_layers"] == [1] and advice["num_layers"] == 2

    # Given bins, Then a sample a few tokens short of its bin is priced at
    # that bin's S (the convention the table was built from); without bins
    # the kernel's floor rule runs it dense; the S=1 guard row is dense too.
    assert tool.lookup_scale([[4096, 3072]], 4000, bins=[4096, 8192]) == 3072
    assert tool.lookup_scale([[4096, 3072]], 4000) is None
    assert tool.lookup_scale([[4096, 1]], 4200, bins=[4096]) is None

    # Given only the A-shaped layer, Then the advice is ON.
    only_a = [p for p in scatter if p["layer"] == "A"]
    assert tool.advise_shape(only_a, [[4096, 3072]])["verdict"] == "on"
    # Given a table the kernel guards everywhere (S >= L), Then there is no advice.
    assert tool.advise_shape(scatter, [[0, 8192]]) is None

    # Given no deployable table (the hybrid ladder: sparsity never reaches
    # the ask) and a flat layer that tops out at 30% with 30% of the mass,
    # When read at the 75% ask, Then the point is read at its most sparse
    # trial and priced OFF — the advisor still speaks.
    capped = [_point("F", 4096, [0.1, 0.2, 0.3], [0.1, 0.2, 0.3])]
    advice = tool.advise_shape(capped, None, target=0.75)
    assert advice["sparsity"] == pytest.approx(0.3)
    assert advice["mass_share_tail"] == pytest.approx(0.3)
    assert advice["verdict"] == "off"


def test_record_calibration_writes_and_clears_shape_advice(tmp_path):
    # Given a calibration recorded with advisor output,
    # When config.json is read back and then re-recorded without advice,
    # Then skip_softmax_prefill_shape_advice is present, then removed, while the
    # deployed table is unaffected either way.
    tool = _load_tool()
    model_dir = _write_checkpoint_config(tmp_path / "ckpt")
    advice = {"verdict": "on", "sparsity": 0.75, "mass_share_tail": 0.02}

    cfg_path = tool.record_calibration(model_dir,
                                       _A,
                                       _B,
                                       0.75, [[8192, 750]],
                                       {"75%": [[8192, 750]]},
                                       phase="prefill",
                                       shape_advice=advice)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["skip_softmax_prefill_shape_advice"] == advice
    assert cfg["skip_softmax_prefill_scale_table"] == [[8192, 750]]

    tool.record_calibration(model_dir,
                            _A,
                            _B,
                            0.75, [[8192, 750]],
                            phase="prefill")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert "skip_softmax_prefill_shape_advice" not in cfg
    assert cfg["skip_softmax_prefill_scale_table"] == [[8192, 750]]


def test_parse_inference_ttft_takes_the_last_average_line():
    # Given llm_inference --dumpProfile output with a warm-up and a measured run
    # When the TTFT is parsed
    # Then the last "Average Time per Run" wins and a run without it yields None
    tool = _load_tool()
    text = ("Average Time per Run: 300.5 ms\nsomething else\n"
            "Average Time per Run: 182.25 ms\n")
    assert tool.parse_inference_ttft_ms(text) == 182.25
    assert tool.parse_inference_ttft_ms("no profile here") is None


def test_parse_nsys_kernel_sum_covers_both_fmha_families():
    # Given an nsys cuda_gpu_kern_sum CSV with a Blackwell FMHA kernel, an FMHA-v2
    # kernel and a GEMM
    # When the attention sum is taken with the default regex
    # Then both attention families are summed, the GEMM is not, and totals are in ms
    tool = _load_tool()
    csv_text = (
        "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),"
        "StdDev (ns),Name\n"
        "50,3000000,1,3000000,3000000,3000000,3000000,0,"
        "kernel_cutlass_kernel___main__BlackwellFusedMultiHeadAttentionForward_x\n"
        "30,2000000,1,2000000,2000000,2000000,2000000,0,"
        "kernel_cutlass_kernel___main__FMHAV2Ampere_object_at_y\n"
        "20,1000000,1,1000000,1000000,1000000,1000000,0,sm100_gemm_fp16\n")
    attention, total = tool.parse_nsys_kernel_sum(
        csv_text, tool.DEFAULT_ATTENTION_KERNEL_REGEX)
    assert attention == pytest.approx(5.0)
    assert total == pytest.approx(6.0)
    only_v2, _ = tool.parse_nsys_kernel_sum(csv_text, "FMHAV2Ampere")
    assert only_v2 == pytest.approx(2.0)


class _WordTokenizer:
    """One token per whitespace-separated word; decode joins with spaces."""

    class _Encoded:

        def __init__(self, ids):
            self.input_ids = ids

    def __call__(self, text, add_special_tokens=False):
        return self._Encoded(text.split())

    def decode(self, ids):
        return " ".join(ids)


def test_real_text_requests_hit_the_length_and_use_distinct_prefixes():
    # Given 40 short articles and a word-level tokenizer
    # When 4 requests of 96 tokens are built
    # Then every request is exactly 96 - 64 body tokens of real text, one message each,
    # generation is one token, and no two requests start with the same article
    tool = _load_tool()
    articles = [
        f"article{i} " + " ".join(f"w{i}_{j}" for j in range(10))
        for i in range(40)
    ]
    payload = tool.build_real_text_requests(articles, _WordTokenizer(), 96)
    assert payload["max_generate_length"] == 1
    assert payload["batch_size"] == 1
    assert len(payload["requests"]) == 4
    firsts = set()
    for req in payload["requests"]:
        assert len(req["messages"]) == 1
        words = req["messages"][0]["content"].split()
        assert len(words) == 96 - 64
        firsts.add(words[0])
    assert len(firsts) == 4


def test_real_text_requests_split_messages_at_the_byte_cap():
    # Given a tiny per-message byte cap
    # When a request's text exceeds it
    # Then the text is split into several messages whose bytes add up to the original
    tool = _load_tool()
    articles = ["a" * 5 + " " + "b" * 5 + " " + "c" * 5 + " " + "d" * 5] * 3
    payload = tool.build_real_text_requests(articles,
                                            _WordTokenizer(),
                                            96,
                                            n_requests=1,
                                            byte_cap=7)
    messages = payload["requests"][0]["messages"]
    assert len(messages) > 1
    joined = "".join(m["content"] for m in messages)
    assert joined == " ".join(["a" * 5, "b" * 5, "c" * 5, "d" * 5] * 8)


def _price_list_fixture():
    return {
        "lengths": [4096, 8192, 32768],
        "meta": {
            "model": "Qwen3-8B",
            "platform": "GB10",
            "ttft_source": "real",
            "kernel_profile": True,
            "deploy_tier": "75%"
        },
        "dense": {
            "4096": {
                "ttft_ms": 800.0,
                "attn_ms": 240.0,
                "gpu_ms": 3200.0,
                "attn_share": 0.075
            },
            "8192": {
                "ttft_ms": 1790.0,
                "attn_ms": 930.0,
                "gpu_ms": 7000.0,
                "attn_share": 0.133,
                "ruler": 0.80,
                "longbench": 0.40
            },
            "32768": {
                "ttft_ms": 9900.0,
                "attn_ms": 14700.0,
                "gpu_ms": 39000.0,
                "attn_share": 0.377,
                "longbench": 0.38
            }
        },
        "tiers": {
            "75%": {
                "4096": {
                    "ttft_ms": 810.0,
                    "speedup": 0.988,
                    "attn_ms": 243.0,
                    "gpu_ms": 3210.0,
                    "kernel_speedup": 0.988
                },
                "8192": {
                    "ttft_ms": 1750.0,
                    "speedup": 1.023,
                    "attn_ms": 705.0,
                    "gpu_ms": 6800.0,
                    "kernel_speedup": 1.319,
                    "ruler": 0.79,
                    "ruler_drop": 0.01,
                    "longbench": 0.395,
                    "longbench_drop": 0.005
                },
                "32768": {
                    "ttft_ms": 9000.0,
                    "speedup": 1.1,
                    "attn_ms": 11100.0,
                    "gpu_ms": 33000.0,
                    "kernel_speedup": 1.324,
                    "longbench": 0.37,
                    "longbench_drop": 0.01
                }
            }
        }
    }


def test_price_report_html_carries_every_measured_number():
    # Given a priced model with TTFT, kernel profile and accuracy at some lengths
    # When the HTML report is rendered
    # Then the figure annotates the deploy tier's kernel and TTFT peaks with their
    # lengths, the table lists speedups, kernel speedups and drops per length, and
    # lengths without accuracy show "-" instead of a number
    tool = _load_tool()
    html_text = tool.render_price_report_html([_price_list_fixture()])
    assert "75% peak 1.324x @ 32k" in html_text
    assert "75% peak 1.100x @ 32k" in html_text
    assert "attention share of GPU time 8% -&gt; 38% (dense)" in html_text
    assert "<td>1.319x</td>" in html_text and "<td>0.988x</td>" in html_text
    assert "<td>-0.010</td>" in html_text  # RULER drop at 8k (dense minus tier)
    assert html_text.count("<tr><td>75%</td><td>RULER drop</td>") == 1
    row = html_text.split("<tr><td>75%</td><td>RULER drop</td>")[1].split(
        "</tr>")[0]
    assert row == "<td>-</td><td>-0.010</td><td>-</td>"


def test_price_report_text_shows_kernel_speedup_next_to_ttft():
    # Given the same priced model
    # When the console price list is formatted
    # Then each cell carries the TTFT speedup and the kernel speedup
    tool = _load_tool()
    lines = "\n".join(tool.format_price_report(_price_list_fixture()))
    assert "1.02x k1.32x" in lines and "1.10x k1.32x" in lines
    assert "k = attention-kernel speedup" in lines


def test_report_merges_runs_of_the_same_model_and_platform():
    # Given two price runs of the same model on the same GPU, one per tier, plus a
    # run of another model
    # When the report merges them
    # Then the same-model runs become one row carrying both tiers (dense from the
    # first run) and the other model stays a separate row
    tool = _load_tool()
    first = _price_list_fixture()
    second = _price_list_fixture()
    second["tiers"] = {"60%": second["tiers"]["75%"]}
    other = _price_list_fixture()
    other["meta"]["model"] = "Llama-3.1-8B"
    merged = tool.merge_price_lists([first, second, other])
    assert [m["meta"]["model"] for m in merged] == ["Qwen3-8B", "Llama-3.1-8B"]
    assert sorted(merged[0]["tiers"]) == ["60%", "75%"]
    assert merged[0]["dense"] == first["dense"]

    # Given a `--no-perf` accuracy run and a `--no-accuracy` perf run of the same
    # tier at partly different lengths
    # When merged
    # Then each length's row carries both the perf and the accuracy fields, the
    # perf run (first) wins on a repeated field, and lengths are unioned
    perf = _price_list_fixture()
    acc = _price_list_fixture()
    length = str(perf["lengths"][0])
    acc["lengths"] = [int(length), 65536]
    acc["tiers"]["75%"] = {
        length: {
            "ruler": 0.5,
            "ruler_drop": 0.01,
            "speedup": 9.0
        },
        "65536": {
            "longbench": 0.4
        }
    }
    acc["dense"] = {length: {"ruler": 0.51}}
    both = tool.merge_price_lists([perf, acc])[0]
    row = both["tiers"]["75%"][length]
    assert row["ruler"] == 0.5 and row["ruler_drop"] == 0.01
    assert row["speedup"] == perf["tiers"]["75%"][length]["speedup"]
    assert both["tiers"]["75%"]["65536"] == {"longbench": 0.4}
    assert both["dense"][length]["ruler"] == 0.51
    assert both["lengths"] == sorted(set(perf["lengths"]) | {65536})
    html_text = tool.render_price_report_html([first, second])
    assert html_text.count("<h2>Qwen3-8B") == 1
    assert "<tr><td>60%</td><td>TTFT speedup</td>" in html_text
