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
"""Contracts for the ``checks`` block of a model-owned case."""

import csv
import io
from types import SimpleNamespace

import pytest
from defs.config import ModelType
from defs.utils import command_execution
from defs.utils.baseline import get_baseline

_PARAM = "llm-Qwen3-0.6B-fp16-mxsl:1024-bs:1-llm_basic"

_ACCURACY_CHECK = {
    "accuracy": {
        "required": ["rouge1", "rouge_l"],
        "max_regression": 0.30,
    },
}


@pytest.fixture(autouse=True)
def _isolated_baseline_env(monkeypatch):
    """Drop inherited baseline settings and the cached BaselineData."""
    for variable in ("BASELINE_CSV", "BASELINE_AUTOSAVE", "PROMOTE_BASELINE"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(command_execution,
                        "_detected_compute_capability",
                        lambda logger=None: 80)
    yield
    get_baseline()


def _case(checks=None):
    return SimpleNamespace(
        metric_checks=checks or _ACCURACY_CHECK,
        param_str=_PARAM,
        model_type=ModelType.LLM,
    )


def _result(rouge1=0.62, rouge_l=0.48):
    return {
        "success": True,
        "output": "",
        "rouge_score": {
            "rouge1": rouge1,
            "rougeL": rouge_l
        },
    }


def _check(config, result):
    command_execution.check_declared_metrics(config, result,
                                             "test_build_and_run")
    return result.get("baseline_regressions", [])


def _baseline_file(path, **columns):
    """Write a baseline CSV holding one recorded run of the case."""
    case_name = (f"tests/defs/test_builder_pipeline.py::"
                 f"test_build_and_run[{_PARAM}]")
    headers = ["Case Name", *columns]
    with open(path, "w", newline="") as handle:
        handle.write("#" + _csv_row(headers))
        handle.write(_csv_row([case_name, *columns.values()]))
    return str(path)


def _csv_row(values):
    buffer = io.StringIO()
    csv.writer(buffer).writerow(list(values))
    return buffer.getvalue()


def test_missing_baseline_skips_comparison():
    """A first measurement skips comparison when the reference is absent."""
    failures = _check(_case(), _result())
    assert failures == []


def test_missing_metric_fails_with_missing_reference(monkeypatch):
    """The harness still requires declared metrics with a missing reference."""
    result = _result()
    result["rouge_score"].pop("rougeL")
    failures = _check(_case(), result)
    assert any("required metric rouge_l was not produced" in f
               for f in failures)


def test_zero_accuracy_fails_with_missing_reference(monkeypatch):
    """The harness still rejects unusable metrics without a comparison."""
    failures = _check(_case(), _result(rouge1=0.0))
    assert any("required metric rouge1 has invalid value" in f
               for f in failures)


def test_missing_configured_baseline_skips_comparison(tmp_path, monkeypatch):
    """An absent configured CSV permits recording a first measurement."""
    monkeypatch.setenv("BASELINE_CSV", str(tmp_path / "absent.csv"))
    failures = _check(_case(), _result())
    assert failures == []


def test_absent_case_skips_comparison(tmp_path, monkeypatch):
    """An unknown case in a managed baseline means the case was never recorded."""
    monkeypatch.setenv(
        "BASELINE_CSV",
        _baseline_file(tmp_path / "baseline.csv", accuracy_rouge_1="0.60"))
    other = _case()
    other.param_str = "llm-Qwen3-8B-fp16-mxsl:1024-bs:1-llm_basic"
    failures = _check(other, _result())
    assert failures == []


def test_autosave_seeds_a_configured_baseline(tmp_path, monkeypatch):
    """Explicit autosave lets a first measurement establish its baseline."""
    csv_path = tmp_path / "baseline.csv"
    monkeypatch.setenv("BASELINE_CSV", str(csv_path))
    monkeypatch.setenv("BASELINE_AUTOSAVE", "1")
    assert _check(_case(), _result()) == []
    assert csv_path.is_file()
    assert _PARAM in csv_path.read_text()


def test_autosave_seeds_missing_required_reference(tmp_path, monkeypatch):
    """An explicitly enabled seeding run records its first reference."""
    monkeypatch.setenv("BASELINE_CSV", str(tmp_path / "absent.csv"))
    monkeypatch.setenv("BASELINE_AUTOSAVE", "1")
    failures = _check(_case(), _result())
    assert failures == []


def test_recorded_baseline_still_detects_a_regression(tmp_path, monkeypatch):
    """The comparison itself is unchanged once a baseline exists."""
    monkeypatch.setenv(
        "BASELINE_CSV",
        _baseline_file(tmp_path / "baseline.csv",
                       accuracy_rouge_1="0.60",
                       accuracy_rouge_l="0.50"))
    failures = _check(_case(), _result(rouge1=0.20, rouge_l=0.48))
    assert any("REGRESSION rouge1" in f for f in failures)


def test_recorded_baseline_accepts_a_result_within_tolerance(
        tmp_path, monkeypatch):
    """A small drop stays inside the case's declared max_regression."""
    monkeypatch.setenv(
        "BASELINE_CSV",
        _baseline_file(tmp_path / "baseline.csv",
                       accuracy_rouge_1="0.60",
                       accuracy_rouge_l="0.50"))
    assert _check(_case(), _result(rouge1=0.58, rouge_l=0.48)) == []


@pytest.mark.parametrize('missing_accuracy,regressed', [(False, False),
                                                        (True, False),
                                                        (False, True)])
def test_missing_performance_reference_keeps_accuracy_independent(
        tmp_path, monkeypatch, missing_accuracy, regressed):
    """Performance gaps preserve accuracy comparison and threshold semantics."""
    columns = {'accuracy_rouge_1': '0.60'}
    if not missing_accuracy:
        columns['accuracy_rouge_l'] = '0.50'
    monkeypatch.setenv('BASELINE_CSV',
                       _baseline_file(tmp_path / 'baseline.csv', **columns))
    perf_column = 'llm_prefill_tokens_per_second (tokens/s)'
    monkeypatch.setattr(command_execution, 'parse_perf_from_output',
                        lambda *args: {perf_column: 100.0})
    checks = dict(_ACCURACY_CHECK,
                  performance={
                      'required': ['prefill_tokens_per_second'],
                      'max_regression': .30
                  })
    result = _result(rouge1=.20 if regressed else .58)
    result['threshold_failure'] = 'Static accuracy threshold failed'
    failures = _check(_case(checks), result)
    assert ('threshold_failure' in result) == missing_accuracy
    assert bool(failures) == regressed
    assert perf_column in result['baseline_comparison_skipped'][
        'missing_baseline']
    assert result['baseline_comparison_metrics'][perf_column] == 100.0
    if missing_accuracy or regressed:
        with pytest.raises(pytest.fail.Exception):
            command_execution.check_result_failures(result)
    else:
        command_execution.check_result_failures(result)
