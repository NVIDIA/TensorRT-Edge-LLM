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
"""Optional baseline comparison, safe seeding, and suite timeout contracts."""

import concurrent.futures
import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from defs import test_common
from defs.utils import baseline, command_execution

METRIC = 'spec_decode_acceptance_rate'
TIMING = 'spec_decode_draft_proposal_avg_time (ms)'


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """Keep environment and cached baselines isolated between cases."""
    for key in ('BASELINE_AUTOSAVE', 'PROMOTE_BASELINE', 'BASELINE_CSV'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(baseline, '_baseline_csv_path', None)
    monkeypatch.setattr(command_execution, '_detected_compute_capability',
                        lambda logger: 87)


def config():
    return SimpleNamespace(param_str='mtp-case',
                           model_type=SimpleNamespace(value='llm'),
                           is_eagle=False,
                           is_mtp=True,
                           is_dflash=False,
                           is_jetspec=False,
                           is_domino=False,
                           is_dspark=False,
                           test_case='mtbench')


def compare(monkeypatch,
            entry,
            perf,
            accuracy=None,
            missing_file=False,
            result=None):
    data = baseline.BaselineData.__new__(baseline.BaselineData)
    data.find_by_param = lambda *args, **kwargs: entry
    monkeypatch.setattr(command_execution, 'get_baseline', lambda: None
                        if missing_file else data)
    monkeypatch.setattr(command_execution,
                        '_required_spec_decode_perf_columns',
                        lambda c: {METRIC, TIMING})
    monkeypatch.setattr(command_execution, 'parse_perf_from_output',
                        lambda *args: dict(perf))
    monkeypatch.setattr(command_execution, 'map_accuracy_result_to_csv',
                        lambda r: dict(accuracy or {}))
    monkeypatch.setattr(command_execution, 'peak_gpu_memory_is_comparable',
                        lambda *args: True)
    if result is None:
        result = {'success': True, 'output': 'raw measurements'}
    command_execution._check_baseline_regression(config(), 'test_e2e_bench',
                                                 result,
                                                 logging.getLogger(__name__),
                                                 True)
    return result


@pytest.mark.parametrize('entry,perf', [({}, {
    METRIC: .5
}), ({
    METRIC: .5
}, {}), ({
    METRIC: .5
}, {
    METRIC: .5
}), ({
    METRIC: .5,
    TIMING: 2
}, {
    METRIC: .5,
    TIMING: 2
})])
def test_missing_metrics_skip_and_record(monkeypatch, caplog, entry, perf):
    """Available metrics persist in logs while missing pairs skip comparison."""
    with caplog.at_level(logging.INFO):
        result = compare(monkeypatch, entry, perf)
    assert not result.get('baseline_regressions')
    assert result['baseline_comparison_metrics'] == perf
    payload = next(
        r.message.split('=', 1)[1] for r in caplog.records
        if r.message.startswith('BASELINE_COMPARISON_DATA='))
    recorded = json.loads(payload)
    assert recorded['current_metrics'] == perf
    assert set(recorded['skipped']['missing_baseline']) == {METRIC, TIMING
                                                            } - entry.keys()
    assert set(recorded['skipped']['missing_current']) == {METRIC, TIMING
                                                           } - perf.keys()


@pytest.mark.parametrize('missing_file', [False, True])
def test_missing_entry_preserves_threshold_and_measurements(
        monkeypatch, missing_file):
    """Missing references retain current values and independent threshold failures."""
    result = compare(monkeypatch,
                     None, {METRIC: .5},
                     missing_file=missing_file,
                     result={
                         'success': True,
                         'threshold_failure': 'accuracy too low'
                     })
    assert not result.get('baseline_regressions')
    assert result['threshold_failure'] == 'accuracy too low'
    assert result['baseline_comparison_metrics'] == {METRIC: .5}
    assert result['baseline_comparison_skipped']['reason'] == (
        'missing_baseline_file' if missing_file else 'missing_baseline_entry')


def test_present_regression_survives_missing_timing(monkeypatch):
    """Acceptance regression remains a failure with an absent timing metric."""
    result = compare(monkeypatch, {METRIC: .5}, {METRIC: .1})
    assert any('REGRESSION' in r for r in result['baseline_regressions'])


def test_accuracy_regression_survives_missing_spec_metrics(monkeypatch):
    """Measured accuracy regression remains a failure with partial perf data."""
    result = compare(monkeypatch, {'accuracy_overall_accuracy': .6}, {},
                     {'accuracy_overall_accuracy': .4})
    assert any('REGRESSION' in r for r in result['baseline_regressions'])


@pytest.mark.parametrize('remote', [None, object()])
def test_unit_suite_timeout_and_failure(monkeypatch, remote):
    """Local and remote suites get 1200 seconds while failed commands still fail."""
    calls = []
    monkeypatch.setattr(
        test_common, 'run_command', lambda **kw: calls.append(kw) or {
            'success': False,
            'error': 'unit failure'
        })
    with pytest.raises(pytest.fail.Exception, match='unit failure'):
        test_common.test_unit_tests(
            SimpleNamespace(build_dir='/workspace/build',
                            trt_package_dir=None), remote,
            logging.getLogger(__name__))
    assert calls[0]['timeout'] == 1200
    assert calls[0]['remote_config'] is remote


def seed(path, param, score):
    return baseline.save_to_baseline(str(path), 'llm', 'test_e2e_bench', param,
                                     {
                                         'success': True,
                                         'accuracy': score,
                                         'output': ''
                                     })


@pytest.fixture
def seeding(monkeypatch):
    """Expose simple metric dictionaries through the real CSV writer."""
    monkeypatch.setattr(baseline, 'map_accuracy_result_to_csv',
                        lambda r: r.get('accuracy', {}))
    monkeypatch.setattr(baseline, 'parse_perf_from_output', lambda *args: {})


def test_seed_fills_only_blank_cells(tmp_path, seeding):
    """Repeated writes preserve existing numeric baselines including zero."""
    path = tmp_path / 'baseline.csv'
    seed(path, 'case', {METRIC: 0, 'accuracy_overall_accuracy': .6})
    seed(path, 'case', {
        METRIC: .5,
        'accuracy_overall_accuracy': .4,
        TIMING: 2
    })
    data = baseline.BaselineData(str(path)).find_by_param(
        'case', 'test_e2e_bench', 'llm')
    assert data[METRIC] == 0
    assert data['accuracy_overall_accuracy'] == .6
    assert data[TIMING] == 2
    assert path.read_text().count('test_e2e_bench[case]') == 1


def test_seed_preserves_other_sections(tmp_path, seeding):
    """Different section schemas and unrelated records survive a fill."""
    path = tmp_path / 'baseline.csv'
    case = baseline._build_case_name('llm', 'test_e2e_bench', 'case')
    path.write_text(
        '# section A\n#Case Name,' + METRIC + '\n' + case + ',\n'
        '# section B\n#Case Name,accuracy_overall_accuracy\nother,0.7\n')
    seed(path, 'case', {METRIC: .5})
    data = baseline.BaselineData(str(path))
    assert data.get('other')['accuracy_overall_accuracy'] == .7
    assert data.get(case)[METRIC] == .5
    assert '# section B' in path.read_text()


def test_parallel_seed_preserves_all_cases(tmp_path, seeding):
    """Concurrent workers append distinct cases without losing measurements."""
    path = tmp_path / 'baseline.csv'
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(lambda i: seed(path, str(i), {METRIC: i / 10}),
                     range(12)))
    data = baseline.BaselineData(str(path))
    for i in range(12):
        assert data.find_by_param(str(i), 'test_e2e_bench',
                                  'llm')[METRIC] == i / 10


@pytest.mark.parametrize('failure', [{
    'success': False
}, {
    'threshold_failure': 'low'
}, {
    'baseline_regressions': ['low']
}])
def test_failed_results_leave_baseline_intact(tmp_path, seeding, failure):
    """Execution, threshold, and regression failures cannot seed references."""
    path = tmp_path / 'baseline.csv'
    seed(path, 'known', {METRIC: .5})
    before = path.read_bytes()
    baseline.save_to_baseline(str(path), 'llm', 'test_e2e_bench', 'bad', {
        'success': True,
        'accuracy': {
            METRIC: .1
        },
        **failure
    })
    assert path.read_bytes() == before


def test_autosave_opt_in_and_existing_entry(monkeypatch):
    """Explicit autosave fills an existing case only after comparisons pass."""
    calls = []
    monkeypatch.setattr(command_execution, 'save_to_baseline',
                        lambda *a: calls.append(a) or True)
    compare(monkeypatch, {METRIC: .5}, {METRIC: .5})
    assert calls == []
    monkeypatch.setenv('BASELINE_AUTOSAVE', '1')
    compare(monkeypatch, {METRIC: .5}, {METRIC: .5})
    assert len(calls) == 1
    assert calls[0][-1] == 87
    compare(monkeypatch, {METRIC: .5}, {METRIC: .1})
    assert len(calls) == 1
    assert calls[0][-1] == 87


def test_records_missing_metrics(monkeypatch):
    """Missing references retain a current-metric audit trail."""
    result = compare(monkeypatch, {}, {METRIC: .5})
    assert result['baseline_comparison_metrics'][METRIC] == .5
    assert METRIC in result['baseline_comparison_skipped']['missing_baseline']


def test_invalid_declared_metric_cannot_seed(monkeypatch, tmp_path):
    """A required invalid metric is rejected before a reference can be saved."""
    path = tmp_path / 'baseline.csv'
    monkeypatch.setenv('BASELINE_CSV', str(path))
    monkeypatch.setenv('BASELINE_AUTOSAVE', '1')
    case = config()
    case.metric_checks = {
        'accuracy': {
            'required': ['rouge1'],
            'max_regression': .05
        }
    }
    result = {'success': True, 'output': '', 'rouge_score': {'rouge1': 0}}
    command_execution.check_declared_metrics(case, result,
                                             'test_build_and_run')
    assert result['baseline_regressions']
    assert not path.exists()


def test_seed_filters_orin_system_memory(monkeypatch, tmp_path):
    """Orin system-pressure memory is excluded from stored references."""
    path = tmp_path / 'baseline.csv'
    result = {
        'success':
        True,
        'output':
        'Peak GPU Memory: 100 MB\n'
        'Peak CPU Memory: 50 MB\nGPU Memory Metric: cuda_free_memory_delta'
    }
    baseline.save_to_baseline(str(path), 'llm', 'test_e2e_bench', 'case',
                              result, 87)
    row = baseline.BaselineData(str(path)).find_by_param(
        'case', 'test_e2e_bench', 'llm')
    assert baseline.PEAK_GPU_MEMORY_COLUMN not in row
    assert row['memory_usage_peak_cpu_memory (MB)'] == 50


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_seed_filters_nonfinite(tmp_path, seeding, bad):
    """Nonfinite samples remain excluded from a usable reference row."""
    path = tmp_path / 'baseline.csv'
    seed(path, 'case', {METRIC: bad, TIMING: 2})
    row = baseline.BaselineData(str(path)).find_by_param(
        'case', 'test_e2e_bench', 'llm')
    assert METRIC not in row
    assert row[TIMING] == 2


@pytest.mark.parametrize('error', [
    OSError('read only'),
    TimeoutError('busy'),
    csv.Error('malformed'),
    UnicodeError('invalid encoding')
])
def test_write_failure_logs_and_preserves_result(monkeypatch, caplog, error):
    """Storage failures remain visible while inference results stay intact."""
    monkeypatch.setenv('BASELINE_AUTOSAVE', '1')

    def fail(*args):
        raise error

    monkeypatch.setattr(command_execution, 'save_to_baseline', fail)
    with caplog.at_level(logging.WARNING):
        result = compare(monkeypatch, None, {METRIC: .5})
    assert result['success']
    assert not result.get('baseline_regressions')
    assert 'Baseline update failed' in caplog.text


def test_promotion_filters_orin_system_memory(monkeypatch, tmp_path):
    """Explicit promotion retains GPU references excluded on Orin."""
    path = tmp_path / 'baseline.csv'
    path.write_text(
        '#Case Name,memory_usage_peak_gpu_memory (MB),memory_usage_peak_cpu_memory (MB)\n'
        'tests/defs/test_llm_pipeline.py::test_e2e_bench[case],200,100\n')
    monkeypatch.setenv('PROMOTE_BASELINE', '1')
    result = {
        'success':
        True,
        'output':
        'Peak GPU Memory: 100 MB\n'
        'Peak CPU Memory: 50 MB\nGPU Memory Metric: cuda_free_memory_delta'
    }
    columns = baseline.promote_baseline_if_better(str(path),
                                                  'llm',
                                                  'test_e2e_bench',
                                                  'case',
                                                  result,
                                                  compute_capability=87)
    assert baseline.PEAK_GPU_MEMORY_COLUMN not in columns
    assert any(
        c.startswith('memory_usage_peak_cpu_memory (MB):') for c in columns)
    assert ',200,' in path.read_text()


@pytest.mark.parametrize('failure', ['permission', 'lock', 'replace'])
def test_promotion_write_failure_preserves_case_and_baseline(
        monkeypatch, tmp_path, caplog, failure):
    """Optional promotion failures preserve completed results and CSV bytes."""
    path = tmp_path / 'baseline.csv'
    original = ('#Case Name,accuracy_overall_accuracy\n'
                'tests/defs/test_llm_pipeline.py::test_e2e_bench[case],0.5\n')
    path.write_text(original)
    monkeypatch.setenv('PROMOTE_BASELINE', '1')
    result = {'success': True, 'accuracy': .8, 'output': ''}

    def fail(*args, **kwargs):
        if failure == 'lock':
            raise TimeoutError(str(path) + '.lock')
        raise PermissionError('read-only baseline directory')

    if failure == 'lock':
        monkeypatch.setattr(baseline, '_baseline_lock', fail)
    elif failure == 'permission':
        monkeypatch.setattr(baseline.os, 'open', fail)
    else:
        monkeypatch.setattr(baseline.os, 'replace', fail)
    with caplog.at_level(logging.WARNING):
        promoted = baseline.promote_baseline_if_better(
            str(path), 'llm', 'test_e2e_bench', 'case', result,
            logging.getLogger('promotion-test'), 87)
    assert promoted == []
    assert result == {'success': True, 'accuracy': .8, 'output': ''}
    assert path.read_text() == original
    assert 'Baseline promotion failed' in caplog.text
    assert list(tmp_path.glob('.baseline-*')) == []


def test_baseline_lock_contention_and_release(tmp_path):
    """A competing writer times out and can acquire after release."""
    path = str(tmp_path / 'baseline.csv.lock')
    with baseline._baseline_lock(path):
        with pytest.raises(TimeoutError):
            with baseline._baseline_lock(path, timeout=0.05):
                pytest.fail('Concurrent writer acquired held lock')
    with baseline._baseline_lock(path, timeout=0):
        pass


def test_baseline_lock_releases_after_error(tmp_path):
    """Writer exceptions release the OS lock."""
    path = str(tmp_path / 'baseline.csv.lock')
    with pytest.raises(ValueError):
        with baseline._baseline_lock(path):
            raise ValueError('writer failed')
    with baseline._baseline_lock(path, timeout=0):
        pass


def test_baseline_lock_blocks_another_process(tmp_path):
    """Independent processes obey the same baseline lock."""
    path = str(tmp_path / 'baseline.csv.lock')
    env = dict(os.environ, PYTHONPATH=str(Path(baseline.__file__).parents[2]))
    script = """
import sys
from defs.utils.baseline import _baseline_lock
try:
    with _baseline_lock(sys.argv[1], timeout=0.05):
        pass
except TimeoutError:
    sys.exit(23)
"""
    with baseline._baseline_lock(path):
        result = subprocess.run([sys.executable, '-c', script, path],
                                env=env,
                                capture_output=True,
                                text=True,
                                timeout=10)
        assert result.returncode == 23, result.stderr
    result = subprocess.run([sys.executable, '-c', script, path],
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=10)
    assert result.returncode == 0, result.stderr
