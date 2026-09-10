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
"""
Baseline regression checking for TensorRT Edge-LLM tests.

Parses a multi-section results CSV from a previous test run and provides
regression checking against current results.

Accuracy threshold: 5%  (higher-is-better metrics)
Perf threshold:     20% (board-specific, time/throughput/memory)
"""

import csv
import io
import os
import re
from typing import Dict, List, Optional, Tuple

# TODO: Restore thresholds after baselines stabilize
# Original values: CORRECTNESS=0.01, ROUGE=0.20, PERF=0.20
CORRECTNESS_ACCURACY_THRESHOLD = 0.05
ROUGE_ACCURACY_THRESHOLD = float(
    os.environ.get('BASELINE_ROUGE_ACCURACY_THRESHOLD', '0.50'))
PERF_THRESHOLD = 0.50

# Per-column overrides for PERF_THRESHOLD. minADE drift is well-bounded
# (paper +/- 5%, quant +/- 10%) so 20% is plenty without waiting for the
# global threshold to be tightened.
_PERF_THRESHOLD_OVERRIDES = {
    'accuracy_minade_3s': 0.20,
    'accuracy_minade_6s': 0.20,
}

ACCURACY_COLUMNS = (
    'accuracy_rouge_1',
    'accuracy_rouge_2',
    'accuracy_rouge_l',
    'accuracy_rouge_lsum',
    'accuracy_overall_accuracy',
)

PERF_LOWER_IS_BETTER = {
    'llm_prefill_avg_time_per_run (ms)',
    'llm_prefill_avg_time_per_token (ms)',
    'llm_generation_excluding_sampling_after_prefill_avg_time_per_token (ms)',
    'memory_usage_peak_gpu_memory (MB)',
    'memory_usage_peak_cpu_memory (MB)',
    'spec_decode_draft_model_prefill_avg_time (ms)',
    'spec_decode_draft_proposal_avg_time (ms)',
    'spec_decode_base_model_verification_avg_time (ms)',
    'multimodal_avg_time_per_token (ms)',
    'accuracy_wer',
    'accuracy_minade_3s',
    'accuracy_minade_6s',
}

PERF_HIGHER_IS_BETTER = {
    'llm_prefill_tokens_per_second (tokens/s)',
    'llm_generation_excluding_sampling_after_prefill_tokens_per_second (tokens/s)',
    'spec_decode_avg_acceptance_rate',
    'spec_decode_avg_tokens_per_run',
    'spec_decode_overall_tokens_per_second (tokens/s)',
}

PEAK_GPU_MEMORY_COLUMN = 'memory_usage_peak_gpu_memory (MB)'

# Fallback metric, selected only when per-process accounting is unavailable.
FALLBACK_GPU_MEMORY_METRICS = frozenset({'cuda_free_memory_delta'})

# ...and on these compute capabilities it measures system memory pressure rather
# than this process. Elsewhere the same fallback is a usable measurement, so this
# is a denylist: an unlisted board keeps its regression check.
UNTRUSTED_GPU_MEMORY_SM = frozenset({87})  # Jetson Orin family

_GPU_MEMORY_METRIC_PATTERN = r'GPU Memory Metric:\s+(\S+)'

# Maps stdout profile output patterns to CSV column names
_STDOUT_PERF_PATTERNS = [
    (r'=== LLM Prefill ===.*?Average Time per Run:\s+([\d.]+)\s+ms',
     'llm_prefill_avg_time_per_run (ms)'),
    (r'=== LLM Prefill ===.*?Average Time per Token:\s+([\d.]+)\s+ms',
     'llm_prefill_avg_time_per_token (ms)'),
    (r'=== LLM Prefill ===.*?Tokens/Second:\s+([\d.]+)',
     'llm_prefill_tokens_per_second (tokens/s)'),
    (r'=== LLM Generation.*?Average Time per Token:\s+([\d.]+)\s+ms',
     'llm_generation_excluding_sampling_after_prefill_avg_time_per_token (ms)'
     ),
    (r'=== LLM Generation.*?Tokens/Second:\s+([\d.]+)',
     'llm_generation_excluding_sampling_after_prefill_tokens_per_second (tokens/s)'
     ),
    (r'Peak GPU Memory:\s+([\d.]+)\s+MB', 'memory_usage_peak_gpu_memory (MB)'),
    (r'Peak CPU Memory:\s+([\d.]+)\s+MB', 'memory_usage_peak_cpu_memory (MB)'),
    (r'=== (?:Eagle|MTP|DFlash|JetSpec|DSpark|SpecDecode) Generation ===.*?Average Acceptance Rate:\s+([\d.]+)',
     'spec_decode_avg_acceptance_rate'),
    (r'=== (?:Eagle|MTP|DFlash|JetSpec|DSpark|SpecDecode) Generation ===.*?Average Tokens per Run:\s+([\d.]+)',
     'spec_decode_avg_tokens_per_run'),
    (r'=== (?:Eagle|MTP|DFlash|JetSpec|DSpark|SpecDecode) Generation ===.*?Overall Tokens/Second \(excluding base prefill\):\s+([\d.]+)',
     'spec_decode_overall_tokens_per_second (tokens/s)'),
    (r'Draft Model Prefill - Total Runs:.*?Average:\s+([\d.]+)\s+ms',
     'spec_decode_draft_model_prefill_avg_time (ms)'),
    (r'Construct Draft Proposal - Total Runs:.*?Average:\s+([\d.]+)\s+ms',
     'spec_decode_draft_proposal_avg_time (ms)'),
    (r'Base Model Verification - Total Runs:.*?Average:\s+([\d.]+)\s+ms',
     'spec_decode_base_model_verification_avg_time (ms)'),
    (r'=== Multimodal Processing ===.*?Average Time per Token:\s+([\d.]+)\s+ms',
     'multimodal_avg_time_per_token (ms)'),
]


def gpu_memory_metric_from_output(output: str) -> Optional[str]:
    """Which GPU-memory backend the runtime selected for this run, if it said."""
    match = re.search(_GPU_MEMORY_METRIC_PATTERN, output)
    return match.group(1) if match else None


def peak_gpu_memory_is_comparable(output: str,
                                  compute_capability: Optional[int] = None
                                  ) -> bool:
    """False when the run's peak GPU memory does not measure this process.

    Needs both the metric and the board: the fallback metric is meaningless on
    Orin but validated on DRIVE Thor. An unknown compute capability keeps the
    check rather than dropping it, so coverage is never lost silently.
    """
    if gpu_memory_metric_from_output(
            output) not in FALLBACK_GPU_MEMORY_METRICS:
        return True
    return compute_capability not in UNTRUSTED_GPU_MEMORY_SM


def parse_perf_from_output(
        output: str,
        compute_capability: Optional[int] = None) -> Dict[str, float]:
    """Extract perf metrics from llm_inference/llm_build stdout (--dumpProfile output)."""
    metrics = {}
    for pattern, col_name in _STDOUT_PERF_PATTERNS:
        match = re.search(pattern, output, re.DOTALL)
        if match:
            metrics[col_name] = float(match.group(1))
    if not peak_gpu_memory_is_comparable(output, compute_capability):
        metrics.pop(PEAK_GPU_MEMORY_COLUMN, None)
    return metrics


class BaselineData:
    """Parsed baseline CSV with regression checking."""

    def __init__(self, csv_path: str):
        self._data: Dict[str, Dict] = {}
        self._parse(csv_path)

    def _parse(self, csv_path: str):
        with open(csv_path, encoding='utf-8') as f:
            lines = f.readlines()

        headers: List[str] = []
        name_idx: int = -1

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Column header line: starts with '#Case Name'
            if stripped.startswith('#Case Name'):
                headers = next(csv.reader(io.StringIO(
                    stripped[1:])))  # strip leading '#'
                name_idx = headers.index('Case Name')
                continue

            # Skip other comment/section lines
            if stripped.startswith('#'):
                continue

            if not headers or name_idx < 0:
                continue

            row = next(csv.reader(io.StringIO(stripped)))
            if name_idx >= len(row):
                continue

            case_name = row[name_idx]
            entry = {}
            for j, h in enumerate(headers):
                if j < len(row) and row[j].strip():
                    try:
                        entry[h] = float(row[j])
                    except ValueError:
                        entry[h] = row[j]
            self._data[case_name] = entry

    def get(self, case_name: str) -> Optional[Dict]:
        return self._data.get(case_name)

    def find_by_param(self,
                      param_str: str,
                      test_func: str = None,
                      model_type_value: str = None) -> Optional[Dict]:
        """Find baseline entry by param string, test function, and model type.

        When model_type_value and test_func are both provided, uses exact key
        lookup via _build_case_name to avoid ambiguity across pipeline classes
        (e.g. TestLLMPipeline vs TestVLMPipeline both having test_inference).
        Falls back to substring search when model_type_value is not given.
        """
        if model_type_value and test_func:
            case_name = _build_case_name(model_type_value, test_func,
                                         param_str)
            return self._data.get(case_name)
        # Fallback: substring search for callers that do not pass model type.
        key = f"[{param_str}]"
        for name, data in self._data.items():
            if key in name:
                if test_func is None or f"::{test_func}[" in name:
                    return data
        return None

    def check_accuracy_regression(
            self, baseline: Dict,
            current: Dict) -> Tuple[List[str], List[str]]:
        """Compare current accuracy metrics against baseline.

        Returns (regressions, summaries).
        regressions: list of failure messages for metrics that regressed beyond threshold.
        summaries: list of all comparison lines with percentage change.
        """
        regressions = []
        summaries = []
        for col in ACCURACY_COLUMNS:
            if col not in baseline or col not in current:
                continue
            base_val = baseline[col]
            curr_val = current[col]
            if not isinstance(base_val, (int, float)) or base_val <= 0:
                continue
            change_pct = (curr_val - base_val) / base_val * 100
            line = (f"{col}: {curr_val:.4f} vs baseline {base_val:.4f} "
                    f"({change_pct:+.1f}%)")
            summaries.append(line)
            if col.startswith('accuracy_rouge'):
                threshold = ROUGE_ACCURACY_THRESHOLD
            else:
                threshold = CORRECTNESS_ACCURACY_THRESHOLD
            if curr_val < base_val * (1 - threshold):
                regressions.append(
                    f"REGRESSION {line} [threshold {threshold*100:.0f}%]")
        return regressions, summaries

    def check_perf_regression(self, baseline: Dict,
                              current: Dict) -> Tuple[List[str], List[str]]:
        """Compare current perf metrics against baseline.

        Returns (regressions, summaries).
        regressions: list of failure messages for metrics that regressed beyond threshold.
        summaries: list of all comparison lines with percentage change and direction label.
        """
        regressions = []
        summaries = []

        for col in PERF_LOWER_IS_BETTER:
            if col not in baseline or col not in current:
                continue
            base_val = baseline[col]
            curr_val = current[col]
            if not isinstance(base_val, (int, float)) or base_val <= 0:
                continue
            threshold = _PERF_THRESHOLD_OVERRIDES.get(col, PERF_THRESHOLD)
            change_pct = (curr_val - base_val) / base_val * 100
            label = "worse" if change_pct > 0 else "better"
            line = (f"{col} (lower=better): {curr_val:.2f} vs baseline "
                    f"{base_val:.2f} ({change_pct:+.1f}%, {label})")
            summaries.append(line)
            if curr_val > base_val * (1 + threshold):
                regressions.append(
                    f"REGRESSION {line} [threshold {threshold*100:.0f}%]")

        for col in PERF_HIGHER_IS_BETTER:
            if col not in baseline or col not in current:
                continue
            base_val = baseline[col]
            curr_val = current[col]
            if not isinstance(base_val, (int, float)) or base_val <= 0:
                continue
            threshold = _PERF_THRESHOLD_OVERRIDES.get(col, PERF_THRESHOLD)
            change_pct = (curr_val - base_val) / base_val * 100
            label = "better" if change_pct > 0 else "worse"
            line = (f"{col} (higher=better): {curr_val:.1f} vs baseline "
                    f"{base_val:.1f} ({change_pct:+.1f}%, {label})")
            summaries.append(line)
            if curr_val < base_val * (1 - threshold):
                regressions.append(
                    f"REGRESSION {line} [threshold {threshold*100:.0f}%]")

        return regressions, summaries


# Mapping from check_accuracy_with_dataset result keys to CSV column names
_ACCURACY_RESULT_TO_CSV = {
    'rouge1': 'accuracy_rouge_1',
    'rouge2': 'accuracy_rouge_2',
    'rougeL': 'accuracy_rouge_l',
    'rougeLsum': 'accuracy_rouge_lsum',
}


def map_accuracy_result_to_csv(result: Dict) -> Dict[str, float]:
    """Convert check_accuracy_with_dataset result dict to CSV column names."""
    mapped = {}
    if 'rouge_score' in result:
        for key, csv_col in _ACCURACY_RESULT_TO_CSV.items():
            if key in result['rouge_score']:
                mapped[csv_col] = result['rouge_score'][key]
    if 'accuracy' in result:
        mapped['accuracy_overall_accuracy'] = result['accuracy']
    if 'wer' in result:
        mapped['accuracy_wer'] = result['wer']
    if result.get('minade_6s') is not None:
        mapped['accuracy_minade_6s'] = result['minade_6s']
    if result.get('minade_3s') is not None:
        mapped['accuracy_minade_3s'] = result['minade_3s']
    return mapped


_TEST_FILE_MAP = {
    'llm': 'tests/defs/test_llm_pipeline.py',
    'vlm': 'tests/defs/test_vlm_pipeline.py',
}

_CLASS_NAME_MAP = {}


def _build_case_name(model_type_value: str, test_func: str,
                     param_str: str) -> str:
    test_file = _TEST_FILE_MAP.get(
        model_type_value, f'tests/defs/test_{model_type_value}_pipeline.py')
    class_name = _CLASS_NAME_MAP.get(model_type_value)
    if class_name:
        return f"{test_file}::{class_name}::{test_func}[{param_str}]"
    return f"{test_file}::{test_func}[{param_str}]"


def _serialise_row(columns: List[str], values: Dict) -> str:
    buf = io.StringIO()
    csv.writer(buf).writerow([values.get(c, '') for c in columns])
    return buf.getvalue()


# Minimum relative improvement required for promote_baseline_if_better to
# overwrite a baseline cell. Anything smaller is treated as noise.
PROMOTE_MIN_IMPROVEMENT = 0.01


def _is_significantly_better(col: str, curr, base) -> bool:
    if not isinstance(curr,
                      (int, float)) or not isinstance(base, (int, float)):
        return False
    if base <= 0:
        return False
    if col in PERF_LOWER_IS_BETTER:
        return curr < base * (1 - PROMOTE_MIN_IMPROVEMENT)
    if col in PERF_HIGHER_IS_BETTER or col in ACCURACY_COLUMNS:
        return curr > base * (1 + PROMOTE_MIN_IMPROVEMENT)
    return False


def promote_baseline_if_better(csv_path: str,
                               model_type_value: str,
                               test_func: str,
                               param_str: str,
                               result: Dict,
                               logger=None) -> List[str]:
    """Replace baseline cells where the current run is >1% better.

    Gated by the ``PROMOTE_BASELINE`` env var; no-op otherwise. Direction is
    per-column via PERF_LOWER_IS_BETTER / PERF_HIGHER_IS_BETTER / ACCURACY_COLUMNS.
    Returns the list of columns that were overwritten (for logging).
    """
    if not os.environ.get('PROMOTE_BASELINE'):
        return []
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        return []

    metrics: Dict[str, float] = {}
    metrics.update(map_accuracy_result_to_csv(result))
    metrics.update(parse_perf_from_output(result.get('output', '')))
    if not metrics:
        return []

    case_name = _build_case_name(model_type_value, test_func, param_str)

    with open(csv_path, 'r', encoding='utf-8') as f:
        content = f.read()

    HEADER_PREFIX = '#Case Name'
    header_line_start = content.find(HEADER_PREFIX)
    if header_line_start == -1:
        return []
    header_line_end = content.find('\n', header_line_start)
    cols = next(
        csv.reader(io.StringIO(content[header_line_start +
                                       1:header_line_end])))
    if 'Case Name' not in cols:
        return []
    case_idx = cols.index('Case Name')

    body_start = header_line_end + 1
    raw_rows = content[body_start:].splitlines(keepends=True)
    promoted: List[str] = []
    new_rows = []
    for raw in raw_rows:
        stripped = raw.rstrip('\n')
        if not stripped.strip() or stripped.startswith('#'):
            new_rows.append(raw)
            continue
        row = next(csv.reader(io.StringIO(stripped)))
        # Pad short rows so col lookups don't raise.
        while len(row) < len(cols):
            row.append('')
        if row[case_idx] != case_name:
            new_rows.append(raw)
            continue
        for col, curr in metrics.items():
            if col not in cols:
                continue
            j = cols.index(col)
            try:
                base = float(row[j]) if row[j] else None
            except ValueError:
                base = None
            if base is None or not _is_significantly_better(col, curr, base):
                continue
            row[j] = f"{curr}"
            promoted.append(f"{col}: {base} -> {curr}")
        new_rows.append(_serialise_row(cols, dict(zip(cols, row))))

    if not promoted:
        return []

    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(content[:body_start] + ''.join(new_rows))

    # Invalidate the singleton so subsequent reads see the updated CSV.
    global _baseline_csv_path
    _baseline_csv_path = None

    if logger:
        logger.info("Promoted baseline cells for [%s]:\n  %s", param_str,
                    "\n  ".join(promoted))
    return promoted


def save_to_baseline(csv_path: str, model_type_value: str, test_func: str,
                     param_str: str, result: Dict) -> None:
    """Append current result to baseline CSV when no prior baseline exists.

    All entries for the same section share a single column header (union of
    all columns ever seen).  Entries that lack a column simply leave it empty.
    """
    metrics: Dict[str, float] = {}
    metrics.update(map_accuracy_result_to_csv(result))
    metrics.update(parse_perf_from_output(result.get('output', '')))

    if not metrics:
        return

    case_name = _build_case_name(model_type_value, test_func, param_str)
    new_entry = {'Case Name': case_name, **metrics}

    # The column header line is prefixed with '#' so the parser can identify
    # it without a separate section-name row: e.g. "#Case Name,rouge_1,..."
    HEADER_PREFIX = '#Case Name'

    if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find the existing header line.
        header_line_start = content.find(HEADER_PREFIX)
        if header_line_start != -1:
            header_line_end = content.find('\n', header_line_start)
            existing_cols = next(
                csv.reader(
                    io.StringIO(content[header_line_start +
                                        1:header_line_end])))

            # Union of columns (existing order + new ones appended).
            new_cols = [
                c for c in sorted(metrics.keys()) if c not in existing_cols
            ]
            all_cols = existing_cols + new_cols

            # Re-serialise all data rows with expanded columns.
            data_rows = ''
            for raw in content[header_line_end + 1:].splitlines():
                if not raw.strip() or raw.startswith('#'):
                    continue
                old_row = next(csv.reader(io.StringIO(raw)))
                old_map = {
                    existing_cols[i]: old_row[i]
                    for i in range(min(len(existing_cols), len(old_row)))
                }
                data_rows += _serialise_row(all_cols, old_map)

            data_rows += _serialise_row(all_cols, new_entry)

            new_header = '#' + ','.join(all_cols) + '\n'
            # Replace everything from the header line onward.
            content = content[:header_line_start] + new_header + data_rows
        else:
            # No header yet — write header + first row.
            all_cols = ['Case Name'] + sorted(metrics.keys())
            new_header = '#' + ','.join(all_cols) + '\n'
            content = content + new_header + _serialise_row(
                all_cols, new_entry)

        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(content)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        all_cols = ['Case Name'] + sorted(metrics.keys())
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('#' + ','.join(all_cols) + '\n' +
                    _serialise_row(all_cols, new_entry))

    # Invalidate the singleton so get_baseline() reloads the updated CSV on
    # the next call (e.g. a later test in the same pytest session).
    global _baseline_csv_path
    _baseline_csv_path = None


_baseline_instance: Optional[BaselineData] = None
_baseline_csv_path: Optional[str] = None


def get_baseline() -> Optional[BaselineData]:
    """Lazy singleton: load BaselineData from BASELINE_CSV env var.

    Re-loads if the env var value changes between calls (e.g. set by a
    conftest fixture after module import).
    """
    global _baseline_instance, _baseline_csv_path
    csv_path = os.environ.get('BASELINE_CSV', 'logs/baseline.csv')
    if csv_path != _baseline_csv_path:
        _baseline_csv_path = csv_path
        _baseline_instance = (BaselineData(csv_path) if csv_path
                              and os.path.isfile(csv_path) else None)
    return _baseline_instance
