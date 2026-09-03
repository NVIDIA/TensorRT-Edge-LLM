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
Test execution functions for TensorRT Edge-LLM tests.

This module contains the simplified test execution functions for build, inference, 
and benchmark tests. Each function is focused on its specific task without 
unnecessary abstraction layers.
"""

import json
import math
import os
import subprocess
from typing import Any, Dict, Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import check_file_exists, run_command, run_with_trt_env

from ..config import ModelType, TaskType, TestConfig
from .accuracy import check_accuracy_with_dataset
from .baseline import (get_baseline, gpu_memory_metric_from_output,
                       map_accuracy_result_to_csv, parse_perf_from_output,
                       peak_gpu_memory_is_comparable,
                       promote_baseline_if_better, save_to_baseline)
from .command_generation import (generate_build_commands,
                                 generate_e2e_bench_commands,
                                 generate_inference_commands,
                                 generate_kernel_bench_commands,
                                 generate_vlmevalkit_commands)

_ALPAMAYO_DATASET_PLACEHOLDER = "$ALPAMAYO_DATASET_DIR"


def _sync_remote_output_file(filepath: str,
                             remote_config: Optional[RemoteConfig],
                             logger) -> str:
    if remote_config is None or os.path.exists(filepath):
        return filepath

    local_dir = os.path.dirname(filepath)
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)

    remote_host = f"{remote_config.user}@{remote_config.host}"
    env = os.environ.copy()
    env["SSHPASS"] = remote_config.password
    cmd = [
        "sshpass", "-e", "rsync", "-a", "-e",
        "ssh -o StrictHostKeyChecking=no", f"{remote_host}:{filepath}",
        filepath
    ]

    if logger:
        logger.info("Copying remote output file back for metrics: %s",
                    filepath)

    result = subprocess.run(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to copy remote output file {filepath}: {result.stdout}")

    return filepath


def _read_context_reuse_profile(
        config: TestConfig,
        logger,
        remote_config: Optional[RemoteConfig] = None) -> int:
    """Return reused prefill tokens and fail when an enabled cache never hits."""
    profile_path = _sync_remote_output_file(config.get_profile_json_file(),
                                            remote_config, logger)
    with open(profile_path, encoding='utf-8') as profile_file:
        profile = json.load(profile_file)

    reused_tokens = profile.get('prefill', {}).get('reused_tokens')
    if not isinstance(reused_tokens, int) or reused_tokens <= 0:
        raise RuntimeError(
            "Context reuse was enabled but the profile did not report a "
            f"positive prefill.reused_tokens value: {reused_tokens!r}")
    return reused_tokens


def _check_vlm_context_reuse_profile(
        config: TestConfig,
        logger,
        remote_config: Optional[RemoteConfig] = None) -> None:
    """Validate media-aware context cache metrics against expected values."""
    profile_path = _sync_remote_output_file(config.get_profile_json_file(),
                                            remote_config, logger)
    with open(profile_path, encoding='utf-8') as profile_file:
        profile = json.load(profile_file)

    with open(config.get_test_case_file(), encoding='utf-8') as tc_file:
        expected = json.load(tc_file).get('expected_context_cache', {})

    cc = profile.get('context_cache', {})
    errors = []

    if expected.get('media_aware_sequences') is not None:
        actual = cc.get('media_aware_sequences', 0)
        want = expected['media_aware_sequences']
        if actual < want:
            errors.append(
                f"media_aware_sequences: got {actual}, want >= {want}")

    if expected.get('hit_sequences') is not None:
        actual = cc.get('hit_sequences', 0)
        want = expected['hit_sequences']
        if actual < want:
            errors.append(f"hit_sequences: got {actual}, want >= {want}")

    if errors:
        raise RuntimeError(
            "VLM context cache profile failed expectations:\n  " +
            "\n  ".join(errors))


def _check_context_reuse_cold_hit_equivalence(config: TestConfig) -> None:
    """Require the cold full prompt and prefix-reused full prompt to match."""
    with open(config.get_output_json_file(), encoding='utf-8') as output_file:
        responses = json.load(output_file).get('responses', [])
    if len(responses) != 3:
        raise RuntimeError(
            "The context-reuse parity fixture must produce exactly three responses, "
            f"got {len(responses)}.")

    producer, _, consumer = responses
    if producer.get('output_text') != consumer.get('output_text'):
        raise RuntimeError(
            "The reused response text differs from the cold producer response."
        )
    if producer.get('finish_reason') != consumer.get('finish_reason'):
        raise RuntimeError(
            "The reused response finish reason differs from the cold producer response."
        )

    def generated_token_ids(response: Dict[str, Any]) -> list[int]:
        logprobs = response.get('logprobs')
        if not isinstance(logprobs, list) or any(
                not isinstance(step, list) or not step
                or not isinstance(step[0].get('token_id'), int)
                for step in logprobs):
            raise RuntimeError(
                "The context-reuse parity fixture requires num_logprobs >= 1.")
        return [step[0]['token_id'] for step in logprobs]

    if generated_token_ids(producer) != generated_token_ids(consumer):
        raise RuntimeError(
            "The reused response token IDs differ from the cold producer response."
        )


def _logprobs_equivalent(a: Any, b: Any, abs_tol: float = 1e-3) -> bool:
    """Compare two logprobs structures, tolerating float noise in 'logprob' values.

    Batch compaction reassigns a surviving sequence to a different batch-slot
    index than an uncompacted replay of the same content, which changes
    kernel-launch/reduction order and perturbs 'logprob' floats by ordinary
    GPU non-associativity (~1e-4 to 1e-7 in practice) without changing the
    generated token IDs. Everything else in the structure must still match
    exactly.
    """
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _logprobs_equivalent(x, y, abs_tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(
            math.isclose(a[key], b[key], abs_tol=abs_tol) if key ==
            'logprob' else _logprobs_equivalent(a[key], b[key], abs_tol)
            for key in a)
    return a == b


def _check_spec_prefill_evict_equivalence(config: TestConfig) -> None:
    """Require a survivor compacted after a first-token stop to match a later uncompacted replay."""
    with open(config.get_output_json_file(), encoding='utf-8') as output_file:
        responses = json.load(output_file).get('responses', [])
    if len(responses) != 4:
        raise RuntimeError(
            "The speculative prefill-eviction fixture must produce exactly four responses, "
            f"got {len(responses)}.")

    stopped, compacted_survivor, replayed_survivor, _ = responses
    if stopped.get('finish_reason') != 'stop-words' or len(
            stopped.get('logprobs', [])) != 1:
        raise RuntimeError(
            "The speculative prefill-eviction fixture did not stop slot 0 on its first token."
        )

    for field in ('output_text', 'finish_reason'):
        if compacted_survivor.get(field) != replayed_survivor.get(field):
            raise RuntimeError(
                f"The compacted survivor differs from its replayed baseline in {field}."
            )
    if not _logprobs_equivalent(compacted_survivor.get('logprobs'),
                                replayed_survivor.get('logprobs')):
        raise RuntimeError(
            "The compacted survivor differs from its replayed baseline in logprobs."
        )


def _source_test_case_file(config: TestConfig) -> Optional[str]:
    """Return the canonical (un-rewritten) test case JSON path, ignoring any
    per-config preprocessed override that this module may have already set.
    """
    override = config._test_case_file_override
    config._test_case_file_override = None
    try:
        return config.get_test_case_file()
    except ValueError:
        return None
    finally:
        config._test_case_file_override = override


def _write_preprocessed_test_case(config: TestConfig, data: Dict[str, Any],
                                  logger) -> str:
    """Write *data* as a per-config preprocessed test case JSON and return its path.

    Output goes under ``config.test_log_dir`` (a per-test directory pytest
    creates afresh per parametrization, so different ASR/OMNI configs never
    contend for the same path even under pytest-xdist). The write itself is
    atomic via ``os.replace`` to defend against partial-write corruption.
    """
    out_dir = config.test_log_dir or os.path.dirname(
        _source_test_case_file(config) or ".") or "."
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "preprocessed_test_case.json")
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
    os.replace(tmp_path, out_path)
    if logger:
        logger.debug("Wrote preprocessed test case: %s", out_path)
    return out_path


def _substitute_placeholder_in_test_case(config: TestConfig, placeholder: str,
                                         replacement: str,
                                         logger) -> Dict[str, Any]:
    """Replace ``placeholder`` with ``replacement`` in every string value of
    the test case JSON, write the result to a per-config preprocessed copy,
    and expose it via ``config._test_case_file_override``.

    The rewrite walks the parsed document recursively rather than doing a
    text-level ``str.replace``, so a placeholder that incidentally appears as
    a JSON key or inside a comment-shaped string elsewhere is not silently
    replaced. The version-controlled source JSON is *never* mutated, which
    keeps pytest-xdist parallel runs safe and avoids stale-state-on-rerun
    that the old in-place sed/text rewrite suffered from. Returns a
    ``run_command``-shaped dict so call sites keep their existing failure
    handling.
    """
    try:
        source_file = _source_test_case_file(config)
        if source_file is None or not os.path.isfile(source_file):
            return {
                "success": False,
                "error": f"test case file not found: {source_file!r}",
                "output": "",
            }

        with open(source_file) as f:
            data = json.load(f)

        def _walk(node: Any) -> Any:
            if isinstance(node, dict):
                return {k: _walk(v) for k, v in node.items()}
            if isinstance(node, list):
                return [_walk(v) for v in node]
            if isinstance(node, str):
                return node.replace(placeholder, replacement)
            return node

        new_data = _walk(data)
        if new_data == data:
            # Placeholder absent — nothing to do, leave override unset so
            # the canonical source path is used downstream.
            return {"success": True, "error": None, "output": ""}

        config._test_case_file_override = _write_preprocessed_test_case(
            config, new_data, logger)
        if logger:
            logger.info("Substituted %s in %s -> %s", placeholder, source_file,
                        config._test_case_file_override)
        return {"success": True, "error": None, "output": ""}
    except (OSError, json.JSONDecodeError) as e:
        return {
            "success": False,
            "error": f"placeholder substitution failed: {e}",
            "output": "",
        }


def check_result_failures(result: Dict[str, Any]) -> None:
    """Check baseline regressions first, then static threshold failures.

    Called by pipeline tests after execute_*_test returns successfully.
    """
    failures = []
    if result.get('baseline_regressions'):
        failures.append("Baseline regression:\n  " +
                        "\n  ".join(result['baseline_regressions']))
    if result.get('threshold_failure'):
        failures.append(result['threshold_failure'])
    if failures:
        pytest.fail("\n\n".join(failures))


def _try_save_baseline(config: TestConfig, test_func: str,
                       result: Dict[str, Any], logger) -> None:
    """Seed the baseline CSV with the current result.

    Opt-in only: requires BASELINE_AUTOSAVE=1. Without that env var, the
    baseline CSV is treated as read-only during regression runs — missing
    entries are reported (see caller) but never written, so baselines are
    not silently polluted by ad-hoc test runs.
    """
    if os.environ.get('BASELINE_AUTOSAVE',
                      '').strip().lower() not in ('1', 'true'):
        return
    csv_path = os.environ.get('BASELINE_CSV', 'logs/baseline.csv')
    if not result.get('success', False):
        return
    if result.get('threshold_failure'):
        return
    save_to_baseline(csv_path, config.model_type.value, test_func,
                     config.param_str, result)
    if logger:
        logger.info(
            "No baseline entry for [%s]. "
            "Saved current result to %s (BASELINE_AUTOSAVE=1)",
            config.param_str, csv_path)


_DEVICE_CONFIG = None


def _detected_compute_capability(logger=None):
    """Compute capability of the board under test, detected once per session."""
    global _DEVICE_CONFIG
    if _DEVICE_CONFIG is None:
        try:
            from .device import DeviceConfig
            _DEVICE_CONFIG = DeviceConfig.auto_detect(None, logger)
        except Exception:
            _DEVICE_CONFIG = False
    return getattr(_DEVICE_CONFIG, "compute_capability", None) or None


def _check_baseline_regression(config: TestConfig,
                               test_func: str,
                               result: Dict[str, Any],
                               logger,
                               check_perf: bool = False) -> bool:
    """Check accuracy (and optionally perf) regression against baseline CSV.

    Returns True if baseline entry was found (regardless of pass/fail).
    When baseline is found, threshold_failure is cleared since baseline takes priority.
    If no baseline exists, the current result is NOT written back — baseline
    CSVs are managed externally. Set BASELINE_AUTOSAVE=1 to opt in to seeding.

    Args:
        check_perf: only True for benchmark tests; inference skips perf comparison.
    """
    baseline = get_baseline()
    if baseline is None:
        if logger:
            logger.info(
                "No baseline loaded for [%s]; skipping regression check",
                config.param_str)
        _try_save_baseline(config, test_func, result, logger)
        return False

    entry = baseline.find_by_param(config.param_str,
                                   test_func,
                                   model_type_value=config.model_type.value)
    if entry is None:
        if logger:
            logger.info(
                "No baseline entry for [%s]; skipping regression check",
                config.param_str)
        _try_save_baseline(config, test_func, result, logger)
        return False

    regressions = []
    all_summaries = []

    current_acc = map_accuracy_result_to_csv(result)
    if current_acc:
        acc_reg, acc_sum = baseline.check_accuracy_regression(
            entry, current_acc)
        regressions.extend(acc_reg)
        all_summaries.extend(acc_sum)

    if check_perf:
        raw_output = result.get('output', '')
        cc = _detected_compute_capability(logger)
        current_perf = parse_perf_from_output(raw_output, cc)
        if not peak_gpu_memory_is_comparable(raw_output, cc):
            all_summaries.append(
                "memory_usage_peak_gpu_memory (MB): skipped - this platform reports "
                f"'{gpu_memory_metric_from_output(raw_output)}', which measures system "
                "memory pressure, not this process's GPU allocation")
        # Merge accuracy metrics into perf dict; check_perf_regression
        # only looks at columns in PERF_LOWER/HIGHER_IS_BETTER, so extras
        # (e.g. rouge scores) are naturally ignored.
        current_perf.update(current_acc)
        if current_perf:
            perf_reg, perf_sum = baseline.check_perf_regression(
                entry, current_perf)
            regressions.extend(perf_reg)
            all_summaries.extend(perf_sum)

    if logger and all_summaries:
        logger.info("Baseline comparison:\n  " + "\n  ".join(all_summaries))

    if regressions:
        result['baseline_regressions'] = regressions
        if logger:
            logger.warning("Baseline regressions detected:\n  " +
                           "\n  ".join(regressions))

    # Baseline found → it takes priority, discard static threshold result
    result.pop('threshold_failure', None)

    # Optional auto-promote: PROMOTE_BASELINE=1 overwrites baseline cells
    # where the current run is >1% better.
    csv_path = os.environ.get('BASELINE_CSV', 'logs/baseline.csv')
    promote_baseline_if_better(csv_path, config.model_type.value, test_func,
                               config.param_str, result, logger)
    return True


def execute_build_test(
        config: TestConfig, executable_files: Dict[str, str],
        remote_config: Optional[RemoteConfig], logger,
        env_config: Optional[EnvironmentConfig]) -> Dict[str, Any]:
    """Execute build test for any model type"""

    # Generate all build commands
    commands = generate_build_commands(config, executable_files)

    all_outputs = []

    engine_file_map = {
        executable_files['llm_build']: ["llm.engine"],
        executable_files['visual_build']: ["visual.engine"],
        executable_files['audio_build']: [
            os.path.join("audio", "audio_encoder.engine"),
            os.path.join("code2wav", "code2wav.engine"),
        ],
        executable_files['action_build']:
        [os.path.join("action", "action.engine")],
    }

    for i, (cmd, timeout) in enumerate(commands):
        task_name = f"Build step {i+1}/{len(commands)}"
        if logger:
            logger.info(f"Starting {task_name}: {' '.join(cmd)}")

        engine_candidates = engine_file_map.get(cmd[0], [])
        engine_dir = next((arg.split('=', 1)[1]
                           for arg in cmd if arg.startswith('--engineDir=')),
                          None) if engine_candidates else None
        if engine_dir:
            skip = False
            for engine_filename in engine_candidates:
                if check_file_exists(os.path.join(engine_dir, engine_filename),
                                     remote_config, logger):
                    if logger:
                        logger.info(
                            f"{engine_filename} already exists in {engine_dir}. Skipping."
                        )
                    all_outputs.append(
                        f"{engine_filename} already exists - skipped")
                    skip = True
                    break
            if skip:
                continue

        result = run_with_trt_env(cmd, remote_config, timeout, logger,
                                  env_config)
        all_outputs.append(result['output'])

        if not result['success']:
            return {
                'success': False,
                'error':
                f"{task_name} failed: {result.get('error', 'Unknown error')}",
                'output': '\n'.join(all_outputs),
                'test_type': TaskType.BUILD.value
            }

    return {
        'success': True,
        'error': None,
        'output': '\n'.join(all_outputs),
        'test_type': TaskType.BUILD.value
    }


def execute_e2e_bench_test(
        config: TestConfig, executable_files: Dict[str, str],
        remote_config: Optional[RemoteConfig], logger,
        env_config: Optional[EnvironmentConfig]) -> Dict[str, Any]:
    """Execute end-to-end benchmark test for any model type"""

    # Handle LoRA weights replacement if needed
    if config.max_lora_rank is not None and config.max_lora_rank > 0:
        # Replace the $LORA_WEIGHTS_DIR placeholder with the resolved path.
        result = _substitute_placeholder_in_test_case(
            config, "$LORA_WEIGHTS_DIR", config.get_lora_weights_dir(), logger)
        if not result['success']:
            result['test_type'] = TaskType.E2E_BENCH.value
            return result

    if config.model_type == ModelType.VLA:
        result = _substitute_placeholder_in_test_case(
            config, _ALPAMAYO_DATASET_PLACEHOLDER,
            config.get_alpamayo_dataset_dir(), logger)
        if not result['success']:
            result['test_type'] = TaskType.E2E_BENCH.value
            return result

    # Generate all e2e benchmark commands
    commands = generate_e2e_bench_commands(config, executable_files)

    all_outputs = []

    for i, (cmd, timeout) in enumerate(commands):
        task_name = f"Benchmark step {i+1}/{len(commands)}"
        if logger:
            logger.info(f"Starting {task_name}: {' '.join(cmd)}")

        result = run_with_trt_env(cmd, remote_config, timeout, logger,
                                  env_config)
        all_outputs.append(result['output'])

        if not result['success']:
            return {
                'success': False,
                'error':
                f"{task_name} failed: {result.get('error', 'Unknown error')}",
                'output': '\n'.join(all_outputs),
                'test_type': TaskType.E2E_BENCH.value
            }

    # Calculate metrics based on dataset type
    final_result = {
        'success': True,
        'error': None,
        'output': '\n'.join(all_outputs),
        'test_type': TaskType.E2E_BENCH.value
    }

    try:
        # Use model-specific reference if available, fallback to generic test case file
        reference_file = config.get_reference_json_file(
        ) or config.get_test_case_file()
        output_file = _sync_remote_output_file(config.get_output_json_file(),
                                               remote_config, logger)
        # Pass file paths directly to the accuracy checker (runs on host only)
        metrics_result = check_accuracy_with_dataset(output_file,
                                                     reference_file,
                                                     config.test_case, logger)

        # Merge metrics result into final result
        final_result.update(metrics_result)
        if config.context_reuse:
            if config.test_case == 'llm_context_reuse':
                _check_context_reuse_cold_hit_equivalence(config)
            elif config.test_case == 'llm_spec_prefill_evict':
                _check_spec_prefill_evict_equivalence(config)
            elif config.test_case == 'vlm_context_reuse':
                _check_vlm_context_reuse_profile(config, logger, remote_config)
            final_result['context_reuse_reused_tokens'] = (
                _read_context_reuse_profile(config, logger, remote_config))

    except Exception as e:
        final_result['error'] = f"Failed to calculate metrics: {str(e)}"
        final_result['success'] = False

    if final_result['success']:
        _check_baseline_regression(config,
                                   'test_e2e_bench',
                                   final_result,
                                   logger,
                                   check_perf=True)

    return final_result


def execute_inference_test(
        config: TestConfig, executable_files: Dict[str, str],
        remote_config: Optional[RemoteConfig], logger,
        env_config: Optional[EnvironmentConfig]) -> Dict[str, Any]:
    """Execute inference test for any model type"""

    # Handle LoRA weights replacement if needed
    if config.max_lora_rank is not None and config.max_lora_rank > 0:
        # Replace the $LORA_WEIGHTS_DIR placeholder with the resolved path.
        result = _substitute_placeholder_in_test_case(
            config, "$LORA_WEIGHTS_DIR", config.get_lora_weights_dir(), logger)
        if not result['success']:
            result['test_type'] = TaskType.INFERENCE.value
            return result

    if config.model_type == ModelType.VLA:
        result = _substitute_placeholder_in_test_case(
            config, _ALPAMAYO_DATASET_PLACEHOLDER,
            config.get_alpamayo_dataset_dir(), logger)
        if not result['success']:
            result['test_type'] = TaskType.INFERENCE.value
            return result

    # Generate all inference commands
    commands = generate_inference_commands(config, executable_files)

    all_outputs = []

    for i, (cmd, timeout) in enumerate(commands):
        task_name = f"Inference step {i+1}/{len(commands)}"
        if logger:
            logger.info(f"Starting {task_name}: {' '.join(cmd)}")

        result = run_with_trt_env(cmd, remote_config, timeout, logger,
                                  env_config)
        all_outputs.append(result['output'])

        if not result['success']:
            return {
                'success': False,
                'error':
                f"{task_name} failed: {result.get('error', 'Unknown error')}",
                'output': '\n'.join(all_outputs),
                'test_type': TaskType.INFERENCE.value
            }

    # Calculate metrics based on dataset type
    final_result = {
        'success': True,
        'error': None,
        'output': '\n'.join(all_outputs),
        'test_type': TaskType.INFERENCE.value
    }

    try:
        # Use model-specific reference if available, fallback to generic test case file
        reference_file = config.get_reference_json_file(
        ) or config.get_test_case_file()
        output_file = _sync_remote_output_file(config.get_output_json_file(),
                                               remote_config, logger)
        # Pass file paths directly to the accuracy checker (runs on host only)
        metrics_result = check_accuracy_with_dataset(output_file,
                                                     reference_file,
                                                     config.test_case, logger)

        # Merge metrics result into final result
        final_result.update(metrics_result)
        if config.context_reuse:
            if config.test_case == 'llm_context_reuse':
                _check_context_reuse_cold_hit_equivalence(config)
            elif config.test_case == 'llm_spec_prefill_evict':
                _check_spec_prefill_evict_equivalence(config)
            elif config.test_case == 'vlm_context_reuse':
                _check_vlm_context_reuse_profile(config, logger, remote_config)
            final_result['context_reuse_reused_tokens'] = (
                _read_context_reuse_profile(config, logger, remote_config))

    except Exception as e:
        final_result['error'] = f"Failed to calculate metrics: {str(e)}"
        final_result['success'] = False

    if final_result['success']:
        _check_baseline_regression(config, 'test_inference', final_result,
                                   logger)

    return final_result


def execute_kernel_bench_test(
        config: TestConfig, executable_files: Dict[str, str],
        remote_config: Optional[RemoteConfig], logger,
        env_config: Optional[EnvironmentConfig]) -> Dict[str, Any]:
    """Execute kernel_bench test - validates that the kernel benchmark runs successfully"""

    commands = generate_kernel_bench_commands(config, executable_files)

    all_outputs = []

    for i, (cmd, timeout) in enumerate(commands):
        task_name = f"kernel_bench step {i+1}/{len(commands)}"
        if logger:
            logger.info(f"Starting {task_name}: {' '.join(cmd)}")

        result = run_with_trt_env(cmd, remote_config, timeout, logger,
                                  env_config)
        all_outputs.append(result['output'])

        if not result['success']:
            return {
                'success': False,
                'error':
                f"{task_name} failed: {result.get('error', 'Unknown error')}",
                'output': '\n'.join(all_outputs),
                'test_type': TaskType.KERNEL_BENCH.value
            }

    return {
        'success': True,
        'error': None,
        'output': '\n'.join(all_outputs),
        'test_type': TaskType.KERNEL_BENCH.value
    }


def execute_vlmevalkit_test(
        config: TestConfig, executable_files: Dict[str, str],
        remote_config: Optional[RemoteConfig], logger,
        env_config: Optional[EnvironmentConfig]) -> Dict[str, Any]:
    """Run VLM inference, then convert and score with VLMEvalKit."""
    all_outputs = []

    if logger:
        logger.info("VLMEvalKit step 1: running inference")

    inference_commands = generate_inference_commands(config, executable_files)
    for i, (cmd, timeout) in enumerate(inference_commands):
        task_name = f"Inference step {i+1}/{len(inference_commands)}"
        if logger:
            logger.info(f"Starting {task_name}: {' '.join(cmd)}")

        result = run_with_trt_env(cmd, remote_config, timeout, logger,
                                  env_config)
        all_outputs.append(result.get('output', ''))

        if not result['success']:
            return {
                'success': False,
                'error':
                f"{task_name} failed: {result.get('error', 'Unknown error')}",
                'output': '\n'.join(all_outputs),
                'test_type': TaskType.VLMEVALKIT.value
            }

    if logger:
        logger.info("VLMEvalKit step 2: preparing xlsx and running evaluation")

    vlmevalkit_commands = generate_vlmevalkit_commands(config, env_config)
    total_steps = len(inference_commands) + len(vlmevalkit_commands)

    for i, (cmd, timeout) in enumerate(vlmevalkit_commands):
        step_num = len(inference_commands) + i + 1
        task_name = f"VLMEvalKit step {step_num}/{total_steps}"
        if logger:
            logger.info(f"Starting {task_name}: {' '.join(cmd)}")

        # These commands are host-side Python post-processing and scoring,
        # so they do not need TensorRT library path setup.
        result = run_command(cmd, remote_config, timeout, logger)
        all_outputs.append(result.get('output', ''))

        if not result['success']:
            return {
                'success': False,
                'error':
                f"{task_name} failed: {result.get('error', 'Unknown error')}",
                'output': '\n'.join(all_outputs),
                'test_type': TaskType.VLMEVALKIT.value
            }

    return {
        'success': True,
        'error': None,
        'output': '\n'.join(all_outputs),
        'test_type': TaskType.VLMEVALKIT.value
    }
