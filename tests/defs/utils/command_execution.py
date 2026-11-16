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

import csv
import json
import math
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import check_file_exists, run_command, run_with_trt_env

from ..config import ModelType, TaskType, TestConfig
from .accuracy import check_accuracy_with_dataset
from .baseline import (MODEL_TEST_METRICS, PERF_HIGHER_IS_BETTER,
                       PERF_LOWER_IS_BETTER, get_baseline,
                       gpu_memory_metric_from_output,
                       map_accuracy_result_to_csv, parse_perf_from_output,
                       peak_gpu_memory_is_comparable,
                       promote_baseline_if_better, save_to_baseline)
from .ci_engine_cache import EngineBundleCache
from .command_generation import (_uses_spec_decode, generate_build_commands,
                                 generate_e2e_bench_commands,
                                 generate_inference_commands,
                                 generate_kernel_bench_commands,
                                 generate_vlmevalkit_commands)

_ALPAMAYO_DATASET_PLACEHOLDER = "$ALPAMAYO_DATASET_DIR"
_TRAILING_SPECIAL_TOKEN_RE = re.compile(
    r"(?:<\|[^<>]*\|>|<[a-z_]+\|>|<(?:eos|bos|pad|unk|mask|start_of_turn|end_of_turn)>)+\s*$"
)


def _strip_trailing_special_tokens(text: str) -> str:
    """Remove tokenizer stop markers appended to otherwise constrained text."""
    return _TRAILING_SPECIAL_TOKEN_RE.sub("", text).rstrip()


_SPEC_DECODE_COMMON_PERF_COLUMNS = frozenset({
    'spec_decode_avg_accept_length',
    'spec_decode_acceptance_rate',
    'spec_decode_avg_tokens_per_run',
    'spec_decode_overall_tokens_per_second (tokens/s)',
    'spec_decode_draft_proposal_avg_time (ms)',
    'spec_decode_base_model_verification_avg_time (ms)',
})

_SPEC_DECODE_DRAFT_PREFILL_COLUMN = (
    'spec_decode_draft_model_prefill_avg_time (ms)')
_SPEC_DECODE_ACCEPTANCE_RATE_COLUMN = 'spec_decode_acceptance_rate'
_LLM_BASIC_ACCEPTANCE_RATE_THRESHOLD = 0.30

# These values prove that the runtime exercised the expected path and remain in
# the baseline report. One-shot prefill and component timings are not stable
# regression gates for a single request. End-to-end speculative throughput and
# average accept length remain gated at 15%. Dataset acceptance rate uses the
# same threshold; the one-prompt llm_basic smoke test allows 30% variance.
_SPEC_DECODE_INFORMATIONAL_PERF_COLUMNS = frozenset({
    'llm_prefill_avg_time_per_run (ms)',
    'llm_prefill_avg_time_per_token (ms)',
    'llm_prefill_tokens_per_second (tokens/s)',
    _SPEC_DECODE_DRAFT_PREFILL_COLUMN,
    'spec_decode_avg_tokens_per_run',
    'spec_decode_draft_proposal_avg_time (ms)',
    'spec_decode_base_model_verification_avg_time (ms)',
})


def _env_flag(name: str) -> bool:
    return os.environ.get(name, '').strip().lower() in ('1', 'true')


def _required_spec_decode_perf_columns(config: TestConfig) -> set[str]:
    required = set(_SPEC_DECODE_COMMON_PERF_COLUMNS)
    native_mtp = (config.is_mtp
                  and not config.model_name.lower().startswith('gemma-'))
    if (config.is_eagle and not config.is_mtp) or native_mtp:
        required.add(_SPEC_DECODE_DRAFT_PREFILL_COLUMN)
    return required


def _engine_files(directory):
    if not directory or not os.path.isdir(directory):
        return set()
    return {
        os.path.relpath(os.path.join(root, name), directory)
        for root, _, names in os.walk(directory)
        for name in names if name.endswith(".engine")
    }


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
    """Validate context-cache metrics and return the reused prefill tokens."""
    profile_path = _sync_remote_output_file(config.get_profile_json_file(),
                                            remote_config, logger)
    with open(profile_path, encoding='utf-8') as profile_file:
        profile = json.load(profile_file)

    reused_tokens = profile.get('prefill', {}).get('reused_tokens')
    if config.test_case == 'llm_spec_prefill_evict':
        context_cache = profile.get('context_cache', {})
        lookup_bypass_sequences = context_cache.get('lookup_bypass_sequences')
        committed_publications = context_cache.get('publications',
                                                   {}).get('committed')
        if (not isinstance(reused_tokens, int) or reused_tokens < 0
                or not isinstance(lookup_bypass_sequences, int)
                or lookup_bypass_sequences <= 0
                or not isinstance(committed_publications, int)
                or committed_publications <= 0):
            raise RuntimeError(
                "The speculative prefill-eviction profile must report a "
                "non-negative prefill.reused_tokens value, at least one "
                "lookup bypass, and at least one committed publication: "
                f"reused_tokens={reused_tokens!r}, "
                f"lookup_bypass_sequences={lookup_bypass_sequences!r}, "
                f"committed_publications={committed_publications!r}")
        return reused_tokens

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


_GUIDED_DECODING_TEST_CASES = ('llm_guided_decoding',
                               'llm_guided_decoding_jump_forward')
_JUMP_FORWARD_TEST_CASE = 'llm_guided_decoding_jump_forward'
# Guides whose output has a finite length, so a response that hits the generation budget is a
# failure rather than a long answer; json_object and structural_tag leave free text to the model.
_BOUNDED_GUIDES = ('json_schema', 'regex', 'ebnf', 'choice')
_JUMP_FORWARD_STATS = re.compile(
    r"Request (\d+) jump-forward: jumps=(\d+) forced=(\d+)")
_JUMP_FORWARD_REFUSAL = "Jump-forward is not supported on"


def _check_jump_forward_ran(output: str, logger=None) -> None:
    """Require the jump-forward fixture to have jumped.

    Conformance holds whether or not the runtime jumps, so a deployment that refuses the flag
    or grafts nothing would still pass it. The runtime logs one ``jump-forward:`` line per
    request that asked for the flag, and the fixture's json_schema and regex requests force
    text, so at least one request must report a jump and no refusal may be logged.
    """
    refusal = next(
        (line.strip()
         for line in output.splitlines() if _JUMP_FORWARD_REFUSAL in line),
        None)
    if refusal is not None:
        raise RuntimeError(f"Jump-forward was refused: {refusal}")
    stats = _JUMP_FORWARD_STATS.findall(output)
    if not stats:
        raise RuntimeError(
            "No 'jump-forward:' statistics line in the runtime output; "
            "the requests did not run with jump-forward.")
    jumps = sum(int(count) for _, count, _ in stats)
    forced = sum(int(count) for _, _, count in stats)
    if jumps == 0:
        raise RuntimeError(
            f"jump-forward ran on {len(stats)} requests but none jumped.")
    if logger:
        logger.info("jump-forward: %d requests, %d jumps, %d forced tokens",
                    len(stats), jumps, forced)


def _check_guided_decoding_conformance(
        config: TestConfig,
        logger=None,
        require_completion: bool = False) -> None:
    """Require every guided response to satisfy the guide that produced it.

    Guided decoding constrains generation token by token, so a violation is a real
    regression rather than a quality dip and no accuracy metric would catch it. The
    expectations are read back from the input fixture so the two cannot drift apart.

    json_schema, json_object, regex and choice are checked against the guide itself. ebnf
    cannot be validated without a grammar parser, and structural_tag only constrains text
    once a trigger appears, so those two are checked for a clean finish plus, for
    structural_tag, a well-formed tag whenever a trigger did fire; their grammar semantics
    are covered by the unit tests.

    With ``require_completion`` (the jump-forward fixture), a response to a bounded guide
    must have finished: hitting the generation budget fails instead of being skipped, and
    at least one response must have been checked, so a run that jumps once and then emits
    invalid text until the budget expires cannot pass.
    """
    with open(config.get_test_case_file(), encoding='utf-8') as input_file:
        requests = json.load(input_file).get('requests', [])
    with open(config.get_output_json_file(), encoding='utf-8') as output_file:
        responses = json.load(output_file).get('responses', [])

    if len(responses) != len(requests):
        raise RuntimeError(
            f"Expected {len(requests)} guided-decoding responses, got {len(responses)}."
        )

    json_types = {
        'string': str,
        'integer': int,
        'number': (int, float),
        'boolean': bool,
        'array': list,
        'object': dict,
    }

    checked = 0
    for index, (request, response) in enumerate(zip(requests, responses)):
        text = response.get('output_text', '')
        guide = request.get('guided_decoding') or {}

        # A grammar that fails to compile, or one that cannot be satisfied, ends the slot
        # with `error`; every fixture here is expected to generate normally.
        finish_reason = response.get('finish_reason')
        if finish_reason == 'error':
            raise RuntimeError(
                f"Response {index} ended with finish_reason=error; text={text!r}"
            )

        # Hitting the generation budget leaves a valid prefix rather than a complete
        # document, so only a run that reached EOS can be parsed as a whole.
        if finish_reason == 'max-length':
            if require_completion and any(key in guide
                                          for key in _BOUNDED_GUIDES):
                raise RuntimeError(
                    f"Response {index} hit the generation budget on a bounded guide; "
                    f"text={text!r}")
            continue

        text = _strip_trailing_special_tokens(text)

        if 'json_schema' in guide or 'json_object' in guide:
            try:
                document = json.loads(text)
            except ValueError as error:
                raise RuntimeError(
                    f"Response {index} is constrained to JSON but did not parse: {error}; "
                    f"text={text!r}") from error
            if not isinstance(document, dict):
                raise RuntimeError(
                    f"Response {index} must be a JSON object, got {type(document).__name__}."
                )
            schema = guide.get('json_schema', {})
            for field in schema.get('required', []):
                if field not in document:
                    raise RuntimeError(
                        f"Response {index} is missing required field {field!r}: {document!r}"
                    )
            for field, spec in schema.get('properties', {}).items():
                expected = json_types.get(spec.get('type'))
                if field in document and expected is not None and not isinstance(
                        document[field], expected):
                    raise RuntimeError(
                        f"Response {index} field {field!r} should be {spec['type']}, "
                        f"got {document[field]!r}")
        elif 'regex' in guide:
            if re.fullmatch(guide['regex'], text) is None:
                raise RuntimeError(
                    f"Response {index} does not match its regex {guide['regex']!r}: {text!r}"
                )
        elif 'choice' in guide:
            if text not in guide['choice']:
                raise RuntimeError(
                    f"Response {index} is not one of its choices {guide['choice']!r}: "
                    f"{text!r}")
        elif 'structural_tag' in guide:
            tag_format = guide['structural_tag'].get('format', {})
            for tag in tag_format.get('tags', []):
                if tag['begin'] in text and tag['end'] not in text:
                    raise RuntimeError(
                        f"Response {index} opened {tag['begin']!r} without closing "
                        f"{tag['end']!r}: {text!r}")
        elif 'ebnf' not in guide:
            raise RuntimeError(
                f"Request {index} of the guided-decoding fixture carries no guide."
            )
        checked += 1
    if require_completion and checked == 0:
        raise RuntimeError(
            "No guided-decoding response was checked: every response hit the generation budget."
        )
    if logger:
        logger.info("Guided-decoding conformance: %d of %d responses checked",
                    checked, len(responses))


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


def _accuracy_reference_file(config: TestConfig) -> Optional[str]:
    """Return the reference input used to locate dataset-side accuracy data."""
    reference_file = config.get_reference_json_file()
    if reference_file is not None:
        return reference_file
    if config.model_type == ModelType.VLA:
        return _source_test_case_file(config) or config.get_test_case_file()
    return config.get_test_case_file()


def _write_preprocessed_test_case(
        config: TestConfig,
        data: Dict[str, Any],
        logger,
        filename="preprocessed_test_case.json") -> str:
    """Write *data* as a per-config preprocessed test case JSON and return its path.

    Output goes under ``config.test_log_dir`` (a per-test directory pytest
    creates afresh per parametrization, so different ASR/OMNI configs never
    contend for the same path even under pytest-xdist). The write itself is
    atomic via ``os.replace`` to defend against partial-write corruption.
    """
    out_dir = config.test_log_dir or os.path.dirname(
        _source_test_case_file(config) or ".") or "."
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
    os.replace(tmp_path, out_path)
    if logger:
        logger.debug("Wrote preprocessed test case: %s", out_path)
    return out_path


def prepare_spec_decode_input(config: TestConfig, logger) -> str:
    """Use a greedy copy of the request for greedy-only speculative paths."""
    source = config.get_test_case_file()
    requires_greedy = (config.is_eagle or config.is_dflash or config.is_domino
                       or config.is_jetspec_tree or config.is_dspark_tree)
    if not requires_greedy:
        return source

    with open(source, encoding="utf-8") as stream:
        request = json.load(stream)
    if request.get("top_k") == 1:
        return source

    request["top_k"] = 1
    config._test_case_file_override = _write_preprocessed_test_case(
        config, request, logger)
    if logger:
        logger.info("Using greedy sampling input for speculative decoding: %s",
                    config._test_case_file_override)
    return config._test_case_file_override


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


def _prepare_alpamayo_action_chat(config: TestConfig,
                                  logger) -> Dict[str, Any]:
    """Prepare six action samples per clip from the chat dataset."""
    if config.test_case != "alpamayo_action_chat":
        return {"success": True, "error": None, "output": ""}
    try:
        source_path = config.get_test_case_file()
        with open(source_path, encoding="utf-8") as source:
            data = json.load(source)
        reference_dir = os.path.dirname(
            _source_test_case_file(config) or source_path)
        with open(os.path.join(reference_dir, "gt.json"),
                  encoding="utf-8") as source:
            ground_truth = json.load(source)
        system_prompt = data.pop("default_system_prompt")
        requests = data["requests"]
        if not requests or len(requests) % 6:
            raise ValueError(
                "Alpamayo action chat requires six requests per clip")
        for offset in range(0, len(requests), 6):
            if len({request["id"]
                    for request in requests[offset:offset + 6]}) != 1:
                raise ValueError(
                    "Alpamayo action samples must be grouped by clip")
        for request in requests:
            messages = request["messages"]
            if messages[0]["role"] != "system":
                messages.insert(0, {
                    "role": "system",
                    "content": system_prompt
                })
            if messages[-1]["role"] != "assistant":
                raise ValueError(
                    "Alpamayo action chat must end with an assistant prefix")
            user_messages = [
                message for message in messages if message["role"] == "user"
            ]
            if (len(user_messages) != 1
                    or not isinstance(user_messages[0]["content"], list)):
                raise ValueError(
                    "Alpamayo action chat requires one structured user message"
                )
            content = user_messages[0]["content"]
            if not any(item.get("type") == "trajectory" for item in content):
                text_item = content[-1]
                if text_item.get("type") != "text":
                    raise ValueError(
                        "Alpamayo action chat has no trajectory history")
                history, marker, instruction = text_item["text"].partition(
                    "<|traj_history_end|>")
                if not marker or not history.startswith(
                        "<|traj_history_start|>"):
                    raise ValueError(
                        "Alpamayo action chat has no trajectory history")
                text_item["text"] = instruction
                content.insert(
                    -1, {
                        "type":
                        "trajectory",
                        "trajectory":
                        ground_truth[request["id"]]["ego_history_xyz"],
                    })
        data["batch_size"] = 6
        data["add_generation_prompt"] = False
        config._test_case_file_override = _write_preprocessed_test_case(
            config, data, logger)
        return {"success": True, "error": None, "output": ""}
    except (OSError, ValueError, KeyError, IndexError) as e:
        return {
            "success": False,
            "error": f"Alpamayo action chat preprocessing failed: {e}",
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


def check_declared_metrics(config: TestConfig,
                           result: Dict[str, Any],
                           test_func: str,
                           logger=None) -> None:
    """Check YAML-selected metrics against the platform baseline CSV.

    Declared current metrics must be usable. Missing references skip comparison
    and retain the measured values for optional baseline seeding.
    """
    checks = config.metric_checks or {}
    current = map_accuracy_result_to_csv(result)
    current.update(
        parse_perf_from_output(result.get('output', ''),
                               _detected_compute_capability(logger)))

    baseline = get_baseline()
    entry = None
    failures = []
    summaries = []
    measured = []
    skipped = []
    if baseline is not None:
        entry = baseline.find_by_param(
            config.param_str,
            test_func,
            model_type_value=config.model_type.value)

    for section_name, section in checks.items():
        if section_name not in MODEL_TEST_METRICS:
            continue
        aliases = MODEL_TEST_METRICS[section_name]
        regression = section["max_regression"]
        for metric in section["required"]:
            column = aliases[metric]
            actual = current.get(column)
            if not isinstance(actual, (int, float)):
                failures.append(f"required metric {metric} was not produced")
                continue
            lower_is_better = column in PERF_LOWER_IS_BETTER
            higher_is_better = (not lower_is_better
                                and (column in PERF_HIGHER_IS_BETTER
                                     or column.startswith("accuracy_")))
            if actual < 0 or (higher_is_better and actual == 0):
                failures.append(
                    f"required metric {metric} has invalid value {actual}")
                continue
            measured.append(f"{metric}={actual:.4f}")
            if entry is None:
                continue
            if column not in entry:
                skipped.append(column)
                continue
            reference = entry.get(column)
            if not isinstance(reference, (int, float)) or reference < 0:
                failures.append(
                    f"baseline metric {metric} is missing or invalid")
                continue
            if reference == 0:
                if higher_is_better:
                    failures.append(
                        f"baseline metric {metric} must be positive")
                elif actual > 0:
                    failures.append(
                        f"REGRESSION {metric}: {actual:.4f} vs baseline 0")
                continue
            change = (actual - reference) / reference
            if lower_is_better:
                regressed = change > regression
                direction = "worse" if change > 0 else "better"
            elif higher_is_better:
                regressed = change < -regression
                direction = "better" if change > 0 else "worse"
            else:
                failures.append(f"metric {metric} has no comparison direction")
                continue
            comparison = (f"{metric}: {actual:.4f} vs baseline "
                          f"{reference:.4f} ({change:+.1%}, {direction})")
            summaries.append(comparison)
            if regressed:
                failures.append(f"REGRESSION {comparison} "
                                f"[threshold {regression:.0%}]")

    if entry is None or skipped:
        reason = ('missing_baseline_file'
                  if baseline is None else 'missing_baseline_entry'
                  if entry is None else 'missing_baseline_metrics')
        result['baseline_comparison_skipped'] = {
            'reason': reason,
            'missing_baseline': skipped,
        }
        result['baseline_comparison_metrics'] = current
        if logger:
            logger.info(
                "BASELINE_COMPARISON_DATA=%s",
                json.dumps(
                    {
                        'case': config.param_str,
                        'current_metrics': current,
                        'skipped': result['baseline_comparison_skipped'],
                    },
                    sort_keys=True))

    if logger and summaries:
        logger.info("Model-owned baseline comparison:\n  " +
                    "\n  ".join(summaries))
    accuracy_columns = {
        MODEL_TEST_METRICS['accuracy'][metric]
        for metric in checks.get('accuracy', {}).get('required', [])
    }
    if (entry is not None and accuracy_columns
            and accuracy_columns.issubset(entry)):
        result.pop('threshold_failure', None)
    if failures:
        result.setdefault('baseline_regressions', []).extend(failures)
    _try_save_baseline(config, test_func, result, logger)


def _try_save_baseline(config: TestConfig, test_func: str,
                       result: Dict[str, Any], logger) -> None:
    """Fill missing baseline cells with a successful current result.

    Opt-in only: requires BASELINE_AUTOSAVE=1. Without that env var, the
    baseline CSV is treated as read-only during regression runs — missing
    entries are reported (see caller) but never written, so baselines are
    not silently polluted by ad-hoc test runs.
    """
    if not _env_flag('BASELINE_AUTOSAVE'):
        return
    csv_path = os.environ.get('BASELINE_CSV', 'logs/baseline.csv')
    if not result.get('success', False):
        return
    if result.get('threshold_failure') or result.get('baseline_regressions'):
        return
    try:
        saved = save_to_baseline(csv_path, config.model_type.value, test_func,
                                 config.param_str, result,
                                 _detected_compute_capability(logger))
    except (OSError, csv.Error, UnicodeError) as error:
        if logger:
            logger.warning("Baseline update failed for [%s] at %s: %s",
                           config.param_str, csv_path, error)
        return
    if saved and logger:
        logger.info(
            "Filled missing baseline cells for [%s] in %s (BASELINE_AUTOSAVE=1)",
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
    If no baseline exists, the current result is NOT written back unless
    BASELINE_AUTOSAVE=1 opts in to seeding. Missing baseline files, entries,
    and speculative metrics are recorded and skipped during comparison.

    Args:
        check_perf: only True for benchmark tests; inference skips perf comparison.
    """
    baseline = get_baseline()
    entry = (baseline.find_by_param(
        config.param_str, test_func, model_type_value=config.model_type.value)
             if baseline is not None else None)
    if entry is None:
        reason = 'missing_baseline_file' if baseline is None else 'missing_baseline_entry'
        current_metrics = map_accuracy_result_to_csv(result)
        if check_perf:
            current_metrics.update(
                parse_perf_from_output(result.get('output', ''),
                                       _detected_compute_capability(logger)))
        result['baseline_comparison_skipped'] = {'reason': reason}
        result['baseline_comparison_metrics'] = current_metrics
        if logger:
            logger.info("Baseline comparison skipped for [%s]: %s",
                        config.param_str, reason)
            logger.info(
                "BASELINE_COMPARISON_DATA=%s",
                json.dumps(
                    {
                        'case': config.param_str,
                        'current_metrics': current_metrics,
                        'skipped': result['baseline_comparison_skipped'],
                    },
                    sort_keys=True))
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
        if any((config.is_eagle, config.is_mtp, config.is_dflash,
                config.is_jetspec, config.is_domino, config.is_dspark)):
            required_metrics = _required_spec_decode_perf_columns(config)
            missing_baseline = sorted(required_metrics - entry.keys())
            missing_current = sorted(required_metrics - current_perf.keys())
            result['baseline_comparison_skipped'] = {
                'missing_baseline': missing_baseline,
                'missing_current': missing_current,
            }
            result['baseline_comparison_metrics'] = dict(current_perf)
            if logger:
                logger.info(
                    "BASELINE_COMPARISON_DATA=%s",
                    json.dumps(
                        {
                            'case': config.param_str,
                            'current_metrics': current_perf,
                            'skipped': result['baseline_comparison_skipped'],
                        },
                        sort_keys=True))
            if missing_baseline:
                all_summaries.append(
                    "Speculative comparison skipped for metrics missing from baseline: "
                    + ", ".join(missing_baseline))
            if missing_current:
                all_summaries.append(
                    "Speculative comparison skipped for metrics missing from current run: "
                    + ", ".join(missing_current))
        if current_perf:
            is_spec_decode = any(
                (config.is_eagle, config.is_mtp, config.is_dflash,
                 config.is_jetspec, config.is_domino, config.is_dspark))
            report_only = (_SPEC_DECODE_INFORMATIONAL_PERF_COLUMNS
                           if is_spec_decode else frozenset())
            threshold_overrides = ({
                _SPEC_DECODE_ACCEPTANCE_RATE_COLUMN:
                _LLM_BASIC_ACCEPTANCE_RATE_THRESHOLD,
            } if is_spec_decode and config.test_case in ('llm_basic',
                                                         'llm_basic_sampled')
                                   else None)
            perf_reg, perf_sum = baseline.check_perf_regression(
                entry,
                current_perf,
                report_only_columns=report_only,
                threshold_overrides=threshold_overrides)
            regressions.extend(perf_reg)
            all_summaries.extend(perf_sum)

    if logger and all_summaries:
        logger.info("Baseline comparison:\n  " + "\n  ".join(all_summaries))

    if regressions:
        result['baseline_regressions'] = regressions
        if logger:
            logger.warning("Baseline regressions detected:\n  " +
                           "\n  ".join(regressions))

    if current_acc and all(column in entry for column in current_acc):
        result.pop('threshold_failure', None)

    _try_save_baseline(config, test_func, result, logger)

    # Optional auto-promote: PROMOTE_BASELINE=1 overwrites baseline cells
    # where the current run is >1% better.
    csv_path = os.environ.get('BASELINE_CSV', 'logs/baseline.csv')
    promote_baseline_if_better(csv_path, config.model_type.value, test_func,
                               config.param_str, result, logger,
                               _detected_compute_capability(logger))
    return True


def _get_arg_value(cmd: List[str], name: str) -> Optional[str]:
    prefix = f"--{name}="
    for index, arg in enumerate(cmd):
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
        if arg == f"--{name}" and index + 1 < len(cmd):
            return cmd[index + 1]
    return None


def _llm_build_engine_candidates(cmd: List[str],
                                 default: List[str]) -> List[str]:
    """Per-rank engine name for a tensor-parallel build, else the default."""
    tp_size = int(_get_arg_value(cmd, "tpSize") or 1)
    tp_rank = int(_get_arg_value(cmd, "tpRank") or 0)
    if tp_size <= 1:
        return default
    return [f"llm_world{tp_size}_rank{tp_rank}.engine"]


def execute_build_test(
        config: TestConfig, executable_files: Dict[str, str],
        remote_config: Optional[RemoteConfig], logger,
        env_config: Optional[EnvironmentConfig]) -> Dict[str, Any]:
    """Execute build test for any model type"""

    # Generate all build commands
    commands = generate_build_commands(config, executable_files)

    all_outputs = []
    remote_runner = None
    trtexec_path = None
    if remote_config is not None:

        def remote_runner(command, timeout):
            return run_with_trt_env(command, remote_config, timeout, logger,
                                    env_config)

        if env_config and env_config.trt_package_dir:
            trtexec_path = os.path.join(env_config.trt_package_dir, 'bin',
                                        'trtexec')
    engine_cache = EngineBundleCache(config,
                                     commands,
                                     logger,
                                     remote_runner=remote_runner,
                                     trtexec_path=trtexec_path)
    if engine_cache.restore_bundle():
        return {
            'success': True,
            'error': None,
            'output': 'complete TensorRT engine bundle restored from cache',
            'test_type': TaskType.BUILD.value
        }
    engine_cache.prepare_build()

    engine_file_map = {
        executable_files['llm_build']: ["llm.engine"],
        executable_files['visual_build']:
        [os.path.join("visual", "visual.engine")],
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

        cache_enabled = engine_cache is not None and engine_cache.enabled
        engine_candidates = engine_file_map.get(cmd[0], [])
        if cmd[0] == executable_files['llm_build']:
            engine_candidates = _llm_build_engine_candidates(
                cmd, engine_candidates)
        engine_dir = next((arg.split('=', 1)[1]
                           for arg in cmd if arg.startswith('--engineDir=')),
                          None)
        engines_before = _engine_files(engine_dir) if cache_enabled else set()
        if engine_dir and not cache_enabled:
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

        if (cache_enabled and engine_dir
                and not _engine_files(engine_dir) - engines_before):
            return {
                'success': False,
                'error': f"{task_name} produced no TensorRT engine",
                'output': '\n'.join(all_outputs),
                'test_type': TaskType.BUILD.value
            }

    if engine_cache is not None and engine_cache.enabled:
        engine_cache.publish_bundle()

    return {
        'success': True,
        'error': None,
        'output': '\n'.join(all_outputs),
        'test_type': TaskType.BUILD.value
    }


def _prepare_spec_greedy_input(config: TestConfig, logger) -> None:
    """Run speculative correctness cases with the supported greedy policy."""
    if not _uses_spec_decode(config):
        return
    with open(config.get_test_case_file()) as stream:
        data = json.load(stream)
    greedy = {"temperature": 0.0, "top_k": 1, "top_p": 1.0}
    data.update(greedy)
    for request in data.get("requests", []):
        request.update(greedy)
    config._test_case_file_override = _write_preprocessed_test_case(
        config, data, logger, filename=f"{config.param_str}.greedy.json")


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

    _prepare_spec_greedy_input(config, logger)

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
        if _uses_spec_decode(
                config) and "Decoder fallback:" in result['output']:
            result['success'] = False
            result[
                'error'] = "Speculative decoding fell back to another decoder"

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
        # The reference keeps its dataset location for sibling assets.
        reference_file = config.get_reference_json_file(
        ) or _source_test_case_file(config)
        output_file = _sync_remote_output_file(config.get_output_json_file(),
                                               remote_config, logger)
        # Pass file paths directly to the accuracy checker (runs on host only)
        metrics_result = check_accuracy_with_dataset(output_file,
                                                     reference_file,
                                                     config.test_case, logger)

        # Merge metrics result into final result
        final_result.update(metrics_result)
        if config.test_case in _GUIDED_DECODING_TEST_CASES:
            _check_guided_decoding_conformance(
                config,
                logger,
                require_completion=config.test_case == _JUMP_FORWARD_TEST_CASE)
        if config.test_case == _JUMP_FORWARD_TEST_CASE:
            _check_jump_forward_ran(final_result.get('output', ''), logger)
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
        result = _prepare_alpamayo_action_chat(config, logger)
        if not result['success']:
            result['test_type'] = TaskType.INFERENCE.value
            return result

    prepare_spec_decode_input(config, logger)

    _prepare_spec_greedy_input(config, logger)

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
        if _uses_spec_decode(
                config) and "Decoder fallback:" in result['output']:
            result['success'] = False
            result[
                'error'] = "Speculative decoding fell back to another decoder"

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
        reference_file = _accuracy_reference_file(config)
        output_file = _sync_remote_output_file(config.get_output_json_file(),
                                               remote_config, logger)
        # Pass file paths directly to the accuracy checker (runs on host only)
        metrics_result = check_accuracy_with_dataset(output_file,
                                                     reference_file,
                                                     config.test_case, logger)

        # Merge metrics result into final result
        final_result.update(metrics_result)
        if config.test_case in _GUIDED_DECODING_TEST_CASES:
            _check_guided_decoding_conformance(
                config,
                logger,
                require_completion=config.test_case == _JUMP_FORWARD_TEST_CASE)
        if config.test_case == _JUMP_FORWARD_TEST_CASE:
            _check_jump_forward_ran(final_result.get('output', ''), logger)
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

    if final_result['success'] and config.metric_checks:
        check_declared_metrics(config, final_result, 'test_inference', logger)
    elif final_result[
            'success'] and config.test_case not in _GUIDED_DECODING_TEST_CASES:
        # A guided fixture without metric checks is guarded by conformance, not by a baseline.
        _check_baseline_regression(config, 'test_inference', final_result,
                                   logger)

    return final_result


def _validate_runtime_bench_report(report, expected_iterations,
                                   expected_batch_size, expected_mode):
    """Reject successful process exits that did not complete the requested experiment."""
    samples = report.get('samples', [])
    if (report.get('schema_version') != 2
            or report.get('mode') != expected_mode
            or report.get('iterations') != expected_iterations
            or len(samples) != expected_iterations):
        raise ValueError("Incomplete benchmark report")
    for sample in samples:
        if not math.isfinite(sample['wall_ms']) or sample['wall_ms'] <= 0:
            raise ValueError("Invalid request wall time")
        counts = sample['output_token_counts']
        if len(counts) != expected_batch_size or any(count <= 0
                                                     for count in counts):
            raise ValueError(
                "Benchmark output does not cover the requested batch")
        if expected_mode == 'fixed_round' and sample[
                'decode_round_counts'] != [1] * expected_batch_size:
            raise ValueError(
                "Fixed-round request did not complete exactly one round per sequence"
            )
        if not any(row['stage'] == 'llm_prefill'
                   for row in sample['invocations']):
            raise ValueError(
                "Benchmark report is missing production prefill timing")


def execute_kernel_bench_test(
        config: TestConfig, executable_files: Dict[str, str],
        remote_config: Optional[RemoteConfig], logger,
        env_config: Optional[EnvironmentConfig]) -> Dict[str, Any]:
    """Validate request-backed benchmark results and round semantics."""

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

    try:
        output_dir = next(
            arg.split('=', 1)[1] for arg in commands[-1][0]
            if arg.startswith('--outputDir='))
        path = _sync_remote_output_file(
            os.path.join(output_dir, 'report.json'), remote_config, logger)
        with open(path, encoding='utf-8') as handle:
            report = json.load(handle)
        _validate_runtime_bench_report(report, 10, config.batch_size or 1,
                                       config.bench_mode or 'fixed_round')
    except (OSError, ValueError, KeyError, StopIteration) as error:
        return {
            'success': False,
            'error': str(error),
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
