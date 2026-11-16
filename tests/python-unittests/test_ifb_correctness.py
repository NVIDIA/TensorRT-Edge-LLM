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
"""Pass/fail decisions of tests/ifb/check_ifb_correctness.py on synthetic trace results."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_ifb_correctness",
    Path(__file__).resolve().parents[2] / "tests" / "ifb" /
    "check_ifb_correctness.py")
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)

TOKENS = [11, 12, 13, 14, 15]


def _req(tokens=TOKENS, gap_at=None, gap=1.0, **fields):
    """A finished request whose top-k logprobs put each token ``gap`` ahead of token 99 at ``gap_at``."""
    steps = [[[t, 0.0], [99, -(gap if i == gap_at else 1.0)], [98, -8.0]]
             for i, t in enumerate(tokens)]
    return dict(status="finished", token_ids=tokens, logprobs=steps) | fields


def _run(requests=None, decode=10, graph_hits=10, **fields):
    metrics = dict(decode_steps=decode, decode_graph_steps=graph_hits)
    requests = dict(r0=_req()) if requests is None else requests
    probe = dict(error="RequestFailedError: input too long", kind="OTHER")
    return dict(label="run",
                requests=requests,
                metrics_delta=metrics,
                failure_kind_probe=probe) | fields


_NEAR_TIE = _run({"r0": _req(gap_at=2, gap=0.0625)})
_DIVERGED = _run({"r0": _req([11, 12, 99, 14, 15])})
# The same argmax (token 11) with token 12's logprob raised from -3.0 to -2.0: raw cosine 0.99936, centered 0.99768.
_TAIL = [[token, -(6.0 + i)] for i, token in enumerate(range(13, 21))]
_PERTURBED_REF = _run({"r0": {**_req(), "logprobs": [[[11, -0.05], [12, -3.0], *_TAIL]]}})  # yapf: disable
_PERTURBED_RUN = _run({"r0": {**_req(), "logprobs": [[[11, -0.05], [12, -2.0], *_TAIL]]}})  # yapf: disable
_LATE_TRIGGER = {"degraded": True, "max_tokens": 12, "limit": 8}
# What run_ifb_trace.py writes when the engine does not start.
_NOT_STARTED = dict(label="reference",
                    startup_error="RuntimeError: no engine",
                    requests={})


def _policy(mode_source):
    """A run that expected mixed but resolved to prefillFirst, as on a deployment without compact kernels."""
    return dict(resolved="prefillFirst",
                expected="mixed",
                match=False,
                details=dict(mode="prefillFirst", mode_source=mode_source))


# yapf: disable
_CASES = {
    "identical": (_run(), _run(), {}),
    "near_tie": (_DIVERGED, _NEAR_TIE, {}),
    "wide_gap": (_DIVERGED, _run({"r0": _req(gap_at=2, gap=0.25)}), {"tie_rule": "FAIL"}),
    "not_a_reference_top2_token": (_run({"r0": _req([11, 12, 98, 14, 15])}), _NEAR_TIE, {"tie_rule": "FAIL"}),
    "perturbed_logit_fails_centered_cosine": (_PERTURBED_RUN, _PERTURBED_REF, {"first_token_cosine": "FAIL"}),
    "finished_not_cancelled": (_run({"r0": _req(), "c0": _req(expected_status="cancelled")}), _run(),
                               {"status_policy": "FAIL"}),
    "reference_request_missing": (_run({}), _run(), {"status_policy": "FAIL"}),
    "reference_did_not_start": (_run(), _NOT_STARTED, {"status_policy": "FAIL", "graph_hit_rate": "SKIP"}),
    "nothing_compared": (_run({}), _run({}), {"status_policy": "FAIL"}),
    "failure_kind_not_attached": (_run(failure_kind_probe={"error": "RequestFailedError: x", "kind": None}), _run(),
                                  {"status_policy": "FAIL"}),
    "requested_mode_resolved_otherwise": (_run(policy=_policy("requested")), _run(), {"status_policy": "FAIL"}),
    "auto_resolved_otherwise": (_run(policy=_policy("auto")), _run(), {}),
    "excluded_request_missing": (_run(requests_exclude=["r1"]), _run({"r0": _req(), "r1": _req()}), {}),
    "cache_reuse_issue": (_run(cache_hint_issues={"r0": "reuse 384 > max 0"}), _run(), {"cache_reuse": "FAIL"}),
    "admission_issue": (_run(admission_issues={"r0": "first token only after p0 finished"}), _run(),
                        {"admission": "FAIL"}),
    "engine_stats_issue": (_run(stats_issues={"engine": "paused_resident_steps 0 < min 1"}), _run(),
                           {"engine_stats": "FAIL"}),
    "late_trigger": (_run(trigger_lag=_LATE_TRIGGER), _run(), {"admission": "SKIP", "engine_stats": "SKIP"}),
    "graph_rate_below_reference": (_run(decode=40, graph_hits=39), _run(), {"graph_hit_rate": "FAIL"}),
    "graph_capture_disabled": (_run(graph_hits=0, engine_kwargs={"capture_cuda_graph": False}), _run(),
                               {"graph_hit_rate": "SKIP"}),
    "shorter_budget": (_run({"r0": _req(TOKENS[:3], max_generate_length=3)}), _run(), {}),
    "stopped_short_of_its_budget": (_run({"r0": _req(TOKENS[:3], max_generate_length=4)}), _run(),
                                    {"tie_rule": "FAIL"}),
    "diverged_within_its_budget": (_run({"r0": _req([11, 12, 99], max_generate_length=3)}), _run(),
                                   {"tie_rule": "FAIL"}),
}
# yapf: enable


@pytest.mark.parametrize("run, reference, verdicts",
                         _CASES.values(),
                         ids=_CASES.keys())
def test_checker_verdicts(tmp_path, monkeypatch, run, reference, verdicts):
    """Each criterion passes unless violated and any FAIL exits 1: divergence only to a reference top-2 token below
    the gap threshold, the centered first-token cosine (a perturbed logit the raw cosine accepts fails), expected
    statuses and resolved policy (a requested mode only; auto may resolve otherwise per deployment), a reference that
    did not start or a run that compared no request, a failed request raised without its RequestFailureKind, the
    driver's reuse, admission and engine-stats issues (n/a after a late trigger), and a CUDA-graph hit rate no lower
    than the reference's (n/a without capture)."""
    paths = []
    for name, value in (("run", run), ("reference", reference)):
        paths += [f"--{name}", str(tmp_path / f"{name}.json")]
        Path(paths[-1]).write_text(json.dumps({"runs": [value]}))
    out = tmp_path / "check.json"
    monkeypatch.setattr(sys, "argv", ["check", *paths, "--json-out", str(out)])
    code = checker.main()
    report = json.loads(out.read_text())["runs"]["run"]
    statuses = {k: v["status"] for k, v in report["criteria"].items()}
    assert statuses == dict.fromkeys(statuses,
                                     "PASS") | verdicts, report["failures"]
    assert code == int("FAIL" in verdicts.values())


def test_a_reference_that_did_not_start_is_reported_as_such():
    """A run checked against a reference whose engine did not start fails with the reference's startup error, not
    only as a run that compared no request."""
    failures, _, _, _ = checker.compare_run(_run(), _NOT_STARTED, None)
    assert any(
        "reference engine failed to start: RuntimeError: no engine" in failure
        for failure in failures), failures
