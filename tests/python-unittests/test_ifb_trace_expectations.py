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
"""Engine-stats and trigger-lag expectations of tests/ifb/run_ifb_trace.py on synthetic request records."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "run_ifb_trace",
    Path(__file__).resolve().parents[2] / "tests" / "ifb" / "run_ifb_trace.py")
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


def _timed(submit, first_token, done, status="finished"):
    """A request record with its submission, first-token and completion times in seconds."""
    tokens = [first_token - submit] if first_token is not None else []
    return dict(status=status, submit_time_s=submit, token_times_s=tokens, done_time_s=done - submit)  # yapf: disable


_P0 = _timed(0.0, 0.1, 4.0)


def _stats(paused=0, chunks=1, admitted_s=1.0):
    """A request record whose engine stats report pauses, prompt chunks and an admission time (None: never)."""
    stats = dict(admitted=admitted_s is not None, queue_time_us=int((admitted_s or 0.0) * 1e6), paused_steps=paused,
                 prompt_chunks=chunks)  # yapf: disable
    return dict(status="finished", submit_time_s=0.0, engine_stats=stats)


_STATS_SPECS = [{"id": "h0", "expect": {"stats": {"paused_steps": {"max": 0}, "prompt_chunks": {"max": 1}},
                                        "admitted_before": ["p2"]}}]  # yapf: disable
_PAUSE_MIN = {"paused_resident_steps": {"min": 1}}

# yapf: disable
_STATS_CASES = {
    "every_expectation_met_at_its_bound": (1, _stats(), _stats(admitted_s=2.0), {}),
    "nothing_paused": (0, _stats(), _stats(admitted_s=2.0), {"engine": "paused_resident_steps 0 < min 1"}),
    "urgent_request_paused": (1, _stats(paused=1), _stats(admitted_s=2.0), {"h0": "paused_steps 1 > max 0"}),
    "admitted_after_the_other": (1, _stats(admitted_s=2.5), _stats(admitted_s=2.0),
                                 {"h0": "admitted 500.0 ms after p2"}),
    "admitted_with_the_other": (1, _stats(admitted_s=2.0), _stats(admitted_s=2.0), {"h0": "admitted 0.0 ms after p2"}),
    "other_never_admitted": (1, _stats(), _stats(admitted_s=None), {}),
    "never_admitted": (1, _stats(chunks=0, admitted_s=None), _stats(admitted_s=2.0), {"h0": "not admitted"}),
    "no_engine_stats": (1, {**_stats(), "engine_stats": None}, _stats(admitted_s=2.0),
                        {"h0": "paused_steps not observed; prompt_chunks not observed; not admitted"}),
}
# yapf: enable


@pytest.mark.parametrize("paused_resident_steps, h0, p2, issues",
                         _STATS_CASES.values(),
                         ids=_STATS_CASES.keys())
def test_engine_stats_expectations(paused_resident_steps, h0, p2, issues):
    """expect_metrics bounds the run's EngineMetrics delta, expect.stats a request's engine stats (a missing stat is
    an issue) and expect.admitted_before requires the engine to admit the request strictly earlier than each listed
    request that it admitted at all; a bound is met at its value."""
    delta = {"paused_resident_steps": paused_resident_steps}
    records = {"h0": h0, "p2": p2}
    assert driver.stats_expectation_issues(_STATS_SPECS, records, delta,
                                           _PAUSE_MIN) == issues


_JOINS = [{"id": "h0", "expect": {"joins_running": ["p0"]}}]
_LATE_JOIN = {"h0": "first token only after p0 finished"}


# yapf: disable
@pytest.mark.parametrize("produced, policy_match, admission, stats", [
    (13, True, _LATE_JOIN, {"engine": "paused_resident_steps 0 < min 1"}),
    (14, True, {}, {}),
    (13, False, _LATE_JOIN, {}),
], ids=["lag_at_the_limit", "lag_over_the_limit", "policy_mismatch"])
# yapf: enable
def test_timing_expectations_skip_after_a_late_trigger(produced, policy_match,
                                                       admission, stats):
    """A token-count trigger's lag is its source's tokens past the count (a finish trigger has none; a run without
    token triggers has lag 0). When the run's largest arrival or cancel lag exceeds the limit, neither joins_running
    nor the engine-stats expectations are evaluated; a resolved policy other than the expected one skips only the
    engine-stats ones."""
    source = driver.ReqState({"id": "p0"})
    source.submitted, source.token_ids = True, [1] * produced
    finish = driver.trigger_lag(source, {"request": "p0", "event": "finish"})
    late = driver.trigger_lag(source, {"request": "p0", "tokens": 5})
    lag = driver.trigger_lag_summary({"a": {"trigger_lag_tokens": {"arrival": 3, "cancel": None}},
                                      "b": {"trigger_lag_tokens": {"arrival": finish, "cancel": late}}}, 8)  # yapf: disable
    assert driver.trigger_lag_summary({}, 8)["max_tokens"] == 0
    records = {"h0": _timed(1.0, 5.0, 6.0), "p0": _P0}
    assert driver.timing_expectation_issues(_JOINS, records,
                                            {"match": policy_match},
                                            {"paused_resident_steps": 0},
                                            _PAUSE_MIN,
                                            lag) == (admission, stats)
