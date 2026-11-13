#!/usr/bin/env python3
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
"""Trace-driven driver for the in-flight-batching ``RequestEngine`` (nanobind).

``--mode run`` runs a trace (``tests/test_cases/ifb_*.json``) on one engine, every variant unless
``--variant``; ``--mode reference`` runs each request without a cancel rule alone on
``max_batch_size=1`` and ``ifb_scheduling_mode=singleCohort``. The result JSON records the engine
options, the resolved policy, the EngineMetrics delta, the trace expectations' issues and per-request
tokens, top-k logprobs, timestamps and ``RequestHandle.stats()``. tests/ifb/README.md holds the trace
schema and what each expectation checks.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCHEMA = "ifb-trace-result/1"
MAX_LOGPROBS = 20
BASE_ENGINE_KWARGS = ("max_batch_size", "max_pending_requests",
                      "command_queue_capacity", "capture_cuda_graph",
                      "context_cache_config", "multimodal_engine_dir")
REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEDULING_KWARGS = ("ifb_scheduling_mode", "max_num_tokens",
                     "max_chunk_tokens", "scheduling_policy", "preemption",
                     "priority_reserved_slots", "max_pause_steps")
MIXED_ONLY_KWARGS = ("max_num_tokens", "max_chunk_tokens", "preemption",
                     "max_pause_steps")
REQUEST_FIELDS = ("max_generate_length", "apply_chat_template",
                  "add_generation_prompt", "enable_thinking", "ignore_eos",
                  "priority", "skip_special_tokens")
POLL_SLEEP_S = 0.0002
DEFAULT_MAX_TRIGGER_LAG_TOKENS = 8
#: The stats_issues key of a run-level (EngineMetrics) expectation.
ENGINE_ISSUE_KEY = "engine"


def log(msg):
    print(f"[ifb-trace] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Runtime module
# ---------------------------------------------------------------------------
def load_runtime_module(nanobind_dir):
    """The nanobind runtime of the build under test, through the package's development-build loader."""
    sys.path.insert(0, str(REPO_ROOT))
    from tensorrt_edgellm._native.load import load_runtime_from_path

    candidates = [
        Path(path)
        for path in (nanobind_dir, os.environ.get("EDGELLM_NANOBIND_DIR"))
        if path
    ]
    if os.environ.get("BUILD_DIR"):
        candidates.append(Path(os.environ["BUILD_DIR"]) / "lib")
    for cand in candidates:
        so_files = sorted(cand.glob("_edgellm_runtime*.so"))
        if so_files:
            log(f"runtime module: {so_files[0]}")
            return load_runtime_from_path(so_files[0])
    raise SystemExit(
        "Cannot find _edgellm_runtime*.so; pass --nanobind-dir <build>/lib "
        f"(searched {[str(c) for c in candidates]})")


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------
def expand_content(content, snippets):
    if isinstance(content, str):
        return content
    parts = []
    for seg in content:
        if isinstance(seg, str):
            parts.append(seg)
        else:
            text = snippets[seg["snippet"]]
            parts.extend([text] * int(seg.get("repeat", 1)))
    return "\n\n".join(parts)


def is_image(seg):
    return isinstance(seg, dict) and "image" in seg


_IMAGES = {}


def load_image(rt, path):
    full = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    if full not in _IMAGES:
        _IMAGES[full] = rt.load_image_from_path(str(full))
    return _IMAGES[full]


def message_contents(rt, content, snippets, images):
    """MessageContent list of one message; the images it names are appended to ``images`` in order."""
    if isinstance(content, str) or not any(is_image(seg) for seg in content):
        return [rt.MessageContent("text", expand_content(content, snippets))]
    out, text = [], []
    for seg in content:
        if is_image(seg):
            if text:
                out.append(
                    rt.MessageContent("text", expand_content(text, snippets)))
                text = []
            out.append(rt.MessageContent("image", ""))
            images.append(load_image(rt, seg["image"]))
        else:
            text.append(seg)
    if text:
        out.append(rt.MessageContent("text", expand_content(text, snippets)))
    return out


def load_trace(path):
    with open(path) as f:
        trace = json.load(f)
    ids = [r["id"] for r in trace["requests"]]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: duplicate request ids")
    idset = set(ids)
    deps = {}
    for r in trace["requests"]:
        if "num_logprobs" in r:
            raise SystemExit(
                f"{path}: request {r['id']} sets num_logprobs; it is a cohort field, "
                "set it once at trace level")
        if ("messages" in r) == ("prompt" in r):
            raise SystemExit(
                f"{path}: request {r['id']} needs exactly one of messages/prompt"
            )
        arrival = r.get("arrival", "at_start")
        deps[r["id"]] = None
        if arrival != "at_start":
            src = arrival["after"]["request"]
            if src not in idset or src == r["id"]:
                raise SystemExit(
                    f"{path}: request {r['id']} arrival refers to {src}")
            deps[r["id"]] = src
        cancel = r.get("cancel_after")
        if cancel is not None:
            src = cancel.get("request", "self")
            if src != "self" and src not in idset:
                raise SystemExit(
                    f"{path}: request {r['id']} cancel refers to {src}")
    for rid in ids:
        seen, cur = set(), rid
        while deps[cur] is not None:
            if cur in seen:
                raise SystemExit(f"{path}: arrival cycle through {rid}")
            seen.add(cur)
            cur = deps[cur]
    trace.setdefault("name", Path(path).stem)
    nlp = int(trace.get("num_logprobs", MAX_LOGPROBS))
    trace["num_logprobs"] = max(2, min(MAX_LOGPROBS, nlp))
    return trace


def request_settings(trace, spec):
    merged = dict(trace.get("request_defaults", {}))
    merged.update({k: spec[k] for k in REQUEST_FIELDS if k in spec})
    if "prompt" in spec:
        merged.setdefault("apply_chat_template", False)
        merged.setdefault("add_generation_prompt", False)
    return merged


def expected_status(spec):
    exp = spec.get("expect", {}).get("status")
    if exp:
        return exp
    return "cancelled" if spec.get("cancel_after") is not None else "finished"


def build_request(rt, trace, spec, num_logprobs, channel, with_priority):
    snippets = trace.get("snippets", {})
    settings = request_settings(trace, spec)
    images = []
    if "messages" in spec:
        msgs = [
            rt.Message(m["role"],
                       message_contents(rt, m["content"], snippets, images))
            for m in spec["messages"]
        ]
    else:
        msgs = [
            rt.Message("user", [
                rt.MessageContent("text",
                                  expand_content(spec["prompt"], snippets))
            ])
        ]
    req = rt.LLMGenerationRequest()
    item = rt.Request(messages=msgs)
    if images:
        item.image_buffers = images
    req.requests = [item]
    if channel is not None:
        req.stream_channels = [channel]
    req.temperature = 1.0
    req.top_k = 1
    req.top_p = 1.0
    req.num_logprobs = num_logprobs
    req.max_generate_length = int(settings.get("max_generate_length", 64))
    req.apply_chat_template = bool(settings.get("apply_chat_template", True))
    req.add_generation_prompt = bool(
        settings.get("add_generation_prompt", True))
    req.enable_thinking = bool(settings.get("enable_thinking", False))
    req.ignore_eos = bool(settings.get("ignore_eos", False))
    if "skip_special_tokens" in settings:
        req.skip_special_tokens = bool(settings["skip_special_tokens"])
    if with_priority and "priority" in settings:
        req.priority = int(settings["priority"])
    return req


# ---------------------------------------------------------------------------
# Engine construction, metrics
# ---------------------------------------------------------------------------
def engine_build_config(engine_dir):
    try:
        with open(Path(engine_dir) / "config.json") as f:
            return json.load(f).get("builder_config", {})
    except (OSError, ValueError):
        return {}


def to_plain(value, depth=0):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [to_plain(v, depth + 1) for v in value]
    if isinstance(value, dict):
        return {str(k): to_plain(v, depth + 1) for k, v in value.items()}
    if hasattr(value, "name") and type(value).__name__ != "str" and hasattr(
            type(value), "__members__"):
        return value.name
    if depth >= 3:
        return str(value)
    out = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(value, name)
        except Exception:  # noqa: BLE001 - a broken property must not stop the dump
            continue
        if callable(attr):
            continue
        out[name] = to_plain(attr, depth + 1)
    return out if out else str(value)


def failure_kind_probe(rt, engine, max_input_len):
    """What get() raises for a request the engine fails at admission: a prompt
    longer than ``max_input_len`` tokens. The binding attaches the request's
    RequestFailureKind to RequestFailedError as ``kind``."""
    req = rt.LLMGenerationRequest()
    req.requests = [
        rt.Request(messages=[
            rt.Message(
                "user",
                [rt.MessageContent("text", "hello " * (2 * max_input_len))])
        ])
    ]
    req.max_generate_length = 1
    try:
        engine.submit(req).get()
    except Exception as exc:  # noqa: BLE001 - the failure is the observation
        kind = getattr(exc, "kind", None)
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "kind": getattr(kind, "name", None)
        }
    return {"error": None, "kind": None}


def snapshot_metrics(engine):
    try:
        return to_plain(engine.metrics())
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def metrics_delta(before, after):
    delta = {}
    for key, val in after.items():
        prev = before.get(key)
        if isinstance(val, (int, float)) and isinstance(
                prev, (int, float)) and not isinstance(val, bool):
            delta[key] = val - prev
    return delta


def policy_record(engine, expected):
    """The engine's resolved scheduling policy and whether its mode matches ``expected``."""
    details = to_plain(engine.resolved_policy)
    resolved = details["mode"]
    return {
        "resolved": resolved,
        "expected": expected,
        "match":
        None if expected is None else resolved.lower() == expected.lower(),
        "details": details,
    }


def binding_value(rt, key, val):
    """Turn a JSON engine option into the binding object it stands for."""
    if key == "context_cache_config" and isinstance(val, dict):
        config = rt.ContextCacheConfig()
        for name, field in val.items():
            setattr(config, name, field)
        return config
    return val


def make_engine(rt, engine_dir, kwargs):
    return rt.RequestEngine(
        engine_dir, **{
            k: binding_value(rt, k, v)
            for k, v in kwargs.items()
        })


# ---------------------------------------------------------------------------
# Request execution
# ---------------------------------------------------------------------------
class ReqState:

    def __init__(self, spec):
        self.spec = spec
        self.id = spec["id"]
        self.handle = None
        self.channel = None
        self.submitted = False
        self.submit_t = None
        self.done_t = None
        self.token_ids = []
        self.stream_logprobs = []
        self.token_times = []
        self.stream_reason = None
        self.prompt_token_count = None
        self.terminal = False
        self.status = "not_submitted"
        self.finish_reason = None
        self.error = None
        self.response_ids = None
        self.response_logprobs = None
        self.text = ""
        self.cancel_issued_at = None
        self.engine_stats = None
        self.arrival_lag = None
        self.cancel_lag = None

    def generated(self):
        return len(self.token_ids)


def logprob_entries(entries):
    return [[int(e.token_id), float(e.logprob)] for e in entries]


def drain_channel(state, now):
    got = False
    while True:
        chunk = state.channel.try_pop()
        if chunk is None:
            return got
        got = True
        ids = list(chunk.token_ids)
        state.token_ids.extend(ids)
        state.token_times.extend([now - state.submit_t] * len(ids))
        state.stream_logprobs.extend(
            logprob_entries(lp) for lp in chunk.logprobs)
        state.text += chunk.text
        if chunk.prompt_token_count >= 0:
            state.prompt_token_count = chunk.prompt_token_count
        if chunk.finished:
            state.stream_reason = chunk.reason.name


def finalize(rt, state, now):
    state.terminal = True
    state.done_t = now - state.submit_t
    try:
        resp = state.handle.get()
        if resp.output_ids:
            state.response_ids = list(resp.output_ids[0])
        if resp.logprobs:
            state.response_logprobs = [
                logprob_entries(step) for step in resp.logprobs[0]
            ]
        if resp.finish_reasons:
            state.finish_reason = resp.finish_reasons[0].name
        if resp.prompt_token_counts:
            state.prompt_token_count = int(resp.prompt_token_counts[0])
    except Exception as exc:  # noqa: BLE001 - get() raises on cancel/error
        state.error = f"{type(exc).__name__}: {exc}"
    reason = state.finish_reason or state.stream_reason
    if state.finish_reason is None:
        state.finish_reason = state.stream_reason
    if reason == "CANCELLED" or (state.error
                                 and "cancel" in state.error.lower()
                                 and reason in (None, "CANCELLED")):
        state.status = "cancelled"
    elif reason == "ERROR" or (state.error and reason is None):
        state.status = "failed"
    elif reason is not None:
        state.status = "finished"
    else:
        state.status = "failed"


def request_engine_stats(handle):
    """RequestHandle.stats() as a dict once the outcome is published, else None."""
    deadline = time.perf_counter() + 5.0
    while not handle.ready() and time.perf_counter() < deadline:
        time.sleep(POLL_SLEEP_S)
    stats = handle.stats()
    return None if stats is None else to_plain(stats)


def submit(rt, engine, trace, state, num_logprobs, with_priority, t0):
    state.channel = rt.StreamChannel.create()
    req = build_request(rt, trace, state.spec, num_logprobs, state.channel,
                        with_priority)
    state.submit_t = time.perf_counter()
    state.submit_rel = state.submit_t - t0
    state.submitted = True
    try:
        state.handle = engine.submit(req)
        state.status = "running"
    except Exception as exc:  # noqa: BLE001 - SubmitError and validation errors
        state.error = f"{type(exc).__name__}: {exc}"
        state.status = "refused"
        state.terminal = True
        state.done_t = 0.0


def trigger_ready(src_state, after):
    """A source request absent from this run (variant ``requests_exclude``) counts as satisfied."""
    if src_state is None:
        return True
    if not src_state.submitted:
        return False
    if after.get("event") == "finish":
        return src_state.terminal
    return src_state.generated() >= int(after.get("tokens",
                                                  0)) or src_state.terminal


def trigger_lag(src_state, after):
    """Tokens the source had produced past a token-count trigger when it fired; None for other triggers."""
    if src_state is None or after.get("event") == "finish":
        return None
    return src_state.generated() - int(after.get("tokens", 0))


def run_requests(rt, engine, trace, specs, num_logprobs, with_priority,
                 timeout_s, sequential):
    states = {s["id"]: ReqState(s) for s in specs}
    order = [s["id"] for s in specs]
    t0 = time.perf_counter()
    deadline = t0 + timeout_s
    timed_out = False

    def step_once(active):
        progressed = False
        now = time.perf_counter()
        for st in active:
            if st.terminal or st.handle is None:
                continue
            if drain_channel(st, now):
                progressed = True
            if st.stream_reason is not None or st.handle.ready():
                drain_channel(st, time.perf_counter())
                finalize(rt, st, time.perf_counter())
                progressed = True
        return progressed

    if sequential:
        for rid in order:
            st = states[rid]
            submit(rt, engine, trace, st, num_logprobs, with_priority, t0)
            while not st.terminal:
                if time.perf_counter() > deadline:
                    timed_out = True
                    break
                if not step_once([st]):
                    time.sleep(POLL_SLEEP_S)
            if timed_out:
                break
    else:
        while True:
            for rid in order:
                st = states[rid]
                if st.submitted:
                    continue
                arrival = st.spec.get("arrival", "at_start")
                if arrival == "at_start":
                    submit(rt, engine, trace, st, num_logprobs, with_priority,
                           t0)
                    continue
                src_state = states.get(arrival["after"]["request"])
                if trigger_ready(src_state, arrival["after"]):
                    st.arrival_lag = trigger_lag(src_state, arrival["after"])
                    submit(rt, engine, trace, st, num_logprobs, with_priority,
                           t0)
            for rid in order:
                st = states[rid]
                cancel = st.spec.get("cancel_after")
                if cancel is None or not st.submitted or st.terminal or st.cancel_issued_at is not None:
                    continue
                src = cancel.get("request", "self")
                src_state = st if src == "self" else states.get(src)
                if trigger_ready(src_state, cancel):
                    st.cancel_lag = trigger_lag(src_state, cancel)
                    st.cancel_issued_at = st.generated()
                    st.handle.cancel()
            submitted = [states[r] for r in order if states[r].submitted]
            progressed = step_once(submitted)
            if all(states[r].terminal for r in order):
                break
            if time.perf_counter() > deadline:
                timed_out = True
                break
            if not progressed:
                time.sleep(POLL_SLEEP_S)
    if timed_out:
        for st in states.values():
            if st.submitted and not st.terminal and st.handle is not None:
                st.handle.cancel()
                st.status = "timeout"
                st.terminal = True
    wall = time.perf_counter() - t0
    # Collected after the run so a request waiting for its outcome does not delay the polling of others.
    for st in states.values():
        if st.handle is not None and st.status != "timeout":
            st.engine_stats = request_engine_stats(st.handle)
    return states, wall, timed_out


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def request_record(st):
    ids = st.response_ids if st.response_ids is not None else st.token_ids
    lps = st.response_logprobs if st.response_logprobs else st.stream_logprobs
    return {
        "status":
        st.status,
        "expected_status":
        expected_status(st.spec),
        "finish_reason":
        st.finish_reason,
        "error":
        st.error,
        "prompt_tokens":
        st.prompt_token_count,
        "num_generated":
        len(ids),
        "token_ids":
        ids,
        "logprobs":
        lps,
        "text":
        st.text,
        "submit_time_s":
        getattr(st, "submit_rel", None),
        "token_times_s":
        st.token_times,
        "done_time_s":
        st.done_t,
        "cancel_issued_at_tokens":
        st.cancel_issued_at,
        "trigger_lag_tokens": {
            "arrival": st.arrival_lag,
            "cancel": st.cancel_lag
        },
        "engine_stats":
        st.engine_stats,
        "stream_matches_response": (st.response_ids is None
                                    or st.response_ids == st.token_ids),
    }


def step_summary(delta, after):
    """Step mix, sizes, CUDA-graph hits and host step time from an EngineMetrics delta."""
    if "decode_steps" not in delta:
        return {}
    decode = delta.get("decode_steps", 0)
    prompt = delta.get("prompt_steps", 0)
    subset = delta.get("decode_subset_steps", 0)
    steps = decode + prompt + subset

    def avg(key):
        return round(delta.get(key, 0) / steps, 2) if steps else None

    return {
        "steps":
        steps,
        "decode_steps":
        decode,
        "prompt_steps":
        prompt,
        "mixed_steps":
        delta.get("mixed_steps", 0),
        "decode_subset_steps":
        subset,
        "prompt_step_share":
        round(prompt / steps, 4) if steps else None,
        "decode_graph_hit_rate":
        (round(delta.get("decode_graph_steps", 0) /
               decode, 4) if decode else None),
        "partial_prompt_chunks":
        delta.get("partial_prompt_chunks", 0),
        "avg_N":
        avg("step_sequences"),
        "avg_Np":
        avg("step_prompt_chunks"),
        "avg_Nd":
        avg("step_decode_rows"),
        "avg_T":
        avg("step_tokens"),
        "avg_L":
        avg("step_sampled_rows"),
        "host_step_ms_avg":
        (round(delta.get("step_time_total_us", 0) / steps /
               1000.0, 4) if steps else None),
        "host_step_ms_max_engine_lifetime":
        round(after.get("step_time_max_us", 0) / 1000.0, 3),
        "paused_resident_steps":
        delta.get("paused_resident_steps", 0),
    }


def request_stats_summary(records):
    """Per-request engine-side stats: paused time, prompt chunks, reuse, engine TTFT."""
    out = {}
    for rid, rec in records.items():
        st = rec.get("engine_stats")
        if not st or not st.get("admitted"):
            continue
        out[rid] = {
            "queue_ms": round(st["queue_time_us"] / 1000.0, 3),
            "ttft_ms": round(st["time_to_first_token_us"] / 1000.0, 3),
            "prompt_chunks": st["prompt_chunks"],
            "reused": st["reused_prompt_tokens"],
            "paused_steps": st["paused_steps"],
            "paused_ms": round(st["paused_time_us"] / 1000.0, 3),
        }
    return out


def engine_is_hybrid(engine_dir):
    """Whether the engine has recurrent (linear-attention) layers, whose context cache hits only at snapshots."""
    try:
        with open(Path(engine_dir) / "config.json") as f:
            return int(json.load(f).get("num_linear_attn_layers", 0) or 0) > 0
    except (OSError, ValueError, TypeError):
        return False


def reuse_expectation_issue(spec, records, hybrid):
    """Why a finished request's reuse breaks its expect.reuse_tokens bounds, or None.

    Returns None as well when the bounds do not apply (see the module docstring).
    """
    want = spec.get("expect", {}).get("reuse_tokens")
    rec = records.get(spec["id"])
    if not want or rec is None or rec["status"] != "finished":
        return None
    bounds = {key: want[key] for key in ("min", "max") if key in want}
    if hybrid:
        bounds.update(want.get("hybrid", {}))
    condition = want.get("if_prompt_cancelled")
    if condition is not None:
        source = records.get(condition)
        stats = (source or {}).get("engine_stats")
        if source is None or source["status"] != "cancelled":
            log(f"reuse bounds of {spec['id']} skipped: {condition} " +
                ("did not run" if source is None else "was not cancelled"))
            return None
        if stats is None:
            log(f"reuse bounds of {spec['id']} skipped: no engine stats tell whether "
                f"{condition}'s prompt completed")
            return None
        if stats.get("time_to_first_token_us", 0) > 0:
            log(f"reuse bounds of {spec['id']} skipped: {condition}'s prompt completed "
                "before its cancel landed")
            return None
    low, high = int(bounds.get("min", 0)), bounds.get("max")
    if low <= 0 and high is None:
        return None
    got = rec.get("cache_reuse_tokens")
    if got is None:
        return "reuse not observed"
    if got < low:
        return f"reuse {got} < min {low}"
    if high is not None and got > int(high):
        return f"reuse {got} > max {int(high)}"
    return None


def attribute_cache_reuse(records, specs, hybrid=False):
    """Attach RequestHandle.stats().reused_prompt_tokens to each admitted request; return the reuse issues."""
    order = sorted(
        (rid
         for rid, rec in records.items() if rec["submit_time_s"] is not None),
        key=lambda rid: records[rid]["submit_time_s"])
    for rid in order:
        rec = records[rid]
        stats = rec.get("engine_stats")
        rec["cache_reuse_tokens"] = (stats["reused_prompt_tokens"] if stats
                                     and stats.get("admitted") else None)
    issues = {}
    for spec in specs:
        issue = reuse_expectation_issue(spec, records, hybrid)
        if issue is not None:
            issues[spec["id"]] = issue
    if any(records[rid].get("cache_reuse_tokens") for rid in order):
        log("context cache reuse: " + ", ".join(
            f"{rid}={records[rid].get('cache_reuse_tokens')}/{records[rid]['prompt_tokens']}"
            for rid in order))
    for rid, issue in issues.items():
        log(f"cache reuse expectation violated by {rid}: {issue}")
    return issues


def apply_request_overrides(specs, overrides):
    """Apply a variant's per-request field overrides; an "expect" mapping is merged into the request's."""
    out = []
    for spec in specs:
        override = overrides.get(spec["id"])
        if override:
            spec = dict(spec)
            for key, value in override.items():
                if key == "expect":
                    spec["expect"] = {**spec.get("expect", {}), **value}
                else:
                    spec[key] = value
        out.append(spec)
    return out


def admission_expectation_issues(specs, records):
    """Per request id, why a request with expect.joins_running did not start beside those requests.

    The request's first token must arrive before each listed request finished: a request that waited for
    the listed ones to leave (a stall at admission) produces its first token only after them.
    """
    issues = {}
    for spec in specs:
        running = spec.get("expect", {}).get("joins_running")
        rec = records.get(spec["id"])
        if not running or rec is None:
            continue
        if (rec["status"] != "finished" or not rec["token_times_s"]
                or rec["submit_time_s"] is None):
            issues[spec["id"]] = "produced no token"
            continue
        first_token = rec["submit_time_s"] + rec["token_times_s"][0]
        finished_before = []
        for other in running:
            other_rec = records.get(other)
            if other_rec is None:
                log(f"joins_running of {spec['id']}: {other} did not run; skipped"
                    )
                continue
            if other_rec["submit_time_s"] is None or other_rec[
                    "done_time_s"] is None:
                continue
            if other_rec["submit_time_s"] + other_rec[
                    "done_time_s"] <= first_token:
                finished_before.append(other)
        if finished_before:
            issues[spec["id"]] = ("first token only after " +
                                  ", ".join(finished_before) + " finished")
    for rid, issue in issues.items():
        log(f"admission expectation violated by {rid}: {issue}")
    return issues


def bound_issue(name, value, bounds):
    """Why ``value`` breaks ``bounds`` ({"min": a, "max": b}, either optional), or None."""
    if value is None:
        return f"{name} not observed"
    if "min" in bounds and value < bounds["min"]:
        return f"{name} {value} < min {bounds['min']}"
    if "max" in bounds and value > bounds["max"]:
        return f"{name} {value} > max {bounds['max']}"
    return None


def admitted_at(rec):
    """When the engine admitted a request, in seconds from the run's start, or None if it never did."""
    stats = rec.get("engine_stats")
    if not stats or not stats.get("admitted") or rec.get(
            "submit_time_s") is None:
        return None
    return rec["submit_time_s"] + stats["queue_time_us"] / 1e6


def stats_expectation_issues(specs, records, delta, expect_metrics):
    """Per request id, and ENGINE_ISSUE_KEY for ``expect_metrics`` over the EngineMetrics ``delta``, why the
    engine's metrics or a request's stats break the trace's expectations."""
    issues = {}
    engine = [
        bound_issue(name, delta.get(name), bounds)
        for name, bounds in sorted((expect_metrics or {}).items())
    ]
    engine = [issue for issue in engine if issue]
    if engine:
        issues[ENGINE_ISSUE_KEY] = "; ".join(engine)
    for spec in specs:
        expect = spec.get("expect", {})
        wanted, before = expect.get("stats") or {}, expect.get(
            "admitted_before") or []
        rec = records.get(spec["id"])
        if rec is None or not (wanted or before):
            continue
        stats = rec.get("engine_stats") or {}
        found = [
            bound_issue(name, stats.get(name), bounds)
            for name, bounds in sorted(wanted.items())
        ]
        mine = admitted_at(rec)
        for other in before:
            other_rec = records.get(other)
            if other_rec is None:
                log(f"admitted_before of {spec['id']}: {other} did not run; skipped"
                    )
                continue
            theirs = admitted_at(other_rec)
            if mine is None:
                found.append("not admitted")
                break
            if theirs is not None and theirs <= mine:
                found.append(
                    f"admitted {(mine - theirs) * 1000.0:.1f} ms after {other}"
                )
        found = [issue for issue in found if issue]
        if found:
            issues[spec["id"]] = "; ".join(found)
    for rid, issue in issues.items():
        log(f"engine stats expectation violated by {rid}: {issue}")
    return issues


def trigger_lag_summary(records, limit):
    """The run's largest trigger lag and whether it exceeds ``limit`` tokens."""
    by_request = {}
    for rid, rec in records.items():
        lags = [
            lag for lag in (rec.get("trigger_lag_tokens") or {}).values()
            if lag is not None
        ]
        if lags:
            by_request[rid] = max(lags)
    largest = max(by_request.values(), default=0)
    return {
        "max_tokens": largest,
        "limit": limit,
        "degraded": largest > limit,
        "by_request": by_request
    }


def timing_expectation_issues(specs, records, policy, delta, expect_metrics,
                              lag):
    """(admission_issues, stats_issues) of a run. Both are skipped when a trigger fired late (``lag``), and the
    stats ones also when the resolved policy is not the expected one."""
    if lag["degraded"]:
        log(f"a trigger fired {lag['max_tokens']} tokens late (limit {lag['limit']}): joins_running, "
            "expect_metrics, expect.stats and admitted_before are skipped")
        return {}, {}
    admission = admission_expectation_issues(specs, records)
    if policy.get("match") is False:
        log(f"resolved policy {policy.get('resolved')} != expected {policy.get('expected')}: "
            "engine stats expectations skipped")
        return admission, {}
    return admission, stats_expectation_issues(specs, records, delta,
                                               expect_metrics)


def check_prompt_hints(spec, prompt_tokens, max_input_len):
    issues = []
    hint = spec.get("expect", {}).get("prompt_tokens", {})
    if prompt_tokens is None:
        return issues
    if max_input_len and prompt_tokens > max_input_len:
        issues.append(
            f"{prompt_tokens} > engine max_input_len {max_input_len}")
    if "min" in hint and prompt_tokens < hint["min"]:
        issues.append(f"{prompt_tokens} < min {hint['min']}")
    if "max" in hint and prompt_tokens > hint["max"]:
        issues.append(f"{prompt_tokens} > max {hint['max']}")
    nm = hint.get("not_multiple_of")
    if nm and prompt_tokens % nm == 0:
        issues.append(f"{prompt_tokens} is a multiple of {nm}")
    return issues


def count_prompts(rt, engine, trace, specs, num_logprobs):
    counts = {}
    for spec in specs:
        req = build_request(rt, trace, spec, num_logprobs, None, False)
        try:
            counts[spec["id"]] = int(engine.count_prompt_tokens(req)[0])
        except Exception as exc:  # noqa: BLE001
            counts[spec["id"]] = None
            log(f"count_prompt_tokens({spec['id']}) failed: {exc}")
    return counts


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def engine_kwargs_for(mode, trace, variant, args, build_cfg):
    kw = {}
    for key in BASE_ENGINE_KWARGS + SCHEDULING_KWARGS:
        if key in trace.get("engine", {}):
            kw[key] = trace["engine"][key]
    if variant:
        kw.update(variant.get("engine", {}))
    if args.multimodal_engine_dir:
        kw["multimodal_engine_dir"] = args.multimodal_engine_dir
    if args.scheduling_mode and mode == "run":
        kw["ifb_scheduling_mode"] = args.scheduling_mode
    if mode == "reference":
        kw["max_batch_size"] = 1
        kw["ifb_scheduling_mode"] = "singleCohort"
        for key in MIXED_ONLY_KWARGS + ("scheduling_policy",
                                        "priority_reserved_slots"):
            kw.pop(key, None)
    built_b = build_cfg.get("max_batch_size")
    kw.setdefault("max_batch_size", built_b or 1)
    if built_b and kw["max_batch_size"] > built_b:
        log(f"max_batch_size {kw['max_batch_size']} > engine build B={built_b}; clamping"
            )
        kw["max_batch_size"] = built_b
    return kw


def run_one(rt, args, trace, mode, variant):
    build_cfg = engine_build_config(args.engine_dir)
    kw = engine_kwargs_for(mode, trace, variant, args, build_cfg)
    num_logprobs = trace["num_logprobs"]
    specs = trace["requests"]
    if variant and variant.get("requests_exclude"):
        specs = [
            s for s in specs if s["id"] not in variant["requests_exclude"]
        ]
    if variant and variant.get("request_overrides"):
        specs = apply_request_overrides(specs, variant["request_overrides"])
    if mode == "reference":
        specs = [s for s in specs if s.get("cancel_after") is None]
    label = f"{mode}" + (f"/{variant['name']}" if variant else "")
    log(f"[{label}] engine kwargs: {kw}")
    t_build = time.perf_counter()
    try:
        engine = make_engine(rt, args.engine_dir, kw)
    except Exception as exc:  # noqa: BLE001 - startup validation failures are a test outcome
        log(f"[{label}] engine construction failed: {exc}")
        return {
            "label": label,
            "variant": variant["name"] if variant else None,
            "engine_kwargs": kw,
            "startup_error": f"{type(exc).__name__}: {exc}",
            "requests": {},
        }
    log(f"[{label}] engine ready in {time.perf_counter() - t_build:.1f}s")
    if mode == "run":
        expected_policy = args.expect_policy or (
            variant or {}).get("expect_policy") or trace.get("expect_policy")
    else:
        expected_policy = "singleCohort"
    policy = policy_record(engine, expected_policy)
    with_priority = mode != "reference"
    prompt_counts = count_prompts(rt, engine, trace, specs, num_logprobs)
    hint_issues = {}
    for spec in specs:
        issues = check_prompt_hints(spec, prompt_counts.get(spec["id"]),
                                    build_cfg.get("max_input_len"))
        if issues:
            hint_issues[spec["id"]] = issues
            log(f"[{label}] prompt hint violation {spec['id']}: {issues}")
    metrics_before = snapshot_metrics(engine)
    states, wall, timed_out = run_requests(rt,
                                           engine,
                                           trace,
                                           specs,
                                           num_logprobs,
                                           with_priority,
                                           args.timeout,
                                           sequential=(mode == "reference"))
    metrics_after = snapshot_metrics(engine)
    probe = (failure_kind_probe(rt, engine, build_cfg["max_input_len"])
             if mode == "run" else None)
    try:
        engine.shutdown()
    except Exception as exc:  # noqa: BLE001
        log(f"shutdown raised: {exc}")
    del engine
    records = {rid: request_record(st) for rid, st in states.items()}
    for rid, rec in records.items():
        if rec["prompt_tokens"] is None:
            rec["prompt_tokens"] = prompt_counts.get(rid)
    cache_issues = attribute_cache_reuse(records, specs,
                                         engine_is_hybrid(args.engine_dir))
    for spec in specs:
        if spec["id"] in records:
            records[spec["id"]]["max_generate_length"] = int(
                request_settings(trace, spec).get("max_generate_length", 64))
    lag = trigger_lag_summary(records, args.max_trigger_lag)
    run_delta = metrics_delta(metrics_before, metrics_after)
    expect_metrics = (variant or {}).get("expect_metrics",
                                         trace.get("expect_metrics"))
    admission_issues, stats_issues = timing_expectation_issues(
        specs, records, policy, run_delta, expect_metrics, lag)
    run = {
        "label": label,
        "variant": variant["name"] if variant else None,
        "requests_exclude": list((variant or {}).get("requests_exclude", [])),
        "startup_error": None,
        "engine_kwargs": kw,
        "num_logprobs": num_logprobs,
        "policy": policy,
        "prompt_hint_issues": hint_issues,
        "cache_hint_issues": cache_issues,
        "admission_issues": admission_issues,
        "stats_issues": stats_issues,
        "trigger_lag": lag,
        "timed_out": timed_out,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "metrics_delta": run_delta,
        "failure_kind_probe": probe,
        "requests": records,
        "summary": {
            "wall_s": round(wall, 4),
            "steps": step_summary(run_delta, metrics_after),
            "request_stats": request_stats_summary(records),
        },
    }
    print_run(run)
    return run


def print_run(run):
    log(f"== {run['label']}: policy resolved={run['policy']['resolved']} "
        f"expected={run['policy']['expected']} match={run['policy']['match']} "
        f"trigger lag max={run['trigger_lag']['max_tokens']} tokens "
        f"(limit {run['trigger_lag']['limit']})")
    print(f"{'id':<20}{'status':<11}{'reason':<10}{'prompt':>7}{'gen':>5}")
    for rid, rec in run["requests"].items():
        print(f"{rid:<20}{rec['status']:<11}{str(rec['finish_reason']):<10}"
              f"{str(rec['prompt_tokens']):>7}{rec['num_generated']:>5}")
    s = run["summary"]
    print(f"wall={s['wall_s']}s")
    steps = s.get("steps") or {}
    if steps:
        print(
            f"steps={steps['steps']} decode={steps['decode_steps']} prompt={steps['prompt_steps']} "
            f"(mixed={steps['mixed_steps']}) decode_subset={steps['decode_subset_steps']} "
            f"prompt_share={steps['prompt_step_share']} graph_hit={steps['decode_graph_hit_rate']} "
            f"avg N/Np/Nd/T/L={steps['avg_N']}/{steps['avg_Np']}/{steps['avg_Nd']}/{steps['avg_T']}/"
            f"{steps['avg_L']} host_step_ms avg={steps['host_step_ms_avg']} "
            f"max={steps['host_step_ms_max_engine_lifetime']}")
    paused = {
        rid: v["paused_ms"]
        for rid, v in (s.get("request_stats") or {}).items()
        if v["paused_steps"]
    }
    if paused:
        print(f"paused_ms per request: {paused}")
    print(f"EngineMetrics delta: {run['metrics_delta']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", required=True)
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--mode", choices=["run", "reference"], default="run")
    ap.add_argument("--out", help="output JSON path")
    ap.add_argument("--nanobind-dir",
                    help="directory holding _edgellm_runtime*.so")
    ap.add_argument(
        "--variant",
        help="comma-separated variants to run (run mode); default all")
    ap.add_argument("--scheduling-mode",
                    choices=["auto", "mixed", "prefillFirst", "singleCohort"],
                    help="override ifb_scheduling_mode (run mode)")
    ap.add_argument("--multimodal-engine-dir",
                    help="encoder engines for traces with images")
    ap.add_argument("--expect-policy", help="override the trace expect_policy")
    ap.add_argument("--timeout",
                    type=float,
                    default=900.0,
                    help="per-run timeout in seconds")
    ap.add_argument(
        "--max-trigger-lag",
        type=int,
        default=DEFAULT_MAX_TRIGGER_LAG_TOKENS,
        help="tokens a token-count trigger may fire late before the run's "
        "timing-dependent expectations are skipped instead of evaluated")
    args = ap.parse_args()

    trace = load_trace(args.trace)
    rt = load_runtime_module(args.nanobind_dir)
    result = {
        "schema": SCHEMA,
        "mode": args.mode,
        "trace": str(Path(args.trace).resolve()),
        "trace_name": trace["name"],
        "engine_dir": str(Path(args.engine_dir).resolve()),
        "runs": [],
    }

    variants = [None]
    if args.mode == "run" and trace.get("variants"):
        variants = list(trace["variants"])
        if args.variant:
            wanted = args.variant.split(",")
            variants = [v for v in variants if v["name"] in wanted]
            if not variants:
                raise SystemExit(f"no variant {args.variant}")
    elif args.variant:
        log("--variant only applies to run mode on traces with variants; ignored"
            )

    for variant in variants:
        result["runs"].append(run_one(rt, args, trace, args.mode, variant))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
        log(f"wrote {args.out}")
    bad = [
        f"{run['label']}:startup" for run in result["runs"]
        if run.get("startup_error")
    ]
    bad += [
        f"{run['label']}:{rid}" for run in result["runs"]
        for rid, rec in run["requests"].items()
        if rec["status"] in ("failed", "refused", "timeout", "not_submitted")
    ]
    if bad:
        log(f"requests that did not complete: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
