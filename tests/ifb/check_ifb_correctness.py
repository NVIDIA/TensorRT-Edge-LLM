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
"""Compare an IFB trace run against the batch-1 reference.

Inputs are ``run_ifb_trace.py`` outputs: ``--run`` (mode run, every run in it is checked) and
``--reference`` (mode reference).

Per request present in the reference (requests with a cancel rule are not in the reference):
  (a) greedy tokens identical, or the first divergence is at a position where the reference top-2
      logprob gap is < --gap-threshold (logprob gap == logit gap) and the run's token there is one of
      the reference's top two; comparison stops there.
  (b) first-token cosine >= --cosine-threshold. The runtime exposes only top-k logprobs, not full
      logits, so this is an APPROXIMATION: cosine over the two top-k logprob vectors of the first
      generated token, aligned on the union of token ids, an id missing from one vector taking
      that vector's minimum logprob, and mean-removed (centered top-k logits). The raw logprob
      cosine is reported for information; its shared log-sum-exp offset makes it about ten times less
      sensitive to a perturbed logit.
  (c) drift (number of diverging requests, earliest divergence position, and per request the largest
      |logprob difference| up to its first divergence). Reported, not gated (see the README).
  (d) with CUDA graph capture enabled, the run's pure-decode full steps replay a graph at least as
      often as the reference's. Skipped when capture is disabled or the run or the reference has no
      pure-decode full step.
  (e) every finished request's context-cache reuse lies within its trace ``expect.reuse_tokens``
      bounds, as the driver evaluated them on this engine (the run's ``cache_hint_issues``).
  (f) every request with a trace ``expect.joins_running`` produced its first token before each listed
      request finished (the run's ``admission_issues``).
  (g) the engine's step metrics and each request's engine stats meet the trace's ``expect_metrics``,
      ``expect.stats`` and ``expect.admitted_before`` (the run's ``stats_issues``).
(f) and (g) depend on when the driver submitted requests; when its trigger lag exceeded its limit
(the run's ``trigger_lag.degraded``), the driver does not evaluate them and they are reported as SKIP.
A request whose run budget (``max_generate_length``, e.g. from a variant's ``request_overrides``) is
shorter than its reference output is compared over the tokens it generated: stopping at its own budget
is not a divergence.
Also: every request must end with its expected status (cancel rule -> CANCELLED, otherwise
finished), the resolved policy must match the expectation when both are known, the run's and the
reference's engines must have started, at least one request must have been compared, and the run's
``failure_kind_probe`` (a prompt longer than the engine's input limit, submitted after the trace)
must have raised RequestFailedError with RequestFailureKind OTHER. Exit code 1 on any failure of
(a), (b), (d), (e), (f), (g) or these checks.
"""

import argparse
import json
import math
import sys


def load(path):
    with open(path) as f:
        data = json.load(f)
    if not data.get("runs"):
        raise SystemExit(f"{path}: no runs")
    return data


def first_divergence(run_ids, ref_ids):
    for i, (a, b) in enumerate(zip(run_ids, ref_ids)):
        if a != b:
            return i
    if len(run_ids) != len(ref_ids):
        return min(len(run_ids), len(ref_ids))
    return None


def top2_gap(step):
    if step is None or len(step) < 2:
        return None
    vals = sorted((lp for _, lp in step), reverse=True)
    return vals[0] - vals[1]


def topk_cosine(a, b, centered=False):
    """Cosine of two top-k logprob vectors aligned on the union of token ids.

    Uncentered, the shared log-sum-exp offset dominates both vectors, so even swapping the top-2
    tokens stays above 0.999. Centering removes that per-vector constant, which makes the value the
    cosine of the centered top-k logits.
    """
    if not a or not b:
        return None
    da = {int(t): float(v) for t, v in a}
    db = {int(t): float(v) for t, v in b}
    fill_a, fill_b = min(da.values()), min(db.values())
    ids = sorted(set(da) | set(db))
    va = [da.get(t, fill_a) for t in ids]
    vb = [db.get(t, fill_b) for t in ids]
    if centered:
        mean_a, mean_b = sum(va) / len(va), sum(vb) / len(vb)
        va = [x - mean_a for x in va]
        vb = [x - mean_b for x in vb]
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    if na == 0 or nb == 0:
        return None
    return sum(x * y for x, y in zip(va, vb)) / (na * nb)


def max_common_abs_diff(a, b):
    if not a or not b:
        return None
    da = {int(t): float(v) for t, v in a}
    db = {int(t): float(v) for t, v in b}
    common = set(da) & set(db)
    if not common:
        return None
    return max(abs(da[t] - db[t]) for t in common)


def compare_run(run, ref_run, args):
    failures, warnings, per_request = [], [], {}
    label = run.get("label", "?")
    if run.get("startup_error"):
        failures.append(
            f"{label}: engine failed to start: {run['startup_error']}")
        return failures, warnings, per_request, None
    if ref_run.get("startup_error"):
        failures.append(
            f"{label}: reference engine failed to start: {ref_run['startup_error']}"
        )
        return failures, warnings, per_request, None
    probe = run.get("failure_kind_probe") or {}
    if probe.get("kind") != "OTHER":
        failures.append(
            f"{label}: a request the engine failed at admission did not raise RequestFailedError "
            f"with RequestFailureKind OTHER: {probe}")
    ref_reqs = ref_run["requests"]
    for rid, rec in run["requests"].items():
        expected = rec.get("expected_status", "finished")
        if rec["status"] != expected:
            failures.append(
                f"{label}:{rid}: status {rec['status']} (reason {rec.get('finish_reason')}, "
                f"error {rec.get('error')}) != expected {expected}")
        if not rec.get("stream_matches_response", True):
            warnings.append(
                f"{label}:{rid}: streamed tokens differ from get() response")
    diverged = []
    for rid, ref in ref_reqs.items():
        rec = run["requests"].get(rid)
        if rec is None:
            if rid not in run.get("requests_exclude", []):
                failures.append(f"{label}:{rid}: in reference but not in run")
            continue
        if ref["status"] != "finished":
            failures.append(f"{label}:{rid}: reference status {ref['status']}")
            continue
        if rec["status"] != "finished":
            continue
        run_ids, ref_ids = rec["token_ids"], ref["token_ids"]
        ref_lps, run_lps = ref.get("logprobs") or [], rec.get("logprobs") or []
        budget = rec.get("max_generate_length")
        if budget is not None and len(run_ids) == budget < len(ref_ids):
            ref_ids, ref_lps = ref_ids[:budget], ref_lps[:budget]
        pos = first_divergence(run_ids, ref_ids)
        entry = {
            "num_tokens_run": len(run_ids),
            "num_tokens_ref": len(ref_ids),
            "first_divergence": pos,
        }
        if pos is not None:
            diverged.append((rid, pos))
            gap = top2_gap(ref_lps[pos] if pos < len(ref_lps) else None)
            entry["ref_top2_gap_at_divergence"] = gap
            if pos < len(run_ids) and pos < len(ref_ids) and pos < len(
                    ref_lps):
                alt = [
                    t for t, _ in sorted(ref_lps[pos], key=lambda e: -e[1])[:2]
                ]
                entry["run_token_is_ref_top2"] = run_ids[pos] in alt
            if gap is None:
                failures.append(
                    f"{label}:{rid}: diverges at {pos} (run {len(run_ids)} vs ref {len(ref_ids)} "
                    "tokens) where the reference has no top-2 logprobs")
            elif gap >= args.gap_threshold:
                failures.append(
                    f"{label}:{rid}: diverges at {pos} with reference top-2 gap {gap:.4f} >= "
                    f"{args.gap_threshold}")
            elif entry.get("run_token_is_ref_top2") is False:
                failures.append(
                    f"{label}:{rid}: diverges at {pos} to token {run_ids[pos]}, which is not among "
                    "the reference top-2")
        prefix = len(run_ids) if pos is None else pos + 1
        diffs = [
            max_common_abs_diff(a, b)
            for a, b in zip(run_lps[:prefix], ref_lps[:prefix])
        ]
        diffs = [d for d in diffs if d is not None]
        entry["max_abs_logprob_diff_to_divergence"] = max(
            diffs) if diffs else None
        first_run = run_lps[0] if run_lps else None
        first_ref = ref_lps[0] if ref_lps else None
        entry["first_token_topk_logprob_cosine"] = topk_cosine(
            first_run, first_ref)
        cos = topk_cosine(first_run, first_ref, centered=True)
        entry["first_token_topk_centered_cosine"] = cos
        entry["first_token_max_abs_logprob_diff"] = max_common_abs_diff(
            first_run, first_ref)
        if cos is None:
            failures.append(
                f"{label}:{rid}: no first-token logprobs to compare")
        elif cos < args.cosine_threshold:
            failures.append(
                f"{label}:{rid}: first-token top-k centered cosine {cos:.6f} < "
                f"{args.cosine_threshold}")
        per_request[rid] = entry
    if not per_request:
        failures.append(f"{label}: no request was compared with the reference")
    drift = {
        "diverged_requests": len(diverged),
        "earliest_divergence": min((p for _, p in diverged), default=None),
        "compared_requests": len(per_request),
    }
    pol = run.get("policy") or {}
    if pol.get("expected") is not None and not pol.get("match"):
        message = f"{label}: resolved policy {pol.get('resolved')} != expected {pol['expected']}"
        # auto resolves per deployment (no compact kernels -> prefillFirst); only a requested mode is a contract.
        if (pol.get("details") or {}).get("mode_source") == "auto":
            warnings.append(
                f"{message} (auto on this deployment; policy-dependent stats skipped)"
            )
        else:
            failures.append(message)
    if run.get("prompt_hint_issues"):
        warnings.append(
            f"{label}: prompt hint issues: {run['prompt_hint_issues']}")
    for rid, issue in sorted((run.get("cache_hint_issues") or {}).items()):
        failures.append(
            f"{label}:{rid}: context-cache reuse outside its expectation: {issue}"
        )
    for rid, issue in sorted((run.get("admission_issues") or {}).items()):
        failures.append(
            f"{label}:{rid}: did not start beside the running requests: {issue}"
        )
    for rid, issue in sorted((run.get("stats_issues") or {}).items()):
        failures.append(
            f"{label}:{rid}: engine stats outside the expectation: {issue}")
    lag = timing_degraded(run)
    if lag:
        warnings.append(
            f"{label}: timing-dependent expectations skipped: {lag}")
    return failures, warnings, per_request, drift


def timing_degraded(run):
    """Why the driver skipped the timing-dependent expectations of ``run``, or None."""
    lag = run.get("trigger_lag") or {}
    if not lag.get("degraded"):
        return None
    return (f"a trigger fired {lag.get('max_tokens')} tokens late "
            f"(limit {lag.get('limit')})")


def graph_capture_enabled(run):
    """Whether the run's engine captured decode CUDA graphs (the binding captures by default)."""
    return bool((run.get("engine_kwargs") or {}).get("capture_cuda_graph",
                                                     True))


def decode_graph_rate(run):
    """(hit rate, pure-decode full steps) of a run, or (None, None) without the step metrics."""
    delta = run.get("metrics_delta") or {}
    if "decode_steps" not in delta or "decode_graph_steps" not in delta:
        return None, None
    decode = delta["decode_steps"]
    return delta["decode_graph_steps"] / decode if decode else None, decode


def graph_criterion(run, ref_run):
    """(status, detail, failure message or None) of the CUDA-graph hit-rate criterion."""
    if run.get("startup_error"):
        return "SKIP", "n/a: engine did not start", None
    if not graph_capture_enabled(run):
        return "SKIP", "n/a: CUDA graph capture disabled", None
    rate, decode = decode_graph_rate(run)
    if decode is None:
        return "SKIP", "n/a: the run exposes no step metrics", None
    if rate is None:
        return "SKIP", "n/a: no pure-decode full steps", None
    required, _ = decode_graph_rate(ref_run)
    if required is None or not graph_capture_enabled(ref_run):
        return "SKIP", "n/a: the reference has no pure-decode CUDA-graph rate", None
    # The reference runs one request at a time, so its rate covers other steps than the run's and
    # is a lower bound, not a value to match exactly.
    detail = f"run {rate:.4f} over {decode} steps, required >= {required:.4f} (reference)"
    if rate < required:
        return "FAIL", detail, (
            f"{run.get('label', '?')}: pure-decode CUDA-graph hit rate {rate:.4f} < "
            f"{required:.4f} (reference)")
    return "PASS", detail, None


CRITERIA = (
    ("tie_rule",
     "divergences only where the reference top-2 gap < {gap}, to a reference top-2 token"
     ),
    ("first_token_cosine", "first-token top-k centered cosine >= {cos}"),
    ("graph_hit_rate",
     "pure-decode CUDA-graph hit rate no lower than the reference"),
    ("cache_reuse",
     "context-cache reuse within each request's expect.reuse_tokens bounds"),
    ("admission",
     "each request with expect.joins_running starts before the listed requests finish"
     ),
    ("engine_stats",
     "engine metrics and request stats within expect_metrics, expect.stats and expect.admitted_before"
     ),
    ("status_policy", "expected statuses, startup and resolved policy"),
)
TIMING_CRITERIA = ("admission", "engine_stats")


def criterion_of(message):
    """The criterion a failure message belongs to."""
    if "diverges at" in message:
        return "tie_rule"
    if "CUDA-graph hit rate" in message:
        return "graph_hit_rate"
    if "cosine" in message or "first-token logprobs" in message:
        return "first_token_cosine"
    if "context-cache reuse" in message:
        return "cache_reuse"
    if "did not start beside the running requests" in message:
        return "admission"
    if "engine stats outside the expectation" in message:
        return "engine_stats"
    return "status_policy"


def criteria_summary(failures, per_request, drift, graph, args, run=None):
    """Pass/fail per criterion with the numbers behind it (reporting only; RESULT uses failures)."""
    by = {
        key: [m for m in failures if criterion_of(m) == key]
        for key, _ in CRITERIA
    }
    gaps = [
        e["ref_top2_gap_at_divergence"] for e in per_request.values()
        if e.get("ref_top2_gap_at_divergence") is not None
    ]
    cosines = [
        e["first_token_topk_centered_cosine"] for e in per_request.values()
        if e["first_token_topk_centered_cosine"] is not None
    ]
    detail = {
        "tie_rule":
        f"diverged {drift['diverged_requests'] if drift else '-'}/"
        f"{drift['compared_requests'] if drift else '-'}, max gap at divergence "
        f"{fmt(max(gaps) if gaps else None, 4)}",
        "first_token_cosine":
        f"min {fmt(min(cosines) if cosines else None)}",
        "graph_hit_rate":
        graph[1],
        "cache_reuse":
        f"{len(by['cache_reuse'])} request(s) outside their bounds",
        "admission":
        f"{len(by['admission'])} request(s) waited for the running ones",
        "engine_stats":
        f"{len(by['engine_stats'])} expectation(s) not met",
        "status_policy":
        "",
    }
    degraded = timing_degraded(run or {})
    out = {}
    for key, text in CRITERIA:
        if by[key]:
            status = "FAIL"
        elif key == "graph_hit_rate":
            status = graph[0]
        elif key in TIMING_CRITERIA and degraded:
            status = "SKIP"
            detail[key] = f"n/a: {degraded}"
        else:
            status = "PASS"
        out[key] = {
            "status":
            status,
            "criterion":
            text.format(gap=args.gap_threshold, cos=args.cosine_threshold),
            "detail":
            detail[key],
            "failures":
            by[key],
        }
    return out


def fmt(v, nd=6):
    return "-" if v is None else (
        f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--gap-threshold", type=float, default=0.1)
    ap.add_argument("--cosine-threshold", type=float, default=0.999)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    run_data = load(args.run)
    ref_data = load(args.reference)
    ref_run = ref_data["runs"][0]
    all_failures, all_warnings, report = [], [], {"runs": {}}

    for run in run_data["runs"]:
        failures, warnings, per_request, drift = compare_run(
            run, ref_run, args)
        graph = graph_criterion(run, ref_run)
        if graph[2]:
            failures.append(graph[2])
        criteria = criteria_summary(failures, per_request, drift, graph, args,
                                    run)
        report["runs"][run.get("label", "?")] = {
            "criteria": criteria,
            "drift": drift,
            "requests": per_request,
            "failures": failures,
            "warnings": warnings,
        }
        all_failures += failures
        all_warnings += warnings

        print(
            f"== {run.get('label')} vs reference ({run_data.get('engine_dir')})"
        )
        print(
            f"{'id':<20}{'run':>5}{'ref':>5}{'div@':>6}{'gap':>9}{'cos(lp)':>11}{'cos(c)':>11}"
            f"{'max|dlp|':>10}{'drift|dlp|':>11}  status")
        for rid, rec in run["requests"].items():
            e = per_request.get(rid, {})
            print(
                f"{rid:<20}{fmt(e.get('num_tokens_run')):>5}{fmt(e.get('num_tokens_ref')):>5}"
                f"{fmt(e.get('first_divergence')):>6}{fmt(e.get('ref_top2_gap_at_divergence'), 4):>9}"
                f"{fmt(e.get('first_token_topk_logprob_cosine')):>11}"
                f"{fmt(e.get('first_token_topk_centered_cosine')):>11}"
                f"{fmt(e.get('first_token_max_abs_logprob_diff'), 4):>10}"
                f"{fmt(e.get('max_abs_logprob_diff_to_divergence'), 4):>11}  {rec['status']}"
            )
        print(f"drift (informational, see README): {drift}")
        print("  max |dlogprob| up to the first divergence: " + ", ".join(
            f"{rid}={fmt(e.get('max_abs_logprob_diff_to_divergence'), 4)}"
            for rid, e in per_request.items()))
        print("criteria:")
        for value in criteria.values():
            print(
                f"  [{value['status']}] {value['criterion']}: {value['detail']}"
            )

    print(
        "note: first-token cosines approximate logits cosine from the top-k logprobs only; cos(c) "
        "(centered) is gated. cos(lp), max|dlp| (first token) and drift|dlp| (every position up to "
        "the first divergence) are informational.")
    for w in all_warnings:
        print(f"WARN  {w}")
    for f in all_failures:
        print(f"FAIL  {f}")
    print("RESULT:", "FAIL" if all_failures else "PASS")
    if args.json_out:
        report["failures"] = all_failures
        report["warnings"] = all_warnings
        report["pass"] = not all_failures
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=1)
    return 1 if all_failures else 0


if __name__ == "__main__":
    sys.exit(main())
