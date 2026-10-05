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
"""Shared load driver for online-serving benchmarks."""

import concurrent.futures
import hashlib
import json
import pathlib
import time
import traceback
from typing import Any, Mapping, Sequence

import httpx

from . import workloads
from .adapters import base


def _event(run_id: str, event_name: str, timestamp_ns: int,
           **fields: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "event": event_name,
        "timestamp_ns": timestamp_ns,
        **fields,
    }


def _wait_until_ready(adapter: base.ServingAdapter, client: httpx.Client,
                      timeout_s: float) -> tuple[int, dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    request = adapter.readiness_request()
    last_error = "not attempted"
    while True:
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            break
        try:
            response = client.request(request.method,
                                      request.url,
                                      content=request.body,
                                      headers=request.headers,
                                      timeout=min(remaining_s, 5.0))
            if 200 <= response.status_code < 300:
                return time.monotonic_ns(), {
                    "url": request.url,
                    "status": response.status_code,
                }
            last_error = f"HTTP {response.status_code}"
        except httpx.HTTPError as error:
            last_error = str(error)
        remaining_s = deadline - time.monotonic()
        if remaining_s > 0:
            time.sleep(min(0.1, remaining_s))
    raise TimeoutError(f"Backend did not become ready: {last_error}")


def _run_request(adapter: base.ServingAdapter, client: httpx.Client,
                 request: Mapping[str, Any], generation: Mapping[str, Any],
                 submit_ns: int, timeout_s: float) -> dict[str, Any]:
    try:
        backend_request = adapter.inference_request(request, generation)
        first_token_ns = None
        usage = None
        text = []
        with client.stream(backend_request.method,
                           backend_request.url,
                           content=backend_request.body,
                           headers=backend_request.headers,
                           timeout=timeout_s) as response:
            if not 200 <= response.status_code < 300:
                raise RuntimeError(
                    f"Completion request failed: HTTP {response.status_code}")
            for line in response.iter_lines():
                observed_ns = time.monotonic_ns()
                stream_event = adapter.parse_stream_line(line)
                if not stream_event:
                    continue
                if (stream_event.get("token_observed")
                        and first_token_ns is None):
                    first_token_ns = observed_ns
                fragment = stream_event.get("text")
                if isinstance(fragment, str):
                    text.append(fragment)
                if isinstance(stream_event.get("usage"), dict):
                    usage = stream_event["usage"]
        completion_ns = time.monotonic_ns()
        if usage is None:
            raise RuntimeError("Completion response did not include usage")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if (not isinstance(prompt_tokens, int)
                or isinstance(prompt_tokens, bool)
                or prompt_tokens != len(request["token_ids"])):
            raise RuntimeError(
                "Completion response reported a different prompt length")
        if (not isinstance(completion_tokens, int)
                or isinstance(completion_tokens, bool) or completion_tokens < 1
                or first_token_ns is None):
            raise RuntimeError(
                "Completion response did not report generated tokens")
        if (generation["ignore_eos"]
                and completion_tokens != request["output_tokens"]):
            raise RuntimeError(
                "Completion response ended before the requested length")
    except Exception as error:
        return {
            "success": False,
            "request_id": request["id"],
            "timestamp_ns": submit_ns,
            "completion_ns": time.monotonic_ns(),
            "input_tokens": len(request["token_ids"]),
            "input_token_ids_sha256": request["input_token_ids_sha256"],
            "requested_output_tokens": request["output_tokens"],
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    return {
        "success": True,
        "request_id": request["id"],
        "timestamp_ns": submit_ns,
        "first_token_ns": first_token_ns,
        "completion_ns": completion_ns,
        "input_tokens": prompt_tokens,
        "input_token_ids_sha256": request["input_token_ids_sha256"],
        "output_tokens": completion_tokens,
        "output_sha256":
        hashlib.sha256("".join(text).encode("utf-8")).hexdigest(),
    }


def _run_requests(adapter: base.ServingAdapter, client: httpx.Client,
                  requests: Sequence[Mapping[str,
                                             Any]], generation: Mapping[str,
                                                                        Any],
                  concurrency: int, arrival_rate_rps: float | None,
                  timeout_s: float) -> list[dict[str, Any]]:
    schedule_start = time.monotonic()
    active: dict[concurrent.futures.Future, int] = {}
    results = {}
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency) as executor:
        for index, request in enumerate(requests):
            while len(active) >= concurrency:
                done, _ = concurrent.futures.wait(
                    active, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    results[active.pop(future)] = future.result()
            if arrival_rate_rps is not None:
                due = schedule_start + index / arrival_rate_rps
                time.sleep(max(0.0, due - time.monotonic()))
            submit_ns = time.monotonic_ns()
            future = executor.submit(_run_request, adapter, client, request,
                                     generation, submit_ns, timeout_s)
            active[future] = index
        for future, index in active.items():
            results[index] = future.result()
    return [results[index] for index in range(len(requests))]


def _warmup_requests(
        workload: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    scope = workload["warmup_scope"]
    if scope == "none":
        return []
    if scope == "canonical_batch":
        return workload["requests"][:workload["concurrency"]]
    return workload["requests"]


def _request_event(run_id: str, result: Mapping[str, Any],
                   phase: str) -> dict[str, Any]:
    event_name = "request_completed" if result["success"] else "request_failed"
    return _event(run_id,
                  event_name,
                  result["timestamp_ns"],
                  source="shared_online_driver",
                  phase=phase,
                  **{
                      name: value
                      for name, value in result.items()
                      if name not in ("success", "timestamp_ns")
                  })


def _start_events(output_path: pathlib.Path, event: Mapping[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(event, sort_keys=True) + "\n",
                           encoding="utf-8")


def _append_events(output_path: pathlib.Path,
                   events: Sequence[Mapping[str, Any]]) -> None:
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.writelines(
            json.dumps(event, sort_keys=True) + "\n" for event in events)
        output_file.flush()


def _request_failure(failures: Sequence[Mapping[str, Any]],
                     output_path: pathlib.Path) -> RuntimeError:
    first = failures[0]
    return RuntimeError(
        f"{len(failures)} online request(s) failed; first failure "
        f"{first['request_id']}: {first['error']}; events: {output_path}")


def run(adapter: base.ServingAdapter,
        workload: Mapping[str, Any],
        output_path: pathlib.Path,
        request_timeout_s: float = 3600,
        ready_timeout_s: float = 300) -> pathlib.Path:
    """Run one workload and write controller-timestamped events."""
    request_timeout_s = workloads.positive_number(request_timeout_s,
                                                  "request_timeout_s")
    ready_timeout_s = workloads.positive_number(ready_timeout_s,
                                                "ready_timeout_s")
    normalized = workloads.normalize_online(workload)
    run_id = normalized["run_id"]
    run_started_ns = time.monotonic_ns()
    _start_events(output_path, _event(run_id, "run_started", run_started_ns))
    limits = httpx.Limits(max_connections=normalized["concurrency"],
                          max_keepalive_connections=normalized["concurrency"])
    with httpx.Client(limits=limits) as client:
        try:
            ready_ns, readiness = _wait_until_ready(adapter, client,
                                                    ready_timeout_s)
        except Exception as error:
            _append_events(output_path, [
                _event(run_id,
                       "run_failed",
                       time.monotonic_ns(),
                       source="shared_online_driver",
                       phase="readiness",
                       error=f"{type(error).__name__}: {error}",
                       traceback=traceback.format_exc())
            ])
            raise
        _append_events(output_path, [
            _event(run_id,
                   "marker",
                   ready_ns,
                   name="ready",
                   source="shared_online_driver",
                   **readiness)
        ])

        warmup_requests = _warmup_requests(normalized)
        for iteration in range(normalized["warmup_iterations"]):
            warmed = _run_requests(adapter, client, warmup_requests,
                                   normalized["generation"],
                                   normalized["concurrency"],
                                   normalized["arrival_rate_rps"],
                                   request_timeout_s)
            failures = [result for result in warmed if not result["success"]]
            if failures:
                failure_events = [
                    _request_event(run_id, result, "warmup")
                    for result in failures
                ]
                failed_ns = max(result["completion_ns"] for result in failures)
                failure_events.append(
                    _event(run_id,
                           "run_failed",
                           failed_ns,
                           source="shared_online_driver",
                           phase="warmup",
                           warmup_iteration=iteration))
                _append_events(output_path, failure_events)
                raise _request_failure(failures, output_path)
        warmup_completed_ns = time.monotonic_ns()
        _append_events(output_path, [
            _event(run_id,
                   "marker",
                   warmup_completed_ns,
                   name="warmup_completed",
                   source="shared_online_driver")
        ])

        measurement_started_ns = time.monotonic_ns()
        _append_events(output_path, [
            _event(run_id,
                   "measurement_started",
                   measurement_started_ns,
                   source="shared_online_driver")
        ])
        completed = _run_requests(adapter, client, normalized["requests"],
                                  normalized["generation"],
                                  normalized["concurrency"],
                                  normalized["arrival_rate_rps"],
                                  request_timeout_s)

    measurement_ended_ns = max(result["completion_ns"] for result in completed)
    final_events = [
        _request_event(run_id, result, "measurement") for result in completed
    ]
    final_events.append(
        _event(run_id,
               "measurement_ended",
               measurement_ended_ns,
               source="shared_online_driver"))
    failures = [result for result in completed if not result["success"]]
    if failures:
        final_events.append(
            _event(run_id,
                   "run_failed",
                   time.monotonic_ns(),
                   source="shared_online_driver",
                   phase="measurement"))
    else:
        final_events.append(_event(run_id, "run_ended", time.monotonic_ns()))
    _append_events(output_path, final_events)
    if failures:
        raise _request_failure(failures, output_path)
    return output_path
