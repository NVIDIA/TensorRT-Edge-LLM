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

import json
import threading
import time

import httpx
import pytest

from scripts.benchmark_runner import online, workloads
from scripts.benchmark_runner.adapters import openai


def _workload():
    return {
        "run_id":
        "run",
        "concurrency":
        2,
        "load": {
            "arrival_rate_rps": None,
        },
        "warmup": {
            "iterations": 1,
            "scope": "canonical_batch",
        },
        "generation": {
            "temperature": 0,
            "top_p": 1,
            "top_k": 0,
            "ignore_eos": True,
        },
        "requests": [{
            "id": f"request-{index}",
            "token_ids": [index + 1],
            "output_tokens": 2,
        } for index in range(4)],
    }


def test_openai_adapter_translates_requests_and_streams():
    adapter = openai.CompletionsAdapter({
        "endpoint": "http://127.0.0.1:8000",
        "model": "example",
    })
    request = adapter.inference_request(_workload()["requests"][0],
                                        _workload()["generation"])

    assert request.url == "http://127.0.0.1:8000/v1/completions"
    assert json.loads(request.body) == {
        "model": "example",
        "prompt": [1],
        "max_tokens": 2,
        "temperature": 0,
        "top_p": 1,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
        "ignore_eos": True,
    }
    assert adapter.parse_stream_line(
        'data: {"choices":[{"text":"x","finish_reason":null}]}\n') == {
            "text": "x",
            "token_observed": True,
        }
    assert adapter.parse_stream_line(
        'data: {"choices":[{"text":"","finish_reason":null}]}\n') is None
    assert adapter.parse_stream_line("data: [DONE]\n") == {"done": True}


def test_online_request_uses_streamed_usage():
    adapter = openai.CompletionsAdapter({
        "endpoint": "http://backend",
        "model": "example",
    })
    workload = workloads.normalize_online(_workload())

    def respond(request):
        assert request.url.path == "/v1/completions"
        return httpx.Response(
            200,
            text=('data: {"choices":[{"text":"x","finish_reason":null}]}\n\n'
                  'data: {"choices":[],"usage":{"prompt_tokens":1,'
                  '"completion_tokens":2,"total_tokens":3}}\n\n'
                  "data: [DONE]\n\n"))

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        submit_ns = time.monotonic_ns()
        result = online._run_request(adapter, client, workload["requests"][0],
                                     workload["generation"], submit_ns, 10)

    assert result["success"]
    assert result["input_tokens"] == 1
    assert result["output_tokens"] == 2
    assert result["first_token_ns"] >= result["timestamp_ns"]


def test_online_driver_owns_warmup_concurrency_and_event_order(
        monkeypatch, tmp_path):
    active = 0
    maximum_active = 0
    observed_ids = []
    clients = []
    event_batches = []
    lock = threading.Lock()
    append_events = online._append_events

    def record_events(output_path, events):
        event_batches.append([event["event"] for event in events])
        append_events(output_path, events)

    def run_request(unused_adapter, client, request, unused_generation,
                    submit_ns, unused_timeout):
        nonlocal active, maximum_active
        del unused_adapter, unused_generation, unused_timeout
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            observed_ids.append(request["id"])
            clients.append(client)
        time.sleep(0.02 if request["id"] == "request-0" else 0.01)
        with lock:
            active -= 1
        return {
            "success": True,
            "request_id": request["id"],
            "timestamp_ns": submit_ns,
            "first_token_ns": submit_ns + 1,
            "completion_ns": time.monotonic_ns(),
            "input_tokens": len(request["token_ids"]),
            "input_token_ids_sha256": request["input_token_ids_sha256"],
            "output_tokens": request["output_tokens"],
            "output_sha256": "output",
        }

    monkeypatch.setattr(online, "_run_request", run_request)
    monkeypatch.setattr(online, "_append_events", record_events)
    monkeypatch.setattr(
        online, "_wait_until_ready",
        lambda unused_adapter, unused_client, unused_timeout:
        (time.monotonic_ns(), {
            "url": "http://backend/health",
            "status": 200,
        }))

    output = online.run(object(), _workload(), tmp_path / "events.jsonl")
    events = [json.loads(line) for line in output.read_text().splitlines()]
    completed = [
        event for event in events if event["event"] == "request_completed"
    ]

    assert set(observed_ids[:2]) == {"request-0", "request-1"}
    assert set(observed_ids[2:]) == {
        "request-0", "request-1", "request-2", "request-3"
    }
    assert maximum_active == 2
    assert len({id(client) for client in clients}) == 1
    assert [event["request_id"] for event in completed
            ] == ["request-0", "request-1", "request-2", "request-3"]
    assert [event["timestamp_ns"] for event in completed
            ] == sorted(event["timestamp_ns"] for event in completed)
    assert event_batches[:3] == [["marker"], ["marker"],
                                 ["measurement_started"]]
    assert event_batches[3] == (["request_completed"] * 4 +
                                ["measurement_ended", "run_ended"])
    assert [
        event.get("name") for event in events if event["event"] == "marker"
    ] == ["ready", "warmup_completed"]
    assert next(
        event for event in events if event["event"] ==
        "measurement_started")["timestamp_ns"] <= completed[0]["timestamp_ns"]


def test_online_driver_rejects_changed_token_hash(tmp_path):
    workload = _workload()
    workload["requests"][0]["input_token_ids_sha256"] = "wrong"

    with pytest.raises(ValueError, match="token hash"):
        online.run(object(), workload, tmp_path / "events.jsonl")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_positive_number_rejects_non_finite_values(value):
    with pytest.raises(ValueError, match="finite and positive"):
        workloads.positive_number(value, "value")


def test_readiness_requests_use_the_remaining_timeout(monkeypatch):
    clock = [0.0]
    timeouts = []

    class Client:

        def request(self, unused_method, unused_url, *, content, headers,
                    timeout):
            del content, headers
            timeouts.append(timeout)
            clock[0] += 4.8 if len(timeouts) == 1 else timeout
            raise httpx.ConnectError("not ready")

    def sleep(duration):
        clock[0] += duration

    monkeypatch.setattr(online.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(online.time, "sleep", sleep)
    adapter = openai.CompletionsAdapter({
        "endpoint": "http://127.0.0.1:8000",
        "model": "example",
    })

    with pytest.raises(TimeoutError, match="not ready"):
        online._wait_until_ready(adapter, Client(), 6.0)

    assert timeouts == pytest.approx([5.0, 1.1])
    assert clock[0] == pytest.approx(6.0)


def test_online_driver_records_request_failures(monkeypatch, tmp_path):
    workload = _workload()
    workload["warmup"] = {"iterations": 0, "scope": "none"}
    workload["requests"] = workload["requests"][:1]

    monkeypatch.setattr(
        online, "_wait_until_ready",
        lambda unused_adapter, unused_client, unused_timeout:
        (time.monotonic_ns(), {
            "url": "http://backend/health",
            "status": 200,
        }))

    def fail_request(unused_adapter, unused_client, request, unused_generation,
                     submit_ns, unused_timeout):
        now = time.monotonic_ns()
        return {
            "success": False,
            "request_id": request["id"],
            "timestamp_ns": submit_ns,
            "completion_ns": now + 1,
            "input_tokens": len(request["token_ids"]),
            "input_token_ids_sha256": request["input_token_ids_sha256"],
            "requested_output_tokens": request["output_tokens"],
            "error": "RuntimeError: backend failed",
            "traceback": "backend traceback",
        }

    monkeypatch.setattr(online, "_run_request", fail_request)
    output = tmp_path / "events.jsonl"

    with pytest.raises(RuntimeError, match="backend failed"):
        online.run(object(), workload, output)

    events = [json.loads(line) for line in output.read_text().splitlines()]
    failure = next(event for event in events
                   if event["event"] == "request_failed")
    assert failure["request_id"] == "request-0"
    assert failure["traceback"] == "backend traceback"
    assert events[-1]["event"] == "run_failed"
