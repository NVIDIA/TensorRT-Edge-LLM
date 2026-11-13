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
"""Failure-oracle tests for the pressure harness and benchmark statistics."""

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import replace

import httpx
import pytest
from server_stress.support import Settings, assert_load, idle, request_one

from scripts.benchmark_runner import __main__ as bench_cli
from scripts.benchmark_runner import online
from scripts.benchmark_runner.adapters.openai import CompletionsAdapter
from scripts.benchmark_runner.metrics import distribution, summarize_events


def sse(*, done=True, finish=True, usage=True):
    events = [{
        'choices': [{
            'delta': {
                'content': 'hello'
            },
            'finish_reason': None
        }]
    }]
    if finish:
        events.append({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
    if usage:
        events.append({
            'choices': [],
            'usage': {
                'completion_tokens': 2,
                'prompt_tokens': 3
            }
        })
    text = ''.join('data: ' + json.dumps(e) + '\n\n' for e in events)
    return text + ('data: [DONE]\n\n' if done else '')


def call_request(response, *, stream=True, mode=None, settings=None):

    async def run():
        async with httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _: response)) as client:
            return await request_one(client, settings or Settings(), 0, stream,
                                     mode)

    return asyncio.run(run())


@pytest.mark.parametrize('missing', ['done', 'finish', 'usage'])
def test_stress_rejects_incomplete_sse(missing):
    """Truncated streams and missing usage are failed requests."""
    row = call_request(httpx.Response(200, text=sse(**{missing: False})))
    assert row['outcome'] == 'error'


def test_stress_distinguishes_empty_role_from_first_token():
    """First-token cancellation waits for generated content and records no completion tokens."""
    text = 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n' + sse()
    row = call_request(httpx.Response(200, text=text), mode='first_token')
    assert row['outcome'] == 'cancelled'
    assert row['ttft_s'] is not None
    assert row['output_tokens'] is None


@pytest.mark.parametrize('status,outcome', [(429, 'rejected'), (500, 'error'),
                                            (401, 'error')])
def test_stress_classifies_http_failures(status, outcome):
    """Overload refusals are distinct from authentication and server errors."""
    row = call_request(httpx.Response(status))
    assert row['outcome'] == outcome
    assert row['status'] == status


class SlowStream(httpx.AsyncByteStream):

    async def __aiter__(self):
        await asyncio.sleep(0.1)
        yield sse().encode()


@pytest.mark.parametrize('mode,outcome', [(None, 'error'),
                                          ('timed', 'cancelled')])
def test_stress_enforces_total_request_deadline(mode, outcome):
    """A slow response becomes either an explicit timeout failure or an intended disconnect."""
    settings = replace(Settings(), request_timeout_s=0.01, cancel_after_s=0.01)
    row = call_request(httpx.Response(200, stream=SlowStream()),
                       mode=mode,
                       settings=settings)
    assert row['outcome'] == outcome
    assert row['latency_s'] < 0.1


def test_stress_valid_sse_reports_usage():
    """Successful streams preserve server token counts and measurable TTFT."""
    row = call_request(httpx.Response(200, text=sse()))
    assert row['outcome'] == 'success'
    assert row['output_tokens'] == 2
    assert 0 <= row['ttft_s'] <= row['latency_s']


def test_stress_completion_before_cancel_is_success():
    """Requests finishing before the cancellation timer count as completed work."""
    row = call_request(httpx.Response(200, text=sse()), mode='timed')
    assert row['outcome'] == 'success'


@pytest.mark.parametrize('gauges', [{
    'queued': 1,
    'resident': 0
}, {
    'queued': 0,
    'resident': 1
}])
def test_stress_idle_requires_native_scheduler_drain(gauges):
    """HTTP admission idle alone is insufficient while native work remains."""
    assert not idle({
        'active_requests': 0,
        'queued_requests': 0,
        'scheduling': gauges
    })


@pytest.mark.parametrize('change,options', [({
    'outcomes': {
        'success': 0
    }
}, {}), ({
    'error_rate': 0.1
}, {}), ({
    'health_errors': 1
}, {}), ({
    'outcomes': {
        'success': 4,
        'rejected': 1
    }
}, {}), ({}, {
    'overload': True
}), ({}, {
    'cancellation': True
})])
def test_stress_gates_reject_missing_coverage_and_errors(change, options):
    """Pressure runs fail on errors, empty work, or unexercised fault scenarios."""
    summary = {'outcomes': {'success': 4}, 'error_rate': 0, 'health_errors': 0}
    summary.update(change)
    with pytest.raises(AssertionError):
        assert_load(summary, Settings(), **options)


def test_benchmark_summary_excludes_warmup_and_uses_measured_interval():
    """Throughput uses completed measured tokens and includes the measurement tail."""
    events = [
        {
            'event': 'request_completed',
            'phase': 'warmup'
        },
        {
            'event': 'measurement_started',
            'timestamp_ns': 0
        },
        {
            'event': 'request_completed',
            'phase': 'measurement',
            'timestamp_ns': 0,
            'first_token_ns': 1_000_000_000,
            'completion_ns': 4_000_000_000,
            'output_tokens': 8
        },
        {
            'event': 'request_failed',
            'phase': 'measurement',
            'timestamp_ns': 1_000_000_000,
            'completion_ns': 5_000_000_000
        },
        {
            'event': 'measurement_ended',
            'timestamp_ns': 5_000_000_000
        },
        {
            'event': 'run_failed'
        },
    ]
    result = summarize_events(events)
    assert result['output_tokens_per_s'] == pytest.approx(1.6)
    assert result['success_rps'] == pytest.approx(0.2)
    assert result['error_rate'] == 0.5
    assert result['ttft_s']['p99'] == 1
    assert not result['run_complete']


def test_benchmark_empty_distribution_is_explicit():
    """Missing samples remain null instead of indicating zero latency."""
    assert distribution([])['p99'] is None
    assert distribution([1, 2, 3, 4])['p50'] == 2.5


def test_failure_rate_excludes_intentional_disconnects_and_overload():
    """Expected refusals and disconnects preserve the failure rate among attempted completions."""
    from scripts.benchmark_runner.metrics import summarize
    rows = [{
        'outcome': name,
        'output_tokens': 1,
        'latency_s': 1
    } for name in ['success', 'error', *(['rejected', 'cancelled'] * 100)]]
    assert summarize(rows, 1)['error_rate'] == 0.5


@pytest.mark.parametrize('value', [{}, [], 3, None])
def test_cli_invalid_workload_leaves_failure_summary(tmp_path, value):
    """Invalid workload input returns failure and a machine-readable artifact."""
    workload = tmp_path / 'workload.json'
    workload.write_text(json.dumps(value))
    output = tmp_path / 'output'
    assert bench_cli.main([
        '--url', 'http://127.0.0.1:1', '--model', 'm', '--workload',
        str(workload), '--output',
        str(output)
    ]) == 1
    result = json.loads((output / 'summary.json').read_text())
    assert result['error']
    assert not result['run_complete']


@pytest.mark.parametrize('increment,passes', [(0, False), (1, True)])
def test_cancellation_gate_requires_native_evidence(increment, passes):
    """IFB churn requires observed engine cancellation beyond the client disconnect count."""
    summary = {
        'outcomes': {
            'success': 4,
            'cancelled': 1
        },
        'error_rate': 0,
        'health_errors': 0,
        'health_before': {
            'capabilities': {
                'in_flight_batching': True
            },
            'scheduling': {
                'cancelled': 5,
                'failed': 0
            }
        },
        'health_after': {
            'scheduling': {
                'cancelled': 5 + increment,
                'failed': 0
            }
        }
    }
    if passes:
        assert_load(summary, Settings(), cancellation=True)
    else:
        with pytest.raises(AssertionError):
            assert_load(summary, Settings(), cancellation=True)


def test_load_stops_when_owned_server_dies(monkeypatch, tmp_path):
    """A crashed owned process stops load generation and leaves a failure summary."""
    from server_stress import support

    class Process:

        def poll(self):
            return 1

    class Owned:
        process = Process()

        def rss_mb(self):
            return 100.0

    def respond(request):
        if request.url.path == '/health':
            return httpx.Response(200,
                                  json={
                                      'active_requests': 0,
                                      'queued_requests': 0
                                  })
        return httpx.Response(200, text=sse())

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        support.httpx, 'AsyncClient',
        lambda **kwargs: client_type(transport=httpx.MockTransport(respond),
                                     **kwargs))
    settings = replace(Settings(), concurrency=[1])
    result = asyncio.run(
        support.run_load(settings, tmp_path, 1, 60, owned=Owned()))
    assert result['early_stop']
    assert result['elapsed_s'] < 1
    assert (tmp_path / 'requests.jsonl').exists()
    with pytest.raises(AssertionError):
        assert_load(result, settings)


@pytest.mark.parametrize('cancelled,failed,passes', [(1, 0, False),
                                                     (10, 0, True),
                                                     (10, 1, False)])
def test_native_gate_covers_most_first_token_cancellations(
        cancelled, failed, passes):
    """First-token churn rejects largely ineffective cancellation and native failures."""
    summary = {
        'outcomes': {
            'success': 4,
            'cancelled': 10
        },
        'error_rate': 0,
        'health_errors': 0,
        'cancellation_mode': 'first_token',
        'health_before': {
            'capabilities': {
                'in_flight_batching': True
            },
            'scheduling': {
                'cancelled': 0,
                'failed': 0
            }
        },
        'health_after': {
            'scheduling': {
                'cancelled': cancelled,
                'failed': failed
            }
        }
    }
    if passes:
        assert_load(summary, Settings(), cancellation=True)
    else:
        with pytest.raises(AssertionError):
            assert_load(summary, Settings(), cancellation=True)


def test_cancel_calibration_reports_inconclusive(monkeypatch, tmp_path):
    """A response too short to separate cancellation from completion produces an explicit skip."""
    from server_stress import test_server_stress as cases

    async def healthy(*args):
        return {'active_requests': 0, 'queued_requests': 0}

    async def respond(*args):
        return {
            'outcome': 'success',
            'latency_s': 1,
            'ttft_s': .8,
            'output_tokens': 16
        }

    monkeypatch.setattr(cases, 'health', healthy)
    monkeypatch.setattr(cases, 'request_one', respond)
    with pytest.raises(pytest.skip.Exception):
        cases.test_cancel_drain(Settings(), tmp_path, None)


@pytest.mark.parametrize('scenario', ['gate', 'response', 'health', 'warmup'])
def test_stress_checks_survive_python_optimization(tmp_path, scenario):
    """Optimized Python preserves acceptance, response, health, and warmup failures."""
    script = """
import asyncio
import sys
from pathlib import Path
import httpx
from server_stress.support import Settings, assert_load, health, request_one, run_load

scenario = sys.argv[1]
async def exercise():
    if scenario == 'gate':
        assert_load({'outcomes': {'success': 4}, 'error_rate': 1,
                     'health_errors': 0}, Settings())
        return
    def respond(request):
        if request.url.path == '/health':
            return httpx.Response(200, json={
                'active_requests': -1 if scenario == 'health' else 0,
                'queued_requests': 0})
        if scenario == 'warmup':
            return httpx.Response(500)
        return httpx.Response(200, text='data: [DONE]\\n\\n')
    original_client = httpx.AsyncClient
    def client(**kwargs):
        return original_client(transport=httpx.MockTransport(respond), **kwargs)
    if scenario == 'warmup':
        httpx.AsyncClient = client
        await run_load(Settings(), Path(sys.argv[2]), 1, 0.01)
        return
    async with client() as connection:
        if scenario == 'health':
            await health(connection, 'http://server')
        else:
            row = await request_one(connection, Settings(), 0, True)
            if row['outcome'] == 'error' and 'Truncated SSE stream' in row.get('error', ''):
                return 'response rejected'
            raise SystemExit('invalid response accepted: ' + repr(row))
try:
    result = asyncio.run(exercise())
except AssertionError as error:
    print(error)
else:
    if result == 'response rejected':
        print(result)
    else:
        raise SystemExit('invalid input accepted under optimization')
"""
    result = subprocess.run(
        [sys.executable, '-O', '-c', script, scenario,
         str(tmp_path)],
        env={
            **os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)
        },
        capture_output=True,
        text=True,
        timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = {
        'gate': 'error_rate',
        'response': 'response rejected',
        'health': 'Invalid health gauge',
        'warmup': 'Warmup failed'
    }
    assert expected[scenario] in result.stdout


@pytest.mark.parametrize('status,valid_body', [(200, True), (429, False),
                                               (503, False), (None, False),
                                               (200, False)])
def test_online_status_survives_events_and_summary(monkeypatch, tmp_path,
                                                   status, valid_body):
    """HTTP status survives success, HTTP errors, bad streams, and transport failures."""

    def respond(request):
        if request.url.path == '/health':
            return httpx.Response(200)
        if status is None:
            raise httpx.ConnectError('connection failed', request=request)
        if valid_body:
            return httpx.Response(
                status,
                text=
                ('data: {"choices":[{"text":"hello"}]}\n\n'
                 'data: {"usage":{"prompt_tokens":1,"completion_tokens":2}}\n\n'
                 'data: [DONE]\n\n'))
        return httpx.Response(status, text='data: [DONE]\n\n')

    original_client = httpx.Client
    monkeypatch.setattr(
        online.httpx, 'Client', lambda **kwargs: original_client(
            transport=httpx.MockTransport(respond), **kwargs))
    workload = {
        'run_id': 'status',
        'concurrency': 1,
        'generation': {
            'temperature': 0,
            'top_p': 1,
            'top_k': 0,
            'ignore_eos': True
        },
        'warmup': {
            'iterations': 0,
            'scope': 'none'
        },
        'requests': [{
            'id': 'one',
            'token_ids': [1],
            'output_tokens': 2
        }]
    }
    adapter = CompletionsAdapter({
        'endpoint': 'http://server',
        'model': 'test'
    })
    output = tmp_path / 'events.jsonl'
    if valid_body:
        online.run(adapter, workload, output)
    else:
        with pytest.raises(RuntimeError, match='online request'):
            online.run(adapter, workload, output)
    events = [json.loads(line) for line in output.read_text().splitlines()]
    measured = [
        event for event in events
        if event['event'] in ('request_completed', 'request_failed')
    ]
    assert len(measured) == 1
    assert measured[0]['status'] == status
    summary = summarize_events(events)
    assert summary['http_status_counts'] == {str(status): 1}
    assert summary['outcomes'] == {'success' if valid_body else 'error': 1}
