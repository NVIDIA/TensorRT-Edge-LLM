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
"""Live pressure cases for the OpenAI server and its owned process lifecycle."""

import asyncio
import json
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from .support import (ServerProcess, assert_load, check, headers, health, idle,
                      request_one, run_load, wait_idle, wait_ready, write_json)

pytestmark = pytest.mark.server_stress


def test_concurrency(stress_settings, case_directory, server, concurrency):
    """Sustained mixed streaming traffic completes within configured error and latency limits."""
    result = asyncio.run(
        run_load(stress_settings,
                 case_directory,
                 concurrency,
                 stress_settings.duration_s,
                 owned=server))
    assert_load(result, stress_settings)


def test_overload_recovery(stress_settings, case_directory, server):
    """Saturation produces explicit 429 responses and releases capacity for new work."""
    result = asyncio.run(
        run_load(stress_settings,
                 case_directory,
                 stress_settings.overload_concurrency,
                 stress_settings.duration_s,
                 owned=server))
    assert_load(result, stress_settings, overload=True)

    async def recover():
        async with httpx.AsyncClient(headers=headers(),
                                     trust_env=False) as client:
            row = await request_one(client, stress_settings, 0, True)
            write_json(case_directory / 'recovery.json', row)
            check(row['outcome'] == 'success', row)

    asyncio.run(recover())


@pytest.mark.parametrize('cancel_mode', ['timed', 'first_token'])
def test_cancellation_churn(stress_settings, case_directory, server,
                            cancel_mode):
    """Random disconnects coexist with successful traffic and leave admission idle."""
    result = asyncio.run(
        run_load(stress_settings,
                 case_directory,
                 max(stress_settings.concurrency),
                 stress_settings.duration_s,
                 cancellation=cancel_mode,
                 owned=server))
    assert_load(result, stress_settings, cancellation=True)


def test_cancel_drain(stress_settings, case_directory, server):
    """Cancellation of active streams releases backend slots within the drain deadline."""

    async def run():
        limits = httpx.Limits(
            max_connections=max(stress_settings.concurrency) + 2)
        async with httpx.AsyncClient(headers=headers(),
                                     limits=limits,
                                     trust_env=False) as client:
            before = await health(client, stress_settings.url)
            check(idle(before), 'Dedicated server must start idle')
            measurements = []
            for attempt in range(stress_settings.load_repeats):
                # Establish that natural completion exceeds the cancellation budget.
                natural = await request_one(
                    client,
                    replace(stress_settings,
                            max_tokens=stress_settings.cancel_max_tokens), 0,
                    True)
                check(natural['outcome'] == 'success', natural)
                natural_tail = natural['latency_s'] - natural['ttft_s']
                write_json(
                    case_directory / 'calibration.json', {
                        'natural_request': natural,
                        'natural_tail_s': natural_tail,
                        'drain_timeout_s': stress_settings.drain_timeout_s
                    })
                before = await health(client, stress_settings.url)
                has_counters = before.get('capabilities',
                                          {}).get('in_flight_batching')
                if not has_counters and natural_tail <= stress_settings.drain_timeout_s * 2:
                    pytest.skip(
                        'Cancellation calibration inconclusive: increase cancel_max_tokens or use a shorter drain timeout to distinguish cancellation from natural completion'
                    )
                row = await request_one(client, stress_settings, 0, True,
                                        'first_token')
                started = time.monotonic()
                check(row['outcome'] == 'cancelled', row)
                drain_s, after = await wait_idle(client, stress_settings)
                measurements.append({
                    'attempt': attempt,
                    'client': row,
                    'drain_s': time.monotonic() - started,
                    'natural_tail_s': natural_tail,
                    'before': before,
                    'after': after
                })
                write_json(case_directory / 'drain.json', measurements)
                if has_counters:
                    check(
                        after['scheduling']['cancelled'] -
                        before['scheduling']['cancelled'] == 1,
                        'Active request completed instead of cancelling')
            row = await request_one(client, stress_settings, 0, False)
            write_json(case_directory / 'recovery.json', row)
            check(row['outcome'] == 'success', row)

    asyncio.run(run())


def test_soak(stress_settings, case_directory, server):
    """Long-running mixed traffic sustains completions and checks configured RSS limits."""
    result = asyncio.run(
        run_load(stress_settings,
                 case_directory,
                 max(stress_settings.concurrency),
                 stress_settings.soak_s,
                 cancellation='first_token',
                 owned=server))
    assert_load(result, stress_settings, cancellation=True)


@pytest.mark.parametrize('sig', [signal.SIGINT, signal.SIGTERM],
                         ids=['sigint', 'sigterm'])
def test_loading_interrupt(stress_settings, case_directory, load_attempt, sig):
    """A signal during cached runtime loading terminates the owned server within its deadline."""
    if not stress_settings.command:
        pytest.skip('Loading interrupts require a managed server command')
    warmup_dir = case_directory / 'warmup'
    warmup_dir.mkdir()
    warmup = ServerProcess(stress_settings, warmup_dir)
    try:
        warmup.start()
        asyncio.run(wait_ready(stress_settings, warmup))
    finally:
        warmup.close()
    owned = ServerProcess(stress_settings, case_directory)
    try:
        owned.start()

        async def interrupt_load():
            delay = stress_settings.load_delays_s[load_attempt % len(
                stress_settings.load_delays_s)]
            await owned.wait_loading(delay)
            await owned.interrupt(sig)

        asyncio.run(interrupt_load())
    finally:
        owned.close()


@pytest.mark.parametrize('sig', [signal.SIGINT, signal.SIGTERM],
                         ids=['sigint', 'sigterm'])
def test_busy_shutdown(stress_settings, case_directory, server, sig):
    """An active server completes lifespan shutdown within its budget and permits a restart."""
    if server is None:
        pytest.skip('Shutdown requires a managed server command')

    async def shutdown():
        async with httpx.AsyncClient(headers=headers(),
                                     trust_env=False) as client:
            first_token = asyncio.Event()
            tasks = [
                asyncio.create_task(
                    request_one(client,
                                stress_settings,
                                i,
                                True,
                                first_token_event=first_token))
                for i in range(max(stress_settings.concurrency))
            ]
            try:
                await asyncio.wait_for(first_token.wait(),
                                       stress_settings.request_timeout_s)
                data = await health(client, stress_settings.url)
                check(data['active_requests'] > 0,
                      'Generation finished before shutdown')
                write_json(case_directory / 'before_shutdown.json', data)
                await server.interrupt(
                    sig,
                    graceful=True,
                    timeout_s=stress_settings.busy_exit_timeout_s)
            finally:
                await asyncio.wait(tasks, timeout=1)
                for task in tasks:
                    if not task.done():
                        task.cancel()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                write_json(case_directory / 'shutdown_requests.json', [
                    row if isinstance(row, dict) else {
                        'outcome': 'test_cleanup',
                        'error': type(row).__name__
                    } for row in results
                ])

    asyncio.run(shutdown())
    restart_dir = case_directory / 'restart'
    restart_dir.mkdir()
    restarted = ServerProcess(stress_settings, restart_dir)
    try:
        restarted.start()
        asyncio.run(wait_ready(stress_settings, restarted))
    finally:
        restarted.close()


def test_server_benchmark(stress_settings, case_directory, server):
    """The shared online benchmark CLI produces measured throughput and token latency."""
    workload = {
        'run_id':
        case_directory.name,
        'concurrency':
        min(4, max(stress_settings.concurrency)),
        'generation': {
            'temperature': 0,
            'top_p': 1,
            'top_k': 0,
            'ignore_eos': True
        },
        'warmup': {
            'iterations': 1,
            'scope': 'canonical_batch'
        },
        'requests': [{
            'id': str(i),
            'token_ids': [1000] * 32,
            'output_tokens': 32
        } for i in range(16)],
    }
    path = case_directory / 'workload.json'
    write_json(path, workload)
    output = case_directory / 'benchmark'
    with (case_directory / 'benchmark.log').open('w') as log:
        result = subprocess.run(
            [
                sys.executable, '-m', 'scripts.benchmark_runner', '--url',
                stress_settings.url, '--model', stress_settings.model,
                '--workload',
                str(path), '--output',
                str(output), '--request-timeout',
                str(stress_settings.request_timeout_s)
            ],
            cwd=Path(__file__).resolve().parents[2],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=stress_settings.request_timeout_s * 20 + 30)
    check(result.returncode == 0,
          f'Benchmark failed: {case_directory / "benchmark.log"}')
    summary = json.loads((output / 'summary.json').read_text())
    check(
        summary['outcomes'] == {'success': 16} and summary['error_rate'] == 0,
        summary)
    check(summary['run_complete'] and summary['requests'] == 16, summary)
    check(summary['output_tokens'] == 16 * 32, summary)
    check(summary['output_tokens_per_s'] > 0 and summary['ttft_s']['p99'] > 0,
          summary)
