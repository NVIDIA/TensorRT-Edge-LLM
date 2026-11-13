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
"""Bounded HTTP load and lifecycle measurement for a dedicated Edge-LLM server."""

import asyncio
import json
import logging
import math
import os
import random
import re
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import psutil

from scripts.benchmark_runner.metrics import summarize


def check(condition, message):
    """Enforce stress contracts even when Python optimization is enabled."""
    if not condition:
        raise AssertionError(message)


@dataclass
class Settings:
    """Workload, process ownership, and explicit pass/fail limits."""

    url: str = 'http://127.0.0.1:18080'
    model: str = 'stress-model'
    command: list = field(default_factory=list)
    cwd: str = ''
    concurrency: list = field(default_factory=lambda: [1, 4, 16])
    duration_s: float = 60
    soak_s: float = 600
    request_timeout_s: float = 60
    ready_timeout_s: float = 600
    drain_timeout_s: float = 5
    exit_timeout_s: float = 2
    busy_exit_timeout_s: float = 30
    load_repeats: int = 3
    load_delays_s: list = field(default_factory=lambda: [0, 0.1, 0.3])
    load_pattern: str = 'Loading runtime bundle from'
    cancellation_fraction: float = 0.3
    cancel_after_s: float = 0.05
    overload_concurrency: int = 64
    max_tokens: int = 128
    cancel_max_tokens: int = 1024
    min_successes: int = 4
    seed: int = 1802
    max_error_rate: float = 0
    max_p99_s: float | None = None
    max_rss_growth_mb: float | None = None
    prompts: list = field(default_factory=lambda: [
        'List the integers from 1 to 500, in order, one per line.',
        'Describe how a computer works in detail. ' * 32,
    ])

    @classmethod
    def read(cls, path):
        """Reject invalid settings before creating processes or sending load."""
        obj = cls(**json.loads(Path(path).read_text()))
        for name in ('duration_s', 'soak_s', 'request_timeout_s',
                     'ready_timeout_s', 'drain_timeout_s', 'exit_timeout_s',
                     'busy_exit_timeout_s', 'cancel_after_s'):
            value = getattr(obj, name)
            if isinstance(value, bool) or not isinstance(
                    value,
                (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('max_p99_s', 'max_rss_growth_mb'):
            value = getattr(obj, name)
            if value is not None and (isinstance(value, bool)
                                      or not isinstance(value, (int, float)) or
                                      not math.isfinite(value) or value <= 0):
                raise ValueError(
                    f'{name} must be finite and positive when set')
        for name in ('load_repeats', 'overload_concurrency', 'max_tokens',
                     'cancel_max_tokens', 'min_successes'):
            value = getattr(obj, name)
            if isinstance(value,
                          bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not obj.concurrency or any(
                isinstance(c, bool) or not isinstance(c, int) or c < 1
                for c in obj.concurrency):
            raise ValueError('concurrency must contain positive integers')
        for name in ('max_error_rate', 'cancellation_fraction'):
            value = getattr(obj, name)
            if isinstance(value, bool) or not isinstance(
                    value,
                (int,
                 float)) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f'{name} must be in [0, 1)')
        if obj.cancellation_fraction == 0:
            raise ValueError('cancellation_fraction must be positive')
        if not obj.prompts or any(not isinstance(p, str) or not p
                                  for p in obj.prompts):
            raise ValueError('prompts must contain nonempty strings')
        if not isinstance(obj.command, list) or any(
                not isinstance(a, str) or not a for a in obj.command):
            raise ValueError('command must be an argv list')
        if not obj.load_delays_s or any(
                isinstance(d, bool) or not isinstance(d, (float, int))
                or not math.isfinite(d) or d < 0 for d in obj.load_delays_s):
            raise ValueError(
                'load_delays_s must contain finite nonnegative delays')
        parts = urlsplit(obj.url)
        if parts.scheme not in (
                'http', 'https') or not parts.hostname or parts.path not in (
                    '', '/') or parts.username or parts.password:
            raise ValueError('url must be an HTTP origin')
        if obj.command and parts.hostname not in ('127.0.0.1', 'localhost'):
            raise ValueError('Managed server requires a loopback URL')
        re.compile(obj.load_pattern)
        return obj


def headers():
    """Read optional credentials from the environment, excluding report metadata."""
    token = os.environ.get('SERVER_BENCH_API_KEY')
    return {'Authorization': f'Bearer {token}'} if token else {}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


class ServerProcess:
    """Own exactly the process group started by this test."""

    def __init__(self, settings, directory):
        self.settings = settings
        self.directory = directory
        self.process = None
        self.log = None
        self.log_path = directory / 'server.log'

    def start(self):
        parts = urlsplit(self.settings.url)
        port = parts.port or (443 if parts.scheme == 'https' else 80)
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', port))
        self.log = self.log_path.open('wb')
        try:
            self.process = subprocess.Popen(
                self.settings.command,
                cwd=self.settings.cwd or None,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={
                    **os.environ, 'PYTHONUNBUFFERED': '1'
                })
        except BaseException:
            self.log.close()
            raise

    def rss_mb(self):
        try:
            root = psutil.Process(self.process.pid)
            return sum(p.memory_info().rss
                       for p in [root, *root.children(recursive=True)]) / 2**20
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    async def wait_loading(self, delay):
        deadline = time.monotonic() + self.settings.ready_timeout_s
        pattern = re.compile(self.settings.load_pattern)
        with self.log_path.open(errors='replace') as stream:
            observed = ''
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise AssertionError(
                        f'Server exited during startup: {self.log_path}')
                observed = (observed + stream.read())[-65536:]
                if pattern.search(observed):
                    await asyncio.sleep(delay)
                    observed += stream.read()
                    check(
                        'Loaded model=' not in observed
                        and 'Application startup complete' not in observed,
                        'Loading window missed')
                    return
                await asyncio.sleep(0.005)
        raise AssertionError(f'Loading marker timeout: {self.log_path}')

    def group_alive(self):
        try:
            os.killpg(self.process.pid, 0)
            return True
        except ProcessLookupError:
            return False

    async def interrupt(self, sig, *, graceful=False, timeout_s=None):
        if self.process.poll() is not None:
            raise AssertionError('Server exited before the measured signal')
        started = time.monotonic()
        timeout_s = self.settings.exit_timeout_s if timeout_s is None else timeout_s
        loaded_at_signal = 'Loaded model=' in self.log_path.read_text(
            errors='replace')
        os.killpg(self.process.pid, sig)
        while time.monotonic() - started < timeout_s:
            if self.process.poll() is not None and not self.group_alive():
                break
            await asyncio.sleep(0.01)
        result = {
            'signal':
            signal.Signals(sig).name,
            'exit_s':
            time.monotonic() - started,
            'timeout_s':
            timeout_s,
            'loaded_at_signal':
            loaded_at_signal,
            'returncode':
            self.process.poll(),
            'timed_out':
            self.process.poll() is None or self.group_alive(),
            'loaded_before_exit':
            'Loaded model=' in self.log_path.read_text(errors='replace')
        }
        write_json(self.directory / 'exit.json', result)
        check(not result['timed_out'], f'Shutdown exceeded {timeout_s}s')
        check(result['returncode'] in (0, 130, 143, -sig),
              f'Unexpected exit: {result}')
        if graceful:
            check(
                'Application shutdown complete'
                in self.log_path.read_text(errors='replace'),
                'Server lifespan shutdown did not complete')
        return result

    def close(self):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logging.error('Cleanup timed out for owned server PID %s',
                              self.process.pid)
        if self.log:
            self.log.close()
        if self.process is not None:
            check(not self.group_alive(),
                  f'Owned process group survives cleanup: {self.process.pid}')


async def health(client, url):
    response = await client.get(url + '/health', timeout=2)
    response.raise_for_status()
    data = response.json()
    for key in ('active_requests', 'queued_requests'):
        check(
            isinstance(data.get(key), int) and data[key] >= 0,
            f'Invalid health gauge: {key}')
    return data


def idle(data):
    scheduling = data.get('scheduling') or {}
    return (data['active_requests'] == 0 and data['queued_requests'] == 0
            and scheduling.get('queued', 0) == 0
            and scheduling.get('resident', 0) == 0)


async def wait_ready(settings, owned=None):
    deadline = time.monotonic() + settings.ready_timeout_s
    async with httpx.AsyncClient(headers=headers(), trust_env=False) as client:
        while time.monotonic() < deadline:
            if owned and owned.process.poll() is not None:
                raise AssertionError(f'Server exited: {owned.log_path}')
            try:
                response = await client.get(settings.url + '/health/ready',
                                            timeout=2)
                if response.status_code == 200 and response.json().get(
                        'status') == 'ready':
                    return await health(client, settings.url)
            except (httpx.HTTPError, ValueError):
                pass
            await asyncio.sleep(0.1)
    raise AssertionError('Server readiness timeout')


async def wait_idle(client, settings):
    started = time.monotonic()
    last = None
    while time.monotonic() - started < settings.drain_timeout_s:
        last = await health(client, settings.url)
        if idle(last):
            return time.monotonic() - started, last
        await asyncio.sleep(0.05)
    raise AssertionError(f'Server failed to drain: {last}')


async def request_one(client,
                      settings,
                      index,
                      stream,
                      cancel_mode=None,
                      first_token_event=None,
                      prompt_index=None):
    started = time.monotonic()
    prompt_index = index % len(
        settings.prompts) if prompt_index is None else prompt_index
    row = {
        'id': index,
        'prompt_index': prompt_index,
        'outcome': 'error',
        'status': None,
        'ttft_s': None,
        'output_tokens': None,
        'stream': stream,
        'cancel_mode': cancel_mode
    }
    payload = {
        'model': settings.model,
        'messages': [{
            'role': 'user',
            'content': settings.prompts[prompt_index]
        }],
        'max_tokens':
        settings.cancel_max_tokens if cancel_mode else settings.max_tokens,
        'temperature': 0,
        'stream': stream
    }
    if stream:
        payload['stream_options'] = {'include_usage': True}

    async def perform():
        async with client.stream('POST',
                                 settings.url + '/v1/chat/completions',
                                 json=payload) as response:
            row['status'] = response.status_code
            if response.status_code == 429:
                row['outcome'] = 'rejected'
                return
            response.raise_for_status()
            if not stream:
                data = json.loads(await response.aread())
                check(
                    data.get('choices')
                    and data['choices'][0].get('finish_reason')
                    in ('stop', 'length', 'tool_calls'), 'Incomplete response')
                usage = data.get('usage') or {}
            else:
                done, finished = False, False
                usage = {}
                async for line in response.aiter_lines():
                    if not line or line.startswith(':'):
                        continue
                    if not line.startswith('data:'):
                        continue
                    raw = line[5:].strip()
                    if raw == '[DONE]':
                        done = True
                        break
                    event = json.loads(raw)
                    check(not event.get('error'),
                          f'Stream error: {event.get("error")}')
                    for choice in event.get('choices', []):
                        delta = choice.get('delta') or {}
                        if delta.get('content') or delta.get(
                                'reasoning_content') or delta.get(
                                    'tool_calls'):
                            if row['ttft_s'] is None:
                                row['ttft_s'] = time.monotonic() - started
                                if first_token_event is not None:
                                    first_token_event.set()
                            if cancel_mode == 'first_token':
                                row['outcome'] = 'cancelled'
                                return
                        if choice.get('finish_reason'):
                            check(
                                choice['finish_reason']
                                in ('stop', 'length', 'tool_calls'),
                                'Generation failed')
                            finished = True
                    if event.get('usage'):
                        usage = event['usage']
                check(done and finished, 'Truncated SSE stream')
                check(row['ttft_s'] is not None, 'Empty generation')
            count = usage.get('completion_tokens')
            check(
                isinstance(count, int) and not isinstance(count, bool)
                and count > 0, 'Missing output token usage')
            row['output_tokens'] = count
            row['outcome'] = 'success'

    try:
        timeout = settings.cancel_after_s if cancel_mode == 'timed' else settings.request_timeout_s
        await asyncio.wait_for(perform(), timeout=timeout)
    except asyncio.TimeoutError:
        row['outcome'] = 'cancelled' if cancel_mode == 'timed' else 'error'
        row['error'] = 'client_disconnect' if cancel_mode == 'timed' else 'request_deadline'
    except (httpx.HTTPError, ValueError, AssertionError, KeyError,
            TypeError) as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
    row['latency_s'] = time.monotonic() - started
    return row


async def run_load(settings,
                   directory,
                   concurrency,
                   duration,
                   *,
                   cancellation=None,
                   owned=None):
    """Apply bounded closed-loop load and persist each result as it completes."""
    limits = httpx.Limits(max_connections=concurrency + 4,
                          max_keepalive_connections=concurrency + 4)
    samples, records = [], []
    stop = asyncio.Event()
    async with httpx.AsyncClient(headers=headers(),
                                 limits=limits,
                                 timeout=settings.request_timeout_s,
                                 trust_env=False) as client:
        baseline = await health(client, settings.url)
        check(idle(baseline), 'Use a dedicated idle server')
        warmed = await asyncio.gather(
            *(request_one(client, settings, i, i % 2 == 0)
              for i in range(min(concurrency, max(settings.concurrency)))))
        write_json(directory / 'warmup.json', warmed)
        check(all(row['outcome'] == 'success' for row in warmed),
              'Warmup failed')
        await wait_idle(client, settings)
        baseline = await health(client, settings.url)
        baseline_rss = owned.rss_mb() if owned else None
        started = time.monotonic()
        deadline = started + duration
        with (directory / 'requests.jsonl').open('w') as output, (
                directory / 'health.jsonl').open('w') as observations:

            async def worker(worker_id):
                rng = random.Random(settings.seed + worker_id)
                sequence = 0
                while time.monotonic() < deadline and not stop.is_set():
                    index = worker_id + sequence * concurrency
                    mode = cancellation if cancellation and rng.random(
                    ) < settings.cancellation_fraction else None
                    stream = True if mode else rng.choice([True, False])
                    row = await request_one(client,
                                            settings,
                                            index,
                                            stream,
                                            mode,
                                            prompt_index=rng.randrange(
                                                len(settings.prompts)))
                    records.append(row)
                    output.write(json.dumps(row) + '\n')
                    output.flush()
                    sequence += 1
                    if owned and owned.process.poll() is not None:
                        stop.set()
                    if row['outcome'] in ('error', 'rejected'):
                        await asyncio.sleep(0.01)

            async def observe():
                while not stop.is_set():
                    sample = {'elapsed_s': time.monotonic() - started}
                    try:
                        sample['health'] = await health(client, settings.url)
                    except (httpx.HTTPError, ValueError,
                            AssertionError) as exc:
                        sample['error'] = str(exc)
                        if len(samples) >= 2 and all(
                                'error' in prior for prior in samples[-2:]):
                            stop.set()
                    if owned:
                        sample['rss_mb'] = owned.rss_mb()
                    samples.append(sample)
                    observations.write(json.dumps(sample) + '\n')
                    observations.flush()
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=0.5)
                    except asyncio.TimeoutError:
                        pass

            observer = asyncio.create_task(observe())
            try:
                await asyncio.gather(*(worker(i) for i in range(concurrency)))
            finally:
                stop.set()
                await observer
        elapsed = time.monotonic() - started
        summary = summarize(records, elapsed)
        summary.update({
            'concurrency': concurrency,
            'cancellation_mode': cancellation,
            'offered_duration_s': duration,
            'health_before': baseline,
            'health_errors': sum('error' in s for s in samples),
            'early_stop': elapsed < duration
        })
        write_json(directory / 'summary.json', summary)
        try:
            summary['cohort_drain_s'], summary[
                'health_after'] = await wait_idle(client, settings)
        finally:
            write_json(directory / 'summary.json', summary)
        rss = [s['rss_mb'] for s in samples if s.get('rss_mb') is not None]
        if rss:
            summary['rss_peak_mb'] = max(rss)
        final_rss = owned.rss_mb() if owned else None
        if baseline_rss is not None and final_rss is not None:
            summary['rss_baseline_mb'] = baseline_rss
            summary['rss_after_drain_mb'] = final_rss
            summary['rss_growth_mb'] = final_rss - baseline_rss
        summary['rss_limit_mb'] = settings.max_rss_growth_mb
        write_json(directory / 'summary.json', summary)
        return summary


def assert_load(summary, settings, *, overload=False, cancellation=False):
    """Require useful work, bounded errors, recovery, and requested fault coverage."""
    counts = summary['outcomes']
    check(
        counts.get('success', 0) >= settings.min_successes,
        'Insufficient completed workload')
    check(not summary.get('early_stop'),
          'Load aborted before its requested duration')
    check(summary['error_rate'] <= settings.max_error_rate, summary)
    check(summary['health_errors'] == 0, 'Health endpoint failed under load')
    before = summary.get('health_before', {})
    after = summary.get('health_after', {})
    native = before.get('capabilities', {}).get('in_flight_batching')
    if native:
        check(after['scheduling']['failed'] == before['scheduling']['failed'],
              'Native engine request failed')
    if overload:
        check(
            counts.get('rejected', 0) > 0,
            'Overload was not reached; increase concurrency or output length')
    else:
        check(counts.get('rejected', 0) == 0, 'Unexpected admission rejection')
    if cancellation:
        check(counts.get('cancelled', 0) > 0, 'Cancellation was not exercised')
        if native:
            cancelled = after['scheduling']['cancelled'] - before[
                'scheduling']['cancelled']
            check(cancelled > 0, 'Native engine cancellation was not observed')
            if summary.get('cancellation_mode') == 'first_token':
                check(
                    cancelled >= 0.9 * counts['cancelled'],
                    'Native cancellation covered fewer than 90% of first-token disconnects'
                )
    if settings.max_p99_s is not None:
        check(summary['latency_s']['p99'] <= settings.max_p99_s, summary)
    if settings.max_rss_growth_mb is not None:
        check('rss_growth_mb' in summary,
              'RSS gate requires a managed local process')
        check(summary['rss_growth_mb'] <= settings.max_rss_growth_mb, summary)
