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

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from experimental.server.api.errors import (ServerOverloadedError,
                                            ServerUnavailableError)
from experimental.server.config import ContextCacheConfig
from experimental.server.runtime.engine import (
    LLM, SamplingParams, _native_context_cache_config,
    _set_context_cache_request_policies)
from experimental.server.runtime.engine_client import (_AdmissionController,
                                                       _iterate_sync)
from experimental.server.runtime.engine_layout import EngineType


class _ConcurrentRuntime:

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def handle_request(self, request):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

        time.sleep(0.02)

        with self.lock:
            self.active -= 1
        return request


def _bare_llm(runtime):
    llm = LLM.__new__(LLM)
    llm._runtime = runtime
    llm._admission_sem = threading.Semaphore(1)
    llm._infer_lock = threading.Lock()
    llm._close_lock = threading.Lock()
    llm._closed = False
    return llm


def test_runtime_requests_are_serialized():
    runtime = _ConcurrentRuntime()
    llm = _bare_llm(runtime)
    results = []

    threads = [
        threading.Thread(target=lambda request=request: results.append(
            LLM._handle_request(llm, request))) for request in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == list(range(8))
    assert runtime.max_active == 1


def test_admission_queue_bounds_and_drains_in_order():

    async def exercise():
        admission = _AdmissionController(max_queued_requests=1, timeout=1.0)
        active = await admission.reserve()
        waiting = asyncio.create_task(admission.reserve())
        while admission.waiting != 1:
            await asyncio.sleep(0)

        with pytest.raises(ServerOverloadedError):
            await admission.reserve()

        active.release()
        queued = await waiting
        assert admission.active == 1
        queued.release()
        assert admission.active == 0
        assert admission.waiting == 0

    asyncio.run(exercise())


def test_admission_shutdown_rejects_new_work_and_waits_for_active():

    async def exercise():
        admission = _AdmissionController(max_queued_requests=1, timeout=1.0)
        active = await admission.reserve()
        closing = asyncio.create_task(admission.close())
        while not admission._closing:
            await asyncio.sleep(0)

        with pytest.raises(ServerUnavailableError):
            await admission.reserve()
        assert not closing.done()

        active.release()
        await closing
        with pytest.raises(ServerUnavailableError):
            await admission.reserve()

    asyncio.run(exercise())


def test_admission_wait_timeout_releases_queue_accounting():

    async def exercise():
        admission = _AdmissionController(max_queued_requests=1, timeout=0.01)
        active = await admission.reserve()
        with pytest.raises(ServerOverloadedError, match="timed out"):
            await admission.reserve()
        assert admission.active == 1
        assert admission.waiting == 0
        active.release()

    asyncio.run(exercise())


def test_cancelled_preparation_holds_admission_until_worker_exits():
    entered = threading.Event()
    release = threading.Event()

    class Runtime:

        def _make_generation_request(self, *_args, **_kwargs):
            entered.set()
            release.wait()
            return object()

    from experimental.server.runtime.engine_client import EngineClient

    client = EngineClient.__new__(EngineClient)
    client._llm = Runtime()
    client._admission = _AdmissionController(max_queued_requests=0,
                                             timeout=1.0)

    async def exercise():
        preparing = asyncio.create_task(
            client.prepare_request([{
                "role": "user",
                "content": "first"
            }], SamplingParams()))
        assert await asyncio.to_thread(entered.wait, 1.0)

        preparing.cancel()
        await asyncio.sleep(0)
        assert client.active_requests == 1
        with pytest.raises(ServerOverloadedError):
            await client.prepare_request([{
                "role": "user",
                "content": "second"
            }], SamplingParams())

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await preparing
        assert client.active_requests == 0

    asyncio.run(exercise())


def test_cancelled_generation_holds_admission_until_worker_exits():
    entered = threading.Event()
    release = threading.Event()

    class Runtime:

        def _make_generation_request(self, *_args, **_kwargs):
            return object()

        def _complete_prepared_request(self, *_args, **_kwargs):
            entered.set()
            release.wait()
            return object()

    from experimental.server.runtime.engine_client import EngineClient

    client = EngineClient.__new__(EngineClient)
    client._llm = Runtime()
    client._admission = _AdmissionController(max_queued_requests=0,
                                             timeout=1.0)

    async def exercise():
        messages = [{"role": "user", "content": "first"}]
        prepared = await client.prepare_request(messages, SamplingParams())
        generating = asyncio.create_task(
            client.generate(messages, SamplingParams(), prepared=prepared))
        assert await asyncio.to_thread(entered.wait, 1.0)

        generating.cancel()
        await asyncio.sleep(0)
        assert client.active_requests == 1
        with pytest.raises(ServerOverloadedError):
            await client.prepare_request([{
                "role": "user",
                "content": "second"
            }], SamplingParams())

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await generating
        assert client.active_requests == 0

    asyncio.run(exercise())


def test_stream_close_cancels_and_joins_native_worker():
    worker_entered = threading.Event()
    worker_may_exit = threading.Event()

    class Chunk:
        text = "partial"
        token_ids = [1]
        prompt_token_count = 1
        finished = False
        reason = None
        logprobs = []

    class Channel:

        def __init__(self):
            self.cancelled = False

        def set_skip_special_tokens(self, _enabled):
            pass

        def wait_pop(self, timeout_ms=0):
            return None if self.cancelled else Chunk()

        def is_finished(self):
            return False

        def is_cancelled(self):
            return self.cancelled

        def cancel(self):
            self.cancelled = True

    channel = Channel()

    class RuntimeModule:

        class StreamChannel:

            @staticmethod
            def create():
                return channel

    llm = _bare_llm(object())
    llm._rt = RuntimeModule()

    def handle(_request):
        worker_entered.set()
        worker_may_exit.wait()

    llm._handle_request = handle
    request = type("Request", (), {"stream_channels": None})()
    stream = llm.generate_stream([], prebuilt_request=request)
    assert next(stream).text == "partial"
    assert worker_entered.wait(1.0)

    closing = threading.Thread(target=stream.close)
    closing.start()
    while not channel.cancelled:
        time.sleep(0.001)
    assert closing.is_alive(), "stream returned while native work was active"

    worker_may_exit.set()
    closing.join(1.0)
    assert not closing.is_alive()


def test_async_disconnect_cancels_before_first_stream_chunk():
    worker_entered = threading.Event()

    class Channel:

        def __init__(self):
            self.cancelled = threading.Event()

        def set_skip_special_tokens(self, _enabled):
            pass

        def wait_pop(self, timeout_ms=0):
            self.cancelled.wait(timeout_ms / 1000)
            return None

        def is_finished(self):
            return False

        def is_cancelled(self):
            return self.cancelled.is_set()

        def cancel(self):
            self.cancelled.set()

    channel = Channel()

    class RuntimeModule:

        class StreamChannel:

            @staticmethod
            def create():
                return channel

    llm = _bare_llm(object())
    llm._rt = RuntimeModule()

    def handle(_request):
        worker_entered.set()
        channel.cancelled.wait()

    llm._handle_request = handle
    request = type("Request", (), {"stream_channels": None})()
    stream = llm.generate_stream([], prebuilt_request=request)

    async def exercise():

        async def consume():
            async for _ in _iterate_sync(stream):
                pass

        task = asyncio.create_task(consume())
        assert await asyncio.to_thread(worker_entered.wait, 1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert channel.cancelled.is_set()


def test_close_waits_for_native_entry_and_releases_runtime_once():
    entered = threading.Event()
    release = threading.Event()

    class Runtime:

        def handle_request(self, _request):
            entered.set()
            release.wait()

    llm = _bare_llm(Runtime())
    inference = threading.Thread(target=llm._handle_request, args=(object(), ))
    inference.start()
    assert entered.wait(1.0)

    closing = threading.Thread(target=llm.close)
    closing.start()
    time.sleep(0.01)
    assert closing.is_alive()
    assert llm._runtime is not None

    release.set()
    inference.join(1.0)
    closing.join(1.0)
    assert llm._runtime is None
    llm.close()
    assert llm._runtime is None


def test_context_cache_metrics_are_read_under_runtime_lock():
    expected = object()

    class Runtime:

        def get_context_cache_metrics(self):
            assert llm._infer_lock.locked()
            return expected

    llm = _bare_llm(Runtime())
    assert llm.get_context_cache_metrics() is expected


def test_context_cache_config_maps_to_native_runtime_value():

    class NativeConfig:

        def __init__(self):
            self.enabled = False
            self.max_records = 0
            self.recurrent_snapshot_pool_bytes = 0
            self.partial_kv_snapshot_pool_bytes = 0

    native = _native_context_cache_config(
        SimpleNamespace(ContextCacheConfig=NativeConfig),
        ContextCacheConfig(
            enabled=True,
            max_records=17,
            recurrent_snapshot_pool_bytes=1024,
            partial_kv_snapshot_pool_bytes=2048,
        ),
    )

    assert native.enabled
    assert native.max_records == 17
    assert native.recurrent_snapshot_pool_bytes == 1024
    assert native.partial_kv_snapshot_pool_bytes == 2048


@pytest.mark.parametrize(
    "reuse_context,cache_generated_tokens,expected_lookup,expected_commit",
    [
        (True, True, "use", "all"),
        (False, True, "bypass", "all"),
        (True, False, "use", "prefill"),
    ],
)
def test_context_cache_request_policies(reuse_context, cache_generated_tokens,
                                        expected_lookup, expected_commit):
    runtime = SimpleNamespace(
        ContextCacheLookupPolicy=SimpleNamespace(USE_CACHE="use",
                                                 BYPASS="bypass"),
        ContextCacheCommitPolicy=SimpleNamespace(
            INCLUDING_GENERATED_TOKENS="all",
            PREFILL_STATE_ONLY="prefill",
        ),
    )
    request = SimpleNamespace()

    _set_context_cache_request_policies(
        runtime,
        request,
        SamplingParams(reuse_context=reuse_context,
                       cache_generated_tokens=cache_generated_tokens),
    )

    assert request.context_cache_lookup_policy == expected_lookup
    assert request.context_cache_commit_policy == expected_commit


@pytest.mark.parametrize("engine_type",
                         [EngineType.LLM, EngineType.SPEC_DECODE])
def test_runtime_load_forwards_context_cache_config(monkeypatch, engine_type):
    from experimental.server.runtime import engine as engine_module

    captured = {}

    class NativeConfig:

        def __init__(self):
            self.enabled = False
            self.max_records = 0
            self.recurrent_snapshot_pool_bytes = 0
            self.partial_kv_snapshot_pool_bytes = 0

    class Runtime:

        def __init__(self, *args):
            captured["args"] = args

        @staticmethod
        def capture_decoding_cuda_graph():
            return True

    runtime_module = SimpleNamespace(ContextCacheConfig=NativeConfig,
                                     LLMRuntime=Runtime)
    monkeypatch.setattr(engine_module, "_import_runtime",
                        lambda: runtime_module)

    llm = LLM.__new__(LLM)
    llm._layout = SimpleNamespace(engine_type=engine_type, has_speech=False)
    llm._bundle_dir = "/bundle"
    llm._media_dir = ""
    llm._model_dir = "/model"
    llm._draft_model_dir = "/draft"
    llm._draft_top_k = 4
    llm._draft_step = 3
    llm._verify_tree_size = 8
    llm._dflash_block_size = 0
    llm._context_cache_config = ContextCacheConfig(enabled=True,
                                                   max_records=23)

    llm._load_runtime()

    native = captured["args"][-2 if engine_type ==
                              EngineType.SPEC_DECODE else -1]
    assert native.enabled
    assert native.max_records == 23
    if engine_type == EngineType.SPEC_DECODE:
        assert captured["args"][-1] == 0
