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
"""Unit tests for non-streaming request batching in the experimental server."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from experimental.server.api.errors import ServerOverloadedError
from experimental.server.config import parse_server_config
from experimental.server.runtime.batching import (BATCH_COMPATIBILITY_FIELDS,
                                                  BatcherOverflow,
                                                  RequestBatcher,
                                                  resolve_batch_size)
from experimental.server.runtime.engine import LLM
from experimental.server.runtime.engine_client import _AdmissionController


class _Row:

    def __init__(self, text: str) -> None:
        self.text = text
        self.image_buffers = []


class _Request:
    """Duck-typed stand-in for the pybind ``LLMGenerationRequest``."""

    def __init__(self, *rows: str, **settings) -> None:
        for field in BATCH_COMPATIBILITY_FIELDS:
            setattr(self, field, settings.get(field, 0))
        self.requests = [_Row(text) for text in rows]
        self.stream_channels = []


class _Response:

    def __init__(self, rows) -> None:
        self.output_texts = [f"echo:{row.text}" for row in rows]
        self.output_ids = [[len(row.text)] for row in rows]
        self.finish_reasons = ["stop" for _ in rows]
        self.logprobs = [None for _ in rows]
        self.prompt_token_counts = [1 for _ in rows]


class _Handler:

    def __init__(self, delay: float = 0.0, gate=None) -> None:
        self.calls = []
        self.lock = threading.Lock()
        self.delay = delay
        self.gate = gate

    def __call__(self, request):
        with self.lock:
            self.calls.append(list(request.requests))
        if self.gate is not None:
            self.gate.wait(5.0)
        if self.delay:
            time.sleep(self.delay)
        return _Response(request.requests)


def _submit_all(batcher, requests):
    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
        return list(pool.map(batcher.submit, requests))


def test_resolve_batch_size_caps_to_engine():
    assert resolve_batch_size(4, None) == 4
    assert resolve_batch_size(4, 2) == 2
    assert resolve_batch_size(4, 8) == 4
    assert resolve_batch_size(0, None) == 1


def test_compatible_requests_share_one_runtime_call():
    gate = threading.Event()
    handler = _Handler(gate=gate)
    batcher = RequestBatcher(handler, max_batch_size=4, timeout_ms=200)
    try:
        requests = [_Request("a"), _Request("b"), _Request("c")]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(batcher.submit, r) for r in requests]
            # Let all three enqueue before the worker's timeout elapses.
            deadline = time.monotonic() + 2.0
            while batcher.pending + len(handler.calls) < 1:
                assert time.monotonic() < deadline
                time.sleep(0.005)
            gate.set()
            results = [f.result(timeout=5.0) for f in futures]
    finally:
        batcher.close()

    assert len(handler.calls) == 1
    assert [row.text for row in handler.calls[0]] == ["a", "b", "c"]
    assert [r.output_texts for r in results] == [["echo:a"], ["echo:b"],
                                                 ["echo:c"]]
    assert [r.output_ids for r in results] == [[[1]], [[1]], [[1]]]
    assert all(r.finish_reasons == ["stop"] for r in results)
    assert all(r.prompt_token_counts == [1] for r in results)


def test_incompatible_settings_run_in_separate_calls():
    handler = _Handler(delay=0.02)
    batcher = RequestBatcher(handler, max_batch_size=4, timeout_ms=100)
    try:
        results = _submit_all(batcher, [
            _Request("a", temperature=0.0),
            _Request("b", temperature=0.7),
            _Request("c", temperature=0.0),
        ])
    finally:
        batcher.close()

    assert len(handler.calls) == 2
    batched = sorted(len(call) for call in handler.calls)
    assert batched == [1, 2]
    assert sorted(r.output_texts[0]
                  for r in results) == ["echo:a", "echo:b", "echo:c"]


def test_batch_size_bounds_rows_per_call():
    handler = _Handler(delay=0.02)
    batcher = RequestBatcher(handler, max_batch_size=2, timeout_ms=100)
    try:
        _submit_all(batcher, [_Request(str(i)) for i in range(4)])
    finally:
        batcher.close()

    assert all(len(call) <= 2 for call in handler.calls)
    assert sum(len(call) for call in handler.calls) == 4


def test_multi_row_requests_are_sliced_back_correctly():
    handler = _Handler(delay=0.02)
    batcher = RequestBatcher(handler, max_batch_size=8, timeout_ms=100)
    try:
        results = _submit_all(
            batcher,
            [_Request("a", "b"),
             _Request("c"),
             _Request("d", "e", "f")])
    finally:
        batcher.close()

    texts = sorted(tuple(r.output_texts) for r in results)
    assert texts == [("echo:a", "echo:b"), ("echo:c", ),
                     ("echo:d", "echo:e", "echo:f")]


def test_pending_queue_overflow_is_reported():
    gate = threading.Event()
    handler = _Handler(gate=gate)
    batcher = RequestBatcher(handler,
                             max_batch_size=1,
                             timeout_ms=0,
                             max_pending=1)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            running = pool.submit(batcher.submit, _Request("a"))
            deadline = time.monotonic() + 2.0
            while not handler.calls:
                assert time.monotonic() < deadline
                time.sleep(0.005)
            queued = pool.submit(batcher.submit, _Request("b"))
            deadline = time.monotonic() + 2.0
            while batcher.pending < 1:
                assert time.monotonic() < deadline
                time.sleep(0.005)
            with pytest.raises(BatcherOverflow):
                batcher.submit(_Request("c"))
            gate.set()
            assert running.result(5.0).output_texts == ["echo:a"]
            assert queued.result(5.0).output_texts == ["echo:b"]
    finally:
        batcher.close()


def test_runtime_failure_propagates_to_every_member():

    def failing(_request):
        raise RuntimeError("boom")

    batcher = RequestBatcher(failing, max_batch_size=4, timeout_ms=50)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(batcher.submit, _Request("a")),
                pool.submit(batcher.submit, _Request("b")),
            ]
            for future in futures:
                with pytest.raises(RuntimeError, match="boom"):
                    future.result(5.0)
    finally:
        batcher.close()
    # The worker survives a failed batch and keeps serving.
    batcher2 = RequestBatcher(_Handler(), max_batch_size=1, timeout_ms=0)
    try:
        assert batcher2.submit(_Request("z")).output_texts == ["echo:z"]
    finally:
        batcher2.close()
    with pytest.raises(RuntimeError, match="closed"):
        batcher2.submit(_Request("late"))


class _Runtime:

    def __init__(self) -> None:
        self.calls = []

    def handle_request(self, request):
        self.calls.append(list(request.requests))
        time.sleep(0.02)
        return _Response(request.requests)


def _bare_llm(runtime):
    llm = LLM.__new__(LLM)
    llm._runtime = runtime
    llm._admission_sem = threading.Semaphore(1)
    llm._infer_lock = threading.Lock()
    llm._close_lock = threading.Lock()
    llm._closed = False
    llm._batcher = None
    return llm


def test_llm_runs_generation_through_batcher_and_closes_it():
    runtime = _Runtime()
    llm = _bare_llm(runtime)
    llm._batcher = RequestBatcher(llm._handle_request,
                                  max_batch_size=4,
                                  timeout_ms=100)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(llm._run_generation, _Request("a")),
            pool.submit(llm._run_generation, _Request("b")),
        ]
        results = [f.result(5.0) for f in futures]

    assert len(runtime.calls) == 1
    assert sorted(r.output_texts[0] for r in results) == ["echo:a", "echo:b"]

    # Streaming requests bypass the batcher even when it is enabled.
    streaming = _Request("s")
    streaming.stream_channels = [object()]
    assert llm._run_generation(streaming).output_texts == ["echo:s"]
    assert len(runtime.calls) == 2

    llm.close()
    assert llm.batcher is None
    assert llm._runtime is None


def test_llm_without_batcher_calls_runtime_directly():
    runtime = _Runtime()
    llm = _bare_llm(runtime)
    assert llm._run_generation(_Request("a")).output_texts == ["echo:a"]
    assert len(runtime.calls) == 1


def test_admission_allows_batch_size_concurrent_leases():

    async def exercise():
        admission = _AdmissionController(max_queued_requests=1,
                                         timeout=1.0,
                                         max_active=2)
        first = await admission.reserve()
        second = await admission.reserve()
        assert admission.active == 2
        waiting = asyncio.create_task(admission.reserve())
        while admission.waiting != 1:
            await asyncio.sleep(0)
        with pytest.raises(ServerOverloadedError):
            await admission.reserve()
        first.release()
        third = await waiting
        assert admission.active == 2
        second.release()
        third.release()
        assert admission.active == 0
        closing = asyncio.create_task(admission.close())
        await closing
        with pytest.raises(Exception):
            await admission.reserve()

    asyncio.run(exercise())


def test_server_config_parses_batching_flags():
    config = parse_server_config([
        "/model", "--enable-batching", "--batch-timeout-ms", "5",
        "--max-queue-batch-size", "2", "--max-batch-size", "4"
    ])
    kwargs = config.model.llm_kwargs()
    assert kwargs["enable_batching"] is True
    assert kwargs["batch_timeout_ms"] == 5.0
    assert kwargs["max_queue_batch_size"] == 2
    assert kwargs["max_batch_size"] == 4

    default = parse_server_config(["/model"]).model.llm_kwargs()
    assert default["enable_batching"] is False
    assert default["max_queue_batch_size"] is None
