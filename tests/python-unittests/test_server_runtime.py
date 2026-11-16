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
import json
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from tensorrt_edgellm.server.api.errors import (ServerOverloadedError,
                                                ServerUnavailableError)
from tensorrt_edgellm.server.config import ContextCacheConfig, PleRuntimeConfig
from tensorrt_edgellm.server.runtime import engine
from tensorrt_edgellm.server.runtime.engine import (
    LLM, SamplingParams, _convert_messages_to_cpp,
    _native_context_cache_config, _resolve_spec_decode_runtime_options,
    _set_context_cache_request_policies)
from tensorrt_edgellm.server.runtime.engine_client import (
    _AdmissionController, _IFBAdmissionGate, _iterate_sync)
from tensorrt_edgellm.server.runtime.engine_layout import EngineType


@pytest.mark.parametrize("top_k, expected_verify", [(1, 4), (2, 13)])
def test_eagle_runtime_default_verify_width_follows_layout(
        tmp_path, top_k, expected_verify):
    # Given an EAGLE artifact supporting both layouts and three proposal tokens.
    (tmp_path / "base_config.json").write_text(
        json.dumps({
            "spec_decode_type": "eagle3",
            "builder_config": {
                "max_spec_verify_tokens": 13
            },
        }))
    (tmp_path / "draft_config.json").write_text("{}")
    # When the caller leaves the verification width unspecified.
    options = _resolve_spec_decode_runtime_options(str(tmp_path), "eagle3",
                                                   None, top_k, 3, None, None)
    # Then chain includes exactly its root, while tree uses the artifact capacity.
    assert options.verify_tokens == expected_verify


@pytest.mark.parametrize("method, expected_top_k, expected_verify",
                         [("eagle3", 10, 13), ("mtp", 1, 4),
                          ("gemma4_mtp", 1, 4)])
def test_iterative_runtime_preserves_default_layout(tmp_path, method,
                                                    expected_top_k,
                                                    expected_verify):
    # Given an iterative speculative artifact and three proposal tokens.
    (tmp_path / "base_config.json").write_text(
        json.dumps({
            "spec_decode_type": method,
            "builder_config": {
                "max_spec_verify_tokens": 13
            },
        }))
    (tmp_path / "draft_config.json").write_text("{}")
    # When neither candidate top-k nor verification width is specified.
    options = _resolve_spec_decode_runtime_options(str(tmp_path), method, None,
                                                   None, 3, None, None)
    # Then EAGLE defaults to tree and MTP variants retain their chain default.
    assert options.top_k == expected_top_k
    assert options.verify_tokens == expected_verify


def test_gemma4_mtp_tree_runtime_options_follow_compiled_profile(tmp_path):
    (tmp_path / "base_config.json").write_text(
        json.dumps({
            "spec_decode_type": "gemma4_mtp",
            "builder_config": {
                "max_spec_verify_tokens": 13
            },
        }))
    (tmp_path / "draft_config.json").write_text("{}")

    options = _resolve_spec_decode_runtime_options(str(tmp_path), "mtp", None,
                                                   4, 3, None, None)

    assert options.top_k == 4
    assert options.draft_step == 3
    assert options.proposal_tokens is None
    assert options.verify_tokens == 13


@pytest.mark.parametrize("builder_config", [{"max_verify_tree_size": 13}, {}])
def test_spec_runtime_rejects_missing_verify_capacity_key(
        tmp_path, builder_config):
    """The unified server contract requires the fixed-ABI verify-capacity key."""
    (tmp_path / "base_config.json").write_text(
        json.dumps({
            "spec_decode_type": "mtp",
            "builder_config": builder_config,
        }))
    (tmp_path / "draft_config.json").write_text("{}")

    with pytest.raises(ValueError, match="max_spec_verify_tokens"):
        _resolve_spec_decode_runtime_options(str(tmp_path), "mtp", None, 2, 3,
                                             None, None)


def _write_cached_draft_bundle(path, method, base_max, draft_max, mode):
    (path / "base_config.json").write_text(
        json.dumps({
            "spec_decode_type": method,
            "builder_config": {
                "max_spec_verify_tokens": base_max
            },
        }))
    (path / "draft_config.json").write_text(
        json.dumps({
            "builder_config": {
                "max_spec_draft_tokens": draft_max
            },
            f"{method}_config": mode,
        }))


@pytest.mark.parametrize("method", ["dflash", "jetspec", "domino", "dspark"])
@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("proposal, verify", [(None, None), (7, None),
                                              (None, 8), (7, 8)])
def test_block_runtime_delegates_geometry_to_deployment_config(
        tmp_path, method, top_k, proposal, verify):
    # Given block metadata whose omitted and explicit proposal limits differ.
    _write_cached_draft_bundle(tmp_path, method, 5, 8, {
        "block_size": 8,
        "sample_from_anchor": False
    })
    # When transporting caller options, including omitted values.
    options = _resolve_spec_decode_runtime_options(str(tmp_path), method, None,
                                                   top_k, None, proposal,
                                                   verify)
    # Then the common C++ resolver owns defaults and capacity validation.
    assert options.top_k == top_k
    assert options.proposal_tokens == proposal
    assert options.verify_tokens == verify


@pytest.mark.parametrize("method", ["dflash", "jetspec", "domino", "dspark"])
def test_block_runtime_transports_normalized_proposal_alias(tmp_path, method):
    # Given a caller using the server proposal-count shorthand.
    _write_cached_draft_bundle(tmp_path, method, 9, 8, {"block_size": 8})
    # When preparing native runtime arguments without an explicit top-k.
    options = _resolve_spec_decode_runtime_options(str(tmp_path), method, 7,
                                                   None, None, None, None)
    # Then chain is selected and the shorthand reaches the common resolver.
    assert options.top_k == 1
    assert options.proposal_tokens == 7
    assert options.verify_tokens is None


@pytest.mark.parametrize("proposal", [1, 3])
def test_iterative_runtime_rejects_explicit_proposal_tokens(
        tmp_path, proposal):
    # Given an iterative bundle, when P is explicitly supplied, then it is
    # rejected even when it equals S or JSON also supplies the step.
    (tmp_path / "base_config.json").write_text(
        json.dumps({
            "spec_decode_type": "mtp",
            "builder_config": {
                "max_spec_verify_tokens": 13
            }
        }))
    with pytest.raises(ValueError, match="use draft_step"):
        _resolve_spec_decode_runtime_options(str(tmp_path), "mtp", 3, None, 3,
                                             proposal, None)


@pytest.mark.parametrize("draft_step", [None, 1, 0, 2])
def test_block_runtime_only_accepts_one_draft_step(tmp_path, draft_step):
    # Given a block bundle, when S is omitted or one it normalizes to one;
    # all other explicit values are rejected.
    _write_cached_draft_bundle(tmp_path, "dflash", 8, 8, {"block_size": 8})
    if draft_step in (None, 1):
        options = _resolve_spec_decode_runtime_options(str(tmp_path), "dflash",
                                                       None, None, draft_step,
                                                       None, None)
        assert options.draft_step == 1
    else:
        with pytest.raises(ValueError, match="only support draft_step=1"):
            _resolve_spec_decode_runtime_options(str(tmp_path), "dflash", None,
                                                 None, draft_step, None, None)


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


class _NativeMessageValue:

    def __init__(self, *values):
        self.values = values


def _message_runtime():
    return SimpleNamespace(Message=_NativeMessageValue,
                           MessageContent=_NativeMessageValue,
                           MessageToolCall=_NativeMessageValue)


def test_message_conversion_passes_tool_call_arguments_as_objects():
    """OpenAI string arguments reach the renderer as JSON object text, and
    empty or non-object arguments as an empty object, as in vLLM."""

    def tool_call(arguments):
        return {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "weather",
                "arguments": arguments
            },
        }

    messages = _convert_messages_to_cpp(
        _message_runtime(),
        [{
            "role":
            "assistant",
            "content":
            None,
            "tool_calls": [
                tool_call('{"city": "Paris", "days": 3}'),
                tool_call(""),
                tool_call("[1, 2]")
            ],
        }],
    )

    assert messages[0].has_tool_calls
    assert messages[0].has_content
    assert messages[0].content_is_null
    assert [call.arguments for call in messages[0].tool_calls
            ] == ['{"city":"Paris","days":3}', "{}", "{}"]


def test_message_conversion_preserves_absent_content_and_object_arguments():
    messages = _convert_messages_to_cpp(
        _message_runtime(),
        [{
            "role":
            "assistant",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": {
                        "city": "Paris"
                    },
                },
            }],
        }],
    )

    assert not messages[0].has_content
    assert not messages[0].content_is_null
    assert messages[0].tool_calls[0].arguments == '{"city":"Paris"}'


def test_message_conversion_ignores_empty_tool_calls():
    messages = _convert_messages_to_cpp(
        _message_runtime(),
        [{
            "role": "assistant",
            "content": "done",
            "tool_calls": []
        }],
    )

    assert not messages[0].has_tool_calls
    assert messages[0].tool_calls == []


def test_message_conversion_rejects_invalid_tool_call_json():
    with pytest.raises(ValueError,
                       match="Invalid JSON in assistant tool-call arguments"):
        _convert_messages_to_cpp(
            _message_runtime(),
            [{
                "role":
                "assistant",
                "content":
                None,
                "tool_calls": [{
                    "function": {
                        "name": "weather",
                        "arguments": "{invalid",
                    },
                }],
            }],
        )


@pytest.mark.parametrize(
    ("content", "message"),
    (({
        "text": "not a content part"
    }, "must be a string, an array, or null"),
     ([7], "must be strings or objects")),
)
def test_message_conversion_rejects_non_openai_content_shape(content, message):
    with pytest.raises(ValueError, match=message):
        _convert_messages_to_cpp(
            _message_runtime(),
            [{
                "role": "user",
                "content": content
            }],
        )


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


def test_completion_request_preserves_raw_input():
    llm = LLM.__new__(LLM)
    prepared_messages = []

    def make_request(messages, _params, **options):
        prepared_messages.append(messages)
        return SimpleNamespace(pre_tokenized_input_ids=[], **options)

    llm._make_generation_request = make_request
    params = SamplingParams()

    token_request = llm._make_completion_request([11, 12, 13], params)
    text_request = llm._make_completion_request("raw prompt", params)

    assert prepared_messages == [[], [{
        "role": "user",
        "content": "raw prompt"
    }]]
    assert token_request.pre_tokenized_input_ids == [[11, 12, 13]]
    assert not token_request.apply_chat_template
    assert not token_request.add_generation_prompt
    assert text_request.pre_tokenized_input_ids == []
    assert not text_request.apply_chat_template
    assert not text_request.add_generation_prompt


def test_generation_request_forwards_ignore_eos():

    class NativeRequest:
        pass

    runtime = SimpleNamespace(
        LLMGenerationRequest=NativeRequest,
        Request=lambda messages: SimpleNamespace(messages=messages),
        ToolChoice=lambda **values: SimpleNamespace(**values),
        ToolChoiceMode=SimpleNamespace(NONE="none",
                                       AUTO="auto",
                                       REQUIRED="required",
                                       FUNCTION="function"),
        ContextCacheLookupPolicy=SimpleNamespace(USE_CACHE="use",
                                                 BYPASS="bypass"),
        ContextCacheCommitPolicy=SimpleNamespace(
            INCLUDING_GENERATED_TOKENS="all",
            PREFILL_STATE_ONLY="prefill",
        ),
    )
    llm = LLM.__new__(LLM)
    llm._rt = runtime
    llm._context_cache_config = None
    llm._prepare_messages_for_runtime = lambda *_args, **_kwargs: ([], [])

    request = llm._make_generation_request([], SamplingParams(ignore_eos=True))

    assert request.ignore_eos


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

    from tensorrt_edgellm.server.runtime.engine_client import EngineClient

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

    from tensorrt_edgellm.server.runtime.engine_client import EngineClient

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

        def try_pop(self):
            return None

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
            def create(*, collect_metrics=False):
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

        def try_pop(self):
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
            def create(*, collect_metrics=False):
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


def test_disconnect_closes_the_stream_only_after_next_returns():
    # Closing a generator another thread is running can crash the interpreter.
    reading = threading.Event()
    cancelled = threading.Event()
    closes_during_read = []

    class Source:

        def __init__(self):
            self.in_next = False
            self.closed = False

        def __next__(self):
            self.in_next = True
            reading.set()
            cancelled.wait()
            time.sleep(0.05)
            self.in_next = False
            raise StopIteration

        def close(self):
            closes_during_read.append(self.in_next)
            self.closed = True

    source = Source()
    stream = engine.CancellableIterator(source, cancelled.set)

    async def exercise():

        async def consume():
            async for _ in _iterate_sync(stream):
                pass

        task = asyncio.create_task(consume())
        assert await asyncio.to_thread(reading.wait, 1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert cancelled.is_set()
    assert source.closed
    assert closes_during_read and not any(closes_during_read)


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


class _NativeCacheMetrics:
    """Every counter reads 0 and every pool 40 free of 96, unless named."""

    def __init__(self, **counters):
        self.__dict__.update(counters)

    def __getattr__(self, name):
        if name.endswith(("_pages", "_snapshots")):
            return SimpleNamespace(free=40, capacity=96)
        return 0


def test_ifb_context_cache_counters_do_not_wait_for_the_runtime_lock():
    """Given in-flight batching and a request holding the runtime lock, when
    the context-cache counters are read, then the engine's copy comes back
    without waiting, pools as free/capacity pairs."""
    native = _NativeCacheMetrics(hit_sequences=3, reused_tokens=384)

    class Engine(_FakeEngine):

        def context_cache_metrics(self):
            return native

    llm = _ifb_llm(Engine())
    read = {}

    def reader():
        read["metrics"] = llm.get_context_cache_metrics()
        read["counters"] = llm.get_context_cache_counters()

    with llm._infer_lock:
        thread = threading.Thread(target=reader)
        thread.start()
        thread.join(1.0)
        finished_while_locked = not thread.is_alive()
    thread.join()

    assert finished_while_locked
    assert read["metrics"] is native
    assert read["counters"]["hit_sequences"] == 3
    assert read["counters"]["reused_tokens"] == 384
    assert read["counters"]["base_kv_pages"] == {"free": 40, "capacity": 96}


def test_health_context_cache_counters_are_absent_on_the_blocking_path():
    """Given the blocking path, when /health asks for context-cache counters,
    then it gets None without reading the runtime a running request holds."""

    class Runtime:

        def get_context_cache_metrics(self):
            raise AssertionError("a health probe must not wait on the runtime")

    llm = _bare_llm(Runtime())
    llm._engine = None
    assert llm.get_context_cache_counters() is None


_STEP_COUNTERS = ("decode_steps", "decode_graph_steps", "prompt_steps",
                  "mixed_steps", "decode_subset_steps", "step_sequences",
                  "step_prompt_chunks", "partial_prompt_chunks",
                  "step_decode_rows", "step_tokens", "step_sampled_rows",
                  "step_time_total_us", "step_time_max_us",
                  "paused_resident_steps", "compact_refusals")


def test_scheduling_metrics_report_exclusive_stalls():
    """Given an engine that held heads back for exclusivity, when /health reads
    the scheduling counters, then stalls_exclusive is reported."""

    class Engine(_FakeEngine):
        queued = 0
        resident = 1

        def metrics(self):
            return SimpleNamespace(submitted=5,
                                   refused=0,
                                   stalls_incompatible=1,
                                   stalls_guided=0,
                                   stalls_exclusive=4,
                                   stalls_no_capacity=0,
                                   admitted_mid_flight=2,
                                   completed=4,
                                   cancelled=0,
                                   failed=0,
                                   queue_latency_total_us=10,
                                   queue_latency_max_us=7,
                                   queue_latency_count=5,
                                   **dict.fromkeys(_STEP_COUNTERS, 0))

    assert _ifb_llm(Engine()).get_scheduling_metrics()["stalls_exclusive"] == 4


@pytest.mark.parametrize("native, expected", [(5, 5), (-1, None)])
def test_ifb_stream_delta_carries_the_reused_prompt_count(native, expected):
    """Given a chunk reporting how much of its prompt was reused (-1 before
    the prefill ran), when the stream yields it, then the delta carries that
    count, or None for -1."""
    engine = _FakeEngine()
    channel = _IFBChannel()
    chunk = _IFBChunk()
    chunk.reused_prompt_token_count = native
    channel.chunks = [chunk]
    llm = _ifb_llm(engine, channel)

    request = type("Request", (), {"stream_channels": None})()
    deltas = list(llm.generate_stream([], prebuilt_request=request))

    assert [d.reused_prompt_tokens for d in deltas] == [expected]


def test_completion_carries_the_reused_prompt_count(tmp_path):
    """Given a response reporting how much of its prompt was reused, when it is
    turned into a completion, then the completion carries that count."""
    from tensorrt_edgellm.server.parsing.tool_calling import ToolConfig

    engine_ = _FakeEngine()
    engine_.handle.get = lambda: SimpleNamespace(output_texts=["x"],
                                                 output_ids=[[1]],
                                                 prompt_token_counts=[7],
                                                 reused_prompt_token_counts=
                                                 [5],
                                                 finish_reasons=[2],
                                                 logprobs=[])
    llm = _ifb_llm(engine_)
    llm._rt = SimpleNamespace(FinishReason=SimpleNamespace(
        NOT_FINISHED=0, END_ID=1, LENGTH=2, CANCELLED=3, ERROR=4,
        STOP_WORDS=5))
    llm._model_dir = str(tmp_path)
    llm._context_cache_config = SimpleNamespace(enabled=False)

    output = llm._complete_prepared_request(object(), SamplingParams(),
                                            ToolConfig())

    assert output.prompt_tokens == 7
    assert output.reused_prompt_tokens == 5


def test_context_cache_config_maps_to_native_runtime_value():

    class NativeConfig:

        def __init__(self):
            self.enabled = False
            self.max_records = 0
            self.max_snapshots = 0

    native = _native_context_cache_config(
        SimpleNamespace(ContextCacheConfig=NativeConfig),
        ContextCacheConfig(enabled=True, max_records=17, max_snapshots=6),
    )

    assert native.enabled
    assert native.max_records == 17
    assert native.max_snapshots == 6


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


@pytest.mark.parametrize(
    "requested, disable_spec_decode, reuse_context, prefill_state_only, "
    "expected",
    [
        (None, False, True, False, True),
        (None, False, True, True, False),
        (None, True, True, True, True),
        (True, False, False, True, True),
    ],
)
def test_cache_generated_tokens_follows_the_deployment(requested,
                                                       disable_spec_decode,
                                                       reuse_context,
                                                       prefill_state_only,
                                                       expected):
    """An omitted commit policy publishes generated tokens unless a
    speculative request looks up context on a deployment that can only
    publish prefill state; vanilla and lookup-bypass requests are exempt."""
    params = SamplingParams(cache_generated_tokens=requested,
                            disable_spec_decode=disable_spec_decode,
                            reuse_context=reuse_context)
    assert engine.resolve_cache_generated_tokens(
        params, prefill_state_only) is expected


def test_native_chat_template_derives_hybrid_mtp_replay_tail(monkeypatch):
    from tensorrt_edgellm.server.runtime import engine as engine_module

    class NativeValue:

        def __init__(self, **values):
            self.__dict__.update(values)

    runtime_module = SimpleNamespace(
        LLMGenerationRequest=NativeValue,
        Request=NativeValue,
        ToolChoice=NativeValue,
        ToolChoiceMode=SimpleNamespace(NONE="none",
                                       AUTO="auto",
                                       REQUIRED="required",
                                       FUNCTION="function"),
        ContextCacheLookupPolicy=SimpleNamespace(USE_CACHE="use",
                                                 BYPASS="bypass"),
        ContextCacheCommitPolicy=SimpleNamespace(
            INCLUDING_GENERATED_TOKENS="all", PREFILL_STATE_ONLY="prefill"),
    )
    llm = _bare_llm(SimpleNamespace(has_draft_model=lambda: True))
    llm._rt = runtime_module
    llm._context_cache_config = ContextCacheConfig(enabled=True)
    llm._prepare_messages_for_runtime = lambda _messages: ([], [])
    monkeypatch.setattr(engine_module, "_load_audio_buffers",
                        lambda *_args: [])

    request = llm._make_generation_request(
        [{
            "role": "user",
            "content": "hello"
        }],
        SamplingParams(cache_generated_tokens=False, seed=7),
    )
    assert request.sampling_seed == 7
    assert request.apply_chat_template is True
    assert request.add_generation_prompt is True
    assert request.context_cache_replay_tail_length == -1
    assert request.requests[0].sampling_seed == 7

    raw_request = llm._make_generation_request(
        [{
            "role": "user",
            "content": "hello"
        }],
        SamplingParams(cache_generated_tokens=False),
        apply_chat_template=False,
    )
    assert raw_request.context_cache_replay_tail_length == 0

    for prompt in ("hello", [11, 12, 13]):
        completion_request = llm._make_completion_request(
            prompt, SamplingParams(cache_generated_tokens=False))
        assert completion_request.apply_chat_template is False
        assert completion_request.add_generation_prompt is False
        assert completion_request.context_cache_replay_tail_length == 0


@pytest.mark.parametrize("proposal, verify", [(None, None), (3, None),
                                              (None, 8), (3, 8)])
@pytest.mark.parametrize("engine_type",
                         [EngineType.LLM, EngineType.SPEC_DECODE])
def test_runtime_load_forwards_context_cache_config(monkeypatch, engine_type,
                                                    proposal, verify):
    # Given native runtime options with independently omitted P/Q values.
    from tensorrt_edgellm.server.runtime import engine as engine_module

    captured = {}

    class NativeConfig:

        def __init__(self):
            self.enabled = False
            self.max_records = 0
            self.max_snapshots = 0

    class Runtime:

        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            captured["ple_config"] = kwargs["ple_config"]

        @staticmethod
        def capture_decoding_cuda_graph():
            return True

    runtime_module = SimpleNamespace(ContextCacheConfig=NativeConfig,
                                     LLMRuntime=Runtime,
                                     PleRuntimeConfig=SimpleNamespace,
                                     PleResidency=SimpleNamespace(
                                         RESIDENT="resident", CACHED="cached"))
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
    llm._proposal_tokens = proposal
    llm._verify_tokens = verify
    llm._context_cache_config = ContextCacheConfig(enabled=True,
                                                   max_records=23)

    llm._ple_config = PleRuntimeConfig(file_path="/model/ple.safetensors")
    # When loading the selected runtime.
    llm._load_runtime()
    assert captured["ple_config"].file_path == "/model/ple.safetensors"
    assert captured["ple_config"].residency == "resident"

    # Then native configuration receives caller values without Python defaults.
    if engine_type == EngineType.SPEC_DECODE:
        assert captured["kwargs"]["draft_step"] == 3
        assert captured["kwargs"]["proposal_tokens"] == proposal
        assert captured["kwargs"]["verify_tokens"] == verify
    native = (captured["kwargs"]["context_cache_config"]
              if "context_cache_config" in captured["kwargs"] else
              captured["args"][-1])
    assert native.enabled
    assert native.max_records == 23


# ---------------------------------------------------------------------------
# In-flight batching: the engine-backed request paths
# ---------------------------------------------------------------------------


class _FakeHandle:

    def __init__(self):
        self.cancelled = False
        self.got = False
        self.error = None
        self.is_ready = False

    def ready(self):
        return self.is_ready

    def cancel(self):
        self.cancelled = True

    def get(self):
        self.got = True
        if self.error is not None:
            raise self.error
        return SimpleNamespace(output_texts=["done"])


class _FakeEngine:

    def __init__(self):
        self.submitted = []
        self.handle = _FakeHandle()
        self.shutdown_modes = []

    def submit(self, request):
        self.submitted.append(request)
        return self.handle

    def shutdown(self, mode):
        self.shutdown_modes.append(mode)


class _IFBChunk:
    text = "partial"
    token_ids = [1]
    prompt_token_count = 1
    finished = False
    reason = None
    logprobs = []


class _IFBChannel:
    """One unfinished chunk, then a finished channel: the loop exits on the
    finished flag rather than a terminal chunk, so the fake runtime module
    needs no FinishReason enum."""

    def __init__(self):
        self.chunks = [_IFBChunk()]
        self.cancelled = False
        self.skip_special = None
        self.silent = False  # neither finished nor cancelled once drained

    def set_skip_special_tokens(self, enabled):
        self.skip_special = enabled

    def wait_pop(self, timeout_ms=0):
        return self.chunks.pop(0) if self.chunks else None

    def try_pop(self):
        return self.wait_pop()

    def is_finished(self):
        return not self.silent and not self.chunks

    def is_cancelled(self):
        return self.cancelled

    def cancel(self):
        self.cancelled = True


def _ifb_llm(engine, channel=None):
    llm = _bare_llm(None)
    llm._engine = engine
    if channel is not None:

        class RuntimeModule:

            class StreamChannel:

                @staticmethod
                def create(*, collect_metrics=False):
                    return channel

        llm._rt = RuntimeModule()
    return llm


def test_ifb_handle_request_does_not_take_the_infer_lock():
    # On the blocking path a held _infer_lock would deadlock this call; under
    # in-flight batching concurrency control lives in the engine, so it must
    # go through even while the lock is held.
    engine = _FakeEngine()
    llm = _ifb_llm(engine)
    with llm._infer_lock:
        response = LLM._handle_request(llm, "req")
    assert engine.submitted == ["req"]
    assert response.output_texts == ["done"]


def test_ifb_stream_attaches_its_channel_and_needs_no_worker():
    engine = _FakeEngine()
    channel = _IFBChannel()
    llm = _ifb_llm(engine, channel)

    request = type("Request", (), {"stream_channels": None})()
    deltas = list(llm.generate_stream([], prebuilt_request=request))

    assert [d.text for d in deltas] == ["partial"]
    assert request.stream_channels == [channel]
    assert engine.submitted == [request]
    # Every stream the caller did not cancel collects its outcome: a terminal
    # chunk says the tokens ended, get() says whether the request succeeded.
    assert engine.handle.got


def test_ifb_stream_close_cancels_the_engine_request():
    engine = _FakeEngine()
    channel = _IFBChannel()
    channel.chunks = [_IFBChunk(), _IFBChunk()]
    llm = _ifb_llm(engine, channel)

    request = type("Request", (), {"stream_channels": None})()
    stream = llm.generate_stream([], prebuilt_request=request)
    assert next(stream).text == "partial"
    stream.close()

    assert engine.handle.cancelled
    assert channel.cancelled
    assert not engine.handle.got


@pytest.mark.parametrize("outcome", ["completed", "cancelled", "failed"])
def test_ifb_final_metrics_wait_for_outcome_without_emitting_more_text(
        outcome):
    # Given an in-flight request, when its outcome arrives, then final metrics precede completion or failure.
    request_engine = _FakeEngine()
    channel = _IFBChannel()
    llm = _ifb_llm(request_engine, channel)
    results, collected = [], []
    native = SimpleNamespace(processed_prompt_tokens=2,
                             generated_tokens=3,
                             decoded_tokens=2,
                             prompt_seconds=0.5,
                             generation_seconds=0.25)

    def ready():
        assert not results
        if not request_engine.handle.is_ready:
            request_engine.handle.is_ready = True
            return False
        channel.chunks.append(
            SimpleNamespace(token_ids=[], prompt_token_count=3,
                            metrics=native))
        return True

    def get():
        assert channel.chunks[-1].metrics is native
        collected.append(True)
        if outcome != "completed":
            raise RuntimeError(outcome)

    request_engine.handle.ready = ready
    request_engine.handle.get = get
    stream = llm.generate_stream([],
                                 prebuilt_request=SimpleNamespace(),
                                 on_metrics=results.append)
    assert next(stream).text == "partial"
    if outcome == "cancelled":
        stream.close()
        assert request_engine.handle.cancelled
    elif outcome == "failed":
        with pytest.raises(RuntimeError, match="failed"):
            list(stream)
    else:
        assert list(stream) == []
    assert collected == [True]
    metrics, = results
    assert metrics.prompt_tokens == 3
    assert metrics.processed_prompt_tokens == 2
    assert metrics.generated_tokens == 3
    assert metrics.prompt_time == 0.5
    assert metrics.generation_time == 0.25
    assert metrics.decoded_tokens == 2
    assert metrics.decode_tokens_per_second == 8.0
    assert not channel.chunks


def test_ifb_stream_raises_an_outcome_error_after_a_clean_finish():
    # The channel finished normally, but the outcome carries an execution
    # error (a sequence that ended in FinishReason.ERROR, or a response that
    # failed to materialize). It must reach the caller, not be dropped behind
    # a normal-looking end of stream.
    engine = _FakeEngine()
    engine.handle.error = RuntimeError("materialize failed")
    channel = _IFBChannel()
    llm = _ifb_llm(engine, channel)
    request = type("Request", (), {"stream_channels": None})()
    with pytest.raises(RuntimeError, match="materialize failed"):
        list(llm.generate_stream([], prebuilt_request=request))
    assert engine.handle.got
    assert not engine.handle.cancelled


def test_ifb_handle_request_hands_the_handle_to_the_caller():
    engine = _FakeEngine()
    llm = _ifb_llm(engine)
    seen = []
    LLM._handle_request(llm, "req", on_handle=seen.append)
    assert seen == [engine.handle]
    assert engine.handle.got


def test_run_sync_cancels_in_the_engine_before_waiting_for_the_worker():
    from tensorrt_edgellm.server.runtime.engine_client import _run_sync

    cancelled = []
    release = threading.Event()

    def blocking():
        release.wait(5)
        return "done"

    def on_cancel():
        cancelled.append(True)
        release.set()  # the engine's cancel is what lets the worker return

    async def exercise():
        task = asyncio.create_task(_run_sync(blocking, on_cancel=on_cancel))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert cancelled == [True]


def test_disconnect_before_submit_returns_still_cancels_the_request():
    # The client goes away while the worker is still inside engine.submit():
    # on_cancel runs before on_handle has anything to cancel. The handle that
    # appears afterwards must be cancelled on publication, not left to decode.
    from tensorrt_edgellm.server.runtime.engine_client import (_HandleLatch,
                                                               _run_sync)

    class Handle:

        def __init__(self):
            self.cancelled = threading.Event()

        def cancel(self):
            self.cancelled.set()

        def get(self):
            assert self.cancelled.wait(5), "get() would run to max_tokens"
            raise RuntimeError("Request 1 was cancelled.")

    handle = Handle()
    task_cancelled = threading.Event()
    latch = _HandleLatch()

    def operation():
        task_cancelled.wait(5)  # still "inside submit()" when the cancel lands
        latch.publish(handle)
        return handle.get()

    def on_cancel():
        latch.cancel()
        task_cancelled.set()

    async def exercise():
        task = asyncio.create_task(_run_sync(operation, on_cancel=on_cancel))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert handle.cancelled.is_set()


def test_load_model_keeps_the_ifb_flag_away_from_standalone_tts(
        monkeypatch, tmp_path):
    """A standalone TTS constructor never sees the IFB flag, PLE or the scheduling options at their defaults;
    a non-default value of any of them is refused."""
    from tensorrt_edgellm.server.config import IFB_SCHEDULING_DEFAULTS
    from tensorrt_edgellm.server.runtime import engine as engine_module
    from tensorrt_edgellm.server.runtime import engine_build

    model_dir = tmp_path / "tts"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "qwen3_tts"}')
    monkeypatch.setattr(engine_build, "resolve_model_dir",
                        lambda model, cache_dir: str(model_dir))
    received = {}

    class FakeTTS:

        def __init__(self, **kwargs):
            received.update(kwargs)

    monkeypatch.setattr(engine_module, "TTS", FakeTTS)
    monkeypatch.setattr(engine_module, "_derive_model_id", lambda model: "tts")

    # The server's llm_kwargs() always carries these options; at their defaults
    # the TTS constructor must simply not see them ...
    engine_module.load_model(model=str(model_dir),
                             enable_in_flight_batching=False,
                             ple_config=PleRuntimeConfig(),
                             **IFB_SCHEDULING_DEFAULTS)
    assert "enable_in_flight_batching" not in received
    assert "ple_config" not in received
    assert not set(IFB_SCHEDULING_DEFAULTS) & set(received)
    with pytest.raises(ValueError, match="do not use PLE"):
        engine_module.load_model(model=str(model_dir),
                                 ple_config={
                                     "residency": "cached",
                                     "cache_slots": 513
                                 })
    # ... and an explicit opt-in is a configuration error, like speculative
    # decoding on a TTS model.
    with pytest.raises(ValueError, match="not supported for this deployment"):
        engine_module.load_model(model=str(model_dir),
                                 enable_in_flight_batching=True)
    with pytest.raises(ValueError, match="enable-in-flight-batching"):
        engine_module.load_model(model=str(model_dir),
                                 scheduling_policy="priority")


def test_ifb_unsupported_reason_names_the_deployment():
    from tensorrt_edgellm.server.runtime.engine import ifb_unsupported_reason

    text = SimpleNamespace(engine_type=EngineType.LLM, has_speech=False)
    speech = SimpleNamespace(engine_type=EngineType.LLM, has_speech=True)
    spec = SimpleNamespace(engine_type=EngineType.SPEC_DECODE,
                           has_speech=False)
    assert ifb_unsupported_reason(text) is None
    assert ifb_unsupported_reason(speech) is None  # text-only, not refused
    assert ifb_unsupported_reason(spec) is None  # the engine seats drafts
    assert "TTS" in ifb_unsupported_reason(None, model_type="qwen3_tts")


def _speculative_ifb_llm(monkeypatch, request_engine):
    """An LLM over a speculative bundle with in-flight batching enabled, whose
    native module builds ``request_engine`` and refuses to build a blocking
    runtime."""
    from tensorrt_edgellm.server.runtime import engine as engine_module

    class Runtime:

        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "in-flight batching must not build a blocking runtime")

    monkeypatch.setattr(
        engine_module, "_import_runtime", lambda: SimpleNamespace(
            ContextCacheConfig=lambda: SimpleNamespace(),
            PleRuntimeConfig=lambda: SimpleNamespace(),
            PleResidency=SimpleNamespace(CACHED="cached", RESIDENT="resident"),
            LLMRuntime=Runtime,
            RequestEngine=request_engine))
    llm = LLM.__new__(LLM)
    llm._layout = SimpleNamespace(engine_type=EngineType.SPEC_DECODE,
                                  has_speech=False)
    llm._bundle_dir = "/bundle"
    llm._media_dir = ""
    llm._model_dir = "/model"
    llm._draft_model_dir = "/draft"
    llm._draft_top_k = 1
    llm._draft_step = 3
    llm._proposal_tokens = None
    llm._verify_tokens = 4
    llm._max_batch_size = 4
    llm._context_cache_config = ContextCacheConfig()
    llm._ple_config = engine_module.PleRuntimeConfig.parse(None)
    llm._enable_in_flight_batching = True
    llm._ifb_scheduling = {}
    llm._runtime = None
    return llm


def test_request_engine_gets_drafting_shape(monkeypatch):
    # Given a speculative bundle and in-flight batching, when the runtime
    # loads, then the request engine is built with the bundle's drafting
    # shape and reports the draft model, and no blocking runtime exists; and
    # when the engine refuses the deployment (a context cache without live
    # admission), startup fails with the engine's reason instead of falling
    # back silently.
    built = {}

    class RequestEngine:
        resolved_policy = SimpleNamespace(mode="prefillFirst",
                                          execution="padded")

        def __init__(self, *args, **kwargs):
            built.update(kwargs)

    llm = _speculative_ifb_llm(monkeypatch, RequestEngine)
    llm._load_runtime()
    assert built["draft_top_k"] == 1
    assert built["draft_step"] == 3
    assert built["proposal_tokens"] is None
    assert built["verify_tokens"] == 4
    assert built["draft_checkpoint_dir"] == "/draft"
    assert built["max_batch_size"] == 4
    assert llm._runtime is None
    assert llm.has_draft_model

    class RefusingEngine:

        def __init__(self, *args, **kwargs):
            raise RuntimeError("this deployment cannot seat")

    llm = _speculative_ifb_llm(monkeypatch, RefusingEngine)
    with pytest.raises(ValueError,
                       match="not supported for this deployment: this "
                       "deployment cannot seat"):
        llm._load_runtime()


def test_omni_bundle_under_ifb_serves_text_only():
    # The Thinker is a text runtime the engine serves; the speech stack is
    # not loaded and every speech entry point refuses, naming the flag.
    engine = _FakeEngine()
    llm = _ifb_llm(engine)
    llm._layout = SimpleNamespace(engine_type=EngineType.LLM,
                                  has_speech=True,
                                  talker_dir="/talker")
    assert not llm.speech_available
    llm._load_omni_runtime()  # no runtime to load into; must not touch one
    assert llm.list_voices() == []
    with pytest.raises(ValueError, match="in-flight batching"):
        llm.generate_speech_stream("hello")
    with pytest.raises(ValueError, match="in-flight batching"):
        llm.generate_stream_with_audio([])


def test_capabilities_do_not_advertise_speech_under_ifb():
    from tensorrt_edgellm.server.runtime.engine_client import _capabilities_for

    class FakeLLM:
        bundle_dir = "/nonexistent"
        model_dir = "/nonexistent"
        model_id = "m"
        runtime_kind = "chat"
        video_capable = False
        has_draft_model = False
        context_cache_enabled = False
        bundle_layout = SimpleNamespace(visual_dir=None,
                                        audio_dir=None,
                                        has_speech=True,
                                        has_transcription=False)
        _engine = object()

    caps = _capabilities_for(FakeLLM())
    assert caps.in_flight_batching
    assert not caps.speech
    assert "audio" not in caps.output_modalities

    FakeLLM._engine = None
    caps = _capabilities_for(FakeLLM())
    assert caps.speech
    assert "audio" in caps.output_modalities


def test_direct_generate_overlaps_under_ifb_up_to_batch_plus_queue():
    # The direct Python API must not serialise engine-backed requests: the
    # gate is batch + queue depth, so with max_batch_size=1 exactly 17 callers
    # are inside the engine at once and the 18th waits at the gate.
    from tensorrt_edgellm.server.config import DEFAULT_MAX_QUEUED_REQUESTS

    release = threading.Event()
    inside = threading.Semaphore(0)

    class Handle:

        def get(self):
            inside.release()
            release.wait(10)
            return SimpleNamespace(output_texts=["x"],
                                   output_ids=[[1]],
                                   prompt_token_counts=[1],
                                   finish_reasons=["stop"])

    class Engine:

        def __init__(self):
            self.submitted = 0
            self.lock = threading.Lock()

        def submit(self, request):
            with self.lock:
                self.submitted += 1
            return Handle()

    engine = Engine()
    llm = _ifb_llm(engine)
    llm._max_batch_size = 1
    limit = llm._max_batch_size + DEFAULT_MAX_QUEUED_REQUESTS
    llm._admission_sem = threading.Semaphore(limit)
    llm._make_generation_request = lambda *a, **k: object()
    llm._context_cache_config = SimpleNamespace(enabled=False)

    def one_call():
        with llm._admission():
            llm._handle_request(llm._make_generation_request())

    threads = [threading.Thread(target=one_call) for _ in range(limit + 3)]
    for t in threads:
        t.start()
    for _ in range(limit):
        assert inside.acquire(timeout=5), "callers were serialised"
    time.sleep(0.2)
    assert engine.submitted == limit, "the gate let more than batch+queue in"
    release.set()
    for t in threads:
        t.join(10)
    assert engine.submitted == limit + 3


def test_ifb_stream_takes_and_releases_the_gate():
    # Streaming under IFB goes through the same gate as generate(): taken
    # before the request is built, released when the stream ends.
    engine = _FakeEngine()
    channel = _IFBChannel()
    llm = _ifb_llm(engine, channel)
    llm._admission_sem = threading.Semaphore(1)
    llm._make_generation_request = lambda *a, **k: object()
    stream = llm.generate_stream([{"role": "user", "content": "hi"}])
    first = next(stream)
    assert first.text == "partial"
    assert not llm._admission_sem.acquire(blocking=False), "gate not held"
    for _ in stream:
        pass
    assert llm._admission_sem.acquire(blocking=False), "gate not released"


def test_use_ifb_is_the_flag_alone():
    llm = LLM.__new__(LLM)
    llm._layout = SimpleNamespace(engine_type=EngineType.LLM, has_speech=True)
    llm._enable_in_flight_batching = False
    assert llm._use_ifb() is False
    llm._enable_in_flight_batching = True
    assert llm._use_ifb() is True  # the deployment check happened at load time


def test_ifb_stream_ends_when_the_outcome_beats_the_channel():
    # A request can retire without its channel ever being touched (rejected, or a
    # founder failing before decoding). The consumer polls the channel, so it must
    # also watch the outcome flag — or it waits out its transport timeout.
    engine = _FakeEngine()
    engine.handle.is_ready = True
    engine.handle.error = RuntimeError("input too long")
    channel = _IFBChannel()
    channel.chunks = []
    channel.silent = True  # silent forever, neither finished nor cancelled
    llm = _ifb_llm(engine, channel)

    request = type("Request", (), {"stream_channels": None})()
    with pytest.raises(RuntimeError, match="input too long"):
        list(llm.generate_stream([], prebuilt_request=request))
    assert engine.handle.got


def test_ifb_stream_surfaces_an_actor_failure():
    engine = _FakeEngine()
    engine.handle.error = RuntimeError("execution failed")
    channel = _IFBChannel()
    channel.chunks = []
    channel.cancelled = True  # the actor, not the caller, cancelled it
    llm = _ifb_llm(engine, channel)

    request = type("Request", (), {"stream_channels": None})()
    with pytest.raises(RuntimeError, match="execution failed"):
        list(llm.generate_stream([], prebuilt_request=request))
    assert engine.handle.got


def test_ifb_close_drains_the_engine_once():
    engine = _FakeEngine()
    llm = _ifb_llm(engine)
    llm._rt = SimpleNamespace(ShutdownMode=SimpleNamespace(DRAIN="drain"))

    llm.close()
    llm.close()

    assert engine.shutdown_modes == ["drain"]
    with pytest.raises(RuntimeError):
        LLM._handle_request(llm, "req")


def test_ifb_gate_limits_without_queueing():

    async def exercise():
        gate = _IFBAdmissionGate(max_queued_requests=1)
        first = await gate.reserve()
        second = await gate.reserve()  # limit is 1 + max_queued
        assert gate.active == 2
        assert gate.waiting == 0

        # No waiting slot exists: the third caller is refused on the spot.
        with pytest.raises(ServerOverloadedError):
            await gate.reserve()

        first.release()
        third = await gate.reserve()
        second.release()
        third.release()
        assert gate.active == 0

    asyncio.run(exercise())


def test_ifb_gate_capacity_scales_with_batch_size():
    # A zero-depth queue must not serialize the engine: the gate's bound is
    # batch capacity plus queue depth, so an mxbs-4 engine with no queue still
    # admits four concurrent requests before refusing the fifth.

    async def exercise():
        gate = _IFBAdmissionGate(max_queued_requests=0, max_concurrency=4)
        leases = [await gate.reserve() for _ in range(4)]
        with pytest.raises(ServerOverloadedError):
            await gate.reserve()
        for lease in leases:
            lease.release()
        assert gate.active == 0

    asyncio.run(exercise())


def test_ifb_gate_close_rejects_new_work_and_waits_for_leases():

    async def exercise():
        gate = _IFBAdmissionGate(max_queued_requests=0)
        lease = await gate.reserve()
        closing = asyncio.create_task(gate.close())
        while not gate._closing:
            await asyncio.sleep(0)

        with pytest.raises(ServerUnavailableError):
            await gate.reserve()
        assert not closing.done(), "close returned while a lease was live"

        lease.release()
        await closing

    asyncio.run(exercise())


def test_engine_refusal_maps_to_overload():
    from tensorrt_edgellm.server.runtime.engine_client import EngineClient

    class SubmitError(Exception):
        pass

    class FakeLLM:
        bundle_dir = "/nonexistent"
        model_dir = "/nonexistent"
        model_id = "m"
        runtime_kind = "chat"
        video_capable = False
        has_draft_model = False
        context_cache_enabled = False
        bundle_layout = SimpleNamespace(visual_dir=None,
                                        audio_dir=None,
                                        has_speech=False,
                                        has_transcription=False)
        _rt = SimpleNamespace(SubmitError=SubmitError)
        _engine = object()

        def _make_generation_request(self, *args, **kwargs):
            return object()

        def _complete_prepared_request(self, *args, **kwargs):
            raise SubmitError("too many requests in flight")

    client = EngineClient(FakeLLM())

    async def exercise():
        with pytest.raises(ServerOverloadedError):
            await client.generate([], SamplingParams())

    asyncio.run(exercise())


def test_context_validation_reserves_output_tokens():
    """Validation counts rendered tools and reserves the output budget."""
    llm = _bare_llm(object())
    llm._max_input_len, llm._max_kv_cache_capacity = 10, 12
    llm._make_generation_request = lambda messages, params, tools: tools or []
    llm._count_prepared_prompt_tokens = lambda tools: 8 + len(tools)
    assert llm.validate_context([], engine.SamplingParams(max_tokens=4)) == 8
    assert llm.validate_context([],
                                engine.SamplingParams(max_tokens=3),
                                tools=[{}]) == 9
    with pytest.raises(ValueError, match="output tokens reserved"):
        llm.validate_context([], engine.SamplingParams(max_tokens=5))
    llm._max_input_len = 7
    with pytest.raises(ValueError):
        llm.validate_context([], engine.SamplingParams(max_tokens=1))


def test_stream_without_metrics_skips_collection(monkeypatch):
    # Given no metrics callback, when a stream runs, then native collection and Python clock reads stay off.
    request_engine = _FakeEngine()
    channel = _IFBChannel()
    llm = _ifb_llm(request_engine, channel)
    requested = []

    def create_channel(*, collect_metrics=False):
        requested.append(collect_metrics)
        return channel

    def unexpected_clock():
        pytest.fail("Metrics clock read without a callback")

    llm._rt.StreamChannel.create = create_channel
    monkeypatch.setattr(engine.time, "perf_counter", unexpected_clock)
    assert "".join(delta.text for delta in llm.generate_stream(
        [], prebuilt_request=SimpleNamespace())) == "partial"
    assert requested == [False]
    assert request_engine.handle.got


@pytest.mark.parametrize("received_at,native_timings", [
    pytest.param([11., 13., 14.], True, id="steady-delivery"),
    pytest.param([100., 100.001, 100.002], True, id="bursty-delivery"),
    pytest.param([11., 13., 14.], False, id="incomplete-native-timings"),
])
def test_stream_metrics_use_native_timings(monkeypatch, received_at,
                                           native_timings):
    # Given delayed chunks, when metrics are requested, then native timing determines throughput.
    native = SimpleNamespace(processed_prompt_tokens=2,
                             generated_tokens=3 if native_timings else 0,
                             decoded_tokens=2 if native_timings else 0,
                             prompt_seconds=0.5 if native_timings else 0.,
                             generation_seconds=0.25 if native_timings else 0.)
    chunks = iter([
        SimpleNamespace(text="first",
                        token_ids=[1, 2],
                        metrics=native,
                        prompt_token_count=3,
                        finished=False,
                        logprobs=[]),
        SimpleNamespace(text="second",
                        token_ids=[3],
                        prompt_token_count=-1,
                        finished=False,
                        logprobs=[]),
        SimpleNamespace(text="",
                        token_ids=[],
                        prompt_token_count=-1,
                        finished=True,
                        reason=None,
                        metrics=native,
                        logprobs=[]),
    ])
    started_at = 10.
    clock = SimpleNamespace(now=started_at)
    deliveries = iter(received_at)

    def wait_pop(**_kwargs):
        chunk = next(chunks)
        clock.now = next(deliveries)
        return chunk

    channel = SimpleNamespace(
        set_skip_special_tokens=lambda _: None,
        wait_pop=wait_pop,
        try_pop=lambda: next(chunks, None),
        is_finished=lambda: True,
        is_cancelled=lambda: False,
        cancel=lambda: None,
    )
    llm = _bare_llm(object())

    def create_channel(*, collect_metrics=False):
        assert collect_metrics
        return channel

    llm._rt = SimpleNamespace(StreamChannel=SimpleNamespace(
        create=create_channel))
    llm._handle_request = lambda _: None
    monkeypatch.setattr(engine.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(engine, "finish_reason_name", lambda *_: "stop")
    results = []
    stream = llm.generate_stream([],
                                 prebuilt_request=SimpleNamespace(),
                                 on_metrics=results.append)
    assert "".join(delta.text for delta in stream) == "firstsecond"
    metrics, = results
    assert metrics.generated_tokens == 3
    assert metrics.time_to_first_token == received_at[0] - started_at
    assert metrics.generation_time == (0.25 if native_timings else 0.)
    assert metrics.decoded_tokens == (2 if native_timings else 0)
    assert metrics.decode_tokens_per_second == (8.0 if native_timings else 0.)
    assert metrics.total_time == received_at[-1] - started_at
    assert metrics.processed_prompt_tokens == 2
    assert metrics.prompt_time == (0.5 if native_timings else 0.)


@pytest.mark.parametrize("generated,decoded", [(1, 0), (3, 2)])
def test_decode_throughput_is_zero_without_decode_timing(generated, decoded):
    # Given no decode timing, when throughput is read, then it is zero and output counts remain available.
    metrics = engine.GenerationMetrics(generated_tokens=generated,
                                       decoded_tokens=decoded)
    assert metrics.decode_tokens_per_second == 0.0
    assert metrics.generated_tokens == generated


@pytest.mark.parametrize("operation", ["cancel", "close"])
def test_unstarted_stream_reports_final_metrics_once(operation):
    # Given an unstarted stream, when cancelled or closed, then empty metrics are reported once.
    llm = _bare_llm(object())
    results = []

    def on_metrics(metrics):
        results.append(metrics)
        if operation == "close":
            stream.close()

    stream = llm.generate_stream([], on_metrics=on_metrics)
    if operation == "cancel":
        stream.cancel()
    else:
        stream.close()
    assert results == [engine.GenerationMetrics()]
    assert list(stream) == []
    stream.cancel()
    stream.close()
    assert len(results) == 1


@pytest.mark.parametrize("partial", [False, True],
                         ids=["before-first-chunk", "after-partial-response"])
def test_cancellation_collects_final_worker_metrics_without_more_text(partial):
    # Given a running producer, when cancelled, then final metrics survive without emitting more text.
    pending = queue.Queue()
    cancelled = threading.Event()
    stopped = threading.Event()

    def chunk(text, tokens, native=None):
        return SimpleNamespace(text=text,
                               token_ids=tokens,
                               prompt_token_count=3,
                               finished=False,
                               logprobs=[],
                               metrics=native)

    def handle(_request):
        try:
            if partial:
                pending.put(chunk("visible", [1]))
            assert cancelled.wait(5)
            pending.put(chunk("hidden", [2]))
            pending.put(
                chunk(
                    "", [],
                    SimpleNamespace(processed_prompt_tokens=2,
                                    generated_tokens=3,
                                    decoded_tokens=2,
                                    prompt_seconds=0.5,
                                    generation_seconds=0.25)))
        finally:
            stopped.set()

    def wait_pop(**_kwargs):
        if not partial:
            stream.cancel()
        return pending.get(timeout=5)

    def try_pop():
        assert stopped.is_set(), "metrics drained before worker stopped"
        return pending.get_nowait() if not pending.empty() else None

    channel = SimpleNamespace(set_skip_special_tokens=lambda _: None,
                              wait_pop=wait_pop,
                              try_pop=try_pop,
                              is_finished=stopped.is_set,
                              is_cancelled=cancelled.is_set,
                              cancel=cancelled.set)
    llm = _bare_llm(object())
    llm._rt = SimpleNamespace(StreamChannel=SimpleNamespace(
        create=lambda **_: channel))
    llm._handle_request = handle
    results = []
    stream = llm.generate_stream([],
                                 prebuilt_request=SimpleNamespace(),
                                 on_metrics=results.append)
    if partial:
        assert next(stream).text == "visible"
        stream.cancel()
        stream.close()
    assert list(stream) == []
    assert stopped.is_set() and pending.empty()
    metrics, = results
    assert metrics.prompt_tokens == 3
    assert metrics.processed_prompt_tokens == 2
    assert metrics.generated_tokens == 3
    assert metrics.prompt_time == 0.5
    assert metrics.generation_time == 0.25
    assert metrics.decoded_tokens == 2
    assert metrics.decode_tokens_per_second == 8.0
    assert metrics.total_time > 0
    assert (metrics.time_to_first_token is not None) == partial


def test_the_server_cli_reports_an_engine_scheduling_rejection_with_flag_names(
        monkeypatch):
    """When the engine refuses the scheduling options at startup, tensorrt-edgellm-serve exits with "invalid server
    configuration" and the option names, as the engine and the Python API spell them, as its command-line flags;
    "preemption" as a plain word is left alone."""
    pytest.importorskip("fastapi")
    from tensorrt_edgellm.server import cli
    from tensorrt_edgellm.server.config import ServerConfigError

    def load_model(**_kwargs):
        raise ServerConfigError(
            "ifbSchedulingMode=mixed is not supported: x. Use ifbSchedulingMode=prefillFirst. maxChunkTokens 200; "
            "priorityReservedSlots requires schedulingPolicy=priority; preemption=pause and no preemption"
        )

    monkeypatch.setattr(cli, "load_model", load_model)
    with pytest.raises(SystemExit) as raised:
        cli.main(["Qwen/Qwen3-0.6B", "--enable-in-flight-batching"])
    assert str(raised.value) == (
        "invalid server configuration: --ifb-scheduling-mode mixed is not supported: x. Use --ifb-scheduling-mode "
        "prefillFirst. --max-chunk-tokens 200; --priority-reserved-slots requires --scheduling-policy priority; "
        "--preemption pause and no preemption")


# yapf: disable
@pytest.mark.parametrize("via_build_options", [False, True], ids=["kwargs", "build_options"])
@pytest.mark.parametrize("reserved, refused", [(3, False), (4, True)], ids=["one_slot_open", "every_slot_reserved"])
# yapf: enable
def test_reserved_slots_are_checked_against_the_batch_the_engine_is_built_with(
        monkeypatch, reserved, refused, via_build_options):
    """An in-flight batching LLM that reserves every slot of the batch it is built with (B = 4, from max_batch_size or,
    when given, from build_options) raises ServerConfigError before the engine build starts; one that leaves a slot
    open reaches the build."""
    from tensorrt_edgellm.server.config import ServerConfigError
    from tensorrt_edgellm.server.runtime import engine_build

    built = []

    def prepare_model(*args, **_kwargs):
        built.append(args)
        raise RuntimeError("build started")

    monkeypatch.setattr(engine_build, "prepare_model", prepare_model)
    limits = dict(max_batch_size=4, max_input_len=1024)
    if via_build_options:
        limits = dict(build_options=engine_build.BuildOptions(**limits))
    expected = (RuntimeError, ServerConfigError)[refused]
    with pytest.raises(
            expected,
            match="priorityReservedSlots 4" if refused else "build started"):
        LLM("model",
            enable_in_flight_batching=True,
            scheduling_policy="priority",
            priority_reserved_slots=reserved,
            **limits)
    assert len(built) == (0 if refused else 1)
