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
import threading
import time
from types import SimpleNamespace

import pytest

from experimental.server.api.errors import (ServerOverloadedError,
                                            ServerUnavailableError)
from experimental.server.config import ContextCacheConfig
from experimental.server.runtime.engine import (
    LLM, SamplingParams, _convert_messages_to_cpp,
    _native_context_cache_config, _resolve_spec_decode_runtime_options,
    _set_context_cache_request_policies)
from experimental.server.runtime.engine_client import (_AdmissionController,
                                                       _IFBAdmissionGate,
                                                       _iterate_sync)
from experimental.server.runtime.engine_layout import EngineType


def test_gemma4_mtp_tree_runtime_options_follow_compiled_profile(tmp_path):
    (tmp_path / "base_config.json").write_text(
        json.dumps({
            "spec_decode_type": "gemma4_mtp",
            "builder_config": {
                "max_verify_tree_size": 13
            },
        }))
    (tmp_path / "draft_config.json").write_text("{}")

    options = _resolve_spec_decode_runtime_options(str(tmp_path), "mtp", None,
                                                   4, 3, None)

    assert options.top_k == 4
    assert options.step == 3
    assert options.verify_size == 13


def _write_cached_draft_bundle(path, method, base_max, draft_max, mode):
    (path / "base_config.json").write_text(
        json.dumps({
            "spec_decode_type": method,
            "builder_config": {
                "max_verify_tree_size": base_max
            },
        }))
    (path / "draft_config.json").write_text(
        json.dumps({
            "builder_config": {
                "max_draft_tree_size": draft_max
            },
            f"{method}_config": mode,
        }))


def test_dflash_runtime_defaults_clamp_to_compiled_draft_profile(tmp_path):
    _write_cached_draft_bundle(tmp_path, "dflash", 9, 4, {"block_size": 8})

    options = _resolve_spec_decode_runtime_options(str(tmp_path), "dflash",
                                                   None, 1, None, None)

    assert options.step == 1
    assert options.verify_size == 4
    assert options.dflash_block_size == 4
    with pytest.raises(ValueError, match="compiled proposal capacity"):
        _resolve_spec_decode_runtime_options(str(tmp_path), "dflash", 5, 1,
                                             None, None)


def test_dspark_runtime_defaults_add_non_anchor_mask_slot(tmp_path):
    _write_cached_draft_bundle(tmp_path, "dspark", 9, 8, {
        "block_size": 8,
        "sample_from_anchor": False
    })

    options = _resolve_spec_decode_runtime_options(str(tmp_path), "dspark",
                                                   None, 1, None, None)

    assert options.verify_size == 9
    assert _resolve_spec_decode_runtime_options(str(tmp_path), "dspark", 8, 1,
                                                None, None).verify_size == 9
    assert _resolve_spec_decode_runtime_options(str(tmp_path), "dspark", 7, 1,
                                                None, None).verify_size == 8


def test_dspark_tree_runtime_defaults_to_compiled_verify_budget(tmp_path):
    _write_cached_draft_bundle(tmp_path, "dspark", 3, 2, {
        "block_size": 8,
        "sample_from_anchor": False
    })

    options = _resolve_spec_decode_runtime_options(str(tmp_path), "dspark",
                                                   None, 2, None, None)

    assert options.top_k == 2
    assert options.step == 1
    assert options.verify_size == 3
    with pytest.raises(ValueError, match="uses verify_tree_size"):
        _resolve_spec_decode_runtime_options(str(tmp_path), "dspark", 2, 2,
                                             None, None)


def test_dspark_tree_replay_rejects_smaller_verify_budget(tmp_path):
    _write_cached_draft_bundle(tmp_path, "dspark", 9, 9, {
        "block_size": 8,
        "sample_from_anchor": False
    })
    base_path = tmp_path / "base_config.json"
    base = json.loads(base_path.read_text())
    base.update({
        "num_linear_attn_layers": 23,
        "recurrent_spec_verify_mode": "replay",
    })
    base_path.write_text(json.dumps(base))

    with pytest.raises(ValueError, match="compiled maximum 9"):
        _resolve_spec_decode_runtime_options(str(tmp_path), "dspark", None, 2,
                                             None, 5)


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


def test_message_conversion_preserves_string_tool_call_arguments():
    messages = _convert_messages_to_cpp(
        _message_runtime(),
        [{
            "role":
            "assistant",
            "content":
            None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": '{"city":"Paris"}',
                },
            }],
        }],
    )

    assert messages[0].has_tool_calls
    assert messages[0].has_content
    assert messages[0].content_is_null
    assert messages[0].tool_calls[0].arguments_is_string
    assert messages[0].tool_calls[0].arguments == '{"city":"Paris"}'


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
    assert not messages[0].tool_calls[0].arguments_is_string
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


def test_native_chat_template_derives_hybrid_mtp_replay_tail(monkeypatch):
    from experimental.server.runtime import engine as engine_module

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
                def create():
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
    from experimental.server.runtime.engine_client import _run_sync

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
    from experimental.server.runtime.engine_client import (_HandleLatch,
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
    from experimental.server.runtime import engine as engine_module
    from experimental.server.runtime import engine_build

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

    # The server's llm_kwargs() always carries the flag; at its default the
    # TTS constructor must simply not see it ...
    engine_module.load_model(model=str(model_dir),
                             enable_in_flight_batching=False)
    assert "enable_in_flight_batching" not in received
    # ... and an explicit opt-in is a configuration error, like speculative
    # decoding on a TTS model.
    with pytest.raises(ValueError, match="not supported for this deployment"):
        engine_module.load_model(model=str(model_dir),
                                 enable_in_flight_batching=True)


def test_ifb_unsupported_reason_names_the_deployment():
    from experimental.server.runtime.engine import ifb_unsupported_reason

    text = SimpleNamespace(engine_type=EngineType.LLM, has_speech=False)
    speech = SimpleNamespace(engine_type=EngineType.LLM, has_speech=True)
    spec = SimpleNamespace(engine_type=EngineType.SPEC_DECODE,
                           has_speech=False)
    assert ifb_unsupported_reason(text) is None
    assert ifb_unsupported_reason(speech) is None  # text-only, not refused
    assert "speculative" in ifb_unsupported_reason(spec)
    assert "TTS" in ifb_unsupported_reason(None, model_type="qwen3_tts")


@pytest.mark.parametrize("layout, needle", [
    (SimpleNamespace(engine_type=EngineType.SPEC_DECODE,
                     has_speech=False), "speculative"),
])
def test_runtime_load_refuses_ifb_for_an_unsupported_deployment(
        monkeypatch, layout, needle):
    # The operator asked for in-flight batching on a deployment that cannot
    # honour it: fail at startup with the reason, never fall back silently.
    from experimental.server.runtime import engine as engine_module

    class Runtime:

        def __init__(self, *args):
            raise AssertionError("no runtime may be built before the check")

    monkeypatch.setattr(
        engine_module, "_import_runtime",
        lambda: SimpleNamespace(ContextCacheConfig=lambda: SimpleNamespace(),
                                LLMRuntime=Runtime))
    llm = LLM.__new__(LLM)
    llm._layout = layout
    llm._bundle_dir = "/bundle"
    llm._media_dir = ""
    llm._model_dir = "/model"
    llm._context_cache_config = ContextCacheConfig()
    llm._enable_in_flight_batching = True
    with pytest.raises(ValueError, match=needle):
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
    from experimental.server.runtime.engine_client import _capabilities_for

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
    from experimental.server.config import DEFAULT_MAX_QUEUED_REQUESTS

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
    from experimental.server.runtime.engine_client import EngineClient

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
