# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""End-to-end protocol tests over the modular server and a fake HLAPI."""

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experimental.server.api.errors import ServerOverloadedError
from experimental.server.config import ApiConfig
from experimental.server.runtime.engine import (CompletionOutput, LogprobEntry,
                                                SamplingParams, StreamDelta)


def _create_app(llm, config=None):
    from experimental.server.api.app import create_app
    from experimental.server.runtime.engine_client import EngineClient

    config = config or ApiConfig()
    return create_app(EngineClient(llm, config), config)


class _FakeLLM:
    runtime_kind = "chat"
    video_capable = False
    has_draft_model = False
    max_batch_size = 1

    def __init__(self, root: Path):
        self.model_dir = str(root / "qwen3")
        self.bundle_dir = str(root / "engines")
        self.model_id = "fake-model"
        self.prepared_messages = None
        self.prepare_count = 0
        self.close_count = 0
        self.last_audio_params = None
        self.last_sampling_params = None
        self.next_text = "answer"
        Path(self.model_dir).mkdir(parents=True)
        visual = Path(self.bundle_dir) / "visual"
        visual.mkdir(parents=True)
        (visual / "visual.engine").touch()
        audio = Path(self.bundle_dir) / "audio"
        audio.mkdir(parents=True)
        (audio / "audio_encoder.engine").touch()
        (audio / "config.json").write_text('{"model_type":"qwen3_asr"}',
                                           encoding="utf-8")
        (Path(self.bundle_dir) / "config.json").write_text(
            json.dumps({
                "builder_config": {
                    "max_input_len": 32,
                    "max_batch_size": 1,
                    "max_kv_cache_capacity": 128,
                },
                "kv_cache_dtype": "fp16",
            }),
            encoding="utf-8",
        )
        for component, engine in (("talker", "llm.engine"), ("code_predictor",
                                                             "llm.engine"),
                                  ("code2wav", "code2wav.engine")):
            directory = Path(self.bundle_dir) / component
            directory.mkdir()
            (directory / engine).touch()
        from experimental.server.runtime.engine_layout import inspect_bundle
        self.bundle_layout = inspect_bundle(self.bundle_dir)

    def _make_generation_request(self, messages, params, **_kwargs):
        self.prepared_messages = messages
        self.last_sampling_params = params
        self.prepare_count += 1
        return object()

    def _count_prepared_prompt_tokens(self, _request):
        return 7

    def _complete_prepared_request(self, _request, _params, tool_config,
                                   **_kwargs):
        if tool_config.parse_output:
            return CompletionOutput(
                token_ids=[1, 2],
                prompt_tokens=7,
                finish_reason="tool_calls",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                }],
            )
        return CompletionOutput(text=self.next_text,
                                reasoning="plan",
                                token_ids=[1, 2, 3],
                                prompt_tokens=7,
                                finish_reason="stop")

    def generate_stream(self, _messages, _params, *, tools=None, **_kwargs):
        if tools:
            yield StreamDelta(text="Before<tool_",
                              token_ids=[1],
                              prompt_tokens=7)
            yield StreamDelta(text=('call>{"name":"get_weather",'
                                    '"arguments":{"city":"Paris"}}'
                                    '</tool_call>'),
                              token_ids=[2])
            yield StreamDelta(text="After",
                              finished=True,
                              finish_reason="stop")
            return
        yield StreamDelta(text="hello",
                          token_ids=[1, 2],
                          prompt_tokens=7,
                          finished=True,
                          finish_reason="stop")

    def generate_stream_with_audio(self, _messages, _params, **_kwargs):
        self.last_audio_params = _kwargs.get("audio_params")
        yield StreamDelta(text="spoken",
                          token_ids=[1],
                          prompt_tokens=7,
                          audio_bytes=b"\x01\x00")
        yield StreamDelta(audio_bytes=b"\x02\x00",
                          prompt_tokens=7,
                          finished=True,
                          finish_reason="stop")

    def generate_speech_stream(self, _text, _params):
        self.last_audio_params = _params
        yield StreamDelta(audio_bytes=b"\x01\x00\x02\x00")

    def list_voices(self):
        return ["Ryan"]

    def close(self):
        self.close_count += 1


def _engine():
    from experimental.server.runtime import engine
    return engine


# ---------------------------------------------------------------------------
# Nemotron-Omni video model-family + engine-minimum accounting
# (ported to the modular runtime.engine / media.video_sampling API)
# ---------------------------------------------------------------------------


class _FakeVideoBuffer:
    """ImageData stand-in exposing the fields the min-profile check reads."""

    def __init__(self, video, frames):
        self.video = video
        self.frames = frames


def test_video_model_family_nemotron(tmp_path):
    # A visual engine whose config.json model_type is the Nemotron-Omni vision
    # encoder resolves to the "nemotron" frame-sampling family.
    eng = _engine()
    root = tmp_path / "nemo"
    root.mkdir()
    (root / "visual").mkdir()
    (root / "visual" / "config.json"
     ).write_text('{"model_type": "nemotron_omni_vision_encoder"}')
    (root / "visual" / "visual.engine").touch()
    llm = eng.LLM.__new__(eng.LLM)
    llm._media_dir = str(root)
    assert llm._video_model_family() == "nemotron"

    (root / "visual" / "config.json").write_text(
        '{"model_type": "nemotron_omni_vision_encoder", '
        '"supports_video": false}')
    image_only = eng.LLM.__new__(eng.LLM)
    image_only._media_dir = str(root)
    with pytest.raises(ValueError, match="video input is not supported"):
        image_only._video_model_family()


def test_load_image_buffers_nemotron_minimum():
    # A Nemotron video buffer is built and its EVS token estimate is honored
    # against the request-wide engine minimum (no cu_seqlens binding).
    eng = _engine()
    limits = {
        "model_type": "nemotron_omni_vision_encoder",
        "min_image_tokens": 256,
        "max_image_tokens": 4096,
        "max_image_tokens_per_image": 4096,
        "video_pruning_rate": 0.0,
        "video_temporal_patch_size": 2,
        "video_target_num_patches": 1024,
        "downsample_ratio": 0.5,
    }

    def fake_load_video_buffer(rt,
                               item,
                               family,
                               frame_limits=None,
                               budget=None,
                               pixel_budget=None,
                               cu_budget=None):
        # 8 frames = 4 tubelets (T=2), 4*256 EVS tokens, no cu_seqlens groups.
        return _FakeVideoBuffer(item["video"], frames=8), 4 * 256, 0, 0

    import experimental.server.media.video_sampling as vs_mod
    orig = vs_mod.load_video_buffer
    vs_mod.load_video_buffer = fake_load_video_buffer
    try:
        bufs = eng._load_image_buffers(
            None, [{
                "role": "user",
                "content": [{
                    "type": "video",
                    "video": "a.mp4"
                }]
            }], lambda: "nemotron", lambda: limits)
        assert [b.video for b in bufs] == ["a.mp4"]
    finally:
        vs_mod.load_video_buffer = orig


def test_load_image_buffers_nemotron_minimum_uses_raw_tubelets():
    # The ViT processes every tubelet, so the engine-minimum check must use the
    # pre-EVS tubelet count, not the pruned estimate: a heavily-pruned clip whose
    # raw tubelets clear the minimum must not be rejected.
    eng = _engine()
    limits = {
        "model_type": "nemotron_omni_vision_encoder",
        "min_image_tokens": 1024,
        "max_image_tokens": 8192,
        "max_image_tokens_per_image": 8192,
        "video_pruning_rate": 0.7,
        "video_temporal_patch_size": 2,
        "video_target_num_patches": 1024,
        "downsample_ratio": 0.5,
    }

    def fake_load_video_buffer(rt,
                               item,
                               family,
                               frame_limits=None,
                               budget=None,
                               pixel_budget=None,
                               cu_budget=None):
        # 8 frames = 4 tubelets: raw 1024 >= min, but EVS(0.7) prunes to ~307.
        return _FakeVideoBuffer(item["video"], frames=8), 307, 0, 0

    import experimental.server.media.video_sampling as vs_mod
    orig = vs_mod.load_video_buffer
    vs_mod.load_video_buffer = fake_load_video_buffer
    try:
        bufs = eng._load_image_buffers(
            None, [{
                "role": "user",
                "content": [{
                    "type": "video",
                    "video": "a.mp4"
                }]
            }], lambda: "nemotron", lambda: limits)
        assert [b.video for b in bufs] == ["a.mp4"]
    finally:
        vs_mod.load_video_buffer = orig


def test_load_image_buffers_nemotron_rejects_multiple_and_mixed():
    # The C++ Nemotron video path is single-video, no mixed images, batch 1; the
    # server rejects other layouts up front instead of crashing the runner.
    eng = _engine()
    two_videos = [{
        "role":
        "user",
        "content": [{
            "type": "video",
            "video": "a.mp4"
        }, {
            "type": "video",
            "video": "b.mp4"
        }],
    }]
    with pytest.raises(ValueError, match="exactly one video"):
        eng._load_image_buffers(None, two_videos, lambda: "nemotron",
                                lambda: {})
    mixed = [{
        "role":
        "user",
        "content": [{
            "type": "image",
            "image": "x.jpg"
        }, {
            "type": "video",
            "video": "a.mp4"
        }],
    }]
    with pytest.raises(ValueError, match="exactly one video"):
        eng._load_image_buffers(None, mixed, lambda: "nemotron", lambda: {})


@pytest.fixture
def client_and_llm(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    llm = _FakeLLM(tmp_path)
    config = ApiConfig(enable_auto_tool_choice=True,
                       allowed_local_media_path=str(tmp_path))
    return TestClient(_create_app(llm, config)), llm


def _tool():
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object"
            },
        },
    }


def _sse_payloads(response):
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines() if line.startswith("data: {")
    ]


def test_engine_client_releases_single_runtime_slot(tmp_path):
    from experimental.server.runtime.engine import SamplingParams
    from experimental.server.runtime.engine_client import EngineClient

    async def exercise():
        client = EngineClient(_FakeLLM(tmp_path),
                              ApiConfig(max_queued_requests=0))
        first = await client.prepare_request([{
            "role": "user",
            "content": "first"
        }], SamplingParams())
        assert client.active_requests == 1
        with pytest.raises(ServerOverloadedError):
            await client.prepare_request([{
                "role": "user",
                "content": "second"
            }], SamplingParams())

        first.release()
        second = await client.prepare_request([{
            "role": "user",
            "content": "second"
        }], SamplingParams())
        second.release()
        assert client.active_requests == 0

    asyncio.run(exercise())


def test_stream_lease_is_held_until_iterator_cleanup_finishes(tmp_path):
    import threading

    from experimental.server.runtime.engine import SamplingParams
    from experimental.server.runtime.engine_client import EngineClient

    llm = _FakeLLM(tmp_path)
    cleanup_entered = threading.Event()
    cleanup_may_finish = threading.Event()

    def blocking_stream(*_args, **_kwargs):
        try:
            yield StreamDelta(text="first", token_ids=[1])
        finally:
            cleanup_entered.set()
            cleanup_may_finish.wait()

    llm.generate_stream = blocking_stream

    async def exercise():
        client = EngineClient(llm, ApiConfig(max_queued_requests=0))
        prepared = await client.prepare_request([{
            "role": "user",
            "content": "first"
        }], SamplingParams())
        stream = client.stream([{
            "role": "user",
            "content": "first"
        }],
                               SamplingParams(),
                               prepared=prepared)
        assert (await anext(stream)).text == "first"

        closing = asyncio.create_task(stream.aclose())
        assert await asyncio.to_thread(cleanup_entered.wait, 1.0)
        assert client.active_requests == 1
        with pytest.raises(ServerOverloadedError):
            await client.prepare_request([{
                "role": "user",
                "content": "second"
            }], SamplingParams())

        cleanup_may_finish.set()
        await closing
        assert client.active_requests == 0

    asyncio.run(exercise())


def test_app_shutdown_closes_runtime_exactly_once(tmp_path):
    from fastapi.testclient import TestClient

    llm = _FakeLLM(tmp_path)
    app = _create_app(llm)
    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 200
        assert llm.close_count == 0
    assert llm.close_count == 1

    asyncio.run(app.state.engine_client.close())
    assert llm.close_count == 1


def test_stream_disconnect_before_headers_releases_admission(tmp_path):
    llm = _FakeLLM(tmp_path)
    app = _create_app(llm)
    body = json.dumps({
        "messages": [{
            "role": "user",
            "content": "hi"
        }],
        "stream": True,
    }).encode()
    scope = {
        "type":
        "http",
        "http_version":
        "1.1",
        "method":
        "POST",
        "path":
        "/v1/chat/completions",
        "raw_path":
        b"/v1/chat/completions",
        "root_path":
        "",
        "scheme":
        "http",
        "query_string":
        b"",
        "headers": [(b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode())],
        "client": ("test", 1),
        "server": ("test", 80),
    }
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    class Disconnect(Exception):
        pass

    async def send(_message):
        raise Disconnect()

    async def run():
        with pytest.raises(Disconnect):
            await app(scope, receive, send)

    asyncio.run(run())
    client = app.state.engine_client
    assert client.active_requests == 0
    assert client.queued_requests == 0


@pytest.mark.parametrize("path,anthropic", [
    ("/v1/chat/completions", False),
    ("/v1/messages", True),
])
def test_invalid_json_returns_protocol_error_without_inference(
        client_and_llm, path, anthropic):
    client, llm = client_and_llm
    response = client.post(path,
                           content=b'{"messages":',
                           headers={"content-type": "application/json"})
    assert response.status_code == 400
    payload = response.json()
    if anthropic:
        assert payload["type"] == "error"
        assert payload["error"]["type"] == "invalid_request_error"
    else:
        assert payload["error"]["type"] == "invalid_request_error"
    assert llm.prepare_count == 0


def test_chat_prepares_multimodal_request_once(client_and_llm):
    client, llm = client_and_llm
    image = base64.b64encode(b"image").decode()
    response = client.post("/v1/chat/completions",
                           json={
                               "model":
                               "fake-model",
                               "messages": [{
                                   "role":
                                   "user",
                                   "content": [{
                                       "type": "image_url",
                                       "image_url": {
                                           "url":
                                           f"data:image/png;base64,{image}"
                                       },
                                   }, {
                                       "type": "text",
                                       "text": "Describe it",
                                   }],
                               }],
                               "max_tokens":
                               8,
                           })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "answer"
    assert body["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }
    assert llm.prepare_count == 1


def test_chat_forwards_context_cache_request_policies(client_and_llm):
    client, llm = client_and_llm
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role":
                                   "user",
                                   "content":
                                   "Do not reuse this prefix"
                               }],
                               "reuse_context":
                               False,
                               "cache_generated_tokens":
                               False,
                           })

    assert response.status_code == 200, response.text
    assert not llm.last_sampling_params.reuse_context
    assert not llm.last_sampling_params.cache_generated_tokens


def test_openai_tools_and_reasoning_fields(client_and_llm):
    client, _ = client_and_llm
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role": "user",
                                   "content": "Weather?"
                               }],
                               "tools": [_tool()],
                               "tool_choice":
                               "required",
                               "parallel_tool_calls":
                               False,
                               "max_tokens":
                               8,
                           })
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == (
        "get_weather")


def test_streaming_usage_and_done_marker(client_and_llm):
    client, _ = client_and_llm
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role": "user",
                                   "content": "Hello"
                               }],
                               "stream": True,
                               "stream_options": {
                                   "include_usage": True
                               },
                               "max_tokens": 8,
                           })
    assert response.status_code == 200
    chunks = [
        line.removeprefix("data: ") for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert chunks[-1] == "[DONE]"
    usage = json.loads(chunks[-2])["usage"]
    assert usage["prompt_tokens"] == 7
    assert usage["completion_tokens"] == 2


def test_streaming_omits_usage_by_default(client_and_llm):
    client, _ = client_and_llm
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role": "user",
                                   "content": "Hello"
                               }],
                               "stream": True,
                           })
    assert response.status_code == 200
    assert all(
        payload.get("usage") is None for payload in _sse_payloads(response))


def test_streaming_tool_call_and_usage_share_one_contract(client_and_llm):
    client, _ = client_and_llm
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role": "user",
                                   "content": "Weather?"
                               }],
                               "tools": [_tool()],
                               "tool_choice":
                               "required",
                               "stream":
                               True,
                               "stream_options": {
                                   "include_usage": True
                               },
                           })
    assert response.status_code == 200, response.text
    payloads = _sse_payloads(response)
    tool_chunks = [
        payload for payload in payloads if payload.get("choices")
        and payload["choices"][0]["delta"].get("tool_calls")
    ]
    choices = [
        payload["choices"][0] for payload in payloads if payload.get("choices")
    ]
    usage_chunks = [payload for payload in payloads if payload.get("usage")]
    initial_call = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert initial_call["index"] == 0
    assert initial_call["type"] == "function"
    assert initial_call["function"] == {
        "name": "get_weather",
        "arguments": "",
    }
    argument_delta = tool_chunks[1]["choices"][0]["delta"]["tool_calls"][0]
    assert argument_delta["index"] == 0
    assert json.loads(argument_delta["function"]["arguments"]) == {
        "city": "Paris"
    }
    tool_position = next(i for i, choice in enumerate(choices)
                         if choice["delta"].get("tool_calls"))
    trailing_content = next(i for i, choice in enumerate(choices)
                            if choice["delta"].get("content") == "After")
    assert tool_position < trailing_content
    assert choices[-1]["finish_reason"] == "tool_calls"
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["choices"] == []
    assert usage_chunks[0]["usage"]["prompt_tokens"] == 7


def test_tool_call_delta_does_not_wait_for_native_stream_end(tmp_path):
    from experimental.server.api.protocol import ChatCompletionRequest
    from experimental.server.api.serving_chat import (OpenAIServingChat,
                                                      PreparedChatRequest)
    from experimental.server.parsing.tool_calling import validate_tool_request

    release_finish = asyncio.Event()

    class BlockingClient:
        llm = SimpleNamespace(model_dir=str(tmp_path))
        model_name = "fake-model"

        async def stream(self, *_args, **_kwargs):
            yield StreamDelta(
                text=('<tool_call>{"name":"get_weather","arguments":'
                      '{"city":"Paris"}}</tool_call>'),
                token_ids=[1],
                prompt_tokens=7,
            )
            await release_finish.wait()
            yield StreamDelta(finished=True, finish_reason="stop")

    request = ChatCompletionRequest(
        messages=[{
            "role": "user",
            "content": "Weather?"
        }],
        tools=[_tool()],
        tool_choice="required",
        stream=True,
    )
    prepared = PreparedChatRequest(
        sampling=SamplingParams(),
        tool_config=validate_tool_request(request.messages, request.tools,
                                          request.tool_choice),
        reasoning_parser="none",
    )

    async def exercise():
        handler = OpenAIServingChat(BlockingClient(),
                                    ApiConfig(enable_auto_tool_choice=True))
        chunks = handler._stream_tools(request, prepared, "chatcmpl-test", 0,
                                       False, None)
        first = await anext(chunks)
        assert not release_finish.is_set()
        payload = json.loads(first.removeprefix("data: "))
        call = payload["choices"][0]["delta"]["tool_calls"][0]
        assert call["function"]["name"] == "get_weather"
        release_finish.set()
        tail = [chunk async for chunk in chunks]
        assert tail[-1] == "data: [DONE]\n\n"

    asyncio.run(exercise())


def test_streaming_with_tools_keeps_plain_text_incremental(client_and_llm):
    # #719 regression: tools + tool_choice=auto + stream must not buffer a
    # plain-text answer until generation ends.
    client, llm = client_and_llm

    def word_stream(_messages, _params, **_kwargs):
        words = ["The ", "weather ", "looks ", "fine ", "today."]
        for i, word in enumerate(words):
            last = i == len(words) - 1
            yield StreamDelta(text=word,
                              token_ids=[i],
                              prompt_tokens=7,
                              finished=last,
                              finish_reason="stop" if last else None)

    llm.generate_stream = word_stream
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role": "user",
                                   "content": "Weather?"
                               }],
                               "tools": [_tool()],
                               "tool_choice":
                               "auto",
                               "stream":
                               True,
                           })
    assert response.status_code == 200, response.text
    payloads = _sse_payloads(response)
    contents = [
        payload["choices"][0]["delta"]["content"] for payload in payloads
        if payload.get("choices")
        and payload["choices"][0]["delta"].get("content")
    ]
    assert len(contents) > 1  # exactly 1 before the fix
    assert "".join(contents) == "The weather looks fine today."
    finish = [
        payload["choices"][0]["finish_reason"] for payload in payloads if
        payload.get("choices") and payload["choices"][0].get("finish_reason")
    ]
    assert finish == ["stop"]


def test_streaming_tool_call_wire_format_and_assembly(client_and_llm):
    # OpenAI wire contract: head chunk carries index/id/type/name with empty
    # arguments; later chunks carry only argument fragments; the documented
    # client-side assembly must reproduce the call.
    client, llm = client_and_llm

    def tool_stream(_messages, _params, **_kwargs):
        pieces = [
            "Sure. ",
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Par',
            'is"}}</tool_call>',
        ]
        for i, piece in enumerate(pieces):
            last = i == len(pieces) - 1
            yield StreamDelta(text=piece,
                              token_ids=[i],
                              prompt_tokens=7,
                              finished=last,
                              finish_reason="stop" if last else None)

    llm.generate_stream = tool_stream
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role": "user",
                                   "content": "Weather?"
                               }],
                               "tools": [_tool()],
                               "tool_choice":
                               "auto",
                               "stream":
                               True,
                           })
    assert response.status_code == 200, response.text
    payloads = _sse_payloads(response)
    deltas = [
        payload["choices"][0]["delta"] for payload in payloads
        if payload.get("choices")
    ]
    tool_deltas = [d["tool_calls"][0] for d in deltas if d.get("tool_calls")]
    head, *fragments = tool_deltas
    assert head["id"].startswith("call_") and head["type"] == "function"
    assert head["function"] == {"name": "get_weather", "arguments": ""}
    assert len(fragments) >= 2  # argument bytes streamed, not one late blob
    for fragment in fragments:
        assert set(fragment["function"]) == {"arguments"}
        assert "id" not in fragment
        assert fragment["index"] == head["index"]
    assembled = "".join(f["function"]["arguments"] for f in fragments)
    assert json.loads(assembled) == {"city": "Paris"}
    contents = [d["content"] for d in deltas if d.get("content")]
    assert "".join(contents) == "Sure. "
    finish = [
        payload["choices"][0]["finish_reason"] for payload in payloads if
        payload.get("choices") and payload["choices"][0].get("finish_reason")
    ]
    assert finish == ["tool_calls"]


def test_streaming_parallel_tool_calls_keep_stable_indices(client_and_llm):
    client, llm = client_and_llm

    def two_calls(_messages, _params, **_kwargs):
        yield StreamDelta(
            text=('<tool_call>{"name": "get_weather", "arguments": '
                  '{"city": "A"}}</tool_call> then '),
            token_ids=[1],
            prompt_tokens=7)
        yield StreamDelta(
            text=('<tool_call>{"name": "get_weather", "arguments": '
                  '{"city": "B"}}</tool_call>'),
            token_ids=[2],
            prompt_tokens=7,
            finished=True,
            finish_reason="stop")

    llm.generate_stream = two_calls
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role": "user",
                                   "content": "Weather?"
                               }],
                               "tools": [_tool()],
                               "tool_choice":
                               "auto",
                               "stream":
                               True,
                           })
    assert response.status_code == 200, response.text
    calls = {}
    for payload in _sse_payloads(response):
        for choice in payload.get("choices") or []:
            for tc in choice["delta"].get("tool_calls") or []:
                slot = calls.setdefault(tc["index"], {"id": None, "args": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                slot["args"] += tc.get("function", {}).get("arguments", "")
    assert sorted(calls) == [0, 1]
    assert calls[0]["id"] != calls[1]["id"]
    assert json.loads(calls[0]["args"]) == {"city": "A"}
    assert json.loads(calls[1]["args"]) == {"city": "B"}


def test_streaming_tool_call_truncation_keeps_length_finish_reason(
        client_and_llm):
    # Design R2: a call cut off by max_tokens must not be re-labelled
    # "tool_calls"; the client needs "length" to treat it as truncated.
    client, llm = client_and_llm

    def truncated(_messages, _params, **_kwargs):
        yield StreamDelta(
            text='<tool_call>{"name": "get_weather", "arguments": {"city": "tr',
            token_ids=[1],
            prompt_tokens=7,
            finished=True,
            finish_reason="length")

    llm.generate_stream = truncated
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role": "user",
                                   "content": "Weather?"
                               }],
                               "tools": [_tool()],
                               "tool_choice":
                               "auto",
                               "stream":
                               True,
                           })
    payloads = _sse_payloads(response)
    finish = [
        payload["choices"][0]["finish_reason"] for payload in payloads if
        payload.get("choices") and payload["choices"][0].get("finish_reason")
    ]
    assert finish == ["length"]
    args = "".join(
        tc.get("function", {}).get("arguments", "") for payload in payloads
        for choice in payload.get("choices") or []
        for tc in choice["delta"].get("tool_calls") or [])
    assert args == '{"city": "tr'


def test_streaming_tools_with_thinking_splits_reasoning(client_and_llm):
    # D5 mainstream case: reasoning closes, then the tool call follows.
    client, llm = client_and_llm

    def thinking_then_tool(_messages, _params, **_kwargs):
        for i, piece in enumerate([
                "plan it</think>",
                "ok ",
                '<tool_call>{"name": "get_weather", "arguments": {}}'
                "</tool_call>",
        ]):
            yield StreamDelta(text=piece,
                              token_ids=[i],
                              prompt_tokens=7,
                              finished=piece.endswith("</tool_call>"),
                              finish_reason="stop"
                              if piece.endswith("</tool_call>") else None)

    llm.generate_stream = thinking_then_tool
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role": "user",
                                   "content": "Weather?"
                               }],
                               "tools": [_tool()],
                               "tool_choice":
                               "auto",
                               "stream":
                               True,
                               "enable_thinking":
                               True,
                           })
    assert response.status_code == 200, response.text
    deltas = [
        payload["choices"][0]["delta"] for payload in _sse_payloads(response)
        if payload.get("choices")
    ]
    reasoning = "".join(d.get("reasoning_content") or "" for d in deltas)
    contents = "".join(d.get("content") or "" for d in deltas)
    heads = [
        tc for d in deltas for tc in d.get("tool_calls") or [] if tc.get("id")
    ]
    assert reasoning == "plan it"
    assert contents == "ok "
    assert len(heads) == 1 and heads[0]["function"]["name"] == "get_weather"


def _stream_with_logprobs(pieces):
    """One delta per piece, each carrying its own single-token logprob step."""

    def stream(_messages, _params, **_kwargs):
        for i, piece in enumerate(pieces):
            last = i == len(pieces) - 1
            yield StreamDelta(text=piece,
                              token_ids=[i],
                              prompt_tokens=7,
                              logprobs=[[
                                  LogprobEntry(token_id=i,
                                               logprob=-0.5,
                                               token=piece,
                                               bytes=list(piece.encode()))
                              ]],
                              finished=last,
                              finish_reason="stop" if last else None)

    return stream


# Deltas short enough that the tool and reasoning parsers withhold them, which
# is the normal shape of token-by-token streaming.
_LOGPROB_CASES = {
    "plain": ["Sun", "ny", " today", "."],
    "thinking": ["The", " sky", " is", "</think>", "Sun", "ny."],
    "tools": [
        "Sure. ",
        "<tool_",
        'call>{"name": "get_weather", "arguments": {"city": "Par',
        'is"}}</tool_call>',
    ],
    "tools+thinking": [
        "plan",
        " it</think>",
        "ok ",
        '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>',
    ],
}


def _stream_logprob_request(client, case, extra):
    body = {
        "messages": [{
            "role": "user",
            "content": "Weather?"
        }],
        "stream": True,
        "enable_thinking": "thinking" in case,
        **extra,
    }
    if "tools" in case:
        body["tools"] = [_tool()]
        body["tool_choice"] = "auto"
    return client.post("/v1/chat/completions", json=body)


@pytest.mark.parametrize("case", sorted(_LOGPROB_CASES))
@pytest.mark.parametrize("extra", [{}, {
    "logprobs": True
}, {
    "logprobs": True,
    "top_logprobs": 2
}],
                         ids=["off", "on", "on+top"])
def test_streaming_logprobs_hold_across_tool_and_thinking_combos(
        client_and_llm, case, extra):
    # #719 follow-up: both streaming paths dropped the logprobs of any delta
    # whose bytes the tool or reasoning parser withheld, while the
    # non-streaming path returned them for the same request.
    client, llm = client_and_llm
    pieces = _LOGPROB_CASES[case]
    llm.generate_stream = _stream_with_logprobs(pieces)

    response = _stream_logprob_request(client, case, extra)
    assert response.status_code == 200, response.text
    choices = [
        payload["choices"][0] for payload in _sse_payloads(response)
        if payload.get("choices")
    ]
    entries = [
        entry for choice in choices if choice["logprobs"]
        for entry in choice["logprobs"]["content"]
    ]
    if extra:
        # Concatenating the chunks reproduces the generated token sequence:
        # every delta delivers its logprobs exactly once, in order.
        assert [entry["token_id"]
                for entry in entries] == list(range(len(pieces)))
        assert all(
            bool(entry["top_logprobs"]) == ("top_logprobs" in extra)
            for entry in entries)
    else:
        assert entries == []
    assert choices[-1]["finish_reason"] == ("tool_calls"
                                            if "tools" in case else "stop")


def test_streaming_tools_carry_logprobs_on_head_and_empty_chunks(
        client_and_llm):
    # A delta reaches the wire in one of two shapes, and both must carry the
    # logprobs: as the first of the chunks it expands into, or -- when the
    # parser withheld all of its bytes -- as an otherwise empty delta.
    client, llm = client_and_llm
    llm.generate_stream = _stream_with_logprobs(_LOGPROB_CASES["tools"])

    response = _stream_logprob_request(client, "tools", {"logprobs": True})
    assert response.status_code == 200, response.text
    carried = [
        choice["delta"] for payload in _sse_payloads(response)
        if payload.get("choices") for choice in [payload["choices"][0]]
        if choice["logprobs"]
    ]
    assert carried[0]["content"] == "Sure. "
    assert not any(carried[1][field] for field in ("content", "tool_calls"))
    assert carried[2]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert carried[3]["tool_calls"][0]["function"]["arguments"]


def test_streaming_audio_orders_text_pcm_usage_and_done(client_and_llm):
    client, _ = client_and_llm
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role": "user",
                                   "content": "Say hello"
                               }],
                               "modalities": ["text", "audio"],
                               "audio": {
                                   "voice": "Ryan",
                                   "format": "pcm16"
                               },
                               "stream":
                               True,
                               "stream_options": {
                                   "include_usage": True
                               },
                           })
    assert response.status_code == 200, response.text
    payloads = _sse_payloads(response)
    choices = [
        payload["choices"][0] for payload in payloads if payload.get("choices")
    ]
    text_index = next(i for i, choice in enumerate(choices)
                      if choice["delta"].get("content") == "spoken")
    audio_chunks = [(i, choice["delta"]["audio"]["data"])
                    for i, choice in enumerate(choices)
                    if choice["delta"].get("audio")]
    assert text_index < audio_chunks[-1][0]
    assert b"".join(base64.b64decode(data)
                    for _, data in audio_chunks) == b"\x01\x00\x02\x00"
    assert len([payload for payload in payloads if payload.get("usage")]) == 1
    assert response.text.rstrip().endswith("data: [DONE]")


def test_streaming_input_error_is_returned_before_sse_headers(client_and_llm):
    client, llm = client_and_llm

    def reject(*_args, **_kwargs):
        raise ValueError("invalid media payload")

    llm._make_generation_request = reject
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role": "user",
                                   "content": "bad media"
                               }],
                               "stream":
                               True,
                           })
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert "invalid media payload" in response.json()["error"]["message"]


def test_anthropic_claude_code_tools_images_and_count(client_and_llm):
    client, llm = client_and_llm
    image = base64.b64encode(b"image").decode()
    body = {
        "model":
        "fake-model",
        "max_tokens":
        16,
        "messages": [{
            "role":
            "user",
            "content": [{
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image,
                },
            }, {
                "type": "text",
                "text": "Weather?"
            }],
        }],
        "tools": [{
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {
                "type": "object"
            },
        }],
        "tool_choice": {
            "type": "any"
        },
    }
    response = client.post("/v1/messages", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["content"][0]["type"] == "tool_use"
    content = llm.prepared_messages[-1]["content"]
    assert content[0]["type"] == "image_url"

    count_body = dict(body)
    count_body.pop("max_tokens")
    count = client.post("/v1/messages/count_tokens", json=count_body)
    assert count.status_code == 200
    assert count.json() == {"input_tokens": 7}


def test_anthropic_stream_uses_valid_event_order(client_and_llm):
    client, _ = client_and_llm
    response = client.post("/v1/messages",
                           json={
                               "model": "fake-model",
                               "max_tokens": 8,
                               "stream": True,
                               "messages": [{
                                   "role": "user",
                                   "content": "Hello"
                               }],
                           })
    assert response.status_code == 200
    events = [
        line.removeprefix("event: ") for line in response.text.splitlines()
        if line.startswith("event: ")
    ]
    assert events[0] == "message_start"
    assert events[-1] == "message_stop"
    assert events.index("content_block_start") < events.index(
        "content_block_stop")


def test_anthropic_stream_orders_tool_and_following_content(client_and_llm):
    client, _ = client_and_llm
    response = client.post(
        "/v1/messages",
        json={
            "model":
            "fake-model",
            "max_tokens":
            8,
            "stream":
            True,
            "messages": [{
                "role": "user",
                "content": "Weather?"
            }],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object"
                },
            }],
            "tool_choice": {
                "type": "any"
            },
        },
    )
    assert response.status_code == 200, response.text
    events = []
    for frame in response.text.strip().split("\n\n"):
        lines = frame.splitlines()
        if len(lines) == 2 and lines[0].startswith("event: "):
            events.append((lines[0].removeprefix("event: "),
                           json.loads(lines[1].removeprefix("data: "))))

    deltas = [(index, data["delta"])
              for index, (event, data) in enumerate(events)
              if event == "content_block_delta"]
    tool_index, tool_delta = next(item for item in deltas
                                  if item[1]["type"] == "input_json_delta")
    after_index, _ = next(item for item in deltas
                          if item[1].get("text") == "After")
    assert json.loads(tool_delta["partial_json"]) == {"city": "Paris"}
    assert tool_index < after_index
    message_delta = next(data for event, data in events
                         if event == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "tool_use"


def test_anthropic_x_api_key_auth(tmp_path):
    from fastapi.testclient import TestClient
    llm = _FakeLLM(tmp_path)
    client = TestClient(_create_app(llm, ApiConfig(api_key="secret")))
    body = {
        "model": "fake-model",
        "max_tokens": 8,
        "messages": [{
            "role": "user",
            "content": "Hello"
        }],
    }
    assert client.post("/v1/messages", json=body).status_code == 401
    assert client.post("/v1/messages",
                       json=body,
                       headers={
                           "x-api-key": "secret"
                       }).status_code == 200


def test_asr_and_omni_audio_routes(client_and_llm):
    client, llm = client_and_llm
    llm.next_text = "language English<asr_text>Hello world"
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        data={
            "model": "fake-model",
            "language": "en",
            "prompt": "Names"
        },
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"text": "Hello world", "language": "English"}
    assert llm.prepared_messages[0] == {
        "role": "system",
        "content": "English",
    }

    speech = client.post("/v1/audio/speech",
                         json={
                             "model": "fake-model",
                             "input": "Hello",
                             "voice": "Ryan",
                             "response_format": "pcm",
                         })
    assert speech.status_code == 200
    assert speech.content == b"\x01\x00\x02\x00"


def test_transcription_text_response_and_validation(client_and_llm):
    client, llm = client_and_llm
    llm.next_text = "transcript"
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        data={
            "model": "fake-model",
            "response_format": "text"
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "transcript"

    invalid_cases = [
        ({
            "model": "other"
        }, "model"),
        ({
            "model": "fake-model",
            "language": "not-a-language"
        }, "language"),
        ({
            "model": "fake-model",
            "temperature": "3"
        }, "temperature"),
        ({
            "model": "fake-model",
            "response_format": "srt"
        }, "response_format"),
    ]
    for data, expected in invalid_cases:
        rejected = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
            data=data,
        )
        assert rejected.status_code in (400, 404)
        assert expected in rejected.text

    empty = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("empty.wav", b"", "audio/wav")},
        data={"model": "fake-model"},
    )
    assert empty.status_code == 400
    assert "empty" in empty.text


def test_transcription_requires_asr_bundle(tmp_path):
    from fastapi.testclient import TestClient

    from experimental.server.runtime.engine_layout import inspect_bundle

    llm = _FakeLLM(tmp_path)
    audio_config = Path(llm.bundle_dir) / "audio" / "config.json"
    audio_config.write_text('{"model_type":"qwen3_omni"}', encoding="utf-8")
    llm.bundle_layout = inspect_bundle(llm.bundle_dir)
    client = TestClient(_create_app(llm))
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        data={"model": "fake-model"},
    )
    assert response.status_code == 400
    assert "not transcription" in response.text


def test_audio_requests_forward_talker_controls(client_and_llm):
    client, llm = client_and_llm
    controls = {
        "voice": "Ryan",
        "talker_temperature": 0.8,
        "talker_top_k": 7,
        "talker_top_p": 0.95,
        "repetition_penalty": 1.1,
        "max_audio_length": 1024,
        "codec_chunk_frames": 5,
        "talker_prefill_threshold": 2,
    }
    response = client.post("/v1/audio/speech",
                           json={
                               "model": "fake-model",
                               "input": "Hello",
                               **controls,
                           })
    assert response.status_code == 200, response.text
    for field, value in controls.items():
        assert getattr(llm.last_audio_params, field) == value

    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role": "user",
                                   "content": "Hello"
                               }],
                               "modalities": ["text", "audio"],
                               "audio": {
                                   "format": "pcm16",
                                   **controls,
                               },
                           })
    assert response.status_code == 200, response.text
    for field, value in controls.items():
        assert getattr(llm.last_audio_params, field) == value


@pytest.mark.parametrize("field,value", [
    ("talker_temperature", -0.1),
    ("talker_top_k", 1.5),
    ("max_audio_length", 0),
    ("codec_chunk_frames", 0),
    ("talker_prefill_threshold", 0),
])
def test_audio_requests_reject_invalid_talker_controls(client_and_llm, field,
                                                       value):
    client, _ = client_and_llm
    response = client.post("/v1/audio/speech",
                           json={
                               "input": "Hello",
                               field: value,
                           })
    assert response.status_code == 400
    assert field in response.text


def test_omni_chat_audio_output(client_and_llm):
    client, _ = client_and_llm
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role": "user",
                                   "content": "Say hello"
                               }],
                               "modalities": ["text", "audio"],
                               "audio": {
                                   "voice": "Ryan",
                                   "format": "pcm16"
                               },
                               "max_tokens":
                               8,
                           })
    assert response.status_code == 200, response.text
    message = response.json()["choices"][0]["message"]
    assert message["content"] == "spoken"
    assert base64.b64decode(message["audio"]["data"]) == b"\x01\x00\x02\x00"


def test_local_media_is_denied_outside_allowed_root(client_and_llm):
    client, _ = client_and_llm
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role":
                                   "user",
                                   "content": [{
                                       "type": "image_url",
                                       "image_url": {
                                           "url": "file:///etc/passwd"
                                       },
                                   }],
                               }],
                               "max_tokens":
                               8,
                           })
    assert response.status_code == 403


def _media_messages(item):
    return [{"role": "user", "content": [item]}]


@pytest.mark.parametrize("item", [
    {
        "type": "image",
        "image": "/etc/passwd.png"
    },
    {
        "type": "image_url",
        "image_url": {
            "url": "/etc/passwd.png"
        }
    },
    {
        "type": "video",
        "video": "/etc/passwd.mp4"
    },
    {
        "type": "video",
        "video": {
            "url": "/etc/passwd.mp4"
        }
    },
    {
        "type": "video_url",
        "video_url": {
            "url": "/etc/passwd.mp4"
        }
    },
    {
        "type": "video",
        "frames": ["/etc/passwd.png"]
    },
    {
        "type": "audio_url",
        "audio_url": {
            "url": "/etc/passwd.wav"
        }
    },
])
def test_local_media_policy_checks_every_loader_spelling(item):
    from experimental.server.media.media_source import \
        enforce_local_media_policy

    with pytest.raises(PermissionError):
        enforce_local_media_policy(_media_messages(item), "")


def test_local_media_policy_allows_root_and_rejects_traversal(tmp_path):
    from experimental.server.media.media_source import \
        enforce_local_media_policy

    root = tmp_path / "media"
    root.mkdir()
    inside = root / "image.png"
    inside.touch()
    enforce_local_media_policy(
        _media_messages({
            "type": "image_url",
            "image_url": {
                "url": f"file://{inside}"
            }
        }), str(root))

    outside = tmp_path / "outside.png"
    outside.touch()
    with pytest.raises(PermissionError):
        enforce_local_media_policy(
            _media_messages({
                "type": "image_url",
                "image_url": {
                    "url": str(root / ".." / outside.name)
                }
            }), str(root))


def test_remote_and_data_media_do_not_require_local_root():
    from experimental.server.media.media_source import \
        enforce_local_media_policy

    for ref in ("https://example.com/image.png",
                "data:image/png;base64,aW1hZ2U="):
        enforce_local_media_policy(
            _media_messages({
                "type": "image_url",
                "image_url": {
                    "url": ref
                }
            }), "")


def test_allowed_local_media_path_is_honored_by_http(tmp_path):
    from fastapi.testclient import TestClient

    root = tmp_path / "media"
    root.mkdir()
    image = root / "image.png"
    image.touch()
    llm = _FakeLLM(tmp_path / "runtime")
    config = ApiConfig(allowed_local_media_path=str(root))
    client = TestClient(_create_app(llm, config))
    response = client.post("/v1/chat/completions",
                           json={
                               "messages":
                               _media_messages({
                                   "type": "image_url",
                                   "image_url": {
                                       "url": str(image)
                                   }
                               })
                           })
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("marker,status", [
    ("EDGELLM_BAD_MEDIA_COUNT: expected one image", 400),
    ("EDGELLM_INPUT_TOO_LONG: rebuild for this prompt", 413),
    ("CUDA error: out of memory", 500),
])
def test_native_errors_map_to_protocol_status(client_and_llm, marker, status):
    client, llm = client_and_llm

    def fail(*_args, **_kwargs):
        raise RuntimeError(marker)

    llm._complete_prepared_request = fail
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{
            "role": "user",
            "content": "Hello"
        }]})
    assert response.status_code == status
    assert marker in response.text
    expected_type = ("invalid_request_error"
                     if status < 500 else "engine_error")
    assert response.json()["error"]["type"] == expected_type


@pytest.mark.parametrize("field,value", [
    ("top_k", 1.5),
    ("max_tokens", 2.5),
    ("temperature", "hot"),
    ("top_p", 2.0),
])
def test_sampling_schema_rejects_invalid_values(client_and_llm, field, value):
    client, _ = client_and_llm
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role": "user",
                                   "content": "Hello"
                               }],
                               field: value,
                           })
    assert response.status_code == 400
    assert field in response.text


def test_stream_include_usage_requires_boolean(client_and_llm):
    client, _ = client_and_llm
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role": "user",
                                   "content": "Hello"
                               }],
                               "stream": True,
                               "stream_options": {
                                   "include_usage": "false"
                               },
                           })
    assert response.status_code == 400
    assert "include_usage" in response.text


def test_request_body_limit_counts_received_bytes(client_and_llm, monkeypatch):
    from experimental.server.api import app as app_module

    client, _ = client_and_llm
    monkeypatch.setattr(app_module, "MAX_REQUEST_BODY_BYTES", 256)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{
            "role": "user",
            "content": "x" * 1024
        }]})
    assert response.status_code == 413


@pytest.mark.parametrize("value,expected", [
    (None, None),
    ("1024", 1024),
    ("abc", -1),
    ("1.5", -1),
    ("10, 20", -1),
])
def test_content_length_parsing(value, expected):
    from experimental.server.api.app import _content_length

    assert _content_length(value) == expected


def test_app_reports_package_version(tmp_path):
    from fastapi.testclient import TestClient

    from tensorrt_edgellm import __version__

    client = TestClient(_create_app(_FakeLLM(tmp_path)))
    assert client.get("/openapi.json").json()["info"]["version"] == __version__


def test_media_is_rejected_by_the_loaded_model_contract(tmp_path):
    from fastapi.testclient import TestClient

    from experimental.server.runtime.engine_layout import inspect_bundle

    llm = _FakeLLM(tmp_path)
    (Path(llm.bundle_dir) / "visual" / "visual.engine").unlink()
    llm.bundle_layout = inspect_bundle(llm.bundle_dir)
    response = TestClient(_create_app(llm)).post(
        "/v1/chat/completions",
        json={
            "messages": [{
                "role":
                "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.png"
                    },
                }],
            }],
            "max_tokens":
            8,
        },
    )
    assert response.status_code == 400
    assert "does not accept image input" in response.json()["error"]["message"]


def test_remote_media_fetch_is_size_bounded(monkeypatch):
    from experimental.server.media import media_source

    class Response:

        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def geturl(self):
            return "https://example.com/image.png"

        def read(self, _limit):
            return b"oversized"

    monkeypatch.setattr(media_source.urllib.request, "urlopen",
                        lambda *_args, **_kwargs: Response())
    item = {
        "type": "image_url",
        "image_url": {
            "url": "https://example.com/image.png"
        },
    }
    assert media_source.resolve_image_message(item) == b"oversized"
    with pytest.raises(ValueError, match="supported maximum"):
        media_source.fetch_remote_media("https://example.com/image.png",
                                        "image", 4)


def test_image_data_url_reaches_bytes_loader():
    from experimental.server.runtime.engine import _load_image_buffers

    class Image:
        do_resize = True

    class Runtime:
        loaded = None

        @classmethod
        def load_image_from_bytes(cls, data):
            cls.loaded = data
            return Image()

    payload = base64.b64encode(b"encoded-image").decode()
    buffers = _load_image_buffers(Runtime, [{
        "role":
        "user",
        "content": [{
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{payload}"
            },
        }],
    }])
    assert len(buffers) == 1
    assert Runtime.loaded == b"encoded-image"
