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
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_edgellm.server.api.errors import (PayloadTooLargeError,
                                                ServerOverloadedError)
from tensorrt_edgellm.server.config import ApiConfig
from tensorrt_edgellm.server.runtime.engine import (CompletionOutput,
                                                    LogprobEntry,
                                                    SamplingParams,
                                                    StreamDelta)


def _create_app(llm, config=None):
    from tensorrt_edgellm.server.api.app import create_app
    from tensorrt_edgellm.server.runtime.engine_client import EngineClient

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
        self.prepared_prompt = None
        self.prepare_count = 0
        self.close_count = 0
        self.last_audio_params = None
        self.last_sampling_params = None
        self.last_request_options = None
        self.next_text = "answer"
        self.next_reused = None
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
        from tensorrt_edgellm.server.runtime.engine_layout import \
            inspect_bundle
        self.bundle_layout = inspect_bundle(self.bundle_dir)

    def _make_generation_request(self, messages, params, **kwargs):
        self.prepared_messages = messages
        self.last_sampling_params = params
        self.last_request_options = kwargs
        self.prepare_count += 1
        return object()

    def _make_completion_request(self, prompt, params):
        self.prepared_prompt = prompt
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
                reused_prompt_tokens=self.next_reused,
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
                                reused_prompt_tokens=self.next_reused,
                                finish_reason="stop")

    def generate_stream(self, _messages, _params, *, tools=None, **_kwargs):
        if tools:
            yield StreamDelta(text="Before<tool_",
                              token_ids=[1],
                              prompt_tokens=7,
                              reused_prompt_tokens=self.next_reused)
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
                          reused_prompt_tokens=self.next_reused,
                          finished=True,
                          finish_reason="stop")

    def generate_stream_with_audio(self, _messages, _params, **_kwargs):
        self.last_audio_params = _kwargs.get("audio_params")
        yield StreamDelta(text="spoken",
                          token_ids=[1],
                          prompt_tokens=7,
                          reused_prompt_tokens=self.next_reused,
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
    from tensorrt_edgellm.server.runtime import engine
    return engine


@pytest.mark.parametrize(("model_type", "family"), [
    ("qwen3_vl", "qwen"),
    ("cosmos3_edge_vision", "qwen"),
    ("internvl_chat", "internvl"),
    ("muse_glimmer_vision", "muse"),
    ("nemotron_omni_vision_encoder", "nemotron"),
    ("qwen3_omni_audio_encoder", None),
    ("unsupported_vision", None),
])
def test_visual_model_family(model_type, family):
    """Visual model types map to their preprocessing family; audio-side and
    unknown types map to None."""
    assert _engine()._visual_model_family(model_type) == family


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


def test_video_model_family_muse(tmp_path):
    eng = _engine()
    root = tmp_path / "muse"
    (root / "visual").mkdir(parents=True)
    (root / "visual" /
     "config.json").write_text('{"model_type": "muse_glimmer_vision"}')
    (root / "visual" / "visual.engine").touch()
    llm = eng.LLM.__new__(eng.LLM)
    llm._media_dir = str(root)
    assert llm._video_model_family() == "muse"


def test_cosmos3_reasoner_routes_native_video_file(monkeypatch, tmp_path):
    # Cosmos3 uses the Qwen-compatible native video decoder and visual runner.
    eng = _engine()
    root = tmp_path / "cosmos3"
    (root / "visual").mkdir(parents=True)
    (root / "visual" /
     "config.json").write_text('{"model_type": "cosmos3_edge_vision"}')
    (root / "visual" / "visual.engine").touch()
    llm = eng.LLM.__new__(eng.LLM)
    llm._media_dir = str(root)
    assert llm._video_model_family() == "qwen"
    captured = {}

    def fake_load_video_buffer(rt,
                               item,
                               family,
                               frame_limits=None,
                               budget=None,
                               pixel_budget=None,
                               cu_budget=None):
        captured.update(item=item, family=family)
        return _FakeVideoBuffer(item["video_url"]["url"], frames=8), 0, 0, 0

    import tensorrt_edgellm.server.media.video_sampling as vs_mod
    monkeypatch.setattr(vs_mod, "load_video_buffer", fake_load_video_buffer)
    buffers = eng._load_image_buffers(None, [{
        "role":
        "user",
        "content": [{
            "type": "video_url",
            "video_url": {
                "url": "example.mp4"
            }
        }],
    }], llm._video_model_family, lambda: {})

    assert captured == {
        "item": {
            "type": "video_url",
            "video_url": {
                "url": "example.mp4"
            }
        },
        "family": "qwen",
    }
    assert [buffer.video for buffer in buffers] == ["example.mp4"]


def test_cosmos3_reasoner_uses_single_frame_temporal_patches(tmp_path):
    eng = _engine()
    visual = tmp_path / "cosmos3" / "visual"
    visual.mkdir(parents=True)
    (visual / "config.json").write_text(
        json.dumps({
            "model_type": "cosmos3_edge_vision",
            "builder_config": {
                "min_image_tokens": 4,
                "max_image_tokens": 1024,
                "max_image_tokens_per_image": 512,
            },
        }))
    # Cosmos3's processor intentionally omits temporal_patch_size. The C++
    # runner defaults it to one, so server-side accounting must do the same.
    (visual / "preprocessor_config.json").write_text(
        json.dumps({
            "patch_size": 16,
            "merge_size": 2,
        }))
    llm = eng.LLM.__new__(eng.LLM)
    llm._media_dir = str(tmp_path / "cosmos3")

    assert llm._video_frame_limits()["temporal_patch_size"] == 1


def test_load_image_buffers_rejects_image_only_request_over_profile(
        monkeypatch, tmp_path):
    """Image-only requests whose summed estimate exceeds max_image_tokens are
    rejected with the estimate and limit, without video routing."""
    eng = _engine()
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.touch()
    second.touch()
    limits = {
        "model_type": "cosmos3_edge_vision",
        "min_image_tokens": 1,
        "max_image_tokens": 1000,
        "max_image_tokens_per_image": 600,
    }

    from tensorrt_edgellm.server.media import video_sampling as vs_mod
    monkeypatch.setattr(vs_mod, "estimate_image_tokens",
                        lambda *_args, **_kwargs: 600)

    class Runtime:

        @staticmethod
        def load_image_from_path(_path):
            return SimpleNamespace()

    with pytest.raises(
            vs_mod.VisualProfileCapacityError,
            match=r"estimated 1200 visual tokens.*max_image_tokens=1000"):
        eng._load_image_buffers(
            Runtime, [{
                "role":
                "user",
                "content": [{
                    "type": "image",
                    "image": str(first)
                }, {
                    "type": "image",
                    "image": str(second)
                }],
            }], lambda: pytest.fail(
                "image-only requests must not use video routing"),
            lambda: limits)


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

    import tensorrt_edgellm.server.media.video_sampling as vs_mod
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

    import tensorrt_edgellm.server.media.video_sampling as vs_mod
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
    from tensorrt_edgellm.server.runtime.engine import SamplingParams
    from tensorrt_edgellm.server.runtime.engine_client import EngineClient

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

    from tensorrt_edgellm.server.runtime.engine import SamplingParams
    from tensorrt_edgellm.server.runtime.engine_client import EngineClient

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


_REFUSED = "a plugin refused compact execution at runtime"
_STOPPED = "ifbSchedulingMode=mixed requires compact execution, and " + _REFUSED

_POLICY = dict(
    scheduling_policy="priority",
    preemption="none",
    max_num_tokens=0,
    max_chunk_tokens=0,
    priority_reserved_slots=1,
    max_pause_steps=0,
    warnings=[
        "preemption=pause is not applied: auto resolved to prefillFirst"
    ])


def _engine_holder(serving, unavailable=None):
    """An LLM over a native engine stand-in that serves or has stopped with ``unavailable``."""
    from tensorrt_edgellm.server.runtime.engine import LLM

    holder = LLM.__new__(LLM)
    # yapf: disable
    holder._engine = SimpleNamespace(
        status=lambda: SimpleNamespace(serving=serving, unavailable_reason=unavailable or "",
                                       compact_unavailable_reason=_REFUSED),
        resolved_policy=SimpleNamespace(mode="prefillFirst", mode_source="auto", execution="single-cohort",
                                        reason=_REFUSED, **_POLICY))
    # yapf: enable
    return holder


@pytest.mark.parametrize("serving", [None, False],
                         ids=["blocking_path", "engine_stopped"])
def test_health_reports_whether_the_engine_still_serves(tmp_path, serving):
    """/health answers 200 "healthy" with no engine block on the blocking path, and 503 "unhealthy" with the
    engine's policy in effect and its reason once it fails every request; /health/ready answers 200 "ready" and 503
    "unavailable" with the reason alike."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    llm = _FakeLLM(tmp_path)
    unavailable = _STOPPED if serving is False else None
    if serving is not None:
        llm.get_engine_status = _engine_holder(serving,
                                               unavailable).get_engine_status
    # yapf: disable
    engine = None if serving is None else dict(
        serving=serving, unavailable_reason=unavailable, scheduling_mode="prefillFirst", scheduling_mode_source="auto",
        scheduling_reason=_REFUSED, execution="single-cohort", compact_unavailable_reason=_REFUSED, **_POLICY)
    # yapf: enable
    client = TestClient(_create_app(llm))

    response = client.get("/health")
    ready = client.get("/health/ready")

    body = response.json()
    assert response.status_code == (503 if serving is False else 200)
    assert body["status"] == ("unhealthy" if serving is False else "healthy")
    assert body["engine"] == engine
    assert body.get("reason") == unavailable
    assert body["model"] == "fake-model"
    assert ready.status_code == (503 if serving is False else 200)
    assert ready.json()["status"] == ("unavailable"
                                      if serving is False else "ready")
    assert ready.json().get("reason") == unavailable


# yapf: disable
@pytest.mark.parametrize("path, body", [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}]}),
    ("/v1/completions", {"prompt": "Hello"}),
], ids=["chat", "completion"])
# yapf: enable
def test_a_stopped_engine_refuses_requests_with_503_before_preparing_them(
        tmp_path, path, body):
    """Once the engine no longer serves, a request gets HTTP 503 of type engine_unavailable naming the engine's
    reason, without Retry-After, and no native request is built for it."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    llm = _FakeLLM(tmp_path)
    llm.engine_unavailable_reason = _engine_holder(
        False, _STOPPED).engine_unavailable_reason
    response = TestClient(_create_app(llm)).post(path, json=body)

    assert response.status_code == 503, response.text
    assert response.json()["error"]["type"] == "engine_unavailable"
    assert _STOPPED in response.json()["error"]["message"]
    assert "retry-after" not in response.headers
    assert llm.prepare_count == 0


# yapf: disable
@pytest.mark.parametrize("kind, status, error_type, retry_after", [
    ("COMPACT_STEP_REFUSED", 503, "compact_step_refused", "1"),
    ("ENGINE_STOPPED", 503, "engine_unavailable", None),
    ("OTHER", 500, "engine_error", None),
    (None, 500, "engine_error", None),
], ids=["refused_step_retryable", "engine_stopped", "other_failure", "no_kind"])
# yapf: enable
def test_engine_failures_map_to_the_status_a_client_can_act_on(
        tmp_path, kind, status, error_type, retry_after):
    """A request the engine failed with RequestFailureKind COMPACT_STEP_REFUSED gets HTTP 503 of type
    compact_step_refused with Retry-After: 1, and a streamed one an error of that type; ENGINE_STOPPED gets 503
    engine_unavailable without Retry-After; any other failure, with or without a kind, stays 500 engine_error. The
    message is the engine's. /v1/messages answers with the same status and Retry-After and the Anthropic api_error
    type."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from tensorrt_edgellm.server.runtime.engine import _import_runtime

    rt = _import_runtime()
    llm = _FakeLLM(tmp_path)
    llm._rt = rt
    message = "Request 3 failed: the runtime session failed: x"

    def fail(*_args, **_kwargs):
        error = rt.RequestFailedError(message)
        if kind is not None:
            error.kind = getattr(rt.RequestFailureKind, kind)
        raise error
        yield  # pragma: no cover - makes the streaming stand-in a generator

    llm._complete_prepared_request = lambda *_args, **_kwargs: next(fail())
    llm.generate_stream = fail
    client = TestClient(_create_app(llm))
    chat = {"messages": [{"role": "user", "content": "Hello"}]}

    response = client.post("/v1/chat/completions", json=chat)
    streamed = client.post("/v1/chat/completions",
                           json={
                               **chat, "stream": True
                           })

    assert response.status_code == status, response.text
    assert response.json()["error"]["type"] == error_type
    assert response.json()["error"]["message"] == message
    assert response.headers.get("retry-after") == retry_after
    assert error_type in streamed.text, streamed.text

    messages = client.post("/v1/messages",
                           json={
                               "model": "fake-model",
                               "max_tokens": 8,
                               **chat
                           })
    assert messages.status_code == status, messages.text
    assert messages.json()["error"]["type"] == "api_error"
    assert messages.headers.get("retry-after") == retry_after


_TOO_LONG = ("EDGELLM_INPUT_TOO_LONG: prefill length 90000 exceeds engine "
             "max_input_len 81920")
_FINISH_REASONS = dict(NOT_FINISHED=0,
                       END_ID=1,
                       LENGTH=2,
                       CANCELLED=3,
                       ERROR=4,
                       STOP_WORDS=5)


class _NativeChannel:
    """A StreamChannel: the chunks one native request pushed, and its end."""

    def __init__(self):
        self.chunks = []
        self.finished = False
        self.cancelled = False

    def set_skip_special_tokens(self, _enabled):
        pass

    def wait_pop(self, timeout_ms=0):
        if not (self.chunks or self.finished or self.cancelled):
            time.sleep(0.001)
        return self.try_pop()

    def try_pop(self):
        return self.chunks.pop(0) if self.chunks else None

    def is_finished(self):
        return self.finished

    def is_cancelled(self):
        return self.cancelled

    def cancel(self):
        self.cancelled = True


def _native_chunk(text="", token_ids=(), finished=False):
    return SimpleNamespace(
        text=text,
        token_ids=list(token_ids),
        prompt_token_count=7,
        finished=finished,
        reason=_FINISH_REASONS["ERROR"] if finished else None,
        logprobs=[])


def _serve_native(llm, server_path, play):
    """Serve ``llm`` through the real LLM.generate_stream over a native stand-in: on the blocking path a worker
    runs the request, under in-flight batching an engine does and its outcome reports a failure.
    ``play(channel)`` is one request: it pushes chunks, then returns or raises."""
    from fastapi.testclient import TestClient

    from tensorrt_edgellm.server.runtime.engine import LLM

    native = LLM.__new__(LLM)
    native._closed = False
    native._rt = SimpleNamespace(
        FinishReason=SimpleNamespace(**_FINISH_REASONS),
        StreamChannel=SimpleNamespace(
            create=lambda collect_metrics=False: _NativeChannel()))
    if server_path == "blocking":
        native._runtime = object()
        native._handle_request = lambda request: play(request.stream_channels[
            0])
    else:

        def submit(request):
            channel = request.stream_channels[0]
            failure = None
            try:
                play(channel)
            except Exception as error:  # noqa: BLE001 - the outcome reports it
                failure = error
            channel.finished = True

            def outcome():
                if failure is not None:
                    raise failure

            return SimpleNamespace(ready=lambda: True,
                                   cancel=channel.cancel,
                                   get=outcome)

        native._runtime = None
        native._engine = SimpleNamespace(submit=submit)
        llm._engine = native._engine
    llm.generate_stream = native.generate_stream
    llm._make_generation_request = lambda *_args, **_kwargs: SimpleNamespace()
    llm._make_completion_request = lambda *_args, **_kwargs: SimpleNamespace()
    llm._complete_prepared_request = lambda request, *_args, **_kwargs: list(
        native.generate_stream([], prebuilt_request=request))
    return TestClient(_create_app(llm,
                                  ApiConfig(enable_auto_tool_choice=True)))


# yapf: disable
_STREAMED = [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}]}),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Weather?"}], "tools": [_tool()],
                              "tool_choice": "auto"}),
    ("/v1/completions", {"prompt": "Hello"}),
    ("/v1/messages", {"model": "fake-model", "max_tokens": 8, "messages": [{"role": "user", "content": "Hello"}]}),
    ("/v1/responses", {"input": "Hello"}),
]
_STREAMED_IDS = ["chat", "chat_tools", "completion", "messages", "responses"]


@pytest.mark.parametrize("server_path, error, status, terminal_chunk", [
    ("blocking", lambda: RuntimeError(_TOO_LONG), 413, False),
    ("blocking", lambda: RuntimeError(_TOO_LONG), 413, True),
    ("in_flight_batching", lambda: RuntimeError(_TOO_LONG), 413, False),
    ("blocking", lambda: ValueError("invalid sampling argument"), 400, False),
], ids=["blocking", "blocking_after_empty_terminal_chunk", "in_flight_batching", "blocking_invalid_argument"])
@pytest.mark.parametrize("path, body", _STREAMED, ids=_STREAMED_IDS)
# yapf: enable
def test_stream_failing_before_its_first_token_answers_with_the_http_status(
        tmp_path, server_path, error, status, terminal_chunk, path, body):
    """A streamed request that fails before its first token, with or without an empty terminal chunk ahead of the
    failure, gets the HTTP status and JSON error the same request gets without streaming, on every streaming
    endpoint and both server paths."""
    pytest.importorskip("fastapi")

    def play(channel):
        if terminal_chunk:
            channel.chunks.append(_native_chunk(finished=True))
        raise error()

    client = _serve_native(_FakeLLM(tmp_path), server_path, play)
    plain = client.post(path, json=body)
    streamed = client.post(path, json={**body, "stream": True})

    assert plain.status_code == status, plain.text
    assert streamed.status_code == status, streamed.text
    assert streamed.headers["content-type"].startswith("application/json")
    assert streamed.json()["error"].items() >= plain.json()["error"].items()


# yapf: disable
@pytest.mark.parametrize("server_path", ["blocking", "in_flight_batching"])
@pytest.mark.parametrize("path, body", _STREAMED, ids=_STREAMED_IDS)
# yapf: enable
def test_stream_failing_after_its_first_token_reports_the_error_in_band(
        tmp_path, server_path, path, body):
    """A streamed request that fails after its first token keeps HTTP 200 and reports the failure as an error
    event after the text it produced."""
    pytest.importorskip("fastapi")

    def play(channel):
        # Two words: a tool parser holds back the last one until more text arrives.
        channel.chunks.append(_native_chunk(text="partial ", token_ids=[1]))
        channel.chunks.append(_native_chunk(text="answer ", token_ids=[2]))
        raise RuntimeError(_TOO_LONG)

    client = _serve_native(_FakeLLM(tmp_path), server_path, play)
    streamed = client.post(path, json={**body, "stream": True})

    assert streamed.status_code == 200, streamed.text
    assert streamed.headers["content-type"].startswith("text/event-stream")
    assert 0 <= streamed.text.find("partial") < streamed.text.find(_TOO_LONG)


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


def test_stream_disconnect_before_its_first_token_cancels_and_completes_the_exchange(
        tmp_path):
    """A client that disconnects while its streamed request waits for the first token cancels the native request
    and releases admission, and the exchange still ends with a response, not an application error."""
    import threading

    from tensorrt_edgellm.server.runtime.engine import CancellableIterator

    llm = _FakeLLM(tmp_path)
    reading = threading.Event()
    cancelled = threading.Event()

    def native_stream(*_args, **_kwargs):

        def iterate():
            reading.set()
            cancelled.wait()
            return
            yield  # pragma: no cover - makes iterate() a generator

        return CancellableIterator(iterate(), cancelled.set)

    llm.generate_stream = native_stream
    app = _create_app(llm)
    # yapf: disable
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True}).encode()
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
             "method": "POST", "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions", "root_path": "",
             "scheme": "http", "query_string": b"", "client": ("test", 1), "server": ("test", 80),
             "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]}
    # yapf: enable
    received = False
    sent = []

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.to_thread(reading.wait, 1.0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))

    assert cancelled.is_set()
    assert app.state.engine_client.active_requests == 0
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 499


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


def test_text_completion_preserves_raw_prompt(client_and_llm):
    client, llm = client_and_llm
    response = client.post("/v1/completions",
                           json={
                               "model": "fake-model",
                               "prompt": "raw prompt",
                               "max_tokens": 8,
                               "temperature": 0.25,
                               "top_p": 0.8,
                               "top_k": 7,
                               "stop": ["END"],
                               "disable_spec_decode": True,
                               "ignore_eos": True,
                               "reuse_context": False,
                               "cache_generated_tokens": False,
                           })

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["text"] == "answer"
    assert body["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }
    assert llm.prepared_prompt == "raw prompt"
    assert llm.prepare_count == 1
    params = llm.last_sampling_params
    assert params.max_tokens == 8
    assert params.temperature == pytest.approx(0.25)
    assert params.top_p == pytest.approx(0.8)
    assert params.top_k == 7
    assert params.stop == ["END"]
    assert params.disable_spec_decode
    assert params.ignore_eos
    assert not params.reuse_context
    assert not params.cache_generated_tokens
    assert params.skip_special_tokens


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("path, payload", [
    ("/v1/completions", {
        "prompt": [11, 12, 13]
    }),
    ("/v1/chat/completions", {
        "messages": [{
            "role": "user",
            "content": "hi"
        }]
    }),
    ("/v1/chat/completions", {
        "messages": [{
            "role": "user",
            "content": "Weather?"
        }],
        "tools": [_tool()],
        "tool_choice": "required",
    }),
    ("/v1/chat/completions", {
        "messages": [{
            "role": "user",
            "content": "Say hello"
        }],
        "modalities": ["text", "audio"],
        "audio": {
            "voice": "Ryan",
            "format": "pcm16"
        },
    }),
])
def test_usage_reports_the_prompt_tokens_the_runtime_reused(
        client_and_llm, path, payload, stream):
    """Given a runtime that served part of the prompt from reused state, when a
    completion, chat, tool-calling chat or spoken chat response reports usage,
    streamed or not, then prompt_tokens_details.cached_tokens is that count."""
    client, llm = client_and_llm
    llm.next_reused = 5
    body = dict(payload, model="fake-model", max_tokens=8)
    if stream:
        body.update(stream=True, stream_options={"include_usage": True})

    response = client.post(path, json=body)

    assert response.status_code == 200, response.text
    if stream:
        usage = next(payload["usage"] for payload in _sse_payloads(response)
                     if payload.get("usage"))
    else:
        usage = response.json()["usage"]
    assert usage["prompt_tokens_details"] == {"cached_tokens": 5}


def test_health_reports_context_cache_counters(client_and_llm):
    """Given a runtime that reports context-cache counters, when /health is
    read, then they appear under context_cache."""
    client, llm = client_and_llm
    counters = {"hit_sequences": 3, "reused_tokens": 384}
    llm.get_context_cache_counters = lambda: counters

    response = client.get("/health")

    assert response.status_code == 200, response.text
    assert response.json()["context_cache"] == counters


def test_token_completion_streams_exact_usage(client_and_llm):
    client, llm = client_and_llm
    response = client.post("/v1/completions",
                           json={
                               "prompt": [11, 12, 13],
                               "stream": True,
                               "stream_options": {
                                   "include_usage": True
                               },
                               "max_tokens": 8,
                           })

    assert response.status_code == 200, response.text
    payloads = _sse_payloads(response)
    text = "".join(payload["choices"][0]["text"] for payload in payloads
                   if payload["choices"])
    usage = next(payload["usage"] for payload in payloads
                 if payload.get("usage"))
    assert text == "hello"
    assert usage == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    params = llm.last_sampling_params
    assert params.temperature == pytest.approx(1.0)
    assert params.top_p == pytest.approx(1.0)
    assert params.top_k == 0
    assert llm.prepared_prompt == [11, 12, 13]
    assert response.text.rstrip().endswith("data: [DONE]")


@pytest.mark.parametrize("prompt", [[], [-1], [[1, 2]]])
def test_completion_rejects_invalid_token_prompts(client_and_llm, prompt):
    client, llm = client_and_llm
    response = client.post("/v1/completions", json={"prompt": prompt})

    assert response.status_code == 400
    assert llm.prepare_count == 0


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


@pytest.mark.parametrize("path, payload", [
    ("/v1/chat/completions", {
        "messages": [{
            "role": "user",
            "content": "hi"
        }]
    }),
    ("/v1/completions", {
        "prompt": "hi"
    }),
])
def test_hybrid_speculative_reuse_defaults_to_prefill_state(
        client_and_llm, path, payload):
    """On a deployment that publishes only prefill state, an omitted commit
    policy is accepted and an explicit generated-token commit is a 400."""
    client, llm = client_and_llm
    llm.publishes_prefill_state_only = True

    response = client.post(path, json=payload)
    assert response.status_code == 200, response.text
    assert llm.last_sampling_params.cache_generated_tokens is None

    response = client.post(path,
                           json={
                               **payload, "cache_generated_tokens": True
                           })
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == "cache_generated_tokens"


def test_chat_forwards_ignore_eos(client_and_llm):
    """Chat accepts ignore_eos, which fixed-length load clients send."""
    client, llm = client_and_llm
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role": "user",
                                   "content": "hi"
                               }],
                               "ignore_eos": True,
                           })

    assert response.status_code == 200, response.text
    assert llm.last_sampling_params.ignore_eos


def test_chat_normalizes_unsupported_sampling_fields_and_forwards_seed(
        client_and_llm, caplog):
    client, llm = client_and_llm
    seed = (1 << 64) - 1
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role": "user",
                                   "content": "Describe the video",
                               }],
                               "seed":
                               seed,
                               "top_k":
                               -1,
                               "min_p":
                               0.2,
                               "repetition_penalty":
                               1.2,
                           })

    assert response.status_code == 200, response.text
    assert llm.last_sampling_params.seed == seed
    assert llm.last_sampling_params.top_k == -1
    assert "min_p=0.2 is unsupported; using default 0.0" in caplog.text
    assert ("repetition_penalty=1.2 is unsupported; using default 1.0"
            in caplog.text)


def test_chat_template_is_default_on_and_request_configurable(client_and_llm):
    client, llm = client_and_llm
    body = {"messages": [{"role": "user", "content": "Hello"}]}

    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    assert llm.last_request_options["apply_chat_template"] is True
    assert llm.last_request_options["add_generation_prompt"] is True

    body.update({
        "apply_chat_template": False,
        "add_generation_prompt": False,
    })
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    assert llm.last_request_options["apply_chat_template"] is False
    assert llm.last_request_options["add_generation_prompt"] is False


def test_qwen38_reasoning_controls_reach_native_template(client_and_llm):
    client, llm = client_and_llm
    body = {
        "messages": [{
            "role": "user",
            "content": "Solve 23 * 9."
        }],
        "enable_thinking": True,
        "reasoning_effort": "low",
    }

    response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200, response.text
    assert llm.last_sampling_params.enable_thinking is True
    assert llm.last_sampling_params.reasoning_effort == "low"

    body.pop("enable_thinking")
    body.pop("reasoning_effort")
    body["chat_template_kwargs"] = {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }
    response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200, response.text
    assert llm.last_sampling_params.enable_thinking is True
    assert llm.last_sampling_params.reasoning_effort == "medium"


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
    from tensorrt_edgellm.server.api.protocol import ChatCompletionRequest
    from tensorrt_edgellm.server.api.serving_chat import (OpenAIServingChat,
                                                          PreparedChatRequest)
    from tensorrt_edgellm.server.parsing.tool_calling import \
        validate_tool_request

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
                                       False,
                                       BlockingClient().stream())
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
    # Tools with automatic selection must not buffer a plain-text streaming
    # answer until generation ends.
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
    # Both streaming paths must retain logprobs for deltas whose bytes the tool
    # or reasoning parser withholds, matching the non-streaming response.
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

    from tensorrt_edgellm.server.runtime.engine_layout import inspect_bundle

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
    from tensorrt_edgellm.server.media.media_source import \
        enforce_local_media_policy

    with pytest.raises(PermissionError):
        enforce_local_media_policy(_media_messages(item), "")


def test_local_media_policy_allows_root_and_rejects_traversal(tmp_path):
    from tensorrt_edgellm.server.media.media_source import \
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
    from tensorrt_edgellm.server.media.media_source import \
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


def test_visual_profile_capacity_error_maps_to_413(tmp_path):
    """A visual profile overflow during request preparation becomes HTTP 413
    and releases the admission lease."""
    from tensorrt_edgellm.server.media import video_sampling
    from tensorrt_edgellm.server.runtime.engine_client import EngineClient

    llm = _FakeLLM(tmp_path)

    def fail(*_args, **_kwargs):
        raise video_sampling.VisualProfileCapacityError(
            "visual engine profile exceeded")

    llm._make_generation_request = fail
    client = EngineClient(llm, ApiConfig())

    async def exercise():
        with pytest.raises(PayloadTooLargeError) as error:
            await client.prepare_request([{
                "role": "user",
                "content": "too many images"
            }], SamplingParams())
        assert error.value.status_code == 413
        assert error.value.param == "messages"
        assert "visual engine profile exceeded" in str(error.value)
        assert client.active_requests == 0

    asyncio.run(exercise())


@pytest.mark.parametrize("field,value", [
    ("top_k", 1.5),
    ("max_tokens", 2.5),
    ("temperature", "hot"),
    ("top_p", 2.0),
    ("min_p", 1.1),
    ("repetition_penalty", 0),
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


@pytest.mark.parametrize("seed", [-1, 1 << 64],
                         ids=["negative", "exceeds-uint64"])
def test_sampling_schema_rejects_seed_outside_uint64(client_and_llm, seed):
    client, _ = client_and_llm
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role": "user",
                                   "content": "Hello"
                               }],
                               "seed": seed,
                           })
    assert response.status_code == 400
    assert "seed" in response.text


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
    from tensorrt_edgellm.server.api import app as app_module

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
    from tensorrt_edgellm.server.api.app import _content_length

    assert _content_length(value) == expected


def test_app_reports_package_version(tmp_path):
    from fastapi.testclient import TestClient

    from tensorrt_edgellm import __version__

    client = TestClient(_create_app(_FakeLLM(tmp_path)))
    assert client.get("/openapi.json").json()["info"]["version"] == __version__


def test_media_is_rejected_by_the_loaded_model_contract(tmp_path):
    from fastapi.testclient import TestClient

    from tensorrt_edgellm.server.runtime.engine_layout import inspect_bundle

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
    from tensorrt_edgellm.server.media import media_source

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


def test_image_data_url_reaches_bytes_loader(monkeypatch):
    from tensorrt_edgellm.server.media import video_sampling
    from tensorrt_edgellm.server.runtime.engine import _load_image_buffers

    class Image:
        do_resize = True

    class Runtime:
        loaded = None

        @classmethod
        def load_image_from_bytes(cls, data):
            cls.loaded = data
            return Image()

    captured = {}

    def estimate(source, family, limits, do_resize=True):
        captured.update(source=source,
                        family=family,
                        limits=limits,
                        do_resize=do_resize)
        return 4

    monkeypatch.setattr(video_sampling, "estimate_image_tokens", estimate)
    limits = {
        "model_type": "muse_glimmer_vision",
        "min_image_tokens": 1,
        "max_image_tokens": 8,
        "max_image_tokens_per_image": 8,
    }
    payload = base64.b64encode(b"encoded-image").decode()
    buffers = _load_image_buffers(
        Runtime, [{
            "role":
            "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{payload}"
                },
            }],
        }], lambda: pytest.fail("image-only request used video routing"),
        lambda: limits)
    assert len(buffers) == 1
    assert Runtime.loaded == b"encoded-image"
    assert captured == {
        "source": b"encoded-image",
        "family": "muse",
        "limits": limits,
        "do_resize": True,
    }


@pytest.mark.parametrize("stream_kind", ["normal", "audio"])
def test_disconnect_during_a_native_read_holds_lease_until_cleanup(
        tmp_path, stream_kind):
    """A cancelled native read retains admission until its cleanup returns."""
    # The common disconnect: the client leaves while next() is waiting in a
    # worker thread for the next token. Starlette re-cancels the response task
    # at every await; the lease must still outlive native teardown.
    import threading

    pytest.importorskip("fastapi")
    from tensorrt_edgellm.server.api.protocol import ChatCompletionRequest
    from tensorrt_edgellm.server.api.routes import _ReleasingStreamingResponse
    from tensorrt_edgellm.server.api.serving_chat import OpenAIServingChat
    from tensorrt_edgellm.server.runtime.engine import CancellableIterator
    from tensorrt_edgellm.server.runtime.engine_client import EngineClient

    llm = _FakeLLM(tmp_path)
    cancelled = threading.Event()
    in_native_read = threading.Event()
    cleanup_entered = threading.Event()
    cleanup_may_finish = threading.Event()

    def native_stream(*_args, **_kwargs):

        def iterate():
            try:
                yield StreamDelta(text="first", token_ids=[1], prompt_tokens=7)
                # Like channel.wait_pop(): inside next() until a token arrives
                # or the stream is cancelled.
                in_native_read.set()
                cancelled.wait()
            finally:
                # Like worker.join(): native handleRequest is still unwinding.
                cleanup_entered.set()
                cleanup_may_finish.wait()

        return CancellableIterator(iterate(), cancelled.set)

    if stream_kind == "audio":
        llm.generate_stream_with_audio = native_stream
    else:
        llm.generate_stream = native_stream

    async def exercise():
        config = ApiConfig(max_queued_requests=0)
        client = EngineClient(llm, config)
        serving = OpenAIServingChat(client, config)
        request_kwargs = {}
        if stream_kind == "audio":
            request_kwargs.update({
                "modalities": ["text", "audio"],
                "audio": {
                    "voice": "Ryan",
                    "format": "pcm16",
                },
            })
        request = ChatCompletionRequest(messages=[{
            "role": "user",
            "content": "first"
        }],
                                        stream=True,
                                        **request_kwargs)
        prepared_chat = serving.prepare_request(request)
        engine_request = await serving.prepare_engine_request(
            request, prepared_chat)
        response = _ReleasingStreamingResponse(
            serving.stream_chat_completion(request, prepared_chat,
                                           engine_request),
            prepared_stream=engine_request,
            media_type="text/event-stream",
        )

        async def send(_message):
            pass

        async def receive():
            await asyncio.to_thread(in_native_read.wait)
            return {"type": "http.disconnect"}

        scope = {
            "type": "http",
            "asgi": {
                "version": "3.0",
                "spec_version": "2.3",
            },
        }
        response_task = asyncio.create_task(response(scope, receive, send))
        unexpected_request = None
        try:
            assert await asyncio.to_thread(cleanup_entered.wait, 1.0)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(response_task), 0.1)
            assert client.active_requests == 1
            try:
                unexpected_request = await client.prepare_request(
                    [{
                        "role": "user",
                        "content": "second"
                    }], prepared_chat.sampling)
            except ServerOverloadedError:
                pass
            else:
                pytest.fail(
                    "disconnect released admission before native cleanup finished"
                )
        finally:
            if unexpected_request is not None:
                unexpected_request.release()
            cleanup_may_finish.set()
            await response_task
        assert client.active_requests == 0

    asyncio.run(exercise())


@pytest.mark.parametrize("stream_kind", ["audio", "speech"])
def test_audio_close_in_cancelled_scope_keeps_lease_until_cleanup(
        tmp_path, stream_kind):
    """Audio and speech closure wait for native cleanup before releasing admission."""
    import threading

    import anyio

    from tensorrt_edgellm.server.runtime.engine import AudioParams
    from tensorrt_edgellm.server.runtime.engine_client import EngineClient

    llm = _FakeLLM(tmp_path)
    cleanup_entered = threading.Event()
    cleanup_may_finish = threading.Event()
    cleanup_finished = threading.Event()

    def native_stream(*_args, **_kwargs):
        try:
            yield StreamDelta(text="first",
                              token_ids=[1],
                              prompt_tokens=7,
                              audio_bytes=b"\x01\x00")
        finally:
            cleanup_entered.set()
            try:
                if not cleanup_may_finish.wait(5):
                    raise RuntimeError("test did not release native cleanup")
            finally:
                cleanup_finished.set()

    llm.generate_stream_with_audio = native_stream
    llm.generate_speech_stream = native_stream

    async def exercise():
        client = EngineClient(llm, ApiConfig(max_queued_requests=0))
        messages = [{"role": "user", "content": "first"}]
        sampling = SamplingParams()
        owned = await client.prepare_request(messages, sampling)
        if stream_kind == "audio":
            stream = client.stream_with_audio(messages,
                                              sampling,
                                              AudioParams(),
                                              prepared=owned)
        else:
            stream = client.stream_speech("first",
                                          AudioParams(),
                                          prepared=owned.lease)
        await stream.__anext__()

        async def close_under_cancellation():
            with anyio.CancelScope() as scope:
                scope.cancel()
                await stream.aclose()

        close_task = asyncio.create_task(close_under_cancellation())
        unexpected = None
        try:
            if not await asyncio.to_thread(cleanup_entered.wait, 1):
                pytest.fail("native cleanup did not start")
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(close_task), 0.05)
            assert client.active_requests == 1
            try:
                unexpected = await client.prepare_request(messages, sampling)
            except ServerOverloadedError:
                pass
            else:
                pytest.fail(
                    "audio close released admission during native cleanup")
        finally:
            if unexpected is not None:
                unexpected.release()
            cleanup_may_finish.set()
            await close_task
            await asyncio.to_thread(cleanup_finished.wait, 1)
        assert client.active_requests == 0

    asyncio.run(exercise())


def test_blocking_cuda_failure_reports_unavailable_and_rejects_waiters(
        tmp_path):
    """A fatal drain rejects queued and new requests before preparation and makes both health endpoints fail."""
    from fastapi.testclient import TestClient

    from tensorrt_edgellm.server.api.errors import EngineUnavailableError
    from tensorrt_edgellm.server.runtime.engine import LLM
    from tensorrt_edgellm.server.runtime.engine_client import EngineClient

    llm = _FakeLLM(tmp_path)
    holder = LLM.__new__(LLM)
    holder._runtime = SimpleNamespace(unavailable_reason="")
    llm.engine_unavailable_reason = holder.engine_unavailable_reason
    llm.get_engine_status = holder.get_engine_status
    reason = "cudaErrorIllegalAddress (700): CUDA request cleanup failed; restart the process"

    async def exercise():
        client = EngineClient(llm, ApiConfig(max_queued_requests=1))
        first = await client.prepare_completion("first", SamplingParams())
        waiting = asyncio.create_task(
            client.prepare_completion("waiting", SamplingParams()))
        while client.queued_requests == 0:
            await asyncio.sleep(0)
        holder._runtime.unavailable_reason = reason
        first.release()
        with pytest.raises(EngineUnavailableError,
                           match="cudaErrorIllegalAddress"):
            await waiting
        with pytest.raises(EngineUnavailableError,
                           match="cudaErrorIllegalAddress"):
            await client.prepare_completion("new", SamplingParams())
        assert llm.prepare_count == 1
        assert client.active_requests == 0

    asyncio.run(exercise())
    client = TestClient(_create_app(llm))
    for path in ("/health", "/health/ready"):
        response = client.get(path)
        assert response.status_code == 503
        assert response.json()["reason"] == reason


@pytest.mark.parametrize("failure_phase", ["before", "queued", "queue_full"])
def test_speech_rejects_failed_runtime_before_headers(tmp_path, failure_phase):
    """Failed speech requests return 503 before headers or native work, even after queueing."""
    import httpx

    from tensorrt_edgellm.server.runtime.engine import LLM

    llm = _FakeLLM(tmp_path)
    holder = LLM.__new__(LLM)
    holder._runtime = SimpleNamespace(unavailable_reason="")
    llm.engine_unavailable_reason = holder.engine_unavailable_reason
    reason = "cudaErrorIllegalAddress (700): restart the process"
    app = _create_app(
        llm,
        ApiConfig(
            max_queued_requests=0 if failure_phase == "queue_full" else 1))
    client = app.state.engine_client

    async def exercise():
        first = None
        pending = None
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as http:
            try:
                if failure_phase != "before":
                    first = await client.prepare_completion(
                        "first", SamplingParams())
                if failure_phase != "queued":
                    holder._runtime.unavailable_reason = reason
                pending = asyncio.create_task(
                    http.post("/v1/audio/speech",
                              json={
                                  "input": "Hello",
                                  "voice": "Ryan",
                                  "response_format": "pcm"
                              }))
                if failure_phase == "queued":

                    async def wait_for_queue():
                        while client.queued_requests == 0:
                            await asyncio.sleep(0)

                    await asyncio.wait_for(wait_for_queue(), 1)
                    holder._runtime.unavailable_reason = reason
                    first.release()
                response = await asyncio.wait_for(pending, 1)
                assert response.status_code == 503
                assert response.json()["error"]["type"] == "engine_unavailable"
                assert reason in response.json()["error"]["message"]
                assert llm.last_audio_params is None
            finally:
                if first is not None:
                    first.release()
                if pending is not None and not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
        assert client.active_requests == 0
        assert client.queued_requests == 0

    asyncio.run(exercise())


def test_direct_speech_rejects_failed_runtime_and_waiters(tmp_path):
    """Direct speech streams reject both queued and new work after a fatal failure."""
    from tensorrt_edgellm.server.api.errors import EngineUnavailableError
    from tensorrt_edgellm.server.runtime.engine import LLM, AudioParams
    from tensorrt_edgellm.server.runtime.engine_client import EngineClient

    llm = _FakeLLM(tmp_path)
    holder = LLM.__new__(LLM)
    holder._runtime = SimpleNamespace(unavailable_reason="")
    llm.engine_unavailable_reason = holder.engine_unavailable_reason

    async def exercise():
        client = EngineClient(llm, ApiConfig(max_queued_requests=1))
        first = await client.prepare_completion("first", SamplingParams())
        stream = client.stream_speech("waiting", AudioParams())
        pending = asyncio.create_task(anext(stream))
        try:

            async def wait_for_queue():
                while client.queued_requests == 0:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_for_queue(), 1)
            holder._runtime.unavailable_reason = "cudaErrorIllegalAddress"
            first.release()
            with pytest.raises(EngineUnavailableError,
                               match="cudaErrorIllegalAddress"):
                await asyncio.wait_for(pending, 1)
            with pytest.raises(EngineUnavailableError,
                               match="cudaErrorIllegalAddress"):
                await anext(client.stream_speech("new", AudioParams()))
            assert llm.last_audio_params is None
            assert client.active_requests == 0
            assert client.queued_requests == 0
        finally:
            first.release()
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await stream.aclose()

    asyncio.run(exercise())


@pytest.mark.parametrize("disconnect_at", ["headers", "body"])
def test_speech_response_closes_before_releasing_admission(
        tmp_path, disconnect_at):
    """Speech disconnects release admission only after closing any started native stream."""
    from tensorrt_edgellm.server.api.protocol import SpeechRequest
    from tensorrt_edgellm.server.api.routes import audio_speech

    llm = _FakeLLM(tmp_path)
    app = _create_app(llm, ApiConfig(max_queued_requests=0))
    client = app.state.engine_client
    cleanup_active = []

    def native_stream(*_args, **_kwargs):
        try:
            yield StreamDelta(audio_bytes=b"\x01\x00")
        finally:
            cleanup_active.append(client.active_requests)

    llm.generate_speech_stream = native_stream

    async def exercise():
        response = await audio_speech(SpeechRequest(input="Hello"),
                                      SimpleNamespace(app=app))
        assert client.active_requests == 1

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            if disconnect_at == "headers" or message[
                    "type"] == "http.response.body":
                raise OSError("client disconnected")

        from starlette.requests import ClientDisconnect
        try:
            with pytest.raises(ClientDisconnect):
                await response(
                    {
                        "type": "http",
                        "asgi": {
                            "spec_version": "2.4"
                        }
                    }, receive, send)
            assert client.active_requests == 0
            assert cleanup_active == ([1] if disconnect_at == "body" else [])
        finally:
            await response.body_iterator.aclose()

    asyncio.run(exercise())
