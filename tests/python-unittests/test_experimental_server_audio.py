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
"""Tests for native audio streaming helpers (no GPU / native module)."""

import threading
from collections import deque
from types import SimpleNamespace

import pytest

from experimental.server.runtime import engine as server_engine
from experimental.server.runtime.engine import (TTS, AudioParams,
                                                _native_audio_params,
                                                _pump_channels, _stream_tts)
from experimental.server.runtime.engine_build import PreparedModel

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeFinishReason:
    NOT_FINISHED = 0
    END_ID = 1
    LENGTH = 2
    CANCELLED = 3
    ERROR = 4
    STOP_WORDS = 5


FAKE_RT = SimpleNamespace(FinishReason=_FakeFinishReason)


class _FakeChannel:
    """Mimics the pybind channel pop/terminal API for one producer run."""

    def __init__(self, chunks, finished_from_start=False):
        self._chunks = deque(chunks)
        self.finished = finished_from_start
        self.cancelled = False

    def wait_pop(self, timeout_ms):
        return self.try_pop()

    def try_pop(self):
        return self._chunks.popleft() if self._chunks else None

    def is_finished(self):
        return self.finished

    def is_cancelled(self):
        return self.cancelled

    def cancel(self):
        self.cancelled = True


def _text_chunk(text,
                finished=False,
                reason=_FakeFinishReason.END_ID,
                prompt_token_count=4):
    return SimpleNamespace(
        text=text,
        token_ids=[1],
        finished=finished,
        reason=reason if finished else _FakeFinishReason.NOT_FINISHED,
        logprobs=[],
        prompt_token_count=prompt_token_count,
    )


def _audio_chunk(pcm16=b"", is_final=False):
    return SimpleNamespace(pcm16=pcm16, is_final=is_final, num_frames=1)


def _run_pump(text_channel, audio_channel, run=lambda: None):
    for chan in (text_channel, audio_channel):
        if chan is not None:
            chan.finished = True
    return list(_pump_channels(FAKE_RT, run, text_channel, audio_channel))


# ---------------------------------------------------------------------------
# _pump_channels
# ---------------------------------------------------------------------------


def test_pump_audio_only_orders_and_terminates():
    audio = _FakeChannel(
        [_audio_chunk(b"a"),
         _audio_chunk(b"b", is_final=True)])
    deltas = _run_pump(None, audio)
    assert [d.audio_bytes for d in deltas] == [b"a", b"b"]
    assert all(not d.text for d in deltas)


def test_pump_empty_final_chunk_yields_nothing():
    audio = _FakeChannel([_audio_chunk(b"", is_final=True)])
    assert _run_pump(None, audio) == []


def test_pump_dual_stream_carries_text_fields():
    text = _FakeChannel(
        [_text_chunk("hello "),
         _text_chunk("world", finished=True)])
    audio = _FakeChannel([_audio_chunk(b"pcm", is_final=True)])
    deltas = _run_pump(text, audio)
    texts = [d for d in deltas if d.text]
    audios = [d for d in deltas if d.audio_bytes]
    assert [d.text for d in texts] == ["hello ", "world"]
    assert texts[0].finish_reason is None
    assert texts[1].finished and texts[1].finish_reason == "stop"
    assert [d.audio_bytes for d in audios] == [b"pcm"]


def test_pump_drains_chunks_pending_after_finish():
    # Producer finished before the consumer's first pop: nothing may be lost.
    audio = _FakeChannel(
        [_audio_chunk(b"x"), _audio_chunk(b"y")], finished_from_start=True)
    deltas = list(_pump_channels(FAKE_RT, lambda: None, None, audio))
    assert [d.audio_bytes for d in deltas] == [b"x", b"y"]


def test_pump_reraises_worker_error():
    audio = _FakeChannel([])

    def _boom():
        raise RuntimeError("talker failed")

    with pytest.raises(RuntimeError, match="talker failed"):
        list(_pump_channels(FAKE_RT, _boom, None, audio))
    assert audio.cancelled


def test_pump_cancels_channels_on_early_close():
    text = _FakeChannel([_text_chunk("partial")])
    audio = _FakeChannel([])
    gen = _pump_channels(FAKE_RT, lambda: None, text, audio)
    assert next(gen).text == "partial"
    gen.close()
    assert text.cancelled and audio.cancelled


# ---------------------------------------------------------------------------
# _native_audio_params / checkpoint-direct TTS construction
# ---------------------------------------------------------------------------


def test_native_audio_params_maps_all_fields():
    captured = SimpleNamespace()
    rt = SimpleNamespace(OmniAudioParams=lambda: captured)
    native = _native_audio_params(
        rt, AudioParams(voice="ryan", talker_top_k=7, codec_chunk_frames=4))
    assert native is captured
    assert captured.speaker_name == "ryan"
    assert captured.talker_top_k == 7
    assert captured.codec_chunk_frames == 4
    assert not hasattr(captured, "voice")


def _mock_prepared_tts(monkeypatch, tmp_path, *, complete=True):
    checkpoint = tmp_path / "checkpoint"
    bundle = tmp_path / "cache" / "bundle"
    checkpoint.mkdir(parents=True)
    for component, filename in (("talker", "llm.engine"), ("code_predictor",
                                                           "llm.engine"),
                                ("code2wav", "code2wav.engine")):
        directory = bundle / component
        directory.mkdir(parents=True, exist_ok=True)
        if complete or component == "talker":
            (directory / filename).touch()
    prepared = PreparedModel(str(bundle), str(checkpoint))
    monkeypatch.setattr(
        "experimental.server.runtime.engine_build.prepare_model",
        lambda *args, **kwargs: prepared)
    return prepared


def test_tts_rejects_incomplete_model_bundle(monkeypatch, tmp_path):
    _mock_prepared_tts(monkeypatch, tmp_path, complete=False)
    with pytest.raises(ValueError, match="complete TTS runtime"):
        TTS(model=str(tmp_path / "checkpoint"))


def test_tts_loads_all_components_from_model_cache(monkeypatch, tmp_path):
    prepared = _mock_prepared_tts(monkeypatch, tmp_path)
    captured = {}
    runtime = SimpleNamespace(get_speaker_names=lambda: ["Ryan"])

    def create_runtime(**kwargs):
        captured.update(kwargs)
        return runtime

    monkeypatch.setattr(
        server_engine,
        "_import_runtime",
        lambda: SimpleNamespace(TTSRuntime=create_runtime),
    )
    tts = TTS(model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
              cache_dir=str(tmp_path / "cache"))

    assert tts.bundle_dir == prepared.bundle_dir
    assert tts.model_dir == prepared.model_dir
    assert captured["checkpoint_dir"] == prepared.model_dir
    assert captured["talker_engine_dir"].endswith("/talker")
    assert captured["code_predictor_engine_dir"].endswith("/code_predictor")
    assert captured["code2wav_engine_dir"].endswith("/code2wav")
    assert tts.list_voices() == ["Ryan"]


# ---------------------------------------------------------------------------
# _stream_tts gate ownership
# ---------------------------------------------------------------------------


def _tts_rt(channel):
    return SimpleNamespace(FinishReason=_FakeFinishReason,
                           AudioStreamChannel=lambda: channel,
                           OmniAudioParams=lambda: SimpleNamespace())


def test_stream_tts_acquires_gate_for_direct_callers():
    channel = _FakeChannel([_audio_chunk(b"pcm", is_final=True)],
                           finished_from_start=True)
    sem = threading.Semaphore(1)
    runtime = SimpleNamespace(handle_request_tts=lambda *a: None)
    deltas = list(
        _stream_tts(_tts_rt(channel), runtime, "hi", AudioParams(), sem))
    assert [d.audio_bytes for d in deltas] == [b"pcm"]
    # Acquired on entry, released by the worker: back to its initial count.
    assert sem.acquire(blocking=False)


def test_stream_tts_holds_infer_guard_during_call():
    """The batcher takes only this lock, so TTS must hold it while the C++
    call runs or batched text would enter the runtime concurrently."""
    channel = _FakeChannel([_audio_chunk(b"pcm", is_final=True)],
                           finished_from_start=True)
    guard = threading.Lock()
    held = []
    runtime = SimpleNamespace(
        handle_request_tts=lambda *a: held.append(guard.locked()))
    list(
        _stream_tts(_tts_rt(channel),
                    runtime,
                    "hi",
                    AudioParams(),
                    None,
                    infer_guard=guard))
    assert held == [True]
    assert not guard.locked()
