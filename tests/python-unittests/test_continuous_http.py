# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the prepared HTTP boundary without a GPU."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from experimental.server.config import ApiConfig
from experimental.server.runtime import engine_client
from experimental.server.runtime.engine import LLM, SamplingParams


class Options(SimpleNamespace):
    def set_queue_timeout_ms(self, value):
        self.timeout = value


class Ticket:
    def __init__(self):
        self.cancelled = threading.Event()

    def cancel(self):
        self.cancelled.set()


class Runtime:
    def __init__(self):
        self.tickets = []
        self.failure = None

    def prepare_continuous_prompt(self, request):
        return [1, 2, 3]

    def submit_continuous(self, prompt, options):
        if self.failure:
            raise RuntimeError(self.failure)
        ticket = Ticket()
        self.tickets.append(ticket)
        return ticket

    def continuous_healthy(self):
        return self.failure != "failed"


class Model:
    model_dir = "test"
    model_id = "openclaw"
    continuous_batching_enabled = True

    def __init__(self):
        self._runtime = Runtime()
        self._rt = SimpleNamespace(ContinuousRequestOptions=Options,
                                   ContinuousSequenceOptions=Options)

    _continuous_options = LLM._continuous_options

    def _make_generation_request(self, *args, **kwargs):
        return SimpleNamespace()

    def _complete_continuous_request(self, *args, ticket, **kwargs):
        assert ticket in self._runtime.tickets
        return "completed"

    def generate_continuous_stream(self, *args, ticket):
        assert ticket in self._runtime.tickets
        yield "streamed"


def client(monkeypatch):
    monkeypatch.setattr(engine_client, "_capabilities_for", lambda llm: None)
    return engine_client.EngineClient(Model(), ApiConfig())


def test_default_stream_storage_fits_native_admission():
    options = Model()._continuous_options(SamplingParams(), stream=True)
    assert options.stream_records * 1024 + options.stream_bytes + 6144 * 4 < 256 * 1024
    assert Model()._continuous_options(SamplingParams(), stream=False).stream_records == 0


@pytest.mark.asyncio
async def test_prepared_requests_overlap_and_use_native_tickets(monkeypatch):
    owner = client(monkeypatch)
    params = SamplingParams(max_tokens=4)
    first = await owner.prepare_request([], params)
    second = await asyncio.wait_for(owner.prepare_request([], params, stream=True), .5)
    assert first.ticket is not second.ticket
    assert await owner.generate([], params, prepared=first) == "completed"
    assert [item async for item in owner.stream([], params, prepared=second)] == ["streamed"]
    assert first.ticket.cancelled.is_set() and second.ticket.cancelled.is_set()
    assert owner._admission.active == 0


@pytest.mark.asyncio
async def test_nonstream_disconnect_cancels_owned_ticket(monkeypatch):
    owner = client(monkeypatch)
    params = SamplingParams(max_tokens=4)
    prepared = await owner.prepare_request([], params)
    entered = threading.Event()

    def blocking(*args, ticket, **kwargs):
        entered.set()
        assert ticket.cancelled.wait(2)

    owner._llm._complete_continuous_request = blocking
    task = asyncio.create_task(owner.generate([], params, prepared=prepared))
    assert await asyncio.to_thread(entered.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert prepared.ticket.cancelled.is_set()
    assert owner._admission.active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,exception", [
    ("Scheduler queue full", engine_client.ServerOverloadedError),
    ("failed", engine_client.ServerUnavailableError)])
async def test_native_rejection_releases_preparation(monkeypatch, failure, exception):
    owner = client(monkeypatch)
    owner._llm._runtime.failure = failure
    with pytest.raises(exception):
        await owner.prepare_request([], SamplingParams())
    assert owner._admission.active == 0
    assert owner.healthy == (failure != "failed")


def test_text_only_flush_does_not_inflate_usage():
    llm = LLM.__new__(LLM)
    llm._rt = SimpleNamespace(SchedulerStatus=SimpleNamespace(COMPLETED=1, DEADLINE=2),
                             SequenceFinish=SimpleNamespace(LENGTH=1, EOS=2, STOP=3))
    reads = iter([
        SimpleNamespace(update=SimpleNamespace(text="A", sample=SimpleNamespace(token=8)), closed=False),
        SimpleNamespace(update=SimpleNamespace(text="!", sample=SimpleNamespace(token=-1)), closed=True)])
    ticket = SimpleNamespace(read=lambda **kwargs: next(reads), cancel=lambda: None,
                             result=lambda: SimpleNamespace(status=1, finish=1, prompt_tokens=3))
    updates = list(llm.generate_continuous_stream(None, SamplingParams(), ticket=ticket))
    assert [token for update in updates for token in update.token_ids] == [8]
    assert "".join(update.text for update in updates) == "A!"
    assert updates[-1].finished


def test_sampled_token_outside_top_logprobs_keeps_probability():
    from experimental.server.api.serving_chat import _format_logprob_steps
    llm = LLM.__new__(LLM)
    llm._runtime = SimpleNamespace(continuous_token_piece=lambda token: b"X" if token == 9 else b"A")
    sample = SimpleNamespace(token=9, logprob=-3.0, top_count=1,
                             top=[SimpleNamespace(token=1, logprob=-.5)])
    result = _format_logprob_steps([9], llm._continuous_logprobs([sample]), True)
    entry = result['content'][0]
    assert entry['token'] == 'X' and entry['logprob'] == -3.0
    assert len(entry['top_logprobs']) == 1 and entry['top_logprobs'][0]['token_id'] == 1
